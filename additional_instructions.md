# Optimizing Kernels

This document outlines the strategy to improve speedup by writing fused and optimized in-line CUDA/Triton/PyTorch kernels using a single-file implementation.

## Goal 

Optimize the inference performance of the `GPTModel` class by fusing and rewriting operations and combinations of operations/methods/functions using PyTorch, in-line CUDA and Triton along with any other optimizations you can think of. The goal is to maximize the end to end system speedup compared to the baseline for a single request inference task. Maintain the same interface for `GPTModel`'s methods. Focus on writing efficient in-line CUDA kernel more than any other optimization

## Requirements

- **Simplicity & Readability:** Write simple, easy-to-understand code and include clear comments.
- **Single-File Implementation:** Develop fused CUDA/Triton kernels within ONE file.
- **Code Generated Format:** The solution must include a class `GPTModel` (an instance of `nn.Module`), with the same interface for the `forward` and `generate` methods. There is no need to write code about instantiating `GPTMode`, functions to get the inputs and any other usage based stuff because the `GPTModel` class is imported and used in my custom script to evaluate the performance of your implementation through the `forward` and `generate` methods. Only focus on writing the `GPTModel` class.
- **Preserve Initialization:** Do not change the initialization of the `GPTModel` class.
- **Optimize for Single Request Inference:** The batch size will always be fixed at 1 as during inference, there will be only 1 request processed at a time. That being said, the batch dimension should still remain in the computations.

## Potential Ideas (Non-Exhaustive)
- **Multiple Kernels Allowed:** You can define more than one kernel in the file if needed.
- **One Large Single Kernel:** You can even try to merge all of the operations into a single class in `GPTModel` and fuse operations into a single CUDA/triton kernel.
- **Optimize For Single Forward and Generation Pass:** Do not try to cache values in the initialization to be used across different forward passes. You may optimize the computation in the initialization to make the forward pass more efficient as long as you are optimizing for each individual forward pass.
- **Avoid Templates:** Use plain fused kernel functions without CPP templates. DO NOT USE PyBind. 
- **No Fallback Implementation:** Do not include any alternative or fallback code.
- **No Need For Logs:** Do not worry about writing logs.
- **Don't Check For Packages:** Assume you have access to all the required packages.
- **Identify and Re-Write Repeated Operations:** `nn.Linear`, `nn.LayerNorm`, `nn.GeLU`, `view`, `transpose` are some operations/classes/methods/functions that are used quite frequently in the code. See if you can write efficient in-line CUDA or triton code/optimizations for these such that they can be made much more efficient. Similarly do this for other commonly used operations you think would improve inference performance. The operations I've mentioned can be greatly sped up by fusing operations together using in-line CUDA.
- **Try Different Frameworks:** When writing a triton kernel, also try the same using in-line CUDA to maximize your chances of finding the framework that works best.
- **Use System-Level Optimizations:** When writing in-line CUDA, use compiler flags that can make the compiled kernel faster. When writing Triton, use things like autotune to improve performance. Take advantage of built-in settings in PyTorch such as `jit`, `torch.compile`, and other things to improve overall performance.

Here's a summary of tips for writing efficient Triton kernels:
1.  **Understand the programming model:**
    *   Each kernel launch creates a grid of independent program instances, each processing a tile of data.
2.  **Plan your kernel:**
    *   Choose tile/block sizes carefully to balance parallelism and resource limits.
    *   Align tiles with memory layout for coalesced access.
3.  **Optimize memory access:**
    *   Use vectorized, aligned loads/stores.
    *   Minimize divergent accesses and bank conflicts.
4.  **Leverage Triton features:**
    *   Utilize vector operations, minimal control flow, and atomics effectively.

## GPU Architecture
You are optimizibng code on an NVIDIA A100 40GB.

## Simple Example (Triton)

### Baseline Code

The baseline implementation performs a sequence of operations including a linear layer and several reductions.

```python
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs a sequence of operations:
        - Matrix multiplication
        - Summation
        - Max
        - Average pooling
        - LogSumExp
        - LogSumExp
    """
    def __init__(self, in_features, out_features):
        super(Model, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1).
        """
        x = self.linear(x)  # (batch_size, out_features)
        x = torch.sum(x, dim=1, keepdim=True) # (batch_size, 1)
        x = torch.max(x, dim=1, keepdim=True)[0] # (batch_size, 1)
        x = torch.mean(x, dim=1, keepdim=True) # (batch_size, 1)
        x = torch.logsumexp(x, dim=1, keepdim=True) # (batch_size, 1)
        x = torch.logsumexp(x, dim=1, keepdim=True) # (batch_size, 1)
        return x
```

### Optimized Code

The optimized version employs a custom Triton kernel to fuse the linear layer and subsequent reduction operations.

```python
import torch
import torch.nn as nn
import triton
import triton.language as tl

# This fused kernel combines the linear operation and a full reduction over its outputs.
# In the original module the sequence of operations
#   x = self.linear(x)
#   x = torch.sum(x, dim=1, keepdim=True)
#   x = torch.max(x, dim=1, keepdim=True)[0]
#   x = torch.mean(x, dim=1, keepdim=True)
#   x = torch.logsumexp(x, dim=1, keepdim=True)
#   x = torch.logsumexp(x, dim=1, keepdim=True)
# mathematically collapses to computing:
#   output = sum_i (dot(x, weight[i, :]) + bias[i])
# Here we fuse the two adjacent operations that are easiest to merge:
# the linear layer and the subsequent summation reduction.
@triton.jit
def fused_kernel(x_ptr, weight_ptr, bias_ptr, out_ptr,
                 stride_x: tl.constexpr, stride_out: tl.constexpr,
                 in_features: tl.constexpr, out_features: tl.constexpr,
                 pad_in: tl.constexpr):
    # Each kernel instance processes one input row.
    pid = tl.program_id(0)

    # Compute pointers to the current row of input and output.
    x_row = x_ptr + pid * stride_x
    out_addr = out_ptr + pid * stride_out

    # Create a padded index range.
    offs = tl.arange(0, pad_in)
    # Load the input row with masking; no mask is used if in_features is already a power of 2.
    x_vals = tl.load(x_row + offs, mask=(offs < in_features), other=0.0)

    # Compute the aggregated weight vector by summing over all weight rows.
    # Each weight row corresponds to one linear output dimension.
    aggr_w = tl.zeros([pad_in], dtype=tl.float32)
    for i in range(out_features):
        w_vals = tl.load(weight_ptr + i * in_features + offs,
                         mask=(offs < in_features), other=0.0)
        aggr_w += w_vals

    # Dot product between the input row and the aggregated weight vector.
    dot_val = tl.sum(x_vals * aggr_w)

    # Sum all bias entries.
    aggr_bias = 0.0
    for i in range(out_features):
        aggr_bias += tl.load(bias_ptr + i)

    result = dot_val + aggr_bias
    tl.store(out_addr, result)


class Model(nn.Module):
    """
    Optimized Model that fuses an nn.Linear layer and subsequent reduction operations.

    Given the original operations:
         x = self.linear(x)
         x = torch.sum(x, dim=1, keepdim=True)
         x = torch.max(x, dim=1, keepdim=True)[0]
         x = torch.mean(x, dim=1, keepdim=True)
         x = torch.logsumexp(x, dim=1, keepdim=True)
         x = torch.logsumexp(x, dim=1, keepdim=True)

    Their net effect is equivalent to:
         output = sum_i (dot(x, weight[i, :]) + bias[i])
    This module exposes the same interface as before.
    """
    def __init__(self, in_features, out_features):
        super(Model, self).__init__()
        # Use a standard linear layer for identical parameter initialization.
        self.linear = nn.Linear(in_features, out_features)
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1).
        """
        batch_size = x.shape[0]
        # Allocate the output tensor.
        output = torch.empty((batch_size, 1), device=x.device, dtype=x.dtype)

        # Ensure input and output tensors are contiguous.
        if not x.is_contiguous():
            x = x.contiguous()
        if not output.is_contiguous():
            output = output.contiguous()

        # For a (batch_size, in_features) tensor, the row stride equals in_features.
        stride_x = x.stride(0)
        # For the output (batch_size, 1) tensor, the row stride is 1.
        stride_out = output.stride(0)

        # Compute the next power-of-2 padded size for in_features.
        pad_in = 1 << ((self.in_features - 1).bit_length())

        # Launch one kernel instance per input row.
        grid = (batch_size,)
        fused_kernel[grid](
            x,
            self.linear.weight,
            self.linear.bias,
            output,
            stride_x,
            stride_out,
            self.in_features,
            self.out_features,
            pad_in
        )
        return output
```

## Simple Example (CUDA)

### Baseline Code

The baseline implementation of the `AddModel` class simply performs an element-wise addition.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class AddModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, a, b):
        return a + b
```

### Optimized Code

The optimized version employs a custom CUDA kernel for fused element-wise addition. The kernel is defined and compiled inline using PyTorch's `load_inline`.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for element-wise addition
elementwise_add_source = '''
#include <torch/extension.h>
#include <cuda_runtime.h>

// CUDA kernel for element-wise addition
__global__ void elementwise_add_kernel(const float* a, const float* b, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] + b[idx];
    }
}

// Launch function for the CUDA kernel
torch::Tensor elementwise_add_cuda(torch::Tensor a, torch::Tensor b) {
    auto size = a.numel();
    auto out = torch::zeros_like(a);
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    elementwise_add_kernel<<<num_blocks, block_size>>>(a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), size);
    return out;
}
'''

# C++ function prototype declaration
elementwise_add_cpp_source = "torch::Tensor elementwise_add_cuda(torch::Tensor a, torch::Tensor b);"

# Compile the inline CUDA code for element-wise addition
elementwise_add = load_inline(
    name="elementwise_add",
    cpp_sources=elementwise_add_cpp_source,
    cuda_sources=elementwise_add_source,
    functions=["elementwise_add_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class AddModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.elementwise_add = elementwise_add

    def forward(self, a, b):
        return self.elementwise_add.elementwise_add_cuda(a, b)
```
