# Optimizing Kernels

This document outlines the strategy to improve system_generation_time by writing fused and optimized in-line CUDA kernels using a single-file implementation.

## Goal 

Optimize the inference performance of the `GPTModel` class by fusing and rewriting operations and combinations of operations/methods/functions using PyTorch and in-line CUDA along with any other optimizations you can think of. The goal is to minimize the end to end system generation time compared to the baseline for a single request inference task.

## Requirements

- **Simplicity & Readability:** Write simple, easy-to-understand code and include clear comments.
- **Single-File Implementation:** Develop fused CUDA kernels within ONE file.
- **Code Generated Format:** The solution must include a class `GPTModel` (an instance of `nn.Module`), with the same interface for the `forward` and `generate` methods. There is no need to write code about instantiating `GPTMode`, functions to get the inputs and any other usage based stuff because the `GPTModel` class is imported and used in my custom script to evaluate the performance of your implementation through the `forward` and `generate` methods. Only focus on writing the `GPTModel` class.
- **Preserve Initialization:** Do not change the initialization of the `GPTModel` class.
- **Optimize for Single Request Inference:** The batch size will always be fixed at 1 as during inference, there will be only 1 request processed at a time. That being said, the batch dimension should still remain in the computations.
- **Replace with in-line CUDA:** Replace the classes `LayerNorm`, `GeLU`, `CausalSelfAttention`, `MLP`, `Block` and the `view` and `transpose` operations with in-line CUDA.
- **Minimize Kernel Launches:** Minimize the number of kernel launches by fusing kernels wherever possible to avoid kernel launch times.

## Potential Ideas (Non-Exhaustive)
- **Multiple Kernels Allowed:** You can define more than one kernel in the file if needed.
- **One Large Single Kernel:** You can even try to merge all of the operations into a single class in `GPTModel` and fuse operations into a single CUDA kernel.
- **Optimize For Single Forward and Generation Pass:** Do not try to cache values in the initialization to be used across different forward passes. You may optimize the computation in the initialization to make the forward pass more efficient as long as you are optimizing for each individual forward pass.
- **Avoid Templates:** Use plain fused kernel functions without CPP templates. DO NOT USE PyBind. 
- **No Fallback Implementation:** Do not include any alternative or fallback code.
- **No Need For Logs:** Do not worry about writing logs.
- **Don't Check For Packages:** Assume you have access to all the required packages.
- **Identify and Re-Write Repeated Operations:** `nn.Linear`, `nn.LayerNorm`, `nn.GeLU`, `view`, `transpose` are some operations/classes/methods/functions that are used quite frequently in the code. See if you can write efficient in-line CUDA ton code/optimizations for these such that they can be made much more efficient. Similarly do this for other commonly used operations you think would improve inference performance. The operations I've mentioned can be greatly sped up by fusing operations together using in-line CUDA.
- **Use System-Level Optimizations:** When writing in-line CUDA, use compiler flags that can make the compiled kernel faster. Take advantage of built-in settings in PyTorch such as `jit`, `torch.compile`, and other things to improve overall performance.
- **Downcast Data:** Use lower precision when possible and downcast weights to lower precision wherever required to optimized throughput. Just remember to cast it back to the original dtype when returning the result in `generate` and `forward`.
- **In-Place Operations:** Replace operations with in-place operations whenever possible and minimize movement of data to maximize throughput.
- **Numerical Approximations:** Make numerical approximations when possible to simplify or eliminate computations. The tests your code is evaluated on will tell you if you've gone too far.
- **Reason & List Ideas:** When planning your next optimization, list out potential optimization ideas, weigh them by their expected benefits based on your knowledge and past experiments, then choose the idea that will most likely yield the best performance and implement it. Don't be afraid to try the same idea more than once if you're combining something that didn't work before with a different idea as the combination may surprise you but use your best judgement.

## GPU Architecture
You are optimizibng code on an NVIDIA A100 40GB.

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
