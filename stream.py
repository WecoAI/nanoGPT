import os
import shutil
import torch
import tiktoken
import time

# -----------------------------------------------------------------------------
model_name = 'gpt2-xl'
model_type = 'baseline'
start = "What is the meaning of life, the universe, and everything?"
max_new_tokens = 500 # number of tokens generated in each sample
temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = True # use PyTorch 2.0 to compile the model to be faster
stream = False
exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0"
shutil.rmtree(os.path.expanduser("~/.cache/torch_extensions"), ignore_errors=True)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype)

if model_type == 'baseline':
    from baseline_model import GPT as BaselineGPT
    model = BaselineGPT.from_pretrained(model_name, dict(dropout=0.0))
elif model_type == 'optimized':
    from optimized_model import GPT as OptimizedGPT
    model = OptimizedGPT.from_pretrained(model_name, dict(dropout=0.0))
else:
    raise ValueError(f"Invalid model type: {model_type}")
model.eval()
model.to(device)
if compile:
    model = torch.compile(model) # requires PyTorch 2.0 (optional)

enc = tiktoken.get_encoding("gpt2")
encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
decode = lambda l: enc.decode(l)

start_ids = encode(start)
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

# ask the user to hit enter to start
input("Press Enter/Return to start...\n")
start_time = time.time()
y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k, stream=stream)
if stream:
    for i in y:
        print(decode(i[0].tolist()), end='', flush=True)
else:
    print(decode(y[0].tolist()))
end_time = time.time()
print(f"\nTime taken: {end_time - start_time:.2f} seconds")
print(f"Tokens per second: {max_new_tokens / (end_time - start_time):.2f}")