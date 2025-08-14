import os
import shutil
import torch
import tiktoken
from triton.testing import do_bench
from baseline_model import GPT as BaselineGPT
from optimized_model import GPT as OptimizedGPT

# -----------------------------------------------------------------------------
model_name = "gpt2"
max_new_tokens = 100 # number of tokens generated in each sample
temperature = 1.0 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 1 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337
device = 'cuda' # examples: 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
atol = 1e-3
warmup_s = 10
rep_s = 50
# -----------------------------------------------------------------------------
correctness_prompts = [
    "Hi",
    "Hello, how are you?",
    "What is the capital of France? I thought it was Nice. I think that wrong.",
    "There once was a man from Nantucket. He went to the store to buy some peanuts. He bought a pound of peanuts.",
    "Explain the concept of quantum entanglement, but explain it in a way that is easy to understand...as if you were explaining it to a 5 year old.",
]
timing_prompt = "What is the meaning of life, the universe, and everything?"
# -----------------------------------------------------------------------------
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0"

# reset torch extensions cache
shutil.rmtree(os.path.expanduser("~/.cache/torch_extensions"), ignore_errors=True)

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# models
baseline_model = BaselineGPT.from_pretrained(model_name, dict(dropout=0.0))
baseline_model.eval()
baseline_model.to(device)

optimized_model = OptimizedGPT.from_pretrained(model_name, dict(dropout=0.0))
optimized_model.eval()
optimized_model.to(device)

enc = tiktoken.get_encoding("gpt2")
encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})  # noqa
decode = lambda l: enc.decode(l)  # noqa

# encode the beginning of the prompt
correctness_start_ids = [(torch.tensor(encode(p), dtype=torch.long, device=device)[None, ...]) for p in correctness_prompts]
timing_start_ids = (torch.tensor(encode(timing_prompt), dtype=torch.long, device=device)[None, ...])

with torch.no_grad():
    with ctx:
        print("Checking logits, tokens and character generation correctness...")
        for i, x in enumerate(correctness_start_ids):
            # check logits correctness
            baseline_logits = baseline_model(x)[0][0]  # logits for batch size of 1
            optimized_logits = optimized_model(x)[0][0]  # logits for batch size of 1
            logits_diff = torch.abs(baseline_logits - optimized_logits)
            print(f"Test {i + 1}: Max float diff: {torch.max(logits_diff)}, Mean float diff: {torch.mean(logits_diff)}")
            if torch.max(logits_diff) > atol:
                print(f"[Error] Test {i + 1}: Logits are not close enough. atol: {atol}")

            # # run generation and check correctness
            # baseline_y = baseline_model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)[0]  # batch size of 1
            # optimized_y = optimized_model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)[0]  # batch size of 1
            # if (baseline_y != optimized_y).any():
            #     print(f"[Warning] Test {i}: Generation is not the same. At least one token is different.")

            # # check generation correctness
            # baseline_generation = decode(baseline_y.tolist())
            # optimized_generation = decode(optimized_y.tolist())
            # if baseline_generation != optimized_generation:
            #     print(f"[Warning] Test {i}: Generation is not the same. Baseline: {baseline_generation}, Optimized: {optimized_generation}")
            
        print("Measuring performance...")
        # measure prefill time
        baseline_prefill_time = do_bench(lambda: baseline_model(timing_start_ids), warmup=warmup_s * 1e3, rep=rep_s * 1e3)
        print(f"Prefill time: {baseline_prefill_time} seconds (baseline)")
        optimized_prefill_time = do_bench(lambda: optimized_model(timing_start_ids), warmup=warmup_s * 1e3, rep=rep_s * 1e3)
        print(f"Prefill time: {optimized_prefill_time} seconds (optimized)")
        prefill_speedup = baseline_prefill_time / optimized_prefill_time
        print(f"Prefill speedup: {prefill_speedup}")


        # measure end to end generation time
        baseline_end_to_end_time = do_bench(lambda: baseline_model.generate(timing_start_ids, max_new_tokens, temperature=temperature, top_k=top_k), warmup=warmup_s * 1e3, rep=rep_s * 1e3)
        print(f"Overall system generation time: {baseline_end_to_end_time} seconds (baseline)")
        optimized_end_to_end_time = do_bench(lambda: optimized_model.generate(timing_start_ids, max_new_tokens, temperature=temperature, top_k=top_k), warmup=warmup_s * 1e3, rep=rep_s * 1e3)
        print(f"Overall system generation time: {optimized_end_to_end_time} seconds (optimized)")
        end_to_end_speedup = baseline_end_to_end_time / optimized_end_to_end_time
        print(f"system_speedup: {end_to_end_speedup}")
