import os
import shutil
import torch
import tiktoken
import argparse
from triton.testing import do_bench

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate GPT model correctness and performance.")
    parser.add_argument('--model_name', type=str, default="gpt2", help='Model name')
    parser.add_argument('--max_new_tokens', type=int, default=100, help='Number of tokens to generate')
    parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature')
    parser.add_argument('--top_k', type=int, default=1, help='Top-k sampling')
    parser.add_argument('--seed', type=int, default=1337, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--dtype', type=str, default=None, help='Data type: float32, bfloat16, or float16')
    parser.add_argument('--atol', type=float, default=1e-3, help='Absolute tolerance for correctness')
    parser.add_argument('--warmup_s', type=int, default=10, help='Warmup seconds for benchmarking')
    parser.add_argument('--rep_s', type=int, default=50, help='Repetition seconds for benchmarking')
    return parser.parse_args()

def setup_environment(seed, device, dtype):
    os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0"
    shutil.rmtree(os.path.expanduser("~/.cache/torch_extensions"), ignore_errors=True)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device_type = 'cuda'
    if dtype is None:
        dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    return ctx, ptdtype

def get_models(model_name, device):
    from baseline_model import GPT as BaselineGPT
    from optimized_model import GPT as OptimizedGPT
    baseline_model = BaselineGPT.from_pretrained(model_name, dict(dropout=0.0))
    baseline_model.eval()
    baseline_model.to(device)
    optimized_model = OptimizedGPT.from_pretrained(model_name, dict(dropout=0.0))
    optimized_model.eval()
    optimized_model.to(device)
    return baseline_model, optimized_model

def get_tokenizer():
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})  # noqa
    decode = lambda l: enc.decode(l)  # noqa
    return encode, decode

def prepare_inputs(encode, device, correctness_prompts, timing_prompt):
    correctness_start_ids = [torch.tensor(encode(p), dtype=torch.long, device=device)[None, ...] for p in correctness_prompts]
    timing_start_ids = torch.tensor(encode(timing_prompt), dtype=torch.long, device=device)[None, ...]
    return correctness_start_ids, timing_start_ids

def check_correctness(baseline_model, optimized_model, correctness_start_ids, atol):
    print("Checking logits, tokens and character generation correctness...")
    for i, x in enumerate(correctness_start_ids):
        baseline_logits = baseline_model(x)[0][0]
        optimized_logits = optimized_model(x)[0][0]
        logits_diff = torch.abs(baseline_logits - optimized_logits)
        print(f"Test {i + 1}: Max float diff: {torch.max(logits_diff)}, Mean float diff: {torch.mean(logits_diff)}")
        if torch.max(logits_diff) > atol:
            print(f"[Error] Test {i + 1}: Logits are not close enough. atol: {atol}")

def measure_performance(baseline_model, optimized_model, timing_start_ids, max_new_tokens, temperature, top_k, warmup_s, rep_s):
    print("Measuring performance...")
    baseline_prefill_time = do_bench(lambda: baseline_model(timing_start_ids), warmup=warmup_s * 1e3, rep=rep_s * 1e3)
    print(f"Prefill time: {baseline_prefill_time} seconds (baseline)")
    optimized_prefill_time = do_bench(lambda: optimized_model(timing_start_ids), warmup=warmup_s * 1e3, rep=rep_s * 1e3)
    print(f"Prefill time: {optimized_prefill_time} seconds (optimized)")
    prefill_speedup = baseline_prefill_time / optimized_prefill_time
    print(f"Prefill speedup: {prefill_speedup}")

    baseline_end_to_end_time = do_bench(
        lambda: baseline_model.generate(timing_start_ids, max_new_tokens, temperature=temperature, top_k=top_k),
        warmup=warmup_s * 1e3, rep=rep_s * 1e3)
    print(f"Overall system generation time: {baseline_end_to_end_time} seconds (baseline)")
    optimized_end_to_end_time = do_bench(
        lambda: optimized_model.generate(timing_start_ids, max_new_tokens, temperature=temperature, top_k=top_k),
        warmup=warmup_s * 1e3, rep=rep_s * 1e3)
    print(f"Overall system generation time: {optimized_end_to_end_time} seconds (optimized)")
    end_to_end_speedup = baseline_end_to_end_time / optimized_end_to_end_time
    print(f"system_speedup: {end_to_end_speedup}")

def main():
    args = parse_args()
    correctness_prompts = [
        "Hi",
        "Hello, how are you?",
        "What is the capital of France? I thought it was Nice. I think that wrong.",
        "There once was a man from Nantucket. He went to the store to buy some peanuts. He bought a pound of peanuts.",
        "Explain the concept of quantum entanglement, but explain it in a way that is easy to understand...as if you were explaining it to a 5 year old.",
    ]
    timing_prompt = "What is the meaning of life, the universe, and everything?"

    ctx, _ = setup_environment(args.seed, args.device, args.dtype)
    # get_models should be called after setup_environment
    baseline_model, optimized_model = get_models(args.model_name, args.device)
    encode, decode = get_tokenizer()
    correctness_start_ids, timing_start_ids = prepare_inputs(encode, args.device, correctness_prompts, timing_prompt)

    with torch.no_grad():
        with ctx:
            check_correctness(baseline_model, optimized_model, correctness_start_ids, args.atol)
            measure_performance(
                baseline_model, optimized_model, timing_start_ids,
                args.max_new_tokens, args.temperature, args.top_k,
                args.warmup_s, args.rep_s
            )

if __name__ == "__main__":
    main()
