import torch
import psutil
import time
import os
from transformers import AutoTokenizer, AutoModelForCausalLM

# ── System Info ──────────────────────────────────────────
def print_system_info():
    print("\n" + "="*50)
    print("TinyLLM-ARM-Pro | Baseline Benchmark")
    print("="*50)
    print(f"Device        : Apple Silicon (ARM64)")
    print(f"PyTorch       : {torch.__version__}")
    print(f"MPS Available : {torch.backends.mps.is_available()}")
    print(f"RAM Total     : {psutil.virtual_memory().total / 1e9:.1f} GB")
    print(f"RAM Available : {psutil.virtual_memory().available / 1e9:.1f} GB")
    print("="*50 + "\n")

# ── Memory Usage ─────────────────────────────────────────
def get_ram_usage():
    # RSS alone misses MPS shared memory (GPU allocations live outside
    # the process heap), so include torch's MPS allocator explicitly.
    process = psutil.Process(os.getpid())
    rss = process.memory_info().rss / 1e9
    mps = 0.0
    if torch.backends.mps.is_available():
        try:
            mps = torch.mps.current_allocated_memory() / 1e9
        except Exception:
            mps = 0.0
    return rss + mps

# ── Benchmark ────────────────────────────────────────────
def run_benchmark():
    print_system_info()

    model_name = "./models/tinyllama"
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    print(f"Loading model: {model_name}")
    print(f"Running on   : {device.upper()}\n")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Measure RAM before model load
    ram_before = get_ram_usage()

    # Load model — explicit dtype + explicit .to(device).
    # NOTE: device_map="auto" does not place reliably on MPS
    # (silently falls back to CPU), and torch_dtype is deprecated
    # in transformers 5.x. Both were breaking FP32 baseline runs.
    load_start = time.time()
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32)
    model.to(device)
    model.eval()
    load_time = time.time() - load_start
    ram_after = get_ram_usage()

    print(f"✅ Model loaded in  : {load_time:.2f}s")
    print(f"📦 RAM used by model: {ram_after - ram_before:.2f} GB (RSS + MPS)\n")

    # Inference benchmark
    prompts = [
        "What is ARM architecture?",
        "Explain quantization in simple terms.",
        "Why is edge AI important?"
    ]

    print("Running inference benchmark...")
    print("-" * 50)

    for prompt in prompts:
        # Chat model — format with the model's chat template. Raw prompts
        # cause the greedy decoder to emit </s> immediately (EOS prediction
        # on an untemplated turn), which was misread as an MPS bug.
        chat = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
        )
        inputs = tokenizer(chat, return_tensors="pt").to(device)
        input_tokens = inputs["input_ids"].shape[1]

        start = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False
            )
        elapsed = time.time() - start

        output_tokens = outputs.shape[1] - input_tokens
        tokens_per_sec = output_tokens / elapsed

        print(f"\nPrompt : {prompt}")
        print(f"Output : {tokenizer.decode(outputs[0], skip_special_tokens=True)}")
        print(f"Tokens : {output_tokens} tokens in {elapsed:.2f}s")
        print(f"Speed  : {tokens_per_sec:.2f} tokens/sec")
        print("-" * 50)

    print("\n✅ Baseline benchmark complete.")
    print("These are your FP32 numbers — everything we optimize will beat this.\n")

if __name__ == "__main__":
    run_benchmark()