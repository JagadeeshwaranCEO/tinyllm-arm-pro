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
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1e9  # GB

# ── Benchmark ────────────────────────────────────────────
def run_benchmark():
    print_system_info()

    model_name = "./models/tinyllama"
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    print(f"Loading model: {model_name}")
    print(f"Running on   : {device.upper()}\n")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Measure RAM before model load
    ram_before = get_ram_usage()

    # Load model
    load_start = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map="auto"
    )
    load_time = time.time() - load_start
    ram_after = get_ram_usage()

    print(f"✅ Model loaded in  : {load_time:.2f}s")
    print(f"📦 RAM used by model: {ram_after - ram_before:.2f} GB\n")

    # Inference benchmark
    prompts = [
        "What is ARM architecture?",
        "Explain quantization in simple terms.",
        "Why is edge AI important?"
    ]

    print("Running inference benchmark...")
    print("-" * 50)

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
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