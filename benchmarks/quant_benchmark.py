import time
import psutil
import os
from llama_cpp import Llama

# ── Memory Usage ─────────────────────────────────────────
def get_ram_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1e9

# ── Benchmark ────────────────────────────────────────────
def run_quant_benchmark():
    print("\n" + "="*50)
    print("TinyLLM-ARM-Pro | Quantization Benchmark")
    print("="*50)
    print(f"Model    : TinyLlama 1.1B Q4_K_M (GGUF)")
    print(f"Backend  : llama.cpp + Apple Metal (ARM64)")
    print(f"RAM Total: {psutil.virtual_memory().total / 1e9:.1f} GB")
    print("="*50 + "\n")

    model_path = "./models/gguf/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"

    ram_before = get_ram_usage()
    load_start = time.time()

    # Load with Metal GPU acceleration
    model = Llama(
        model_path=model_path,
        n_gpu_layers=-1,  # offload all layers to Metal GPU
        n_ctx=2048,
        verbose=False
    )

    load_time = time.time() - load_start
    ram_after = get_ram_usage()

    print(f"✅ Model loaded in  : {load_time:.2f}s")
    print(f"📦 RAM used by model: {ram_after - ram_before:.2f} GB\n")

    prompts = [
        "What is ARM architecture?",
        "Explain quantization in simple terms.",
        "Why is edge AI important?"
    ]

    print("Running inference benchmark...")
    print("-" * 50)

    total_tokens = 0
    total_time = 0

    for prompt in prompts:
        start = time.time()
        output = model(
            prompt,
            max_tokens=100,
            echo=False
        )
        elapsed = time.time() - start

        tokens = output["usage"]["completion_tokens"]
        speed = tokens / elapsed
        total_tokens += tokens
        total_time += elapsed

        print(f"\nPrompt : {prompt}")
        print(f"Output : {output['choices'][0]['text'].strip()}")
        print(f"Tokens : {tokens} tokens in {elapsed:.2f}s")
        print(f"Speed  : {speed:.2f} tokens/sec")
        print("-" * 50)

    avg_speed = total_tokens / total_time
    print(f"\n📊 Average Speed     : {avg_speed:.2f} tokens/sec")
    print(f"📊 FP32 Baseline     : 16.52 tokens/sec")
    print(f"📊 Speedup           : {avg_speed / 16.52:.2f}x")
    print("\n✅ Quantization benchmark complete.\n")

if __name__ == "__main__":
    run_quant_benchmark()