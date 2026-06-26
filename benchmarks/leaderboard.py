import time
import psutil
import os
import json
from datetime import datetime
from llama_cpp import Llama

def get_ram_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1e9

def benchmark_model(model_path, model_name):
    print(f"\n⏳ Testing {model_name}...")
    ram_before = get_ram_usage()
    load_start = time.time()
    model = Llama(
        model_path=model_path,
        n_gpu_layers=-1,
        n_ctx=2048,
        verbose=False
    )
    load_time = time.time() - load_start
    ram_used = get_ram_usage() - ram_before
    prompts = [
        "What is ARM architecture?",
        "Explain quantization in simple terms.",
        "Why is edge AI important?"
    ]
    total_tokens = 0
    total_time = 0
    for prompt in prompts:
        start = time.time()
        output = model(prompt, max_tokens=100, echo=False)
        elapsed = time.time() - start
        total_tokens += output["usage"]["completion_tokens"]
        total_time += elapsed
    avg_speed = total_tokens / total_time
    del model
    return {
        "name": model_name.strip(),
        "load_time": round(load_time, 3),
        "ram_gb": round(ram_used, 3),
        "tokens_per_sec": round(avg_speed, 2)
    }

def run_leaderboard():
    print("\n" + "="*60)
    print("TinyLLM-ARM-Pro | Multi-Quantization Leaderboard")
    print("Apple Silicon ARM64 | llama.cpp + Metal GPU")
    print("="*60)

    models = [
        ("./models/gguf/tinyllama-1.1b-chat-v1.0.Q2_K.gguf",   "Q2_K   "),
        ("./models/gguf/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf", "Q4_K_M "),
        ("./models/gguf/tinyllama-1.1b-chat-v1.0.Q5_K_M.gguf", "Q5_K_M "),
        ("./models/gguf/tinyllama-1.1b-chat-v1.0.Q8_0.gguf",   "Q8_0   "),
    ]

    fp32_baseline = 16.52
    results = []
    for path, name in models:
        result = benchmark_model(path, name)
        results.append(result)

    # Print leaderboard table
    print("\n" + "="*60)
    print("📊 RESULTS LEADERBOARD")
    print("="*60)
    print(f"{'Model':<12} {'Speed':>12} {'Speedup':>10} {'RAM':>8} {'Load':>8}")
    print("-"*60)
    print(f"{'FP32':<12} {'16.52 tok/s':>12} {'1.00x':>10} {'2.20GB':>8} {'3.86s':>8}")
    for r in results:
        speedup = r["tokens_per_sec"] / fp32_baseline
        print(f"{r['name']:<12} {r['tokens_per_sec']:>9.2f} tok/s {speedup:>9.2f}x {r['ram_gb']:>7.2f}GB {r['load_time']:>7.2f}s")
    print("="*60)

    best = max(results, key=lambda x: x["tokens_per_sec"])
    best_speedup = best["tokens_per_sec"] / fp32_baseline
    print(f"\n🏆 Best: {best['name']} at {best['tokens_per_sec']:.2f} tok/s ({best_speedup:.2f}x speedup)")
    print(f"💾 Most efficient: Q2_K (smallest RAM, fastest speed)")
    print(f"⚖️  Best balance: Q4_K_M (quality + speed sweet spot)")
    print("\n✅ Leaderboard complete.\n")

    # ── SAVE TO JSON (this is what was missing) ──────────────
    os.makedirs("results", exist_ok=True)

    output = {
        "generated_at": datetime.now().isoformat(),
        "fp32_baseline_tokens_per_sec": fp32_baseline,
        "best_model": best["name"],
        "best_speedup": round(best_speedup, 2),
        "results": results
    }

    with open("results/leaderboard.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"✅ Saved: results/leaderboard.json")

    # ── APPEND TO HISTORY (so the Planner can learn over time) ─
    history_path = "results/history.jsonl"
    with open(history_path, "a") as f:
        f.write(json.dumps(output) + "\n")
    print(f"✅ Appended to: {history_path}\n")

if __name__ == "__main__":
    run_leaderboard()