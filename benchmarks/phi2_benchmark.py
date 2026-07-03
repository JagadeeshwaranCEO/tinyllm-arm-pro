#!/usr/bin/env python3
"""
TinyLLM-ARM-Pro | Phi-2 (2.7B) Benchmark
═══════════════════════════════════════════════════════════
Same methodology as leaderboard.py — applied to Phi-2.
Proves our pipeline is model-agnostic, not a TinyLlama-only trick.
"""

import time
import psutil
import os
import json
from datetime import datetime
from llama_cpp import Llama

FP32_BASELINE = {
    "note": "FP32 baseline not run for Phi-2 due to 5.5GB size requiring "
            "significant time. F16 GGUF (5.6GB) serves as the reference point."
}

def get_ram_usage():
    return psutil.Process(os.getpid()).memory_info().rss / 1e9

def benchmark_model(model_path, model_name):
    print(f"\n⏳ Testing {model_name}...")

    if not os.path.exists(model_path):
        print(f"   ❌ Model not found: {model_path}")
        return None

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
        "What is ARM architecture and why does it matter for AI?",
        "Explain the difference between INT8 and FP32 quantization.",
        "Why is edge AI important for developing countries?"
    ]

    total_tokens = 0
    total_time = 0

    for prompt in prompts:
        start = time.time()
        output = model(prompt, max_tokens=80, echo=False)
        elapsed = time.time() - start
        total_tokens += output["usage"]["completion_tokens"]
        total_time += elapsed

    avg_speed = total_tokens / total_time
    del model

    result = {
        "name": model_name,
        "load_time": round(load_time, 3),
        "ram_gb": round(ram_used, 3),
        "tokens_per_sec": round(avg_speed, 2),
    }

    print(f"   ✅ Speed     : {avg_speed:.2f} tok/s")
    print(f"   ✅ RAM       : {ram_used:.3f} GB")
    print(f"   ✅ Load time : {load_time:.2f}s")

    return result

def run_phi2_benchmark():
    print("\n" + "="*60)
    print("TinyLLM-ARM-Pro | Phi-2 (2.7B) Benchmark")
    print("Apple Silicon ARM64 | llama.cpp + Metal GPU")
    print("Proves pipeline works on PhiForCausalLM architecture")
    print("="*60)

    models = [
        ("./models/phi2_quantized/phi2-q2k.gguf",  "Phi2-Q2_K"),
        ("./models/phi2_quantized/phi2-q4km.gguf", "Phi2-Q4_K_M"),
        ("./models/phi2_quantized/phi2-q8.gguf",   "Phi2-Q8_0"),
    ]

    results = []
    for path, name in models:
        result = benchmark_model(path, name)
        if result:
            results.append(result)

    if not results:
        print("❌ No models found. Run quantization first.")
        return

    print("\n" + "="*60)
    print("📊 PHI-2 RESULTS LEADERBOARD")
    print("="*60)
    print(f"{'Model':<15} {'Speed':>12} {'RAM':>8} {'Load':>8}")
    print("-"*60)

    best = max(results, key=lambda x: x["tokens_per_sec"])
    for r in results:
        marker = " 👑" if r["name"] == best["name"] else ""
        print(f"{r['name']:<15} {r['tokens_per_sec']:>9.2f} tok/s "
              f"{r['ram_gb']:>6.2f}GB {r['load_time']:>7.2f}s{marker}")

    print("="*60)
    print(f"\n🏆 Best: {best['name']} at {best['tokens_per_sec']:.2f} tok/s")

    # Save JSON
    os.makedirs("results", exist_ok=True)
    output = {
        "generated_at": datetime.now().isoformat(),
        "model_family": "Phi-2 (2.7B, PhiForCausalLM)",
        "note": "Same pipeline as TinyLlama — proves model-agnostic quantization",
        "best_model": best["name"],
        "results": results,
    }

    with open("results/phi2_benchmark.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"✅ Saved: results/phi2_benchmark.json\n")

    # Append to history
    with open("results/history.jsonl", "a") as f:
        f.write(json.dumps(output) + "\n")
    print(f"✅ Appended to: results/history.jsonl\n")

if __name__ == "__main__":
    run_phi2_benchmark()