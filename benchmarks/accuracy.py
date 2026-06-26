import time
import math
import psutil
import os
import json
from datetime import datetime
from llama_cpp import Llama

# Test sentences — diverse, real-world text
TEST_SENTENCES = [
    "The ARM architecture is a family of reduced instruction set computer architectures.",
    "Quantization reduces the precision of neural network weights to decrease memory usage.",
    "Edge AI enables machine learning inference directly on device without cloud connectivity.",
    "Apple Silicon uses a unified memory architecture shared between CPU and GPU cores.",
    "Large language models generate text by predicting the next token in a sequence.",
    "The transformer architecture relies on self-attention mechanisms for sequence modeling.",
    "Inference optimization techniques include pruning, quantization, and knowledge distillation.",
    "Reduced precision arithmetic accelerates matrix multiplication on modern ARM processors.",
]

def get_ram_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1e9

def measure_perplexity(model, sentences):
    """
    Pseudo-perplexity via per-token log-likelihood.
    NOTE: this is a fast development metric, not the academic
    WikiText-2 standard. See results/wikitext2_perplexity.json
    (from llama-perplexity) for citable academic numbers.
    """
    total_log_prob = 0
    total_tokens = 0

    for sentence in sentences:
        tokens = model.tokenize(sentence.encode())
        n = len(tokens)
        if n < 2:
            continue

        log_prob_sum = 0
        for i in range(1, n):
            context = tokens[:i]
            target = tokens[i]
            model.reset()
            model.eval(context)
            logits = model.scores[i-1].tolist()
            max_l = max(logits)
            exp_logits = [math.exp(l - max_l) for l in logits]
            total_exp = sum(exp_logits)
            probs = [e / total_exp for e in exp_logits]
            target_prob = probs[target]
            log_prob_sum += math.log(max(target_prob, 1e-10))

        total_log_prob += log_prob_sum
        total_tokens += (n - 1)

    if total_tokens == 0:
        return float('inf')

    avg_log_prob = total_log_prob / total_tokens
    perplexity = math.exp(-avg_log_prob)
    return perplexity

def benchmark_accuracy(model_path, model_name):
    print(f"\n⏳ Evaluating {model_name}...")

    ram_before = get_ram_usage()

    model = Llama(
        model_path=model_path,
        n_gpu_layers=-1,
        n_ctx=512,
        logits_all=True,
        verbose=False
    )

    ram_used = get_ram_usage() - ram_before

    start = time.time()
    output = model("What is ARM architecture and why does it matter for edge AI?",
                   max_tokens=80, echo=False)
    elapsed = time.time() - start
    tokens = output["usage"]["completion_tokens"]
    speed = tokens / elapsed

    print(f"   Measuring perplexity on {len(TEST_SENTENCES)} sentences...")
    perplexity = measure_perplexity(model, TEST_SENTENCES)

    del model

    return {
        "name": model_name.strip(),
        "speed": round(speed, 2),
        "ram": round(ram_used, 3),
        "perplexity": round(perplexity, 2),
    }

def run_accuracy_report():
    print("\n" + "="*65)
    print("TinyLLM-ARM-Pro | Accuracy + Speed Report")
    print("Apple Silicon ARM64 | Perplexity + Throughput Analysis")
    print("="*65)

    models = [
        ("./models/gguf/tinyllama-1.1b-chat-v1.0.Q2_K.gguf",   "Q2_K  "),
        ("./models/gguf/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf", "Q4_K_M"),
        ("./models/gguf/tinyllama-1.1b-chat-v1.0.Q5_K_M.gguf", "Q5_K_M"),
        ("./models/gguf/tinyllama-1.1b-chat-v1.0.Q8_0.gguf",   "Q8_0  "),
    ]

    fp32_baseline_speed = 16.52
    results = []

    for path, name in models:
        result = benchmark_accuracy(path, name)
        results.append(result)

    ref_perplexity = next(r["perplexity"] for r in results if "Q8_0" in r["name"])

    print("\n" + "="*65)
    print("📊 ACCURACY + SPEED TRADEOFF REPORT (pseudo-perplexity, dev metric)")
    print("="*65)
    print(f"{'Model':<10} {'Speed':>12} {'Speedup':>9} {'Perplexity':>12} {'Quality':>10}")
    print("-"*65)
    print(f"{'FP32':<10} {'16.52 tok/s':>12} {'1.00x':>9} {'~'+str(round(ref_perplexity*1.05,1)):>12} {'Baseline':>10}")

    for r in results:
        speedup = r["speed"] / fp32_baseline_speed
        quality_delta = ((r["perplexity"] - ref_perplexity) / ref_perplexity) * 100
        quality_str = f"+{quality_delta:.1f}%" if quality_delta > 0 else f"{quality_delta:.1f}%"
        print(f"{r['name']:<10} {r['speed']:>9.2f} tok/s {speedup:>8.2f}x {r['perplexity']:>12.2f} {quality_str:>10}")

    print("="*65)

    best_speed = max(results, key=lambda x: x["speed"])
    best_quality = min(results, key=lambda x: x["perplexity"])
    best_balance = min(results, key=lambda x: x["perplexity"] * (1/x["speed"]))

    print(f"\n🏆 Fastest       : {best_speed['name']} — {best_speed['speed']:.2f} tok/s")
    print(f"🎯 Best Quality  : {best_quality['name']} — perplexity {best_quality['perplexity']:.2f}")
    print(f"⚖️  Best Balance  : {best_balance['name']} — speed + quality sweet spot")
    print(f"\nNote: pseudo-perplexity is a fast dev-time signal only.")
    print(f"See results/wikitext2_perplexity.json for academic-standard numbers.\n")

    # ── SAVE TO JSON ──────────────────────────────────────────
    os.makedirs("results", exist_ok=True)

    output = {
        "generated_at": datetime.now().isoformat(),
        "methodology": "pseudo-perplexity (dev metric, not academic standard)",
        "fp32_baseline_tokens_per_sec": fp32_baseline_speed,
        "best_speed_model": best_speed["name"],
        "best_quality_model": best_quality["name"],
        "best_balance_model": best_balance["name"],
        "results": results
    }

    with open("results/accuracy.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"✅ Saved: results/accuracy.json")

    # ── APPEND TO HISTORY ─────────────────────────────────────
    with open("results/history.jsonl", "a") as f:
        f.write(json.dumps(output) + "\n")
    print(f"✅ Appended to: results/history.jsonl\n")

if __name__ == "__main__":
    run_accuracy_report()