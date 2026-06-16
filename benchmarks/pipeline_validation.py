import time
import math
import psutil
import os
import json
from llama_cpp import Llama

# ── Test Suite ───────────────────────────────────────────
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

INFERENCE_PROMPTS = [
    "What is ARM architecture?",
    "Explain quantization in simple terms.",
    "Why is edge AI important for developing countries?",
]

def get_ram_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1e9

def measure_perplexity(model, sentences):
    total_log_prob = 0
    total_tokens = 0
    for sentence in sentences:
        tokens = model.tokenize(sentence.encode())
        n = len(tokens)
        if n < 2:
            continue
        log_prob_sum = 0
        for i in range(1, min(n, 20)):
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
        total_tokens += (min(n, 20) - 1)
    if total_tokens == 0:
        return float('inf')
    return math.exp(-total_log_prob / total_tokens)

def benchmark_model(model_path, model_name, source):
    print(f"\n{'='*55}")
    print(f"Testing : {model_name}")
    print(f"Source  : {source}")
    print(f"Path    : {model_path}")
    print(f"{'='*55}")

    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return None

    ram_before = get_ram_usage()
    load_start = time.time()

    model = Llama(
        model_path=model_path,
        n_gpu_layers=-1,
        n_ctx=512,
        logits_all=True,
        verbose=False
    )

    load_time = time.time() - load_start
    ram_used = get_ram_usage() - ram_before

    # Speed benchmark
    total_tokens = 0
    total_time = 0
    for prompt in INFERENCE_PROMPTS:
        start = time.time()
        output = model(prompt, max_tokens=80, echo=False)
        elapsed = time.time() - start
        total_tokens += output["usage"]["completion_tokens"]
        total_time += elapsed

    avg_speed = total_tokens / total_time

    # Perplexity
    print(f"   Measuring perplexity...")
    perplexity = measure_perplexity(model, TEST_SENTENCES)

    # Sample output
    sample = model(
        "What is the benefit of running AI on ARM devices?",
        max_tokens=60,
        echo=False
    )
    sample_text = sample["choices"][0]["text"].strip()

    del model

    result = {
        "name": model_name,
        "source": source,
        "load_time": round(load_time, 2),
        "ram_gb": round(ram_used, 2),
        "tokens_per_sec": round(avg_speed, 2),
        "perplexity": round(perplexity, 2),
        "sample_output": sample_text,
        "model_size_mb": round(os.path.getsize(model_path) / 1e6, 1)
    }

    print(f"   ✅ Speed      : {avg_speed:.2f} tok/s")
    print(f"   ✅ Perplexity : {perplexity:.2f}")
    print(f"   ✅ RAM        : {ram_used:.2f} GB")
    print(f"   ✅ Load time  : {load_time:.2f}s")

    return result

def run_validation():
    print("\n" + "="*55)
    print("TinyLLM-ARM-Pro | Pipeline Validation Report")
    print("OUR Models vs Reference (TheBloke) Models")
    print("Apple Silicon ARM64 | llama.cpp + Metal GPU")
    print("="*55)

    models = [
        # Our quantized models
        (
            "./models/own_quantized/tinyllama-own-q4km.gguf",
            "Q4_K_M (OURS)",
            "Our Pipeline"
        ),
        (
            "./models/own_quantized/tinyllama-own-q8.gguf",
            "Q8_0 (OURS)",
            "Our Pipeline"
        ),
        (
            "./models/own_quantized/tinyllama-own-q2k.gguf",
            "Q2_K (OURS)",
            "Our Pipeline"
        ),
        # Reference models
        (
            "./models/gguf/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
            "Q4_K_M (REF)",
            "TheBloke Reference"
        ),
        (
            "./models/gguf/tinyllama-1.1b-chat-v1.0.Q8_0.gguf",
            "Q8_0 (REF)",
            "TheBloke Reference"
        ),
        (
            "./models/gguf/tinyllama-1.1b-chat-v1.0.Q2_K.gguf",
            "Q2_K (REF)",
            "TheBloke Reference"
        ),
    ]

    results = []
    for path, name, source in models:
        result = benchmark_model(path, name, source)
        if result:
            results.append(result)

    # Save results to JSON
    os.makedirs("results", exist_ok=True)
    with open("results/pipeline_validation.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Results saved to results/pipeline_validation.json")

    # Print comparison table
    print("\n" + "="*75)
    print("📊 PIPELINE VALIDATION — OUR vs REFERENCE")
    print("="*75)
    print(f"{'Model':<18} {'Source':<22} {'Speed':>10} {'Perplexity':>12} {'RAM':>7}")
    print("-"*75)

    for r in results:
        print(f"{r['name']:<18} {r['source']:<22} "
              f"{r['tokens_per_sec']:>8.2f} t/s "
              f"{r['perplexity']:>12.2f} "
              f"{r['ram_gb']:>6.2f}GB")

    print("="*75)

    # Compare our vs reference
    print("\n📊 QUALITY COMPARISON — OUR PIPELINE vs REFERENCE")
    print("-"*55)

    quant_levels = ["Q4_K_M", "Q8_0", "Q2_K"]
    for q in quant_levels:
        ours = next((r for r in results if q in r["name"] and r["source"] == "Our Pipeline"), None)
        ref = next((r for r in results if q in r["name"] and r["source"] == "TheBloke Reference"), None)
        if ours and ref:
            speed_diff = ((ours["tokens_per_sec"] - ref["tokens_per_sec"]) / ref["tokens_per_sec"]) * 100
            ppl_diff = ((ours["perplexity"] - ref["perplexity"]) / ref["perplexity"]) * 100
            print(f"\n{q}:")
            print(f"  Speed      — Ours: {ours['tokens_per_sec']:.2f} | Ref: {ref['tokens_per_sec']:.2f} | Diff: {speed_diff:+.1f}%")
            print(f"  Perplexity — Ours: {ours['perplexity']:.2f} | Ref: {ref['perplexity']:.2f} | Diff: {ppl_diff:+.1f}%")

    print("\n" + "="*55)
    print("✅ Pipeline validation complete.")
    print("Results saved to results/pipeline_validation.json")
    print("="*55 + "\n")

if __name__ == "__main__":
    run_validation()