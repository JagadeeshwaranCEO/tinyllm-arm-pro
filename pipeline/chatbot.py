#!/usr/bin/env python3
"""
TinyLLM-ARM-Pro | Interactive Chatbot Demo
Uses the Adaptive Inference Planner to auto-select
the best quantization for your device, then starts an
interactive conversation loop.

Usage:
    python pipeline/chatbot.py
    python pipeline/chatbot.py --model phi2
"""

import sys
import os
import argparse
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.planner import load_benchmark_data, recommend, format_explanation
import psutil


def find_model_file(model, quant):
    candidates = [
        ROOT / "models" / "gguf" / f"tinyllama-1.1b-chat-v1.0.{quant}.gguf",
        ROOT / "models" / "gguf_ref2" / f"TinyLlama-1.1B-Chat-v1.0-{quant}.gguf",
        ROOT / "models" / "own_quantized" / f"tinyllama-own-q{quant.lower().replace('_', '').replace('k', 'k')}.gguf",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return None


def find_phi2_model_file(quant):
    name_map = {
        "Phi2-Q4_K_M": "phi2-q4km.gguf",
        "Phi2-Q8_0": "phi2-q8.gguf",
        "Phi2-Q2_K": "phi2-q2k.gguf",
    }
    name = name_map.get(quant)
    if not name:
        return None
    path = ROOT / "models" / "phi2_quantized" / name
    return str(path) if path.exists() else None


def chat_loop(llm, quant_info):
    print("\n" + "=" * 60)
    print(" TinyLLM-ARM-Pro Chatbot Demo")
    print(f" Model loaded: {quant_info}")
    print(" Type 'quit' to exit, 'clear' to reset context")
    print("=" * 60)

    messages = []
    while True:
        try:
            user = input("\n You: ")
        except (EOFError, KeyboardInterrupt):
            print("\n")
            break

        if user.lower() in ("quit", "exit"):
            break
        if user.lower() == "clear":
            messages.clear()
            print(" Context cleared.")
            continue

        messages.append({"role": "user", "content": user})
        prompt = llm.tokenizer.apply_chat_template(messages, tokenize=False)

        print(" Bot: ", end="", flush=True)
        output = llm(
            prompt,
            max_tokens=256,
            temperature=0.7,
            top_p=0.9,
            stop=["</s>", "User:", "user:"],
            echo=False,
        )
        response = output["choices"][0]["text"].strip()
        print(response)
        messages.append({"role": "assistant", "content": response})


def main():
    parser = argparse.ArgumentParser(
        description="TinyLLM-ARM-Pro Interactive Chatbot"
    )
    parser.add_argument("--model", default="tinyllama",
                        choices=["tinyllama", "phi2"])
    parser.add_argument("--quant", default=None,
                        help="Force a specific quantization (skip planner)")
    parser.add_argument("--list-models", action="store_true",
                        help="Show available models and exit")
    args = parser.parse_args()

    if args.list_models:
        print("Available GGUF models in ./models/gguf/:")
        gguf_dir = ROOT / "models" / "gguf"
        if gguf_dir.exists():
            for f in sorted(gguf_dir.iterdir()):
                if f.suffix == ".gguf":
                    print(f"  {f.name}")
        else:
            print("  No models found. Download first.")
        return

    ram = psutil.virtual_memory().total / 1e9

    if args.quant:
        quant = args.quant
        if args.model == "phi2":
            model_path = find_phi2_model_file(quant)
        else:
            model_path = find_model_file(args.model, quant)
        if not model_path:
            print(f"Model file for {quant} not found.")
            return
        quant_info = f"{quant} (manual)"
        print(f"Using manual quant: {quant}")
    else:
        print(f"Detected RAM: {ram:.1f}GB")
        print(f"Loading benchmark data for {args.model}...")
        try:
            data = load_benchmark_data(model=args.model)
        except FileNotFoundError:
            print("No benchmark data found. Run benchmarks first, or use --quant.")
            return

        plan = recommend(total_ram_gb=ram, benchmark_data=data)
        print(format_explanation(plan))
        rec = plan["recommendation"]
        quant = rec["name"]
        quant_info = f"{quant} (auto-selected by planner)"

        if args.model == "phi2":
            model_path = find_phi2_model_file(quant)
        else:
            model_path = find_model_file(args.model, quant)

        if not model_path:
            print(f"Model file for recommended {quant} not found.")
            return

    print(f"\nLoading {quant_info}...")
    from llama_cpp import Llama
    llm = Llama(
        model_path=model_path,
        n_gpu_layers=-1,
        n_ctx=2048,
        verbose=False,
    )
    print(f" Loaded: {model_path}")
    print(f" GPU: Metal (all layers offloaded)")

    chat_loop(llm, quant_info)
    del llm
    print("Goodbye.")


if __name__ == "__main__":
    main()
