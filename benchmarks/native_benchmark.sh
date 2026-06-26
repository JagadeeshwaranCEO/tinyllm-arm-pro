#!/bin/bash
# TinyLLM-ARM-Pro | Native llama-bench Benchmark
# Industry-standard benchmark using llama.cpp's built-in tool
#
# Paths are resolved relative to this script (not hardcoded to one
# user's home directory) so this works on any machine that clones
# the repo. Override llama.cpp location with: LLAMA_CPP_DIR=/path bash ...

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-$HOME/llama.cpp}"
LLAMA_BENCH="$LLAMA_CPP_DIR/build/bin/llama-bench"

MODELS_DIR="$PROJECT_ROOT/models/own_quantized"
RESULTS_DIR="$PROJECT_ROOT/results"
mkdir -p "$RESULTS_DIR"

if [ ! -x "$LLAMA_BENCH" ]; then
  echo "❌ llama-bench not found at: $LLAMA_BENCH"
  echo "   If llama.cpp is elsewhere, run:"
  echo "   LLAMA_CPP_DIR=/path/to/llama.cpp bash $0"
  exit 1
fi

echo "=================================================="
echo "TinyLLM-ARM-Pro | Native llama-bench Benchmark"
echo "Apple Silicon ARM64 | Metal GPU + Flash Attention"
echo "=================================================="

LABELS=("Q4_K_M" "Q8_0" "Q2_K")
FILES=(
  "$MODELS_DIR/tinyllama-own-q4km.gguf"
  "$MODELS_DIR/tinyllama-own-q8.gguf"
  "$MODELS_DIR/tinyllama-own-q2k.gguf"
)

for i in "${!LABELS[@]}"; do
  label="${LABELS[$i]}"
  model_path="${FILES[$i]}"
  log_path="$RESULTS_DIR/.tmp_native_${label}.txt"

  if [ ! -f "$model_path" ]; then
    echo ""
    echo "⚠️  Skipping $label — model not found: $model_path"
    : > "$log_path"
    continue
  fi

  echo ""
  echo "▶ Benchmarking OUR $label..."

  "$LLAMA_BENCH" -m "$model_path" \
    -ngl 99 -fa 1 \
    -b 2048 -ub 2048 \
    --cache-type-k q8_0 --cache-type-v q8_0 \
    -p 512 -n 128 -r 5 2>&1 | tee "$log_path"
done

echo ""
echo "=================================================="
echo "✅ Native benchmark complete."
echo "=================================================="

# Parsing free-text benchmark tables is fiddly — isolated into its
# own script so it's simple, testable, and doesn't clutter this file.
python3 "$SCRIPT_DIR/_parse_native_benchmark.py" \
  --results-dir "$RESULTS_DIR" \
  --labels "${LABELS[@]}"