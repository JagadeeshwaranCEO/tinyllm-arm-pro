#!/bin/bash
# TinyLLM-ARM-Pro | Own Quantization Pipeline
# Uses llama.cpp built from source to quantize FP32 → INT4/INT8

set -e

LLAMA_CPP=~/llama.cpp
MODEL_FP32=./models/tinyllama
OUTPUT_DIR=./models/own_quantized
QUANTIZE=$LLAMA_CPP/build/bin/llama-quantize

echo "=================================================="
echo "TinyLLM-ARM-Pro | Own Quantization Pipeline"
echo "ARM64 Apple Silicon | llama.cpp native build"
echo "=================================================="

mkdir -p $OUTPUT_DIR

# Step 1 — Convert FP32 to GGUF F16
echo ""
echo "Step 1: Converting FP32 model to GGUF F16..."
python3 $LLAMA_CPP/convert_hf_to_gguf.py \
    $MODEL_FP32 \
    --outfile $OUTPUT_DIR/tinyllama-f16.gguf \
    --outtype f16
echo "✅ F16 GGUF created"

# Step 2 — Quantize to Q4_K_M (our best performer)
echo ""
echo "Step 2: Quantizing F16 → Q4_K_M..."
$QUANTIZE \
    $OUTPUT_DIR/tinyllama-f16.gguf \
    $OUTPUT_DIR/tinyllama-own-q4km.gguf \
    Q4_K_M
echo "✅ Q4_K_M quantization complete"

# Step 3 — Quantize to Q8_0
echo ""
echo "Step 3: Quantizing F16 → Q8_0..."
$QUANTIZE \
    $OUTPUT_DIR/tinyllama-f16.gguf \
    $OUTPUT_DIR/tinyllama-own-q8.gguf \
    Q8_0
echo "✅ Q8_0 quantization complete"

# Step 4 — Quantize to Q2_K
echo ""
echo "Step 4: Quantizing F16 → Q2_K..."
$QUANTIZE \
    $OUTPUT_DIR/tinyllama-f16.gguf \
    $OUTPUT_DIR/tinyllama-own-q2k.gguf \
    Q2_K
echo "✅ Q2_K quantization complete"

# Step 5 — Show file sizes
echo ""
echo "=================================================="
echo "📦 OUR QUANTIZED MODEL SIZES"
echo "=================================================="
ls -lh $OUTPUT_DIR/*.gguf | awk '{print $5, $9}'

echo ""
echo "✅ Own quantization pipeline complete."
echo "These models were quantized by us — not downloaded."
echo "=================================================="