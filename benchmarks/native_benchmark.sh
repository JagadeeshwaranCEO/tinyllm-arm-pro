#!/bin/bash
# TinyLLM-ARM-Pro | Native llama-bench Benchmark
# Industry-standard benchmark using llama.cpp's built-in tool

set -e

LLAMA_BENCH=~/llama.cpp/build/bin/llama-bench
OWN_Q4=~/tinyllm-arm-pro/tinyllm-arm-pro/models/own_quantized/tinyllama-own-q4km.gguf
OWN_Q8=~/tinyllm-arm-pro/tinyllm-arm-pro/models/own_quantized/tinyllama-own-q8.gguf
OWN_Q2=~/tinyllm-arm-pro/tinyllm-arm-pro/models/own_quantized/tinyllama-own-q2k.gguf

echo "=================================================="
echo "TinyLLM-ARM-Pro | Native llama-bench Benchmark"
echo "Apple Silicon ARM64 | Metal GPU + Flash Attention"
echo "=================================================="

echo ""
echo "▶ Benchmarking OUR Q4_K_M..."
$LLAMA_BENCH -m $OWN_Q4 \
  -ngl 99 -fa 1 \
  -b 2048 -ub 2048 \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  -p 512 -n 128 -r 5

echo ""
echo "▶ Benchmarking OUR Q8_0..."
$LLAMA_BENCH -m $OWN_Q8 \
  -ngl 99 -fa 1 \
  -b 2048 -ub 2048 \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  -p 512 -n 128 -r 5

echo ""
echo "▶ Benchmarking OUR Q2_K..."
$LLAMA_BENCH -m $OWN_Q2 \
  -ngl 99 -fa 1 \
  -b 2048 -ub 2048 \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  -p 512 -n 128 -r 5

echo ""
echo "=================================================="
echo "✅ Native benchmark complete."
echo "=================================================="