#!/bin/bash
# TinyLLM-ARM-Pro | Compile NEON Benchmark
# Compiles C kernels with ARM NEON support for native execution

set -e

echo "=================================================="
echo "Compiling NEON kernels for ARM64..."
echo "=================================================="

clang -O3 -march=native \
    kernels/naive_gemm.c \
    kernels/neon_gemm.c \
    kernels/neon_gemm_v2.c \
    kernels/benchmark_neon.c \
    -o kernels/neon_benchmark \
    -lm

echo "✅ Compiled: kernels/neon_benchmark"
echo ""
echo "Running benchmark..."
echo "=================================================="

./kernels/neon_benchmark