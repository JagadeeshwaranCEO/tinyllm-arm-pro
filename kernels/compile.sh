#!/bin/bash
set -e

echo "=================================================="
echo "Compiling NEON kernels for ARM64..."
echo "=================================================="

# FP32 NEON kernels (v1 + v2)
clang -O3 -march=native \
    kernels/naive_gemm.c \
    kernels/neon_gemm.c \
    kernels/neon_gemm_v2.c \
    kernels/benchmark_neon.c \
    -o kernels/neon_benchmark \
    -lm

echo "✅ Compiled: kernels/neon_benchmark"

# INT8 I8MM kernel
clang -O3 -march=native+i8mm \
    kernels/neon_i8mm.c \
    -o kernels/i8mm_benchmark \
    -lm

echo "✅ Compiled: kernels/i8mm_benchmark"
echo ""
echo "Running FP32 NEON benchmark..."
echo "=================================================="
./kernels/neon_benchmark

echo ""
echo "Running INT8 I8MM benchmark..."
echo "=================================================="
./kernels/i8mm_benchmark