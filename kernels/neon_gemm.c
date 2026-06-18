// TinyLLM-ARM-Pro | NEON-Optimized Matrix Multiply
// Hand-tuned using ARM NEON SIMD intrinsics
// Processes 4 floats per instruction instead of 1

#include <arm_neon.h>
#include <stdio.h>
#include <stdlib.h>

void neon_matmul(const float *A, const float *B, float *C, int M, int N, int K) {
    // C[M x N] = A[M x K] * B[K x N]
    // NEON processes 4 float32 values per SIMD register (128-bit)

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j += 4) {
            // Accumulator for 4 output values at once
            float32x4_t sum_vec = vdupq_n_f32(0.0f);

            for (int k = 0; k < K; k++) {
                // Broadcast single A value across all 4 lanes
                float32x4_t a_val = vdupq_n_f32(A[i * K + k]);

                // Load 4 consecutive B values
                float32x4_t b_vec = vld1q_f32(&B[k * N + j]);

                // Multiply-accumulate: sum += a * b (fused, single instruction)
                sum_vec = vmlaq_f32(sum_vec, a_val, b_vec);
            }

            // Store 4 computed results back to C
            vst1q_f32(&C[i * N + j], sum_vec);
        }
    }
}