// TinyLLM-ARM-Pro | NEON-Optimized Matrix Multiply v2
// Multiple accumulators + 4-row blocking to break FMA dependency chains
// and improve cache reuse on Apple Silicon NEON pipeline

#include <arm_neon.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void neon_matmul_v2(const float *A, const float *B, float *C, int M, int N, int K) {
    // Process 4 rows of A at a time -> reuses each loaded B vector 4x
    // before it leaves registers/cache. Each row gets its own
    // accumulator -> breaks the single dependency chain from v1.

    int i = 0;
    for (; i + 4 <= M; i += 4) {
        for (int j = 0; j < N; j += 4) {
            // 4 independent accumulators -- one per row of A
            float32x4_t sum0 = vdupq_n_f32(0.0f);
            float32x4_t sum1 = vdupq_n_f32(0.0f);
            float32x4_t sum2 = vdupq_n_f32(0.0f);
            float32x4_t sum3 = vdupq_n_f32(0.0f);

            const float *a_row0 = &A[(i + 0) * K];
            const float *a_row1 = &A[(i + 1) * K];
            const float *a_row2 = &A[(i + 2) * K];
            const float *a_row3 = &A[(i + 3) * K];

            for (int k = 0; k < K; k++) {
                // Load B vector ONCE, reuse across all 4 rows
                float32x4_t b_vec = vld1q_f32(&B[k * N + j]);

                float32x4_t a0 = vdupq_n_f32(a_row0[k]);
                float32x4_t a1 = vdupq_n_f32(a_row1[k]);
                float32x4_t a2 = vdupq_n_f32(a_row2[k]);
                float32x4_t a3 = vdupq_n_f32(a_row3[k]);

                // 4 independent FMA chains -- no waiting on each other,
                // CPU can pipeline these back to back
                sum0 = vmlaq_f32(sum0, a0, b_vec);
                sum1 = vmlaq_f32(sum1, a1, b_vec);
                sum2 = vmlaq_f32(sum2, a2, b_vec);
                sum3 = vmlaq_f32(sum3, a3, b_vec);
            }

            vst1q_f32(&C[(i + 0) * N + j], sum0);
            vst1q_f32(&C[(i + 1) * N + j], sum1);
            vst1q_f32(&C[(i + 2) * N + j], sum2);
            vst1q_f32(&C[(i + 3) * N + j], sum3);
        }
    }

    // Handle remaining rows (M not divisible by 4) with v1-style fallback
    for (; i < M; i++) {
        for (int j = 0; j < N; j += 4) {
            float32x4_t sum_vec = vdupq_n_f32(0.0f);
            for (int k = 0; k < K; k++) {
                float32x4_t a_val = vdupq_n_f32(A[i * K + k]);
                float32x4_t b_vec = vld1q_f32(&B[k * N + j]);
                sum_vec = vmlaq_f32(sum_vec, a_val, b_vec);
            }
            vst1q_f32(&C[i * N + j], sum_vec);
        }
    }
}