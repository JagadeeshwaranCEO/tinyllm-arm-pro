// TinyLLM-ARM-Pro | Naive Matrix Multiply (Baseline)
// Standard scalar implementation — no SIMD optimization

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void naive_matmul(const float *A, const float *B, float *C, int M, int N, int K) {
    // C[M x N] = A[M x K] * B[K x N]
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}