// TinyLLM-ARM-Pro | NEON vs Naive Benchmark
// Measures real speedup from hand-tuned NEON intrinsics

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>

extern void naive_matmul(const float *A, const float *B, float *C, int M, int N, int K);
extern void neon_matmul(const float *A, const float *B, float *C, int M, int N, int K);
extern void neon_matmul_v2(const float *A, const float *B, float *C, int M, int N, int K);

void fill_random(float *arr, int size) {
    for (int i = 0; i < size; i++) {
        arr[i] = (float)(rand() % 100) / 10.0f;
    }
}

double get_time_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

int verify_match(float *a, float *b, int size, float tolerance) {
    for (int i = 0; i < size; i++) {
        if (fabsf(a[i] - b[i]) > tolerance) return 0;
    }
    return 1;
}

void run_benchmark(int M, int N, int K, int iterations) {
    printf("\n========================================\n");
    printf("Matrix size: %dx%d * %dx%d\n", M, K, K, N);
    printf("========================================\n");

    float *A = malloc(M * K * sizeof(float));
    float *B = malloc(K * N * sizeof(float));
    float *C_naive = malloc(M * N * sizeof(float));
    float *C_neon = malloc(M * N * sizeof(float));
    float *C_neon_v2 = malloc(M * N * sizeof(float));

    fill_random(A, M * K);
    fill_random(B, K * N);

    // Warmup
    naive_matmul(A, B, C_naive, M, N, K);
    neon_matmul(A, B, C_neon, M, N, K);
    neon_matmul_v2(A, B, C_neon_v2, M, N, K);

    // Verify correctness
    int match_v1 = verify_match(C_naive, C_neon, M * N, 0.01f);
    int match_v2 = verify_match(C_naive, C_neon_v2, M * N, 0.01f);
    printf("NEON v1 correctness: %s\n", match_v1 ? "PASSED ✅" : "FAILED ❌");
    printf("NEON v2 correctness: %s\n", match_v2 ? "PASSED ✅" : "FAILED ❌");

    // Benchmark naive
    double start = get_time_ms();
    for (int iter = 0; iter < iterations; iter++) {
        naive_matmul(A, B, C_naive, M, N, K);
    }
    double naive_time = (get_time_ms() - start) / iterations;

    // Benchmark NEON v1
    start = get_time_ms();
    for (int iter = 0; iter < iterations; iter++) {
        neon_matmul(A, B, C_neon, M, N, K);
    }
    double neon_time = (get_time_ms() - start) / iterations;

    // Benchmark NEON v2
    start = get_time_ms();
    for (int iter = 0; iter < iterations; iter++) {
        neon_matmul_v2(A, B, C_neon_v2, M, N, K);
    }
    double neon_v2_time = (get_time_ms() - start) / iterations;

    printf("\nNaive (scalar)     : %.4f ms\n", naive_time);
    printf("NEON v1 (basic)    : %.4f ms  -> %.2fx speedup\n", neon_time, naive_time / neon_time);
    printf("NEON v2 (blocked)  : %.4f ms  -> %.2fx speedup\n", neon_v2_time, naive_time / neon_v2_time);
    printf("v2 vs v1 improvement: %.2fx\n", neon_time / neon_v2_time);

    free(A); free(B); free(C_naive); free(C_neon); free(C_neon_v2);

}

int main() {
    printf("\n");
    printf("==================================================\n");
    printf("TinyLLM-ARM-Pro | NEON Intrinsics Benchmark\n");
    printf("Hand-tuned SIMD matrix multiply vs naive scalar\n");
    printf("==================================================\n");

    srand(42);

    // Sizes relevant to transformer matrix multiplies
    run_benchmark(64, 64, 64, 1000);
    run_benchmark(256, 256, 256, 100);
    run_benchmark(512, 512, 512, 20);
    run_benchmark(2048, 2048, 64, 10);  // TinyLlama-like dimensions

    printf("\n==================================================\n");
    printf("✅ NEON benchmark complete.\n");
    printf("==================================================\n\n");

    return 0;
}