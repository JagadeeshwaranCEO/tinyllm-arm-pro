// TinyLLM-ARM-Pro | NEON I8MM INT8 Matrix Multiply
// Uses ARMv8.6-A SMMLA (vmmlaq_s32) instruction
// This is the SAME instruction llama.cpp uses internally
// for quantized LLM inference on Apple Silicon M4
//
// Your M4 confirmed I8MM support during our cmake build:
// "Performing Test GGML_MACHINE_SUPPORTS_i8mm - Success"

#include <arm_neon.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

// ── Naive INT8 baseline ───────────────────────────────
void naive_i8_matmul(
    const int8_t *A, const int8_t *B,
    int32_t *C, int M, int N, int K)
{
    memset(C, 0, M * N * sizeof(int32_t));
    for (int i = 0; i < M; i++)
        for (int k = 0; k < K; k++)
            for (int j = 0; j < N; j++)
                C[i*N+j] += (int32_t)A[i*K+k] * (int32_t)B[k*N+j];
}

// ── I8MM SMMLA kernel ─────────────────────────────────
// vmmlaq_s32 processes a 2×8 block of INT8 A and
// an 8×2 block of INT8 B, producing a 2×2 INT32 result.
// That's 32 multiply-accumulates in a SINGLE instruction.
void i8mm_matmul(
    const int8_t *A, const int8_t *B,
    int32_t *C, int M, int N, int K)
{
    memset(C, 0, M * N * sizeof(int32_t));

    // Process 2 rows of A and 8 columns of B at a time
    for (int i = 0; i + 1 < M; i += 2) {
        for (int j = 0; j + 7 < N; j += 8) {
            // 4 accumulators cover a 2×8 output tile
            // Each int32x4_t holds a 2×2 result block
            int32x4_t acc0 = vdupq_n_s32(0);
            int32x4_t acc1 = vdupq_n_s32(0);
            int32x4_t acc2 = vdupq_n_s32(0);
            int32x4_t acc3 = vdupq_n_s32(0);

            for (int k = 0; k + 7 < K; k += 8) {
                // Load 2 rows of A, each 8 INT8 values
                // Interleaved into a single 16-byte register
                int8_t a_buf[16];
                memcpy(a_buf,     &A[(i+0)*K + k], 8);
                memcpy(a_buf + 8, &A[(i+1)*K + k], 8);
                int8x16_t a_vec = vld1q_s8(a_buf);

                // Process 4 pairs of B columns (4 × 2-col blocks)
                // Each b_vec is 8 rows × 2 cols of INT8
                int8_t b_buf[16];

                // Block 0: columns j+0, j+1
                for (int kk = 0; kk < 8; kk++) {
                    b_buf[kk]   = B[(k+kk)*N + j+0];
                    b_buf[kk+8] = B[(k+kk)*N + j+1];
                }
                acc0 = vmmlaq_s32(acc0, a_vec, vld1q_s8(b_buf));

                // Block 1: columns j+2, j+3
                for (int kk = 0; kk < 8; kk++) {
                    b_buf[kk]   = B[(k+kk)*N + j+2];
                    b_buf[kk+8] = B[(k+kk)*N + j+3];
                }
                acc1 = vmmlaq_s32(acc1, a_vec, vld1q_s8(b_buf));

                // Block 2: columns j+4, j+5
                for (int kk = 0; kk < 8; kk++) {
                    b_buf[kk]   = B[(k+kk)*N + j+4];
                    b_buf[kk+8] = B[(k+kk)*N + j+5];
                }
                acc2 = vmmlaq_s32(acc2, a_vec, vld1q_s8(b_buf));

                // Block 3: columns j+6, j+7
                for (int kk = 0; kk < 8; kk++) {
                    b_buf[kk]   = B[(k+kk)*N + j+6];
                    b_buf[kk+8] = B[(k+kk)*N + j+7];
                }
                acc3 = vmmlaq_s32(acc3, a_vec, vld1q_s8(b_buf));
            }

            // Store 2×8 output tile back to C
            // acc layout from vmmlaq_s32:
            //   lane 0: C[i][j+c0], lane 1: C[i][j+c1]
            //   lane 2: C[i+1][j+c0], lane 3: C[i+1][j+c1]
            C[(i+0)*N+j+0] += vgetq_lane_s32(acc0, 0);
            C[(i+0)*N+j+1] += vgetq_lane_s32(acc0, 1);
            C[(i+1)*N+j+0] += vgetq_lane_s32(acc0, 2);
            C[(i+1)*N+j+1] += vgetq_lane_s32(acc0, 3);

            C[(i+0)*N+j+2] += vgetq_lane_s32(acc1, 0);
            C[(i+0)*N+j+3] += vgetq_lane_s32(acc1, 1);
            C[(i+1)*N+j+2] += vgetq_lane_s32(acc1, 2);
            C[(i+1)*N+j+3] += vgetq_lane_s32(acc1, 3);

            C[(i+0)*N+j+4] += vgetq_lane_s32(acc2, 0);
            C[(i+0)*N+j+5] += vgetq_lane_s32(acc2, 1);
            C[(i+1)*N+j+4] += vgetq_lane_s32(acc2, 2);
            C[(i+1)*N+j+5] += vgetq_lane_s32(acc2, 3);

            C[(i+0)*N+j+6] += vgetq_lane_s32(acc3, 0);
            C[(i+0)*N+j+7] += vgetq_lane_s32(acc3, 1);
            C[(i+1)*N+j+6] += vgetq_lane_s32(acc3, 2);
            C[(i+1)*N+j+7] += vgetq_lane_s32(acc3, 3);
        }
    }
}

// ── Timer ─────────────────────────────────────────────
double get_time_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

// ── Correctness check ─────────────────────────────────
int verify(int32_t *ref, int32_t *test, int size) {
    for (int i = 0; i < size; i++)
        if (abs(ref[i] - test[i]) > 2) return 0;
    return 1;
}

// ── Benchmark ─────────────────────────────────────────
void run_i8mm_benchmark(int M, int N, int K, int iters) {
    printf("\n========================================\n");
    printf("Matrix: %dx%d * %dx%d  (INT8 inputs)\n", M, K, K, N);
    printf("========================================\n");

    int8_t  *A     = malloc(M * K);
    int8_t  *B     = malloc(K * N);
    int32_t *C_ref = calloc(M * N, sizeof(int32_t));
    int32_t *C_i8  = calloc(M * N, sizeof(int32_t));

    // Fill with small values to avoid int8 overflow
    for (int i = 0; i < M*K; i++) A[i] = (int8_t)((rand() % 20) - 10);
    for (int i = 0; i < K*N; i++) B[i] = (int8_t)((rand() % 20) - 10);

    // Reference
    naive_i8_matmul(A, B, C_ref, M, N, K);

    // I8MM
    i8mm_matmul(A, B, C_i8, M, N, K);

    int ok = verify(C_ref, C_i8, M*N);
    printf("Correctness: %s\n", ok ? "PASSED ✅" : "FAILED ❌");

    // Benchmark naive INT8
    double start = get_time_ms();
    for (int it = 0; it < iters; it++) {
        memset(C_ref, 0, M*N*sizeof(int32_t));
        naive_i8_matmul(A, B, C_ref, M, N, K);
    }
    double naive_t = (get_time_ms() - start) / iters;

    // Benchmark I8MM
    start = get_time_ms();
    for (int it = 0; it < iters; it++) {
        memset(C_i8, 0, M*N*sizeof(int32_t));
        i8mm_matmul(A, B, C_i8, M, N, K);
    }
    double i8mm_t = (get_time_ms() - start) / iters;

    printf("Naive INT8       : %.4f ms\n", naive_t);
    printf("I8MM (SMMLA)     : %.4f ms  ->  %.2fx speedup\n",
           i8mm_t, naive_t / i8mm_t);

    free(A); free(B); free(C_ref); free(C_i8);
}

int main() {
    printf("\n");
    printf("==================================================\n");
    printf("TinyLLM-ARM-Pro | I8MM (SMMLA) Benchmark\n");
    printf("ARMv8.6-A vmmlaq_s32 — hardware INT8 matrix mul\n");
    printf("Apple Silicon M4 | MTLGPUFamilyApple9\n");
    printf("==================================================\n");

    srand(42);

    // Sizes relevant to quantized LLM inference
    run_i8mm_benchmark(64,   64,   64,   5000);
    run_i8mm_benchmark(256,  256,  256,  500);
    run_i8mm_benchmark(512,  512,  512,  50);
    run_i8mm_benchmark(2048, 2048, 64,   20);

    printf("\n==================================================\n");
    printf("✅ I8MM benchmark complete.\n");
    printf("vmmlaq_s32: 32 INT8 multiply-accumulates per instruction\n");
    printf("This is what powers quantized inference in llama.cpp\n");
    printf("==================================================\n\n");

    return 0;
}