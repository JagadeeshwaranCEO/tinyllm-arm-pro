// TinyLLM-ARM-Pro | NEON I8MM INT8 Matrix Multiply
// Uses ARMv8.6-A SMMLA (vmmlaq_s32) instruction
// This is the SAME instruction llama.cpp uses internally
// for quantized LLM inference on Apple Silicon M4
//
// Your M4 confirmed I8MM support during our cmake build:
// "Performing Test GGML_MACHINE_SUPPORTS_i8mm - Success"
//
// v2: Pre-packed B matrix eliminates runtime interleaving overhead
// Production engines (llama.cpp) pack weights at quantization time.
// We simulate that here — B is packed ONCE, hot loop is clean SMMLA.

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

// ── Pre-pack B for SMMLA layout ──────────────────────
// Pack B from row-major into interleaved format that
// vmmlaq_s32 expects. Done ONCE before inference —
// exactly how llama.cpp stores quantized weights on disk.
//
// Input:  B[K][N]  row-major
// Output: B_packed — tiles of [col0_rows0-7 | col1_rows0-7]
void pack_b_i8mm(
    const int8_t *B, int8_t *B_packed,
    int K, int N)
{
    for (int j = 0; j + 1 < N; j += 2) {
        for (int k = 0; k + 7 < K; k += 8) {
            int8_t *dst = &B_packed[(j/2) * K * 2 + k * 2];
            for (int kk = 0; kk < 8; kk++) {
                dst[kk]   = B[(k+kk)*N + j];
                dst[kk+8] = B[(k+kk)*N + j+1];
            }
        }
    }
}

// ── I8MM with pre-packed B (production-grade) ────────
// Hot loop: vld1q_s8 + vmmlaq_s32 + vld1q_s8
// No runtime interleaving. This is what real engines do.
void i8mm_matmul_packed(
    const int8_t *A, const int8_t *B_packed,
    int32_t *C, int M, int N, int K)
{
    memset(C, 0, M * N * sizeof(int32_t));

    for (int i = 0; i + 1 < M; i += 2) {
        for (int j = 0; j + 1 < N; j += 2) {
            int32x4_t acc = vdupq_n_s32(0);

            // B_packed tile for this (j, j+1) column pair
            const int8_t *b_tile = &B_packed[(j/2) * K * 2];

            for (int k = 0; k + 7 < K; k += 8) {
                // Pack 2 rows of A (stays in registers, cheap)
                int8_t a_buf[16];
                memcpy(a_buf,     &A[(i+0)*K + k], 8);
                memcpy(a_buf + 8, &A[(i+1)*K + k], 8);
                int8x16_t a_vec = vld1q_s8(a_buf);

                // Load pre-packed B tile — sequential, cache-friendly
                int8x16_t b_vec = vld1q_s8(&b_tile[k * 2]);

                // SMMLA — 32 INT8 multiply-accumulates, ONE instruction
                // This is the entire point of I8MM on ARMv8.6-A
                acc = vmmlaq_s32(acc, a_vec, b_vec);
            }

            // Extract 2x2 output block from accumulator
            // vmmlaq_s32 lane layout:
            //   lane 0: C[row0][col0]
            //   lane 1: C[row0][col1]
            //   lane 2: C[row1][col0]
            //   lane 3: C[row1][col1]
            C[(i+0)*N + j+0] += vgetq_lane_s32(acc, 0);
            C[(i+0)*N + j+1] += vgetq_lane_s32(acc, 1);
            C[(i+1)*N + j+0] += vgetq_lane_s32(acc, 2);
            C[(i+1)*N + j+1] += vgetq_lane_s32(acc, 3);
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

    int8_t  *A        = malloc(M * K);
    int8_t  *B        = malloc(K * N);
    int8_t  *B_packed = malloc(K * N);  // same size, different layout
    int32_t *C_ref    = calloc(M * N, sizeof(int32_t));
    int32_t *C_packed = calloc(M * N, sizeof(int32_t));

    // Fill with small values to avoid INT8 overflow
    for (int i = 0; i < M*K; i++) A[i] = (int8_t)((rand() % 20) - 10);
    for (int i = 0; i < K*N; i++) B[i] = (int8_t)((rand() % 20) - 10);

    // Pre-pack B ONCE — simulates quantization-time weight packing
    pack_b_i8mm(B, B_packed, K, N);

    // Correctness check
    naive_i8_matmul(A, B, C_ref, M, N, K);
    i8mm_matmul_packed(A, B_packed, C_packed, M, N, K);
    printf("Correctness (packed): %s\n",
           verify(C_ref, C_packed, M*N) ? "PASSED ✅" : "FAILED ❌");

    // Benchmark naive INT8
    double start = get_time_ms();
    for (int it = 0; it < iters; it++) {
        memset(C_ref, 0, M*N*sizeof(int32_t));
        naive_i8_matmul(A, B, C_ref, M, N, K);
    }
    double naive_t = (get_time_ms() - start) / iters;

    // Benchmark I8MM with pre-packed B
    start = get_time_ms();
    for (int it = 0; it < iters; it++) {
        memset(C_packed, 0, M*N*sizeof(int32_t));
        i8mm_matmul_packed(A, B_packed, C_packed, M, N, K);
    }
    double packed_t = (get_time_ms() - start) / iters;

    printf("Naive INT8          : %.4f ms\n", naive_t);
    printf("I8MM pre-packed     : %.4f ms  ->  %.2fx speedup\n",
           packed_t, naive_t / packed_t);
    printf("(B packed once at init — not counted in benchmark time)\n");

    free(A); free(B); free(B_packed); free(C_ref); free(C_packed);
}

int main() {
    printf("\n");
    printf("==================================================\n");
    printf("TinyLLM-ARM-Pro | I8MM (SMMLA) Benchmark v2\n");
    printf("ARMv8.6-A vmmlaq_s32 + pre-packed weight layout\n");
    printf("Apple Silicon M4 | MTLGPUFamilyApple9\n");
    printf("Production-grade: same approach as llama.cpp\n");
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
    printf("Pre-packed B: eliminates runtime interleaving overhead\n");
    printf("This matches how llama.cpp stores quantized weights.\n");
    printf("==================================================\n\n");

    return 0;
}