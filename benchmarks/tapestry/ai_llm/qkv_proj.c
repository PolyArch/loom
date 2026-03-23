/*
 * QKV Projection -- matmul for transformer self-attention.
 * Computes C[M][N] = A[M][K] * B[K][N]
 * where M=seq_len, N=3*d_model, K=d_model (d_model=512).
 * Tiled: tile_M=32, tile_N=64, tile_K=32.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "tile_utils.h"

#define D_MODEL   512
#define SEQ_LEN   128
#define N_OUT     (3 * D_MODEL)
#define TILE_M    32
#define TILE_N    64
#define TILE_K    32

void qkv_proj(const float *A, const float *B, float *C,
              int M, int N, int K) {
    memset(C, 0, (size_t)M * N * sizeof(float));

    TILE_FOR(tm, 0, M, TILE_M) {
        int tm_end = TILE_END(tm, M, TILE_M);
        TILE_FOR(tn, 0, N, TILE_N) {
            int tn_end = TILE_END(tn, N, TILE_N);
            TILE_FOR(tk, 0, K, TILE_K) {
                int tk_end = TILE_END(tk, K, TILE_K);
                for (int i = tm; i < tm_end; i++) {
                    for (int j = tn; j < tn_end; j++) {
                        float sum = 0.0f;
                        for (int k = tk; k < tk_end; k++) {
                            sum += A[i * K + k] * B[k * N + j];
                        }
                        C[i * N + j] += sum;
                    }
                }
            }
        }
    }
}

void qkv_proj_ref(const float *A, const float *B, float *C,
                  int M, int N, int K) {
    memset(C, 0, (size_t)M * N * sizeof(float));
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

int main(void) {
    int M = SEQ_LEN, N = N_OUT, K = D_MODEL;
    float *A = (float *)malloc((size_t)M * K * sizeof(float));
    float *B = (float *)malloc((size_t)K * N * sizeof(float));
    float *C_tiled = (float *)malloc((size_t)M * N * sizeof(float));
    float *C_ref   = (float *)malloc((size_t)M * N * sizeof(float));

    if (!A || !B || !C_tiled || !C_ref) {
        fprintf(stderr, "Allocation failed\n");
        return 1;
    }

    /* Generate test data */
    for (int i = 0; i < M * K; i++)
        A[i] = ((float)(i % 97) - 48.0f) / 48.0f;
    for (int i = 0; i < K * N; i++)
        B[i] = ((float)(i % 83) - 41.0f) / 41.0f;

    qkv_proj(A, B, C_tiled, M, N, K);
    qkv_proj_ref(A, B, C_ref, M, N, K);

    /* Compare results */
    float max_err = 0.0f;
    for (int i = 0; i < M * N; i++) {
        float err = fabsf(C_tiled[i] - C_ref[i]);
        if (err > max_err) max_err = err;
    }

    printf("qkv_proj: max_error = %e\n", max_err);
    int pass = (max_err < 1e-3f);
    printf("qkv_proj: %s\n", pass ? "PASS" : "FAIL");

    free(A); free(B); free(C_tiled); free(C_ref);
    return pass ? 0 : 1;
}
