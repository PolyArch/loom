/*
 * QKV Projection -- matmul for transformer self-attention.
 * Computes C[M][N] = A[M][K] * B[K][N]
 * where M=seq_len, N=3*d_model, K=d_model (d_model=512).
 *
 * Variants: tile32, tile64, tile128, separate (Q/K/V computed independently).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "tile_utils.h"

#define D_MODEL   512
#define SEQ_LEN   128
#define N_OUT     (3 * D_MODEL)

/* --- Primary kernel: default tile sizes --- */

static void gemm_tiled(const float *A, const float *B, float *C,
                       int M, int N, int K,
                       int tile_m, int tile_n, int tile_k) {
    memset(C, 0, (size_t)M * N * sizeof(float));
    TILE_FOR(tm, 0, M, tile_m) {
        int tm_end = TILE_END(tm, M, tile_m);
        TILE_FOR(tn, 0, N, tile_n) {
            int tn_end = TILE_END(tn, N, tile_n);
            TILE_FOR(tk, 0, K, tile_k) {
                int tk_end = TILE_END(tk, K, tile_k);
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

void qkv_proj(const float *A, const float *B, float *C,
              int M, int N, int K) {
    gemm_tiled(A, B, C, M, N, K, 32, 64, 32);
}

/* --- Tile-size variants --- */

void qkv_proj_tile32(const float *A, const float *B, float *C,
                     int M, int N, int K) {
    gemm_tiled(A, B, C, M, N, K, 32, 32, 32);
}

void qkv_proj_tile64(const float *A, const float *B, float *C,
                     int M, int N, int K) {
    gemm_tiled(A, B, C, M, N, K, 64, 64, 64);
}

void qkv_proj_tile128(const float *A, const float *B, float *C,
                      int M, int N, int K) {
    gemm_tiled(A, B, C, M, N, K, 128, 128, 64);
}

/* Separate variant: compute Q, K, V projections independently */
void qkv_proj_separate(const float *input, const float *Wq,
                       const float *Wk, const float *Wv,
                       float *Q, float *K_out, float *V,
                       int M, int D) {
    gemm_tiled(input, Wq, Q, M, D, D, 32, 64, 32);
    gemm_tiled(input, Wk, K_out, M, D, D, 32, 64, 32);
    gemm_tiled(input, Wv, V, M, D, D, 32, 64, 32);
}

/* --- Reference implementation --- */

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

/* --- Self-test --- */

int main(void) {
    int M = SEQ_LEN, N = N_OUT, K = D_MODEL;
    float *A       = (float *)malloc((size_t)M * K * sizeof(float));
    float *B       = (float *)malloc((size_t)K * N * sizeof(float));
    float *C_tiled = (float *)malloc((size_t)M * N * sizeof(float));
    float *C_ref   = (float *)malloc((size_t)M * N * sizeof(float));
    float *C_v32   = (float *)malloc((size_t)M * N * sizeof(float));
    float *C_v64   = (float *)malloc((size_t)M * N * sizeof(float));
    float *C_v128  = (float *)malloc((size_t)M * N * sizeof(float));

    if (!A || !B || !C_tiled || !C_ref || !C_v32 || !C_v64 || !C_v128) {
        fprintf(stderr, "Allocation failed\n");
        return 1;
    }

    for (int i = 0; i < M * K; i++)
        A[i] = ((float)(i % 97) - 48.0f) / 48.0f;
    for (int i = 0; i < K * N; i++)
        B[i] = ((float)(i % 83) - 41.0f) / 41.0f;

    qkv_proj(A, B, C_tiled, M, N, K);
    qkv_proj_ref(A, B, C_ref, M, N, K);
    qkv_proj_tile32(A, B, C_v32, M, N, K);
    qkv_proj_tile64(A, B, C_v64, M, N, K);
    qkv_proj_tile128(A, B, C_v128, M, N, K);

    float max_err = 0.0f;
    float max_err32 = 0.0f, max_err64 = 0.0f, max_err128 = 0.0f;
    for (int i = 0; i < M * N; i++) {
        float err = fabsf(C_tiled[i] - C_ref[i]);
        if (err > max_err) max_err = err;
        err = fabsf(C_v32[i] - C_ref[i]);
        if (err > max_err32) max_err32 = err;
        err = fabsf(C_v64[i] - C_ref[i]);
        if (err > max_err64) max_err64 = err;
        err = fabsf(C_v128[i] - C_ref[i]);
        if (err > max_err128) max_err128 = err;
    }

    printf("qkv_proj: max_error = %e\n", max_err);
    printf("qkv_proj_tile32: max_error = %e\n", max_err32);
    printf("qkv_proj_tile64: max_error = %e\n", max_err64);
    printf("qkv_proj_tile128: max_error = %e\n", max_err128);

    int pass = (max_err < 1e-3f) && (max_err32 < 1e-3f) &&
               (max_err64 < 1e-3f) && (max_err128 < 1e-3f);
    printf("qkv_proj: %s\n", pass ? "PASS" : "FAIL");

    free(A); free(B); free(C_tiled); free(C_ref);
    free(C_v32); free(C_v64); free(C_v128);
    return pass ? 0 : 1;
}
