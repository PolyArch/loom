/*
 * Attention Score -- batched matmul Q * K^T for multi-head attention.
 * Computes score[H][N][N] = Q[H][N][D] * K[H][N][D]^T / sqrt(D)
 * H=8 heads, N=seq_len, D=d_model/H=64.
 * Tiled per head, tile_N=32 for the sequence dimension.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "tile_utils.h"

#define NUM_HEADS  8
#define SEQ_LEN    128
#define HEAD_DIM   64
#define TILE_N     32

void attn_score(const float *Q, const float *K, float *score,
                int H, int N, int D) {
    float scale = 1.0f / sqrtf((float)D);

    for (int h = 0; h < H; h++) {
        const float *Qh = Q + (size_t)h * N * D;
        const float *Kh = K + (size_t)h * N * D;
        float *Sh = score + (size_t)h * N * N;

        memset(Sh, 0, (size_t)N * N * sizeof(float));

        TILE_FOR(ti, 0, N, TILE_N) {
            int ti_end = TILE_END(ti, N, TILE_N);
            TILE_FOR(tj, 0, N, TILE_N) {
                int tj_end = TILE_END(tj, N, TILE_N);
                for (int i = ti; i < ti_end; i++) {
                    for (int j = tj; j < tj_end; j++) {
                        float sum = 0.0f;
                        for (int k = 0; k < D; k++) {
                            sum += Qh[i * D + k] * Kh[j * D + k];
                        }
                        Sh[i * N + j] = sum * scale;
                    }
                }
            }
        }
    }
}

void attn_score_ref(const float *Q, const float *K, float *score,
                    int H, int N, int D) {
    float scale = 1.0f / sqrtf((float)D);
    for (int h = 0; h < H; h++) {
        const float *Qh = Q + (size_t)h * N * D;
        const float *Kh = K + (size_t)h * N * D;
        float *Sh = score + (size_t)h * N * N;
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                float sum = 0.0f;
                for (int k = 0; k < D; k++) {
                    sum += Qh[i * D + k] * Kh[j * D + k];
                }
                Sh[i * N + j] = sum * scale;
            }
        }
    }
}

int main(void) {
    int H = NUM_HEADS, N = SEQ_LEN, D = HEAD_DIM;
    size_t qk_size = (size_t)H * N * D;
    size_t s_size  = (size_t)H * N * N;

    float *Q      = (float *)malloc(qk_size * sizeof(float));
    float *K      = (float *)malloc(qk_size * sizeof(float));
    float *S_tile = (float *)malloc(s_size * sizeof(float));
    float *S_ref  = (float *)malloc(s_size * sizeof(float));

    if (!Q || !K || !S_tile || !S_ref) {
        fprintf(stderr, "Allocation failed\n");
        return 1;
    }

    for (size_t i = 0; i < qk_size; i++) {
        Q[i] = ((float)(i % 67) - 33.0f) / 33.0f;
        K[i] = ((float)(i % 73) - 36.0f) / 36.0f;
    }

    attn_score(Q, K, S_tile, H, N, D);
    attn_score_ref(Q, K, S_ref, H, N, D);

    float max_err = 0.0f;
    for (size_t i = 0; i < s_size; i++) {
        float err = fabsf(S_tile[i] - S_ref[i]);
        if (err > max_err) max_err = err;
    }

    printf("attn_score: max_error = %e\n", max_err);
    int pass = (max_err < 1e-3f);
    printf("attn_score: %s\n", pass ? "PASS" : "FAIL");

    free(Q); free(K); free(S_tile); free(S_ref);
    return pass ? 0 : 1;
}
