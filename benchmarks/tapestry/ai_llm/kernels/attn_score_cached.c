/*
 * Attention Score (Cached) -- Q * K_cache^T for decode step.
 * Computes score[B][H][1][C] = Q[B][H][1][D_h] * K_cache[B][H_kv][C][D_h]^T / sqrt(D_h)
 * This is a vector-matrix multiply (S=1): for each head, the single query
 * vector is dotted against all C cached key vectors.
 * B=batch, H=query heads, H_kv=KV heads, C=cache_len, D_h=64.
 *
 * Variants: full, blocked, sparse.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "tile_utils.h"

#define BATCH        4
#define NUM_HEADS    8
#define NUM_HEADS_KV 8
#define HEAD_DIM     64
#define CACHE_LEN    256
#define TILE_C       32

/* --- Core: Q * K_cache^T with tiled cache dimension --- */

static void attn_score_cached_impl(const float *Q, const float *K_cache,
                                   float *score,
                                   int B, int H, int H_kv, int C, int D_h,
                                   int tile_c) {
    float scale = 1.0f / sqrtf((float)D_h);
    int kv_group = H / H_kv;

    for (int b = 0; b < B; b++) {
        for (int h = 0; h < H; h++) {
            const float *q = Q + ((size_t)b * H + h) * D_h;
            int h_kv = h / kv_group;
            const float *k = K_cache + ((size_t)b * H_kv + h_kv) * C * D_h;
            float *s = score + ((size_t)b * H + h) * C;

            TILE_FOR(tc, 0, C, tile_c) {
                int tc_end = TILE_END(tc, C, tile_c);
                for (int c = tc; c < tc_end; c++) {
                    float sum = 0.0f;
                    for (int d = 0; d < D_h; d++) {
                        sum += q[d] * k[c * D_h + d];
                    }
                    s[c] = sum * scale;
                }
            }
        }
    }
}

void attn_score_cached(const float *Q, const float *K_cache, float *score,
                       int B, int H, int H_kv, int C, int D_h) {
    attn_score_cached_impl(Q, K_cache, score, B, H, H_kv, C, D_h, TILE_C);
}

/* --- Variants --- */

void attn_score_cached_full(const float *Q, const float *K_cache,
                            float *score,
                            int B, int H, int H_kv, int C, int D_h) {
    attn_score_cached_impl(Q, K_cache, score, B, H, H_kv, C, D_h, TILE_C);
}

/* Blocked: larger cache tile for better locality */
void attn_score_cached_blocked(const float *Q, const float *K_cache,
                               float *score,
                               int B, int H, int H_kv, int C, int D_h) {
    attn_score_cached_impl(Q, K_cache, score, B, H, H_kv, C, D_h, 64);
}

/* Sparse: only compute scores for non-masked positions (simulated) */
void attn_score_cached_sparse(const float *Q, const float *K_cache,
                              float *score,
                              int B, int H, int H_kv, int C, int D_h) {
    /* Sparse attention: compute all but could skip positions.
       For DFG compilation, produce the same structure. */
    attn_score_cached_impl(Q, K_cache, score, B, H, H_kv, C, D_h, TILE_C);
}

/* --- Reference implementation --- */

void attn_score_cached_ref(const float *Q, const float *K_cache, float *score,
                           int B, int H, int H_kv, int C, int D_h) {
    float scale = 1.0f / sqrtf((float)D_h);
    int kv_group = H / H_kv;

    for (int b = 0; b < B; b++) {
        for (int h = 0; h < H; h++) {
            const float *q = Q + ((size_t)b * H + h) * D_h;
            int h_kv = h / kv_group;
            const float *k = K_cache + ((size_t)b * H_kv + h_kv) * C * D_h;
            float *s = score + ((size_t)b * H + h) * C;

            for (int c = 0; c < C; c++) {
                float sum = 0.0f;
                for (int d = 0; d < D_h; d++) {
                    sum += q[d] * k[c * D_h + d];
                }
                s[c] = sum * scale;
            }
        }
    }
}

/* --- Self-test --- */

int main(void) {
    int B = BATCH, H = NUM_HEADS, H_kv = NUM_HEADS_KV;
    int C = CACHE_LEN, D_h = HEAD_DIM;

    size_t q_size = (size_t)B * H * D_h;
    size_t k_size = (size_t)B * H_kv * C * D_h;
    size_t s_size = (size_t)B * H * C;

    float *Q       = (float *)malloc(q_size * sizeof(float));
    float *K_cache = (float *)malloc(k_size * sizeof(float));
    float *S_tile  = (float *)malloc(s_size * sizeof(float));
    float *S_ref   = (float *)malloc(s_size * sizeof(float));

    if (!Q || !K_cache || !S_tile || !S_ref) {
        fprintf(stderr, "Allocation failed\n");
        return 1;
    }

    for (size_t i = 0; i < q_size; i++)
        Q[i] = ((float)(i % 67) - 33.0f) / 33.0f;
    for (size_t i = 0; i < k_size; i++)
        K_cache[i] = ((float)(i % 73) - 36.0f) / 36.0f;

    attn_score_cached(Q, K_cache, S_tile, B, H, H_kv, C, D_h);
    attn_score_cached_ref(Q, K_cache, S_ref, B, H, H_kv, C, D_h);

    float max_err = 0.0f;
    for (size_t i = 0; i < s_size; i++) {
        float err = fabsf(S_tile[i] - S_ref[i]);
        if (err > max_err) max_err = err;
    }

    printf("attn_score_cached: max_error = %e\n", max_err);
    int pass = (max_err < 1e-3f);
    printf("attn_score_cached: %s\n", pass ? "PASS" : "FAIL");

    free(Q); free(K_cache); free(S_tile); free(S_ref);
    return pass ? 0 : 1;
}
