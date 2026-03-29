/*
 * Attention Apply (Cached) -- score * V_cache for decode step.
 * Computes out[B][H][1][D_h] = score[B][H][1][C] * V_cache[B][H_kv][C][D_h]
 * Vector-matrix multiply: weighted sum of cached value vectors by attention scores.
 * B=batch, H=query heads, H_kv=KV heads, C=cache_len, D_h=64.
 *
 * Variants: full, blocked, fused.
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

/* --- Core: score * V_cache with tiled cache dimension --- */

static void attn_apply_cached_impl(const float *score, const float *V_cache,
                                   float *output,
                                   int B, int H, int H_kv, int C, int D_h,
                                   int tile_c) {
    int kv_group = H / H_kv;

    for (int b = 0; b < B; b++) {
        for (int h = 0; h < H; h++) {
            const float *s = score + ((size_t)b * H + h) * C;
            int h_kv = h / kv_group;
            const float *v = V_cache + ((size_t)b * H_kv + h_kv) * C * D_h;
            float *o = output + ((size_t)b * H + h) * D_h;

            memset(o, 0, (size_t)D_h * sizeof(float));

            TILE_FOR(tc, 0, C, tile_c) {
                int tc_end = TILE_END(tc, C, tile_c);
                for (int c = tc; c < tc_end; c++) {
                    float sc = s[c];
                    for (int d = 0; d < D_h; d++) {
                        o[d] += sc * v[c * D_h + d];
                    }
                }
            }
        }
    }
}

void attn_apply_cached(const float *score, const float *V_cache,
                       float *output,
                       int B, int H, int H_kv, int C, int D_h) {
    attn_apply_cached_impl(score, V_cache, output, B, H, H_kv, C, D_h, TILE_C);
}

/* --- Variants --- */

void attn_apply_cached_full(const float *score, const float *V_cache,
                            float *output,
                            int B, int H, int H_kv, int C, int D_h) {
    attn_apply_cached_impl(score, V_cache, output, B, H, H_kv, C, D_h, TILE_C);
}

/* Blocked: larger cache tile */
void attn_apply_cached_blocked(const float *score, const float *V_cache,
                               float *output,
                               int B, int H, int H_kv, int C, int D_h) {
    attn_apply_cached_impl(score, V_cache, output, B, H, H_kv, C, D_h, 64);
}

/* Fused: combined score*V in a single loop (same structure, different tiling) */
void attn_apply_cached_fused(const float *score, const float *V_cache,
                             float *output,
                             int B, int H, int H_kv, int C, int D_h) {
    attn_apply_cached_impl(score, V_cache, output, B, H, H_kv, C, D_h, C);
}

/* --- Reference implementation --- */

void attn_apply_cached_ref(const float *score, const float *V_cache,
                           float *output,
                           int B, int H, int H_kv, int C, int D_h) {
    int kv_group = H / H_kv;

    for (int b = 0; b < B; b++) {
        for (int h = 0; h < H; h++) {
            const float *s = score + ((size_t)b * H + h) * C;
            int h_kv = h / kv_group;
            const float *v = V_cache + ((size_t)b * H_kv + h_kv) * C * D_h;
            float *o = output + ((size_t)b * H + h) * D_h;

            for (int d = 0; d < D_h; d++) {
                float sum = 0.0f;
                for (int c = 0; c < C; c++) {
                    sum += s[c] * v[c * D_h + d];
                }
                o[d] = sum;
            }
        }
    }
}

/* --- Self-test --- */

int main(void) {
    int B = BATCH, H = NUM_HEADS, H_kv = NUM_HEADS_KV;
    int C = CACHE_LEN, D_h = HEAD_DIM;

    size_t s_size = (size_t)B * H * C;
    size_t v_size = (size_t)B * H_kv * C * D_h;
    size_t o_size = (size_t)B * H * D_h;

    float *score   = (float *)malloc(s_size * sizeof(float));
    float *V_cache = (float *)malloc(v_size * sizeof(float));
    float *O_tile  = (float *)malloc(o_size * sizeof(float));
    float *O_ref   = (float *)malloc(o_size * sizeof(float));

    if (!score || !V_cache || !O_tile || !O_ref) {
        fprintf(stderr, "Allocation failed\n");
        return 1;
    }

    /* Generate softmax-like scores per head */
    for (int b = 0; b < B; b++) {
        for (int h = 0; h < H; h++) {
            float *s = score + ((size_t)b * H + h) * C;
            float sum = 0.0f;
            for (int c = 0; c < C; c++) {
                float val = 1.0f / (1.0f + (float)(c * c) / 100.0f);
                s[c] = val;
                sum += val;
            }
            for (int c = 0; c < C; c++) s[c] /= sum;
        }
    }

    for (size_t i = 0; i < v_size; i++)
        V_cache[i] = ((float)(i % 89) - 44.0f) / 44.0f;

    attn_apply_cached(score, V_cache, O_tile, B, H, H_kv, C, D_h);
    attn_apply_cached_ref(score, V_cache, O_ref, B, H, H_kv, C, D_h);

    float max_err = 0.0f;
    for (size_t i = 0; i < o_size; i++) {
        float err = fabsf(O_tile[i] - O_ref[i]);
        if (err > max_err) max_err = err;
    }

    printf("attn_apply_cached: max_error = %e\n", max_err);
    int pass = (max_err < 1e-3f);
    printf("attn_apply_cached: %s\n", pass ? "PASS" : "FAIL");

    free(score); free(V_cache); free(O_tile); free(O_ref);
    return pass ? 0 : 1;
}
