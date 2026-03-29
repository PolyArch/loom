/*
 * KV Cache Update -- append new K/V vectors to cache and read back.
 * For each batch and KV head:
 *   cache[B][H_kv][pos][D_h] = new_kv[B][H_kv][D_h]  (store at position pos)
 *   Then read back full cache [B][H_kv][C][D_h] for attention.
 * B=batch, H_kv=KV heads, C=cache_len, D_h=64.
 *
 * Variants: linear, ring, paged.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "tile_utils.h"

#define BATCH     4
#define NUM_HEADS_KV 8
#define HEAD_DIM  64
#define CACHE_LEN 256
#define TILE_C    32

/* --- Primary kernel: linear cache append --- */

void kv_cache_update(float *cache, const float *new_kv,
                     int B, int H_kv, int D_h, int C, int pos) {
    for (int b = 0; b < B; b++) {
        for (int h = 0; h < H_kv; h++) {
            const float *src = new_kv + ((size_t)b * H_kv + h) * D_h;
            float *dst = cache + (((size_t)b * H_kv + h) * C + pos) * D_h;

            TILE_FOR(td, 0, D_h, 32) {
                int td_end = TILE_END(td, D_h, 32);
                for (int d = td; d < td_end; d++) {
                    dst[d] = src[d];
                }
            }
        }
    }
}

/* --- Variants --- */

void kv_cache_update_linear(float *cache, const float *new_kv,
                            int B, int H_kv, int D_h, int C, int pos) {
    kv_cache_update(cache, new_kv, B, H_kv, D_h, C, pos);
}

/* Ring buffer: position wraps around when cache is full */
void kv_cache_update_ring(float *cache, const float *new_kv,
                          int B, int H_kv, int D_h, int C, int pos) {
    int ring_pos = pos % C;
    kv_cache_update(cache, new_kv, B, H_kv, D_h, C, ring_pos);
}

/* Paged: cache is organized in pages; append within current page */
void kv_cache_update_paged(float *cache, const float *new_kv,
                           int B, int H_kv, int D_h, int C, int pos) {
    /* Paged layout: same linear layout but with page-aligned access.
       For simplicity, same as linear (page management is external). */
    kv_cache_update(cache, new_kv, B, H_kv, D_h, C, pos);
}

/* Read back full cache for a batch and head (utility, not a kernel variant) */
void kv_cache_read(const float *cache, float *output,
                   int B, int H_kv, int D_h, int C, int valid_len) {
    for (int b = 0; b < B; b++) {
        for (int h = 0; h < H_kv; h++) {
            const float *src = cache + (((size_t)b * H_kv + h) * C) * D_h;
            float *dst = output + ((size_t)b * H_kv + h) * valid_len * D_h;

            TILE_FOR(tc, 0, valid_len, TILE_C) {
                int tc_end = TILE_END(tc, valid_len, TILE_C);
                for (int c = tc; c < tc_end; c++) {
                    for (int d = 0; d < D_h; d++) {
                        dst[c * D_h + d] = src[c * D_h + d];
                    }
                }
            }
        }
    }
}

/* --- Reference implementation --- */

void kv_cache_update_ref(float *cache, const float *new_kv,
                         int B, int H_kv, int D_h, int C, int pos) {
    for (int b = 0; b < B; b++) {
        for (int h = 0; h < H_kv; h++) {
            const float *src = new_kv + ((size_t)b * H_kv + h) * D_h;
            float *dst = cache + (((size_t)b * H_kv + h) * C + pos) * D_h;
            for (int d = 0; d < D_h; d++) {
                dst[d] = src[d];
            }
        }
    }
}

/* --- Self-test --- */

int main(void) {
    int B = BATCH, H_kv = NUM_HEADS_KV, D_h = HEAD_DIM, C = CACHE_LEN;
    size_t cache_size = (size_t)B * H_kv * C * D_h;
    size_t kv_size = (size_t)B * H_kv * D_h;

    float *cache_t = (float *)malloc(cache_size * sizeof(float));
    float *cache_r = (float *)malloc(cache_size * sizeof(float));
    float *new_kv  = (float *)malloc(kv_size * sizeof(float));

    if (!cache_t || !cache_r || !new_kv) {
        fprintf(stderr, "Allocation failed\n");
        return 1;
    }

    /* Initialize caches with zeros */
    memset(cache_t, 0, cache_size * sizeof(float));
    memset(cache_r, 0, cache_size * sizeof(float));

    /* Append several KV vectors to test multi-step update */
    int pass = 1;
    for (int step = 0; step < 16; step++) {
        for (size_t i = 0; i < kv_size; i++)
            new_kv[i] = ((float)((i + step * 7) % 101) - 50.0f) / 50.0f;

        kv_cache_update(cache_t, new_kv, B, H_kv, D_h, C, step);
        kv_cache_update_ref(cache_r, new_kv, B, H_kv, D_h, C, step);
    }

    /* Compare full caches */
    float max_err = 0.0f;
    for (size_t i = 0; i < cache_size; i++) {
        float err = fabsf(cache_t[i] - cache_r[i]);
        if (err > max_err) max_err = err;
    }

    printf("kv_cache_update: max_error = %e\n", max_err);
    pass = (max_err == 0.0f);
    printf("kv_cache_update: %s\n", pass ? "PASS" : "FAIL");

    free(cache_t); free(cache_r); free(new_kv);
    return pass ? 0 : 1;
}
