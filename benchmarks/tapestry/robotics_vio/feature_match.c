/*
 * Brute-Force Feature Matching for Visual-Inertial Odometry.
 * Matches 500 query descriptors against 500 train descriptors.
 * Each descriptor is 256 bits (8 x uint32). Distance metric is
 * Hamming distance computed via XOR + popcount.
 * Applies Lowe's ratio test (best/second_best < 0.75).
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include "tile_utils.h"

#define NUM_QUERY    500
#define NUM_TRAIN    500
#define DESC_WORDS   8
#define TILE_Q       32
#define TILE_T       64
#define RATIO_THRESH 0.75f

/* Popcount for 32-bit integer */
static inline int popcount32(uint32_t x) {
    x = x - ((x >> 1) & 0x55555555U);
    x = (x & 0x33333333U) + ((x >> 2) & 0x33333333U);
    x = (x + (x >> 4)) & 0x0F0F0F0FU;
    return (int)((x * 0x01010101U) >> 24);
}

/* Hamming distance between two 256-bit descriptors */
static inline int hamming_distance(const uint32_t *a, const uint32_t *b) {
    int dist = 0;
    int w;
    for (w = 0; w < DESC_WORDS; w++) {
        dist += popcount32(a[w] ^ b[w]);
    }
    return dist;
}

/*
 * Tiled brute-force matching.
 * For each query, find best and second-best match among train descriptors.
 * Returns number of good matches (passing ratio test).
 */
int feature_match_tiled(const uint32_t *query, const uint32_t *train,
                        int nq, int nt, int *match_idx, float *match_ratio) {
    int num_good = 0;
    int q;

    TILE_FOR(tq, 0, nq, TILE_Q) {
        int tq_end = TILE_END(tq, nq, TILE_Q);
        for (q = tq; q < tq_end; q++) {
            int best_dist = DESC_WORDS * 32 + 1;
            int second_dist = DESC_WORDS * 32 + 1;
            int best_idx = -1;

            TILE_FOR(tt, 0, nt, TILE_T) {
                int tt_end = TILE_END(tt, nt, TILE_T);
                int t;
                for (t = tt; t < tt_end; t++) {
                    int d = hamming_distance(&query[q * DESC_WORDS],
                                             &train[t * DESC_WORDS]);
                    if (d < best_dist) {
                        second_dist = best_dist;
                        best_dist = d;
                        best_idx = t;
                    } else if (d < second_dist) {
                        second_dist = d;
                    }
                }
            }

            match_idx[q] = best_idx;
            float ratio = (second_dist > 0)
                          ? (float)best_dist / (float)second_dist
                          : 1.0f;
            match_ratio[q] = ratio;
            if (ratio < RATIO_THRESH) {
                num_good++;
            }
        }
    }
    return num_good;
}

/* Reference (non-tiled) matching */
int feature_match_ref(const uint32_t *query, const uint32_t *train,
                      int nq, int nt, int *match_idx, float *match_ratio) {
    int num_good = 0;
    int q, t;
    for (q = 0; q < nq; q++) {
        int best_dist = DESC_WORDS * 32 + 1;
        int second_dist = DESC_WORDS * 32 + 1;
        int best_idx = -1;

        for (t = 0; t < nt; t++) {
            int d = hamming_distance(&query[q * DESC_WORDS],
                                     &train[t * DESC_WORDS]);
            if (d < best_dist) {
                second_dist = best_dist;
                best_dist = d;
                best_idx = t;
            } else if (d < second_dist) {
                second_dist = d;
            }
        }

        match_idx[q] = best_idx;
        float ratio = (second_dist > 0)
                      ? (float)best_dist / (float)second_dist
                      : 1.0f;
        match_ratio[q] = ratio;
        if (ratio < RATIO_THRESH) {
            num_good++;
        }
    }
    return num_good;
}

int main(void) {
    int nq = NUM_QUERY, nt = NUM_TRAIN;

    uint32_t *query = (uint32_t *)malloc((size_t)nq * DESC_WORDS * sizeof(uint32_t));
    uint32_t *train = (uint32_t *)malloc((size_t)nt * DESC_WORDS * sizeof(uint32_t));
    int *idx_t = (int *)malloc((size_t)nq * sizeof(int));
    int *idx_r = (int *)malloc((size_t)nq * sizeof(int));
    float *ratio_t = (float *)malloc((size_t)nq * sizeof(float));
    float *ratio_r = (float *)malloc((size_t)nq * sizeof(float));

    if (!query || !train || !idx_t || !idx_r || !ratio_t || !ratio_r) {
        fprintf(stderr, "Allocation failed\n");
        return 1;
    }

    /* Generate pseudo-random descriptors */
    unsigned int state = 42;
    int i;
    for (i = 0; i < nq * DESC_WORDS; i++) {
        state = (state * 1103515245 + 12345) & 0x7FFFFFFF;
        query[i] = (uint32_t)state;
    }
    for (i = 0; i < nt * DESC_WORDS; i++) {
        state = (state * 1103515245 + 12345) & 0x7FFFFFFF;
        train[i] = (uint32_t)state;
    }

    /* Make some query descriptors similar to train descriptors */
    for (i = 0; i < 50; i++) {
        int w;
        for (w = 0; w < DESC_WORDS; w++) {
            /* Copy with small perturbation */
            query[i * DESC_WORDS + w] = train[(i * 3) * DESC_WORDS + w];
            if (w < 2) {
                query[i * DESC_WORDS + w] ^= (uint32_t)(1 << (i % 16));
            }
        }
    }

    int good_t = feature_match_tiled(query, train, nq, nt, idx_t, ratio_t);
    int good_r = feature_match_ref(query, train, nq, nt, idx_r, ratio_r);

    /* Verify results match */
    int idx_mismatches = 0;
    float max_ratio_err = 0.0f;
    for (i = 0; i < nq; i++) {
        if (idx_t[i] != idx_r[i]) idx_mismatches++;
        float err = ratio_t[i] - ratio_r[i];
        if (err < 0) err = -err;
        if (err > max_ratio_err) max_ratio_err = err;
    }

    printf("feature_match: good_tiled=%d, good_ref=%d, "
           "idx_mismatches=%d, max_ratio_err=%e\n",
           good_t, good_r, idx_mismatches, max_ratio_err);

    int pass = (good_t == good_r) && (idx_mismatches == 0)
               && (max_ratio_err < 1e-6f);
    printf("feature_match: %s\n", pass ? "PASS" : "FAIL");

    free(query); free(train); free(idx_t); free(idx_r);
    free(ratio_t); free(ratio_r);
    return pass ? 0 : 1;
}
