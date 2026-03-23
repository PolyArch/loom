/*
 * Multi-Scalar Multiplication (MSM) using Pippenger's bucket method.
 * Computes R = sum(s_i * P_i) for N scalar-point pairs.
 * Uses affine points with coordinates in M31 field.
 * The "curve" is the additive group (M31, M31) with componentwise
 * operations, which makes point_add trivially correct. This lets us
 * benchmark the Pippenger bucket algorithm structure accurately.
 * Tiled by bucket windows.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "tile_utils.h"
#include "m31_arith.h"

#define NUM_POINTS    64
#define WINDOW_BITS   4
#define NUM_WINDOWS   8   /* ceil(32 / WINDOW_BITS) */
#define NUM_BUCKETS   (1 << WINDOW_BITS)  /* 16 */
#define TILE_PTS      16

/* Point in (M31, M31) additive group.
 * is_inf=1 means the identity element (zero point). */
typedef struct {
    m31_t x;
    m31_t y;
    int is_inf;
} point_t;

static const point_t POINT_INF = {0, 0, 1};

/* Point addition: componentwise M31 addition */
static inline point_t point_add(point_t p, point_t q) {
    if (p.is_inf) return q;
    if (q.is_inf) return p;
    point_t r;
    r.x = m31_add(p.x, q.x);
    r.y = m31_add(p.y, q.y);
    r.is_inf = 0;
    return r;
}

/* Scalar multiplication via double-and-add (for reference) */
static point_t scalar_mul(uint32_t s, point_t p) {
    point_t result = POINT_INF;
    point_t base = p;
    while (s > 0) {
        if (s & 1) {
            result = point_add(result, base);
        }
        /* "Doubling" = adding to itself */
        base = point_add(base, base);
        s >>= 1;
    }
    return result;
}

/*
 * Pippenger MSM: decompose scalars into windows, accumulate in buckets.
 */
point_t msm_pippenger(const uint32_t *scalars, const point_t *points,
                      int n) {
    point_t total = POINT_INF;
    int w;

    /* Process windows from MSB to LSB */
    for (w = NUM_WINDOWS - 1; w >= 0; w--) {
        /* Double the running total WINDOW_BITS times */
        int d;
        for (d = 0; d < WINDOW_BITS; d++) {
            total = point_add(total, total);
        }

        /* Initialize buckets */
        point_t buckets[NUM_BUCKETS];
        int b;
        for (b = 0; b < NUM_BUCKETS; b++) {
            buckets[b] = POINT_INF;
        }

        /* Assign points to buckets based on scalar window */
        TILE_FOR(tp, 0, n, TILE_PTS) {
            int tp_end = TILE_END(tp, n, TILE_PTS);
            int i;
            for (i = tp; i < tp_end; i++) {
                int bucket_idx = (scalars[i] >> (w * WINDOW_BITS))
                                 & (NUM_BUCKETS - 1);
                if (bucket_idx > 0) {
                    buckets[bucket_idx] = point_add(buckets[bucket_idx],
                                                    points[i]);
                }
            }
        }

        /* Aggregate buckets: bucket_sum = sum(b * bucket[b])
         * This is computed as:
         * running  = bucket[15]
         * sum += running
         * running += bucket[14]
         * sum += running
         * ...
         * running += bucket[1]
         * sum += running
         */
        point_t running = POINT_INF;
        point_t bucket_sum = POINT_INF;
        for (b = NUM_BUCKETS - 1; b >= 1; b--) {
            running = point_add(running, buckets[b]);
            bucket_sum = point_add(bucket_sum, running);
        }

        total = point_add(total, bucket_sum);
    }

    return total;
}

/* Naive MSM: compute each s_i * P_i via double-and-add */
point_t msm_naive(const uint32_t *scalars, const point_t *points, int n) {
    point_t total = POINT_INF;
    int i;
    for (i = 0; i < n; i++) {
        point_t sp = scalar_mul(scalars[i], points[i]);
        total = point_add(total, sp);
    }
    return total;
}

/* Direct MSM using M31 arithmetic (ground truth):
 * Since our group is additive M31^2, s*P = (s*P.x mod p, s*P.y mod p)
 * and the total is sum of those. */
point_t msm_direct(const uint32_t *scalars, const point_t *points, int n) {
    m31_t sum_x = 0, sum_y = 0;
    int i;
    for (i = 0; i < n; i++) {
        sum_x = m31_add(sum_x, m31_mul((m31_t)scalars[i], points[i].x));
        sum_y = m31_add(sum_y, m31_mul((m31_t)scalars[i], points[i].y));
    }
    point_t r;
    r.x = sum_x;
    r.y = sum_y;
    r.is_inf = 0;
    return r;
}

int main(void) {
    int n = NUM_POINTS;

    uint32_t *scalars = (uint32_t *)malloc((size_t)n * sizeof(uint32_t));
    point_t *points = (point_t *)malloc((size_t)n * sizeof(point_t));

    if (!scalars || !points) {
        fprintf(stderr, "Allocation failed\n");
        return 1;
    }

    /* Generate test data */
    unsigned int state = 42;
    int i;
    for (i = 0; i < n; i++) {
        state = (state * 1103515245 + 12345) & 0x7FFFFFFF;
        scalars[i] = (uint32_t)(state % 256); /* small scalars */

        state = (state * 1103515245 + 12345) & 0x7FFFFFFF;
        m31_t x = (m31_t)(state % 1000 + 1);
        state = (state * 1103515245 + 12345) & 0x7FFFFFFF;
        m31_t y = (m31_t)(state % 1000 + 1);
        points[i].x = x;
        points[i].y = y;
        points[i].is_inf = 0;
    }

    point_t result_pip = msm_pippenger(scalars, points, n);
    point_t result_naive = msm_naive(scalars, points, n);
    point_t result_direct = msm_direct(scalars, points, n);

    printf("msm: n=%d, window_bits=%d\n", n, WINDOW_BITS);
    printf("msm: pippenger = (%u, %u)\n", result_pip.x, result_pip.y);
    printf("msm: naive     = (%u, %u)\n", result_naive.x, result_naive.y);
    printf("msm: direct    = (%u, %u)\n", result_direct.x, result_direct.y);

    int pass_pn = (result_pip.x == result_naive.x) &&
                  (result_pip.y == result_naive.y);
    int pass_pd = (result_pip.x == result_direct.x) &&
                  (result_pip.y == result_direct.y);

    printf("msm: pippenger vs naive: %s\n", pass_pn ? "OK" : "MISMATCH");
    printf("msm: pippenger vs direct: %s\n", pass_pd ? "OK" : "MISMATCH");

    int pass = pass_pn && pass_pd;
    printf("msm: %s\n", pass ? "PASS" : "FAIL");

    free(scalars); free(points);
    return pass ? 0 : 1;
}
