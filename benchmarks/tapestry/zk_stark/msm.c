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

static inline void point_set_inf(point_t *p) {
    p->x = 0;
    p->y = 0;
    p->is_inf = 1;
}

static inline void point_copy(point_t *dst, const point_t *src) {
    dst->x = src->x;
    dst->y = src->y;
    dst->is_inf = src->is_inf;
}

/* Point addition: componentwise M31 addition */
static inline void point_add(const point_t *p, const point_t *q, point_t *out) {
    m31_t px = p->x;
    m31_t py = p->y;
    int p_inf = p->is_inf;
    m31_t qx = q->x;
    m31_t qy = q->y;
    int q_inf = q->is_inf;

    if (p_inf) {
        out->x = qx;
        out->y = qy;
        out->is_inf = q_inf;
        return;
    }
    if (q_inf) {
        out->x = px;
        out->y = py;
        out->is_inf = p_inf;
        return;
    }

    out->x = m31_add(px, qx);
    out->y = m31_add(py, qy);
    out->is_inf = 0;
}

/* Scalar multiplication via double-and-add (for reference) */
static void scalar_mul(uint32_t s, const point_t *p, point_t *out) {
    point_t result;
    point_t base;
    point_t tmp;
    point_set_inf(&result);
    point_copy(&base, p);
    while (s > 0) {
        if (s & 1) {
            point_add(&result, &base, &tmp);
            point_copy(&result, &tmp);
        }
        /* "Doubling" = adding to itself */
        point_add(&base, &base, &tmp);
        point_copy(&base, &tmp);
        s >>= 1;
    }
    point_copy(out, &result);
}

/*
 * Pippenger MSM: decompose scalars into windows, accumulate in buckets.
 */
void msm_pippenger(const uint32_t *scalars, const point_t *points,
                   int n, point_t *out) {
    point_t total;
    point_t tmp;
    point_set_inf(&total);
    int w;

    /* Process windows from MSB to LSB */
    for (w = NUM_WINDOWS - 1; w >= 0; w--) {
        /* Double the running total WINDOW_BITS times */
        int d;
        for (d = 0; d < WINDOW_BITS; d++) {
            point_add(&total, &total, &tmp);
            point_copy(&total, &tmp);
        }

        /* Initialize buckets */
        point_t buckets[NUM_BUCKETS];
        int b;
        for (b = 0; b < NUM_BUCKETS; b++) {
            point_set_inf(&buckets[b]);
        }

        /* Assign points to buckets based on scalar window */
        TILE_FOR(tp, 0, n, TILE_PTS) {
            int tp_end = TILE_END(tp, n, TILE_PTS);
            int i;
            for (i = tp; i < tp_end; i++) {
                int bucket_idx = (scalars[i] >> (w * WINDOW_BITS))
                                 & (NUM_BUCKETS - 1);
                if (bucket_idx > 0) {
                    point_add(&buckets[bucket_idx], &points[i], &tmp);
                    point_copy(&buckets[bucket_idx], &tmp);
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
        point_t running;
        point_t bucket_sum;
        point_set_inf(&running);
        point_set_inf(&bucket_sum);
        for (b = NUM_BUCKETS - 1; b >= 1; b--) {
            point_add(&running, &buckets[b], &tmp);
            point_copy(&running, &tmp);
            point_add(&bucket_sum, &running, &tmp);
            point_copy(&bucket_sum, &tmp);
        }

        point_add(&total, &bucket_sum, &tmp);
        point_copy(&total, &tmp);
    }
    point_copy(out, &total);
}

/* Naive MSM: compute each s_i * P_i via double-and-add */
void msm_naive(const uint32_t *scalars, const point_t *points, int n,
               point_t *out) {
    point_t total;
    point_t sp;
    point_t tmp;
    point_set_inf(&total);
    int i;
    for (i = 0; i < n; i++) {
        scalar_mul(scalars[i], &points[i], &sp);
        point_add(&total, &sp, &tmp);
        point_copy(&total, &tmp);
    }
    point_copy(out, &total);
}

/* Direct MSM using M31 arithmetic (ground truth):
 * Since our group is additive M31^2, s*P = (s*P.x mod p, s*P.y mod p)
 * and the total is sum of those. */
void msm_direct(const uint32_t *scalars, const point_t *points, int n,
                point_t *out) {
    m31_t sum_x = 0, sum_y = 0;
    int i;
    for (i = 0; i < n; i++) {
        sum_x = m31_add(sum_x, m31_mul((m31_t)scalars[i], points[i].x));
        sum_y = m31_add(sum_y, m31_mul((m31_t)scalars[i], points[i].y));
    }
    out->x = sum_x;
    out->y = sum_y;
    out->is_inf = 0;
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

    point_t result_pip, result_naive, result_direct;
    msm_pippenger(scalars, points, n, &result_pip);
    msm_naive(scalars, points, n, &result_naive);
    msm_direct(scalars, points, n, &result_direct);

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
