/*
 * Polynomial Evaluation using Horner's method over M31 field.
 * Evaluates a degree-N polynomial at M points using:
 *   result = c[n]; for(i=n-1..0) result = result*x + c[i]
 * Tiled by batches of evaluation points.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "tile_utils.h"
#include "m31_arith.h"

#define POLY_DEGREE   256
#define NUM_EVAL_PTS  128
#define TILE_PTS      16

/*
 * Horner evaluation of polynomial at a single point.
 * coeffs[0] is constant term, coeffs[degree] is leading coefficient.
 */
static inline m31_t horner_eval(const m31_t *coeffs, int degree, m31_t x) {
    m31_t result = coeffs[degree];
    int i;
    for (i = degree - 1; i >= 0; i--) {
        result = m31_add(m31_mul(result, x), coeffs[i]);
    }
    return result;
}

/* Tiled polynomial evaluation at multiple points */
void poly_eval_tiled(const m31_t *coeffs, int degree,
                     const m31_t *points, m31_t *results, int n_pts) {
    TILE_FOR(tp, 0, n_pts, TILE_PTS) {
        int tp_end = TILE_END(tp, n_pts, TILE_PTS);
        int i;
        for (i = tp; i < tp_end; i++) {
            results[i] = horner_eval(coeffs, degree, points[i]);
        }
    }
}

/* Reference (non-tiled) polynomial evaluation */
void poly_eval_ref(const m31_t *coeffs, int degree,
                   const m31_t *points, m31_t *results, int n_pts) {
    int i;
    for (i = 0; i < n_pts; i++) {
        results[i] = horner_eval(coeffs, degree, points[i]);
    }
}

/*
 * Naive polynomial evaluation (for correctness check on small polynomials).
 * Computes sum(coeffs[i] * x^i) directly.
 */
static m31_t naive_eval(const m31_t *coeffs, int degree, m31_t x) {
    m31_t result = 0;
    m31_t x_pow = 1;
    int i;
    for (i = 0; i <= degree; i++) {
        result = m31_add(result, m31_mul(coeffs[i], x_pow));
        x_pow = m31_mul(x_pow, x);
    }
    return result;
}

int main(void) {
    int degree = POLY_DEGREE;
    int n_pts = NUM_EVAL_PTS;

    m31_t *coeffs = (m31_t *)malloc((size_t)(degree + 1) * sizeof(m31_t));
    m31_t *points = (m31_t *)malloc((size_t)n_pts * sizeof(m31_t));
    m31_t *results_t = (m31_t *)malloc((size_t)n_pts * sizeof(m31_t));
    m31_t *results_r = (m31_t *)malloc((size_t)n_pts * sizeof(m31_t));

    if (!coeffs || !points || !results_t || !results_r) {
        fprintf(stderr, "Allocation failed\n");
        return 1;
    }

    /* Generate polynomial coefficients */
    unsigned int state = 42;
    int i;
    for (i = 0; i <= degree; i++) {
        state = (state * 1103515245 + 12345) & 0x7FFFFFFF;
        coeffs[i] = (m31_t)(state % 1000);
    }

    /* Generate evaluation points */
    for (i = 0; i < n_pts; i++) {
        state = (state * 1103515245 + 12345) & 0x7FFFFFFF;
        coeffs[i % (degree + 1)] = (m31_t)(state % 1000);
        points[i] = (m31_t)(i + 1); /* simple distinct points */
    }

    /* Re-generate coefficients (previous loop modified some) */
    state = 42;
    for (i = 0; i <= degree; i++) {
        state = (state * 1103515245 + 12345) & 0x7FFFFFFF;
        coeffs[i] = (m31_t)(state % 1000);
    }

    /* Evaluate */
    poly_eval_tiled(coeffs, degree, points, results_t, n_pts);
    poly_eval_ref(coeffs, degree, points, results_r, n_pts);

    /* Verify tiled matches reference */
    int mismatches = 0;
    for (i = 0; i < n_pts; i++) {
        if (results_t[i] != results_r[i]) {
            mismatches++;
        }
    }

    printf("poly_eval: degree=%d, points=%d, mismatches=%d\n",
           degree, n_pts, mismatches);

    /* Verify Horner against naive for a small polynomial */
    int small_deg = 10;
    m31_t small_coeffs[11];
    for (i = 0; i <= small_deg; i++) {
        small_coeffs[i] = (m31_t)(i + 1);
    }

    int naive_ok = 1;
    m31_t test_points[] = {0, 1, 2, 3, 5, 10, 100, 1000};
    int n_test = sizeof(test_points) / sizeof(test_points[0]);
    for (i = 0; i < n_test; i++) {
        m31_t h = horner_eval(small_coeffs, small_deg, test_points[i]);
        m31_t n_val = naive_eval(small_coeffs, small_deg, test_points[i]);
        if (h != n_val) {
            printf("poly_eval: horner(%u)=%u, naive=%u\n",
                   test_points[i], h, n_val);
            naive_ok = 0;
        }
    }
    printf("poly_eval: horner vs naive test: %s\n",
           naive_ok ? "OK" : "FAILED");

    /* Verify polynomial identity: p(0) = coeffs[0] */
    m31_t p0 = horner_eval(coeffs, degree, 0);
    int zero_ok = (p0 == coeffs[0]);
    printf("poly_eval: p(0)==coeffs[0] test: %s\n",
           zero_ok ? "OK" : "FAILED");

    int pass = (mismatches == 0) && naive_ok && zero_ok;
    printf("poly_eval: %s\n", pass ? "PASS" : "FAIL");

    free(coeffs); free(points); free(results_t); free(results_r);
    return pass ? 0 : 1;
}
