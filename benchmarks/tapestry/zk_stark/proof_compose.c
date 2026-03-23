/*
 * Proof Composition for STARK pipeline.
 * Combines multiple polynomial commitments via random linear combination
 * using Fiat-Shamir challenge generation (hash-based).
 * Operations are over the M31 field.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "tile_utils.h"
#include "m31_arith.h"

#define NUM_POLYS      8
#define POLY_LEN       256
#define TILE_ELEMS     32

/*
 * Simple Fiat-Shamir challenge: hash inputs to produce a challenge value.
 * Uses a lightweight hash (not cryptographically secure, for benchmark only).
 * Mixes the input using M31 arithmetic.
 */
static m31_t fiat_shamir_challenge(const m31_t *data, int len, int domain_sep) {
    m31_t hash = (m31_t)domain_sep;
    int i;
    for (i = 0; i < len; i++) {
        /* Mix step: hash = hash * 0x9E3779B9 + data[i] (mod p) */
        hash = m31_mul(hash, 0x9E3779B9U % M31_P);
        hash = m31_add(hash, data[i]);
        /* Additional mixing */
        hash = m31_add(m31_mul(hash, hash), (m31_t)(i + 1));
    }
    return hash;
}

/*
 * Generate multiple challenges from a seed using a chain.
 */
static void generate_challenges(m31_t seed, m31_t *challenges, int n) {
    m31_t state = seed;
    int i;
    for (i = 0; i < n; i++) {
        state = m31_add(m31_mul(state, 0x9E3779B9U % M31_P), (m31_t)(i + 1));
        state = m31_mul(state, state);
        state = m31_add(state, seed);
        /* Ensure non-zero */
        if (state == 0) state = 1;
        challenges[i] = state;
    }
}

/*
 * Tiled linear combination: result[i] = sum_j (alpha_j * poly_j[i])
 * Combines NUM_POLYS polynomials using random challenges alpha_j.
 */
void linear_combine_tiled(const m31_t *polys, int n_polys, int poly_len,
                          const m31_t *alphas, m31_t *result) {
    memset(result, 0, (size_t)poly_len * sizeof(m31_t));

    TILE_FOR(te, 0, poly_len, TILE_ELEMS) {
        int te_end = TILE_END(te, poly_len, TILE_ELEMS);
        int i, j;
        for (i = te; i < te_end; i++) {
            m31_t sum = 0;
            for (j = 0; j < n_polys; j++) {
                sum = m31_add(sum, m31_mul(alphas[j],
                              polys[j * poly_len + i]));
            }
            result[i] = sum;
        }
    }
}

/* Reference linear combination */
void linear_combine_ref(const m31_t *polys, int n_polys, int poly_len,
                        const m31_t *alphas, m31_t *result) {
    memset(result, 0, (size_t)poly_len * sizeof(m31_t));
    int i, j;
    for (i = 0; i < poly_len; i++) {
        m31_t sum = 0;
        for (j = 0; j < n_polys; j++) {
            sum = m31_add(sum, m31_mul(alphas[j],
                          polys[j * poly_len + i]));
        }
        result[i] = sum;
    }
}

/*
 * Full proof composition:
 * 1. Commit polynomials (simplified: hash each polynomial)
 * 2. Generate Fiat-Shamir challenge from commitments
 * 3. Compute random linear combination
 * 4. Output composed polynomial + challenge
 */
void proof_compose(const m31_t *polys, int n_polys, int poly_len,
                   m31_t *composed, m31_t *challenge_out) {
    /* Commit: hash each polynomial */
    m31_t commitments[NUM_POLYS];
    int j;
    for (j = 0; j < n_polys; j++) {
        commitments[j] = fiat_shamir_challenge(
            &polys[j * poly_len], poly_len, j);
    }

    /* Generate challenges from commitments */
    m31_t seed = fiat_shamir_challenge(commitments, n_polys, 0xFF);
    m31_t alphas[NUM_POLYS];
    generate_challenges(seed, alphas, n_polys);

    *challenge_out = seed;

    /* Linear combination */
    linear_combine_tiled(polys, n_polys, poly_len, alphas, composed);
}

int main(void) {
    int n_polys = NUM_POLYS;
    int poly_len = POLY_LEN;

    m31_t *polys = (m31_t *)malloc((size_t)n_polys * poly_len * sizeof(m31_t));
    m31_t *composed = (m31_t *)malloc((size_t)poly_len * sizeof(m31_t));
    m31_t *result_t = (m31_t *)malloc((size_t)poly_len * sizeof(m31_t));
    m31_t *result_r = (m31_t *)malloc((size_t)poly_len * sizeof(m31_t));

    if (!polys || !composed || !result_t || !result_r) {
        fprintf(stderr, "Allocation failed\n");
        return 1;
    }

    /* Generate test polynomial data */
    unsigned int state = 42;
    int i;
    for (i = 0; i < n_polys * poly_len; i++) {
        state = (state * 1103515245 + 12345) & 0x7FFFFFFF;
        polys[i] = (m31_t)(state % 1000);
    }

    /* Test proof composition */
    m31_t challenge;
    proof_compose(polys, n_polys, poly_len, composed, &challenge);

    /* Test determinism: same inputs -> same outputs */
    m31_t *composed2 = (m31_t *)malloc((size_t)poly_len * sizeof(m31_t));
    m31_t challenge2;
    if (composed2) {
        proof_compose(polys, n_polys, poly_len, composed2, &challenge2);
        int determ_ok = 1;
        for (i = 0; i < poly_len; i++) {
            if (composed[i] != composed2[i]) determ_ok = 0;
        }
        if (challenge != challenge2) determ_ok = 0;
        printf("proof_compose: determinism test: %s\n",
               determ_ok ? "OK" : "FAILED");
        free(composed2);
    }

    /* Test tiled vs reference linear combination */
    m31_t alphas[NUM_POLYS];
    generate_challenges(challenge, alphas, n_polys);

    linear_combine_tiled(polys, n_polys, poly_len, alphas, result_t);
    linear_combine_ref(polys, n_polys, poly_len, alphas, result_r);

    int lc_mismatches = 0;
    for (i = 0; i < poly_len; i++) {
        if (result_t[i] != result_r[i]) lc_mismatches++;
    }
    printf("proof_compose: linear_combine mismatches=%d\n", lc_mismatches);

    /* Test Fiat-Shamir: different inputs -> different challenges */
    m31_t polys_mod[NUM_POLYS * POLY_LEN];
    memcpy(polys_mod, polys, (size_t)n_polys * poly_len * sizeof(m31_t));
    polys_mod[0] = m31_add(polys_mod[0], 1); /* modify one element */
    m31_t challenge_mod;
    m31_t *composed_mod = (m31_t *)malloc((size_t)poly_len * sizeof(m31_t));
    int fs_ok = 0;
    if (composed_mod) {
        proof_compose(polys_mod, n_polys, poly_len, composed_mod,
                      &challenge_mod);
        if (challenge != challenge_mod) fs_ok = 1;
        free(composed_mod);
    }
    printf("proof_compose: fiat-shamir sensitivity: %s\n",
           fs_ok ? "OK" : "FAILED");

    printf("proof_compose: challenge=%u, composed[0]=%u\n",
           challenge, composed[0]);

    int pass = (lc_mismatches == 0) && fs_ok;
    printf("proof_compose: %s\n", pass ? "PASS" : "FAIL");

    free(polys); free(composed); free(result_t); free(result_r);
    return pass ? 0 : 1;
}
