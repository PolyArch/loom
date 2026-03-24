/*
 * Entry function for auto_analyze: STARK Proof pipeline.
 * Contains calls to all kernel functions with shared buffer arguments.
 * auto_analyze should detect 5 kernels and 5 edges.
 */

#include <stdlib.h>
#include <string.h>

typedef unsigned int m31_t;
#define M31_P 0x7FFFFFFFu

#define NTT_N       1024
#define NTT_LOG_N   10
#define NUM_QUERIES 256
#define HASH_RATE   4
#define MSM_SIZE    1024

static m31_t m31_add(m31_t a, m31_t b) {
    unsigned long long s = (unsigned long long)a + b;
    return (m31_t)(s >= M31_P ? s - M31_P : s);
}

static m31_t m31_mul(m31_t a, m31_t b) {
    unsigned long long p = (unsigned long long)a * b;
    return (m31_t)(p % M31_P);
}

__attribute__((noinline))
void ntt_forward(m31_t *data, int n, int log_n, const m31_t *twiddles) {
    int stage, j;
    for (stage = 0; stage < log_n; stage++) {
        int half = 1 << stage;
        int full = half << 1;
        int stride = n / full;
        int k;
        for (k = 0; k < n; k += full) {
            for (j = 0; j < half; j++) {
                m31_t w = twiddles[j * stride];
                m31_t wb = m31_mul(w, data[k+j+half]);
                m31_t a = data[k+j];
                data[k+j] = m31_add(a, wb);
                data[k+j+half] = m31_add(a, M31_P - wb);
            }
        }
    }
}

__attribute__((noinline))
void msm(const m31_t *scalars, const m31_t *bases, m31_t *result, int n) {
    int i;
    result[0] = 0; result[1] = 0; result[2] = 1;
    for (i = 0; i < n; i++) {
        m31_t s = scalars[i];
        result[0] = m31_add(result[0], m31_mul(s, bases[i*3+0]));
        result[1] = m31_add(result[1], m31_mul(s, bases[i*3+1]));
        result[2] = m31_mul(result[2], bases[i*3+2]);
    }
}

__attribute__((noinline))
void poseidon_hash(const m31_t *input, m31_t *output,
                   int n_blocks, int rate) {
    int b, r;
    m31_t state[4] = {0, 0, 0, 0};
    for (b = 0; b < n_blocks; b++) {
        for (r = 0; r < rate; r++)
            state[r] = m31_add(state[r], input[b * rate + r]);
        /* Simplified permutation */
        int i;
        for (i = 0; i < 4; i++)
            state[i] = m31_mul(state[i], m31_mul(state[i], state[i]));
    }
    int i;
    for (i = 0; i < rate; i++) output[i] = state[i];
}

__attribute__((noinline))
void poly_eval(const m31_t *coeffs, const m31_t *points,
               m31_t *results, int degree, int n_points) {
    int p, d;
    for (p = 0; p < n_points; p++) {
        m31_t val = coeffs[degree - 1];
        for (d = degree - 2; d >= 0; d--)
            val = m31_add(m31_mul(val, points[p]), coeffs[d]);
        results[p] = val;
    }
}

__attribute__((noinline))
void proof_compose(const m31_t *poly_vals, const m31_t *hash_out,
                   const m31_t *msm_out, m31_t *proof, int n) {
    int i;
    for (i = 0; i < n; i++)
        proof[i] = m31_add(poly_vals[i], m31_mul(hash_out[i % 4], msm_out[i % 3]));
}

/* Entry function for auto_analyze */
void stark_pipeline(m31_t *trace, const m31_t *twiddles,
                    const m31_t *scalars, const m31_t *bases,
                    const m31_t *query_pts, m31_t *proof) {
    m31_t *ntt_out = (m31_t *)malloc(NTT_N * sizeof(m31_t));
    m31_t *poly_vals = (m31_t *)malloc(NUM_QUERIES * sizeof(m31_t));
    m31_t hash_out[HASH_RATE];
    m31_t msm_result[3];

    memcpy(ntt_out, trace, NTT_N * sizeof(m31_t));
    ntt_forward(ntt_out, NTT_N, NTT_LOG_N, twiddles);
    msm(scalars, bases, msm_result, MSM_SIZE);
    poseidon_hash(ntt_out, hash_out, NTT_N / HASH_RATE, HASH_RATE);
    poly_eval(ntt_out, query_pts, poly_vals, NTT_N, NUM_QUERIES);
    proof_compose(poly_vals, hash_out, msm_result, proof, NUM_QUERIES);

    free(ntt_out); free(poly_vals);
}
