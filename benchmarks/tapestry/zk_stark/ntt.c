/*
 * Number Theoretic Transform (NTT) over the M31 field.
 * Iterative Cooley-Tukey butterfly with pre-computed twiddle factors.
 * N must be a power of 2. Tiled by NTT stages.
 * Core butterfly: a' = a + w*b (mod p), b' = a - w*b (mod p)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "tile_utils.h"
#include "m31_arith.h"

#define NTT_LOG_N   10
#define NTT_N       (1 << NTT_LOG_N)  /* 1024 */
#define TILE_STAGE  1   /* process one stage at a time */

/*
 * Find a primitive root of unity of order n in M31.
 * For M31 (p=2^31-1), the group order is p-1 = 2^31-2.
 * We need n | (p-1). For n = 2^k, this works since 2^k | 2*(2^30-1).
 * g = 3 is a known primitive root of M31.
 */
static m31_t find_root_of_unity(int n) {
    /* g = 3 is a primitive root mod p */
    /* omega_n = g^((p-1)/n) mod p */
    uint32_t exp = (M31_P - 1) / (uint32_t)n;
    return m31_pow(3, exp);
}

/* Bit-reverse permutation */
static void bit_reverse(m31_t *data, int n, int log_n) {
    int i;
    for (i = 0; i < n; i++) {
        int rev = 0;
        int tmp = i;
        int b;
        for (b = 0; b < log_n; b++) {
            rev = (rev << 1) | (tmp & 1);
            tmp >>= 1;
        }
        if (rev > i) {
            m31_t t = data[i];
            data[i] = data[rev];
            data[rev] = t;
        }
    }
}

/* Pre-compute twiddle factors for all stages */
static void precompute_twiddles(m31_t *twiddles, int n) {
    m31_t omega = find_root_of_unity(n);
    twiddles[0] = 1;
    int i;
    for (i = 1; i < n / 2; i++) {
        twiddles[i] = m31_mul(twiddles[i - 1], omega);
    }
}

/* Forward NTT (tiled by stages) */
void ntt_forward_tiled(m31_t *data, int n, int log_n,
                       const m31_t *twiddles) {
    bit_reverse(data, n, log_n);

    int stage;
    for (stage = 0; stage < log_n; stage++) {
        int half_len = 1 << stage;
        int full_len = half_len << 1;
        int stride = n / full_len; /* twiddle stride for this stage */

        TILE_FOR(tk, 0, n, full_len) {
            int j;
            for (j = 0; j < half_len; j++) {
                int idx_a = tk + j;
                int idx_b = tk + j + half_len;
                m31_t w = twiddles[j * stride];
                m31_t wb = m31_mul(w, data[idx_b]);
                m31_t a = data[idx_a];
                data[idx_a] = m31_add(a, wb);
                data[idx_b] = m31_sub(a, wb);
            }
        }
    }
}

/* Pre-compute inverse twiddle factors */
static void precompute_inv_twiddles(m31_t *inv_twiddles, int n) {
    /* omega_inv = omega^(-1) = omega^(n-1) mod p */
    m31_t omega = find_root_of_unity(n);
    m31_t omega_inv = m31_inv(omega);
    inv_twiddles[0] = 1;
    int i;
    for (i = 1; i < n / 2; i++) {
        inv_twiddles[i] = m31_mul(inv_twiddles[i - 1], omega_inv);
    }
}

/* Inverse NTT */
void ntt_inverse_tiled(m31_t *data, int n, int log_n,
                       const m31_t *twiddles) {
    m31_t *inv_tw = (m31_t *)malloc((size_t)(n / 2) * sizeof(m31_t));
    if (!inv_tw) return;

    precompute_inv_twiddles(inv_tw, n);

    (void)twiddles; /* inverse uses its own twiddle table */

    bit_reverse(data, n, log_n);

    int stage;
    for (stage = 0; stage < log_n; stage++) {
        int half_len = 1 << stage;
        int full_len = half_len << 1;
        int stride = n / full_len;

        TILE_FOR(tk, 0, n, full_len) {
            int j;
            for (j = 0; j < half_len; j++) {
                int idx_a = tk + j;
                int idx_b = tk + j + half_len;
                m31_t w = inv_tw[j * stride];
                m31_t wb = m31_mul(w, data[idx_b]);
                m31_t a = data[idx_a];
                data[idx_a] = m31_add(a, wb);
                data[idx_b] = m31_sub(a, wb);
            }
        }
    }

    /* Scale by 1/n */
    m31_t n_inv = m31_inv((m31_t)n);
    int i;
    for (i = 0; i < n; i++) {
        data[i] = m31_mul(data[i], n_inv);
    }

    free(inv_tw);
}

/* Reference (non-tiled) forward NTT */
void ntt_forward_ref(m31_t *data, int n, int log_n,
                     const m31_t *twiddles) {
    bit_reverse(data, n, log_n);

    int stage;
    for (stage = 0; stage < log_n; stage++) {
        int half_len = 1 << stage;
        int full_len = half_len << 1;
        int stride = n / full_len;
        int k, j;

        for (k = 0; k < n; k += full_len) {
            for (j = 0; j < half_len; j++) {
                int idx_a = k + j;
                int idx_b = k + j + half_len;
                m31_t w = twiddles[j * stride];
                m31_t wb = m31_mul(w, data[idx_b]);
                m31_t a = data[idx_a];
                data[idx_a] = m31_add(a, wb);
                data[idx_b] = m31_sub(a, wb);
            }
        }
    }
}

int main(void) {
    int n = NTT_N;
    int log_n = NTT_LOG_N;

    m31_t *data = (m31_t *)malloc((size_t)n * sizeof(m31_t));
    m31_t *orig = (m31_t *)malloc((size_t)n * sizeof(m31_t));
    m31_t *data_ref = (m31_t *)malloc((size_t)n * sizeof(m31_t));
    m31_t *twiddles = (m31_t *)malloc((size_t)(n / 2) * sizeof(m31_t));

    if (!data || !orig || !data_ref || !twiddles) {
        fprintf(stderr, "Allocation failed\n");
        return 1;
    }

    /* Pre-compute twiddle factors */
    precompute_twiddles(twiddles, n);

    /* Generate test data: small values in M31 */
    int i;
    unsigned int state = 42;
    for (i = 0; i < n; i++) {
        state = (state * 1103515245 + 12345) & 0x7FFFFFFF;
        data[i] = (m31_t)(state % 1000);
        orig[i] = data[i];
        data_ref[i] = data[i];
    }

    /* Forward NTT (tiled) */
    ntt_forward_tiled(data, n, log_n, twiddles);

    /* Forward NTT (reference) */
    ntt_forward_ref(data_ref, n, log_n, twiddles);

    /* Verify tiled matches reference */
    int fwd_mismatches = 0;
    for (i = 0; i < n; i++) {
        if (data[i] != data_ref[i]) fwd_mismatches++;
    }

    /* Inverse NTT on tiled result */
    ntt_inverse_tiled(data, n, log_n, twiddles);

    /* Verify round-trip: INTT(NTT(x)) == x */
    int rt_mismatches = 0;
    for (i = 0; i < n; i++) {
        if (data[i] != orig[i]) rt_mismatches++;
    }

    printf("ntt: n=%d, fwd_mismatches=%d, roundtrip_mismatches=%d\n",
           n, fwd_mismatches, rt_mismatches);

    /* Also verify M31 arithmetic on small values */
    int arith_ok = 1;
    m31_t a, b;
    for (a = 0; a < 100; a++) {
        for (b = 0; b < 100; b++) {
            uint64_t expected = ((uint64_t)a * (uint64_t)b) % M31_P;
            m31_t got = m31_mul(a, b);
            if (got != (m31_t)expected) {
                printf("ntt: m31_mul(%u,%u) = %u, expected %llu\n",
                       a, b, got, (unsigned long long)expected);
                arith_ok = 0;
            }
        }
    }
    printf("ntt: m31_arith exhaustive test (0..99): %s\n",
           arith_ok ? "OK" : "FAILED");

    int pass = (fwd_mismatches == 0) && (rt_mismatches == 0) && arith_ok;
    printf("ntt: %s\n", pass ? "PASS" : "FAIL");

    free(data); free(orig); free(data_ref); free(twiddles);
    return pass ? 0 : 1;
}
