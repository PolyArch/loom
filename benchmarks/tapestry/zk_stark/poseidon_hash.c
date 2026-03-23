/*
 * Poseidon Hash Function over M31 field.
 * State width = 12 elements, rate = 8, capacity = 4.
 * Full rounds: S-box (x^5) applied to all elements + MDS + round constant.
 * Partial rounds: S-box applied to first element only + MDS + round constant.
 * Standard configuration: 8 full rounds + 22 partial rounds.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "tile_utils.h"
#include "m31_arith.h"

#define STATE_WIDTH  12
#define RATE         8
#define CAPACITY     (STATE_WIDTH - RATE)
#define FULL_ROUNDS  8
#define PARTIAL_ROUNDS 22
#define TOTAL_ROUNDS (FULL_ROUNDS + PARTIAL_ROUNDS)
#define HALF_FULL    (FULL_ROUNDS / 2)

/* S-box: x^5 mod p */
static inline m31_t sbox(m31_t x) {
    m31_t x2 = m31_mul(x, x);
    m31_t x4 = m31_mul(x2, x2);
    return m31_mul(x4, x);
}

/* Generate deterministic round constants */
static void gen_round_constants(m31_t *rc, int n_rounds, int width) {
    unsigned int state = 12345;
    int i;
    for (i = 0; i < n_rounds * width; i++) {
        state = (state * 1103515245 + 12345) & 0x7FFFFFFF;
        rc[i] = (m31_t)(state % (M31_P - 1)) + 1;
    }
}

/* Generate deterministic MDS matrix (Cauchy construction) */
static void gen_mds_matrix(m31_t mds[STATE_WIDTH][STATE_WIDTH]) {
    int i, j;
    for (i = 0; i < STATE_WIDTH; i++) {
        for (j = 0; j < STATE_WIDTH; j++) {
            /* Cauchy matrix: M[i][j] = 1/(x_i + y_j) where
             * x_i = i+1, y_j = STATE_WIDTH + j + 1 */
            m31_t xi = (m31_t)(i + 1);
            m31_t yj = (m31_t)(STATE_WIDTH + j + 1);
            m31_t sum = m31_add(xi, yj);
            mds[i][j] = m31_inv(sum);
        }
    }
}

/* MDS matrix-vector multiplication */
static void mds_multiply(m31_t state[STATE_WIDTH],
                         const m31_t mds[STATE_WIDTH][STATE_WIDTH]) {
    m31_t tmp[STATE_WIDTH];
    int i, j;
    for (i = 0; i < STATE_WIDTH; i++) {
        m31_t sum = 0;
        for (j = 0; j < STATE_WIDTH; j++) {
            sum = m31_add(sum, m31_mul(mds[i][j], state[j]));
        }
        tmp[i] = sum;
    }
    memcpy(state, tmp, sizeof(tmp));
}

/* Add round constants */
static void add_round_constants(m31_t state[STATE_WIDTH],
                                const m31_t *rc, int round) {
    int i;
    for (i = 0; i < STATE_WIDTH; i++) {
        state[i] = m31_add(state[i], rc[round * STATE_WIDTH + i]);
    }
}

/*
 * Poseidon permutation: apply full and partial rounds.
 * Round structure: HALF_FULL full -> PARTIAL partial -> HALF_FULL full
 */
void poseidon_permutation(m31_t state[STATE_WIDTH],
                          const m31_t mds[STATE_WIDTH][STATE_WIDTH],
                          const m31_t *rc) {
    int r, i;
    int round = 0;

    /* First half of full rounds */
    for (r = 0; r < HALF_FULL; r++) {
        add_round_constants(state, rc, round);
        for (i = 0; i < STATE_WIDTH; i++) {
            state[i] = sbox(state[i]);
        }
        mds_multiply(state, mds);
        round++;
    }

    /* Partial rounds */
    for (r = 0; r < PARTIAL_ROUNDS; r++) {
        add_round_constants(state, rc, round);
        state[0] = sbox(state[0]); /* S-box on first element only */
        mds_multiply(state, mds);
        round++;
    }

    /* Second half of full rounds */
    for (r = 0; r < HALF_FULL; r++) {
        add_round_constants(state, rc, round);
        for (i = 0; i < STATE_WIDTH; i++) {
            state[i] = sbox(state[i]);
        }
        mds_multiply(state, mds);
        round++;
    }
}

/*
 * Poseidon hash: absorb input blocks and squeeze output.
 * Input length must be a multiple of RATE.
 */
void poseidon_hash(const m31_t *input, int input_len, m31_t *output,
                   const m31_t mds[STATE_WIDTH][STATE_WIDTH],
                   const m31_t *rc) {
    m31_t state[STATE_WIDTH];
    int i;

    /* Initialize state to zero */
    for (i = 0; i < STATE_WIDTH; i++) state[i] = 0;

    /* Absorb phase */
    int block;
    for (block = 0; block < input_len; block += RATE) {
        int elems = (block + RATE <= input_len) ? RATE : (input_len - block);
        for (i = 0; i < elems; i++) {
            state[i] = m31_add(state[i], input[block + i]);
        }
        poseidon_permutation(state, mds, rc);
    }

    /* Squeeze: output capacity elements */
    for (i = 0; i < CAPACITY; i++) {
        output[i] = state[i];
    }
}

int main(void) {
    /* Generate MDS matrix and round constants */
    m31_t mds[STATE_WIDTH][STATE_WIDTH];
    m31_t *rc = (m31_t *)malloc(TOTAL_ROUNDS * STATE_WIDTH * sizeof(m31_t));
    if (!rc) {
        fprintf(stderr, "Allocation failed\n");
        return 1;
    }

    gen_mds_matrix(mds);
    gen_round_constants(rc, TOTAL_ROUNDS, STATE_WIDTH);

    /* Test 1: Deterministic - same input gives same output */
    m31_t input1[RATE] = {1, 2, 3, 4, 5, 6, 7, 8};
    m31_t output1a[CAPACITY], output1b[CAPACITY];

    poseidon_hash(input1, RATE, output1a, mds, rc);
    poseidon_hash(input1, RATE, output1b, mds, rc);

    int determ_ok = 1;
    int i;
    for (i = 0; i < CAPACITY; i++) {
        if (output1a[i] != output1b[i]) determ_ok = 0;
    }
    printf("poseidon_hash: determinism test: %s\n",
           determ_ok ? "OK" : "FAILED");

    /* Test 2: Different inputs give different outputs */
    m31_t input2[RATE] = {1, 2, 3, 4, 5, 6, 7, 9}; /* last byte differs */
    m31_t output2[CAPACITY];
    poseidon_hash(input2, RATE, output2, mds, rc);

    int diff_ok = 0;
    for (i = 0; i < CAPACITY; i++) {
        if (output1a[i] != output2[i]) diff_ok = 1;
    }
    printf("poseidon_hash: avalanche test: %s\n",
           diff_ok ? "OK" : "FAILED");

    /* Test 3: S-box correctness (x^5 mod p) */
    int sbox_ok = 1;
    m31_t test_vals[] = {0, 1, 2, 3, 100, 1000, M31_P - 1};
    int n_test = sizeof(test_vals) / sizeof(test_vals[0]);
    for (i = 0; i < n_test; i++) {
        m31_t x = test_vals[i];
        m31_t got = sbox(x);
        m31_t expected = m31_pow(x, 5);
        if (got != expected) {
            printf("poseidon_hash: sbox(%u) = %u, expected %u\n",
                   x, got, expected);
            sbox_ok = 0;
        }
    }
    printf("poseidon_hash: sbox test: %s\n", sbox_ok ? "OK" : "FAILED");

    /* Test 4: Multi-block input */
    m31_t input3[RATE * 2];
    for (i = 0; i < RATE * 2; i++) {
        input3[i] = (m31_t)(i + 10);
    }
    m31_t output3[CAPACITY];
    poseidon_hash(input3, RATE * 2, output3, mds, rc);

    /* Verify output is non-zero */
    int nonzero = 0;
    for (i = 0; i < CAPACITY; i++) {
        if (output3[i] != 0) nonzero = 1;
    }
    printf("poseidon_hash: multi-block test: %s\n",
           nonzero ? "OK" : "FAILED");

    printf("poseidon_hash: output = [%u, %u, %u, %u]\n",
           output1a[0], output1a[1], output1a[2], output1a[3]);

    int pass = determ_ok && diff_ok && sbox_ok && nonzero;
    printf("poseidon_hash: %s\n", pass ? "PASS" : "FAIL");

    free(rc);
    return pass ? 0 : 1;
}
