/*
 * Viterbi Decoder -- convolutional code decoder for OFDM receiver.
 * Constraint length K=7, rate 1/2, 64 states.
 * Processes soft input bits, outputs hard-decision decoded bits.
 * Tiled: per-traceback-window (TRACEBACK_LEN).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "tile_utils.h"

#define CONSTRAINT_LEN  7
#define NUM_STATES      64
#define RATE_INV        2
#define TRACEBACK_LEN   35
#define INPUT_BITS      3600
#define OUTPUT_BITS     (INPUT_BITS / RATE_INV)

/* Generator polynomials for K=7, rate 1/2 (octal: 171, 133) */
#define GEN_POLY_0  0x6D  /* 1101101 = 109 */
#define GEN_POLY_1  0x4F  /* 1001111 = 79  */

/* Count set bits (popcount) */
static int popcount(int x) {
    int count = 0;
    while (x) {
        count += x & 1;
        x >>= 1;
    }
    return count;
}

/* Compute expected output bits for a given state and input */
static void encoder_output(int state, int input_bit, int *out0, int *out1) {
    int reg = (input_bit << (CONSTRAINT_LEN - 1)) | state;
    *out0 = popcount(reg & GEN_POLY_0) & 1;
    *out1 = popcount(reg & GEN_POLY_1) & 1;
}

void viterbi_decode(const int *input_bits, int *output_bits,
                    int num_input, int num_output) {
    (void)num_input; /* num_output * RATE_INV = num_input */
    int *path_metric = (int *)calloc((size_t)NUM_STATES, sizeof(int));
    int *new_metric  = (int *)malloc((size_t)NUM_STATES * sizeof(int));
    /* Survivor paths: for each time step and state, store the previous state */
    int *survivor    = (int *)malloc((size_t)num_output * NUM_STATES * sizeof(int));

    if (!path_metric || !new_metric || !survivor) {
        free(path_metric); free(new_metric); free(survivor);
        return;
    }

    /* Initialize: state 0 starts at metric 0, others at large value */
    int iter_var0;
    for (iter_var0 = 1; iter_var0 < NUM_STATES; iter_var0++) {
        path_metric[iter_var0] = 10000;
    }

    /* Forward pass: ACS (Add-Compare-Select) */
    int t;
    for (t = 0; t < num_output; t++) {
        int rx_bit0 = input_bits[t * RATE_INV + 0];
        int rx_bit1 = input_bits[t * RATE_INV + 1];

        int s;
        for (s = 0; s < NUM_STATES; s++) {
            new_metric[s] = 100000;
        }

        for (s = 0; s < NUM_STATES; s++) {
            /* Try input 0 and input 1 */
            int input_bit;
            for (input_bit = 0; input_bit < 2; input_bit++) {
                int next_state = ((s >> 1) | (input_bit << (CONSTRAINT_LEN - 2)));
                int exp0, exp1;
                encoder_output(s, input_bit, &exp0, &exp1);
                int branch_metric = (rx_bit0 ^ exp0) + (rx_bit1 ^ exp1);
                int total = path_metric[s] + branch_metric;
                if (total < new_metric[next_state]) {
                    new_metric[next_state] = total;
                    survivor[t * NUM_STATES + next_state] = s;
                }
            }
        }

        memcpy(path_metric, new_metric, (size_t)NUM_STATES * sizeof(int));
    }

    /* Traceback: find best final state */
    int best_state = 0;
    for (iter_var0 = 1; iter_var0 < NUM_STATES; iter_var0++) {
        if (path_metric[iter_var0] < path_metric[best_state]) {
            best_state = iter_var0;
        }
    }

    /* Trace back through survivor paths */
    int state = best_state;
    for (t = num_output - 1; t >= 0; t--) {
        int prev_state = survivor[t * NUM_STATES + state];
        /* The input bit that caused this transition */
        output_bits[t] = (state >> (CONSTRAINT_LEN - 2)) & 1;
        state = prev_state;
    }

    free(path_metric); free(new_metric); free(survivor);
}

/* Simple convolutional encoder for test data generation */
void conv_encode(const int *input, int *output, int num_bits) {
    int shift_reg = 0;
    int t;
    for (t = 0; t < num_bits; t++) {
        shift_reg = ((shift_reg >> 1) | (input[t] << (CONSTRAINT_LEN - 1))) &
                    ((1 << CONSTRAINT_LEN) - 1);
        output[t * RATE_INV + 0] = popcount(shift_reg & GEN_POLY_0) & 1;
        output[t * RATE_INV + 1] = popcount(shift_reg & GEN_POLY_1) & 1;
    }
}

int main(void) {
    int num_output = OUTPUT_BITS;
    int num_input  = INPUT_BITS;

    int *orig_bits   = (int *)malloc((size_t)num_output * sizeof(int));
    int *encoded     = (int *)malloc((size_t)num_input * sizeof(int));
    int *decoded     = (int *)malloc((size_t)num_output * sizeof(int));

    if (!orig_bits || !encoded || !decoded) {
        fprintf(stderr, "Allocation failed\n");
        return 1;
    }

    /* Generate random-ish data */
    for (int i = 0; i < num_output; i++) {
        orig_bits[i] = ((i * 37 + 13) % 100) > 49 ? 1 : 0;
    }

    conv_encode(orig_bits, encoded, num_output);
    viterbi_decode(encoded, decoded, num_input, num_output);

    /* Compare (skip first CONSTRAINT_LEN bits due to encoder startup) */
    int errors = 0;
    int compared = 0;
    for (int i = CONSTRAINT_LEN; i < num_output - CONSTRAINT_LEN; i++) {
        if (decoded[i] != orig_bits[i]) errors++;
        compared++;
    }

    float ber = (compared > 0) ? (float)errors / (float)compared : 0.0f;
    printf("viterbi: bit_errors = %d / %d (BER = %e)\n", errors, compared, ber);
    int pass = (ber < 0.01f);
    printf("viterbi: %s\n", pass ? "PASS" : "FAIL");

    free(orig_bits); free(encoded); free(decoded);
    return pass ? 0 : 1;
}
