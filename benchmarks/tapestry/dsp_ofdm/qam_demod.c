/*
 * 64-QAM Demodulation for OFDM receiver.
 * Maps 1200 complex subcarrier symbols to 7200 hard-decision bits.
 * 64-QAM: 6 bits per symbol, 8-level PAM per axis (3 bits I, 3 bits Q).
 * Tiled: process TILE_SC subcarriers at a time.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "tile_utils.h"
#include "complex.h"

#define NUM_SC       1200
#define BITS_PER_SYM 6
#define NUM_BITS     (NUM_SC * BITS_PER_SYM)
#define TILE_SC      128

/* 64-QAM constellation levels for one axis (Gray-coded) */
static const float qam_levels[8] = {
    -7.0f, -5.0f, -3.0f, -1.0f, 1.0f, 3.0f, 5.0f, 7.0f
};

/* Gray code mapping for 3 bits */
static const int gray_map[8] = { 0, 1, 3, 2, 6, 7, 5, 4 };

/* Find closest constellation point for one axis (hard decision) */
static int slice_axis(float val) {
    int best = 0;
    float best_dist = fabsf(val - qam_levels[0]);
    int iter_var0;
    for (iter_var0 = 1; iter_var0 < 8; iter_var0++) {
        float dist = fabsf(val - qam_levels[iter_var0]);
        if (dist < best_dist) {
            best_dist = dist;
            best = iter_var0;
        }
    }
    return best;
}

void qam_demod(const cmplx_t *symbols, int *bits, int num_sc) {
    /* Normalization factor for 64-QAM: 1/sqrt(42) */
    float norm = sqrtf(42.0f);

    TILE_FOR(ts, 0, num_sc, TILE_SC) {
        int ts_end = TILE_END(ts, num_sc, TILE_SC);
        int k;
        for (k = ts; k < ts_end; k++) {
            /* Denormalize */
            float i_val = symbols[k].re * norm;
            float q_val = symbols[k].im * norm;

            /* Slice to nearest constellation point */
            int i_idx = slice_axis(i_val);
            int q_idx = slice_axis(q_val);

            /* Gray-coded bits */
            int i_bits = gray_map[i_idx];
            int q_bits = gray_map[q_idx];

            /* Pack 6 bits: 3 from I, 3 from Q */
            int base = k * BITS_PER_SYM;
            bits[base + 0] = (i_bits >> 2) & 1;
            bits[base + 1] = (i_bits >> 1) & 1;
            bits[base + 2] = (i_bits >> 0) & 1;
            bits[base + 3] = (q_bits >> 2) & 1;
            bits[base + 4] = (q_bits >> 1) & 1;
            bits[base + 5] = (q_bits >> 0) & 1;
        }
    }
}

void qam_demod_ref(const cmplx_t *symbols, int *bits, int num_sc) {
    float norm = sqrtf(42.0f);
    int k;
    for (k = 0; k < num_sc; k++) {
        float i_val = symbols[k].re * norm;
        float q_val = symbols[k].im * norm;
        int i_idx = slice_axis(i_val);
        int q_idx = slice_axis(q_val);
        int i_bits = gray_map[i_idx];
        int q_bits = gray_map[q_idx];
        int base = k * BITS_PER_SYM;
        bits[base + 0] = (i_bits >> 2) & 1;
        bits[base + 1] = (i_bits >> 1) & 1;
        bits[base + 2] = (i_bits >> 0) & 1;
        bits[base + 3] = (q_bits >> 2) & 1;
        bits[base + 4] = (q_bits >> 1) & 1;
        bits[base + 5] = (q_bits >> 0) & 1;
    }
}

int main(void) {
    int num_sc = NUM_SC;
    int num_bits = NUM_BITS;
    float norm_inv = 1.0f / sqrtf(42.0f);

    cmplx_t *symbols   = (cmplx_t *)malloc((size_t)num_sc * sizeof(cmplx_t));
    int     *bits_tile = (int *)malloc((size_t)num_bits * sizeof(int));
    int     *bits_ref  = (int *)malloc((size_t)num_bits * sizeof(int));

    if (!symbols || !bits_tile || !bits_ref) {
        fprintf(stderr, "Allocation failed\n");
        return 1;
    }

    /* Generate valid 64-QAM symbols */
    for (int k = 0; k < num_sc; k++) {
        int i_level = (k * 7 + 3) % 8;
        int q_level = (k * 5 + 1) % 8;
        symbols[k] = cmplx_make(qam_levels[i_level] * norm_inv,
                                qam_levels[q_level] * norm_inv);
    }

    qam_demod(symbols, bits_tile, num_sc);
    qam_demod_ref(symbols, bits_ref, num_sc);

    int errors = 0;
    for (int i = 0; i < num_bits; i++) {
        if (bits_tile[i] != bits_ref[i]) errors++;
    }

    printf("qam_demod: bit_errors = %d / %d\n", errors, num_bits);
    int pass = (errors == 0);
    printf("qam_demod: %s\n", pass ? "PASS" : "FAIL");

    free(symbols); free(bits_tile); free(bits_ref);
    return pass ? 0 : 1;
}
