/*
 * Zero-Forcing Equalization for OFDM receiver.
 * Computes X_eq[k] = Y[k] / H[k] for each subcarrier k.
 * 1200 subcarriers, complex division.
 * Tiled: process TILE_SC subcarriers at a time.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "tile_utils.h"
#include "complex.h"

#define NUM_SC     1200
#define TILE_SC    128

void equalizer(const cmplx_t *Y, const cmplx_t *H, cmplx_t *X,
               int num_sc) {
    TILE_FOR(ts, 0, num_sc, TILE_SC) {
        int ts_end = TILE_END(ts, num_sc, TILE_SC);
        int k;
        for (k = ts; k < ts_end; k++) {
            X[k] = cmplx_div(Y[k], H[k]);
        }
    }
}

void equalizer_ref(const cmplx_t *Y, const cmplx_t *H, cmplx_t *X,
                   int num_sc) {
    int k;
    for (k = 0; k < num_sc; k++) {
        X[k] = cmplx_div(Y[k], H[k]);
    }
}

int main(void) {
    int num_sc = NUM_SC;
    cmplx_t *Y      = (cmplx_t *)malloc((size_t)num_sc * sizeof(cmplx_t));
    cmplx_t *H      = (cmplx_t *)malloc((size_t)num_sc * sizeof(cmplx_t));
    cmplx_t *X_tile = (cmplx_t *)malloc((size_t)num_sc * sizeof(cmplx_t));
    cmplx_t *X_ref  = (cmplx_t *)malloc((size_t)num_sc * sizeof(cmplx_t));

    if (!Y || !H || !X_tile || !X_ref) {
        fprintf(stderr, "Allocation failed\n");
        return 1;
    }

    /* Generate test data: known transmitted + channel response */
    for (int k = 0; k < num_sc; k++) {
        float phase_h = 0.3f * (float)k / (float)num_sc;
        float mag_h = 0.5f + 0.5f * cosf(2.0f * 3.14159f * (float)k / 200.0f);
        if (mag_h < 0.1f) mag_h = 0.1f;
        H[k] = cmplx_make(mag_h * cosf(phase_h), mag_h * sinf(phase_h));

        /* Transmitted symbol (QPSK-like) */
        float tx_re = (k % 2 == 0) ? 1.0f : -1.0f;
        float tx_im = (k % 3 == 0) ? 1.0f : -1.0f;
        cmplx_t tx = cmplx_make(tx_re, tx_im);

        /* Received = H * X + noise(small) */
        Y[k] = cmplx_mul(H[k], tx);
    }

    equalizer(Y, H, X_tile, num_sc);
    equalizer_ref(Y, H, X_ref, num_sc);

    float max_err = 0.0f;
    for (int k = 0; k < num_sc; k++) {
        float err = fabsf(X_tile[k].re - X_ref[k].re) +
                    fabsf(X_tile[k].im - X_ref[k].im);
        if (err > max_err) max_err = err;
    }

    printf("equalizer: max_error = %e\n", max_err);
    int pass = (max_err < 1e-5f);
    printf("equalizer: %s\n", pass ? "PASS" : "FAIL");

    free(Y); free(H); free(X_tile); free(X_ref);
    return pass ? 0 : 1;
}
