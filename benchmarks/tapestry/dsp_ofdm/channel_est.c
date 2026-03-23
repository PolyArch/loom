/*
 * Channel Estimation -- pilot-based channel estimation with linear
 * interpolation for OFDM receiver.
 * 1200 subcarriers, pilot spacing=6 (200 pilots).
 * Tiled: process TILE_SC subcarriers at a time.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "tile_utils.h"
#include "complex.h"

#define NUM_SC        1200
#define PILOT_SPACING 6
#define NUM_PILOTS    (NUM_SC / PILOT_SPACING)
#define TILE_SC       128

void channel_est(const cmplx_t *rx_pilots, const cmplx_t *tx_pilots,
                 cmplx_t *H_est, int num_sc, int pilot_spacing) {
    int num_pilots = num_sc / pilot_spacing;

    /* Estimate channel at pilot positions: H_pilot = Rx / Tx */
    cmplx_t *H_pilot = (cmplx_t *)malloc((size_t)num_pilots * sizeof(cmplx_t));
    if (!H_pilot) return;

    int iter_var0;
    for (iter_var0 = 0; iter_var0 < num_pilots; iter_var0++) {
        H_pilot[iter_var0] = cmplx_div(rx_pilots[iter_var0], tx_pilots[iter_var0]);
    }

    /* Linear interpolation between pilots */
    TILE_FOR(ts, 0, num_sc, TILE_SC) {
        int ts_end = TILE_END(ts, num_sc, TILE_SC);
        int sc;
        for (sc = ts; sc < ts_end; sc++) {
            int pilot_idx = sc / pilot_spacing;
            int sc_in_segment = sc % pilot_spacing;

            if (pilot_idx >= num_pilots - 1) {
                /* Last segment: use last pilot */
                H_est[sc] = H_pilot[num_pilots - 1];
            } else {
                /* Linear interpolation */
                float alpha = (float)sc_in_segment / (float)pilot_spacing;
                cmplx_t h0 = H_pilot[pilot_idx];
                cmplx_t h1 = H_pilot[pilot_idx + 1];
                H_est[sc].re = h0.re + alpha * (h1.re - h0.re);
                H_est[sc].im = h0.im + alpha * (h1.im - h0.im);
            }
        }
    }

    free(H_pilot);
}

void channel_est_ref(const cmplx_t *rx_pilots, const cmplx_t *tx_pilots,
                     cmplx_t *H_est, int num_sc, int pilot_spacing) {
    int num_pilots = num_sc / pilot_spacing;

    cmplx_t *H_pilot = (cmplx_t *)malloc((size_t)num_pilots * sizeof(cmplx_t));
    if (!H_pilot) return;

    int iter_var0;
    for (iter_var0 = 0; iter_var0 < num_pilots; iter_var0++) {
        H_pilot[iter_var0] = cmplx_div(rx_pilots[iter_var0], tx_pilots[iter_var0]);
    }

    int sc;
    for (sc = 0; sc < num_sc; sc++) {
        int pilot_idx = sc / pilot_spacing;
        int sc_in_segment = sc % pilot_spacing;

        if (pilot_idx >= num_pilots - 1) {
            H_est[sc] = H_pilot[num_pilots - 1];
        } else {
            float alpha = (float)sc_in_segment / (float)pilot_spacing;
            cmplx_t h0 = H_pilot[pilot_idx];
            cmplx_t h1 = H_pilot[pilot_idx + 1];
            H_est[sc].re = h0.re + alpha * (h1.re - h0.re);
            H_est[sc].im = h0.im + alpha * (h1.im - h0.im);
        }
    }

    free(H_pilot);
}

int main(void) {
    int num_sc = NUM_SC;
    int pilot_spacing = PILOT_SPACING;
    int num_pilots = NUM_PILOTS;

    cmplx_t *rx_pilots = (cmplx_t *)malloc((size_t)num_pilots * sizeof(cmplx_t));
    cmplx_t *tx_pilots = (cmplx_t *)malloc((size_t)num_pilots * sizeof(cmplx_t));
    cmplx_t *H_tiled   = (cmplx_t *)malloc((size_t)num_sc * sizeof(cmplx_t));
    cmplx_t *H_ref     = (cmplx_t *)malloc((size_t)num_sc * sizeof(cmplx_t));

    if (!rx_pilots || !tx_pilots || !H_tiled || !H_ref) {
        fprintf(stderr, "Allocation failed\n");
        return 1;
    }

    /* Generate test pilots */
    for (int i = 0; i < num_pilots; i++) {
        float phase = (float)i * 0.1f;
        float atten = 0.5f + 0.5f * cosf(phase);
        tx_pilots[i] = cmplx_make(1.0f, 0.0f);
        rx_pilots[i] = cmplx_make(atten * cosf(phase), atten * sinf(phase));
    }

    channel_est(rx_pilots, tx_pilots, H_tiled, num_sc, pilot_spacing);
    channel_est_ref(rx_pilots, tx_pilots, H_ref, num_sc, pilot_spacing);

    float max_err = 0.0f;
    for (int i = 0; i < num_sc; i++) {
        float err = fabsf(H_tiled[i].re - H_ref[i].re) +
                    fabsf(H_tiled[i].im - H_ref[i].im);
        if (err > max_err) max_err = err;
    }

    printf("channel_est: max_error = %e\n", max_err);
    int pass = (max_err < 1e-5f);
    printf("channel_est: %s\n", pass ? "PASS" : "FAIL");

    free(rx_pilots); free(tx_pilots); free(H_tiled); free(H_ref);
    return pass ? 0 : 1;
}
