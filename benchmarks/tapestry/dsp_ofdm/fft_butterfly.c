/*
 * FFT Butterfly -- Cooley-Tukey radix-2 decimation-in-time FFT.
 * N=4096 complex points.
 * In-place computation with bit-reversal permutation.
 * Tiled: process butterflies per stage in tile-sized blocks.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "tile_utils.h"
#include "complex.h"

#define FFT_N       4096
#define TILE_SIZE   256

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Bit-reversal permutation */
static void bit_reverse(cmplx_t *data, int n) {
    int bits = 0;
    int tmp = n;
    while (tmp > 1) { bits++; tmp >>= 1; }

    for (int i = 0; i < n; i++) {
        int j = 0;
        int x = i;
        int iter_var0;
        for (iter_var0 = 0; iter_var0 < bits; iter_var0++) {
            j = (j << 1) | (x & 1);
            x >>= 1;
        }
        if (j > i) {
            cmplx_t temp = data[i];
            data[i] = data[j];
            data[j] = temp;
        }
    }
}

void fft_butterfly(cmplx_t *data, int n) {
    bit_reverse(data, n);

    /* Iterate over FFT stages */
    int stage_size;
    for (stage_size = 2; stage_size <= n; stage_size *= 2) {
        int half = stage_size / 2;
        float angle_step = -2.0f * (float)M_PI / (float)stage_size;

        /* Process butterfly groups in tiles */
        TILE_FOR(tg, 0, n, TILE_SIZE) {
            int tg_end = TILE_END(tg, n, TILE_SIZE);
            int group_start;
            for (group_start = tg - (tg % stage_size);
                 group_start < tg_end;
                 group_start += stage_size) {
                if (group_start + stage_size <= tg) continue;
                int k_start = MAX(0, tg - group_start);
                int k_end_val = MIN(half, tg_end - group_start);
                int k;
                for (k = k_start; k < k_end_val; k++) {
                    int idx = group_start + k;
                    if (idx < 0 || idx >= n || idx + half >= n) continue;
                    float angle = angle_step * (float)k;
                    cmplx_t twiddle = cmplx_make(cosf(angle), sinf(angle));
                    cmplx_t t = cmplx_mul(twiddle, data[idx + half]);
                    cmplx_t u = data[idx];
                    data[idx] = cmplx_add(u, t);
                    data[idx + half] = cmplx_sub(u, t);
                }
            }
        }
    }
}

void fft_butterfly_ref(cmplx_t *data, int n) {
    bit_reverse(data, n);

    int stage_size;
    for (stage_size = 2; stage_size <= n; stage_size *= 2) {
        int half = stage_size / 2;
        float angle_step = -2.0f * (float)M_PI / (float)stage_size;
        int group;
        for (group = 0; group < n; group += stage_size) {
            int k;
            for (k = 0; k < half; k++) {
                float angle = angle_step * (float)k;
                cmplx_t twiddle = cmplx_make(cosf(angle), sinf(angle));
                cmplx_t t = cmplx_mul(twiddle, data[group + k + half]);
                cmplx_t u = data[group + k];
                data[group + k] = cmplx_add(u, t);
                data[group + k + half] = cmplx_sub(u, t);
            }
        }
    }
}

int main(void) {
    int n = FFT_N;
    cmplx_t *data_tiled = (cmplx_t *)malloc((size_t)n * sizeof(cmplx_t));
    cmplx_t *data_ref   = (cmplx_t *)malloc((size_t)n * sizeof(cmplx_t));

    if (!data_tiled || !data_ref) {
        fprintf(stderr, "Allocation failed\n");
        return 1;
    }

    /* Generate test signal: sum of a few sinusoids */
    for (int i = 0; i < n; i++) {
        float t = (float)i / (float)n;
        float val = sinf(2.0f * (float)M_PI * 3.0f * t) +
                    0.5f * cosf(2.0f * (float)M_PI * 7.0f * t);
        data_tiled[i] = cmplx_make(val, 0.0f);
        data_ref[i]   = cmplx_make(val, 0.0f);
    }

    fft_butterfly(data_tiled, n);
    fft_butterfly_ref(data_ref, n);

    float max_err = 0.0f;
    for (int i = 0; i < n; i++) {
        float err_re = fabsf(data_tiled[i].re - data_ref[i].re);
        float err_im = fabsf(data_tiled[i].im - data_ref[i].im);
        float err = (err_re > err_im) ? err_re : err_im;
        if (err > max_err) max_err = err;
    }

    printf("fft_butterfly: max_error = %e\n", max_err);
    int pass = (max_err < 1e-1f);
    printf("fft_butterfly: %s\n", pass ? "PASS" : "FAIL");

    free(data_tiled); free(data_ref);
    return pass ? 0 : 1;
}
