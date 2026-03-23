/*
 * Layer Normalization for transformer.
 * For each token (row): compute mean, variance, normalize, scale+bias.
 * N=seq_len, D=512.
 * Tiled: process TILE_N=32 rows at a time.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "tile_utils.h"

#define SEQ_LEN    128
#define D_MODEL    512
#define TILE_N     32
#define LN_EPS     1e-5f

void layernorm(const float *input, const float *gamma, const float *beta,
               float *output, int N, int D) {
    TILE_FOR(tn, 0, N, TILE_N) {
        int tn_end = TILE_END(tn, N, TILE_N);
        for (int i = tn; i < tn_end; i++) {
            const float *row_in = input + (size_t)i * D;
            float *row_out = output + (size_t)i * D;

            /* Compute mean */
            float mean = 0.0f;
            for (int j = 0; j < D; j++) {
                mean += row_in[j];
            }
            mean /= (float)D;

            /* Compute variance */
            float var = 0.0f;
            for (int j = 0; j < D; j++) {
                float diff = row_in[j] - mean;
                var += diff * diff;
            }
            var /= (float)D;

            /* Normalize and apply scale+bias */
            float inv_std = 1.0f / sqrtf(var + LN_EPS);
            for (int j = 0; j < D; j++) {
                row_out[j] = (row_in[j] - mean) * inv_std * gamma[j] + beta[j];
            }
        }
    }
}

void layernorm_ref(const float *input, const float *gamma, const float *beta,
                   float *output, int N, int D) {
    for (int i = 0; i < N; i++) {
        const float *row_in = input + (size_t)i * D;
        float *row_out = output + (size_t)i * D;

        float mean = 0.0f;
        for (int j = 0; j < D; j++) {
            mean += row_in[j];
        }
        mean /= (float)D;

        float var = 0.0f;
        for (int j = 0; j < D; j++) {
            float diff = row_in[j] - mean;
            var += diff * diff;
        }
        var /= (float)D;

        float inv_std = 1.0f / sqrtf(var + LN_EPS);
        for (int j = 0; j < D; j++) {
            row_out[j] = (row_in[j] - mean) * inv_std * gamma[j] + beta[j];
        }
    }
}

int main(void) {
    int N = SEQ_LEN, D = D_MODEL;

    float *input  = (float *)malloc((size_t)N * D * sizeof(float));
    float *gamma  = (float *)malloc((size_t)D * sizeof(float));
    float *beta   = (float *)malloc((size_t)D * sizeof(float));
    float *out_t  = (float *)malloc((size_t)N * D * sizeof(float));
    float *out_r  = (float *)malloc((size_t)N * D * sizeof(float));

    if (!input || !gamma || !beta || !out_t || !out_r) {
        fprintf(stderr, "Allocation failed\n");
        return 1;
    }

    for (int i = 0; i < N * D; i++)
        input[i] = ((float)(i % 113) - 56.0f) / 56.0f;
    for (int i = 0; i < D; i++) {
        gamma[i] = 0.8f + 0.4f * ((float)(i % 17) / 17.0f);
        beta[i]  = ((float)(i % 23) - 11.0f) / 110.0f;
    }

    layernorm(input, gamma, beta, out_t, N, D);
    layernorm_ref(input, gamma, beta, out_r, N, D);

    float max_err = 0.0f;
    for (int i = 0; i < N * D; i++) {
        float err = fabsf(out_t[i] - out_r[i]);
        if (err > max_err) max_err = err;
    }

    printf("layernorm: max_error = %e\n", max_err);
    int pass = (max_err < 1e-5f);
    printf("layernorm: %s\n", pass ? "PASS" : "FAIL");

    free(input); free(gamma); free(beta); free(out_t); free(out_r);
    return pass ? 0 : 1;
}
