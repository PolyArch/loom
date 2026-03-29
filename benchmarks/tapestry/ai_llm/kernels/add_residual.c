/*
 * Residual Addition -- element-wise addition for skip connections.
 * Computes output[i] = input[i] + residual[i] for all elements.
 * Shape: [N][D] where N=seq_len, D=d_model.
 *
 * Variants: inplace (writes back to residual), separate (new output buffer).
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "tile_utils.h"

#define SEQ_LEN   128
#define D_MODEL   512
#define TILE_SIZE 256

/* --- Primary kernel: separate output buffer --- */

void add_residual(const float *input, const float *residual,
                  float *output, int total) {
    TILE_FOR(t, 0, total, TILE_SIZE) {
        int t_end = TILE_END(t, total, TILE_SIZE);
        for (int i = t; i < t_end; i++) {
            output[i] = input[i] + residual[i];
        }
    }
}

/* --- Variants --- */

/* In-place: accumulate into residual buffer */
void add_residual_inplace(float *residual, const float *input, int total) {
    TILE_FOR(t, 0, total, TILE_SIZE) {
        int t_end = TILE_END(t, total, TILE_SIZE);
        for (int i = t; i < t_end; i++) {
            residual[i] += input[i];
        }
    }
}

/* Separate: identical to primary but as a named variant */
void add_residual_separate(const float *input, const float *residual,
                           float *output, int total) {
    TILE_FOR(t, 0, total, TILE_SIZE) {
        int t_end = TILE_END(t, total, TILE_SIZE);
        for (int i = t; i < t_end; i++) {
            output[i] = input[i] + residual[i];
        }
    }
}

/* --- Reference implementation --- */

void add_residual_ref(const float *input, const float *residual,
                      float *output, int total) {
    for (int i = 0; i < total; i++) {
        output[i] = input[i] + residual[i];
    }
}

/* --- Self-test --- */

int main(void) {
    int total = SEQ_LEN * D_MODEL;

    float *input    = (float *)malloc((size_t)total * sizeof(float));
    float *residual = (float *)malloc((size_t)total * sizeof(float));
    float *out_t    = (float *)malloc((size_t)total * sizeof(float));
    float *out_r    = (float *)malloc((size_t)total * sizeof(float));

    if (!input || !residual || !out_t || !out_r) {
        fprintf(stderr, "Allocation failed\n");
        return 1;
    }

    for (int i = 0; i < total; i++) {
        input[i]    = ((float)(i % 97) - 48.0f) / 48.0f;
        residual[i] = ((float)(i % 71) - 35.0f) / 35.0f;
    }

    add_residual(input, residual, out_t, total);
    add_residual_ref(input, residual, out_r, total);

    float max_err = 0.0f;
    for (int i = 0; i < total; i++) {
        float err = fabsf(out_t[i] - out_r[i]);
        if (err > max_err) max_err = err;
    }

    printf("add_residual: max_error = %e\n", max_err);
    int pass = (max_err == 0.0f);
    printf("add_residual: %s\n", pass ? "PASS" : "FAIL");

    free(input); free(residual); free(out_t); free(out_r);
    return pass ? 0 : 1;
}
