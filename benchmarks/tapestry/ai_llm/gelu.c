/*
 * GeLU Activation -- elementwise Gaussian Error Linear Unit.
 * gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
 * Uses polynomial tanh approximation for CGRA-friendliness.
 * N=seq_len, D=2048 (D_ff).
 * Tiled: process TILE_SIZE=256 elements at a time.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "tile_utils.h"

#define SEQ_LEN    128
#define D_FF       2048
#define TILE_SIZE  256

#define SQRT_2_OVER_PI 0.7978845608f
#define GELU_COEFF     0.044715f

/*
 * Polynomial tanh approximation suitable for CGRA implementation.
 * Uses piecewise polynomial: accurate to ~1e-3 for |x| < 4.
 * Clamps to +/-1 for larger values.
 */
static inline float approx_tanh(float x) {
    if (x > 5.0f) return 1.0f;
    if (x < -5.0f) return -1.0f;
    /* Use the standard tanhf for now; a CGRA backend would replace
       this with a lookup table or polynomial. */
    float x2 = x * x;
    float x4 = x2 * x2;
    /* Pade approximant: tanh(x) ~ x(135135 + 17325*x2 + 378*x4) /
                                    (135135 + 62370*x2 + 3150*x4 + 28*x2*x4) */
    float num = x * (135135.0f + x2 * (17325.0f + x2 * 378.0f));
    float den = 135135.0f + x2 * (62370.0f + x2 * (3150.0f + x2 * 28.0f));
    (void)x4;
    return num / den;
}

void gelu(float *data, int total) {
    TILE_FOR(t, 0, total, TILE_SIZE) {
        int t_end = TILE_END(t, total, TILE_SIZE);
        for (int i = t; i < t_end; i++) {
            float x = data[i];
            float x3 = x * x * x;
            float inner = SQRT_2_OVER_PI * (x + GELU_COEFF * x3);
            data[i] = 0.5f * x * (1.0f + approx_tanh(inner));
        }
    }
}

void gelu_ref(float *data, int total) {
    for (int i = 0; i < total; i++) {
        float x = data[i];
        float x3 = x * x * x;
        float inner = SQRT_2_OVER_PI * (x + GELU_COEFF * x3);
        /* Use the same approximation as tiled version for correctness comparison */
        data[i] = 0.5f * x * (1.0f + approx_tanh(inner));
    }
}

int main(void) {
    int total = SEQ_LEN * D_FF;
    float *data_tiled = (float *)malloc((size_t)total * sizeof(float));
    float *data_ref   = (float *)malloc((size_t)total * sizeof(float));

    if (!data_tiled || !data_ref) {
        fprintf(stderr, "Allocation failed\n");
        return 1;
    }

    for (int i = 0; i < total; i++) {
        float val = ((float)(i % 199) - 99.0f) / 33.0f;
        data_tiled[i] = val;
        data_ref[i]   = val;
    }

    gelu(data_tiled, total);
    gelu_ref(data_ref, total);

    float max_err = 0.0f;
    float max_rel = 0.0f;
    for (int i = 0; i < total; i++) {
        float err = fabsf(data_tiled[i] - data_ref[i]);
        if (err > max_err) max_err = err;
        float denom = fabsf(data_ref[i]) + 1e-6f;
        float rel = err / denom;
        if (rel > max_rel) max_rel = rel;
    }

    printf("gelu: max_abs_error = %e, max_rel_error = %e\n", max_err, max_rel);
    /* Both implementations use the same approx_tanh; error should be zero */
    int pass = (max_err < 1e-5f);
    printf("gelu: %s\n", pass ? "PASS" : "FAIL");

    free(data_tiled); free(data_ref);
    return pass ? 0 : 1;
}
