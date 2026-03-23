/*
 * FFN Second Layer -- matmul for transformer feed-forward network.
 * Computes C[N][D] = A[N][D_ff] * B[D_ff][D]
 * N=seq_len, D_ff=2048, D=512.
 * Tiled: tile_N=32, tile_D=64, tile_Dff=32.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "tile_utils.h"

#define SEQ_LEN   128
#define D_MODEL   512
#define D_FF      2048
#define TILE_N    32
#define TILE_D    64
#define TILE_DFF  32

void ffn2(const float *input, const float *weight, const float *bias,
          float *output, int N, int D_ff, int D) {
    /* Initialize output with bias */
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < D; j++) {
            output[i * D + j] = bias[j];
        }
    }

    TILE_FOR(tn, 0, N, TILE_N) {
        int tn_end = TILE_END(tn, N, TILE_N);
        TILE_FOR(td, 0, D, TILE_D) {
            int td_end = TILE_END(td, D, TILE_D);
            TILE_FOR(tk, 0, D_ff, TILE_DFF) {
                int tk_end = TILE_END(tk, D_ff, TILE_DFF);
                for (int i = tn; i < tn_end; i++) {
                    for (int j = td; j < td_end; j++) {
                        float sum = 0.0f;
                        for (int k = tk; k < tk_end; k++) {
                            sum += input[i * D_ff + k] * weight[k * D + j];
                        }
                        output[i * D + j] += sum;
                    }
                }
            }
        }
    }
}

void ffn2_ref(const float *input, const float *weight, const float *bias,
              float *output, int N, int D_ff, int D) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < D; j++) {
            float sum = bias[j];
            for (int k = 0; k < D_ff; k++) {
                sum += input[i * D_ff + k] * weight[k * D + j];
            }
            output[i * D + j] = sum;
        }
    }
}

int main(void) {
    int N = SEQ_LEN, Dff = D_FF, D = D_MODEL;

    float *input   = (float *)malloc((size_t)N * Dff * sizeof(float));
    float *weight  = (float *)malloc((size_t)Dff * D * sizeof(float));
    float *bias    = (float *)malloc((size_t)D * sizeof(float));
    float *out_t   = (float *)malloc((size_t)N * D * sizeof(float));
    float *out_r   = (float *)malloc((size_t)N * D * sizeof(float));

    if (!input || !weight || !bias || !out_t || !out_r) {
        fprintf(stderr, "Allocation failed\n");
        return 1;
    }

    for (int i = 0; i < N * Dff; i++)
        input[i] = ((float)(i % 79) - 39.0f) / 390.0f;
    for (int i = 0; i < Dff * D; i++)
        weight[i] = ((float)(i % 61) - 30.0f) / 300.0f;
    for (int i = 0; i < D; i++)
        bias[i] = ((float)(i % 37) - 18.0f) / 18.0f;

    ffn2(input, weight, bias, out_t, N, Dff, D);
    ffn2_ref(input, weight, bias, out_r, N, Dff, D);

    float max_err = 0.0f;
    for (int i = 0; i < N * D; i++) {
        float err = fabsf(out_t[i] - out_r[i]);
        if (err > max_err) max_err = err;
    }

    printf("ffn2: max_error = %e\n", max_err);
    int pass = (max_err < 1e-2f);
    printf("ffn2: %s\n", pass ? "PASS" : "FAIL");

    free(input); free(weight); free(bias); free(out_t); free(out_r);
    return pass ? 0 : 1;
}
