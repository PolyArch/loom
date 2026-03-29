/*
 * Expert FFN -- per-expert feed-forward network for MoE.
 * Composed kernel: up-project -> activation -> down-project.
 * Operates on a gathered subset of B_e tokens assigned to expert e.
 * B_e=tokens_per_expert, D=512, D_ff=2048.
 *
 * Variants: full, narrow (D_ff/2), shared_up, tile64.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "tile_utils.h"

#define D_MODEL      512
#define D_FF         2048
#define TOKENS_PER_E 32
#define TILE_N       16
#define TILE_D       32

#define SQRT_2_OVER_PI 0.7978845608f
#define GELU_COEFF     0.044715f

static inline float approx_tanh(float x) {
    if (x > 5.0f) return 1.0f;
    if (x < -5.0f) return -1.0f;
    float x2 = x * x;
    float num = x * (135135.0f + x2 * (17325.0f + x2 * 378.0f));
    float den = 135135.0f + x2 * (62370.0f + x2 * (3150.0f + x2 * 28.0f));
    return num / den;
}

static inline float gelu_fn(float x) {
    float x3 = x * x * x;
    float inner = SQRT_2_OVER_PI * (x + GELU_COEFF * x3);
    return 0.5f * x * (1.0f + approx_tanh(inner));
}

/* --- Core: composed FFN (up + gelu + down) --- */

static void expert_ffn_core(const float *input,
                            const float *w_up, const float *b_up,
                            const float *w_down, const float *b_down,
                            float *output, float *hidden,
                            int B_e, int D, int D_ff,
                            int tile_n, int tile_d) {
    /* Up-project: hidden[B_e][D_ff] = input[B_e][D] * w_up[D][D_ff] + b_up */
    for (int i = 0; i < B_e; i++)
        for (int j = 0; j < D_ff; j++)
            hidden[i * D_ff + j] = b_up[j];

    TILE_FOR(tn, 0, B_e, tile_n) {
        int tn_end = TILE_END(tn, B_e, tile_n);
        TILE_FOR(td, 0, D_ff, tile_d) {
            int td_end = TILE_END(td, D_ff, tile_d);
            TILE_FOR(tk, 0, D, tile_d) {
                int tk_end = TILE_END(tk, D, tile_d);
                for (int i = tn; i < tn_end; i++) {
                    for (int j = td; j < td_end; j++) {
                        float sum = 0.0f;
                        for (int k = tk; k < tk_end; k++) {
                            sum += input[i * D + k] * w_up[k * D_ff + j];
                        }
                        hidden[i * D_ff + j] += sum;
                    }
                }
            }
        }
    }

    /* Activation (GeLU) */
    for (int i = 0; i < B_e * D_ff; i++) {
        hidden[i] = gelu_fn(hidden[i]);
    }

    /* Down-project: output[B_e][D] = hidden[B_e][D_ff] * w_down[D_ff][D] + b_down */
    for (int i = 0; i < B_e; i++)
        for (int j = 0; j < D; j++)
            output[i * D + j] = b_down[j];

    TILE_FOR(tn, 0, B_e, tile_n) {
        int tn_end = TILE_END(tn, B_e, tile_n);
        TILE_FOR(td, 0, D, tile_d) {
            int td_end = TILE_END(td, D, tile_d);
            TILE_FOR(tk, 0, D_ff, tile_d) {
                int tk_end = TILE_END(tk, D_ff, tile_d);
                for (int i = tn; i < tn_end; i++) {
                    for (int j = td; j < td_end; j++) {
                        float sum = 0.0f;
                        for (int k = tk; k < tk_end; k++) {
                            sum += hidden[i * D_ff + k] * w_down[k * D + j];
                        }
                        output[i * D + j] += sum;
                    }
                }
            }
        }
    }
}

void expert_ffn(const float *input,
                const float *w_up, const float *b_up,
                const float *w_down, const float *b_down,
                float *output, int B_e, int D, int D_ff) {
    float *hidden = (float *)malloc((size_t)B_e * D_ff * sizeof(float));
    if (!hidden) return;
    expert_ffn_core(input, w_up, b_up, w_down, b_down,
                    output, hidden, B_e, D, D_ff, TILE_N, TILE_D);
    free(hidden);
}

/* --- Variants --- */

void expert_ffn_full(const float *input,
                     const float *w_up, const float *b_up,
                     const float *w_down, const float *b_down,
                     float *output, int B_e, int D, int D_ff) {
    expert_ffn(input, w_up, b_up, w_down, b_down, output, B_e, D, D_ff);
}

/* Narrow: half the hidden dimension */
void expert_ffn_narrow(const float *input,
                       const float *w_up, const float *b_up,
                       const float *w_down, const float *b_down,
                       float *output, int B_e, int D, int D_ff) {
    int narrow_ff = D_ff / 2;
    float *hidden = (float *)malloc((size_t)B_e * narrow_ff * sizeof(float));
    if (!hidden) return;
    expert_ffn_core(input, w_up, b_up, w_down, b_down,
                    output, hidden, B_e, D, narrow_ff, TILE_N, TILE_D);
    free(hidden);
}

/* Shared up-projection: assumes w_up is shared across experts */
void expert_ffn_shared_up(const float *input,
                          const float *w_up, const float *b_up,
                          const float *w_down, const float *b_down,
                          float *output, int B_e, int D, int D_ff) {
    expert_ffn(input, w_up, b_up, w_down, b_down, output, B_e, D, D_ff);
}

void expert_ffn_tile64(const float *input,
                       const float *w_up, const float *b_up,
                       const float *w_down, const float *b_down,
                       float *output, int B_e, int D, int D_ff) {
    float *hidden = (float *)malloc((size_t)B_e * D_ff * sizeof(float));
    if (!hidden) return;
    expert_ffn_core(input, w_up, b_up, w_down, b_down,
                    output, hidden, B_e, D, D_ff, 32, 64);
    free(hidden);
}

/* --- Reference implementation --- */

void expert_ffn_ref(const float *input,
                    const float *w_up, const float *b_up,
                    const float *w_down, const float *b_down,
                    float *output, int B_e, int D, int D_ff) {
    float *hidden = (float *)malloc((size_t)B_e * D_ff * sizeof(float));
    if (!hidden) return;

    /* Up-project */
    for (int i = 0; i < B_e; i++) {
        for (int j = 0; j < D_ff; j++) {
            float sum = b_up[j];
            for (int k = 0; k < D; k++) {
                sum += input[i * D + k] * w_up[k * D_ff + j];
            }
            hidden[i * D_ff + j] = sum;
        }
    }

    /* GeLU */
    for (int i = 0; i < B_e * D_ff; i++) {
        hidden[i] = gelu_fn(hidden[i]);
    }

    /* Down-project */
    for (int i = 0; i < B_e; i++) {
        for (int j = 0; j < D; j++) {
            float sum = b_down[j];
            for (int k = 0; k < D_ff; k++) {
                sum += hidden[i * D_ff + k] * w_down[k * D + j];
            }
            output[i * D + j] = sum;
        }
    }

    free(hidden);
}

/* --- Self-test --- */

int main(void) {
    int B_e = TOKENS_PER_E, D = D_MODEL, Dff = D_FF;

    float *input  = (float *)malloc((size_t)B_e * D * sizeof(float));
    float *w_up   = (float *)malloc((size_t)D * Dff * sizeof(float));
    float *b_up   = (float *)malloc((size_t)Dff * sizeof(float));
    float *w_down = (float *)malloc((size_t)Dff * D * sizeof(float));
    float *b_down = (float *)malloc((size_t)D * sizeof(float));
    float *out_t  = (float *)malloc((size_t)B_e * D * sizeof(float));
    float *out_r  = (float *)malloc((size_t)B_e * D * sizeof(float));

    if (!input || !w_up || !b_up || !w_down || !b_down || !out_t || !out_r) {
        fprintf(stderr, "Allocation failed\n");
        return 1;
    }

    for (int i = 0; i < B_e * D; i++)
        input[i] = ((float)(i % 79) - 39.0f) / 79.0f;
    for (int i = 0; i < D * Dff; i++)
        w_up[i] = ((float)(i % 61) - 30.0f) / 610.0f;
    for (int i = 0; i < Dff; i++)
        b_up[i] = ((float)(i % 37) - 18.0f) / 180.0f;
    for (int i = 0; i < Dff * D; i++)
        w_down[i] = ((float)(i % 53) - 26.0f) / 530.0f;
    for (int i = 0; i < D; i++)
        b_down[i] = ((float)(i % 29) - 14.0f) / 140.0f;

    expert_ffn(input, w_up, b_up, w_down, b_down, out_t, B_e, D, Dff);
    expert_ffn_ref(input, w_up, b_up, w_down, b_down, out_r, B_e, D, Dff);

    float max_err = 0.0f;
    for (int i = 0; i < B_e * D; i++) {
        float err = fabsf(out_t[i] - out_r[i]);
        if (err > max_err) max_err = err;
    }

    printf("expert_ffn: max_error = %e\n", max_err);
    int pass = (max_err < 1e-2f);
    printf("expert_ffn: %s\n", pass ? "PASS" : "FAIL");

    free(input); free(w_up); free(b_up); free(w_down); free(b_down);
    free(out_t); free(out_r);
    return pass ? 0 : 1;
}
