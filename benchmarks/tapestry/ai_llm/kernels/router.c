/*
 * Router / Gating Network -- MoE expert routing.
 * Computes logits[N][E] = input[N][D] * gate_weight[D][E]
 * then applies softmax over E experts per token to produce gate values.
 * N=seq_len, D=512, E=8 (number of experts).
 *
 * Variants: dense, noisy_topk, sigmoid, hash.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "tile_utils.h"

#define SEQ_LEN    128
#define D_MODEL    512
#define NUM_EXPERT 8
#define TILE_N     32
#define TILE_D     32

/* --- Core: dense gating (matmul + softmax) --- */

static void router_matmul(const float *input, const float *gate_weight,
                          float *logits, int N, int D, int E,
                          int tile_n, int tile_d) {
    memset(logits, 0, (size_t)N * E * sizeof(float));
    TILE_FOR(tn, 0, N, tile_n) {
        int tn_end = TILE_END(tn, N, tile_n);
        TILE_FOR(td, 0, D, tile_d) {
            int td_end = TILE_END(td, D, tile_d);
            for (int i = tn; i < tn_end; i++) {
                for (int k = td; k < td_end; k++) {
                    float a = input[i * D + k];
                    for (int e = 0; e < E; e++) {
                        logits[i * E + e] += a * gate_weight[k * E + e];
                    }
                }
            }
        }
    }
}

static void row_softmax(float *data, int N, int E) {
    for (int i = 0; i < N; i++) {
        float *row = data + i * E;
        float row_max = row[0];
        for (int e = 1; e < E; e++) {
            if (row[e] > row_max) row_max = row[e];
        }
        float sum = 0.0f;
        for (int e = 0; e < E; e++) {
            row[e] = expf(row[e] - row_max);
            sum += row[e];
        }
        float inv = 1.0f / sum;
        for (int e = 0; e < E; e++) {
            row[e] *= inv;
        }
    }
}

void router(const float *input, const float *gate_weight,
            float *gate_values, int N, int D, int E) {
    router_matmul(input, gate_weight, gate_values, N, D, E, TILE_N, TILE_D);
    row_softmax(gate_values, N, E);
}

/* --- Variants --- */

void router_dense(const float *input, const float *gate_weight,
                  float *gate_values, int N, int D, int E) {
    router(input, gate_weight, gate_values, N, D, E);
}

/* Noisy top-k: add noise before softmax for load balancing */
void router_noisy_topk(const float *input, const float *gate_weight,
                       float *gate_values, int N, int D, int E) {
    router_matmul(input, gate_weight, gate_values, N, D, E, TILE_N, TILE_D);
    /* Add deterministic pseudo-noise for reproducibility */
    for (int i = 0; i < N * E; i++) {
        float noise = ((float)((i * 7 + 13) % 101) - 50.0f) / 500.0f;
        gate_values[i] += noise;
    }
    row_softmax(gate_values, N, E);
}

/* Sigmoid gating: independent sigmoid per expert instead of softmax */
void router_sigmoid(const float *input, const float *gate_weight,
                    float *gate_values, int N, int D, int E) {
    router_matmul(input, gate_weight, gate_values, N, D, E, TILE_N, TILE_D);
    for (int i = 0; i < N * E; i++) {
        gate_values[i] = 1.0f / (1.0f + expf(-gate_values[i]));
    }
}

/* Hash routing: deterministic assignment based on token index */
void router_hash(const float *input, const float *gate_weight,
                 float *gate_values, int N, int D, int E) {
    (void)input; (void)gate_weight; (void)D;
    memset(gate_values, 0, (size_t)N * E * sizeof(float));
    for (int i = 0; i < N; i++) {
        int primary   = i % E;
        int secondary = (i * 7 + 3) % E;
        gate_values[i * E + primary]   = 0.6f;
        gate_values[i * E + secondary] = 0.4f;
    }
}

/* --- Reference implementation --- */

void router_ref(const float *input, const float *gate_weight,
                float *gate_values, int N, int D, int E) {
    for (int i = 0; i < N; i++) {
        for (int e = 0; e < E; e++) {
            float sum = 0.0f;
            for (int k = 0; k < D; k++) {
                sum += input[i * D + k] * gate_weight[k * E + e];
            }
            gate_values[i * E + e] = sum;
        }
    }
    row_softmax(gate_values, N, E);
}

/* --- Self-test --- */

int main(void) {
    int N = SEQ_LEN, D = D_MODEL, E = NUM_EXPERT;

    float *input  = (float *)malloc((size_t)N * D * sizeof(float));
    float *gw     = (float *)malloc((size_t)D * E * sizeof(float));
    float *gv_t   = (float *)malloc((size_t)N * E * sizeof(float));
    float *gv_r   = (float *)malloc((size_t)N * E * sizeof(float));

    if (!input || !gw || !gv_t || !gv_r) {
        fprintf(stderr, "Allocation failed\n");
        return 1;
    }

    for (int i = 0; i < N * D; i++)
        input[i] = ((float)(i % 79) - 39.0f) / 39.0f;
    for (int i = 0; i < D * E; i++)
        gw[i] = ((float)(i % 53) - 26.0f) / 260.0f;

    router(input, gw, gv_t, N, D, E);
    router_ref(input, gw, gv_r, N, D, E);

    float max_err = 0.0f;
    for (int i = 0; i < N * E; i++) {
        float err = fabsf(gv_t[i] - gv_r[i]);
        if (err > max_err) max_err = err;
    }

    /* Verify each row sums to ~1 */
    int row_ok = 1;
    for (int i = 0; i < N; i++) {
        float sum = 0.0f;
        for (int e = 0; e < E; e++)
            sum += gv_t[i * E + e];
        if (fabsf(sum - 1.0f) > 1e-4f) row_ok = 0;
    }

    printf("router: max_error = %e, row_sums_ok = %d\n", max_err, row_ok);
    int pass = (max_err < 1e-4f) && row_ok;
    printf("router: %s\n", pass ? "PASS" : "FAIL");

    free(input); free(gw); free(gv_t); free(gv_r);
    return pass ? 0 : 1;
}
