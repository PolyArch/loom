/*
 * Combine -- weighted aggregation of expert outputs for MoE.
 * For each token i, gathers K expert outputs and computes:
 *   output[i][D] = sum_k( weights[i][k] * expert_out[indices[i][k]][D] )
 * Uses scatter-gather indexing for the irregular access pattern.
 * N=seq_len, D=512, K=2 experts.
 *
 * Variants: weighted_sum, renormalized, top1_passthrough.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "tile_utils.h"

#define SEQ_LEN    128
#define D_MODEL    512
#define TOP_K      2
#define NUM_EXPERT 8
#define TILE_N     32

/* --- Primary kernel: weighted sum of expert outputs --- */

void combine(const float *expert_outputs, const int *top_indices,
             const float *top_weights, float *output,
             int N, int D, int K, int E) {
    (void)E;
    TILE_FOR(tn, 0, N, TILE_N) {
        int tn_end = TILE_END(tn, N, TILE_N);
        for (int i = tn; i < tn_end; i++) {
            const int *idx = top_indices + i * K;
            const float *wt = top_weights + i * K;

            for (int d = 0; d < D; d++) {
                float sum = 0.0f;
                for (int k = 0; k < K; k++) {
                    int expert_id = idx[k];
                    sum += wt[k] * expert_outputs[(size_t)expert_id * N * D +
                                                   (size_t)i * D + d];
                }
                output[i * D + d] = sum;
            }
        }
    }
}

/* --- Variants --- */

void combine_weighted_sum(const float *expert_outputs, const int *top_indices,
                          const float *top_weights, float *output,
                          int N, int D, int K, int E) {
    combine(expert_outputs, top_indices, top_weights, output, N, D, K, E);
}

/* Renormalized: re-normalize weights within combine (in case they don't sum to 1) */
void combine_renormalized(const float *expert_outputs, const int *top_indices,
                          const float *top_weights, float *output,
                          int N, int D, int K, int E) {
    (void)E;
    TILE_FOR(tn, 0, N, TILE_N) {
        int tn_end = TILE_END(tn, N, TILE_N);
        for (int i = tn; i < tn_end; i++) {
            const int *idx = top_indices + i * K;
            const float *wt = top_weights + i * K;

            float wt_sum = 0.0f;
            for (int k = 0; k < K; k++) wt_sum += wt[k];
            float inv = (wt_sum > 0.0f) ? (1.0f / wt_sum) : 0.0f;

            for (int d = 0; d < D; d++) {
                float sum = 0.0f;
                for (int k = 0; k < K; k++) {
                    int expert_id = idx[k];
                    sum += (wt[k] * inv) * expert_outputs[
                        (size_t)expert_id * N * D + (size_t)i * D + d];
                }
                output[i * D + d] = sum;
            }
        }
    }
}

/* Top-1 passthrough: only use the top expert (K=1 effective) */
void combine_top1_passthrough(const float *expert_outputs,
                              const int *top_indices,
                              const float *top_weights, float *output,
                              int N, int D, int K, int E) {
    (void)top_weights; (void)E;
    TILE_FOR(tn, 0, N, TILE_N) {
        int tn_end = TILE_END(tn, N, TILE_N);
        for (int i = tn; i < tn_end; i++) {
            int expert_id = top_indices[i * K];  /* top-1 only */
            for (int d = 0; d < D; d++) {
                output[i * D + d] = expert_outputs[
                    (size_t)expert_id * N * D + (size_t)i * D + d];
            }
        }
    }
}

/* --- Reference implementation --- */

void combine_ref(const float *expert_outputs, const int *top_indices,
                 const float *top_weights, float *output,
                 int N, int D, int K, int E) {
    (void)E;
    for (int i = 0; i < N; i++) {
        const int *idx = top_indices + i * K;
        const float *wt = top_weights + i * K;
        for (int d = 0; d < D; d++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                int expert_id = idx[k];
                sum += wt[k] * expert_outputs[(size_t)expert_id * N * D +
                                               (size_t)i * D + d];
            }
            output[i * D + d] = sum;
        }
    }
}

/* --- Self-test --- */

int main(void) {
    int N = SEQ_LEN, D = D_MODEL, K = TOP_K, E = NUM_EXPERT;

    /* Expert outputs: E experts, each producing [N][D] */
    float *expert_out = (float *)malloc((size_t)E * N * D * sizeof(float));
    int *top_idx      = (int *)malloc((size_t)N * K * sizeof(int));
    float *top_wt     = (float *)malloc((size_t)N * K * sizeof(float));
    float *out_t      = (float *)malloc((size_t)N * D * sizeof(float));
    float *out_r      = (float *)malloc((size_t)N * D * sizeof(float));

    if (!expert_out || !top_idx || !top_wt || !out_t || !out_r) {
        fprintf(stderr, "Allocation failed\n");
        return 1;
    }

    /* Deterministic expert outputs */
    for (size_t i = 0; i < (size_t)E * N * D; i++)
        expert_out[i] = ((float)(i % 127) - 63.0f) / 63.0f;

    /* Fixed gate assignments: expert 0 and 1 with weights 0.6, 0.4 */
    for (int i = 0; i < N; i++) {
        top_idx[i * K + 0] = i % E;
        top_idx[i * K + 1] = (i + 1) % E;
        top_wt[i * K + 0] = 0.6f;
        top_wt[i * K + 1] = 0.4f;
    }

    combine(expert_out, top_idx, top_wt, out_t, N, D, K, E);
    combine_ref(expert_out, top_idx, top_wt, out_r, N, D, K, E);

    float max_err = 0.0f;
    for (int i = 0; i < N * D; i++) {
        float err = fabsf(out_t[i] - out_r[i]);
        if (err > max_err) max_err = err;
    }

    printf("combine: max_error = %e\n", max_err);
    int pass = (max_err < 1e-6f);
    printf("combine: %s\n", pass ? "PASS" : "FAIL");

    free(expert_out); free(top_idx); free(top_wt); free(out_t); free(out_r);
    return pass ? 0 : 1;
}
