/*
 * Top-K Expert Selection -- selects top-K experts per token.
 * Given gate_values[N][E], produces:
 *   top_indices[N][K]  -- indices of top-K experts per token
 *   top_weights[N][K]  -- corresponding gate weights (renormalized)
 * N=seq_len, E=8 experts, K=2 selected.
 *
 * Variants: linear_scan, heap, threshold.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "tile_utils.h"

#define SEQ_LEN    128
#define NUM_EXPERT 8
#define TOP_K      2
#define TILE_N     32

/* --- Primary kernel: linear scan top-K --- */

void topk_select(const float *gate_values, int *top_indices,
                 float *top_weights, int N, int E, int K) {
    TILE_FOR(tn, 0, N, TILE_N) {
        int tn_end = TILE_END(tn, N, TILE_N);
        for (int i = tn; i < tn_end; i++) {
            const float *row = gate_values + i * E;
            int *idx = top_indices + i * K;
            float *wt = top_weights + i * K;

            /* Initialize with -inf */
            for (int k = 0; k < K; k++) {
                idx[k] = -1;
                wt[k] = -1e30f;
            }

            /* Linear scan: find top-K values */
            for (int e = 0; e < E; e++) {
                /* Find the minimum in current top-K */
                int min_pos = 0;
                for (int k = 1; k < K; k++) {
                    if (wt[k] < wt[min_pos]) min_pos = k;
                }
                if (row[e] > wt[min_pos]) {
                    wt[min_pos] = row[e];
                    idx[min_pos] = e;
                }
            }

            /* Sort top-K by descending weight (simple insertion sort) */
            for (int a = 1; a < K; a++) {
                float wt_tmp = wt[a];
                int idx_tmp = idx[a];
                int b = a - 1;
                while (b >= 0 && wt[b] < wt_tmp) {
                    wt[b + 1] = wt[b];
                    idx[b + 1] = idx[b];
                    b--;
                }
                wt[b + 1] = wt_tmp;
                idx[b + 1] = idx_tmp;
            }

            /* Renormalize weights to sum to 1 */
            float sum = 0.0f;
            for (int k = 0; k < K; k++) sum += wt[k];
            if (sum > 0.0f) {
                float inv = 1.0f / sum;
                for (int k = 0; k < K; k++) wt[k] *= inv;
            }
        }
    }
}

/* --- Variants --- */

void topk_select_linear_scan(const float *gate_values, int *top_indices,
                             float *top_weights, int N, int E, int K) {
    topk_select(gate_values, top_indices, top_weights, N, E, K);
}

/* Heap-based variant: use a min-heap of size K (for large E) */
void topk_select_heap(const float *gate_values, int *top_indices,
                      float *top_weights, int N, int E, int K) {
    /* For small K and E, heap is overkill but structurally different.
       Uses bottom-up heapify approach. */
    TILE_FOR(tn, 0, N, TILE_N) {
        int tn_end = TILE_END(tn, N, TILE_N);
        for (int i = tn; i < tn_end; i++) {
            const float *row = gate_values + i * E;
            int *idx = top_indices + i * K;
            float *wt = top_weights + i * K;

            /* Initialize heap with first K elements */
            for (int k = 0; k < K && k < E; k++) {
                wt[k] = row[k];
                idx[k] = k;
            }
            /* Heapify (min-heap by weight) */
            for (int k = K / 2 - 1; k >= 0; k--) {
                int pos = k;
                while (2 * pos + 1 < K) {
                    int child = 2 * pos + 1;
                    if (child + 1 < K && wt[child + 1] < wt[child]) child++;
                    if (wt[pos] <= wt[child]) break;
                    float tw = wt[pos]; wt[pos] = wt[child]; wt[child] = tw;
                    int ti = idx[pos]; idx[pos] = idx[child]; idx[child] = ti;
                    pos = child;
                }
            }

            /* Scan remaining elements */
            for (int e = K; e < E; e++) {
                if (row[e] > wt[0]) {
                    wt[0] = row[e];
                    idx[0] = e;
                    /* Sift down */
                    int pos = 0;
                    while (2 * pos + 1 < K) {
                        int child = 2 * pos + 1;
                        if (child + 1 < K && wt[child + 1] < wt[child]) child++;
                        if (wt[pos] <= wt[child]) break;
                        float tw = wt[pos]; wt[pos] = wt[child]; wt[child] = tw;
                        int ti = idx[pos]; idx[pos] = idx[child]; idx[child] = ti;
                        pos = child;
                    }
                }
            }

            /* Sort descending and renormalize */
            for (int a = 1; a < K; a++) {
                float wt_tmp = wt[a];
                int idx_tmp = idx[a];
                int b = a - 1;
                while (b >= 0 && wt[b] < wt_tmp) {
                    wt[b + 1] = wt[b];
                    idx[b + 1] = idx[b];
                    b--;
                }
                wt[b + 1] = wt_tmp;
                idx[b + 1] = idx_tmp;
            }

            float sum = 0.0f;
            for (int k = 0; k < K; k++) sum += wt[k];
            if (sum > 0.0f) {
                float inv = 1.0f / sum;
                for (int k = 0; k < K; k++) wt[k] *= inv;
            }
        }
    }
}

/* Threshold variant: select all experts above a threshold, cap at K */
void topk_select_threshold(const float *gate_values, int *top_indices,
                           float *top_weights, int N, int E, int K) {
    float threshold = 1.0f / (float)E;

    TILE_FOR(tn, 0, N, TILE_N) {
        int tn_end = TILE_END(tn, N, TILE_N);
        for (int i = tn; i < tn_end; i++) {
            const float *row = gate_values + i * E;
            int *idx = top_indices + i * K;
            float *wt = top_weights + i * K;

            /* Collect candidates above threshold */
            int count = 0;
            for (int e = 0; e < E && count < K; e++) {
                if (row[e] >= threshold) {
                    idx[count] = e;
                    wt[count] = row[e];
                    count++;
                }
            }
            /* If fewer than K found, fill with best remaining */
            if (count < K) {
                for (int e = 0; e < E && count < K; e++) {
                    int already = 0;
                    for (int c = 0; c < count; c++) {
                        if (idx[c] == e) { already = 1; break; }
                    }
                    if (!already) {
                        idx[count] = e;
                        wt[count] = row[e];
                        count++;
                    }
                }
            }

            /* Sort descending and renormalize */
            for (int a = 1; a < K; a++) {
                float wt_tmp = wt[a];
                int idx_tmp = idx[a];
                int b = a - 1;
                while (b >= 0 && wt[b] < wt_tmp) {
                    wt[b + 1] = wt[b];
                    idx[b + 1] = idx[b];
                    b--;
                }
                wt[b + 1] = wt_tmp;
                idx[b + 1] = idx_tmp;
            }

            float sum = 0.0f;
            for (int k = 0; k < K; k++) sum += wt[k];
            if (sum > 0.0f) {
                float inv = 1.0f / sum;
                for (int k = 0; k < K; k++) wt[k] *= inv;
            }
        }
    }
}

/* --- Reference implementation --- */

void topk_select_ref(const float *gate_values, int *top_indices,
                     float *top_weights, int N, int E, int K) {
    for (int i = 0; i < N; i++) {
        const float *row = gate_values + i * E;
        int *idx = top_indices + i * K;
        float *wt = top_weights + i * K;

        for (int k = 0; k < K; k++) {
            idx[k] = -1;
            wt[k] = -1e30f;
        }
        for (int e = 0; e < E; e++) {
            int min_pos = 0;
            for (int k = 1; k < K; k++) {
                if (wt[k] < wt[min_pos]) min_pos = k;
            }
            if (row[e] > wt[min_pos]) {
                wt[min_pos] = row[e];
                idx[min_pos] = e;
            }
        }
        /* Sort descending */
        for (int a = 1; a < K; a++) {
            float wt_tmp = wt[a];
            int idx_tmp = idx[a];
            int b = a - 1;
            while (b >= 0 && wt[b] < wt_tmp) {
                wt[b + 1] = wt[b];
                idx[b + 1] = idx[b];
                b--;
            }
            wt[b + 1] = wt_tmp;
            idx[b + 1] = idx_tmp;
        }
        float sum = 0.0f;
        for (int k = 0; k < K; k++) sum += wt[k];
        if (sum > 0.0f) {
            float inv = 1.0f / sum;
            for (int k = 0; k < K; k++) wt[k] *= inv;
        }
    }
}

/* --- Self-test --- */

int main(void) {
    int N = SEQ_LEN, E = NUM_EXPERT, K = TOP_K;

    float *gv   = (float *)malloc((size_t)N * E * sizeof(float));
    int *idx_t   = (int *)malloc((size_t)N * K * sizeof(int));
    float *wt_t  = (float *)malloc((size_t)N * K * sizeof(float));
    int *idx_r   = (int *)malloc((size_t)N * K * sizeof(int));
    float *wt_r  = (float *)malloc((size_t)N * K * sizeof(float));

    if (!gv || !idx_t || !wt_t || !idx_r || !wt_r) {
        fprintf(stderr, "Allocation failed\n");
        return 1;
    }

    /* Generate softmax-like gate values */
    for (int i = 0; i < N; i++) {
        float sum = 0.0f;
        for (int e = 0; e < E; e++) {
            float val = expf(((float)((i * 7 + e * 13) % 31) - 15.0f) / 5.0f);
            gv[i * E + e] = val;
            sum += val;
        }
        for (int e = 0; e < E; e++)
            gv[i * E + e] /= sum;
    }

    topk_select(gv, idx_t, wt_t, N, E, K);
    topk_select_ref(gv, idx_r, wt_r, N, E, K);

    int idx_match = 1;
    float max_wt_err = 0.0f;
    for (int i = 0; i < N * K; i++) {
        if (idx_t[i] != idx_r[i]) idx_match = 0;
        float err = fabsf(wt_t[i] - wt_r[i]);
        if (err > max_wt_err) max_wt_err = err;
    }

    printf("topk_select: idx_match=%d, max_weight_error=%e\n",
           idx_match, max_wt_err);
    int pass = idx_match && (max_wt_err < 1e-5f);
    printf("topk_select: %s\n", pass ? "PASS" : "FAIL");

    free(gv); free(idx_t); free(wt_t); free(idx_r); free(wt_r);
    return pass ? 0 : 1;
}
