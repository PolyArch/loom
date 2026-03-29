/*
 * Row-wise Softmax for attention scores.
 * For each row: find max, subtract max, exponentiate, sum, divide.
 * H=8 heads, N=seq_len.
 *
 * Variants: 3pass, online_2pass, tiled_32, fused_with_score.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "tile_utils.h"

#define NUM_HEADS  8
#define SEQ_LEN    128

/* --- Primary kernel: 3-pass softmax with tiled row processing --- */

void softmax(float *data, int H, int N) {
    for (int h = 0; h < H; h++) {
        float *head = data + (size_t)h * N * N;
        TILE_FOR(tr, 0, N, 32) {
            int tr_end = TILE_END(tr, N, 32);
            for (int i = tr; i < tr_end; i++) {
                float *row = head + (size_t)i * N;
                float row_max = row[0];
                for (int j = 1; j < N; j++) {
                    if (row[j] > row_max) row_max = row[j];
                }
                float sum = 0.0f;
                for (int j = 0; j < N; j++) {
                    row[j] = expf(row[j] - row_max);
                    sum += row[j];
                }
                float inv_sum = 1.0f / sum;
                for (int j = 0; j < N; j++) {
                    row[j] *= inv_sum;
                }
            }
        }
    }
}

/* --- Variants --- */

/* Explicit 3-pass: separate max, exp+sum, normalize passes */
void softmax_3pass(float *data, int H, int N) {
    for (int h = 0; h < H; h++) {
        float *head = data + (size_t)h * N * N;
        TILE_FOR(tr, 0, N, 32) {
            int tr_end = TILE_END(tr, N, 32);
            for (int i = tr; i < tr_end; i++) {
                float *row = head + (size_t)i * N;
                /* pass 1: find max */
                float row_max = row[0];
                for (int j = 1; j < N; j++) {
                    if (row[j] > row_max) row_max = row[j];
                }
                /* pass 2: exp and sum */
                float sum = 0.0f;
                for (int j = 0; j < N; j++) {
                    row[j] = expf(row[j] - row_max);
                    sum += row[j];
                }
                /* pass 3: normalize */
                float inv_sum = 1.0f / sum;
                for (int j = 0; j < N; j++) {
                    row[j] *= inv_sum;
                }
            }
        }
    }
}

/* Online 2-pass: fused max+exp+sum in first pass, normalize in second */
void softmax_online_2pass(float *data, int H, int N) {
    for (int h = 0; h < H; h++) {
        float *head = data + (size_t)h * N * N;
        TILE_FOR(tr, 0, N, 32) {
            int tr_end = TILE_END(tr, N, 32);
            for (int i = tr; i < tr_end; i++) {
                float *row = head + (size_t)i * N;
                /* pass 1: online max tracking with running sum correction */
                float running_max = row[0];
                float running_sum = 1.0f;
                row[0] = 1.0f;
                for (int j = 1; j < N; j++) {
                    if (row[j] > running_max) {
                        float correction = expf(running_max - row[j]);
                        running_sum *= correction;
                        running_max = row[j];
                        row[j] = 1.0f;
                    } else {
                        row[j] = expf(row[j] - running_max);
                    }
                    running_sum += row[j];
                }
                /* pass 2: normalize */
                float inv_sum = 1.0f / running_sum;
                for (int j = 0; j < N; j++) {
                    row[j] *= inv_sum;
                }
            }
        }
    }
}

/* Tiled with tile_rows=32 */
void softmax_tiled_32(float *data, int H, int N) {
    softmax(data, H, N);
}

/* Fused with score: accepts raw QK^T scores and applies softmax in-place */
void softmax_fused_with_score(float *data, int H, int N) {
    softmax(data, H, N);
}

/* --- Reference implementation --- */

void softmax_ref(float *data, int H, int N) {
    for (int h = 0; h < H; h++) {
        float *head = data + (size_t)h * N * N;
        for (int i = 0; i < N; i++) {
            float *row = head + (size_t)i * N;
            float row_max = row[0];
            for (int j = 1; j < N; j++) {
                if (row[j] > row_max) row_max = row[j];
            }
            float sum = 0.0f;
            for (int j = 0; j < N; j++) {
                row[j] = expf(row[j] - row_max);
                sum += row[j];
            }
            float inv_sum = 1.0f / sum;
            for (int j = 0; j < N; j++) {
                row[j] *= inv_sum;
            }
        }
    }
}

/* --- Self-test --- */

int main(void) {
    int H = NUM_HEADS, N = SEQ_LEN;
    size_t total = (size_t)H * N * N;

    float *data_tiled = (float *)malloc(total * sizeof(float));
    float *data_ref   = (float *)malloc(total * sizeof(float));

    if (!data_tiled || !data_ref) {
        fprintf(stderr, "Allocation failed\n");
        return 1;
    }

    for (size_t i = 0; i < total; i++) {
        float val = ((float)(i % 101) - 50.0f) / 10.0f;
        data_tiled[i] = val;
        data_ref[i]   = val;
    }

    softmax(data_tiled, H, N);
    softmax_ref(data_ref, H, N);

    float max_err = 0.0f;
    for (size_t i = 0; i < total; i++) {
        float err = fabsf(data_tiled[i] - data_ref[i]);
        if (err > max_err) max_err = err;
    }

    /* Verify rows sum to ~1.0 */
    int row_sum_ok = 1;
    for (int h = 0; h < H; h++) {
        for (int i = 0; i < N; i++) {
            float sum = 0.0f;
            for (int j = 0; j < N; j++) {
                sum += data_tiled[(size_t)h * N * N + (size_t)i * N + j];
            }
            if (fabsf(sum - 1.0f) > 1e-4f) row_sum_ok = 0;
        }
    }

    printf("softmax: max_error = %e, row_sums_ok = %d\n", max_err, row_sum_ok);
    int pass = (max_err < 1e-5f) && row_sum_ok;
    printf("softmax: %s\n", pass ? "PASS" : "FAIL");

    free(data_tiled); free(data_ref);
    return pass ? 0 : 1;
}
