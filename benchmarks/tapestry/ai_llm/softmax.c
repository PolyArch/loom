/*
 * Row-wise Softmax for attention scores.
 * For each row: find max, subtract max, exponentiate, sum, divide.
 * H=8 heads, N=seq_len.
 * Tiled: process tile_N=32 rows at a time.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "tile_utils.h"

#define NUM_HEADS  8
#define SEQ_LEN    128
#define TILE_ROWS  32

void softmax(float *data, int H, int N) {
    for (int h = 0; h < H; h++) {
        float *head = data + (size_t)h * N * N;

        TILE_FOR(tr, 0, N, TILE_ROWS) {
            int tr_end = TILE_END(tr, N, TILE_ROWS);
            for (int i = tr; i < tr_end; i++) {
                float *row = head + (size_t)i * N;

                /* Find row max for numerical stability */
                float row_max = row[0];
                for (int j = 1; j < N; j++) {
                    if (row[j] > row_max) row_max = row[j];
                }

                /* Exponentiate and sum */
                float sum = 0.0f;
                for (int j = 0; j < N; j++) {
                    row[j] = expf(row[j] - row_max);
                    sum += row[j];
                }

                /* Normalize */
                float inv_sum = 1.0f / sum;
                for (int j = 0; j < N; j++) {
                    row[j] *= inv_sum;
                }
            }
        }
    }
}

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

int main(void) {
    int H = NUM_HEADS, N = SEQ_LEN;
    size_t total = (size_t)H * N * N;

    float *data_tiled = (float *)malloc(total * sizeof(float));
    float *data_ref   = (float *)malloc(total * sizeof(float));

    if (!data_tiled || !data_ref) {
        fprintf(stderr, "Allocation failed\n");
        return 1;
    }

    /* Generate test data (attention scores) */
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

    printf("softmax: max_error = %e\n", max_err);
    int pass = (max_err < 1e-5f);
    printf("softmax: %s\n", pass ? "PASS" : "FAIL");

    free(data_tiled); free(data_ref);
    return pass ? 0 : 1;
}
