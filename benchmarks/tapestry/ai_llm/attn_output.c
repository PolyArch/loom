/*
 * Attention Output -- batched matmul score * V.
 * Computes out[H][N][D] = score[H][N][N] * V[H][N][D]
 * H=8 heads, N=seq_len, D=64.
 * Tiled per head, tile_N=32 for sequence dimension.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "tile_utils.h"

#define NUM_HEADS  8
#define SEQ_LEN    128
#define HEAD_DIM   64
#define TILE_N     32

void attn_output(const float *score, const float *V, float *out,
                 int H, int N, int D) {
    for (int h = 0; h < H; h++) {
        const float *Sh = score + (size_t)h * N * N;
        const float *Vh = V + (size_t)h * N * D;
        float *Oh = out + (size_t)h * N * D;

        memset(Oh, 0, (size_t)N * D * sizeof(float));

        TILE_FOR(ti, 0, N, TILE_N) {
            int ti_end = TILE_END(ti, N, TILE_N);
            TILE_FOR(tj, 0, N, TILE_N) {
                int tj_end = TILE_END(tj, N, TILE_N);
                for (int i = ti; i < ti_end; i++) {
                    for (int j = tj; j < tj_end; j++) {
                        float s = Sh[i * N + j];
                        for (int d = 0; d < D; d++) {
                            Oh[i * D + d] += s * Vh[j * D + d];
                        }
                    }
                }
            }
        }
    }
}

void attn_output_ref(const float *score, const float *V, float *out,
                     int H, int N, int D) {
    for (int h = 0; h < H; h++) {
        const float *Sh = score + (size_t)h * N * N;
        const float *Vh = V + (size_t)h * N * D;
        float *Oh = out + (size_t)h * N * D;
        for (int i = 0; i < N; i++) {
            for (int d = 0; d < D; d++) {
                float sum = 0.0f;
                for (int j = 0; j < N; j++) {
                    sum += Sh[i * N + j] * Vh[j * D + d];
                }
                Oh[i * D + d] = sum;
            }
        }
    }
}

int main(void) {
    int H = NUM_HEADS, N = SEQ_LEN, D = HEAD_DIM;
    size_t s_size  = (size_t)H * N * N;
    size_t vd_size = (size_t)H * N * D;

    float *score  = (float *)malloc(s_size * sizeof(float));
    float *V      = (float *)malloc(vd_size * sizeof(float));
    float *O_tile = (float *)malloc(vd_size * sizeof(float));
    float *O_ref  = (float *)malloc(vd_size * sizeof(float));

    if (!score || !V || !O_tile || !O_ref) {
        fprintf(stderr, "Allocation failed\n");
        return 1;
    }

    /* Generate softmax-like scores (positive, sum~1 per row) */
    for (int h = 0; h < H; h++) {
        for (int i = 0; i < N; i++) {
            float sum = 0.0f;
            for (int j = 0; j < N; j++) {
                float val = 1.0f / (1.0f + (float)((i - j) * (i - j)));
                score[h * N * N + i * N + j] = val;
                sum += val;
            }
            for (int j = 0; j < N; j++) {
                score[h * N * N + i * N + j] /= sum;
            }
        }
    }

    for (size_t i = 0; i < vd_size; i++)
        V[i] = ((float)(i % 89) - 44.0f) / 44.0f;

    attn_output(score, V, O_tile, H, N, D);
    attn_output_ref(score, V, O_ref, H, N, D);

    float max_err = 0.0f;
    for (size_t i = 0; i < vd_size; i++) {
        float err = fabsf(O_tile[i] - O_ref[i]);
        if (err > max_err) max_err = err;
    }

    printf("attn_output: max_error = %e\n", max_err);
    int pass = (max_err < 1e-3f);
    printf("attn_output: %s\n", pass ? "PASS" : "FAIL");

    free(score); free(V); free(O_tile); free(O_ref);
    return pass ? 0 : 1;
}
