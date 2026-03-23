/*
 * Bilateral Filter -- joint bilateral post-processing filter for disparity.
 * 5x5 kernel, guided by image intensity.
 * Smooths disparity map while preserving edges aligned with the guide image.
 * Tiled: process TILE_W x TILE_H blocks.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "tile_utils.h"

#define IMG_W       640
#define IMG_H       480
#define FILTER_R    2   /* 5x5 kernel: radius=2 */
#define SIGMA_S     2.0f
#define SIGMA_R     30.0f
#define TILE_W      64
#define TILE_H      32

/* Use smaller sizes for testing */
#define TEST_W      160
#define TEST_H      120

static inline int clamp_idx(int v, int lo, int hi) {
    if (v < lo) return lo;
    if (v >= hi) return hi - 1;
    return v;
}

void post_filter(const float *disparity, const float *guide,
                 float *output, int W, int H) {
    float inv_sigma_s2 = 1.0f / (2.0f * SIGMA_S * SIGMA_S);
    float inv_sigma_r2 = 1.0f / (2.0f * SIGMA_R * SIGMA_R);

    TILE_FOR(ty, 0, H, TILE_H) {
        int ty_end = TILE_END(ty, H, TILE_H);
        TILE_FOR(tx, 0, W, TILE_W) {
            int tx_end = TILE_END(tx, W, TILE_W);

            int y, x;
            for (y = ty; y < ty_end; y++) {
                for (x = tx; x < tx_end; x++) {
                    float center_guide = guide[y * W + x];
                    float sum_w = 0.0f;
                    float sum_v = 0.0f;

                    int fy, fx;
                    for (fy = -FILTER_R; fy <= FILTER_R; fy++) {
                        for (fx = -FILTER_R; fx <= FILTER_R; fx++) {
                            int ny = clamp_idx(y + fy, 0, H);
                            int nx = clamp_idx(x + fx, 0, W);

                            /* Spatial weight */
                            float ds2 = (float)(fx * fx + fy * fy);
                            float ws = expf(-ds2 * inv_sigma_s2);

                            /* Range (intensity) weight */
                            float di = guide[ny * W + nx] - center_guide;
                            float wr = expf(-di * di * inv_sigma_r2);

                            float w = ws * wr;
                            sum_w += w;
                            sum_v += w * disparity[ny * W + nx];
                        }
                    }

                    output[y * W + x] = sum_v / (sum_w + 1e-10f);
                }
            }
        }
    }
}

void post_filter_ref(const float *disparity, const float *guide,
                     float *output, int W, int H) {
    float inv_sigma_s2 = 1.0f / (2.0f * SIGMA_S * SIGMA_S);
    float inv_sigma_r2 = 1.0f / (2.0f * SIGMA_R * SIGMA_R);

    int y, x;
    for (y = 0; y < H; y++) {
        for (x = 0; x < W; x++) {
            float center_guide = guide[y * W + x];
            float sum_w = 0.0f;
            float sum_v = 0.0f;

            int fy, fx;
            for (fy = -FILTER_R; fy <= FILTER_R; fy++) {
                for (fx = -FILTER_R; fx <= FILTER_R; fx++) {
                    int ny = clamp_idx(y + fy, 0, H);
                    int nx = clamp_idx(x + fx, 0, W);

                    float ds2 = (float)(fx * fx + fy * fy);
                    float ws = expf(-ds2 * inv_sigma_s2);

                    float di = guide[ny * W + nx] - center_guide;
                    float wr = expf(-di * di * inv_sigma_r2);

                    float w = ws * wr;
                    sum_w += w;
                    sum_v += w * disparity[ny * W + nx];
                }
            }

            output[y * W + x] = sum_v / (sum_w + 1e-10f);
        }
    }
}

int main(void) {
    int W = TEST_W, H = TEST_H;
    size_t img_size = (size_t)W * H;

    float *disparity = (float *)malloc(img_size * sizeof(float));
    float *guide     = (float *)malloc(img_size * sizeof(float));
    float *out_tile  = (float *)malloc(img_size * sizeof(float));
    float *out_ref   = (float *)malloc(img_size * sizeof(float));

    if (!disparity || !guide || !out_tile || !out_ref) {
        fprintf(stderr, "Allocation failed\n");
        return 1;
    }

    /* Generate test data */
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            /* Disparity with some noise */
            float base_d = 10.0f + 20.0f * (float)x / (float)W;
            float noise = (float)((x * 13 + y * 7) % 11) - 5.0f;
            disparity[y * W + x] = base_d + noise;

            /* Guide image (edge-preserving reference) */
            guide[y * W + x] = (float)((x / 40 + y / 40) % 2) * 200.0f + 28.0f;
        }
    }

    post_filter(disparity, guide, out_tile, W, H);
    post_filter_ref(disparity, guide, out_ref, W, H);

    float max_err = 0.0f;
    for (size_t i = 0; i < img_size; i++) {
        float err = fabsf(out_tile[i] - out_ref[i]);
        if (err > max_err) max_err = err;
    }

    printf("post_filter: max_error = %e\n", max_err);
    int pass = (max_err < 1e-4f);
    printf("post_filter: %s\n", pass ? "PASS" : "FAIL");

    free(disparity); free(guide); free(out_tile); free(out_ref);
    return pass ? 0 : 1;
}
