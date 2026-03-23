/*
 * Stereo Disparity -- winner-take-all disparity selection with sub-pixel
 * refinement from SAD cost volume.
 * 640x480 image, 64 disparity levels.
 * Tiled: process TILE_W x TILE_H pixel blocks.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "tile_utils.h"

#define IMG_W      640
#define IMG_H      480
#define MAX_DISP   64
#define TILE_W     64
#define TILE_H     32

/* Use smaller sizes for testing */
#define TEST_W     160
#define TEST_H     120

void stereo_disparity(const float *cost, float *disparity,
                      int W, int H, int max_disp) {
    TILE_FOR(ty, 0, H, TILE_H) {
        int ty_end = TILE_END(ty, H, TILE_H);
        TILE_FOR(tx, 0, W, TILE_W) {
            int tx_end = TILE_END(tx, W, TILE_W);

            int y, x;
            for (y = ty; y < ty_end; y++) {
                for (x = tx; x < tx_end; x++) {
                    const float *pixel_cost = cost + (y * W + x) * max_disp;

                    /* Winner-take-all: find minimum cost disparity */
                    int best_d = 0;
                    float best_cost = pixel_cost[0];
                    int d;
                    for (d = 1; d < max_disp; d++) {
                        if (pixel_cost[d] < best_cost) {
                            best_cost = pixel_cost[d];
                            best_d = d;
                        }
                    }

                    /* Sub-pixel refinement using parabola fitting */
                    float sub_disp = (float)best_d;
                    if (best_d > 0 && best_d < max_disp - 1) {
                        float c_prev = pixel_cost[best_d - 1];
                        float c_next = pixel_cost[best_d + 1];
                        float denom = 2.0f * (c_prev + c_next - 2.0f * best_cost);
                        if (fabsf(denom) > 1e-6f) {
                            sub_disp = (float)best_d + (c_prev - c_next) / denom;
                        }
                    }

                    disparity[y * W + x] = sub_disp;
                }
            }
        }
    }
}

void stereo_disparity_ref(const float *cost, float *disparity,
                          int W, int H, int max_disp) {
    int y, x;
    for (y = 0; y < H; y++) {
        for (x = 0; x < W; x++) {
            const float *pixel_cost = cost + (y * W + x) * max_disp;

            int best_d = 0;
            float best_cost = pixel_cost[0];
            int d;
            for (d = 1; d < max_disp; d++) {
                if (pixel_cost[d] < best_cost) {
                    best_cost = pixel_cost[d];
                    best_d = d;
                }
            }

            float sub_disp = (float)best_d;
            if (best_d > 0 && best_d < max_disp - 1) {
                float c_prev = pixel_cost[best_d - 1];
                float c_next = pixel_cost[best_d + 1];
                float denom = 2.0f * (c_prev + c_next - 2.0f * best_cost);
                if (fabsf(denom) > 1e-6f) {
                    sub_disp = (float)best_d + (c_prev - c_next) / denom;
                }
            }

            disparity[y * W + x] = sub_disp;
        }
    }
}

int main(void) {
    int W = TEST_W, H = TEST_H, max_disp = MAX_DISP;
    size_t cost_size = (size_t)W * H * max_disp;
    size_t img_size  = (size_t)W * H;

    float *cost      = (float *)malloc(cost_size * sizeof(float));
    float *disp_tile = (float *)malloc(img_size * sizeof(float));
    float *disp_ref  = (float *)malloc(img_size * sizeof(float));

    if (!cost || !disp_tile || !disp_ref) {
        fprintf(stderr, "Allocation failed\n");
        return 1;
    }

    /* Generate synthetic cost volume with known disparity */
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            int true_disp = 10 + (x * 20) / W;
            for (int d = 0; d < max_disp; d++) {
                float diff = (float)(d - true_disp);
                cost[(y * W + x) * max_disp + d] = diff * diff + 5.0f;
            }
        }
    }

    stereo_disparity(cost, disp_tile, W, H, max_disp);
    stereo_disparity_ref(cost, disp_ref, W, H, max_disp);

    float max_err = 0.0f;
    for (size_t i = 0; i < img_size; i++) {
        float err = fabsf(disp_tile[i] - disp_ref[i]);
        if (err > max_err) max_err = err;
    }

    printf("stereo_disparity: max_error = %e\n", max_err);
    int pass = (max_err < 1e-5f);
    printf("stereo_disparity: %s\n", pass ? "PASS" : "FAIL");

    free(cost); free(disp_tile); free(disp_ref);
    return pass ? 0 : 1;
}
