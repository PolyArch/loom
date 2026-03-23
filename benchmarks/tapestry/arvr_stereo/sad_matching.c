/*
 * SAD Feature Matching -- Sum of Absolute Differences for stereo matching.
 * 640x480 images, 64 disparity levels, 7x7 matching window.
 * Computes SAD cost volume: cost[y][x][d] = sum of |left[y+wy][x+wx] - right[y+wy][x+wx-d]|
 * Tiled: per column block of TILE_W.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "tile_utils.h"

#define IMG_W        640
#define IMG_H        480
#define MAX_DISP     64
#define WIN_SIZE     7
#define WIN_HALF     (WIN_SIZE / 2)
#define TILE_W       64
#define TILE_H       32

/* Use smaller sizes for testing to keep runtime reasonable */
#define TEST_W       160
#define TEST_H       120

static inline int clamp(int v, int lo, int hi) {
    if (v < lo) return lo;
    if (v >= hi) return hi - 1;
    return v;
}

void sad_matching(const float *left, const float *right, float *cost,
                  int W, int H, int max_disp) {
    TILE_FOR(ty, 0, H, TILE_H) {
        int ty_end = TILE_END(ty, H, TILE_H);
        TILE_FOR(tx, 0, W, TILE_W) {
            int tx_end = TILE_END(tx, W, TILE_W);

            int y, x, d;
            for (y = ty; y < ty_end; y++) {
                for (x = tx; x < tx_end; x++) {
                    for (d = 0; d < max_disp; d++) {
                        float sad = 0.0f;
                        int wy, wx;
                        for (wy = -WIN_HALF; wy <= WIN_HALF; wy++) {
                            for (wx = -WIN_HALF; wx <= WIN_HALF; wx++) {
                                int ly = clamp(y + wy, 0, H);
                                int lx = clamp(x + wx, 0, W);
                                int rx = clamp(x + wx - d, 0, W);
                                sad += fabsf(left[ly * W + lx] - right[ly * W + rx]);
                            }
                        }
                        cost[(y * W + x) * max_disp + d] = sad;
                    }
                }
            }
        }
    }
}

void sad_matching_ref(const float *left, const float *right, float *cost,
                      int W, int H, int max_disp) {
    int y, x, d;
    for (y = 0; y < H; y++) {
        for (x = 0; x < W; x++) {
            for (d = 0; d < max_disp; d++) {
                float sad = 0.0f;
                int wy, wx;
                for (wy = -WIN_HALF; wy <= WIN_HALF; wy++) {
                    for (wx = -WIN_HALF; wx <= WIN_HALF; wx++) {
                        int ly = clamp(y + wy, 0, H);
                        int lx = clamp(x + wx, 0, W);
                        int rx = clamp(x + wx - d, 0, W);
                        sad += fabsf(left[ly * W + lx] - right[ly * W + rx]);
                    }
                }
                cost[(y * W + x) * max_disp + d] = sad;
            }
        }
    }
}

int main(void) {
    int W = TEST_W, H = TEST_H, max_disp = MAX_DISP;
    size_t img_size  = (size_t)W * H;
    size_t cost_size = (size_t)W * H * max_disp;

    float *left      = (float *)malloc(img_size * sizeof(float));
    float *right     = (float *)malloc(img_size * sizeof(float));
    float *cost_tile = (float *)malloc(cost_size * sizeof(float));
    float *cost_ref  = (float *)malloc(cost_size * sizeof(float));

    if (!left || !right || !cost_tile || !cost_ref) {
        fprintf(stderr, "Allocation failed\n");
        return 1;
    }

    /* Generate stereo pair: right image is left shifted by ~10 pixels */
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            float val = (float)((x * 3 + y * 7 + 42) % 256);
            left[y * W + x] = val;
            int sx = clamp(x + 10, 0, W);
            right[y * W + x] = (float)((sx * 3 + y * 7 + 42) % 256);
        }
    }

    sad_matching(left, right, cost_tile, W, H, max_disp);
    sad_matching_ref(left, right, cost_ref, W, H, max_disp);

    float max_err = 0.0f;
    for (size_t i = 0; i < cost_size; i++) {
        float err = fabsf(cost_tile[i] - cost_ref[i]);
        if (err > max_err) max_err = err;
    }

    printf("sad_matching: max_error = %e\n", max_err);
    int pass = (max_err < 1e-4f);
    printf("sad_matching: %s\n", pass ? "PASS" : "FAIL");

    free(left); free(right); free(cost_tile); free(cost_ref);
    return pass ? 0 : 1;
}
