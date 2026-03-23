/*
 * Image Warping -- bilinear interpolation using disparity map.
 * Warps the right image to match the left viewpoint.
 * 640x480 output, bilinear interpolation for sub-pixel accuracy.
 * Tiled: process TILE_W x TILE_H blocks.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "tile_utils.h"

#define IMG_W    640
#define IMG_H    480
#define TILE_W   64
#define TILE_H   32

/* Use smaller sizes for testing */
#define TEST_W   160
#define TEST_H   120

static inline float clampf(float v, float lo, float hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

void image_warp(const float *src, const float *disparity, float *dst,
                int W, int H) {
    TILE_FOR(ty, 0, H, TILE_H) {
        int ty_end = TILE_END(ty, H, TILE_H);
        TILE_FOR(tx, 0, W, TILE_W) {
            int tx_end = TILE_END(tx, W, TILE_W);

            int y, x;
            for (y = ty; y < ty_end; y++) {
                for (x = tx; x < tx_end; x++) {
                    float d = disparity[y * W + x];
                    float src_x = (float)x - d;

                    /* Clamp source coordinates */
                    src_x = clampf(src_x, 0.0f, (float)(W - 1));

                    /* Bilinear interpolation */
                    int x0 = (int)floorf(src_x);
                    int x1 = x0 + 1;
                    if (x1 >= W) x1 = W - 1;
                    float fx = src_x - (float)x0;

                    float v0 = src[y * W + x0];
                    float v1 = src[y * W + x1];
                    dst[y * W + x] = v0 * (1.0f - fx) + v1 * fx;
                }
            }
        }
    }
}

void image_warp_ref(const float *src, const float *disparity, float *dst,
                    int W, int H) {
    int y, x;
    for (y = 0; y < H; y++) {
        for (x = 0; x < W; x++) {
            float d = disparity[y * W + x];
            float src_x = (float)x - d;
            src_x = clampf(src_x, 0.0f, (float)(W - 1));

            int x0 = (int)floorf(src_x);
            int x1 = x0 + 1;
            if (x1 >= W) x1 = W - 1;
            float fx = src_x - (float)x0;

            float v0 = src[y * W + x0];
            float v1 = src[y * W + x1];
            dst[y * W + x] = v0 * (1.0f - fx) + v1 * fx;
        }
    }
}

int main(void) {
    int W = TEST_W, H = TEST_H;
    size_t img_size = (size_t)W * H;

    float *src       = (float *)malloc(img_size * sizeof(float));
    float *disp      = (float *)malloc(img_size * sizeof(float));
    float *dst_tile  = (float *)malloc(img_size * sizeof(float));
    float *dst_ref   = (float *)malloc(img_size * sizeof(float));

    if (!src || !disp || !dst_tile || !dst_ref) {
        fprintf(stderr, "Allocation failed\n");
        return 1;
    }

    /* Generate test image and disparity */
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            src[y * W + x] = (float)((x * 3 + y * 7 + 42) % 256);
            disp[y * W + x] = 5.0f + 10.0f * sinf((float)x / 30.0f);
        }
    }

    image_warp(src, disp, dst_tile, W, H);
    image_warp_ref(src, disp, dst_ref, W, H);

    float max_err = 0.0f;
    for (size_t i = 0; i < img_size; i++) {
        float err = fabsf(dst_tile[i] - dst_ref[i]);
        if (err > max_err) max_err = err;
    }

    printf("image_warp: max_error = %e\n", max_err);
    int pass = (max_err < 1e-5f);
    printf("image_warp: %s\n", pass ? "PASS" : "FAIL");

    free(src); free(disp); free(dst_tile); free(dst_ref);
    return pass ? 0 : 1;
}
