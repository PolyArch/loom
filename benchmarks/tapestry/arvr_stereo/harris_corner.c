/*
 * Harris Corner Detection for stereo vision pipeline.
 * 640x480 grayscale image, processed in 64x64 tiles with 1-pixel halo.
 * Sobel gradients -> structure tensor -> corner response.
 * R = det(M) - k * trace(M)^2, k=0.04
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "tile_utils.h"

#define IMG_W      640
#define IMG_H      480
#define TILE_W     64
#define TILE_H     64
#define HARRIS_K   0.04f
#define THRESHOLD  1000.0f

/* Clamp pixel access */
static inline int clamp_idx(int v, int lo, int hi) {
    if (v < lo) return lo;
    if (v >= hi) return hi - 1;
    return v;
}

static inline float px(const float *img, int x, int y, int w, int h) {
    return img[clamp_idx(y, 0, h) * w + clamp_idx(x, 0, w)];
}

void harris_corner(const float *image, float *response, int *corners,
                   int *num_corners, int W, int H) {
    float *Ix  = (float *)calloc((size_t)W * H, sizeof(float));
    float *Iy  = (float *)calloc((size_t)W * H, sizeof(float));
    if (!Ix || !Iy) { free(Ix); free(Iy); return; }

    *num_corners = 0;

    /* Pass 1: Compute Sobel gradients for the entire image (tiled) */
    TILE_FOR(ty, 0, H, TILE_H) {
        int ty_end = TILE_END(ty, H, TILE_H);
        TILE_FOR(tx, 0, W, TILE_W) {
            int tx_end = TILE_END(tx, W, TILE_W);
            int y, x;
            for (y = ty; y < ty_end; y++) {
                for (x = tx; x < tx_end; x++) {
                    /* Sobel X: [-1 0 1; -2 0 2; -1 0 1] */
                    float gx = -px(image, x-1, y-1, W, H)
                               + px(image, x+1, y-1, W, H)
                               - 2.0f * px(image, x-1, y, W, H)
                               + 2.0f * px(image, x+1, y, W, H)
                               - px(image, x-1, y+1, W, H)
                               + px(image, x+1, y+1, W, H);
                    /* Sobel Y: [-1 -2 -1; 0 0 0; 1 2 1] */
                    float gy = -px(image, x-1, y-1, W, H)
                               - 2.0f * px(image, x, y-1, W, H)
                               - px(image, x+1, y-1, W, H)
                               + px(image, x-1, y+1, W, H)
                               + 2.0f * px(image, x, y+1, W, H)
                               + px(image, x+1, y+1, W, H);
                    Ix[y * W + x] = gx;
                    Iy[y * W + x] = gy;
                }
            }
        }
    }

    /* Pass 2: Compute structure tensor and corner response (tiled) */
    TILE_FOR(ty, 0, H, TILE_H) {
        int ty_end = TILE_END(ty, H, TILE_H);
        TILE_FOR(tx, 0, W, TILE_W) {
            int tx_end = TILE_END(tx, W, TILE_W);
            int y, x;
            for (y = ty; y < ty_end; y++) {
                for (x = tx; x < tx_end; x++) {
                    /* Sum over 3x3 window */
                    float sxx = 0.0f, syy = 0.0f, sxy = 0.0f;
                    int wy, wx;
                    for (wy = -1; wy <= 1; wy++) {
                        for (wx = -1; wx <= 1; wx++) {
                            int cx = clamp_idx(x + wx, 0, W);
                            int cy = clamp_idx(y + wy, 0, H);
                            float ix = Ix[cy * W + cx];
                            float iy = Iy[cy * W + cx];
                            sxx += ix * ix;
                            syy += iy * iy;
                            sxy += ix * iy;
                        }
                    }
                    float det = sxx * syy - sxy * sxy;
                    float trace = sxx + syy;
                    float r = det - HARRIS_K * trace * trace;
                    response[y * W + x] = r;

                    if (r > THRESHOLD) {
                        int idx = *num_corners;
                        corners[idx * 2 + 0] = x;
                        corners[idx * 2 + 1] = y;
                        (*num_corners)++;
                    }
                }
            }
        }
    }

    free(Ix); free(Iy);
}

void harris_corner_ref(const float *image, float *response,
                       int W, int H) {
    float *Ix = (float *)calloc((size_t)W * H, sizeof(float));
    float *Iy = (float *)calloc((size_t)W * H, sizeof(float));
    if (!Ix || !Iy) { free(Ix); free(Iy); return; }

    int y, x;
    for (y = 0; y < H; y++) {
        for (x = 0; x < W; x++) {
            float gx = -px(image, x-1, y-1, W, H) + px(image, x+1, y-1, W, H)
                       - 2.0f * px(image, x-1, y, W, H) + 2.0f * px(image, x+1, y, W, H)
                       - px(image, x-1, y+1, W, H) + px(image, x+1, y+1, W, H);
            float gy = -px(image, x-1, y-1, W, H) - 2.0f * px(image, x, y-1, W, H)
                       - px(image, x+1, y-1, W, H) + px(image, x-1, y+1, W, H)
                       + 2.0f * px(image, x, y+1, W, H) + px(image, x+1, y+1, W, H);
            Ix[y * W + x] = gx;
            Iy[y * W + x] = gy;
        }
    }

    for (y = 0; y < H; y++) {
        for (x = 0; x < W; x++) {
            float sxx = 0.0f, syy = 0.0f, sxy = 0.0f;
            int wy, wx;
            for (wy = -1; wy <= 1; wy++) {
                for (wx = -1; wx <= 1; wx++) {
                    int cx = clamp_idx(x + wx, 0, W);
                    int cy = clamp_idx(y + wy, 0, H);
                    float ix = Ix[cy * W + cx];
                    float iy = Iy[cy * W + cx];
                    sxx += ix * ix;
                    syy += iy * iy;
                    sxy += ix * iy;
                }
            }
            float det = sxx * syy - sxy * sxy;
            float trace = sxx + syy;
            response[y * W + x] = det - HARRIS_K * trace * trace;
        }
    }

    free(Ix); free(Iy);
}

int main(void) {
    int W = IMG_W, H = IMG_H;
    size_t img_size = (size_t)W * H;

    float *image     = (float *)malloc(img_size * sizeof(float));
    float *resp_tile = (float *)calloc(img_size, sizeof(float));
    float *resp_ref  = (float *)calloc(img_size, sizeof(float));
    int   *corners   = (int *)malloc(img_size * 2 * sizeof(int));

    if (!image || !resp_tile || !resp_ref || !corners) {
        fprintf(stderr, "Allocation failed\n");
        return 1;
    }

    /* Generate test image: checkerboard pattern with gradients */
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            float checker = ((x / 32 + y / 32) % 2) ? 200.0f : 50.0f;
            float noise = (float)((x * 7 + y * 13) % 17) / 17.0f * 5.0f;
            image[y * W + x] = checker + noise;
        }
    }

    int num_corners = 0;
    harris_corner(image, resp_tile, corners, &num_corners, W, H);
    harris_corner_ref(image, resp_ref, W, H);

    float max_err = 0.0f;
    for (size_t i = 0; i < img_size; i++) {
        float err = fabsf(resp_tile[i] - resp_ref[i]);
        if (err > max_err) max_err = err;
    }

    printf("harris_corner: max_error = %e, corners_found = %d\n", max_err, num_corners);
    int pass = (max_err < 1e-2f);
    printf("harris_corner: %s\n", pass ? "PASS" : "FAIL");

    free(image); free(resp_tile); free(resp_ref); free(corners);
    return pass ? 0 : 1;
}
