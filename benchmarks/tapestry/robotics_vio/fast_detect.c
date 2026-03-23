/*
 * FAST Keypoint Detection for Visual-Inertial Odometry.
 * Processes a 320x240 grayscale image using the FAST-9 algorithm
 * with a 16-point Bresenham circle. Tiled processing in 32x32 blocks.
 * A pixel is a corner if 12 of 16 contiguous circle pixels are all
 * brighter (or all darker) than center +/- threshold.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "tile_utils.h"

#define IMG_W       320
#define IMG_H       240
#define TILE_W      32
#define TILE_H      32
#define FAST_THRESH 20.0f
#define FAST_N      12     /* need 12 contiguous of 16 */
#define MAX_CORNERS 10000

/* 16-point Bresenham circle offsets (radius 3) */
static const int circle_dx[16] = { 0,  1,  2,  3,  3,  3,  2,  1,
                                    0, -1, -2, -3, -3, -3, -2, -1};
static const int circle_dy[16] = {-3, -3, -2, -1,  0,  1,  2,  3,
                                    3,  3,  2,  1,  0, -1, -2, -3};

static inline float get_pixel(const float *img, int x, int y, int w, int h) {
    if (x < 0 || x >= w || y < 0 || y >= h) return 0.0f;
    return img[y * w + x];
}

/* Check if pixel (x,y) is a FAST corner */
static int is_fast_corner(const float *img, int x, int y, int w, int h,
                          float threshold) {
    float center = get_pixel(img, x, y, w, h);
    float hi = center + threshold;
    float lo = center - threshold;

    /* Quick rejection: check pixels at 0, 4, 8, 12 (cardinal directions) */
    int cardinal_bright = 0, cardinal_dark = 0;
    int card_idx[4] = {0, 4, 8, 12};
    int k;
    for (k = 0; k < 4; k++) {
        float p = get_pixel(img, x + circle_dx[card_idx[k]],
                           y + circle_dy[card_idx[k]], w, h);
        if (p > hi) cardinal_bright++;
        if (p < lo) cardinal_dark++;
    }
    /* At least 3 of 4 cardinal must be bright or dark */
    if (cardinal_bright < 3 && cardinal_dark < 3) return 0;

    /* Full check: look for 12 contiguous bright or dark pixels */
    int bright[16], dark[16];
    for (k = 0; k < 16; k++) {
        float p = get_pixel(img, x + circle_dx[k], y + circle_dy[k], w, h);
        bright[k] = (p > hi) ? 1 : 0;
        dark[k] = (p < lo) ? 1 : 0;
    }

    /* Check for N contiguous bright */
    int max_bright = 0, count = 0;
    for (k = 0; k < 32; k++) { /* wrap around */
        if (bright[k % 16]) {
            count++;
            if (count > max_bright) max_bright = count;
        } else {
            count = 0;
        }
    }
    if (max_bright >= FAST_N) return 1;

    /* Check for N contiguous dark */
    int max_dark = 0;
    count = 0;
    for (k = 0; k < 32; k++) {
        if (dark[k % 16]) {
            count++;
            if (count > max_dark) max_dark = count;
        } else {
            count = 0;
        }
    }
    if (max_dark >= FAST_N) return 1;

    return 0;
}

/* Tiled FAST detection */
int fast_detect_tiled(const float *img, int *corners_x, int *corners_y,
                      int max_corners, int W, int H, float threshold) {
    int num_corners = 0;

    TILE_FOR(ty, 0, H, TILE_H) {
        int ty_end = TILE_END(ty, H, TILE_H);
        TILE_FOR(tx, 0, W, TILE_W) {
            int tx_end = TILE_END(tx, W, TILE_W);
            int y, x;
            /* Skip border pixels (need radius 3 for circle) */
            int y_start = (ty < 3) ? 3 : ty;
            int x_start = (tx < 3) ? 3 : tx;
            int y_stop = (ty_end > H - 3) ? H - 3 : ty_end;
            int x_stop = (tx_end > W - 3) ? W - 3 : tx_end;

            for (y = y_start; y < y_stop; y++) {
                for (x = x_start; x < x_stop; x++) {
                    if (is_fast_corner(img, x, y, W, H, threshold)) {
                        if (num_corners < max_corners) {
                            corners_x[num_corners] = x;
                            corners_y[num_corners] = y;
                            num_corners++;
                        }
                    }
                }
            }
        }
    }
    return num_corners;
}

/* Reference (non-tiled) FAST detection */
int fast_detect_ref(const float *img, int *corners_x, int *corners_y,
                    int max_corners, int W, int H, float threshold) {
    int num_corners = 0;
    int y, x;
    for (y = 3; y < H - 3; y++) {
        for (x = 3; x < W - 3; x++) {
            if (is_fast_corner(img, x, y, W, H, threshold)) {
                if (num_corners < max_corners) {
                    corners_x[num_corners] = x;
                    corners_y[num_corners] = y;
                    num_corners++;
                }
            }
        }
    }
    return num_corners;
}

int main(void) {
    int W = IMG_W, H = IMG_H;
    size_t img_size = (size_t)W * H;

    float *image = (float *)malloc(img_size * sizeof(float));
    int *cx_t = (int *)malloc(MAX_CORNERS * sizeof(int));
    int *cy_t = (int *)malloc(MAX_CORNERS * sizeof(int));
    int *cx_r = (int *)malloc(MAX_CORNERS * sizeof(int));
    int *cy_r = (int *)malloc(MAX_CORNERS * sizeof(int));

    if (!image || !cx_t || !cy_t || !cx_r || !cy_r) {
        fprintf(stderr, "Allocation failed\n");
        return 1;
    }

    /* Generate test image with corner-like features */
    int y, x;
    for (y = 0; y < H; y++) {
        for (x = 0; x < W; x++) {
            /* Checkerboard with sharp edges */
            float val = ((x / 16 + y / 16) % 2) ? 200.0f : 50.0f;
            /* Add small gradient */
            val += (float)(x % 7) * 0.5f;
            image[y * W + x] = val;
        }
    }

    int n_tiled = fast_detect_tiled(image, cx_t, cy_t, MAX_CORNERS,
                                     W, H, FAST_THRESH);
    int n_ref = fast_detect_ref(image, cx_r, cy_r, MAX_CORNERS,
                                 W, H, FAST_THRESH);

    printf("fast_detect: tiled=%d corners, ref=%d corners\n", n_tiled, n_ref);

    /* Verify counts match */
    int pass = (n_tiled == n_ref);

    /* Verify corner positions match */
    if (pass && n_tiled > 0) {
        int i;
        for (i = 0; i < n_tiled; i++) {
            if (cx_t[i] != cx_r[i] || cy_t[i] != cy_r[i]) {
                pass = 0;
                printf("fast_detect: mismatch at corner %d: "
                       "tiled=(%d,%d) ref=(%d,%d)\n",
                       i, cx_t[i], cy_t[i], cx_r[i], cy_r[i]);
                break;
            }
        }
    }

    printf("fast_detect: %s\n", pass ? "PASS" : "FAIL");

    free(image); free(cx_t); free(cy_t); free(cx_r); free(cy_r);
    return pass ? 0 : 1;
}
