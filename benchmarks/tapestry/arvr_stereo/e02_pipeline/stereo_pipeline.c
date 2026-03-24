/*
 * Entry function for auto_analyze: Stereo Vision pipeline.
 * Contains calls to all kernel functions with shared buffer arguments.
 * auto_analyze should detect 5 kernels and 4 edges.
 */

#include <stdlib.h>
#include <string.h>
#include <math.h>

#define IMG_W  640
#define IMG_H  480
#define MAX_D  64

__attribute__((noinline))
void harris_corner(const float *image, float *response,
                   int *corners, int *num_corners, int W, int H) {
    int x, y;
    *num_corners = 0;
    for (y = 1; y < H - 1; y++) {
        for (x = 1; x < W - 1; x++) {
            float gx = image[y*W+x+1] - image[y*W+x-1];
            float gy = image[(y+1)*W+x] - image[(y-1)*W+x];
            float r = gx*gx*gy*gy - (gx*gy)*(gx*gy) -
                      0.04f*(gx*gx+gy*gy)*(gx*gx+gy*gy);
            response[y * W + x] = r;
            if (r > 1000.0f) {
                int idx = *num_corners;
                corners[idx*2] = x;
                corners[idx*2+1] = y;
                (*num_corners)++;
            }
        }
    }
}

__attribute__((noinline))
void sad_matching(const float *left, const float *right, float *cost,
                  int W, int H, int max_disp) {
    int x, y, d, dx, dy;
    for (y = 0; y < H; y++)
        for (x = 0; x < W; x++)
            for (d = 0; d < max_disp; d++) {
                float sad = 0.0f;
                for (dy = -2; dy <= 2; dy++)
                    for (dx = -2; dx <= 2; dx++) {
                        int ly = y+dy, lx = x+dx, rx = x+dx-d;
                        if (ly >= 0 && ly < H && lx >= 0 && lx < W &&
                            rx >= 0 && rx < W)
                            sad += fabsf(left[ly*W+lx] - right[ly*W+rx]);
                    }
                cost[(y*W+x)*max_disp+d] = sad;
            }
}

__attribute__((noinline))
void stereo_disparity(const float *cost, float *disp, int W, int H) {
    int x, y, d;
    for (y = 0; y < H; y++)
        for (x = 0; x < W; x++) {
            float best = cost[(y*W+x)*MAX_D];
            int best_d = 0;
            for (d = 1; d < MAX_D; d++) {
                float c = cost[(y*W+x)*MAX_D+d];
                if (c < best) { best = c; best_d = d; }
            }
            disp[y*W+x] = (float)best_d;
        }
}

__attribute__((noinline))
void image_warp(const float *src, const float *disp, float *dst,
                int W, int H) {
    int x, y;
    for (y = 0; y < H; y++)
        for (x = 0; x < W; x++) {
            int sx = x - (int)disp[y*W+x];
            if (sx >= 0 && sx < W) dst[y*W+x] = src[y*W+sx];
            else dst[y*W+x] = 0.0f;
        }
}

__attribute__((noinline))
void post_filter(float *image, int W, int H, int radius) {
    int x, y, dx, dy;
    float *tmp = (float *)malloc((size_t)W * H * sizeof(float));
    for (y = 0; y < H; y++)
        for (x = 0; x < W; x++) {
            float sum = 0.0f;
            int cnt = 0;
            for (dy = -radius; dy <= radius; dy++)
                for (dx = -radius; dx <= radius; dx++) {
                    int ny = y+dy, nx = x+dx;
                    if (ny >= 0 && ny < H && nx >= 0 && nx < W) {
                        sum += image[ny*W+nx]; cnt++;
                    }
                }
            tmp[y*W+x] = sum / (float)cnt;
        }
    memcpy(image, tmp, (size_t)W * H * sizeof(float));
    free(tmp);
}

/* Entry function for auto_analyze */
void stereo_pipeline(const float *left, const float *right,
                     float *result) {
    size_t sz = (size_t)IMG_W * IMG_H;
    float *response = (float *)malloc(sz * sizeof(float));
    int *corners = (int *)malloc(sz * 2 * sizeof(int));
    int num_corners = 0;
    float *cost = (float *)malloc(sz * MAX_D * sizeof(float));
    float *disp = (float *)malloc(sz * sizeof(float));

    harris_corner(left, response, corners, &num_corners, IMG_W, IMG_H);
    sad_matching(left, right, cost, IMG_W, IMG_H, MAX_D);
    stereo_disparity(cost, disp, IMG_W, IMG_H);
    image_warp(left, disp, result, IMG_W, IMG_H);
    post_filter(result, IMG_W, IMG_H, 3);

    free(response); free(corners); free(cost); free(disp);
}
