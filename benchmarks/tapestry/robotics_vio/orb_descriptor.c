/*
 * ORB Descriptor Computation for Visual-Inertial Odometry.
 * Computes oriented BRIEF descriptors (256-bit) for keypoints.
 * Each descriptor is 8 x uint32 words. Uses a deterministic
 * sampling pattern rotated by keypoint orientation.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include "tile_utils.h"

#define IMG_W        320
#define IMG_H        240
#define NUM_KP       500
#define DESC_BITS    256
#define DESC_WORDS   (DESC_BITS / 32)
#define PATCH_RADIUS 15
#define NUM_PAIRS    256
#define TILE_KP      32

/* Sampling pair offsets (pre-defined, deterministic pattern) */
/* Each pair: (dx1, dy1, dx2, dy2) relative to keypoint, within patch */
static void generate_pairs(int pairs[][4], int n, int seed) {
    unsigned int state = (unsigned int)seed;
    int i;
    for (i = 0; i < n; i++) {
        state = (state * 1103515245 + 12345) & 0x7FFFFFFF;
        pairs[i][0] = (state % (2 * PATCH_RADIUS + 1)) - PATCH_RADIUS;
        state = (state * 1103515245 + 12345) & 0x7FFFFFFF;
        pairs[i][1] = (state % (2 * PATCH_RADIUS + 1)) - PATCH_RADIUS;
        state = (state * 1103515245 + 12345) & 0x7FFFFFFF;
        pairs[i][2] = (state % (2 * PATCH_RADIUS + 1)) - PATCH_RADIUS;
        state = (state * 1103515245 + 12345) & 0x7FFFFFFF;
        pairs[i][3] = (state % (2 * PATCH_RADIUS + 1)) - PATCH_RADIUS;
    }
}

static inline float safe_pixel(const float *img, int x, int y, int w, int h) {
    if (x < 0) x = 0;
    if (x >= w) x = w - 1;
    if (y < 0) y = 0;
    if (y >= h) y = h - 1;
    return img[y * w + x];
}

/* Compute intensity centroid orientation for a keypoint */
static float compute_orientation(const float *img, int cx, int cy,
                                 int w, int h) {
    float m01 = 0.0f, m10 = 0.0f;
    int dy, dx;
    for (dy = -PATCH_RADIUS; dy <= PATCH_RADIUS; dy++) {
        for (dx = -PATCH_RADIUS; dx <= PATCH_RADIUS; dx++) {
            if (dx * dx + dy * dy <= PATCH_RADIUS * PATCH_RADIUS) {
                float p = safe_pixel(img, cx + dx, cy + dy, w, h);
                m10 += dx * p;
                m01 += dy * p;
            }
        }
    }
    return atan2f(m01, m10);
}

/* Rotate a 2D point by angle */
static inline void rotate_point(int dx, int dy, float cos_a, float sin_a,
                                int *rx, int *ry) {
    *rx = (int)roundf(cos_a * dx - sin_a * dy);
    *ry = (int)roundf(sin_a * dx + cos_a * dy);
}

/* Compute ORB descriptor for a single keypoint */
static void compute_orb_single(const float *img, int kx, int ky,
                               int w, int h, const int pairs[][4],
                               uint32_t *desc) {
    float angle = compute_orientation(img, kx, ky, w, h);
    float cos_a = cosf(angle);
    float sin_a = sinf(angle);

    int word;
    for (word = 0; word < DESC_WORDS; word++) {
        desc[word] = 0;
    }

    int i;
    for (i = 0; i < NUM_PAIRS; i++) {
        int rx1, ry1, rx2, ry2;
        rotate_point(pairs[i][0], pairs[i][1], cos_a, sin_a, &rx1, &ry1);
        rotate_point(pairs[i][2], pairs[i][3], cos_a, sin_a, &rx2, &ry2);

        float p1 = safe_pixel(img, kx + rx1, ky + ry1, w, h);
        float p2 = safe_pixel(img, kx + rx2, ky + ry2, w, h);

        if (p1 < p2) {
            desc[i / 32] |= (1U << (i % 32));
        }
    }
}

/* Tiled ORB descriptor computation */
void orb_descriptor_tiled(const float *img, const int *kp_x, const int *kp_y,
                          int n_kp, uint32_t *descriptors, int w, int h,
                          const int pairs[][4]) {
    TILE_FOR(tk, 0, n_kp, TILE_KP) {
        int tk_end = TILE_END(tk, n_kp, TILE_KP);
        int k;
        for (k = tk; k < tk_end; k++) {
            compute_orb_single(img, kp_x[k], kp_y[k], w, h, pairs,
                             &descriptors[k * DESC_WORDS]);
        }
    }
}

/* Reference (non-tiled) implementation */
void orb_descriptor_ref(const float *img, const int *kp_x, const int *kp_y,
                        int n_kp, uint32_t *descriptors, int w, int h,
                        const int pairs[][4]) {
    int k;
    for (k = 0; k < n_kp; k++) {
        compute_orb_single(img, kp_x[k], kp_y[k], w, h, pairs,
                         &descriptors[k * DESC_WORDS]);
    }
}

int main(void) {
    int W = IMG_W, H = IMG_H;
    int n_kp = NUM_KP;

    float *image = (float *)malloc((size_t)W * H * sizeof(float));
    int *kp_x = (int *)malloc((size_t)n_kp * sizeof(int));
    int *kp_y = (int *)malloc((size_t)n_kp * sizeof(int));
    uint32_t *desc_t = (uint32_t *)calloc((size_t)n_kp * DESC_WORDS,
                                           sizeof(uint32_t));
    uint32_t *desc_r = (uint32_t *)calloc((size_t)n_kp * DESC_WORDS,
                                           sizeof(uint32_t));
    int (*pairs)[4] = (int (*)[4])malloc(NUM_PAIRS * 4 * sizeof(int));

    if (!image || !kp_x || !kp_y || !desc_t || !desc_r || !pairs) {
        fprintf(stderr, "Allocation failed\n");
        return 1;
    }

    /* Generate test image */
    int y, x;
    for (y = 0; y < H; y++) {
        for (x = 0; x < W; x++) {
            image[y * W + x] = (float)((x * 7 + y * 13 + 42) % 256);
        }
    }

    /* Generate sampling pairs */
    generate_pairs(pairs, NUM_PAIRS, 12345);

    /* Generate keypoint positions (avoiding borders) */
    int i;
    for (i = 0; i < n_kp; i++) {
        kp_x[i] = PATCH_RADIUS + (i * 37 + 11) % (W - 2 * PATCH_RADIUS);
        kp_y[i] = PATCH_RADIUS + (i * 53 + 7) % (H - 2 * PATCH_RADIUS);
    }

    orb_descriptor_tiled(image, kp_x, kp_y, n_kp, desc_t, W, H,
                         (const int (*)[4])pairs);
    orb_descriptor_ref(image, kp_x, kp_y, n_kp, desc_r, W, H,
                       (const int (*)[4])pairs);

    /* Compare descriptors */
    int mismatches = 0;
    for (i = 0; i < n_kp * DESC_WORDS; i++) {
        if (desc_t[i] != desc_r[i]) mismatches++;
    }

    printf("orb_descriptor: %d keypoints, %d word mismatches\n",
           n_kp, mismatches);
    int pass = (mismatches == 0);
    printf("orb_descriptor: %s\n", pass ? "PASS" : "FAIL");

    free(image); free(kp_x); free(kp_y);
    free(desc_t); free(desc_r); free(pairs);
    return pass ? 0 : 1;
}
