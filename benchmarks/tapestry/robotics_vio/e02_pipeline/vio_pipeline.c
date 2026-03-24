/*
 * Entry function for auto_analyze: Visual-Inertial Odometry pipeline.
 * Contains calls to all kernel functions with shared buffer arguments.
 * auto_analyze should detect 5 kernels and 4 edges.
 */

#include <stdlib.h>
#include <string.h>
#include <math.h>

#define IMG_W       320
#define IMG_H       240
#define IMU_SAMPLES 200
#define MAX_CORNERS 500
#define DESC_WORDS  8

__attribute__((noinline))
void imu_integration(const float *imu_data, float *state,
                     int n_samples, float dt) {
    int i;
    float vx = 0, vy = 0, vz = 0;
    for (i = 0; i < n_samples; i++) {
        float ax = imu_data[i * 6 + 0];
        float ay = imu_data[i * 6 + 1];
        float az = imu_data[i * 6 + 2];
        vx += ax * dt; vy += ay * dt; vz += az * dt;
        state[i * 3 + 0] = vx;
        state[i * 3 + 1] = vy;
        state[i * 3 + 2] = vz;
    }
}

__attribute__((noinline))
int fast_detect(const float *img, int *cx, int *cy,
                int max_corners, int W, int H, float threshold) {
    int num = 0, x, y;
    for (y = 3; y < H - 3; y++)
        for (x = 3; x < W - 3; x++) {
            float c = img[y * W + x];
            float u = img[(y-3)*W+x], d = img[(y+3)*W+x];
            float l = img[y*W+x-3], r = img[y*W+x+3];
            int bright = (u > c+threshold) + (d > c+threshold) +
                         (l > c+threshold) + (r > c+threshold);
            int dark = (u < c-threshold) + (d < c-threshold) +
                       (l < c-threshold) + (r < c-threshold);
            if ((bright >= 3 || dark >= 3) && num < max_corners) {
                cx[num] = x; cy[num] = y; num++;
            }
        }
    return num;
}

__attribute__((noinline))
void orb_descriptor(const float *img, const int *cx, const int *cy,
                    unsigned *desc, int n_corners, int W, int H) {
    int i, w;
    for (i = 0; i < n_corners; i++) {
        int x = cx[i], y = cy[i];
        for (w = 0; w < DESC_WORDS; w++) {
            unsigned bits = 0;
            int b;
            for (b = 0; b < 32; b++) {
                int ox1 = (b * 3 + w * 7) % 15 - 7;
                int oy1 = (b * 5 + w * 3) % 15 - 7;
                int ox2 = (b * 7 + w * 11) % 15 - 7;
                int oy2 = (b * 11 + w * 5) % 15 - 7;
                int x1 = x+ox1, y1 = y+oy1, x2 = x+ox2, y2 = y+oy2;
                if (x1 >= 0 && x1 < W && y1 >= 0 && y1 < H &&
                    x2 >= 0 && x2 < W && y2 >= 0 && y2 < H) {
                    if (img[y1*W+x1] < img[y2*W+x2])
                        bits |= (1u << b);
                }
            }
            desc[i * DESC_WORDS + w] = bits;
        }
    }
}

__attribute__((noinline))
void feature_match(const unsigned *desc_a, const unsigned *desc_b,
                   int *matches, int n_a, int n_b, int desc_words) {
    int i, j, w;
    for (i = 0; i < n_a; i++) {
        int best = -1, best_dist = 256 * desc_words + 1;
        for (j = 0; j < n_b; j++) {
            int dist = 0;
            for (w = 0; w < desc_words; w++) {
                unsigned x = desc_a[i*desc_words+w] ^ desc_b[j*desc_words+w];
                while (x) { dist++; x &= x - 1; }
            }
            if (dist < best_dist) { best_dist = dist; best = j; }
        }
        matches[i] = best;
    }
}

__attribute__((noinline))
void pose_estimate(const float *imu_state, const int *matches,
                   float *pose, int n_imu, int n_matches) {
    int i;
    float sum_x = 0, sum_y = 0, sum_z = 0;
    for (i = 0; i < n_imu; i++) {
        sum_x += imu_state[i*3+0];
        sum_y += imu_state[i*3+1];
        sum_z += imu_state[i*3+2];
    }
    pose[0] = sum_x / (float)n_imu;
    pose[1] = sum_y / (float)n_imu;
    pose[2] = sum_z / (float)n_imu;
    int valid_matches = 0;
    for (i = 0; i < n_matches; i++)
        if (matches[i] >= 0) valid_matches++;
    pose[3] = (float)valid_matches;
}

/* Entry function for auto_analyze */
void vio_pipeline(const float *image, const float *imu_data,
                  float *pose_out) {
    float *imu_state = (float *)malloc(IMU_SAMPLES * 3 * sizeof(float));
    int *cx = (int *)malloc(MAX_CORNERS * sizeof(int));
    int *cy = (int *)malloc(MAX_CORNERS * sizeof(int));
    unsigned *desc_curr = (unsigned *)malloc(
        MAX_CORNERS * DESC_WORDS * sizeof(unsigned));
    unsigned *desc_prev = (unsigned *)malloc(
        MAX_CORNERS * DESC_WORDS * sizeof(unsigned));
    int *matches = (int *)malloc(MAX_CORNERS * sizeof(int));

    memset(desc_prev, 0, MAX_CORNERS * DESC_WORDS * sizeof(unsigned));

    imu_integration(imu_data, imu_state, IMU_SAMPLES, 0.005f);
    int n = fast_detect(image, cx, cy, MAX_CORNERS, IMG_W, IMG_H, 20.0f);
    orb_descriptor(image, cx, cy, desc_curr, n, IMG_W, IMG_H);
    feature_match(desc_curr, desc_prev, matches, n, n, DESC_WORDS);
    pose_estimate(imu_state, matches, pose_out, IMU_SAMPLES, n);

    free(imu_state); free(cx); free(cy);
    free(desc_curr); free(desc_prev); free(matches);
}
