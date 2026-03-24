/* Pragma-annotated C -- Visual-Inertial Odometry (Robotics domain)
 * E01 Productivity Comparison: pragma-based baseline format
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define IMG_W       320
#define IMG_H       240
#define IMU_SAMPLES 200
#define MAX_CORNERS 500
#define DESC_WORDS  8

#pragma tapestry graph(vio_pipeline)

#pragma tapestry kernel(imu_integration, target=CGRA, source="imu_integration.c")
void imu_integration(const float *imu_data, float *state,
                     int n_samples, float dt);

#pragma tapestry kernel(fast_detect, target=CGRA, source="fast_detect.c")
int fast_detect(const float *img, int *cx, int *cy,
                int max_corners, int W, int H, float threshold);

#pragma tapestry kernel(orb_descriptor, target=CGRA, source="orb_descriptor.c")
void orb_descriptor(const float *img, const int *cx, const int *cy,
                    unsigned *desc, int n_corners, int W, int H);

#pragma tapestry kernel(feature_match, target=CGRA, source="feature_match.c")
void feature_match(const unsigned *desc_a, const unsigned *desc_b,
                   int *matches, int n_a, int n_b, int desc_words);

#pragma tapestry kernel(pose_estimate, target=CGRA, source="pose_estimate.c")
void pose_estimate(const float *imu_state, const int *matches,
                   float *pose, int n_imu, int n_matches);

#pragma tapestry connect(imu_integration, pose_estimate, \
    ordering=FIFO, data_type=f32, rate=600, \
    tile_shape=[200,3], visibility=LOCAL_SPM, \
    double_buffering=false, backpressure=BLOCK, \
    may_fuse=true, may_replicate=true, may_pipeline=true, \
    may_reorder=false, may_retile=true)

#pragma tapestry connect(fast_detect, orb_descriptor, \
    ordering=FIFO, data_type=i32, rate=1000, \
    tile_shape=[500,2], visibility=LOCAL_SPM, \
    double_buffering=false, backpressure=BLOCK, \
    may_fuse=true, may_replicate=true, may_pipeline=true, \
    may_reorder=false, may_retile=true)

#pragma tapestry connect(orb_descriptor, feature_match, \
    ordering=FIFO, data_type=u32, rate=4000, \
    tile_shape=[500,8], visibility=LOCAL_SPM, \
    double_buffering=true, backpressure=BLOCK, \
    may_fuse=true, may_replicate=true, may_pipeline=true, \
    may_reorder=false, may_retile=true)

#pragma tapestry connect(feature_match, pose_estimate, \
    ordering=FIFO, data_type=f32, rate=400, \
    tile_shape=[100,4], visibility=LOCAL_SPM, \
    double_buffering=false, backpressure=BLOCK, \
    may_fuse=true, may_replicate=true, may_pipeline=true, \
    may_reorder=false, may_retile=true)

void vio_pipeline(const float *image, const float *imu_data,
                  float *pose_out) {
    float imu_state[IMU_SAMPLES * 3];
    int cx[MAX_CORNERS], cy[MAX_CORNERS];
    unsigned desc_curr[MAX_CORNERS * DESC_WORDS];
    unsigned desc_prev[MAX_CORNERS * DESC_WORDS];
    int matches[MAX_CORNERS];

    imu_integration(imu_data, imu_state, IMU_SAMPLES, 0.005f);
    int n = fast_detect(image, cx, cy, MAX_CORNERS, IMG_W, IMG_H, 20.0f);
    orb_descriptor(image, cx, cy, desc_curr, n, IMG_W, IMG_H);
    feature_match(desc_curr, desc_prev, matches, n, n, DESC_WORDS);
    pose_estimate(imu_state, matches, pose_out, IMU_SAMPLES, n);
}
