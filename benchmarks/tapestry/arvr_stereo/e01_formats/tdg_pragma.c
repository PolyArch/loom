/* Pragma-annotated C -- Stereo Vision pipeline (AR/VR domain)
 * E01 Productivity Comparison: pragma-based baseline format
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define IMG_W  640
#define IMG_H  480
#define MAX_D  64

#pragma tapestry graph(stereo_vision)

#pragma tapestry kernel(harris_corner, target=CGRA, source="harris_corner.c")
void harris_corner(const float *image, float *response, int *corners,
                   int *num_corners, int W, int H);

#pragma tapestry kernel(sad_matching, target=CGRA, source="sad_matching.c")
void sad_matching(const float *left, const float *right, float *cost,
                  int W, int H, int max_disp);

#pragma tapestry kernel(stereo_disparity, target=CGRA, source="stereo_disparity.c")
void stereo_disparity(const float *cost, float *disp, int W, int H);

#pragma tapestry kernel(image_warp, target=CGRA, source="image_warp.c")
void image_warp(const float *src, const float *disp, float *dst,
                int W, int H);

#pragma tapestry kernel(post_filter, target=CGRA, source="post_filter.c")
void post_filter(float *image, int W, int H, int radius);

#pragma tapestry connect(harris_corner, sad_matching, \
    ordering=FIFO, data_type=f32, rate=4096, \
    tile_shape=[64,64], visibility=LOCAL_SPM, \
    double_buffering=true, backpressure=BLOCK, \
    may_fuse=true, may_replicate=true, may_pipeline=true, \
    may_reorder=false, may_retile=true)

#pragma tapestry connect(sad_matching, stereo_disparity, \
    ordering=FIFO, data_type=f32, rate=262144, \
    tile_shape=[64,64,64], visibility=LOCAL_SPM, \
    double_buffering=true, backpressure=BLOCK, \
    may_fuse=true, may_replicate=true, may_pipeline=true, \
    may_reorder=false, may_retile=true)

#pragma tapestry connect(stereo_disparity, image_warp, \
    ordering=FIFO, data_type=f32, rate=4096, \
    tile_shape=[64,64], visibility=LOCAL_SPM, \
    double_buffering=false, backpressure=BLOCK, \
    may_fuse=true, may_replicate=true, may_pipeline=true, \
    may_reorder=false, may_retile=true)

#pragma tapestry connect(image_warp, post_filter, \
    ordering=FIFO, data_type=f32, rate=4096, \
    tile_shape=[64,64], visibility=LOCAL_SPM, \
    double_buffering=false, backpressure=BLOCK, \
    may_fuse=true, may_replicate=true, may_pipeline=true, \
    may_reorder=false, may_retile=true)

void stereo_pipeline(const float *left, const float *right,
                     float *result) {
    size_t sz = (size_t)IMG_W * IMG_H;
    float *response = (float *)malloc(sz * sizeof(float));
    int *corners = (int *)malloc(sz * 2 * sizeof(int));
    int num_corners = 0;
    float *cost = (float *)malloc(sz * MAX_D * sizeof(float));
    float *disp = (float *)malloc(sz * sizeof(float));
    float *warped = (float *)malloc(sz * sizeof(float));

    harris_corner(left, response, corners, &num_corners, IMG_W, IMG_H);
    sad_matching(left, right, cost, IMG_W, IMG_H, MAX_D);
    stereo_disparity(cost, disp, IMG_W, IMG_H);
    image_warp(left, disp, warped, IMG_W, IMG_H);
    post_filter(warped, IMG_W, IMG_H, 3);
    memcpy(result, warped, sz * sizeof(float));

    free(response); free(corners); free(cost); free(disp); free(warped);
}
