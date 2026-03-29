// TaskGraph C++ DSL -- Stereo Vision pipeline (AR/VR domain)
// E01 Productivity Comparison: Tier 1 DSL format

#include "tapestry/task_graph.h"

extern "C" {
void harris_corner(const float *, float *, int *, int *, int, int);
void sad_matching(const float *, const float *, float *, int, int, int);
void stereo_disparity(const float *, float *, int, int);
void image_warp(const float *, const float *, float *, int, int);
void post_filter(float *, int, int, int);
}

tapestry::TaskGraph buildStereoTDG() {
  tapestry::TaskGraph tg("stereo_vision");

  auto k_harris = tg.kernel("harris_corner", harris_corner);
  auto k_sad = tg.kernel("sad_matching", sad_matching);
  auto k_disp = tg.kernel("stereo_disparity", stereo_disparity);
  auto k_warp = tg.kernel("image_warp", image_warp);
  auto k_filt = tg.kernel("post_filter", post_filter);

  tg.connect(k_harris, k_sad)
      .ordering(tapestry::Ordering::FIFO)
      .data_type<float>()
      .shape("64x64")
      .data_volume(4096);

  tg.connect(k_sad, k_disp)
      .ordering(tapestry::Ordering::FIFO)
      .data_type<float>()
      .shape("64x64x64")
      .data_volume(262144);

  tg.connect(k_disp, k_warp)
      .ordering(tapestry::Ordering::FIFO)
      .data_type<float>()
      .shape("64x64")
      .data_volume(4096);

  tg.connect(k_warp, k_filt)
      .ordering(tapestry::Ordering::FIFO)
      .data_type<float>()
      .shape("64x64")
      .data_volume(4096);

  return tg;
}
