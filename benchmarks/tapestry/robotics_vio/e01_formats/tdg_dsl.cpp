// TaskGraph C++ DSL -- Visual-Inertial Odometry (Robotics domain)
// E01 Productivity Comparison: Tier 1 DSL format

#include "tapestry/task_graph.h"

extern "C" {
void imu_integration(const float *, float *, int, float);
void fast_detect(const float *, int *, int *, int, int, int, float);
void orb_descriptor(const float *, const int *, const int *,
                    unsigned *, int, int, int);
void feature_match(const unsigned *, const unsigned *,
                   int *, int, int, int);
void pose_estimate(const float *, const int *, float *, int, int);
}

tapestry::TaskGraph buildVIOTDG() {
  tapestry::TaskGraph tg("vio_pipeline");

  auto k_imu = tg.kernel("imu_integration", imu_integration);
  auto k_fast = tg.kernel("fast_detect", fast_detect);
  auto k_orb = tg.kernel("orb_descriptor", orb_descriptor);
  auto k_match = tg.kernel("feature_match", feature_match);
  auto k_pose = tg.kernel("pose_estimate", pose_estimate);

  tg.connect(k_imu, k_pose)
      .ordering(tapestry::Ordering::FIFO)
      .data_type<float>()
      .tile_shape({200, 3})
      .rate(600);

  tg.connect(k_fast, k_orb)
      .ordering(tapestry::Ordering::FIFO)
      .data_type<int32_t>()
      .tile_shape({500, 2})
      .rate(1000);

  tg.connect(k_orb, k_match)
      .ordering(tapestry::Ordering::FIFO)
      .data_type<uint32_t>()
      .tile_shape({500, 8})
      .rate(4000)
      .double_buffering(true);

  tg.connect(k_match, k_pose)
      .ordering(tapestry::Ordering::FIFO)
      .data_type<float>()
      .tile_shape({100, 4})
      .rate(400);

  return tg;
}
