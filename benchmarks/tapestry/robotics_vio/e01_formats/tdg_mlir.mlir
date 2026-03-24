// Hand-written TDG MLIR -- Visual-Inertial Odometry (Robotics domain)
// E01 Productivity Comparison: hand-written MLIR format

module @vio_pipeline {
  tdg.graph @vio_pipeline {

    tdg.kernel @imu_integration {
      execution_target = "CGRA",
      source = "imu_integration.c",
      function = "imu_integration"
    }

    tdg.kernel @fast_detect {
      execution_target = "CGRA",
      source = "fast_detect.c",
      function = "fast_detect"
    }

    tdg.kernel @orb_descriptor {
      execution_target = "CGRA",
      source = "orb_descriptor.c",
      function = "orb_descriptor"
    }

    tdg.kernel @feature_match {
      execution_target = "CGRA",
      source = "feature_match.c",
      function = "feature_match"
    }

    tdg.kernel @pose_estimate {
      execution_target = "CGRA",
      source = "pose_estimate.c",
      function = "pose_estimate"
    }

    tdg.contract @imu_integration -> @pose_estimate {
      ordering = "FIFO",
      data_type = "f32",
      rate = 600 : i64,
      tile_shape = [200, 3],
      visibility = "LOCAL_SPM",
      double_buffering = false,
      backpressure = "BLOCK",
      may_fuse = true,
      may_replicate = true,
      may_pipeline = true,
      may_reorder = false,
      may_retile = true
    }

    tdg.contract @fast_detect -> @orb_descriptor {
      ordering = "FIFO",
      data_type = "i32",
      rate = 1000 : i64,
      tile_shape = [500, 2],
      visibility = "LOCAL_SPM",
      double_buffering = false,
      backpressure = "BLOCK",
      may_fuse = true,
      may_replicate = true,
      may_pipeline = true,
      may_reorder = false,
      may_retile = true
    }

    tdg.contract @orb_descriptor -> @feature_match {
      ordering = "FIFO",
      data_type = "u32",
      rate = 4000 : i64,
      tile_shape = [500, 8],
      visibility = "LOCAL_SPM",
      double_buffering = true,
      backpressure = "BLOCK",
      may_fuse = true,
      may_replicate = true,
      may_pipeline = true,
      may_reorder = false,
      may_retile = true
    }

    tdg.contract @feature_match -> @pose_estimate {
      ordering = "FIFO",
      data_type = "f32",
      rate = 400 : i64,
      tile_shape = [100, 4],
      visibility = "LOCAL_SPM",
      double_buffering = false,
      backpressure = "BLOCK",
      may_fuse = true,
      may_replicate = true,
      may_pipeline = true,
      may_reorder = false,
      may_retile = true
    }
  }
}
