// Hand-written TDG MLIR -- Stereo Vision pipeline (AR/VR domain)
// E01 Productivity Comparison: hand-written MLIR format

module @stereo_vision {
  tdg.graph @stereo_vision {

    tdg.kernel @harris_corner {
      execution_target = "CGRA",
      source = "harris_corner.c",
      function = "harris_corner"
    }

    tdg.kernel @sad_matching {
      execution_target = "CGRA",
      source = "sad_matching.c",
      function = "sad_matching"
    }

    tdg.kernel @stereo_disparity {
      execution_target = "CGRA",
      source = "stereo_disparity.c",
      function = "stereo_disparity"
    }

    tdg.kernel @image_warp {
      execution_target = "CGRA",
      source = "image_warp.c",
      function = "image_warp"
    }

    tdg.kernel @post_filter {
      execution_target = "CGRA",
      source = "post_filter.c",
      function = "post_filter"
    }

    tdg.contract @harris_corner -> @sad_matching {
      ordering = "FIFO",
      data_type = "f32",
      rate = 4096 : i64,
      tile_shape = [64, 64],
      visibility = "LOCAL_SPM",
      double_buffering = true,
      backpressure = "BLOCK",
      may_fuse = true,
      may_replicate = true,
      may_pipeline = true,
      may_reorder = false,
      may_retile = true
    }

    tdg.contract @sad_matching -> @stereo_disparity {
      ordering = "FIFO",
      data_type = "f32",
      rate = 262144 : i64,
      tile_shape = [64, 64, 64],
      visibility = "LOCAL_SPM",
      double_buffering = true,
      backpressure = "BLOCK",
      may_fuse = true,
      may_replicate = true,
      may_pipeline = true,
      may_reorder = false,
      may_retile = true
    }

    tdg.contract @stereo_disparity -> @image_warp {
      ordering = "FIFO",
      data_type = "f32",
      rate = 4096 : i64,
      tile_shape = [64, 64],
      visibility = "LOCAL_SPM",
      double_buffering = false,
      backpressure = "BLOCK",
      may_fuse = true,
      may_replicate = true,
      may_pipeline = true,
      may_reorder = false,
      may_retile = true
    }

    tdg.contract @image_warp -> @post_filter {
      ordering = "FIFO",
      data_type = "f32",
      rate = 4096 : i64,
      tile_shape = [64, 64],
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
