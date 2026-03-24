// Hand-written TDG MLIR -- OFDM Receiver Chain (DSP domain)
// E01 Productivity Comparison: hand-written MLIR format

module @ofdm_receiver {
  tdg.graph @ofdm_receiver {

    tdg.kernel @fft_butterfly {
      execution_target = "CGRA",
      source = "fft_butterfly.c",
      function = "fft_butterfly"
    }

    tdg.kernel @channel_est {
      execution_target = "CGRA",
      source = "channel_est.c",
      function = "channel_est"
    }

    tdg.kernel @equalizer {
      execution_target = "CGRA",
      source = "equalizer.c",
      function = "equalizer"
    }

    tdg.kernel @qam_demod {
      execution_target = "CGRA",
      source = "qam_demod.c",
      function = "qam_demod"
    }

    tdg.kernel @viterbi {
      execution_target = "CGRA",
      source = "viterbi.c",
      function = "viterbi"
    }

    tdg.kernel @crc_check {
      execution_target = "CGRA",
      source = "crc_check.c",
      function = "crc_check"
    }

    tdg.contract @fft_butterfly -> @channel_est {
      ordering = "FIFO",
      data_type = "complex64",
      rate = 4096 : i64,
      tile_shape = [4096],
      visibility = "LOCAL_SPM",
      double_buffering = true,
      backpressure = "BLOCK",
      may_fuse = true,
      may_replicate = true,
      may_pipeline = true,
      may_reorder = false,
      may_retile = true
    }

    tdg.contract @channel_est -> @equalizer {
      ordering = "FIFO",
      data_type = "complex64",
      rate = 1200 : i64,
      tile_shape = [1200],
      visibility = "LOCAL_SPM",
      double_buffering = false,
      backpressure = "BLOCK",
      may_fuse = true,
      may_replicate = true,
      may_pipeline = true,
      may_reorder = false,
      may_retile = true
    }

    tdg.contract @equalizer -> @qam_demod {
      ordering = "FIFO",
      data_type = "complex64",
      rate = 1200 : i64,
      tile_shape = [128],
      visibility = "LOCAL_SPM",
      double_buffering = false,
      backpressure = "BLOCK",
      may_fuse = true,
      may_replicate = true,
      may_pipeline = true,
      may_reorder = false,
      may_retile = true
    }

    tdg.contract @qam_demod -> @viterbi {
      ordering = "FIFO",
      data_type = "i32",
      rate = 7200 : i64,
      tile_shape = [7200],
      visibility = "LOCAL_SPM",
      double_buffering = true,
      backpressure = "BLOCK",
      may_fuse = true,
      may_replicate = true,
      may_pipeline = true,
      may_reorder = false,
      may_retile = true
    }

    tdg.contract @viterbi -> @crc_check {
      ordering = "FIFO",
      data_type = "i32",
      rate = 1800 : i64,
      tile_shape = [3600],
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
