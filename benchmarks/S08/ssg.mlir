// Hand-written TDG MLIR -- STARK Proof pipeline (ZK domain)
// E01 Productivity Comparison: hand-written MLIR format

module @stark_proof {
  tdg.graph @stark_proof {

    tdg.kernel @ntt {
      execution_target = "CGRA",
      source = "ntt.c",
      function = "ntt_forward_tiled"
    }

    tdg.kernel @msm {
      execution_target = "CGRA",
      source = "msm.c",
      function = "msm"
    }

    tdg.kernel @poseidon_hash {
      execution_target = "CGRA",
      source = "poseidon_hash.c",
      function = "poseidon_hash"
    }

    tdg.kernel @poly_eval {
      execution_target = "CGRA",
      source = "poly_eval.c",
      function = "poly_eval"
    }

    tdg.kernel @proof_compose {
      execution_target = "CGRA",
      source = "proof_compose.c",
      function = "proof_compose"
    }

    tdg.contract @ntt -> @poly_eval {
      ordering = "FIFO",
      data_type = "u32",
      rate = 1024 : i64,
      tile_shape = [1024],
      visibility = "LOCAL_SPM",
      double_buffering = true,
      backpressure = "BLOCK",
      may_fuse = true,
      may_replicate = true,
      may_pipeline = true,
      may_reorder = false,
      may_retile = true
    }

    tdg.contract @poly_eval -> @proof_compose {
      ordering = "FIFO",
      data_type = "u32",
      rate = 256 : i64,
      tile_shape = [256],
      visibility = "LOCAL_SPM",
      double_buffering = false,
      backpressure = "BLOCK",
      may_fuse = true,
      may_replicate = true,
      may_pipeline = true,
      may_reorder = false,
      may_retile = true
    }

    tdg.contract @poseidon_hash -> @proof_compose {
      ordering = "FIFO",
      data_type = "u32",
      rate = 4 : i64,
      tile_shape = [4],
      visibility = "LOCAL_SPM",
      double_buffering = false,
      backpressure = "BLOCK",
      may_fuse = true,
      may_replicate = true,
      may_pipeline = true,
      may_reorder = false,
      may_retile = true
    }

    tdg.contract @msm -> @proof_compose {
      ordering = "FIFO",
      data_type = "u32",
      rate = 3 : i64,
      tile_shape = [3],
      visibility = "LOCAL_SPM",
      double_buffering = false,
      backpressure = "BLOCK",
      may_fuse = true,
      may_replicate = true,
      may_pipeline = true,
      may_reorder = false,
      may_retile = true
    }

    tdg.contract @ntt -> @poseidon_hash {
      ordering = "FIFO",
      data_type = "u32",
      rate = 8 : i64,
      tile_shape = [8],
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
