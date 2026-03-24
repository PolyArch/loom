// Hand-written TDG MLIR -- Transformer Layer pipeline (AI/LLM domain)
// E01 Productivity Comparison: hand-written MLIR format

module @transformer_layer {
  tdg.graph @transformer_layer {

    tdg.kernel @qkv_proj {
      execution_target = "CGRA",
      source = "qkv_proj.c",
      function = "qkv_proj"
    }

    tdg.kernel @attn_score {
      execution_target = "CGRA",
      source = "attn_score.c",
      function = "attn_score"
    }

    tdg.kernel @softmax {
      execution_target = "CGRA",
      source = "softmax.c",
      function = "softmax"
    }

    tdg.kernel @attn_output {
      execution_target = "CGRA",
      source = "attn_output.c",
      function = "attn_output"
    }

    tdg.kernel @ffn1 {
      execution_target = "CGRA",
      source = "ffn1.c",
      function = "ffn1"
    }

    tdg.kernel @gelu {
      execution_target = "CGRA",
      source = "gelu.c",
      function = "gelu"
    }

    tdg.kernel @ffn2 {
      execution_target = "CGRA",
      source = "ffn2.c",
      function = "ffn2"
    }

    tdg.kernel @layernorm {
      execution_target = "CGRA",
      source = "layernorm.c",
      function = "layernorm"
    }

    tdg.contract @qkv_proj -> @attn_score {
      ordering = "FIFO",
      data_type = "f32",
      rate = 2048 : i64,
      tile_shape = [32, 64],
      visibility = "LOCAL_SPM",
      double_buffering = false,
      backpressure = "BLOCK",
      may_fuse = true,
      may_replicate = true,
      may_pipeline = true,
      may_reorder = false,
      may_retile = true
    }

    tdg.contract @attn_score -> @softmax {
      ordering = "FIFO",
      data_type = "f32",
      rate = 4096 : i64,
      tile_shape = [32, 128],
      visibility = "LOCAL_SPM",
      double_buffering = false,
      backpressure = "BLOCK",
      may_fuse = true,
      may_replicate = true,
      may_pipeline = true,
      may_reorder = false,
      may_retile = true
    }

    tdg.contract @softmax -> @attn_output {
      ordering = "FIFO",
      data_type = "f32",
      rate = 4096 : i64,
      tile_shape = [32, 128],
      visibility = "LOCAL_SPM",
      double_buffering = false,
      backpressure = "BLOCK",
      may_fuse = true,
      may_replicate = true,
      may_pipeline = true,
      may_reorder = false,
      may_retile = true
    }

    tdg.contract @attn_output -> @ffn1 {
      ordering = "FIFO",
      data_type = "f32",
      rate = 16384 : i64,
      tile_shape = [32, 512],
      visibility = "LOCAL_SPM",
      double_buffering = true,
      backpressure = "BLOCK",
      may_fuse = true,
      may_replicate = true,
      may_pipeline = true,
      may_reorder = false,
      may_retile = true
    }

    tdg.contract @ffn1 -> @gelu {
      ordering = "FIFO",
      data_type = "f32",
      rate = 65536 : i64,
      tile_shape = [32, 2048],
      visibility = "LOCAL_SPM",
      double_buffering = false,
      backpressure = "BLOCK",
      may_fuse = true,
      may_replicate = true,
      may_pipeline = true,
      may_reorder = false,
      may_retile = true
    }

    tdg.contract @gelu -> @ffn2 {
      ordering = "FIFO",
      data_type = "f32",
      rate = 65536 : i64,
      tile_shape = [32, 2048],
      visibility = "LOCAL_SPM",
      double_buffering = false,
      backpressure = "BLOCK",
      may_fuse = true,
      may_replicate = true,
      may_pipeline = true,
      may_reorder = false,
      may_retile = true
    }

    tdg.contract @ffn2 -> @layernorm {
      ordering = "FIFO",
      data_type = "f32",
      rate = 16384 : i64,
      tile_shape = [32, 512],
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
