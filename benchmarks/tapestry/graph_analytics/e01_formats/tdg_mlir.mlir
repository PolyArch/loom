// Hand-written TDG MLIR -- Graph Analytics pipeline (Graph domain)
// E01 Productivity Comparison: hand-written MLIR format

module @graph_analytics {
  tdg.graph @graph_analytics {

    tdg.kernel @bfs_traversal {
      execution_target = "CGRA",
      source = "bfs_traversal.c",
      function = "bfs_tiled"
    }

    tdg.kernel @pagerank_spmv {
      execution_target = "CGRA",
      source = "pagerank_spmv.c",
      function = "pagerank_spmv"
    }

    tdg.kernel @triangle_count {
      execution_target = "CGRA",
      source = "triangle_count.c",
      function = "triangle_count"
    }

    tdg.kernel @label_prop {
      execution_target = "CGRA",
      source = "label_prop.c",
      function = "label_prop"
    }

    tdg.contract @bfs_traversal -> @pagerank_spmv {
      ordering = "FIFO",
      data_type = "i32",
      rate = 1024 : i64,
      tile_shape = [1024],
      visibility = "EXTERNAL_DRAM",
      double_buffering = false,
      backpressure = "BLOCK",
      may_fuse = true,
      may_replicate = true,
      may_pipeline = true,
      may_reorder = false,
      may_retile = true
    }

    tdg.contract @pagerank_spmv -> @label_prop {
      ordering = "FIFO",
      data_type = "f32",
      rate = 1024 : i64,
      tile_shape = [1024],
      visibility = "EXTERNAL_DRAM",
      double_buffering = false,
      backpressure = "BLOCK",
      may_fuse = true,
      may_replicate = true,
      may_pipeline = true,
      may_reorder = false,
      may_retile = true
    }

    tdg.contract @bfs_traversal -> @triangle_count {
      ordering = "FIFO",
      data_type = "i32",
      rate = 1024 : i64,
      tile_shape = [1024],
      visibility = "EXTERNAL_DRAM",
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
