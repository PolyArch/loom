// TaskGraph C++ DSL -- Graph Analytics pipeline (Graph domain)
// E01 Productivity Comparison: Tier 1 DSL format

#include "tapestry/task_graph.h"

extern "C" {
int bfs_tiled(const void *, int, int *);
void pagerank_spmv(const void *, const float *, float *, int, int);
int triangle_count(const void *, int);
void label_prop(const void *, int *, int, int);
}

tapestry::TaskGraph buildGraphTDG() {
  tapestry::TaskGraph tg("graph_analytics");

  auto k_bfs = tg.kernel("bfs_traversal", bfs_tiled);
  auto k_pr = tg.kernel("pagerank_spmv", pagerank_spmv);
  auto k_tc = tg.kernel("triangle_count", triangle_count);
  auto k_lp = tg.kernel("label_prop", label_prop);

  tg.connect(k_bfs, k_pr)
      .ordering(tapestry::Ordering::FIFO)
      .data_type<int32_t>()
      .tile_shape({1024})
      .rate(1024)
      .visibility(tapestry::Visibility::EXTERNAL_DRAM);

  tg.connect(k_pr, k_lp)
      .ordering(tapestry::Ordering::FIFO)
      .data_type<float>()
      .tile_shape({1024})
      .rate(1024)
      .visibility(tapestry::Visibility::EXTERNAL_DRAM);

  tg.connect(k_bfs, k_tc)
      .ordering(tapestry::Ordering::FIFO)
      .data_type<int32_t>()
      .tile_shape({1024})
      .rate(1024)
      .visibility(tapestry::Visibility::EXTERNAL_DRAM);

  return tg;
}
