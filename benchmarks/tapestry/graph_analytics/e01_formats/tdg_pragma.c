/* Pragma-annotated C -- Graph Analytics pipeline (Graph domain)
 * E01 Productivity Comparison: pragma-based baseline format
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    int num_vertices;
    int num_edges;
    int *row_ptr;
    int *col_idx;
} csr_graph_t;

#define NUM_VERTICES 1024
#define NUM_ITERS    20

#pragma tapestry graph(graph_analytics)

#pragma tapestry kernel(bfs_traversal, target=CGRA, source="bfs_traversal.c")
int bfs_tiled(const csr_graph_t *g, int source, int *level);

#pragma tapestry kernel(pagerank_spmv, target=CGRA, source="pagerank_spmv.c")
void pagerank_spmv(const csr_graph_t *g, const float *rank_in,
                   float *rank_out, int nv, int iters);

#pragma tapestry kernel(triangle_count, target=CGRA, source="triangle_count.c")
int triangle_count(const csr_graph_t *g, int nv);

#pragma tapestry kernel(label_prop, target=CGRA, source="label_prop.c")
void label_prop(const csr_graph_t *g, int *labels, int nv, int iters);

#pragma tapestry connect(bfs_traversal, pagerank_spmv, \
    ordering=FIFO, data_type=i32, rate=1024, \
    tile_shape=[1024], visibility=EXTERNAL_DRAM, \
    double_buffering=false, backpressure=BLOCK, \
    may_fuse=true, may_replicate=true, may_pipeline=true, \
    may_reorder=false, may_retile=true)

#pragma tapestry connect(pagerank_spmv, label_prop, \
    ordering=FIFO, data_type=f32, rate=1024, \
    tile_shape=[1024], visibility=EXTERNAL_DRAM, \
    double_buffering=false, backpressure=BLOCK, \
    may_fuse=true, may_replicate=true, may_pipeline=true, \
    may_reorder=false, may_retile=true)

#pragma tapestry connect(bfs_traversal, triangle_count, \
    ordering=FIFO, data_type=i32, rate=1024, \
    tile_shape=[1024], visibility=EXTERNAL_DRAM, \
    double_buffering=false, backpressure=BLOCK, \
    may_fuse=true, may_replicate=true, may_pipeline=true, \
    may_reorder=false, may_retile=true)

void graph_pipeline(const csr_graph_t *g) {
    int nv = g->num_vertices;
    int *bfs_levels = (int *)malloc((size_t)nv * sizeof(int));
    float *rank_in = (float *)malloc((size_t)nv * sizeof(float));
    float *rank_out = (float *)malloc((size_t)nv * sizeof(float));
    int *labels = (int *)malloc((size_t)nv * sizeof(int));

    float init_rank = 1.0f / (float)nv;
    int i;
    for (i = 0; i < nv; i++) rank_in[i] = init_rank;

    bfs_tiled(g, 0, bfs_levels);
    pagerank_spmv(g, rank_in, rank_out, nv, NUM_ITERS);
    triangle_count(g, nv);
    label_prop(g, labels, nv, NUM_ITERS);

    free(bfs_levels); free(rank_in); free(rank_out); free(labels);
}
