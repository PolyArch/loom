/*
 * Entry function for auto_analyze: Graph Analytics pipeline.
 * Contains calls to all kernel functions with shared buffer arguments.
 * auto_analyze should detect 4 kernels and 3 edges.
 */

#include <stdlib.h>
#include <string.h>

#define NUM_VERTICES 1024
#define NUM_ITERS    20

typedef struct {
    int num_vertices;
    int num_edges;
    int *row_ptr;
    int *col_idx;
} csr_graph_t;

__attribute__((noinline))
int bfs_tiled(const csr_graph_t *g, int source, int *level) {
    int nv = g->num_vertices;
    int *frontier = (int *)malloc((size_t)nv * sizeof(int));
    int *next = (int *)malloc((size_t)nv * sizeof(int));
    int i;
    for (i = 0; i < nv; i++) level[i] = -1;
    level[source] = 0;
    frontier[0] = source;
    int f_size = 1, visited = 1, cur = 0;
    while (f_size > 0) {
        int nf = 0; cur++;
        for (i = 0; i < f_size; i++) {
            int u = frontier[i];
            int start = g->row_ptr[u], end = g->row_ptr[u+1];
            int j;
            for (j = start; j < end; j++) {
                int v = g->col_idx[j];
                if (level[v] == -1) {
                    level[v] = cur;
                    next[nf++] = v;
                    visited++;
                }
            }
        }
        int *tmp = frontier; frontier = next; next = tmp;
        f_size = nf;
    }
    free(frontier); free(next);
    return visited;
}

__attribute__((noinline))
void pagerank_spmv(const csr_graph_t *g, const float *rank_in,
                   float *rank_out, int nv, int iters) {
    int iter, i, j;
    float *src = (float *)malloc((size_t)nv * sizeof(float));
    float *dst = rank_out;
    memcpy(src, rank_in, (size_t)nv * sizeof(float));
    for (iter = 0; iter < iters; iter++) {
        for (i = 0; i < nv; i++) {
            float sum = 0.0f;
            int start = g->row_ptr[i], end = g->row_ptr[i+1];
            for (j = start; j < end; j++)
                sum += src[g->col_idx[j]];
            dst[i] = 0.15f / (float)nv + 0.85f * sum;
        }
        float *tmp = src; src = dst; dst = tmp;
    }
    if (dst != rank_out)
        memcpy(rank_out, src, (size_t)nv * sizeof(float));
    free(src);
}

__attribute__((noinline))
int triangle_count(const csr_graph_t *g, int nv) {
    int count = 0, u, i, j;
    for (u = 0; u < nv; u++) {
        int u_start = g->row_ptr[u], u_end = g->row_ptr[u+1];
        for (i = u_start; i < u_end; i++) {
            int v = g->col_idx[i];
            if (v <= u) continue;
            int v_start = g->row_ptr[v], v_end = g->row_ptr[v+1];
            int ui = u_start, vi = v_start;
            while (ui < u_end && vi < v_end) {
                if (g->col_idx[ui] == g->col_idx[vi]) {
                    count++; ui++; vi++;
                } else if (g->col_idx[ui] < g->col_idx[vi]) {
                    ui++;
                } else {
                    vi++;
                }
            }
        }
    }
    return count;
}

__attribute__((noinline))
void label_prop(const csr_graph_t *g, int *labels, int nv, int iters) {
    int iter, u, j;
    for (u = 0; u < nv; u++) labels[u] = u;
    for (iter = 0; iter < iters; iter++) {
        for (u = 0; u < nv; u++) {
            int start = g->row_ptr[u], end = g->row_ptr[u+1];
            int best_label = labels[u], best_count = 0;
            for (j = start; j < end; j++) {
                int nl = labels[g->col_idx[j]];
                int cnt = 0, k;
                for (k = start; k < end; k++)
                    if (labels[g->col_idx[k]] == nl) cnt++;
                if (cnt > best_count) { best_count = cnt; best_label = nl; }
            }
            labels[u] = best_label;
        }
    }
}

/* Entry function for auto_analyze */
void graph_pipeline(const csr_graph_t *g) {
    int nv = g->num_vertices;
    int *bfs_levels = (int *)malloc((size_t)nv * sizeof(int));
    float *rank_in = (float *)malloc((size_t)nv * sizeof(float));
    float *rank_out = (float *)malloc((size_t)nv * sizeof(float));
    int *labels = (int *)malloc((size_t)nv * sizeof(int));

    float init = 1.0f / (float)nv;
    int i;
    for (i = 0; i < nv; i++) rank_in[i] = init;

    bfs_tiled(g, 0, bfs_levels);
    pagerank_spmv(g, rank_in, rank_out, nv, NUM_ITERS);
    triangle_count(g, nv);
    label_prop(g, labels, nv, NUM_ITERS);

    free(bfs_levels); free(rank_in); free(rank_out); free(labels);
}
