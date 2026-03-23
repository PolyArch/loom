/*
 * PageRank via Sparse Matrix-Vector Multiply (SpMV).
 * Iteratively computes PageRank on a CSR graph.
 * pr_new[v] = (1-d)/N + d * sum_{u in in_neighbors(v)} pr[u] / out_deg(u)
 * Damping factor d = 0.85, max 20 iterations, convergence threshold 1e-6.
 * Tiled by vertex partition.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "tile_utils.h"
#include "csr_graph.h"

#define TEST_VERTICES  20
#define TEST_AVG_DEG   4
#define MAX_ITERS      20
#define DAMPING        0.85f
#define CONV_THRESH    1e-6f
#define TILE_V         8

/*
 * PageRank with tiled SpMV.
 * Uses the CSR graph as a column-oriented adjacency (out-edges).
 * For undirected graphs, in-neighbors == neighbors.
 */
int pagerank_tiled(const csr_graph_t *g, float *pr) {
    int nv = g->num_vertices;
    float *pr_new = (float *)malloc((size_t)nv * sizeof(float));
    if (!pr_new) return 0;

    float init_val = 1.0f / nv;
    int i;
    for (i = 0; i < nv; i++) pr[i] = init_val;

    int iter;
    for (iter = 0; iter < MAX_ITERS; iter++) {
        float base = (1.0f - DAMPING) / nv;

        /* Initialize new ranks */
        for (i = 0; i < nv; i++) pr_new[i] = base;

        /* SpMV: accumulate contributions from neighbors */
        TILE_FOR(tv, 0, nv, TILE_V) {
            int tv_end = TILE_END(tv, nv, TILE_V);
            int v;
            for (v = tv; v < tv_end; v++) {
                int deg = csr_degree(g, v);
                const int *nbrs = csr_neighbors(g, v);
                float sum = 0.0f;
                int j;
                for (j = 0; j < deg; j++) {
                    int u = nbrs[j];
                    int u_deg = csr_degree(g, u);
                    if (u_deg > 0) {
                        sum += pr[u] / (float)u_deg;
                    }
                }
                pr_new[v] += DAMPING * sum;
            }
        }

        /* Check convergence */
        float diff = 0.0f;
        for (i = 0; i < nv; i++) {
            float d = fabsf(pr_new[i] - pr[i]);
            if (d > diff) diff = d;
        }

        /* Copy new to old */
        memcpy(pr, pr_new, (size_t)nv * sizeof(float));

        if (diff < CONV_THRESH) {
            free(pr_new);
            return iter + 1;
        }
    }

    free(pr_new);
    return MAX_ITERS;
}

/* Reference (non-tiled) PageRank */
int pagerank_ref(const csr_graph_t *g, float *pr) {
    int nv = g->num_vertices;
    float *pr_new = (float *)malloc((size_t)nv * sizeof(float));
    if (!pr_new) return 0;

    float init_val = 1.0f / nv;
    int i;
    for (i = 0; i < nv; i++) pr[i] = init_val;

    int iter;
    for (iter = 0; iter < MAX_ITERS; iter++) {
        float base = (1.0f - DAMPING) / nv;
        for (i = 0; i < nv; i++) pr_new[i] = base;

        int v;
        for (v = 0; v < nv; v++) {
            int deg = csr_degree(g, v);
            const int *nbrs = csr_neighbors(g, v);
            float sum = 0.0f;
            int j;
            for (j = 0; j < deg; j++) {
                int u = nbrs[j];
                int u_deg = csr_degree(g, u);
                if (u_deg > 0) {
                    sum += pr[u] / (float)u_deg;
                }
            }
            pr_new[v] += DAMPING * sum;
        }

        float diff = 0.0f;
        for (i = 0; i < nv; i++) {
            float d = fabsf(pr_new[i] - pr[i]);
            if (d > diff) diff = d;
        }
        memcpy(pr, pr_new, (size_t)nv * sizeof(float));
        if (diff < CONV_THRESH) {
            free(pr_new);
            return iter + 1;
        }
    }

    free(pr_new);
    return MAX_ITERS;
}

int main(void) {
    csr_graph_t *g = csr_build_test_graph(TEST_VERTICES, TEST_AVG_DEG, 42);
    if (!g) {
        fprintf(stderr, "Graph construction failed\n");
        return 1;
    }

    int nv = g->num_vertices;
    float *pr_t = (float *)malloc((size_t)nv * sizeof(float));
    float *pr_r = (float *)malloc((size_t)nv * sizeof(float));
    if (!pr_t || !pr_r) {
        fprintf(stderr, "Allocation failed\n");
        csr_free(g);
        return 1;
    }

    int iters_t = pagerank_tiled(g, pr_t);
    int iters_r = pagerank_ref(g, pr_r);

    /* Verify PR values match */
    float max_err = 0.0f;
    float sum_pr = 0.0f;
    int i;
    for (i = 0; i < nv; i++) {
        float err = fabsf(pr_t[i] - pr_r[i]);
        if (err > max_err) max_err = err;
        sum_pr += pr_t[i];
    }

    printf("pagerank_spmv: %d vertices, %d edges\n",
           g->num_vertices, g->num_edges);
    printf("pagerank_spmv: iters_tiled=%d, iters_ref=%d, "
           "max_err=%e, sum_pr=%.4f\n",
           iters_t, iters_r, max_err, sum_pr);

    /* Print top-3 ranked vertices */
    if (nv <= 30) {
        printf("  PR values: ");
        for (i = 0; i < nv; i++) printf("%.4f ", pr_t[i]);
        printf("\n");
    }

    int pass = (iters_t == iters_r) && (max_err < 1e-6f);
    printf("pagerank_spmv: %s\n", pass ? "PASS" : "FAIL");

    free(pr_t); free(pr_r);
    csr_free(g);
    return pass ? 0 : 1;
}
