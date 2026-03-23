/*
 * Triangle Counting on CSR graph.
 * For each edge (u,v) where u < v, count common neighbors
 * using merge-based set intersection on sorted adjacency lists.
 * Total triangles = sum of intersections / 1 (since we only count u < v < w).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "tile_utils.h"
#include "csr_graph.h"

#define TEST_VERTICES  20
#define TEST_AVG_DEG   6
#define TILE_V         8

/* Tiled triangle counting */
long long triangle_count_tiled(const csr_graph_t *g) {
    int nv = g->num_vertices;
    long long total = 0;

    TILE_FOR(tv, 0, nv, TILE_V) {
        int tv_end = TILE_END(tv, nv, TILE_V);
        int u;
        for (u = tv; u < tv_end; u++) {
            int u_start = g->row_ptr[u];
            int u_end = g->row_ptr[u + 1];
            int j;
            for (j = u_start; j < u_end; j++) {
                int v = g->col_idx[j];
                if (v <= u) continue; /* only count u < v */

                /* Intersect adj(u) and adj(v) for vertices > v */
                int v_start = g->row_ptr[v];
                int v_end = g->row_ptr[v + 1];

                /* Find start position in u's neighbors where w > v */
                int u_scan = j + 1; /* already past v in u's sorted list */
                /* Find start position in v's neighbors where w > v */
                int v_scan = v_start;
                while (v_scan < v_end && g->col_idx[v_scan] <= v) {
                    v_scan++;
                }

                /* Count intersection of remaining parts */
                int iu = u_scan, iv = v_scan;
                while (iu < u_end && iv < v_end) {
                    int wu = g->col_idx[iu];
                    int wv = g->col_idx[iv];
                    if (wu == wv) {
                        total++;
                        iu++;
                        iv++;
                    } else if (wu < wv) {
                        iu++;
                    } else {
                        iv++;
                    }
                }
            }
        }
    }

    return total;
}

/* Reference (non-tiled) triangle counting */
long long triangle_count_ref(const csr_graph_t *g) {
    int nv = g->num_vertices;
    long long total = 0;
    int u, j;

    for (u = 0; u < nv; u++) {
        int u_start = g->row_ptr[u];
        int u_end = g->row_ptr[u + 1];
        for (j = u_start; j < u_end; j++) {
            int v = g->col_idx[j];
            if (v <= u) continue;

            int v_start = g->row_ptr[v];
            int v_end = g->row_ptr[v + 1];

            int u_scan = j + 1;
            int v_scan = v_start;
            while (v_scan < v_end && g->col_idx[v_scan] <= v) {
                v_scan++;
            }

            int iu = u_scan, iv = v_scan;
            while (iu < u_end && iv < v_end) {
                int wu = g->col_idx[iu];
                int wv = g->col_idx[iv];
                if (wu == wv) {
                    total++;
                    iu++;
                    iv++;
                } else if (wu < wv) {
                    iu++;
                } else {
                    iv++;
                }
            }
        }
    }

    return total;
}

/*
 * Brute-force triangle counting for verification.
 * Check all triples (u, v, w) with u < v < w.
 */
long long triangle_count_brute(const csr_graph_t *g) {
    int nv = g->num_vertices;

    /* Build adjacency matrix for small graph */
    int *adj_mat = (int *)calloc((size_t)nv * nv, sizeof(int));
    if (!adj_mat) return -1;

    int u;
    for (u = 0; u < nv; u++) {
        int start = g->row_ptr[u];
        int end = g->row_ptr[u + 1];
        int j;
        for (j = start; j < end; j++) {
            adj_mat[u * nv + g->col_idx[j]] = 1;
        }
    }

    long long count = 0;
    int v, w;
    for (u = 0; u < nv; u++) {
        for (v = u + 1; v < nv; v++) {
            if (!adj_mat[u * nv + v]) continue;
            for (w = v + 1; w < nv; w++) {
                if (adj_mat[u * nv + w] && adj_mat[v * nv + w]) {
                    count++;
                }
            }
        }
    }

    free(adj_mat);
    return count;
}

int main(void) {
    csr_graph_t *g = csr_build_test_graph(TEST_VERTICES, TEST_AVG_DEG, 42);
    if (!g) {
        fprintf(stderr, "Graph construction failed\n");
        return 1;
    }

    long long tc_tiled = triangle_count_tiled(g);
    long long tc_ref = triangle_count_ref(g);
    long long tc_brute = triangle_count_brute(g);

    printf("triangle_count: %d vertices, %d edges\n",
           g->num_vertices, g->num_edges);
    printf("triangle_count: tiled=%lld, ref=%lld, brute_force=%lld\n",
           tc_tiled, tc_ref, tc_brute);

    int pass = (tc_tiled == tc_ref) && (tc_tiled == tc_brute);
    printf("triangle_count: %s\n", pass ? "PASS" : "FAIL");

    csr_free(g);
    return pass ? 0 : 1;
}
