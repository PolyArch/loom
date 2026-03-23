/*
 * BFS Traversal on CSR graph.
 * Frontier-based breadth-first search with vertex partitioning.
 * Uses two arrays (current/next frontier) for level-synchronous BFS.
 * Tiled by vertex partitions when scanning the frontier.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "tile_utils.h"
#include "csr_graph.h"

#define TEST_VERTICES  20
#define TEST_AVG_DEG   4
#define TILE_VERTICES  8

/*
 * BFS with tiled frontier processing.
 * Returns the number of vertices visited.
 * level[] is filled with BFS levels (-1 = unvisited).
 */
int bfs_tiled(const csr_graph_t *g, int source, int *level) {
    int nv = g->num_vertices;
    int *frontier = (int *)malloc((size_t)nv * sizeof(int));
    int *next_frontier = (int *)malloc((size_t)nv * sizeof(int));
    if (!frontier || !next_frontier) {
        free(frontier); free(next_frontier);
        return 0;
    }

    int i;
    for (i = 0; i < nv; i++) level[i] = -1;

    level[source] = 0;
    frontier[0] = source;
    int f_size = 1;
    int visited = 1;
    int cur_level = 0;

    while (f_size > 0) {
        int nf_size = 0;
        cur_level++;

        /* Process frontier in tiles */
        TILE_FOR(tf, 0, f_size, TILE_VERTICES) {
            int tf_end = TILE_END(tf, f_size, TILE_VERTICES);
            int fi;
            for (fi = tf; fi < tf_end; fi++) {
                int u = frontier[fi];
                int deg = csr_degree(g, u);
                const int *nbrs = csr_neighbors(g, u);
                int j;
                for (j = 0; j < deg; j++) {
                    int v = nbrs[j];
                    if (level[v] == -1) {
                        level[v] = cur_level;
                        next_frontier[nf_size++] = v;
                        visited++;
                    }
                }
            }
        }

        /* Swap frontiers */
        int *tmp = frontier;
        frontier = next_frontier;
        next_frontier = tmp;
        f_size = nf_size;
    }

    free(frontier);
    free(next_frontier);
    return visited;
}

/* Reference (non-tiled) BFS */
int bfs_ref(const csr_graph_t *g, int source, int *level) {
    int nv = g->num_vertices;
    int *frontier = (int *)malloc((size_t)nv * sizeof(int));
    int *next_frontier = (int *)malloc((size_t)nv * sizeof(int));
    if (!frontier || !next_frontier) {
        free(frontier); free(next_frontier);
        return 0;
    }

    int i;
    for (i = 0; i < nv; i++) level[i] = -1;

    level[source] = 0;
    frontier[0] = source;
    int f_size = 1;
    int visited = 1;
    int cur_level = 0;

    while (f_size > 0) {
        int nf_size = 0;
        cur_level++;
        int fi;
        for (fi = 0; fi < f_size; fi++) {
            int u = frontier[fi];
            int deg = csr_degree(g, u);
            const int *nbrs = csr_neighbors(g, u);
            int j;
            for (j = 0; j < deg; j++) {
                int v = nbrs[j];
                if (level[v] == -1) {
                    level[v] = cur_level;
                    next_frontier[nf_size++] = v;
                    visited++;
                }
            }
        }
        int *tmp = frontier;
        frontier = next_frontier;
        next_frontier = tmp;
        f_size = nf_size;
    }

    free(frontier);
    free(next_frontier);
    return visited;
}

int main(void) {
    csr_graph_t *g = csr_build_test_graph(TEST_VERTICES, TEST_AVG_DEG, 42);
    if (!g) {
        fprintf(stderr, "Graph construction failed\n");
        return 1;
    }

    int nv = g->num_vertices;
    int *level_t = (int *)malloc((size_t)nv * sizeof(int));
    int *level_r = (int *)malloc((size_t)nv * sizeof(int));
    if (!level_t || !level_r) {
        fprintf(stderr, "Allocation failed\n");
        csr_free(g);
        return 1;
    }

    int source = 0;
    int visited_t = bfs_tiled(g, source, level_t);
    int visited_r = bfs_ref(g, source, level_r);

    /* Verify levels match */
    int mismatches = 0;
    int i;
    for (i = 0; i < nv; i++) {
        if (level_t[i] != level_r[i]) {
            mismatches++;
        }
    }

    printf("bfs_traversal: %d vertices, %d edges, source=%d\n",
           g->num_vertices, g->num_edges, source);
    printf("bfs_traversal: visited_tiled=%d, visited_ref=%d, mismatches=%d\n",
           visited_t, visited_r, mismatches);

    /* Print BFS levels for small graph */
    if (nv <= 30) {
        printf("  BFS levels: ");
        for (i = 0; i < nv; i++) printf("%d ", level_t[i]);
        printf("\n");
    }

    int pass = (visited_t == visited_r) && (mismatches == 0);
    printf("bfs_traversal: %s\n", pass ? "PASS" : "FAIL");

    free(level_t); free(level_r);
    csr_free(g);
    return pass ? 0 : 1;
}
