/*
 * Label Propagation for community detection on CSR graph.
 * Each vertex adopts the most frequent label among its neighbors.
 * Uses histogram-based majority vote. Iterates up to 10 rounds
 * or until convergence (no label changes).
 * Tiled by vertex partition.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "tile_utils.h"
#include "csr_graph.h"

#define TEST_VERTICES  20
#define TEST_AVG_DEG   4
#define MAX_ITERS      10
#define TILE_V         8

/*
 * Find the most frequent label among a vertex's neighbors.
 * Uses a simple histogram approach (labels are vertex indices).
 */
static int majority_label(const csr_graph_t *g, int v, const int *labels,
                          int nv) {
    int deg = csr_degree(g, v);
    if (deg == 0) return labels[v];

    const int *nbrs = csr_neighbors(g, v);

    /* Use a small histogram; labels are in [0, nv) */
    int *hist = (int *)calloc((size_t)nv, sizeof(int));
    if (!hist) return labels[v];

    int j;
    for (j = 0; j < deg; j++) {
        hist[labels[nbrs[j]]]++;
    }

    /* Find maximum */
    int best_label = labels[v];
    int best_count = 0;
    for (j = 0; j < nv; j++) {
        if (hist[j] > best_count) {
            best_count = hist[j];
            best_label = j;
        } else if (hist[j] == best_count && j < best_label) {
            /* Tie-breaking: prefer smaller label */
            best_label = j;
        }
    }

    free(hist);
    return best_label;
}

/* Tiled label propagation */
int label_prop_tiled(const csr_graph_t *g, int *labels) {
    int nv = g->num_vertices;
    int *new_labels = (int *)malloc((size_t)nv * sizeof(int));
    if (!new_labels) return 0;

    /* Initialize: each vertex is its own community */
    int i;
    for (i = 0; i < nv; i++) labels[i] = i;

    int iter;
    for (iter = 0; iter < MAX_ITERS; iter++) {
        int changes = 0;

        TILE_FOR(tv, 0, nv, TILE_V) {
            int tv_end = TILE_END(tv, nv, TILE_V);
            int v;
            for (v = tv; v < tv_end; v++) {
                new_labels[v] = majority_label(g, v, labels, nv);
                if (new_labels[v] != labels[v]) changes++;
            }
        }

        memcpy(labels, new_labels, (size_t)nv * sizeof(int));

        if (changes == 0) {
            free(new_labels);
            return iter + 1;
        }
    }

    free(new_labels);
    return MAX_ITERS;
}

/* Reference (non-tiled) label propagation */
int label_prop_ref(const csr_graph_t *g, int *labels) {
    int nv = g->num_vertices;
    int *new_labels = (int *)malloc((size_t)nv * sizeof(int));
    if (!new_labels) return 0;

    int i;
    for (i = 0; i < nv; i++) labels[i] = i;

    int iter;
    for (iter = 0; iter < MAX_ITERS; iter++) {
        int changes = 0;
        int v;
        for (v = 0; v < nv; v++) {
            new_labels[v] = majority_label(g, v, labels, nv);
            if (new_labels[v] != labels[v]) changes++;
        }
        memcpy(labels, new_labels, (size_t)nv * sizeof(int));
        if (changes == 0) {
            free(new_labels);
            return iter + 1;
        }
    }

    free(new_labels);
    return MAX_ITERS;
}

int main(void) {
    csr_graph_t *g = csr_build_test_graph(TEST_VERTICES, TEST_AVG_DEG, 42);
    if (!g) {
        fprintf(stderr, "Graph construction failed\n");
        return 1;
    }

    int nv = g->num_vertices;
    int *labels_t = (int *)malloc((size_t)nv * sizeof(int));
    int *labels_r = (int *)malloc((size_t)nv * sizeof(int));
    if (!labels_t || !labels_r) {
        fprintf(stderr, "Allocation failed\n");
        csr_free(g);
        return 1;
    }

    int iters_t = label_prop_tiled(g, labels_t);
    int iters_r = label_prop_ref(g, labels_r);

    /* Count number of distinct communities */
    int *seen = (int *)calloc((size_t)nv, sizeof(int));
    int n_communities = 0;
    int i;
    if (seen) {
        for (i = 0; i < nv; i++) {
            if (!seen[labels_t[i]]) {
                seen[labels_t[i]] = 1;
                n_communities++;
            }
        }
        free(seen);
    }

    /* Verify labels match */
    int mismatches = 0;
    for (i = 0; i < nv; i++) {
        if (labels_t[i] != labels_r[i]) mismatches++;
    }

    printf("label_prop: %d vertices, %d edges\n",
           g->num_vertices, g->num_edges);
    printf("label_prop: iters_tiled=%d, iters_ref=%d, "
           "communities=%d, mismatches=%d\n",
           iters_t, iters_r, n_communities, mismatches);

    if (nv <= 30) {
        printf("  Labels: ");
        for (i = 0; i < nv; i++) printf("%d ", labels_t[i]);
        printf("\n");
    }

    int pass = (iters_t == iters_r) && (mismatches == 0);
    printf("label_prop: %s\n", pass ? "PASS" : "FAIL");

    free(labels_t); free(labels_r);
    csr_free(g);
    return pass ? 0 : 1;
}
