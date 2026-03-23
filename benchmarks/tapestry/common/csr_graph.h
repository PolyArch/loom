#ifndef CSR_GRAPH_H
#define CSR_GRAPH_H

/*
 * CSR (Compressed Sparse Row) graph data structure for graph analytics.
 * Provides creation, neighbor iteration, and utility functions.
 */

#include <stdlib.h>
#include <string.h>

typedef struct {
    int num_vertices;
    int num_edges;
    int *row_ptr;   /* size: num_vertices + 1 */
    int *col_idx;   /* size: num_edges */
    float *values;  /* size: num_edges (optional edge weights) */
} csr_graph_t;

/* Return degree of vertex v */
static inline int csr_degree(const csr_graph_t *g, int v) {
    return g->row_ptr[v + 1] - g->row_ptr[v];
}

/* Return pointer to adjacency list of vertex v */
static inline const int *csr_neighbors(const csr_graph_t *g, int v) {
    return &g->col_idx[g->row_ptr[v]];
}

/* Allocate a CSR graph with given sizes */
static inline csr_graph_t *csr_alloc(int nv, int ne) {
    csr_graph_t *g = (csr_graph_t *)malloc(sizeof(csr_graph_t));
    if (!g) return NULL;
    g->num_vertices = nv;
    g->num_edges = ne;
    g->row_ptr = (int *)calloc((size_t)(nv + 1), sizeof(int));
    g->col_idx = (int *)malloc((size_t)ne * sizeof(int));
    g->values = (float *)malloc((size_t)ne * sizeof(float));
    if (!g->row_ptr || !g->col_idx || !g->values) {
        free(g->row_ptr); free(g->col_idx); free(g->values); free(g);
        return NULL;
    }
    return g;
}

/* Free a CSR graph */
static inline void csr_free(csr_graph_t *g) {
    if (g) {
        free(g->row_ptr);
        free(g->col_idx);
        free(g->values);
        free(g);
    }
}

/*
 * Build a small deterministic graph for testing.
 * Creates a graph with nv vertices where edges are generated
 * pseudo-randomly with roughly avg_deg average degree.
 * Adjacency lists are sorted for algorithms that require it.
 */
static inline csr_graph_t *csr_build_test_graph(int nv, int avg_deg, int seed) {
    /* First pass: count edges */
    int ne = 0;
    unsigned int state = (unsigned int)seed;
    int u, v;
    for (u = 0; u < nv; u++) {
        for (v = u + 1; v < nv; v++) {
            state = (state * 1103515245 + 12345) & 0x7FFFFFFF;
            if ((int)(state % (unsigned)nv) < avg_deg) {
                ne += 2; /* undirected: both directions */
            }
        }
    }
    if (ne == 0) ne = 2; /* ensure at least one edge */

    csr_graph_t *g = csr_alloc(nv, ne);
    if (!g) return NULL;

    /* Second pass: fill degree counts */
    int *deg = (int *)calloc((size_t)nv, sizeof(int));
    if (!deg) { csr_free(g); return NULL; }

    state = (unsigned int)seed;
    for (u = 0; u < nv; u++) {
        for (v = u + 1; v < nv; v++) {
            state = (state * 1103515245 + 12345) & 0x7FFFFFFF;
            if ((int)(state % (unsigned)nv) < avg_deg) {
                deg[u]++;
                deg[v]++;
            }
        }
    }

    /* Build row_ptr from degrees */
    g->row_ptr[0] = 0;
    for (u = 0; u < nv; u++) {
        g->row_ptr[u + 1] = g->row_ptr[u] + deg[u];
    }

    /* Third pass: fill col_idx */
    int *pos = (int *)calloc((size_t)nv, sizeof(int));
    if (!pos) { free(deg); csr_free(g); return NULL; }

    state = (unsigned int)seed;
    for (u = 0; u < nv; u++) {
        for (v = u + 1; v < nv; v++) {
            state = (state * 1103515245 + 12345) & 0x7FFFFFFF;
            if ((int)(state % (unsigned)nv) < avg_deg) {
                g->col_idx[g->row_ptr[u] + pos[u]] = v;
                g->col_idx[g->row_ptr[v] + pos[v]] = u;
                g->values[g->row_ptr[u] + pos[u]] = 1.0f;
                g->values[g->row_ptr[v] + pos[v]] = 1.0f;
                pos[u]++;
                pos[v]++;
            }
        }
    }

    /* Sort each adjacency list (insertion sort, small lists) */
    for (u = 0; u < nv; u++) {
        int start = g->row_ptr[u];
        int end = g->row_ptr[u + 1];
        int i, j;
        for (i = start + 1; i < end; i++) {
            int key = g->col_idx[i];
            float kv = g->values[i];
            j = i - 1;
            while (j >= start && g->col_idx[j] > key) {
                g->col_idx[j + 1] = g->col_idx[j];
                g->values[j + 1] = g->values[j];
                j--;
            }
            g->col_idx[j + 1] = key;
            g->values[j + 1] = kv;
        }
    }

    free(deg);
    free(pos);
    return g;
}

#endif /* CSR_GRAPH_H */
