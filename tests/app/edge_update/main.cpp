#include <cstdio>

#include "edge_update.h"

int main() {
    const uint32_t num_nodes = 8;
    const uint32_t num_edges = 16;

    // CSR representation of a graph
    uint32_t row_ptr[num_nodes + 1] = {0, 2, 4, 7, 10, 12, 14, 15, 16};

    // Column indices (adjacency list)
    uint32_t col_indices[num_edges] = {1, 2, 0, 3, 0, 4, 5, 1, 2, 6, 3, 7, 4, 6, 7, 5};

    // Input weights (shared by both CPU and DSA)
    uint32_t input_weights[num_edges];
    for (uint32_t i = 0; i < num_edges; i++) {
        input_weights[i] = i + 1;
    }

    // Output edge weights (separate for CPU and DSA)
    uint32_t expect_weights[num_edges];
    uint32_t calculated_weights[num_edges];

    // Single edge update
    uint32_t src = 2;
    uint32_t dst = 4;
    uint32_t new_weight = 100;

    // Perform single edge update
    edge_update_cpu(row_ptr, col_indices, input_weights, expect_weights, 
                   src, dst, new_weight, num_nodes, num_edges);
    edge_update_dsa(row_ptr, col_indices, input_weights, calculated_weights, 
                   src, dst, new_weight, num_nodes, num_edges);

    // Compare results
    for (uint32_t i = 0; i < num_edges; i++) {
        if (expect_weights[i] != calculated_weights[i]) {
            printf("edge_update: FAILED\n");
            return 1;
        }
    }

    printf("edge_update: PASSED\n");
    return 0;
}

