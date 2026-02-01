#include <cstdio>

#include "edge_update_batch.h"

int main() {
    const uint32_t num_nodes = 8;
    const uint32_t num_edges = 16;
    const uint32_t num_updates = 4;
    
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
    
    // Updates to perform
    uint32_t src_nodes[num_updates] = {0, 2, 4, 6};
    uint32_t dst_nodes[num_updates] = {1, 4, 7, 7};
    uint32_t new_weights[num_updates] = {100, 200, 300, 400};
    
    // Perform batch updates with CPU version
    edge_update_batch_cpu(row_ptr, col_indices, input_weights, expect_weights,
                         src_nodes, dst_nodes, new_weights,
                         num_updates, num_nodes, num_edges);
    
    // Perform batch updates with accelerator version
    edge_update_batch_dsa(row_ptr, col_indices, input_weights, calculated_weights,
                         src_nodes, dst_nodes, new_weights,
                         num_updates, num_nodes, num_edges);
    
    // Compare results
    for (uint32_t i = 0; i < num_edges; i++) {
        if (expect_weights[i] != calculated_weights[i]) {
            printf("edge_update_batch: FAILED\n");
            return 1;
        }
    }
    
    printf("edge_update_batch: PASSED\n");
    return 0;
}

