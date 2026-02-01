// Loom kernel implementation: edge_update_batch
#include "edge_update_batch.h"
#include "loom/loom.h"
#include <cmath>
#include <cstdlib>

// Full pipeline test from C++ source: Batch edge weight update in CSR format
// Tests complete compilation chain with graph data structure manipulation
// Test: CSR graph (3 nodes, 4 edges), batch update 2 edges → updated weights=[99,20,88,40]

// CPU implementation of batch edge weight update
void edge_update_batch_cpu(const uint32_t* __restrict__ row_ptr,
                           const uint32_t* __restrict__ input_col_indices,
                           const uint32_t* __restrict__ input_weights,
                           uint32_t* __restrict__ output_weights,
                           const uint32_t* __restrict__ src_nodes,
                           const uint32_t* __restrict__ dst_nodes,
                           const uint32_t* __restrict__ new_weights,
                           const uint32_t num_updates,
                           const uint32_t num_nodes,
                           const uint32_t num_edges) {
    // Copy all weights first
    for (uint32_t i = 0; i < num_edges; i++) {
        output_weights[i] = input_weights[i];
    }

    // Apply batch updates
    for (uint32_t u = 0; u < num_updates; u++) {
        uint32_t src_node = src_nodes[u];
        uint32_t dst_node = dst_nodes[u];
        uint32_t new_weight = new_weights[u];

        if (src_node >= num_nodes) continue;

        uint32_t row_start = row_ptr[src_node];
        uint32_t row_end = row_ptr[src_node + 1];

        for (uint32_t i = row_start; i < row_end; i++) {
            if (input_col_indices[i] == dst_node) {
                output_weights[i] = new_weight;
                break;
            }
        }
    }
}

// Accelerator implementation of batch update
LOOM_ACCEL()
void edge_update_batch_dsa(LOOM_MEMORY_BANK(8) LOOM_STREAM const uint32_t* __restrict__ row_ptr,
                           LOOM_STREAM const uint32_t* __restrict__ input_col_indices,
                           const uint32_t* __restrict__ input_weights,
                           uint32_t* __restrict__ output_weights,
                           const uint32_t* __restrict__ src_nodes,
                           const uint32_t* __restrict__ dst_nodes,
                           const uint32_t* __restrict__ new_weights,
                           const uint32_t num_updates,
                           const uint32_t num_nodes,
                           const uint32_t num_edges) {
    // Copy all weights first
    for (uint32_t i = 0; i < num_edges; i++) {
        output_weights[i] = input_weights[i];
    }

    // Apply batch updates
    for (uint32_t u = 0; u < num_updates; u++) {
        uint32_t src_node = src_nodes[u];
        uint32_t dst_node = dst_nodes[u];
        uint32_t new_weight = new_weights[u];

        if (src_node >= num_nodes) continue;

        uint32_t row_start = row_ptr[src_node];
        uint32_t row_end = row_ptr[src_node + 1];

        for (uint32_t i = row_start; i < row_end; i++) {
            if (input_col_indices[i] == dst_node) {
                output_weights[i] = new_weight;
                break;
            }
        }
    }
}

