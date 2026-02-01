// Loom kernel implementation: edge_update
#include "edge_update.h"
#include "loom/loom.h"
#include <cmath>
#include <cstdlib>


// Full pipeline test from C++ source: Single edge weight update in CSR format
// Tests complete compilation chain with graph data structure manipulation
// Test: CSR graph (3 nodes, 4 edges), update edge (0,1) to 99 → updated weights=[99,20,30,40]






// CPU implementation of single edge weight update in CSR format
// Updates one edge weight in a CSR graph representation
// CSR format:
//   - input_col_indices: column indices of edges (read-only)
//   - row_ptr: row_ptr[i] is the start of row i in col_indices array
//   - input_weights: original weights of edges (read-only)
//   - output_weights: updated weights (write-only, same as input except one edge)
//   - num_edges: total number of edges in the graph
void edge_update_cpu(const uint32_t* __restrict__ row_ptr,
                     const uint32_t* __restrict__ input_col_indices,
                     const uint32_t* __restrict__ input_weights,
                     uint32_t* __restrict__ output_weights,
                     const uint32_t src_node,
                     const uint32_t dst_node,
                     const uint32_t new_weight,
                     const uint32_t num_nodes,
                     const uint32_t num_edges) {
    // Copy all weights first
    for (uint32_t i = 0; i < num_edges; i++) {
        output_weights[i] = input_weights[i];
    }
    
    if (src_node >= num_nodes) {
        return;  // Invalid source node
    }
    
    uint32_t row_start = row_ptr[src_node];
    uint32_t row_end = row_ptr[src_node + 1];
    
    // Find the edge (src_node, dst_node) and update its weight
    for (uint32_t i = row_start; i < row_end; i++) {
        if (input_col_indices[i] == dst_node) {
            output_weights[i] = new_weight;
            return;
        }
    }
}

// Accelerator implementation of single edge weight update
LOOM_ACCEL()
void edge_update_dsa(LOOM_MEMORY_BANK(8) LOOM_STREAM const uint32_t* __restrict__ row_ptr,
                     LOOM_STREAM const uint32_t* __restrict__ input_col_indices,
                     const uint32_t* __restrict__ input_weights,
                     uint32_t* __restrict__ output_weights,
                     const uint32_t src_node,
                     const uint32_t dst_node,
                     const uint32_t new_weight,
                     const uint32_t num_nodes,
                     const uint32_t num_edges) {
    // Copy all weights first
    for (uint32_t i = 0; i < num_edges; i++) {
        output_weights[i] = input_weights[i];
    }
    
    if (src_node >= num_nodes) {
        return;
    }
    
    uint32_t row_start = row_ptr[src_node];
    uint32_t row_end = row_ptr[src_node + 1];
    
    // Find the edge (src_node, dst_node) and update its weight
    for (uint32_t i = row_start; i < row_end; i++) {
        if (input_col_indices[i] == dst_node) {
            output_weights[i] = new_weight;
            return;
        }
    }
}



