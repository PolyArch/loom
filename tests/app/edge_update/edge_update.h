// Loom kernel: edge_update
#ifndef EDGE_UPDATE_H
#define EDGE_UPDATE_H

#include <cstdint>
#include <cstddef>

void edge_update_cpu(const uint32_t* __restrict__ row_ptr, const uint32_t* __restrict__ input_col_indices, const uint32_t* __restrict__ input_weights, uint32_t* __restrict__ output_weights, const uint32_t src_node, const uint32_t dst_node, const uint32_t new_weight, const uint32_t num_nodes, const uint32_t num_edges);

void edge_update_dsa(const uint32_t* __restrict__ row_ptr, const uint32_t* __restrict__ input_col_indices, const uint32_t* __restrict__ input_weights, uint32_t* __restrict__ output_weights, const uint32_t src_node, const uint32_t dst_node, const uint32_t new_weight, const uint32_t num_nodes, const uint32_t num_edges);

#endif // EDGE_UPDATE_H
