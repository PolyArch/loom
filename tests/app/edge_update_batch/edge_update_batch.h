// Loom kernel: edge_update_batch
#ifndef EDGE_UPDATE_BATCH_H
#define EDGE_UPDATE_BATCH_H

#include <cstdint>
#include <cstddef>

void edge_update_batch_cpu(const uint32_t* __restrict__ row_ptr, const uint32_t* __restrict__ input_col_indices, const uint32_t* __restrict__ input_weights, uint32_t* __restrict__ output_weights, const uint32_t* __restrict__ src_nodes, const uint32_t* __restrict__ dst_nodes, const uint32_t* __restrict__ new_weights, const uint32_t num_updates, const uint32_t num_nodes, const uint32_t num_edges);

void edge_update_batch_dsa(const uint32_t* __restrict__ row_ptr, const uint32_t* __restrict__ input_col_indices, const uint32_t* __restrict__ input_weights, uint32_t* __restrict__ output_weights, const uint32_t* __restrict__ src_nodes, const uint32_t* __restrict__ dst_nodes, const uint32_t* __restrict__ new_weights, const uint32_t num_updates, const uint32_t num_nodes, const uint32_t num_edges);

#endif // EDGE_UPDATE_BATCH_H
