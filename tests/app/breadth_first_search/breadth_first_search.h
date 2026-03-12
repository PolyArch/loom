// Loom kernel: breadth_first_search
#ifndef BREADTH_FIRST_SEARCH_H
#define BREADTH_FIRST_SEARCH_H

#include <cstdint>
#include <cstddef>

void bfs_cpu(const uint32_t* __restrict__ row_ptr, const uint32_t* __restrict__ col_idx, int32_t* __restrict__ distance, uint32_t* __restrict__ queue, uint32_t* __restrict__ visited, const uint32_t num_nodes, const uint32_t num_edges, const uint32_t queue_size, const uint32_t source);

void bfs_dsa(const uint32_t* __restrict__ row_ptr, const uint32_t* __restrict__ col_idx, int32_t* __restrict__ distance, uint32_t* __restrict__ queue, uint32_t* __restrict__ visited, const uint32_t num_nodes, const uint32_t num_edges, const uint32_t queue_size, const uint32_t source);

#endif // BREADTH_FIRST_SEARCH_H
