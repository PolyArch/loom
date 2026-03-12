// Loom kernel implementation: breadth_first_search
#include "breadth_first_search.h"
#include "loom/loom.h"
#include <cmath>
#include <cstdlib>

// Full pipeline test from C++ source: Breadth-First Search on CSR graph
// Tests complete compilation chain with queue-based traversal and irregular memory access
// Graph (6 nodes, 11 edges):
//   0 -> 1, 3
//   1 -> 2, 4
//   2 -> 3, 5
//   3 -> 4, 5
//   4 -> 0, 2
//   5 -> 1
// BFS from node 0: distances = [0, 1, 2, 1, 2, 2]

// CPU implementation of BFS on CSR graph
// queue_size must be >= num_nodes to guarantee no overflow
// Note: programmer must guarantee arrays with __restrict__ do not overlap
void bfs_cpu(const uint32_t* __restrict__ row_ptr,
             const uint32_t* __restrict__ col_idx,
             int32_t* __restrict__ distance,
             uint32_t* __restrict__ queue,
             uint32_t* __restrict__ visited,
             const uint32_t num_nodes,
             const uint32_t num_edges,
             const uint32_t queue_size,
             const uint32_t source) {
    // Initialize distances to -1 (unvisited)
    for (uint32_t i = 0; i < num_nodes; i++) {
        distance[i] = -1;
        visited[i] = 0;
    }

    // Initialize BFS from source
    uint32_t queue_head = 0;
    uint32_t queue_tail = 0;

    distance[source] = 0;
    visited[source] = 1;
    queue[queue_tail++] = source;

    // BFS traversal (queue_tail bounded by num_nodes due to visited check)
    while (queue_head < queue_tail && queue_tail <= queue_size) {
        // dequeue current node
        uint32_t current = queue[queue_head++];
        int32_t current_dist = distance[current];

        // Iterate over neighbors
        uint32_t start = row_ptr[current];
        uint32_t end = row_ptr[current + 1];

        for (uint32_t edge = start; edge < end; edge++) {
            uint32_t neighbor = col_idx[edge];

            if (visited[neighbor] == 0 && queue_tail < queue_size) {
                visited[neighbor] = 1;
                distance[neighbor] = current_dist + 1;
                queue[queue_tail++] = neighbor;
            }
        }
    }
}

// Accelerator implementation of BFS on CSR graph
// queue_size must be >= num_nodes to guarantee no overflow
LOOM_ACCEL()
void bfs_dsa(const uint32_t* __restrict__ row_ptr,
             const uint32_t* __restrict__ col_idx,
             int32_t* __restrict__ distance,
             uint32_t* __restrict__ queue,
             uint32_t* __restrict__ visited,
             const uint32_t num_nodes,
             const uint32_t num_edges,
             const uint32_t queue_size,
             const uint32_t source) {
    // Initialize distances to -1 (unvisited)
    LOOM_NO_PARALLEL
    LOOM_NO_UNROLL
    for (uint32_t i = 0; i < num_nodes; i++) {
        distance[i] = -1;
        visited[i] = 0;
    }

    // Initialize BFS from source
    uint32_t queue_head = 0;
    uint32_t queue_tail = 0;

    distance[source] = 0;
    visited[source] = 1;
    queue[queue_tail++] = source;

    // BFS traversal (queue_tail bounded by num_nodes due to visited check)
    while (queue_head < queue_tail && queue_tail <= queue_size) {
        // dequeue current node
        uint32_t current = queue[queue_head++];
        int32_t current_dist = distance[current];

        // Iterate over neighbors
        uint32_t start = row_ptr[current];
        uint32_t end = row_ptr[current + 1];

        for (uint32_t edge = start; edge < end; edge++) {
            uint32_t neighbor = col_idx[edge];

            if (visited[neighbor] == 0 && queue_tail < queue_size) {
                visited[neighbor] = 1;
                distance[neighbor] = current_dist + 1;
                queue[queue_tail++] = neighbor;
            }
        }
    }
}
