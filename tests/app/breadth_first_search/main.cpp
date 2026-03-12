#include <cstdio>

#include "breadth_first_search.h"

int main() {
    // Graph setup: 6 nodes, 11 edges
    // 0 -> 1, 3
    // 1 -> 2, 4
    // 2 -> 3, 5
    // 3 -> 4, 5
    // 4 -> 0, 2
    // 5 -> 1
    const uint32_t num_nodes = 6;
    const uint32_t num_edges = 11;
    const uint32_t queue_size = 6;
    const uint32_t source = 0;

    // CSR format: row_ptr (node offsets) and col_idx (edge destinations)
    uint32_t row_ptr[7] = {0, 2, 5, 7, 9, 11, 11};  // num_nodes + 1
    uint32_t col_idx[11] = {1, 3, 2, 4, 3, 5, 4, 5, 0, 2, 1};

    // Output arrays
    int32_t expect_distance[6];
    int32_t calculated_distance[6];
    uint32_t expect_queue[6];
    uint32_t calculated_queue[6];
    uint32_t expect_visited[6];
    uint32_t calculated_visited[6];

    // Compute expected result with CPU version
    bfs_cpu(row_ptr, col_idx, expect_distance, expect_queue, expect_visited, num_nodes, num_edges, queue_size, source);

    // Compute result with DSA version
    bfs_dsa(row_ptr, col_idx, calculated_distance, calculated_queue, calculated_visited, num_nodes, num_edges, queue_size, source);

    // Compare results (distances)
    for (uint32_t i = 0; i < num_nodes; i++) {
        if (expect_distance[i] != calculated_distance[i]) {
            printf("breadth_first_search: FAILED (distance mismatch at node %u: expected %d, got %d)\n",
                   i, expect_distance[i], calculated_distance[i]);
            return 1;
        }
    }

    printf("breadth_first_search: PASSED\n");
    return 0;
}
