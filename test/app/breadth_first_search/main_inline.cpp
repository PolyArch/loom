// Queue-based CSR breadth-first search inline variant.

#include <array>
#include <cstdint>
#include <cstdio>

namespace {

constexpr uint32_t kNumNodes = 6;
constexpr uint32_t kNumEdges = 11;
constexpr uint32_t kQueueSize = 6;
constexpr uint32_t kSource = 0;
constexpr std::array<uint32_t, kNumNodes + 1> kRowPtr = {
    0u, 2u, 5u, 7u, 9u, 11u, 11u};
constexpr std::array<uint32_t, kNumEdges> kColIdx = {
    1u, 3u, 2u, 4u, 3u, 5u, 4u, 5u, 0u, 2u, 1u};
constexpr std::array<int32_t, kNumNodes> kExpectedDistance = {
    0, 1, 2, 1, 2, 2};

void breadth_first_search_inline(const uint32_t *row_ptr,
                                 const uint32_t *col_idx,
                                 int32_t *distance, uint32_t *queue,
                                 uint32_t *visited, uint32_t num_nodes,
                                 uint32_t queue_size, uint32_t source) {
    for (uint32_t i = 0; i < num_nodes; ++i) {
        distance[i] = -1;
        visited[i] = 0;
    }

    uint32_t queue_head = 0;
    uint32_t queue_tail = 0;
    distance[source] = 0;
    visited[source] = 1;
    queue[queue_tail++] = source;

    while (queue_head < queue_tail && queue_tail <= queue_size) {
        const uint32_t current = queue[queue_head++];
        const int32_t current_dist = distance[current];
        const uint32_t start = row_ptr[current];
        const uint32_t end = row_ptr[current + 1u];

        for (uint32_t edge = start; edge < end; ++edge) {
            const uint32_t neighbor = col_idx[edge];
            if (visited[neighbor] == 0 && queue_tail < queue_size) {
                visited[neighbor] = 1;
                distance[neighbor] = current_dist + 1;
                queue[queue_tail++] = neighbor;
            }
        }
    }
}

uint64_t checksum(const std::array<int32_t, kNumNodes> &distance) {
    uint64_t sum = 0;
    for (uint32_t i = 0; i < kNumNodes; ++i) {
        sum += static_cast<uint64_t>(i + 1u) *
               static_cast<uint64_t>(distance[i]);
    }
    return sum;
}

} // namespace

int main() {
    std::array<int32_t, kNumNodes> candidate_distance = {};
    std::array<uint32_t, kQueueSize> candidate_queue = {};
    std::array<uint32_t, kNumNodes> candidate_visited = {};

    breadth_first_search_inline(kRowPtr.data(), kColIdx.data(),
                                candidate_distance.data(), candidate_queue.data(),
                                candidate_visited.data(), kNumNodes, kQueueSize,
                                kSource);

    for (uint32_t i = 0; i < kNumNodes; ++i) {
        if (candidate_distance[i] != kExpectedDistance[i]) {
            std::puts("FAILED");
            return 1;
        }
    }

    std::printf("breadth_first_search checksum: %llu\n",
                static_cast<unsigned long long>(checksum(candidate_distance)));
    std::puts("PASSED");
    return 0;
}
