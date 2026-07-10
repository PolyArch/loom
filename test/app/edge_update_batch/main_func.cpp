
#include <array>
#include <cstdint>
#include <cstdio>

namespace {

constexpr uint32_t kNodes = 8;
constexpr uint32_t kEdges = 16;
constexpr uint32_t kUpdates = 4;
constexpr std::array<uint32_t, kNodes + 1u> kRowPtr = {
    0, 2, 4, 7, 10, 12, 14, 15, 16};
constexpr std::array<uint32_t, kEdges> kCols = {
    1, 2, 0, 3, 0, 4, 5, 1, 2, 6, 3, 7, 4, 6, 7, 5};
constexpr std::array<uint32_t, kUpdates> kSrc = {0, 2, 4, 6};
constexpr std::array<uint32_t, kUpdates> kDst = {1, 4, 7, 7};
constexpr std::array<uint32_t, kUpdates> kNewWeights = {100, 200, 300, 400};

void initialize_weights(std::array<uint32_t, kEdges> &weights) {
    for (uint32_t i = 0; i < kEdges; ++i) {
        weights[i] = i + 1u;
    }
}

void edge_update_batch_ref(const uint32_t *row_ptr, const uint32_t *cols,
                           const uint32_t *input, uint32_t *output,
                           const uint32_t *src_nodes,
                           const uint32_t *dst_nodes,
                           const uint32_t *new_weights,
                           uint32_t updates, uint32_t nodes,
                           uint32_t edges) {
    for (uint32_t i = 0; i < edges; ++i) {
        output[i] = input[i];
    }
    for (uint32_t u = 0; u < updates; ++u) {
        const uint32_t src = src_nodes[u];
        if (src >= nodes) {
            continue;
        }
        for (uint32_t i = row_ptr[src]; i < row_ptr[src + 1u]; ++i) {
            if (cols[i] == dst_nodes[u]) {
                output[i] = new_weights[u];
                break;
            }
        }
    }
}

uint64_t checksum(const std::array<uint32_t, kEdges> &weights) {
    uint64_t sum = 0;
    for (uint32_t i = 0; i < kEdges; ++i) {
        sum += static_cast<uint64_t>(i + 1u) * weights[i];
    }
    return sum;
}

} // namespace

extern "C" __attribute__((noinline))
void edge_update_batch_kernel(const uint32_t *row_ptr, const uint32_t *cols,
                              const uint32_t *input, uint32_t *output,
                              const uint32_t *src_nodes,
                              const uint32_t *dst_nodes,
                              const uint32_t *new_weights,
                              uint32_t updates, uint32_t nodes,
                              uint32_t edges) {
    for (uint32_t i = 0; i < edges; ++i) {
        output[i] = input[i];
    }
    for (uint32_t u = 0; u < updates; ++u) {
        const uint32_t src = src_nodes[u];
        if (src >= nodes) {
            continue;
        }
        for (uint32_t i = row_ptr[src]; i < row_ptr[src + 1u]; ++i) {
            if (cols[i] == dst_nodes[u]) {
                output[i] = new_weights[u];
                break;
            }
        }
    }
}

int main() {
    std::array<uint32_t, kEdges> input = {};
    std::array<uint32_t, kEdges> reference = {};
    std::array<uint32_t, kEdges> candidate = {};

    initialize_weights(input);
    edge_update_batch_ref(kRowPtr.data(), kCols.data(), input.data(),
                          reference.data(), kSrc.data(), kDst.data(),
                          kNewWeights.data(), kUpdates, kNodes, kEdges);
    edge_update_batch_kernel(kRowPtr.data(), kCols.data(), input.data(),
                             candidate.data(), kSrc.data(), kDst.data(),
                             kNewWeights.data(), kUpdates, kNodes, kEdges);

    for (uint32_t i = 0; i < kEdges; ++i) {
        if (reference[i] != candidate[i]) {
            std::puts("FAILED");
            return 1;
        }
    }

    std::printf("edge_update_batch checksum: %llu\n",
                static_cast<unsigned long long>(checksum(candidate)));
    std::puts("PASSED");
    return 0;
}
