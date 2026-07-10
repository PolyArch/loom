
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
constexpr std::array<uint32_t, kEdges> kExpected = {
    100, 2, 3, 4, 5, 200, 7, 8, 9, 10, 11, 300, 13, 14, 400, 16};

uint64_t checksum(const std::array<uint32_t, kEdges> &weights) {
    uint64_t sum = 0;
    for (uint32_t i = 0; i < kEdges; ++i) {
        sum += static_cast<uint64_t>(i + 1u) * weights[i];
    }
    return sum;
}

} // namespace

int main() {
    std::array<uint32_t, kEdges> input = {};
    std::array<uint32_t, kEdges> output = {};
    for (uint32_t i = 0; i < kEdges; ++i) {
        input[i] = i + 1u;
    }

    for (uint32_t i = 0; i < kEdges; ++i) {
        output[i] = input[i];
    }
    for (uint32_t u = 0; u < kUpdates; ++u) {
        const uint32_t src = kSrc[u];
        if (src >= kNodes) {
            continue;
        }
        for (uint32_t i = kRowPtr[src]; i < kRowPtr[src + 1u]; ++i) {
            if (kCols[i] == kDst[u]) {
                output[i] = kNewWeights[u];
                break;
            }
        }
    }

    for (uint32_t i = 0; i < kEdges; ++i) {
        if (output[i] != kExpected[i]) {
            std::puts("FAILED");
            return 1;
        }
    }

    std::printf("edge_update_batch checksum: %llu\n",
                static_cast<unsigned long long>(checksum(output)));
    std::puts("PASSED");
    return 0;
}
