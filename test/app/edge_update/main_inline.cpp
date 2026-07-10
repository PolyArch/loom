
#include <array>
#include <cstdint>
#include <cstdio>

namespace {

constexpr uint32_t kNodes = 8;
constexpr uint32_t kEdges = 16;
constexpr std::array<uint32_t, kNodes + 1u> kRowPtr = {
    0, 2, 4, 7, 10, 12, 14, 15, 16};
constexpr std::array<uint32_t, kEdges> kCols = {
    1, 2, 0, 3, 0, 4, 5, 1, 2, 6, 3, 7, 4, 6, 7, 5};
constexpr std::array<uint32_t, kEdges> kExpected = {
    1, 2, 3, 4, 5, 100, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};

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
    const uint32_t src = 2;
    const uint32_t dst = 4;
    const uint32_t new_weight = 100;
    if (src < kNodes) {
        for (uint32_t i = kRowPtr[src]; i < kRowPtr[src + 1u]; ++i) {
            if (kCols[i] == dst) {
                output[i] = new_weight;
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

    std::printf("edge_update checksum: %llu\n",
                static_cast<unsigned long long>(checksum(output)));
    std::puts("PASSED");
    return 0;
}
