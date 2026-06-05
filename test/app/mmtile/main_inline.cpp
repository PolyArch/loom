// Tiled dense matrix multiplication inline variant.

#include <array>
#include <cstdint>
#include <cstdio>

namespace {

constexpr uint32_t kRows = 4;
constexpr uint32_t kInner = 4;
constexpr uint32_t kCols = 3;
constexpr uint32_t kTileRows = 2;
constexpr uint32_t kTileInner = 2;
constexpr uint32_t kTileCols = 2;
constexpr std::array<uint32_t, kRows * kInner> kInputA = {
    1u, 2u, 0u, 1u,
    3u, 1u, 2u, 0u,
    0u, 1u, 4u, 2u,
    2u, 0u, 1u, 3u};
constexpr std::array<uint32_t, kInner * kCols> kInputB = {
    1u, 0u, 2u,
    0u, 3u, 1u,
    4u, 1u, 0u,
    2u, 2u, 1u};
constexpr std::array<uint32_t, kRows * kCols> kExpected = {
    3u, 8u, 5u,
    11u, 5u, 7u,
    20u, 11u, 3u,
    12u, 7u, 7u};

uint32_t min_u32(uint32_t lhs, uint32_t rhs) {
    return lhs < rhs ? lhs : rhs;
}

uint64_t checksum(const std::array<uint32_t, kRows * kCols> &values) {
    uint64_t sum = 0;
    for (uint32_t i = 0; i < values.size(); ++i) {
        sum += static_cast<uint64_t>(i + 1u) * values[i];
    }
    return sum;
}

} // namespace

int main() {
    std::array<uint32_t, kRows * kCols> output = {};

    for (uint32_t i = 0; i < output.size(); ++i) {
        output[i] = 0;
    }
    for (uint32_t i0 = 0; i0 < kRows; i0 += kTileRows) {
        for (uint32_t j0 = 0; j0 < kCols; j0 += kTileCols) {
            for (uint32_t k0 = 0; k0 < kInner; k0 += kTileInner) {
                const uint32_t i_end = min_u32(i0 + kTileRows, kRows);
                const uint32_t j_end = min_u32(j0 + kTileCols, kCols);
                const uint32_t k_end = min_u32(k0 + kTileInner, kInner);
                for (uint32_t i = i0; i < i_end; ++i) {
                    for (uint32_t j = j0; j < j_end; ++j) {
                        uint32_t sum = 0;
                        for (uint32_t k = k0; k < k_end; ++k) {
                            sum += kInputA[i * kInner + k] * kInputB[k * kCols + j];
                        }
                        output[i * kCols + j] += sum;
                    }
                }
            }
        }
    }

    for (uint32_t i = 0; i < output.size(); ++i) {
        if (output[i] != kExpected[i]) {
            std::puts("FAILED");
            return 1;
        }
    }

    std::printf("mmtile checksum: %llu\n",
                static_cast<unsigned long long>(checksum(output)));
    std::puts("PASSED");
    return 0;
}
