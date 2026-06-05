// Tiled dense matrix multiplication function variant.

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

extern "C" __attribute__((noinline))
void mmtile_kernel(const uint32_t *a, const uint32_t *b, uint32_t *c,
                   uint32_t rows, uint32_t inner, uint32_t cols,
                   uint32_t tile_rows, uint32_t tile_inner,
                   uint32_t tile_cols) {
    for (uint32_t i = 0; i < rows * cols; ++i) {
        c[i] = 0;
    }
    for (uint32_t i0 = 0; i0 < rows; i0 += tile_rows) {
        for (uint32_t j0 = 0; j0 < cols; j0 += tile_cols) {
            for (uint32_t k0 = 0; k0 < inner; k0 += tile_inner) {
                const uint32_t i_end = min_u32(i0 + tile_rows, rows);
                const uint32_t j_end = min_u32(j0 + tile_cols, cols);
                const uint32_t k_end = min_u32(k0 + tile_inner, inner);
                for (uint32_t i = i0; i < i_end; ++i) {
                    for (uint32_t j = j0; j < j_end; ++j) {
                        uint32_t sum = 0;
                        for (uint32_t k = k0; k < k_end; ++k) {
                            sum += a[i * inner + k] * b[k * cols + j];
                        }
                        c[i * cols + j] += sum;
                    }
                }
            }
        }
    }
}

int main() {
    std::array<uint32_t, kRows * kCols> candidate = {};

    mmtile_kernel(kInputA.data(), kInputB.data(), candidate.data(),
                  kRows, kInner, kCols, kTileRows, kTileInner, kTileCols);

    for (uint32_t i = 0; i < candidate.size(); ++i) {
        if (candidate[i] != kExpected[i]) {
            std::puts("FAILED");
            return 1;
        }
    }

    std::printf("mmtile checksum: %llu\n",
                static_cast<unsigned long long>(checksum(candidate)));
    std::puts("PASSED");
    return 0;
}
