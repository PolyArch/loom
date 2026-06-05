// Dense matrix multiplication inline variant.

#include <array>
#include <cstdint>
#include <cstdio>

namespace {

constexpr uint32_t kRows = 2;
constexpr uint32_t kInner = 3;
constexpr uint32_t kCols = 2;
constexpr std::array<uint32_t, kRows * kInner> kInputA = {
    1u, 2u, 3u,
    4u, 5u, 6u};
constexpr std::array<uint32_t, kInner * kCols> kInputB = {
    7u, 8u,
    9u, 10u,
    11u, 12u};
constexpr std::array<uint32_t, kRows * kCols> kExpected = {
    58u, 64u, 139u, 154u};

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

    for (uint32_t i = 0; i < kRows; ++i) {
        for (uint32_t j = 0; j < kCols; ++j) {
            uint32_t sum = 0;
            for (uint32_t k = 0; k < kInner; ++k) {
                sum += kInputA[i * kInner + k] * kInputB[k * kCols + j];
            }
            output[i * kCols + j] = sum;
        }
    }

    for (uint32_t i = 0; i < output.size(); ++i) {
        if (output[i] != kExpected[i]) {
            std::puts("FAILED");
            return 1;
        }
    }

    std::printf("matmul checksum: %llu\n",
                static_cast<unsigned long long>(checksum(output)));
    std::puts("PASSED");
    return 0;
}
