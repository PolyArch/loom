// CSR sparse matrix times dense matrix inline variant.

#include <array>
#include <cstdint>
#include <cstdio>

namespace {

constexpr uint32_t kRows = 2;
constexpr uint32_t kCols = 3;
constexpr uint32_t kOutCols = 2;
constexpr uint32_t kNnz = 4;
constexpr std::array<uint32_t, kNnz> kValues = {
    1u, 2u, 3u, 4u};
constexpr std::array<uint32_t, kNnz> kColIndices = {
    0u, 2u, 1u, 2u};
constexpr std::array<uint32_t, kRows + 1> kRowPtr = {
    0u, 2u, 4u};
constexpr std::array<uint32_t, kCols * kOutCols> kDense = {
    1u, 2u,
    3u, 4u,
    5u, 6u};
constexpr std::array<uint32_t, kRows * kOutCols> kExpected = {
    11u, 14u,
    29u, 36u};

uint64_t checksum(const std::array<uint32_t, kRows * kOutCols> &values) {
    uint64_t sum = 0;
    for (uint32_t i = 0; i < values.size(); ++i) {
        sum += static_cast<uint64_t>(i + 1u) * values[i];
    }
    return sum;
}

} // namespace

int main() {
    std::array<uint32_t, kRows * kOutCols> output = {};

    for (uint32_t i = 0; i < kRows * kOutCols; ++i) {
        output[i] = 0;
    }
    for (uint32_t row = 0; row < kRows; ++row) {
        const uint32_t begin = kRowPtr[row];
        const uint32_t end = kRowPtr[row + 1u];
        for (uint32_t idx = begin; idx < end; ++idx) {
            const uint32_t value = kValues[idx];
            const uint32_t dense_row = kColIndices[idx];
            for (uint32_t col = 0; col < kOutCols; ++col) {
                output[row * kOutCols + col] +=
                    value * kDense[dense_row * kOutCols + col];
            }
        }
    }

    for (uint32_t i = 0; i < output.size(); ++i) {
        if (output[i] != kExpected[i]) {
            std::puts("FAILED");
            return 1;
        }
    }

    std::printf("spmm checksum: %llu\n",
                static_cast<unsigned long long>(checksum(output)));
    std::puts("PASSED");
    return 0;
}
