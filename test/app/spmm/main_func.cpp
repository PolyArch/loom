// CSR sparse matrix times dense matrix function variant.

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

extern "C" __attribute__((noinline))
void spmm_kernel(const uint32_t *values, const uint32_t *col_indices,
                 const uint32_t *row_ptr, const uint32_t *dense,
                 uint32_t *out, uint32_t rows, uint32_t out_cols) {
    for (uint32_t i = 0; i < rows * out_cols; ++i) {
        out[i] = 0;
    }
    for (uint32_t row = 0; row < rows; ++row) {
        const uint32_t begin = row_ptr[row];
        const uint32_t end = row_ptr[row + 1u];
        for (uint32_t idx = begin; idx < end; ++idx) {
            const uint32_t value = values[idx];
            const uint32_t dense_row = col_indices[idx];
            for (uint32_t col = 0; col < out_cols; ++col) {
                out[row * out_cols + col] +=
                    value * dense[dense_row * out_cols + col];
            }
        }
    }
}

int main() {
    std::array<uint32_t, kRows * kOutCols> candidate = {};

    spmm_kernel(kValues.data(), kColIndices.data(), kRowPtr.data(),
                kDense.data(), candidate.data(), kRows, kOutCols);

    for (uint32_t i = 0; i < candidate.size(); ++i) {
        if (candidate[i] != kExpected[i]) {
            std::puts("FAILED");
            return 1;
        }
    }

    std::printf("spmm checksum: %llu\n",
                static_cast<unsigned long long>(checksum(candidate)));
    std::puts("PASSED");
    return 0;
}
