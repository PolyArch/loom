// Sparse matrix times sparse-vector function variant.

#include <array>
#include <cstdint>
#include <cstdio>

namespace {

constexpr uint32_t kRows = 4;
constexpr uint32_t kCols = 5;
constexpr uint32_t kMatrixNnz = 9;
constexpr uint32_t kVectorNnz = 3;
constexpr std::array<uint32_t, kMatrixNnz> kValues = {
    2u, 3u, 4u, 1u, 5u, 6u, 7u, 2u, 3u};
constexpr std::array<uint32_t, kMatrixNnz> kColIndices = {
    0u, 2u, 1u, 3u, 0u, 4u, 1u, 2u, 4u};
constexpr std::array<uint32_t, kRows + 1> kRowPtr = {
    0u, 2u, 4u, 6u, 9u};
constexpr std::array<uint32_t, kVectorNnz> kVectorValues = {
    3u, 2u, 5u};
constexpr std::array<uint32_t, kVectorNnz> kVectorIndices = {
    0u, 2u, 3u};
constexpr std::array<uint32_t, kRows> kExpected = {
    12u, 5u, 15u, 4u};

uint64_t checksum(const std::array<uint32_t, kRows> &values) {
    uint64_t sum = 0;
    for (uint32_t i = 0; i < kRows; ++i) {
        sum += static_cast<uint64_t>(i + 1u) * values[i];
    }
    return sum;
}

} // namespace

extern "C" __attribute__((noinline))
void spmspv_kernel(const uint32_t *values, const uint32_t *col_indices,
                   const uint32_t *row_ptr, const uint32_t *x_values,
                   const uint32_t *x_indices, uint32_t x_nnz,
                   uint32_t *y, uint32_t *x_dense,
                   uint32_t rows, uint32_t cols) {
    for (uint32_t i = 0; i < cols; ++i) {
        x_dense[i] = 0;
    }
    for (uint32_t i = 0; i < x_nnz; ++i) {
        x_dense[x_indices[i]] = x_values[i];
    }
    for (uint32_t row = 0; row < rows; ++row) {
        uint32_t sum = 0;
        const uint32_t begin = row_ptr[row];
        const uint32_t end = row_ptr[row + 1u];
        for (uint32_t idx = begin; idx < end; ++idx) {
            sum += values[idx] * x_dense[col_indices[idx]];
        }
        y[row] = sum;
    }
}

int main() {
    std::array<uint32_t, kRows> candidate = {};
    std::array<uint32_t, kCols> dense = {};

    spmspv_kernel(kValues.data(), kColIndices.data(), kRowPtr.data(),
                  kVectorValues.data(), kVectorIndices.data(), kVectorNnz,
                  candidate.data(), dense.data(), kRows, kCols);

    for (uint32_t i = 0; i < kRows; ++i) {
        if (candidate[i] != kExpected[i]) {
            std::puts("FAILED");
            return 1;
        }
    }

    std::printf("spmspv checksum: %llu\n",
                static_cast<unsigned long long>(checksum(candidate)));
    std::puts("PASSED");
    return 0;
}
