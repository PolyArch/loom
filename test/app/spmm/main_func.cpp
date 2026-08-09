// CSR sparse matrix times dense matrix function variant.

#include <stdint.h>
#include <stdio.h>

namespace {

constexpr uint32_t kRows = 2;
constexpr uint32_t kCols = 3;
constexpr uint32_t kOutCols = 2;
constexpr uint32_t kNnz = 4;
constexpr uint32_t kValues[kNnz] = {1u, 2u, 3u, 4u};
constexpr uint32_t kColIndices[kNnz] = {0u, 2u, 1u, 2u};
constexpr uint32_t kRowPtr[kRows + 1] = {0u, 2u, 4u};
constexpr uint32_t kDense[kCols * kOutCols] = {1u, 2u, 3u, 4u, 5u, 6u};
constexpr uint32_t kExpected[kRows * kOutCols] = {11u, 14u, 29u, 36u};

uint64_t checksum(const uint32_t *values) {
  uint64_t sum = 0;
  for (uint32_t i = 0; i < kRows * kOutCols; ++i) {
    sum += static_cast<uint64_t>(i + 1u) * values[i];
  }
  return sum;
}

} // namespace

extern "C" __attribute__((noinline)) void
spmm_kernel(const uint32_t *values, const uint32_t *col_indices,
            const uint32_t *row_ptr, const uint32_t *dense, uint32_t *out,
            uint32_t rows, uint32_t out_cols) {
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
        out[row * out_cols + col] += value * dense[dense_row * out_cols + col];
      }
    }
  }
}

int main() {
  uint32_t candidate[kRows * kOutCols] = {};

  spmm_kernel(kValues, kColIndices, kRowPtr, kDense, candidate, kRows,
              kOutCols);

  for (uint32_t i = 0; i < kRows * kOutCols; ++i) {
    if (candidate[i] != kExpected[i]) {
      printf("FAILED\n");
      return 1;
    }
  }

  printf("spmm checksum: %llu\n",
         static_cast<unsigned long long>(checksum(candidate)));
  printf("PASSED\n");
  return 0;
}
