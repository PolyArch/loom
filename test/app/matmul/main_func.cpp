// Dense matrix multiplication function variant.

#include <stdint.h>
#include <stdio.h>

namespace {

constexpr uint32_t kRows = 2;
constexpr uint32_t kInner = 3;
constexpr uint32_t kCols = 2;
constexpr uint32_t kInputA[kRows * kInner] = {1u, 2u, 3u, 4u, 5u, 6u};
constexpr uint32_t kInputB[kInner * kCols] = {7u, 8u, 9u, 10u, 11u, 12u};
constexpr uint32_t kExpected[kRows * kCols] = {58u, 64u, 139u, 154u};

uint64_t checksum(const uint32_t *values) {
  uint64_t sum = 0;
  for (uint32_t i = 0; i < kRows * kCols; ++i) {
    sum += static_cast<uint64_t>(i + 1u) * values[i];
  }
  return sum;
}

} // namespace

extern "C" __attribute__((noinline)) void
matmul_kernel(const uint32_t *a, const uint32_t *b, uint32_t *c, uint32_t rows,
              uint32_t cols) {
  const uint32_t output_count = rows * cols;
  for (uint32_t output = 0; output < output_count; ++output) {
    const uint32_t row = output / cols;
    const uint32_t col = output % cols;
    const uint32_t a_base = row * kInner;
    c[output] = a[a_base] * b[col] + a[a_base + 1u] * b[cols + col] +
                a[a_base + 2u] * b[2u * cols + col];
  }
}

int main() {
  uint32_t candidate[kRows * kCols] = {};

  matmul_kernel(kInputA, kInputB, candidate, kRows, kCols);

  for (uint32_t i = 0; i < kRows * kCols; ++i) {
    if (candidate[i] != kExpected[i]) {
      printf("FAILED\n");
      return 1;
    }
  }

  printf("matmul checksum: %llu\n",
         static_cast<unsigned long long>(checksum(candidate)));
  printf("PASSED\n");
  return 0;
}
