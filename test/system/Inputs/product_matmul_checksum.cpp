namespace {

constexpr unsigned kRows = 2;
constexpr unsigned kInner = 3;
constexpr unsigned kCols = 2;
constexpr unsigned kInputA[kRows * kInner] = {1, 2, 3, 4, 5, 6};
constexpr unsigned kInputB[kInner * kCols] = {7, 8, 9, 10, 11, 12};
constexpr unsigned kExpected[kRows * kCols] = {58, 64, 139, 154};
constexpr unsigned kExpectedChecksum = 1219;

unsigned checksum(const unsigned *values) {
  unsigned sum = 0;
  for (unsigned index = 0; index < kRows * kCols; ++index)
    sum += (index + 1) * values[index];
  return sum;
}

} // namespace

extern "C" __attribute__((noinline)) void
matmul_kernel(const unsigned *a, const unsigned *b, unsigned *c,
              unsigned rows, unsigned cols) {
  const unsigned outputCount = rows * cols;
  for (unsigned output = 0; output < outputCount; ++output) {
    const unsigned row = output / cols;
    const unsigned col = output % cols;
    const unsigned aBase = row * kInner;
    c[output] = a[aBase] * b[col] + a[aBase + 1] * b[cols + col] +
                a[aBase + 2] * b[2 * cols + col];
  }
}

int main() {
  unsigned candidate[kRows * kCols] = {};
  matmul_kernel(kInputA, kInputB, candidate, kRows, kCols);
  for (unsigned index = 0; index < kRows * kCols; ++index)
    if (candidate[index] != kExpected[index])
      return -1;
  const unsigned result = checksum(candidate);
  return result == kExpectedChecksum ? static_cast<int>(result) : -1;
}
