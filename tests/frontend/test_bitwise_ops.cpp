// T3: Bitwise shift and logic operations.
// Tests: shli, shrsi/shrui, xori, ori, andi
#include <cstdint>

void bitwise_butterfly(const uint32_t *x, const uint32_t *y,
                       const uint32_t *mask, uint32_t *z, int n) {
  for (int i = 0; i < n; ++i) {
    // ZK-style butterfly: z = (x << 16) ^ (y >> 3) | (x & mask)
    z[i] = (x[i] << 16) ^ (y[i] >> 3) | (x[i] & mask[i]);
  }
}

int main() {
  constexpr int n = 4;
  uint32_t x[n] = {0x1234, 0x5678, 0x9ABC, 0xDEF0};
  uint32_t y[n] = {0xAAAA, 0xBBBB, 0xCCCC, 0xDDDD};
  uint32_t mask[n] = {0xFF, 0xFF00, 0xFF0000, 0xFF000000};
  uint32_t z[n] = {};

  bitwise_butterfly(x, y, mask, z, n);
  return 0;
}
