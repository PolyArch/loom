// T5: Saturating arithmetic (manual clamp pattern).
// Tests: add + compare + select clamping sequence
#include <cstdint>

void saturating_add(const uint32_t *a, const uint32_t *b,
                    uint32_t *result, int n) {
  for (int i = 0; i < n; ++i) {
    // Manual unsigned saturation add pattern
    uint32_t sum = a[i] + b[i];
    // Detect overflow: if sum < a, it wrapped around
    result[i] = (sum < a[i]) ? 0xFFFFFFFF : sum;
  }
}

int main() {
  constexpr int n = 4;
  uint32_t a[n] = {100, 0xFFFFFF00, 200, 0x80000000};
  uint32_t b[n] = {200, 0x200, 300, 0x80000001};
  uint32_t result[n] = {};

  saturating_add(a, b, result, n);
  return 0;
}
