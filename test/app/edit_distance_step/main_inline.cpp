
#include <array>
#include <cstdint>
#include <cstdio>

namespace {

constexpr uint32_t kSize = 64;

uint32_t checksum(const std::array<uint32_t, kSize> &values) {
  uint32_t sum = 0;
  for (uint32_t i = 0; i < kSize; ++i)
    sum += (i + 1u) * values[i];
  return sum;
}

void initialize_inputs(std::array<uint32_t, kSize> &left,
                       std::array<uint32_t, kSize> &top,
                       std::array<uint32_t, kSize> &diag,
                       std::array<uint32_t, kSize> &char_a,
                       std::array<uint32_t, kSize> &char_b) {
  for (uint32_t i = 0; i < kSize; ++i) {
    left[i] = i + 1u;
    top[i] = i + 2u;
    diag[i] = i;
    char_a[i] = static_cast<uint32_t>('a') + (i % 2u);
    char_b[i] = static_cast<uint32_t>('a') + ((i + 1u) % 2u);
  }
}

} // namespace

int main() {
  std::array<uint32_t, kSize> left = {};
  std::array<uint32_t, kSize> top = {};
  std::array<uint32_t, kSize> diag = {};
  std::array<uint32_t, kSize> char_a = {};
  std::array<uint32_t, kSize> char_b = {};
  std::array<uint32_t, kSize> expected = {};
  std::array<uint32_t, kSize> candidate = {};

  initialize_inputs(left, top, diag, char_a, char_b);

  for (uint32_t i = 0; i < kSize; ++i) {
    const uint32_t cost = (char_a[i] == char_b[i]) ? 0u : 1u;
    const uint32_t insert_cost = top[i] + 1u;
    const uint32_t delete_cost = left[i] + 1u;
    const uint32_t subst_cost = diag[i] + cost;
    uint32_t min_val = insert_cost < delete_cost ? insert_cost : delete_cost;
    min_val = min_val < subst_cost ? min_val : subst_cost;
    expected[i] = min_val;
  }

  for (uint32_t i = 0; i < kSize; ++i) {
    const uint32_t cost = (char_a[i] == char_b[i]) ? 0u : 1u;
    const uint32_t insert_cost = top[i] + 1u;
    const uint32_t delete_cost = left[i] + 1u;
    const uint32_t subst_cost = diag[i] + cost;
    uint32_t min_val = insert_cost < delete_cost ? insert_cost : delete_cost;
    min_val = min_val < subst_cost ? min_val : subst_cost;
    candidate[i] = min_val;
  }

  for (uint32_t i = 0; i < kSize; ++i) {
    if (expected[i] != candidate[i]) {
      std::puts("FAILED");
      return 1;
    }
  }

  std::printf("edit_distance_step checksum: %u\n", checksum(candidate));
  std::puts("PASSED");
  return 0;
}
