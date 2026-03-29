// T9: Indirect addressing (gather pattern).
// Tests: data[index[i]] generates chained loads
#include <cstdint>

int32_t indirect_sum(const int32_t *data, const int32_t *index, int n) {
  int32_t sum = 0;
  for (int i = 0; i < n; ++i) {
    sum += data[index[i]];
  }
  return sum;
}

int main() {
  constexpr int n = 4;
  int32_t data[8] = {10, 20, 30, 40, 50, 60, 70, 80};
  int32_t index[n] = {3, 1, 7, 0};

  int32_t result = indirect_sum(data, index, n);
  (void)result;
  return 0;
}
