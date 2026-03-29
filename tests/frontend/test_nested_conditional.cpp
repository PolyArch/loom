// T6: Nested conditional with multiple results.
// Tests: nested if/else chain with multiple live-out values
#include <cstdint>

void nested_cond(const int32_t *c1, const int32_t *c2,
                 const int32_t *a, const int32_t *b,
                 int32_t *r1, int32_t *r2, int n) {
  for (int i = 0; i < n; ++i) {
    int32_t v1, v2;
    if (c1[i]) {
      if (c2[i]) {
        v1 = a[i];
        v2 = b[i];
      } else {
        v1 = b[i];
        v2 = a[i];
      }
    } else {
      v1 = a[i] + b[i];
      v2 = a[i] - b[i];
    }
    r1[i] = v1;
    r2[i] = v2;
  }
}

int main() {
  constexpr int n = 4;
  int32_t c1[n] = {1, 0, 1, 0};
  int32_t c2[n] = {1, 1, 0, 0};
  int32_t a[n] = {10, 20, 30, 40};
  int32_t b[n] = {5, 15, 25, 35};
  int32_t r1[n] = {};
  int32_t r2[n] = {};

  nested_cond(c1, c2, a, b, r1, r2, n);
  return 0;
}
