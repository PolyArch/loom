#include <cstdio>

int scalar_add(int a, int b) { return a + b; }

int main() {
  constexpr int lhs = 17;
  constexpr int rhs = 25;
  constexpr int golden = 42;

  int sum = scalar_add(lhs, rhs);
  if (sum != golden) {
    std::printf("FAIL scalar-add %d %d\n", sum, golden);
    return 1;
  }

  std::puts("PASS");
  return 0;
}
