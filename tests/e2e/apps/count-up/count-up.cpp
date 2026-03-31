#include <cstdio>

int count_up(int n) {
  int acc = 0;
  for (int i = 0; i < n; ++i)
    acc = acc + 1;
  return acc;
}

int main() {
  constexpr int n = 8;
  int value = count_up(n);
  if (value != n) {
    std::printf("FAIL count-up %d %d\n", value, n);
    return 1;
  }
  std::puts("PASS");
  return 0;
}
