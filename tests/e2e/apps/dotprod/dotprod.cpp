#include <cstdio>

int dotprod(const int *a, const int *b, int n) {
  int acc = 0;
  for (int i = 0; i < n; ++i)
    acc += a[i] * b[i];
  return acc;
}

int main() {
  constexpr int n = 6;
  int a[n] = {1, 2, 3, 4, 5, 6};
  int b[n] = {6, 5, 4, 3, 2, 1};
  int golden = 56;
  int got = dotprod(a, b, n);
  if (got != golden) {
    std::printf("FAIL dotprod %d %d\n", got, golden);
    return 1;
  }
  std::puts("PASS");
  return 0;
}
