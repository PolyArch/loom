#include <cstdio>

void vecadd(const int *a, const int *b, int *c, int n) {
  for (int i = 0; i < n; ++i)
    c[i] = a[i] + b[i];
}

int main() {
  constexpr int n = 8;
  int a[n] = {1, 2, 3, 4, 5, 6, 7, 8};
  int b[n] = {8, 7, 6, 5, 4, 3, 2, 1};
  int c[n] = {};
  int golden[n] = {9, 9, 9, 9, 9, 9, 9, 9};

  vecadd(a, b, c, n);

  for (int i = 0; i < n; ++i) {
    if (c[i] != golden[i]) {
      std::printf("FAIL vecadd %d %d %d\n", i, c[i], golden[i]);
      return 1;
    }
  }
  std::puts("PASS");
  return 0;
}
