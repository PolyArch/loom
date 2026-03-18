#include <cstdio>

void gemm(const int *a, const int *b, int *c, int m, int n, int k) {
  for (int row = 0; row < m; ++row) {
    for (int col = 0; col < n; ++col) {
      int acc = 0;
      for (int inner = 0; inner < k; ++inner)
        acc += a[row * k + inner] * b[inner * n + col];
      c[row * n + col] = acc;
    }
  }
}

int main() {
  constexpr int m = 2;
  constexpr int n = 3;
  constexpr int k = 2;
  int a[m * k] = {1, 2, 3, 4};
  int b[k * n] = {
      2, 1, 0,
      1, 3, 2,
  };
  int c[m * n] = {};
  int golden[m * n] = {4, 7, 4, 10, 15, 8};

  gemm(a, b, c, m, n, k);

  for (int idx = 0; idx < m * n; ++idx) {
    if (c[idx] != golden[idx]) {
      std::printf("FAIL gemm %d %d %d\n", idx, c[idx], golden[idx]);
      return 1;
    }
  }
  std::puts("PASS");
  return 0;
}
