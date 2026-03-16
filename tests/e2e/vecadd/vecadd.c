#include <stdio.h>
#include <stdlib.h>

void vecadd(int *restrict a, int *restrict b, int *restrict c, int n) {
  for (int i = 0; i < n; i++)
    c[i] = a[i] + b[i];
}

int main() {
  const int N = 4;
  int a[4] = {1, 2, 3, 4};
  int b[4] = {5, 6, 7, 8};
  int c[4] = {0, 0, 0, 0};
  int golden[4] = {6, 8, 10, 12};

  vecadd(a, b, c, N);

  int pass = 1;
  for (int i = 0; i < N; i++) {
    if (c[i] != golden[i]) {
      printf("FAIL: c[%d] = %d, expected %d\n", i, c[i], golden[i]);
      pass = 0;
    }
  }

  if (pass)
    printf("PASS\n");

  return pass ? 0 : 1;
}
