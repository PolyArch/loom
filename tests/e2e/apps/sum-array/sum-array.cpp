#include <cstdio>

int sum_array(const int *a, int n) {
  int acc = 0;
  for (int i = 0; i < n; ++i)
    acc += a[i];
  return acc;
}

int main(int argc, char **) {
  constexpr int n = 8;
  int a[n];
  int golden = 0;
  for (int i = 0; i < n; ++i) {
    a[i] = argc + i;
    golden += a[i];
  }

  int value = sum_array(a, n);
  if (value != golden) {
    std::printf("FAIL sum-array %d %d\n", value, golden);
    return 1;
  }
  std::puts("PASS");
  return 0;
}
