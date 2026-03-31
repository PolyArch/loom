#include <cstdio>

static void mergeRuns(const int *input, int *output, int left, int mid, int right) {
  int lhs = left;
  int rhs = mid;
  int dst = left;
  while (lhs < mid && rhs < right) {
    if (input[lhs] <= input[rhs])
      output[dst++] = input[lhs++];
    else
      output[dst++] = input[rhs++];
  }
  while (lhs < mid)
    output[dst++] = input[lhs++];
  while (rhs < right)
    output[dst++] = input[rhs++];
}

void mergeSort(int *data, int n) {
  int temp[16] = {};
  for (int width = 1; width < n; width *= 2) {
    for (int left = 0; left < n; left += 2 * width) {
      int mid = left + width;
      int right = left + 2 * width;
      if (mid > n)
        mid = n;
      if (right > n)
        right = n;
      mergeRuns(data, temp, left, mid, right);
    }
    for (int i = 0; i < n; ++i)
      data[i] = temp[i];
  }
}

int main() {
  int data[8] = {9, 1, 7, 3, 5, 2, 8, 4};
  int golden[8] = {1, 2, 3, 4, 5, 7, 8, 9};

  mergeSort(data, 8);

  for (int idx = 0; idx < 8; ++idx) {
    if (data[idx] != golden[idx]) {
      std::printf("FAIL merge-sort %d %d %d\n", idx, data[idx], golden[idx]);
      return 1;
    }
  }
  std::puts("PASS");
  return 0;
}
