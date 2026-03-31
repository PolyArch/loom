#include <cstdio>

void spmm(const int *rowPtr, const int *colIdx, const int *values,
          const int *dense, int *output, int rows, int cols) {
  for (int row = 0; row < rows; ++row) {
    for (int col = 0; col < cols; ++col)
      output[row * cols + col] = 0;
    for (int edge = rowPtr[row]; edge < rowPtr[row + 1]; ++edge) {
      int srcCol = colIdx[edge];
      int weight = values[edge];
      for (int col = 0; col < cols; ++col)
        output[row * cols + col] += weight * dense[srcCol * cols + col];
    }
  }
}

int main() {
  constexpr int rows = 3;
  constexpr int cols = 2;
  int rowPtr[rows + 1] = {0, 2, 3, 5};
  int colIdx[5] = {0, 2, 1, 0, 2};
  int values[5] = {1, 2, 3, 4, 1};
  int dense[3 * cols] = {
      1, 2,
      0, 3,
      2, 1,
  };
  int output[rows * cols] = {};
  int golden[rows * cols] = {
      5, 4,
      0, 9,
      6, 9,
  };

  spmm(rowPtr, colIdx, values, dense, output, rows, cols);

  for (int idx = 0; idx < rows * cols; ++idx) {
    if (output[idx] != golden[idx]) {
      std::printf("FAIL spmm %d %d %d\n", idx, output[idx], golden[idx]);
      return 1;
    }
  }
  std::puts("PASS");
  return 0;
}
