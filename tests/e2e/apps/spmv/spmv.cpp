#include <cstdio>

void spmv(const int *rowPtr, const int *colIdx, const int *values,
          const int *vector, int *output, int rows) {
  for (int row = 0; row < rows; ++row) {
    int acc = 0;
    for (int edge = rowPtr[row]; edge < rowPtr[row + 1]; ++edge)
      acc += values[edge] * vector[colIdx[edge]];
    output[row] = acc;
  }
}

int main() {
  constexpr int rows = 4;
  int rowPtr[rows + 1] = {0, 2, 4, 5, 7};
  int colIdx[7] = {0, 2, 1, 3, 2, 0, 3};
  int values[7] = {3, 1, 4, 2, 5, 2, 1};
  int vector[4] = {2, 1, 3, 4};
  int output[rows] = {};
  int golden[rows] = {9, 12, 15, 8};

  spmv(rowPtr, colIdx, values, vector, output, rows);

  for (int row = 0; row < rows; ++row) {
    if (output[row] != golden[row]) {
      std::printf("FAIL spmv %d %d %d\n", row, output[row], golden[row]);
      return 1;
    }
  }
  std::puts("PASS");
  return 0;
}
