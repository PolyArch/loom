#include <cstdio>

void gemv(const int *matrix, const int *vector, int *output,
          int rows, int cols) {
  for (int row = 0; row < rows; ++row) {
    int acc = 0;
    for (int col = 0; col < cols; ++col)
      acc += matrix[row * cols + col] * vector[col];
    output[row] = acc;
  }
}

int main() {
  constexpr int rows = 3;
  constexpr int cols = 4;
  int matrix[rows * cols] = {
      1, 2, 3, 4,
      2, 0, 1, 1,
      3, 1, 0, 2,
  };
  int vector[cols] = {1, 2, 1, 3};
  int output[rows] = {};
  int golden[rows] = {19, 6, 11};

  gemv(matrix, vector, output, rows, cols);

  for (int row = 0; row < rows; ++row) {
    if (output[row] != golden[row]) {
      std::printf("FAIL gemv %d %d %d\n", row, output[row], golden[row]);
      return 1;
    }
  }
  std::puts("PASS");
  return 0;
}
