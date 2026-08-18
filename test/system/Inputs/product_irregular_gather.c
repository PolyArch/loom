__attribute__((noinline)) static void
irregular_gather(const int *source, const unsigned int *indices, int *output) {
  for (unsigned int lane = 0; lane < 4; ++lane)
    output[lane] = source[indices[lane]] + (int)lane;
}

int main(void) {
  static const int source[5] = {7, 11, 19, 23, 31};
  static const unsigned int indices[4] = {3, 0, 4, 1};
  static const int expected[4] = {23, 8, 33, 14};
  int output[4] = {0, 0, 0, 0};
  irregular_gather(source, indices, output);
  for (unsigned int lane = 0; lane < 4; ++lane)
    if (output[lane] != expected[lane])
      return 1;
  return 0;
}
