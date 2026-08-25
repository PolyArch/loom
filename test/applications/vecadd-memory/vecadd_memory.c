enum { VECTOR_LENGTH = 64 };

__attribute__((noinline)) static void
vecadd_memory(const int *restrict lhs, const int *restrict rhs,
              int *restrict result, int length) {
  for (int index = 0; index < length; ++index)
    result[index] = lhs[index] + rhs[index];
}

int main(void) {
  int lhs[VECTOR_LENGTH];
  int rhs[VECTOR_LENGTH];
  int result[VECTOR_LENGTH];
  for (int index = 0; index < VECTOR_LENGTH; ++index) {
    lhs[index] = index;
    rhs[index] = index * 2;
  }
  vecadd_memory(lhs, rhs, result, VECTOR_LENGTH);
  for (int index = 0; index < VECTOR_LENGTH; ++index)
    if (result[index] != index * 3)
      return 1;
  return 0;
}
