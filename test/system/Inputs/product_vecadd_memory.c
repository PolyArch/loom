__attribute__((noinline)) static void vecadd_memory(const int *lhs,
                                                     const int *rhs,
                                                     int *result) {
  result[0] = lhs[0] + rhs[0];
}

int main(void) {
  int lhs[1] = {19};
  int rhs[1] = {23};
  int result[1] = {0};
  vecadd_memory(lhs, rhs, result);
  return result[0] != 42;
}
