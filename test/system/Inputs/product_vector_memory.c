typedef int v4i __attribute__((vector_size(16)));

__attribute__((noinline)) static int vector_memory(v4i *output,
                                                   const v4i *lhs,
                                                   const v4i *rhs) {
  *output = *lhs + *rhs;
  return 0;
}

int main(void) {
  v4i lhs = {1, 2, 3, 4};
  v4i rhs = {5, 6, 7, 8};
  v4i output = {0, 0, 0, 0};
  const int status = vector_memory(&output, &lhs, &rhs);
  return status == 0 && output[0] == 6 && output[3] == 12 ? 0 : 1;
}
