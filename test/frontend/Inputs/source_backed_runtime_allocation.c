#include <stddef.h>
#include <stdlib.h>

__attribute__((alloc_size(1), noinline)) static void *
allocate_bytes(size_t byte_count) {
  return malloc(byte_count);
}

__attribute__((noinline)) static void fill_values(int *values, int count) {
  for (int index = 0; index < count; ++index)
    values[index] = index * 3 + 1;
}

int main(void) {
  const int count = 16;
  int *values = (int *)allocate_bytes((size_t)count * sizeof(int));
  if (!values)
    return 1;
  fill_values(values, count);
  const int result = values[0] != 1 || values[count - 1] != 46;
  free(values);
  return result;
}
