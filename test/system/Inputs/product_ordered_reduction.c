__attribute__((noinline)) static unsigned int
ordered_reduction(const unsigned int *values, unsigned int count) {
  unsigned int accumulator = 0;
  for (unsigned int index = 0; index < count; ++index)
    accumulator = accumulator * 33u + values[index];
  return accumulator;
}

int main(void) {
  unsigned int ordinary[4] = {3u, 5u, 7u, 11u};
  unsigned int boundary[1] = {~0u};
  volatile unsigned int ordinary_count = 4;
  volatile unsigned int boundary_count = 1;
  volatile unsigned int empty_count = 0;

  unsigned int ordinary_result =
      ordered_reduction(ordinary, ordinary_count);
  unsigned int boundary_result =
      ordered_reduction(boundary, boundary_count);
  unsigned int empty_result = ordered_reduction(boundary, empty_count);
  return (ordinary_result != 113498u) | (boundary_result != ~0u) |
         (empty_result != 0u);
}
