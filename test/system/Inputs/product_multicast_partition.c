__attribute__((noinline)) static void
multicast_partition(const unsigned int *source, unsigned int *left,
                    unsigned int *right) {
  for (unsigned int index = 0; index < 4; ++index) {
    unsigned int shared = source[index] * 3u + 1u;
    left[index] = shared + 5u;
    right[index] = shared ^ 7u;
  }
}

int main(void) {
  unsigned int source[4] = {2u, 5u, 11u, 17u};
  unsigned int left[4] = {0u, 0u, 0u, 0u};
  unsigned int right[4] = {0u, 0u, 0u, 0u};
  const unsigned int expected_left[4] = {12u, 21u, 39u, 57u};
  const unsigned int expected_right[4] = {0u, 23u, 37u, 51u};
  multicast_partition(source, left, right);
  for (unsigned int index = 0; index < 4; ++index)
    if (left[index] != expected_left[index] ||
        right[index] != expected_right[index])
      return 1;
  return 0;
}
