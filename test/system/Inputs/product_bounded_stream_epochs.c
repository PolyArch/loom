__attribute__((noinline)) static void
stream_producer(const unsigned int *source, unsigned int *temporary) {
  temporary[0] = source[0] + 3u;
  temporary[1] = source[1] + 3u;
}

__attribute__((noinline)) static void
stream_consumer(const unsigned int *temporary, unsigned int *result) {
  result[0] = temporary[0] * 5u;
  result[1] = temporary[1] * 5u;
}

__attribute__((noinline)) static void
bounded_stream_chain(const unsigned int *source, unsigned int *result) {
  unsigned int temporary[2];
  stream_producer(source, temporary);
  stream_consumer(temporary, result);
}

int main(void) {
  const unsigned int sources[2][2] = {{1u, 4u}, {6u, 9u}};
  const unsigned int expected[2][2] = {{20u, 35u}, {45u, 60u}};
  unsigned int result[2] = {0u, 0u};
  for (unsigned int epoch = 0; epoch < 2; ++epoch) {
    bounded_stream_chain(sources[epoch], result);
    for (unsigned int index = 0; index < 2; ++index)
      if (result[index] != expected[epoch][index])
        return 1;
  }
  return 0;
}
