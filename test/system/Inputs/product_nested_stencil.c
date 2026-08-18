#include <stdint.h>

enum { ROWS = 2, COLUMNS = 2, INPUT_STRIDE = 3 };

static inline uint32_t transform_value(uint32_t value, uint32_t row,
                                       uint32_t column) {
  value = value * 3u + 1u;
  value ^= value >> 3u;
  value += 0x9e3779b9u;
  value = (value << 5u) ^ (value >> 13u);
  value *= 5u;
  value ^= value << 7u;
  value += 17u;
  value ^= value >> 11u;
  value = value * 3u + row;
  value ^= column;
  value += 1u;
  value = value * 3u + row;
  value ^= column;
  value += 1u;
  value = value * 3u + row;
  value ^= column;
  value += 1u;
  return value;
}

__attribute__((noinline)) void
nested_stencil(const uint32_t *restrict input, uint32_t *restrict output,
               int32_t rows, int32_t columns, int32_t input_stride) {
  for (int32_t y = 0; y < rows; ++y)
    for (int32_t x = 0; x < columns; ++x) {
      const int32_t input_index = y * input_stride + x;
      const int32_t output_index = y * columns + x;
      const uint32_t halo = input[input_index] + input[input_index + 1];
      output[output_index] =
          transform_value(halo, (uint32_t)y, (uint32_t)x);
    }
}

int main(void) {
  static const uint32_t input[6] = {2, 5, 11, 17, 23, 31};
  uint32_t output[ROWS * COLUMNS] = {0};
  nested_stencil(input, output, ROWS, COLUMNS, INPUT_STRIDE);
  for (uint32_t row = 0; row < ROWS; ++row)
    for (uint32_t column = 0; column < COLUMNS; ++column) {
      const uint32_t input_index = row * INPUT_STRIDE + column;
      const uint32_t expected =
          transform_value(input[input_index] + input[input_index + 1], row,
                          column);
      if (output[row * COLUMNS + column] != expected)
        return 1;
    }
  return 0;
}
