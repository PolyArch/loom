#include <stdint.h>

void compare_swap(const int32_t *restrict input_a,
                  const int32_t *restrict input_b,
                  int32_t *restrict output_min,
                  int32_t *restrict output_max, int32_t n) {
  for (int32_t i = 0; i < n; ++i) {
    int32_t a = input_a[i];
    int32_t b = input_b[i];
    if (a <= b) {
      output_min[i] = a;
      output_max[i] = b;
    } else {
      output_min[i] = b;
      output_max[i] = a;
    }
  }
}

int main(void) {
  int32_t input_a[4] = {3, 1, 5, 2};
  int32_t input_b[4] = {2, 4, 3, 6};
  int32_t output_min[4] = {0, 0, 0, 0};
  int32_t output_max[4] = {0, 0, 0, 0};
  compare_swap(input_a, input_b, output_min, output_max, 4);
  return (output_min[0] == 2 && output_min[1] == 1 &&
          output_min[2] == 3 && output_min[3] == 2 &&
          output_max[0] == 3 && output_max[1] == 4 &&
          output_max[2] == 5 && output_max[3] == 6)
             ? 0
             : 1;
}
