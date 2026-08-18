__attribute__((noinline)) static void special_math(const float *input,
                                                    float *output) {
  output[0] = __builtin_sqrtf(input[0]);
}

int main(void) {
  float input[1] = {4.0f};
  float output[1] = {-1.0f};
  special_math(input, output);
  return output[0] != 2.0f;
}
