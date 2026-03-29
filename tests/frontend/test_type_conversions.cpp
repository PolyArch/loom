// T1 + T2: Integer and float type conversions produce valid DFG.
// Tests: sext, trunc, sitofp, fptosi
#include <cstdint>

void type_conversions(const int32_t *input, int16_t *out_trunc,
                      int64_t *out_ext, float *out_float,
                      int32_t *out_int, int n) {
  for (int i = 0; i < n; ++i) {
    int32_t y = input[i];
    // Truncation: int32 -> int16
    out_trunc[i] = static_cast<int16_t>(y);
    // Sign extension: int32 -> int64
    out_ext[i] = static_cast<int64_t>(y);
    // Int to float
    out_float[i] = static_cast<float>(y);
    // Float to int
    float f = static_cast<float>(y);
    out_int[i] = static_cast<int32_t>(f);
  }
}

int main() {
  constexpr int n = 4;
  int32_t input[n] = {100, -200, 300, -400};
  int16_t out_trunc[n] = {};
  int64_t out_ext[n] = {};
  float out_float[n] = {};
  int32_t out_int[n] = {};

  type_conversions(input, out_trunc, out_ext, out_float, out_int, n);
  return 0;
}
