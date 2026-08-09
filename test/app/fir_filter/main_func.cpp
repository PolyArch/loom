
#include <stdint.h>
#include <stdio.h>

constexpr uint32_t kInputSize = 8;
constexpr uint32_t kTapCount = 4;

void fir_filter_ref(const float *input, const float *coeffs, float *output,
                    uint32_t input_size, uint32_t tap_count) {
  for (uint32_t n = 0; n < input_size; ++n) {
    float sum = 0.0f;
    for (uint32_t k = 0; k < tap_count; ++k) {
      int32_t idx = static_cast<int32_t>(n) - static_cast<int32_t>(k);
      if (idx >= 0) {
        sum += coeffs[k] * input[idx];
      }
    }
    output[n] = sum;
  }
}

extern "C" __attribute__((noinline)) void
fir_filter_kernel(const float *input, const float *coeffs, float *output,
                  uint32_t input_size, uint32_t tap_count) {
  for (uint32_t n = 0; n < input_size; ++n) {
    float sum = 0.0f;
    for (uint32_t k = 0; k < tap_count; ++k) {
      int32_t idx = static_cast<int32_t>(n) - static_cast<int32_t>(k);
      if (idx >= 0) {
        sum += coeffs[k] * input[idx];
      }
    }
    output[n] = sum;
  }
}

float absolute(float value) { return value < 0.0f ? -value : value; }

float checksum(const float *values) {
  float sum = 0.0f;
  for (uint32_t i = 0; i < kInputSize; ++i) {
    sum += values[i];
  }
  return sum;
}

bool same(const float *lhs, const float *rhs) {
  for (uint32_t i = 0; i < kInputSize; ++i) {
    if (absolute(lhs[i] - rhs[i]) > 1e-5f) {
      return false;
    }
  }
  return true;
}

int main() {
  const float input[kInputSize] = {
      1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
  };
  const float coeffs[kTapCount] = {0.125f, 0.25f, 0.375f, 0.25f};
  float reference[kInputSize] = {};
  float candidate[kInputSize] = {};

  fir_filter_ref(input, coeffs, reference, kInputSize, kTapCount);
  fir_filter_kernel(input, coeffs, candidate, kInputSize, kTapCount);

  if (!same(reference, candidate)) {
    printf("FAILED\n");
    return 1;
  }

  printf("fir_filter checksum: %.6f\n", checksum(candidate));
  printf("PASSED\n");
  return 0;
}
