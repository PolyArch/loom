#ifndef LOOM_TEST_APPLICATIONS_MLPERF_TINY_VWW_SMOKE_H
#define LOOM_TEST_APPLICATIONS_MLPERF_TINY_VWW_SMOKE_H

#include "mlperf-tiny-classification-stub.h"

inline constexpr int kVwwInputSize = 16;
inline const unsigned char g_person_detect_model_data[] = {0};

extern "C" __attribute__((noinline)) void
mlperf_tiny_vww_preprocess_kernel(const uint8_t *input, int8_t *centered,
                                  int count) {
  for (int index = 0; index < count; ++index)
    centered[index] = static_cast<int8_t>(static_cast<int>(input[index]) - 128);
}

int main() {
  uint8_t input[kVwwInputSize] = {0,   1,   7,   31,  63,  95,  126, 127,
                                  128, 129, 160, 192, 224, 248, 254, 255};
  int8_t centered[kVwwInputSize]{};
  mlperf_tiny_vww_preprocess_kernel(input, centered, kVwwInputSize);
  for (int index = 0; index < kVwwInputSize; ++index)
    if (static_cast<int>(centered[index]) !=
        static_cast<int>(input[index]) - 128)
      return 1;
  return 0;
}

#endif
