#ifndef LOOM_TEST_APPLICATIONS_MLPERF_TINY_KWS_SMOKE_H
#define LOOM_TEST_APPLICATIONS_MLPERF_TINY_KWS_SMOKE_H

#define V0_1_KWS_KWS_INPUTS_H_
#define kws_model_data_h
#define V0_1_KWS_MODEL_SETTINGS_H_

#include "mlperf-tiny-classification-stub.h"

inline constexpr int kKwsInputSize = 16;
inline constexpr int kCategoryCount = 12;
inline const unsigned char g_kws_model_data[] = {0};

extern "C" __attribute__((noinline)) int
mlperf_tiny_kws_postprocess_kernel(const int8_t *quantized,
                                   float *probabilities, int count, float scale,
                                   int zeroPoint) {
  int best = 0;
  for (int index = 0; index < count; ++index) {
    probabilities[index] =
        DequantizeInt8ToFloat(quantized[index], scale, zeroPoint);
    if (probabilities[index] > probabilities[best])
      best = index;
  }
  return best;
}

int main() {
  int8_t quantized[kCategoryCount] = {-8, -4, 1, 3, 7, 12, 5, 2, 0, 9, 4, 6};
  float probabilities[kCategoryCount]{};
  const int best = mlperf_tiny_kws_postprocess_kernel(
      quantized, probabilities, kCategoryCount, 0.125f, 0);
  return best == 5 && probabilities[5] == 1.5f && probabilities[0] == -1.0f ? 0
                                                                            : 1;
}

#endif
