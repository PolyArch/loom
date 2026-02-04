// Stream nested kernel implementations

#include "stream_nested.h"
#include "loom/loom.h"

void stream_nested_cpu(const uint32_t* __restrict input,
                       uint32_t* __restrict output, uint32_t n) {
  uint32_t acc = 0;
  for (uint32_t i = 1; i < n; i <<= 1) {
    for (uint32_t j = n; j != 0; j >>= 1) {
      for (uint32_t k = 1; k <= n; k *= 2) {
        uint32_t idx = (i + j + k) % n;
        acc += input[idx];
      }
    }
  }
  output[0] = acc;
}

LOOM_ACCEL("stream_nested")
void stream_nested_dsa(const uint32_t* __restrict input,
                       uint32_t* __restrict output, uint32_t n) {
  uint32_t acc = 0;
  outer_loop:
  LOOM_TRIPCOUNT_FULL(8, 8, 1, 16)
  LOOM_PARALLEL(2, contiguous)
  for (uint32_t i = 1; i < n; i <<= 1) {
    middle_loop:
    LOOM_TRIPCOUNT_FULL(8, 8, 1, 16)
    LOOM_UNROLL(2)
    for (uint32_t j = n; j != 0; j >>= 1) {
      inner_loop:
      LOOM_TRIPCOUNT_FULL(8, 8, 1, 16)
      LOOM_UNROLL(2)
      for (uint32_t k = 1; k <= n; k *= 2) {
        uint32_t idx = (i + j + k) % n;
        acc += input[idx];
      }
    }
  }
  output[0] = acc;
}
