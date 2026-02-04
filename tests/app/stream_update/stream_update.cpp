// Stream update kernel implementations

#include "stream_update.h"
#include "loom/loom.h"

void stream_update_cpu(const uint32_t* __restrict input,
                       uint32_t* __restrict output,
                       uint32_t n, uint32_t step) {
  uint32_t acc = 0;
  for (uint32_t i = 0; i + step < n; i += step) {
    for (uint32_t j = n; j > 0; j >>= 1) {
      uint32_t idx = (i + j) % n;
      acc += input[idx];
    }
  }
  output[0] = acc;
}

LOOM_ACCEL("stream_update")
void stream_update_dsa(const uint32_t* __restrict input,
                       uint32_t* __restrict output,
                       uint32_t n, uint32_t step) {
  uint32_t acc = 0;
  outer_loop:
  LOOM_TRIPCOUNT_FULL(8, 8, 1, 32)
  LOOM_PARALLEL(2, contiguous)
  for (uint32_t i = 0; i + step < n; i += step) {
    inner_loop:
    LOOM_TRIPCOUNT_FULL(8, 8, 1, 16)
    LOOM_UNROLL(2)
    for (uint32_t j = n; j > 0; j >>= 1) {
      uint32_t idx = (i + j) % n;
      acc += input[idx];
    }
  }
  output[0] = acc;
}
