#include "loom/loom.h"

#include <cstdint>

LOOM_ACCEL()
void unused_memref_kernel(const float *__restrict__ input,
                          const float *__restrict__ dead_input,
                          float *__restrict__ output,
                          uint32_t n) {
  LOOM_PARALLEL()
  for (uint32_t i = 0; i < n; ++i)
    output[i] = input[i];
}
