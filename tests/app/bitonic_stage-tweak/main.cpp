// Loom app test driver: bitonic_stage-tweak
#include "bitonic_stage-tweak.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>

int main() {
  constexpr uint32_t N = 8;
  constexpr uint32_t stage = 1;
  constexpr uint32_t pass = 0;

  float input[N] = {3.0f, 1.0f, 4.0f, 2.0f, 8.0f, 6.0f, 7.0f, 5.0f};
  float cpu[N];
  float accel[N];

  for (uint32_t i = 0; i < N; ++i) {
    cpu[i] = input[i];
    accel[i] = input[i];
  }

  bitonic_stage_cpu(cpu, N, stage, pass);
  bitonic_stage_dsa(accel, N, stage, pass);

  bool passed = true;
  for (uint32_t i = 0; i < N; ++i) {
    if (std::fabs(cpu[i] - accel[i]) > 1e-5f) {
      printf("FAILED at index %u: cpu=%.4f accel=%.4f\n", i, cpu[i],
             accel[i]);
      passed = false;
    }
  }

  if (passed) {
    printf("bitonic_stage-tweak: PASSED\n");
    return 0;
  }
  return 1;
}
