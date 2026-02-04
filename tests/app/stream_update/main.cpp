// Main test for stream_update kernel

#include "stream_update.h"
#include <cstdio>

int main() {
  const uint32_t n = 32;
  const uint32_t step = 3;
  uint32_t input[n];
  for (uint32_t i = 0; i < n; ++i)
    input[i] = (i + 1) * 2;

  uint32_t cpu_out[1] = {0};
  uint32_t dsa_out[1] = {0};

  stream_update_cpu(input, cpu_out, n, step);
  stream_update_dsa(input, dsa_out, n, step);

  bool passed = (cpu_out[0] == dsa_out[0]);
  if (!passed) {
    printf("FAILED: cpu=%u dsa=%u\n", cpu_out[0], dsa_out[0]);
    return 1;
  }
  printf("PASSED: stream_update result=%u\n", dsa_out[0]);
  return 0;
}
