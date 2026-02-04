// Main test for stream_nested kernel

#include "stream_nested.h"
#include <cstdio>

int main() {
  const uint32_t n = 16;
  uint32_t input[n];
  for (uint32_t i = 0; i < n; ++i)
    input[i] = i + 1;

  uint32_t cpu_out[1] = {0};
  uint32_t dsa_out[1] = {0};

  stream_nested_cpu(input, cpu_out, n);
  stream_nested_dsa(input, dsa_out, n);

  bool passed = (cpu_out[0] == dsa_out[0]);
  if (!passed) {
    printf("FAILED: cpu=%u dsa=%u\n", cpu_out[0], dsa_out[0]);
    return 1;
  }
  printf("PASSED: stream_nested result=%u\n", dsa_out[0]);
  return 0;
}
