#include <stdint.h>
#include <stdio.h>

typedef uint8_t byte4 __attribute__((vector_size(4)));

static volatile byte4 runtime_input = {0x12u, 0x34u, 0x56u, 0x78u};
static volatile byte4 runtime_lane_bias = {0u, 2u, 4u, 6u};

union packed_byte4 {
  byte4 vector;
  uint32_t integer;
};

__attribute__((noinline)) uint32_t vector_pack_kernel(uint32_t packed_input,
                                                      uint32_t packed_bias) {
  const union packed_byte4 lane_bias = {.integer = packed_bias};
  union packed_byte4 value = {
      .vector = ((union packed_byte4){.integer = packed_input}).vector +
                lane_bias.vector};
  value.integer ^= UINT32_C(0x01010101);
  value.vector -= lane_bias.vector;
  return value.integer;
}

int main(void) {
  const union packed_byte4 input = {.vector = runtime_input};
  const union packed_byte4 lane_bias = {.vector = runtime_lane_bias};
  const byte4 expected = {0x13u, 0x35u, 0x57u, 0x79u};
  const union packed_byte4 output = {
      .integer = vector_pack_kernel(input.integer, lane_bias.integer)};
  uint32_t checksum = 0;

  for (uint32_t lane = 0; lane < 4; ++lane) {
    if (output.vector[lane] != expected[lane]) {
      printf("FAILED\n");
      return 1;
    }
    checksum += (lane + 1u) * output.vector[lane];
  }

  printf("vector_pack checksum: %u\n", checksum);
  printf("PASSED\n");
  return 0;
}
