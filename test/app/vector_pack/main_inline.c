#include <stdint.h>
#include <stdio.h>

typedef uint8_t byte4 __attribute__((vector_size(4)));

union packed_byte4 {
  byte4 vector;
  uint32_t integer;
};

int main(void) {
  const byte4 input = {0x12u, 0x34u, 0x56u, 0x78u};
  const byte4 expected = {0x13u, 0x35u, 0x57u, 0x79u};
  union packed_byte4 value = {.vector = input};
  value.integer ^= UINT32_C(0x01010101);
  const byte4 output = value.vector;
  uint32_t checksum = 0;

  for (uint32_t lane = 0; lane < 4; ++lane) {
    if (output[lane] != expected[lane]) {
      printf("FAILED\n");
      return 1;
    }
    checksum += (lane + 1u) * output[lane];
  }

  printf("vector_pack checksum: %u\n", checksum);
  printf("PASSED\n");
  return 0;
}
