#include <stdint.h>

void byte_swap(const uint32_t *restrict input_data,
               uint32_t *restrict output_swapped, uint32_t n) {
  for (uint32_t i = 0; i < n; ++i) {
    uint32_t value = input_data[i];
    uint32_t byte0 = (value >> 0u) & 0xFFu;
    uint32_t byte1 = (value >> 8u) & 0xFFu;
    uint32_t byte2 = (value >> 16u) & 0xFFu;
    uint32_t byte3 = (value >> 24u) & 0xFFu;
    output_swapped[i] =
        (byte0 << 24u) | (byte1 << 16u) | (byte2 << 8u) | byte3;
  }
}

int main(void) {
  uint32_t input_data[4] = {0x12345678u, 0xAABBCCDDu, 0x11223344u, 0xFF00FF00u};
  uint32_t output_swapped[4] = {0, 0, 0, 0};
  byte_swap(input_data, output_swapped, 4);
  return (output_swapped[0] == 0x78563412u &&
          output_swapped[1] == 0xDDCCBBAAu &&
          output_swapped[2] == 0x44332211u &&
          output_swapped[3] == 0x00FF00FFu)
             ? 0
             : 1;
}
