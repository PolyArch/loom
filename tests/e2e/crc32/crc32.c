#include <stdint.h>

__attribute__((noinline)) void crc32(const uint32_t *restrict input_data,
                                     uint32_t *restrict output_checksum,
                                     uint32_t n) {
  volatile uint32_t polynomial = 0xEDB88320u;
  uint32_t crc = 0xFFFFFFFFu;

#pragma clang loop vectorize(disable) interleave(disable)
  for (uint32_t i = 0; i < n; ++i) {
    uint32_t data = input_data[i];

#pragma clang loop vectorize(disable) interleave(disable)
    for (uint32_t byte_idx = 0; byte_idx < 4u; ++byte_idx) {
      uint32_t byte = (data >> (byte_idx * 8u)) & 0xFFu;
      crc ^= byte;

#pragma clang loop vectorize(disable) interleave(disable)
      for (uint32_t bit = 0; bit < 8u; ++bit) {
        if ((crc & 1u) != 0u)
          crc = (crc >> 1u) ^ polynomial;
        else
          crc = crc >> 1u;
      }
    }
  }

  *output_checksum = ~crc;
}

int main(void) {
  uint32_t input_data[2];
  uint32_t output_checksum = 0u;
  input_data[0] = 0x12345678u;
  input_data[1] = 0xABCDEF00u;
  crc32(input_data, &output_checksum, 2u);
  return output_checksum == 1119308902u ? 0 : 1;
}
