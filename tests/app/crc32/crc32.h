// Loom kernel: crc32
#ifndef CRC32_H
#define CRC32_H

#include <cstdint>
#include <cstddef>

void crc32_cpu(const uint32_t* __restrict__ input_data, uint32_t* __restrict__ output_checksum, const uint32_t N);

void crc32_dsa(const uint32_t* __restrict__ input_data, uint32_t* __restrict__ output_checksum, const uint32_t N);

#endif // CRC32_H
