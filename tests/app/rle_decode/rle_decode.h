// Loom kernel: rle_decode
#ifndef RLE_DECODE_H
#define RLE_DECODE_H

#include <cstdint>
#include <cstddef>

void rle_decode_cpu(const uint32_t* __restrict__ input_values, const uint32_t* __restrict__ input_counts, uint32_t* __restrict__ output_data, const uint32_t encoded_length);

void rle_decode_dsa(const uint32_t* __restrict__ input_values, const uint32_t* __restrict__ input_counts, uint32_t* __restrict__ output_data, const uint32_t encoded_length);

#endif // RLE_DECODE_H
