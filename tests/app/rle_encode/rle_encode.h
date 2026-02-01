// Loom kernel: rle_encode
#ifndef RLE_ENCODE_H
#define RLE_ENCODE_H

#include <cstdint>
#include <cstddef>

void rle_encode_cpu(const uint32_t* __restrict__ input_data, uint32_t* __restrict__ output_values, uint32_t* __restrict__ output_counts, uint32_t* __restrict__ output_length, const uint32_t N);

void rle_encode_dsa(const uint32_t* __restrict__ input_data, uint32_t* __restrict__ output_values, uint32_t* __restrict__ output_counts, uint32_t* __restrict__ output_length, const uint32_t N);

#endif // RLE_ENCODE_H
