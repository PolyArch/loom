// Loom kernel: lz77_compress
#ifndef LZ77_COMPRESS_H
#define LZ77_COMPRESS_H

#include <cstdint>
#include <cstddef>

void lz77_compress_cpu(const uint32_t* __restrict__ input_data, uint32_t* __restrict__ output_offsets, uint32_t* __restrict__ output_lengths, uint32_t* __restrict__ output_literals, uint32_t* __restrict__ output_count, const uint32_t N, const uint32_t window_size);

void lz77_compress_dsa(const uint32_t* __restrict__ input_data, uint32_t* __restrict__ output_offsets, uint32_t* __restrict__ output_lengths, uint32_t* __restrict__ output_literals, uint32_t* __restrict__ output_count, const uint32_t N, const uint32_t window_size);

#endif // LZ77_COMPRESS_H
