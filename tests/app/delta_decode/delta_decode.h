// Loom kernel: delta_decode
#ifndef DELTA_DECODE_H
#define DELTA_DECODE_H

#include <cstdint>
#include <cstddef>

void delta_decode_cpu(const uint32_t* __restrict__ input_deltas, uint32_t* __restrict__ output_data, const uint32_t N);

void delta_decode_dsa(const uint32_t* __restrict__ input_deltas, uint32_t* __restrict__ output_data, const uint32_t N);

#endif // DELTA_DECODE_H
