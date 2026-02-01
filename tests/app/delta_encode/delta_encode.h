// Loom kernel: delta_encode
#ifndef DELTA_ENCODE_H
#define DELTA_ENCODE_H

#include <cstdint>
#include <cstddef>

void delta_encode_cpu(const uint32_t* __restrict__ input_data, uint32_t* __restrict__ output_deltas, const uint32_t N);

void delta_encode_dsa(const uint32_t* __restrict__ input_data, uint32_t* __restrict__ output_deltas, const uint32_t N);

#endif // DELTA_ENCODE_H
