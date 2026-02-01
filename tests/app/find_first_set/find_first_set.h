// Loom kernel: find_first_set
#ifndef FIND_FIRST_SET_H
#define FIND_FIRST_SET_H

#include <cstdint>
#include <cstddef>

void find_first_set_cpu(const uint32_t* __restrict__ input_data, uint32_t* __restrict__ output_position, const uint32_t N);

void find_first_set_dsa(const uint32_t* __restrict__ input_data, uint32_t* __restrict__ output_position, const uint32_t N);

#endif // FIND_FIRST_SET_H
