// Loom kernel: bitonic_stage-tweak
#ifndef BITONIC_STAGE_TWEAK_H
#define BITONIC_STAGE_TWEAK_H

#include <cstdint>
#include <cstddef>

void bitonic_stage_cpu(float* __restrict__ inplace, const uint32_t N, const uint32_t stage, const uint32_t pass);

void bitonic_stage_dsa(float* __restrict__ inplace, const uint32_t N, const uint32_t stage, const uint32_t pass);

#endif // BITONIC_STAGE_TWEAK_H
