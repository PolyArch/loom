// Loom kernel: vecnorm_l1
#ifndef VECNORM_L1_H
#define VECNORM_L1_H

#include <cstdint>
#include <cstddef>

uint32_t vecnorm_l1_cpu(const uint32_t* __restrict__ A, const uint32_t N);

uint32_t vecnorm_l1_dsa(const uint32_t* __restrict__ A, const uint32_t N);

#endif // VECNORM_L1_H
