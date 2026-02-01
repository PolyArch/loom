// Loom kernel: vecnorm_l2
#ifndef VECNORM_L2_H
#define VECNORM_L2_H

#include <cstdint>
#include <cstddef>

uint32_t vecnorm_l2_cpu(const uint32_t* __restrict__ A, const uint32_t N);

uint32_t vecnorm_l2_dsa(const uint32_t* __restrict__ A, const uint32_t N);

#endif // VECNORM_L2_H
