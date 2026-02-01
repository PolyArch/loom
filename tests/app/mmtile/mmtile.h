// Loom kernel: mmtile
#ifndef MMTILE_H
#define MMTILE_H

#include <cstdint>
#include <cstddef>

void mmtile_cpu(const uint32_t* __restrict__ A, const uint32_t* __restrict__ B, uint32_t* __restrict__ C, const uint32_t M, const uint32_t N, const uint32_t K, const uint32_t TILE_M, const uint32_t TILE_N, const uint32_t TILE_K);

void mmtile_dsa(const uint32_t* __restrict__ A, const uint32_t* __restrict__ B, uint32_t* __restrict__ C, const uint32_t M, const uint32_t N, const uint32_t K, const uint32_t TILE_M, const uint32_t TILE_N, const uint32_t TILE_K);

#endif // MMTILE_H
