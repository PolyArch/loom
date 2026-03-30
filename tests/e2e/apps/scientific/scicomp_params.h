#ifndef TAPESTRY_SCICOMP_PARAMS_H
#define TAPESTRY_SCICOMP_PARAMS_H

#include <cstdint>

namespace scicomp {

struct JacobiParams {
  unsigned tileRows = 16;
  unsigned tileCols = 16;
  unsigned haloWidth = 1;
};

struct CGParams {
  unsigned n = 100;
  unsigned nnz = 300;
  unsigned nnzPerRow = 3;
};

struct NBodyParams {
  unsigned nParticles = 64;
  unsigned nNeighbors = 10;
  unsigned rebuildInterval = 8;
};

inline constexpr uint64_t fp32Bytes = 4;

inline uint64_t tileBytes(const JacobiParams &p) {
  return static_cast<uint64_t>(p.tileRows) * p.tileCols * fp32Bytes;
}

inline uint64_t haloBytes(const JacobiParams &p) {
  const uint64_t rows = static_cast<uint64_t>(p.tileRows) + 2 * p.haloWidth;
  const uint64_t cols = static_cast<uint64_t>(p.tileCols) + 2 * p.haloWidth;
  return rows * cols * fp32Bytes;
}

inline uint64_t cgVectorBytes(const CGParams &p) {
  return static_cast<uint64_t>(p.n) * fp32Bytes;
}

inline uint64_t cgScalarBytes() { return fp32Bytes; }

inline uint64_t nbodyParticleBytes(const NBodyParams &p) {
  return static_cast<uint64_t>(p.nParticles) * 7 * fp32Bytes;
}

inline uint64_t nbodyNeighborBytes(const NBodyParams &p) {
  return static_cast<uint64_t>(p.nParticles) * p.nNeighbors * sizeof(uint32_t);
}

} // namespace scicomp

#endif // TAPESTRY_SCICOMP_PARAMS_H
