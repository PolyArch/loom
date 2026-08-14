#ifndef LOOM_LIB_PNR_SPATIALTAGCOLORING_H
#define LOOM_LIB_PNR_SPATIALTAGCOLORING_H

#include "PnR/PnrIndex.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace loom::pnr::detail {

inline constexpr std::size_t spatialTagExactColoringVertexLimit = 64;

struct SpatialTagColoringVertex final {
  std::uint32_t tagWidthBits = 0;
  bool restricted = false;
};

struct SpatialTagColoringInterval final {
  llvm::APInt lower;
  llvm::APInt upper;
};

struct SpatialTagColoringProblemView final {
  llvm::ArrayRef<SpatialTagColoringVertex> vertices;
  llvm::ArrayRef<PnrIndex> vertexDomainOffsets;
  llvm::ArrayRef<PnrIndex> vertexDomains;
  llvm::ArrayRef<PnrIndex> vertexIntervalOffsets;
  llvm::ArrayRef<SpatialTagColoringInterval> intervals;
  PnrIndex domainCount = 0;
};

struct SpatialTagColoringResult final {
  std::vector<std::optional<llvm::APInt>> values;
  std::uint64_t unassignedCount = 0;
  std::uint64_t conflictCount = 0;
};

llvm::Expected<SpatialTagColoringResult>
colorSpatialTagInterference(const SpatialTagColoringProblemView &problem);

} // namespace loom::pnr::detail

#endif // LOOM_LIB_PNR_SPATIALTAGCOLORING_H
