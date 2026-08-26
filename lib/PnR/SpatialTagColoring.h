#ifndef LOOM_LIB_PNR_SPATIALTAGCOLORING_H
#define LOOM_LIB_PNR_SPATIALTAGCOLORING_H

#include "PnR/PnrIndex.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <tuple>
#include <vector>

namespace loom::pnr::detail {

inline constexpr std::size_t spatialTagExactColoringVertexLimit = 64;

struct SpatialTagColoringVertex final {
  std::uint32_t tagWidthBits = 0;
  bool restricted = false;

  friend bool operator==(SpatialTagColoringVertex lhs,
                         SpatialTagColoringVertex rhs) {
    return lhs.tagWidthBits == rhs.tagWidthBits &&
           lhs.restricted == rhs.restricted;
  }
};

struct SpatialTagColoringInterval final {
  llvm::APInt lower;
  llvm::APInt upper;

  friend bool operator==(const SpatialTagColoringInterval &lhs,
                         const SpatialTagColoringInterval &rhs) {
    return lhs.lower == rhs.lower && lhs.upper == rhs.upper;
  }
};

struct SpatialTagColoringVertexIdentity final {
  std::uint64_t owner = 0;
  std::uint64_t originKind = 0;
  std::uint64_t origin = 0;

  friend bool operator==(SpatialTagColoringVertexIdentity lhs,
                         SpatialTagColoringVertexIdentity rhs) {
    return lhs.owner == rhs.owner && lhs.originKind == rhs.originKind &&
           lhs.origin == rhs.origin;
  }
  friend bool operator<(SpatialTagColoringVertexIdentity lhs,
                        SpatialTagColoringVertexIdentity rhs) {
    return std::tie(lhs.owner, lhs.originKind, lhs.origin) <
           std::tie(rhs.owner, rhs.originKind, rhs.origin);
  }
};

struct SpatialTagColoringComponentCache final {
  std::vector<SpatialTagColoringVertexIdentity> identities;
  std::vector<SpatialTagColoringVertex> vertices;
  std::vector<PnrIndex> domainOffsets;
  std::vector<PnrIndex> domains;
  std::vector<PnrIndex> intervalOffsets;
  std::vector<SpatialTagColoringInterval> intervals;
  std::vector<PnrIndex> conflictOffsets;
  std::vector<PnrIndex> conflicts;
  std::vector<std::optional<llvm::APInt>> values;
  std::uint64_t exactWorkBefore = 0;
  std::uint64_t exactWorkAfter = 0;
  std::uint64_t unassignedCount = 0;
  std::uint64_t conflictCount = 0;
};

struct SpatialTagColoringCache final {
  std::vector<SpatialTagColoringComponentCache> components;
};

struct SpatialTagColoringProblemView final {
  llvm::ArrayRef<SpatialTagColoringVertex> vertices;
  llvm::ArrayRef<PnrIndex> vertexDomainOffsets;
  llvm::ArrayRef<PnrIndex> vertexDomains;
  llvm::ArrayRef<PnrIndex> vertexIntervalOffsets;
  llvm::ArrayRef<SpatialTagColoringInterval> intervals;
  PnrIndex domainCount = 0;
  llvm::ArrayRef<PnrIndex> vertexConflictOffsets;
  llvm::ArrayRef<PnrIndex> vertexConflicts;
};

struct SpatialTagColoringResult final {
  std::vector<std::optional<llvm::APInt>> values;
  std::vector<SpatialTagColoringVertexIdentity> recomputedIdentities;
  std::uint64_t unassignedCount = 0;
  std::uint64_t conflictCount = 0;
  SpatialTagColoringCache cache;
};

llvm::Expected<SpatialTagColoringResult> colorSpatialTagInterference(
    const SpatialTagColoringProblemView &problem,
    llvm::ArrayRef<SpatialTagColoringVertexIdentity> identities = {},
    const SpatialTagColoringCache *previous = nullptr);

} // namespace loom::pnr::detail

#endif // LOOM_LIB_PNR_SPATIALTAGCOLORING_H
