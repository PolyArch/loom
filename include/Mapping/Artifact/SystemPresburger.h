#ifndef LOOM_MAPPING_ARTIFACT_SYSTEMPRESBURGER_H
#define LOOM_MAPPING_ARTIFACT_SYSTEMPRESBURGER_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <vector>

namespace loom::mapping {

/// One convex integer-set disjunct over a Dataflow-owned logical signature.
/// Rows use MLIR Presburger order: dimensions, symbols, then the constant.
struct SystemPresburgerCell final {
  std::uint32_t dimensionCount = 0;
  std::uint32_t symbolCount = 0;
  std::vector<std::vector<std::int64_t>> equalities;
  std::vector<std::vector<std::int64_t>> inequalities;

  friend bool operator==(const SystemPresburgerCell &lhs,
                         const SystemPresburgerCell &rhs) {
    return lhs.dimensionCount == rhs.dimensionCount &&
           lhs.symbolCount == rhs.symbolCount &&
           lhs.equalities == rhs.equalities &&
           lhs.inequalities == rhs.inequalities;
  }
  friend bool operator!=(const SystemPresburgerCell &lhs,
                         const SystemPresburgerCell &rhs) {
    return !(lhs == rhs);
  }
};

struct SystemPresburgerPartitionAnalysis final {
  bool liesWithinLegalDomain = false;
  bool cellsAreDisjoint = false;
  bool coversLegalDomain = false;
};

llvm::Expected<SystemPresburgerCell>
canonicalizeSystemPresburgerCell(const SystemPresburgerCell &input);

llvm::Expected<bool>
systemPresburgerCellsIntersect(const SystemPresburgerCell &lhs,
                               const SystemPresburgerCell &rhs);

llvm::Expected<SystemPresburgerPartitionAnalysis>
analyzeSystemPresburgerPartition(llvm::ArrayRef<SystemPresburgerCell> cells,
                                 const SystemPresburgerCell &legalDomain);

llvm::Expected<bool>
systemPresburgerSetIsSubsetOf(llvm::ArrayRef<SystemPresburgerCell> subset,
                              llvm::ArrayRef<SystemPresburgerCell> superset);

} // namespace loom::mapping

#endif // LOOM_MAPPING_ARTIFACT_SYSTEMPRESBURGER_H
