#ifndef LOOM_FRONTEND_ANALYSIS_SCALARGUARDDOMAIN_H
#define LOOM_FRONTEND_ANALYSIS_SCALARGUARDDOMAIN_H

#include "mlir/IR/Value.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLFunctionalExtras.h"

#include <cstdint>
#include <optional>
#include <set>
#include <utility>
#include <vector>

namespace loom::frontend::analysis {

/// One integer value and the finite value set that a taken structured branch
/// admits for it.
struct GuardedValueDomain final {
  mlir::Value value;
  std::set<std::int64_t> values;
};

/// Projections the guard reader needs from its owner: constant folding of an
/// integer expression, forwarding of a value through proven reloads, and the
/// finite value set of a comparison bound that is not itself a constant.
struct ScalarGuardContext final {
  llvm::function_ref<std::optional<llvm::APInt>(mlir::Value)> constantValue;
  llvm::function_ref<mlir::Value(mlir::Value)> canonicalValue;
  llvm::function_ref<std::optional<std::set<std::int64_t>>(mlir::Value)>
      boundValues;
  std::uint64_t maximumValues = 0;
};

/// Value domains admitted by a set of taken structured branch conditions. Each
/// pair names one condition and the truth its taken region proves. Only
/// unsigned comparisons contribute an upper bound, because a signed comparison
/// leaves the compared value unbounded below. `interested` is the value the
/// caller is asking about; a bound that is known only through `boundValues` is
/// read for that value alone, which keeps the recursion finite.
std::vector<GuardedValueDomain> projectGuardedValueDomains(
    llvm::ArrayRef<std::pair<mlir::Value, bool>> takenConditions,
    const ScalarGuardContext &context, mlir::Value interested = {});

} // namespace loom::frontend::analysis

#endif // LOOM_FRONTEND_ANALYSIS_SCALARGUARDDOMAIN_H
