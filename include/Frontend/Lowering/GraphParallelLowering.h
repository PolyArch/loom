#ifndef LOOM_FRONTEND_LOWERING_GRAPH_PARALLEL_LOWERING_H
#define LOOM_FRONTEND_LOWERING_GRAPH_PARALLEL_LOWERING_H

#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <optional>
#include <string>

namespace loom::lowering {

struct FixedParallelDomain {
  ::llvm::SmallVector<int64_t, 4> lower;
  ::llvm::SmallVector<int64_t, 4> upper;
  ::llvm::SmallVector<int64_t, 4> step;
};

std::optional<FixedParallelDomain>
getFixedParallelDomain(::mlir::Operation *op);

void forEachParallelPoint(
    const FixedParallelDomain &domain,
    ::llvm::function_ref<void(::llvm::ArrayRef<int64_t>)> callback);

::mlir::LogicalResult checkGraphOwnedParallelPreconditions(
    ::llvm::ArrayRef<::mlir::Operation *> parallelOps);

/// Proves that one effect-form forall can become an unordered logical thread
/// domain. Dynamic read-only and atomic domains are admissible; plain writes
/// require the same fixed-domain disjointness proof used by graph-owned
/// parallel lowering.
::mlir::LogicalResult
checkLogicalThreadParallelPreconditions(::mlir::Operation *forall);

/// Applies the graph-owned parallel legality proof to every parallel domain
/// inside one just-materialized Spatial carrier. Diagnostics are captured and
/// returned as a candidate-local explanation instead of being emitted. This
/// lets ownership search reject a non-finalizable coordinate before graph
/// publication while the lowering pass retains the same proof as its strict
/// verifier.
std::optional<std::string>
explainSpatialCarrierParallelRejection(::mlir::Operation *spatialCarrier);

} // namespace loom::lowering

#endif // LOOM_FRONTEND_LOWERING_GRAPH_PARALLEL_LOWERING_H
