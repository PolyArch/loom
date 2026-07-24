#ifndef LOOM_FRONTEND_LOWERING_GRAPH_PARALLEL_LOWERING_H
#define LOOM_FRONTEND_LOWERING_GRAPH_PARALLEL_LOWERING_H

#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <optional>

namespace loom {
namespace lowering {

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

} // namespace lowering
} // namespace loom

#endif // LOOM_FRONTEND_LOWERING_GRAPH_PARALLEL_LOWERING_H
