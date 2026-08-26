#ifndef LOOM_FRONTEND_LOWERING_EXACTMEMREFLAYOUT_H
#define LOOM_FRONTEND_LOWERING_EXACTMEMREFLAYOUT_H

#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>

namespace loom::lowering {

struct ExactMemRefLayout final {
  std::int64_t offset = 0;
  llvm::SmallVector<std::int64_t, 4> strides;
  std::optional<std::uint64_t> staticElementSpan;
};

/// Resolves the one exact static-stride interpretation admitted by D0. A
/// dynamic rank-one identity memref retains its canonical unit stride; all
/// higher-rank layouts require a static positive shape. Every address that is
/// valid for a static shape must fit the resolved signed index domain.
llvm::Expected<ExactMemRefLayout>
resolveExactMemRefLayout(mlir::MemRefType type, unsigned indexBits);

/// Returns true only when the ranked layout is proven not to map two distinct
/// in-bounds logical coordinates to the same physical element.
bool isProvablyInjectiveMemRefLayout(mlir::MemRefType type);

/// Returns dimension ordinals in major-to-minor storage order when the exact
/// type is a positive, statically shaped dense permutation.
llvm::Expected<llvm::SmallVector<unsigned, 4>>
resolveDenseMemRefStorageOrder(mlir::MemRefType type, unsigned indexBits);

} // namespace loom::lowering

#endif // LOOM_FRONTEND_LOWERING_EXACTMEMREFLAYOUT_H
