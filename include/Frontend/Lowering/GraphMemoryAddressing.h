#ifndef LOOM_FRONTEND_LOWERING_GRAPHMEMORYADDRESSING_H
#define LOOM_FRONTEND_LOWERING_GRAPHMEMORYADDRESSING_H

#include "Dataflow/IR/DataflowOps.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <optional>

namespace loom::lowering {

struct ExactElementStrideScale {
  std::int64_t scale = 1;
  unsigned exactSignedDivideShift = 0;
};

std::optional<ExactElementStrideScale>
resolveExactElementStrideScale(mlir::Value index, std::uint64_t byteStride,
                               std::uint64_t elementBytes);

struct LinearByteTerm {
  mlir::Value index;
  std::int64_t byteStride = 1;
};

struct LinearElementTerm {
  mlir::Value index;
  std::int64_t scale = 1;
  unsigned exactSignedDivideShift = 0;
};

struct ResolvedLinearMemoryAddress {
  mlir::Value root;
  llvm::SmallVector<LinearByteTerm, 4> terms;
  llvm::SmallVector<LinearElementTerm, 4> elementTerms;
  mlir::Type indexType;
  std::int64_t byteBias = 0;
  std::int64_t elementBias = 0;
  unsigned byteToElementShift = 0;
  std::uint64_t elementAllocByteCount = 0;
  std::uint64_t accessByteCount = 0;
  unsigned addressBitWidth = 0;
  llvm::SmallVector<mlir::Operation *, 4> gepsLeafToRoot;
};

std::optional<ResolvedLinearMemoryAddress>
resolveLinearMemoryAddress(mlir::Value pointer, mlir::Type accessType,
                           unsigned canonicalIndexBits);

/// Resolves an exact root-relative address while stopping at the service root
/// owned by the caller's projection boundary. The root predicate changes only
/// where the shared GEP walk stops; DataLayout and element-index proofs remain
/// identical to graph lowering.
std::optional<ResolvedLinearMemoryAddress> resolveLinearMemoryAddress(
    mlir::Value pointer, mlir::Type accessType, unsigned canonicalIndexBits,
    llvm::function_ref<bool(mlir::Value)> isBoundaryRoot);

/// Resolves one typed LLVM GEP chain as an exact DataLayout byte address.
/// Unlike the RootRelative overload above, this projection derives its
/// arithmetic width from the pointer address space and does not require a
/// synthetic canonical element-index representation.
std::optional<ResolvedLinearMemoryAddress>
resolveLinearPointerAddress(mlir::Value pointer, mlir::Type accessType);

std::optional<ResolvedLinearMemoryAddress>
resolveLinearMemoryAddress(mlir::Value pointer, dataflow::GraphOp graph,
                           mlir::Type accessType, unsigned canonicalIndexBits);

/// Resolves the one memory-service boundary root of an LLVM pointer lineage.
/// `isBoundaryRoot` is the only context-dependent policy: graph lowering uses
/// exact pointer-valued graph inputs, while ownership preflight uses exact
/// values crossing the selected scope. The lineage rules themselves have one
/// owner so preflight cannot drift from lowering.
mlir::Value resolveMemoryServiceBoundaryRoot(
    mlir::Value pointer, llvm::function_ref<bool(mlir::Value)> isBoundaryRoot);

} // namespace loom::lowering

#endif // LOOM_FRONTEND_LOWERING_GRAPHMEMORYADDRESSING_H
