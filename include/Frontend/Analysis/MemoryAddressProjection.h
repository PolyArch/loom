#ifndef LOOM_FRONTEND_ANALYSIS_MEMORYADDRESSPROJECTION_H
#define LOOM_FRONTEND_ANALYSIS_MEMORYADDRESSPROJECTION_H

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <optional>

namespace loom::frontend::analysis {

/// Finite allocation extent from the same ABI DataLayout that owns pointer
/// indexing. The element stride includes allocation padding, unlike a store's
/// representation width. Dynamic, scalable, empty, and overflowing extents
/// have no projection.
std::optional<std::uint64_t>
projectFixedAllocationByteCount(mlir::LLVM::AllocaOp allocation);

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
  /// A no-unsigned-wrap step whose whole computed offset is this one scaled
  /// index proves the index itself non-negative.
  bool nonNegative = false;
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

std::optional<ResolvedLinearMemoryAddress> resolveLinearPointerAddress(
    mlir::Value pointer, mlir::Type accessType,
    llvm::function_ref<bool(mlir::Value)> isBoundaryRoot);

} // namespace loom::frontend::analysis

#endif
