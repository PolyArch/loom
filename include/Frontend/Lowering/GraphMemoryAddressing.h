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
#include <variant>

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

/// One direct scalar LLVM access whose complete byte geometry is exactly one
/// point-coordinate partition of an enclosing loop domain. The projection is
/// shared by SCoP admission and independent-iteration proofs so neither owner
/// can silently interpret a GEP as an unscaled element index.
struct ExactPointerPointAccess {
  mlir::Operation *operation = nullptr;
  mlir::Value root;
  mlir::LLVM::GEPOp address;
  bool writes = false;
  std::uint64_t elementBytes = 0;
};

enum class ExactPointerPointAccessRefusal {
  NotMemoryAccess,
  UnsupportedEffect,
  UnsupportedElementType,
  NonDirectInboundsAddress,
  AddressRelationNotEstablished,
  NonLocalRoot,
};

using ExactPointerPointAccessOutcome =
    std::variant<ExactPointerPointAccess, ExactPointerPointAccessRefusal>;

ExactPointerPointAccessOutcome projectExactPointerPointAccess(
    mlir::Operation *operation, mlir::Operation *enclosingRoot,
    llvm::function_ref<bool(mlir::Value)> isPointCoordinate);

/// The complete write-bearing pair result for exact point accesses. A
/// same-root pair is iteration-local only when both byte partitions are
/// identical; different element widths can overlap across point coordinates
/// and therefore never acquire an identity dependence or independence proof.
enum class ExactPointerPointAccessPairKind {
  NoDependence,
  SameRootIterationLocal,
  ByteRelationNotEstablished,
  AliasNotEstablished,
};

ExactPointerPointAccessPairKind classifyExactPointerPointAccessPair(
    const ExactPointerPointAccess &lhs,
    const ExactPointerPointAccess &rhs);

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
