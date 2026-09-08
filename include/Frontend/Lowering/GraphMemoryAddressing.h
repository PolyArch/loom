#ifndef LOOM_FRONTEND_LOWERING_GRAPHMEMORYADDRESSING_H
#define LOOM_FRONTEND_LOWERING_GRAPHMEMORYADDRESSING_H

#include "Frontend/Analysis/MemoryAddressProjection.h"
#include "Frontend/Analysis/StoredMemoryProvenance.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <optional>
#include <variant>

namespace loom::lowering {

/// One direct scalar LLVM access whose complete byte geometry is exactly one
/// point-coordinate partition of an enclosing loop domain. The projection is
/// shared by SCoP admission and independent-iteration proofs so neither owner
/// can silently interpret a GEP as an unscaled element index.
/// Its root is the loop-invariant direct GEP base, which may itself be an
/// enclosing loop's row address; alias proofs still follow its provenance.
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

/// Compares signed coordinates through lossless casts, including truncation
/// whose source IR explicitly guarantees no signed overflow.
bool isSameSignedMemoryCoordinate(mlir::Value value, mlir::Value expected,
                                  mlir::Operation *anchor);

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

ExactPointerPointAccessPairKind
classifyExactPointerPointAccessPair(const ExactPointerPointAccess &lhs,
                                    const ExactPointerPointAccess &rhs);

/// Temporary compiler projection from a descriptor input to the object input
/// represented by every pointer read from that descriptor in the selected
/// body. It is derived from immutable source IR, mapped only through explicit
/// IR cloning, and consumed during the same publication. It is never serialized
/// or used as an independent authority for another input artifact.
using PointerServiceBindings = llvm::DenseMap<mlir::Value, mlir::Value>;
using PointerServiceBindingsOutcome =
    std::variant<PointerServiceBindings,
                 frontend::analysis::StoredPointerRefusal>;

PointerServiceBindingsOutcome projectPointerServiceBindings(
    llvm::ArrayRef<mlir::Operation *> selectedBody,
    llvm::ArrayRef<mlir::Value> boundaryValues,
    frontend::analysis::StoredMemoryProvenance &provenance);

/// Resolves the one memory-service boundary root of an LLVM pointer lineage.
/// `isBoundaryRoot` is the only context-dependent policy: graph lowering uses
/// exact pointer-valued graph inputs, while ownership preflight uses exact
/// values crossing the selected scope. The lineage rules themselves have one
/// owner so preflight cannot drift from lowering.
mlir::Value resolveMemoryServiceBoundaryRoot(
    mlir::Value pointer, llvm::function_ref<bool(mlir::Value)> isBoundaryRoot,
    const PointerServiceBindings &bindings = PointerServiceBindings());

/// Returns whether a pointer lineage contains an LLVM pointer value loaded
/// from memory. Source-origin completion is required only for this case;
/// graph-memory lowering owns direct branch selections among boundary roots.
bool usesLoadedPointerService(mlir::Value pointer);

} // namespace loom::lowering

#endif // LOOM_FRONTEND_LOWERING_GRAPHMEMORYADDRESSING_H
