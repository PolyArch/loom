#ifndef LOOM_FRONTEND_ANALYSIS_STOREDMEMORYPROVENANCE_H
#define LOOM_FRONTEND_ANALYSIS_STOREDMEMORYPROVENANCE_H

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

#include <memory>
#include <variant>

namespace loom::frontend::analysis {

enum class StoredPointerRefusal {
  OpenInvocationDomain,
  UnsupportedMemoryEffect,
  UnknownByteAddress,
  OutOfBoundsAccess,
  UnknownAlias,
  UnknownIntegerDomain,
  IncompleteInitialization,
  PartialPointerWrite,
  UnsupportedPointerOrigin,
  UnsupportedPointerRepresentation,
  DistinctPointerOrigins,
  NullPointerOrigin,
  OriginNotInBoundary,
};

llvm::StringRef storedPointerRefusalSpelling(StoredPointerRefusal refusal);

struct StoredPointerTarget final {
  mlir::Value root;
  bool mayBeNull = false;
};

using ReachingPointerValueOutcome =
    std::variant<mlir::Value, StoredPointerRefusal>;

using StoredPointerTargetOutcome =
    std::variant<StoredPointerTarget, StoredPointerRefusal>;

/// Invocation-local analysis of pointer representations stored in finite LLVM
/// objects. Byte geometry, reaching writes, scalar guards, and complete byte
/// copies share one memoized source projection. The analysis never reads
/// runtime memory or changes the source IR. The root callable, its module,
/// all reachable callees, and their SSA values must remain alive and unchanged
/// for this object's lifetime. Destroy the analysis before any mutating
/// rewrite; a later compiler phase constructs a fresh analysis of its own input
/// IR. Effects that may change a queried representation, unknown aliases,
/// partial representations, and non-unique non-null targets remain typed
/// refusals.
class StoredMemoryProvenance final {
public:
  explicit StoredMemoryProvenance(mlir::LLVM::LLVMFuncOp rootCallable);
  ~StoredMemoryProvenance();
  StoredMemoryProvenance(StoredMemoryProvenance &&) noexcept;
  StoredMemoryProvenance &operator=(StoredMemoryProvenance &&) noexcept;

  StoredPointerTargetOutcome projectPointerTarget(mlir::Value pointer);

  /// Returns one SSA pointer equal to this read at its exact invocation.
  /// Capture uses this stronger relation when a fixed backing view is needed;
  /// it shares the same reaching-write and alias proof as target projection.
  ReachingPointerValueOutcome projectReachingPointerValue(
      mlir::LLVM::LoadOp read,
      llvm::ArrayRef<mlir::LLVM::CallOp> invocationPath = {});

  /// Projects the proven object into an explicit SSA input boundary. Each
  /// accepted input denotes that object's base in the same exact invocation;
  /// the returned value is one of boundaryValues, never an inferred host
  /// address or an unbound value from a different callable.
  StoredPointerTargetOutcome
  projectPointerTarget(mlir::Value pointer,
                       llvm::ArrayRef<mlir::Value> boundaryValues);

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace loom::frontend::analysis

#endif
