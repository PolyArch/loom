#ifndef LOOM_LIB_DATAFLOW_IR_DATAFLOW_CANONICAL_LABELING_H
#define LOOM_LIB_DATAFLOW_IR_DATAFLOW_CANONICAL_LABELING_H

#include "Common/Artifact.h"
#include "Dataflow/IR/DataflowCanonicalEntity.h"

#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>
#include <vector>

namespace mlir {
class Operation;
} // namespace mlir

namespace dataflow::detail {

/// One entity carrier discovered by the canonical relation graph, with the
/// artifact-global EntityId assigned by its canonical slot. This is the single
/// derivation shared by the finalizer (which materializes the IDs) and the
/// importer (which independently recomputes and verifies them).
struct EntityCarrier {
  CanonicalDataflowEntityKind kind;
  std::uint64_t id;
  // For an op carrier this is the carrier operation. For an imported memory
  // root it is the owning dataflow.thread, and `formalArgIndex` selects the
  // memory formal argument; a fresh-allocation root carrier is the alloc op.
  mlir::Operation *op = nullptr;
  // The owning graph for an actor; null for a graph, launch, or memory root.
  mlir::Operation *graphOp = nullptr;
  // Set for an imported thread memory formal: the function-input ordinal, which
  // is also its entry-block-argument index.
  std::optional<unsigned> formalArgIndex = std::nullopt;
  // The resolved callee definition for a launch; null otherwise.
  mlir::Operation *calleeOp = nullptr;
};

/// The canonical bytes of the whole program plus the entity carriers ordered by
/// their dense artifact-global ID. `canonicalOperationOrder` lists every
/// operation in canonical-label order, so a consumer can derive an unordered
/// region's structural ordinals from canonical semantics rather than native
/// traversal order.
struct CanonicalLabeling {
  ::loom::CanonicalSemanticBytes bytes;
  std::vector<EntityCarrier> carriers;
  std::vector<mlir::Operation *> canonicalOperationOrder;
};

/// Build the exact typed semantic relation graph of `module`, compute a
/// canonical labeling that is invariant to nonsemantic presentation, assign
/// dense IDs by canonical slot, and emit the canonical bytes. The derived
/// entity-id carrier attribute is excluded from labeling. Fails on an
/// unresolved symbol or memory-root relation.
llvm::Expected<CanonicalLabeling>
computeCanonicalLabeling(mlir::ModuleOp module);

} // namespace dataflow::detail

#endif // LOOM_LIB_DATAFLOW_IR_DATAFLOW_CANONICAL_LABELING_H
