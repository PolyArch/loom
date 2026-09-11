#include "GraphActorSharing.h"

#include "GraphEventDependencies.h"

#include "Dataflow/IR/DataflowOps.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <cstddef>

namespace {

// The actors whose behaviour is fixed by their inputs alone, so one instance
// can serve every role that asked for the same replication, selection, or
// rendezvous. SSA multi-use is token broadcast, so each surviving use still
// observes the full ordered token sequence that its own actor produced.
//
// `dataflow.stream`, `dataflow.carry`, and `dataflow.gate` are deliberately
// absent. Lowering replicates a stream on purpose when a loop's iterations are
// proven independent, and a carry or gate owns one recurrence whose close
// signal the finalized-graph retirement proofs key on by identity.
bool isShareableControlActor(::mlir::Operation *op) {
  return ::llvm::isa<::dataflow::ConstantOp, ::dataflow::SyncOp,
                     ::dataflow::InvariantOp, ::dataflow::DemuxOp,
                     ::dataflow::MuxOp>(op);
}

std::size_t controlActorSignature(::mlir::Operation *op) {
  ::llvm::hash_code code =
      ::llvm::hash_value(op->getName().getAsOpaquePointer());
  for (::mlir::Value operand : op->getOperands())
    code = ::llvm::hash_combine(code, operand.getAsOpaquePointer());
  for (::mlir::Type type : op->getResultTypes())
    code = ::llvm::hash_combine(code, type.getAsOpaquePointer());
  return static_cast<std::size_t>(
      ::llvm::hash_combine(code, op->getAttrDictionary().getAsOpaquePointer()));
}

bool haveSameControlActorStructure(::mlir::Operation *lhs,
                                   ::mlir::Operation *rhs) {
  return lhs->getName() == rhs->getName() &&
         lhs->getNumOperands() == rhs->getNumOperands() &&
         lhs->getNumResults() == rhs->getNumResults() &&
         std::equal(lhs->operand_begin(), lhs->operand_end(),
                    rhs->operand_begin()) &&
         std::equal(lhs->result_type_begin(), lhs->result_type_end(),
                    rhs->result_type_begin()) &&
         lhs->getAttrDictionary() == rhs->getAttrDictionary();
}

// A frontier join is only as strict as the events it still has to wait for.
// Lowering builds each join from the frontier it held at that point, and a
// later projection can turn one of those inputs into a prerequisite of
// another: an iteration's read frontier becomes its write frontier once the
// body is proven to leave that write frontier alone, and the access completion
// that follows already depends on it. `reduceEvents` owns that relation, so a
// control-only join it reduces to one event is that event.
bool collapseCoveredEventJoins(::mlir::Block &body) {
  ::llvm::SmallVector<::dataflow::SyncOp, 4> collapsed;
  for (::mlir::Operation &op : body) {
    auto sync = ::llvm::dyn_cast<::dataflow::SyncOp>(&op);
    if (!sync || sync.getInputs().size() < 2)
      continue;
    if (!::llvm::all_of(sync.getInputs(), [](::mlir::Value input) {
          return ::llvm::isa<::mlir::NoneType>(input.getType());
        }))
      continue;
    if (!::llvm::all_of(
            sync.getOutputs().drop_front(),
            [](::mlir::Value output) { return output.use_empty(); }))
      continue;
    ::llvm::SmallVector<::mlir::Value, 4> reduced =
        ::loom::lowering::reduceEvents(sync.getInputs());
    if (reduced.size() != 1)
      continue;
    sync.getOutputs().front().replaceAllUsesWith(reduced.front());
    collapsed.push_back(sync);
  }
  for (::dataflow::SyncOp sync : collapsed)
    sync.erase();
  return !collapsed.empty();
}

// Sharing one actor exposes the next: two replays that become one actor make
// their two projections structurally equal, and those in turn make the joins
// that consumed them a rendezvous with one distinct prerequisite.
bool shareOnce(::mlir::Block &body) {
  ::llvm::DenseMap<std::size_t, ::llvm::SmallVector<::mlir::Operation *, 2>>
      representatives;
  ::llvm::SmallVector<::mlir::Operation *, 8> shared;
  for (::mlir::Operation &op : body) {
    if (!isShareableControlActor(&op))
      continue;
    auto &bucket = representatives[controlActorSignature(&op)];
    ::mlir::Operation *representative = nullptr;
    for (::mlir::Operation *candidate : bucket)
      if (haveSameControlActorStructure(candidate, &op)) {
        representative = candidate;
        break;
      }
    if (!representative) {
      bucket.push_back(&op);
      continue;
    }
    op.replaceAllUsesWith(representative);
    shared.push_back(&op);
  }
  for (::mlir::Operation *op : shared)
    op->erase();
  return !shared.empty();
}

} // namespace

void loom::lowering::shareCanonicalControlActors(::mlir::Block &body) {
  bool changed = true;
  while (changed) {
    changed = shareOnce(body);
    changed |= collapseCoveredEventJoins(body);
  }
  // The retirement frontier is a join like any other. Sharing can make two of
  // its witnesses the same event, and the graph's completion set names each
  // witness once.
  auto returnOp =
      ::llvm::dyn_cast<::dataflow::GraphReturnOp>(body.getTerminator());
  if (!returnOp)
    return;
  ::llvm::SmallVector<::mlir::Value, 4> retirement =
      ::loom::lowering::reduceEvents(returnOp.getComplete());
  if (!retirement.empty() &&
      retirement.size() != returnOp.getComplete().size())
    returnOp.getCompleteMutable().assign(retirement);
}
