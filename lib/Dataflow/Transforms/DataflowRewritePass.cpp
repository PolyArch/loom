// Optional typed Dataflow-only rewrites. Each rule owns one source pattern,
// one complete set of legality preconditions, and one deterministic result
// construction; a single closed switch over DataflowRewriteKind dispatches
// them. There is no rule registry, predicate language, optimizer controller,
// or persistent rewrite plan, and mandatory canonical finalization stays
// outside this file.

#include "Dataflow/Transforms/DataflowRewrite.h"

#include "Dataflow/IR/DataflowActorSemantics.h"
#include "Dataflow/IR/DataflowDialect.h"
#include "Dataflow/IR/DataflowGraphValidation.h"
#include "Dataflow/IR/OperationSchema.h"
#include "Dataflow/IR/DataflowOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>

namespace {

namespace semantics = ::dataflow::semantics;
using ::dataflow::DataflowRewriteKind;

//===----------------------------------------------------------------------===//
// PackUnpackRoundTripEliminate
//===----------------------------------------------------------------------===//

// `dataflow.pack` and `dataflow.unpack` are stateless exact-one bit
// representation adapters. Exact total width, row-major lane order and
// floating payload bit preservation are already owned by their verifiers,
// which whole-module pre-verification has run before any match here. The rule
// therefore adds no width arithmetic of its own: it only proves adjacency,
// exact type recovery, and that the intermediate token has no other consumer.
// It never inserts a replacement adapter, so no physical adaptation can be
// invented by eliminating one.
bool applyPackUnpackRoundTripEliminate(::mlir::Operation *op) {
  ::mlir::Operation *inner = nullptr;
  if (auto unpack = ::llvm::dyn_cast<::dataflow::UnpackOp>(op))
    inner = unpack.getPacked().getDefiningOp<::dataflow::PackOp>();
  else if (auto pack = ::llvm::dyn_cast<::dataflow::PackOp>(op))
    inner = pack.getVector().getDefiningOp<::dataflow::UnpackOp>();
  if (!inner)
    return false;
  if (!semantics::isStatelessOneTokenVectorBoundary(op) ||
      !semantics::isStatelessOneTokenVectorBoundary(inner))
    return false;

  // The intermediate is `op`'s only operand, so a single use is exactly "the
  // outer op is its sole consumer".
  ::mlir::Value intermediate = op->getOperand(0);
  if (!intermediate.hasOneUse())
    return false;

  // The outer result must recover the original source type exactly. Equal
  // total bit width is not a reason to forward a differently shaped value.
  ::mlir::Value source = inner->getOperand(0);
  if (op->getResult(0).getType() != source.getType())
    return false;

  op->getResult(0).replaceAllUsesWith(source);
  op->erase();
  inner->erase();
  return true;
}

//===----------------------------------------------------------------------===//
// ParallelizeSerializeRoundTripEliminate
//===----------------------------------------------------------------------===//

// Constant-work guard over the shared adapter transition semantics. The rule
// claims that, for one fixed group width, serialize composed with parallelize
// is the identity on the scalar data and scalar phase streams. That claim
// rests on the complete case split of one activation, so the guard reads back
// exactly those canonical transitions from the shared evaluators:
//
//   * a false scalar phase with nothing pending closes without a group;
//   * the width-completing true item publishes one full group and resets;
//   * a true item below the width only accumulates;
//   * a false scalar phase with a pending item flushes that partial tail
//     under a true group phase and then closes;
//   * serialize replays one true group and turns one false group phase into
//     one false scalar phase.
//
// The guard defines no state machine and no transition table of its own: it
// only asserts on the values the shared evaluators return, so if the owning
// semantics ever change shape the rewrite stops firing instead of silently
// relying on a stale assumption. The work is constant in the group width; the
// two width-sensitive points are probed at their boundary states rather than
// by enumerating lanes.
bool provesScalarStreamRoundTrip(std::uint64_t groupWidth) {
  if (groupWidth == 0)
    return false;

  const semantics::SemanticInputMask phaseOnly =
      semantics::semanticInput(semantics::ParallelizeInput::Phase);
  const semantics::SemanticInputMask phaseAndData =
      phaseOnly | semantics::semanticInput(semantics::ParallelizeInput::Data);

  const semantics::ParallelizeSemanticState idle;
  semantics::ParallelizeTransition emptyClose =
      semantics::evaluateParallelizeTransition(idle, groupWidth,
                                               /*scalarPhase=*/false,
                                               /*dataAvailable=*/false);
  if (!emptyClose.firing.ready ||
      emptyClose.firing.consumedInputs != phaseOnly || emptyClose.emitGroup ||
      emptyClose.emitTruePhase || !emptyClose.emitFalsePhase ||
      emptyClose.nextState.pendingItems != 0)
    return false;

  semantics::ParallelizeSemanticState nearlyFull;
  nearlyFull.pendingItems = groupWidth - 1;
  semantics::ParallelizeTransition fullGroup =
      semantics::evaluateParallelizeTransition(nearlyFull, groupWidth,
                                               /*scalarPhase=*/true,
                                               /*dataAvailable=*/true);
  if (!fullGroup.firing.ready ||
      fullGroup.firing.consumedInputs != phaseAndData || !fullGroup.emitGroup ||
      fullGroup.activeItems != groupWidth || !fullGroup.emitTruePhase ||
      fullGroup.emitFalsePhase || fullGroup.nextState.pendingItems != 0)
    return false;

  // A width of one has no accumulating state and no partial tail: every true
  // item completes a group, which the check above already covers.
  if (groupWidth >= 2) {
    semantics::ParallelizeTransition accumulate =
        semantics::evaluateParallelizeTransition(idle, groupWidth,
                                                 /*scalarPhase=*/true,
                                                 /*dataAvailable=*/true);
    if (!accumulate.firing.ready ||
        accumulate.firing.consumedInputs != phaseAndData ||
        accumulate.emitGroup || accumulate.emitTruePhase ||
        accumulate.emitFalsePhase || accumulate.nextState.pendingItems != 1)
      return false;

    semantics::ParallelizeSemanticState partial;
    partial.pendingItems = 1;
    semantics::ParallelizeTransition tailFlush =
        semantics::evaluateParallelizeTransition(partial, groupWidth,
                                                 /*scalarPhase=*/false,
                                                 /*dataAvailable=*/false);
    if (!tailFlush.firing.ready ||
        tailFlush.firing.consumedInputs != phaseOnly || !tailFlush.emitGroup ||
        tailFlush.activeItems != 1 || !tailFlush.emitTruePhase ||
        !tailFlush.emitFalsePhase || tailFlush.nextState.pendingItems != 0)
      return false;
  }

  semantics::SerializeTransition trueGroup =
      semantics::evaluateSerializeTransition(/*groupPhase=*/true,
                                             /*vectorAvailable=*/true,
                                             /*maskAvailable=*/true);
  if (!trueGroup.firing.ready ||
      trueGroup.firing.consumedInputs !=
          (semantics::semanticInput(semantics::SerializeInput::Phase) |
           semantics::semanticInput(semantics::SerializeInput::Vector) |
           semantics::semanticInput(semantics::SerializeInput::Mask)) ||
      !trueGroup.emitActiveItems || trueGroup.emitFalsePhase)
    return false;

  semantics::SerializeTransition falseClose =
      semantics::evaluateSerializeTransition(/*groupPhase=*/false,
                                             /*vectorAvailable=*/false,
                                             /*maskAvailable=*/false);
  return falseClose.firing.ready &&
         falseClose.firing.consumedInputs ==
             semantics::semanticInput(semantics::SerializeInput::Phase) &&
         !falseClose.emitActiveItems && falseClose.emitFalsePhase;
}

// Only `serialize(parallelize(data, phase))` is matched. The reverse shape is
// never an identity, because serialize drops inactive lanes and a following
// parallelize compacts the survivors across the original group boundaries.
bool applyParallelizeSerializeRoundTripEliminate(::mlir::Operation *op) {
  auto serialize = ::llvm::dyn_cast<::dataflow::SerializeOp>(op);
  if (!serialize)
    return false;

  auto groupVectorResult =
      ::llvm::dyn_cast<::mlir::OpResult>(serialize.getVector());
  auto groupMaskResult =
      ::llvm::dyn_cast<::mlir::OpResult>(serialize.getMask());
  auto groupPhaseResult =
      ::llvm::dyn_cast<::mlir::OpResult>(serialize.getGroupPhase());
  if (!groupVectorResult || !groupMaskResult || !groupPhaseResult)
    return false;

  // All three group operands must be the exact result numbers of one
  // parallelize, and that parallelize must have no other consumer of them.
  auto parallelize =
      ::llvm::dyn_cast<::dataflow::ParallelizeOp>(groupVectorResult.getOwner());
  if (!parallelize || groupMaskResult.getOwner() != parallelize ||
      groupPhaseResult.getOwner() != parallelize)
    return false;
  if (groupVectorResult.getResultNumber() != 0 ||
      groupMaskResult.getResultNumber() != 1 ||
      groupPhaseResult.getResultNumber() != 2)
    return false;
  if (!groupVectorResult.hasOneUse() || !groupMaskResult.hasOneUse() ||
      !groupPhaseResult.hasOneUse())
    return false;

  // Phase connection through the canonical vector-boundary projection: the
  // group phase parallelize publishes is exactly the one serialize consumes.
  std::optional<::mlir::Value> publishedGroupPhase =
      semantics::getVectorBoundaryOutputPhase(parallelize);
  std::optional<::mlir::Value> consumedGroupPhase =
      semantics::getVectorBoundaryInputPhase(serialize);
  std::optional<::mlir::Value> originalScalarPhase =
      semantics::getVectorBoundaryInputPhase(parallelize);
  std::optional<::mlir::Value> replayedScalarPhase =
      semantics::getVectorBoundaryOutputPhase(serialize);
  if (!publishedGroupPhase || !consumedGroupPhase || !originalScalarPhase ||
      !replayedScalarPhase || *publishedGroupPhase != *consumedGroupPhase)
    return false;

  // The payloads carried by a true phase on each side are exactly the values
  // being forwarded, so no unrelated token crosses the eliminated boundary.
  ::mlir::ValueRange groupPayloads =
      semantics::getVectorBoundaryTruePhaseInputPayloads(serialize);
  ::mlir::ValueRange scalarPayloads =
      semantics::getVectorBoundaryTruePhaseInputPayloads(parallelize);
  if (groupPayloads.size() != 2 || scalarPayloads.size() != 1 ||
      groupPayloads[0] != groupVectorResult ||
      groupPayloads[1] != groupMaskResult ||
      scalarPayloads[0] != parallelize.getData())
    return false;
  if (!semantics::isVectorBoundaryTruePhaseOutputPayload(
          groupVectorResult, *publishedGroupPhase) ||
      !semantics::isVectorBoundaryTruePhaseOutputPayload(
          groupMaskResult, *publishedGroupPhase) ||
      !semantics::isVectorBoundaryTruePhaseOutputPayload(serialize.getData(),
                                                         *replayedScalarPhase))
    return false;

  // The one fixed group width, read through the shared rank-one analysis.
  ::llvm::Expected<::mlir::VectorType> groupVector =
      semantics::analyzeFixedRankDataVector(groupVectorResult.getType(),
                                            semantics::VectorRank::One);
  if (!groupVector) {
    ::llvm::consumeError(groupVector.takeError());
    return false;
  }
  if (!provesScalarStreamRoundTrip(
          static_cast<std::uint64_t>(groupVector->getDimSize(0))))
    return false;

  serialize.getData().replaceAllUsesWith(parallelize.getData());
  serialize.getScalarPhase().replaceAllUsesWith(*originalScalarPhase);
  serialize->erase();
  parallelize->erase();
  return true;
}

//===----------------------------------------------------------------------===//
// ActivationPreservingConstantFold
//===----------------------------------------------------------------------===//

// The replacement constant must fire on the same activation stream as the
// subgraph it replaces, so every source constant must be triggered by the
// exact same ctrl SSA value. Two ctrl values that happen to be exact-one are
// still two activation streams and are never merged.
bool applyActivationPreservingConstantFold(::mlir::Operation *op) {
  // Canonical actor classification owns the selector, control, stateful and
  // memory exclusions; this rule adds no operation taxonomy of its own.
  if (!::dataflow::isCanonicalDataflowActor(
          op, ::dataflow::CanonicalDataflowActorKind::Compute))
    return false;
  if (op->getNumOperands() == 0 || op->getNumResults() != 1 ||
      op->getNumRegions() != 0 || op->getNumSuccessors() != 0)
    return false;
  // No observable effect, and totality established mechanically. An operation
  // whose speculatability is merely unknown is rejected, not assumed total.
  if (!::mlir::isPure(op))
    return false;

  ::mlir::Value ctrl;
  ::llvm::SmallVector<::mlir::Attribute, 4> operandValues;
  ::llvm::SmallSetVector<::mlir::Operation *, 4> sources;
  for (::mlir::Value operand : op->getOperands()) {
    auto constant = operand.getDefiningOp<::dataflow::ConstantOp>();
    if (!constant)
      return false;
    if (!ctrl)
      ctrl = constant.getCtrl();
    else if (ctrl != constant.getCtrl())
      return false;
    // No token of a source constant may escape. One constant legally feeding
    // several operands of this one actor is still fully consumed by it.
    if (!::llvm::all_of(constant.getValue().getUsers(),
                        [&](::mlir::Operation *user) { return user == op; }))
      return false;
    operandValues.push_back(constant.getConstValue());
    sources.insert(constant.getOperation());
  }

  // Operation::fold is not observationally read-only: it reports success with
  // an empty result vector when it has folded the operation in place. The
  // standard hook is therefore driven on a detached clone, never on the
  // candidate, so a fold this rule goes on to reject cannot leave a mutation
  // behind for the module-wide publication to pick up.
  //
  // The clone is regionless by the structural check above and is never
  // inserted into a block, so the only IR it touches is the use list of the
  // operands it shares with the candidate. Every use-based condition is
  // already settled above, and the probe is destroyed before any mutation
  // below, so no observer can see those transient uses. The retained value is
  // a context-uniqued Attribute, which outlives the probe; a Value result
  // would not, and is rejected along with an in-place empty result.
  ::mlir::TypedAttr typed;
  {
    ::mlir::Operation *probe = op->clone();
    ::llvm::SmallVector<::mlir::OpFoldResult, 1> folded;
    if (::mlir::succeeded(probe->fold(operandValues, folded)) &&
        folded.size() == 1)
      typed = ::llvm::dyn_cast_if_present<::mlir::TypedAttr>(
          ::llvm::dyn_cast_if_present<::mlir::Attribute>(folded.front()));
    probe->erase();
  }
  // An untyped or mistyped attribute cannot build a canonical
  // dataflow.constant.
  if (!typed || typed.getType() != op->getResult(0).getType())
    return false;

  ::mlir::OpBuilder builder(op);
  auto replacement = ::dataflow::ConstantOp::create(
      builder, op->getLoc(), typed.getType(), ctrl,
      ::llvm::cast<::mlir::Attribute>(typed));
  op->getResult(0).replaceAllUsesWith(replacement.getValue());
  op->erase();
  // Deduplicated, so a constant feeding several operands is erased once.
  for (::mlir::Operation *source : sources)
    source->erase();
  return true;
}

//===----------------------------------------------------------------------===//
// Dispatch and pass
//===----------------------------------------------------------------------===//

// The one closed dispatch over the catalog. Every kind resolves here to its
// own matcher, legality checker and builder; there is no callback table.
bool applySelectedRewrite(::mlir::Operation *op, DataflowRewriteKind kind) {
  switch (kind) {
  case DataflowRewriteKind::PackUnpackRoundTripEliminate:
    return applyPackUnpackRoundTripEliminate(op);
  case DataflowRewriteKind::ParallelizeSerializeRoundTripEliminate:
    return applyParallelizeSerializeRoundTripEliminate(op);
  case DataflowRewriteKind::ActivationPreservingConstantFold:
    return applyActivationPreservingConstantFold(op);
  }
  return false;
}

struct DataflowRewritePass
    : public ::mlir::PassWrapper<DataflowRewritePass,
                                 ::mlir::OperationPass<::mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DataflowRewritePass)

  DataflowRewritePass() = default;
  explicit DataflowRewritePass(DataflowRewriteKind selected) {
    kind = selected;
  }
  DataflowRewritePass(const DataflowRewritePass &other)
      : ::mlir::PassWrapper<DataflowRewritePass,
                            ::mlir::OperationPass<::mlir::ModuleOp>>(other) {
    // Assigning a pass option runs its callback and marks it as supplied, so
    // an unset kind must be left alone to survive pass cloning as unset.
    if (other.kind.hasValue())
      kind = other.kind.getValue();
  }

  ::llvm::StringRef getArgument() const final { return "dataflow-rewrite"; }

  ::llvm::StringRef getDescription() const final {
    return "Apply one selected typed Dataflow-only rewrite to a canonical "
           "Dataflow program.";
  }

  void getDependentDialects(::mlir::DialectRegistry &registry) const final {
    registry.insert<::dataflow::DataflowDialect>();
  }

  // The selected kind is the entire decision this pass makes, so it is
  // required rather than defaulted: silently picking one catalog entry would
  // apply a rewrite the caller never selected.
  ::mlir::Pass::Option<DataflowRewriteKind> kind{
      *this, "kind", ::llvm::cl::desc("typed Dataflow rewrite kind to apply"),
      ::llvm::cl::values(
          clEnumValN(DataflowRewriteKind::PackUnpackRoundTripEliminate,
                     "pack-unpack-round-trip-eliminate",
                     "eliminate an exact unpack(pack(vector)) or "
                     "pack(unpack(bits))"),
          clEnumValN(
              DataflowRewriteKind::ParallelizeSerializeRoundTripEliminate,
              "parallelize-serialize-round-trip-eliminate",
              "eliminate an exact serialize(parallelize(data, phase))"),
          clEnumValN(DataflowRewriteKind::ActivationPreservingConstantFold,
                     "activation-preserving-constant-fold",
                     "fold a same-ctrl constant Compute actor into one "
                     "dataflow.constant"))};

  void runOnOperation() final {
    if (!kind.hasValue()) {
      getOperation().emitError(
          "dataflow-rewrite requires an explicit 'kind' option");
      return signalPassFailure();
    }

    ::mlir::ModuleOp module = getOperation();
    ::mlir::OwningOpRef<::mlir::ModuleOp> candidate(
        ::mlir::cast<::mlir::ModuleOp>(module->clone()));

    // The rules read exact types recovered by the pack/unpack and
    // parallelize/serialize verifiers, so malformed input must be rejected
    // before any match rather than erased into apparent validity.
    if (::mlir::failed(validateCandidate(*candidate, "input program")))
      return signalPassFailure();

    if (!applySelectedRewrites(*candidate))
      return;

    if (::mlir::failed(validateCandidate(*candidate, "rewritten candidate")))
      return signalPassFailure();

    module->setAttrs((*candidate)->getAttrs());
    module.getBodyRegion().takeBody(candidate->getBodyRegion());
  }

  ::mlir::LogicalResult validateCandidate(::mlir::ModuleOp candidate,
                                          ::llvm::StringRef stage) {
    if (::mlir::failed(::mlir::verify(candidate))) {
      getOperation().emitError("dataflow-rewrite ")
          << stage << " failed native verification";
      return ::mlir::failure();
    }
    if (auto error = ::dataflow::validateFinalizedProgram(candidate)) {
      getOperation().emitError("dataflow-rewrite ")
          << stage << " failed canonical Dataflow validation: "
          << ::llvm::toString(std::move(error));
      return ::mlir::failure();
    }
    return ::mlir::success();
  }

  // Deterministic order: graph definitions in module order, then operations in
  // program order inside each graph body. Graph definitions are the iteration
  // domain and are never erased by a rule, while every match is recomputed
  // from live SSA at the moment it is applied. Overlapping round trips are
  // therefore resolved in favour of the earliest outer operation, and a match
  // invalidated by an earlier application simply stops matching instead of
  // being dereferenced. Each operation is considered once; the pass runs no
  // fixpoint loop and keeps no rewrite state between operations.
  bool applySelectedRewrites(::mlir::ModuleOp candidate) {
    ::llvm::SmallVector<::dataflow::GraphOp, 4> graphs;
    candidate.walk([&](::dataflow::GraphOp graph) {
      if (!graph.isExternal())
        graphs.push_back(graph);
    });

    bool changed = false;
    for (::dataflow::GraphOp graph : graphs)
      for (::mlir::Operation &op :
           ::llvm::make_early_inc_range(graph.getBody().front()))
        changed |= applySelectedRewrite(&op, kind.getValue());
    return changed;
  }
};

} // namespace

std::unique_ptr<::mlir::Pass>
dataflow::createDataflowRewritePass(DataflowRewriteKind kind) {
  return std::make_unique<DataflowRewritePass>(kind);
}

void dataflow::registerDataflowTransformsPasses() {
  static bool once = []() {
    ::mlir::PassRegistration<DataflowRewritePass>();
    return true;
  }();
  (void)once;
}
