//===- DataflowCanonicalArtifact.cpp - finalizer and importer ------------===//
//
// Failure-atomic finalization of a Canonical Dataflow Program and the
// independent read-only importer view. The finalizer operates on a private
// clone, strips author-supplied derived IDs, validates the whole program,
// computes the canonical labeling, materializes the derived entity IDs, and
// frames the result with the Common finalizer. The importer reconstructs the
// canonical labeling from scratch, verifies every materialized ID, and resolves
// the typed reference maps used by consumers with no Mapping Artifact.
//
//===----------------------------------------------------------------------===//

#include "Dataflow/IR/DataflowCanonicalArtifact.h"

#include "Common/ArtifactFinalizer.h"
#include "Dataflow/IR/DataflowAttrs.h"
#include "Dataflow/IR/DataflowGraphValidation.h"
#include "Dataflow/IR/DataflowOps.h"

#include "DataflowCanonicalLabeling.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"

#include <optional>
#include <utility>

using namespace mlir;

namespace dataflow {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(), message);
}

// The derived entity-id carrier is a finalizer output, never trusted on input.
// Strip every occurrence before validation and labeling: operation attributes
// and both function-like argument and result dictionaries.
void stripDerivedIds(ModuleOp module) {
  StringAttr name = StringAttr::get(module.getContext(), kEntityIdAttrName);
  module.walk([&](Operation *op) {
    op->removeAttr(name);
    if (auto fn = dyn_cast<FunctionOpInterface>(op)) {
      for (unsigned index = 0; index < fn.getNumArguments(); ++index)
        fn.removeArgAttr(index, name);
      for (unsigned index = 0; index < fn.getNumResults(); ++index)
        fn.removeResultAttr(index, name);
    }
  });
}

// Publication removes unreachable private symbols. A definition is live when it
// is an externally visible (public) symbol, a dataflow.graph (a
// target-independent workload root a consumer such as DFG-sim addresses
// directly), or a host definition that issues a root dataflow.thread.launch;
// liveness then flows along symbol uses. Every unreachable private definition
// is erased so it contributes no canonical bytes and no entity ID. Operates on
// the private clone; the source module is never mutated.
void pruneUnreachablePrivateSymbols(ModuleOp module) {
  llvm::DenseSet<Operation *> live;
  llvm::SmallVector<Operation *> worklist;
  auto markLive = [&](Operation *op) {
    if (op && op->getParentOp() == module.getOperation() &&
        live.insert(op).second)
      worklist.push_back(op);
  };
  for (Operation &op : module.getBody()->getOperations()) {
    auto symbol = dyn_cast<SymbolOpInterface>(&op);
    if (!symbol)
      continue;
    bool issuesRootLaunch = false;
    op.walk([&](ThreadLaunchOp) { issuesRootLaunch = true; });
    if (symbol.getVisibility() == SymbolTable::Visibility::Public ||
        isa<GraphOp>(op) || issuesRootLaunch)
      markLive(&op);
  }
  while (!worklist.empty()) {
    Operation *op = worklist.pop_back_val();
    std::optional<SymbolTable::UseRange> uses = SymbolTable::getSymbolUses(op);
    if (!uses)
      continue;
    for (const SymbolTable::SymbolUse &use : *uses)
      markLive(SymbolTable::lookupNearestSymbolFrom(op, use.getSymbolRef()));
  }
  llvm::SmallVector<Operation *> dead;
  for (Operation &op : module.getBody()->getOperations()) {
    auto symbol = dyn_cast<SymbolOpInterface>(&op);
    if (symbol && symbol.getVisibility() == SymbolTable::Visibility::Private &&
        !live.contains(&op))
      dead.push_back(&op);
  }
  for (Operation *op : dead)
    op->erase();
}

llvm::Error validateProgram(ModuleOp module) {
  if (failed(mlir::verify(module.getOperation())))
    return invalid("canonical dataflow: module failed verification");
  return validateFinalizedProgram(module);
}

void materialize(const detail::EntityCarrier &carrier, MLIRContext *ctx) {
  auto id = EntityIdAttr::get(ctx, carrier.id);
  if (carrier.formalArgIndex) {
    cast<FunctionOpInterface>(carrier.op)
        .setArgAttr(*carrier.formalArgIndex,
                    StringAttr::get(ctx, kEntityIdAttrName), id);
    return;
  }
  carrier.op->setAttr(kEntityIdAttrName, id);
}

std::optional<std::uint64_t>
readMaterializedId(const detail::EntityCarrier &carrier) {
  Attribute attr;
  if (carrier.formalArgIndex)
    attr = cast<FunctionOpInterface>(carrier.op)
               .getArgAttr(*carrier.formalArgIndex, kEntityIdAttrName);
  else
    attr = carrier.op->getAttr(kEntityIdAttrName);
  if (auto entity = dyn_cast_or_null<EntityIdAttr>(attr))
    return entity.getId();
  return std::nullopt;
}

} // namespace

//===----------------------------------------------------------------------===//
// finalizeCanonicalDataflow
//===----------------------------------------------------------------------===//

llvm::Expected<CanonicalDataflowArtifact>
finalizeCanonicalDataflow(ModuleOp source) {
  OwningOpRef<ModuleOp> clone(cast<ModuleOp>(source.getOperation()->clone()));

  stripDerivedIds(clone.get());
  pruneUnreachablePrivateSymbols(clone.get());
  if (llvm::Error error = validateProgram(clone.get()))
    return std::move(error);

  llvm::Expected<detail::CanonicalLabeling> labeling =
      detail::computeCanonicalLabeling(clone.get());
  if (!labeling)
    return labeling.takeError();

  for (const detail::EntityCarrier &carrier : labeling->carriers)
    materialize(carrier, clone.get().getContext());

  ::loom::ArtifactIdentity identity = ::loom::finalizeArtifactIdentity(
      canonicalDataflowSchema, labeling->bytes);

  // Validate the complete closed structural relation set before publishing,
  // reusing the labeling already computed rather than recomputing it. Any
  // unresolved memory root, exposure, or owner relation fails finalization here
  // so a defective artifact is never published.
  if (llvm::Error error = CanonicalDataflowProgramView::buildView(
                              clone.get(), identity, *labeling)
                              .takeError())
    return std::move(error);

  return CanonicalDataflowArtifact(identity, std::move(clone),
                                   std::move(labeling->bytes));
}

//===----------------------------------------------------------------------===//
// CanonicalDataflowProgramView
//===----------------------------------------------------------------------===//

llvm::Expected<std::uint64_t> CanonicalDataflowProgramView::requireKind(
    const ::loom::ArtifactIdentity &artifact, std::uint64_t id,
    CanonicalDataflowEntityKind kind) const {
  if (artifact != identity_)
    return invalid("canonical dataflow: foreign-artifact reference");
  if (id >= kindOfId_.size())
    return invalid("canonical dataflow: entity ID out of range");
  if (kindOfId_[id] != kind)
    return invalid("canonical dataflow: wrong-kind reference");
  return slotOfId_[id];
}

llvm::Expected<CanonicalGraphView>
CanonicalDataflowProgramView::resolve(GraphRef ref) const {
  auto slot = requireKind(ref.artifact, ref.entity.value(),
                          CanonicalDataflowEntityKind::Graph);
  if (!slot)
    return slot.takeError();
  return graphs_[*slot];
}

llvm::Expected<CanonicalActorView>
CanonicalDataflowProgramView::resolve(ActorRef ref) const {
  auto slot = requireKind(ref.artifact, ref.entity.value(),
                          CanonicalDataflowEntityKind::Actor);
  if (!slot)
    return slot.takeError();
  return actors_[*slot];
}

llvm::Expected<CanonicalRootThreadLaunchView>
CanonicalDataflowProgramView::resolve(RootThreadLaunchRef ref) const {
  auto slot = requireKind(ref.artifact, ref.entity.value(),
                          CanonicalDataflowEntityKind::RootThreadLaunch);
  if (!slot)
    return slot.takeError();
  return rootThreadLaunches_[*slot];
}

llvm::Expected<CanonicalStaticGraphLaunchView>
CanonicalDataflowProgramView::resolve(StaticGraphLaunchRef ref) const {
  auto slot = requireKind(ref.artifact, ref.entity.value(),
                          CanonicalDataflowEntityKind::StaticGraphLaunch);
  if (!slot)
    return slot.takeError();
  return staticGraphLaunches_[*slot];
}

llvm::Expected<CanonicalLogicalMemoryRootView>
CanonicalDataflowProgramView::resolve(LogicalMemoryRootRef ref) const {
  auto slot = requireKind(ref.artifact, ref.entity.value(),
                          CanonicalDataflowEntityKind::LogicalMemoryRoot);
  if (!slot)
    return slot.takeError();
  return logicalMemoryRoots_[*slot];
}

llvm::Expected<CanonicalDataflowProgramView>
CanonicalDataflowProgramView::buildView(
    ModuleOp module, const ::loom::ArtifactIdentity &identity,
    const detail::CanonicalLabeling &labeling) {
  const std::size_t count = labeling.carriers.size();
  CanonicalDataflowProgramView view(identity);
  view.kindOfId_.resize(count, CanonicalDataflowEntityKind::Graph);
  view.slotOfId_.resize(count, 0);

  llvm::DenseMap<Operation *, std::uint64_t> graphIdOf;
  for (const detail::EntityCarrier &carrier : labeling.carriers)
    if (carrier.kind == CanonicalDataflowEntityKind::Graph)
      graphIdOf[carrier.op] = carrier.id;

  // Checked resolution of an owning or callee graph; a missing entry is an
  // unresolved or wrong-kind relation, never a silent GraphId(0).
  auto graphRef = [&](Operation *graphOp) -> llvm::Expected<GraphRef> {
    auto found = graphIdOf.find(graphOp);
    if (found == graphIdOf.end())
      return invalid("canonical dataflow: unresolved graph relation");
    return GraphRef{identity, GraphId(found->second)};
  };

  for (const detail::EntityCarrier &carrier : labeling.carriers) {
    view.kindOfId_[carrier.id] = carrier.kind;
    switch (carrier.kind) {
    case CanonicalDataflowEntityKind::Graph:
      view.graphIdByOp_[carrier.op] = carrier.id;
      view.slotOfId_[carrier.id] = view.graphs_.size();
      view.graphs_.push_back(
          {GraphRef{identity, GraphId(carrier.id)}, carrier.op});
      break;
    case CanonicalDataflowEntityKind::Actor: {
      llvm::Expected<GraphRef> owner = graphRef(carrier.graphOp);
      if (!owner)
        return owner.takeError();
      view.actorIdByOp_[carrier.op] = carrier.id;
      view.slotOfId_[carrier.id] = view.actors_.size();
      view.actors_.push_back({ActorRef{identity, ActorId(carrier.id)},
                              carrier.op, *owner,
                              *classifyCanonicalDataflowActor(carrier.op)});
      break;
    }
    case CanonicalDataflowEntityKind::RootThreadLaunch:
      view.rootThreadLaunchIdByOp_[carrier.op] = carrier.id;
      view.slotOfId_[carrier.id] = view.rootThreadLaunches_.size();
      view.rootThreadLaunches_.push_back(
          {RootThreadLaunchRef{identity, RootThreadLaunchId(carrier.id)},
           carrier.op, carrier.calleeOp});
      break;
    case CanonicalDataflowEntityKind::StaticGraphLaunch: {
      llvm::Expected<GraphRef> callee = graphRef(carrier.calleeOp);
      if (!callee)
        return callee.takeError();
      view.staticGraphLaunchIdByOp_[carrier.op] = carrier.id;
      view.slotOfId_[carrier.id] = view.staticGraphLaunches_.size();
      view.staticGraphLaunches_.push_back(
          {StaticGraphLaunchRef{identity, StaticGraphLaunchId(carrier.id)},
           carrier.op, *callee});
      break;
    }
    case CanonicalDataflowEntityKind::LogicalMemoryRoot: {
      LogicalMemoryRootRef ref{identity, LogicalMemoryRootId(carrier.id)};
      // The root-defining value: an imported thread memory formal is that
      // thread's entry-block argument at its input ordinal (function inputs are
      // the leading block arguments); a fresh allocation is its result.
      mlir::Value rootValue =
          carrier.formalArgIndex
              ? mlir::Value(cast<ThreadOp>(carrier.op)
                                .getBody()
                                .front()
                                .getArgument(*carrier.formalArgIndex))
              : mlir::Value(carrier.op->getResult(0));
      view.memoryRootIdByValue_[rootValue] = carrier.id;
      view.slotOfId_[carrier.id] = view.logicalMemoryRoots_.size();
      view.logicalMemoryRoots_.push_back(
          {ref, carrier.op, carrier.formalArgIndex});
      break;
    }
    }
  }

  // Generate and validate every closed structural inventory and derived
  // relation once. An unresolved memory root or exposure relation fails here.
  if (llvm::Error error = view.buildStructuralInventories(
          module, labeling.canonicalOperationOrder))
    return std::move(error);
  return view;
}

llvm::Expected<CanonicalDataflowProgramView>
CanonicalDataflowProgramView::import(
    ModuleOp finalizedModule,
    const ::loom::ArtifactIdentity &expectedIdentity) {
  if (llvm::Error error = validateProgram(finalizedModule))
    return std::move(error);

  llvm::Expected<detail::CanonicalLabeling> labeling =
      detail::computeCanonicalLabeling(finalizedModule);
  if (!labeling)
    return labeling.takeError();

  ::loom::ArtifactIdentity identity = ::loom::finalizeArtifactIdentity(
      canonicalDataflowSchema, labeling->bytes);
  if (identity != expectedIdentity)
    return invalid("canonical dataflow: identity does not match the artifact");

  // Verify every materialized ID against the independently recomputed canonical
  // assignment. Missing, stale, duplicate, out-of-range, and noncanonical
  // values all fail this equality.
  llvm::DenseSet<Operation *> opCarriers;
  llvm::DenseSet<std::pair<Operation *, unsigned>> argCarriers;
  for (const detail::EntityCarrier &carrier : labeling->carriers) {
    std::optional<std::uint64_t> stored = readMaterializedId(carrier);
    if (!stored)
      return invalid("canonical dataflow: missing materialized entity ID");
    if (*stored != carrier.id)
      return invalid("canonical dataflow: noncanonical materialized entity ID");
    if (carrier.formalArgIndex)
      argCarriers.insert({carrier.op, *carrier.formalArgIndex});
    else
      opCarriers.insert(carrier.op);
  }

  // Reject any derived carrier at an illegal location or of the wrong type. An
  // extra ID must never survive merely because the expected carriers verified.
  llvm::Error scanError = llvm::Error::success();
  finalizedModule.walk([&](Operation *op) {
    if (scanError)
      return;
    if (Attribute attr = op->getAttr(kEntityIdAttrName)) {
      if (!opCarriers.contains(op))
        scanError =
            invalid("canonical dataflow: entity ID on a non-carrier operation");
      else if (!isa<EntityIdAttr>(attr))
        scanError = invalid("canonical dataflow: illegal entity-id attribute");
    }
    auto fn = dyn_cast<FunctionOpInterface>(op);
    if (scanError || !fn)
      return;
    for (unsigned index = 0; index < fn.getNumArguments(); ++index)
      if (Attribute attr = fn.getArgAttr(index, kEntityIdAttrName)) {
        if (!argCarriers.contains({op, index}))
          scanError = invalid(
              "canonical dataflow: entity ID on a non-carrier argument");
        else if (!isa<EntityIdAttr>(attr))
          scanError =
              invalid("canonical dataflow: illegal entity-id attribute");
        if (scanError)
          return;
      }
    for (unsigned index = 0; index < fn.getNumResults(); ++index)
      if (fn.getResultAttr(index, kEntityIdAttrName)) {
        scanError =
            invalid("canonical dataflow: entity ID on a result dictionary");
        return;
      }
  });
  if (scanError)
    return std::move(scanError);

  return buildView(finalizedModule, identity, *labeling);
}

} // namespace dataflow
