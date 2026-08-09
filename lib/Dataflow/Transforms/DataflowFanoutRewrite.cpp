#include "DataflowRewriteInternal.h"

#include "Dataflow/IR/DataflowCanonicalEntity.h"
#include "Dataflow/IR/DataflowOps.h"
#include "Dataflow/IR/OperationSchema.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <iterator>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

namespace dataflow::detail {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "dataflow_fanout_rewrite_invalid: " + message);
}

struct Replica final {
  CanonicalActorView actor;
};

llvm::Expected<std::optional<Replica>>
classifyReplica(const CanonicalActorView &actor) {
  std::optional<OperationSchemaId> schema = operationSchemaOf(actor.op);
  if (!schema || actor.kind != CanonicalDataflowActorKind::Compute ||
      !isDeterministic(*schema) || actor.op->getNumResults() != 1 ||
      actor.op->getNumRegions() != 0 || actor.op->getNumSuccessors() != 0 ||
      !mlir::isPure(actor.op))
    return std::optional<Replica>{};
  return std::optional<Replica>{Replica{actor}};
}

bool haveSameReplicaInputs(const Replica &lhs, const Replica &rhs) {
  return lhs.actor.graph == rhs.actor.graph &&
         haveIdenticalDeterministicComputeInstance(lhs.actor.op->getResult(0),
                                                   rhs.actor.op->getResult(0));
}

llvm::Expected<std::vector<Replica>>
collectReplicas(const CanonicalDataflowProgramView &view) {
  std::vector<Replica> result;
  for (const CanonicalActorView &actor : view.actors()) {
    auto replica = classifyReplica(actor);
    if (!replica)
      return replica.takeError();
    if (*replica)
      result.push_back(std::move(**replica));
  }
  return result;
}

llvm::Expected<mlir::OpOperand *>
resolveConsumerOperand(const CanonicalDataflowProgramView &view,
                       const CanonicalGraphConsumerEndpointRef &endpoint) {
  if (const auto *operand = std::get_if<ActorTokenOperandRef>(&endpoint)) {
    auto actor = view.resolve(operand->actor);
    if (!actor)
      return actor.takeError();
    if (operand->ordinal >= actor->op->getNumOperands())
      return invalid("actor consumer operand is out of range");
    return &actor->op->getOpOperand(operand->ordinal);
  }

  const GraphEgressTokenRef &egress = std::get<GraphEgressTokenRef>(endpoint);
  GraphRef graphRef = std::visit(
      [](const auto &token) -> GraphRef { return token.graph; }, egress);
  auto graphView = view.resolve(graphRef);
  if (!graphView)
    return graphView.takeError();
  auto graph = llvm::dyn_cast<GraphOp>(graphView->op);
  if (!graph || graph.isExternal())
    return invalid("graph egress does not belong to a graph definition");
  auto ret =
      llvm::dyn_cast<GraphReturnOp>(graph.getBody().front().getTerminator());
  if (!ret)
    return invalid("graph egress has no graph.return owner");

  const unsigned flatOrdinal = std::visit(
      [&](const auto &token) -> unsigned {
        using Token = std::decay_t<decltype(token)>;
        if constexpr (std::is_same_v<Token, GraphValueOutputTokenRef>)
          return token.ordinal;
        if constexpr (std::is_same_v<Token, GraphStreamOutputTokenRef>)
          return static_cast<unsigned>(ret.getValues().size()) + token.ordinal;
        return static_cast<unsigned>(ret.getValues().size() +
                                     ret.getStreams().size() +
                                     ret.getMemories().size()) +
               token.ordinal;
      },
      egress);
  if (flatOrdinal >= ret->getNumOperands())
    return invalid("graph egress operand is out of range");
  return &ret->getOpOperand(flatOrdinal);
}

llvm::Expected<std::vector<mlir::OpOperand *>>
completeSinkOperands(const CanonicalDataflowProgramView &view,
                     const Replica &source) {
  CanonicalGraphProducerEndpointRef producer =
      ActorTokenResultRef{source.actor.ref, 0};
  auto consumers = view.graphConsumers(producer);
  if (!consumers)
    return consumers.takeError();
  std::vector<mlir::OpOperand *> sinks;
  sinks.reserve(consumers->size());
  for (const CanonicalGraphConsumerEndpointRef &consumer : *consumers) {
    auto operand = resolveConsumerOperand(view, consumer);
    if (!operand)
      return operand.takeError();
    if ((*operand)->get() != source.actor.op->getResult(0))
      return invalid("canonical sink does not consume the selected result");
    sinks.push_back(*operand);
  }
  if (sinks.size() != static_cast<std::size_t>(std::distance(
                          source.actor.op->getResult(0).use_begin(),
                          source.actor.op->getResult(0).use_end())))
    return invalid("canonical sink set does not cover every result use");
  return sinks;
}

std::vector<ActorId> completeReplicaGroup(const Replica &source,
                                          llvm::ArrayRef<Replica> replicas) {
  std::vector<ActorId> ids;
  for (const Replica &candidate : replicas)
    if (candidate.actor.op->getResult(0).hasOneUse() &&
        haveSameReplicaInputs(source, candidate))
      ids.push_back(candidate.actor.ref.entity);
  llvm::sort(
      ids, [](ActorId lhs, ActorId rhs) { return lhs.value() < rhs.value(); });
  return ids;
}

const Replica *findReplica(llvm::ArrayRef<Replica> replicas, ActorId id) {
  auto found = llvm::find_if(replicas, [&](const Replica &replica) {
    return replica.actor.ref.entity == id;
  });
  return found == replicas.end() ? nullptr : &*found;
}

llvm::Expected<std::optional<CanonicalDataflowArtifact>>
finalizeChanged(const CanonicalDataflowArtifact &parent,
                mlir::ModuleOp candidate) {
  auto finalized = finalizeCanonicalDataflow(candidate);
  if (!finalized)
    return finalized.takeError();
  if (finalized->identity() == parent.identity())
    return std::optional<CanonicalDataflowArtifact>{};
  return std::optional<CanonicalDataflowArtifact>(std::move(*finalized));
}

} // namespace

llvm::Expected<std::vector<DataflowRewriteDecision>>
enumeratePureComputeFanoutDecisions(const CanonicalDataflowArtifact &parent) {
  auto view = parent.view();
  if (!view)
    return view.takeError();
  auto replicas = collectReplicas(*view);
  if (!replicas)
    return replicas.takeError();

  std::vector<DataflowRewriteDecision> decisions;
  for (const Replica &replica : *replicas) {
    auto sinks = completeSinkOperands(*view, replica);
    if (!sinks)
      return sinks.takeError();
    if (sinks->size() >= 2)
      decisions.emplace_back(
          PureComputeFanoutReplicateRewrite{replica.actor.ref.entity});
  }

  llvm::DenseSet<mlir::Operation *> grouped;
  for (const Replica &replica : *replicas) {
    if (!replica.actor.op->getResult(0).hasOneUse() ||
        !grouped.insert(replica.actor.op).second)
      continue;
    std::vector<ActorId> group = completeReplicaGroup(replica, *replicas);
    for (ActorId id : group) {
      const Replica *member = findReplica(*replicas, id);
      if (member)
        grouped.insert(member->actor.op);
    }
    if (group.size() >= 2)
      decisions.emplace_back(PureComputeFanoutFactorRewrite{std::move(group)});
  }
  llvm::sort(decisions, dataflowRewriteDecisionLess);
  return decisions;
}

llvm::Expected<std::optional<CanonicalDataflowArtifact>>
materializePureComputeFanoutRewrite(const CanonicalDataflowArtifact &parent,
                                    const DataflowRewriteDecision &decision) {
  auto view = parent.view();
  if (!view)
    return view.takeError();
  auto replicas = collectReplicas(*view);
  if (!replicas)
    return replicas.takeError();

  mlir::IRMapping mapping;
  mlir::OwningOpRef<mlir::ModuleOp> candidate(
      mlir::cast<mlir::ModuleOp>(parent.module()->clone(mapping)));

  if (const auto *replicate =
          std::get_if<PureComputeFanoutReplicateRewrite>(&decision)) {
    const Replica *source = findReplica(*replicas, replicate->compute);
    if (!source)
      return invalid("replication source is not an eligible Compute actor");
    auto sinks = completeSinkOperands(*view, *source);
    if (!sinks)
      return sinks.takeError();
    if (sinks->size() < 2)
      return invalid("replication source has fewer than two result sinks");
    mlir::Operation *clonedSource = mapping.lookupOrNull(source->actor.op);
    if (!clonedSource)
      return invalid("replication source was not cloned");
    mlir::OpBuilder builder(clonedSource);
    for (mlir::OpOperand *sink : *sinks) {
      mlir::Operation *clonedSinkOwner = mapping.lookupOrNull(sink->getOwner());
      if (!clonedSinkOwner)
        return invalid("replication sink was not cloned");
      mlir::Operation *clone = clonedSource->clone();
      clone->removeAttr(kEntityIdAttrName);
      builder.insert(clone);
      clonedSinkOwner->getOpOperand(sink->getOperandNumber())
          .set(clone->getResult(0));
    }
    clonedSource->erase();
    return finalizeChanged(parent, candidate.get());
  }

  const auto *factor = std::get_if<PureComputeFanoutFactorRewrite>(&decision);
  if (!factor)
    return invalid("decision is not a fanout variant");
  const Replica *source = findReplica(*replicas, factor->replicas.front());
  if (!source || !source->actor.op->getResult(0).hasOneUse())
    return invalid("factor source is not an eligible one-sink replica");
  std::vector<ActorId> complete = completeReplicaGroup(*source, *replicas);
  if (complete != factor->replicas)
    return invalid("factor decision does not name the complete replica group");

  mlir::Operation *clonedSource = mapping.lookupOrNull(source->actor.op);
  if (!clonedSource)
    return invalid("factor source was not cloned");
  for (ActorId id : llvm::drop_begin(factor->replicas)) {
    const Replica *replica = findReplica(*replicas, id);
    if (!replica)
      return invalid("factor member is not an eligible replica");
    mlir::Operation *clonedReplica = mapping.lookupOrNull(replica->actor.op);
    if (!clonedReplica)
      return invalid("factor member was not cloned");
    clonedReplica->getResult(0).replaceAllUsesWith(clonedSource->getResult(0));
    clonedReplica->erase();
  }
  return finalizeChanged(parent, candidate.get());
}

} // namespace dataflow::detail
