#include "SpatialRecurrenceTimingInternal.h"
#include "SpatialRecurrenceTimingPersistent.h"

#include "Dataflow/IR/DataflowActorSemantics.h"
#include "Dataflow/IR/DataflowServiceSchema.h"
#include "Fabric/IR/MemoryServiceContract.h"
#include "Fabric/IR/OperationResourceContract.h"
#include "Fabric/Identity/FabricRefBytes.h"
#include "Mapping/Artifact/MappingArtifact.h"
#include "PnR/RouteTreeState.h"
#include "PnR/SpatialCandidateState.h"
#include "PnR/SpatialPnrProblem.h"
#include "SpatialPhysicalTiming.h"
#include "StaticSchedulePressure.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <optional>
#include <queue>
#include <set>
#include <string>
#include <system_error>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

using namespace loom;
using namespace loom::fabric;
using namespace loom::mapping;
using namespace loom::pnr;
using namespace loom::pnr::detail;

namespace {

using ActorKey = std::uint64_t;
using EdgeKey = std::tuple<ActorKey, std::uint64_t, ActorKey, std::uint64_t>;

ActorKey actorKey(::dataflow::ActorRef actor) { return actor.entity.value(); }

EdgeKey edgeKey(const ::dataflow::ActorTokenResultRef &producer,
                const ::dataflow::ActorTokenOperandRef &consumer) {
  return {actorKey(producer.actor), producer.ordinal, actorKey(consumer.actor),
          consumer.ordinal};
}

llvm::Error freezeInvalid(const llvm::Twine &message) {
  return llvm::make_error<SpatialPnrFreezeFailure>(
      SpatialPnrFreezeFailureKind::Invalid,
      ("recurrence timing: " + message).str());
}

llvm::Error projectionInvalid(const llvm::Twine &message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      "spatial_recurrence_timing_invalid: " + message);
}

llvm::Expected<PnrIndex> checkedIndex(std::size_t value,
                                      llvm::StringRef subject) {
  if (value > getPnrIndexMax())
    return freezeInvalid(subject + " exceeds PnrIndex");
  return static_cast<PnrIndex>(value);
}

llvm::Expected<std::uint64_t> checkedAdd(std::uint64_t lhs, std::uint64_t rhs,
                                         llvm::StringRef subject) {
  if (rhs > std::numeric_limits<std::uint64_t>::max() - lhs)
    return projectionInvalid(subject + " exceeds u64");
  return lhs + rhs;
}

struct ActorOwner final {
  FrozenRecurrenceActorOwnerKind kind = FrozenRecurrenceActorOwnerKind::Compute;
  PnrIndex realization = 0;
  PnrIndex ownerActor = 0;
};

llvm::Expected<std::vector<std::optional<std::uint32_t>>>
resultPublicationLatencies(
    const ::dataflow::CanonicalActorView &actor,
    const ::fabric::ResourceContract &contract,
    std::optional<std::uint32_t> &carryNextStateLatency) {
  const ::dataflow::OperationSchemaId schema =
      ::dataflow::requireOperationSchema(actor.op);
  auto cases = ::dataflow::semantics::projectActorHandshakeCases(
      schema, actor.op->getNumOperands(), actor.op->getNumResults());
  if (!cases)
    return cases.takeError();

  std::vector<bool> produced(actor.op->getNumResults(), false);
  std::vector<bool> proven(actor.op->getNumResults(), true);
  std::vector<std::uint32_t> maximum(actor.op->getNumResults(), 0);
  for (const ::dataflow::semantics::ActorHandshakeCase &transition : *cases) {
    auto timing = ::fabric::projectOperationTransitionArchitecturalTiming(
        contract, schema, transition.ordinal);
    if (!timing)
      return timing.takeError();
    for (std::uint32_t result : transition.activeResults) {
      if (result >= produced.size())
        return freezeInvalid("actor transition publishes a foreign result");
      produced[result] = true;
      if (!*timing) {
        proven[result] = false;
        continue;
      }
      maximum[result] =
          std::max(maximum[result], (*timing)->resultPublicationLatencyCycles);
    }
    if (schema == ::dataflow::OperationSchemaId::DataflowCarry &&
        transition.ordinal == static_cast<std::uint32_t>(
                                  ::dataflow::semantics::CarryCase::Next)) {
      if (*timing)
        carryNextStateLatency = (*timing)->nextStateLatencyCycles;
      else
        carryNextStateLatency.reset();
    }
  }

  std::vector<std::optional<std::uint32_t>> result(produced.size());
  for (std::size_t ordinal = 0; ordinal < result.size(); ++ordinal)
    if (produced[ordinal] && proven[ordinal])
      result[ordinal] = maximum[ordinal];
  return result;
}

const TechComputeActorView *
findComputeActor(const TechComputeRealizationView &realization,
                 ::dataflow::ActorRef actor) {
  const TechComputeActorView *result = nullptr;
  for (const TechComputeActorView &candidate : realization.actors) {
    if (candidate.actor != actor)
      continue;
    if (result)
      return nullptr;
    result = &candidate;
  }
  return result;
}

struct ProjectionValue final {
  std::optional<SpatialRecurrenceTimingEdgeWitness> witness;
  std::string missingProof;
};

SpatialRecurrenceTimingProjection proofNotEstablished(llvm::StringRef reason) {
  SpatialRecurrenceTimingProjection result;
  result.kind = SpatialRecurrenceTimingProofKind::ProofNotEstablished;
  result.diagnostic = reason.str();
  return result;
}

std::string missingPublicationTiming(const FrozenRecurrenceActor &actor,
                                     std::uint64_t resultOrdinal) {
  const llvm::StringRef owner =
      actor.ownerKind == FrozenRecurrenceActorOwnerKind::Compute ? "compute"
                                                                 : "memory";
  return (llvm::Twine("actor_publication_timing_not_established:actor=") +
          llvm::Twine(actor.actor.entity.value()) +
          ":result=" + llvm::Twine(resultOrdinal) + ":owner=" + owner)
      .str();
}

llvm::Expected<std::optional<std::uint64_t>> computePublicationLatency(
    const SpatialCandidateState &candidate, PnrIndex recurrenceActor,
    const FrozenRecurrenceActor &actor, std::uint64_t resultOrdinal) {
  const auto &index = candidate.problem().recurrenceTiming();
  if (actor.ownerKind != FrozenRecurrenceActorOwnerKind::Compute)
    return projectionInvalid("compute publication query names memory");
  const PnrIndex placement = candidate.computeBinding(actor.owner).placement;
  const auto offsets = index.computeTimingOffsets();
  if (recurrenceActor + 1 >= offsets.size() ||
      offsets[recurrenceActor] > offsets[recurrenceActor + 1] ||
      offsets[recurrenceActor + 1] > index.computeTimings().size())
    return projectionInvalid("compute timing CSR is malformed");
  for (const FrozenComputeActorArchitecturalTiming &timing :
       index.computeTimings().slice(offsets[recurrenceActor],
                                    offsets[recurrenceActor + 1] -
                                        offsets[recurrenceActor])) {
    if (timing.placement != placement)
      continue;
    if (resultOrdinal >= timing.resultCount ||
        timing.resultOffset >
            index.computeResultPublicationLatencies().size() ||
        timing.resultCount > index.computeResultPublicationLatencies().size() -
                                 timing.resultOffset)
      return projectionInvalid("compute result timing is out of range");
    const auto projected =
        index.computeResultPublicationLatencies()[timing.resultOffset +
                                                  resultOrdinal];
    return projected ? std::optional<std::uint64_t>{*projected}
                     : std::optional<std::uint64_t>{};
  }
  return projectionInvalid("selected compute placement has no timing row");
}

llvm::Expected<std::optional<std::uint64_t>>
memoryPublicationLatency(const SpatialCandidateState &candidate,
                         const FrozenRecurrenceActor &actor) {
  if (actor.ownerKind != FrozenRecurrenceActorOwnerKind::Memory)
    return projectionInvalid("memory publication query names compute");
  const auto &problem = candidate.problem();
  const PnrIndex planOrdinal = candidate.memoryOperationPlan(actor.ownerActor);
  const auto plans = problem.handshake().memoryOperationPlans();
  if (planOrdinal >= plans.size())
    return projectionInvalid("memory actor selects a foreign operation plan");
  if (!plans[planOrdinal].issueLatencyCycles)
    return std::optional<std::uint64_t>{};

  const auto offsets = problem.memory().actorUseOffsets();
  if (actor.ownerActor + 1 >= offsets.size() ||
      offsets[actor.ownerActor] > offsets[actor.ownerActor + 1] ||
      offsets[actor.ownerActor + 1] > problem.memory().rootedUses().size())
    return projectionInvalid("memory actor use CSR is malformed");
  std::uint64_t maximumCompletion = 0;
  for (PnrIndex use = offsets[actor.ownerActor];
       use < offsets[actor.ownerActor + 1]; ++use) {
    const PnrIndex optionOrdinal = candidate.memoryUseDispatch(use);
    if (optionOrdinal >= problem.memory().dispatchOptions().size())
      return projectionInvalid("memory use selects a foreign dispatch option");
    const FrozenSpatialMemoryDispatchOption &option =
        problem.memory().dispatchOptions()[optionOrdinal];
    if (!option.maxIssueToRetireCycles)
      return std::optional<std::uint64_t>{};
    maximumCompletion =
        std::max(maximumCompletion, *option.maxIssueToRetireCycles);
  }
  auto latency = checkedAdd(*plans[planOrdinal].issueLatencyCycles,
                            maximumCompletion, "memory publication latency");
  if (!latency)
    return latency.takeError();
  return std::optional<std::uint64_t>{*latency};
}

llvm::Expected<std::optional<std::uint64_t>>
carryNextStateLatency(const SpatialCandidateState &candidate,
                      PnrIndex recurrenceActor,
                      const FrozenRecurrenceActor &actor) {
  if (actor.ownerKind != FrozenRecurrenceActorOwnerKind::Compute)
    return std::optional<std::uint64_t>{};
  const auto &index = candidate.problem().recurrenceTiming();
  const PnrIndex placement = candidate.computeBinding(actor.owner).placement;
  const auto offsets = index.computeTimingOffsets();
  if (recurrenceActor + 1 >= offsets.size())
    return projectionInvalid("carry timing CSR is malformed");
  for (const FrozenComputeActorArchitecturalTiming &timing :
       index.computeTimings().slice(offsets[recurrenceActor],
                                    offsets[recurrenceActor + 1] -
                                        offsets[recurrenceActor]))
    if (timing.placement == placement) {
      if (!timing.carryNextStateLatencyCycles)
        return std::optional<std::uint64_t>{};
      return static_cast<std::uint64_t>(*timing.carryNextStateLatencyCycles);
    }
  return projectionInvalid("selected carry placement has no timing row");
}

llvm::Expected<std::uint64_t>
traversalLatency(const FrozenSpatialPnrProblem &problem, PnrIndex traversal) {
  if (traversal >= problem.routing().traversals().size())
    return projectionInvalid("recurrence path names a foreign traversal");
  return problem.routing().traversals()[traversal].architecturalLatencyCycles;
}

llvm::Expected<std::optional<std::uint64_t>>
residualTransportLatency(const SpatialCandidateState &candidate,
                         const FrozenRecurrenceEdge &edge,
                         llvm::ArrayRef<const RouteTreeState *> routeTrees,
                         SpatialRecurrenceEdgeDisposition &disposition) {
  const FrozenSpatialPnrProblem &problem = candidate.problem();
  if (edge.logicalNet >= problem.transfers().logicalNets().size() ||
      edge.logicalNet >= routeTrees.size())
    return projectionInvalid("recurrence residual edge is out of range");
  const PnrIndex local = candidate.registerFifoTransfer(edge.logicalNet);
  if (local != getInvalidPnrIndex()) {
    disposition = SpatialRecurrenceEdgeDisposition::RegisterFifo;
    if (local >= problem.localTransfers().options().size())
      return projectionInvalid("recurrence RegFIFO selection is out of range");
    const auto &option = problem.localTransfers().options()[local];
    auto write = traversalLatency(problem, option.writeTraversal);
    auto read = traversalLatency(problem, option.readTraversal);
    if (!write)
      return write.takeError();
    if (!read)
      return read.takeError();
    return checkedAdd(*write, *read, "RegFIFO architectural latency");
  }

  disposition = SpatialRecurrenceEdgeDisposition::ExternalRouteTree;
  const RouteTreeState *route = routeTrees[edge.logicalNet];
  if (!route || &route->routingGraph() != &problem.routing())
    return projectionInvalid("recurrence RouteTree has a foreign owner");
  const auto sinkSlot = route->sinkNode(edge.sink);
  if (!sinkSlot)
    return std::optional<std::uint64_t>{};

  std::uint64_t total = 0;
  const auto sourceBindings = problem.transfers().logicalNetSourceBindings();
  if (edge.logicalNet >= sourceBindings.size())
    return projectionInvalid("recurrence source binding is absent");
  auto sourceLocal = projectSelectedSpatialTerminalTraversal(
      problem, sourceBindings[edge.logicalNet],
      candidate.portAttachmentSelections(),
      candidate.graphBoundaryAttachmentSelections());
  if (!sourceLocal)
    return sourceLocal.takeError();
  if (*sourceLocal) {
    auto latency = traversalLatency(problem, **sourceLocal);
    if (!latency)
      return latency.takeError();
    total = *latency;
  }

  const auto &routing = problem.routing();
  const auto arcs = routing.routingArcs();
  PnrIndex slot = *sinkSlot;
  for (std::size_t depth = 0;; ++depth) {
    if (depth > route->nodeStorage().size())
      return projectionInvalid("recurrence RouteTree contains a cycle");
    const RouteTreeNode &node = route->node(slot);
    if (node.parentArc == getInvalidPnrIndex())
      break;
    if (node.parentArc >= arcs.size())
      return projectionInvalid("recurrence RouteTree arc is out of range");
    auto latency = traversalLatency(problem, arcs[node.parentArc].traversal);
    if (!latency)
      return latency.takeError();
    auto next = checkedAdd(total, *latency, "route architectural latency");
    if (!next)
      return next.takeError();
    total = *next;
    const auto parent = route->parentNodeSlot(slot);
    if (!parent)
      return projectionInvalid("recurrence RouteTree parent is absent");
    slot = *parent;
  }

  const FrozenSpatialLogicalNet &net =
      problem.transfers().logicalNets()[edge.logicalNet];
  const auto sinkBindings = problem.transfers().logicalNetSinkBindings();
  if (edge.sink >= net.sinkCount ||
      net.sinkOffset + edge.sink >= sinkBindings.size())
    return projectionInvalid("recurrence sink binding is absent");
  auto sinkLocal = projectSelectedSpatialTerminalTraversal(
      problem, sinkBindings[net.sinkOffset + edge.sink],
      candidate.portAttachmentSelections(),
      candidate.graphBoundaryAttachmentSelections());
  if (!sinkLocal)
    return sinkLocal.takeError();
  if (*sinkLocal) {
    auto latency = traversalLatency(problem, **sinkLocal);
    if (!latency)
      return latency.takeError();
    auto next = checkedAdd(total, *latency, "route architectural latency");
    if (!next)
      return next.takeError();
    total = *next;
  }
  return total;
}

llvm::Expected<ProjectionValue>
projectEdge(const SpatialCandidateState &candidate,
            const FrozenRecurrenceEdge &edge,
            llvm::ArrayRef<const RouteTreeState *> routeTrees) {
  const auto actors = candidate.problem().recurrenceTiming().actors();
  if (edge.producerActor >= actors.size() ||
      edge.consumerActor >= actors.size())
    return projectionInvalid("recurrence edge has a foreign actor");
  const FrozenRecurrenceActor &producer = actors[edge.producerActor];
  const FrozenRecurrenceActor &consumer = actors[edge.consumerActor];

  llvm::Expected<std::optional<std::uint64_t>> publication =
      producer.ownerKind == FrozenRecurrenceActorOwnerKind::Compute
          ? computePublicationLatency(candidate, edge.producerActor, producer,
                                      edge.producer.ordinal)
          : memoryPublicationLatency(candidate, producer);
  if (!publication)
    return publication.takeError();
  if (!*publication)
    return ProjectionValue{std::nullopt, missingPublicationTiming(
                                             producer, edge.producer.ordinal)};

  SpatialRecurrenceEdgeDisposition disposition =
      SpatialRecurrenceEdgeDisposition::ComputeInternal;
  std::optional<std::uint64_t> transport = 0;
  switch (edge.disposition) {
  case FrozenRecurrenceEdgeDisposition::ComputeInternal:
    disposition = SpatialRecurrenceEdgeDisposition::ComputeInternal;
    break;
  case FrozenRecurrenceEdgeDisposition::MemoryInternal:
    disposition = SpatialRecurrenceEdgeDisposition::MemoryInternal;
    break;
  case FrozenRecurrenceEdgeDisposition::Residual: {
    auto projected =
        residualTransportLatency(candidate, edge, routeTrees, disposition);
    if (!projected)
      return projected.takeError();
    transport = *projected;
    break;
  }
  }
  if (!transport)
    return ProjectionValue{std::nullopt, "recurrence_route_not_established"};

  std::uint64_t nextState = 0;
  if (edge.feedback) {
    auto next = carryNextStateLatency(candidate, edge.consumerActor, consumer);
    if (!next)
      return next.takeError();
    if (!*next)
      return ProjectionValue{std::nullopt,
                             "carry_next_state_timing_not_established"};
    nextState = **next;
  }
  auto partial =
      checkedAdd(**publication, *transport, "recurrence edge latency");
  if (!partial)
    return partial.takeError();
  auto total = checkedAdd(*partial, nextState, "recurrence edge latency");
  if (!total)
    return total.takeError();
  return ProjectionValue{SpatialRecurrenceTimingEdgeWitness{
                             edge.producer, edge.consumer, disposition,
                             **publication, *transport, nextState, *total},
                         {}};
}

template <typename Index>
const FrozenRecurrenceGraph *findGraph(const Index &index,
                                       ::dataflow::GraphRef graph) {
  const auto found = llvm::find_if(index.graphs(),
                                   [&](const FrozenRecurrenceGraph &candidate) {
                                     return candidate.graph == graph;
                                   });
  return found == index.graphs().end() ? nullptr : &*found;
}

} // namespace

llvm::Expected<std::shared_ptr<const SpatialRecurrenceTimingIndex>>
SpatialRecurrenceTimingIndex::build(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const TechMappingView &techMapping, const FabricArtifactView &fabric,
    const FrozenSpatialRealizationIndex &realizations,
    const FrozenSpatialMemoryIndex &memory,
    const FrozenSpatialTransferIndex &transfers,
    const FrozenSpatialHandshakeIndex &handshake,
    const StaticScheduleAnalysis &schedule) {
  (void)memory;
  (void)handshake;
  auto result = std::make_shared<SpatialRecurrenceTimingIndex>();

  std::map<ActorKey, ActorOwner> owners;
  for (auto [realizationOrdinalValue, realization] :
       llvm::enumerate(realizations.computeRealizations())) {
    const PnrIndex realizationOrdinal =
        static_cast<PnrIndex>(realizationOrdinalValue);
    for (PnrIndex local = 0; local < realization.actorCount; ++local) {
      const PnrIndex actorOrdinal = realization.actorOffset + local;
      if (actorOrdinal >= realizations.computeActors().size() ||
          !owners
               .emplace(actorKey(realizations.computeActors()[actorOrdinal]),
                        ActorOwner{FrozenRecurrenceActorOwnerKind::Compute,
                                   realizationOrdinal, actorOrdinal})
               .second)
        return freezeInvalid("compute actor owner is inconsistent");
    }
  }
  for (auto [realizationOrdinalValue, realization] :
       llvm::enumerate(realizations.memoryRealizations())) {
    const PnrIndex realizationOrdinal =
        static_cast<PnrIndex>(realizationOrdinalValue);
    for (PnrIndex local = 0; local < realization.actorCount; ++local) {
      const PnrIndex actorOrdinal = realization.actorOffset + local;
      if (actorOrdinal >= realizations.memoryActors().size() ||
          !owners
               .emplace(
                   actorKey(realizations.memoryActors()[actorOrdinal].actor),
                   ActorOwner{FrozenRecurrenceActorOwnerKind::Memory,
                              realizationOrdinal, actorOrdinal})
               .second)
        return freezeInvalid("memory actor owner is inconsistent");
    }
  }

  std::map<ActorKey, PnrIndex> actorOrdinals;
  result->actors_.reserve(schedule.actors().size());
  for (const StaticActorCriticality &actor : schedule.actors()) {
    const auto owner = owners.find(actorKey(actor.actor));
    if (owner == owners.end())
      return freezeInvalid("scheduled actor has no Tech realization owner");
    auto ordinal = checkedIndex(result->actors_.size(), "recurrence actors");
    if (!ordinal)
      return ordinal.takeError();
    actorOrdinals.emplace(actorKey(actor.actor), *ordinal);
    result->actors_.push_back({actor.actor, actor.graph, owner->second.kind,
                               owner->second.realization,
                               owner->second.ownerActor});
  }

  result->computeTimingOffsets_.reserve(result->actors_.size() + 1);
  result->computeTimingOffsets_.push_back(0);
  for (const FrozenRecurrenceActor &actor : result->actors_) {
    if (actor.ownerKind == FrozenRecurrenceActorOwnerKind::Compute) {
      if (actor.owner >= techMapping.computeRealizations().size() ||
          actor.owner >= realizations.computeRealizations().size())
        return freezeInvalid("compute timing owner is out of range");
      const TechComputeRealizationView &tech =
          techMapping.computeRealizations()[actor.owner];
      const FrozenSpatialComputeRealization &frozen =
          realizations.computeRealizations()[actor.owner];
      const TechComputeActorView *techActor =
          findComputeActor(tech, actor.actor);
      if (!techActor)
        return freezeInvalid("compute timing actor is absent");
      auto dataflowActor = dataflow.resolve(actor.actor);
      if (!dataflowActor)
        return dataflowActor.takeError();
      for (PnrIndex placement = frozen.placementOffset;
           placement < frozen.placementOffset + frozen.placementCount;
           ++placement) {
        if (placement >= realizations.computePlacements().size())
          return freezeInvalid("compute timing placement is out of range");
        auto occurrence = deriveFabricFuOccurrenceNode(
            fabric, techActor->fabricOperation,
            realizations.computePlacements()[placement].fu);
        if (!occurrence)
          return occurrence.takeError();
        const ResolvedFabricOpCapabilityView *capability =
            fabric.resolvedFabricOpCapability(*occurrence);
        if (!capability)
          return freezeInvalid("compute placement has no operation contract");
        std::optional<std::uint32_t> carryNextState;
        auto publications = resultPublicationLatencies(
            *dataflowActor, capability->resourceStateAndTimingContract,
            carryNextState);
        if (!publications)
          return publications.takeError();
        auto offset =
            checkedIndex(result->computeResultPublicationLatencies_.size(),
                         "compute result timing");
        if (!offset)
          return offset.takeError();
        auto count =
            checkedIndex(publications->size(), "compute result timing");
        if (!count)
          return count.takeError();
        result->computeResultPublicationLatencies_.insert(
            result->computeResultPublicationLatencies_.end(),
            publications->begin(), publications->end());
        result->computeTimings_.push_back(
            {placement, *offset, *count, carryNextState});
      }
    }
    auto end =
        checkedIndex(result->computeTimings_.size(), "compute timing rows");
    if (!end)
      return end.takeError();
    result->computeTimingOffsets_.push_back(*end);
  }

  std::map<EdgeKey, std::pair<PnrIndex, PnrIndex>> residualEdges;
  for (auto [netOrdinalValue, net] : llvm::enumerate(transfers.logicalNets())) {
    const auto *producer =
        std::get_if<::dataflow::ActorTokenResultRef>(&net.producer);
    if (!producer)
      continue;
    for (PnrIndex sink = 0; sink < net.sinkCount; ++sink) {
      const auto *consumer = std::get_if<::dataflow::ActorTokenOperandRef>(
          &transfers.logicalNetSinks()[net.sinkOffset + sink]);
      if (!consumer)
        continue;
      if (!residualEdges
               .emplace(edgeKey(*producer, *consumer),
                        std::pair{static_cast<PnrIndex>(netOrdinalValue), sink})
               .second)
        return freezeInvalid("residual edge disposition is duplicated");
    }
  }
  std::set<EdgeKey> memoryInternalEdges;
  for (const TechMemoryRealizationView &realization :
       techMapping.memoryRealizations())
    for (const TechMemoryInternalEdgeView &edge : realization.internalEdges) {
      const auto *producer =
          std::get_if<::dataflow::ActorTokenResultRef>(&edge.producer);
      const auto *consumer =
          std::get_if<::dataflow::ActorTokenOperandRef>(&edge.consumer);
      if (producer && consumer &&
          !memoryInternalEdges.emplace(edgeKey(*producer, *consumer)).second)
        return freezeInvalid("memory internal edge is duplicated");
    }
  std::map<EdgeKey, std::uint64_t> feedbackDistance;
  for (const StaticRecurrenceFeedback &feedback : schedule.feedbacks())
    if (!feedbackDistance
             .emplace(edgeKey(feedback.producer, feedback.consumer),
                      feedback.dependenceDistance)
             .second)
      return freezeInvalid("recurrence feedback edge is duplicated");

  result->edges_.reserve(schedule.edges().size());
  for (const StaticActorEdgeCriticality &edge : schedule.edges()) {
    const auto producer = actorOrdinals.find(actorKey(edge.producer.actor));
    const auto consumer = actorOrdinals.find(actorKey(edge.consumer.actor));
    if (producer == actorOrdinals.end() || consumer == actorOrdinals.end())
      return freezeInvalid("recurrence edge has an unmapped actor");
    FrozenRecurrenceEdgeDisposition disposition;
    PnrIndex logicalNet = getInvalidPnrIndex();
    PnrIndex sink = getInvalidPnrIndex();
    const auto residual =
        residualEdges.find(edgeKey(edge.producer, edge.consumer));
    if (residual != residualEdges.end()) {
      disposition = FrozenRecurrenceEdgeDisposition::Residual;
      logicalNet = residual->second.first;
      sink = residual->second.second;
    } else if (memoryInternalEdges.count(
                   edgeKey(edge.producer, edge.consumer))) {
      disposition = FrozenRecurrenceEdgeDisposition::MemoryInternal;
    } else {
      const FrozenRecurrenceActor &source = result->actors_[producer->second];
      const FrozenRecurrenceActor &target = result->actors_[consumer->second];
      if (source.ownerKind != FrozenRecurrenceActorOwnerKind::Compute ||
          target.ownerKind != FrozenRecurrenceActorOwnerKind::Compute ||
          source.owner != target.owner)
        return freezeInvalid("actor edge has no physical disposition");
      disposition = FrozenRecurrenceEdgeDisposition::ComputeInternal;
    }
    const auto feedback =
        feedbackDistance.find(edgeKey(edge.producer, edge.consumer));
    auto ordinal = checkedIndex(result->edges_.size(), "recurrence edges");
    if (!ordinal)
      return ordinal.takeError();
    result->edges_.push_back(
        {edge.producer, edge.consumer, edge.graph, producer->second,
         consumer->second, disposition, logicalNet, sink,
         edge.initializedFeedback, feedback != feedbackDistance.end(),
         feedback == feedbackDistance.end() ? 0 : feedback->second});
    if (feedback != feedbackDistance.end())
      result->feedbackEdges_.push_back(*ordinal);
  }

  for (const StaticGraphRecurrenceTopology &topology :
       schedule.recurrenceTopologies()) {
    std::vector<PnrIndex> graphActors;
    std::vector<PnrIndex> graphEdges;
    for (auto [ordinal, actor] : llvm::enumerate(result->actors_))
      if (actor.graph == topology.graph)
        graphActors.push_back(static_cast<PnrIndex>(ordinal));
    for (auto [ordinal, edge] : llvm::enumerate(result->edges_))
      if (edge.graph == topology.graph)
        graphEdges.push_back(static_cast<PnrIndex>(ordinal));
    auto actorOffset =
        checkedIndex(result->graphActors_.size(), "recurrence graph actors");
    auto actorCount =
        checkedIndex(graphActors.size(), "recurrence graph actors");
    auto edgeOffset =
        checkedIndex(result->graphEdges_.size(), "recurrence graph edges");
    auto edgeCount = checkedIndex(graphEdges.size(), "recurrence graph edges");
    auto topologicalOffset = checkedIndex(
        result->graphTopologicalActors_.size(), "recurrence topology");
    if (!actorOffset)
      return actorOffset.takeError();
    if (!actorCount)
      return actorCount.takeError();
    if (!edgeOffset)
      return edgeOffset.takeError();
    if (!edgeCount)
      return edgeCount.takeError();
    if (!topologicalOffset)
      return topologicalOffset.takeError();
    result->graphActors_.insert(result->graphActors_.end(), graphActors.begin(),
                                graphActors.end());
    result->graphEdges_.insert(result->graphEdges_.end(), graphEdges.begin(),
                               graphEdges.end());

    std::vector<PnrIndex> topological;
    if (topology.postInitializationAcyclic) {
      std::vector<PnrIndex> indegree(result->actors_.size(), 0);
      std::vector<std::vector<PnrIndex>> successors(result->actors_.size());
      for (PnrIndex edgeOrdinal : graphEdges) {
        const FrozenRecurrenceEdge &edge = result->edges_[edgeOrdinal];
        if (edge.initializedFeedback)
          continue;
        successors[edge.producerActor].push_back(edge.consumerActor);
        ++indegree[edge.consumerActor];
      }
      std::priority_queue<PnrIndex, std::vector<PnrIndex>,
                          std::greater<PnrIndex>>
          ready;
      for (PnrIndex actor : graphActors)
        if (indegree[actor] == 0)
          ready.push(actor);
      while (!ready.empty()) {
        const PnrIndex actor = ready.top();
        ready.pop();
        topological.push_back(actor);
        for (PnrIndex successor : successors[actor])
          if (--indegree[successor] == 0)
            ready.push(successor);
      }
      if (topological.size() != graphActors.size())
        return freezeInvalid(
            "static recurrence topology disagrees with its DAG");
    }
    result->graphTopologicalActors_.insert(
        result->graphTopologicalActors_.end(), topological.begin(),
        topological.end());
    auto topologicalCount =
        checkedIndex(topological.size(), "recurrence topology");
    if (!topologicalCount)
      return topologicalCount.takeError();
    result->graphs_.push_back({topology.graph,
                               topology.postInitializationAcyclic, *actorOffset,
                               *actorCount, *edgeOffset, *edgeCount,
                               *topologicalOffset, *topologicalCount});
  }
  return std::shared_ptr<const SpatialRecurrenceTimingIndex>(std::move(result));
}

namespace {

template <typename Index, typename EdgeProjector>
llvm::Expected<SpatialRecurrenceTimingProjection>
projectRecurrenceCycles(const Index &index, EdgeProjector projectEdgeOrdinal) {
  SpatialRecurrenceTimingProjection result;
  if (index.feedbackEdges().empty())
    return result;
  std::vector<std::optional<SpatialRecurrenceTimingEdgeWitness>> edgeCache(
      index.edges().size());
  std::vector<bool> edgeProjected(index.edges().size(), false);
  std::vector<std::string> missingProof(index.edges().size());
  const auto getEdge = [&](PnrIndex ordinal)
      -> llvm::Expected<const SpatialRecurrenceTimingEdgeWitness *> {
    if (ordinal >= index.edges().size())
      return projectionInvalid("recurrence witness edge is out of range");
    if (!edgeProjected[ordinal]) {
      auto projected = projectEdgeOrdinal(ordinal);
      if (!projected)
        return projected.takeError();
      edgeProjected[ordinal] = true;
      edgeCache[ordinal] = std::move(projected->witness);
      missingProof[ordinal] = std::move(projected->missingProof);
    }
    return edgeCache[ordinal]
               ? &*edgeCache[ordinal]
               : static_cast<const SpatialRecurrenceTimingEdgeWitness *>(
                     nullptr);
  };

  for (PnrIndex feedbackOrdinal : index.feedbackEdges()) {
    if (feedbackOrdinal >= index.edges().size())
      return projectionInvalid("feedback edge is out of range");
    const FrozenRecurrenceEdge &feedback = index.edges()[feedbackOrdinal];
    const FrozenRecurrenceGraph *graph = findGraph(index, feedback.graph);
    if (!graph)
      return projectionInvalid("feedback edge has no graph topology");
    if (!graph->postInitializationAcyclic)
      return proofNotEstablished(
          "post_initialization_cycle_timing_not_established");
    const auto graphEdges =
        index.graphEdges().slice(graph->edgeOffset, graph->edgeCount);
    const auto topological = index.graphTopologicalActors().slice(
        graph->topologicalOffset, graph->topologicalCount);
    std::vector<std::vector<PnrIndex>> outgoing(index.actors().size());
    for (PnrIndex edgeOrdinal : graphEdges) {
      const FrozenRecurrenceEdge &edge = index.edges()[edgeOrdinal];
      if (edge.initializedFeedback)
        continue;
      outgoing[edge.producerActor].push_back(edgeOrdinal);
    }

    std::vector<bool> reachesTarget(index.actors().size(), false);
    reachesTarget[feedback.producerActor] = true;
    for (PnrIndex actor : llvm::reverse(topological))
      for (PnrIndex edgeOrdinal : outgoing[actor])
        if (reachesTarget[index.edges()[edgeOrdinal].consumerActor])
          reachesTarget[actor] = true;
    if (!reachesTarget[feedback.consumerActor])
      continue;

    std::vector<std::optional<std::uint64_t>> distance(index.actors().size());
    std::vector<std::vector<PnrIndex>> paths(index.actors().size());
    distance[feedback.consumerActor] = 0;
    for (PnrIndex actor : topological) {
      if (!distance[actor])
        continue;
      for (PnrIndex edgeOrdinal : outgoing[actor]) {
        const FrozenRecurrenceEdge &edge = index.edges()[edgeOrdinal];
        if (!reachesTarget[edge.consumerActor])
          continue;
        auto witness = getEdge(edgeOrdinal);
        if (!witness)
          return witness.takeError();
        if (!*witness)
          return proofNotEstablished(missingProof[edgeOrdinal]);
        auto candidateDistance =
            checkedAdd(*distance[actor], (*witness)->totalLatencyCycles,
                       "recurrence path latency");
        if (!candidateDistance)
          return candidateDistance.takeError();
        std::vector<PnrIndex> candidatePath = paths[actor];
        candidatePath.push_back(edgeOrdinal);
        if (!distance[edge.consumerActor] ||
            *candidateDistance > *distance[edge.consumerActor] ||
            (*candidateDistance == *distance[edge.consumerActor] &&
             candidatePath < paths[edge.consumerActor])) {
          distance[edge.consumerActor] = *candidateDistance;
          paths[edge.consumerActor] = std::move(candidatePath);
        }
      }
    }
    if (!distance[feedback.producerActor])
      return projectionInvalid("recurrence DAG reachability is inconsistent");
    auto feedbackWitness = getEdge(feedbackOrdinal);
    if (!feedbackWitness)
      return feedbackWitness.takeError();
    if (!*feedbackWitness)
      return proofNotEstablished(missingProof[feedbackOrdinal]);
    auto latency = checkedAdd(*distance[feedback.producerActor],
                              (*feedbackWitness)->totalLatencyCycles,
                              "recurrence cycle latency");
    if (!latency)
      return latency.takeError();
    if (feedback.dependenceDistance == 0)
      return projectionInvalid("recurrence dependence distance is zero");
    const std::uint64_t recurrenceMinimumInitiationInterval = std::max(
        std::uint64_t{1}, *latency / feedback.dependenceDistance +
                              (*latency % feedback.dependenceDistance != 0));

    SpatialRecurrenceTimingWitness witness{feedback.graph,
                                           feedback.producer,
                                           feedback.consumer,
                                           feedback.dependenceDistance,
                                           *latency,
                                           recurrenceMinimumInitiationInterval,
                                           {}};
    for (PnrIndex edgeOrdinal : paths[feedback.producerActor])
      witness.edges.push_back(*edgeCache[edgeOrdinal]);
    witness.edges.push_back(**feedbackWitness);
    result.witnesses.push_back(std::move(witness));
    result.recurrenceMinimumInitiationIntervalCycles =
        std::max(result.recurrenceMinimumInitiationIntervalCycles,
                 recurrenceMinimumInitiationInterval);
  }
  return result;
}

} // namespace

llvm::Expected<SpatialRecurrenceTimingProjection>
loom::pnr::detail::projectSpatialRecurrenceTiming(
    const SpatialCandidateState &candidate,
    llvm::ArrayRef<const RouteTreeState *> routeTrees) {
  const SpatialRecurrenceTimingIndex &index =
      candidate.problem().recurrenceTiming();
  if (routeTrees.size() != candidate.problem().transfers().logicalNets().size())
    return projectionInvalid("recurrence RouteTree domain is incomplete");
  return projectRecurrenceCycles(index, [&](PnrIndex ordinal) {
    return projectEdge(candidate, index.edges()[ordinal], routeTrees);
  });
}

namespace {

struct PersistentRecurrenceIndex final {
  std::vector<FrozenRecurrenceActor> actorRecords;
  std::vector<FrozenRecurrenceEdge> edgeRecords;
  std::vector<FrozenRecurrenceGraph> graphRecords;
  std::vector<PnrIndex> graphActorOrdinals;
  std::vector<PnrIndex> graphEdgeOrdinals;
  std::vector<PnrIndex> topologicalActorOrdinals;
  std::vector<PnrIndex> feedbackEdgeOrdinals;

  llvm::ArrayRef<FrozenRecurrenceActor> actors() const { return actorRecords; }
  llvm::ArrayRef<FrozenRecurrenceEdge> edges() const { return edgeRecords; }
  llvm::ArrayRef<FrozenRecurrenceGraph> graphs() const { return graphRecords; }
  llvm::ArrayRef<PnrIndex> graphActors() const { return graphActorOrdinals; }
  llvm::ArrayRef<PnrIndex> graphEdges() const { return graphEdgeOrdinals; }
  llvm::ArrayRef<PnrIndex> graphTopologicalActors() const {
    return topologicalActorOrdinals;
  }
  llvm::ArrayRef<PnrIndex> feedbackEdges() const {
    return feedbackEdgeOrdinals;
  }
};

llvm::Expected<PersistentRecurrenceIndex> buildPersistentRecurrenceIndex(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const TechMappingView &techMapping,
    llvm::ArrayRef<::dataflow::GraphRef> covers) {
  auto schedule = deriveStaticScheduleAnalysis(dataflow, covers);
  if (!schedule)
    return schedule.takeError();

  PersistentRecurrenceIndex result;
  std::map<ActorKey, ActorOwner> owners;
  PnrIndex computeActorOrdinal = 0;
  for (auto [realizationOrdinalValue, realization] :
       llvm::enumerate(techMapping.computeRealizations())) {
    auto realizationOrdinal =
        checkedIndex(realizationOrdinalValue, "persistent compute owner");
    if (!realizationOrdinal)
      return realizationOrdinal.takeError();
    for (const TechComputeActorView &actor : realization.actors) {
      if (!owners
               .emplace(actorKey(actor.actor),
                        ActorOwner{FrozenRecurrenceActorOwnerKind::Compute,
                                   *realizationOrdinal, computeActorOrdinal})
               .second)
        return projectionInvalid("persistent compute actor owner is repeated");
      if (computeActorOrdinal == getPnrIndexMax())
        return projectionInvalid("persistent compute actor domain overflows");
      ++computeActorOrdinal;
    }
  }
  PnrIndex memoryActorOrdinal = 0;
  for (auto [realizationOrdinalValue, realization] :
       llvm::enumerate(techMapping.memoryRealizations())) {
    auto realizationOrdinal =
        checkedIndex(realizationOrdinalValue, "persistent memory owner");
    if (!realizationOrdinal)
      return realizationOrdinal.takeError();
    for (const TechMemoryActorView &actor : realization.actors) {
      if (!owners
               .emplace(actorKey(actor.actor),
                        ActorOwner{FrozenRecurrenceActorOwnerKind::Memory,
                                   *realizationOrdinal, memoryActorOrdinal})
               .second)
        return projectionInvalid("persistent memory actor owner is repeated");
      if (memoryActorOrdinal == getPnrIndexMax())
        return projectionInvalid("persistent memory actor domain overflows");
      ++memoryActorOrdinal;
    }
  }

  std::map<ActorKey, PnrIndex> actorOrdinals;
  result.actorRecords.reserve(schedule->actors().size());
  for (const StaticActorCriticality &actor : schedule->actors()) {
    const auto owner = owners.find(actorKey(actor.actor));
    if (owner == owners.end())
      return projectionInvalid("persistent recurrence actor has no owner");
    auto ordinal = checkedIndex(result.actorRecords.size(),
                                "persistent recurrence actors");
    if (!ordinal)
      return ordinal.takeError();
    if (!actorOrdinals.emplace(actorKey(actor.actor), *ordinal).second)
      return projectionInvalid("persistent recurrence actor is repeated");
    result.actorRecords.push_back({actor.actor, actor.graph, owner->second.kind,
                                   owner->second.realization,
                                   owner->second.ownerActor});
  }

  std::set<EdgeKey> residualEdges;
  for (const TechResidualLogicalNetView &net :
       techMapping.residualLogicalNets()) {
    const auto *producer =
        std::get_if<::dataflow::ActorTokenResultRef>(&net.producer);
    if (!producer)
      continue;
    for (const auto &sink : net.sinks) {
      const auto *consumer =
          std::get_if<::dataflow::ActorTokenOperandRef>(&sink);
      if (consumer &&
          !residualEdges.emplace(edgeKey(*producer, *consumer)).second)
        return projectionInvalid("persistent residual edge is repeated");
    }
  }
  std::set<EdgeKey> memoryInternalEdges;
  for (const TechMemoryRealizationView &realization :
       techMapping.memoryRealizations())
    for (const TechMemoryInternalEdgeView &edge : realization.internalEdges) {
      const auto *producer =
          std::get_if<::dataflow::ActorTokenResultRef>(&edge.producer);
      const auto *consumer =
          std::get_if<::dataflow::ActorTokenOperandRef>(&edge.consumer);
      if (producer && consumer &&
          !memoryInternalEdges.emplace(edgeKey(*producer, *consumer)).second)
        return projectionInvalid("persistent memory internal edge is repeated");
    }
  std::map<EdgeKey, std::uint64_t> feedbackDistance;
  for (const StaticRecurrenceFeedback &feedback : schedule->feedbacks())
    if (!feedbackDistance
             .emplace(edgeKey(feedback.producer, feedback.consumer),
                      feedback.dependenceDistance)
             .second)
      return projectionInvalid("persistent feedback edge is repeated");

  result.edgeRecords.reserve(schedule->edges().size());
  for (const StaticActorEdgeCriticality &edge : schedule->edges()) {
    const auto producer = actorOrdinals.find(actorKey(edge.producer.actor));
    const auto consumer = actorOrdinals.find(actorKey(edge.consumer.actor));
    if (producer == actorOrdinals.end() || consumer == actorOrdinals.end())
      return projectionInvalid("persistent recurrence edge has no actor");
    FrozenRecurrenceEdgeDisposition disposition;
    const EdgeKey key = edgeKey(edge.producer, edge.consumer);
    if (residualEdges.count(key)) {
      disposition = FrozenRecurrenceEdgeDisposition::Residual;
    } else if (memoryInternalEdges.count(key)) {
      disposition = FrozenRecurrenceEdgeDisposition::MemoryInternal;
    } else {
      const FrozenRecurrenceActor &source =
          result.actorRecords[producer->second];
      const FrozenRecurrenceActor &target =
          result.actorRecords[consumer->second];
      if (source.ownerKind != FrozenRecurrenceActorOwnerKind::Compute ||
          target.ownerKind != FrozenRecurrenceActorOwnerKind::Compute ||
          source.owner != target.owner)
        return projectionInvalid(
            "persistent actor edge has no physical disposition");
      disposition = FrozenRecurrenceEdgeDisposition::ComputeInternal;
    }
    const auto feedback = feedbackDistance.find(key);
    auto ordinal =
        checkedIndex(result.edgeRecords.size(), "persistent recurrence edges");
    if (!ordinal)
      return ordinal.takeError();
    result.edgeRecords.push_back(
        {edge.producer, edge.consumer, edge.graph, producer->second,
         consumer->second, disposition, getInvalidPnrIndex(),
         getInvalidPnrIndex(), edge.initializedFeedback,
         feedback != feedbackDistance.end(),
         feedback == feedbackDistance.end() ? 0 : feedback->second});
    if (feedback != feedbackDistance.end())
      result.feedbackEdgeOrdinals.push_back(*ordinal);
  }

  for (const StaticGraphRecurrenceTopology &topology :
       schedule->recurrenceTopologies()) {
    std::vector<PnrIndex> graphActors;
    std::vector<PnrIndex> graphEdges;
    for (auto [ordinal, actor] : llvm::enumerate(result.actorRecords))
      if (actor.graph == topology.graph)
        graphActors.push_back(static_cast<PnrIndex>(ordinal));
    for (auto [ordinal, edge] : llvm::enumerate(result.edgeRecords))
      if (edge.graph == topology.graph)
        graphEdges.push_back(static_cast<PnrIndex>(ordinal));
    auto actorOffset = checkedIndex(result.graphActorOrdinals.size(),
                                    "persistent graph actors");
    auto actorCount =
        checkedIndex(graphActors.size(), "persistent graph actors");
    auto edgeOffset =
        checkedIndex(result.graphEdgeOrdinals.size(), "persistent graph edges");
    auto edgeCount = checkedIndex(graphEdges.size(), "persistent graph edges");
    auto topologicalOffset = checkedIndex(
        result.topologicalActorOrdinals.size(), "persistent topology");
    if (!actorOffset)
      return actorOffset.takeError();
    if (!actorCount)
      return actorCount.takeError();
    if (!edgeOffset)
      return edgeOffset.takeError();
    if (!edgeCount)
      return edgeCount.takeError();
    if (!topologicalOffset)
      return topologicalOffset.takeError();
    result.graphActorOrdinals.insert(result.graphActorOrdinals.end(),
                                     graphActors.begin(), graphActors.end());
    result.graphEdgeOrdinals.insert(result.graphEdgeOrdinals.end(),
                                    graphEdges.begin(), graphEdges.end());

    std::vector<PnrIndex> topologicalActors;
    if (topology.postInitializationAcyclic) {
      std::vector<PnrIndex> indegree(result.actorRecords.size(), 0);
      std::vector<std::vector<PnrIndex>> successors(result.actorRecords.size());
      for (PnrIndex edgeOrdinal : graphEdges) {
        const FrozenRecurrenceEdge &edge = result.edgeRecords[edgeOrdinal];
        if (edge.initializedFeedback)
          continue;
        successors[edge.producerActor].push_back(edge.consumerActor);
        ++indegree[edge.consumerActor];
      }
      std::priority_queue<PnrIndex, std::vector<PnrIndex>,
                          std::greater<PnrIndex>>
          ready;
      for (PnrIndex actor : graphActors)
        if (indegree[actor] == 0)
          ready.push(actor);
      while (!ready.empty()) {
        const PnrIndex actor = ready.top();
        ready.pop();
        topologicalActors.push_back(actor);
        for (PnrIndex successor : successors[actor])
          if (--indegree[successor] == 0)
            ready.push(successor);
      }
      if (topologicalActors.size() != graphActors.size())
        return projectionInvalid(
            "persistent recurrence topology is inconsistent");
    }
    result.topologicalActorOrdinals.insert(
        result.topologicalActorOrdinals.end(), topologicalActors.begin(),
        topologicalActors.end());
    auto topologicalCount =
        checkedIndex(topologicalActors.size(), "persistent topology");
    if (!topologicalCount)
      return topologicalCount.takeError();
    result.graphRecords.push_back(
        {topology.graph, topology.postInitializationAcyclic, *actorOffset,
         *actorCount, *edgeOffset, *edgeCount, *topologicalOffset,
         *topologicalCount});
  }
  return result;
}

const SpatialComputeBindingView *
findPersistentComputeBinding(const SpatialMappingView &mapping,
                             std::uint64_t realization) {
  const SpatialComputeBindingView *result = nullptr;
  for (const SpatialComputeBindingView &binding : mapping.computeBindings()) {
    if (binding.realization != realization)
      continue;
    if (result)
      return nullptr;
    result = &binding;
  }
  return result;
}

const SpatialMemoryEngineBindingView *
findPersistentMemoryBinding(const SpatialMappingView &mapping,
                            std::uint64_t realization) {
  const SpatialMemoryEngineBindingView *result = nullptr;
  for (const SpatialMemoryEngineBindingView &binding :
       mapping.memoryEngineBindings()) {
    if (binding.realization != realization)
      continue;
    if (result)
      return nullptr;
    result = &binding;
  }
  return result;
}

const SpatialMemoryOperationView *
findPersistentMemoryOperation(const SpatialMemoryEngineBindingView &binding,
                              ::dataflow::ActorRef actor) {
  const SpatialMemoryOperationView *result = nullptr;
  for (const SpatialMemoryOperationView &operation : binding.operations) {
    const ::dataflow::ActorRef candidate =
        std::visit([](const auto &typed) { return typed.actor; }, operation);
    if (candidate != actor)
      continue;
    if (result)
      return nullptr;
    result = &operation;
  }
  return result;
}

llvm::Expected<FabricUsePatternRef> findPersistentMemoryPattern(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const SpatialMappingView &mapping, ::dataflow::ActorRef actor,
    bool engineOwner, std::uint64_t ownerOrdinal) {
  auto issue = deriveSpatialMemoryIssueEvent(dataflow, actor);
  if (!issue)
    return issue.takeError();
  const FabricUsePatternRef *result = nullptr;
  for (const SpatialResourceUseView &use : mapping.resourceUses()) {
    bool matchesOwner = false;
    if (engineOwner) {
      const auto *owner =
          std::get_if<SpatialMemoryEngineResourceOwnerRef>(&use.owner);
      matchesOwner = owner && owner->realization == ownerOrdinal;
    } else {
      const auto *owner =
          std::get_if<SpatialMemoryBindingResourceOwnerRef>(&use.owner);
      matchesOwner = owner && owner->binding == ownerOrdinal;
    }
    if (!matchesOwner)
      continue;
    const auto *event = std::get_if<SpatialActorTransitionEventRef>(
        &use.activation.trigger.event);
    if (!event || !(*event == *issue))
      continue;
    if (result)
      return projectionInvalid(
          "persistent memory activation selects multiple UsePatterns");
    result = &use.useSite;
  }
  if (!result)
    return projectionInvalid(
        "persistent memory activation has no exact UsePattern");
  return *result;
}

struct PersistentMemoryActorProjection final {
  ::dataflow::CanonicalActorSchemaProjection actor;
  std::optional<::dataflow::semantics::CanonicalMemoryAccessView> access;
};

llvm::Expected<PersistentMemoryActorProjection>
projectPersistentMemoryActor(const ::dataflow::CanonicalActorView &actor) {
  auto projection =
      ::dataflow::projectRegisteredActorSchemaProjection(actor.op);
  if (!projection)
    return projection.takeError();
  auto service = ::dataflow::semantics::CanonicalService::forActor(actor.op);
  if (!service)
    return service.takeError();
  std::optional<::dataflow::semantics::CanonicalMemoryAccessView> access;
  if (service->kind() != ::dataflow::semantics::ServiceKind::MemoryFence) {
    auto projected =
        ::dataflow::semantics::getCanonicalMemoryAccessView(actor.op);
    if (!projected)
      return projected.takeError();
    access.emplace(std::move(*projected));
  }
  return PersistentMemoryActorProjection{std::move(*projection),
                                         std::move(access)};
}

std::optional<std::uint64_t> persistentCompletionBound(
    const ::fabric::MemoryServiceCapabilityDeclaration &capability) {
  const auto *local = std::get_if<::fabric::LocalProviderConsistency>(
      &capability.consistencyBinding);
  if (!local)
    return std::nullopt;
  const auto *bounded =
      std::get_if<::fabric::LocalBoundedCompletionCycles>(&local->progress);
  return bounded ? std::optional<std::uint64_t>(bounded->maxIssueToRetireCycles)
                 : std::nullopt;
}

llvm::Expected<std::optional<std::uint64_t>>
projectPersistentLocalCompletion(const FabricArtifactView &fabric,
                                 const ::dataflow::CanonicalActorView &actor,
                                 const LocalMemoryServiceRef &target,
                                 const FabricUsePatternRef &pattern) {
  if (target.underlying().kind() != FabricMemoryServiceKind::Local)
    return projectionInvalid("persistent local dispatch is not local");
  const auto occurrence =
      std::get<FabricMemoryOccurrenceRef>(target.underlying().payload);
  const auto *service = fabric.localMemoryService(occurrence);
  if (!service)
    return projectionInvalid("persistent local dispatch has no service");
  auto projectedActor = projectPersistentMemoryActor(actor);
  if (!projectedActor)
    return projectedActor.takeError();
  auto matches = service->matchingCapabilities(projectedActor->actor,
                                               projectedActor->access);
  if (!matches)
    return matches.takeError();
  bool found = false;
  std::optional<std::uint64_t> result;
  const ::fabric::UsePatternKey key(
      static_cast<std::uint32_t>(pattern.ordinal));
  for (std::uint64_t capabilityOrdinal : *matches) {
    if (capabilityOrdinal >= service->capabilities().size())
      return projectionInvalid("persistent service capability is out of range");
    const auto &capability = service->capabilities()[capabilityOrdinal];
    if (!llvm::is_contained(capability.admissibleUsePatterns, key))
      continue;
    const auto bound = persistentCompletionBound(capability);
    if (found && result != bound)
      return projectionInvalid(
          "persistent local dispatch has inconsistent completion timing");
    found = true;
    result = bound;
  }
  if (!found)
    return projectionInvalid(
        "persistent local dispatch UsePattern is not admitted");
  return result;
}

llvm::Expected<FrozenPersistentRecurrenceActorTiming>
projectPersistentComputeTiming(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const TechMappingView &techMapping, const FabricArtifactView &fabric,
    const SpatialMappingView &mapping, const FrozenRecurrenceActor &actor) {
  if (actor.owner >= techMapping.computeRealizations().size())
    return projectionInvalid("persistent compute owner is out of range");
  const TechComputeRealizationView &realization =
      techMapping.computeRealizations()[actor.owner];
  const SpatialComputeBindingView *binding =
      findPersistentComputeBinding(mapping, realization.entityId);
  const TechComputeActorView *techActor =
      findComputeActor(realization, actor.actor);
  if (!binding || !techActor)
    return projectionInvalid("persistent compute binding is absent");
  auto dataflowActor = dataflow.resolve(actor.actor);
  if (!dataflowActor)
    return dataflowActor.takeError();
  auto occurrence = deriveFabricFuOccurrenceNode(
      fabric, techActor->fabricOperation, binding->occurrence);
  if (!occurrence)
    return occurrence.takeError();
  const ResolvedFabricOpCapabilityView *capability =
      fabric.resolvedFabricOpCapability(*occurrence);
  if (!capability)
    return projectionInvalid("persistent compute capability is absent");
  std::optional<std::uint32_t> nextState;
  auto publications = resultPublicationLatencies(
      *dataflowActor, capability->resourceStateAndTimingContract, nextState);
  if (!publications)
    return publications.takeError();
  FrozenPersistentRecurrenceActorTiming result;
  result.fixedPublications.reserve(publications->size());
  for (const auto publication : *publications)
    result.fixedPublications.push_back(
        publication ? std::optional<std::uint64_t>(*publication)
                    : std::optional<std::uint64_t>{});
  if (nextState)
    result.nextState = *nextState;
  return result;
}

llvm::Expected<FrozenPersistentRecurrenceActorTiming>
projectPersistentMemoryTiming(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const TechMappingView &techMapping, const FabricArtifactView &fabric,
    const SpatialMappingView &mapping, const FrozenRecurrenceActor &actor) {
  if (actor.owner >= techMapping.memoryRealizations().size())
    return projectionInvalid("persistent memory owner is out of range");
  const TechMemoryRealizationView &realization =
      techMapping.memoryRealizations()[actor.owner];
  const SpatialMemoryEngineBindingView *binding =
      findPersistentMemoryBinding(mapping, realization.entityId);
  if (!binding)
    return projectionInvalid("persistent memory binding is absent");
  const SpatialMemoryOperationView *operation =
      findPersistentMemoryOperation(*binding, actor.actor);
  if (!operation)
    return projectionInvalid("persistent memory operation is absent");
  auto dataflowActor = dataflow.resolve(actor.actor);
  if (!dataflowActor)
    return dataflowActor.takeError();
  const SpatialMemoryOperationPlacementView &placement = std::visit(
      [](const auto &typed) -> const SpatialMemoryOperationPlacementView & {
        return typed.placement;
      },
      *operation);
  const FabricMemoryOperationPortRef port = std::visit(
      [](const auto &typed) {
        using Type = std::decay_t<decltype(typed)>;
        if constexpr (std::is_same_v<Type, FabricMemoryOperationPortRef>)
          return typed;
        else
          return typed.port;
      },
      placement);
  const auto *portRecord = fabric.memoryOperationPort(port);
  if (!portRecord)
    return projectionInvalid("persistent memory operation port is absent");
  auto operationPattern = findPersistentMemoryPattern(
      dataflow, mapping, actor.actor, true, realization.entityId);
  if (!operationPattern)
    return operationPattern.takeError();
  auto issue = projectMemoryOperationIssueLatency(
      *portRecord, ::fabric::UsePatternKey(
                       static_cast<std::uint32_t>(operationPattern->ordinal)));
  if (!issue)
    return issue.takeError();

  FrozenPersistentRecurrenceActorTiming result;
  result.fixedPublications.resize(dataflowActor->op->getNumResults());
  result.memoryIssueLatencyCycles = *issue;
  if (const auto *addressed =
          std::get_if<SpatialAddressedMemoryOperationView>(operation)) {
    for (const SpatialAddressedMemoryUseView &use : addressed->uses) {
      const auto *local = std::get_if<LocalMemoryServiceRef>(&use.dispatch);
      if (!local) {
        result.memoryUses.push_back({use.launch, std::nullopt, true});
        continue;
      }
      auto servicePattern = findPersistentMemoryPattern(
          dataflow, mapping, actor.actor, false, use.binding);
      if (!servicePattern)
        return servicePattern.takeError();
      auto completion = projectPersistentLocalCompletion(
          fabric, *dataflowActor, *local, *servicePattern);
      if (!completion)
        return completion.takeError();
      result.memoryUses.push_back({use.launch, *completion, false});
    }
  } else {
    const auto &fence = std::get<SpatialFenceMemoryOperationView>(*operation);
    for (const SpatialFenceMemoryUseView &use : fence.uses) {
      const bool boundary =
          std::holds_alternative<::loom::fabric::ManagerEndpointRef>(
              use.consistency);
      result.memoryUses.push_back({use.launch, std::nullopt, boundary});
    }
  }
  if (result.memoryUses.empty())
    return projectionInvalid("persistent memory actor has no rooted use");
  return result;
}

llvm::Expected<std::uint64_t>
persistentTraversalLatency(const FabricArtifactView &fabric,
                           const FabricPhysicalTraversalRef &reference) {
  const FabricPhysicalTraversalView *found = nullptr;
  for (const FabricPhysicalTraversalView &traversal :
       fabric.physicalTraversals()) {
    if (traversal.reference != reference)
      continue;
    if (found)
      return projectionInvalid("persistent traversal is repeated");
    found = &traversal;
  }
  if (!found)
    return projectionInvalid("persistent traversal does not resolve");
  return found->timing.architecturalLatencyCycles;
}

llvm::Expected<std::uint64_t>
addPersistentTraversal(const FabricArtifactView &fabric,
                       const FabricPhysicalTraversalRef &traversal,
                       std::uint64_t total) {
  auto latency = persistentTraversalLatency(fabric, traversal);
  if (!latency)
    return latency.takeError();
  return checkedAdd(total, *latency, "persistent route latency");
}

llvm::Expected<std::pair<SpatialRecurrenceEdgeDisposition, std::uint64_t>>
projectPersistentResidualTransport(const FabricArtifactView &fabric,
                                   const SpatialMappingView &mapping,
                                   const FrozenRecurrenceEdge &edge) {
  const SpatialRegisterFifoTransferView *fifo = nullptr;
  for (const SpatialRegisterFifoTransferView &candidate :
       mapping.registerFifoTransfers()) {
    const auto *producer =
        std::get_if<::dataflow::ActorTokenResultRef>(&candidate.logicalNet);
    const auto *consumer =
        std::get_if<::dataflow::ActorTokenOperandRef>(&candidate.sink);
    if (!producer || !consumer || *producer != edge.producer ||
        *consumer != edge.consumer)
      continue;
    if (fifo)
      return projectionInvalid("persistent recurrence RegFIFO is repeated");
    fifo = &candidate;
  }
  if (fifo) {
    auto total = addPersistentTraversal(fabric, fifo->writeTraversal, 0);
    if (!total)
      return total.takeError();
    total = addPersistentTraversal(fabric, fifo->readTraversal, *total);
    if (!total)
      return total.takeError();
    return std::pair{SpatialRecurrenceEdgeDisposition::RegisterFifo, *total};
  }

  const SpatialRouteTreeView *route = nullptr;
  const SpatialRouteSinkView *sink = nullptr;
  for (const SpatialRouteTreeView &candidate : mapping.routeTrees()) {
    const auto *producer =
        std::get_if<::dataflow::ActorTokenResultRef>(&candidate.logicalNet);
    if (!producer || *producer != edge.producer)
      continue;
    for (const SpatialRouteSinkView &candidateSink : candidate.sinks) {
      const auto *consumer =
          std::get_if<::dataflow::ActorTokenOperandRef>(&candidateSink.sink);
      if (!consumer || *consumer != edge.consumer)
        continue;
      if (route)
        return projectionInvalid("persistent recurrence route is repeated");
      route = &candidate;
      sink = &candidateSink;
    }
  }
  if (!route || !sink)
    return projectionInvalid("persistent recurrence route is absent");

  std::uint64_t total = 0;
  if (route->localTraversal) {
    auto next = addPersistentTraversal(fabric, *route->localTraversal, total);
    if (!next)
      return next.takeError();
    total = *next;
  }
  std::map<std::uint64_t, const SpatialRouteNodeView *> nodes;
  for (const SpatialRouteNodeView &node : route->nodes)
    if (!nodes.emplace(node.ordinal, &node).second)
      return projectionInvalid("persistent recurrence route repeats a node");
  auto current = nodes.find(sink->nodeOrdinal);
  if (current == nodes.end())
    return projectionInvalid("persistent recurrence sink node is absent");
  for (std::size_t depth = 0;; ++depth) {
    if (depth > route->nodes.size())
      return projectionInvalid("persistent recurrence route contains a cycle");
    const SpatialRouteNodeView &node = *current->second;
    if (!node.parentOrdinal) {
      if (node.incomingTraversal)
        return projectionInvalid(
            "persistent recurrence root has an incoming traversal");
      break;
    }
    if (!node.incomingTraversal)
      return projectionInvalid(
          "persistent recurrence route has an incomplete arc");
    auto next = addPersistentTraversal(fabric, *node.incomingTraversal, total);
    if (!next)
      return next.takeError();
    total = *next;
    current = nodes.find(*node.parentOrdinal);
    if (current == nodes.end())
      return projectionInvalid("persistent recurrence route parent is absent");
  }
  if (sink->localTraversal) {
    auto next = addPersistentTraversal(fabric, *sink->localTraversal, total);
    if (!next)
      return next.takeError();
    total = *next;
  }
  return std::pair{SpatialRecurrenceEdgeDisposition::ExternalRouteTree, total};
}

llvm::Expected<std::optional<std::uint64_t>> projectFrozenActorPublication(
    const FrozenSpatialRecurrenceTimingDemand &demand, PnrIndex actorOrdinal,
    std::uint64_t resultOrdinal,
    std::optional<::dataflow::RootedGraphLaunchRef> exactLaunch,
    SpatialBoundaryMemoryCompletionResolver boundaryCompletion) {
  if (actorOrdinal >= demand.actors().size() ||
      actorOrdinal >= demand.actorTimings().size())
    return projectionInvalid("persistent recurrence actor timing is absent");
  const FrozenRecurrenceActor &actor = demand.actors()[actorOrdinal];
  const FrozenPersistentRecurrenceActorTiming &timing =
      demand.actorTimings()[actorOrdinal];
  if (resultOrdinal >= timing.fixedPublications.size())
    return projectionInvalid(
        "persistent recurrence result timing is out of range");
  if (actor.ownerKind == FrozenRecurrenceActorOwnerKind::Compute)
    return timing.fixedPublications[resultOrdinal];
  if (!timing.memoryIssueLatencyCycles)
    return std::optional<std::uint64_t>{};

  bool matchedUse = false;
  std::uint64_t maximumCompletion = 0;
  for (const FrozenPersistentMemoryUseTiming &use : timing.memoryUses) {
    if (exactLaunch && use.launch != *exactLaunch)
      continue;
    matchedUse = true;
    std::optional<std::uint64_t> completion = use.localCompletionCycles;
    if (use.requiresBoundaryCompletion) {
      auto resolved = boundaryCompletion({use.launch, actor.actor});
      if (!resolved)
        return resolved.takeError();
      completion = *resolved;
    }
    if (!completion)
      return std::optional<std::uint64_t>{};
    maximumCompletion = std::max(maximumCompletion, *completion);
  }
  if (!matchedUse)
    return projectionInvalid(
        "persistent recurrence actor has no use in the selected launch");
  return checkedAdd(*timing.memoryIssueLatencyCycles, maximumCompletion,
                    "persistent memory publication latency");
}

} // namespace

std::uint64_t
loom::pnr::detail::FrozenSpatialRecurrenceTimingDemand::retainedBytes() const {
  std::uint64_t bytes =
      sizeof(*this) + actors_.capacity() * sizeof(FrozenRecurrenceActor) +
      edges_.capacity() * sizeof(FrozenRecurrenceEdge) +
      graphs_.capacity() * sizeof(FrozenRecurrenceGraph) +
      graphActors_.capacity() * sizeof(PnrIndex) +
      graphEdges_.capacity() * sizeof(PnrIndex) +
      graphTopologicalActors_.capacity() * sizeof(PnrIndex) +
      feedbackEdges_.capacity() * sizeof(PnrIndex) +
      actorTimings_.capacity() * sizeof(FrozenPersistentRecurrenceActorTiming) +
      edgeTimings_.capacity() * sizeof(FrozenPersistentRecurrenceEdgeTiming);
  for (const FrozenPersistentRecurrenceActorTiming &timing : actorTimings_) {
    bytes += timing.fixedPublications.capacity() *
             sizeof(std::optional<std::uint64_t>);
    bytes +=
        timing.memoryUses.capacity() * sizeof(FrozenPersistentMemoryUseTiming);
  }
  return bytes;
}

llvm::Expected<std::shared_ptr<const FrozenSpatialRecurrenceTimingDemand>>
loom::pnr::detail::freezeSpatialMappingGraphRecurrenceTimingDemand(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const TechMappingView &techMapping, const FabricArtifactView &fabric,
    const SpatialMappingView &mapping, ::dataflow::GraphRef graph) {
  if (mapping.dataflowIdentity() != dataflow.identity() ||
      mapping.techMappingIdentity() != techMapping.identity() ||
      mapping.fabricIdentity() != fabric.identity())
    return projectionInvalid("persistent recurrence dependency tuple differs");
  const std::array<::dataflow::GraphRef, 1> covers{graph};
  auto index = buildPersistentRecurrenceIndex(dataflow, techMapping, covers);
  if (!index)
    return index.takeError();

  auto demand = std::make_shared<FrozenSpatialRecurrenceTimingDemand>();
  demand->actors_ = std::move(index->actorRecords);
  demand->edges_ = std::move(index->edgeRecords);
  demand->graphs_ = std::move(index->graphRecords);
  demand->graphActors_ = std::move(index->graphActorOrdinals);
  demand->graphEdges_ = std::move(index->graphEdgeOrdinals);
  demand->graphTopologicalActors_ = std::move(index->topologicalActorOrdinals);
  demand->feedbackEdges_ = std::move(index->feedbackEdgeOrdinals);
  demand->actorTimings_.reserve(demand->actors_.size());
  for (const FrozenRecurrenceActor &actor : demand->actors_) {
    auto timing = actor.ownerKind == FrozenRecurrenceActorOwnerKind::Compute
                      ? projectPersistentComputeTiming(dataflow, techMapping,
                                                       fabric, mapping, actor)
                      : projectPersistentMemoryTiming(dataflow, techMapping,
                                                      fabric, mapping, actor);
    if (!timing)
      return timing.takeError();
    demand->actorTimings_.push_back(std::move(*timing));
  }
  demand->edgeTimings_.reserve(demand->edges_.size());
  for (const FrozenRecurrenceEdge &edge : demand->edges_) {
    FrozenPersistentRecurrenceEdgeTiming timing;
    if (edge.disposition == FrozenRecurrenceEdgeDisposition::MemoryInternal) {
      timing.disposition = SpatialRecurrenceEdgeDisposition::MemoryInternal;
    } else if (edge.disposition == FrozenRecurrenceEdgeDisposition::Residual) {
      auto projected =
          projectPersistentResidualTransport(fabric, mapping, edge);
      if (!projected)
        return projected.takeError();
      timing.disposition = projected->first;
      timing.transportLatencyCycles = projected->second;
    }
    demand->edgeTimings_.push_back(timing);
  }
  return std::shared_ptr<const FrozenSpatialRecurrenceTimingDemand>(
      std::move(demand));
}

llvm::Expected<SpatialRecurrenceTimingProjection>
loom::pnr::detail::projectFrozenSpatialRecurrenceTimingDemand(
    const FrozenSpatialRecurrenceTimingDemand &demand,
    std::optional<::dataflow::RootedGraphLaunchRef> exactLaunch,
    SpatialBoundaryMemoryCompletionResolver boundaryCompletion) {
  if (demand.actorTimings().size() != demand.actors().size() ||
      demand.edgeTimings().size() != demand.edges().size())
    return projectionInvalid("persistent recurrence demand is incomplete");
  return projectRecurrenceCycles(
      demand, [&](PnrIndex ordinal) -> llvm::Expected<ProjectionValue> {
        if (ordinal >= demand.edges().size())
          return projectionInvalid(
              "persistent recurrence edge is out of range");
        const FrozenRecurrenceEdge &edge = demand.edges()[ordinal];
        if (edge.producerActor >= demand.actorTimings().size() ||
            edge.consumerActor >= demand.actorTimings().size())
          return projectionInvalid(
              "persistent recurrence timing is out of range");
        auto publication = projectFrozenActorPublication(
            demand, edge.producerActor, edge.producer.ordinal, exactLaunch,
            boundaryCompletion);
        if (!publication)
          return publication.takeError();
        if (!*publication)
          return ProjectionValue{
              std::nullopt,
              missingPublicationTiming(demand.actors()[edge.producerActor],
                                       edge.producer.ordinal)};
        const FrozenPersistentRecurrenceEdgeTiming &edgeTiming =
            demand.edgeTimings()[ordinal];
        std::uint64_t nextState = 0;
        if (edge.feedback) {
          const auto next = demand.actorTimings()[edge.consumerActor].nextState;
          if (!next)
            return ProjectionValue{std::nullopt,
                                   "carry_next_state_timing_not_established"};
          nextState = *next;
        }
        auto partial =
            checkedAdd(**publication, edgeTiming.transportLatencyCycles,
                       "persistent recurrence edge latency");
        if (!partial)
          return partial.takeError();
        auto total = checkedAdd(*partial, nextState,
                                "persistent recurrence edge latency");
        if (!total)
          return total.takeError();
        return ProjectionValue{SpatialRecurrenceTimingEdgeWitness{
                                   edge.producer, edge.consumer,
                                   edgeTiming.disposition, **publication,
                                   edgeTiming.transportLatencyCycles, nextState,
                                   *total},
                               {}};
      });
}

llvm::Expected<SpatialRecurrenceTimingProjection>
loom::pnr::projectSpatialMappingRecurrenceTiming(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const TechMappingView &techMapping, const FabricArtifactView &fabric,
    const SpatialMappingView &mapping) {
  SpatialRecurrenceTimingProjection result;
  for (const ::dataflow::GraphRef graph : techMapping.covers()) {
    auto demand = freezeSpatialMappingGraphRecurrenceTimingDemand(
        dataflow, techMapping, fabric, mapping, graph);
    if (!demand)
      return demand.takeError();
    auto projection = projectFrozenSpatialRecurrenceTimingDemand(
        **demand, std::nullopt,
        [](const ::dataflow::ContextualActorRef &)
            -> llvm::Expected<std::optional<std::uint64_t>> {
          return std::optional<std::uint64_t>{};
        });
    if (!projection)
      return projection.takeError();
    if (projection->kind ==
        SpatialRecurrenceTimingProofKind::ProofNotEstablished)
      return std::move(*projection);
    result.recurrenceMinimumInitiationIntervalCycles =
        std::max(result.recurrenceMinimumInitiationIntervalCycles,
                 projection->recurrenceMinimumInitiationIntervalCycles);
    result.witnesses.insert(
        result.witnesses.end(),
        std::make_move_iterator(projection->witnesses.begin()),
        std::make_move_iterator(projection->witnesses.end()));
  }
  return result;
}

llvm::Expected<SpatialRecurrenceTimingProjection>
loom::pnr::projectSpatialMappingGraphRecurrenceTiming(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const TechMappingView &techMapping, const FabricArtifactView &fabric,
    const SpatialMappingView &mapping, ::dataflow::GraphRef graph) {
  auto demand = freezeSpatialMappingGraphRecurrenceTimingDemand(
      dataflow, techMapping, fabric, mapping, graph);
  if (!demand)
    return demand.takeError();
  return projectFrozenSpatialRecurrenceTimingDemand(
      **demand, std::nullopt,
      [](const ::dataflow::ContextualActorRef &)
          -> llvm::Expected<std::optional<std::uint64_t>> {
        return std::optional<std::uint64_t>{};
      });
}
