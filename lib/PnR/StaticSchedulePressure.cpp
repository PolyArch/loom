#include "StaticSchedulePressure.h"

#include "PnR/SpatialCandidateState.h"
#include "PnR/SpatialPnrProblem.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <system_error>
#include <utility>
#include <vector>

using namespace loom::pnr;

namespace {

using loom::pnr::detail::SpatialSchedulePressureEdge;
using loom::pnr::detail::StaticActorCriticality;
using loom::pnr::detail::StaticActorEdgeCriticality;

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      "static_schedule_pressure_invalid: " + message);
}
llvm::Expected<std::uint64_t>
checkedSum(std::uint64_t left, std::uint64_t right, llvm::StringRef subject) {
  if (right > std::numeric_limits<std::uint64_t>::max() - left)
    return invalid(subject + " exceeds u64");
  return left + right;
}

llvm::Error addTo(std::uint64_t &value, std::uint64_t increment,
                  llvm::StringRef subject) {
  auto sum = checkedSum(value, increment, subject);
  if (!sum)
    return sum.takeError();
  value = *sum;
  return llvm::Error::success();
}

using ActorKey = std::uint64_t;
using GraphKey = std::uint64_t;

ActorKey actorKey(::dataflow::ActorRef actor) { return actor.entity.value(); }

GraphKey graphKey(::dataflow::GraphRef graph) { return graph.entity.value(); }

llvm::Expected<std::uint64_t>
actorPlacementContribution(const StaticActorCriticality &actor,
                           ::fabric::Schedule schedule) {
  if (schedule == ::fabric::Schedule::Spatial)
    return actor.temporalStateCarrier ? 1 : 0;
  return checkedSum(actor.graphCriticalLength, actor.recurrenceCriticalLength,
                    "actor Temporal pressure");
}

PnrIndex flatRoot(PnrIndex computeRootCount, bool memory, PnrIndex ordinal) {
  return memory ? computeRootCount + ordinal : ordinal;
}

llvm::Expected<::fabric::Schedule>
selectedRootSchedule(const SpatialCandidateState &candidate, PnrIndex root,
                     std::optional<std::pair<PnrIndex, PnrIndex>> override) {
  const auto &problem = candidate.problem();
  const PnrIndex computeCount = problem.schedulePressure().computeRootCount();
  if (root < computeCount) {
    PnrIndex placement = candidate.computeBinding(root).placement;
    if (override && override->first == root)
      placement = override->second;
    if (placement >= problem.realizations().computePlacements().size())
      return invalid("selected compute placement is out of range");
    return problem.realizations().computePlacements()[placement].schedule;
  }
  const PnrIndex memory = root - computeCount;
  PnrIndex placement = candidate.memoryBinding(memory).placement;
  if (override && override->first == root)
    placement = override->second;
  if (placement >= problem.realizations().memoryPlacements().size())
    return invalid("selected memory placement is out of range");
  return problem.realizations().memoryPlacements()[placement].schedule;
}

llvm::Expected<std::uint64_t>
projectAfterRootChange(const SpatialCandidateState &candidate, PnrIndex root,
                       PnrIndex oldPlacement, PnrIndex newPlacement,
                       bool memory) {
  const auto &index = candidate.problem().schedulePressure();
  if (root >= index.rootCount())
    return invalid("changed schedule root is out of range");
  const std::uint64_t oldBase =
      memory ? index.memoryPlacementContribution(oldPlacement)
             : index.computePlacementContribution(oldPlacement);
  const std::uint64_t newBase =
      memory ? index.memoryPlacementContribution(newPlacement)
             : index.computePlacementContribution(newPlacement);
  std::uint64_t result = candidate.staticSchedulePressure();
  if (oldBase > result)
    return invalid("actor placement pressure exceeds its total");
  result -= oldBase;
  if (llvm::Error error = addTo(result, newBase, "static schedule pressure"))
    return std::move(error);

  auto oldSchedule = selectedRootSchedule(candidate, root, std::nullopt);
  if (!oldSchedule)
    return oldSchedule.takeError();
  auto newSchedule =
      selectedRootSchedule(candidate, root, std::pair{root, newPlacement});
  if (!newSchedule)
    return newSchedule.takeError();
  for (PnrIndex edgeOrdinal : index.incidentEdges(root)) {
    if (edgeOrdinal >= index.edges().size())
      return invalid("schedule-pressure incidence is out of range");
    const SpatialSchedulePressureEdge &edge = index.edges()[edgeOrdinal];
    const PnrIndex other =
        edge.firstRoot == root ? edge.secondRoot : edge.firstRoot;
    auto otherSchedule = selectedRootSchedule(candidate, other, std::nullopt);
    if (!otherSchedule)
      return otherSchedule.takeError();
    const bool oldCrossing = *oldSchedule != *otherSchedule;
    const bool newCrossing = *newSchedule != *otherSchedule;
    if (oldCrossing == newCrossing)
      continue;
    if (oldCrossing) {
      if (edge.weight > result)
        return invalid("edge pressure exceeds its total");
      result -= edge.weight;
    } else if (llvm::Error error =
                   addTo(result, edge.weight, "static schedule pressure")) {
      return std::move(error);
    }
  }
  return result;
}

} // namespace

llvm::Expected<std::shared_ptr<const detail::SpatialSchedulePressureIndex>>
detail::SpatialSchedulePressureIndex::build(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ::loom::mapping::TechMappingView &techMapping,
    const FrozenSpatialRealizationIndex &realizations) {
  auto analysis = deriveStaticScheduleAnalysis(dataflow, techMapping.covers());
  if (!analysis)
    return analysis.takeError();
  auto result = std::make_shared<SpatialSchedulePressureIndex>();
  result->analysis_ = std::move(*analysis);
  if (realizations.computeRealizations().size() > getPnrIndexMax() ||
      realizations.memoryRealizations().size() > getPnrIndexMax() ||
      realizations.computeRealizations().size() >
          getPnrIndexMax() - realizations.memoryRealizations().size())
    return invalid("schedule root inventory exceeds PnrIndex");
  result->computeRootCount_ =
      static_cast<PnrIndex>(realizations.computeRealizations().size());
  result->rootCount_ =
      static_cast<PnrIndex>(realizations.computeRealizations().size() +
                            realizations.memoryRealizations().size());

  std::map<ActorKey, PnrIndex> rootByActor;
  for (const auto &[ordinal, realization] :
       llvm::enumerate(techMapping.computeRealizations())) {
    const PnrIndex root = static_cast<PnrIndex>(ordinal);
    for (const auto &actor : realization.actors)
      if (!rootByActor.emplace(actorKey(actor.actor), root).second)
        return invalid("TechMapping actor belongs to multiple realizations");
  }
  for (const auto &[ordinal, realization] :
       llvm::enumerate(techMapping.memoryRealizations())) {
    const PnrIndex root = flatRoot(result->computeRootCount_, true,
                                   static_cast<PnrIndex>(ordinal));
    for (const auto &actor : realization.actors)
      if (!rootByActor.emplace(actorKey(actor.actor), root).second)
        return invalid("TechMapping actor belongs to multiple realizations");
  }
  if (rootByActor.size() != result->analysis_.actors().size())
    return invalid("TechMapping does not partition every covered actor");

  result->computePlacementContributions_.reserve(
      realizations.computePlacements().size());
  for (const auto &placement : realizations.computePlacements()) {
    if (placement.realization >= realizations.computeRealizations().size())
      return invalid("compute placement has a foreign realization");
    const auto &realization =
        realizations.computeRealizations()[placement.realization];
    std::uint64_t contribution = 0;
    for (const ::dataflow::ActorRef actor : realizations.computeActors().slice(
             realization.actorOffset, realization.actorCount)) {
      const StaticActorCriticality *criticality =
          result->analysis_.findActor(actor);
      if (!criticality)
        return invalid("compute realization has an unanalyzed actor");
      auto value = actorPlacementContribution(*criticality, placement.schedule);
      if (!value)
        return value.takeError();
      if (llvm::Error error =
              addTo(contribution, *value, "compute placement pressure"))
        return std::move(error);
    }
    result->computePlacementContributions_.push_back(contribution);
  }
  result->memoryPlacementContributions_.reserve(
      realizations.memoryPlacements().size());
  for (const auto &placement : realizations.memoryPlacements()) {
    if (placement.realization >= realizations.memoryRealizations().size())
      return invalid("memory placement has a foreign realization");
    const auto &realization =
        realizations.memoryRealizations()[placement.realization];
    std::uint64_t contribution = 0;
    for (const auto &actor : realizations.memoryActors().slice(
             realization.actorOffset, realization.actorCount)) {
      const StaticActorCriticality *criticality =
          result->analysis_.findActor(actor.actor);
      if (!criticality)
        return invalid("memory realization has an unanalyzed actor");
      auto value = actorPlacementContribution(*criticality, placement.schedule);
      if (!value)
        return value.takeError();
      if (llvm::Error error =
              addTo(contribution, *value, "memory placement pressure"))
        return std::move(error);
    }
    result->memoryPlacementContributions_.push_back(contribution);
  }

  std::map<std::pair<PnrIndex, PnrIndex>, std::uint64_t> edgeWeights;
  for (const StaticActorEdgeCriticality &edge : result->analysis_.edges()) {
    if (edge.weight == 0)
      continue;
    const auto source = rootByActor.find(actorKey(edge.producer.actor));
    const auto sink = rootByActor.find(actorKey(edge.consumer.actor));
    if (source == rootByActor.end() || sink == rootByActor.end())
      return invalid("critical edge has an unmapped actor");
    if (source->second == sink->second)
      continue;
    const auto roots = std::minmax(source->second, sink->second);
    if (llvm::Error error =
            addTo(edgeWeights[{roots.first, roots.second}], edge.weight,
                  "root-pair critical edge weight"))
      return std::move(error);
  }
  result->edges_.reserve(edgeWeights.size());
  for (const auto &[roots, weight] : edgeWeights)
    result->edges_.push_back({roots.first, roots.second, weight});
  std::vector<std::vector<PnrIndex>> incidence(result->rootCount_);
  for (const auto &[ordinal, edge] : llvm::enumerate(result->edges_)) {
    if (ordinal > getPnrIndexMax())
      return invalid("schedule-pressure edge inventory exceeds PnrIndex");
    const PnrIndex index = static_cast<PnrIndex>(ordinal);
    incidence[edge.firstRoot].push_back(index);
    incidence[edge.secondRoot].push_back(index);
  }
  result->incidenceOffsets_.reserve(result->rootCount_ + 1);
  for (const auto &rootEdges : incidence) {
    if (result->incidenceEdges_.size() > getPnrIndexMax())
      return invalid("schedule-pressure incidence exceeds PnrIndex");
    result->incidenceOffsets_.push_back(
        static_cast<PnrIndex>(result->incidenceEdges_.size()));
    result->incidenceEdges_.insert(result->incidenceEdges_.end(),
                                   rootEdges.begin(), rootEdges.end());
  }
  if (result->incidenceEdges_.size() > getPnrIndexMax())
    return invalid("schedule-pressure incidence exceeds PnrIndex");
  result->incidenceOffsets_.push_back(
      static_cast<PnrIndex>(result->incidenceEdges_.size()));
  return std::shared_ptr<const SpatialSchedulePressureIndex>(std::move(result));
}

std::uint64_t
detail::SpatialSchedulePressureIndex::computePlacementContribution(
    PnrIndex placement) const {
  return computePlacementContributions_.at(placement);
}

std::uint64_t detail::SpatialSchedulePressureIndex::memoryPlacementContribution(
    PnrIndex placement) const {
  return memoryPlacementContributions_.at(placement);
}

llvm::ArrayRef<PnrIndex>
detail::SpatialSchedulePressureIndex::incidentEdges(PnrIndex root) const {
  if (root >= rootCount_)
    return {};
  return llvm::ArrayRef(incidenceEdges_)
      .slice(incidenceOffsets_[root],
             incidenceOffsets_[root + 1] - incidenceOffsets_[root]);
}

llvm::Expected<std::uint64_t> loom::pnr::detail::measureStaticSchedulePressure(
    const SpatialCandidateState &candidate) {
  const auto &index = candidate.problem().schedulePressure();
  std::uint64_t result = 0;
  for (PnrIndex root = 0; root < index.computeRootCount(); ++root)
    if (llvm::Error error = addTo(result,
                                  index.computePlacementContribution(
                                      candidate.computeBinding(root).placement),
                                  "static schedule pressure"))
      return std::move(error);
  for (PnrIndex root = index.computeRootCount(); root < index.rootCount();
       ++root)
    if (llvm::Error error =
            addTo(result,
                  index.memoryPlacementContribution(
                      candidate.memoryBinding(root - index.computeRootCount())
                          .placement),
                  "static schedule pressure"))
      return std::move(error);
  for (const SpatialSchedulePressureEdge &edge : index.edges()) {
    auto first = selectedRootSchedule(candidate, edge.firstRoot, std::nullopt);
    if (!first)
      return first.takeError();
    auto second =
        selectedRootSchedule(candidate, edge.secondRoot, std::nullopt);
    if (!second)
      return second.takeError();
    if (*first != *second)
      if (llvm::Error error =
              addTo(result, edge.weight, "static schedule pressure"))
        return std::move(error);
  }
  return result;
}

llvm::Expected<std::uint64_t>
loom::pnr::detail::projectStaticSchedulePressureAfterComputeChange(
    const SpatialCandidateState &candidate, PnrIndex realization,
    PnrIndex placement) {
  return projectAfterRootChange(candidate, realization,
                                candidate.computeBinding(realization).placement,
                                placement, false);
}

llvm::Expected<std::uint64_t>
loom::pnr::detail::projectStaticSchedulePressureAfterMemoryChange(
    const SpatialCandidateState &candidate, PnrIndex realization,
    PnrIndex placement) {
  const PnrIndex root =
      candidate.problem().schedulePressure().computeRootCount() + realization;
  return projectAfterRootChange(candidate, root,
                                candidate.memoryBinding(realization).placement,
                                placement, true);
}

llvm::Expected<std::vector<std::uint64_t>>
loom::pnr::detail::projectStaticSchedulePressureByGraph(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ::loom::mapping::TechMappingView &techMapping,
    const ::loom::fabric::FabricArtifactView &fabric,
    const ::loom::mapping::SpatialMappingView &mapping) {
  if (mapping.dataflowIdentity() != dataflow.identity() ||
      mapping.techMappingIdentity() != techMapping.identity() ||
      mapping.fabricIdentity() != fabric.identity())
    return invalid("SpatialMapping lineage differs from its pressure owners");
  auto analysis = deriveStaticScheduleAnalysis(dataflow, techMapping.covers());
  if (!analysis)
    return analysis.takeError();
  std::map<GraphKey, std::size_t> graphOrdinals;
  for (const auto &[ordinal, graph] : llvm::enumerate(techMapping.covers()))
    if (!graphOrdinals.emplace(graphKey(graph), ordinal).second)
      return invalid("TechMapping cover inventory contains a duplicate");
  std::vector<std::uint64_t> result(techMapping.covers().size(), 0);

  std::map<std::uint64_t, std::size_t> computeOrdinals;
  for (const auto &[ordinal, realization] :
       llvm::enumerate(techMapping.computeRealizations()))
    if (!computeOrdinals.emplace(realization.entityId, ordinal).second)
      return invalid("TechMapping compute entity ID is duplicated");
  std::map<std::uint64_t, std::size_t> memoryOrdinals;
  for (const auto &[ordinal, realization] :
       llvm::enumerate(techMapping.memoryRealizations()))
    if (!memoryOrdinals.emplace(realization.entityId, ordinal).second)
      return invalid("TechMapping memory entity ID is duplicated");
  std::vector<std::optional<::fabric::Schedule>> computeSchedules(
      techMapping.computeRealizations().size());
  for (const auto &binding : mapping.computeBindings()) {
    const auto realization = computeOrdinals.find(binding.realization);
    if (realization == computeOrdinals.end() ||
        computeSchedules[realization->second])
      return invalid("SpatialMapping compute binding is foreign or duplicated");
    const auto parent = fabric.parentPeOf(binding.occurrence);
    if (!parent)
      return invalid("SpatialMapping compute occurrence has no parent PE");
    const auto schedule = fabric.peSchedule(*parent);
    if (!schedule)
      return invalid("SpatialMapping compute PE has no schedule");
    computeSchedules[realization->second] = *schedule;
  }
  std::vector<std::optional<::fabric::Schedule>> memorySchedules(
      techMapping.memoryRealizations().size());
  for (const auto &binding : mapping.memoryEngineBindings()) {
    const auto realization = memoryOrdinals.find(binding.realization);
    if (realization == memoryOrdinals.end() ||
        memorySchedules[realization->second])
      return invalid("SpatialMapping memory binding is foreign or duplicated");
    const auto schedule = fabric.memorySchedule(binding.occurrence);
    if (!schedule)
      return invalid("SpatialMapping memory occurrence has no schedule");
    memorySchedules[realization->second] = *schedule;
  }
  if (llvm::any_of(computeSchedules,
                   [](const auto &value) { return !value; }) ||
      llvm::any_of(memorySchedules, [](const auto &value) { return !value; }))
    return invalid("SpatialMapping schedule projection is incomplete");

  struct Root final {
    bool memory = false;
    std::size_t ordinal = 0;
  };
  std::map<ActorKey, Root> rootByActor;
  for (const auto &[ordinal, realization] :
       llvm::enumerate(techMapping.computeRealizations()))
    for (const auto &actor : realization.actors)
      if (!rootByActor.emplace(actorKey(actor.actor), Root{false, ordinal})
               .second)
        return invalid("TechMapping actor belongs to multiple realizations");
  for (const auto &[ordinal, realization] :
       llvm::enumerate(techMapping.memoryRealizations()))
    for (const auto &actor : realization.actors)
      if (!rootByActor.emplace(actorKey(actor.actor), Root{true, ordinal})
               .second)
        return invalid("TechMapping actor belongs to multiple realizations");
  const auto scheduleOf = [&](Root root) {
    return root.memory ? *memorySchedules[root.ordinal]
                       : *computeSchedules[root.ordinal];
  };
  for (const StaticActorCriticality &actor : analysis->actors()) {
    const auto root = rootByActor.find(actorKey(actor.actor));
    const auto graph = graphOrdinals.find(graphKey(actor.graph));
    if (root == rootByActor.end() || graph == graphOrdinals.end())
      return invalid("analyzed actor has no Mapping root or graph");
    auto contribution =
        actorPlacementContribution(actor, scheduleOf(root->second));
    if (!contribution)
      return contribution.takeError();
    if (llvm::Error error = addTo(result[graph->second], *contribution,
                                  "graph static schedule pressure"))
      return std::move(error);
  }
  for (const StaticActorEdgeCriticality &edge : analysis->edges()) {
    if (edge.weight == 0)
      continue;
    const auto source = rootByActor.find(actorKey(edge.producer.actor));
    const auto sink = rootByActor.find(actorKey(edge.consumer.actor));
    const auto graph = graphOrdinals.find(graphKey(edge.graph));
    if (source == rootByActor.end() || sink == rootByActor.end() ||
        graph == graphOrdinals.end())
      return invalid("analyzed edge has no Mapping root or graph");
    if (source->second.memory == sink->second.memory &&
        source->second.ordinal == sink->second.ordinal)
      continue;
    if (scheduleOf(source->second) != scheduleOf(sink->second))
      if (llvm::Error error = addTo(result[graph->second], edge.weight,
                                    "graph static schedule pressure"))
        return std::move(error);
  }
  return result;
}
