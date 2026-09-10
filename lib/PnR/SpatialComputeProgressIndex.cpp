#include "SpatialComputeProgressIndex.h"

#include "Mapping/Artifact/ResourceCapacityVerification.h"
#include "Mapping/Artifact/SpatialResourceEventProjection.h"
#include "PnR/SpatialCandidateState.h"
#include "SpatialProgressAnalysis.h"
#include "SpatialProgressIndex.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>

namespace loom::pnr::detail {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "spatial_compute_progress_invalid: " +
                                     message);
}

llvm::Expected<PnrIndex> index(std::size_t value) {
  return checkedPnrIndex({"FrozenSpatialPnrProblem", "compute_progress",
                          "activation_inventory", PnrCapacityMeasure::Offset},
                         value);
}

llvm::Expected<bool> durable(const SpatialCandidateState &candidate,
                             PnrIndex logicalNet, PnrIndex sink) {
  if (candidate.usesRegisterFifo(logicalNet))
    return true;
  const auto &problem = candidate.problem();
  const auto &transfers = problem.transfers();
  for (auto terminal :
       {transfers.logicalNetSourceBindings()[logicalNet],
        transfers.logicalNetSinkBindings()
            [transfers.logicalNets()[logicalNet].sinkOffset + sink]}) {
    auto boundary =
        spatialTerminalProvidesLocalProgressBoundary(candidate, terminal);
    if (!boundary)
      return boundary.takeError();
    if (*boundary)
      return true;
  }
  const auto &tree = candidate.routeTree(logicalNet);
  auto endpoint = tree.sinkEndpoint(sink);
  auto slot = endpoint ? tree.findNode(*endpoint) : std::nullopt;
  for (std::size_t visited = 0; slot; ++visited) {
    if (visited >= tree.activeNodeCount())
      return invalid("result branch has cyclic ancestry");
    const auto &node = tree.node(*slot);
    if (node.parentArc == getInvalidPnrIndex())
      break;
    const auto traversal =
        problem.routing().routingArcs()[node.parentArc].traversal;
    if (problem.progressIndex().traversalOwner(traversal) !=
        getInvalidPnrIndex())
      return true;
    slot = tree.parentNodeSlot(*slot);
  }
  return false;
}

} // namespace

llvm::Expected<std::shared_ptr<const SpatialComputeProgressIndex>>
SpatialComputeProgressIndex::build(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ::loom::mapping::TechMappingView &techMapping,
    const ::loom::fabric::FabricArtifactView &fabric,
    const FrozenSpatialCapacityIndex &capacity,
    const FrozenSpatialResourceIndex &resources,
    const FrozenSpatialTransferIndex &transfers) {
  using namespace ::loom::mapping;
  namespace capacity_detail = ::loom::mapping::detail;
  auto connections =
      deriveSpatialComputeResultConnections(dataflow, techMapping);
  if (!connections)
    return connections.takeError();
  std::vector<capacity_detail::ResourceCapacityPatternSource> patterns;
  for (const auto &use : capacity.resourceUses())
    if (capacity.resourceEvents()[use.event].ownerKind ==
        FrozenSpatialResourceEventOwnerKind::ComputeRealization)
      patterns.push_back({0, resources.usePatterns()[use.pattern].reference});
  const capacity_detail::ResourceCapacityNamespaceView space{
      &fabric, capacity_detail::rootResourceCapacityQualifier(fabric)};
  auto physical = capacity_detail::freezeResourceCapacityIndex(
      llvm::ArrayRef(space), patterns, {});
  if (!physical)
    return physical.takeError();
  std::vector<PnrIndex> dimensions;
  for (const auto &cell : physical->cells()) {
    const auto state =
        llvm::find_if(resources.resourceStates(), [&](const auto &entry) {
          return entry.reference.owner.catalog() == cell.owner &&
                 entry.reference.ordinal == cell.state.ordinal();
        });
    if (state == resources.resourceStates().end() ||
        cell.dimension.ordinal() >= state->capacityCount)
      return invalid("compute capacity has no native resource dimension");
    const PnrIndex dimension = state->capacityOffset + cell.dimension.ordinal();
    const auto &native = resources.capacityDimensions()[dimension];
    if (native.capacity != cell.capacity ||
        native.initialOccupancy != cell.initialOccupancy)
      return invalid("native compute capacity diverges from Fabric");
    dimensions.push_back(dimension);
  }
  std::vector<::dataflow::RootedGraphLaunchRef> launches;
  dataflow.forEachRootedGraphLaunch(
      [&](auto launch) { launches.push_back(launch); });
  std::vector<::dataflow::EventFamilyKey> events;
  std::vector<Activation> activations;
  std::vector<PnrIndex> offsets{0};
  for (const auto &use : capacity.resourceUses()) {
    const auto &event = capacity.resourceEvents()[use.event];
    if (event.ownerKind ==
        FrozenSpatialResourceEventOwnerKind::ComputeRealization) {
      auto ownerGraph =
          resolveSpatialActivityEventGraph(dataflow, event.reference);
      if (!ownerGraph)
        return ownerGraph.takeError();
      auto patternOrdinal = physical->patternOrdinal(
          0, resources.usePatterns()[use.pattern].reference);
      if (!patternOrdinal)
        return patternOrdinal.takeError();
      const auto &pattern = physical->patterns()[*patternOrdinal];
      for (const auto &launch : launches) {
        auto graph = dataflow.resolve(launch);
        if (!graph)
          return graph.takeError();
        if (*graph != *ownerGraph)
          continue;
        auto triggers = projectRootedSpatialActivityEvent(dataflow, launch,
                                                          event.reference);
        if (!triggers)
          return triggers.takeError();
        events.insert(events.end(), triggers->begin(), triggers->end());
        Activation activation{{std::nullopt,
                               launch.rootThreadLaunch,
                               {SystemPresburgerCell{}},
                               std::move(*triggers),
                               {},
                               {},
                               pattern.progressUse},
                              {}};
        for (const auto &claim : pattern.claims)
          activation.projection.capacityClaims.push_back(
              {claim.cell, claim.amount});
        for (const auto &release : capacity.resourceReleaseEvents().slice(
                 use.releaseOffset, use.releaseCount)) {
          auto releaseGraph =
              resolveSpatialActivityEventGraph(dataflow, release);
          if (!releaseGraph)
            return releaseGraph.takeError();
          if (*releaseGraph != *graph)
            continue;
          const auto *producer =
              std::get_if<::dataflow::CanonicalGraphProducerEndpointRef>(
                  &release);
          const auto *result =
              producer ? std::get_if<::dataflow::ActorTokenResultRef>(producer)
                       : nullptr;
          if (!result)
            return invalid("compute release has no actor result");
          const auto connection =
              llvm::find_if(*connections, [&](const auto &entry) {
                return entry.producer == *result;
              });
          if (connection == connections->end())
            return invalid("compute release has no logical connections");
          for (const auto &sink : connection->sinks) {
            auto alternatives =
                dataflow.projectRootedGraphEndpointEventFamilies(launch,
                                                                 sink.sink);
            if (!alternatives)
              return alternatives.takeError();
            events.insert(events.end(), alternatives->begin(),
                          alternatives->end());
            PnrIndex sinkOrdinal = getInvalidPnrIndex();
            if (sink.residualBranch) {
              const auto &branch = *sink.residualBranch;
              if (branch.logicalNetOrdinal >= transfers.logicalNets().size())
                return invalid("compute release has an absent residual net");
              const auto &net =
                  transfers.logicalNets()[branch.logicalNetOrdinal];
              if (net.producer != *producer ||
                  branch.sinkOrdinal >= net.sinkCount ||
                  transfers.logicalNetSinks()[net.sinkOffset +
                                              branch.sinkOrdinal] != sink.sink)
                return invalid("compute release residual ordinal diverges");
              sinkOrdinal = net.sinkOffset + branch.sinkOrdinal;
            }
            activation.release.push_back(
                {{std::move(*alternatives)}, sinkOrdinal});
          }
        }
        activations.push_back(std::move(activation));
      }
    }
    auto end = index(activations.size());
    if (!end)
      return end.takeError();
    offsets.push_back(*end);
  }
  auto model = freezeMappingProgressModel(dataflow, events);
  if (!model)
    return model.takeError();
  return std::shared_ptr<const SpatialComputeProgressIndex>(
      new SpatialComputeProgressIndex(std::move(*model), std::move(dimensions),
                                      std::move(activations),
                                      std::move(offsets)));
}

llvm::Expected<SpatialComputeProgressStateHandle>
SpatialComputeProgressIndex::project(
    const SpatialCandidateState &candidate,
    const SpatialComputeProgressStateHandle &previous) const {
  using namespace ::loom::mapping;
  const auto &problem = candidate.problem();
  std::vector<std::uint64_t> key;
  const auto realizationCount =
      problem.realizations().computeRealizations().size();
  const auto sinkCount = problem.transfers().logicalNetSinks().size();
  const auto sinkOffset = realizationCount + capacityDimensions_.size();
  const auto keySize = sinkOffset + sinkCount;
  bool changed = !previous || previous->selectionKey.size() != keySize;
  std::size_t selectionOrdinal = 0;
  if (changed)
    key.reserve(keySize);
  // Cache hits remain allocation-free. A changed key retains its already
  // compared prefix and records only the remaining selections.
  const auto appendSelection = [&](std::uint64_t value) {
    if (!changed && previous->selectionKey[selectionOrdinal] != value) {
      key.reserve(keySize);
      key.insert(key.end(), previous->selectionKey.begin(),
                 previous->selectionKey.begin() + selectionOrdinal);
      changed = true;
    }
    if (changed)
      key.push_back(value);
    ++selectionOrdinal;
  };
  for (PnrIndex realization = 0; realization < realizationCount; ++realization)
    appendSelection(candidate.computeBinding(realization).instructionContext);
  for (auto dimension : capacityDimensions_)
    appendSelection(candidate.routeCapacityUsageRaw(dimension));
  for (auto [netOrdinal, net] :
       llvm::enumerate(problem.transfers().logicalNets()))
    for (PnrIndex sink = 0; sink < net.sinkCount; ++sink) {
      auto cut = durable(candidate, netOrdinal, sink);
      if (!cut)
        return cut.takeError();
      appendSelection(*cut);
    }
  if (!changed)
    return previous;
  MappingProgressProjection projection;
  projection.basis = problem.progressBasis();
  for (auto [ordinal, dimension] : llvm::enumerate(capacityDimensions_))
    projection.capacityCells.push_back(
        {problem.resources().capacityDimensions()[dimension].capacity,
         key[realizationCount + ordinal]});
  std::vector<PnrIndex> activationRealizations;
  const auto &capacity = problem.capacity();
  const auto envelopeOffsets =
      capacity.computeInstructionContextEnvelopeOffsets();
  for (PnrIndex realization = 0; realization < realizationCount;
       ++realization) {
    const auto context =
        candidate.computeBinding(realization).instructionContext;
    for (const auto &envelope : capacity.resourceTimeEnvelopes().slice(
             envelopeOffsets[context],
             envelopeOffsets[context + 1] - envelopeOffsets[context]))
      for (PnrIndex use = envelope.useOffset;
           use < envelope.useOffset + envelope.useCount; ++use)
        for (PnrIndex ordinal = useActivationOffsets_[use];
             ordinal < useActivationOffsets_[use + 1]; ++ordinal) {
          const auto &frozen = activations_[ordinal];
          auto activation = frozen.projection;
          for (const auto &release : frozen.release)
            if (release.sink == getInvalidPnrIndex() ||
                !key[sinkOffset + release.sink])
              activation.causalRelease.push_back(release.clause);
          projection.resourceActivations.push_back(std::move(activation));
          activationRealizations.push_back(realization);
        }
  }
  auto closure = deriveMappingProgressClosure(model_, projection);
  if (!closure)
    return closure.takeError();
  auto result = std::make_shared<SpatialComputeProgressState>();
  result->selectionKey = std::move(key);
  result->objective = projectMappingProgressObjective(*closure);
  result->closure = std::move(*closure);
  for (const auto &node : result->closure.possibleWaitCycle)
    for (auto activation : node.activationOrdinals)
      result->witnessRealizations.push_back(activationRealizations[activation]);
  llvm::sort(result->witnessRealizations);
  result->witnessRealizations.erase(
      std::unique(result->witnessRealizations.begin(),
                  result->witnessRealizations.end()),
      result->witnessRealizations.end());
  return SpatialComputeProgressStateHandle(std::move(result));
}

std::size_t SpatialComputeProgressIndex::retainedStorageBytes() const {
  std::size_t bytes = model_.retainedStorageBytes() +
                      capacityDimensions_.capacity() * sizeof(PnrIndex) +
                      useActivationOffsets_.capacity() * sizeof(PnrIndex) +
                      activations_.capacity() * sizeof(Activation);
  for (const auto &activation : activations_) {
    const auto &projection = activation.projection;
    bytes +=
        projection.triggerAlternatives.capacity() *
            sizeof(::dataflow::EventFamilyKey) +
        projection.capacityClaims.capacity() *
            sizeof(::loom::mapping::MappingProgressCapacityClaimProjection) +
        projection.relationDomain.capacity() *
            sizeof(::loom::mapping::SystemPresburgerCell) +
        projection.arbitration.physicalOwnerKey.capacity() +
        activation.release.capacity() * sizeof(Release);
    for (const auto &release : activation.release)
      bytes += release.clause.alternatives.capacity() *
               sizeof(::dataflow::EventFamilyKey);
  }
  return bytes;
}

} // namespace loom::pnr::detail

std::size_t
loom::pnr::SpatialComputeProgressState::retainedStorageBytes() const {
  std::size_t bytes = sizeof(*this) +
                      selectionKey.capacity() * sizeof(std::uint64_t) +
                      witnessRealizations.capacity() * sizeof(PnrIndex) +
                      closure.possibleWaitCycle.capacity() *
                          sizeof(loom::mapping::MappingProgressWaitCycleNode);
  for (const auto &node : closure.possibleWaitCycle)
    bytes += (node.activationOrdinals.capacity() +
              node.capacityCellOrdinals.capacity()) *
                 sizeof(std::uint64_t) +
             (node.triggerEventOrdinals.capacity() +
              node.causalReleaseEventOrdinals.capacity()) *
                 sizeof(std::uint32_t);
  return bytes;
}
