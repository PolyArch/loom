#include "SpatialCandidateTopologyPreference.h"

#include "SpatialBindingRelationModel.h"
#include "StaticSchedulePressure.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <limits>
#include <system_error>
#include <variant>

using namespace loom::pnr;

namespace {

llvm::Error topologyError(const llvm::Twine &message) {
  return llvm::make_error<llvm::StringError>(
      ("invalid Spatial candidate topology preference: " + message).str(),
      std::make_error_code(std::errc::invalid_argument));
}

} // namespace

llvm::Expected<detail::SpatialCandidateTopologyPreference>
detail::SpatialCandidateTopologyPreference::create(
    const FrozenSpatialPnrProblem &problem) {
  SpatialCandidateTopologyPreference result(problem);
  const auto &transfers = problem.transfers();
  const auto &ports = problem.ports();
  const auto nets = transfers.logicalNets();
  const auto sources = transfers.logicalNetSourceBindings();
  const auto sinks = transfers.logicalNetSinkBindings();
  const auto demands = ports.portDemands();
  const PnrIndex computeCount =
      problem.bindingRelations().computeDecisionCount();
  const PnrIndex memoryCount = problem.bindingRelations().memoryDecisionCount();
  const PnrIndex rootCount = computeCount + memoryCount;
  if (sources.size() != nets.size())
    return topologyError("logical-net source binding domain is malformed");

  const auto demandRoot =
      [&](const FrozenSpatialPortDemand &demand) -> PnrIndex {
    if (demand.kind == FrozenSpatialPortDemandKind::Compute)
      return demand.realization < computeCount ? demand.realization
                                               : getInvalidPnrIndex();
    return demand.realization < memoryCount ? computeCount + demand.realization
                                            : getInvalidPnrIndex();
  };

  result.incidences_.resize(rootCount);
  for (PnrIndex logicalNet = 0; logicalNet < nets.size(); ++logicalNet) {
    const FrozenSpatialTerminalBinding source = sources[logicalNet];
    const FrozenSpatialPortDemand *sourceDemand = nullptr;
    PnrIndex sourceRoot = getInvalidPnrIndex();
    if (source.kind == FrozenSpatialTerminalBindingKind::PortDemand) {
      if (source.index >= demands.size())
        return topologyError("logical-net source PortDemand is out of range");
      sourceDemand = &demands[source.index];
      sourceRoot = demandRoot(*sourceDemand);
      if (sourceRoot == getInvalidPnrIndex())
        sourceDemand = nullptr;
    } else if (source.index >= ports.graphBoundaries().size()) {
      return topologyError("logical-net source graph boundary is out of range");
    }

    const FrozenSpatialLogicalNet &net = nets[logicalNet];
    if (net.sinkOffset > sinks.size() ||
        net.sinkCount > sinks.size() - net.sinkOffset)
      return topologyError("logical-net sink binding domain is malformed");
    for (PnrIndex sinkOrdinal = 0; sinkOrdinal < net.sinkCount; ++sinkOrdinal) {
      const auto graphSinks = transfers.logicalNetSinks();
      if (net.sinkOffset + sinkOrdinal >= graphSinks.size())
        return topologyError("logical-net sink inventory is malformed");
      std::uint64_t distanceWeight = 1;
      if (const auto *producer =
              std::get_if<::dataflow::ActorTokenResultRef>(&net.producer))
        if (const auto *consumer =
                std::get_if<::dataflow::ActorTokenOperandRef>(
                    &graphSinks[net.sinkOffset + sinkOrdinal])) {
          const std::uint64_t critical =
              problem.schedulePressure().edgeWeight(*producer, *consumer);
          if (critical == std::numeric_limits<std::uint64_t>::max())
            return topologyError("critical edge weight exceeds u64");
          distanceWeight += critical;
        }

      const FrozenSpatialTerminalBinding sink =
          sinks[net.sinkOffset + sinkOrdinal];
      const FrozenSpatialPortDemand *sinkDemand = nullptr;
      PnrIndex sinkRoot = getInvalidPnrIndex();
      if (sink.kind == FrozenSpatialTerminalBindingKind::PortDemand) {
        if (sink.index >= demands.size())
          return topologyError("logical-net sink PortDemand is out of range");
        sinkDemand = &demands[sink.index];
        sinkRoot = demandRoot(*sinkDemand);
        if (sinkRoot == getInvalidPnrIndex())
          sinkDemand = nullptr;
      } else if (sink.index >= ports.graphBoundaries().size()) {
        return topologyError("logical-net sink graph boundary is out of range");
      }

      if (sourceDemand && sinkDemand) {
        if (sinkRoot == sourceRoot)
          continue;
        if (sinkDemand->payloadWidthBits != sourceDemand->payloadWidthBits)
          return topologyError("root-neighbor payload widths disagree");
        result.incidences_[sourceRoot].push_back(
            {sinkRoot, getInvalidPnrIndex(), source.index, sink.index,
             sourceDemand->payloadWidthBits, distanceWeight, true});
        result.incidences_[sinkRoot].push_back(
            {sourceRoot, getInvalidPnrIndex(), sink.index, source.index,
             sourceDemand->payloadWidthBits, distanceWeight, false});
        continue;
      }

      if (!sourceDemand && sinkDemand &&
          source.kind == FrozenSpatialTerminalBindingKind::GraphBoundary) {
        const FrozenSpatialGraphBoundary &boundary =
            ports.graphBoundaries()[source.index];
        if (boundary.payloadWidthBits != sinkDemand->payloadWidthBits)
          return topologyError("graph-ingress payload widths disagree");
        result.incidences_[sinkRoot].push_back(
            {getInvalidPnrIndex(), source.index, sink.index,
             getInvalidPnrIndex(), sinkDemand->payloadWidthBits, distanceWeight,
             false});
        continue;
      }

      if (sourceDemand && !sinkDemand &&
          sink.kind == FrozenSpatialTerminalBindingKind::GraphBoundary) {
        const FrozenSpatialGraphBoundary &boundary =
            ports.graphBoundaries()[sink.index];
        if (boundary.payloadWidthBits != sourceDemand->payloadWidthBits)
          return topologyError("graph-egress payload widths disagree");
        result.incidences_[sourceRoot].push_back(
            {getInvalidPnrIndex(), sink.index, source.index,
             getInvalidPnrIndex(), sourceDemand->payloadWidthBits,
             distanceWeight, true});
      }
    }
  }
  return result;
}

llvm::Expected<llvm::ArrayRef<FrozenSpatialAttachmentOption>>
detail::SpatialCandidateTopologyPreference::attachmentOptionsForPlacement(
    PnrIndex demandOrdinal, PnrIndex placement) const {
  const auto &ports = problem_->ports();
  const auto demands = ports.portDemands();
  const auto domains = ports.placementDomains();
  const auto options = ports.attachmentOptions();
  if (demandOrdinal >= demands.size())
    return topologyError("foreign PortDemand");
  const FrozenSpatialPortDemand &demand = demands[demandOrdinal];
  if (demand.placementDomainOffset > domains.size() ||
      demand.placementDomainCount >
          domains.size() - demand.placementDomainOffset)
    return topologyError("PortDemand domain is malformed");
  for (const FrozenSpatialPortPlacementDomain &domain : domains.slice(
           demand.placementDomainOffset, demand.placementDomainCount)) {
    if (domain.placement != placement)
      continue;
    if (domain.attachmentOptionOffset > options.size() ||
        domain.attachmentOptionCount >
            options.size() - domain.attachmentOptionOffset)
      return topologyError("attachment domain is malformed");
    return options.slice(domain.attachmentOptionOffset,
                         domain.attachmentOptionCount);
  }
  return topologyError("placement has no attachment domain");
}

llvm::Error detail::SpatialCandidateTopologyPreference::fillHopDistances(
    llvm::ArrayRef<FrozenSpatialAttachmentOption> fixedOptions,
    std::uint32_t payloadWidthBits, bool forward) {
  const FrozenSpatialRoutingGraph &routing = problem_->routing();
  const auto endpoints = routing.routingEndpoints();
  const auto arcs = routing.routingArcs();
  const auto arcSources = routing.arcSources();
  const auto adjacency = routing.adjacencyOffsets();
  const auto reverseAdjacency = routing.reverseAdjacencyOffsets();
  const auto reverseArcs = routing.reverseArcOrdinals();
  constexpr std::uint32_t unreachable =
      std::numeric_limits<std::uint32_t>::max();
  hopDistances_.assign(endpoints.size(), unreachable);
  hopWorklist_.clear();
  for (const FrozenSpatialAttachmentOption &option : fixedOptions) {
    if (option.endpoint >= endpoints.size())
      return topologyError("attachment endpoint is out of range");
    if (hopDistances_[option.endpoint] == 0)
      continue;
    hopDistances_[option.endpoint] = 0;
    hopWorklist_.push_back(option.endpoint);
  }
  if (hopWorklist_.empty())
    return topologyError("no fixed endpoints");

  for (std::size_t cursor = 0; cursor < hopWorklist_.size(); ++cursor) {
    const PnrIndex endpoint = hopWorklist_[cursor];
    if (hopDistances_[endpoint] == unreachable - 1)
      return topologyError("hop distance exceeds u32");
    const std::uint32_t nextDistance = hopDistances_[endpoint] + 1;
    if (forward) {
      if (endpoint + 1 >= adjacency.size())
        return topologyError("forward adjacency is malformed");
      for (PnrIndex arc = adjacency[endpoint]; arc < adjacency[endpoint + 1];
           ++arc) {
        if (arc >= arcs.size() || arcs[arc].target >= hopDistances_.size())
          return topologyError("forward arc is malformed");
        if (!problem_->activeRouting().arcIsActive(arc) ||
            arcs[arc].payloadCapacityBits < payloadWidthBits ||
            hopDistances_[arcs[arc].target] != unreachable)
          continue;
        hopDistances_[arcs[arc].target] = nextDistance;
        hopWorklist_.push_back(arcs[arc].target);
      }
      continue;
    }

    if (endpoint + 1 >= reverseAdjacency.size())
      return topologyError("reverse adjacency is malformed");
    for (PnrIndex entry = reverseAdjacency[endpoint];
         entry < reverseAdjacency[endpoint + 1]; ++entry) {
      if (entry >= reverseArcs.size() || reverseArcs[entry] >= arcs.size() ||
          reverseArcs[entry] >= arcSources.size())
        return topologyError("reverse arc is malformed");
      const PnrIndex arc = reverseArcs[entry];
      const PnrIndex source = arcSources[arc];
      if (source >= hopDistances_.size())
        return topologyError("reverse source is malformed");
      if (!problem_->activeRouting().arcIsActive(arc) ||
          arcs[arc].payloadCapacityBits < payloadWidthBits ||
          hopDistances_[source] != unreachable)
        continue;
      hopDistances_[source] = nextDistance;
      hopWorklist_.push_back(source);
    }
  }
  return llvm::Error::success();
}

llvm::Expected<llvm::ArrayRef<std::uint32_t>>
detail::SpatialCandidateTopologyPreference::hopDistancesFrom(
    llvm::ArrayRef<FrozenSpatialAttachmentOption> fixedOptions,
    std::uint32_t payloadWidthBits, bool forward) {
  if (llvm::Error error =
          fillHopDistances(fixedOptions, payloadWidthBits, forward))
    return std::move(error);
  return llvm::ArrayRef(hopDistances_);
}

llvm::Expected<PnrIndex>
detail::SpatialCandidateTopologyPreference::selectedRootPlacement(
    PnrIndex root, llvm::ArrayRef<PnrIndex> selectedChoiceOrdinals) const {
  const auto &bindings = problem_->bindingRelations();
  if (root >= selectedChoiceOrdinals.size())
    return topologyError("neighbor root is out of range");
  const PnrIndex computeCount = bindings.computeDecisionCount();
  const PnrIndex choice = selectedChoiceOrdinals[root];
  if (choice == getInvalidPnrIndex())
    return getInvalidPnrIndex();
  if (root < computeCount) {
    const auto choices = bindings.computeChoices(root);
    if (choice >= choices.size())
      return topologyError("neighbor compute choice is out of range");
    return choices[choice].placement;
  }
  const PnrIndex memory = root - computeCount;
  if (memory >= bindings.memoryDecisionCount())
    return topologyError("neighbor memory root is out of range");
  const auto choices = bindings.memoryChoices(memory);
  if (choice >= choices.size())
    return topologyError("neighbor memory choice is out of range");
  return choices[choice].placement;
}

llvm::Expected<detail::SpatialCandidateTopologyScores>
detail::SpatialCandidateTopologyPreference::scoreChoices(
    PnrIndex root, llvm::ArrayRef<PnrIndex> choicePlacements,
    llvm::ArrayRef<PnrIndex> selectedChoiceOrdinals) {
  SpatialCandidateTopologyScores result;
  result.distances.assign(choicePlacements.size(), 0);
  result.unreachable.assign(choicePlacements.size(), 0);
  if (root >= incidences_.size())
    return topologyError("candidate root is out of range");
  const auto &bindings = problem_->bindingRelations();

  for (const Incidence &incidence : incidences_[root]) {
    llvm::ArrayRef<FrozenSpatialAttachmentOption> fixedOptions;
    if (incidence.graphBoundary != getInvalidPnrIndex()) {
      if (incidence.graphBoundary >= problem_->ports().graphBoundaries().size())
        return topologyError("graph boundary is out of range");
      fixedOptionScratch_.clear();
      for (PnrIndex option :
           bindings.graphBoundaryAttachmentChoices(incidence.graphBoundary)) {
        if (option >= problem_->ports().attachmentOptions().size())
          return topologyError("graph-boundary option is out of range");
        const FrozenSpatialAttachmentOption &record =
            problem_->ports().attachmentOptions()[option];
        if (record.ownerKind !=
                FrozenSpatialAttachmentOwnerKind::GraphBoundary ||
            record.owner != incidence.graphBoundary)
          return topologyError("graph-boundary option owner is malformed");
        fixedOptionScratch_.push_back(record);
      }
      fixedOptions = fixedOptionScratch_;
      if (result.activeBoundaryAnchorIncidences ==
          std::numeric_limits<std::uint64_t>::max())
        return topologyError("boundary-anchor incidence count exceeds u64");
      ++result.activeBoundaryAnchorIncidences;
    } else {
      auto neighborPlacement =
          selectedRootPlacement(incidence.neighbor, selectedChoiceOrdinals);
      if (!neighborPlacement)
        return neighborPlacement.takeError();
      if (*neighborPlacement == getInvalidPnrIndex())
        continue;
      auto selectedOptions = attachmentOptionsForPlacement(
          incidence.neighborDemand, *neighborPlacement);
      if (!selectedOptions)
        return selectedOptions.takeError();
      fixedOptions = *selectedOptions;
    }
    if (llvm::Error error =
            fillHopDistances(fixedOptions, incidence.payloadWidthBits,
                             !incidence.candidateIsSource))
      return std::move(error);

    llvm::DenseMap<PnrIndex, PnrIndex> placementDomains;
    const auto &ports = problem_->ports();
    if (incidence.candidateDemand >= ports.portDemands().size())
      return topologyError("candidate PortDemand is out of range");
    const FrozenSpatialPortDemand &demand =
        ports.portDemands()[incidence.candidateDemand];
    const auto domains = ports.placementDomains();
    if (demand.placementDomainOffset > domains.size() ||
        demand.placementDomainCount >
            domains.size() - demand.placementDomainOffset)
      return topologyError("candidate placement domain is malformed");
    for (PnrIndex local = 0; local < demand.placementDomainCount; ++local) {
      const PnrIndex domainOrdinal = demand.placementDomainOffset + local;
      const PnrIndex placement = domains[domainOrdinal].placement;
      if (!placementDomains.try_emplace(placement, domainOrdinal).second)
        return topologyError("candidate placement is duplicated");
    }

    constexpr std::uint32_t unreachable =
        std::numeric_limits<std::uint32_t>::max();
    const auto options = ports.attachmentOptions();
    for (std::size_t localChoice = 0; localChoice < choicePlacements.size();
         ++localChoice) {
      const auto found = placementDomains.find(choicePlacements[localChoice]);
      if (found == placementDomains.end())
        return topologyError("choice has no candidate attachment domain");
      const FrozenSpatialPortPlacementDomain &domain = domains[found->second];
      if (domain.attachmentOptionOffset > options.size() ||
          domain.attachmentOptionCount >
              options.size() - domain.attachmentOptionOffset)
        return topologyError("candidate attachments are malformed");
      std::uint32_t minimum = unreachable;
      for (const FrozenSpatialAttachmentOption &option : options.slice(
               domain.attachmentOptionOffset, domain.attachmentOptionCount)) {
        if (option.endpoint >= hopDistances_.size())
          return topologyError("candidate endpoint is out of range");
        minimum = std::min(minimum, hopDistances_[option.endpoint]);
      }
      if (minimum == unreachable) {
        result.unreachable[localChoice] = 1;
        continue;
      }
      if (minimum != 0 &&
          incidence.distanceWeight >
              std::numeric_limits<std::uint64_t>::max() / minimum)
        return topologyError("weighted distance exceeds u64");
      const std::uint64_t weighted = minimum * incidence.distanceWeight;
      if (weighted > std::numeric_limits<std::uint64_t>::max() -
                         result.distances[localChoice])
        return topologyError("distance exceeds u64");
      result.distances[localChoice] += weighted;
    }
    if (result.activeIncidences == std::numeric_limits<std::uint64_t>::max())
      return topologyError("incidence count exceeds u64");
    ++result.activeIncidences;
  }
  return result;
}
