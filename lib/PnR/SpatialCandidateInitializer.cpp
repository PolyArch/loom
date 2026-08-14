#include "PnR/SpatialCandidateInitializer.h"

#include "Common/MappingDebugLog.h"
#include "InitializerChoiceOrder.h"
#include "InitializerRelationSolver.h"
#include "SpatialBindingRelationModel.h"
#include "SpatialMemoryCompatibility.h"
#include "SpatialMemoryConstraintModel.h"
#include "StaticSchedulePressure.h"

#include "Fabric/Identity/FabricRefText.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <optional>
#include <system_error>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

using namespace loom::pnr;

namespace {

using loom::pnr::detail::InitializerRelationSolveFailure;
using loom::pnr::detail::InitializerRelationSolveFailureKind;

llvm::Error initializerError(const llvm::Twine &message) {
  return llvm::make_error<llvm::StringError>(
      ("invalid Spatial candidate initialization: " + message).str(),
      std::make_error_code(std::errc::invalid_argument));
}

const FrozenSpatialMemoryDispatchDomain *
dispatchDomain(const FrozenSpatialPnrProblem &problem,
               llvm::ArrayRef<SpatialMemoryBindingSelection> memoryBindings,
               const FrozenSpatialMemoryRootedUse &use) {
  const auto &realizations = problem.realizations();
  if (use.actor >= realizations.memoryActors().size())
    return nullptr;
  const PnrIndex realization =
      realizations.memoryActorRealizations()[use.actor];
  if (realization >= memoryBindings.size())
    return nullptr;
  const auto &owner = realizations.memoryRealizations()[realization];
  if (use.actor < owner.actorOffset ||
      use.actor - owner.actorOffset >= owner.actorCount)
    return nullptr;
  const PnrIndex placement = memoryBindings[realization].placement;
  const auto offsets = problem.memory().memoryPlacementDomainOffsets();
  if (placement + 1 >= offsets.size())
    return nullptr;
  const PnrIndex domain = offsets[placement] + use.actor - owner.actorOffset;
  if (domain >= offsets[placement + 1] ||
      domain >= problem.memory().dispatchDomains().size())
    return nullptr;
  return &problem.memory().dispatchDomains()[domain];
}

void appendMatchingDispatches(
    const FrozenSpatialPnrProblem &problem,
    llvm::ArrayRef<SpatialMemoryBindingSelection> memoryBindings,
    const FrozenSpatialMemoryRootedUse &use,
    const FrozenSpatialMemoryBindingTargetOption *bindingTarget,
    std::vector<PnrIndex> &choices) {
  const auto *domain = dispatchDomain(problem, memoryBindings, use);
  if (!domain)
    return;
  const auto &memory = problem.memory();
  for (PnrIndex optionOrdinal = domain->optionOffset;
       optionOrdinal != domain->optionOffset + domain->optionCount;
       ++optionOrdinal) {
    const auto &option = memory.dispatchOptions()[optionOrdinal];
    if (!bindingTarget) {
      if (!std::holds_alternative<::loom::fabric::LocalMemoryServiceRef>(
              option.target))
        choices.push_back(optionOrdinal);
      continue;
    }
    if (detail::memoryDispatchMatchesTarget(memory, option, *bindingTarget))
      choices.push_back(optionOrdinal);
  }
}

llvm::Error initializerFailure(InitializerRelationSolveFailureKind kind,
                               const llvm::Twine &message) {
  return llvm::make_error<InitializerRelationSolveFailure>(
      kind, ("Spatial initializer " + message).str());
}

struct PreferredRootAssignment final {
  std::vector<PnrIndex> choices;
  std::uint64_t assignmentAttempts = 0;
  std::uint64_t changedComputeRoots = 0;
  std::uint64_t selectedTemporalComputeRoots = 0;
  std::uint64_t maximumContextSelections = 0;
  std::uint64_t maximumComputeOccurrenceSelections = 0;
  std::uint64_t distinctComputeOccurrences = 0;
  std::uint64_t changedMemoryRoots = 0;
  std::uint64_t selectedTemporalMemoryRoots = 0;
  std::uint64_t maximumMemorySelections = 0;
  std::uint64_t distinctMemoryOccurrences = 0;
  std::uint64_t topologyScoredRoots = 0;
  std::uint64_t topologyBoundaryAnchorIncidences = 0;
  std::uint64_t topologyHopSum = 0;
  std::uint64_t topologyUnreachableSelections = 0;
  std::uint64_t topologyRefinedComputeRoots = 0;
  std::uint64_t topologyRefinementHopSum = 0;
  std::uint64_t topologyRefinementUnreachableSelections = 0;
  std::uint64_t changedPortAttachments = 0;
  std::uint64_t changedGraphBoundaryAttachments = 0;
  std::uint64_t maximumEndpointSelections = 0;
  bool applied = false;
  llvm::StringRef status = "unchanged";
};

struct ComputeTopologyIncidence final {
  PnrIndex neighbor = getInvalidPnrIndex();
  PnrIndex graphBoundary = getInvalidPnrIndex();
  PnrIndex candidateDemand = 0;
  PnrIndex neighborDemand = 0;
  std::uint32_t payloadWidthBits = 0;
  std::uint64_t distanceWeight = 1;
  bool candidateIsSource = false;
};

struct ComputeTopologyScores final {
  std::vector<std::uint64_t> distances;
  std::vector<std::uint8_t> unreachable;
  std::uint64_t activeIncidences = 0;
  std::uint64_t activeBoundaryAnchorIncidences = 0;
};

llvm::Expected<std::vector<std::vector<ComputeTopologyIncidence>>>
buildComputeTopologyIncidences(const FrozenSpatialPnrProblem &problem) {
  const auto &transfers = problem.transfers();
  const auto &ports = problem.ports();
  const auto nets = transfers.logicalNets();
  const auto sources = transfers.logicalNetSourceBindings();
  const auto sinks = transfers.logicalNetSinkBindings();
  const auto demands = ports.portDemands();
  const PnrIndex computeCount =
      problem.bindingRelations().computeDecisionCount();
  if (sources.size() != nets.size())
    return initializerError("logical-net source binding domain is malformed");

  std::vector<std::vector<ComputeTopologyIncidence>> result(computeCount);
  for (PnrIndex logicalNet = 0; logicalNet < nets.size(); ++logicalNet) {
    const FrozenSpatialTerminalBinding source = sources[logicalNet];
    const FrozenSpatialPortDemand *sourceDemand = nullptr;
    if (source.kind == FrozenSpatialTerminalBindingKind::PortDemand) {
      if (source.index >= demands.size())
        return initializerError(
            "logical-net source PortDemand is out of range");
      sourceDemand = &demands[source.index];
      if (sourceDemand->kind != FrozenSpatialPortDemandKind::Compute ||
          sourceDemand->realization >= computeCount)
        sourceDemand = nullptr;
    } else if (source.index >= ports.graphBoundaries().size()) {
      return initializerError(
          "logical-net source graph boundary is out of range");
    }
    const FrozenSpatialLogicalNet &net = nets[logicalNet];
    if (net.sinkOffset > sinks.size() ||
        net.sinkCount > sinks.size() - net.sinkOffset)
      return initializerError("logical-net sink binding domain is malformed");
    for (PnrIndex sinkOrdinal = 0; sinkOrdinal < net.sinkCount; ++sinkOrdinal) {
      const auto graphSinks = transfers.logicalNetSinks();
      if (net.sinkOffset + sinkOrdinal >= graphSinks.size())
        return initializerError("logical-net sink inventory is malformed");
      std::uint64_t distanceWeight = 1;
      if (const auto *producer =
              std::get_if<::dataflow::ActorTokenResultRef>(&net.producer))
        if (const auto *consumer =
                std::get_if<::dataflow::ActorTokenOperandRef>(
                    &graphSinks[net.sinkOffset + sinkOrdinal])) {
          const std::uint64_t critical =
              problem.schedulePressure().edgeWeight(*producer, *consumer);
          if (critical == std::numeric_limits<std::uint64_t>::max())
            return initializerError(
                "topology preference critical edge weight exceeds u64");
          distanceWeight += critical;
        }
      const FrozenSpatialTerminalBinding sink =
          sinks[net.sinkOffset + sinkOrdinal];
      const FrozenSpatialPortDemand *sinkDemand = nullptr;
      if (sink.kind == FrozenSpatialTerminalBindingKind::PortDemand) {
        if (sink.index >= demands.size())
          return initializerError(
              "logical-net sink PortDemand is out of range");
        sinkDemand = &demands[sink.index];
        if (sinkDemand->kind != FrozenSpatialPortDemandKind::Compute ||
            sinkDemand->realization >= computeCount)
          sinkDemand = nullptr;
      } else if (sink.index >= ports.graphBoundaries().size()) {
        return initializerError(
            "logical-net sink graph boundary is out of range");
      }

      if (sourceDemand && sinkDemand) {
        if (sinkDemand->realization == sourceDemand->realization)
          continue;
        if (sinkDemand->payloadWidthBits != sourceDemand->payloadWidthBits)
          return initializerError(
              "compute-neighbor terminal payload widths disagree");
        result[sourceDemand->realization].push_back(
            {sinkDemand->realization, getInvalidPnrIndex(), source.index,
             sink.index, sourceDemand->payloadWidthBits, distanceWeight, true});
        result[sinkDemand->realization].push_back(
            {sourceDemand->realization, getInvalidPnrIndex(), sink.index,
             source.index, sourceDemand->payloadWidthBits, distanceWeight,
             false});
        continue;
      }

      if (!sourceDemand && sinkDemand &&
          source.kind == FrozenSpatialTerminalBindingKind::GraphBoundary) {
        const FrozenSpatialGraphBoundary &boundary =
            ports.graphBoundaries()[source.index];
        if (boundary.payloadWidthBits != sinkDemand->payloadWidthBits)
          return initializerError(
              "graph-ingress terminal payload widths disagree");
        result[sinkDemand->realization].push_back(
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
          return initializerError(
              "graph-egress terminal payload widths disagree");
        result[sourceDemand->realization].push_back(
            {getInvalidPnrIndex(), sink.index, source.index,
             getInvalidPnrIndex(), sourceDemand->payloadWidthBits,
             distanceWeight, true});
      }
    }
  }
  return result;
}

llvm::Expected<llvm::ArrayRef<FrozenSpatialAttachmentOption>>
attachmentOptionsForPlacement(const FrozenSpatialPnrProblem &problem,
                              PnrIndex demandOrdinal, PnrIndex placement) {
  const auto &ports = problem.ports();
  const auto demands = ports.portDemands();
  const auto domains = ports.placementDomains();
  const auto options = ports.attachmentOptions();
  if (demandOrdinal >= demands.size())
    return initializerError("topology preference names a foreign PortDemand");
  const FrozenSpatialPortDemand &demand = demands[demandOrdinal];
  if (demand.placementDomainOffset > domains.size() ||
      demand.placementDomainCount >
          domains.size() - demand.placementDomainOffset)
    return initializerError(
        "topology preference PortDemand domain is malformed");
  for (const FrozenSpatialPortPlacementDomain &domain : domains.slice(
           demand.placementDomainOffset, demand.placementDomainCount)) {
    if (domain.placement != placement)
      continue;
    if (domain.attachmentOptionOffset > options.size() ||
        domain.attachmentOptionCount >
            options.size() - domain.attachmentOptionOffset)
      return initializerError(
          "topology preference attachment domain is malformed");
    return options.slice(domain.attachmentOptionOffset,
                         domain.attachmentOptionCount);
  }
  return initializerError(
      "topology preference placement has no attachment domain");
}

llvm::Error fillTopologyHopDistances(
    const FrozenSpatialPnrProblem &problem,
    llvm::ArrayRef<FrozenSpatialAttachmentOption> fixedOptions,
    std::uint32_t payloadWidthBits, bool forward,
    std::vector<std::uint32_t> &distances, std::vector<PnrIndex> &worklist) {
  const FrozenSpatialRoutingGraph &routing = problem.routing();
  const auto endpoints = routing.routingEndpoints();
  const auto arcs = routing.routingArcs();
  const auto arcSources = routing.arcSources();
  const auto adjacency = routing.adjacencyOffsets();
  const auto reverseAdjacency = routing.reverseAdjacencyOffsets();
  const auto reverseArcs = routing.reverseArcOrdinals();
  constexpr std::uint32_t unreachable =
      std::numeric_limits<std::uint32_t>::max();
  distances.assign(endpoints.size(), unreachable);
  worklist.clear();
  for (const FrozenSpatialAttachmentOption &option : fixedOptions) {
    if (option.endpoint >= endpoints.size())
      return initializerError(
          "topology preference attachment endpoint is out of range");
    if (distances[option.endpoint] == 0)
      continue;
    distances[option.endpoint] = 0;
    worklist.push_back(option.endpoint);
  }
  if (worklist.empty())
    return initializerError("topology preference has no fixed endpoints");

  for (std::size_t cursor = 0; cursor < worklist.size(); ++cursor) {
    const PnrIndex endpoint = worklist[cursor];
    if (distances[endpoint] == unreachable - 1)
      return initializerError("topology preference hop distance overflows u32");
    const std::uint32_t nextDistance = distances[endpoint] + 1;
    if (forward) {
      if (endpoint + 1 >= adjacency.size())
        return initializerError(
            "topology preference forward adjacency is malformed");
      for (PnrIndex arc = adjacency[endpoint]; arc < adjacency[endpoint + 1];
           ++arc) {
        if (arc >= arcs.size() || arcs[arc].target >= distances.size())
          return initializerError(
              "topology preference forward arc is malformed");
        if (arcs[arc].payloadCapacityBits < payloadWidthBits ||
            distances[arcs[arc].target] != unreachable)
          continue;
        distances[arcs[arc].target] = nextDistance;
        worklist.push_back(arcs[arc].target);
      }
      continue;
    }

    if (endpoint + 1 >= reverseAdjacency.size())
      return initializerError(
          "topology preference reverse adjacency is malformed");
    for (PnrIndex entry = reverseAdjacency[endpoint];
         entry < reverseAdjacency[endpoint + 1]; ++entry) {
      if (entry >= reverseArcs.size() || reverseArcs[entry] >= arcs.size() ||
          reverseArcs[entry] >= arcSources.size())
        return initializerError("topology preference reverse arc is malformed");
      const PnrIndex arc = reverseArcs[entry];
      const PnrIndex source = arcSources[arc];
      if (source >= distances.size())
        return initializerError(
            "topology preference reverse source is malformed");
      if (arcs[arc].payloadCapacityBits < payloadWidthBits ||
          distances[source] != unreachable)
        continue;
      distances[source] = nextDistance;
      worklist.push_back(source);
    }
  }
  return llvm::Error::success();
}

llvm::Expected<ComputeTopologyScores> scoreComputeTopologyChoices(
    const FrozenSpatialPnrProblem &problem, PnrIndex realization,
    llvm::ArrayRef<detail::SpatialComputeBindingChoice> choices,
    llvm::ArrayRef<PnrIndex> selectedChoiceOrdinals,
    llvm::ArrayRef<std::vector<ComputeTopologyIncidence>> incidences,
    std::vector<std::uint32_t> &hopDistances,
    std::vector<PnrIndex> &hopWorklist,
    std::vector<FrozenSpatialAttachmentOption> &fixedOptionScratch) {
  ComputeTopologyScores result;
  result.distances.assign(choices.size(), 0);
  result.unreachable.assign(choices.size(), 0);
  if (realization >= incidences.size())
    return initializerError("topology preference realization is out of range");
  const auto &bindings = problem.bindingRelations();

  for (const ComputeTopologyIncidence &incidence : incidences[realization]) {
    llvm::ArrayRef<FrozenSpatialAttachmentOption> fixedOptions;
    if (incidence.graphBoundary != getInvalidPnrIndex()) {
      if (incidence.graphBoundary >= problem.ports().graphBoundaries().size())
        return initializerError(
            "topology preference graph boundary is out of range");
      fixedOptionScratch.clear();
      for (PnrIndex option :
           bindings.graphBoundaryAttachmentChoices(incidence.graphBoundary)) {
        if (option >= problem.ports().attachmentOptions().size())
          return initializerError(
              "topology preference graph-boundary option is out of range");
        const FrozenSpatialAttachmentOption &record =
            problem.ports().attachmentOptions()[option];
        if (record.ownerKind !=
                FrozenSpatialAttachmentOwnerKind::GraphBoundary ||
            record.owner != incidence.graphBoundary)
          return initializerError(
              "topology preference graph-boundary owner is malformed");
        fixedOptionScratch.push_back(record);
      }
      fixedOptions = fixedOptionScratch;
      if (result.activeBoundaryAnchorIncidences ==
          std::numeric_limits<std::uint64_t>::max())
        return initializerError(
            "topology boundary-anchor incidence count exceeds u64");
      ++result.activeBoundaryAnchorIncidences;
    } else {
      if (incidence.neighbor >= selectedChoiceOrdinals.size())
        return initializerError("topology preference neighbor is out of range");
      const PnrIndex neighborChoice =
          selectedChoiceOrdinals[incidence.neighbor];
      if (neighborChoice == getInvalidPnrIndex())
        continue;
      const auto neighborChoices = bindings.computeChoices(incidence.neighbor);
      if (neighborChoice >= neighborChoices.size())
        return initializerError(
            "topology preference neighbor choice is out of range");
      auto selectedOptions = attachmentOptionsForPlacement(
          problem, incidence.neighborDemand,
          neighborChoices[neighborChoice].placement);
      if (!selectedOptions)
        return selectedOptions.takeError();
      fixedOptions = *selectedOptions;
    }
    if (llvm::Error error = fillTopologyHopDistances(
            problem, fixedOptions, incidence.payloadWidthBits,
            !incidence.candidateIsSource, hopDistances, hopWorklist))
      return std::move(error);

    llvm::DenseMap<PnrIndex, PnrIndex> placementDomains;
    const auto &ports = problem.ports();
    if (incidence.candidateDemand >= ports.portDemands().size())
      return initializerError(
          "topology preference candidate demand is out of range");
    const FrozenSpatialPortDemand &demand =
        ports.portDemands()[incidence.candidateDemand];
    const auto domains = ports.placementDomains();
    if (demand.placementDomainOffset > domains.size() ||
        demand.placementDomainCount >
            domains.size() - demand.placementDomainOffset)
      return initializerError(
          "topology preference candidate domain is malformed");
    for (PnrIndex local = 0; local < demand.placementDomainCount; ++local) {
      const PnrIndex domainOrdinal = demand.placementDomainOffset + local;
      const PnrIndex placement = domains[domainOrdinal].placement;
      if (!placementDomains.try_emplace(placement, domainOrdinal).second)
        return initializerError(
            "topology preference candidate placement is duplicated");
    }

    constexpr std::uint32_t unreachable =
        std::numeric_limits<std::uint32_t>::max();
    const auto options = ports.attachmentOptions();
    for (std::size_t localChoice = 0; localChoice < choices.size();
         ++localChoice) {
      const auto found = placementDomains.find(choices[localChoice].placement);
      if (found == placementDomains.end())
        return initializerError(
            "topology preference choice has no candidate attachment domain");
      const FrozenSpatialPortPlacementDomain &domain = domains[found->second];
      if (domain.attachmentOptionOffset > options.size() ||
          domain.attachmentOptionCount >
              options.size() - domain.attachmentOptionOffset)
        return initializerError(
            "topology preference candidate attachments are malformed");
      std::uint32_t minimum = unreachable;
      for (const FrozenSpatialAttachmentOption &option : options.slice(
               domain.attachmentOptionOffset, domain.attachmentOptionCount)) {
        if (option.endpoint >= hopDistances.size())
          return initializerError(
              "topology preference candidate endpoint is out of range");
        minimum = std::min(minimum, hopDistances[option.endpoint]);
      }
      if (minimum == unreachable) {
        result.unreachable[localChoice] = 1;
        continue;
      }
      if (minimum != 0 &&
          incidence.distanceWeight >
              std::numeric_limits<std::uint64_t>::max() / minimum)
        return initializerError(
            "topology preference weighted distance exceeds u64");
      const std::uint64_t weighted = minimum * incidence.distanceWeight;
      if (weighted > std::numeric_limits<std::uint64_t>::max() -
                         result.distances[localChoice])
        return initializerError("topology preference distance exceeds u64");
      result.distances[localChoice] += weighted;
    }
    if (result.activeIncidences == std::numeric_limits<std::uint64_t>::max())
      return initializerError(
          "topology preference incidence count exceeds u64");
    ++result.activeIncidences;
  }
  return result;
}

llvm::Expected<PreferredRootAssignment> preferScheduleAwareRootPlacements(
    const FrozenSpatialPnrProblem &problem, std::uint32_t attemptOrdinal,
    detail::InitializerRelationSolver &solver,
    detail::InitializerRelationSolveResult baseline,
    std::uint64_t assignmentLimit) {
  PreferredRootAssignment result;
  result.choices = std::move(baseline.choices);
  result.assignmentAttempts = baseline.assignmentAttempts;
  const detail::SpatialBindingRelationModel &bindings =
      problem.bindingRelations();
  if (result.choices.size() != bindings.decisionCount() ||
      result.assignmentAttempts > assignmentLimit)
    return initializerError("root relation baseline has invalid accounting");

  const auto emitPreference = [&] {
    loom::mapping_debug::emit(
        loom::mapping_debug::Level::Decision,
        loom::mapping_debug::Stage::SpatialPnr,
        loom::mapping_debug::Event::ContextChoice,
        [&](llvm::json::Object &fields) {
          fields["operation"] = "initializer_root_preference";
          fields["attempt"] = attemptOrdinal;
          fields["changed_compute_roots"] = result.changedComputeRoots;
          fields["selected_temporal_compute_roots"] =
              result.selectedTemporalComputeRoots;
          fields["maximum_context_selections"] =
              result.maximumContextSelections;
          fields["maximum_compute_occurrence_selections"] =
              result.maximumComputeOccurrenceSelections;
          fields["distinct_compute_occurrences"] =
              result.distinctComputeOccurrences;
          fields["changed_memory_roots"] = result.changedMemoryRoots;
          fields["selected_temporal_memory_roots"] =
              result.selectedTemporalMemoryRoots;
          fields["maximum_memory_selections"] = result.maximumMemorySelections;
          fields["distinct_memory_occurrences"] =
              result.distinctMemoryOccurrences;
          fields["topology_scored_roots"] = result.topologyScoredRoots;
          fields["topology_boundary_anchor_incidences"] =
              result.topologyBoundaryAnchorIncidences;
          fields["topology_hop_sum"] = result.topologyHopSum;
          fields["topology_unreachable_selections"] =
              result.topologyUnreachableSelections;
          fields["topology_refined_compute_roots"] =
              result.topologyRefinedComputeRoots;
          fields["topology_refinement_hop_sum"] =
              result.topologyRefinementHopSum;
          fields["topology_refinement_unreachable_selections"] =
              result.topologyRefinementUnreachableSelections;
          fields["changed_port_attachments"] = result.changedPortAttachments;
          fields["changed_graph_boundary_attachments"] =
              result.changedGraphBoundaryAttachments;
          fields["maximum_endpoint_selections"] =
              result.maximumEndpointSelections;
          fields["assignment_attempts"] = result.assignmentAttempts;
          fields["status"] = result.status;
        });
  };

  using ContextKey =
      std::pair<::loom::fabric::FabricEntityId, ::loom::fabric::FabricOrdinal>;
  using OccurrenceKey = ::loom::fabric::FabricEntityId;
  std::map<ContextKey, std::uint64_t> selectedCounts;
  std::map<OccurrenceKey, std::uint64_t> selectedOccurrenceCounts;
  std::vector<PnrIndex> fixedChoices(bindings.decisionCount(),
                                     getInvalidPnrIndex());
  const auto &realizations = problem.realizations();
  const auto contexts = realizations.computeInstructionContexts();
  auto topologyIncidences = buildComputeTopologyIncidences(problem);
  if (!topologyIncidences)
    return topologyIncidences.takeError();
  std::vector<PnrIndex> selectedChoiceOrdinals(bindings.computeDecisionCount(),
                                               getInvalidPnrIndex());
  std::vector<std::uint32_t> hopDistances;
  std::vector<PnrIndex> hopWorklist;
  std::vector<FrozenSpatialAttachmentOption> fixedOptionScratch;
  const auto contextKey = [&](const detail::SpatialComputeBindingChoice &choice)
      -> llvm::Expected<ContextKey> {
    if (choice.instructionContext >= contexts.size())
      return initializerError(
          "compute preference resolved a foreign instruction context");
    const auto &context = contexts[choice.instructionContext];
    return ContextKey{context.pe.id(), context.ordinal};
  };
  const auto computePlacement =
      [&](const detail::SpatialComputeBindingChoice &choice)
      -> llvm::Expected<const FrozenSpatialComputePlacement *> {
    if (choice.placement >= realizations.computePlacements().size())
      return initializerError(
          "compute preference resolved a foreign placement");
    return &realizations.computePlacements()[choice.placement];
  };
  const auto computeSchedule =
      [&](const detail::SpatialComputeBindingChoice &choice)
      -> llvm::Expected<::fabric::Schedule> {
    auto placement = computePlacement(choice);
    if (!placement)
      return placement.takeError();
    return (*placement)->schedule;
  };
  // The scarce physical compute resource is the PE occurrence: every resident
  // context inside one Temporal PE shares that PE's operand and result ports
  // and the switch domain attached to them. Counting selections per resident
  // context reports no pressure until a PE is completely full, which lets
  // canonical enumeration order bind an entire graph to the first few PE
  // occurrences of an arbitrarily large Fabric. Occurrence load prices the
  // frozen-topology distance of every choice on that occurrence, so a locality
  // preference degrades as the occurrence fills instead of collapsing onto it.
  // Resident-context exclusivity remains the hard relation owned by the
  // relation solver, and occurrence load remains a search preference that
  // cannot prove infeasibility or remove a legal choice.
  const auto countComputeOccurrence =
      [&](const detail::SpatialComputeBindingChoice &choice) -> llvm::Error {
    auto placement = computePlacement(choice);
    if (!placement)
      return placement.takeError();
    std::uint64_t &count =
        selectedOccurrenceCounts[(*placement)->parentPe.id()];
    if (count == std::numeric_limits<std::uint64_t>::max())
      return initializerError("compute occurrence count overflows u64");
    result.maximumComputeOccurrenceSelections =
        std::max(result.maximumComputeOccurrenceSelections, ++count);
    return llvm::Error::success();
  };
  const auto releaseComputeOccurrence =
      [&](const detail::SpatialComputeBindingChoice &choice) -> llvm::Error {
    auto placement = computePlacement(choice);
    if (!placement)
      return placement.takeError();
    auto found = selectedOccurrenceCounts.find((*placement)->parentPe.id());
    if (found == selectedOccurrenceCounts.end() || found->second == 0)
      return initializerError(
          "topology refinement lost its current compute occurrence");
    --found->second;
    return llvm::Error::success();
  };
  const auto computeOccurrenceLoad =
      [&](const detail::SpatialComputeBindingChoice &choice)
      -> llvm::Expected<std::uint64_t> {
    auto placement = computePlacement(choice);
    if (!placement)
      return placement.takeError();
    const auto found =
        selectedOccurrenceCounts.find((*placement)->parentPe.id());
    return found == selectedOccurrenceCounts.end() ? std::uint64_t{0}
                                                   : found->second;
  };
  const auto pricedDistance =
      [&](std::uint64_t distance,
          std::uint64_t load) -> llvm::Expected<std::uint64_t> {
    const std::uint64_t price = load + 1;
    if (distance != 0 &&
        price > std::numeric_limits<std::uint64_t>::max() / distance)
      return initializerError("occurrence-priced distance exceeds u64");
    return distance * price;
  };
  const auto participatesInHardRelation = [&](PnrIndex decision) {
    return llvm::any_of(bindings.decisionRelations(decision),
                        [&](PnrIndex relation) {
                          return bindings.relationIsConstraint(relation);
                        });
  };
  const auto preferenceOrigin = [&](PnrIndex baselineChoice) {
    return attemptOrdinal == 0 ? PnrIndex{0} : baselineChoice;
  };

  for (PnrIndex realization = 0; realization < bindings.computeDecisionCount();
       ++realization) {
    if (!participatesInHardRelation(realization))
      continue;
    const auto choices = bindings.computeChoices(realization);
    const PnrIndex selected = result.choices[realization];
    if (selected >= choices.size())
      return initializerError(
          "root relation baseline selected a foreign compute choice");
    fixedChoices[realization] = selected;
    selectedChoiceOrdinals[realization] = selected;
    auto key = contextKey(choices[selected]);
    if (!key)
      return key.takeError();
    auto schedule = computeSchedule(choices[selected]);
    if (!schedule)
      return schedule.takeError();
    result.selectedTemporalComputeRoots +=
        *schedule == ::fabric::Schedule::Temporal;
    std::uint64_t &count = selectedCounts[*key];
    if (count == std::numeric_limits<std::uint64_t>::max())
      return initializerError("compute preference count overflows u64");
    result.maximumContextSelections =
        std::max(result.maximumContextSelections, ++count);
    if (llvm::Error error = countComputeOccurrence(choices[selected]))
      return std::move(error);
  }

  for (PnrIndex realization = 0; realization < bindings.computeDecisionCount();
       ++realization) {
    if (fixedChoices[realization] != getInvalidPnrIndex())
      continue;
    const auto choices = bindings.computeChoices(realization);
    const PnrIndex baselineChoice = result.choices[realization];
    if (choices.empty() || baselineChoice >= choices.size())
      return initializerError(
          "root relation baseline selected a foreign compute choice");

    auto topologyScores = scoreComputeTopologyChoices(
        problem, realization, choices, selectedChoiceOrdinals,
        *topologyIncidences, hopDistances, hopWorklist, fixedOptionScratch);
    if (!topologyScores)
      return topologyScores.takeError();
    if (topologyScores->activeIncidences != 0) {
      if (result.topologyScoredRoots ==
          std::numeric_limits<std::uint64_t>::max())
        return initializerError("topology-scored root count exceeds u64");
      ++result.topologyScoredRoots;
    }
    if (topologyScores->activeBoundaryAnchorIncidences >
        std::numeric_limits<std::uint64_t>::max() -
            result.topologyBoundaryAnchorIncidences)
      return initializerError(
          "topology boundary-anchor incidence total exceeds u64");
    result.topologyBoundaryAnchorIncidences +=
        topologyScores->activeBoundaryAnchorIncidences;

    PnrIndex selected = baselineChoice;
    auto selectedScore = std::make_tuple(
        std::numeric_limits<std::uint64_t>::max(), true,
        std::numeric_limits<std::uint64_t>::max(),
        std::numeric_limits<std::uint64_t>::max(),
        std::numeric_limits<std::uint64_t>::max(),
        std::numeric_limits<std::uint8_t>::max(), choices.size());
    std::uint64_t selectedDistance = std::numeric_limits<std::uint64_t>::max();
    bool selectedUnreachable = true;
    std::uint64_t selectedCount = std::numeric_limits<std::uint64_t>::max();
    std::uint64_t selectedOccurrenceLoad =
        std::numeric_limits<std::uint64_t>::max();
    std::uint64_t selectedSchedulePressure =
        std::numeric_limits<std::uint64_t>::max();
    bool selectedTemporal = false;
    const PnrIndex origin = preferenceOrigin(baselineChoice);
    for (std::size_t rank = 0; rank != choices.size(); ++rank) {
      const PnrIndex local = static_cast<PnrIndex>(
          (static_cast<std::size_t>(origin) + rank) % choices.size());
      auto key = contextKey(choices[local]);
      if (!key)
        return key.takeError();
      const auto found = selectedCounts.find(*key);
      const std::uint64_t count =
          found == selectedCounts.end() ? 0 : found->second;
      auto occurrenceLoad = computeOccurrenceLoad(choices[local]);
      if (!occurrenceLoad)
        return occurrenceLoad.takeError();
      const bool unreachable = topologyScores->unreachable[local] != 0;
      const std::uint64_t distance = topologyScores->distances[local];
      auto locality = pricedDistance(distance, *occurrenceLoad);
      if (!locality)
        return locality.takeError();
      auto schedule = computeSchedule(choices[local]);
      if (!schedule)
        return schedule.takeError();
      const bool temporal = *schedule == ::fabric::Schedule::Temporal;
      const std::uint64_t schedulePressure =
          problem.schedulePressure().computePlacementContribution(
              choices[local].placement);
      const auto score = std::make_tuple(
          count, unreachable, unreachable ? std::uint64_t{0} : *locality,
          *occurrenceLoad, schedulePressure,
          static_cast<std::uint8_t>(temporal ? 0 : 1), rank);
      if (score < selectedScore) {
        selected = local;
        selectedScore = score;
        selectedCount = count;
        selectedOccurrenceLoad = *occurrenceLoad;
        selectedSchedulePressure = schedulePressure;
        selectedDistance = distance;
        selectedUnreachable = unreachable;
        selectedTemporal = temporal;
      }
    }
    fixedChoices[realization] = selected;
    selectedChoiceOrdinals[realization] = selected;
    auto key = contextKey(choices[selected]);
    if (!key)
      return key.takeError();
    std::uint64_t &count = selectedCounts[*key];
    if (count == std::numeric_limits<std::uint64_t>::max())
      return initializerError("compute preference count overflows u64");
    result.maximumContextSelections =
        std::max(result.maximumContextSelections, ++count);
    if (llvm::Error error = countComputeOccurrence(choices[selected]))
      return std::move(error);
    result.changedComputeRoots += selected != baselineChoice;
    result.selectedTemporalComputeRoots += selectedTemporal;
    if (topologyScores->activeIncidences != 0) {
      if (selectedUnreachable) {
        if (result.topologyUnreachableSelections ==
            std::numeric_limits<std::uint64_t>::max())
          return initializerError(
              "topology-unreachable selection count exceeds u64");
        ++result.topologyUnreachableSelections;
      } else {
        if (selectedDistance >
            std::numeric_limits<std::uint64_t>::max() - result.topologyHopSum)
          return initializerError("topology preference hop sum exceeds u64");
        result.topologyHopSum += selectedDistance;
      }
    }
    loom::mapping_debug::emit(
        loom::mapping_debug::Level::Detail,
        loom::mapping_debug::Stage::SpatialPnr,
        loom::mapping_debug::Event::ContextChoice,
        [&](llvm::json::Object &fields) {
          const auto &context = contexts[choices[selected].instructionContext];
          const auto &placement =
              realizations.computePlacements()[choices[selected].placement];
          fields["operation"] = "initializer_compute_root_choice";
          fields["attempt"] = attemptOrdinal;
          fields["realization"] = realization;
          fields["baseline_choice"] = baselineChoice;
          fields["selected_choice"] = selected;
          fields["selected_count_before"] = selectedCount;
          fields["selected_occurrence_load_before"] = selectedOccurrenceLoad;
          fields["pe_ref"] = loom::fabric::printFabricRef(placement.parentPe);
          fields["selected_static_schedule_pressure"] =
              selectedSchedulePressure;
          fields["selected_temporal"] = selectedTemporal;
          fields["topology_incidence_count"] = topologyScores->activeIncidences;
          fields["topology_boundary_anchor_incidence_count"] =
              topologyScores->activeBoundaryAnchorIncidences;
          fields["topology_reachable"] = !selectedUnreachable;
          fields["topology_hops"] = selectedUnreachable ? 0 : selectedDistance;
          fields["instruction_context_ref"] =
              loom::fabric::printFabricRef(context);
          fields["fu_ref"] = loom::fabric::printFabricRef(placement.fu);
        });
  }

  for (PnrIndex realization = 0; realization < bindings.computeDecisionCount();
       ++realization) {
    if (participatesInHardRelation(realization))
      continue;
    const auto choices = bindings.computeChoices(realization);
    const PnrIndex current = fixedChoices[realization];
    if (current >= choices.size())
      return initializerError(
          "topology refinement has a foreign compute choice");
    auto currentKey = contextKey(choices[current]);
    if (!currentKey)
      return currentKey.takeError();
    auto count = selectedCounts.find(*currentKey);
    if (count == selectedCounts.end() || count->second == 0)
      return initializerError(
          "topology refinement lost its current instruction context");
    --count->second;
    if (llvm::Error error = releaseComputeOccurrence(choices[current]))
      return std::move(error);

    auto currentSchedule = computeSchedule(choices[current]);
    if (!currentSchedule)
      return currentSchedule.takeError();
    auto topologyScores = scoreComputeTopologyChoices(
        problem, realization, choices, selectedChoiceOrdinals,
        *topologyIncidences, hopDistances, hopWorklist, fixedOptionScratch);
    if (!topologyScores)
      return topologyScores.takeError();
    PnrIndex selected = current;
    auto selectedScore = std::make_tuple(
        std::numeric_limits<std::uint64_t>::max(), true,
        std::numeric_limits<std::uint64_t>::max(),
        std::numeric_limits<std::uint64_t>::max(), choices.size());
    std::uint64_t selectedDistance = 0;
    bool selectedUnreachable = false;
    const PnrIndex origin = preferenceOrigin(current);
    for (std::size_t rank = 0; rank != choices.size(); ++rank) {
      const PnrIndex local = static_cast<PnrIndex>(
          (static_cast<std::size_t>(origin) + rank) % choices.size());
      auto schedule = computeSchedule(choices[local]);
      if (!schedule)
        return schedule.takeError();
      if (*schedule != *currentSchedule)
        continue;
      auto key = contextKey(choices[local]);
      if (!key)
        return key.takeError();
      const auto found = selectedCounts.find(*key);
      const std::uint64_t selectedCount =
          found == selectedCounts.end() ? 0 : found->second;
      auto occurrenceLoad = computeOccurrenceLoad(choices[local]);
      if (!occurrenceLoad)
        return occurrenceLoad.takeError();
      const bool unreachable = topologyScores->unreachable[local] != 0;
      const std::uint64_t distance = topologyScores->distances[local];
      auto locality = pricedDistance(distance, *occurrenceLoad);
      if (!locality)
        return locality.takeError();
      const auto score = std::make_tuple(
          selectedCount, unreachable,
          unreachable ? std::uint64_t{0} : *locality, *occurrenceLoad, rank);
      if (score < selectedScore) {
        selected = local;
        selectedScore = score;
        selectedDistance = distance;
        selectedUnreachable = unreachable;
      }
    }
    fixedChoices[realization] = selected;
    selectedChoiceOrdinals[realization] = selected;
    auto selectedKey = contextKey(choices[selected]);
    if (!selectedKey)
      return selectedKey.takeError();
    std::uint64_t &selectedCount = selectedCounts[*selectedKey];
    if (selectedCount == std::numeric_limits<std::uint64_t>::max())
      return initializerError("topology refinement count overflows u64");
    ++selectedCount;
    if (llvm::Error error = countComputeOccurrence(choices[selected]))
      return std::move(error);
    result.topologyRefinedComputeRoots += selected != current;
    result.topologyRefinementUnreachableSelections += selectedUnreachable;
    if (!selectedUnreachable) {
      if (selectedDistance > std::numeric_limits<std::uint64_t>::max() -
                                 result.topologyRefinementHopSum)
        return initializerError("topology refinement hop sum exceeds u64");
      result.topologyRefinementHopSum += selectedDistance;
    }
  }
  result.changedComputeRoots = 0;
  for (PnrIndex realization = 0; realization < bindings.computeDecisionCount();
       ++realization)
    result.changedComputeRoots +=
        fixedChoices[realization] != result.choices[realization];

  const PnrIndex memoryOffset = bindings.computeDecisionCount();
  using MemoryKey = ::loom::fabric::FabricEntityId;
  std::map<MemoryKey, std::uint64_t> selectedMemoryCounts;
  const auto memoryPlacement =
      [&](const detail::SpatialMemoryBindingChoice &choice)
      -> llvm::Expected<const FrozenSpatialMemoryPlacement *> {
    if (choice.placement >= realizations.memoryPlacements().size())
      return initializerError("memory preference resolved a foreign placement");
    return &realizations.memoryPlacements()[choice.placement];
  };
  const auto countMemory =
      [&](const FrozenSpatialMemoryPlacement &placement) -> llvm::Error {
    std::uint64_t &count = selectedMemoryCounts[placement.memory.id()];
    if (count == std::numeric_limits<std::uint64_t>::max())
      return initializerError("memory preference count overflows u64");
    result.maximumMemorySelections =
        std::max(result.maximumMemorySelections, ++count);
    result.selectedTemporalMemoryRoots +=
        placement.schedule == ::fabric::Schedule::Temporal;
    return llvm::Error::success();
  };

  for (PnrIndex memory = 0; memory < bindings.memoryDecisionCount(); ++memory) {
    const PnrIndex decision = memoryOffset + memory;
    if (!participatesInHardRelation(decision))
      continue;
    const auto choices = bindings.memoryChoices(memory);
    const PnrIndex selected = result.choices[decision];
    if (selected >= choices.size())
      return initializerError(
          "root relation baseline selected a foreign memory choice");
    auto placement = memoryPlacement(choices[selected]);
    if (!placement)
      return placement.takeError();
    fixedChoices[decision] = selected;
    if (llvm::Error error = countMemory(**placement))
      return std::move(error);
  }

  for (PnrIndex memory = 0; memory < bindings.memoryDecisionCount(); ++memory) {
    const PnrIndex decision = memoryOffset + memory;
    if (fixedChoices[decision] != getInvalidPnrIndex())
      continue;
    const auto choices = bindings.memoryChoices(memory);
    const PnrIndex baselineChoice = result.choices[decision];
    if (choices.empty() || baselineChoice >= choices.size())
      return initializerError(
          "root relation baseline selected a foreign memory choice");

    PnrIndex selected = baselineChoice;
    auto selectedScore = std::make_tuple(
        std::numeric_limits<std::uint64_t>::max(),
        std::numeric_limits<std::uint64_t>::max(),
        std::numeric_limits<std::uint8_t>::max(), choices.size());
    std::uint64_t selectedCount = std::numeric_limits<std::uint64_t>::max();
    std::uint64_t selectedSchedulePressure =
        std::numeric_limits<std::uint64_t>::max();
    bool selectedTemporal = false;
    const PnrIndex origin = preferenceOrigin(baselineChoice);
    for (std::size_t rank = 0; rank != choices.size(); ++rank) {
      const PnrIndex local = static_cast<PnrIndex>(
          (static_cast<std::size_t>(origin) + rank) % choices.size());
      auto placement = memoryPlacement(choices[local]);
      if (!placement)
        return placement.takeError();
      const auto found = selectedMemoryCounts.find((*placement)->memory.id());
      const std::uint64_t count =
          found == selectedMemoryCounts.end() ? 0 : found->second;
      const bool temporal =
          (*placement)->schedule == ::fabric::Schedule::Temporal;
      const std::uint64_t schedulePressure =
          problem.schedulePressure().memoryPlacementContribution(
              choices[local].placement);
      // A Spatial memory occurrence serves one static realization per
      // operation port, so occurrence load is the physical pressure the seed
      // must spread before it optimizes schedule quality.
      const auto score = std::make_tuple(
          count, schedulePressure, static_cast<std::uint8_t>(temporal ? 0 : 1),
          rank);
      if (score < selectedScore) {
        selected = local;
        selectedScore = score;
        selectedCount = count;
        selectedSchedulePressure = schedulePressure;
        selectedTemporal = temporal;
      }
    }
    auto placement = memoryPlacement(choices[selected]);
    if (!placement)
      return placement.takeError();
    fixedChoices[decision] = selected;
    result.changedMemoryRoots += selected != baselineChoice;
    if (llvm::Error error = countMemory(**placement))
      return std::move(error);
    loom::mapping_debug::emit(
        loom::mapping_debug::Level::Detail,
        loom::mapping_debug::Stage::SpatialPnr,
        loom::mapping_debug::Event::ContextChoice,
        [&](llvm::json::Object &fields) {
          fields["operation"] = "initializer_memory_root_choice";
          fields["attempt"] = attemptOrdinal;
          fields["realization"] = memory;
          fields["baseline_choice"] = baselineChoice;
          fields["selected_choice"] = selected;
          fields["selected_count_before"] = selectedCount;
          fields["selected_static_schedule_pressure"] =
              selectedSchedulePressure;
          fields["selected_temporal"] = selectedTemporal;
          fields["memory_ref"] =
              loom::fabric::printFabricRef((*placement)->memory);
        });
  }

  for (const auto &[occurrence, count] : selectedOccurrenceCounts) {
    (void)occurrence;
    result.distinctComputeOccurrences += count != 0;
  }
  result.distinctMemoryOccurrences = selectedMemoryCounts.size();

  const auto &ports = problem.ports();
  const auto attachmentOptions = ports.attachmentOptions();
  const auto placementDomains = ports.placementDomains();
  std::map<PnrIndex, std::uint64_t> endpointSelectionCounts;
  const auto countEndpoint = [&](PnrIndex option) -> llvm::Error {
    if (option >= attachmentOptions.size())
      return initializerError(
          "attachment preference selected a foreign attachment option");
    std::uint64_t &count =
        endpointSelectionCounts[attachmentOptions[option].endpoint];
    if (count == std::numeric_limits<std::uint64_t>::max())
      return initializerError("attachment preference count overflows u64");
    result.maximumEndpointSelections =
        std::max(result.maximumEndpointSelections, ++count);
    return llvm::Error::success();
  };
  const auto selectedPlacement =
      [&](const FrozenSpatialPortDemand &demand) -> llvm::Expected<PnrIndex> {
    if (demand.kind == FrozenSpatialPortDemandKind::Compute) {
      if (demand.realization >= bindings.computeDecisionCount())
        return initializerError(
            "port preference names a foreign compute realization");
      const PnrIndex selected = fixedChoices[demand.realization];
      const auto choices = bindings.computeChoices(demand.realization);
      if (selected >= choices.size())
        return initializerError(
            "port preference compute choice is out of range");
      return choices[selected].placement;
    }
    if (demand.realization >= bindings.memoryDecisionCount())
      return initializerError(
          "port preference names a foreign memory realization");
    const PnrIndex decision = memoryOffset + demand.realization;
    const PnrIndex selected = fixedChoices[decision];
    const auto choices = bindings.memoryChoices(demand.realization);
    if (selected >= choices.size())
      return initializerError("port preference memory choice is out of range");
    return choices[selected].placement;
  };

  for (PnrIndex demand = 0; demand < ports.portDemands().size(); ++demand) {
    const PnrIndex decision = bindings.portDecisionOffset() + demand;
    const auto choices = bindings.portAttachmentChoices(demand);
    const PnrIndex baselineChoice = result.choices[decision];
    if (choices.empty() || baselineChoice >= choices.size())
      return initializerError(
          "port preference baseline choice is out of range");
    auto placement = selectedPlacement(ports.portDemands()[demand]);
    if (!placement)
      return placement.takeError();

    PnrIndex selected = getInvalidPnrIndex();
    auto selectedScore = std::make_pair(
        std::numeric_limits<std::uint64_t>::max(), choices.size());
    const PnrIndex origin = preferenceOrigin(baselineChoice);
    for (std::size_t rank = 0; rank < choices.size(); ++rank) {
      const PnrIndex local = static_cast<PnrIndex>(
          (static_cast<std::size_t>(origin) + rank) % choices.size());
      const PnrIndex option = choices[local];
      if (option >= attachmentOptions.size())
        return initializerError(
            "port preference attachment choice is out of range");
      const FrozenSpatialAttachmentOption &record = attachmentOptions[option];
      if (record.ownerKind !=
              FrozenSpatialAttachmentOwnerKind::PlacementDomain ||
          record.owner >= placementDomains.size())
        return initializerError(
            "port preference attachment owner is malformed");
      if (placementDomains[record.owner].placement != *placement)
        continue;
      if (participatesInHardRelation(decision) && local != baselineChoice)
        continue;
      const auto found = endpointSelectionCounts.find(record.endpoint);
      const std::uint64_t count =
          found == endpointSelectionCounts.end() ? 0 : found->second;
      const auto score = std::make_pair(count, rank);
      if (score < selectedScore) {
        selected = local;
        selectedScore = score;
      }
    }
    if (selected == getInvalidPnrIndex())
      return initializerError(
          "port preference selected placement has no attachment choice");
    fixedChoices[decision] = selected;
    result.changedPortAttachments += selected != baselineChoice;
    if (llvm::Error error = countEndpoint(choices[selected]))
      return std::move(error);
  }

  for (PnrIndex boundary = 0; boundary < bindings.graphBoundaryDecisionCount();
       ++boundary) {
    const PnrIndex decision = bindings.graphBoundaryDecisionOffset() + boundary;
    const auto choices = bindings.graphBoundaryAttachmentChoices(boundary);
    const PnrIndex baselineChoice = result.choices[decision];
    if (choices.empty() || baselineChoice >= choices.size())
      return initializerError(
          "attachment preference graph-boundary baseline is out of range");
    const FrozenSpatialGraphBoundary &boundaryRecord =
        ports.graphBoundaries()[boundary];
    if (boundaryRecord.logicalNet >= problem.transfers().logicalNets().size())
      return initializerError(
          "attachment preference graph boundary names a foreign logical net");

    const auto selectedTerminalOption =
        [&](FrozenSpatialTerminalBinding terminal) -> llvm::Expected<PnrIndex> {
      PnrIndex terminalDecision = 0;
      llvm::ArrayRef<PnrIndex> terminalChoices;
      if (terminal.kind == FrozenSpatialTerminalBindingKind::PortDemand) {
        if (terminal.index >= bindings.portDecisionCount())
          return initializerError(
              "attachment preference names a foreign port terminal");
        terminalDecision = bindings.portDecisionOffset() + terminal.index;
        terminalChoices = bindings.portAttachmentChoices(terminal.index);
      } else {
        if (terminal.index >= bindings.graphBoundaryDecisionCount())
          return initializerError(
              "attachment preference names a foreign boundary terminal");
        terminalDecision =
            bindings.graphBoundaryDecisionOffset() + terminal.index;
        terminalChoices =
            bindings.graphBoundaryAttachmentChoices(terminal.index);
      }
      PnrIndex selected = fixedChoices[terminalDecision];
      if (selected == getInvalidPnrIndex())
        selected = result.choices[terminalDecision];
      if (selected >= terminalChoices.size())
        return initializerError(
            "attachment preference terminal choice is out of range");
      return terminalChoices[selected];
    };

    fixedOptionScratch.clear();
    const auto &transfers = problem.transfers();
    const auto sources = transfers.logicalNetSourceBindings();
    const auto sinks = transfers.logicalNetSinkBindings();
    const FrozenSpatialLogicalNet &net =
        transfers.logicalNets()[boundaryRecord.logicalNet];
    if (boundaryRecord.logicalNet >= sources.size() ||
        net.sinkOffset > sinks.size() ||
        net.sinkCount > sinks.size() - net.sinkOffset)
      return initializerError(
          "attachment preference logical-net terminals are malformed");
    const FrozenSpatialTerminalBinding source =
        sources[boundaryRecord.logicalNet];
    const bool boundaryIsSource =
        source.kind == FrozenSpatialTerminalBindingKind::GraphBoundary &&
        source.index == boundary;
    if (boundaryIsSource) {
      for (FrozenSpatialTerminalBinding sink :
           sinks.slice(net.sinkOffset, net.sinkCount)) {
        auto option = selectedTerminalOption(sink);
        if (!option)
          return option.takeError();
        if (*option >= attachmentOptions.size())
          return initializerError(
              "attachment preference sink option is out of range");
        fixedOptionScratch.push_back(attachmentOptions[*option]);
      }
    } else {
      bool foundBoundary = false;
      for (FrozenSpatialTerminalBinding sink :
           sinks.slice(net.sinkOffset, net.sinkCount))
        foundBoundary |=
            sink.kind == FrozenSpatialTerminalBindingKind::GraphBoundary &&
            sink.index == boundary;
      if (!foundBoundary)
        return initializerError(
            "attachment preference boundary is absent from its logical net");
      auto option = selectedTerminalOption(source);
      if (!option)
        return option.takeError();
      if (*option >= attachmentOptions.size())
        return initializerError(
            "attachment preference source option is out of range");
      fixedOptionScratch.push_back(attachmentOptions[*option]);
    }
    if (fixedOptionScratch.empty())
      return initializerError(
          "attachment preference boundary has no opposite terminal");

    std::vector<std::uint64_t> unreachableCounts(choices.size(), 0);
    std::vector<std::uint64_t> distanceSums(choices.size(), 0);
    constexpr std::uint32_t unreachable =
        std::numeric_limits<std::uint32_t>::max();
    for (const FrozenSpatialAttachmentOption &peer : fixedOptionScratch) {
      if (llvm::Error error = fillTopologyHopDistances(
              problem, llvm::ArrayRef(peer), boundaryRecord.payloadWidthBits,
              !boundaryIsSource, hopDistances, hopWorklist))
        return std::move(error);
      for (std::size_t local = 0; local < choices.size(); ++local) {
        const PnrIndex option = choices[local];
        if (option >= attachmentOptions.size() ||
            attachmentOptions[option].endpoint >= hopDistances.size())
          return initializerError(
              "attachment preference boundary option is out of range");
        const PnrIndex endpoint = attachmentOptions[option].endpoint;
        const std::uint32_t distance = hopDistances[endpoint];
        if (distance == unreachable) {
          ++unreachableCounts[local];
          continue;
        }
        if (distance >
            std::numeric_limits<std::uint64_t>::max() - distanceSums[local])
          return initializerError(
              "attachment preference boundary distance exceeds u64");
        distanceSums[local] += distance;
      }
    }

    PnrIndex selected = baselineChoice;
    auto selectedScore = std::make_tuple(
        std::numeric_limits<std::uint64_t>::max(),
        std::numeric_limits<std::uint64_t>::max(),
        std::numeric_limits<std::uint64_t>::max(), choices.size());
    const PnrIndex origin = preferenceOrigin(baselineChoice);
    for (std::size_t rank = 0; rank < choices.size(); ++rank) {
      const PnrIndex local = static_cast<PnrIndex>(
          (static_cast<std::size_t>(origin) + rank) % choices.size());
      const PnrIndex option = choices[local];
      if (option >= attachmentOptions.size())
        return initializerError(
            "attachment preference graph-boundary choice is out of range");
      const FrozenSpatialAttachmentOption &record = attachmentOptions[option];
      if (record.ownerKind != FrozenSpatialAttachmentOwnerKind::GraphBoundary ||
          record.owner != boundary)
        return initializerError(
            "attachment preference graph-boundary owner is malformed");
      if (participatesInHardRelation(decision) && local != baselineChoice)
        continue;
      const auto found = endpointSelectionCounts.find(record.endpoint);
      const std::uint64_t count =
          found == endpointSelectionCounts.end() ? 0 : found->second;
      const auto score = std::make_tuple(unreachableCounts[local],
                                         distanceSums[local], count, rank);
      if (score < selectedScore) {
        selected = local;
        selectedScore = score;
      }
    }
    fixedChoices[decision] = selected;
    result.changedGraphBoundaryAttachments += selected != baselineChoice;
    if (llvm::Error error = countEndpoint(choices[selected]))
      return std::move(error);
  }

  if (result.changedComputeRoots == 0 && result.changedMemoryRoots == 0 &&
      result.changedPortAttachments == 0 &&
      result.changedGraphBoundaryAttachments == 0) {
    emitPreference();
    return result;
  }
  const std::uint64_t remaining = assignmentLimit - result.assignmentAttempts;
  if (remaining == 0) {
    result.status = "work_limit";
    emitPreference();
    return result;
  }

  auto preferred =
      solver.solveCanonicalWithFixedChoices(remaining, fixedChoices);
  const std::uint64_t preferenceAttempts = solver.assignmentAttempts();
  if (preferenceAttempts > remaining)
    return initializerError("compute preference exceeded its remaining work");
  result.assignmentAttempts += preferenceAttempts;
  if (!preferred) {
    llvm::StringRef fallbackStatus = "invalid";
    llvm::Error unhandled = llvm::handleErrors(
        preferred.takeError(),
        [&](const InitializerRelationSolveFailure &failure) -> llvm::Error {
          if (failure.kind() ==
              InitializerRelationSolveFailureKind::FixedRootInfeasible) {
            fallbackStatus = "fixed_root_infeasible";
            return llvm::Error::success();
          }
          if (failure.kind() ==
              InitializerRelationSolveFailureKind::WorkLimit) {
            fallbackStatus = "work_limit";
            return llvm::Error::success();
          }
          std::string message;
          llvm::raw_string_ostream stream(message);
          failure.log(stream);
          return llvm::make_error<InitializerRelationSolveFailure>(
              failure.kind(), std::move(message));
        });
    if (unhandled)
      return std::move(unhandled);
    result.status = fallbackStatus;
    emitPreference();
    return result;
  }

  result.choices = std::move(preferred->choices);
  result.applied = true;
  result.status = "applied";
  emitPreference();
  return result;
}

class SpatialInitializerAttemptBuilder final {
public:
  SpatialInitializerAttemptBuilder(
      const FrozenSpatialPnrProblem &problem,
      DeterministicPnrRandomStream *diversificationStream,
      std::uint64_t assignmentLimit, std::uint64_t assignmentAttempts,
      std::vector<SpatialComputeBindingSelection> computeBindings,
      std::vector<SpatialMemoryBindingSelection> memoryBindings,
      std::vector<PnrIndex> portAttachments,
      std::vector<PnrIndex> graphBoundaryAttachments)
      : problem_(problem), diversificationStream_(diversificationStream),
        assignmentLimit_(assignmentLimit),
        assignmentAttempts_(assignmentAttempts),
        computeBindings_(std::move(computeBindings)),
        memoryBindings_(std::move(memoryBindings)),
        portAttachments_(std::move(portAttachments)),
        graphBoundaryAttachments_(std::move(graphBoundaryAttachments)) {}

  llvm::Error build() {
    if (llvm::Error error = prepareDecisionInventory())
      return error;
    auto completed = search();
    if (!completed)
      return completed.takeError();
    if (!*completed)
      return initializerFailure(
          InitializerRelationSolveFailureKind::FixedRootInfeasible,
          "has no complete dependent-decision assignment");
    return llvm::Error::success();
  }

  SpatialCandidateInitialization initialization() const {
    return {computeBindings_,      memoryBindings_,
            portAttachments_,      graphBoundaryAttachments_,
            memoryOperationPlans_, logicalMemoryBindings_,
            memoryUseDispatches_,  memoryExposureSelections_};
  }

  std::uint64_t assignmentAttempts() const { return assignmentAttempts_; }

private:
  enum class DecisionKind : std::uint8_t {
    MemoryOperationPlan,
    LogicalMemoryBinding,
    MemoryUseDispatch,
    MemoryExposure,
  };

  struct DecisionRecord final {
    DecisionKind kind = DecisionKind::MemoryOperationPlan;
    PnrIndex index = 0;
    std::size_t choiceOffset = 0;
    PnrIndex choiceCapacity = 0;
  };

  const FrozenSpatialMemoryOperationHandshakeDomain *
  memoryPlanDomain(PnrIndex actor) const {
    const auto &realizations = problem_.realizations();
    if (actor >= realizations.memoryActors().size())
      return nullptr;
    const PnrIndex realization = realizations.memoryActorRealizations()[actor];
    const auto &owner = realizations.memoryRealizations()[realization];
    const PnrIndex placement = memoryBindings_[realization].placement;
    if (actor < owner.actorOffset ||
        actor - owner.actorOffset >= owner.actorCount)
      return nullptr;
    const PnrIndex domainOffset =
        problem_.handshake().memoryPlacementDomainOffsets()[placement];
    return &problem_.handshake().memoryOperationDomains()[domainOffset + actor -
                                                          owner.actorOffset];
  }

  llvm::Error appendDecision(DecisionKind kind, PnrIndex index,
                             PnrIndex choiceCapacity) {
    if (choiceCapacity >
        std::numeric_limits<std::size_t>::max() - choiceStorageSize_)
      return initializerError("dependent choice storage size overflows");
    decisions_.push_back({kind, index, choiceStorageSize_, choiceCapacity});
    choiceStorageSize_ += choiceCapacity;
    return llvm::Error::success();
  }

  llvm::Error prepareDecisionInventory() {
    const auto &ports = problem_.ports();
    const auto &realizations = problem_.realizations();
    const auto &memory = problem_.memory();

    if (portAttachments_.size() != ports.portDemands().size())
      return initializerError("root decision solver omitted PortAttachments");
    if (graphBoundaryAttachments_.size() != ports.graphBoundaries().size())
      return initializerError(
          "root decision solver omitted graph-boundary attachments");
    memoryOperationPlans_.assign(realizations.memoryActors().size(),
                                 getInvalidPnrIndex());
    logicalMemoryBindings_.assign(memory.logicalBindings().size(), {});
    for (auto &binding : logicalMemoryBindings_)
      binding.target = getInvalidPnrIndex();
    memoryUseDispatches_.assign(memory.rootedUses().size(),
                                getInvalidPnrIndex());
    memoryExposureSelections_.assign(memory.exposures().size(),
                                     getInvalidPnrIndex());

    for (PnrIndex actor = 0; actor < realizations.memoryActors().size();
         ++actor) {
      const auto *domain = memoryPlanDomain(actor);
      if (!domain)
        return initializerError(
            "memory actor has no domain for its selected placement");
      if (llvm::Error error = appendDecision(DecisionKind::MemoryOperationPlan,
                                             actor, domain->planCount))
        return error;
    }
    for (PnrIndex binding = 0; binding < memory.logicalBindings().size();
         ++binding) {
      auto capacity =
          problem_.memoryConstraints().logicalBindingChoiceCapacity(binding);
      if (!capacity)
        return capacity.takeError();
      if (llvm::Error error = appendDecision(DecisionKind::LogicalMemoryBinding,
                                             binding, *capacity))
        return error;
    }
    for (PnrIndex use = 0; use < memory.rootedUses().size(); ++use) {
      const auto *domain =
          dispatchDomain(problem_, memoryBindings_, memory.rootedUses()[use]);
      if (!domain)
        return initializerError(
            "memory use has no domain for its selected placement");
      if (llvm::Error error = appendDecision(DecisionKind::MemoryUseDispatch,
                                             use, domain->optionCount))
        return error;
    }
    for (PnrIndex exposure = 0; exposure < memory.exposures().size();
         ++exposure) {
      if (memory.exposureOptions().size() > getPnrIndexMax())
        return initializerError("memory exposure domain is too large");
      if (llvm::Error error = appendDecision(
              DecisionKind::MemoryExposure, exposure,
              static_cast<PnrIndex>(memory.exposureOptions().size())))
        return error;
    }

    canonicalChoices_.resize(choiceStorageSize_);
    choiceOrder_.resize(choiceStorageSize_);
    choiceFenwick_.resize(choiceStorageSize_);
    logicalMemoryChoices_.resize(choiceStorageSize_);
    assignmentJournal_.reserve(decisions_.size());
    compatibilityChoices_.reserve(memory.dispatchOptions().size());
    groupCompatibilityChoices_.reserve(memory.dispatchOptions().size());
    return llvm::Error::success();
  }

  llvm::Error consumeAssignmentAttempt() {
    if (assignmentAttempts_ == assignmentLimit_)
      return initializerFailure(InitializerRelationSolveFailureKind::WorkLimit,
                                "exhausted its assignment work limit");
    ++assignmentAttempts_;
    return llvm::Error::success();
  }

  bool
  targetSupportsBinding(PnrIndex binding,
                        const FrozenSpatialMemoryBindingTargetOption &target) {
    const auto &memory = problem_.memory();
    const auto patterns = problem_.capacity().memoryDispatchOptionPatterns();
    const auto uses =
        memory.bindingUses().slice(memory.bindingUseOffsets()[binding],
                                   memory.bindingUseOffsets()[binding + 1] -
                                       memory.bindingUseOffsets()[binding]);
    for (PnrIndex use : uses) {
      const PnrIndex group = memory.rootedUseServiceGroups()[use];
      if (group != getInvalidPnrIndex()) {
        if (group >= memory.serviceUseGroups().size())
          return false;
        const FrozenSpatialMemoryServiceUseGroup &record =
            memory.serviceUseGroups()[group];
        const auto members =
            memory.serviceGroupUses().slice(record.useOffset, record.useCount);
        if (members.empty() || members.front() != use)
          continue;
        compatibilityChoices_.clear();
        appendMatchingDispatches(problem_, memoryBindings_,
                                 memory.rootedUses()[members.front()], &target,
                                 compatibilityChoices_);
        bool commonPattern = false;
        for (PnrIndex option : compatibilityChoices_) {
          const PnrIndex pattern = patterns[option];
          bool common = true;
          for (PnrIndex member : members) {
            groupCompatibilityChoices_.clear();
            appendMatchingDispatches(problem_, memoryBindings_,
                                     memory.rootedUses()[member], &target,
                                     groupCompatibilityChoices_);
            if (!llvm::any_of(groupCompatibilityChoices_,
                              [&](PnrIndex memberOption) {
                                return patterns[memberOption] == pattern;
                              })) {
              common = false;
              break;
            }
          }
          if (common) {
            commonPattern = true;
            break;
          }
        }
        if (!commonPattern)
          return false;
        continue;
      }
      compatibilityChoices_.clear();
      appendMatchingDispatches(problem_, memoryBindings_,
                               memory.rootedUses()[use], &target,
                               compatibilityChoices_);
      if (compatibilityChoices_.empty())
        return false;
    }
    const auto exposures = memory.bindingExposures().slice(
        memory.bindingExposureOffsets()[binding],
        memory.bindingExposureOffsets()[binding + 1] -
            memory.bindingExposureOffsets()[binding]);
    for (PnrIndex exposure : exposures) {
      (void)exposure;
      bool supported = false;
      for (const auto &option : memory.exposureOptions())
        supported |= detail::memoryExposureMatchesTarget(target, option);
      if (!supported)
        return false;
    }
    return !uses.empty() || !exposures.empty();
  }

  bool decisionAssigned(const DecisionRecord &decision) const {
    switch (decision.kind) {
    case DecisionKind::MemoryOperationPlan:
      return memoryOperationPlans_[decision.index] != getInvalidPnrIndex();
    case DecisionKind::LogicalMemoryBinding:
      return logicalMemoryBindings_[decision.index].target !=
             getInvalidPnrIndex();
    case DecisionKind::MemoryUseDispatch:
      return memoryUseDispatches_[decision.index] != getInvalidPnrIndex();
    case DecisionKind::MemoryExposure:
      return memoryExposureSelections_[decision.index] != getInvalidPnrIndex();
    }
    llvm_unreachable("unknown Spatial initializer decision kind");
  }

  bool decisionActive(const DecisionRecord &decision) const {
    const auto &memory = problem_.memory();
    switch (decision.kind) {
    case DecisionKind::LogicalMemoryBinding:
      return decision.index == 0 ||
             logicalMemoryBindings_[decision.index - 1].target !=
                 getInvalidPnrIndex();
    case DecisionKind::MemoryUseDispatch: {
      const auto binding = memory.rootedUses()[decision.index].logicalBinding;
      return !binding ||
             logicalMemoryBindings_[*binding].target != getInvalidPnrIndex();
    }
    case DecisionKind::MemoryExposure:
      return logicalMemoryBindings_[memory.exposures()[decision.index]
                                        .logicalBinding]
                 .target != getInvalidPnrIndex();
    default:
      return true;
    }
  }

  llvm::Expected<PnrIndex> fillChoices(const DecisionRecord &decision) {
    auto choices = llvm::MutableArrayRef(canonicalChoices_)
                       .slice(decision.choiceOffset, decision.choiceCapacity);
    PnrIndex count = 0;
    switch (decision.kind) {
    case DecisionKind::MemoryOperationPlan: {
      const auto *domain = memoryPlanDomain(decision.index);
      if (!domain)
        return initializerError("memory operation domain disappeared");
      for (PnrIndex local = 0; local < domain->planCount; ++local)
        choices[count++] = domain->planOffset + local;
      break;
    }
    case DecisionKind::LogicalMemoryBinding: {
      const auto &memory = problem_.memory();
      auto values = llvm::MutableArrayRef(logicalMemoryChoices_)
                        .slice(decision.choiceOffset, decision.choiceCapacity);
      auto generated =
          problem_.memoryConstraints().collectLogicalBindingChoices(
              decision.index, logicalMemoryBindings_, values);
      if (!generated)
        return generated.takeError();
      for (PnrIndex choice = 0; choice < *generated; ++choice) {
        const auto value = values[choice];
        if (value.target >= memory.bindingTargets().size())
          return initializerError(
              "memory constraint owner produced a foreign target");
        if (!targetSupportsBinding(decision.index,
                                   memory.bindingTargets()[value.target]))
          continue;
        values[count] = value;
        choices[count] = count;
        ++count;
      }
      break;
    }
    case DecisionKind::MemoryUseDispatch: {
      const auto &memory = problem_.memory();
      const auto &use = memory.rootedUses()[decision.index];
      const auto *domain = dispatchDomain(problem_, memoryBindings_, use);
      if (!domain)
        return initializerError("memory dispatch domain disappeared");
      const FrozenSpatialMemoryBindingTargetOption *target = nullptr;
      if (use.logicalBinding)
        target =
            &memory.bindingTargets()[logicalMemoryBindings_[*use.logicalBinding]
                                         .target];
      compatibilityChoices_.clear();
      appendMatchingDispatches(problem_, memoryBindings_, use, target,
                               compatibilityChoices_);
      std::optional<PnrIndex> requiredPattern;
      const PnrIndex group = memory.rootedUseServiceGroups()[decision.index];
      const auto patterns = problem_.capacity().memoryDispatchOptionPatterns();
      if (group != getInvalidPnrIndex()) {
        if (group >= memory.serviceUseGroups().size())
          return initializerError("memory dispatch has a foreign use group");
        const auto &record = memory.serviceUseGroups()[group];
        for (PnrIndex member : memory.serviceGroupUses().slice(
                 record.useOffset, record.useCount)) {
          const PnrIndex selected = memoryUseDispatches_[member];
          if (selected == getInvalidPnrIndex())
            continue;
          const PnrIndex pattern = patterns[selected];
          if (requiredPattern && *requiredPattern != pattern)
            return PnrIndex{0};
          requiredPattern = pattern;
        }
      }
      for (PnrIndex option : compatibilityChoices_)
        if (!requiredPattern || patterns[option] == *requiredPattern)
          choices[count++] = option;
      break;
    }
    case DecisionKind::MemoryExposure: {
      const auto &memory = problem_.memory();
      const auto &exposure = memory.exposures()[decision.index];
      const auto &target =
          memory.bindingTargets()
              [logicalMemoryBindings_[exposure.logicalBinding].target];
      for (PnrIndex option = 0; option < memory.exposureOptions().size();
           ++option)
        if (detail::memoryExposureMatchesTarget(
                target, memory.exposureOptions()[option]))
          choices[count++] = option;
      break;
    }
    }
    if (count > decision.choiceCapacity)
      return initializerError("dependent choice domain exceeds its frozen cap");
    return count;
  }

  void assignDecision(std::size_t decisionOrdinal, PnrIndex choice) {
    const auto &decision = decisions_[decisionOrdinal];
    switch (decision.kind) {
    case DecisionKind::MemoryOperationPlan:
      memoryOperationPlans_[decision.index] = choice;
      break;
    case DecisionKind::LogicalMemoryBinding: {
      logicalMemoryBindings_[decision.index] =
          logicalMemoryChoices_[decision.choiceOffset + choice];
      break;
    }
    case DecisionKind::MemoryUseDispatch:
      memoryUseDispatches_[decision.index] = choice;
      break;
    case DecisionKind::MemoryExposure:
      memoryExposureSelections_[decision.index] = choice;
      break;
    }
    assignmentJournal_.push_back(decisionOrdinal);
  }

  void rollback(std::size_t journalMark) {
    while (assignmentJournal_.size() > journalMark) {
      const std::size_t ordinal = assignmentJournal_.back();
      assignmentJournal_.pop_back();
      const auto &decision = decisions_[ordinal];
      switch (decision.kind) {
      case DecisionKind::MemoryOperationPlan:
        memoryOperationPlans_[decision.index] = getInvalidPnrIndex();
        break;
      case DecisionKind::LogicalMemoryBinding: {
        logicalMemoryBindings_[decision.index] = {};
        logicalMemoryBindings_[decision.index].target = getInvalidPnrIndex();
        break;
      }
      case DecisionKind::MemoryUseDispatch:
        memoryUseDispatches_[decision.index] = getInvalidPnrIndex();
        break;
      case DecisionKind::MemoryExposure:
        memoryExposureSelections_[decision.index] = getInvalidPnrIndex();
        break;
      }
    }
  }

  bool completeAssignmentValid() {
    if (llvm::Error error =
            problem_.memoryConstraints().verify(logicalMemoryBindings_)) {
      llvm::consumeError(std::move(error));
      return false;
    }
    const auto &memory = problem_.memory();
    const auto patterns = problem_.capacity().memoryDispatchOptionPatterns();
    for (const FrozenSpatialMemoryServiceUseGroup &record :
         memory.serviceUseGroups()) {
      std::optional<PnrIndex> selectedPattern;
      for (PnrIndex use :
           memory.serviceGroupUses().slice(record.useOffset, record.useCount)) {
        const PnrIndex option = memoryUseDispatches_[use];
        if (option >= patterns.size())
          return false;
        const PnrIndex pattern = patterns[option];
        if (selectedPattern && *selectedPattern != pattern)
          return false;
        selectedPattern = pattern;
      }
    }
    return true;
  }

  llvm::Expected<bool> search() {
    while (true) {
      bool allAssigned = true;
      bool propagated = false;
      for (std::size_t ordinal = 0; ordinal < decisions_.size(); ++ordinal) {
        const auto &decision = decisions_[ordinal];
        if (decisionAssigned(decision))
          continue;
        allAssigned = false;
        if (!decisionActive(decision))
          continue;
        auto count = fillChoices(decision);
        if (!count)
          return count.takeError();
        if (*count == 0)
          return false;
        if (*count == 1) {
          assignDecision(ordinal, canonicalChoices_[decision.choiceOffset]);
          propagated = true;
          break;
        }
      }
      if (allAssigned) {
        return completeAssignmentValid();
      }
      if (!propagated)
        break;
    }

    std::size_t selected = std::numeric_limits<std::size_t>::max();
    PnrIndex selectedCount = getInvalidPnrIndex();
    for (std::size_t ordinal = 0; ordinal < decisions_.size(); ++ordinal) {
      const auto &decision = decisions_[ordinal];
      if (decisionAssigned(decision) || !decisionActive(decision))
        continue;
      auto count = fillChoices(decision);
      if (!count)
        return count.takeError();
      if (*count == 0)
        return false;
      if (*count < selectedCount) {
        selected = ordinal;
        selectedCount = *count;
      }
    }
    if (selected == std::numeric_limits<std::size_t>::max())
      return initializerError("dependent decision prerequisites form a cycle");

    const auto &decision = decisions_[selected];
    auto count = fillChoices(decision);
    if (!count)
      return count.takeError();
    auto canonical =
        llvm::ArrayRef(canonicalChoices_).slice(decision.choiceOffset, *count);
    auto order = llvm::MutableArrayRef(choiceOrder_)
                     .slice(decision.choiceOffset, *count);
    if (llvm::Error error = detail::buildInitializerChoiceOrder(
            canonical, diversificationStream_, order,
            llvm::MutableArrayRef(choiceFenwick_)
                .slice(decision.choiceOffset, *count)))
      return std::move(error);

    for (PnrIndex choice : order) {
      if (llvm::Error error = consumeAssignmentAttempt())
        return std::move(error);
      const std::size_t journalMark = assignmentJournal_.size();
      assignDecision(selected, choice);
      auto completed = search();
      if (!completed)
        return completed.takeError();
      if (*completed)
        return true;
      rollback(journalMark);
    }
    return false;
  }

  const FrozenSpatialPnrProblem &problem_;
  DeterministicPnrRandomStream *diversificationStream_ = nullptr;
  std::uint64_t assignmentLimit_ = 0;
  std::uint64_t assignmentAttempts_ = 0;
  std::vector<SpatialComputeBindingSelection> computeBindings_;
  std::vector<SpatialMemoryBindingSelection> memoryBindings_;
  std::vector<PnrIndex> portAttachments_;
  std::vector<PnrIndex> graphBoundaryAttachments_;
  std::vector<PnrIndex> memoryOperationPlans_;
  std::vector<SpatialLogicalMemoryBindingSelection> logicalMemoryBindings_;
  std::vector<PnrIndex> memoryUseDispatches_;
  std::vector<PnrIndex> memoryExposureSelections_;
  std::vector<DecisionRecord> decisions_;
  std::vector<PnrIndex> canonicalChoices_;
  std::vector<PnrIndex> choiceOrder_;
  std::vector<PnrIndex> choiceFenwick_;
  std::vector<PnrIndex> compatibilityChoices_;
  std::vector<PnrIndex> groupCompatibilityChoices_;
  std::vector<SpatialLogicalMemoryBindingSelection> logicalMemoryChoices_;
  std::vector<std::size_t> assignmentJournal_;
  std::size_t choiceStorageSize_ = 0;
};

} // namespace

llvm::Expected<SpatialCandidateInitializerAttempt>
loom::pnr::createSpatialCandidateInitializerAttempt(
    FrozenSpatialPnrProblemHandle problem, std::uint32_t attemptOrdinal,
    std::uint64_t &assignmentAttempts) {
  assignmentAttempts = 0;
  if (!problem)
    return initializerError("FrozenSpatialPnrProblem owner is null");
  const auto &policy = problem->config().policy();
  if (attemptOrdinal >= policy.search.initializer.seedAttemptCount)
    return initializerError("initializer attempt ordinal is out of range");

  const detail::SpatialBindingRelationModel &bindingRelations =
      problem->bindingRelations();
  if (const auto deferred = bindingRelations.deferredProjection())
    return initializerError(
        "hard equality or disjointness for projection '" +
        ::mapping::stringifySpatialConstraintProjection(*deferred) +
        "' requires its owning decision model");

  detail::InitializerRelationSolver relationSolver(
      bindingRelations.relations());
  std::optional<DeterministicPnrRandomStream> diversificationStream;
  if (attemptOrdinal != 0)
    diversificationStream.emplace(DeterministicPnrRandomStream::create(
        policy.determinism.masterSeed, attemptOrdinal,
        PnrRandomStreamPurpose::InitializerDiversification));
  auto relationChoices =
      diversificationStream
          ? relationSolver.solveDiversified(
                policy.search.initializer.assignmentAttemptLimitPerSeed,
                *diversificationStream)
          : relationSolver.solveCanonical(
                policy.search.initializer.assignmentAttemptLimitPerSeed);
  assignmentAttempts = relationSolver.assignmentAttempts();
  if (!relationChoices)
    return relationChoices.takeError();

  auto preferred = preferScheduleAwareRootPlacements(
      *problem, attemptOrdinal, relationSolver, std::move(*relationChoices),
      policy.search.initializer.assignmentAttemptLimitPerSeed);
  if (!preferred)
    return preferred.takeError();
  assignmentAttempts = preferred->assignmentAttempts;
  const std::vector<PnrIndex> &rootChoices = preferred->choices;

  const FrozenSpatialRealizationIndex &realizations = problem->realizations();
  std::vector<SpatialComputeBindingSelection> computeBindings;
  computeBindings.reserve(realizations.computeRealizations().size());
  for (PnrIndex realization = 0;
       realization < realizations.computeRealizations().size(); ++realization) {
    const auto choices = bindingRelations.computeChoices(realization);
    const PnrIndex selected = rootChoices[realization];
    if (selected >= choices.size())
      return initializerError(
          "relation solver returned a foreign compute choice");
    computeBindings.push_back(
        {choices[selected].placement, choices[selected].instructionContext});
  }

  std::vector<SpatialMemoryBindingSelection> memoryBindings;
  memoryBindings.reserve(realizations.memoryRealizations().size());
  for (PnrIndex realization = 0;
       realization < realizations.memoryRealizations().size(); ++realization) {
    const auto choices = bindingRelations.memoryChoices(realization);
    const PnrIndex selected =
        rootChoices[bindingRelations.computeDecisionCount() + realization];
    if (selected >= choices.size())
      return initializerError(
          "relation solver returned a foreign memory choice");
    memoryBindings.push_back({choices[selected].placement});
  }

  std::vector<PnrIndex> portAttachments;
  portAttachments.reserve(problem->ports().portDemands().size());
  for (PnrIndex demand = 0; demand < problem->ports().portDemands().size();
       ++demand) {
    const auto choices = bindingRelations.portAttachmentChoices(demand);
    const PnrIndex selected =
        rootChoices[bindingRelations.portDecisionOffset() + demand];
    if (selected >= choices.size())
      return initializerError(
          "relation solver returned a foreign PortAttachment choice");
    portAttachments.push_back(choices[selected]);
  }
  std::vector<PnrIndex> graphBoundaryAttachments;
  graphBoundaryAttachments.reserve(problem->ports().graphBoundaries().size());
  for (PnrIndex boundary = 0;
       boundary < problem->ports().graphBoundaries().size(); ++boundary) {
    const auto choices =
        bindingRelations.graphBoundaryAttachmentChoices(boundary);
    const PnrIndex selected =
        rootChoices[bindingRelations.graphBoundaryDecisionOffset() + boundary];
    if (selected >= choices.size())
      return initializerError(
          "relation solver returned a foreign graph-boundary choice");
    graphBoundaryAttachments.push_back(choices[selected]);
  }

  if (loom::mapping_debug::enabled(loom::mapping_debug::Level::Decision)) {
    std::map<PnrIndex, std::uint64_t> endpointSelections;
    const auto options = problem->ports().attachmentOptions();
    const auto endpoints = problem->routing().routingEndpoints();
    std::uint64_t maximumSelections = 0;
    const auto countSelection =
        [&](PnrIndex option) -> llvm::Expected<PnrIndex> {
      if (option >= options.size() ||
          options[option].endpoint >= endpoints.size())
        return initializerError(
            "selected attachment diagnostic reference is out of range");
      const PnrIndex endpoint = options[option].endpoint;
      std::uint64_t &count = endpointSelections[endpoint];
      if (count == std::numeric_limits<std::uint64_t>::max())
        return initializerError(
            "selected attachment diagnostic count exceeds u64");
      maximumSelections = std::max(maximumSelections, ++count);
      return endpoint;
    };
    for (PnrIndex demand = 0; demand < portAttachments.size(); ++demand) {
      const PnrIndex option = portAttachments[demand];
      auto endpoint = countSelection(option);
      if (!endpoint)
        return endpoint.takeError();
      loom::mapping_debug::emit(
          loom::mapping_debug::Level::Detail,
          loom::mapping_debug::Stage::SpatialPnr,
          loom::mapping_debug::Event::ContextChoice,
          [&](llvm::json::Object &fields) {
            const FrozenSpatialPortDemand &record =
                problem->ports().portDemands()[demand];
            fields["operation"] = "initializer_port_attachment_choice";
            fields["attempt"] = attemptOrdinal;
            fields["demand"] = demand;
            fields["demand_kind"] = static_cast<std::uint64_t>(record.kind);
            fields["terminal_kind"] = record.terminal.index();
            fields["realization"] = record.realization;
            fields["logical_net"] = record.logicalNet;
            fields["attachment_option"] = option;
            fields["endpoint"] = *endpoint;
            fields["endpoint_ref"] =
                loom::fabric::printFabricRef(endpoints[*endpoint].reference);
          });
    }
    for (PnrIndex boundary = 0; boundary < graphBoundaryAttachments.size();
         ++boundary) {
      const PnrIndex option = graphBoundaryAttachments[boundary];
      auto endpoint = countSelection(option);
      if (!endpoint)
        return endpoint.takeError();
      loom::mapping_debug::emit(
          loom::mapping_debug::Level::Detail,
          loom::mapping_debug::Stage::SpatialPnr,
          loom::mapping_debug::Event::ContextChoice,
          [&](llvm::json::Object &fields) {
            const FrozenSpatialGraphBoundary &record =
                problem->ports().graphBoundaries()[boundary];
            fields["operation"] = "initializer_graph_boundary_choice";
            fields["attempt"] = attemptOrdinal;
            fields["boundary"] = boundary;
            fields["terminal_kind"] = record.terminal.index();
            fields["logical_net"] = record.logicalNet;
            fields["attachment_option"] = option;
            fields["endpoint"] = *endpoint;
            fields["endpoint_ref"] =
                loom::fabric::printFabricRef(endpoints[*endpoint].reference);
          });
    }
    std::uint64_t duplicatedEndpoints = 0;
    for (const auto &[endpoint, count] : endpointSelections) {
      (void)endpoint;
      duplicatedEndpoints += count > 1;
    }
    loom::mapping_debug::emit(
        loom::mapping_debug::Level::Decision,
        loom::mapping_debug::Stage::SpatialPnr,
        loom::mapping_debug::Event::ContextChoice,
        [&](llvm::json::Object &fields) {
          fields["operation"] = "initializer_attachment_summary";
          fields["attempt"] = attemptOrdinal;
          fields["port_demand_count"] = portAttachments.size();
          fields["graph_boundary_count"] = graphBoundaryAttachments.size();
          fields["attachment_count"] =
              portAttachments.size() + graphBoundaryAttachments.size();
          fields["distinct_endpoint_count"] = endpointSelections.size();
          fields["duplicated_endpoint_count"] = duplicatedEndpoints;
          fields["maximum_endpoint_selections"] = maximumSelections;
        });
  }

  SpatialInitializerAttemptBuilder builder(
      *problem, diversificationStream ? &*diversificationStream : nullptr,
      policy.search.initializer.assignmentAttemptLimitPerSeed,
      preferred->assignmentAttempts, std::move(computeBindings),
      std::move(memoryBindings), std::move(portAttachments),
      std::move(graphBoundaryAttachments));
  llvm::Error buildError = builder.build();
  assignmentAttempts = builder.assignmentAttempts();
  if (buildError)
    return std::move(buildError);
  auto candidate =
      SpatialCandidateState::create(problem, builder.initialization());
  if (!candidate)
    return candidate.takeError();
  return SpatialCandidateInitializerAttempt{std::move(*candidate)};
}

llvm::Expected<SpatialCandidateStateHandle>
loom::pnr::createCanonicalSpatialCandidate(
    FrozenSpatialPnrProblemHandle problem) {
  std::uint64_t assignmentAttempts = 0;
  auto attempt = createSpatialCandidateInitializerAttempt(std::move(problem), 0,
                                                          assignmentAttempts);
  if (!attempt)
    return attempt.takeError();
  return std::move(attempt->candidate);
}
