#include "Mapping/Artifact/SpatialPhysicalDemandProjection.h"

#include "ConfiguredHardwareProjectionInternal.h"

#include "Dataflow/IR/DataflowServiceSchema.h"
#include "Fabric/Identity/FabricMemoryInternalConnection.h"
#include "Fabric/Identity/FabricRefBytes.h"
#include "Fabric/Identity/FabricTemporalPeConfiguration.h"
#include "Fabric/Identity/FabricTemporalSwitchRoute.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <system_error>
#include <tuple>
#include <vector>

namespace loom::mapping {
namespace {

using MemoryRole = ::dataflow::semantics::ServiceValueRole;

constexpr std::size_t memoryRoleCount =
    static_cast<std::size_t>(MemoryRole::Completion) + 1;

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      "spatial_physical_demand_projection_invalid: " + message);
}

const TechComputeRealizationView *
findComputeRealization(const TechMappingView &techMapping,
                       ::dataflow::ActorRef actor) {
  const TechComputeRealizationView *result = nullptr;
  for (const TechComputeRealizationView &realization :
       techMapping.computeRealizations()) {
    if (!llvm::any_of(realization.actors, [&](const auto &candidate) {
          return candidate.actor == actor;
        }))
      continue;
    if (result)
      return nullptr;
    result = &realization;
  }
  return result;
}

const SpatialComputeBindingView *
findComputeBinding(llvm::ArrayRef<SpatialComputeBindingView> bindings,
                   std::uint64_t realization) {
  const SpatialComputeBindingView *result = nullptr;
  for (const SpatialComputeBindingView &binding : bindings) {
    if (binding.realization != realization)
      continue;
    if (result)
      return nullptr;
    result = &binding;
  }
  return result;
}

const TechComputeBoundaryView *
findInputBoundary(const TechComputeRealizationView &realization,
                  const ::dataflow::ActorTokenOperandRef &operand) {
  const TechComputeBoundaryView *result = nullptr;
  for (const TechComputeBoundaryView &boundary : realization.boundaries) {
    if (boundary.actor != operand.actor ||
        boundary.direction != ::loom::fabric::FabricPortDirection::Input ||
        boundary.portOrdinal != operand.ordinal)
      continue;
    if (result)
      return nullptr;
    result = &boundary;
  }
  return result;
}

const TechComputeBoundaryView *
findOutputBoundary(const TechComputeRealizationView &realization,
                   const ::dataflow::ActorTokenResultRef &result) {
  const TechComputeBoundaryView *boundary = nullptr;
  for (const TechComputeBoundaryView &candidate : realization.boundaries) {
    if (candidate.actor != result.actor ||
        candidate.direction != ::loom::fabric::FabricPortDirection::Output ||
        candidate.portOrdinal != result.ordinal)
      continue;
    if (boundary)
      return nullptr;
    boundary = &candidate;
  }
  return boundary;
}

std::vector<std::uint8_t>
matchGroupKey(const ::loom::fabric::FabricTransportEndpointRef &ingress,
              const llvm::APInt &tag) {
  const std::vector<std::uint8_t> ingressBytes =
      ::loom::fabric::canonicalFabricBytes(ingress);
  std::vector<std::uint8_t> result;
  result.reserve(8 + ingressBytes.size() + 4 + (tag.getBitWidth() + 7) / 8);
  for (int shift = 56; shift >= 0; shift -= 8)
    result.push_back(static_cast<std::uint8_t>(ingressBytes.size() >> shift));
  result.insert(result.end(), ingressBytes.begin(), ingressBytes.end());
  const unsigned width = tag.getBitWidth();
  for (unsigned byte = 0; byte != 4; ++byte)
    result.push_back(static_cast<std::uint8_t>(width >> (24 - byte * 8)));
  const std::size_t byteCount = (width + 7) / 8;
  for (std::size_t byte = 0; byte != byteCount; ++byte) {
    const unsigned bit = static_cast<unsigned>(byte * 8);
    result.push_back(static_cast<std::uint8_t>(
        tag.extractBitsAsZExtValue(std::min<unsigned>(8, width - bit), bit)));
  }
  return result;
}

llvm::Error
verifyMatchGroupCapacity(const ::loom::fabric::FabricArtifactView &fabric,
                         SpatialPeOperandQueueMatchGroupView &group) {
  if (group.matches.empty())
    return invalid("PE operand queue match group is empty");
  const ::loom::fabric::FabricPeOccurrenceRef pe =
      group.matches.front().queue.context.pe;
  if (group.ingress.owner !=
          ::loom::fabric::FabricTransportEndpointOwnerRef::of(pe) ||
      fabric.transportEndpointDirection(group.ingress) !=
          ::loom::fabric::FabricPortDirection::Input)
    return invalid("PE operand queue group has a foreign ingress");
  auto mode = fabric.peOperandBufferMode(pe);
  auto schema = fabric.temporalPeConfigurationSchema(pe);
  if (!mode || !schema)
    return invalid("PE operand queue group has no Temporal buffer schema");
  std::vector<std::uint32_t> fuInputCounts;
  fuInputCounts.reserve(schema->layout().fus.size());
  for (const auto &fu : schema->layout().fus)
    fuInputCounts.push_back(fu.inputCount);
  auto contract = ::fabric::TemporalOperandBufferContract::create(
      ::fabric::TemporalOperandBufferDeclaration{
          pe, schema->layout().contextCount, fuInputCounts, *mode,
          fabric.peOperandBufferSize(pe)});
  if (!contract)
    return contract.takeError();

  std::vector<std::uint32_t> queues;
  queues.reserve(group.matches.size());
  for (SpatialPeOperandQueueMatchView &match : group.matches) {
    const ::fabric::LogicalOperandQueueKey &queue = match.queue;
    if (queue.context.pe != pe)
      return invalid("one PE operand match group spans multiple PEs");
    const auto found = llvm::find(contract->logicalQueues(), queue);
    if (found == contract->logicalQueues().end())
      return invalid("PE operand match group names an absent logical queue");
    const std::uint32_t ordinal = static_cast<std::uint32_t>(
        std::distance(contract->logicalQueues().begin(), found));
    match.allocationUnit = contract->allocationUnitOf(ordinal);
    match.entryCapacity = contract->entriesPerAllocationUnit().value();
    queues.push_back(ordinal);
  }
  if (!contract->admitsIngressEnqueueSet(queues))
    return invalid("PE operand match group exceeds one enqueue service");
  return llvm::Error::success();
}

llvm::Expected<::loom::fabric::FabricOrdinal> memoryConnectionOrdinal(
    const ::loom::fabric::FabricArtifactView &fabric,
    ::loom::fabric::FabricMemoryOccurrenceRef occurrence,
    const ::loom::fabric::FabricMemoryEngineTemplateInternalConnectionRef
        &selected) {
  const auto *connectivity = fabric.memoryConnectivity(occurrence);
  if (!connectivity)
    return invalid("memory occurrence has no connectivity contract");
  std::optional<::loom::fabric::FabricOrdinal> result;
  for (auto [ordinal, candidate] :
       llvm::enumerate(connectivity->internalConnections())) {
    if (candidate.sourceEndpointOrdinal != selected.source.ordinal ||
        candidate.sinkEndpointOrdinal != selected.sink.ordinal)
      continue;
    if (result)
      return invalid("memory occurrence repeats a template connection");
    result = static_cast<::loom::fabric::FabricOrdinal>(ordinal);
  }
  if (!result)
    return invalid("memory occurrence omits a selected template connection");
  return *result;
}

bool isResidualProducer(
    const TechMappingView &techMapping,
    const ::dataflow::CanonicalGraphProducerEndpointRef &producer) {
  return llvm::any_of(techMapping.residualLogicalNets(),
                      [&](const TechResidualLogicalNetView &net) {
                        return net.producer == producer;
                      });
}

const TechMemoryInternalEdgeView *findInternalSource(
    const TechMemoryRealizationView &realization,
    const ::dataflow::CanonicalGraphConsumerEndpointRef &consumer) {
  const TechMemoryInternalEdgeView *result = nullptr;
  for (const TechMemoryInternalEdgeView &edge : realization.internalEdges) {
    if (edge.consumer != consumer)
      continue;
    if (result)
      return nullptr;
    result = &edge;
  }
  return result;
}

struct SwitchTraversalUse final {
  std::vector<std::uint8_t> occurrenceKey;
  ::loom::fabric::FabricSwitchOccurrenceRef occurrence;
  std::uint64_t routeTreeOrdinal = 0;
  std::uint64_t segmentOrdinal = 0;
  ::loom::fabric::FabricOrdinal input = 0;
  ::loom::fabric::FabricOrdinal output = 0;
  ::loom::fabric::FabricPhysicalTraversalRef traversal;
  llvm::APInt tag = llvm::APInt(1, 0);
};

void saturatingIncrement(std::uint64_t &value) {
  if (value != std::numeric_limits<std::uint64_t>::max())
    ++value;
}

bool augmentComputeContextSupply(
    std::size_t demand, llvm::ArrayRef<std::vector<std::size_t>> domains,
    std::vector<std::optional<std::size_t>> &ownerByValue,
    std::vector<std::uint8_t> &visited, std::uint64_t &deterministicWork) {
  for (const std::size_t value : domains[demand]) {
    saturatingIncrement(deterministicWork);
    if (visited[value])
      continue;
    visited[value] = 1;
    if (!ownerByValue[value] ||
        augmentComputeContextSupply(*ownerByValue[value], domains, ownerByValue,
                                    visited, deterministicWork)) {
      ownerByValue[value] = demand;
      return true;
    }
  }
  return false;
}

using MemoryExclusiveKey = std::pair<std::uint8_t, std::vector<std::uint8_t>>;

MemoryExclusiveKey
memoryExclusiveKey(const SpatialMemoryExclusiveResourceView &resource) {
  return {static_cast<std::uint8_t>(resource.kind), resource.key};
}

struct PreparedMemoryOccurrenceDemand final {
  std::vector<std::size_t> choices;
  std::vector<MemoryExclusiveKey> resources;
  std::uint64_t residentDemand = 0;
};

bool memoryChoiceIsLegal(
    const PreparedMemoryOccurrenceDemand &demand, std::size_t occurrence,
    llvm::ArrayRef<std::uint64_t> capacities,
    llvm::ArrayRef<std::uint64_t> usedCapacity,
    llvm::ArrayRef<std::set<MemoryExclusiveKey>> usedResources,
    std::uint64_t &deterministicWork) {
  saturatingIncrement(deterministicWork);
  if (demand.residentDemand > capacities[occurrence] - usedCapacity[occurrence])
    return false;
  for (const MemoryExclusiveKey &resource : demand.resources) {
    saturatingIncrement(deterministicWork);
    if (usedResources[occurrence].count(resource))
      return false;
  }
  return true;
}

bool searchMemoryOccurrenceAssignment(
    llvm::ArrayRef<PreparedMemoryOccurrenceDemand> demands,
    llvm::ArrayRef<std::uint64_t> capacities,
    std::vector<std::optional<std::size_t>> &assignment,
    std::vector<std::uint64_t> &usedCapacity,
    std::vector<std::set<MemoryExclusiveKey>> &usedResources,
    SpatialMemoryOccurrenceSupplyAnalysis &analysis) {
  std::size_t selected = demands.size();
  std::vector<std::size_t> selectedChoices;
  for (std::size_t demandOrdinal = 0; demandOrdinal != demands.size();
       ++demandOrdinal) {
    if (assignment[demandOrdinal])
      continue;
    std::vector<std::size_t> legalChoices;
    for (const std::size_t occurrence : demands[demandOrdinal].choices)
      if (memoryChoiceIsLegal(demands[demandOrdinal], occurrence, capacities,
                              usedCapacity, usedResources,
                              analysis.deterministicWork))
        legalChoices.push_back(occurrence);
    if (legalChoices.empty())
      return false;
    if (selected == demands.size() ||
        legalChoices.size() < selectedChoices.size()) {
      selected = demandOrdinal;
      selectedChoices = std::move(legalChoices);
    }
  }
  if (selected == demands.size())
    return true;

  const PreparedMemoryOccurrenceDemand &demand = demands[selected];
  for (const std::size_t occurrence : selectedChoices) {
    saturatingIncrement(analysis.assignmentAttempts);
    assignment[selected] = occurrence;
    usedCapacity[occurrence] += demand.residentDemand;
    for (const MemoryExclusiveKey &resource : demand.resources)
      usedResources[occurrence].insert(resource);
    if (searchMemoryOccurrenceAssignment(demands, capacities, assignment,
                                         usedCapacity, usedResources, analysis))
      return true;
    for (const MemoryExclusiveKey &resource : demand.resources)
      usedResources[occurrence].erase(resource);
    usedCapacity[occurrence] -= demand.residentDemand;
    assignment[selected].reset();
  }
  return false;
}

} // namespace

llvm::Expected<std::vector<SpatialComputeContextPlacementDomainView>>
deriveSpatialComputeContextPlacementDomain(
    ::loom::fabric::FabricFuCapabilityTemplateRef capabilityTemplate,
    const ::loom::fabric::FabricArtifactView &fabric) {
  std::vector<SpatialComputeContextPlacementDomainView> placements;
  for (const ::loom::fabric::FabricFuOccurrenceRef fu :
       fabric.fuOccurrences()) {
    const std::optional<::loom::fabric::FabricFuTemplateRef> definition =
        fabric.fuTemplateOf(fu);
    if (!definition || *definition != capabilityTemplate.fu)
      continue;
    const std::optional<::loom::fabric::FabricPeOccurrenceRef> parent =
        fabric.parentPeOf(fu);
    if (!parent)
      return invalid("a Fabric FU occurrence has no parent PE relation");
    const std::optional<::fabric::Schedule> schedule =
        fabric.peSchedule(*parent);
    if (!schedule)
      return invalid("a Fabric PE occurrence has no scheduling contract");

    SpatialComputeContextPlacementDomainView placement{
        fu, *parent, *schedule, {}};
    const std::uint64_t contextCount = fabric.peResidentContextCount(*parent);
    placement.contexts.reserve(contextCount);
    for (std::uint64_t ordinal = 0; ordinal != contextCount; ++ordinal)
      placement.contexts.push_back({*parent, ordinal});
    placements.push_back(std::move(placement));
  }
  return placements;
}

llvm::Expected<std::vector<SpatialComputeContextDemandView>>
deriveSpatialComputeContextDemands(
    const TechMappingView &techMapping,
    const ::loom::fabric::FabricArtifactView &fabric) {
  if (techMapping.fabricIdentity() != fabric.identity())
    return invalid("TechMapping is bound to a foreign Fabric");

  std::vector<SpatialComputeContextDemandView> demands;
  std::map<std::vector<std::uint8_t>,
           std::vector<SpatialComputeContextPlacementDomainView>>
      basePlacements;
  demands.reserve(techMapping.computeRealizations().size());
  for (auto [realizationOrdinal, realization] :
       llvm::enumerate(techMapping.computeRealizations())) {
    SpatialComputeContextDemandView demand{
        static_cast<std::uint64_t>(realizationOrdinal),
        realization.capabilityTemplate,
        {}};
    const std::vector<std::uint8_t> capabilityKey =
        ::loom::fabric::canonicalFabricBytes(realization.capabilityTemplate);
    auto [found, inserted] = basePlacements.try_emplace(capabilityKey);
    if (inserted) {
      auto placements = deriveSpatialComputeContextPlacementDomain(
          realization.capabilityTemplate, fabric);
      if (!placements)
        return placements.takeError();
      found->second = std::move(*placements);
    }
    demand.candidatePlacementCount = found->second.size();
    demand.placements = found->second;
    demands.push_back(std::move(demand));
  }
  return demands;
}

llvm::Expected<SpatialComputeContextSupplyAnalysis>
analyzeSpatialComputeContextSupply(
    llvm::ArrayRef<std::vector<std::size_t>> domains, std::size_t valueCount) {
  SpatialComputeContextSupplyAnalysis result;
  result.demandCount = domains.size();
  result.valueCount = valueCount;
  for (const auto &domain : domains) {
    if (!llvm::is_sorted(domain) ||
        std::adjacent_find(domain.begin(), domain.end()) != domain.end())
      return invalid("compute-context domain is not a canonical set");
    if (!domain.empty() && domain.back() >= valueCount)
      return invalid("compute-context domain contains an unknown value");
    if (domain.size() >
        std::numeric_limits<std::uint64_t>::max() - result.edgeCount)
      result.edgeCount = std::numeric_limits<std::uint64_t>::max();
    else
      result.edgeCount += domain.size();
  }

  std::vector<std::optional<std::size_t>> ownerByValue(valueCount);
  for (std::size_t demand = 0; demand != domains.size(); ++demand) {
    std::vector<std::uint8_t> visited(valueCount, 0);
    result.maximumMatching += augmentComputeContextSupply(
        demand, domains, ownerByValue, visited, result.deterministicWork);
  }
  if (result.admissible())
    return result;

  std::vector<std::optional<std::size_t>> valueByDemand(domains.size());
  for (auto [value, owner] : llvm::enumerate(ownerByValue))
    if (owner)
      valueByDemand[*owner] = value;

  std::vector<std::uint8_t> reachedDemands(domains.size(), 0);
  std::vector<std::uint8_t> reachedValues(valueCount, 0);
  llvm::SmallVector<std::size_t, 16> pending;
  for (std::size_t demand = 0; demand != domains.size(); ++demand) {
    if (valueByDemand[demand])
      continue;
    reachedDemands[demand] = 1;
    pending.push_back(demand);
  }
  for (std::size_t cursor = 0; cursor != pending.size(); ++cursor) {
    const std::size_t demand = pending[cursor];
    for (const std::size_t value : domains[demand]) {
      saturatingIncrement(result.deterministicWork);
      if (valueByDemand[demand] == value || reachedValues[value])
        continue;
      reachedValues[value] = 1;
      const std::optional<std::size_t> owner = ownerByValue[value];
      if (!owner || reachedDemands[*owner])
        continue;
      reachedDemands[*owner] = 1;
      pending.push_back(*owner);
    }
  }
  for (auto [demand, reached] : llvm::enumerate(reachedDemands))
    if (reached)
      result.hallDemands.push_back(demand);
  result.hallValueCount =
      llvm::count(reachedValues, static_cast<std::uint8_t>(1));
  return result;
}

llvm::Expected<SpatialMemoryOccurrenceDemandView>
deriveSpatialMemoryOccurrenceDemand(
    const TechMemoryRealizationView &realization,
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ::loom::fabric::FabricArtifactView &fabric) {
  const auto *engine = fabric.memoryEngineTemplate(realization.engine);
  if (!engine)
    return invalid("memory realization names a foreign engine template");

  SpatialMemoryOccurrenceDemandView demand;
  demand.engine = realization.engine;
  demand.schedule = engine->schedule;
  demand.residentDemand = engine->schedule == ::fabric::Schedule::Temporal
                              ? realization.actors.size()
                              : 0;
  saturatingIncrement(demand.projectionWork);
  for (const ::loom::fabric::FabricMemoryOccurrenceRef occurrence :
       fabric.memoryOccurrences()) {
    saturatingIncrement(demand.projectionWork);
    const auto definition = fabric.memoryEngineTemplateOf(occurrence);
    if (!definition || *definition != realization.engine)
      continue;
    if (fabric.memorySchedule(occurrence) != engine->schedule)
      return invalid("memory occurrence disagrees with its engine schedule");
    demand.occurrences.push_back(
        {occurrence, engine->residentContextCount.value_or(0)});
  }
  llvm::sort(demand.occurrences, [](const auto &lhs, const auto &rhs) {
    return ::loom::fabric::canonicalFabricBytes(lhs.occurrence) <
           ::loom::fabric::canonicalFabricBytes(rhs.occurrence);
  });

  if (engine->schedule == ::fabric::Schedule::Spatial) {
    for (const TechMemoryActorView &actor : realization.actors) {
      saturatingIncrement(demand.projectionWork);
      demand.exclusiveResources.push_back(
          {SpatialMemoryExclusiveResourceKind::SpatialOperationPort,
           ::loom::fabric::canonicalFabricBytes(actor.operationPort)});
    }
  } else {
    auto ingresses = deriveTechMemoryExternalIngresses(realization, dataflow);
    if (!ingresses)
      return ingresses.takeError();
    for (const TechMemoryExternalIngressView &ingress : *ingresses) {
      saturatingIncrement(demand.projectionWork);
      auto key =
          canonicalTechMemoryExternalIngressKey(ingress, dataflow.identity());
      if (!key)
        return key.takeError();
      demand.exclusiveResources.push_back(
          {SpatialMemoryExclusiveResourceKind::TemporalExternalIngress,
           std::move(*key)});
    }
  }
  for (const TechMemoryInternalEdgeView &edge : realization.internalEdges) {
    saturatingIncrement(demand.projectionWork);
    demand.exclusiveResources.push_back(
        {SpatialMemoryExclusiveResourceKind::InternalConnection,
         ::loom::fabric::canonicalFabricBytes(edge.connection)});
  }
  llvm::sort(demand.exclusiveResources, [](const auto &lhs, const auto &rhs) {
    return memoryExclusiveKey(lhs) < memoryExclusiveKey(rhs);
  });
  demand.exclusiveResources.erase(
      std::unique(demand.exclusiveResources.begin(),
                  demand.exclusiveResources.end(),
                  [](const auto &lhs, const auto &rhs) {
                    return lhs.kind == rhs.kind && lhs.key == rhs.key;
                  }),
      demand.exclusiveResources.end());
  return demand;
}

llvm::StringRef spatialMemoryExclusiveResourceKindSpelling(
    SpatialMemoryExclusiveResourceKind kind) {
  switch (kind) {
  case SpatialMemoryExclusiveResourceKind::SpatialOperationPort:
    return "spatial_operation_port";
  case SpatialMemoryExclusiveResourceKind::TemporalExternalIngress:
    return "temporal_external_ingress";
  case SpatialMemoryExclusiveResourceKind::InternalConnection:
    return "internal_connection";
  }
  llvm_unreachable("unknown memory exclusive-resource kind");
}

llvm::StringRef spatialMemoryOccurrenceSupplyFailureKindSpelling(
    SpatialMemoryOccurrenceSupplyFailureKind failure) {
  switch (failure) {
  case SpatialMemoryOccurrenceSupplyFailureKind::None:
    return "none";
  case SpatialMemoryOccurrenceSupplyFailureKind::EmptyOccurrenceDomain:
    return "empty_occurrence_domain";
  case SpatialMemoryOccurrenceSupplyFailureKind::ExclusiveResourceDeficit:
    return "exclusive_resource_deficit";
  case SpatialMemoryOccurrenceSupplyFailureKind::ResidentCapacityDeficit:
    return "resident_capacity_deficit";
  case SpatialMemoryOccurrenceSupplyFailureKind::JointAssignmentInfeasible:
    return "joint_assignment_infeasible";
  }
  llvm_unreachable("unknown memory occurrence-supply failure");
}

llvm::Expected<std::vector<SpatialMemoryOccurrenceDemandView>>
deriveSpatialMemoryOccurrenceDemands(
    const TechMappingView &techMapping,
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ::loom::fabric::FabricArtifactView &fabric) {
  if (techMapping.fabricIdentity() != fabric.identity())
    return invalid("TechMapping is bound to a foreign Fabric");
  if (techMapping.dataflowIdentity() != dataflow.identity())
    return invalid("TechMapping is bound to a foreign Dataflow");

  std::vector<SpatialMemoryOccurrenceDemandView> demands;
  demands.reserve(techMapping.memoryRealizations().size());
  for (auto [ordinal, realization] :
       llvm::enumerate(techMapping.memoryRealizations())) {
    auto demand =
        deriveSpatialMemoryOccurrenceDemand(realization, dataflow, fabric);
    if (!demand)
      return demand.takeError();
    demand->realization = ordinal;
    demands.push_back(std::move(*demand));
  }
  return demands;
}

llvm::Expected<SpatialMemoryOccurrenceSupplyAnalysis>
analyzeSpatialMemoryOccurrenceSupply(
    llvm::ArrayRef<const SpatialMemoryOccurrenceDemandView *> demands) {
  SpatialMemoryOccurrenceSupplyAnalysis analysis;
  analysis.demandCount = demands.size();
  std::map<std::vector<std::uint8_t>, std::size_t> occurrenceOrdinals;
  std::vector<std::uint64_t> capacities;
  std::vector<PreparedMemoryOccurrenceDemand> prepared;
  prepared.reserve(demands.size());
  std::map<MemoryExclusiveKey, std::vector<std::size_t>> resourceUsers;
  std::map<std::vector<std::uint8_t>, std::vector<std::size_t>> engineUsers;

  for (auto [demandOrdinal, source] : llvm::enumerate(demands)) {
    if (!source)
      return invalid("memory occurrence supply contains a null demand");
    PreparedMemoryOccurrenceDemand demand;
    demand.residentDemand = source->residentDemand;
    if (source->occurrences.empty()) {
      analysis.failure =
          SpatialMemoryOccurrenceSupplyFailureKind::EmptyOccurrenceDomain;
      analysis.failingDemandCount = 1;
      return analysis;
    }
    std::vector<std::uint8_t> previousOccurrence;
    bool firstOccurrence = true;
    for (const SpatialMemoryOccurrenceSupplyView &supply :
         source->occurrences) {
      const std::vector<std::uint8_t> key =
          ::loom::fabric::canonicalFabricBytes(supply.occurrence);
      if (!firstOccurrence && key <= previousOccurrence)
        return invalid("memory occurrence domain is not a canonical set");
      firstOccurrence = false;
      previousOccurrence = key;
      auto [found, inserted] =
          occurrenceOrdinals.try_emplace(key, occurrenceOrdinals.size());
      if (inserted)
        capacities.push_back(supply.residentCapacity);
      else if (capacities[found->second] != supply.residentCapacity)
        return invalid("one memory occurrence has inconsistent capacity");
      demand.choices.push_back(found->second);
      saturatingIncrement(analysis.occurrenceChoiceCount);
    }
    for (const SpatialMemoryExclusiveResourceView &resource :
         source->exclusiveResources) {
      const MemoryExclusiveKey key = memoryExclusiveKey(resource);
      if (!demand.resources.empty() && key <= demand.resources.back())
        return invalid("memory exclusive resources are not a canonical set");
      demand.resources.push_back(key);
      resourceUsers[key].push_back(demandOrdinal);
    }
    engineUsers[::loom::fabric::canonicalFabricBytes(source->engine)].push_back(
        demandOrdinal);
    prepared.push_back(std::move(demand));
  }
  analysis.occurrenceValueCount = occurrenceOrdinals.size();

  for (const auto &[resource, users] : resourceUsers) {
    if (users.size() < 2)
      continue;
    saturatingIncrement(analysis.exclusiveRelationCount);
    std::set<std::size_t> values;
    for (const std::size_t demand : users)
      values.insert(prepared[demand].choices.begin(),
                    prepared[demand].choices.end());
    analysis.deterministicWork += std::min<std::uint64_t>(
        users.size() + values.size(),
        std::numeric_limits<std::uint64_t>::max() - analysis.deterministicWork);
    if (users.size() <= values.size())
      continue;
    analysis.failure =
        SpatialMemoryOccurrenceSupplyFailureKind::ExclusiveResourceDeficit;
    analysis.failingResourceKind =
        static_cast<SpatialMemoryExclusiveResourceKind>(resource.first);
    analysis.failingDemandCount = users.size();
    analysis.failingOccurrenceCount = values.size();
    return analysis;
  }

  for (const auto &[engine, users] : engineUsers) {
    (void)engine;
    std::uint64_t residentDemand = 0;
    std::map<std::size_t, std::uint64_t> available;
    for (const std::size_t demand : users) {
      if (prepared[demand].residentDemand >
          std::numeric_limits<std::uint64_t>::max() - residentDemand)
        residentDemand = std::numeric_limits<std::uint64_t>::max();
      else
        residentDemand += prepared[demand].residentDemand;
      for (const std::size_t occurrence : prepared[demand].choices)
        available.try_emplace(occurrence, capacities[occurrence]);
    }
    std::uint64_t residentCapacity = 0;
    for (const auto &[occurrence, capacity] : available) {
      (void)occurrence;
      if (capacity >
          std::numeric_limits<std::uint64_t>::max() - residentCapacity)
        residentCapacity = std::numeric_limits<std::uint64_t>::max();
      else
        residentCapacity += capacity;
    }
    if (residentDemand <= residentCapacity)
      continue;
    analysis.failure =
        SpatialMemoryOccurrenceSupplyFailureKind::ResidentCapacityDeficit;
    analysis.failingDemandCount = users.size();
    analysis.failingOccurrenceCount = available.size();
    analysis.failingResidentDemand = residentDemand;
    analysis.failingResidentCapacity = residentCapacity;
    return analysis;
  }

  std::vector<std::optional<std::size_t>> assignment(prepared.size());
  std::vector<std::uint64_t> usedCapacity(capacities.size(), 0);
  std::vector<std::set<MemoryExclusiveKey>> usedResources(capacities.size());
  if (!searchMemoryOccurrenceAssignment(prepared, capacities, assignment,
                                        usedCapacity, usedResources, analysis))
    analysis.failure =
        SpatialMemoryOccurrenceSupplyFailureKind::JointAssignmentInfeasible;
  return analysis;
}

llvm::Expected<::loom::fabric::FabricOrdinal>
deriveSpatialMemoryInternalConnectionOrdinal(
    const ::loom::fabric::FabricArtifactView &fabric,
    ::loom::fabric::FabricMemoryOccurrenceRef occurrence,
    const ::loom::fabric::FabricMemoryEngineTemplateInternalConnectionRef
        &selected) {
  return memoryConnectionOrdinal(fabric, occurrence, selected);
}

llvm::Expected<std::vector<SpatialTemporalPeDispatchDomainView>>
deriveSpatialTemporalPeDispatchDomains(
    const ::loom::fabric::FabricArtifactView &fabric,
    llvm::ArrayRef<SpatialComputeBindingView> computeBindings) {
  struct PreparedPe final {
    ::fabric::TemporalPeResourceContract contract;
    std::map<std::uint64_t, std::uint32_t> fuOrdinals;
  };

  std::map<std::vector<std::uint8_t>, PreparedPe> prepared;
  using DispatchDomainKey = std::pair<std::vector<std::uint8_t>, std::uint32_t>;
  std::map<DispatchDomainKey, SpatialTemporalPeDispatchDomainView> domains;
  for (const SpatialComputeBindingView &binding : computeBindings) {
    const auto pe = fabric.parentPeOf(binding.occurrence);
    if (!pe)
      return invalid("compute binding FU has no parent PE");
    if (fabric.peSchedule(*pe) != ::fabric::Schedule::Temporal)
      continue;
    if (binding.context.pe != *pe ||
        binding.context.ordinal >= fabric.peResidentContextCount(*pe))
      return invalid("temporal compute binding has a foreign context");

    const std::vector<std::uint8_t> peKey =
        ::loom::fabric::canonicalFabricBytes(*pe);
    auto foundPe = prepared.find(peKey);
    if (foundPe == prepared.end()) {
      auto schema = fabric.temporalPeConfigurationSchema(*pe);
      const auto mode = fabric.peOperandBufferMode(*pe);
      if (!schema || !mode)
        return invalid("temporal compute binding has no PE contract schema");
      std::vector<std::uint32_t> fuInputCounts;
      std::map<std::uint64_t, std::uint32_t> fuOrdinals;
      fuInputCounts.reserve(schema->layout().fus.size());
      for (auto [ordinal, shape] : llvm::enumerate(schema->layout().fus)) {
        if (!fuOrdinals
                 .emplace(shape.fu.id(), static_cast<std::uint32_t>(ordinal))
                 .second)
          return invalid("temporal PE repeats one FU occurrence");
        fuInputCounts.push_back(shape.inputCount);
      }
      auto contract = ::fabric::TemporalPeResourceContract::create(
          {*pe, schema->layout().contextCount, fuInputCounts, *mode,
           fabric.peOperandBufferSize(*pe), schema->layout().registerFifoCount,
           fabric.peRegisterFifoDepth(*pe), fabric.peRegisterFifoPorts(*pe)});
      if (!contract)
        return contract.takeError();
      foundPe = prepared
                    .emplace(peKey, PreparedPe{std::move(*contract),
                                               std::move(fuOrdinals)})
                    .first;
    }

    const auto fu = foundPe->second.fuOrdinals.find(binding.occurrence.id());
    if (fu == foundPe->second.fuOrdinals.end())
      return invalid("temporal compute binding selects a foreign FU");
    const std::uint64_t candidate64 =
        static_cast<std::uint64_t>(binding.context.ordinal) *
            foundPe->second.fuOrdinals.size() +
        fu->second;
    if (candidate64 >= foundPe->second.contract.dispatchCandidates().size())
      return invalid("temporal compute dispatch candidate is out of range");
    const std::uint32_t candidate = static_cast<std::uint32_t>(candidate64);
    const auto &declared =
        foundPe->second.contract.dispatchCandidates()[candidate];
    if (declared.context != binding.context ||
        declared.fuOccurrence != fu->second)
      return invalid("temporal compute dispatch candidate drifted");
    auto pattern = ::fabric::resolveTemporalPeDispatchPattern(
        fabric, binding.context, binding.occurrence);
    if (!pattern)
      return pattern.takeError();
    const auto owner = pattern->owner.catalog();
    const ::fabric::ResourceContract *resource = fabric.resourceContract(owner);
    if (!resource || pattern->ordinal >= resource->usePatternCount())
      return invalid("temporal dispatch has no imported ResourceContract");
    const ::fabric::UsePattern use =
        resource->usePattern(::fabric::UsePatternKey(pattern->ordinal));

    auto [domain, inserted] =
        domains.try_emplace(DispatchDomainKey{peKey, declared.allocationUnit},
                            SpatialTemporalPeDispatchDomainView{
                                *pe, declared.allocationUnit, 0, {}});
    (void)inserted;
    if (llvm::any_of(domain->second.candidates, [&](const auto &existing) {
          return existing.context == binding.context &&
                 existing.fu == binding.occurrence;
        }))
      return invalid("one temporal dispatch candidate selects multiple "
                     "compute realizations");
    domain->second.candidates.push_back({binding.realization, binding.context,
                                         binding.occurrence, *pattern,
                                         use.requester});
  }

  std::vector<SpatialTemporalPeDispatchDomainView> result;
  result.reserve(domains.size());
  for (auto &[key, domain] : domains) {
    (void)key;
    if (domain.candidates.size() > 1) {
      const auto owner = ::loom::fabric::FabricInventoryOwnerRef::of(domain.pe);
      const ::fabric::ResourceContract *resource =
          fabric.resourceContract(owner);
      const auto policy = resource ? resource->grantPolicy() : std::nullopt;
      const auto *roundRobin =
          policy ? std::get_if<::fabric::RoundRobinView>(&*policy) : nullptr;
      if (!roundRobin)
        return invalid("contended temporal dispatch domain has no exact "
                       "round-robin policy");
      const auto cycle = roundRobin->requesterCycle();
      const auto reset = llvm::find(cycle, roundRobin->resetCursor());
      if (reset == cycle.end())
        return invalid("temporal dispatch reset is outside its policy cycle");
      std::vector<SpatialTemporalPeDispatchCandidateView> ordered;
      ordered.reserve(domain.candidates.size());
      for (::fabric::RequesterKey requester : cycle) {
        const auto candidate =
            llvm::find_if(domain.candidates, [&](const auto &selected) {
              return selected.requester == requester;
            });
        if (candidate == domain.candidates.end())
          continue;
        if (llvm::any_of(ordered, [&](const auto &selected) {
              return selected.requester == requester;
            }))
          return invalid("temporal dispatch policy repeats an active "
                         "candidate");
        ordered.push_back(*candidate);
      }
      if (ordered.size() != domain.candidates.size())
        return invalid("temporal dispatch policy omits an active candidate");
      bool foundReset = false;
      for (std::size_t scanned = 0; scanned != cycle.size(); ++scanned) {
        const ::fabric::RequesterKey requester =
            cycle[(std::distance(cycle.begin(), reset) + scanned) %
                  cycle.size()];
        const auto selected =
            llvm::find_if(ordered, [&](const auto &candidate) {
              return candidate.requester == requester;
            });
        if (selected == ordered.end())
          continue;
        domain.resetPosition = static_cast<std::uint32_t>(
            std::distance(ordered.begin(), selected));
        foundReset = true;
        break;
      }
      if (!foundReset)
        return invalid("temporal dispatch policy cannot reset into its active "
                       "candidate cycle");
      domain.candidates = std::move(ordered);
    }
    result.push_back(std::move(domain));
  }
  return result;
}

llvm::Expected<std::vector<SpatialTemporalSwitchPackedRowView>>
deriveSpatialTemporalSwitchPackedRows(
    const ::loom::fabric::FabricArtifactView &fabric,
    llvm::ArrayRef<SpatialRouteTreeView> routes,
    llvm::ArrayRef<SpatialResourceUseView> resourceUses,
    llvm::ArrayRef<SpatialPhysicalTagSegmentView> physicalTagSegments) {
  const std::uint64_t noSegment = std::numeric_limits<std::uint64_t>::max();
  std::vector<std::vector<std::uint64_t>> nodeSegments;
  nodeSegments.reserve(routes.size());
  for (const SpatialRouteTreeView &route : routes)
    nodeSegments.emplace_back(route.nodes.size(), noSegment);
  for (const SpatialPhysicalTagSegmentView &segment : physicalTagSegments) {
    if (segment.routeTreeOrdinal >= routes.size())
      return invalid("switch row segment names an absent RouteTree");
    auto &segments = nodeSegments[segment.routeTreeOrdinal];
    for (std::uint64_t node : segment.nodeOrdinals) {
      if (node >= segments.size() || segments[node] != noSegment)
        return invalid("switch row segment has a noncanonical node relation");
      segments[node] = segment.segmentOrdinal;
    }
  }

  std::vector<SwitchTraversalUse> selected;
  for (auto [routeOrdinal, route] : llvm::enumerate(routes)) {
    for (const SpatialRouteNodeView &node : route.nodes) {
      if (!node.incomingTraversal)
        continue;
      const auto *payload =
          std::get_if<::loom::fabric::FabricSwitchTraversalPayload>(
              &node.incomingTraversal->payload);
      if (!payload ||
          fabric.switchSchedule(payload->owner) != ::fabric::Schedule::Temporal)
        continue;
      if (node.ordinal >= nodeSegments[routeOrdinal].size())
        return invalid("Temporal switch traversal has an absent route node");
      const std::uint64_t segment = nodeSegments[routeOrdinal][node.ordinal];
      if (segment == noSegment)
        return invalid("Temporal switch traversal has no Physical Tag segment");
      auto tag = detail::resolveConfiguredHardwarePhysicalTag(
          fabric, routes, resourceUses, physicalTagSegments, routeOrdinal,
          node.ordinal);
      if (!tag)
        return tag.takeError();
      const ::loom::fabric::FabricTransportEndpointRef ingress{
          ::loom::fabric::FabricTransportEndpointOwnerRef::of(payload->owner),
          payload->input};
      const auto path = fabric.transportEndpointDataPath(ingress);
      if (!path || path->kind != ::fabric::DataPathKind::BitsTag ||
          path->tagWidthBits != tag->getBitWidth())
        return invalid("Temporal switch signature has an incompatible tag");
      selected.push_back({::loom::fabric::canonicalFabricBytes(payload->owner),
                          payload->owner, routeOrdinal, segment, payload->input,
                          payload->output, *node.incomingTraversal,
                          std::move(*tag)});
    }
  }
  llvm::sort(selected, [](const auto &lhs, const auto &rhs) {
    return std::tie(lhs.occurrenceKey, lhs.routeTreeOrdinal, lhs.segmentOrdinal,
                    lhs.input, lhs.output) <
           std::tie(rhs.occurrenceKey, rhs.routeTreeOrdinal, rhs.segmentOrdinal,
                    rhs.input, rhs.output);
  });

  std::vector<SpatialTemporalSwitchRouteSignatureView> signatures;
  for (std::size_t begin = 0; begin != selected.size();) {
    std::size_t end = begin + 1;
    while (end != selected.size() &&
           std::tie(selected[end].occurrenceKey, selected[end].routeTreeOrdinal,
                    selected[end].segmentOrdinal, selected[end].input) ==
               std::tie(selected[begin].occurrenceKey,
                        selected[begin].routeTreeOrdinal,
                        selected[begin].segmentOrdinal, selected[begin].input))
      ++end;
    SpatialTemporalSwitchRouteSignatureView signature{
        selected[begin].occurrence,
        selected[begin].routeTreeOrdinal,
        selected[begin].segmentOrdinal,
        selected[begin].input,
        {},
        {},
        selected[begin].tag};
    for (std::size_t ordinal = begin; ordinal != end; ++ordinal) {
      if (selected[ordinal].tag != signature.tag)
        return invalid("one Temporal switch signature has multiple tags");
      if (!signature.outputs.empty() &&
          signature.outputs.back() == selected[ordinal].output)
        return invalid("one Temporal switch signature repeats a crosspoint");
      signature.outputs.push_back(selected[ordinal].output);
      signature.traversals.push_back(selected[ordinal].traversal);
    }
    signatures.push_back(std::move(signature));
    begin = end;
  }

  std::vector<std::vector<SpatialTemporalSwitchRouteSignatureView>>
      demandSignatures;
  for (std::size_t begin = 0; begin != signatures.size();) {
    std::size_t end = begin + 1;
    while (end != signatures.size() &&
           std::tie(signatures[end].occurrence,
                    signatures[end].routeTreeOrdinal,
                    signatures[end].segmentOrdinal) ==
               std::tie(signatures[begin].occurrence,
                        signatures[begin].routeTreeOrdinal,
                        signatures[begin].segmentOrdinal))
      ++end;
    demandSignatures.emplace_back(
        std::make_move_iterator(signatures.begin() + begin),
        std::make_move_iterator(signatures.begin() + end));
    begin = end;
  }

  std::vector<
      std::vector<::loom::fabric::FabricTemporalSwitchRouteSignatureView>>
      ownerSignatureStorage;
  std::vector<::loom::fabric::FabricTemporalSwitchTaggedRouteDemandView>
      ownerDemands;
  ownerSignatureStorage.reserve(demandSignatures.size());
  ownerDemands.reserve(demandSignatures.size());
  for (const auto &demand : demandSignatures) {
    if (demand.empty())
      return invalid("Temporal switch route demand has no signature");
    ownerSignatureStorage.emplace_back();
    auto &ownerSignatures = ownerSignatureStorage.back();
    ownerSignatures.reserve(demand.size());
    for (const SpatialTemporalSwitchRouteSignatureView &signature : demand) {
      if (signature.tag != demand.front().tag)
        return invalid("one Temporal switch demand has multiple tags");
      ownerSignatures.push_back(
          {signature.occurrence, signature.input, signature.outputs});
    }
    ownerDemands.push_back({{ownerSignatures}, demand.front().tag});
  }
  auto ownerRows =
      ::loom::fabric::projectFabricTemporalSwitchRouteRows(ownerDemands);
  if (!ownerRows)
    return ownerRows.takeError();

  std::vector<SpatialTemporalSwitchPackedRowView> rows;
  rows.reserve(ownerRows->size());
  for (const ::loom::fabric::FabricTemporalSwitchPackedRouteRow &ownerRow :
       *ownerRows) {
    if (!ownerRow.compatible)
      return invalid("equal-tag Temporal switch demands cannot share one "
                     "resident row");
    SpatialTemporalSwitchPackedRowView row{
        ownerRow.occurrence, ownerRow.tag, {}, {}};
    for (std::uint64_t demandOrdinal : ownerRow.demandOrdinals) {
      if (demandOrdinal >= demandSignatures.size())
        return invalid("Fabric switch row names an absent route demand");
      for (SpatialTemporalSwitchRouteSignatureView &signature :
           demandSignatures[demandOrdinal]) {
        row.traversals.insert(row.traversals.end(),
                              signature.traversals.begin(),
                              signature.traversals.end());
        row.signatures.push_back(std::move(signature));
      }
    }
    rows.push_back(std::move(row));
  }
  for (SpatialTemporalSwitchPackedRowView &row : rows) {
    llvm::sort(row.traversals, [](const auto &lhs, const auto &rhs) {
      return ::loom::fabric::canonicalFabricBytes(lhs) <
             ::loom::fabric::canonicalFabricBytes(rhs);
    });
    row.traversals.erase(
        std::unique(row.traversals.begin(), row.traversals.end()),
        row.traversals.end());
  }
  for (std::size_t begin = 0; begin != rows.size();) {
    std::size_t end = begin + 1;
    while (end != rows.size() && rows[end].occurrence == rows[begin].occurrence)
      ++end;
    if (end - begin > fabric.switchRouteTableSize(rows[begin].occurrence))
      return invalid("Temporal switch packed rows exceed resident capacity");
    begin = end;
  }
  return rows;
}

llvm::Expected<std::vector<SpatialPeLocalTransferOptionView>>
deriveSpatialPeLocalTransferOptions(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const TechMappingView &techMapping,
    const ::loom::fabric::FabricArtifactView &fabric,
    llvm::ArrayRef<SpatialComputeBindingView> computeBindings,
    const TechResidualLogicalNetView &logicalNet) {
  std::vector<SpatialPeLocalTransferOptionView> options;
  if (logicalNet.sinks.size() != 1)
    return options;
  const auto *producer =
      std::get_if<::dataflow::ActorTokenResultRef>(&logicalNet.producer);
  const auto *consumer =
      std::get_if<::dataflow::ActorTokenOperandRef>(&logicalNet.sinks.front());
  if (!producer || !consumer)
    return options;

  const TechComputeRealizationView *producerRealization =
      findComputeRealization(techMapping, producer->actor);
  const TechComputeRealizationView *consumerRealization =
      findComputeRealization(techMapping, consumer->actor);
  if (!producerRealization || !consumerRealization)
    return options;
  return deriveSpatialPeLocalTransferOptionsForRealizations(
      dataflow, *producerRealization, *consumerRealization, fabric,
      computeBindings, logicalNet);
}

llvm::Expected<std::vector<SpatialPeLocalTransferOptionView>>
deriveSpatialPeLocalTransferOptionsForRealizations(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const TechComputeRealizationView &producerRealization,
    const TechComputeRealizationView &consumerRealization,
    const ::loom::fabric::FabricArtifactView &fabric,
    llvm::ArrayRef<SpatialComputeBindingView> computeBindings,
    const TechResidualLogicalNetView &logicalNet) {
  std::vector<SpatialPeLocalTransferOptionView> options;
  if (logicalNet.sinks.size() != 1)
    return options;
  const auto *producer =
      std::get_if<::dataflow::ActorTokenResultRef>(&logicalNet.producer);
  const auto *consumer =
      std::get_if<::dataflow::ActorTokenOperandRef>(&logicalNet.sinks.front());
  if (!producer || !consumer)
    return options;

  const TechComputeBoundaryView *producerBoundary =
      findOutputBoundary(producerRealization, *producer);
  const TechComputeBoundaryView *consumerBoundary =
      findInputBoundary(consumerRealization, *consumer);
  const SpatialComputeBindingView *producerBinding =
      findComputeBinding(computeBindings, producerRealization.entityId);
  const SpatialComputeBindingView *consumerBinding =
      findComputeBinding(computeBindings, consumerRealization.entityId);
  if (!producerBoundary || !consumerBoundary || !producerBinding ||
      !consumerBinding)
    return options;

  const auto producerPe = fabric.parentPeOf(producerBinding->occurrence);
  const auto consumerPe = fabric.parentPeOf(consumerBinding->occurrence);
  if (!producerPe || !consumerPe || *producerPe != *consumerPe ||
      fabric.peSchedule(*producerPe) != ::fabric::Schedule::Temporal ||
      producerBinding->context.pe != *producerPe ||
      consumerBinding->context.pe != *consumerPe)
    return options;

  const ::loom::fabric::FabricFuOccurrencePortRef sourcePort{
      producerBinding->occurrence, ::loom::fabric::FabricPortDirection::Output,
      producerBoundary->fabricPort.ordinal};
  const ::loom::fabric::FabricFuOccurrencePortRef sinkPort{
      consumerBinding->occurrence, ::loom::fabric::FabricPortDirection::Input,
      consumerBoundary->fabricPort.ordinal};
  const auto sourceEndpoint = fabric.fuOccurrenceTransportEndpoint(sourcePort);
  const auto sinkEndpoint = fabric.fuOccurrenceTransportEndpoint(sinkPort);
  if (!sourceEndpoint || !sinkEndpoint)
    return options;
  const auto sourcePath = fabric.transportEndpointDataPath(*sourceEndpoint);
  const auto sinkPath = fabric.transportEndpointDataPath(*sinkEndpoint);
  if (!sourcePath || !sinkPath || sourcePath->kind != sinkPath->kind ||
      sourcePath->payloadWidthBits != sinkPath->payloadWidthBits ||
      sourcePath->tagWidthBits != sinkPath->tagWidthBits)
    return options;

  auto producerType = dataflow.tokenType(logicalNet.producer);
  auto consumerType = dataflow.tokenType(logicalNet.sinks.front());
  if (!producerType)
    return producerType.takeError();
  if (!consumerType)
    return consumerType.takeError();
  if (*producerType != *consumerType)
    return options;

  auto schema = fabric.temporalPeConfigurationSchema(*producerPe);
  if (!schema)
    return schema.takeError();
  const auto &layout = schema->layout();
  if (layout.tagWidthBits == 0)
    return invalid("Temporal PE register FIFO has no Physical Tag width");
  options.reserve(layout.registerFifoCount);
  for (::loom::fabric::FabricOrdinal fifo = 0; fifo != layout.registerFifoCount;
       ++fifo) {
    const auto write =
        ::loom::fabric::FabricPhysicalTraversalRef::peRegisterFifo(
            *producerPe, fifo,
            ::loom::fabric::FabricRegisterFifoPathRole::Write);
    const auto read =
        ::loom::fabric::FabricPhysicalTraversalRef::peRegisterFifo(
            *producerPe, fifo,
            ::loom::fabric::FabricRegisterFifoPathRole::Read);
    if (llvm::Error error = ::loom::fabric::validateFabricRef(fabric, write))
      return std::move(error);
    if (llvm::Error error = ::loom::fabric::validateFabricRef(fabric, read))
      return std::move(error);
    options.push_back(SpatialPeLocalTransferOptionView{
        logicalNet.producer, logicalNet.sinks.front(), *producerPe, fifo, write,
        read, llvm::APInt(layout.tagWidthBits, 0)});
  }
  return options;
}

llvm::Expected<std::vector<SpatialMemoryActorRoleDemandView>>
deriveSpatialMemoryActorRoleDemands(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const TechMappingView &techMapping,
    const ::loom::fabric::FabricArtifactView &fabric,
    const TechMemoryRealizationView &realization,
    ::loom::fabric::FabricMemoryOccurrenceRef occurrence) {
  const auto selectedTemplate = fabric.memoryEngineTemplateOf(occurrence);
  if (!selectedTemplate || *selectedTemplate != realization.engine)
    return invalid("memory occurrence selects a different engine template");
  std::vector<SpatialMemoryActorRoleDemandView> result;
  result.reserve(realization.actors.size());
  for (const TechMemoryActorView &actor : realization.actors) {
    auto resolved = dataflow.resolve(actor.actor);
    if (!resolved)
      return resolved.takeError();
    auto service =
        ::dataflow::semantics::CanonicalService::forActor(resolved->op);
    if (!service)
      return service.takeError();
    if (actor.operandPorts.size() != service->arguments().size() ||
        actor.resultPorts.size() != service->results().size())
      return invalid("memory actor role correspondence has the wrong shape");

    SpatialMemoryActorRoleDemandView demand{actor.actor, occurrence, {}, {}};
    demand.sources.resize(memoryRoleCount);
    demand.destinations.resize(memoryRoleCount);
    for (auto [ordinal, argument] : llvm::enumerate(service->arguments())) {
      auto value = service->argumentValue(resolved->op, ordinal);
      if (!value)
        return value.takeError();
      const ::dataflow::CanonicalGraphConsumerEndpointRef consumer(
          ::dataflow::ActorTokenOperandRef{
              actor.actor, static_cast<::dataflow::StructuralOrdinal>(
                               (*value)->getOperandNumber())});
      const std::size_t role = static_cast<std::size_t>(argument.role);
      if (role >= demand.sources.size() || demand.sources[role])
        return invalid("memory actor repeats an active input role");
      const TechMemoryInternalEdgeView *internal =
          findInternalSource(realization, consumer);
      if (internal) {
        auto connection = deriveSpatialMemoryInternalConnectionOrdinal(
            fabric, occurrence, internal->connection);
        if (!connection)
          return connection.takeError();
        demand.sources[role] =
            ::loom::fabric::FabricMemoryHandshakeInternalRoleSource{
                *connection};
      } else {
        demand.sources[role] =
            ::loom::fabric::FabricMemoryHandshakeExternalRoleSource{
                actor.operandPorts[ordinal].ordinal};
      }
    }

    for (auto [ordinal, output] : llvm::enumerate(service->results())) {
      auto value = service->resultValue(resolved->op, ordinal);
      if (!value)
        return value.takeError();
      const ::dataflow::CanonicalGraphProducerEndpointRef producer(
          ::dataflow::ActorTokenResultRef{
              actor.actor, static_cast<::dataflow::StructuralOrdinal>(
                               value->getResultNumber())});
      const std::size_t role = static_cast<std::size_t>(output.role);
      if (role >= demand.destinations.size() || demand.destinations[role])
        return invalid("memory actor repeats an active output role");
      ::loom::fabric::FabricMemoryHandshakeRoleDestination destination;
      if (isResidualProducer(techMapping, producer))
        destination.externalEndpoint = actor.resultPorts[ordinal].ordinal;
      for (const TechMemoryInternalEdgeView &edge : realization.internalEdges) {
        if (edge.producer != producer)
          continue;
        auto connection = deriveSpatialMemoryInternalConnectionOrdinal(
            fabric, occurrence, edge.connection);
        if (!connection)
          return connection.takeError();
        destination.internalConnections.push_back(*connection);
      }
      llvm::sort(destination.internalConnections);
      destination.internalConnections.erase(
          std::unique(destination.internalConnections.begin(),
                      destination.internalConnections.end()),
          destination.internalConnections.end());
      if (!destination.externalEndpoint &&
          destination.internalConnections.empty())
        return invalid("memory result has no physical destination");
      demand.destinations[role] = std::move(destination);
    }
    result.push_back(std::move(demand));
  }

  std::vector<::loom::fabric::FabricMemoryInternalConnectionUse> uses;
  for (const SpatialMemoryActorRoleDemandView &demand : result) {
    for (const auto &source : demand.sources) {
      if (!source)
        continue;
      const auto *internal =
          std::get_if<::loom::fabric::FabricMemoryHandshakeInternalRoleSource>(
              &*source);
      if (internal)
        uses.push_back(
            {occurrence, internal->connection,
             ::loom::fabric::FabricMemoryInternalConnectionUseKind::Consumer});
    }
    for (const auto &destination : demand.destinations) {
      if (!destination)
        continue;
      for (::loom::fabric::FabricOrdinal connection :
           destination->internalConnections)
        uses.push_back(
            {occurrence, connection,
             ::loom::fabric::FabricMemoryInternalConnectionUseKind::Producer});
    }
  }
  switch (::loom::fabric::deriveFabricMemoryInternalConnectionClosure(uses)) {
  case ::loom::fabric::FabricMemoryInternalConnectionClosure::Closed:
    return result;
  case ::loom::fabric::FabricMemoryInternalConnectionClosure::Open:
    return invalid("memory internal connection is not closed");
  case ::loom::fabric::FabricMemoryInternalConnectionClosure::MultipleProducers:
    return invalid("memory internal connection has multiple producers");
  }
  llvm_unreachable("closed memory connection closure domain");
}

llvm::Expected<SpatialDurableProgressBoundaryKind>
classifySpatialAttachmentDurableProgressBoundary(
    const ::loom::fabric::FabricArtifactView &fabric,
    const ::loom::fabric::FabricPhysicalTraversalRef &attachment,
    std::optional<::loom::fabric::FabricFuOccurrencePortRef> fuPort) {
  if (llvm::Error error = ::loom::fabric::validateFabricRef(fabric, attachment))
    return std::move(error);

  if (const auto *fifo =
          std::get_if<::loom::fabric::FabricFifoTraversalPayload>(
              &attachment.payload))
    return fifo->mode == ::loom::fabric::FabricFifoTraversalMode::Buffered
               ? SpatialDurableProgressBoundaryKind::BufferedFifo
               : SpatialDurableProgressBoundaryKind::None;

  const auto *selector =
      std::get_if<::loom::fabric::FabricPeSelectorPayload>(&attachment.payload);
  if (!selector || !fuPort ||
      fuPort->direction != ::loom::fabric::FabricPortDirection::Input)
    return SpatialDurableProgressBoundaryKind::None;

  const auto pe = fabric.parentPeOf(fuPort->fu);
  const auto fixed = fabric.fuOccurrenceTransportEndpoint(*fuPort);
  if (!pe || !fixed || selector->owner != *pe ||
      selector->destination != *fixed)
    return invalid("PE input attachment does not reach its concrete FU port");
  if (fabric.peSchedule(*pe) != ::fabric::Schedule::Temporal ||
      !fabric.peOperandBufferMode(*pe) || fabric.peOperandBufferSize(*pe) == 0)
    return SpatialDurableProgressBoundaryKind::None;

  const auto attachments = fabric.fuOccurrencePortAttachments(*fuPort);
  if (!llvm::any_of(attachments, [&](const auto &candidate) {
        return candidate.localTraversal == attachment;
      }))
    return invalid("PE input attachment is absent from its Fabric domain");
  return SpatialDurableProgressBoundaryKind::TemporalPeOperandQueue;
}

llvm::Expected<std::optional<SpatialDurableProgressBoundaryView>>
deriveSpatialSinkDurableProgressBoundary(
    const TechMappingView &techMapping,
    const ::loom::fabric::FabricArtifactView &fabric,
    llvm::ArrayRef<SpatialComputeBindingView> computeBindings,
    const SpatialRouteTreeView &route, const SpatialRouteSinkView &sink) {
  if (sink.nodeOrdinal >= route.nodes.size())
    return invalid("durable route sink names an unknown terminal node");
  std::optional<::loom::fabric::FabricPhysicalTraversalRef> attachment =
      sink.localTraversal;
  if (attachment) {
    auto traversalOwned = classifySpatialAttachmentDurableProgressBoundary(
        fabric, *attachment, std::nullopt);
    if (!traversalOwned)
      return traversalOwned.takeError();
    if (*traversalOwned == SpatialDurableProgressBoundaryKind::BufferedFifo)
      return std::optional<SpatialDurableProgressBoundaryView>(
          SpatialDurableProgressBoundaryView{*traversalOwned, *attachment,
                                             std::nullopt});
  }

  const auto *operand =
      std::get_if<::dataflow::ActorTokenOperandRef>(&sink.sink);
  if (!operand)
    return std::optional<SpatialDurableProgressBoundaryView>();
  const TechComputeRealizationView *realization =
      findComputeRealization(techMapping, operand->actor);
  if (!realization)
    return std::optional<SpatialDurableProgressBoundaryView>();
  const TechComputeBoundaryView *boundary =
      findInputBoundary(*realization, *operand);
  const SpatialComputeBindingView *binding =
      findComputeBinding(computeBindings, realization->entityId);
  if (!boundary || !binding)
    return invalid("PE operand sink has no unique realization binding");

  const ::loom::fabric::FabricFuOccurrencePortRef port{
      binding->occurrence, ::loom::fabric::FabricPortDirection::Input,
      boundary->fabricPort.ordinal};
  if (!attachment) {
    const auto attachments = fabric.fuOccurrencePortAttachments(port);
    for (const auto &candidate : attachments) {
      if (candidate.endpoint != route.nodes[sink.nodeOrdinal].endpoint)
        continue;
      if (attachment)
        return invalid("shared PE ingress resolves multiple FU attachments");
      attachment = candidate.localTraversal;
    }
    if (!attachment)
      return std::optional<SpatialDurableProgressBoundaryView>();
  }
  auto kind = classifySpatialAttachmentDurableProgressBoundary(
      fabric, *attachment, port);
  if (!kind)
    return kind.takeError();
  if (*kind == SpatialDurableProgressBoundaryKind::None)
    return std::optional<SpatialDurableProgressBoundaryView>();
  if (*kind != SpatialDurableProgressBoundaryKind::TemporalPeOperandQueue)
    return invalid("PE operand sink acquired an incompatible boundary kind");

  const auto pe = fabric.parentPeOf(binding->occurrence);
  if (!pe || binding->context.pe != *pe ||
      binding->context.ordinal >= fabric.peResidentContextCount(*pe))
    return invalid("PE operand queue has an incompatible resident context");
  auto schema = fabric.temporalPeConfigurationSchema(*pe);
  if (!schema)
    return schema.takeError();
  const auto shape = llvm::find_if(schema->layout().fus, [&](const auto &row) {
    return row.fu == binding->occurrence;
  });
  if (shape == schema->layout().fus.end() ||
      boundary->fabricPort.ordinal >= shape->inputCount)
    return invalid("PE operand queue has no concrete FU input");
  const ::loom::fabric::FabricOrdinal fuOrdinal =
      static_cast<::loom::fabric::FabricOrdinal>(
          std::distance(schema->layout().fus.begin(), shape));
  return std::optional<SpatialDurableProgressBoundaryView>(
      SpatialDurableProgressBoundaryView{
          *kind, *attachment,
          ::fabric::LogicalOperandQueueKey{binding->context, fuOrdinal,
                                           boundary->fabricPort.ordinal}});
}

llvm::Expected<std::vector<SpatialPeOperandQueueMatchGroupView>>
deriveSpatialPeOperandQueueMatchGroups(
    const TechMappingView &techMapping,
    const ::loom::fabric::FabricArtifactView &fabric,
    llvm::ArrayRef<SpatialComputeBindingView> computeBindings,
    llvm::ArrayRef<SpatialRouteTreeView> routes,
    llvm::ArrayRef<SpatialResourceUseView> resourceUses,
    llvm::ArrayRef<SpatialPhysicalTagSegmentView> physicalTagSegments) {
  std::map<std::vector<std::uint8_t>, SpatialPeOperandQueueMatchGroupView>
      groups;
  for (auto [routeOrdinal, route] : llvm::enumerate(routes)) {
    for (const SpatialRouteSinkView &sink : route.sinks) {
      auto boundary = deriveSpatialSinkDurableProgressBoundary(
          techMapping, fabric, computeBindings, route, sink);
      if (!boundary)
        return boundary.takeError();
      if (!*boundary ||
          (*boundary)->kind !=
              SpatialDurableProgressBoundaryKind::TemporalPeOperandQueue)
        continue;
      if (!(*boundary)->operandQueue || sink.nodeOrdinal >= route.nodes.size())
        return invalid("PE operand boundary has no queue or route node");
      const auto *selector =
          std::get_if<::loom::fabric::FabricPeSelectorPayload>(
              &(*boundary)->attachment.payload);
      if (!selector)
        return invalid("PE operand boundary has no selector traversal");
      auto tag = detail::resolveConfiguredHardwarePhysicalTag(
          fabric, routes, resourceUses, physicalTagSegments, routeOrdinal,
          sink.nodeOrdinal);
      if (!tag)
        return tag.takeError();
      const auto dataPath = fabric.transportEndpointDataPath(selector->source);
      if (!dataPath || dataPath->kind != ::fabric::DataPathKind::BitsTag ||
          dataPath->tagWidthBits != tag->getBitWidth())
        return invalid("PE operand match group has an incompatible tag");

      const std::vector<std::uint8_t> key =
          matchGroupKey(selector->source, *tag);
      auto queueSchema = fabric.temporalPeConfigurationSchema(
          (*boundary)->operandQueue->context.pe);
      if (!queueSchema)
        return queueSchema.takeError();
      if ((*boundary)->operandQueue->fuOccurrence >=
          queueSchema->layout().fus.size())
        return invalid("PE operand queue has no concrete FU occurrence");
      const auto concreteFu =
          queueSchema->layout().fus[(*boundary)->operandQueue->fuOccurrence].fu;
      auto [found, inserted] = groups.try_emplace(
          key, SpatialPeOperandQueueMatchGroupView{
                   route.logicalNet, selector->source, *tag, {}});
      if (!inserted && found->second.logicalNet != route.logicalNet)
        return invalid("PE operand match group spans multiple logical nets");
      auto queue = llvm::find_if(found->second.matches, [&](const auto &match) {
        return match.queue == *(*boundary)->operandQueue;
      });
      if (queue == found->second.matches.end()) {
        found->second.matches.push_back(
            {{sink.sink}, *(*boundary)->operandQueue, concreteFu, 0, 0});
      } else {
        if (llvm::is_contained(queue->consumers, sink.sink))
          return invalid("PE operand match group repeats a consumer");
        queue->consumers.push_back(sink.sink);
      }
    }
  }

  std::vector<SpatialPeOperandQueueMatchGroupView> result;
  result.reserve(groups.size());
  for (auto &[key, group] : groups) {
    (void)key;
    llvm::sort(group.matches, [](const auto &lhs, const auto &rhs) {
      return lhs.queue < rhs.queue;
    });
    for (auto &match : group.matches) {
      if (llvm::any_of(match.consumers, [](const auto &consumer) {
            return !std::holds_alternative<::dataflow::ActorTokenOperandRef>(
                consumer);
          }))
        return invalid("PE operand queue has a non-actor consumer");
      llvm::sort(match.consumers, [](const auto &lhs, const auto &rhs) {
        const auto &lhsOperand =
            std::get<::dataflow::ActorTokenOperandRef>(lhs);
        const auto &rhsOperand =
            std::get<::dataflow::ActorTokenOperandRef>(rhs);
        return std::make_pair(lhsOperand.actor.entity.value(),
                              lhsOperand.ordinal) <
               std::make_pair(rhsOperand.actor.entity.value(),
                              rhsOperand.ordinal);
      });
    }
    if (llvm::Error error = verifyMatchGroupCapacity(fabric, group))
      return std::move(error);
    result.push_back(std::move(group));
  }
  return result;
}

llvm::Expected<SpatialPeOperandProgressFeedback>
deriveSpatialPeOperandProgressFeedback(
    llvm::ArrayRef<SpatialPeOperandQueueMatchGroupView> groups) {
  if (groups.empty())
    return SpatialPeOperandProgressFeedback{
        SpatialPeOperandProgressStatus::Safe,
        SpatialPeOperandProgressSupport::Exact, 0, 0, 0, 0, 0, 0, 0, 0, {},
        {}, std::nullopt};
  SpatialPeOperandProgressFeedback result;
  result.status = SpatialPeOperandProgressStatus::Safe;
  result.support = SpatialPeOperandProgressSupport::Exact;
  result.groupCount = groups.size();
  std::set<std::vector<std::uint8_t>> ingressKeys;
  std::map<std::vector<std::uint8_t>, SpatialPeOperandPairingProjection>
      pairingProjections;
  for (const SpatialPeOperandQueueMatchGroupView &group : groups) {
    ingressKeys.insert(::loom::fabric::canonicalFabricBytes(group.ingress));
    result.sharedIngressCount += group.matches.size() > 1;
    for (const SpatialPeOperandQueueMatchView &match : group.matches) {
      SpatialPeOperandQualifiedPairingKey key{match.queue.context, match.fu,
                                              group.tag};
      const auto contextBytes =
          ::loom::fabric::canonicalFabricBytes(match.queue.context);
      const auto fuBytes = ::loom::fabric::canonicalFabricBytes(match.fu);
      std::vector<std::uint8_t> encoded;
      for (int shift = 56; shift >= 0; shift -= 8)
        encoded.push_back(static_cast<std::uint8_t>(contextBytes.size() >> shift));
      encoded.insert(encoded.end(), contextBytes.begin(), contextBytes.end());
      for (int shift = 56; shift >= 0; shift -= 8)
        encoded.push_back(static_cast<std::uint8_t>(fuBytes.size() >> shift));
      encoded.insert(encoded.end(), fuBytes.begin(), fuBytes.end());
      encoded.push_back(static_cast<std::uint8_t>(group.tag.getBitWidth()));
      const unsigned byteCount = (group.tag.getBitWidth() + 7) / 8;
      for (unsigned byte = 0; byte != byteCount; ++byte)
        encoded.push_back(static_cast<std::uint8_t>(
            group.tag.extractBitsAsZExtValue(
                std::min<unsigned>(8, group.tag.getBitWidth() - byte * 8),
                byte * 8)));
      auto [found, inserted] = pairingProjections.try_emplace(
          std::move(encoded),
          SpatialPeOperandPairingProjection{key, {}, {}, {}});
      (void)inserted;
      found->second.requiredInputRoles.push_back(match.queue.fuInput);
      found->second.ingresses.push_back(group.ingress);
      found->second.allocationUnits.push_back(match.allocationUnit);
    }
  }
  result.distinctIngressCount = ingressKeys.size();
  result.pairingKeyCount = 0;
  for (const auto &group : groups)
    result.pairingKeyCount += group.matches.size();
  result.distinctPairingKeyCount = pairingProjections.size();
  for (auto &[key, value] : pairingProjections) {
    (void)key;
    llvm::sort(value.requiredInputRoles);
    value.requiredInputRoles.erase(
        std::unique(value.requiredInputRoles.begin(),
                    value.requiredInputRoles.end()),
        value.requiredInputRoles.end());
    llvm::sort(value.ingresses, [](const auto &lhs, const auto &rhs) {
      return ::loom::fabric::canonicalFabricBytes(lhs) <
             ::loom::fabric::canonicalFabricBytes(rhs);
    });
    value.ingresses.erase(
        std::unique(value.ingresses.begin(), value.ingresses.end()),
        value.ingresses.end());
    llvm::sort(value.allocationUnits);
    value.allocationUnits.erase(
        std::unique(value.allocationUnits.begin(),
                    value.allocationUnits.end()),
        value.allocationUnits.end());
    result.pairingKeys.push_back(value.key);
    result.pairingOpportunityCount += value.ingresses.size() > 1;
    const auto qualifiedContext = value.key.context;
    const auto qualifiedFu = value.key.fu;
    const auto qualifiedTag = value.key.tag;
    if (value.requiredInputRoles.size() > 1 && value.ingresses.size() == 1) {
      const bool orderedKnown = llvm::any_of(groups, [&](const auto &group) {
        return group.orderedCorrespondenceKnown && group.tag == qualifiedTag &&
               llvm::any_of(group.matches, [&](const auto &match) {
                 return match.queue.context == qualifiedContext &&
                        match.fu == qualifiedFu;
               });
      });
      if (orderedKnown) {
        ++result.potentiallyBlockingGroupCount;
        result.status = SpatialPeOperandProgressStatus::LikelyRisk;
        result.support = SpatialPeOperandProgressSupport::Analytic;
      } else {
        ++result.unknownPairingGroupCount;
        result.status = SpatialPeOperandProgressStatus::ProofNotEstablished;
        result.support = SpatialPeOperandProgressSupport::Unsupported;
      }
    }
    result.pairings.push_back(std::move(value));
  }
  static constexpr llvm::StringLiteral descriptor{
      "loom.mapping.temporal_operand_progress_projection.1"};
  std::vector<std::uint8_t> canonical;
  const auto appendU64 = [&](std::uint64_t value) {
    for (int shift = 56; shift >= 0; shift -= 8)
      canonical.push_back(static_cast<std::uint8_t>(value >> shift));
  };
  appendU64(result.groupCount);
  appendU64(result.pairingOpportunityCount);
  appendU64(result.pairingKeyCount);
  appendU64(result.distinctPairingKeyCount);
  for (const auto &pairing : result.pairings) {
    const auto context =
        ::loom::fabric::canonicalFabricBytes(pairing.key.context);
    const auto fu = ::loom::fabric::canonicalFabricBytes(pairing.key.fu);
    appendU64(context.size());
    canonical.insert(canonical.end(), context.begin(), context.end());
    appendU64(fu.size());
    canonical.insert(canonical.end(), fu.begin(), fu.end());
    appendU64(pairing.key.tag.getBitWidth());
    for (unsigned byte = 0;
         byte != (pairing.key.tag.getBitWidth() + 7) / 8; ++byte)
      canonical.push_back(static_cast<std::uint8_t>(
          pairing.key.tag.extractBitsAsZExtValue(
              std::min<unsigned>(8, pairing.key.tag.getBitWidth() - byte * 8),
              byte * 8)));
    appendU64(pairing.requiredInputRoles.size());
    for (std::uint32_t role : pairing.requiredInputRoles)
      appendU64(role);
    appendU64(pairing.ingresses.size());
    for (const auto &ingress : pairing.ingresses) {
      const auto bytes = ::loom::fabric::canonicalFabricBytes(ingress);
      appendU64(bytes.size());
      canonical.insert(canonical.end(), bytes.begin(), bytes.end());
    }
    appendU64(pairing.allocationUnits.size());
    for (std::uint32_t unit : pairing.allocationUnits)
      appendU64(unit);
  }
  auto digest = ::loom::computeComponentViewDigest(
      {reinterpret_cast<const std::uint8_t *>(descriptor.data()),
       descriptor.size()},
      canonical);
  if (!digest)
    return digest.takeError();
  result.projectionDigest = *digest;
  return result;
}

llvm::Expected<SpatialPeOperandRuntimeWitness>
deriveSpatialPeOperandRuntimeWitness(
    const SpatialPeOperandProgressFeedback &projection,
    llvm::ArrayRef<SpatialPeOperandRuntimeHeadView> heads) {
  if (!projection.projectionDigest)
    return invalid("PE operand runtime witness has no Mapping projection "
                   "digest");
  SpatialPeOperandRuntimeWitness result;
  result.projectionDigest = projection.projectionDigest;
  result.observedHeadCount = heads.size();
  if (projection.pairings.empty() && !heads.empty())
    return invalid("PE operand runtime heads exist without a selected "
                   "pairing projection");

  std::vector<SpatialPeOperandRuntimeHeadView> canonicalHeads(heads.begin(),
                                                              heads.end());
  llvm::sort(canonicalHeads, [](const auto &lhs, const auto &rhs) {
    if (lhs.queue != rhs.queue)
      return lhs.queue < rhs.queue;
    const auto lhsFu = ::loom::fabric::canonicalFabricBytes(lhs.fu);
    const auto rhsFu = ::loom::fabric::canonicalFabricBytes(rhs.fu);
    if (lhsFu != rhsFu)
      return lhsFu < rhsFu;
    if (lhs.tag.getBitWidth() != rhs.tag.getBitWidth())
      return lhs.tag.getBitWidth() < rhs.tag.getBitWidth();
    return lhs.tag.ult(rhs.tag);
  });

  std::set<std::tuple<std::vector<std::uint8_t>,
                      std::vector<std::uint8_t>, std::uint32_t>>
      observedQueues;
  for (const SpatialPeOperandRuntimeHeadView &head : canonicalHeads) {
    const auto context =
        ::loom::fabric::canonicalFabricBytes(head.queue.context);
    const auto fu = ::loom::fabric::canonicalFabricBytes(head.fu);
    if (!observedQueues.emplace(context, fu, head.queue.fuInput).second)
      return invalid("PE operand runtime witness repeats a QueueKey");
    if (head.occupancy > head.capacity ||
        head.reservations > head.capacity - head.occupancy)
      return invalid("PE operand runtime witness has invalid occupancy");
    if (head.occupancy == head.capacity)
      ++result.fullQueueCount;
    if (head.exactHead)
      ++result.exactHeadCount;
    if (head.occupancy != 0 && !head.exactHead)
      ++result.mismatchedHeadCount;
  }

  for (const SpatialPeOperandPairingProjection &pairing :
       projection.pairings) {
    std::optional<std::uint64_t> expectedSequence;
    bool complete = true;
    bool pairingMismatch = false;
    for (std::uint32_t role : pairing.requiredInputRoles) {
      const auto found = llvm::find_if(canonicalHeads, [&](const auto &head) {
        return head.queue.context == pairing.key.context &&
               head.fu == pairing.key.fu &&
               head.tag.getBitWidth() == pairing.key.tag.getBitWidth() &&
               head.tag == pairing.key.tag &&
               head.queue.fuInput == role;
      });
      if (found == canonicalHeads.end() || !found->exactHead) {
        complete = false;
        continue;
      }
      if (expectedSequence &&
          *expectedSequence != found->headProducerSequenceOrdinal) {
        ++result.mismatchedHeadCount;
        pairingMismatch = true;
      } else {
        expectedSequence = found->headProducerSequenceOrdinal;
      }
    }
    if (complete && expectedSequence && !pairingMismatch)
      ++result.matchedPairingKeyCount;
    else
      ++result.unmatchedPairingKeyCount;
  }

  if (result.mismatchedHeadCount != 0) {
    result.status = SpatialPeOperandRuntimeWitnessStatus::Unsupported;
    result.support = SpatialPeOperandProgressSupport::Unsupported;
  } else if (result.unmatchedPairingKeyCount != 0 ||
             result.exactHeadCount != result.observedHeadCount) {
    result.status =
        SpatialPeOperandRuntimeWitnessStatus::ProofNotEstablished;
    result.support = SpatialPeOperandProgressSupport::Unsupported;
  } else {
    result.status = SpatialPeOperandRuntimeWitnessStatus::Exact;
    result.support = SpatialPeOperandProgressSupport::Exact;
  }
  static constexpr llvm::StringLiteral descriptor{
      "loom.mapping.temporal_operand_runtime_witness.1"};
  std::vector<std::uint8_t> canonical;
  const auto appendU64 = [&](std::uint64_t value) {
    for (int shift = 56; shift >= 0; shift -= 8)
      canonical.push_back(static_cast<std::uint8_t>(value >> shift));
  };
  canonical.insert(canonical.end(), projection.projectionDigest->bytes().begin(),
                   projection.projectionDigest->bytes().end());
  appendU64(canonicalHeads.size());
  for (const SpatialPeOperandRuntimeHeadView &head : canonicalHeads) {
    const auto context =
        ::loom::fabric::canonicalFabricBytes(head.queue.context);
    const auto fu = ::loom::fabric::canonicalFabricBytes(head.fu);
    appendU64(context.size());
    canonical.insert(canonical.end(), context.begin(), context.end());
    appendU64(fu.size());
    canonical.insert(canonical.end(), fu.begin(), fu.end());
    appendU64(head.queue.fuOccurrence);
    appendU64(head.queue.fuInput);
    appendU64(head.tag.getBitWidth());
    for (unsigned byte = 0; byte != (head.tag.getBitWidth() + 7) / 8; ++byte)
      canonical.push_back(static_cast<std::uint8_t>(
          head.tag.extractBitsAsZExtValue(
              std::min<unsigned>(8, head.tag.getBitWidth() - byte * 8),
              byte * 8)));
    appendU64(head.allocationUnit);
    appendU64(head.capacity);
    appendU64(head.occupancy);
    appendU64(head.reservations);
    appendU64(head.headBindingOrdinal);
    appendU64(head.headOccurrenceOrdinal);
    appendU64(head.headProducerSequenceOrdinal);
    canonical.push_back(head.exactHead ? 1 : 0);
  }
  auto digest = ::loom::computeComponentViewDigest(
      {reinterpret_cast<const std::uint8_t *>(descriptor.data()),
       descriptor.size()},
      canonical);
  if (!digest)
    return digest.takeError();
  result.projectionDigest = *digest;
  return result;
}

} // namespace loom::mapping
