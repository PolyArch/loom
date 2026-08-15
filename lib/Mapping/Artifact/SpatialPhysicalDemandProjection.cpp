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

} // namespace

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
    const SpatialRouteSinkView &sink) {
  if (!sink.localTraversal)
    return std::optional<SpatialDurableProgressBoundaryView>();

  auto traversalOwned = classifySpatialAttachmentDurableProgressBoundary(
      fabric, *sink.localTraversal, std::nullopt);
  if (!traversalOwned)
    return traversalOwned.takeError();
  if (*traversalOwned == SpatialDurableProgressBoundaryKind::BufferedFifo)
    return std::optional<SpatialDurableProgressBoundaryView>(
        SpatialDurableProgressBoundaryView{*traversalOwned,
                                           *sink.localTraversal, std::nullopt});

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
  auto kind = classifySpatialAttachmentDurableProgressBoundary(
      fabric, *sink.localTraversal, port);
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
          *kind, *sink.localTraversal,
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
          techMapping, fabric, computeBindings, sink);
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
      auto [found, inserted] = groups.try_emplace(
          key, SpatialPeOperandQueueMatchGroupView{
                   route.logicalNet, selector->source, *tag, {}});
      if (!inserted && found->second.logicalNet != route.logicalNet)
        return invalid("PE operand match group spans multiple logical nets");
      if (llvm::any_of(found->second.matches, [&](const auto &match) {
            return match.queue == *(*boundary)->operandQueue;
          }))
        return invalid("PE operand match group repeats a logical queue");
      found->second.matches.push_back(
          {sink.sink, *(*boundary)->operandQueue, 0, 0});
    }
  }

  std::vector<SpatialPeOperandQueueMatchGroupView> result;
  result.reserve(groups.size());
  for (auto &[key, group] : groups) {
    (void)key;
    llvm::sort(group.matches, [](const auto &lhs, const auto &rhs) {
      return lhs.queue < rhs.queue;
    });
    if (llvm::Error error = verifyMatchGroupCapacity(fabric, group))
      return std::move(error);
    result.push_back(std::move(group));
  }
  return result;
}

} // namespace loom::mapping
