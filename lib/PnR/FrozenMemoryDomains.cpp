#include "FrozenMemoryDomains.h"
#include "EndpointMatching.h"

#include "Mapping/FabricOccurrenceIndex.h"
#include "Mapping/Verifier.h"

#include "llvm/Support/Error.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

using namespace loom::mapping;
using namespace loom::mapping::detail;
using namespace loom::pnr;
using namespace loom::pnr::detail;

namespace {

constexpr llvm::StringLiteral frozenArtifact = "FrozenRealizationGraph";
constexpr PnrCapacityContext memoryCountContext{
    frozenArtifact, "memory_realizations", "memory_realizations",
    PnrCapacityMeasure::Count};
constexpr PnrCapacityContext occurrenceCountContext{
    frozenArtifact, "fabric_memory_occurrences", "memory_occurrences",
    PnrCapacityMeasure::Count};
constexpr PnrCapacityContext endpointCountContext{
    frozenArtifact, "memory_physical_endpoints", "memory_endpoints",
    PnrCapacityMeasure::Count};
constexpr PnrCapacityContext endpointIndexContext{
    frozenArtifact, "memory_physical_endpoints", "memory_endpoints",
    PnrCapacityMeasure::Index};
constexpr PnrCapacityContext endpointOffsetContext{
    frozenArtifact, "fabric_memory_occurrences", "memory_physical_endpoints",
    PnrCapacityMeasure::Offset};
constexpr PnrCapacityContext endpointTypeCountContext{
    frozenArtifact, "memory_physical_endpoint_compatible_types", "types",
    PnrCapacityMeasure::Count};
constexpr PnrCapacityContext endpointTypeOffsetContext{
    frozenArtifact, "memory_physical_endpoints",
    "memory_physical_endpoint_compatible_types", PnrCapacityMeasure::Offset};
constexpr PnrCapacityContext localArcCountContext{
    frozenArtifact, "memory_local_arcs", "memory_local_arcs",
    PnrCapacityMeasure::Count};
constexpr PnrCapacityContext localArcOffsetContext{
    frozenArtifact, "fabric_memory_occurrences", "memory_local_arcs",
    PnrCapacityMeasure::Offset};
constexpr PnrCapacityContext implementationCountContext{
    frozenArtifact, "memory_implementation_occurrences",
    "memory_implementation_occurrences", PnrCapacityMeasure::Count};
constexpr PnrCapacityContext implementationIndexContext{
    frozenArtifact, "memory_implementation_occurrences",
    "memory_implementation_occurrences", PnrCapacityMeasure::Index};
constexpr PnrCapacityContext implementationOffsetContext{
    frozenArtifact, "memory_realizations", "memory_implementation_occurrences",
    PnrCapacityMeasure::Offset};
constexpr PnrCapacityContext portDemandCountContext{
    frozenArtifact, "memory_port_demands", "memory_port_demands",
    PnrCapacityMeasure::Count};
constexpr PnrCapacityContext portDemandOffsetContext{
    frozenArtifact, "memory_implementation_occurrences", "memory_port_demands",
    PnrCapacityMeasure::Offset};
constexpr PnrCapacityContext endpointDomainCountContext{
    frozenArtifact, "compatible_memory_endpoints", "memory_physical_endpoints",
    PnrCapacityMeasure::Count};
constexpr PnrCapacityContext endpointDomainOffsetContext{
    frozenArtifact, "memory_port_demands", "compatible_memory_endpoints",
    PnrCapacityMeasure::Offset};
constexpr PnrCapacityContext realizationIndexContext{
    frozenArtifact, "actor_ownerships", "realizations",
    PnrCapacityMeasure::Index};
constexpr PnrCapacityContext portIndexContext{
    frozenArtifact, "memory_terminals", "ports", PnrCapacityMeasure::Index};

llvm::Error freezeError(std::string message) {
  return llvm::make_error<llvm::StringError>(
      std::move(message), std::make_error_code(std::errc::invalid_argument));
}

llvm::Error mappingInfeasibility(FrozenMappingInfeasibilityCode code,
                                 MemoryRealizationId realization,
                                 std::string message) {
  return llvm::make_error<FrozenMappingInfeasibility>(
      code, FrozenRealizationId{realization}, std::move(message));
}

std::uint64_t sizeValue(std::size_t size) {
  static_assert(sizeof(std::size_t) <= sizeof(std::uint64_t));
  return static_cast<std::uint64_t>(size);
}

llvm::Expected<PnrIndex> checked(PnrCapacityContext context,
                                 std::size_t value) {
  return checkedPnrIndex(context, sizeValue(value));
}

llvm::Expected<PnrIndex> checkedPort(std::uint32_t value) {
  return checkedPnrIndex(portIndexContext, value);
}

struct TypeKeyLess {
  bool operator()(TypeKey lhs, TypeKey rhs) const {
    return lhs.value() < rhs.value();
  }
};

bool supportsDemand(const ValidatedMemoryEndpoint &endpoint,
                    llvm::ArrayRef<TypeKey> compatibleTypes,
                    const ValidatedMemoryLocalArc &arc,
                    const ValidatedMemoryBoundaryPort &demand) {
  const PortDescriptor &descriptor = demand.descriptor;
  return endpoint.direction == demand.direction &&
         endpoint.kind == descriptor.kind && endpoint.role == descriptor.role &&
         endpoint.payloadCapacityBits >= descriptor.payloadWidthBits &&
         endpoint.tagCapacityBits >= descriptor.tagWidthBits &&
         arc.payloadCapacityBits >= descriptor.payloadWidthBits &&
         arc.tagCapacityBits >= descriptor.tagWidthBits &&
         std::binary_search(compatibleTypes.begin(), compatibleTypes.end(),
                            descriptor.type, TypeKeyLess{});
}

} // namespace

llvm::Error loom::pnr::detail::preflightFrozenMemoryDomainsCapacity(
    std::uint64_t memoryRealizationCount, std::uint64_t memoryOccurrenceCount,
    std::uint64_t memoryEndpointCount, std::uint64_t compatibleTypeCount,
    std::uint64_t localArcCount) {
  if (llvm::Error error =
          preflightPnrIndexCapacity(memoryCountContext, memoryRealizationCount))
    return error;
  if (llvm::Error error = preflightPnrIndexCapacity(occurrenceCountContext,
                                                    memoryOccurrenceCount))
    return error;
  if (llvm::Error error =
          preflightPnrIndexCapacity(endpointCountContext, memoryEndpointCount))
    return error;
  if (llvm::Error error = preflightPnrIndexCapacity(endpointTypeCountContext,
                                                    compatibleTypeCount))
    return error;
  return preflightPnrIndexCapacity(localArcCountContext, localArcCount);
}

llvm::Expected<FrozenMemoryDomains> loom::pnr::detail::buildFrozenMemoryDomains(
    const FabricHardwareView &fabric, const ValidatedTechMapping &mapping,
    llvm::ArrayRef<const MemoryRealizationDraft *> realizations) {
  const ValidatedFabricProjection &projection =
      ValidatedTechMappingAccess::fabricProjection(mapping);
  const ValidatedTechMappingProjection &mappingProjection =
      ValidatedTechMappingAccess::mappingProjection(mapping);
  if (projection.identity != fabric.identity)
    return freezeError("cannot freeze memory domains: validated Fabric "
                       "projection identity does not match the input");
  if (mappingProjection.memoryRealizations.size() != realizations.size())
    return freezeError("cannot freeze memory domains: validated Mapping "
                       "projection is incomplete");

  if (llvm::Error error = preflightFrozenMemoryDomainsCapacity(
          sizeValue(realizations.size()),
          sizeValue(projection.memoryOccurrences.size()),
          sizeValue(projection.memoryEndpoints.size()),
          sizeValue(projection.memoryEndpointCompatibleTypes.size()),
          sizeValue(projection.memoryLocalArcs.size())))
    return std::move(error);

  FrozenMemoryDomains result;
  result.occurrences.reserve(projection.memoryOccurrences.size());
  result.endpoints.reserve(projection.memoryEndpoints.size());
  result.endpointCompatibleTypes = projection.memoryEndpointCompatibleTypes;
  result.localArcs.reserve(projection.memoryLocalArcs.size());

  for (const ValidatedMemoryOccurrence &occurrence :
       projection.memoryOccurrences) {
    const FabricMemoryOccurrenceRef occurrenceRef{occurrence.id};
    auto endpointOffset =
        checked(endpointOffsetContext, occurrence.endpointOffset);
    if (!endpointOffset)
      return endpointOffset.takeError();
    auto endpointCount =
        checked(endpointCountContext, occurrence.endpointCount);
    if (!endpointCount)
      return endpointCount.takeError();
    if (llvm::Error error = preflightFrozenRangeCapacity(
            endpointOffsetContext, *endpointOffset, *endpointCount))
      return std::move(error);
    auto localArcOffset =
        checked(localArcOffsetContext, occurrence.localArcOffset);
    if (!localArcOffset)
      return localArcOffset.takeError();
    auto localArcCount =
        checked(localArcCountContext, occurrence.localArcCount);
    if (!localArcCount)
      return localArcCount.takeError();
    if (llvm::Error error = preflightFrozenRangeCapacity(
            localArcOffsetContext, *localArcOffset, *localArcCount))
      return std::move(error);
    result.occurrences.push_back({occurrenceRef, occurrence.implementation,
                                  *endpointOffset, *endpointCount,
                                  *localArcOffset, *localArcCount});

    llvm::ArrayRef<ValidatedMemoryEndpoint> endpoints(
        projection.memoryEndpoints);
    endpoints =
        endpoints.slice(occurrence.endpointOffset, occurrence.endpointCount);
    for (const ValidatedMemoryEndpoint &endpoint : endpoints) {
      auto typeOffset =
          checked(endpointTypeOffsetContext, endpoint.compatibleTypeOffset);
      if (!typeOffset)
        return typeOffset.takeError();
      auto typeCount =
          checked(endpointTypeCountContext, endpoint.compatibleTypeCount);
      if (!typeCount)
        return typeCount.takeError();
      if (llvm::Error error = preflightFrozenRangeCapacity(
              endpointTypeOffsetContext, *typeOffset, *typeCount))
        return std::move(error);
      result.endpoints.push_back(
          {occurrenceRef, endpoint.id, endpoint.direction, endpoint.kind,
           endpoint.payloadCapacityBits, endpoint.tagCapacityBits, *typeOffset,
           *typeCount, endpoint.role});
    }

    llvm::ArrayRef<ValidatedMemoryLocalArc> localArcs(
        projection.memoryLocalArcs);
    localArcs =
        localArcs.slice(occurrence.localArcOffset, occurrence.localArcCount);
    for (const ValidatedMemoryLocalArc &arc : localArcs) {
      auto port = checkedPort(arc.port);
      if (!port)
        return port.takeError();
      auto endpoint = checkedPnrIndexAdd(endpointIndexContext,
                                         sizeValue(occurrence.endpointOffset),
                                         sizeValue(arc.endpoint));
      if (!endpoint)
        return endpoint.takeError();
      result.localArcs.push_back({occurrenceRef, arc.operation, arc.direction,
                                  *port, *endpoint, arc.payloadCapacityBits,
                                  arc.tagCapacityBits});
    }
  }

  result.realizations.reserve(realizations.size());
  EndpointMatchingScratch scratch;
  for (std::size_t realizationIndex = 0; realizationIndex < realizations.size();
       ++realizationIndex) {
    const MemoryRealizationDraft &realization = *realizations[realizationIndex];
    const ValidatedMemoryRealizationProjection &selected =
        mappingProjection.memoryRealizations[realizationIndex];
    if (selected.id != realization.id ||
        selected.encoding != realization.encoding.entity)
      return freezeError("cannot freeze memory domains: validated Mapping "
                         "projection does not match the realization");
    auto frozenRealizationIndex =
        checked(realizationIndexContext, realizationIndex);
    if (!frozenRealizationIndex)
      return frozenRealizationIndex.takeError();
    auto implDomainOffset = checked(implementationOffsetContext,
                                    result.implementationOccurrences.size());
    if (!implDomainOffset)
      return implDomainOffset.takeError();
    const llvm::ArrayRef<std::size_t> candidates =
        findMemoryOccurrences(projection, selected.implementation);
    if (candidates.empty())
      return mappingInfeasibility(
          FrozenMappingInfeasibilityCode::EmptyConcreteMemoryDomain,
          realization.id,
          "memory realization has an empty concrete occurrence domain");

    bool hasUnaryEligibleOccurrence = false;
    for (std::size_t memoryOccurrenceIndex : candidates) {
      const ValidatedMemoryOccurrence &occurrence =
          projection.memoryOccurrences[memoryOccurrenceIndex];
      auto implementationIndex = checked(
          implementationIndexContext, result.implementationOccurrences.size());
      if (!implementationIndex)
        return implementationIndex.takeError();
      auto portDemandOffset =
          checked(portDemandOffsetContext, result.portDemands.size());
      if (!portDemandOffset)
        return portDemandOffset.takeError();
      scratch.reset(occurrence.endpointCount);

      for (const ValidatedMemoryBoundaryPort &demand :
           selected.activeBoundaryPorts) {
        auto endpointDomainOffset = checked(endpointDomainOffsetContext,
                                            result.compatibleEndpoints.size());
        if (!endpointDomainOffset)
          return endpointDomainOffset.takeError();
        const std::size_t scratchOffset = scratch.beginDomain();
        for (const ValidatedMemoryLocalArc &arc : findMemoryPortArcs(
                 projection, memoryOccurrenceIndex, demand.operation,
                 demand.direction, demand.port)) {
          const ValidatedMemoryEndpoint &endpoint =
              projection
                  .memoryEndpoints[occurrence.endpointOffset + arc.endpoint];
          llvm::ArrayRef<TypeKey> compatibleTypes(
              projection.memoryEndpointCompatibleTypes);
          compatibleTypes = compatibleTypes.slice(endpoint.compatibleTypeOffset,
                                                  endpoint.compatibleTypeCount);
          if (supportsDemand(endpoint, compatibleTypes, arc, demand))
            scratch.addEndpoint(arc.endpoint);
        }
        const EndpointDomainRange domain = scratch.endDomain(scratchOffset);
        auto endpointDomainCount =
            checked(endpointDomainCountContext, domain.count);
        if (!endpointDomainCount)
          return endpointDomainCount.takeError();
        if (llvm::Error error = preflightFrozenRangeCapacity(
                endpointDomainOffsetContext, *endpointDomainOffset,
                *endpointDomainCount))
          return std::move(error);
        for (std::size_t localEndpoint : scratch.endpoints(domain)) {
          auto endpoint = checkedPnrIndexAdd(
              endpointIndexContext, sizeValue(occurrence.endpointOffset),
              sizeValue(localEndpoint));
          if (!endpoint)
            return endpoint.takeError();
          result.compatibleEndpoints.push_back(*endpoint);
        }
        auto port = checkedPort(demand.port);
        if (!port)
          return port.takeError();
        const PortDescriptor &descriptor = demand.descriptor;
        result.portDemands.push_back(
            {*implementationIndex, demand.operation, demand.direction, *port,
             descriptor.kind, descriptor.type, descriptor.role,
             descriptor.payloadWidthBits, descriptor.tagWidthBits,
             *endpointDomainOffset, *endpointDomainCount});
      }

      const bool unaryEligible = scratch.hasInjectiveBinding();
      auto portDemandCount =
          checked(portDemandCountContext, selected.activeBoundaryPorts.size());
      if (!portDemandCount)
        return portDemandCount.takeError();
      if (llvm::Error error = preflightFrozenRangeCapacity(
              portDemandOffsetContext, *portDemandOffset, *portDemandCount))
        return std::move(error);
      result.implementationOccurrences.push_back(
          {*frozenRealizationIndex, FabricMemoryOccurrenceRef{occurrence.id},
           *portDemandOffset, *portDemandCount, unaryEligible});
      hasUnaryEligibleOccurrence |= unaryEligible;
    }

    if (!hasUnaryEligibleOccurrence)
      return mappingInfeasibility(
          FrozenMappingInfeasibilityCode::EmptyMemoryUnaryEligibleDomain,
          realization.id,
          "memory realization has no unary-eligible occurrence");
    auto implDomainCount =
        checked(implementationCountContext, candidates.size());
    if (!implDomainCount)
      return implDomainCount.takeError();
    if (llvm::Error error = preflightFrozenRangeCapacity(
            implementationOffsetContext, *implDomainOffset, *implDomainCount))
      return std::move(error);
    result.realizations.push_back({realization.id, selected.encoding,
                                   selected.implementation, selected.service,
                                   *implDomainOffset, *implDomainCount});
  }

  return result;
}
