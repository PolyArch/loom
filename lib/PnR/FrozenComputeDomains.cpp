#include "FrozenComputeDomains.h"

#include "Mapping/FabricOccurrenceIndex.h"

#include "llvm/Support/Error.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
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
constexpr PnrCapacityContext computeCountContext{
    frozenArtifact, "compute_realizations", "compute_realizations",
    PnrCapacityMeasure::Count};
constexpr PnrCapacityContext occurrenceCountContext{
    frozenArtifact, "compute_occurrences", "compute_occurrences",
    PnrCapacityMeasure::Count};
constexpr PnrCapacityContext occurrenceIndexContext{
    frozenArtifact, "compute_occurrences", "compute_occurrences",
    PnrCapacityMeasure::Index};
constexpr PnrCapacityContext occurrenceMembershipCountContext{
    frozenArtifact, "compute_occurrence_fu_memberships", "functional_units",
    PnrCapacityMeasure::Count};
constexpr PnrCapacityContext occurrenceMembershipOffsetContext{
    frozenArtifact, "compute_occurrences", "compute_occurrence_fu_memberships",
    PnrCapacityMeasure::Offset};
constexpr PnrCapacityContext endpointCountContext{
    frozenArtifact, "physical_endpoints", "compute_endpoints",
    PnrCapacityMeasure::Count};
constexpr PnrCapacityContext endpointIndexContext{
    frozenArtifact, "physical_endpoints", "compute_endpoints",
    PnrCapacityMeasure::Index};
constexpr PnrCapacityContext endpointOffsetContext{
    frozenArtifact, "compute_occurrences", "physical_endpoints",
    PnrCapacityMeasure::Offset};
constexpr PnrCapacityContext endpointTypeCountContext{
    frozenArtifact, "physical_endpoint_compatible_types", "types",
    PnrCapacityMeasure::Count};
constexpr PnrCapacityContext endpointTypeOffsetContext{
    frozenArtifact, "physical_endpoints", "physical_endpoint_compatible_types",
    PnrCapacityMeasure::Offset};
constexpr PnrCapacityContext localArcCountContext{
    frozenArtifact, "compute_local_arcs", "compute_local_arcs",
    PnrCapacityMeasure::Count};
constexpr PnrCapacityContext localArcOffsetContext{
    frozenArtifact, "compute_occurrences", "compute_local_arcs",
    PnrCapacityMeasure::Offset};
constexpr PnrCapacityContext implementationCountContext{
    frozenArtifact, "implementation_occurrences", "implementation_occurrences",
    PnrCapacityMeasure::Count};
constexpr PnrCapacityContext implementationIndexContext{
    frozenArtifact, "implementation_occurrences", "implementation_occurrences",
    PnrCapacityMeasure::Index};
constexpr PnrCapacityContext implementationOffsetContext{
    frozenArtifact, "compute_realizations", "implementation_occurrences",
    PnrCapacityMeasure::Offset};
constexpr PnrCapacityContext portDemandCountContext{
    frozenArtifact, "port_demands", "port_demands", PnrCapacityMeasure::Count};
constexpr PnrCapacityContext portDemandOffsetContext{
    frozenArtifact, "implementation_occurrences", "port_demands",
    PnrCapacityMeasure::Offset};
constexpr PnrCapacityContext endpointDomainCountContext{
    frozenArtifact, "compatible_endpoints", "physical_endpoints",
    PnrCapacityMeasure::Count};
constexpr PnrCapacityContext endpointDomainOffsetContext{
    frozenArtifact, "port_demands", "compatible_endpoints",
    PnrCapacityMeasure::Offset};
constexpr PnrCapacityContext realizationIndexContext{
    frozenArtifact, "actor_ownerships", "realizations",
    PnrCapacityMeasure::Index};
constexpr PnrCapacityContext portIndexContext{
    frozenArtifact, "terminals", "ports", PnrCapacityMeasure::Index};

llvm::Error freezeError(std::string message) {
  return llvm::make_error<llvm::StringError>(
      std::move(message), std::make_error_code(std::errc::invalid_argument));
}

llvm::Error mappingInfeasibility(FrozenMappingInfeasibilityCode code,
                                 ComputeRealizationId realization,
                                 std::string message) {
  return llvm::make_error<FrozenMappingInfeasibility>(code, realization,
                                                      std::move(message));
}

std::uint64_t sizeValue(std::size_t size) {
  static_assert(sizeof(std::size_t) <= sizeof(std::uint64_t));
  return static_cast<std::uint64_t>(size);
}

llvm::Error preflight(PnrCapacityContext context, std::size_t size) {
  return preflightPnrIndexCapacity(context, sizeValue(size));
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

struct PortDemandDraft {
  PortDirection direction;
  std::uint32_t port;
  const PortDescriptor *descriptor;
};

bool supportsDemand(const ValidatedComputeEndpoint &endpoint,
                    llvm::ArrayRef<TypeKey> compatibleTypes,
                    const ValidatedComputeLocalArc &arc,
                    const PortDemandDraft &demand) {
  const PortDescriptor &descriptor = *demand.descriptor;
  return endpoint.direction == demand.direction &&
         endpoint.kind == descriptor.kind && endpoint.role == descriptor.role &&
         endpoint.payloadCapacityBits >= descriptor.payloadWidthBits &&
         endpoint.tagCapacityBits >= descriptor.tagWidthBits &&
         arc.payloadCapacityBits >= descriptor.payloadWidthBits &&
         arc.tagCapacityBits >= descriptor.tagWidthBits &&
         std::binary_search(compatibleTypes.begin(), compatibleTypes.end(),
                            descriptor.type, TypeKeyLess{});
}

struct DomainRange {
  std::size_t offset;
  std::size_t count;
};

class EndpointMatchingScratch {
public:
  void reset(std::size_t endpointCount) {
    endpoints_.clear();
    domains_.clear();
    activeEndpointCount_ = endpointCount;
    if (matchedDemand_.size() < endpointCount) {
      matchedDemand_.resize(endpointCount);
      matchedGeneration_.resize(endpointCount);
      visitedGeneration_.resize(endpointCount);
    }
  }

  std::size_t beginDomain() const { return endpoints_.size(); }

  void addEndpoint(std::size_t endpoint) { endpoints_.push_back(endpoint); }

  DomainRange endDomain(std::size_t offset) {
    DomainRange range{offset, endpoints_.size() - offset};
    domains_.push_back(range);
    return range;
  }

  llvm::ArrayRef<std::size_t> endpoints(DomainRange range) const {
    return llvm::ArrayRef<std::size_t>(endpoints_)
        .slice(range.offset, range.count);
  }

  bool allDomainsNonEmpty() const {
    return llvm::all_of(domains_,
                        [](DomainRange range) { return range.count != 0; });
  }

  bool hasInjectiveBinding() {
    if (!allDomainsNonEmpty())
      return false;
    advanceGeneration(matchingGeneration_, matchedGeneration_);
    for (std::size_t demand = 0; demand < domains_.size(); ++demand) {
      advanceGeneration(probeGeneration_, visitedGeneration_);
      if (!augment(demand))
        return false;
    }
    return true;
  }

private:
  static void advanceGeneration(std::uint64_t &generation,
                                std::vector<std::uint64_t> &marks) {
    if (generation == std::numeric_limits<std::uint64_t>::max()) {
      std::fill(marks.begin(), marks.end(), 0);
      generation = 1;
      return;
    }
    ++generation;
  }

  bool augment(std::size_t demand) {
    const DomainRange range = domains_[demand];
    for (std::size_t endpoint : endpoints(range)) {
      assert(endpoint < activeEndpointCount_);
      if (visitedGeneration_[endpoint] == probeGeneration_)
        continue;
      visitedGeneration_[endpoint] = probeGeneration_;
      if (matchedGeneration_[endpoint] != matchingGeneration_ ||
          augment(matchedDemand_[endpoint])) {
        matchedDemand_[endpoint] = demand;
        matchedGeneration_[endpoint] = matchingGeneration_;
        return true;
      }
    }
    return false;
  }

  std::vector<std::size_t> endpoints_;
  std::vector<DomainRange> domains_;
  std::vector<std::size_t> matchedDemand_;
  std::vector<std::uint64_t> matchedGeneration_;
  std::vector<std::uint64_t> visitedGeneration_;
  std::size_t activeEndpointCount_ = 0;
  std::uint64_t matchingGeneration_ = 0;
  std::uint64_t probeGeneration_ = 0;
};

} // namespace

llvm::Expected<FrozenComputeDomains>
loom::pnr::detail::buildFrozenComputeDomains(
    const FabricHardwareView &fabric, const ValidatedTechMapping &mapping,
    llvm::ArrayRef<const ComputeRealizationDraft *> realizations) {
  const ValidatedFabricProjection &projection =
      ValidatedTechMappingAccess::fabricProjection(mapping);
  const ValidatedTechMappingProjection &mappingProjection =
      ValidatedTechMappingAccess::mappingProjection(mapping);
  if (projection.identity != fabric.identity)
    return freezeError("cannot freeze compute domains: validated Fabric "
                       "projection identity does not match the input");
  if (mappingProjection.computeRealizations.size() != realizations.size())
    return freezeError("cannot freeze compute domains: validated Mapping "
                       "projection is incomplete");

  if (llvm::Error error = preflight(computeCountContext, realizations.size()))
    return std::move(error);
  if (llvm::Error error = preflight(occurrenceCountContext,
                                    projection.computeOccurrences.size()))
    return std::move(error);
  if (llvm::Error error =
          preflight(occurrenceMembershipCountContext,
                    projection.computeOccurrenceFuMemberships.size()))
    return std::move(error);
  if (llvm::Error error =
          preflight(endpointCountContext, projection.computeEndpoints.size()))
    return std::move(error);
  if (llvm::Error error =
          preflight(endpointTypeCountContext,
                    projection.computeEndpointCompatibleTypes.size()))
    return std::move(error);
  if (llvm::Error error =
          preflight(localArcCountContext, projection.computeLocalArcs.size()))
    return std::move(error);

  FrozenComputeDomains result;
  result.occurrences.reserve(projection.computeOccurrences.size());
  result.occurrenceFuMemberships = projection.computeOccurrenceFuMemberships;
  result.endpoints.reserve(projection.computeEndpoints.size());
  result.endpointCompatibleTypes = projection.computeEndpointCompatibleTypes;
  result.localArcs.reserve(projection.computeLocalArcs.size());

  for (std::size_t occurrenceIndex = 0;
       occurrenceIndex < projection.computeOccurrences.size();
       ++occurrenceIndex) {
    const ValidatedComputeOccurrence &occurrence =
        projection.computeOccurrences[occurrenceIndex];
    auto frozenOccurrenceIndex =
        checked(occurrenceIndexContext, occurrenceIndex);
    if (!frozenOccurrenceIndex)
      return frozenOccurrenceIndex.takeError();
    auto membershipOffset = checked(occurrenceMembershipOffsetContext,
                                    occurrence.fuMembershipOffset);
    if (!membershipOffset)
      return membershipOffset.takeError();
    auto membershipCount =
        checked(occurrenceMembershipCountContext, occurrence.fuMembershipCount);
    if (!membershipCount)
      return membershipCount.takeError();
    if (llvm::Error error =
            preflightFrozenRangeCapacity(occurrenceMembershipOffsetContext,
                                         *membershipOffset, *membershipCount))
      return std::move(error);
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
    result.occurrences.push_back({occurrence.id, occurrence.schedule,
                                  *membershipOffset, *membershipCount,
                                  *endpointOffset, *endpointCount,
                                  *localArcOffset, *localArcCount});

    llvm::ArrayRef<ValidatedComputeEndpoint> endpoints(
        projection.computeEndpoints);
    endpoints =
        endpoints.slice(occurrence.endpointOffset, occurrence.endpointCount);
    for (const ValidatedComputeEndpoint &endpoint : endpoints) {
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
          {*frozenOccurrenceIndex, endpoint.id, endpoint.direction,
           endpoint.kind, endpoint.payloadCapacityBits,
           endpoint.tagCapacityBits, *typeOffset, *typeCount, endpoint.role});
    }

    llvm::ArrayRef<ValidatedComputeLocalArc> localArcs(
        projection.computeLocalArcs);
    localArcs =
        localArcs.slice(occurrence.localArcOffset, occurrence.localArcCount);
    for (const ValidatedComputeLocalArc &arc : localArcs) {
      auto port = checkedPort(arc.port);
      if (!port)
        return port.takeError();
      auto endpoint = checkedPnrIndexAdd(endpointIndexContext,
                                         sizeValue(occurrence.endpointOffset),
                                         sizeValue(arc.endpoint));
      if (!endpoint)
        return endpoint.takeError();
      result.localArcs.push_back({*frozenOccurrenceIndex, arc.fu, arc.direction,
                                  *port, *endpoint, arc.payloadCapacityBits,
                                  arc.tagCapacityBits});
    }
  }

  result.realizations.reserve(realizations.size());
  EndpointMatchingScratch scratch;
  for (std::size_t realizationIndex = 0; realizationIndex < realizations.size();
       ++realizationIndex) {
    const ComputeRealizationDraft &realization =
        *realizations[realizationIndex];
    auto frozenRealizationIndex =
        checked(realizationIndexContext, realizationIndex);
    if (!frozenRealizationIndex)
      return frozenRealizationIndex.takeError();
    const ValidatedComputeRealizationProjection &selected =
        mappingProjection.computeRealizations[realizationIndex];
    if (selected.id != realization.id || selected.fu != realization.fu.entity ||
        selected.encoding != realization.encoding.entity)
      return freezeError("cannot freeze compute domains: validated Mapping "
                         "projection does not match the realization");

    std::vector<PortDemandDraft> demands;
    demands.reserve(selected.activeBoundaryPorts.size());
    for (const ValidatedConfiguredBoundaryPort &port :
         selected.activeBoundaryPorts)
      demands.push_back({port.direction, port.fuPort, &port.descriptor});

    auto implDomainOffset = checked(implementationOffsetContext,
                                    result.implementationOccurrences.size());
    if (!implDomainOffset)
      return implDomainOffset.takeError();
    const llvm::ArrayRef<std::size_t> candidates =
        findFuOccurrences(projection, realization.fu.entity);
    if (candidates.empty())
      return mappingInfeasibility(
          FrozenMappingInfeasibilityCode::EmptyImplementationDomain,
          realization.id,
          "compute realization has an empty exact implementation domain");

    bool hasUnaryEligibleOccurrence = false;
    for (std::size_t occurrenceIndex : candidates) {
      const ValidatedComputeOccurrence &occurrence =
          projection.computeOccurrences[occurrenceIndex];
      auto implementationIndex = checked(
          implementationIndexContext, result.implementationOccurrences.size());
      if (!implementationIndex)
        return implementationIndex.takeError();
      auto portDemandOffset =
          checked(portDemandOffsetContext, result.portDemands.size());
      if (!portDemandOffset)
        return portDemandOffset.takeError();
      scratch.reset(occurrence.endpointCount);

      for (const PortDemandDraft &demand : demands) {
        auto endpointDomainOffset = checked(endpointDomainOffsetContext,
                                            result.compatibleEndpoints.size());
        if (!endpointDomainOffset)
          return endpointDomainOffset.takeError();
        const std::size_t scratchOffset = scratch.beginDomain();
        for (const ValidatedComputeLocalArc &arc : findComputePortArcs(
                 projection, occurrenceIndex, realization.fu.entity,
                 demand.direction, demand.port)) {
          const ValidatedComputeEndpoint &endpoint =
              projection
                  .computeEndpoints[occurrence.endpointOffset + arc.endpoint];
          llvm::ArrayRef<TypeKey> compatibleTypes(
              projection.computeEndpointCompatibleTypes);
          compatibleTypes = compatibleTypes.slice(endpoint.compatibleTypeOffset,
                                                  endpoint.compatibleTypeCount);
          if (supportsDemand(endpoint, compatibleTypes, arc, demand))
            scratch.addEndpoint(arc.endpoint);
        }
        const DomainRange domain = scratch.endDomain(scratchOffset);
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
        const PortDescriptor &descriptor = *demand.descriptor;
        result.portDemands.push_back(
            {*implementationIndex, realization.fu.entity, demand.direction,
             *port, descriptor.kind, descriptor.type, descriptor.role,
             descriptor.payloadWidthBits, descriptor.tagWidthBits,
             *endpointDomainOffset, *endpointDomainCount});
      }

      bool unaryEligible = scratch.allDomainsNonEmpty();
      if (unaryEligible && occurrence.schedule == ComputeScheduleKind::Spatial)
        unaryEligible = scratch.hasInjectiveBinding();
      auto portDemandCount = checked(portDemandCountContext, demands.size());
      if (!portDemandCount)
        return portDemandCount.takeError();
      if (llvm::Error error = preflightFrozenRangeCapacity(
              portDemandOffsetContext, *portDemandOffset, *portDemandCount))
        return std::move(error);
      auto frozenOccurrenceIndex =
          checked(occurrenceIndexContext, occurrenceIndex);
      if (!frozenOccurrenceIndex)
        return frozenOccurrenceIndex.takeError();
      result.implementationOccurrences.push_back(
          {*frozenRealizationIndex, *frozenOccurrenceIndex, *portDemandOffset,
           *portDemandCount, unaryEligible});
      hasUnaryEligibleOccurrence |= unaryEligible;
    }

    if (!hasUnaryEligibleOccurrence)
      return mappingInfeasibility(
          FrozenMappingInfeasibilityCode::EmptyUnaryEligibleDomain,
          realization.id,
          "compute realization has no unary-eligible occurrence");
    auto implDomainCount =
        checked(implementationCountContext, candidates.size());
    if (!implDomainCount)
      return implDomainCount.takeError();
    if (llvm::Error error = preflightFrozenRangeCapacity(
            implementationOffsetContext, *implDomainOffset, *implDomainCount))
      return std::move(error);
    result.realizations.push_back({realization.id, realization.fu.entity,
                                   realization.encoding.entity,
                                   *implDomainOffset, *implDomainCount});
  }

  return result;
}
