#include "FabricOccurrenceIndex.h"

#include <algorithm>
#include <map>
#include <tuple>
#include <utility>
#include <vector>

using namespace loom::mapping;
using namespace loom::mapping::detail;

namespace {

struct TypeKeyLess {
  bool operator()(TypeKey lhs, TypeKey rhs) const {
    return lhs.value() < rhs.value();
  }
};

struct ResolvedArc {
  const ComputeLocalArcDescriptor *descriptor;
  FuId fu;
  ComputeEndpointId endpointId;
  std::size_t endpoint;
};

bool sameArc(const ResolvedArc &lhs, const ResolvedArc &rhs) {
  return lhs.fu == rhs.fu &&
         lhs.descriptor->fuPort.direction == rhs.descriptor->fuPort.direction &&
         lhs.descriptor->fuPort.index == rhs.descriptor->fuPort.index &&
         lhs.endpointId == rhs.endpointId;
}

bool samePort(const ResolvedArc &lhs, const ResolvedArc &rhs) {
  return lhs.fu == rhs.fu &&
         lhs.descriptor->fuPort.direction == rhs.descriptor->fuPort.direction &&
         lhs.descriptor->fuPort.index == rhs.descriptor->fuPort.index;
}

llvm::Error invalidComputeOccurrence(const llvm::Twine &message) {
  return mappingError(MappingErrorCode::InvalidComputeOccurrence, message);
}

} // namespace

llvm::Expected<std::unique_ptr<ValidatedFabricProjection>>
loom::mapping::detail::buildValidatedFabricProjection(
    const FabricHardwareView &fabric, EntityKinds &kinds,
    const std::map<std::uint64_t, const FuDescriptor *> &functionalUnits) {
  std::vector<const ComputeOccurrenceDescriptor *> occurrences;
  occurrences.reserve(fabric.computeOccurrences.size());
  for (const ComputeOccurrenceDescriptor &occurrence :
       fabric.computeOccurrences)
    occurrences.push_back(&occurrence);
  std::sort(occurrences.begin(), occurrences.end(),
            [](const ComputeOccurrenceDescriptor *lhs,
               const ComputeOccurrenceDescriptor *rhs) {
              return lhs->id.value() < rhs->id.value();
            });

  std::map<std::uint64_t, const ComputeEndpointDescriptor *> endpointsById;
  std::map<std::uint64_t, const ComputeOccurrenceDescriptor *> endpointOwners;
  for (const ComputeOccurrenceDescriptor *occurrence : occurrences) {
    if (llvm::Error error = addEntity(kinds, occurrence->id.value(),
                                      EntityKind::ComputeOccurrence))
      return std::move(error);
    std::vector<const ComputeEndpointDescriptor *> endpoints;
    endpoints.reserve(occurrence->endpoints.size());
    for (const ComputeEndpointDescriptor &endpoint : occurrence->endpoints)
      endpoints.push_back(&endpoint);
    std::sort(endpoints.begin(), endpoints.end(),
              [](const ComputeEndpointDescriptor *lhs,
                 const ComputeEndpointDescriptor *rhs) {
                return lhs->id.value() < rhs->id.value();
              });
    for (const ComputeEndpointDescriptor *endpoint : endpoints) {
      if (llvm::Error error = addEntity(kinds, endpoint->id.value(),
                                        EntityKind::ComputeEndpoint))
        return std::move(error);
      endpointsById.emplace(endpoint->id.value(), endpoint);
      endpointOwners.emplace(endpoint->id.value(), occurrence);
    }
  }

  auto projection =
      std::make_unique<ValidatedFabricProjection>(fabric.identity);
  projection->computeOccurrences.reserve(occurrences.size());
  std::vector<std::pair<FuId, std::size_t>> fuOccurrencePairs;

  for (const ComputeOccurrenceDescriptor *occurrence : occurrences) {
    if ((occurrence->schedule != ComputeScheduleKind::Spatial &&
         occurrence->schedule != ComputeScheduleKind::Temporal) ||
        occurrence->functionalUnits.empty())
      return invalidComputeOccurrence(
          "compute occurrence has an invalid schedule or empty FU membership");

    std::vector<FuId> members;
    members.reserve(occurrence->functionalUnits.size());
    for (const FuRef &reference : occurrence->functionalUnits) {
      auto fu = resolveReference(reference, fabric.identity, kinds,
                                 EntityKind::Fu, functionalUnits);
      if (!fu)
        return fu.takeError();
      members.push_back((*fu)->id);
    }
    std::sort(members.begin(), members.end(),
              [](FuId lhs, FuId rhs) { return lhs.value() < rhs.value(); });
    if (std::adjacent_find(members.begin(), members.end()) != members.end())
      return invalidComputeOccurrence(
          "compute occurrence repeats an FU member");
    const std::size_t membershipOffset =
        projection->computeOccurrenceFuMemberships.size();
    projection->computeOccurrenceFuMemberships.insert(
        projection->computeOccurrenceFuMemberships.end(), members.begin(),
        members.end());

    std::vector<const ComputeEndpointDescriptor *> endpoints;
    endpoints.reserve(occurrence->endpoints.size());
    for (const ComputeEndpointDescriptor &endpoint : occurrence->endpoints)
      endpoints.push_back(&endpoint);
    std::sort(endpoints.begin(), endpoints.end(),
              [](const ComputeEndpointDescriptor *lhs,
                 const ComputeEndpointDescriptor *rhs) {
                return lhs->id.value() < rhs->id.value();
              });
    const std::size_t endpointOffset = projection->computeEndpoints.size();
    std::map<std::uint64_t, std::size_t> localEndpointIndices;
    for (std::size_t endpointIndex = 0; endpointIndex < endpoints.size();
         ++endpointIndex) {
      const ComputeEndpointDescriptor &endpoint = *endpoints[endpointIndex];
      if ((endpoint.direction != PortDirection::Input &&
           endpoint.direction != PortDirection::Output) ||
          (endpoint.kind != PortKind::Value &&
           endpoint.kind != PortKind::Stream &&
           endpoint.kind != PortKind::Memory) ||
          endpoint.compatibleTypes.empty())
        return invalidComputeOccurrence(
            "compute occurrence endpoint has an invalid signature");
      std::vector<TypeKey> compatibleTypes = endpoint.compatibleTypes;
      std::sort(compatibleTypes.begin(), compatibleTypes.end(), TypeKeyLess{});
      if (std::adjacent_find(compatibleTypes.begin(), compatibleTypes.end()) !=
          compatibleTypes.end())
        return invalidComputeOccurrence(
            "compute occurrence endpoint repeats a compatible type");
      const std::size_t compatibleTypeOffset =
          projection->computeEndpointCompatibleTypes.size();
      projection->computeEndpointCompatibleTypes.insert(
          projection->computeEndpointCompatibleTypes.end(),
          compatibleTypes.begin(), compatibleTypes.end());
      localEndpointIndices.emplace(endpoint.id.value(), endpointIndex);
      projection->computeEndpoints.push_back(
          {endpoint.id, endpoint.direction, endpoint.kind,
           endpoint.payloadCapacityBits, endpoint.tagCapacityBits,
           compatibleTypeOffset, compatibleTypes.size(), endpoint.role,
           endpoint.transportKind});
    }

    std::vector<ResolvedArc> arcs;
    arcs.reserve(occurrence->localArcs.size());
    for (const ComputeLocalArcDescriptor &arc : occurrence->localArcs) {
      if (arc.fuPort.direction != PortDirection::Input &&
          arc.fuPort.direction != PortDirection::Output)
        return invalidComputeOccurrence(
            "compute local arc has an invalid direction");
      auto fu = resolveReference(arc.fuPort.fu, fabric.identity, kinds,
                                 EntityKind::Fu, functionalUnits);
      if (!fu)
        return fu.takeError();
      auto endpoint =
          resolveReference(arc.endpoint, fabric.identity, kinds,
                           EntityKind::ComputeEndpoint, endpointsById);
      if (!endpoint)
        return endpoint.takeError();
      const auto &ports = arc.fuPort.direction == PortDirection::Input
                              ? (*fu)->inputPorts
                              : (*fu)->outputPorts;
      const auto localEndpoint =
          localEndpointIndices.find((*endpoint)->id.value());
      if (!std::binary_search(
              members.begin(), members.end(), (*fu)->id,
              [](FuId lhs, FuId rhs) { return lhs.value() < rhs.value(); }) ||
          arc.fuPort.index >= ports.size() ||
          endpointOwners.at((*endpoint)->id.value()) != occurrence ||
          (*endpoint)->direction != arc.fuPort.direction ||
          localEndpoint == localEndpointIndices.end())
        return invalidComputeOccurrence(
            "compute local arc does not belong to its occurrence");
      arcs.push_back({&arc, (*fu)->id, (*endpoint)->id, localEndpoint->second});
    }
    std::sort(arcs.begin(), arcs.end(),
              [](const ResolvedArc &lhs, const ResolvedArc &rhs) {
                return std::make_tuple(lhs.fu.value(),
                                       lhs.descriptor->fuPort.direction,
                                       lhs.descriptor->fuPort.index,
                                       lhs.endpointId.value()) <
                       std::make_tuple(rhs.fu.value(),
                                       rhs.descriptor->fuPort.direction,
                                       rhs.descriptor->fuPort.index,
                                       rhs.endpointId.value());
              });
    if (std::adjacent_find(arcs.begin(), arcs.end(), sameArc) != arcs.end())
      return invalidComputeOccurrence("compute occurrence repeats a local arc");

    const std::size_t localArcOffset = projection->computeLocalArcs.size();
    for (const ResolvedArc &arc : arcs) {
      projection->computeLocalArcs.push_back(
          {arc.fu, arc.descriptor->fuPort.direction,
           arc.descriptor->fuPort.index, arc.endpoint,
           arc.descriptor->payloadCapacityBits,
           arc.descriptor->tagCapacityBits});
    }
    const std::size_t portArcRangeOffset =
        projection->computePortArcRanges.size();
    for (std::size_t begin = 0; begin < arcs.size();) {
      std::size_t end = begin + 1;
      while (end < arcs.size() && samePort(arcs[begin], arcs[end]))
        ++end;
      projection->computePortArcRanges.push_back(
          {arcs[begin].fu, arcs[begin].descriptor->fuPort.direction,
           arcs[begin].descriptor->fuPort.index, localArcOffset + begin,
           end - begin});
      begin = end;
    }

    const std::size_t occurrenceIndex = projection->computeOccurrences.size();
    projection->computeOccurrences.push_back(
        {occurrence->id, occurrence->schedule, membershipOffset, members.size(),
         endpointOffset, endpoints.size(), localArcOffset, arcs.size(),
         portArcRangeOffset,
         projection->computePortArcRanges.size() - portArcRangeOffset});
    for (FuId member : members)
      fuOccurrencePairs.emplace_back(member, occurrenceIndex);
  }

  std::sort(fuOccurrencePairs.begin(), fuOccurrencePairs.end(),
            [](const auto &lhs, const auto &rhs) {
              return std::make_tuple(lhs.first.value(), lhs.second) <
                     std::make_tuple(rhs.first.value(), rhs.second);
            });
  for (std::size_t begin = 0; begin < fuOccurrencePairs.size();) {
    std::size_t end = begin + 1;
    while (end < fuOccurrencePairs.size() &&
           fuOccurrencePairs[end].first == fuOccurrencePairs[begin].first)
      ++end;
    const std::size_t occurrenceOffset = projection->fuOccurrences.size();
    for (std::size_t index = begin; index < end; ++index)
      projection->fuOccurrences.push_back(fuOccurrencePairs[index].second);
    projection->fuOccurrenceRanges.push_back(
        {fuOccurrencePairs[begin].first, occurrenceOffset, end - begin});
    begin = end;
  }

  return projection;
}

llvm::ArrayRef<std::size_t> loom::mapping::detail::findFuOccurrences(
    const ValidatedFabricProjection &projection, FuId fu) {
  const auto range = std::lower_bound(
      projection.fuOccurrenceRanges.begin(),
      projection.fuOccurrenceRanges.end(), fu,
      [](const ValidatedFuOccurrenceRange &candidate, FuId expected) {
        return candidate.fu.value() < expected.value();
      });
  if (range == projection.fuOccurrenceRanges.end() || range->fu != fu)
    return {};
  return llvm::ArrayRef<std::size_t>(projection.fuOccurrences)
      .slice(range->occurrenceOffset, range->occurrenceCount);
}

llvm::ArrayRef<ValidatedComputeLocalArc>
loom::mapping::detail::findComputePortArcs(
    const ValidatedFabricProjection &projection, std::size_t occurrence,
    FuId fu, PortDirection direction, std::uint32_t port) {
  if (occurrence >= projection.computeOccurrences.size())
    return {};
  const ValidatedComputeOccurrence &owner =
      projection.computeOccurrences[occurrence];
  llvm::ArrayRef<ValidatedComputePortArcRange> ranges(
      projection.computePortArcRanges);
  ranges = ranges.slice(owner.portArcRangeOffset, owner.portArcRangeCount);
  const auto range = std::lower_bound(
      ranges.begin(), ranges.end(),
      std::make_tuple(fu.value(), direction, port),
      [](const ValidatedComputePortArcRange &candidate, const auto &expected) {
        return std::make_tuple(candidate.fu.value(), candidate.direction,
                               candidate.port) < expected;
      });
  if (range == ranges.end() || range->fu != fu ||
      range->direction != direction || range->port != port)
    return {};
  return llvm::ArrayRef<ValidatedComputeLocalArc>(projection.computeLocalArcs)
      .slice(range->arcOffset, range->arcCount);
}
