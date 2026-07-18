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
  std::size_t fuOccurrence;
  ComputeEndpointId endpointId;
  std::size_t endpoint;
};

struct ResolvedMemoryArc {
  const MemoryLocalArcDescriptor *descriptor;
  MemoryOperationPortTemplateId operation;
  PortDirection direction;
  MemoryEndpointId endpointId;
  std::size_t endpoint;
};

bool sameArc(const ResolvedArc &lhs, const ResolvedArc &rhs) {
  return lhs.fuOccurrence == rhs.fuOccurrence &&
         lhs.descriptor->fuPort.direction == rhs.descriptor->fuPort.direction &&
         lhs.descriptor->fuPort.index == rhs.descriptor->fuPort.index &&
         lhs.endpointId == rhs.endpointId;
}

bool samePort(const ResolvedArc &lhs, const ResolvedArc &rhs) {
  return lhs.fuOccurrence == rhs.fuOccurrence &&
         lhs.descriptor->fuPort.direction == rhs.descriptor->fuPort.direction &&
         lhs.descriptor->fuPort.index == rhs.descriptor->fuPort.index;
}

bool sameMemoryArc(const ResolvedMemoryArc &lhs, const ResolvedMemoryArc &rhs) {
  return lhs.operation == rhs.operation && lhs.direction == rhs.direction &&
         lhs.descriptor->operationPort.index ==
             rhs.descriptor->operationPort.index &&
         lhs.endpointId == rhs.endpointId;
}

bool sameMemoryPort(const ResolvedMemoryArc &lhs,
                    const ResolvedMemoryArc &rhs) {
  return lhs.operation == rhs.operation && lhs.direction == rhs.direction &&
         lhs.descriptor->operationPort.index ==
             rhs.descriptor->operationPort.index;
}

llvm::Error invalidComputeOccurrence(const llvm::Twine &message) {
  return mappingError(MappingErrorCode::InvalidComputeOccurrence, message);
}

llvm::Error invalidMemoryOccurrence(const llvm::Twine &message) {
  return mappingError(MappingErrorCode::InvalidMemoryOccurrence, message);
}

llvm::Error malformedFuParentLinkage(const llvm::Twine &message) {
  return mappingError(MappingErrorCode::MalformedFuParentLinkage, message);
}

llvm::Expected<const FuDescriptor *> resolveFuImplementation(
    const FuRef &reference, const ArtifactIdentity &fabric,
    const EntityKinds &kinds,
    const std::map<std::uint64_t, const FuDescriptor *> &functionalUnits) {
  if (reference.artifact != fabric)
    return mappingError(MappingErrorCode::ForeignEntityReference,
                        "FU occurrence names a foreign implementation");
  const auto kind = kinds.find(reference.entity.value());
  if (kind == kinds.end())
    return mappingError(MappingErrorCode::MissingFuImplementation,
                        "FU occurrence names a missing implementation");
  if (kind->second != EntityKind::Fu)
    return mappingError(MappingErrorCode::WrongEntityKind,
                        "FU occurrence names an entity of the wrong kind");
  return functionalUnits.at(reference.entity.value());
}

llvm::Expected<MemoryImplementationId>
resolveMemoryImplementation(const MemoryImplementationRef &reference,
                            const ArtifactIdentity &fabric,
                            const EntityKinds &kinds) {
  if (reference.artifact != fabric)
    return mappingError(MappingErrorCode::ForeignEntityReference,
                        "memory occurrence names a foreign implementation");
  const auto kind = kinds.find(reference.entity.value());
  if (kind == kinds.end())
    return mappingError(MappingErrorCode::UnresolvedEntityId,
                        "memory occurrence names a missing implementation");
  if (kind->second != EntityKind::MemoryImplementation)
    return mappingError(MappingErrorCode::WrongEntityKind,
                        "memory occurrence names an entity of the wrong kind");
  return reference.entity;
}

} // namespace

llvm::Expected<std::unique_ptr<ValidatedFabricProjection>>
loom::mapping::detail::buildValidatedFabricProjection(
    const FabricHardwareView &fabric, EntityKinds &kinds,
    const std::map<std::uint64_t, const FuDescriptor *> &functionalUnits,
    const std::map<std::uint64_t, const MemoryOperationPortTemplateDescriptor *>
        &memoryOperationPortTemplates) {
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
  projection->peOccurrences.reserve(occurrences.size());
  std::vector<std::pair<FuId, std::size_t>> fuOccurrencePairs;

  for (const ComputeOccurrenceDescriptor *occurrence : occurrences) {
    if ((occurrence->schedule != ComputeScheduleKind::Spatial &&
         occurrence->schedule != ComputeScheduleKind::Temporal) ||
        occurrence->functionalUnits.empty())
      return invalidComputeOccurrence(
          "compute occurrence has an invalid schedule or empty FU membership");
    if (occurrence->instructionContextCapacity <= 0 ||
        (occurrence->schedule == ComputeScheduleKind::Spatial &&
         occurrence->instructionContextCapacity != 1))
      return mappingError(
          MappingErrorCode::InvalidInstructionContextCapacity,
          "PE occurrence has an invalid instruction context capacity");

    std::vector<FuId> members;
    members.reserve(occurrence->functionalUnits.size());
    for (const FuRef &reference : occurrence->functionalUnits) {
      auto fu = resolveFuImplementation(reference, fabric.identity, kinds,
                                        functionalUnits);
      if (!fu)
        return fu.takeError();
      members.push_back((*fu)->id);
    }
    std::sort(members.begin(), members.end(),
              [](FuId lhs, FuId rhs) { return lhs.value() < rhs.value(); });
    if (std::adjacent_find(members.begin(), members.end()) != members.end())
      return malformedFuParentLinkage(
          "PE occurrence repeats one FU implementation");

    const std::size_t peOccurrence = projection->peOccurrences.size();
    const std::size_t fuOccurrenceOffset = projection->fuOccurrences.size();
    std::map<std::uint64_t, std::size_t> localFuOccurrences;
    for (FuId member : members) {
      const std::size_t fuOccurrence = projection->fuOccurrences.size();
      projection->fuOccurrences.push_back({member, peOccurrence});
      localFuOccurrences.emplace(member.value(), fuOccurrence);
      fuOccurrencePairs.emplace_back(member, fuOccurrence);
    }

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
      auto fu = resolveFuImplementation(arc.fuPort.fu, fabric.identity, kinds,
                                        functionalUnits);
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
      const auto localFu = localFuOccurrences.find((*fu)->id.value());
      const auto localEndpoint =
          localEndpointIndices.find((*endpoint)->id.value());
      if (localFu == localFuOccurrences.end() ||
          endpointOwners.at((*endpoint)->id.value()) != occurrence)
        return malformedFuParentLinkage(
            "compute local arc crosses its FU occurrence parent PE");
      if (arc.fuPort.index >= ports.size() ||
          (*endpoint)->direction != arc.fuPort.direction ||
          localEndpoint == localEndpointIndices.end())
        return invalidComputeOccurrence(
            "compute local arc does not belong to its occurrence");
      arcs.push_back(
          {&arc, localFu->second, (*endpoint)->id, localEndpoint->second});
    }
    std::sort(arcs.begin(), arcs.end(),
              [](const ResolvedArc &lhs, const ResolvedArc &rhs) {
                return std::make_tuple(lhs.fuOccurrence,
                                       lhs.descriptor->fuPort.direction,
                                       lhs.descriptor->fuPort.index,
                                       lhs.endpointId.value()) <
                       std::make_tuple(rhs.fuOccurrence,
                                       rhs.descriptor->fuPort.direction,
                                       rhs.descriptor->fuPort.index,
                                       rhs.endpointId.value());
              });
    if (std::adjacent_find(arcs.begin(), arcs.end(), sameArc) != arcs.end())
      return invalidComputeOccurrence("compute occurrence repeats a local arc");

    const std::size_t localArcOffset = projection->computeLocalArcs.size();
    for (const ResolvedArc &arc : arcs) {
      projection->computeLocalArcs.push_back(
          {arc.fuOccurrence, arc.descriptor->fuPort.direction,
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
          {arcs[begin].fuOccurrence, arcs[begin].descriptor->fuPort.direction,
           arcs[begin].descriptor->fuPort.index, localArcOffset + begin,
           end - begin});
      begin = end;
    }

    projection->peOccurrences.push_back(
        {occurrence->id, occurrence->schedule,
         static_cast<std::uint64_t>(occurrence->instructionContextCapacity),
         fuOccurrenceOffset, members.size(), endpointOffset, endpoints.size(),
         localArcOffset, arcs.size(), portArcRangeOffset,
         projection->computePortArcRanges.size() - portArcRangeOffset});
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
    const std::size_t occurrenceOffset =
        projection->implementationFuOccurrences.size();
    for (std::size_t index = begin; index < end; ++index)
      projection->implementationFuOccurrences.push_back(
          fuOccurrencePairs[index].second);
    projection->implementationFuOccurrenceRanges.push_back(
        {fuOccurrencePairs[begin].first, occurrenceOffset, end - begin});
    begin = end;
  }

  std::vector<const MemoryOccurrenceDescriptor *> memoryOccurrences;
  memoryOccurrences.reserve(fabric.memoryOccurrences.size());
  for (const MemoryOccurrenceDescriptor &occurrence : fabric.memoryOccurrences)
    memoryOccurrences.push_back(&occurrence);
  std::sort(memoryOccurrences.begin(), memoryOccurrences.end(),
            [](const MemoryOccurrenceDescriptor *lhs,
               const MemoryOccurrenceDescriptor *rhs) {
              return lhs->id.value() < rhs->id.value();
            });

  std::map<std::uint64_t, const MemoryEndpointDescriptor *> memoryEndpointsById;
  std::map<std::uint64_t, const MemoryOccurrenceDescriptor *>
      memoryEndpointOwners;
  for (const MemoryOccurrenceDescriptor *occurrence : memoryOccurrences) {
    if (llvm::Error error = addEntity(kinds, occurrence->id.value(),
                                      EntityKind::MemoryOccurrence))
      return std::move(error);
    std::vector<const MemoryEndpointDescriptor *> endpoints;
    endpoints.reserve(occurrence->endpoints.size());
    for (const MemoryEndpointDescriptor &endpoint : occurrence->endpoints)
      endpoints.push_back(&endpoint);
    std::sort(endpoints.begin(), endpoints.end(),
              [](const MemoryEndpointDescriptor *lhs,
                 const MemoryEndpointDescriptor *rhs) {
                return lhs->id.value() < rhs->id.value();
              });
    for (const MemoryEndpointDescriptor *endpoint : endpoints) {
      if (llvm::Error error = addEntity(kinds, endpoint->id.value(),
                                        EntityKind::MemoryEndpoint))
        return std::move(error);
      memoryEndpointsById.emplace(endpoint->id.value(), endpoint);
      memoryEndpointOwners.emplace(endpoint->id.value(), occurrence);
    }
  }

  projection->memoryOccurrences.reserve(memoryOccurrences.size());
  std::vector<std::pair<MemoryImplementationId, std::size_t>>
      memoryOccurrencePairs;
  for (const MemoryOccurrenceDescriptor *occurrence : memoryOccurrences) {
    auto implementation = resolveMemoryImplementation(
        occurrence->implementation, fabric.identity, kinds);
    if (!implementation)
      return implementation.takeError();
    const std::size_t memoryOccurrence = projection->memoryOccurrences.size();
    const std::size_t endpointOffset = projection->memoryEndpoints.size();
    std::vector<const MemoryEndpointDescriptor *> endpoints;
    endpoints.reserve(occurrence->endpoints.size());
    for (const MemoryEndpointDescriptor &endpoint : occurrence->endpoints)
      endpoints.push_back(&endpoint);
    std::sort(endpoints.begin(), endpoints.end(),
              [](const MemoryEndpointDescriptor *lhs,
                 const MemoryEndpointDescriptor *rhs) {
                return lhs->id.value() < rhs->id.value();
              });
    std::map<std::uint64_t, std::size_t> localEndpointIndices;
    for (std::size_t endpointIndex = 0; endpointIndex < endpoints.size();
         ++endpointIndex) {
      const MemoryEndpointDescriptor &endpoint = *endpoints[endpointIndex];
      if ((endpoint.direction != PortDirection::Input &&
           endpoint.direction != PortDirection::Output) ||
          (endpoint.kind != PortKind::Value &&
           endpoint.kind != PortKind::Stream) ||
          endpoint.compatibleTypes.empty())
        return invalidMemoryOccurrence(
            "memory occurrence endpoint has an invalid signature");
      std::vector<TypeKey> compatibleTypes = endpoint.compatibleTypes;
      std::sort(compatibleTypes.begin(), compatibleTypes.end(), TypeKeyLess{});
      if (std::adjacent_find(compatibleTypes.begin(), compatibleTypes.end()) !=
          compatibleTypes.end())
        return invalidMemoryOccurrence(
            "memory occurrence endpoint repeats a compatible type");
      const std::size_t compatibleTypeOffset =
          projection->memoryEndpointCompatibleTypes.size();
      projection->memoryEndpointCompatibleTypes.insert(
          projection->memoryEndpointCompatibleTypes.end(),
          compatibleTypes.begin(), compatibleTypes.end());
      localEndpointIndices.emplace(endpoint.id.value(), endpointIndex);
      projection->memoryEndpoints.push_back(
          {endpoint.id, endpoint.direction, endpoint.kind,
           endpoint.payloadCapacityBits, endpoint.tagCapacityBits,
           compatibleTypeOffset, compatibleTypes.size(), endpoint.role,
           endpoint.transportKind});
    }

    std::vector<ResolvedMemoryArc> arcs;
    arcs.reserve(occurrence->localArcs.size());
    for (const MemoryLocalArcDescriptor &arc : occurrence->localArcs) {
      auto operation =
          resolveReference(arc.operationPort.operation, fabric.identity, kinds,
                           EntityKind::MemoryOperationPortTemplate,
                           memoryOperationPortTemplates);
      if (!operation)
        return operation.takeError();
      auto endpoint =
          resolveReference(arc.endpoint, fabric.identity, kinds,
                           EntityKind::MemoryEndpoint, memoryEndpointsById);
      if (!endpoint)
        return endpoint.takeError();
      const auto localEndpoint =
          localEndpointIndices.find((*endpoint)->id.value());
      if ((*operation)->implementation != *implementation ||
          arc.operationPort.index >= (*operation)->ports.size() ||
          memoryEndpointOwners.at((*endpoint)->id.value()) != occurrence ||
          localEndpoint == localEndpointIndices.end())
        return invalidMemoryOccurrence(
            "memory local arc does not belong to its occurrence");
      const MemoryOperationPortDescriptor &port =
          (*operation)->ports[arc.operationPort.index];
      if ((*endpoint)->direction != port.direction)
        return invalidMemoryOccurrence(
            "memory local arc reverses operation port direction");
      arcs.push_back({&arc, (*operation)->id, port.direction, (*endpoint)->id,
                      localEndpoint->second});
    }
    std::sort(arcs.begin(), arcs.end(),
              [](const ResolvedMemoryArc &lhs, const ResolvedMemoryArc &rhs) {
                return std::make_tuple(lhs.operation.value(), lhs.direction,
                                       lhs.descriptor->operationPort.index,
                                       lhs.endpointId.value()) <
                       std::make_tuple(rhs.operation.value(), rhs.direction,
                                       rhs.descriptor->operationPort.index,
                                       rhs.endpointId.value());
              });
    if (std::adjacent_find(arcs.begin(), arcs.end(), sameMemoryArc) !=
        arcs.end())
      return invalidMemoryOccurrence("memory occurrence repeats a local arc");

    const std::size_t localArcOffset = projection->memoryLocalArcs.size();
    for (const ResolvedMemoryArc &arc : arcs) {
      projection->memoryLocalArcs.push_back(
          {memoryOccurrence, arc.operation, arc.direction,
           arc.descriptor->operationPort.index, arc.endpoint,
           arc.descriptor->payloadCapacityBits,
           arc.descriptor->tagCapacityBits});
    }
    const std::size_t portArcRangeOffset =
        projection->memoryPortArcRanges.size();
    for (std::size_t begin = 0; begin < arcs.size();) {
      std::size_t end = begin + 1;
      while (end < arcs.size() && sameMemoryPort(arcs[begin], arcs[end]))
        ++end;
      projection->memoryPortArcRanges.push_back(
          {memoryOccurrence, arcs[begin].operation, arcs[begin].direction,
           arcs[begin].descriptor->operationPort.index, localArcOffset + begin,
           end - begin});
      begin = end;
    }
    projection->memoryOccurrences.push_back(
        {occurrence->id, *implementation, endpointOffset, endpoints.size(),
         localArcOffset, arcs.size(), portArcRangeOffset,
         projection->memoryPortArcRanges.size() - portArcRangeOffset});
    memoryOccurrencePairs.emplace_back(*implementation, memoryOccurrence);
  }

  std::sort(memoryOccurrencePairs.begin(), memoryOccurrencePairs.end(),
            [](const auto &lhs, const auto &rhs) {
              return std::make_tuple(lhs.first.value(), lhs.second) <
                     std::make_tuple(rhs.first.value(), rhs.second);
            });
  for (std::size_t begin = 0; begin < memoryOccurrencePairs.size();) {
    std::size_t end = begin + 1;
    while (end < memoryOccurrencePairs.size() &&
           memoryOccurrencePairs[end].first ==
               memoryOccurrencePairs[begin].first)
      ++end;
    const std::size_t occurrenceOffset =
        projection->implementationMemoryOccurrences.size();
    for (std::size_t index = begin; index < end; ++index)
      projection->implementationMemoryOccurrences.push_back(
          memoryOccurrencePairs[index].second);
    projection->implementationMemoryOccurrenceRanges.push_back(
        {memoryOccurrencePairs[begin].first, occurrenceOffset, end - begin});
    begin = end;
  }

  return projection;
}

llvm::ArrayRef<std::size_t> loom::mapping::detail::findFuOccurrences(
    const ValidatedFabricProjection &projection, FuId implementation) {
  const auto range = std::lower_bound(
      projection.implementationFuOccurrenceRanges.begin(),
      projection.implementationFuOccurrenceRanges.end(), implementation,
      [](const ValidatedFuOccurrenceRange &candidate, FuId expected) {
        return candidate.implementation.value() < expected.value();
      });
  if (range == projection.implementationFuOccurrenceRanges.end() ||
      range->implementation != implementation)
    return {};
  return llvm::ArrayRef<std::size_t>(projection.implementationFuOccurrences)
      .slice(range->occurrenceOffset, range->occurrenceCount);
}

llvm::ArrayRef<std::size_t> loom::mapping::detail::findMemoryOccurrences(
    const ValidatedFabricProjection &projection,
    MemoryImplementationId implementation) {
  const auto range = std::lower_bound(
      projection.implementationMemoryOccurrenceRanges.begin(),
      projection.implementationMemoryOccurrenceRanges.end(), implementation,
      [](const ValidatedMemoryOccurrenceRange &candidate,
         MemoryImplementationId expected) {
        return candidate.implementation.value() < expected.value();
      });
  if (range == projection.implementationMemoryOccurrenceRanges.end() ||
      range->implementation != implementation)
    return {};
  return llvm::ArrayRef<std::size_t>(projection.implementationMemoryOccurrences)
      .slice(range->occurrenceOffset, range->occurrenceCount);
}

llvm::ArrayRef<ValidatedComputeLocalArc>
loom::mapping::detail::findComputePortArcs(
    const ValidatedFabricProjection &projection, std::size_t fuOccurrence,
    PortDirection direction, std::uint32_t port) {
  if (fuOccurrence >= projection.fuOccurrences.size())
    return {};
  const ValidatedFuOccurrence &fu = projection.fuOccurrences[fuOccurrence];
  const ValidatedPeOccurrence &owner = projection.peOccurrences[fu.parentPe];
  llvm::ArrayRef<ValidatedComputePortArcRange> ranges(
      projection.computePortArcRanges);
  ranges = ranges.slice(owner.portArcRangeOffset, owner.portArcRangeCount);
  const auto range = std::lower_bound(
      ranges.begin(), ranges.end(),
      std::make_tuple(fuOccurrence, direction, port),
      [](const ValidatedComputePortArcRange &candidate, const auto &expected) {
        return std::make_tuple(candidate.fuOccurrence, candidate.direction,
                               candidate.port) < expected;
      });
  if (range == ranges.end() || range->fuOccurrence != fuOccurrence ||
      range->direction != direction || range->port != port)
    return {};
  return llvm::ArrayRef<ValidatedComputeLocalArc>(projection.computeLocalArcs)
      .slice(range->arcOffset, range->arcCount);
}

llvm::ArrayRef<ValidatedMemoryLocalArc>
loom::mapping::detail::findMemoryPortArcs(
    const ValidatedFabricProjection &projection, std::size_t memoryOccurrence,
    MemoryOperationPortTemplateId operation, PortDirection direction,
    std::uint32_t port) {
  if (memoryOccurrence >= projection.memoryOccurrences.size())
    return {};
  const ValidatedMemoryOccurrence &occurrence =
      projection.memoryOccurrences[memoryOccurrence];
  llvm::ArrayRef<ValidatedMemoryPortArcRange> ranges(
      projection.memoryPortArcRanges);
  ranges =
      ranges.slice(occurrence.portArcRangeOffset, occurrence.portArcRangeCount);
  const auto range = std::lower_bound(
      ranges.begin(), ranges.end(),
      std::make_tuple(memoryOccurrence, operation.value(), direction, port),
      [](const ValidatedMemoryPortArcRange &candidate, const auto &expected) {
        return std::make_tuple(candidate.memoryOccurrence,
                               candidate.operation.value(), candidate.direction,
                               candidate.port) < expected;
      });
  if (range == ranges.end() || range->memoryOccurrence != memoryOccurrence ||
      range->operation != operation || range->direction != direction ||
      range->port != port)
    return {};
  return llvm::ArrayRef<ValidatedMemoryLocalArc>(projection.memoryLocalArcs)
      .slice(range->arcOffset, range->arcCount);
}
