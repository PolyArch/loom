#ifndef LOOM_LIB_MAPPING_FABRICOCCURRENCEINDEX_H
#define LOOM_LIB_MAPPING_FABRICOCCURRENCEINDEX_H

#include "FabricRoutingIndex.h"
#include "VerifierInternal.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstddef>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace loom::mapping::detail {

struct ValidatedComputeEndpoint {
  ComputeEndpointId id;
  PortDirection direction;
  PortKind kind;
  std::uint32_t payloadCapacityBits;
  std::uint32_t tagCapacityBits;
  std::size_t compatibleTypeOffset;
  std::size_t compatibleTypeCount;
  PortRoleKey role;
  fabric::DataPathKind transportKind;
};

struct ValidatedComputeLocalArc {
  std::size_t fuOccurrence;
  PortDirection direction;
  std::uint32_t port;
  std::size_t endpoint;
  std::uint32_t payloadCapacityBits;
  std::uint32_t tagCapacityBits;
};

struct ValidatedComputePortArcRange {
  std::size_t fuOccurrence;
  PortDirection direction;
  std::uint32_t port;
  std::size_t arcOffset;
  std::size_t arcCount;
};

struct ValidatedPeOccurrence {
  ComputeOccurrenceId id;
  ComputeScheduleKind schedule;
  std::uint64_t instructionContextCapacity;
  std::size_t fuOccurrenceOffset;
  std::size_t fuOccurrenceCount;
  std::size_t endpointOffset;
  std::size_t endpointCount;
  std::size_t localArcOffset;
  std::size_t localArcCount;
  std::size_t portArcRangeOffset;
  std::size_t portArcRangeCount;
};

struct ValidatedFuOccurrence {
  FuId implementation;
  std::size_t parentPe;
};

struct ValidatedFuOccurrenceRange {
  FuId implementation;
  std::size_t occurrenceOffset;
  std::size_t occurrenceCount;
};

struct ValidatedMemoryEndpoint {
  MemoryEndpointId id;
  PortDirection direction;
  PortKind kind;
  std::uint32_t payloadCapacityBits;
  std::uint32_t tagCapacityBits;
  std::size_t compatibleTypeOffset;
  std::size_t compatibleTypeCount;
  PortRoleKey role;
  fabric::DataPathKind transportKind;
};

struct ValidatedMemoryLocalArc {
  std::size_t memoryOccurrence;
  MemoryOperationPortTemplateId operation;
  PortDirection direction;
  std::uint32_t port;
  std::size_t endpoint;
  std::uint32_t payloadCapacityBits;
  std::uint32_t tagCapacityBits;
};

struct ValidatedMemoryPortArcRange {
  std::size_t memoryOccurrence;
  MemoryOperationPortTemplateId operation;
  PortDirection direction;
  std::uint32_t port;
  std::size_t arcOffset;
  std::size_t arcCount;
};

struct ValidatedMemoryOccurrence {
  MemoryOccurrenceId id;
  MemoryImplementationId implementation;
  std::size_t endpointOffset;
  std::size_t endpointCount;
  std::size_t localArcOffset;
  std::size_t localArcCount;
  std::size_t portArcRangeOffset;
  std::size_t portArcRangeCount;
};

struct ValidatedMemoryOccurrenceRange {
  MemoryImplementationId implementation;
  std::size_t occurrenceOffset;
  std::size_t occurrenceCount;
};

struct ValidatedFabricProjection {
  explicit ValidatedFabricProjection(ArtifactIdentity identity)
      : identity(std::move(identity)) {}
  ValidatedFabricProjection(const ValidatedFabricProjection &) = delete;
  ValidatedFabricProjection &
  operator=(const ValidatedFabricProjection &) = delete;
  ValidatedFabricProjection(ValidatedFabricProjection &&) = default;
  ValidatedFabricProjection &operator=(ValidatedFabricProjection &&) = default;

  ArtifactIdentity identity;
  std::vector<ValidatedPeOccurrence> peOccurrences;
  std::vector<ValidatedFuOccurrence> fuOccurrences;
  std::vector<ValidatedComputeEndpoint> computeEndpoints;
  std::vector<TypeKey> computeEndpointCompatibleTypes;
  std::vector<ValidatedComputeLocalArc> computeLocalArcs;
  std::vector<ValidatedComputePortArcRange> computePortArcRanges;
  std::vector<ValidatedFuOccurrenceRange> implementationFuOccurrenceRanges;
  std::vector<std::size_t> implementationFuOccurrences;
  std::vector<ValidatedMemoryOccurrence> memoryOccurrences;
  std::vector<ValidatedMemoryEndpoint> memoryEndpoints;
  std::vector<TypeKey> memoryEndpointCompatibleTypes;
  std::vector<ValidatedMemoryLocalArc> memoryLocalArcs;
  std::vector<ValidatedMemoryPortArcRange> memoryPortArcRanges;
  std::vector<ValidatedMemoryOccurrenceRange>
      implementationMemoryOccurrenceRanges;
  std::vector<std::size_t> implementationMemoryOccurrences;
  ValidatedFabricRoutingProjection routing = {};
};

struct ValidatedActorPairedLaneProjection {
  ActorId actor;
  FabricOpId operation;
  std::vector<std::uint32_t> laneIndices;
  std::string bitmask;
};

struct ValidatedComputeRealizationProjection {
  ComputeRealizationId id;
  FuId fu;
  EncodingId encoding;
  std::vector<ValidatedConfiguredBoundaryPort> activeBoundaryPorts;
  std::vector<ValidatedActorPairedLaneProjection> pairedLaneProjections;
};

struct ValidatedMemoryBoundaryPort {
  MemoryOperationPortTemplateId operation;
  PortDirection direction;
  std::uint32_t port;
  PortDescriptor descriptor;
};

struct ValidatedMemoryRealizationProjection {
  MemoryRealizationId id;
  MemorySemanticEncodingId encoding;
  MemoryImplementationId implementation;
  MemoryServiceDomainId service;
  std::vector<ValidatedMemoryBoundaryPort> activeBoundaryPorts;
};

struct ValidatedTechMappingProjection {
  std::vector<ValidatedComputeRealizationProjection> computeRealizations;
  std::vector<ValidatedMemoryRealizationProjection> memoryRealizations;
};

class ValidatedTechMappingAccess {
public:
  static const ValidatedFabricProjection &
  fabricProjection(const ValidatedTechMapping &mapping) {
    return *mapping.fabricProjection_;
  }

  static const ValidatedTechMappingProjection &
  mappingProjection(const ValidatedTechMapping &mapping) {
    return *mapping.mappingProjection_;
  }
};

llvm::Expected<std::unique_ptr<ValidatedFabricProjection>>
buildValidatedFabricProjection(
    const FabricHardwareView &fabric, EntityKinds &kinds,
    const std::map<std::uint64_t, const FuDescriptor *> &functionalUnits,
    const std::map<std::uint64_t, const MemoryOperationPortTemplateDescriptor *>
        &memoryOperationPortTemplates);

llvm::ArrayRef<std::size_t>
findFuOccurrences(const ValidatedFabricProjection &projection,
                  FuId implementation);

llvm::ArrayRef<std::size_t>
findMemoryOccurrences(const ValidatedFabricProjection &projection,
                      MemoryImplementationId implementation);

llvm::ArrayRef<ValidatedComputeLocalArc>
findComputePortArcs(const ValidatedFabricProjection &projection,
                    std::size_t fuOccurrence, PortDirection direction,
                    std::uint32_t port);

llvm::ArrayRef<ValidatedMemoryLocalArc>
findMemoryPortArcs(const ValidatedFabricProjection &projection,
                   std::size_t memoryOccurrence,
                   MemoryOperationPortTemplateId operation,
                   PortDirection direction, std::uint32_t port);

} // namespace loom::mapping::detail

#endif // LOOM_LIB_MAPPING_FABRICOCCURRENCEINDEX_H
