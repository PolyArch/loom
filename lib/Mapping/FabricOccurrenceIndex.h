#ifndef LOOM_LIB_MAPPING_FABRICOCCURRENCEINDEX_H
#define LOOM_LIB_MAPPING_FABRICOCCURRENCEINDEX_H

#include "VerifierInternal.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstddef>
#include <map>
#include <memory>
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
};

struct ValidatedComputeLocalArc {
  FuId fu;
  PortDirection direction;
  std::uint32_t port;
  std::size_t endpoint;
  std::uint32_t payloadCapacityBits;
  std::uint32_t tagCapacityBits;
};

struct ValidatedComputePortArcRange {
  FuId fu;
  PortDirection direction;
  std::uint32_t port;
  std::size_t arcOffset;
  std::size_t arcCount;
};

struct ValidatedComputeOccurrence {
  ComputeOccurrenceId id;
  ComputeScheduleKind schedule;
  std::size_t fuMembershipOffset;
  std::size_t fuMembershipCount;
  std::size_t endpointOffset;
  std::size_t endpointCount;
  std::size_t localArcOffset;
  std::size_t localArcCount;
  std::size_t portArcRangeOffset;
  std::size_t portArcRangeCount;
};

struct ValidatedFuOccurrenceRange {
  FuId fu;
  std::size_t occurrenceOffset;
  std::size_t occurrenceCount;
};

struct ValidatedFabricProjection {
  ArtifactIdentity identity;
  std::vector<ValidatedComputeOccurrence> computeOccurrences;
  std::vector<FuId> computeOccurrenceFuMemberships;
  std::vector<ValidatedComputeEndpoint> computeEndpoints;
  std::vector<TypeKey> computeEndpointCompatibleTypes;
  std::vector<ValidatedComputeLocalArc> computeLocalArcs;
  std::vector<ValidatedComputePortArcRange> computePortArcRanges;
  std::vector<ValidatedFuOccurrenceRange> fuOccurrenceRanges;
  std::vector<std::size_t> fuOccurrences;
};

class ValidatedTechMappingAccess {
public:
  static const ValidatedFabricProjection &
  fabricProjection(const ValidatedTechMapping &mapping) {
    return *mapping.fabricProjection_;
  }
};

llvm::Expected<std::shared_ptr<const ValidatedFabricProjection>>
buildValidatedFabricProjection(
    const FabricHardwareView &fabric, EntityKinds &kinds,
    const std::map<std::uint64_t, const FuDescriptor *> &functionalUnits);

llvm::ArrayRef<std::size_t>
findFuOccurrences(const ValidatedFabricProjection &projection, FuId fu);

llvm::ArrayRef<ValidatedComputeLocalArc>
findComputePortArcs(const ValidatedFabricProjection &projection,
                    std::size_t occurrence, FuId fu, PortDirection direction,
                    std::uint32_t port);

} // namespace loom::mapping::detail

#endif // LOOM_LIB_MAPPING_FABRICOCCURRENCEINDEX_H
