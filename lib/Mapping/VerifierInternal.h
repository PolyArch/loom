#ifndef LOOM_LIB_MAPPING_VERIFIERINTERNAL_H
#define LOOM_LIB_MAPPING_VERIFIERINTERNAL_H

#include "Mapping/Verifier.h"

#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace loom::mapping::detail {

enum class EntityKind {
  Graph,
  Actor,
  Edge,
  LogicalMemoryRoot,
  Fu,
  ComputeOccurrence,
  ComputeEndpoint,
  TransportEndpoint,
  TransportResource,
  FabricOp,
  Encoding,
  ComputeRealization,
  MemoryServiceDomain,
  MemoryImplementation,
  MemoryOperationPortTemplate,
  MemoryInternalConnection,
  MemorySemanticEncoding,
  MemoryRealization,
};

using EntityKinds = std::map<std::uint64_t, EntityKind>;

llvm::Error mappingError(MappingErrorCode code, const llvm::Twine &message);
llvm::Error addEntity(EntityKinds &entities, std::uint64_t id, EntityKind kind);
llvm::Error requireLocalKind(const EntityKinds &entities, std::uint64_t id,
                             EntityKind expected);
llvm::Error validatePairedLaneCapability(const FabricOpDescriptor &operation);
bool validPairedConfiguredPorts(const ConfiguredFabricOpDescriptor &configured,
                                const FabricOpDescriptor &operation);

struct PairedLaneProjection {
  std::vector<std::uint32_t> laneIndices;
  std::string bitmask;
};

struct ValidatedConfiguredBoundaryPort {
  PortDirection direction;
  std::uint32_t fuPort;
  PortDescriptor descriptor;
};

llvm::Expected<PairedLaneProjection>
validateAndProjectPairedLaneSelection(const ArtifactIdentity &fabricIdentity,
                                      const FabricOpDescriptor &operation,
                                      const ActorToFabricOp &correspondence);

std::vector<ValidatedConfiguredBoundaryPort>
deriveActiveConfiguredBoundaryPorts(
    const EncodingDescriptor &encoding,
    const std::map<std::uint64_t, const ConfiguredFabricOpDescriptor *>
        &actorToOp,
    const std::map<std::uint64_t, PairedLaneProjection>
        &actorToLaneProjections);

template <typename Id, typename Descriptor>
llvm::Expected<const Descriptor *>
resolveReference(const EntityReference<Id> &reference,
                 const ArtifactIdentity &artifact, const EntityKinds &kinds,
                 EntityKind expected,
                 const std::map<std::uint64_t, const Descriptor *> &entities) {
  if (reference.artifact != artifact)
    return mappingError(MappingErrorCode::ForeignEntityReference,
                        "reference names a foreign artifact");
  const auto kind = kinds.find(reference.entity.value());
  if (kind == kinds.end())
    return mappingError(MappingErrorCode::UnresolvedEntityId,
                        "reference names an unresolved entity ID");
  if (kind->second != expected)
    return mappingError(MappingErrorCode::WrongEntityKind,
                        "reference names an entity of the wrong kind");
  return entities.at(reference.entity.value());
}

} // namespace loom::mapping::detail

#endif // LOOM_LIB_MAPPING_VERIFIERINTERNAL_H
