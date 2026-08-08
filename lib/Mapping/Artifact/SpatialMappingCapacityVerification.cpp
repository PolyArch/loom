#include "SpatialMappingCapacityVerification.h"

#include "Dataflow/IR/DataflowReferenceCodec.h"
#include "Fabric/IR/UsePatternValue.h"
#include "ResourceCapacityVerification.h"

#include <cstdint>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace loom::mapping::detail {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "spatial_capacity_verification_invalid: " +
                                     message);
}

void appendU32(std::string &bytes, std::uint32_t value) {
  for (unsigned shift = 24;; shift -= 8) {
    bytes.push_back(static_cast<char>(value >> shift));
    if (shift == 0)
      break;
  }
}

void appendU64(std::string &bytes, std::uint64_t value) {
  for (unsigned shift = 56;; shift -= 8) {
    bytes.push_back(static_cast<char>(value >> shift));
    if (shift == 0)
      break;
  }
}

llvm::Expected<std::string>
resourceOwnerKey(const ArtifactIdentity &dataflowIdentity,
                 const SpatialResourceOwnerRef &owner) {
  std::string result;
  return std::visit(
      [&](const auto &typed) -> llvm::Expected<std::string> {
        using Owner = std::decay_t<decltype(typed)>;
        if constexpr (std::is_same_v<Owner, SpatialComputeResourceOwnerRef>) {
          appendU32(result, 0);
          appendU64(result, typed.realization);
        } else if constexpr (std::is_same_v<
                                 Owner, SpatialMemoryEngineResourceOwnerRef>) {
          appendU32(result, 1);
          appendU64(result, typed.realization);
        } else if constexpr (std::is_same_v<
                                 Owner, SpatialMemoryBindingResourceOwnerRef>) {
          appendU32(result, 2);
          appendU64(result, typed.binding);
        } else {
          appendU32(result, 3);
          auto logicalNet = ::dataflow::encodeDataflowReference(
              dataflowIdentity, typed.logicalNet);
          if (!logicalNet)
            return logicalNet.takeError();
          appendU64(result, logicalNet->size());
          result.append(reinterpret_cast<const char *>(logicalNet->data()),
                        logicalNet->size());
          appendU64(result, typed.nodeOrdinal);
        }
        return result;
      },
      owner);
}

} // namespace

llvm::Expected<std::string> deriveSpatialCapacityActivationKey(
    const ::loom::fabric::FabricArtifactView &fabric,
    const ArtifactIdentity &dataflowIdentity,
    const SpatialResourceUseView &use) {
  auto result = resourceOwnerKey(dataflowIdentity, use.owner);
  if (!result)
    return result.takeError();
  auto event = encodeSpatialActivityEventKey(dataflowIdentity,
                                             use.activation.trigger.event);
  if (!event)
    return event.takeError();
  appendU64(*result, event->size());
  result->append(reinterpret_cast<const char *>(event->data()), event->size());

  const auto owner = use.useSite.owner.catalog();
  const ::fabric::ResourceContract *contract = fabric.resourceContract(owner);
  if (!contract || use.useSite.ordinal >= contract->usePatternCount())
    return invalid("ResourceUse does not resolve a Fabric pattern");
  const ::fabric::UsePattern pattern =
      contract->usePattern(::fabric::UsePatternKey(use.useSite.ordinal));
  if (pattern.parameters.size() != use.parameters.size())
    return invalid("ResourceUse parameter count disagrees with its pattern");
  appendU64(*result, use.parameters.size());
  for (std::size_t ordinal = 0; ordinal < use.parameters.size(); ++ordinal) {
    auto encoded = ::fabric::encodeUsePatternValue(pattern.parameters[ordinal],
                                                   use.parameters[ordinal]);
    if (!encoded)
      return encoded.takeError();
    appendU64(*result, encoded->size());
    result->append(reinterpret_cast<const char *>(encoded->data()),
                   encoded->size());
  }
  return result;
}

llvm::Expected<SpatialCapacityOveruseProjection> deriveSpatialCapacityOveruse(
    const ::loom::fabric::FabricArtifactView &fabric,
    const ArtifactIdentity &dataflowIdentity,
    llvm::ArrayRef<SpatialResourceUseView> resourceUses,
    llvm::ArrayRef<std::vector<::loom::fabric::FabricPhysicalTraversalRef>>
        routeTraversals) {
  std::vector<ResourceCapacityUseProjection> uses;
  uses.reserve(resourceUses.size());
  for (const auto &use : resourceUses) {
    auto key =
        deriveSpatialCapacityActivationKey(fabric, dataflowIdentity, use);
    if (!key)
      return key.takeError();
    uses.push_back(
        ResourceCapacityUseProjection{0, use.useSite, std::move(*key)});
  }

  std::vector<ResourceCapacityRouteProjection> routes;
  routes.reserve(routeTraversals.size());
  for (const auto &route : routeTraversals)
    routes.push_back(ResourceCapacityRouteProjection{0, route});

  const ResourceCapacityNamespaceView capacityNamespace{
      &fabric, std::vector<std::uint8_t>(fabric.identity().bytes().begin(),
                                         fabric.identity().bytes().end())};
  auto projected = deriveResourceCapacityOveruse(
      llvm::ArrayRef<ResourceCapacityNamespaceView>(capacityNamespace), uses,
      routes);
  if (!projected)
    return projected.takeError();

  SpatialCapacityOveruseProjection result;
  result.total = projected->total;
  if (projected->firstWitness) {
    const auto &witness = *projected->firstWitness;
    if (witness.namespaceOrdinal != 0)
      return invalid("standalone capacity witness has a foreign namespace");
    result.firstWitness = SpatialCapacityOveruseWitness{
        witness.owner, witness.state,    witness.dimension,
        witness.usage, witness.capacity, witness.canonicalOccupancyKey};
  }
  return result;
}

} // namespace loom::mapping::detail
