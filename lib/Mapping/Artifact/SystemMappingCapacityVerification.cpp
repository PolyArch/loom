#include "SystemMappingCapacityVerification.h"

#include "Common/ArtifactLocalReference.h"
#include "Dataflow/IR/DataflowReferenceCodec.h"
#include "Fabric/Identity/FabricRefBytes.h"
#include "Fabric/Identity/FabricRefText.h"
#include "Mapping/Artifact/MappingArtifact.h"
#include "ResourceCapacityVerification.h"
#include "SpatialMappingCapacityVerification.h"
#include "SystemMappingExecutionProjection.h"

#include "llvm/ADT/STLExtras.h"

#include <map>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace loom::mapping::detail {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "system_mapping_capacity_invalid: " + message);
}

void appendU64(std::string &bytes, std::uint64_t value) {
  for (unsigned shift = 56;; shift -= 8) {
    bytes.push_back(static_cast<char>(value >> shift));
    if (shift == 0)
      break;
  }
}

void appendSized(std::string &bytes, llvm::ArrayRef<std::uint8_t> value) {
  appendU64(bytes, value.size());
  bytes.append(reinterpret_cast<const char *>(value.data()), value.size());
}

std::string byteKey(llvm::ArrayRef<std::uint8_t> bytes) {
  return std::string(reinterpret_cast<const char *>(bytes.data()),
                     bytes.size());
}

llvm::Error
validateDirectOwner(const ::loom::fabric::FabricInventoryOwnerRef &owner) {
  auto physical =
      ::loom::fabric::FabricPhysicalOccurrenceOwnerRef::create(owner);
  if (!physical)
    return invalid("System capacity owner is not a direct physical owner: " +
                   llvm::toString(physical.takeError()));
  return llvm::Error::success();
}

llvm::Expected<::loom::fabric::FabricModulePhysicalTargetRef>
projectModuleOwner(const ::loom::fabric::FabricInventoryOwnerRef &owner) {
  using namespace ::loom::fabric;
  llvm::Expected<FabricModulePhysicalOwnerRef> physical =
      [&]() -> llvm::Expected<FabricModulePhysicalOwnerRef> {
    switch (owner.kind()) {
    case FabricInventoryOwnerKind::PeOccurrence:
      return FabricModulePhysicalOwnerRef::create(
          std::get<FabricPeOccurrenceRef>(owner.payload));
    case FabricInventoryOwnerKind::FuOccurrence:
      return FabricModulePhysicalOwnerRef::create(
          std::get<FabricFuOccurrenceRef>(owner.payload));
    case FabricInventoryOwnerKind::FuOccurrenceNode:
      return FabricModulePhysicalOwnerRef::create(
          std::get<FabricFuOccurrenceNodeRef>(owner.payload));
    case FabricInventoryOwnerKind::MemoryOccurrence:
      return FabricModulePhysicalOwnerRef::create(
          std::get<FabricMemoryOccurrenceRef>(owner.payload));
    case FabricInventoryOwnerKind::MemoryOperationPort:
      return FabricModulePhysicalOwnerRef::create(
          std::get<FabricMemoryOperationPortRef>(owner.payload));
    case FabricInventoryOwnerKind::MemoryService:
      return FabricModulePhysicalOwnerRef::create(LocalMemoryServiceRef(
          std::get<FabricMemoryServiceRef>(owner.payload)));
    case FabricInventoryOwnerKind::SwitchOccurrence:
      return FabricModulePhysicalOwnerRef::create(
          std::get<FabricSwitchOccurrenceRef>(owner.payload));
    case FabricInventoryOwnerKind::FifoOccurrence:
      return FabricModulePhysicalOwnerRef::create(
          std::get<FabricFifoOccurrenceRef>(owner.payload));
    case FabricInventoryOwnerKind::BoundaryOccurrence:
      return FabricModulePhysicalOwnerRef::create(
          std::get<FabricBoundaryOccurrenceRef>(owner.payload));
    case FabricInventoryOwnerKind::InstructionContext:
      return FabricModulePhysicalOwnerRef::create(
          std::get<InstructionContextRef>(owner.payload));
    default:
      return invalid("capacity owner is not declared inside a Module");
    }
  }();
  if (!physical)
    return physical.takeError();
  return FabricModulePhysicalTargetRef::create(*physical);
}

llvm::Error
validateInternalOwner(::loom::fabric::SpatialCoreOccurrenceRef spatialCore,
                      const ::loom::fabric::FabricInventoryOwnerRef &owner) {
  auto target = projectModuleOwner(owner);
  if (!target)
    return invalid("imported Spatial capacity owner is not Module-local: " +
                   llvm::toString(target.takeError()));
  const ::loom::fabric::SpatialCoreInternalOccurrenceRef internal{
      spatialCore, std::move(*target)};
  auto physical =
      ::loom::fabric::FabricPhysicalOccurrenceOwnerRef::create(internal);
  if (!physical)
    return invalid("imported Spatial capacity owner cannot be occurrence-"
                   "qualified: " +
                   llvm::toString(physical.takeError()));
  return llvm::Error::success();
}

const ::loom::fabric::FabricPhysicalTraversalView *
findTraversal(const ::loom::fabric::FabricArtifactView &fabric,
              const ::loom::fabric::FabricPhysicalTraversalRef &reference) {
  const auto found =
      llvm::find_if(fabric.physicalTraversals(), [&](const auto &view) {
        return view.reference == reference;
      });
  return found == fabric.physicalTraversals().end() ? nullptr : &*found;
}

llvm::Error validateDirectTraversalOwners(
    const ::loom::fabric::FabricArtifactView &fabric,
    llvm::ArrayRef<::loom::fabric::FabricPhysicalTraversalRef> traversals) {
  for (const auto &reference : traversals) {
    const auto *traversal = findTraversal(fabric, reference);
    if (!traversal)
      return invalid("System route names an absent Fabric traversal");
    for (const auto &use : traversal->impliedUses) {
      if (llvm::Error error = validateDirectOwner(use.activationGroup.owner))
        return error;
      if (llvm::Error error = validateDirectOwner(use.pattern.owner.catalog()))
        return error;
    }
  }
  return llvm::Error::success();
}

llvm::Error validateInternalTraversalOwners(
    const ::loom::fabric::FabricArtifactView &fabric,
    ::loom::fabric::SpatialCoreOccurrenceRef spatialCore,
    llvm::ArrayRef<::loom::fabric::FabricPhysicalTraversalRef> traversals) {
  for (const auto &reference : traversals) {
    const auto *traversal = findTraversal(fabric, reference);
    if (!traversal)
      return invalid("imported Spatial route names an absent traversal");
    for (const auto &use : traversal->impliedUses) {
      if (llvm::Error error =
              validateInternalOwner(spatialCore, use.activationGroup.owner))
        return error;
      if (llvm::Error error =
              validateInternalOwner(spatialCore, use.pattern.owner.catalog()))
        return error;
    }
  }
  return llvm::Error::success();
}

const ::loom::fabric::FabricArtifactView *
resolveOccurrenceModule(const ::loom::fabric::FabricSystemRootView &fabric,
                        ::loom::fabric::AccCoreOccurrenceRef core,
                        const SpatialMappingView &mapping) {
  const auto target = fabric.spatialCoreTarget(core);
  if (!target ||
      target->dependencyOrdinal >= fabric.artifact().importedModules().size())
    return nullptr;
  const auto &module =
      fabric.artifact().importedModules()[target->dependencyOrdinal];
  const auto root = module.moduleRootTemplate();
  if (!root || *root != target->target ||
      module.identity() != mapping.fabricIdentity())
    return nullptr;
  return &module;
}

void collectSpatialRouteTraversals(
    const SpatialMappingView &mapping,
    std::vector<ResourceCapacityRouteProjection> &routes,
    std::size_t namespaceOrdinal) {
  for (const auto &route : mapping.routeTrees()) {
    ResourceCapacityRouteProjection projection;
    projection.namespaceOrdinal = namespaceOrdinal;
    if (route.localTraversal)
      projection.traversals.push_back(*route.localTraversal);
    for (const auto &node : route.nodes)
      if (node.incomingTraversal)
        projection.traversals.push_back(*node.incomingTraversal);
    for (const auto &sink : route.sinks)
      if (sink.localTraversal)
        projection.traversals.push_back(*sink.localTraversal);
    routes.push_back(std::move(projection));
  }
}

struct NamespaceMetadata final {
  std::optional<::loom::fabric::SpatialCoreOccurrenceRef> spatialCore;
};

} // namespace

llvm::Error verifySystemMappingCapacity(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ::loom::fabric::FabricSystemRootView &fabric,
    const SystemExecutionBindingView &execution,
    llvm::ArrayRef<SystemServiceRealizationView> services,
    llvm::ArrayRef<SystemResourceUseView> resourceUses,
    llvm::ArrayRef<std::string> resourceUseActivationKeys,
    const ArtifactStore &store) {
  if (resourceUses.size() != resourceUseActivationKeys.size())
    return invalid("System ResourceUse activation projection is incomplete");

  std::vector<ResourceCapacityNamespaceView> namespaces{
      ResourceCapacityNamespaceView{
          &fabric.artifact(),
          rootResourceCapacityQualifier(fabric.artifact())}};
  std::vector<NamespaceMetadata> metadata(1);
  std::vector<ResourceCapacityUseProjection> uses;
  uses.reserve(resourceUses.size());
  for (const auto &[ordinal, use] : llvm::enumerate(resourceUses)) {
    if (llvm::Error error = validateDirectOwner(use.useSite.owner.catalog()))
      return error;
    uses.push_back(ResourceCapacityUseProjection{
        0, use.useSite, resourceUseActivationKeys[ordinal]});
  }

  std::vector<ResourceCapacityRouteProjection> routes;
  for (const auto &service : services)
    for (const auto &plan : service.plans)
      for (const auto &leg : plan.transferLegs) {
        ResourceCapacityRouteProjection route;
        route.namespaceOrdinal = 0;
        for (const auto &node : leg.nodes)
          route.traversals.push_back(node.incomingTraversal);
        if (llvm::Error error = validateDirectTraversalOwners(fabric.artifact(),
                                                              route.traversals))
          return error;
        routes.push_back(std::move(route));
      }

  auto contexts = projectSystemExecutionContexts(dataflow, execution);
  if (!contexts)
    return contexts.takeError();

  std::map<std::string, SpatialMappingView> mappings;
  std::map<std::string, std::size_t> namespaceByOccurrence;
  std::set<std::string> routedMappings;
  for (const auto &context : contexts->spatialDomains) {
    const std::string mappingKey =
        byteKey(encodeArtifactRootReference(context.spatialMapping));
    auto foundMapping = mappings.find(mappingKey);
    if (foundMapping == mappings.end()) {
      auto imported = importSpatialMapping(context.spatialMapping, store);
      if (!imported)
        return imported.takeError();
      foundMapping = mappings.emplace(mappingKey, imported->view()).first;
    }
    const SpatialMappingView &mapping = foundMapping->second;
    const auto spatialCore =
        ::loom::fabric::SpatialCoreOccurrenceRef{context.context.accCore};
    const auto *module =
        resolveOccurrenceModule(fabric, context.context.accCore, mapping);
    if (!module)
      return invalid("imported SpatialMapping does not match its exact AccCore "
                     "occurrence");

    const std::string occurrenceKey =
        byteKey(::loom::fabric::canonicalFabricBytes(spatialCore));
    auto [namespacePosition, inserted] =
        namespaceByOccurrence.try_emplace(occurrenceKey, namespaces.size());
    if (inserted) {
      namespaces.push_back(ResourceCapacityNamespaceView{
          module,
          occurrenceResourceCapacityQualifier(fabric.artifact(), spatialCore)});
      metadata.push_back(NamespaceMetadata{spatialCore});
    } else if (namespaces[namespacePosition->second].fabric->identity() !=
               module->identity()) {
      return invalid("one SpatialCore occurrence resolves two Module owners");
    }
    const std::size_t namespaceOrdinal = namespacePosition->second;

    auto graph =
        ::dataflow::encodeDataflowReference(dataflow.identity(), context.graph);
    if (!graph)
      return graph.takeError();
    for (const auto &use : mapping.resourceUses()) {
      if (llvm::Error error =
              validateInternalOwner(spatialCore, use.useSite.owner.catalog()))
        return error;
      auto local =
          deriveSpatialCapacityActivationKey(*module, dataflow.identity(), use);
      if (!local)
        return local.takeError();
      std::string activation;
      appendSized(activation, *graph);
      appendSized(activation, mapping.identity().bytes());
      appendU64(activation, local->size());
      activation.append(*local);
      uses.push_back(ResourceCapacityUseProjection{
          namespaceOrdinal, use.useSite, std::move(activation)});
    }

    std::string routedKey = occurrenceKey;
    appendSized(routedKey, mapping.identity().bytes());
    if (routedMappings.insert(routedKey).second) {
      const std::size_t routeBegin = routes.size();
      collectSpatialRouteTraversals(mapping, routes, namespaceOrdinal);
      for (std::size_t ordinal = routeBegin; ordinal < routes.size(); ++ordinal)
        if (llvm::Error error = validateInternalTraversalOwners(
                *module, spatialCore, routes[ordinal].traversals))
          return error;
    }
  }

  auto overuse = deriveResourceCapacityOveruse(namespaces, uses, routes);
  if (!overuse)
    return overuse.takeError();
  if (overuse->total == 0)
    return llvm::Error::success();
  if (!overuse->firstWitness ||
      overuse->firstWitness->namespaceOrdinal >= metadata.size())
    return invalid("CapacityOveruse has no canonical physical witness");
  const auto &witness = *overuse->firstWitness;
  std::string owner;
  if (metadata[witness.namespaceOrdinal].spatialCore) {
    auto target = projectModuleOwner(witness.owner);
    if (!target)
      return target.takeError();
    owner = ::loom::fabric::printFabricRef(
        ::loom::fabric::SpatialCoreInternalOccurrenceRef{
            *metadata[witness.namespaceOrdinal].spatialCore,
            std::move(*target)});
  } else {
    owner = ::loom::fabric::printFabricRef(witness.owner);
  }
  return invalid(llvm::Twine("CapacityOveruse at ") + owner + " state " +
                 llvm::Twine(witness.state.ordinal()) + " dimension " +
                 llvm::Twine(witness.dimension.ordinal()) + " uses " +
                 llvm::Twine(witness.usage) + " of " +
                 llvm::Twine(witness.capacity));
}

} // namespace loom::mapping::detail
