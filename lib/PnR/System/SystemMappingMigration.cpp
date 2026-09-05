#include "PnR/System/SystemMappingMigration.h"

#include "ResourceTimeTransitionInternal.h"

#include "Common/ArtifactLocalReference.h"
#include "Common/ArtifactStore.h"
#include "Common/BlobStore.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Dataflow/IR/DataflowEventDerivation.h"
#include "Dataflow/IR/DataflowReferenceCodec.h"
#include "Deployment/Deployment.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/Artifact/FabricSystemReferenceRemapper.h"
#include "Fabric/Artifact/FabricSystemRootView.h"
#include "Fabric/Identity/FabricRefBytes.h"
#include "Mapping/Artifact/MappingArtifact.h"
#include "Mapping/Artifact/SystemMappingConstraintSet.h"
#include "Mapping/Artifact/SystemMappingExecutionProjection.h"
#include "PnR/System/SystemPnrProblem.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <limits>
#include <optional>
#include <set>
#include <system_error>
#include <tuple>

namespace loom::pnr {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      "system_mapping_migration_invalid: " + message);
}

bool rootLess(::dataflow::RootThreadLaunchRef lhs,
              ::dataflow::RootThreadLaunchRef rhs) {
  if (lhs.artifact != rhs.artifact)
    return lhs.artifact.bytes() < rhs.artifact.bytes();
  return lhs.entity.value() < rhs.entity.value();
}

bool graphLess(::dataflow::GraphRef lhs, ::dataflow::GraphRef rhs) {
  if (lhs.artifact != rhs.artifact)
    return lhs.artifact.bytes() < rhs.artifact.bytes();
  return lhs.entity.value() < rhs.entity.value();
}

std::vector<std::uint8_t> canonicalResourceBytes(
    const ::loom::fabric::FabricPhysicalOccurrenceOwnerRef &resource) {
  return ::loom::fabric::canonicalFabricBytes(resource);
}

bool allocationsEquivalent(llvm::ArrayRef<ResourceTimeRegionAllocation> lhs,
                           llvm::ArrayRef<ResourceTimeRegionAllocation> rhs) {
  if (lhs.size() != rhs.size())
    return false;
  for (const ResourceTimeRegionAllocation &left : lhs) {
    const auto right = llvm::find_if(rhs, [&](const auto &candidate) {
      return candidate.region == left.region;
    });
    if (right == rhs.end() || left.resources.size() != right->resources.size())
      return false;
    std::vector<std::vector<std::uint8_t>> leftResources;
    std::vector<std::vector<std::uint8_t>> rightResources;
    leftResources.reserve(left.resources.size());
    rightResources.reserve(right->resources.size());
    for (const auto &resource : left.resources)
      leftResources.push_back(canonicalResourceBytes(resource));
    for (const auto &resource : right->resources)
      rightResources.push_back(canonicalResourceBytes(resource));
    llvm::sort(leftResources);
    llvm::sort(rightResources);
    if (leftResources != rightResources)
      return false;
  }
  return true;
}

void canonicalizeResources(
    std::vector<::loom::fabric::FabricPhysicalOccurrenceOwnerRef> &resources) {
  llvm::sort(resources, [](const auto &lhs, const auto &rhs) {
    return canonicalResourceBytes(lhs) < canonicalResourceBytes(rhs);
  });
  resources.erase(std::unique(resources.begin(), resources.end()),
                  resources.end());
}

void appendU64(std::vector<std::uint8_t> &bytes, std::uint64_t value) {
  for (int shift = 56; shift >= 0; shift -= 8)
    bytes.push_back(static_cast<std::uint8_t>(value >> shift));
}

llvm::Expected<std::uint64_t> readU64(llvm::ArrayRef<std::uint8_t> bytes,
                                      std::size_t &offset) {
  if (offset > bytes.size() || bytes.size() - offset < 8)
    return invalid("migration seed is truncated");
  std::uint64_t value = 0;
  for (unsigned index = 0; index != 8; ++index)
    value = (value << 8) | bytes[offset++];
  return value;
}

void appendBlob(std::vector<std::uint8_t> &bytes,
                llvm::ArrayRef<std::uint8_t> value) {
  appendU64(bytes, value.size());
  bytes.insert(bytes.end(), value.begin(), value.end());
}

llvm::Expected<llvm::ArrayRef<std::uint8_t>>
readBlob(llvm::ArrayRef<std::uint8_t> bytes, std::size_t &offset) {
  auto size = readU64(bytes, offset);
  if (!size)
    return size.takeError();
  if (offset > bytes.size() || *size > bytes.size() - offset)
    return invalid("migration seed blob is truncated");
  auto result = bytes.slice(offset, static_cast<std::size_t>(*size));
  offset += static_cast<std::size_t>(*size);
  return result;
}

llvm::Expected<ArtifactRootReference>
readRootReference(llvm::ArrayRef<std::uint8_t> bytes, std::size_t &offset) {
  if (offset > bytes.size())
    return invalid("migration root-reference offset is outside payload");
  auto decoded = decodeArtifactRootReferencePrefix(bytes.drop_front(offset));
  if (!decoded)
    return decoded.takeError();
  if (decoded->byteCount > bytes.size() - offset)
    return invalid("migration root reference is truncated");
  offset += decoded->byteCount;
  return std::move(decoded->reference);
}

void appendCorrespondence(
    std::vector<std::uint8_t> &bytes,
    const SystemExecutionBindingCorrespondence &correspondence) {
  appendU64(bytes, correspondence.entities().size());
  for (const auto &entry : correspondence.entities()) {
    appendU64(bytes, static_cast<std::uint64_t>(entry.source.kind));
    appendU64(bytes, entry.source.id);
    appendU64(bytes, entry.target.id);
  }
  appendU64(bytes, correspondence.transferPatterns().size());
  for (const auto &entry : correspondence.transferPatterns()) {
    appendBlob(bytes, ::loom::fabric::canonicalFabricBytes(entry.source));
    appendBlob(bytes, ::loom::fabric::canonicalFabricBytes(entry.target));
  }
  appendU64(bytes, correspondence.modules().size());
  for (const SystemModuleCorrespondence &entry : correspondence.modules()) {
    const auto parent = encodeArtifactRootReference(entry.parent);
    const auto child = encodeArtifactRootReference(entry.child);
    bytes.insert(bytes.end(), parent.begin(), parent.end());
    bytes.insert(bytes.end(), child.begin(), child.end());
  }
}

std::vector<std::uint8_t>
canonicalSeedBytes(const ArtifactRootReference &checkpoint,
                   const SystemExecutionBindingCorrespondence &correspondence,
                   const SystemMappingMigrationContext &context,
                   ::loom::fabric::AccCoreOccurrenceRef reopenedParentAccCore) {
  std::vector<std::uint8_t> bytes = encodeArtifactRootReference(checkpoint);
  const auto child = encodeArtifactRootReference(correspondence.childSystem());
  bytes.insert(bytes.end(), child.begin(), child.end());
  const auto constraints =
      encodeArtifactRootReference(context.childConstraints());
  bytes.insert(bytes.end(), constraints.begin(), constraints.end());
  appendBlob(bytes, context.resolvedPnrConfigDigest().bytes());
  appendU64(bytes, context.spatialMappings().size());
  for (const ArtifactRootReference &mapping : context.spatialMappings()) {
    const auto encoded = encodeArtifactRootReference(mapping);
    bytes.insert(bytes.end(), encoded.begin(), encoded.end());
  }
  appendBlob(bytes,
             ::loom::fabric::canonicalFabricBytes(reopenedParentAccCore));
  appendCorrespondence(bytes, correspondence);
  return bytes;
}

llvm::Expected<std::vector<std::uint8_t>> canonicalFinalizedSeedBytes(
    const ArtifactRootReference &parentMapping,
    const SystemExecutionBindingCorrespondence &correspondence,
    const SystemMappingMigrationContext &context,
    llvm::ArrayRef<::dataflow::RootThreadLaunchRef> reopenedRoots) {
  std::vector<std::uint8_t> bytes = encodeArtifactRootReference(parentMapping);
  const auto child = encodeArtifactRootReference(correspondence.childSystem());
  bytes.insert(bytes.end(), child.begin(), child.end());
  const auto constraints =
      encodeArtifactRootReference(context.childConstraints());
  bytes.insert(bytes.end(), constraints.begin(), constraints.end());
  appendBlob(bytes, context.resolvedPnrConfigDigest().bytes());
  appendU64(bytes, context.spatialMappings().size());
  for (const ArtifactRootReference &mapping : context.spatialMappings()) {
    const auto encoded = encodeArtifactRootReference(mapping);
    bytes.insert(bytes.end(), encoded.begin(), encoded.end());
  }
  appendU64(bytes, reopenedRoots.size());
  for (const ::dataflow::RootThreadLaunchRef root : reopenedRoots) {
    auto encoded = ::dataflow::encodeDataflowReference(root.artifact, root);
    if (!encoded)
      return encoded.takeError();
    appendBlob(bytes, *encoded);
  }
  appendCorrespondence(bytes, correspondence);
  return bytes;
}

llvm::Expected<SystemMappingMigrationContext>
readMigrationContext(llvm::ArrayRef<std::uint8_t> bytes, std::size_t &offset) {
  auto constraints = readRootReference(bytes, offset);
  if (!constraints)
    return constraints.takeError();
  auto digestBytes = readBlob(bytes, offset);
  if (!digestBytes)
    return digestBytes.takeError();
  auto digest = ComponentViewDigest::fromBytes(*digestBytes);
  if (!digest)
    return digest.takeError();
  auto mappingCount = readU64(bytes, offset);
  if (!mappingCount)
    return mappingCount.takeError();
  if (*mappingCount > bytes.size())
    return invalid("migration SpatialMapping count exceeds payload size");
  std::vector<ArtifactRootReference> mappings;
  mappings.reserve(static_cast<std::size_t>(*mappingCount));
  for (std::uint64_t ordinal = 0; ordinal != *mappingCount; ++ordinal) {
    auto mapping = readRootReference(bytes, offset);
    if (!mapping)
      return mapping.takeError();
    mappings.push_back(std::move(*mapping));
  }
  return SystemMappingMigrationContext::get(
      std::move(*constraints), std::move(mappings), std::move(*digest));
}

struct DecodedCorrespondence final {
  std::vector<::loom::fabric::FabricSystemEntityCorrespondence> entities;
  std::vector<::loom::fabric::FabricSystemTransferPatternCorrespondence>
      transferPatterns;
  std::vector<SystemModuleCorrespondence> modules;
};

llvm::Expected<DecodedCorrespondence>
readCorrespondence(llvm::ArrayRef<std::uint8_t> bytes, std::size_t &offset) {
  auto count = readU64(bytes, offset);
  if (!count)
    return count.takeError();
  if (*count > bytes.size())
    return invalid("migration entity count exceeds payload size");
  std::vector<::loom::fabric::FabricSystemEntityCorrespondence> entities;
  entities.reserve(static_cast<std::size_t>(*count));
  for (std::uint64_t ordinal = 0; ordinal != *count; ++ordinal) {
    auto rawKind = readU64(bytes, offset);
    if (!rawKind)
      return rawKind.takeError();
    if (*rawKind >=
        ::loom::fabric::fabricClosedBound(::loom::fabric::FabricEntityKind()))
      return invalid("migration entity has an unknown kind");
    auto source = readU64(bytes, offset);
    if (!source)
      return source.takeError();
    auto target = readU64(bytes, offset);
    if (!target)
      return target.takeError();
    const auto kind = static_cast<::loom::fabric::FabricEntityKind>(*rawKind);
    entities.push_back({{kind, *source}, {kind, *target}});
  }
  auto patternCount = readU64(bytes, offset);
  if (!patternCount)
    return patternCount.takeError();
  if (*patternCount > bytes.size())
    return invalid("migration transfer-pattern count exceeds payload size");
  std::vector<::loom::fabric::FabricSystemTransferPatternCorrespondence>
      transferPatterns;
  transferPatterns.reserve(static_cast<std::size_t>(*patternCount));
  for (std::uint64_t ordinal = 0; ordinal != *patternCount; ++ordinal) {
    auto sourceBytes = readBlob(bytes, offset);
    if (!sourceBytes)
      return sourceBytes.takeError();
    auto source = ::loom::fabric::decodeFabricRef<
        ::loom::fabric::FabricTransferPatternRef>(*sourceBytes);
    if (!source)
      return source.takeError();
    auto targetBytes = readBlob(bytes, offset);
    if (!targetBytes)
      return targetBytes.takeError();
    auto target = ::loom::fabric::decodeFabricRef<
        ::loom::fabric::FabricTransferPatternRef>(*targetBytes);
    if (!target)
      return target.takeError();
    transferPatterns.push_back({std::move(*source), std::move(*target)});
  }
  auto moduleCount = readU64(bytes, offset);
  if (!moduleCount)
    return moduleCount.takeError();
  if (*moduleCount > bytes.size())
    return invalid("migration Module count exceeds payload size");
  std::vector<SystemModuleCorrespondence> modules;
  modules.reserve(static_cast<std::size_t>(*moduleCount));
  for (std::uint64_t ordinal = 0; ordinal != *moduleCount; ++ordinal) {
    auto parent = readRootReference(bytes, offset);
    if (!parent)
      return parent.takeError();
    auto child = readRootReference(bytes, offset);
    if (!child)
      return child.takeError();
    modules.push_back({std::move(*parent), std::move(*child)});
  }
  return DecodedCorrespondence{std::move(entities), std::move(transferPatterns),
                               std::move(modules)};
}

llvm::Expected<ArtifactRootReference>
targetModule(const ::loom::fabric::FinalizedFabricRoot &system,
             const ::loom::fabric::FabricSystemRootView &view,
             ::loom::fabric::AccCoreOccurrenceRef core) {
  if (!llvm::is_contained(view.artifact().accCoreOccurrences(), core))
    return invalid("migration correspondence names a foreign AccCore");
  const auto target = view.spatialCoreTarget(core);
  if (!target ||
      target->dependencyOrdinal >= system.directDependencies().size())
    return invalid("migration AccCore has no imported Module target");
  const auto &dependency =
      system.directDependencies()[target->dependencyOrdinal];
  if (dependency.role != ::loom::fabric::FabricDependencyRole::ImportedModule)
    return invalid("migration AccCore target is not an imported Module");
  return dependency.root;
}

llvm::Error validateCorrespondence(
    const SystemExecutionBindingCorrespondence &correspondence,
    const ArtifactStore &store) {
  auto parent = ::loom::fabric::importEntireFabricRoot(
      correspondence.parentSystem(), store);
  if (!parent)
    return parent.takeError();
  auto child = ::loom::fabric::importEntireFabricRoot(
      correspondence.childSystem(), store);
  if (!child)
    return child.takeError();
  auto parentView = ::loom::fabric::requireSystemRoot(parent->view());
  if (!parentView)
    return parentView.takeError();
  auto childView = ::loom::fabric::requireSystemRoot(child->view());
  if (!childView)
    return childView.takeError();
  for (const auto &entry : correspondence.entities()) {
    const auto parentKind = parentView->artifact().entityKind(entry.source.id);
    const auto childKind = childView->artifact().entityKind(entry.target.id);
    if (!parentKind || *parentKind != entry.source.kind)
      return invalid("migration correspondence names a foreign parent entity");
    if (!childKind || *childKind != entry.target.kind)
      return invalid("migration correspondence names a foreign child entity");
    if (entry.source.kind != entry.target.kind)
      return invalid("migration correspondence changes an entity kind");
  }
  for (const auto &entry : correspondence.transferPatterns()) {
    if (!parentView->transferPattern(entry.source))
      return invalid(
          "migration correspondence names a foreign parent transfer pattern");
    if (!childView->transferPattern(entry.target))
      return invalid(
          "migration correspondence names a foreign child transfer pattern");
  }
  std::vector<SystemModuleCorrespondence> observedModules;
  observedModules.reserve(correspondence.accCores().size());
  for (const SystemAccCoreCorrespondence &entry : correspondence.accCores()) {
    auto parentModule = targetModule(*parent, *parentView, entry.parent);
    if (!parentModule)
      return parentModule.takeError();
    auto childModule = targetModule(*child, *childView, entry.child);
    if (!childModule)
      return childModule.takeError();
    observedModules.push_back({*parentModule, *childModule});
    const auto module = llvm::find_if(
        correspondence.modules(), [&](const SystemModuleCorrespondence &value) {
          return value.parent == *parentModule;
        });
    if (module == correspondence.modules().end() ||
        module->child != *childModule)
      return invalid(
          "migration correspondence changes an AccCore Module without exact "
          "Module lineage");
  }
  const auto moduleLess = [](const SystemModuleCorrespondence &lhs,
                             const SystemModuleCorrespondence &rhs) {
    if (artifactRootReferenceLess(lhs.parent, rhs.parent))
      return true;
    if (artifactRootReferenceLess(rhs.parent, lhs.parent))
      return false;
    return artifactRootReferenceLess(lhs.child, rhs.child);
  };
  llvm::sort(observedModules, moduleLess);
  observedModules.erase(
      std::unique(observedModules.begin(), observedModules.end()),
      observedModules.end());
  if (llvm::ArrayRef<SystemModuleCorrespondence>(observedModules) !=
      correspondence.modules())
    return invalid("Module lineage differs from the preserved AccCore targets");
  return llvm::Error::success();
}

template <typename Binding, typename Key>
void bindingsForKey(llvm::ArrayRef<Binding> bindings, const Key &key,
                    llvm::SmallVectorImpl<Binding> &storage) {
  for (const Binding &binding : bindings)
    if (binding.key == key)
      storage.push_back(binding);
}

template <typename Target, typename Binding>
std::optional<Target>
exactCellTarget(llvm::ArrayRef<Binding> bindings,
                const ::loom::mapping::SystemPresburgerCell &cell,
                bool &ambiguous) {
  std::optional<Target> result;
  for (const Binding &binding : bindings) {
    for (const auto &clause : binding.clauses) {
      if (!llvm::is_contained(clause.cells, cell))
        continue;
      if (result && *result != clause.target) {
        ambiguous = true;
        return std::nullopt;
      }
      result = clause.target;
    }
    if (!result && binding.defaultTarget)
      result = *binding.defaultTarget;
  }
  return result;
}

std::optional<::loom::fabric::AccCoreOccurrenceRef>
mapAccCore(const SystemExecutionBindingCorrespondence &correspondence,
           ::loom::fabric::AccCoreOccurrenceRef parent, bool &ambiguous) {
  std::optional<::loom::fabric::AccCoreOccurrenceRef> result;
  for (const SystemAccCoreCorrespondence &entry : correspondence.accCores()) {
    if (entry.parent != parent)
      continue;
    if (result && *result != entry.child) {
      ambiguous = true;
      return std::nullopt;
    }
    result = entry.child;
  }
  return result;
}

std::optional<PnrIndex>
terminalOrdinal(const FrozenSystemPnrProblem &problem,
                const ::loom::mapping::SystemTransferTerminalKey &key) {
  const auto found =
      llvm::find_if(problem.serviceTerminals(),
                    [&](const auto &row) { return row.key == key; });
  if (found == problem.serviceTerminals().end())
    return std::nullopt;
  return static_cast<PnrIndex>(found - problem.serviceTerminals().begin());
}

const ::loom::mapping::SystemTransferLegView *
findParentRoute(const ::loom::mapping::SystemMappingView &mapping,
                const ::loom::mapping::CanonicalServiceLegKey &key) {
  const ::loom::mapping::SystemTransferLegView *result = nullptr;
  for (const auto &service : mapping.serviceRealizations())
    for (const auto &plan : service.plans)
      for (const auto &route : plan.transferLegs) {
        if (route.leg != key)
          continue;
        if (result)
          return nullptr;
        result = &route;
      }
  return result;
}

struct ProjectedRoute final {
  PnrIndex rootEndpoint = getInvalidPnrIndex();
  std::vector<SystemServiceRouteNodeSelection> nodes;
  std::vector<SystemServiceRouteSinkSelection> sinks;
};

std::optional<ProjectedRoute> projectParentRoute(
    const ::loom::mapping::SystemTransferLegView &parent,
    const ::loom::fabric::FabricSystemReferenceRemapper &remapper,
    const FrozenSystemPnrProblem &problem) {
  auto mappedRoot = remapper.remap(parent.rootEndpoint);
  if (!mappedRoot) {
    llvm::consumeError(mappedRoot.takeError());
    return std::nullopt;
  }
  auto root = problem.routingTopology().endpointOrdinal(*mappedRoot);
  if (!root)
    return std::nullopt;
  ProjectedRoute result;
  result.rootEndpoint = *root;
  result.nodes.push_back({*root, getInvalidPnrIndex(), getInvalidPnrIndex()});

  std::vector<const ::loom::mapping::SystemTransferRouteNodeView *> nodes;
  nodes.reserve(parent.nodes.size());
  for (const auto &node : parent.nodes)
    nodes.push_back(&node);
  llvm::sort(nodes, [](const auto *lhs, const auto *rhs) {
    return lhs->ordinal < rhs->ordinal;
  });
  const auto &topology = problem.routingTopology();
  for (const auto *node : nodes) {
    if (node->ordinal != result.nodes.size() ||
        node->parentOrdinal >= node->ordinal)
      return std::nullopt;
    auto mappedTraversal = remapper.remap(node->incomingTraversal);
    if (!mappedTraversal) {
      llvm::consumeError(mappedTraversal.takeError());
      return std::nullopt;
    }
    auto traversal = topology.traversalOrdinal(*mappedTraversal);
    if (!traversal)
      return std::nullopt;
    const PnrIndex parentEndpoint = result.nodes[node->parentOrdinal].endpoint;
    if (parentEndpoint + 1 >= topology.adjacencyOffsets().size())
      return std::nullopt;
    std::optional<PnrIndex> target;
    for (PnrIndex arc = topology.adjacencyOffsets()[parentEndpoint];
         arc != topology.adjacencyOffsets()[parentEndpoint + 1]; ++arc) {
      if (arc >= topology.arcs().size() ||
          topology.arcs()[arc].traversal != *traversal)
        continue;
      if (target && *target != topology.arcs()[arc].target)
        return std::nullopt;
      target = topology.arcs()[arc].target;
    }
    if (!target)
      return std::nullopt;
    result.nodes.push_back(
        {*target, static_cast<PnrIndex>(node->parentOrdinal), *traversal});
  }
  result.sinks.reserve(parent.sinks.size());
  for (const auto &sink : parent.sinks) {
    auto terminal = terminalOrdinal(problem, sink.terminal);
    if (!terminal || sink.nodeOrdinal >= result.nodes.size())
      return std::nullopt;
    result.sinks.push_back(
        {*terminal, static_cast<PnrIndex>(sink.nodeOrdinal)});
  }
  llvm::sort(result.sinks, [](const auto &lhs, const auto &rhs) {
    return std::tie(lhs.terminal, lhs.node) < std::tie(rhs.terminal, rhs.node);
  });
  return result;
}

llvm::Expected<SystemCandidateRouteSeed> projectFinalizedRoutes(
    const ::loom::mapping::SystemMappingView &mapping,
    const SystemExecutionBindingCorrespondence &correspondence,
    const FrozenSystemPnrProblem &problem) {
  auto remapper = ::loom::fabric::FabricSystemReferenceRemapper::get(
      correspondence.entities(), correspondence.transferPatterns());
  if (!remapper)
    return remapper.takeError();
  SystemCandidateRouteSeed seed;
  seed.routes.reserve(problem.serviceLegs().size());
  for (PnrIndex leg = 0; leg < problem.serviceLegs().size(); ++leg) {
    const auto *parent =
        findParentRoute(mapping, problem.serviceLegs()[leg].key);
    std::optional<ProjectedRoute> projected;
    if (parent)
      projected = projectParentRoute(*parent, *remapper, problem);
    const PnrIndex nodeOffset = static_cast<PnrIndex>(seed.nodes.size());
    const PnrIndex sinkOffset = static_cast<PnrIndex>(seed.sinks.size());
    if (!projected) {
      seed.routes.push_back(
          {leg, getInvalidPnrIndex(), nodeOffset, 0, sinkOffset, 0});
      seed.reroutedLegs.push_back(leg);
      continue;
    }
    seed.nodes.insert(seed.nodes.end(), projected->nodes.begin(),
                      projected->nodes.end());
    seed.sinks.insert(seed.sinks.end(), projected->sinks.begin(),
                      projected->sinks.end());
    seed.routes.push_back({leg, projected->rootEndpoint, nodeOffset,
                           static_cast<PnrIndex>(projected->nodes.size()),
                           sinkOffset,
                           static_cast<PnrIndex>(projected->sinks.size())});
  }
  return seed;
}

} // namespace

llvm::StringRef
resourceTimeTransitionStatusSpelling(ResourceTimeTransitionStatus status) {
  switch (status) {
  case ResourceTimeTransitionStatus::Verified:
    return "verified";
  case ResourceTimeTransitionStatus::Unsupported:
    return "unsupported";
  case ResourceTimeTransitionStatus::ProofNotEstablished:
    return "proof_not_established";
  case ResourceTimeTransitionStatus::CancelledOrTimeout:
    return "cancelled_or_timeout";
  }
  llvm_unreachable("unknown resource-time transition status");
}

llvm::StringRef
resourceTimeLiveStateClassSpelling(ResourceTimeLiveStateClass stateClass) {
  switch (stateClass) {
  case ResourceTimeLiveStateClass::LogicalMemory:
    return "logical_memory";
  case ResourceTimeLiveStateClass::OrderedChannel:
    return "ordered_channel";
  case ResourceTimeLiveStateClass::DynamicWork:
    return "dynamic_work";
  }
  llvm_unreachable("unknown resource-time live-state class");
}

llvm::StringRef resourceTimeLiveStateMigrationSpelling(
    ResourceTimeLiveStateMigration migration) {
  switch (migration) {
  case ResourceTimeLiveStateMigration::RetainedInPlace:
    return "retained_in_place";
  case ResourceTimeLiveStateMigration::Copied:
    return "copied";
  }
  llvm_unreachable("unknown resource-time live-state migration");
}

llvm::StringRef resourceTimeTransitionRefusalReasonSpelling(
    ResourceTimeTransitionRefusalReason reason) {
  switch (reason) {
  case ResourceTimeTransitionRefusalReason::OrderedChannelState:
    return "ordered_channel_state";
  case ResourceTimeTransitionRefusalReason::DynamicWorkState:
    return "dynamic_work_state";
  case ResourceTimeTransitionRefusalReason::LogicalMemoryUnbound:
    return "logical_memory_unbound";
  case ResourceTimeTransitionRefusalReason::LogicalMemoryExtentUnknown:
    return "logical_memory_extent_unknown";
  case ResourceTimeTransitionRefusalReason::LogicalMemoryCopyShapeUnsupported:
    return "logical_memory_copy_shape_unsupported";
  case ResourceTimeTransitionRefusalReason::LogicalMemoryReinitialized:
    return "logical_memory_reinitialized";
  case ResourceTimeTransitionRefusalReason::HardwareBindingChanged:
    return "hardware_binding_changed";
  case ResourceTimeTransitionRefusalReason::
      RuntimeTransitionCapabilityUnavailable:
    return "runtime_transition_capability_unavailable";
  case ResourceTimeTransitionRefusalReason::CompletionFrontierInadmissible:
    return "completion_frontier_inadmissible";
  }
  llvm_unreachable("unknown resource-time transition refusal reason");
}

char ResourceTimeTransitionRefusal::ID = 0;

void ResourceTimeTransitionRefusal::log(llvm::raw_ostream &stream) const {
  stream << "resource_time_transition_refused("
         << resourceTimeTransitionRefusalReasonSpelling(reason_)
         << "): " << message_;
}

std::error_code ResourceTimeTransitionRefusal::convertToErrorCode() const {
  return llvm::inconvertibleErrorCode();
}

llvm::StringRef
resourceTimeSafePointKindSpelling(ResourceTimeSafePointKind kind) {
  switch (kind) {
  case ResourceTimeSafePointKind::Completion:
    return "completion";
  case ResourceTimeSafePointKind::Explicit:
    return "explicit";
  }
  llvm_unreachable("unknown resource-time safe-point kind");
}

llvm::StringRef
resourceTimeReadinessKindSpelling(ResourceTimeReadinessKind kind) {
  switch (kind) {
  case ResourceTimeReadinessKind::Completion:
    return "completion";
  case ResourceTimeReadinessKind::FifoToken:
    return "fifo_token";
  }
  llvm_unreachable("unknown resource-time readiness kind");
}

llvm::StringRef resourceTimeConcurrencyBoundStatusSpelling(
    ResourceTimeConcurrencyBoundStatus status) {
  switch (status) {
  case ResourceTimeConcurrencyBoundStatus::Exact:
    return "exact";
  case ResourceTimeConcurrencyBoundStatus::ProofNotEstablished:
    return "proof_not_established";
  }
  llvm_unreachable("unknown resource-time concurrency bound status");
}

llvm::Error
validateResourceTimeTransition(const ResourceTimeTransition &transition) {
  const auto validateMappingReference =
      [](const ArtifactRootReference &root,
         llvm::StringRef name) -> llvm::Error {
    if (root.schemaIdentity !=
            ::loom::mapping::mappingArtifactSchema.identity ||
        root.schemaVersion != ::loom::mapping::mappingArtifactSchema.version)
      return invalid(name + " is not a Mapping artifact reference");
    return llvm::Error::success();
  };
  if (llvm::Error error =
          validateMappingReference(transition.parent.mapping, "parent_mapping"))
    return error;
  if (llvm::Error error =
          validateMappingReference(transition.child.mapping, "child_mapping"))
    return error;
  const auto validateDeploymentReference =
      [](const std::optional<ArtifactRootReference> &root,
         llvm::StringRef name) -> llvm::Error {
    if (root &&
        (root->schemaIdentity !=
             ::loom::deployment::deploymentSchema.identity ||
         root->schemaVersion != ::loom::deployment::deploymentSchema.version))
      return invalid(name + " is not a Deployment artifact reference");
    return llvm::Error::success();
  };
  if (llvm::Error error = validateDeploymentReference(
          transition.parent.deployment, "parent_deployment"))
    return error;
  if (llvm::Error error = validateDeploymentReference(
          transition.child.deployment, "child_deployment"))
    return error;
  const auto validateAllocations =
      [](llvm::ArrayRef<ResourceTimeRegionAllocation> values,
         llvm::StringRef name) -> llvm::Error {
    std::vector<::loom::fabric::FabricPhysicalOccurrenceOwnerRef> usedResources;
    for (std::size_t index = 0; index != values.size(); ++index) {
      const auto &allocation = values[index];
      if (allocation.resources.empty())
        return invalid(name + " contains a region with no resources");
      for (std::size_t prior = 0; prior != index; ++prior)
        if (values[prior].region == allocation.region)
          return invalid(name + " contains a duplicate region");
      for (std::size_t resource = 0; resource != allocation.resources.size();
           ++resource) {
        for (std::size_t prior = 0; prior != resource; ++prior)
          if (allocation.resources[prior] == allocation.resources[resource])
            return invalid(name + " contains a duplicate resource");
        if (llvm::is_contained(usedResources, allocation.resources[resource]))
          return invalid(name + " assigns one physical resource to "
                                "multiple active regions");
        usedResources.push_back(allocation.resources[resource]);
      }
    }
    return llvm::Error::success();
  };
  if (llvm::Error error =
          validateAllocations(transition.beforeActive, "before_active"))
    return error;
  if (llvm::Error error =
          validateAllocations(transition.afterActive, "after_active"))
    return error;
  std::optional<ArtifactIdentity> activeDataflow;
  for (const auto *allocations :
       {&transition.beforeActive, &transition.afterActive})
    for (const ResourceTimeRegionAllocation &allocation : *allocations) {
      if (activeDataflow && *activeDataflow != allocation.region.artifact)
        return invalid("resource-time transition spans multiple Dataflow "
                       "identities without typed correspondence");
      activeDataflow = allocation.region.artifact;
    }
  for (std::size_t index = 0; index != transition.completedBefore.size();
       ++index) {
    const ::dataflow::RootThreadLaunchRef completed =
        transition.completedBefore[index];
    if (activeDataflow && *activeDataflow != completed.artifact)
      return invalid("resource-time completion frontier names a foreign "
                     "Dataflow root");
    activeDataflow = completed.artifact;
    const auto containsCompleted =
        [&](llvm::ArrayRef<ResourceTimeRegionAllocation> allocations) {
          return llvm::any_of(allocations, [&](const auto &allocation) {
            return allocation.region == completed;
          });
        };
    if (containsCompleted(transition.beforeActive) ||
        containsCompleted(transition.afterActive))
      return invalid("resource-time completion frontier contains an active "
                     "region");
    for (std::size_t prior = 0; prior != index; ++prior)
      if (transition.completedBefore[prior] == completed)
        return invalid("resource-time completion frontier contains a duplicate "
                       "region");
  }
  for (std::size_t index = 0; index != transition.logicalMemories.size();
       ++index) {
    const ResourceTimeLogicalMemoryCorrespondence &memory =
        transition.logicalMemories[index];
    if (activeDataflow && *activeDataflow != memory.memory.artifact)
      return invalid("resource-time live-state correspondence names a "
                     "foreign Dataflow memory");
    for (std::size_t prior = 0; prior != index; ++prior)
      if (transition.logicalMemories[prior].memory == memory.memory)
        return invalid("resource-time live-state correspondence repeats a "
                       "logical memory");
    if (memory.migration == ResourceTimeLiveStateMigration::RetainedInPlace) {
      if (memory.parentBinding != memory.childBinding ||
          memory.migrationTimePicoseconds != 0)
        return invalid("retained-in-place live state must keep its physical "
                       "binding at exact zero migration cost");
    } else if (memory.migration == ResourceTimeLiveStateMigration::Copied) {
      if (memory.parentBinding == memory.childBinding ||
          memory.migrationTimePicoseconds == 0)
        return invalid("copied live state must change physical binding at a "
                       "nonzero migration cost");
    } else {
      return invalid("resource-time live-state correspondence has an unknown "
                     "migration disposition");
    }
  }
  if (transition.status == ResourceTimeTransitionStatus::Verified) {
    if (!transition.safePoint)
      return invalid("verified resource-time transition has no compiler-known "
                     "safe point");
    if (!transition.parent.deployment || !transition.child.deployment)
      return invalid("verified resource-time transition has no exact parent "
                     "and child Deployment references");
    if (!transition.reprogrammingTimePicoseconds ||
        !transition.migrationTimePicoseconds)
      return invalid("verified resource-time transition has no exact "
                     "reprogramming and migration costs");
    if (!transition.resourceDeltaDigest ||
        !transition.configurationDeltaDigest || !transition.routeDeltaDigest)
      return invalid("verified resource-time transition lacks derived delta "
                     "or route digests");
  }
  if (transition.safePoint) {
    if (transition.safePoint->artifact.schemaIdentity.empty())
      return invalid("resource-time transition has an empty safe-point "
                     "artifact");
    if (transition.safePoint->kind == ResourceTimeSafePointKind::Completion) {
      if (transition.safePoint->artifact.schemaIdentity !=
              ::dataflow::canonicalDataflowSchema.identity ||
          transition.safePoint->artifact.schemaVersion !=
              ::dataflow::canonicalDataflowSchema.version)
        return invalid("completion safe point must be owned by Canonical "
                       "Dataflow");
      const auto completing =
          llvm::find_if(transition.beforeActive,
                        [&](const ResourceTimeRegionAllocation &allocation) {
                          return allocation.region.artifact ==
                                     transition.safePoint->artifact.artifact &&
                                 ::dataflow::rootThreadCompletionEventFamily(
                                     allocation.region) == transition.trigger;
                        });
      if (completing == transition.beforeActive.end())
        return invalid("completion safe point is not the completion event of "
                       "an active parent region (active region count " +
                       llvm::Twine(transition.beforeActive.size()) + ")");
    } else if (transition.safePoint->artifact.schemaIdentity ==
                   ::dataflow::canonicalDataflowSchema.identity &&
               transition.safePoint->artifact.schemaVersion ==
                   ::dataflow::canonicalDataflowSchema.version) {
      return invalid("explicit safe point requires a compiler proof artifact, "
                     "not only a Canonical Dataflow root");
    }
  }
  return llvm::Error::success();
}

llvm::Error
verifyResourceTimeTransitionClosure(const ResourceTimeTransition &transition,
                                    const ArtifactStore &artifacts,
                                    const BlobStore &blobs) {
  if (llvm::Error error = validateResourceTimeTransition(transition))
    return error;
  if (transition.status != ResourceTimeTransitionStatus::Verified)
    return invalid("resource-time transition closure requires a verified "
                   "edge status");
  if (!transition.safePoint || !transition.parent.deployment ||
      !transition.child.deployment)
    return invalid("verified resource-time transition lost required closure "
                   "references");

  auto parentDeployment = ::loom::deployment::importDeployment(
      *transition.parent.deployment, artifacts, blobs);
  if (!parentDeployment)
    return parentDeployment.takeError();
  auto childDeployment = ::loom::deployment::importDeployment(
      *transition.child.deployment, artifacts, blobs);
  if (!childDeployment)
    return childDeployment.takeError();
  if (parentDeployment->deployment().systemMapping() !=
      transition.parent.mapping)
    return invalid("parent Deployment does not select the parent "
                   "SystemMapping");
  if (childDeployment->deployment().systemMapping() != transition.child.mapping)
    return invalid("child Deployment does not select the child "
                   "SystemMapping");

  auto parentMapping = ::loom::mapping::importSystemMapping(
      transition.parent.mapping, artifacts);
  if (!parentMapping)
    return parentMapping.takeError();
  auto childMapping =
      ::loom::mapping::importSystemMapping(transition.child.mapping, artifacts);
  if (!childMapping)
    return childMapping.takeError();
  if (parentMapping->view().dataflowIdentity() !=
      childMapping->view().dataflowIdentity())
    return invalid("resource-time transition changes Canonical Dataflow "
                   "without a typed live-state correspondence owner");
  if (parentMapping->view().fabricIdentity() !=
      childMapping->view().fabricIdentity())
    return invalid("resource-time transition changes the immutable Fabric");

  const ArtifactRootReference dataflowReference{
      ::dataflow::canonicalDataflowSchema.identity.str(),
      ::dataflow::canonicalDataflowSchema.version,
      parentMapping->view().dataflowIdentity()};
  auto dataflowArtifact =
      ::dataflow::importCanonicalDataflow(dataflowReference, artifacts);
  if (!dataflowArtifact)
    return dataflowArtifact.takeError();
  auto dataflow = dataflowArtifact->view();
  if (!dataflow)
    return dataflow.takeError();
  if (llvm::Error error = dataflow->validate(transition.trigger))
    return invalid("resource-time transition trigger is not owned by the "
                   "endpoint "
                   "Dataflow: " +
                   llvm::toString(std::move(error)));
  if (transition.safePoint->kind == ResourceTimeSafePointKind::Explicit)
    return invalid("explicit resource-time safe-point closure is not "
                   "established by a typed compiler proof importer");

  const ArtifactRootReference fabricReference{
      ::loom::fabric::fabricArtifactSchema.identity.str(),
      ::loom::fabric::fabricArtifactSchema.version,
      parentMapping->view().fabricIdentity()};
  auto fabricArtifact =
      ::loom::fabric::importEntireFabricRoot(fabricReference, artifacts);
  if (!fabricArtifact)
    return fabricArtifact.takeError();
  auto system = ::loom::fabric::requireSystemRoot(fabricArtifact->view());
  if (!system)
    return system.takeError();
  auto parentContexts = ::loom::mapping::projectSystemExecutionContexts(
      *dataflow, parentMapping->view().executionBindings());
  if (!parentContexts)
    return parentContexts.takeError();
  auto childContexts = ::loom::mapping::projectSystemExecutionContexts(
      *dataflow, childMapping->view().executionBindings());
  if (!childContexts)
    return childContexts.takeError();

  const auto verifyAllocations =
      [&](llvm::ArrayRef<ResourceTimeRegionAllocation> allocations,
          const ::loom::mapping::SystemExecutionContextProjection &contexts,
          llvm::StringRef name) -> llvm::Error {
    for (const ResourceTimeRegionAllocation &allocation : allocations) {
      if (allocation.region.artifact != dataflowReference.artifact)
        return invalid(name + " names a foreign Dataflow region");
      auto root = dataflow->resolve(allocation.region);
      if (!root)
        return root.takeError();
      auto expected =
          projectResourceTimeMappingResources(contexts, allocation.region);
      if (!expected)
        return expected.takeError();
      std::vector<::loom::fabric::FabricPhysicalOccurrenceOwnerRef> observed =
          allocation.resources;
      canonicalizeResources(observed);
      if (observed != *expected)
        return invalid(name + " disagrees with the independently imported "
                              "SystemMapping execution binding");
      for (const auto &resource : observed) {
        auto resolved = system->resolvePhysicalOwner(resource);
        if (!resolved)
          return invalid(name + " names a resource outside the endpoint "
                                "Fabric");
      }
    }
    return llvm::Error::success();
  };
  if (llvm::Error error = verifyAllocations(
          transition.beforeActive, *parentContexts, "parent allocation"))
    return error;
  if (llvm::Error error = verifyAllocations(transition.afterActive,
                                            *childContexts, "child allocation"))
    return error;

  if (llvm::Error error = verifyResourceTimeTransitionDeltaDigests(
          transition, artifacts, blobs))
    return error;
  return llvm::Error::success();
}

llvm::Error validateResourceTimeTransitionSequence(
    const ResourceTimeTransitionSequence &sequence) {
  for (std::size_t index = 0; index != sequence.transitions.size(); ++index) {
    const ResourceTimeTransition &transition = sequence.transitions[index];
    if (llvm::Error error = validateResourceTimeTransition(transition))
      return error;
    if (index != 0 &&
        sequence.transitions[index - 1].child != transition.parent)
      return invalid("resource-time transition sequence is not chained by "
                     "Mapping and Deployment reference");
  }
  return llvm::Error::success();
}

llvm::Error validateResourceTimeScheduleWitness(
    const ResourceTimeScheduleWitness &witness) {
  if (witness.regions.empty())
    return invalid("resource-time schedule witness has no regions");
  if (witness.scenarios.empty())
    return invalid("resource-time schedule witness has no scenarios");
  if (witness.minimumConcurrentRegions == 0 ||
      witness.maximumConcurrentRegions == 0 ||
      witness.minimumConcurrentRegions > witness.maximumConcurrentRegions ||
      witness.maximumConcurrentRegions > witness.regions.size())
    return invalid("resource-time schedule witness has an invalid concurrency "
                   "bound");

  const auto hasRegion = [&](::dataflow::RootThreadLaunchRef reference) {
    return llvm::is_contained(witness.regions, reference);
  };
  for (std::size_t index = 0; index != witness.regions.size(); ++index) {
    if (index != 0 &&
        witness.regions[index].artifact != witness.regions.front().artifact)
      return invalid("resource-time schedule witness spans multiple Dataflow "
                     "identities without typed correspondence");
    for (std::size_t prior = 0; prior != index; ++prior)
      if (witness.regions[prior] == witness.regions[index])
        return invalid("resource-time schedule witness contains a duplicate "
                       "region");
  }

  const auto mappingReference = [](const ArtifactRootReference &reference) {
    return reference.schemaIdentity ==
               ::loom::mapping::mappingArtifactSchema.identity &&
           reference.schemaVersion ==
               ::loom::mapping::mappingArtifactSchema.version;
  };
  for (auto indexedScenario : llvm::enumerate(witness.scenarios)) {
    const std::size_t scenarioOrdinal = indexedScenario.index();
    const ResourceTimeScheduleScenario &scenario = indexedScenario.value();
    if (scenario.executions.empty())
      return invalid("resource-time schedule scenario has no executions");
    if (scenario.states.empty())
      return invalid("resource-time schedule scenario has no event state");

    const auto findExecution = [&](::dataflow::RootThreadLaunchRef region)
        -> const ResourceTimeRegionExecution * {
      for (const ResourceTimeRegionExecution &execution : scenario.executions)
        if (execution.region == region)
          return &execution;
      return nullptr;
    };
    for (std::size_t index = 0; index != scenario.executions.size(); ++index) {
      const ResourceTimeRegionExecution &execution = scenario.executions[index];
      if (!hasRegion(execution.region))
        return invalid("resource-time execution references a foreign region");
      if (execution.readyPicoseconds > execution.startPicoseconds ||
          execution.startPicoseconds >= execution.completionPicoseconds)
        return invalid(
            "resource-time execution interval is not nonempty and ordered");
      for (const ResourceTimeRegionPrerequisite &prerequisite :
           execution.prerequisites) {
        if (!hasRegion(prerequisite.region) ||
            prerequisite.region == execution.region)
          return invalid("resource-time execution has an invalid prerequisite");
        const ResourceTimeRegionExecution *producer =
            findExecution(prerequisite.region);
        if (!producer)
          return invalid("resource-time prerequisite has no execution");
        if (prerequisite.readiness == ResourceTimeReadinessKind::Completion &&
            producer->completionPicoseconds > execution.readyPicoseconds)
          return invalid("completion-gated resource-time execution starts "
                         "before its prerequisite completes");
      }
      for (std::size_t prior = 0; prior != index; ++prior)
        if (scenario.executions[prior].region == execution.region)
          return invalid("resource-time scenario contains a duplicate "
                         "execution region");
    }

    std::vector<ArtifactRootReference> mappingReferences;
    mappingReferences.reserve(scenario.states.size() * 2);
    std::uint64_t maximumCompletion = 0;
    for (const ResourceTimeRegionExecution &execution : scenario.executions)
      maximumCompletion =
          std::max(maximumCompletion, execution.completionPicoseconds);

    std::optional<std::uint64_t> previousTime;
    std::vector<::dataflow::RootThreadLaunchRef> orderedActive;
    bool orderedSawAdmission = false;
    for (std::size_t stateOrdinal = 0; stateOrdinal != scenario.states.size();
         ++stateOrdinal) {
      const ResourceTimeScheduleState &state = scenario.states[stateOrdinal];
      if (!mappingReference(state.mapping))
        return invalid("resource-time state has a non-Mapping reference");
      if (previousTime && *previousTime > state.timePicoseconds)
        return invalid("resource-time state times are not monotonic");
      previousTime = state.timePicoseconds;
      if (!llvm::is_contained(mappingReferences, state.mapping))
        mappingReferences.push_back(state.mapping);
      if (state.active.size() > witness.maximumConcurrentRegions)
        return invalid("resource-time state exceeds the concurrency bound");

      const ResourceTimeRegionExecution *boundaryExecution = nullptr;
      bool boundaryIsStart = false;
      for (const ResourceTimeRegionExecution &execution : scenario.executions) {
        const bool isStart =
            execution.startPicoseconds == state.timePicoseconds &&
            state.event ==
                ::dataflow::rootThreadStartEventFamily(execution.region);
        const bool isCompletion =
            execution.completionPicoseconds == state.timePicoseconds &&
            state.event ==
                ::dataflow::rootThreadCompletionEventFamily(execution.region);
        if (!isStart && !isCompletion)
          continue;
        if (boundaryExecution)
          return invalid("resource-time state event matches multiple "
                         "execution boundaries");
        boundaryExecution = &execution;
        boundaryIsStart = isStart;
      }
      if (!boundaryExecution)
        return invalid("resource-time state event is not a start or completion "
                       "boundary at its timestamp");

      std::vector<::dataflow::RootThreadLaunchRef> expectedActive;
      const bool orderedTimestamp =
          (stateOrdinal != 0 &&
           scenario.states[stateOrdinal - 1].timePicoseconds ==
               state.timePicoseconds) ||
          (stateOrdinal + 1 != scenario.states.size() &&
           scenario.states[stateOrdinal + 1].timePicoseconds ==
               state.timePicoseconds);
      const bool firstAtTimestamp =
          stateOrdinal == 0 ||
          scenario.states[stateOrdinal - 1].timePicoseconds !=
              state.timePicoseconds;
      if (orderedTimestamp) {
        if (firstAtTimestamp) {
          orderedActive.clear();
          orderedSawAdmission = false;
          for (const ResourceTimeRegionExecution &execution :
               scenario.executions)
            if (execution.startPicoseconds < state.timePicoseconds &&
                state.timePicoseconds <= execution.completionPicoseconds)
              orderedActive.push_back(execution.region);
        }
        if (boundaryIsStart) {
          if (llvm::is_contained(orderedActive, boundaryExecution->region))
            return invalid("resource-time ordered boundary starts an active "
                           "region");
          orderedActive.push_back(boundaryExecution->region);
          orderedSawAdmission = true;
        } else {
          if (orderedSawAdmission)
            return invalid("resource-time ordered timestamp completes a "
                           "region after same-time admission");
          auto active = llvm::find(orderedActive, boundaryExecution->region);
          if (active == orderedActive.end())
            return invalid("resource-time ordered boundary completes an "
                           "inactive region");
          orderedActive.erase(active);
        }
        expectedActive = orderedActive;
      } else {
        for (const ResourceTimeRegionExecution &execution :
             scenario.executions) {
          const bool active =
              execution.startPicoseconds <= state.timePicoseconds &&
              state.timePicoseconds < execution.completionPicoseconds;
          if (active)
            expectedActive.push_back(execution.region);
        }
      }
      std::vector<::dataflow::RootThreadLaunchRef> observedActive;
      std::vector<::loom::fabric::FabricPhysicalOccurrenceOwnerRef>
          usedResources;
      for (const ResourceTimeRegionAllocation &allocation : state.active) {
        if (!hasRegion(allocation.region))
          return invalid("resource-time state contains a foreign region");
        if (!findExecution(allocation.region))
          return invalid("resource-time state has no matching execution");
        if (allocation.resources.empty())
          return invalid("resource-time state has an unallocated region");
        if (llvm::is_contained(observedActive, allocation.region))
          return invalid("resource-time state contains a duplicate region");
        observedActive.push_back(allocation.region);
        for (const auto &resource : allocation.resources) {
          if (llvm::is_contained(usedResources, resource))
            return invalid("resource-time state assigns one physical resource "
                           "to multiple regions");
          usedResources.push_back(resource);
        }
      }
      llvm::sort(expectedActive, rootLess);
      llvm::sort(observedActive, rootLess);
      if (expectedActive != observedActive)
        return invalid("resource-time state active set disagrees with "
                       "execution intervals in scenario " +
                       llvm::Twine(scenarioOrdinal) + " at time " +
                       llvm::Twine(state.timePicoseconds) + " (expected " +
                       llvm::Twine(expectedActive.size()) + ", observed " +
                       llvm::Twine(observedActive.size()) + ")");
      const bool lastAtTimestamp =
          stateOrdinal + 1 == scenario.states.size() ||
          scenario.states[stateOrdinal + 1].timePicoseconds !=
              state.timePicoseconds;
      if (lastAtTimestamp && orderedTimestamp) {
        std::vector<::dataflow::RootThreadLaunchRef> rightOpenActive;
        for (const ResourceTimeRegionExecution &execution : scenario.executions)
          if (execution.startPicoseconds <= state.timePicoseconds &&
              state.timePicoseconds < execution.completionPicoseconds)
            rightOpenActive.push_back(execution.region);
        auto orderedFinal = orderedActive;
        llvm::sort(orderedFinal, rootLess);
        llvm::sort(rightOpenActive, rootLess);
        if (orderedFinal != rightOpenActive)
          return invalid("resource-time ordered timestamp does not close to "
                         "the right-open execution state");
      }
    }
    for (const ResourceTimeRegionExecution &execution : scenario.executions) {
      const std::uint64_t starts = llvm::count_if(
          scenario.states, [&](const ResourceTimeScheduleState &state) {
            return state.timePicoseconds == execution.startPicoseconds &&
                   state.event ==
                       ::dataflow::rootThreadStartEventFamily(execution.region);
          });
      const std::uint64_t completions = llvm::count_if(
          scenario.states, [&](const ResourceTimeScheduleState &state) {
            return state.timePicoseconds == execution.completionPicoseconds &&
                   state.event == ::dataflow::rootThreadCompletionEventFamily(
                                      execution.region);
          });
      if (starts != 1 || completions != 1)
        return invalid("resource-time schedule omits or repeats an execution "
                       "boundary");
    }
    if (scenario.makespanPicoseconds < maximumCompletion)
      return invalid("resource-time makespan precedes execution completion");

    if (mappingReferences.size() > 1 &&
        scenario.transitions.transitions.empty())
      return invalid("resource-time mapping change has no transition sequence");
    if (llvm::Error error =
            validateResourceTimeTransitionSequence(scenario.transitions))
      return error;
    std::vector<std::pair<const ResourceTimeScheduleState *,
                          const ResourceTimeScheduleState *>>
        mappingChanges;
    for (std::size_t index = 1; index != scenario.states.size(); ++index)
      if (scenario.states[index - 1].mapping != scenario.states[index].mapping)
        mappingChanges.emplace_back(&scenario.states[index - 1],
                                    &scenario.states[index]);
    if (mappingChanges.size() != scenario.transitions.transitions.size())
      return invalid("resource-time Mapping changes do not match the finite "
                     "transition sequence");
    for (auto paired :
         llvm::zip(mappingChanges, scenario.transitions.transitions)) {
      const auto &change = std::get<0>(paired);
      const ResourceTimeTransition &transition = std::get<1>(paired);
      if (!llvm::is_contained(mappingReferences, transition.parent.mapping) ||
          !llvm::is_contained(mappingReferences, transition.child.mapping))
        return invalid("resource-time transition is absent from its schedule "
                       "states");
      if (transition.parent.mapping != change.first->mapping ||
          transition.child.mapping != change.second->mapping)
        return invalid("resource-time transition endpoints disagree with "
                       "their adjacent schedule states");
      if (transition.trigger != change.second->event)
        return invalid("resource-time transition trigger disagrees with its "
                       "child schedule event");
      if (!allocationsEquivalent(transition.beforeActive,
                                 change.first->active) ||
          !allocationsEquivalent(transition.afterActive, change.second->active))
        return invalid("resource-time transition allocation evidence "
                       "disagrees with its adjacent schedule states");
    }
  }
  return llvm::Error::success();
}

llvm::Expected<SystemMappingMigrationConePartition>
projectSystemMappingMigrationConePartition(
    const ::loom::mapping::SystemMappingView &mapping,
    llvm::ArrayRef<::dataflow::RootThreadLaunchRef> reopenedRoots,
    const ArtifactStore &store) {
  SystemMappingMigrationConePartition result;
  result.reopenedRoots.assign(reopenedRoots.begin(), reopenedRoots.end());
  llvm::sort(result.reopenedRoots, rootLess);
  if (std::adjacent_find(result.reopenedRoots.begin(),
                         result.reopenedRoots.end()) !=
      result.reopenedRoots.end())
    return invalid("Mapping cone repeats a reopened root");

  const ::loom::mapping::SystemExecutionBindingView &execution =
      mapping.executionBindings();
  for (const ::dataflow::RootThreadLaunchRef root : result.reopenedRoots) {
    if (root.artifact != mapping.dataflowIdentity() ||
        !llvm::is_contained(execution.rootThreadLaunches(), root))
      return invalid("Mapping cone names a foreign reopened root");
  }

  auto dataflowArtifact = ::dataflow::importCanonicalDataflow(
      {::dataflow::canonicalDataflowSchema.identity.str(),
       ::dataflow::canonicalDataflowSchema.version, mapping.dataflowIdentity()},
      store);
  if (!dataflowArtifact)
    return dataflowArtifact.takeError();
  auto dataflow = dataflowArtifact->view();
  if (!dataflow)
    return dataflow.takeError();
  std::vector<::dataflow::GraphRef> reopenedGraphCandidates;
  llvm::Error graphError = llvm::Error::success();
  dataflow->forEachRootedGraphLaunch(
      [&](::dataflow::RootedGraphLaunchRef launch) {
        if (graphError)
          return;
        auto graph = dataflow->resolve(launch);
        if (!graph) {
          graphError = graph.takeError();
          return;
        }
        std::vector<::dataflow::GraphRef> &partition =
            llvm::is_contained(result.reopenedRoots, launch.rootThreadLaunch)
                ? reopenedGraphCandidates
                : result.preservedGraphs;
        if (!llvm::is_contained(partition, *graph))
          partition.push_back(*graph);
      });
  if (graphError)
    return std::move(graphError);
  llvm::sort(result.preservedGraphs, graphLess);
  llvm::sort(reopenedGraphCandidates, graphLess);
  for (const ::dataflow::GraphRef graph : reopenedGraphCandidates)
    if (!llvm::is_contained(result.preservedGraphs, graph))
      result.reopenedGraphs.push_back(graph);

  for (const ::loom::mapping::SystemThreadExecutionBindingView &binding :
       execution.threadBindings()) {
    if (llvm::is_contained(result.reopenedRoots, binding.key))
      ++result.reopenedThreadBindings;
    else
      ++result.preservedThreadBindings;
  }

  std::vector<ArtifactRootReference> preservedSpatialTargets;
  std::vector<ArtifactRootReference> reopenedSpatialTargets;
  const auto appendTargets = [](const auto &binding,
                                std::vector<ArtifactRootReference> &targets) {
    for (const auto &clause : binding.clauses)
      targets.push_back(clause.target);
    if (binding.defaultTarget)
      targets.push_back(*binding.defaultTarget);
    for (const auto &entry : binding.stableKeyEntries)
      targets.push_back(entry.target);
  };
  for (const ::loom::mapping::SystemGraphExecutionBindingView &binding :
       execution.graphBindings()) {
    if (llvm::is_contained(result.reopenedRoots,
                           binding.key.rootThreadLaunch)) {
      ++result.reopenedGraphBindings;
      appendTargets(binding, reopenedSpatialTargets);
    } else {
      ++result.preservedGraphBindings;
      appendTargets(binding, preservedSpatialTargets);
    }
  }
  const auto canonicalize = [](std::vector<ArtifactRootReference> &roots) {
    llvm::sort(roots, artifactRootReferenceLess);
    roots.erase(std::unique(roots.begin(), roots.end()), roots.end());
  };
  canonicalize(preservedSpatialTargets);
  canonicalize(reopenedSpatialTargets);

  std::vector<ArtifactRootReference> selectedSpatialMappings =
      preservedSpatialTargets;
  selectedSpatialMappings.insert(selectedSpatialMappings.end(),
                                 reopenedSpatialTargets.begin(),
                                 reopenedSpatialTargets.end());
  canonicalize(selectedSpatialMappings);
  std::vector<ArtifactRootReference> importedSpatialMappings(
      execution.spatialMappingImports().begin(),
      execution.spatialMappingImports().end());
  canonicalize(importedSpatialMappings);
  if (selectedSpatialMappings != importedSpatialMappings)
    return invalid("Mapping cone does not cover the exact SpatialMapping "
                   "import range");

  result.preservedSpatialMappings = std::move(preservedSpatialTargets);
  for (const ArtifactRootReference &mappingReference : reopenedSpatialTargets)
    if (!llvm::is_contained(result.preservedSpatialMappings,
                            mappingReference))
      result.reopenedSpatialMappings.push_back(mappingReference);

  const auto appendTechMappings =
      [&](llvm::ArrayRef<ArtifactRootReference> spatialMappings,
          std::vector<ArtifactRootReference> &techMappings) -> llvm::Error {
    for (const ArtifactRootReference &spatialReference : spatialMappings) {
      auto spatial =
          ::loom::mapping::importSpatialMapping(spatialReference, store);
      if (!spatial)
        return spatial.takeError();
      if (spatial->view().dataflowIdentity() != mapping.dataflowIdentity())
        return invalid("Mapping cone contains a foreign SpatialMapping");
      techMappings.push_back(
          {::loom::mapping::mappingArtifactSchema.identity.str(),
           ::loom::mapping::mappingArtifactSchema.version,
           spatial->view().techMappingIdentity()});
    }
    canonicalize(techMappings);
    return llvm::Error::success();
  };
  if (llvm::Error error =
          appendTechMappings(result.preservedSpatialMappings,
                             result.preservedTechMappings))
    return std::move(error);
  std::vector<ArtifactRootReference> reopenedTechMappings;
  if (llvm::Error error = appendTechMappings(result.reopenedSpatialMappings,
                                             reopenedTechMappings))
    return std::move(error);
  for (const ArtifactRootReference &techMapping : reopenedTechMappings)
    if (!llvm::is_contained(result.preservedTechMappings, techMapping))
      result.reopenedTechMappings.push_back(techMapping);
  return result;
}

bool SystemMappingMigrationConePartition::admitsReplacementGraphs(
    llvm::ArrayRef<::dataflow::GraphRef> coveredGraphs) const {
  return !coveredGraphs.empty() &&
         llvm::all_of(coveredGraphs, [&](::dataflow::GraphRef graph) {
           return llvm::is_contained(reopenedGraphs, graph);
         });
}

static llvm::Error validateFinalizedMigrationSeedRelations(
    const ::loom::mapping::FinalizedSystemMapping &parentMapping,
    const SystemExecutionBindingCorrespondence &correspondence,
    const SystemMappingMigrationContext &context,
    llvm::ArrayRef<::dataflow::RootThreadLaunchRef> reopenedRoots,
    const ArtifactStore &store) {
  if (parentMapping.view().fabricIdentity() !=
      correspondence.parentSystem().artifact)
    return invalid("parent Mapping and correspondence have different Systems");
  if (llvm::Error error = validateCorrespondence(correspondence, store))
    return error;

  auto constraints = ::loom::mapping::importSystemMappingConstraintSet(
      context.childConstraints(), store);
  if (!constraints)
    return constraints.takeError();
  if (constraints->view().dataflowIdentity() !=
          parentMapping.view().dataflowIdentity() ||
      constraints->view().fabricIdentity() !=
          correspondence.childSystem().artifact)
    return invalid("migration context constraints bind foreign owners");

  if (!llvm::is_sorted(reopenedRoots, rootLess) ||
      std::adjacent_find(reopenedRoots.begin(), reopenedRoots.end()) !=
          reopenedRoots.end())
    return invalid("schedule migration roots are not canonical");
  for (const auto root : reopenedRoots) {
    if (root.artifact != parentMapping.view().dataflowIdentity() ||
        !llvm::is_contained(
            parentMapping.view().executionBindings().rootThreadLaunches(),
            root) ||
        !llvm::is_contained(constraints->view().rootThreadLaunches(), root))
      return invalid("schedule migration names a foreign invalidation root");
  }

  auto cone = projectSystemMappingMigrationConePartition(parentMapping.view(),
                                                         reopenedRoots, store);
  if (!cone)
    return cone.takeError();
  if (!llvm::all_of(cone->preservedSpatialMappings, [&](const auto &mapping) {
        return llvm::is_contained(context.spatialMappings(), mapping);
      }))
    return invalid("migration context omits a preserved-cone SpatialMapping");

  for (const ArtifactRootReference &mappingReference :
       context.spatialMappings()) {
    auto mapping =
        ::loom::mapping::importSpatialMapping(mappingReference, store);
    if (!mapping)
      return mapping.takeError();
    if (mapping->view().dataflowIdentity() !=
        parentMapping.view().dataflowIdentity())
      return invalid("migration SpatialMapping binds a foreign Dataflow");
    if (llvm::is_contained(cone->preservedSpatialMappings, mappingReference))
      continue;
    auto tech = ::loom::mapping::importTechMapping(
        {::loom::mapping::mappingArtifactSchema.identity.str(),
         ::loom::mapping::mappingArtifactSchema.version,
         mapping->view().techMappingIdentity()},
        store);
    if (!tech)
      return tech.takeError();
    if (!cone->admitsReplacementGraphs(tech->view().covers()))
      return invalid("migration context replaces a SpatialMapping outside "
                     "the reopened graph cone");
  }
  return llvm::Error::success();
}

llvm::Expected<SystemMappingMigrationContext>
SystemMappingMigrationContext::get(
    ArtifactRootReference childConstraints,
    std::vector<ArtifactRootReference> spatialMappings,
    ComponentViewDigest resolvedPnrConfigDigest) {
  if (childConstraints.schemaIdentity !=
          ::loom::mapping::mappingConstraintSetSchema.identity ||
      childConstraints.schemaVersion !=
          ::loom::mapping::mappingConstraintSetSchema.version)
    return invalid("migration context has a non-System constraint root");
  if (spatialMappings.empty())
    return invalid("migration context has no SpatialMapping frontier");
  for (const ArtifactRootReference &mapping : spatialMappings)
    if (mapping.schemaIdentity !=
            ::loom::mapping::mappingArtifactSchema.identity ||
        mapping.schemaVersion != ::loom::mapping::mappingArtifactSchema.version)
      return invalid("migration context has a non-Mapping frontier member");
  llvm::sort(spatialMappings, artifactRootReferenceLess);
  if (std::adjacent_find(spatialMappings.begin(), spatialMappings.end()) !=
      spatialMappings.end())
    return invalid("migration context has duplicate SpatialMappings");
  return SystemMappingMigrationContext(std::move(childConstraints),
                                       std::move(spatialMappings),
                                       std::move(resolvedPnrConfigDigest));
}

llvm::Expected<SystemExecutionBindingCorrespondence>
SystemExecutionBindingCorrespondence::get(
    ArtifactRootReference parentSystem, ArtifactRootReference childSystem,
    std::vector<::loom::fabric::FabricSystemEntityCorrespondence> entities,
    std::vector<::loom::fabric::FabricSystemTransferPatternCorrespondence>
        transferPatterns,
    std::vector<SystemModuleCorrespondence> modules,
    const ArtifactStore &store) {
  if (parentSystem.schemaIdentity !=
          ::loom::fabric::fabricArtifactSchema.identity ||
      parentSystem.schemaVersion !=
          ::loom::fabric::fabricArtifactSchema.version ||
      childSystem.schemaIdentity !=
          ::loom::fabric::fabricArtifactSchema.identity ||
      childSystem.schemaVersion != ::loom::fabric::fabricArtifactSchema.version)
    return invalid("parent or child is not an exact Fabric root");
  const bool identity = parentSystem == childSystem;
  if (entities.empty())
    return invalid("System entity correspondence is empty");
  llvm::sort(entities, [](const auto &lhs, const auto &rhs) {
    return std::tie(lhs.source.kind, lhs.source.id) <
           std::tie(rhs.source.kind, rhs.source.id);
  });
  std::set<std::pair<::loom::fabric::FabricEntityKind,
                     ::loom::fabric::FabricEntityId>>
      childEntities;
  for (std::size_t index = 0; index < entities.size(); ++index) {
    const auto &entry = entities[index];
    if (entry.source.kind != entry.target.kind)
      return invalid("System entity correspondence changes an entity kind");
    if (identity && entry.source != entry.target)
      return invalid("identity System correspondence changes an entity");
    if (index != 0 && entities[index - 1].source == entry.source)
      return invalid("System entity correspondence repeats a parent");
    if (!childEntities.insert({entry.target.kind, entry.target.id}).second)
      return invalid("System entity correspondence repeats a child");
  }
  llvm::sort(transferPatterns, [](const auto &lhs, const auto &rhs) {
    return ::loom::fabric::canonicalFabricBytes(lhs.source) <
           ::loom::fabric::canonicalFabricBytes(rhs.source);
  });
  std::set<std::vector<std::uint8_t>> childPatterns;
  for (std::size_t index = 0; index < transferPatterns.size(); ++index) {
    if (index != 0 &&
        transferPatterns[index - 1].source == transferPatterns[index].source)
      return invalid("transfer-pattern correspondence repeats a parent");
    if (!childPatterns
             .insert(::loom::fabric::canonicalFabricBytes(
                 transferPatterns[index].target))
             .second)
      return invalid("transfer-pattern correspondence repeats a child");
    if (identity &&
        transferPatterns[index].source != transferPatterns[index].target)
      return invalid("identity System correspondence changes a transfer "
                     "pattern");
  }
  std::vector<SystemAccCoreCorrespondence> accCores;
  for (const auto &entry : entities)
    if (entry.source.kind ==
        ::loom::fabric::FabricEntityKind::AccCoreOccurrence)
      accCores.push_back(
          {::loom::fabric::AccCoreOccurrenceRef(entry.source.id),
           ::loom::fabric::AccCoreOccurrenceRef(entry.target.id)});
  if (accCores.empty())
    return invalid("System correspondence has no preserved AccCore");
  llvm::sort(modules, [](const auto &lhs, const auto &rhs) {
    return artifactRootReferenceLess(lhs.parent, rhs.parent);
  });
  std::set<ArtifactIdentity::Storage> childModules;
  for (std::size_t index = 0; index < modules.size(); ++index) {
    const SystemModuleCorrespondence &entry = modules[index];
    if (entry.parent.schemaIdentity !=
            ::loom::fabric::fabricArtifactSchema.identity ||
        entry.parent.schemaVersion !=
            ::loom::fabric::fabricArtifactSchema.version ||
        entry.child.schemaIdentity !=
            ::loom::fabric::fabricArtifactSchema.identity ||
        entry.child.schemaVersion !=
            ::loom::fabric::fabricArtifactSchema.version)
      return invalid("Module correspondence has a foreign schema");
    if (index != 0 && modules[index - 1].parent == entry.parent)
      return invalid("Module correspondence repeats a parent");
    if (!childModules.insert(entry.child.artifact.bytes()).second)
      return invalid("Module correspondence repeats a child");
    if (identity && entry.parent != entry.child)
      return invalid("identity System correspondence changes a Module");
    auto parentModule =
        ::loom::fabric::importEntireFabricRoot(entry.parent, store);
    if (!parentModule)
      return parentModule.takeError();
    auto childModule =
        ::loom::fabric::importEntireFabricRoot(entry.child, store);
    if (!childModule)
      return childModule.takeError();
    if (parentModule->view().rootKind() !=
            ::loom::fabric::FabricRootKind::Module ||
        childModule->view().rootKind() !=
            ::loom::fabric::FabricRootKind::Module)
      return invalid("Module correspondence names a non-Module root");
  }
  if (modules.empty())
    return invalid("System correspondence has no Module lineage");
  SystemExecutionBindingCorrespondence result(
      std::move(parentSystem), std::move(childSystem), std::move(entities),
      std::move(transferPatterns), std::move(modules), std::move(accCores));
  if (llvm::Error error = validateCorrespondence(result, store))
    return std::move(error);
  return result;
}

llvm::Expected<SystemExecutionBindingCorrespondence>
SystemExecutionBindingCorrespondence::getIdentity(
    const ArtifactRootReference &systemReference, const ArtifactStore &store) {
  auto root = ::loom::fabric::importEntireFabricRoot(systemReference, store);
  if (!root)
    return root.takeError();
  auto system = ::loom::fabric::requireSystemRoot(root->view());
  if (!system)
    return system.takeError();

  std::vector<::loom::fabric::FabricSystemEntityCorrespondence> entities;
  for (::loom::fabric::FabricEntityId id = 0;; ++id) {
    const auto kind = system->artifact().entityKind(id);
    if (!kind)
      break;
    const ::loom::fabric::FabricSystemEntityReference reference{*kind, id};
    entities.push_back({reference, reference});
  }

  std::vector<::loom::fabric::FabricSystemTransferPatternCorrespondence>
      transferPatterns;
  for (const auto resource : system->transportResources())
    for (const auto pattern : system->transferPatterns(resource))
      transferPatterns.push_back({pattern, pattern});

  std::vector<SystemModuleCorrespondence> modules;
  for (const auto core : system->artifact().accCoreOccurrences()) {
    auto module = targetModule(*root, *system, core);
    if (!module)
      return module.takeError();
    modules.push_back({*module, *module});
  }
  llvm::sort(modules, [](const auto &lhs, const auto &rhs) {
    return artifactRootReferenceLess(lhs.parent, rhs.parent);
  });
  modules.erase(std::unique(modules.begin(), modules.end()), modules.end());
  return get(systemReference, systemReference, std::move(entities),
             std::move(transferPatterns), std::move(modules), store);
}

llvm::Expected<SystemExecutionBindingCorrespondence>
composeSystemExecutionBindingCorrespondence(
    const SystemExecutionBindingCorrespondence &first,
    const SystemExecutionBindingCorrespondence &second,
    const ArtifactStore &store) {
  if (first.childSystem() != second.parentSystem())
    return invalid("System correspondence composition is not consecutive");
  std::vector<::loom::fabric::FabricSystemEntityCorrespondence> entities;
  for (const auto &entry : first.entities()) {
    const auto next = llvm::find_if(second.entities(), [&](const auto &value) {
      return value.source == entry.target;
    });
    if (next != second.entities().end())
      entities.push_back({entry.source, next->target});
  }
  std::vector<::loom::fabric::FabricSystemTransferPatternCorrespondence>
      transferPatterns;
  for (const auto &entry : first.transferPatterns()) {
    const auto next =
        llvm::find_if(second.transferPatterns(), [&](const auto &value) {
          return value.source == entry.target;
        });
    if (next != second.transferPatterns().end())
      transferPatterns.push_back({entry.source, next->target});
  }
  std::vector<SystemModuleCorrespondence> modules;
  for (const SystemModuleCorrespondence &entry : first.modules()) {
    const auto next = llvm::find_if(second.modules(), [&](const auto &value) {
      return value.parent == entry.child;
    });
    if (next != second.modules().end())
      modules.push_back({entry.parent, next->child});
  }
  return SystemExecutionBindingCorrespondence::get(
      first.parentSystem(), second.childSystem(), std::move(entities),
      std::move(transferPatterns), std::move(modules), store);
}

llvm::Expected<FinalizedSystemMappingMigrationSeed>
finalizeSystemMappingMigrationSeed(
    const ArtifactRootReference &parentMappingReference,
    const SystemExecutionBindingCorrespondence &correspondence,
    const SystemMappingMigrationContext &context, const ArtifactStore &store) {
  return finalizeSystemMappingMigrationSeed(parentMappingReference,
                                            correspondence, context, {}, store);
}

llvm::Expected<FinalizedSystemMappingMigrationSeed>
finalizeSystemMappingMigrationSeed(
    const ArtifactRootReference &parentMappingReference,
    const SystemExecutionBindingCorrespondence &correspondence,
    const SystemMappingMigrationContext &context,
    llvm::ArrayRef<::dataflow::RootThreadLaunchRef> reopenedRoots,
    const ArtifactStore &store) {
  auto parentMapping =
      ::loom::mapping::importSystemMapping(parentMappingReference, store);
  if (!parentMapping)
    return parentMapping.takeError();
  std::vector<::dataflow::RootThreadLaunchRef> canonicalReopenedRoots(
      reopenedRoots.begin(), reopenedRoots.end());
  llvm::sort(canonicalReopenedRoots, rootLess);
  if (std::adjacent_find(canonicalReopenedRoots.begin(),
                         canonicalReopenedRoots.end()) !=
      canonicalReopenedRoots.end())
    return invalid("schedule migration repeats an invalidation root");
  if (llvm::Error error = validateFinalizedMigrationSeedRelations(
          *parentMapping, correspondence, context, canonicalReopenedRoots,
          store))
    return std::move(error);
  auto canonicalBytes = canonicalFinalizedSeedBytes(
      parentMappingReference, correspondence, context, canonicalReopenedRoots);
  if (!canonicalBytes)
    return canonicalBytes.takeError();
  CanonicalSemanticBytes canonical(std::move(*canonicalBytes));
  auto identity =
      store.put(systemMappingFinalizedMigrationSeedArtifactSchema, canonical);
  if (!identity)
    return identity.takeError();
  return importSystemMappingMigrationSeed(
      {systemMappingFinalizedMigrationSeedArtifactSchema.identity.str(),
       systemMappingFinalizedMigrationSeedArtifactSchema.version, *identity},
      store);
}

llvm::Expected<FinalizedSystemMappingMigrationSeed>
importSystemMappingMigrationSeed(const ArtifactRootReference &reference,
                                 const ArtifactStore &store) {
  if (reference.schemaIdentity !=
          systemMappingFinalizedMigrationSeedArtifactSchema.identity ||
      reference.schemaVersion !=
          systemMappingFinalizedMigrationSeedArtifactSchema.version)
    return invalid("finalized migration seed root has the wrong schema");
  auto stored = store.get(systemMappingFinalizedMigrationSeedArtifactSchema,
                          reference.artifact);
  if (!stored)
    return stored.takeError();
  llvm::ArrayRef<std::uint8_t> bytes = stored->bytes();
  std::size_t offset = 0;
  auto parentMappingReference = readRootReference(bytes, offset);
  if (!parentMappingReference)
    return parentMappingReference.takeError();
  auto childSystem = readRootReference(bytes, offset);
  if (!childSystem)
    return childSystem.takeError();
  auto context = readMigrationContext(bytes, offset);
  if (!context)
    return context.takeError();
  auto parentMapping =
      ::loom::mapping::importSystemMapping(*parentMappingReference, store);
  if (!parentMapping)
    return parentMapping.takeError();
  auto reopenedRootCount = readU64(bytes, offset);
  if (!reopenedRootCount)
    return reopenedRootCount.takeError();
  if (*reopenedRootCount > bytes.size())
    return invalid("schedule migration root count exceeds payload size");
  std::vector<::dataflow::RootThreadLaunchRef> reopenedRoots;
  reopenedRoots.reserve(static_cast<std::size_t>(*reopenedRootCount));
  for (std::uint64_t ordinal = 0; ordinal != *reopenedRootCount; ++ordinal) {
    auto rootBytes = readBlob(bytes, offset);
    if (!rootBytes)
      return rootBytes.takeError();
    auto root =
        ::dataflow::decodeDataflowReference<::dataflow::RootThreadLaunchRef>(
            *rootBytes, parentMapping->view().dataflowIdentity());
    if (!root)
      return root.takeError();
    reopenedRoots.push_back(*root);
  }
  if (!llvm::is_sorted(reopenedRoots, rootLess) ||
      std::adjacent_find(reopenedRoots.begin(), reopenedRoots.end()) !=
          reopenedRoots.end())
    return invalid("schedule migration roots are not canonical");
  auto decodedCorrespondence = readCorrespondence(bytes, offset);
  if (!decodedCorrespondence)
    return decodedCorrespondence.takeError();
  if (offset != bytes.size())
    return invalid("finalized migration seed payload has trailing bytes");
  ArtifactRootReference parentSystem{
      ::loom::fabric::fabricArtifactSchema.identity.str(),
      ::loom::fabric::fabricArtifactSchema.version,
      parentMapping->view().fabricIdentity()};
  auto correspondence = SystemExecutionBindingCorrespondence::get(
      std::move(parentSystem), std::move(*childSystem),
      std::move(decodedCorrespondence->entities),
      std::move(decodedCorrespondence->transferPatterns),
      std::move(decodedCorrespondence->modules), store);
  if (!correspondence)
    return correspondence.takeError();
  if (llvm::Error error = validateFinalizedMigrationSeedRelations(
          *parentMapping, *correspondence, *context, reopenedRoots, store))
    return std::move(error);
  auto canonical = canonicalFinalizedSeedBytes(
      *parentMappingReference, *correspondence, *context, reopenedRoots);
  if (!canonical)
    return canonical.takeError();
  if (llvm::ArrayRef<std::uint8_t>(*canonical) != bytes)
    return invalid("finalized migration seed payload is not canonical");
  return FinalizedSystemMappingMigrationSeed(
      reference, std::move(*parentMapping), std::move(*correspondence),
      std::move(*context), std::move(reopenedRoots));
}

llvm::Expected<FinalizedSystemMappingCheckpointMigrationSeed>
finalizeSystemMappingCheckpointMigrationSeed(
    const ArtifactRootReference &checkpointReference,
    const SystemExecutionBindingCorrespondence &correspondence,
    const SystemMappingMigrationContext &context,
    ::loom::fabric::AccCoreOccurrenceRef reopenedParentAccCore,
    const ArtifactStore &store) {
  auto checkpoint = ::loom::mapping::importSystemExecutionBindingCheckpoint(
      checkpointReference, store);
  if (!checkpoint)
    return checkpoint.takeError();
  if (checkpoint->system() != correspondence.parentSystem())
    return invalid("checkpoint and correspondence have different parents");
  if (checkpoint->resolvedPnrConfigDigest() !=
      context.resolvedPnrConfigDigest())
    return invalid("checkpoint and child migration use different PnR configs");
  if (checkpoint->incomplete().witnessAccCore != reopenedParentAccCore)
    return invalid("reopened AccCore is not the checkpoint capacity witness");
  if (llvm::Error error = validateCorrespondence(correspondence, store))
    return std::move(error);
  auto constraints = ::loom::mapping::importSystemMappingConstraintSet(
      context.childConstraints(), store);
  if (!constraints)
    return constraints.takeError();
  if (constraints->view().dataflowIdentity() !=
          checkpoint->dataflow().artifact ||
      constraints->view().fabricIdentity() !=
          correspondence.childSystem().artifact)
    return invalid("migration context constraints bind foreign owners");
  for (const ArtifactRootReference &mappingReference :
       context.spatialMappings()) {
    auto mapping =
        ::loom::mapping::importSpatialMapping(mappingReference, store);
    if (!mapping)
      return mapping.takeError();
    if (mapping->view().dataflowIdentity() != checkpoint->dataflow().artifact)
      return invalid("migration SpatialMapping binds a foreign Dataflow");
  }
  if (!llvm::any_of(correspondence.accCores(), [&](const auto &entry) {
        return entry.parent == reopenedParentAccCore;
      }))
    return invalid("reopened AccCore is absent from parent-child lineage");
  if (!llvm::any_of(checkpoint->threadBindings(), [&](const auto &binding) {
        return binding.target == reopenedParentAccCore;
      }))
    return invalid("reopened AccCore owns no checkpoint thread binding");
  CanonicalSemanticBytes canonical(canonicalSeedBytes(
      checkpointReference, correspondence, context, reopenedParentAccCore));
  auto identity =
      store.put(systemMappingCheckpointMigrationSeedArtifactSchema, canonical);
  if (!identity)
    return identity.takeError();
  ArtifactRootReference reference{
      systemMappingCheckpointMigrationSeedArtifactSchema.identity.str(),
      systemMappingCheckpointMigrationSeedArtifactSchema.version, *identity};
  return importSystemMappingCheckpointMigrationSeed(reference, store);
}

llvm::Expected<FinalizedSystemMappingCheckpointMigrationSeed>
importSystemMappingCheckpointMigrationSeed(
    const ArtifactRootReference &reference, const ArtifactStore &store) {
  if (reference.schemaIdentity !=
          systemMappingCheckpointMigrationSeedArtifactSchema.identity ||
      reference.schemaVersion !=
          systemMappingCheckpointMigrationSeedArtifactSchema.version)
    return invalid("migration seed root has the wrong schema");
  auto stored = store.get(systemMappingCheckpointMigrationSeedArtifactSchema,
                          reference.artifact);
  if (!stored)
    return stored.takeError();
  llvm::ArrayRef<std::uint8_t> bytes = stored->bytes();
  std::size_t offset = 0;
  auto checkpointReference = readRootReference(bytes, offset);
  if (!checkpointReference)
    return checkpointReference.takeError();
  auto childSystem = readRootReference(bytes, offset);
  if (!childSystem)
    return childSystem.takeError();
  auto context = readMigrationContext(bytes, offset);
  if (!context)
    return context.takeError();
  auto reopenedParentBytes = readBlob(bytes, offset);
  if (!reopenedParentBytes)
    return reopenedParentBytes.takeError();
  auto reopenedParentAccCore =
      ::loom::fabric::decodeFabricRef<::loom::fabric::AccCoreOccurrenceRef>(
          *reopenedParentBytes);
  if (!reopenedParentAccCore)
    return reopenedParentAccCore.takeError();
  auto decodedCorrespondence = readCorrespondence(bytes, offset);
  if (!decodedCorrespondence)
    return decodedCorrespondence.takeError();
  if (offset != bytes.size())
    return invalid("migration seed payload has trailing bytes");
  auto checkpoint = ::loom::mapping::importSystemExecutionBindingCheckpoint(
      *checkpointReference, store);
  if (!checkpoint)
    return checkpoint.takeError();
  auto correspondence = SystemExecutionBindingCorrespondence::get(
      checkpoint->system(), std::move(*childSystem),
      std::move(decodedCorrespondence->entities),
      std::move(decodedCorrespondence->transferPatterns),
      std::move(decodedCorrespondence->modules), store);
  if (!correspondence)
    return correspondence.takeError();
  if (!llvm::any_of(correspondence->accCores(), [&](const auto &entry) {
        return entry.parent == *reopenedParentAccCore;
      }))
    return invalid("reopened AccCore is absent from parent-child lineage");
  if (!llvm::any_of(checkpoint->threadBindings(), [&](const auto &binding) {
        return binding.target == *reopenedParentAccCore;
      }))
    return invalid("reopened AccCore owns no checkpoint thread binding");
  if (llvm::ArrayRef<std::uint8_t>(
          canonicalSeedBytes(*checkpointReference, *correspondence, *context,
                             *reopenedParentAccCore)) != bytes)
    return invalid("migration seed payload is not canonical");
  return FinalizedSystemMappingCheckpointMigrationSeed(
      reference, std::move(*checkpoint), std::move(*correspondence),
      std::move(*context), *reopenedParentAccCore);
}

SystemMappingMigrationProjectionOutcome projectSystemMappingMigrationSeed(
    const FinalizedSystemMappingMigrationSeed &seed,
    const FrozenSystemPnrProblem &childProblem) {
  const auto &mapping = seed.parentMapping().view();
  if (mapping.dataflowIdentity() != childProblem.dataflowIdentity())
    return SystemMappingMigrationFallback{
        SystemMappingMigrationFallbackReason::ParentMappingDataflowMismatch};
  if (mapping.fabricIdentity() != seed.correspondence().parentSystem().artifact)
    return SystemMappingMigrationFallback{
        SystemMappingMigrationFallbackReason::ParentMappingFabricMismatch};
  if (childProblem.fabricIdentity() !=
      seed.correspondence().childSystem().artifact)
    return SystemMappingMigrationFallback{
        SystemMappingMigrationFallbackReason::ChildFabricMismatch};

  const auto &execution = mapping.executionBindings();
  SystemMappingMigrationProjection result;
  result.fixedChoices.reserve(childProblem.threadDecisions().size() +
                              childProblem.graphDecisions().size());

  for (PnrIndex decision = 0; decision < childProblem.threadDecisions().size();
       ++decision) {
    const FrozenSystemThreadExecutionDecision &frozen =
        childProblem.threadDecisions()[decision];
    if (llvm::is_contained(seed.reopenedRoots(), frozen.root)) {
      result.fixedChoices.push_back(getInvalidPnrIndex());
      result.releasedChoices.push_back(decision);
      continue;
    }
    llvm::SmallVector<::loom::mapping::SystemThreadExecutionBindingView, 1>
        matching;
    bindingsForKey(execution.threadBindings(), frozen.root, matching);
    bool ambiguous = false;
    auto parentTarget = exactCellTarget<::loom::fabric::AccCoreOccurrenceRef>(
        llvm::ArrayRef(matching), frozen.cell, ambiguous);
    if (ambiguous)
      return SystemMappingMigrationFallback{
          SystemMappingMigrationFallbackReason::AmbiguousThreadBinding};
    if (!parentTarget)
      return SystemMappingMigrationFallback{
          SystemMappingMigrationFallbackReason::MissingThreadBinding};
    auto childTarget =
        mapAccCore(seed.correspondence(), *parentTarget, ambiguous);
    if (ambiguous || !childTarget)
      return SystemMappingMigrationFallback{
          SystemMappingMigrationFallbackReason::UnmatchedAccCore};
    const auto domain = childProblem.threadChoiceCatalogOrdinals(decision);
    auto choice = llvm::find_if(domain, [&](PnrIndex core) {
      return childProblem.accCores()[core] == *childTarget;
    });
    if (choice == domain.end())
      return SystemMappingMigrationFallback{
          SystemMappingMigrationFallbackReason::UnmatchedAccCore};
    result.fixedChoices.push_back(
        static_cast<PnrIndex>(choice - domain.begin()));
    ++result.preservedThreadBindings;
  }

  for (PnrIndex decision = 0; decision < childProblem.graphDecisions().size();
       ++decision) {
    const FrozenSystemGraphExecutionDecision &frozen =
        childProblem.graphDecisions()[decision];
    if (llvm::is_contained(seed.reopenedRoots(),
                           frozen.launch.rootThreadLaunch)) {
      result.fixedChoices.push_back(getInvalidPnrIndex());
      result.releasedChoices.push_back(childProblem.threadDecisions().size() +
                                       decision);
      continue;
    }
    llvm::SmallVector<::loom::mapping::SystemGraphExecutionBindingView, 1>
        matching;
    bindingsForKey(execution.graphBindings(), frozen.launch, matching);
    bool ambiguous = false;
    auto target = exactCellTarget<ArtifactRootReference>(
        llvm::ArrayRef(matching), frozen.cell, ambiguous);
    if (ambiguous)
      return SystemMappingMigrationFallback{
          SystemMappingMigrationFallbackReason::AmbiguousGraphBinding};
    if (!target)
      return SystemMappingMigrationFallback{
          SystemMappingMigrationFallbackReason::MissingGraphBinding};
    const auto domain = childProblem.graphChoiceCatalogOrdinals(decision);
    auto choice = llvm::find_if(domain, [&](PnrIndex mappingOrdinal) {
      return childProblem.spatialMappings()[mappingOrdinal] == *target;
    });
    if (choice == domain.end())
      return SystemMappingMigrationFallback{
          SystemMappingMigrationFallbackReason::UnmatchedSpatialMapping};
    result.fixedChoices.push_back(
        static_cast<PnrIndex>(choice - domain.begin()));
    ++result.preservedGraphBindings;
  }
  auto routeSeed =
      projectFinalizedRoutes(mapping, seed.correspondence(), childProblem);
  if (!routeSeed) {
    llvm::consumeError(routeSeed.takeError());
    return SystemMappingMigrationFallback{
        SystemMappingMigrationFallbackReason::ChildRebaseRejected};
  }
  for (PnrIndex leg = 0; leg != childProblem.serviceLegs().size(); ++leg) {
    const FrozenSystemServiceLeg &serviceLeg = childProblem.serviceLegs()[leg];
    if (serviceLeg.serviceContext >= childProblem.serviceContexts().size())
      return SystemMappingMigrationFallback{
          SystemMappingMigrationFallbackReason::ChildRebaseRejected};
    const FrozenSystemServiceContext &context =
        childProblem.serviceContexts()[serviceLeg.serviceContext];
    bool reopened = false;
    if (context.threadDecision != getInvalidPnrIndex()) {
      if (context.threadDecision >= childProblem.threadDecisions().size())
        return SystemMappingMigrationFallback{
            SystemMappingMigrationFallbackReason::ChildRebaseRejected};
      reopened |= llvm::is_contained(
          seed.reopenedRoots(),
          childProblem.threadDecisions()[context.threadDecision].root);
    }
    if (context.graphDecision != getInvalidPnrIndex()) {
      if (context.graphDecision >= childProblem.graphDecisions().size())
        return SystemMappingMigrationFallback{
            SystemMappingMigrationFallbackReason::ChildRebaseRejected};
      reopened |= llvm::is_contained(
          seed.reopenedRoots(),
          childProblem.graphDecisions()[context.graphDecision]
              .launch.rootThreadLaunch);
    }
    if (reopened && !llvm::is_contained(routeSeed->reroutedLegs, leg))
      routeSeed->reroutedLegs.push_back(leg);
  }
  llvm::sort(routeSeed->reroutedLegs);
  result.preservedServiceLegs =
      childProblem.serviceLegs().size() - routeSeed->reroutedLegs.size();
  result.reopenedServiceLegs = routeSeed->reroutedLegs.size();
  if (result.preservedServiceLegs != 0)
    result.routeSeed = std::move(*routeSeed);
  if (!seed.reopenedRoots().empty() && result.releasedChoices.empty())
    return SystemMappingMigrationFallback{
        SystemMappingMigrationFallbackReason::EmptyReopenScope};
  return result;
}

SystemMappingMigrationProjectionOutcome projectSystemMappingMigrationSeed(
    const FinalizedSystemMappingCheckpointMigrationSeed &seed,
    const FrozenSystemPnrProblem &childProblem) {
  const auto &checkpoint = seed.checkpoint();
  const auto &correspondence = seed.correspondence();
  if (checkpoint.dataflow().artifact != childProblem.dataflowIdentity())
    return SystemMappingMigrationFallback{
        SystemMappingMigrationFallbackReason::ParentMappingDataflowMismatch};
  if (checkpoint.system() != correspondence.parentSystem())
    return SystemMappingMigrationFallback{
        SystemMappingMigrationFallbackReason::ParentMappingFabricMismatch};
  if (childProblem.fabricIdentity() != correspondence.childSystem().artifact)
    return SystemMappingMigrationFallback{
        SystemMappingMigrationFallbackReason::ChildFabricMismatch};

  SystemMappingMigrationProjection result;
  result.fixedChoices.reserve(childProblem.threadDecisions().size() +
                              childProblem.graphDecisions().size());
  for (PnrIndex decision = 0; decision < childProblem.threadDecisions().size();
       ++decision) {
    const FrozenSystemThreadExecutionDecision &frozen =
        childProblem.threadDecisions()[decision];
    const ::loom::mapping::SystemThreadExecutionCheckpoint *selected = nullptr;
    for (const auto &binding : checkpoint.threadBindings()) {
      if (binding.root != frozen.root || binding.cell != frozen.cell)
        continue;
      if (selected)
        return SystemMappingMigrationFallback{
            SystemMappingMigrationFallbackReason::AmbiguousThreadBinding};
      selected = &binding;
    }
    if (!selected)
      return SystemMappingMigrationFallback{
          SystemMappingMigrationFallbackReason::MissingThreadBinding};
    bool ambiguous = false;
    auto childTarget = mapAccCore(correspondence, selected->target, ambiguous);
    if (ambiguous || !childTarget)
      return SystemMappingMigrationFallback{
          SystemMappingMigrationFallbackReason::UnmatchedAccCore};
    const auto domain = childProblem.threadChoiceCatalogOrdinals(decision);
    auto choice = llvm::find_if(domain, [&](PnrIndex core) {
      return childProblem.accCores()[core] == *childTarget;
    });
    if (choice == domain.end())
      return SystemMappingMigrationFallback{
          SystemMappingMigrationFallbackReason::UnmatchedAccCore};
    if (selected->target == seed.reopenedParentAccCore()) {
      result.fixedChoices.push_back(getInvalidPnrIndex());
      result.releasedChoices.push_back(decision);
    } else {
      result.fixedChoices.push_back(
          static_cast<PnrIndex>(choice - domain.begin()));
      ++result.preservedThreadBindings;
    }
  }
  if (result.releasedChoices.empty())
    return SystemMappingMigrationFallback{
        SystemMappingMigrationFallbackReason::EmptyReopenScope};
  for (PnrIndex decision = 0; decision < childProblem.graphDecisions().size();
       ++decision) {
    const FrozenSystemGraphExecutionDecision &frozen =
        childProblem.graphDecisions()[decision];
    const ::loom::mapping::SystemGraphExecutionCheckpoint *selected = nullptr;
    for (const auto &binding : checkpoint.graphBindings()) {
      if (binding.launch != frozen.launch || binding.cell != frozen.cell)
        continue;
      if (selected)
        return SystemMappingMigrationFallback{
            SystemMappingMigrationFallbackReason::AmbiguousGraphBinding};
      selected = &binding;
    }
    if (!selected)
      return SystemMappingMigrationFallback{
          SystemMappingMigrationFallbackReason::MissingGraphBinding};
    const auto domain = childProblem.graphChoiceCatalogOrdinals(decision);
    auto choice = llvm::find_if(domain, [&](PnrIndex mappingOrdinal) {
      return childProblem.spatialMappings()[mappingOrdinal] == selected->target;
    });
    if (choice == domain.end())
      return SystemMappingMigrationFallback{
          SystemMappingMigrationFallbackReason::UnmatchedSpatialMapping};
    result.fixedChoices.push_back(
        static_cast<PnrIndex>(choice - domain.begin()));
    ++result.preservedGraphBindings;
  }
  return result;
}

llvm::StringRef systemMappingMigrationFallbackReasonSpelling(
    SystemMappingMigrationFallbackReason reason) {
  switch (reason) {
  case SystemMappingMigrationFallbackReason::ParentMappingDataflowMismatch:
    return "parent_mapping_dataflow_mismatch";
  case SystemMappingMigrationFallbackReason::ParentMappingFabricMismatch:
    return "parent_mapping_fabric_mismatch";
  case SystemMappingMigrationFallbackReason::ChildFabricMismatch:
    return "child_fabric_mismatch";
  case SystemMappingMigrationFallbackReason::MissingThreadBinding:
    return "missing_thread_binding";
  case SystemMappingMigrationFallbackReason::AmbiguousThreadBinding:
    return "ambiguous_thread_binding";
  case SystemMappingMigrationFallbackReason::UnmatchedAccCore:
    return "unmatched_acc_core";
  case SystemMappingMigrationFallbackReason::MissingGraphBinding:
    return "missing_graph_binding";
  case SystemMappingMigrationFallbackReason::AmbiguousGraphBinding:
    return "ambiguous_graph_binding";
  case SystemMappingMigrationFallbackReason::UnmatchedSpatialMapping:
    return "unmatched_spatial_mapping";
  case SystemMappingMigrationFallbackReason::EmptyReopenScope:
    return "empty_reopen_scope";
  case SystemMappingMigrationFallbackReason::ChildRebaseRejected:
    return "child_rebase_rejected";
  case SystemMappingMigrationFallbackReason::ChildInitializerRejected:
    return "child_initializer_rejected";
  }
  llvm_unreachable("unknown SystemMapping migration fallback reason");
}

} // namespace loom::pnr
