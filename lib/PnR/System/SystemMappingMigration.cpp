#include "PnR/System/SystemMappingMigration.h"

#include "Common/ArtifactLocalReference.h"
#include "Common/ArtifactStore.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/Artifact/FabricSystemRootView.h"
#include "Fabric/Identity/FabricRefBytes.h"
#include "PnR/System/SystemPnrProblem.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <limits>
#include <optional>
#include <system_error>

namespace loom::pnr {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      "system_mapping_migration_invalid: " + message);
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

std::vector<std::uint8_t>
canonicalSeedBytes(const ArtifactRootReference &checkpoint,
                   const SystemExecutionBindingCorrespondence &correspondence,
                   ::loom::fabric::AccCoreOccurrenceRef reopenedParentAccCore) {
  std::vector<std::uint8_t> bytes = encodeArtifactRootReference(checkpoint);
  const auto child = encodeArtifactRootReference(correspondence.childSystem());
  bytes.insert(bytes.end(), child.begin(), child.end());
  appendBlob(bytes,
             ::loom::fabric::canonicalFabricBytes(reopenedParentAccCore));
  appendU64(bytes, correspondence.accCores().size());
  for (const SystemAccCoreCorrespondence &entry : correspondence.accCores()) {
    appendBlob(bytes, ::loom::fabric::canonicalFabricBytes(entry.parent));
    appendBlob(bytes, ::loom::fabric::canonicalFabricBytes(entry.child));
  }
  return bytes;
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
  for (const SystemAccCoreCorrespondence &entry : correspondence.accCores()) {
    auto parentModule = targetModule(*parent, *parentView, entry.parent);
    if (!parentModule)
      return parentModule.takeError();
    auto childModule = targetModule(*child, *childView, entry.child);
    if (!childModule)
      return childModule.takeError();
    if (*parentModule != *childModule)
      return invalid("migration correspondence changes an AccCore Module");
  }
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

} // namespace

llvm::Expected<SystemExecutionBindingCorrespondence>
SystemExecutionBindingCorrespondence::get(
    ArtifactRootReference parentSystem, ArtifactRootReference childSystem,
    std::vector<SystemAccCoreCorrespondence> accCores,
    const ArtifactStore &store) {
  if (parentSystem.schemaIdentity !=
          ::loom::fabric::fabricArtifactSchema.identity ||
      parentSystem.schemaVersion !=
          ::loom::fabric::fabricArtifactSchema.version ||
      childSystem.schemaIdentity !=
          ::loom::fabric::fabricArtifactSchema.identity ||
      childSystem.schemaVersion != ::loom::fabric::fabricArtifactSchema.version)
    return invalid("parent or child is not an exact Fabric root");
  if (parentSystem == childSystem)
    return invalid("parent and child System identities are equal");
  if (accCores.empty())
    return invalid("AccCore correspondence is empty");
  llvm::sort(accCores, [](const auto &lhs, const auto &rhs) {
    if (lhs.parent != rhs.parent)
      return lhs.parent.id() < rhs.parent.id();
    return lhs.child.id() < rhs.child.id();
  });
  std::vector<::loom::fabric::FabricEntityId> childIds;
  childIds.reserve(accCores.size());
  for (const SystemAccCoreCorrespondence &entry : accCores)
    childIds.push_back(entry.child.id());
  llvm::sort(childIds);
  for (std::size_t index = 1; index < accCores.size(); ++index)
    if (accCores[index - 1].parent == accCores[index].parent ||
        childIds[index - 1] == childIds[index])
      return invalid("AccCore correspondence is not one-to-one");
  SystemExecutionBindingCorrespondence result(
      std::move(parentSystem), std::move(childSystem), std::move(accCores));
  if (llvm::Error error = validateCorrespondence(result, store))
    return std::move(error);
  return result;
}

llvm::Expected<FinalizedSystemMappingCheckpointMigrationSeed>
finalizeSystemMappingCheckpointMigrationSeed(
    const ArtifactRootReference &checkpointReference,
    const SystemExecutionBindingCorrespondence &correspondence,
    ::loom::fabric::AccCoreOccurrenceRef reopenedParentAccCore,
    const ArtifactStore &store) {
  auto checkpoint = ::loom::mapping::importSystemExecutionBindingCheckpoint(
      checkpointReference, store);
  if (!checkpoint)
    return checkpoint.takeError();
  if (checkpoint->system() != correspondence.parentSystem())
    return invalid("checkpoint and correspondence have different parents");
  if (llvm::Error error = validateCorrespondence(correspondence, store))
    return std::move(error);
  if (!llvm::any_of(correspondence.accCores(), [&](const auto &entry) {
        return entry.parent == reopenedParentAccCore;
      }))
    return invalid("reopened AccCore is absent from parent-child lineage");
  if (!llvm::any_of(checkpoint->threadBindings(), [&](const auto &binding) {
        return binding.target == reopenedParentAccCore;
      }))
    return invalid("reopened AccCore owns no checkpoint thread binding");
  CanonicalSemanticBytes canonical(canonicalSeedBytes(
      checkpointReference, correspondence, reopenedParentAccCore));
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
  auto reopenedParentBytes = readBlob(bytes, offset);
  if (!reopenedParentBytes)
    return reopenedParentBytes.takeError();
  auto reopenedParentAccCore =
      ::loom::fabric::decodeFabricRef<::loom::fabric::AccCoreOccurrenceRef>(
          *reopenedParentBytes);
  if (!reopenedParentAccCore)
    return reopenedParentAccCore.takeError();
  auto count = readU64(bytes, offset);
  if (!count)
    return count.takeError();
  if (*count > bytes.size())
    return invalid("migration correspondence count exceeds payload size");
  std::vector<SystemAccCoreCorrespondence> pairs;
  pairs.reserve(static_cast<std::size_t>(*count));
  for (std::uint64_t ordinal = 0; ordinal != *count; ++ordinal) {
    auto parentBytes = readBlob(bytes, offset);
    if (!parentBytes)
      return parentBytes.takeError();
    auto parent =
        ::loom::fabric::decodeFabricRef<::loom::fabric::AccCoreOccurrenceRef>(
            *parentBytes);
    if (!parent)
      return parent.takeError();
    auto childBytes = readBlob(bytes, offset);
    if (!childBytes)
      return childBytes.takeError();
    auto child =
        ::loom::fabric::decodeFabricRef<::loom::fabric::AccCoreOccurrenceRef>(
            *childBytes);
    if (!child)
      return child.takeError();
    pairs.push_back({std::move(*parent), std::move(*child)});
  }
  if (offset != bytes.size())
    return invalid("migration seed payload has trailing bytes");
  auto checkpoint = ::loom::mapping::importSystemExecutionBindingCheckpoint(
      *checkpointReference, store);
  if (!checkpoint)
    return checkpoint.takeError();
  auto correspondence = SystemExecutionBindingCorrespondence::get(
      checkpoint->system(), std::move(*childSystem), std::move(pairs), store);
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
  if (llvm::ArrayRef<std::uint8_t>(canonicalSeedBytes(
          *checkpointReference, *correspondence, *reopenedParentAccCore)) !=
      bytes)
    return invalid("migration seed payload is not canonical");
  return FinalizedSystemMappingCheckpointMigrationSeed(
      reference, std::move(*checkpoint), std::move(*correspondence),
      *reopenedParentAccCore);
}

SystemMappingMigrationProjectionOutcome projectSystemMappingMigrationSeed(
    const FinalizedSystemMappingMigrationSeed &seed,
    const FrozenSystemPnrProblem &childProblem) {
  const auto &mapping = seed.parentMapping.view();
  if (mapping.dataflowIdentity() != childProblem.dataflowIdentity())
    return SystemMappingMigrationFallback{
        SystemMappingMigrationFallbackReason::ParentMappingDataflowMismatch};
  if (mapping.fabricIdentity() != seed.correspondence.parentSystem().artifact)
    return SystemMappingMigrationFallback{
        SystemMappingMigrationFallbackReason::ParentMappingFabricMismatch};
  if (childProblem.fabricIdentity() !=
      seed.correspondence.childSystem().artifact)
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
        mapAccCore(seed.correspondence, *parentTarget, ambiguous);
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
  case SystemMappingMigrationFallbackReason::ChildInitializerRejected:
    return "child_initializer_rejected";
  }
  llvm_unreachable("unknown SystemMapping migration fallback reason");
}

} // namespace loom::pnr
