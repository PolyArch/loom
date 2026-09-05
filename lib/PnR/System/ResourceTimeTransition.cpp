#include "PnR/System/SystemMappingMigration.h"

#include "ResourceTimeTransitionInternal.h"

#include "Common/ArtifactLocalReference.h"
#include "Common/ArtifactStore.h"
#include "Common/BlobStore.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Dataflow/IR/DataflowEventDerivation.h"
#include "Dataflow/IR/DataflowOps.h"
#include "Dataflow/IR/DataflowReferenceCodec.h"
#include "Deployment/Deployment.h"
#include "Deployment/HardwareConfigurationImage.h"
#include "Fabric/Identity/FabricRefBytes.h"
#include "Mapping/Artifact/MappingArtifact.h"
#include "Mapping/Artifact/MappingProgressAnalysis.h"
#include "Mapping/Artifact/SystemMappingExecutionProjection.h"
#include "Runtime/RuntimePlatformBinding.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <array>
#include <limits>
#include <map>
#include <optional>
#include <system_error>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace loom::pnr {
namespace {

constexpr llvm::StringLiteral resourceDeltaDescriptor{
    "loom.resource_time.resource_delta.v1"};
constexpr llvm::StringLiteral configurationDeltaDescriptor{
    "loom.resource_time.configuration_delta.v1"};
constexpr llvm::StringLiteral routeDeltaDescriptor{
    "loom.resource_time.route_delta.v1"};

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      "resource_time_transition_invalid: " + message);
}

void appendU64(std::vector<std::uint8_t> &bytes, std::uint64_t value) {
  for (int shift = 56; shift >= 0; shift -= 8)
    bytes.push_back(static_cast<std::uint8_t>(value >> shift));
}

void appendBlob(std::vector<std::uint8_t> &bytes,
                llvm::ArrayRef<std::uint8_t> value) {
  appendU64(bytes, value.size());
  bytes.insert(bytes.end(), value.begin(), value.end());
}

void appendRoot(std::vector<std::uint8_t> &bytes,
                const ArtifactRootReference &root) {
  const auto encoded = encodeArtifactRootReference(root);
  appendBlob(bytes, encoded);
}

llvm::Expected<ComponentViewDigest>
digestPair(llvm::StringRef descriptor, llvm::ArrayRef<std::uint8_t> parent,
           llvm::ArrayRef<std::uint8_t> child) {
  std::vector<std::uint8_t> bytes;
  appendBlob(bytes, parent);
  appendBlob(bytes, child);
  return computeComponentViewDigest(
      {reinterpret_cast<const std::uint8_t *>(descriptor.data()),
       descriptor.size()},
      bytes);
}

llvm::Expected<std::vector<std::uint8_t>>
allocationStateBytes(
    llvm::ArrayRef<ResourceTimeRegionAllocation> allocations,
    const ::loom::mapping::SystemExecutionContextProjection &contexts,
    llvm::ArrayRef<std::uint8_t> mappingClosure) {
  std::vector<std::pair<std::vector<std::uint8_t>,
                        std::vector<std::vector<std::uint8_t>>>>
      rows;
  rows.reserve(allocations.size());
  for (const ResourceTimeRegionAllocation &allocation : allocations) {
    auto region = ::dataflow::encodeDataflowReference(
        allocation.region.artifact, allocation.region);
    if (!region)
      return region.takeError();
    auto projected =
        projectResourceTimeMappingResources(contexts, allocation.region);
    if (!projected)
      return projected.takeError();
    std::vector<std::vector<std::uint8_t>> resources;
    resources.reserve(projected->size());
    for (const auto &resource : *projected)
      resources.push_back(::loom::fabric::canonicalFabricBytes(resource));
    llvm::sort(resources);
    rows.emplace_back(std::move(*region), std::move(resources));
  }
  llvm::sort(rows, [](const auto &lhs, const auto &rhs) {
    return lhs.first < rhs.first;
  });
  std::vector<std::uint8_t> bytes;
  appendBlob(bytes, mappingClosure);
  appendU64(bytes, rows.size());
  for (const auto &[region, resources] : rows) {
    appendBlob(bytes, region);
    appendU64(bytes, resources.size());
    for (const auto &resource : resources)
      appendBlob(bytes, resource);
  }
  return bytes;
}

struct ConfigurationRow final {
  std::vector<std::uint8_t> key;
  std::vector<std::uint8_t> value;
};

llvm::Expected<std::vector<std::uint8_t>> configurationImageStateBytes(
    const ::loom::deployment::FinalizedDeployment &deployment,
    const ArtifactStore &artifacts) {
  std::vector<ConfigurationRow> rows;
  rows.reserve(deployment.deployment().configurationImages().size());
  for (const ArtifactRootReference &reference :
       deployment.deployment().configurationImages()) {
    auto image = ::loom::deployment::importHardwareConfigurationImage(
        reference, artifacts);
    if (!image)
      return image.takeError();
    std::vector<std::uint8_t> key;
    appendRoot(key, image->image().configurationAbi());
    appendU64(key, image->image().programmingUnitId());
    std::vector<std::uint8_t> value;
    appendU64(value, image->image().payloadBitCount());
    appendBlob(value, image->image().payload());
    rows.push_back({std::move(key), std::move(value)});
  }
  llvm::sort(rows,
             [](const ConfigurationRow &lhs, const ConfigurationRow &rhs) {
               return lhs.key < rhs.key;
             });
  if (std::adjacent_find(rows.begin(), rows.end(),
                         [](const auto &lhs, const auto &rhs) {
                           return lhs.key == rhs.key;
                         }) != rows.end())
    return invalid("Deployment configuration repeats one programming unit");
  std::vector<std::uint8_t> bytes;
  appendU64(bytes, rows.size());
  for (const ConfigurationRow &row : rows) {
    appendBlob(bytes, row.key);
    appendBlob(bytes, row.value);
  }
  return bytes;
}

struct DeploymentConfigurationState final {
  std::vector<std::uint8_t> complete;
  std::vector<std::uint8_t> hardwareBindings;
};

llvm::Expected<DeploymentConfigurationState> deploymentConfigurationStateBytes(
    const ::loom::deployment::FinalizedDeployment &deployment,
    const ArtifactStore &artifacts) {
  const auto &canonical = deployment.canonicalBytes().bytes();
  const llvm::StringRef text(reinterpret_cast<const char *>(canonical.data()),
                             canonical.size());
  auto parsed = llvm::json::parse(text);
  if (!parsed)
    return invalid("Deployment canonical bytes are not JSON");
  const llvm::json::Object *root = parsed->getAsObject();
  if (!root)
    return invalid("Deployment canonical bytes are not an object");

  constexpr std::array<llvm::StringLiteral, 7> fields = {
      "host_program", "instruction_core_binary_refs", "hardware_bindings",
      "configuration_image_refs", "static_memory_images",
      "thread_dispatch_image", "admission_image"};
  llvm::SmallString<4096> storage;
  llvm::raw_svector_ostream output(storage);
  llvm::json::OStream json(output);
  bool missing = false;
  json.object([&] {
    for (llvm::StringRef field : fields) {
      const llvm::json::Value *value = root->get(field);
      if (!value) {
        missing = true;
        continue;
      }
      json.attribute(field, *value);
    }
    if (const llvm::json::Value *spatial = root->get("spatial_launch_image"))
      json.attribute("spatial_launch_image", *spatial);
    else
      json.attribute("spatial_launch_image", nullptr);
  });
  if (missing)
    return invalid("Deployment configuration closure is incomplete");
  auto images = configurationImageStateBytes(deployment, artifacts);
  if (!images)
    return images.takeError();
  std::vector<std::uint8_t> complete;
  appendBlob(complete, llvm::ArrayRef<std::uint8_t>(
                           reinterpret_cast<const std::uint8_t *>(storage.data()),
                           storage.size()));
  appendBlob(complete, *images);

  const llvm::json::Value *hardwareBindings = root->get("hardware_bindings");
  if (!hardwareBindings)
    return invalid("Deployment hardware-binding closure is incomplete");
  llvm::SmallString<2048> hardwareStorage;
  llvm::raw_svector_ostream hardwareOutput(hardwareStorage);
  llvm::json::OStream hardwareJson(hardwareOutput);
  hardwareJson.object([&] {
    hardwareJson.attribute("hardware_bindings", *hardwareBindings);
  });
  std::vector<std::uint8_t> hardwareBindingBytes;
  appendBlob(hardwareBindingBytes,
             llvm::ArrayRef<std::uint8_t>(
                 reinterpret_cast<const std::uint8_t *>(hardwareStorage.data()),
                 hardwareStorage.size()));

  return DeploymentConfigurationState{std::move(complete),
                                      std::move(hardwareBindingBytes)};
}

llvm::Expected<std::vector<std::uint8_t>>
mappingClosureStateBytes(
    const ::loom::mapping::FinalizedSystemMapping &mapping,
    const ArtifactStore &artifacts) {
  std::vector<std::uint8_t> bytes;
  appendBlob(bytes, mapping.canonicalBytes().bytes());
  std::vector<ArtifactRootReference> spatialMappings(
      mapping.view().executionBindings().spatialMappingImports().begin(),
      mapping.view().executionBindings().spatialMappingImports().end());
  llvm::sort(spatialMappings, artifactRootReferenceLess);
  appendU64(bytes, spatialMappings.size());
  for (const ArtifactRootReference &reference : spatialMappings) {
    auto spatial = ::loom::mapping::importSpatialMapping(reference, artifacts);
    if (!spatial)
      return spatial.takeError();
    appendRoot(bytes, reference);
    appendBlob(bytes, spatial->canonicalBytes().bytes());
  }
  return bytes;
}

bool containsChannelType(mlir::Type type) {
  return type
      .walk<mlir::WalkOrder::PreOrder>([](mlir::Type nested) {
        return llvm::isa<::dataflow::ChannelType>(nested)
                   ? mlir::WalkResult::interrupt()
                   : mlir::WalkResult::advance();
      })
      .wasInterrupted();
}

llvm::Error refuse(ResourceTimeTransitionRefusalReason reason,
                   const llvm::Twine &message) {
  return llvm::make_error<ResourceTimeTransitionRefusal>(reason,
                                                         message.str());
}

/// Channel-typed and DynamicWork state has no correspondence owner, so a
/// Dataflow carrying either cannot cross a verified edge.
llvm::Error refuseUnownedLiveState(
    const ::dataflow::CanonicalDataflowArtifact &artifact) {
  std::optional<ResourceTimeLiveStateClass> unowned;
  artifact.module()->walk([&](mlir::Operation *operation) {
    if (auto thread = llvm::dyn_cast<::dataflow::ThreadOp>(operation)) {
      if (thread.getDomain().getKind() ==
          ::dataflow::ThreadDomainKind::DynamicWork)
        unowned = ResourceTimeLiveStateClass::DynamicWork;
      for (mlir::Type type : thread.getFunctionType().getInputs())
        if (containsChannelType(type) && !unowned)
          unowned = ResourceTimeLiveStateClass::OrderedChannel;
    }
    for (mlir::Value operand : operation->getOperands())
      if (containsChannelType(operand.getType()) && !unowned)
        unowned = ResourceTimeLiveStateClass::OrderedChannel;
    for (mlir::Value result : operation->getResults())
      if (containsChannelType(result.getType()) && !unowned)
        unowned = ResourceTimeLiveStateClass::OrderedChannel;
  });
  if (!unowned)
    return llvm::Error::success();
  return refuse(*unowned == ResourceTimeLiveStateClass::DynamicWork
                    ? ResourceTimeTransitionRefusalReason::DynamicWorkState
                    : ResourceTimeTransitionRefusalReason::OrderedChannelState,
                llvm::Twine("Canonical Dataflow carries ") +
                    resourceTimeLiveStateClassSpelling(*unowned) +
                    " live state without a migration correspondence owner");
}

constexpr llvm::StringLiteral memoryBindingDescriptor{
    "loom.resource_time.memory_binding.v2"};

::dataflow::LogicalMemoryRootRef
logicalMemoryRootOf(const ::dataflow::LogicalMemoryRootOrViewRef &memory) {
  return std::visit(
      [](const auto &value) -> ::dataflow::LogicalMemoryRootRef {
        using Value = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<Value, ::dataflow::LogicalMemoryRootRef>)
          return value;
        else
          return value.root;
      },
      memory);
}

void appendMemoryInterval(
    std::vector<std::uint8_t> &bytes,
    const ::loom::mapping::SpatialMemoryIntervalView &interval) {
  if (const auto *range =
          std::get_if<::loom::mapping::SpatialMemoryByteRangeView>(&interval)) {
    appendU64(bytes, 1);
    appendU64(bytes, range->offsetBytes);
    appendU64(bytes, range->sizeBytes);
    return;
  }
  appendU64(bytes, 0);
}

llvm::Expected<std::vector<std::uint8_t>>
memoryTargetBytes(const ResourceTimeMemoryTarget &target) {
  std::vector<std::uint8_t> bytes;
  if (const auto *local =
          std::get_if<ResourceTimeSpatialMemoryTarget>(&target)) {
    appendU64(bytes, 0);
    appendBlob(bytes, ::loom::fabric::canonicalFabricBytes(local->accCore));
    appendMemoryInterval(bytes, local->interval);
    appendBlob(bytes, ::loom::fabric::canonicalFabricBytes(
                          local->region.serviceRegion));
    appendU64(bytes, local->region.physicalOffsetBytes);
    return bytes;
  }
  appendU64(bytes, 1);
  const auto &system =
      std::get<::loom::mapping::SystemMemoryRegionElementView>(target);
  auto logical = ::dataflow::encodeDataflowReference(system.logicalMemory);
  if (!logical)
    return logical.takeError();
  appendBlob(bytes, *logical);
  appendMemoryInterval(bytes, system.interval);
  appendBlob(bytes, ::loom::fabric::canonicalFabricBytes(system.serviceRegion));
  appendU64(bytes, system.transformPath.size());
  for (const auto &transform : system.transformPath)
    appendBlob(bytes, ::loom::fabric::canonicalFabricBytes(transform));
  return bytes;
}

bool selectionMatchesRoots(
    const ::loom::mapping::ServicePlanSelectionAnchor &anchor,
    llvm::ArrayRef<::dataflow::RootThreadLaunchRef> roots) {
  return std::visit(
      [&](const auto &typed) {
        using Anchor = std::decay_t<decltype(typed)>;
        if constexpr (std::is_same_v<
                          Anchor,
                          ::loom::mapping::MemoryExposurePlanSelectionAnchor>) {
          return llvm::is_contained(roots,
                                    typed.exposure.launch.rootThreadLaunch);
        } else {
          return std::visit(
              [&](const auto &member) {
                using Member = std::decay_t<decltype(member)>;
                if constexpr (std::is_same_v<
                                  Member,
                                  ::dataflow::AddressedMemoryActorMemberRef> ||
                              std::is_same_v<Member,
                                             ::dataflow::FenceActorMemberRef>)
                  return llvm::is_contained(
                      roots, member.actor.launch.rootThreadLaunch);
                else
                  return false;
              },
              typed.member);
        }
      },
      anchor);
}

bool planCanBeSelected(
    const ::loom::mapping::SystemServicePlanSelectionView &selection,
    std::uint64_t planOrdinal) {
  if (selection.defaultPlanOrdinal == planOrdinal)
    return true;
  return llvm::any_of(selection.clauses, [&](const auto &clause) {
    return clause.target == planOrdinal;
  });
}

llvm::Expected<ResourceTimeLogicalMemoryBindingProjection>
projectLogicalMemoryBinding(
    const ::loom::mapping::FinalizedSystemMapping &mapping,
    ::dataflow::LogicalMemoryRootRef memory,
    llvm::ArrayRef<::dataflow::RootThreadLaunchRef> roots,
    const ArtifactStore &artifacts) {
  const ArtifactRootReference dataflowReference{
      ::dataflow::canonicalDataflowSchema.identity.str(),
      ::dataflow::canonicalDataflowSchema.version,
      mapping.view().dataflowIdentity()};
  auto dataflowArtifact =
      ::dataflow::importCanonicalDataflow(dataflowReference, artifacts);
  if (!dataflowArtifact)
    return dataflowArtifact.takeError();
  auto dataflow = dataflowArtifact->view();
  if (!dataflow)
    return dataflow.takeError();
  auto extent = dataflow->staticMemoryByteExtent(
      ::dataflow::LogicalMemoryRootOrViewRef{memory});
  if (!extent)
    return extent.takeError();
  auto contexts = ::loom::mapping::projectSystemExecutionContexts(
      *dataflow, mapping.view().executionBindings());
  if (!contexts)
    return contexts.takeError();

  std::vector<ResourceTimeMemoryTarget> candidates;
  bool usesBoundaryProxy = false;
  for (const auto &domain : contexts->spatialDomains) {
    if (!llvm::is_contained(roots, domain.graph.rootThreadLaunch))
      continue;
    auto spatial =
        ::loom::mapping::importSpatialMapping(domain.spatialMapping, artifacts);
    if (!spatial)
      return spatial.takeError();
    for (const auto &binding : spatial->view().memoryBindings()) {
      if (logicalMemoryRootOf(binding.logicalMemory) != memory)
        continue;
      if (const auto *local =
              std::get_if<::loom::mapping::SpatialMemoryLocalRegionView>(
                  &binding.target))
        candidates.emplace_back(ResourceTimeSpatialMemoryTarget{
            domain.context.accCore, binding.interval, *local});
      else
        usesBoundaryProxy = true;
    }
  }

  if (usesBoundaryProxy) {
    for (const auto &realization : mapping.view().serviceRealizations()) {
      const auto *operation =
          std::get_if<::loom::mapping::OperationServiceObligationFamilyKey>(
              &realization.key);
      const auto *logical =
          operation
              ? std::get_if<::dataflow::LogicalMemoryRootOrViewRef>(operation)
              : nullptr;
      if (!logical || logicalMemoryRootOf(*logical) != memory)
        continue;
      for (const auto &plan : realization.plans) {
        const bool selected =
            llvm::any_of(realization.selections, [&](const auto &selection) {
              return selectionMatchesRoots(selection.key.anchor, roots) &&
                     planCanBeSelected(selection, plan.ordinal);
            });
        if (!selected)
          continue;
        for (const auto &target : plan.memoryTargets)
          if (logicalMemoryRootOf(target.element.logicalMemory) == memory)
            candidates.emplace_back(target.element);
      }
    }
  }

  std::map<std::vector<std::uint8_t>, ResourceTimeMemoryTarget> canonical;
  for (ResourceTimeMemoryTarget &target : candidates) {
    auto bytes = memoryTargetBytes(target);
    if (!bytes)
      return bytes.takeError();
    canonical.try_emplace(std::move(*bytes), std::move(target));
  }
  std::vector<std::uint8_t> digestBytes;
  appendU64(digestBytes, canonical.size());
  std::vector<ResourceTimeMemoryTarget> targets;
  targets.reserve(canonical.size());
  for (auto &[bytes, target] : canonical) {
    appendBlob(digestBytes, bytes);
    targets.push_back(std::move(target));
  }
  auto digest = computeComponentViewDigest(
      {reinterpret_cast<const std::uint8_t *>(memoryBindingDescriptor.data()),
       memoryBindingDescriptor.size()},
      digestBytes);
  if (!digest)
    return digest.takeError();
  return ResourceTimeLogicalMemoryBindingProjection{
      memory, *extent, std::move(targets), *digest};
}

bool targetCoversObject(const ResourceTimeMemoryTarget &target,
                        std::uint64_t byteCount) {
  const auto &interval = std::visit(
      [](const auto &typed) -> const auto & { return typed.interval; }, target);
  const auto *range =
      std::get_if<::loom::mapping::SpatialMemoryByteRangeView>(&interval);
  return !range || (range->offsetBytes == 0 && range->sizeBytes == byteCount);
}

struct ConfigurationImageRecord final {
  ArtifactRootReference reference;
  std::uint64_t payloadBitCount = 0;
  std::vector<std::uint8_t> payload;
};

llvm::Expected<std::map<std::vector<std::uint8_t>, ConfigurationImageRecord>>
configurationImageCatalog(
    const ::loom::deployment::FinalizedDeployment &deployment,
    const ArtifactStore &artifacts) {
  std::map<std::vector<std::uint8_t>, ConfigurationImageRecord> result;
  for (const ArtifactRootReference &reference :
       deployment.deployment().configurationImages()) {
    auto image = ::loom::deployment::importHardwareConfigurationImage(
        reference, artifacts);
    if (!image)
      return image.takeError();
    std::vector<std::uint8_t> key;
    appendRoot(key, image->image().configurationAbi());
    appendU64(key, image->image().programmingUnitId());
    ConfigurationImageRecord record{
        reference, image->image().payloadBitCount(),
        std::vector<std::uint8_t>(image->image().payload().begin(),
                                  image->image().payload().end())};
    if (!result.try_emplace(std::move(key), std::move(record)).second)
      return invalid("Deployment configuration repeats one programming unit");
  }
  return result;
}

llvm::Expected<std::optional<::loom::runtime::RuntimeProviderBinding>>
deploymentProviderBinding(
    const ::loom::deployment::FinalizedDeployment &deployment,
    const ArtifactStore &artifacts, const BlobStore &blobs) {
  std::optional<::loom::runtime::RuntimeProviderBinding> result;
  for (const auto &hardware : deployment.deployment().hardwareBindings()) {
    auto binding = ::loom::runtime::importRuntimePlatformBinding(
        hardware.runtimePlatformBinding, artifacts, blobs);
    if (!binding)
      return binding.takeError();
    const auto &provider = binding->binding().providerBinding();
    if (result && !(*result == provider))
      return invalid("Deployment hardware bindings select multiple runtime "
                     "providers");
    result = provider;
  }
  return result;
}

llvm::Expected<std::optional<::loom::runtime::RuntimeResourceTimeCostModel>>
transitionCostModel(const ::loom::deployment::FinalizedDeployment &parent,
                    const ::loom::deployment::FinalizedDeployment &child,
                    const ArtifactStore &artifacts, const BlobStore &blobs) {
  auto parentProvider = deploymentProviderBinding(parent, artifacts, blobs);
  if (!parentProvider)
    return parentProvider.takeError();
  auto childProvider = deploymentProviderBinding(child, artifacts, blobs);
  if (!childProvider)
    return childProvider.takeError();
  if (parentProvider->has_value() != childProvider->has_value() ||
      (parentProvider->has_value() && !(**parentProvider == **childProvider)))
    return refuse(ResourceTimeTransitionRefusalReason::
                      RuntimeTransitionCapabilityUnavailable,
                  "transition endpoints select different runtime providers");
  if (!*parentProvider)
    return std::optional<::loom::runtime::RuntimeResourceTimeCostModel>{};
  return (*parentProvider)->resourceTimeCostModel;
}

std::uint32_t configurationImageWord(llvm::ArrayRef<std::uint8_t> image,
                                     std::uint64_t word) {
  std::uint32_t result = 0;
  for (unsigned byte = 0; byte != 4; ++byte) {
    const std::uint64_t index = word * 4 + byte;
    if (index < image.size())
      result |= std::uint32_t(image[static_cast<std::size_t>(index)])
                << (byte * 8);
  }
  return result;
}

llvm::Expected<std::uint64_t> scaledCost(std::uint64_t count,
                                         std::uint64_t unitCost,
                                         std::uint64_t fixedCost,
                                         llvm::StringRef operation) {
  if (unitCost == 0)
    return invalid(operation + " cost has a zero unit price");
  if (count >
      (std::numeric_limits<std::uint64_t>::max() - fixedCost) / unitCost)
    return invalid(operation + " cost exceeds the u64 picosecond domain");
  return fixedCost + count * unitCost;
}

llvm::Expected<
    std::pair<std::vector<ResourceTimeConfigurationImageDelta>, std::uint64_t>>
deriveConfigurationExecutionPlan(
    const ::loom::deployment::FinalizedDeployment &parent,
    const ::loom::deployment::FinalizedDeployment &child,
    const std::optional<::loom::runtime::RuntimeResourceTimeCostModel> &cost,
    const ArtifactStore &artifacts) {
  auto parentImages = configurationImageCatalog(parent, artifacts);
  if (!parentImages)
    return parentImages.takeError();
  auto childImages = configurationImageCatalog(child, artifacts);
  if (!childImages)
    return childImages.takeError();
  for (const auto &[key, parentImage] : *parentImages) {
    (void)parentImage;
    if (childImages->find(key) == childImages->end())
      return refuse(
          ResourceTimeTransitionRefusalReason::
              RuntimeTransitionCapabilityUnavailable,
          "parent configuration image has no child image to replace it");
  }

  std::vector<ResourceTimeConfigurationImageDelta> images;
  std::uint64_t changedWords = 0;
  for (const auto &[key, childImage] : *childImages) {
    const auto parentImage = parentImages->find(key);
    if (parentImage != parentImages->end() &&
        parentImage->second.payloadBitCount != childImage.payloadBitCount)
      return invalid("configuration image payload extent changes within one "
                     "ConfigurationABI programming unit");
    const std::uint64_t wordCount = childImage.payloadBitCount / 32 +
                                    (childImage.payloadBitCount % 32 != 0);
    ResourceTimeConfigurationImageDelta image{childImage.reference, {}};
    image.changedWordOrdinals.reserve(static_cast<std::size_t>(wordCount));
    for (std::uint64_t word = 0; word != wordCount; ++word) {
      const bool changed =
          parentImage == parentImages->end() ||
          configurationImageWord(parentImage->second.payload, word) !=
              configurationImageWord(childImage.payload, word);
      if (changed)
        image.changedWordOrdinals.push_back(word);
    }
    if (image.changedWordOrdinals.empty())
      continue;
    if (changedWords > std::numeric_limits<std::uint64_t>::max() -
                           image.changedWordOrdinals.size())
      return invalid("configuration word delta exceeds the u64 domain");
    changedWords += image.changedWordOrdinals.size();
    images.push_back(std::move(image));
  }
  if (images.empty())
    return std::make_pair(std::move(images), std::uint64_t{0});
  if (!cost)
    return refuse(
        ResourceTimeTransitionRefusalReason::
            RuntimeTransitionCapabilityUnavailable,
        "changed hardware programming has no runtime-provider cost model");
  auto words = scaledCost(changedWords, cost->configurationWordPicoseconds, 0,
                          "configuration word");
  if (!words)
    return words.takeError();
  auto total = scaledCost(images.size(), cost->configurationCommitPicoseconds,
                          *words, "configuration commit");
  if (!total)
    return total.takeError();
  return std::make_pair(std::move(images), *total);
}

struct DerivedLogicalMemoryPlan final {
  std::vector<ResourceTimeLogicalMemoryCorrespondence> correspondence;
  std::vector<ResourceTimeLogicalMemoryCopyPlan> copies;
  std::uint64_t migrationTimePicoseconds = 0;
};

llvm::Expected<DerivedLogicalMemoryPlan> deriveLogicalMemoryExecutionPlan(
    const ResourceTimeTransition &transition,
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ::loom::mapping::FinalizedSystemMapping &parentMapping,
    const ::loom::mapping::FinalizedSystemMapping &childMapping,
    const ::loom::deployment::FinalizedDeployment &childDeployment,
    const std::optional<::loom::runtime::RuntimeResourceTimeCostModel> &cost,
    const ArtifactStore &artifacts) {
  std::vector<::dataflow::RootThreadLaunchRef> sourceRoots =
      transition.completedBefore;
  for (const ResourceTimeRegionAllocation &allocation : transition.beforeActive)
    if (!llvm::is_contained(sourceRoots, allocation.region))
      sourceRoots.push_back(allocation.region);
  std::vector<::dataflow::RootThreadLaunchRef> destinationRoots;
  for (const auto root :
       childMapping.view().executionBindings().rootThreadLaunches())
    if (!llvm::is_contained(sourceRoots, root))
      destinationRoots.push_back(root);

  DerivedLogicalMemoryPlan result;
  for (const auto &memory : dataflow.logicalMemoryRoots()) {
    auto parent = projectLogicalMemoryBinding(parentMapping, memory.ref,
                                              sourceRoots, artifacts);
    if (!parent)
      return parent.takeError();
    auto child = projectLogicalMemoryBinding(childMapping, memory.ref,
                                             destinationRoots, artifacts);
    if (!child)
      return child.takeError();
    if (child->targets.empty())
      continue;
    if (parent->targets.empty())
      return refuse(ResourceTimeTransitionRefusalReason::LogicalMemoryUnbound,
                    "a live logical memory has no source target before the "
                    "completion safe point");
    for (const auto &image : childDeployment.deployment().staticMemoryImages())
      if (image.logicalMemoryRoot() == memory.ref)
        return refuse(
            ResourceTimeTransitionRefusalReason::LogicalMemoryReinitialized,
            "child Deployment would reinitialize a live logical memory");
    if (parent->digest == child->digest) {
      result.correspondence.push_back(
          {memory.ref, parent->digest, child->digest,
           ResourceTimeLiveStateMigration::RetainedInPlace, 0});
      continue;
    }
    if (!parent->byteCount || !child->byteCount)
      return refuse(
          ResourceTimeTransitionRefusalReason::LogicalMemoryExtentUnknown,
          "relocated logical memory has no static byte extent");
    if (*parent->byteCount == 0 || *parent->byteCount != *child->byteCount ||
        parent->targets.size() != 1 || child->targets.size() != 1 ||
        !targetCoversObject(parent->targets.front(), *parent->byteCount) ||
        !targetCoversObject(child->targets.front(), *child->byteCount))
      return refuse(
          ResourceTimeTransitionRefusalReason::
              LogicalMemoryCopyShapeUnsupported,
          "logical-memory copy requires one complete equal-extent source and "
          "destination target");
    if (!cost)
      return refuse(
          ResourceTimeTransitionRefusalReason::
              RuntimeTransitionCapabilityUnavailable,
          "relocated logical memory has no runtime-provider copy cost model");
    auto migration =
        scaledCost(*parent->byteCount, cost->memoryCopyBytePicoseconds,
                   cost->memoryCopySetupPicoseconds, "logical-memory copy");
    if (!migration)
      return migration.takeError();
    if (result.migrationTimePicoseconds >
        std::numeric_limits<std::uint64_t>::max() - *migration)
      return invalid("logical-memory migration cost exceeds the u64 "
                     "picosecond domain");
    result.migrationTimePicoseconds += *migration;
    result.correspondence.push_back({memory.ref, parent->digest, child->digest,
                                     ResourceTimeLiveStateMigration::Copied,
                                     *migration});
    result.copies.push_back(ResourceTimeLogicalMemoryCopyPlan{
        memory.ref, *parent->byteCount, parent->targets.front(),
        child->targets.front()});
  }
  llvm::sort(result.correspondence, [](const auto &lhs, const auto &rhs) {
    return lhs.memory.entity.value() < rhs.memory.entity.value();
  });
  llvm::sort(result.copies, [](const auto &lhs, const auto &rhs) {
    return lhs.memory.entity.value() < rhs.memory.entity.value();
  });
  return result;
}

llvm::Expected<ResourceTimeTransitionExecutionPlan> deriveExecutionPlan(
    const ResourceTimeTransition &transition,
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ::loom::mapping::FinalizedSystemMapping &parentMapping,
    const ::loom::mapping::FinalizedSystemMapping &childMapping,
    const ::loom::deployment::FinalizedDeployment &parentDeployment,
    const ::loom::deployment::FinalizedDeployment &childDeployment,
    const ArtifactStore &artifacts, const BlobStore &blobs,
    std::vector<ResourceTimeLogicalMemoryCorrespondence> &correspondence) {
  auto cost =
      transitionCostModel(parentDeployment, childDeployment, artifacts, blobs);
  if (!cost)
    return cost.takeError();
  auto configuration = deriveConfigurationExecutionPlan(
      parentDeployment, childDeployment, *cost, artifacts);
  if (!configuration)
    return configuration.takeError();
  auto logical = deriveLogicalMemoryExecutionPlan(
      transition, dataflow, parentMapping, childMapping, childDeployment, *cost,
      artifacts);
  if (!logical)
    return logical.takeError();
  correspondence = std::move(logical->correspondence);
  return ResourceTimeTransitionExecutionPlan{
      std::move(configuration->first), std::move(logical->copies),
      configuration->second, logical->migrationTimePicoseconds};
}

struct DerivedTransitionDigests final {
  ComponentViewDigest resources;
  ComponentViewDigest configuration;
  ComponentViewDigest routes;
  std::vector<ResourceTimeLogicalMemoryCorrespondence> logicalMemories;
  ResourceTimeTransitionExecutionPlan execution;
};

bool sameRootSet(
    llvm::ArrayRef<::dataflow::RootThreadLaunchRef> lhs,
    llvm::ArrayRef<::dataflow::RootThreadLaunchRef> rhs) {
  return lhs.size() == rhs.size() && llvm::all_of(lhs, [&](const auto root) {
           return llvm::is_contained(rhs, root);
         });
}

llvm::Expected<bool> completionFrontierIsCausallyAdmissible(
    const ResourceTimeTransition &transition,
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    llvm::ArrayRef<::dataflow::RootThreadLaunchRef> mappedRoots) {
  if (transition.beforeActive.size() != 1)
    return false;
  const ::dataflow::RootThreadLaunchRef completing =
      transition.beforeActive.front().region;
  if (transition.trigger !=
      ::dataflow::rootThreadCompletionEventFamily(completing))
    return false;
  std::vector<::dataflow::RootThreadLaunchRef> activeAfter;
  activeAfter.reserve(transition.afterActive.size());
  for (const ResourceTimeRegionAllocation &allocation : transition.afterActive)
    activeAfter.push_back(allocation.region);
  return ::loom::mapping::mappingCompletionFrontierIsAdmissible(
      dataflow, mappedRoots, transition.completedBefore, completing,
      activeAfter);
}

llvm::Expected<DerivedTransitionDigests>
deriveTransitionDigests(const ResourceTimeTransition &transition,
                        const ArtifactStore &artifacts,
                        const BlobStore &blobs) {
  if (!transition.parent.deployment || !transition.child.deployment)
    return invalid("transition delta derivation requires exact Deployments");
  auto parentDeployment = ::loom::deployment::importDeployment(
      *transition.parent.deployment, artifacts, blobs);
  if (!parentDeployment)
    return parentDeployment.takeError();
  auto childDeployment = ::loom::deployment::importDeployment(
      *transition.child.deployment, artifacts, blobs);
  if (!childDeployment)
    return childDeployment.takeError();
  if (parentDeployment->deployment().systemMapping() !=
          transition.parent.mapping ||
      childDeployment->deployment().systemMapping() != transition.child.mapping)
    return invalid("transition Deployment endpoint selects another Mapping");

  auto parentMapping = ::loom::mapping::importSystemMapping(
      transition.parent.mapping, artifacts);
  if (!parentMapping)
    return parentMapping.takeError();
  auto childMapping =
      ::loom::mapping::importSystemMapping(transition.child.mapping, artifacts);
  if (!childMapping)
    return childMapping.takeError();
  if (parentMapping->view().dataflowIdentity() !=
          childMapping->view().dataflowIdentity() ||
      parentMapping->view().fabricIdentity() !=
          childMapping->view().fabricIdentity())
    return invalid("transition endpoints do not share Dataflow and Fabric");
  if (!sameRootSet(
          parentMapping->view().executionBindings().rootThreadLaunches(),
          childMapping->view().executionBindings().rootThreadLaunches()))
    return invalid("transition endpoint Mapping root scopes differ");

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

  auto parentContexts = ::loom::mapping::projectSystemExecutionContexts(
      *dataflow, parentMapping->view().executionBindings());
  if (!parentContexts)
    return parentContexts.takeError();
  auto childContexts = ::loom::mapping::projectSystemExecutionContexts(
      *dataflow, childMapping->view().executionBindings());
  if (!childContexts)
    return childContexts.takeError();
  auto parentMappingClosure =
      mappingClosureStateBytes(*parentMapping, artifacts);
  auto childMappingClosure = mappingClosureStateBytes(*childMapping, artifacts);
  if (!parentMappingClosure)
    return parentMappingClosure.takeError();
  if (!childMappingClosure)
    return childMappingClosure.takeError();
  auto beforeResources =
      allocationStateBytes(transition.beforeActive, *parentContexts,
                           *parentMappingClosure);
  auto afterResources =
      allocationStateBytes(transition.afterActive, *childContexts,
                           *childMappingClosure);
  auto beforeConfiguration =
      deploymentConfigurationStateBytes(*parentDeployment, artifacts);
  auto afterConfiguration =
      deploymentConfigurationStateBytes(*childDeployment, artifacts);
  if (!beforeResources)
    return beforeResources.takeError();
  if (!afterResources)
    return afterResources.takeError();
  if (!beforeConfiguration)
    return beforeConfiguration.takeError();
  if (!afterConfiguration)
    return afterConfiguration.takeError();

  auto resources =
      digestPair(resourceDeltaDescriptor, *beforeResources, *afterResources);
  auto configuration = digestPair(configurationDeltaDescriptor,
                                  beforeConfiguration->complete,
                                  afterConfiguration->complete);
  auto routes = digestPair(routeDeltaDescriptor, *parentMappingClosure,
                           *childMappingClosure);
  if (!resources)
    return resources.takeError();
  if (!configuration)
    return configuration.takeError();
  if (!routes)
    return routes.takeError();
  auto completionFrontier = completionFrontierIsCausallyAdmissible(
      transition, *dataflow,
      parentMapping->view().executionBindings().rootThreadLaunches());
  if (!completionFrontier)
    return completionFrontier.takeError();
  if (!*completionFrontier)
    return refuse(
        ResourceTimeTransitionRefusalReason::CompletionFrontierInadmissible,
        "completion frontier is not causally admissible under the canonical "
        "Dataflow event relation");
  if (llvm::Error error = refuseUnownedLiveState(*dataflowArtifact))
    return std::move(error);
  if (beforeConfiguration->hardwareBindings !=
      afterConfiguration->hardwareBindings)
    return refuse(ResourceTimeTransitionRefusalReason::HardwareBindingChanged,
                  "transition endpoints select different immutable hardware "
                  "or runtime bindings");
  std::vector<ResourceTimeLogicalMemoryCorrespondence> logicalMemories;
  auto execution = deriveExecutionPlan(
      transition, *dataflow, *parentMapping, *childMapping, *parentDeployment,
      *childDeployment, artifacts, blobs, logicalMemories);
  if (!execution)
    return execution.takeError();
  return DerivedTransitionDigests{*resources, *configuration, *routes,
                                  std::move(logicalMemories),
                                  std::move(*execution)};
}

llvm::Error
requireCompletionSafePointDraft(const ResourceTimeTransition &transition) {
  if (!transition.safePoint ||
      transition.safePoint->kind != ResourceTimeSafePointKind::Completion)
    return invalid("transition finalization requires a completion safe point");
  if (transition.parent.mapping == transition.child.mapping ||
      transition.parent.deployment == transition.child.deployment)
    return invalid("transition finalization requires distinct Mapping and "
                   "Deployment endpoints");
  if (transition.beforeActive.size() != 1)
    return invalid("completion transition requires one active completing "
                   "parent region");
  if (llvm::any_of(transition.afterActive, [&](const auto &allocation) {
        return allocation.region == transition.beforeActive.front().region;
      }))
    return invalid("completing region remains active after its completion "
                   "safe point");
  return llvm::Error::success();
}

} // namespace

llvm::Expected<std::vector<::loom::fabric::FabricPhysicalOccurrenceOwnerRef>>
projectResourceTimeMappingResources(
    const ::loom::mapping::SystemExecutionContextProjection &contexts,
    ::dataflow::RootThreadLaunchRef root) {
  std::vector<::loom::fabric::FabricPhysicalOccurrenceOwnerRef> resources;
  const auto appendCore = [&](::loom::fabric::AccCoreOccurrenceRef core)
      -> llvm::Error {
    auto physical = ::loom::fabric::FabricPhysicalOccurrenceOwnerRef::create(
        ::loom::fabric::FabricInventoryOwnerRef::of(core));
    if (!physical)
      return physical.takeError();
    resources.push_back(std::move(*physical));
    return llvm::Error::success();
  };
  for (const auto &domain : contexts.instructionDomains) {
    if (domain.root != root)
      continue;
    if (llvm::Error error = appendCore(domain.context.accCore))
      return std::move(error);
  }
  for (const auto &domain : contexts.spatialDomains) {
    if (domain.graph.rootThreadLaunch != root)
      continue;
    if (llvm::Error error = appendCore(domain.context.accCore))
      return std::move(error);
  }
  llvm::sort(resources, [](const auto &lhs, const auto &rhs) {
    return ::loom::fabric::canonicalFabricBytes(lhs) <
           ::loom::fabric::canonicalFabricBytes(rhs);
  });
  resources.erase(std::unique(resources.begin(), resources.end()),
                  resources.end());
  return resources;
}

llvm::Expected<ResourceTimeLogicalMemoryBindingProjection>
projectResourceTimeLogicalMemoryBinding(
    const ::loom::mapping::FinalizedSystemMapping &mapping,
    ::dataflow::LogicalMemoryRootRef memory,
    llvm::ArrayRef<::dataflow::RootThreadLaunchRef> roots,
    const ArtifactStore &artifacts) {
  return projectLogicalMemoryBinding(mapping, memory, roots, artifacts);
}

llvm::Expected<std::vector<std::uint8_t>>
canonicalResourceTimeMemoryTargetBytes(const ResourceTimeMemoryTarget &target) {
  return memoryTargetBytes(target);
}

llvm::Expected<ResourceTimeTransitionExecutionPlan>
deriveResourceTimeTransitionExecutionPlan(
    const ResourceTimeTransition &transition, const ArtifactStore &artifacts,
    const BlobStore &blobs) {
  if (transition.status != ResourceTimeTransitionStatus::Verified)
    return invalid("execution plan requires a verified transition");
  auto derived = deriveTransitionDigests(transition, artifacts, blobs);
  if (!derived)
    return derived.takeError();
  if (transition.logicalMemories != derived->logicalMemories ||
      transition.reprogrammingTimePicoseconds !=
          std::optional<std::uint64_t>(
              derived->execution.reprogrammingTimePicoseconds) ||
      transition.migrationTimePicoseconds !=
          std::optional<std::uint64_t>(
              derived->execution.migrationTimePicoseconds))
    return invalid("verified transition disagrees with its executable live-"
                   "state and configuration projection");
  return std::move(derived->execution);
}

llvm::Expected<ResourceTimeTransition>
finalizeResourceTimeTransition(ResourceTimeTransition draft,
                               const ArtifactStore &artifacts,
                               const BlobStore &blobs) {
  if (draft.status != ResourceTimeTransitionStatus::ProofNotEstablished)
    return invalid("transition draft has an authored terminal status");
  if (draft.resourceDeltaDigest || draft.configurationDeltaDigest ||
      draft.routeDeltaDigest)
    return invalid("transition draft has authored delta digests");
  if (draft.reprogrammingTimePicoseconds || draft.migrationTimePicoseconds)
    return invalid("transition draft has authored cost components");
  if (!draft.logicalMemories.empty())
    return invalid("transition draft has authored live-state correspondence");
  if (llvm::Error error = requireCompletionSafePointDraft(draft))
    return std::move(error);
  if (llvm::Error error = validateResourceTimeTransition(draft))
    return std::move(error);
  auto digests = deriveTransitionDigests(draft, artifacts, blobs);
  if (!digests)
    return digests.takeError();
  draft.resourceDeltaDigest = digests->resources;
  draft.configurationDeltaDigest = digests->configuration;
  draft.routeDeltaDigest = digests->routes;
  draft.logicalMemories = std::move(digests->logicalMemories);
  draft.reprogrammingTimePicoseconds =
      digests->execution.reprogrammingTimePicoseconds;
  draft.migrationTimePicoseconds = digests->execution.migrationTimePicoseconds;
  draft.status = ResourceTimeTransitionStatus::Verified;
  if (llvm::Error error =
          verifyResourceTimeTransitionClosure(draft, artifacts, blobs))
    return std::move(error);
  return draft;
}

llvm::Error verifyResourceTimeTransitionDeltaDigests(
    const ResourceTimeTransition &transition, const ArtifactStore &artifacts,
    const BlobStore &blobs) {
  if (llvm::Error error = requireCompletionSafePointDraft(transition))
    return error;
  auto expected = deriveTransitionDigests(transition, artifacts, blobs);
  if (!expected)
    return expected.takeError();
  if (!transition.reprogrammingTimePicoseconds ||
      !transition.migrationTimePicoseconds)
    return invalid("verified transition has no exact cost components");
  if (*transition.reprogrammingTimePicoseconds !=
      expected->execution.reprogrammingTimePicoseconds)
    return invalid("reprogramming time disagrees with the exact changed-word "
                   "projection");
  if (transition.logicalMemories != expected->logicalMemories)
    return invalid("live-state correspondence disagrees with endpoint memory "
                   "bindings");
  if (*transition.migrationTimePicoseconds !=
      expected->execution.migrationTimePicoseconds)
    return invalid("migration time disagrees with the live-state "
                   "correspondence");
  if (!transition.resourceDeltaDigest ||
      *transition.resourceDeltaDigest != expected->resources)
    return invalid("resource delta digest disagrees with endpoint semantics");
  if (!transition.configurationDeltaDigest ||
      *transition.configurationDeltaDigest != expected->configuration)
    return invalid(
        "configuration delta digest disagrees with endpoint semantics");
  if (!transition.routeDeltaDigest ||
      *transition.routeDeltaDigest != expected->routes)
    return invalid("route delta digest disagrees with endpoint semantics");
  return llvm::Error::success();
}

} // namespace loom::pnr
