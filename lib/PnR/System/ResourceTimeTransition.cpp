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

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <array>
#include <optional>
#include <system_error>
#include <utility>
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
  std::vector<std::uint8_t> hardwareProgramming;
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
    return invalid("Deployment hardware-programming closure is incomplete");
  llvm::SmallString<2048> hardwareStorage;
  llvm::raw_svector_ostream hardwareOutput(hardwareStorage);
  llvm::json::OStream hardwareJson(hardwareOutput);
  hardwareJson.object([&] {
    hardwareJson.attribute("hardware_bindings", *hardwareBindings);
  });
  std::vector<std::uint8_t> hardwareProgramming;
  appendBlob(hardwareProgramming, llvm::ArrayRef<std::uint8_t>(
                                       reinterpret_cast<const std::uint8_t *>(
                                           hardwareStorage.data()),
                                       hardwareStorage.size()));
  appendBlob(hardwareProgramming, *images);
  return DeploymentConfigurationState{std::move(complete),
                                      std::move(hardwareProgramming)};
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

bool hasNoPersistentLiveState(
    const ::dataflow::CanonicalDataflowArtifact &artifact,
    const ::dataflow::CanonicalDataflowProgramView &dataflow) {
  if (!dataflow.logicalMemoryRoots().empty())
    return false;
  bool closed = true;
  artifact.module()->walk([&](mlir::Operation *operation) {
    if (auto thread = llvm::dyn_cast<::dataflow::ThreadOp>(operation)) {
      if (thread.getDomain().getKind() ==
              ::dataflow::ThreadDomainKind::DynamicWork)
        closed = false;
      for (mlir::Type type : thread.getFunctionType().getInputs())
        if (containsChannelType(type))
          closed = false;
    }
    for (mlir::Value operand : operation->getOperands())
      if (containsChannelType(operand.getType()))
        closed = false;
    for (mlir::Value result : operation->getResults())
      if (containsChannelType(result.getType()))
        closed = false;
  });
  return closed;
}

struct DerivedTransitionDigests final {
  ComponentViewDigest resources;
  ComponentViewDigest configuration;
  ComponentViewDigest routes;
  bool hardwareProgrammingChanged = false;
  bool noPersistentLiveState = false;
  bool completionFrontierIsCausallyAdmissible = false;
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
  return DerivedTransitionDigests{
      *resources,
      *configuration,
      *routes,
      beforeConfiguration->hardwareProgramming !=
          afterConfiguration->hardwareProgramming,
      hasNoPersistentLiveState(*dataflowArtifact, *dataflow),
      *completionFrontier};
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
  if (!transition.beforeLiveWork.empty() || !transition.afterLiveWork.empty() ||
      transition.tokenLiveStateCorrespondence)
    return invalid("completion transition has unproved live or token state");
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
  if (llvm::Error error = requireCompletionSafePointDraft(draft))
    return std::move(error);
  if (llvm::Error error = validateResourceTimeTransition(draft))
    return std::move(error);
  auto digests = deriveTransitionDigests(draft, artifacts, blobs);
  if (!digests)
    return digests.takeError();
  if (!digests->completionFrontierIsCausallyAdmissible)
    return invalid("completion frontier is not causally admissible under the "
                   "canonical Dataflow event relation");
  if (!digests->noPersistentLiveState)
    return invalid("Canonical Dataflow has persistent live state without a "
                   "migration proof owner");
  if (digests->hardwareProgrammingChanged)
    return invalid("changed Deployment hardware-programming state has no exact "
                   "reprogramming-time owner");
  draft.resourceDeltaDigest = digests->resources;
  draft.configurationDeltaDigest = digests->configuration;
  draft.routeDeltaDigest = digests->routes;
  draft.reprogrammingTimePicoseconds = 0;
  draft.migrationTimePicoseconds = 0;
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
  if (!expected->completionFrontierIsCausallyAdmissible)
    return invalid("completion frontier is not causally admissible under the "
                   "canonical Dataflow event relation");
  if (!expected->noPersistentLiveState)
    return invalid("Canonical Dataflow has persistent live state without a "
                   "migration proof owner");
  if (expected->hardwareProgrammingChanged)
    return invalid("changed Deployment hardware-programming state has no exact "
                   "reprogramming-time owner");
  if (!transition.reprogrammingTimePicoseconds ||
      !transition.migrationTimePicoseconds)
    return invalid("verified transition has no exact cost components");
  if (*transition.reprogrammingTimePicoseconds != 0)
    return invalid("unchanged Deployment hardware-programming state has "
                   "nonzero "
                   "reprogramming time");
  if (*transition.migrationTimePicoseconds != 0)
    return invalid("completion-only transition cannot claim live-state "
                   "migration work");
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
