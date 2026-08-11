#include "DeploymentInternal.h"

#include "Common/ArtifactLocalReference.h"
#include "Common/ArtifactStore.h"
#include "Common/ArtifactText.h"
#include "Common/BlobStore.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Dataflow/IR/DataflowReferenceCodec.h"
#include "Deployment/HardwareConfigurationImage.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/Artifact/FabricArtifactCodec.h"
#include "Fabric/Artifact/FabricSystemRootView.h"
#include "Frontend/Executable/CompilerTargetBinding.h"
#include "Frontend/Executable/InstructionCoreBinary.h"
#include "Hardware/Configuration/ConfigurationABI.h"
#include "Hardware/Implementation/HardwareImplementation.h"
#include "Mapping/Artifact/SystemMappingArtifact.h"
#include "Mapping/IR/MappingSchema.h"
#include "Runtime/RuntimePlatformBinding.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/JSON.h"

#include <algorithm>
#include <iterator>
#include <set>
#include <string>
#include <tuple>
#include <utility>

namespace loom::deployment {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "deployment_invalid: " + message);
}

bool hardwareBindingLess(const DeploymentHardwareBinding &lhs,
                         const DeploymentHardwareBinding &rhs) {
  if (artifactRootReferenceLess(lhs.hardwareImplementation,
                                rhs.hardwareImplementation))
    return true;
  if (artifactRootReferenceLess(rhs.hardwareImplementation,
                                lhs.hardwareImplementation))
    return false;
  return artifactRootReferenceLess(lhs.runtimePlatformBinding,
                                   rhs.runtimePlatformBinding);
}

llvm::Expected<std::vector<std::uint8_t>>
memoryKey(const StaticMemoryImageLeaf &memory) {
  auto launch = ::dataflow::encodeDataflowReference(
      memory.canonicalDataflow().artifact, memory.rootedGraphLaunch());
  if (!launch)
    return launch.takeError();
  auto root = ::dataflow::encodeDataflowReference(
      memory.canonicalDataflow().artifact, memory.logicalMemoryRoot());
  if (!root)
    return root.takeError();
  std::vector<std::uint8_t> result;
  result.insert(result.end(),
                memory.canonicalDataflow().artifact.bytes().begin(),
                memory.canonicalDataflow().artifact.bytes().end());
  result.insert(result.end(), launch->begin(), launch->end());
  result.insert(result.end(), root->begin(), root->end());
  return result;
}

llvm::Error canonicalizeRoots(std::vector<ArtifactRootReference> &roots,
                              llvm::StringRef context) {
  llvm::sort(roots, artifactRootReferenceLess);
  if (std::adjacent_find(roots.begin(), roots.end()) != roots.end())
    return invalid(context + " contains a duplicate reference");
  return llvm::Error::success();
}

llvm::Error requireCanonicalRoots(llvm::ArrayRef<ArtifactRootReference> roots,
                                  llvm::StringRef context) {
  if (!llvm::is_sorted(roots, artifactRootReferenceLess) ||
      std::adjacent_find(roots.begin(), roots.end()) != roots.end())
    return invalid(context + " is not sorted and unique");
  return llvm::Error::success();
}

llvm::Expected<std::vector<std::uint8_t>>
configurationImageKey(const ArtifactRootReference &reference,
                      const ArtifactStore &artifacts) {
  auto image = importHardwareConfigurationImage(reference, artifacts);
  if (!image)
    return image.takeError();
  return hardware::encodeProgrammingUnitRef(
      {image->image().configurationAbi(), image->image().programmingUnitId()});
}

llvm::Error
canonicalizeConfigurationImages(std::vector<ArtifactRootReference> &images,
                                const ArtifactStore &artifacts) {
  std::vector<std::pair<std::vector<std::uint8_t>, ArtifactRootReference>>
      keyed;
  keyed.reserve(images.size());
  for (ArtifactRootReference &image : images) {
    auto key = configurationImageKey(image, artifacts);
    if (!key)
      return key.takeError();
    keyed.emplace_back(std::move(*key), std::move(image));
  }
  llvm::sort(keyed, [](const auto &lhs, const auto &rhs) {
    return lhs.first < rhs.first;
  });
  for (std::size_t ordinal = 1; ordinal < keyed.size(); ++ordinal)
    if (keyed[ordinal - 1].first == keyed[ordinal].first)
      return invalid("configuration_image_refs repeats a programming unit");
  images.clear();
  images.reserve(keyed.size());
  for (auto &entry : keyed)
    images.push_back(std::move(entry.second));
  return llvm::Error::success();
}

llvm::Error requireCanonicalConfigurationImages(
    llvm::ArrayRef<ArtifactRootReference> images,
    const ArtifactStore &artifacts) {
  std::vector<std::uint8_t> previous;
  bool first = true;
  for (const ArtifactRootReference &image : images) {
    auto key = configurationImageKey(image, artifacts);
    if (!key)
      return key.takeError();
    if (!first && previous >= *key)
      return invalid(
          "configuration_image_refs is not sorted by programming unit");
    previous = std::move(*key);
    first = false;
  }
  return llvm::Error::success();
}

llvm::Error
canonicalizeHardwareBindings(std::vector<DeploymentHardwareBinding> &bindings) {
  llvm::sort(bindings, hardwareBindingLess);
  if (std::adjacent_find(bindings.begin(), bindings.end()) != bindings.end())
    return invalid("hardware_bindings contains a duplicate binding");
  return llvm::Error::success();
}

llvm::Error requireCanonicalHardwareBindings(
    llvm::ArrayRef<DeploymentHardwareBinding> bindings) {
  if (!llvm::is_sorted(bindings, hardwareBindingLess) ||
      std::adjacent_find(bindings.begin(), bindings.end()) != bindings.end())
    return invalid("hardware_bindings is not sorted and unique");
  return llvm::Error::success();
}

llvm::Error
canonicalizeStaticMemory(std::vector<StaticMemoryImageLeaf> &memories) {
  std::vector<std::pair<std::vector<std::uint8_t>, StaticMemoryImageLeaf>>
      keyed;
  keyed.reserve(memories.size());
  for (StaticMemoryImageLeaf &memory : memories) {
    auto key = memoryKey(memory);
    if (!key)
      return key.takeError();
    keyed.emplace_back(std::move(*key), std::move(memory));
  }
  llvm::sort(keyed, [](const auto &lhs, const auto &rhs) {
    return lhs.first < rhs.first;
  });
  for (std::size_t ordinal = 1; ordinal < keyed.size(); ++ordinal)
    if (keyed[ordinal - 1].first == keyed[ordinal].first)
      return invalid("static_memory_images contains a duplicate logical image");
  memories.clear();
  memories.reserve(keyed.size());
  for (auto &entry : keyed)
    memories.push_back(std::move(entry.second));
  return llvm::Error::success();
}

llvm::Error
requireCanonicalStaticMemory(llvm::ArrayRef<StaticMemoryImageLeaf> memories) {
  std::vector<std::uint8_t> previous;
  bool first = true;
  for (const StaticMemoryImageLeaf &memory : memories) {
    auto key = memoryKey(memory);
    if (!key)
      return key.takeError();
    if (!first && previous >= *key)
      return invalid("static_memory_images is not sorted and unique");
    previous = std::move(*key);
    first = false;
  }
  return llvm::Error::success();
}

llvm::Expected<llvm::json::Value>
parseInlineBytes(const CanonicalSemanticBytes &bytes,
                 const ArtifactSchemaDescriptor &schema) {
  llvm::StringRef text(reinterpret_cast<const char *>(bytes.bytes().data()),
                       bytes.bytes().size());
  auto value = llvm::json::parse(text);
  if (!value)
    return invalid(schema.identity + " is not valid JSON");
  const llvm::json::Object *object = value->getAsObject();
  if (!object)
    return invalid(schema.identity + " is not a JSON object");
  const auto identity = object->getString("schema");
  const auto version = object->getString("schema_version");
  if (!identity || !version || *identity != schema.identity ||
      *version != formatSchemaVersion(schema.version))
    return invalid(schema.identity + " has the wrong descriptor");
  return std::move(*value);
}

llvm::Expected<detail::ParsedDeployment>
parsedFromDraft(DeploymentDraft draft) {
  auto thread =
      parseInlineBytes(draft.threadDispatchImage, threadDispatchImageSchema);
  auto admission = parseInlineBytes(draft.admissionImage, admissionImageSchema);
  if (!thread)
    return thread.takeError();
  if (!admission)
    return admission.takeError();
  std::optional<llvm::json::Value> spatial;
  if (draft.spatialLaunchImage) {
    auto parsed =
        parseInlineBytes(*draft.spatialLaunchImage, spatialLaunchImageSchema);
    if (!parsed)
      return parsed.takeError();
    spatial = std::move(*parsed);
  }
  return detail::ParsedDeployment{std::move(draft.systemMapping),
                                  std::move(draft.hostProgram),
                                  std::move(draft.instructionCoreBinaries),
                                  std::move(draft.hardwareBindings),
                                  std::move(draft.configurationImages),
                                  std::move(draft.staticMemoryImages),
                                  std::move(*thread),
                                  std::move(spatial),
                                  std::move(*admission)};
}

llvm::Expected<
    std::pair<dataflow::CanonicalDataflowArtifact, fabric::FinalizedFabricRoot>>
importMappingOwners(const mapping::FinalizedSystemMapping &mapping,
                    const ArtifactStore &artifacts) {
  const ArtifactRootReference dataflowReference{
      dataflow::canonicalDataflowSchema.identity.str(),
      dataflow::canonicalDataflowSchema.version,
      mapping.view().dataflowIdentity()};
  const ArtifactRootReference fabricReference{
      fabric::fabricArtifactSchema.identity.str(),
      fabric::fabricArtifactSchema.version, mapping.view().fabricIdentity()};
  auto dataflow =
      dataflow::importCanonicalDataflow(dataflowReference, artifacts);
  auto fabric = fabric::importEntireFabricRoot(fabricReference, artifacts);
  if (!dataflow)
    return dataflow.takeError();
  if (!fabric)
    return fabric.takeError();
  return std::make_pair(std::move(*dataflow), std::move(*fabric));
}

llvm::Error
validateExecutableAndHardwareClosure(const detail::ParsedDeployment &deployment,
                                     const ArtifactStore &artifacts,
                                     const BlobStore &blobs) {
  if (deployment.systemMapping.schemaIdentity !=
          mapping::mappingArtifactSchema.identity ||
      deployment.systemMapping.schemaVersion !=
          mapping::mappingArtifactSchema.version)
    return invalid("system_mapping_ref has the wrong schema descriptor");
  auto systemMapping =
      mapping::importSystemMapping(deployment.systemMapping, artifacts);
  if (!systemMapping)
    return systemMapping.takeError();
  auto owners = importMappingOwners(*systemMapping, artifacts);
  if (!owners)
    return owners.takeError();
  auto dataflowView = owners->first.view();
  if (!dataflowView)
    return dataflowView.takeError();
  auto system = fabric::requireSystemRoot(owners->second.view());
  if (!system)
    return system.takeError();

  if (llvm::Error error =
          validateHostProgramLeaf(deployment.hostProgram, artifacts, blobs))
    return error;
  if (system->artifact().hostCoreOccurrences().size() != 1)
    return invalid("Fabric System does not contain exactly one HostCore");
  auto hostTarget = importCompilerTargetBinding(
      deployment.hostProgram.compilerTargetBinding(), artifacts);
  if (!hostTarget)
    return hostTarget.takeError();
  const CompilerProcessorArchitectureRef hostProcessor =
      CompilerProcessorArchitectureRef::host(
          {systemMapping->view().fabricIdentity(),
           system->artifact().hostCoreOccurrences().front()});
  if (llvm::Error error = requireCompilerTargetCompatibility(
          hostTarget->binding(), hostProcessor, artifacts))
    return error;

  if (deployment.instructionCoreBinaries.empty() &&
      !systemMapping->view().executionBindings().rootThreadLaunches().empty())
    return invalid("instruction_core_binary_refs is empty");
  for (const ArtifactRootReference &reference :
       deployment.instructionCoreBinaries) {
    auto binary = importInstructionCoreBinary(reference, artifacts, blobs);
    if (!binary)
      return binary.takeError();
    if (binary->binary().canonicalDataflow() !=
        ArtifactRootReference{dataflow::canonicalDataflowSchema.identity.str(),
                              dataflow::canonicalDataflowSchema.version,
                              dataflowView->identity()})
      return invalid("InstructionCoreBinary has a foreign Dataflow owner");
  }

  for (const StaticMemoryImageLeaf &memory : deployment.staticMemoryImages) {
    if (memory.canonicalDataflow() !=
        ArtifactRootReference{dataflow::canonicalDataflowSchema.identity.str(),
                              dataflow::canonicalDataflowSchema.version,
                              dataflowView->identity()})
      return invalid("static memory image has a foreign Dataflow owner");
    if (llvm::Error error =
            validateStaticMemoryImageLeaf(memory, artifacts, blobs))
      return error;
    const bool selected = llvm::any_of(
        systemMapping->view().executionBindings().graphBindings(),
        [&](const mapping::SystemGraphExecutionBindingView &binding) {
          return binding.key == memory.rootedGraphLaunch();
        });
    if (!selected)
      return invalid("static memory image names an unselected rooted graph "
                     "launch");
  }

  if (deployment.hardwareBindings.size() != 1)
    return invalid("hardware_bindings must select exactly one complete System "
                   "implementation");
  const DeploymentHardwareBinding &binding =
      deployment.hardwareBindings.front();
  auto implementation = hardware::importHardwareImplementation(
      binding.hardwareImplementation, artifacts, blobs);
  if (!implementation)
    return implementation.takeError();
  const ArtifactRootReference fabricReference{
      fabric::fabricArtifactSchema.identity.str(),
      fabric::fabricArtifactSchema.version,
      systemMapping->view().fabricIdentity()};
  if (implementation->implementation().fabric() != fabricReference)
    return invalid("HardwareImplementation has a foreign Fabric System");
  auto runtimeBinding = runtime::importRuntimePlatformBinding(
      binding.runtimePlatformBinding, artifacts, blobs);
  if (!runtimeBinding)
    return runtimeBinding.takeError();
  if (runtimeBinding->binding().hardwareImplementation() !=
      binding.hardwareImplementation)
    return invalid("RuntimePlatformBinding names another implementation");

  const ArtifactRootReference abiReference =
      implementation->implementation().configurationAbi();
  auto abi = hardware::importConfigurationABI(abiReference, artifacts);
  if (!abi)
    return abi.takeError();
  if (deployment.configurationImages.size() !=
      abi->abi().programmingUnits().size())
    return invalid("configuration_image_refs does not cover every programming "
                   "unit exactly once");
  std::set<hardware::ProgrammingUnitId> coveredUnits;
  for (const ArtifactRootReference &reference :
       deployment.configurationImages) {
    auto image = importHardwareConfigurationImage(reference, artifacts);
    if (!image)
      return image.takeError();
    if (image->image().configurationAbi() != abiReference)
      return invalid("configuration image has a foreign ConfigurationABI");
    if (image->image().sourceMapping().kind !=
            ConfigurationImageSourceKind::SystemMapping ||
        image->image().sourceMapping().mapping != deployment.systemMapping)
      return invalid("configuration image is not derived from the exact "
                     "SystemMapping");
    if (!coveredUnits.insert(image->image().programmingUnitId()).second)
      return invalid("configuration image set repeats a programming unit");
  }
  for (const hardware::ProgrammingUnit &unit : abi->abi().programmingUnits())
    if (!coveredUnits.count(unit.id))
      return invalid("configuration image set omits a programming unit");
  return llvm::Error::success();
}

llvm::Error requireCanonicalTopLevel(const detail::ParsedDeployment &value,
                                     const ArtifactStore &artifacts) {
  if (llvm::Error error = requireCanonicalRoots(value.instructionCoreBinaries,
                                                "instruction_core_binary_refs"))
    return error;
  if (llvm::Error error =
          requireCanonicalHardwareBindings(value.hardwareBindings))
    return error;
  if (llvm::Error error = requireCanonicalConfigurationImages(
          value.configurationImages, artifacts))
    return error;
  return requireCanonicalStaticMemory(value.staticMemoryImages);
}

llvm::Error canonicalizeDraft(DeploymentDraft &draft,
                              const ArtifactStore &artifacts) {
  if (llvm::Error error = canonicalizeRoots(draft.instructionCoreBinaries,
                                            "instruction_core_binary_refs"))
    return error;
  if (llvm::Error error = canonicalizeHardwareBindings(draft.hardwareBindings))
    return error;
  if (llvm::Error error =
          canonicalizeConfigurationImages(draft.configurationImages, artifacts))
    return error;
  return canonicalizeStaticMemory(draft.staticMemoryImages);
}

bool sameBytes(const CanonicalSemanticBytes &lhs,
               const CanonicalSemanticBytes &rhs) {
  return lhs.bytes().equals(rhs.bytes());
}

llvm::Expected<FinalizedDeployment> finishFinalizedDeployment(
    detail::ParsedDeployment parsed, detail::DerivedRuntimeImages images,
    const ArtifactStore &artifacts) {
  auto canonical = detail::serializeDeployment(parsed, images);
  if (!canonical)
    return canonical.takeError();
  auto strict = detail::parseDeployment(canonical->bytes());
  if (!strict)
    return strict.takeError();
  auto roundTrip = detail::serializeDeployment(*strict, images);
  if (!roundTrip)
    return roundTrip.takeError();
  if (!sameBytes(*canonical, *roundTrip))
    return invalid("finalized Deployment did not round-trip canonically");
  auto identity = artifacts.put(deploymentSchema, *canonical);
  if (!identity)
    return identity.takeError();
  ArtifactRootReference reference{deploymentSchema.identity.str(),
                                  deploymentSchema.version, *identity};
  Deployment deployment =
      detail::materializeDeployment(std::move(*strict), std::move(images));
  return detail::DeploymentCodecAccess::finalized(
      std::move(reference), std::move(*canonical), std::move(deployment));
}

} // namespace

namespace detail {

llvm::Expected<DerivedRuntimeImages>
validateDeploymentClosure(const ParsedDeployment &deployment,
                          const ArtifactStore &artifacts,
                          const BlobStore &blobs) {
  if (llvm::Error error = requireCanonicalTopLevel(deployment, artifacts))
    return error;
  if (llvm::Error error =
          validateExecutableAndHardwareClosure(deployment, artifacts, blobs))
    return error;
  return deriveRuntimeImages(deployment.systemMapping,
                             deployment.instructionCoreBinaries,
                             deployment.configurationImages, artifacts, blobs);
}

Deployment materializeDeployment(ParsedDeployment deployment,
                                 DerivedRuntimeImages images) {
  return DeploymentCodecAccess::deployment(
      std::move(deployment.systemMapping), std::move(deployment.hostProgram),
      std::move(deployment.instructionCoreBinaries),
      std::move(deployment.hardwareBindings),
      std::move(deployment.configurationImages),
      std::move(deployment.staticMemoryImages),
      DeploymentCodecAccess::runtimeImage(threadDispatchImageSchema,
                                          std::move(images.threadDispatch)),
      images.spatialLaunch
          ? std::optional<InlineRuntimeImage>(
                DeploymentCodecAccess::runtimeImage(
                    spatialLaunchImageSchema, std::move(*images.spatialLaunch)))
          : std::nullopt,
      DeploymentCodecAccess::runtimeImage(admissionImageSchema,
                                          std::move(images.admission)));
}

} // namespace detail

llvm::Expected<FinalizedDeployment>
finalizeDeployment(DeploymentDraft draft, const ArtifactStore &artifacts,
                   const BlobStore &blobs) {
  const CanonicalSemanticBytes authoredThread = draft.threadDispatchImage;
  const std::optional<CanonicalSemanticBytes> authoredSpatial =
      draft.spatialLaunchImage;
  const CanonicalSemanticBytes authoredAdmission = draft.admissionImage;
  if (llvm::Error error = canonicalizeDraft(draft, artifacts))
    return error;
  auto parsed = parsedFromDraft(std::move(draft));
  if (!parsed)
    return parsed.takeError();
  auto expected = detail::validateDeploymentClosure(*parsed, artifacts, blobs);
  if (!expected)
    return expected.takeError();
  if (!sameBytes(authoredThread, expected->threadDispatch))
    return invalid("thread_dispatch_image is not the exact derived child");
  if (authoredSpatial.has_value() != expected->spatialLaunch.has_value())
    return invalid("spatial_launch_image presence does not match the exact "
                   "derived closure");
  if (authoredSpatial && !sameBytes(*authoredSpatial, *expected->spatialLaunch))
    return invalid("spatial_launch_image is not the exact derived child");
  if (!sameBytes(authoredAdmission, expected->admission))
    return invalid("admission_image is not the exact derived child");
  return finishFinalizedDeployment(std::move(*parsed), std::move(*expected),
                                   artifacts);
}

llvm::Expected<FinalizedDeployment>
importDeployment(const ArtifactRootReference &reference,
                 const ArtifactStore &artifacts, const BlobStore &blobs) {
  if (reference.schemaIdentity != deploymentSchema.identity ||
      reference.schemaVersion != deploymentSchema.version)
    return invalid("root reference has the wrong schema descriptor");
  auto bytes = artifacts.get(deploymentSchema, reference.artifact);
  if (!bytes)
    return bytes.takeError();
  auto parsed = detail::parseDeployment(bytes->bytes());
  if (!parsed)
    return parsed.takeError();
  auto images = detail::validateDeploymentClosure(*parsed, artifacts, blobs);
  if (!images)
    return images.takeError();
  auto canonical = detail::serializeDeployment(*parsed, *images);
  if (!canonical)
    return canonical.takeError();
  if (!sameBytes(*bytes, *canonical))
    return invalid("stored Deployment is not canonical or has stale derived "
                   "runtime images");
  Deployment deployment =
      detail::materializeDeployment(std::move(*parsed), std::move(*images));
  return detail::DeploymentCodecAccess::finalized(reference, std::move(*bytes),
                                                  std::move(deployment));
}

llvm::Expected<FinalizedDeployment>
buildDeployment(ExactDeploymentInputs inputs, const ArtifactStore &artifacts,
                const BlobStore &blobs) {
  if (llvm::Error error = canonicalizeRoots(inputs.instructionCoreBinaries,
                                            "instruction_core_binary_refs"))
    return error;
  if (llvm::Error error = canonicalizeHardwareBindings(inputs.hardwareBindings))
    return error;
  if (llvm::Error error = canonicalizeStaticMemory(inputs.staticMemoryImages))
    return error;
  if (inputs.hardwareBindings.size() != 1)
    return invalid("buildDeployment requires one complete System hardware "
                   "binding");
  auto implementation = hardware::importHardwareImplementation(
      inputs.hardwareBindings.front().hardwareImplementation, artifacts, blobs);
  if (!implementation)
    return implementation.takeError();
  const ArtifactRootReference abiReference =
      implementation->implementation().configurationAbi();
  auto abi = hardware::importConfigurationABI(abiReference, artifacts);
  if (!abi)
    return abi.takeError();

  std::vector<ArtifactRootReference> images;
  images.reserve(abi->abi().programmingUnits().size());
  for (const hardware::ProgrammingUnit &unit : abi->abi().programmingUnits()) {
    auto image = finalizeHardwareConfigurationImage(
        {abiReference,
         unit.id,
         {ConfigurationImageSourceKind::SystemMapping, inputs.systemMapping}},
        artifacts);
    if (!image)
      return image.takeError();
    images.push_back(image->reference());
  }
  if (llvm::Error error = canonicalizeConfigurationImages(images, artifacts))
    return error;
  auto runtimeImages = detail::deriveRuntimeImages(
      inputs.systemMapping, inputs.instructionCoreBinaries, images, artifacts,
      blobs);
  if (!runtimeImages)
    return runtimeImages.takeError();
  auto thread =
      parseInlineBytes(runtimeImages->threadDispatch, threadDispatchImageSchema);
  auto admission =
      parseInlineBytes(runtimeImages->admission, admissionImageSchema);
  if (!thread)
    return thread.takeError();
  if (!admission)
    return admission.takeError();
  std::optional<llvm::json::Value> spatial;
  if (runtimeImages->spatialLaunch) {
    auto value = parseInlineBytes(*runtimeImages->spatialLaunch,
                                  spatialLaunchImageSchema);
    if (!value)
      return value.takeError();
    spatial = std::move(*value);
  }
  detail::ParsedDeployment parsed{
      std::move(inputs.systemMapping),
      std::move(inputs.hostProgram),
      std::move(inputs.instructionCoreBinaries),
      std::move(inputs.hardwareBindings),
      std::move(images),
      std::move(inputs.staticMemoryImages),
      std::move(*thread),
      std::move(spatial),
      std::move(*admission)};
  if (llvm::Error error = requireCanonicalTopLevel(parsed, artifacts))
    return error;
  if (llvm::Error error =
          validateExecutableAndHardwareClosure(parsed, artifacts, blobs))
    return error;
  return finishFinalizedDeployment(std::move(parsed),
                                   std::move(*runtimeImages), artifacts);
}

} // namespace loom::deployment
