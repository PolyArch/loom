#include "DeploymentInternal.h"

#include "Common/ArtifactLocalReference.h"
#include "Common/ArtifactStore.h"
#include "Common/ArtifactText.h"
#include "Common/BlobStore.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Dataflow/IR/DataflowReferenceCodec.h"
#include "Deployment/DeploymentDiagnostics.h"
#include "Deployment/HardwareConfigurationImage.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/Artifact/FabricArtifactCodec.h"
#include "Fabric/Artifact/FabricSystemRootView.h"
#include "Fabric/Identity/FabricRefBytes.h"
#include "Frontend/Executable/CompilerTargetBinding.h"
#include "Frontend/Executable/InstructionCoreBinary.h"
#include "Hardware/Configuration/ConfigurationABI.h"
#include "Hardware/Implementation/HardwareImplementation.h"
#include "Mapping/Artifact/SystemMappingArtifact.h"
#include "Mapping/Artifact/SystemMappingExecutionProjection.h"
#include "Mapping/IR/MappingSchema.h"
#include "Runtime/RuntimePlatformBinding.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/JSON.h"

#include <sys/resource.h>
#include <time.h>

#include <algorithm>
#include <chrono>
#include <iterator>
#include <limits>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <utility>

namespace loom::deployment {
namespace {

using MonotonicClock = std::chrono::steady_clock;

struct OperationResourceSnapshot final {
  MonotonicClock::time_point wall;
  std::optional<std::uint64_t> selfCpuNanoseconds;
  std::optional<std::uint64_t> childCpuNanoseconds;
};

std::optional<std::uint64_t> timevalNanoseconds(const timeval &value) {
  if (value.tv_sec < 0 || value.tv_usec < 0 || value.tv_usec >= 1'000'000)
    return std::nullopt;
  constexpr std::uint64_t nanosecondsPerSecond = 1'000'000'000;
  const std::uint64_t subsecond =
      static_cast<std::uint64_t>(value.tv_usec) * 1000;
  const std::uint64_t seconds = value.tv_sec;
  if (seconds > (std::numeric_limits<std::uint64_t>::max() - subsecond) /
                    nanosecondsPerSecond)
    return std::nullopt;
  return seconds * nanosecondsPerSecond + subsecond;
}

std::optional<std::uint64_t> processCpuNanoseconds() {
  timespec current{};
  if (::clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &current) != 0 ||
      current.tv_sec < 0 || current.tv_nsec < 0 ||
      current.tv_nsec >= 1'000'000'000)
    return std::nullopt;
  constexpr std::uint64_t nanosecondsPerSecond = 1'000'000'000;
  const std::uint64_t seconds = current.tv_sec;
  if (seconds > (std::numeric_limits<std::uint64_t>::max() -
                 static_cast<std::uint64_t>(current.tv_nsec)) /
                    nanosecondsPerSecond)
    return std::nullopt;
  return seconds * nanosecondsPerSecond + current.tv_nsec;
}

std::optional<std::uint64_t> childCpuNanoseconds() {
  rusage usage{};
  if (::getrusage(RUSAGE_CHILDREN, &usage) != 0)
    return std::nullopt;
  auto user = timevalNanoseconds(usage.ru_utime);
  auto system = timevalNanoseconds(usage.ru_stime);
  if (!user || !system ||
      *system > std::numeric_limits<std::uint64_t>::max() - *user)
    return std::nullopt;
  return *user + *system;
}

OperationResourceSnapshot captureOperationResources() {
  return {MonotonicClock::now(), processCpuNanoseconds(),
          childCpuNanoseconds()};
}

std::optional<std::uint64_t> elapsedCpu(std::optional<std::uint64_t> end,
                                        std::optional<std::uint64_t> begin) {
  if (!begin || !end || *end < *begin)
    return std::nullopt;
  return *end - *begin;
}

void emitElapsed(DeploymentConstructionMode mode,
                 DeploymentConstructionOperation operation,
                 const OperationResourceSnapshot &begin,
                 std::uint64_t deterministicWork = 1) {
  const OperationResourceSnapshot end = captureOperationResources();
  const std::uint64_t duration =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end.wall -
                                                           begin.wall)
          .count();
  emitDeploymentConstructionOperationStatistics(
      {mode, operation, duration, deterministicWork,
       elapsedCpu(end.selfCpuNanoseconds, begin.selfCpuNanoseconds),
       elapsedCpu(end.childCpuNanoseconds, begin.childCpuNanoseconds)});
}

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

llvm::Expected<std::optional<hardware::FinalizedConfigurationABI>>
validateHardwareBindingCoverage(
    llvm::ArrayRef<DeploymentHardwareBinding> bindings,
    llvm::ArrayRef<fabric::SpatialCoreOccurrenceRef> requiredSubjects,
    const ArtifactRootReference &fabricReference,
    const ArtifactStore &artifacts, const BlobStore &blobs) {
  if (bindings.size() != requiredSubjects.size())
    return invalid("hardware_bindings does not exactly cover the SpatialCore "
                   "occurrences selected by SystemMapping");

  std::vector<fabric::SpatialCoreOccurrenceRef> actualSubjects;
  actualSubjects.reserve(bindings.size());
  std::optional<ArtifactRootReference> commonAbi;
  std::optional<runtime::RuntimeProviderBinding> commonProvider;
  for (const DeploymentHardwareBinding &binding : bindings) {
    auto implementation = hardware::importHardwareImplementation(
        binding.hardwareImplementation, artifacts, blobs);
    if (!implementation)
      return implementation.takeError();
    if (implementation->implementation().fabric() != fabricReference)
      return invalid("HardwareImplementation has a foreign Fabric System");
    actualSubjects.push_back(implementation->implementation().subject());
    if (commonAbi &&
        *commonAbi != implementation->implementation().configurationAbi())
      return invalid("hardware_bindings do not share one ConfigurationABI");
    commonAbi = implementation->implementation().configurationAbi();

    auto runtimeBinding = runtime::importRuntimePlatformBinding(
        binding.runtimePlatformBinding, artifacts, blobs);
    if (!runtimeBinding)
      return runtimeBinding.takeError();
    if (runtimeBinding->binding().hardwareImplementation() !=
        binding.hardwareImplementation)
      return invalid("RuntimePlatformBinding names another implementation");
    if (commonProvider &&
        !(*commonProvider == runtimeBinding->binding().providerBinding()))
      return invalid("hardware_bindings do not share one Runtime provider "
                     "contract");
    commonProvider = runtimeBinding->binding().providerBinding();
  }

  llvm::sort(actualSubjects, [](fabric::SpatialCoreOccurrenceRef lhs,
                                fabric::SpatialCoreOccurrenceRef rhs) {
    return fabric::canonicalFabricBytes(lhs) <
           fabric::canonicalFabricBytes(rhs);
  });
  if (std::adjacent_find(actualSubjects.begin(), actualSubjects.end()) !=
      actualSubjects.end())
    return invalid("hardware_bindings repeat a SpatialCore occurrence");
  if (llvm::ArrayRef<fabric::SpatialCoreOccurrenceRef>(actualSubjects) !=
      requiredSubjects)
    return invalid("hardware_bindings select the wrong SpatialCore "
                   "occurrences");
  if (!commonAbi)
    return std::optional<hardware::FinalizedConfigurationABI>{};
  auto abi = hardware::importConfigurationABI(*commonAbi, artifacts);
  if (!abi)
    return abi.takeError();
  return std::optional<hardware::FinalizedConfigurationABI>(std::move(*abi));
}

bool containsSubject(llvm::ArrayRef<fabric::SpatialCoreOccurrenceRef> subjects,
                     fabric::SpatialCoreOccurrenceRef subject) {
  return llvm::is_contained(subjects, subject);
}

bool hasDirectSystemConfiguration(const fabric::FabricSystemRootView &system) {
  for (fabric::SystemTransportResourceRef resource :
       system.transportResources()) {
    const fabric::FabricInventoryOwnerRef owner =
        fabric::FabricInventoryOwnerRef::of(resource);
    if (system.artifact().inventorySize(
            owner, fabric::FabricInventoryKind::SemanticConfigField) != 0)
      return true;
  }
  return false;
}

llvm::Expected<std::vector<hardware::ProgrammingUnitId>>
requiredProgrammingUnits(
    const hardware::ConfigurationABI &abi,
    llvm::ArrayRef<fabric::SpatialCoreOccurrenceRef> requiredSubjects) {
  std::vector<hardware::ProgrammingUnitId> result;
  for (const hardware::ProgrammingUnit &unit : abi.programmingUnits()) {
    const hardware::ProgrammingUnitOccurrenceScope scope =
        hardware::deriveProgrammingUnitOccurrenceScope(unit);
    if (scope.includesDirectSystemResources) {
      if (!scope.spatialCores.empty())
        return invalid("a direct System programming unit also names a "
                       "SpatialCore occurrence");
      result.push_back(unit.id);
      continue;
    }
    const bool touchesRequired = llvm::any_of(
        scope.spatialCores, [&](fabric::SpatialCoreOccurrenceRef subject) {
          return containsSubject(requiredSubjects, subject);
        });
    if (!touchesRequired)
      continue;
    if (scope.spatialCores.size() != 1)
      return invalid("a programming unit required by SystemMapping crosses "
                     "its SpatialCore occurrence");
    result.push_back(unit.id);
  }
  return result;
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

  const ArtifactRootReference fabricReference{
      fabric::fabricArtifactSchema.identity.str(),
      fabric::fabricArtifactSchema.version,
      systemMapping->view().fabricIdentity()};
  auto subjects = mapping::projectSystemExecutionSpatialCoreSubjects(
      *dataflowView, systemMapping->view().executionBindings());
  if (!subjects)
    return subjects.takeError();
  auto abi =
      validateHardwareBindingCoverage(deployment.hardwareBindings, *subjects,
                                      fabricReference, artifacts, blobs);
  if (!abi)
    return abi.takeError();
  if (!*abi) {
    if (!deployment.configurationImages.empty())
      return invalid("configuration_image_refs is nonempty without a selected "
                     "SpatialCore implementation");
    if (hasDirectSystemConfiguration(*system))
      return invalid("direct System configuration requires a selected "
                     "ConfigurationABI provider");
    return llvm::Error::success();
  }
  auto requiredUnits = requiredProgrammingUnits((*abi)->abi(), *subjects);
  if (!requiredUnits)
    return requiredUnits.takeError();
  if (deployment.configurationImages.size() != requiredUnits->size())
    return invalid("configuration_image_refs does not cover every programming "
                   "unit required by the selected SpatialCore occurrences "
                   "exactly once");
  const ArtifactRootReference abiReference = (*abi)->reference();
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
  for (hardware::ProgrammingUnitId unit : *requiredUnits)
    if (!coveredUnits.count(unit))
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

llvm::Expected<FinalizedDeployment>
finishFinalizedDeployment(detail::ParsedDeployment parsed,
                          detail::DerivedRuntimeImages images,
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

llvm::Expected<detail::DerivedRuntimeImages> validateDeploymentClosureImpl(
    const detail::ParsedDeployment &deployment, const ArtifactStore &artifacts,
    const BlobStore &blobs,
    std::optional<DeploymentConstructionMode> statisticsMode) {
  auto operationBegin = captureOperationResources();
  if (llvm::Error error = requireCanonicalTopLevel(deployment, artifacts)) {
    if (statisticsMode)
      emitElapsed(*statisticsMode,
                  DeploymentConstructionOperation::ExecutableClosureValidation,
                  operationBegin);
    return std::move(error);
  }
  if (llvm::Error error =
          validateExecutableAndHardwareClosure(deployment, artifacts, blobs)) {
    if (statisticsMode)
      emitElapsed(*statisticsMode,
                  DeploymentConstructionOperation::ExecutableClosureValidation,
                  operationBegin);
    return std::move(error);
  }
  if (statisticsMode)
    emitElapsed(*statisticsMode,
                DeploymentConstructionOperation::ExecutableClosureValidation,
                operationBegin);

  operationBegin = captureOperationResources();
  auto images = detail::deriveRuntimeImages(
      deployment.systemMapping, deployment.instructionCoreBinaries,
      deployment.configurationImages, artifacts, blobs);
  if (statisticsMode)
    emitElapsed(*statisticsMode,
                DeploymentConstructionOperation::RuntimeImageDerivation,
                operationBegin);
  return images;
}

} // namespace

namespace detail {

llvm::Expected<DerivedRuntimeImages>
validateDeploymentClosure(const ParsedDeployment &deployment,
                          const ArtifactStore &artifacts,
                          const BlobStore &blobs) {
  return validateDeploymentClosureImpl(deployment, artifacts, blobs,
                                       std::nullopt);
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
  auto operationBegin = captureOperationResources();
  auto bytes = artifacts.get(deploymentSchema, reference.artifact);
  if (!bytes)
    return bytes.takeError();
  auto parsed = detail::parseDeployment(bytes->bytes());
  emitElapsed(DeploymentConstructionMode::Import,
              DeploymentConstructionOperation::InputCanonicalization,
              operationBegin);
  if (!parsed)
    return parsed.takeError();
  auto images = validateDeploymentClosureImpl(
      *parsed, artifacts, blobs, DeploymentConstructionMode::Import);
  if (!images)
    return images.takeError();
  operationBegin = captureOperationResources();
  auto canonical = detail::serializeDeployment(*parsed, *images);
  if (!canonical)
    return canonical.takeError();
  if (!sameBytes(*bytes, *canonical))
    return invalid("stored Deployment is not canonical or has stale derived "
                   "runtime images");
  Deployment deployment =
      detail::materializeDeployment(std::move(*parsed), std::move(*images));
  emitElapsed(DeploymentConstructionMode::Import,
              DeploymentConstructionOperation::ArtifactFinalization,
              operationBegin);
  return detail::DeploymentCodecAccess::finalized(reference, std::move(*bytes),
                                                  std::move(deployment));
}

llvm::Expected<FinalizedDeployment>
buildDeployment(ExactDeploymentInputs inputs, const ArtifactStore &artifacts,
                const BlobStore &blobs) {
  auto operationBegin = captureOperationResources();
  if (llvm::Error error = canonicalizeRoots(inputs.instructionCoreBinaries,
                                            "instruction_core_binary_refs"))
    return error;
  if (llvm::Error error = canonicalizeHardwareBindings(inputs.hardwareBindings))
    return error;
  if (llvm::Error error = canonicalizeStaticMemory(inputs.staticMemoryImages))
    return error;
  emitElapsed(DeploymentConstructionMode::Build,
              DeploymentConstructionOperation::InputCanonicalization,
              operationBegin);
  operationBegin = captureOperationResources();
  auto systemMapping =
      mapping::importSystemMapping(inputs.systemMapping, artifacts);
  if (!systemMapping)
    return systemMapping.takeError();
  auto owners = importMappingOwners(*systemMapping, artifacts);
  if (!owners)
    return owners.takeError();
  auto dataflow = owners->first.view();
  if (!dataflow)
    return dataflow.takeError();
  auto system = fabric::requireSystemRoot(owners->second.view());
  if (!system)
    return system.takeError();
  auto subjects = mapping::projectSystemExecutionSpatialCoreSubjects(
      *dataflow, systemMapping->view().executionBindings());
  if (!subjects)
    return subjects.takeError();
  emitElapsed(DeploymentConstructionMode::Build,
              DeploymentConstructionOperation::MappingOwnerImport,
              operationBegin);
  const ArtifactRootReference fabricReference{
      fabric::fabricArtifactSchema.identity.str(),
      fabric::fabricArtifactSchema.version,
      systemMapping->view().fabricIdentity()};
  operationBegin = captureOperationResources();
  auto abi = validateHardwareBindingCoverage(inputs.hardwareBindings, *subjects,
                                             fabricReference, artifacts, blobs);
  emitElapsed(DeploymentConstructionMode::Build,
              DeploymentConstructionOperation::HardwareClosureValidation,
              operationBegin, inputs.hardwareBindings.size());
  if (!abi)
    return abi.takeError();
  if (!*abi && hasDirectSystemConfiguration(*system))
    return invalid("direct System configuration requires a selected "
                   "ConfigurationABI provider");

  std::vector<ArtifactRootReference> images;
  operationBegin = captureOperationResources();
  if (*abi) {
    auto units = requiredProgrammingUnits((*abi)->abi(), *subjects);
    if (!units)
      return units.takeError();
    images.reserve(units->size());
    for (hardware::ProgrammingUnitId unit : *units) {
      auto image = finalizeHardwareConfigurationImage(
          {(*abi)->reference(),
           unit,
           {ConfigurationImageSourceKind::SystemMapping, inputs.systemMapping}},
          artifacts);
      if (!image)
        return image.takeError();
      images.push_back(image->reference());
    }
  }
  if (llvm::Error error = canonicalizeConfigurationImages(images, artifacts))
    return error;
  emitElapsed(DeploymentConstructionMode::Build,
              DeploymentConstructionOperation::ConfigurationImageDerivation,
              operationBegin, images.size());
  operationBegin = captureOperationResources();
  auto runtimeImages = detail::deriveRuntimeImages(
      inputs.systemMapping, inputs.instructionCoreBinaries, images, artifacts,
      blobs);
  if (!runtimeImages)
    return runtimeImages.takeError();
  emitElapsed(DeploymentConstructionMode::Build,
              DeploymentConstructionOperation::RuntimeImageDerivation,
              operationBegin);
  operationBegin = captureOperationResources();
  auto thread = parseInlineBytes(runtimeImages->threadDispatch,
                                 threadDispatchImageSchema);
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
  detail::ParsedDeployment parsed{std::move(inputs.systemMapping),
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
  emitElapsed(DeploymentConstructionMode::Build,
              DeploymentConstructionOperation::ExecutableClosureValidation,
              operationBegin);
  operationBegin = captureOperationResources();
  auto finalized = finishFinalizedDeployment(
      std::move(parsed), std::move(*runtimeImages), artifacts);
  emitElapsed(DeploymentConstructionMode::Build,
              DeploymentConstructionOperation::ArtifactFinalization,
              operationBegin);
  return finalized;
}

} // namespace loom::deployment
