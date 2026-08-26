#include "Runtime/DeploymentLoader.h"

#include "Common/ArtifactLocalReference.h"
#include "Common/ArtifactStore.h"
#include "Common/BlobStore.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Dataflow/IR/DataflowEventDerivation.h"
#include "Dataflow/IR/DataflowReferenceCodec.h"
#include "Deployment/HardwareConfigurationImage.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/Artifact/FabricSystemRootView.h"
#include "Fabric/Identity/FabricRefBytes.h"
#include "Hardware/Configuration/ConfigurationABI.h"
#include "Hardware/Implementation/HardwareImplementation.h"
#include "Hardware/RTL/ConfigurationTransport.h"
#include "Mapping/Artifact/MappingArtifact.h"
#include "Mapping/Artifact/SystemMappingClosureProjection.h"
#include "Runtime/RuntimePlatformBinding.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <system_error>
#include <tuple>
#include <utility>

namespace loom::runtime {
namespace {

using ByteVector = std::vector<std::uint8_t>;

llvm::Error loadError(RuntimeLoadFailureKind kind, const llvm::Twine &message,
                      bool quarantined = false) {
  return llvm::make_error<RuntimeLoadError>(kind, message.str(), quarantined);
}

llvm::Error
activationReplacementError(RuntimeActivationReplacementErrorReason reason,
                           const llvm::Twine &message) {
  return llvm::make_error<RuntimeActivationReplacementError>(reason,
                                                             message.str());
}

void appendU32(ByteVector &bytes, std::uint32_t value) {
  for (unsigned shift = 24;; shift -= 8) {
    bytes.push_back(static_cast<std::uint8_t>(value >> shift));
    if (shift == 0)
      break;
  }
}

void appendU64(ByteVector &bytes, std::uint64_t value) {
  for (unsigned shift = 56;; shift -= 8) {
    bytes.push_back(static_cast<std::uint8_t>(value >> shift));
    if (shift == 0)
      break;
  }
}

void appendFramed(ByteVector &bytes, llvm::ArrayRef<std::uint8_t> value) {
  appendU64(bytes, value.size());
  bytes.insert(bytes.end(), value.begin(), value.end());
}

ByteVector
transportLayoutKey(const hardware::rtl::ConfigurationTransportLayout &layout) {
  ByteVector bytes;
  appendU64(bytes, layout.units.size());
  for (const hardware::rtl::ConfigurationTransportUnitLayout &unit :
       layout.units) {
    appendU64(bytes, unit.payloadBitCount);
    appendU64(bytes, unit.payloadByteCount);
    appendU64(bytes, unit.payloadWordCount);
    appendU32(bytes, unit.baseAddress);
    appendU32(bytes, unit.commitAddress);
    appendU32(bytes, unit.statusAddress);
    appendFramed(bytes, unit.inactiveImage);
  }
  appendU64(bytes, layout.byteSpan);
  return bytes;
}

std::uint32_t imageWord(llvm::ArrayRef<std::uint8_t> image,
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

std::uint8_t imageStrobe(llvm::ArrayRef<std::uint8_t> image,
                         std::uint64_t word) {
  const std::uint64_t firstByte = word * 4;
  if (firstByte >= image.size())
    return 0;
  const unsigned count = static_cast<unsigned>(
      std::min<std::uint64_t>(4, image.size() - firstByte));
  return static_cast<std::uint8_t>((1U << count) - 1U);
}

const RuntimeProgrammingBinding *
findProgrammingBinding(const RuntimePlatformBinding &binding,
                       const hardware::ProgrammingUnitRef &unit) {
  const auto found =
      llvm::find_if(binding.programmingBindings(),
                    [&](const RuntimeProgrammingBinding &candidate) {
                      return candidate.programmingUnit == unit;
                    });
  return found == binding.programmingBindings().end() ? nullptr : &*found;
}

struct ImportedRuntimeClosure final {
  std::vector<hardware::FinalizedHardwareImplementation> implementations;
  hardware::FinalizedConfigurationABI configurationAbi;
  std::vector<FinalizedRuntimePlatformBinding> runtimeBindings;
  mapping::FinalizedSystemMapping systemMapping;
  dataflow::CanonicalDataflowArtifact dataflowArtifact;
  dataflow::CanonicalDataflowProgramView dataflow;
  fabric::FinalizedFabricRoot fabricArtifact;
  fabric::FabricSystemRootView fabric;
  mapping::SystemMappingClosureProjection mappingClosure;
};

llvm::Expected<ImportedRuntimeClosure>
importRuntimeClosure(const deployment::FinalizedDeployment &deployment,
                     const ArtifactStore &artifacts, const BlobStore &blobs) {
  if (deployment.deployment().hardwareBindings().empty())
    return loadError(RuntimeLoadFailureKind::InvalidDeployment,
                     "Deployment has no runtime hardware binding");
  std::vector<hardware::FinalizedHardwareImplementation> implementations;
  std::vector<FinalizedRuntimePlatformBinding> runtimeBindings;
  implementations.reserve(deployment.deployment().hardwareBindings().size());
  runtimeBindings.reserve(deployment.deployment().hardwareBindings().size());
  std::optional<ArtifactRootReference> abiReference;
  for (const deployment::DeploymentHardwareBinding &hardwareBinding :
       deployment.deployment().hardwareBindings()) {
    auto implementation = hardware::importHardwareImplementation(
        hardwareBinding.hardwareImplementation, artifacts, blobs);
    if (!implementation)
      return loadError(RuntimeLoadFailureKind::InvalidDeployment,
                       "cannot import HardwareImplementation: " +
                           llvm::toString(implementation.takeError()));
    if (abiReference &&
        *abiReference != implementation->implementation().configurationAbi())
      return loadError(RuntimeLoadFailureKind::InvalidDeployment,
                       "HardwareImplementations do not share one "
                       "ConfigurationABI");
    abiReference = implementation->implementation().configurationAbi();
    auto runtimeBinding = importRuntimePlatformBinding(
        hardwareBinding.runtimePlatformBinding, artifacts, blobs);
    if (!runtimeBinding)
      return loadError(RuntimeLoadFailureKind::InvalidDeployment,
                       "cannot import RuntimePlatformBinding: " +
                           llvm::toString(runtimeBinding.takeError()));
    if (runtimeBinding->binding().hardwareImplementation() !=
        implementation->reference())
      return loadError(RuntimeLoadFailureKind::InvalidDeployment,
                       "RuntimePlatformBinding names another implementation");
    implementations.push_back(std::move(*implementation));
    runtimeBindings.push_back(std::move(*runtimeBinding));
  }
  assert(abiReference && "nonempty implementation set has one ABI");
  auto abi = hardware::importConfigurationABI(*abiReference, artifacts);
  if (!abi)
    return loadError(RuntimeLoadFailureKind::InvalidDeployment,
                     "cannot import ConfigurationABI: " +
                         llvm::toString(abi.takeError()));

  auto systemMapping = mapping::importSystemMapping(
      deployment.deployment().systemMapping(), artifacts);
  if (!systemMapping)
    return loadError(RuntimeLoadFailureKind::InvalidDeployment,
                     "cannot import SystemMapping: " +
                         llvm::toString(systemMapping.takeError()));
  const ArtifactRootReference dataflowReference{
      dataflow::canonicalDataflowSchema.identity.str(),
      dataflow::canonicalDataflowSchema.version,
      systemMapping->view().dataflowIdentity()};
  auto dataflowArtifact =
      dataflow::importCanonicalDataflow(dataflowReference, artifacts);
  if (!dataflowArtifact)
    return loadError(RuntimeLoadFailureKind::InvalidDeployment,
                     "cannot import Canonical Dataflow: " +
                         llvm::toString(dataflowArtifact.takeError()));
  auto dataflowView = dataflowArtifact->view();
  if (!dataflowView)
    return loadError(RuntimeLoadFailureKind::InvalidDeployment,
                     "cannot reconstruct Canonical Dataflow: " +
                         llvm::toString(dataflowView.takeError()));

  const ArtifactRootReference fabricReference{
      fabric::fabricArtifactSchema.identity.str(),
      fabric::fabricArtifactSchema.version,
      systemMapping->view().fabricIdentity()};
  for (const hardware::FinalizedHardwareImplementation &implementation :
       implementations)
    if (implementation.implementation().fabric() != fabricReference)
      return loadError(RuntimeLoadFailureKind::InvalidDeployment,
                       "SystemMapping and HardwareImplementation use "
                       "different Fabric roots");
  auto fabricArtifact =
      fabric::importEntireFabricRoot(fabricReference, artifacts);
  if (!fabricArtifact)
    return loadError(RuntimeLoadFailureKind::InvalidDeployment,
                     "cannot import Fabric System: " +
                         llvm::toString(fabricArtifact.takeError()));
  auto system = fabric::requireSystemRoot(fabricArtifact->view());
  if (!system)
    return loadError(RuntimeLoadFailureKind::InvalidDeployment,
                     "Fabric root is not a System: " +
                         llvm::toString(system.takeError()));
  auto closure = mapping::projectSystemMappingClosure(
      *dataflowView, *system, systemMapping->view(), artifacts);
  if (!closure)
    return loadError(RuntimeLoadFailureKind::InvalidDeployment,
                     "cannot project SystemMapping closure: " +
                         llvm::toString(closure.takeError()));

  return ImportedRuntimeClosure{
      std::move(implementations),   std::move(*abi),
      std::move(runtimeBindings),   std::move(*systemMapping),
      std::move(*dataflowArtifact), std::move(*dataflowView),
      std::move(*fabricArtifact),   std::move(*system),
      std::move(*closure)};
}

llvm::Error
verifyProviderIdentity(RuntimeProviderInstance &provider,
                       const RuntimeDeviceHandle &device,
                       const RuntimeIdentityVerification &verification,
                       const ArtifactIdentity &expectedImplementation) {
  if (const auto *reported =
          std::get_if<HardwareReportedIdentity>(&verification)) {
    auto identity = provider.readImplementationIdentity(
        device, reported->implementationIdentityEndpoint);
    if (!identity)
      return loadError(RuntimeLoadFailureKind::IdentityVerification,
                       "provider identity read failed: " +
                           llvm::toString(identity.takeError()));
    if (*identity != expectedImplementation)
      return loadError(RuntimeLoadFailureKind::IdentityVerification,
                       "provider reported a foreign HardwareImplementation");
    return llvm::Error::success();
  }
  const BlobDigest &expected =
      std::get<TrustedImmutableIdentity>(verification).attestationBlob;
  auto actual = provider.readTrustedAttestation(device);
  if (!actual)
    return loadError(RuntimeLoadFailureKind::IdentityVerification,
                     "provider attestation read failed: " +
                         llvm::toString(actual.takeError()));
  if (*actual != expected)
    return loadError(RuntimeLoadFailureKind::IdentityVerification,
                     "provider returned a stale trusted attestation");
  return llvm::Error::success();
}

struct RuntimeIdentityClaim final {
  RuntimeIdentityVerification verification;
  ArtifactIdentity expectedImplementation;
};

llvm::Error
verifyProviderIdentities(RuntimeProviderInstance &provider,
                         const RuntimeDeviceHandle &device,
                         llvm::ArrayRef<RuntimeIdentityClaim> claims) {
  for (const RuntimeIdentityClaim &claim : claims)
    if (llvm::Error error = verifyProviderIdentity(
            provider, device, claim.verification, claim.expectedImplementation))
      return error;
  return llvm::Error::success();
}

llvm::Expected<std::vector<RuntimeConfigurationTarget>>
configurationTargets(const deployment::FinalizedDeployment &deployment,
                     const ImportedRuntimeClosure &closure,
                     const ArtifactStore &artifacts) {
  std::vector<RuntimeConfigurationTarget> result;
  result.reserve(deployment.deployment().configurationImages().size());
  for (const ArtifactRootReference &imageReference :
       deployment.deployment().configurationImages()) {
    auto image =
        deployment::importHardwareConfigurationImage(imageReference, artifacts);
    if (!image)
      return loadError(RuntimeLoadFailureKind::InvalidDeployment,
                       "cannot import configuration image: " +
                           llvm::toString(image.takeError()));
    const hardware::ProgrammingUnitRef unitRef{
        closure.configurationAbi.reference(),
        image->image().programmingUnitId()};
    const hardware::ProgrammingUnit *programmingUnit =
        closure.configurationAbi.abi().findProgrammingUnit(unitRef.unitId);
    if (!programmingUnit)
      return loadError(RuntimeLoadFailureKind::InvalidDeployment,
                       "configuration image names a missing programming "
                       "unit");
    const hardware::ProgrammingUnitOccurrenceScope scope =
        hardware::deriveProgrammingUnitOccurrenceScope(*programmingUnit);
    if (scope.includesDirectSystemResources)
      return loadError(RuntimeLoadFailureKind::ProviderMismatch,
                       "runtime provider has no direct System configuration "
                       "binding");
    const RuntimeProgrammingBinding *binding = nullptr;
    const hardware::FinalizedHardwareImplementation *implementation = nullptr;
    for (const auto indexed : llvm::enumerate(closure.runtimeBindings)) {
      const RuntimeProgrammingBinding *candidate =
          findProgrammingBinding(indexed.value().binding(), unitRef);
      if (!candidate)
        continue;
      if (binding)
        return loadError(RuntimeLoadFailureKind::InvalidDeployment,
                         "configuration image has multiple runtime "
                         "programming bindings");
      binding = candidate;
      implementation = &closure.implementations[indexed.index()];
    }
    if (!binding)
      return loadError(RuntimeLoadFailureKind::InvalidDeployment,
                       "configuration image has no runtime programming "
                       "binding");
    assert(implementation && "programming binding has an implementation");
    const fabric::SpatialCoreOccurrenceRef selectedCore =
        implementation->implementation().subject();
    auto selectedLayout =
        hardware::rtl::derivePortableConfigurationTransportLayout(
            closure.configurationAbi, selectedCore);
    if (!selectedLayout)
      return loadError(RuntimeLoadFailureKind::InvalidDeployment,
                       "cannot derive configuration transport layout: " +
                           llvm::toString(selectedLayout.takeError()));
    if (!selectedLayout->find(unitRef.unitId))
      return loadError(RuntimeLoadFailureKind::InvalidDeployment,
                       "configuration image is absent from its implementation "
                       "transport");
    const hardware::rtl::ConfigurationTransportUnitLayout *unit =
        selectedLayout->find(unitRef.unitId);
    assert(unit && "selected layout must contain the programming unit");
    if (image->image().payloadBitCount() != unit->payloadBitCount ||
        image->image().payload().size() != unit->payloadByteCount)
      return loadError(RuntimeLoadFailureKind::InvalidDeployment,
                       "configuration image disagrees with its transport "
                       "layout");
    if (unit->payloadBitCount == 0 || unit->payloadWordCount == 0)
      return loadError(RuntimeLoadFailureKind::InvalidDeployment,
                       "portable configuration target has an empty payload");

    auto activation = detail::configurationActivationEventKey(
        closure.dataflow.identity(),
        closure.mappingClosure.executionContexts.spatialDomains, selectedCore);
    if (!activation)
      return loadError(RuntimeLoadFailureKind::InvalidDeployment,
                       "cannot derive configuration activation events: " +
                           llvm::toString(activation.takeError()));
    RuntimeConfigurationTarget target{imageReference,
                                      selectedCore,
                                      binding->providerEndpoint,
                                      unit->payloadBitCount,
                                      unit->commitAddress,
                                      unit->statusAddress,
                                      {},
                                      transportLayoutKey(*selectedLayout),
                                      std::move(*activation)};
    for (std::uint64_t word = 0; word != unit->payloadWordCount; ++word) {
      target.words.push_back(RuntimeConfigurationWord{
          unit->baseAddress + static_cast<std::uint32_t>(word * 4),
          imageWord(image->image().payload(), word),
          imageStrobe(image->image().payload(), word)});
    }
    result.push_back(std::move(target));
  }
  return result;
}

ByteVector programmingEndpointKey(const RuntimeProviderEndpointRef &endpoint) {
  ByteVector key;
  appendU32(key, endpoint.kind);
  appendFramed(key, endpoint.payload);
  return key;
}

llvm::Error validateConfigurationEndpointOwnership(
    llvm::ArrayRef<RuntimeConfigurationTarget> targets) {
  std::map<ByteVector, fabric::SpatialCoreOccurrenceRef> owners;
  for (const RuntimeConfigurationTarget &target : targets) {
    auto [found, inserted] = owners.try_emplace(
        programmingEndpointKey(target.endpoint), target.spatialCore);
    if (!inserted && !(found->second == target.spatialCore))
      return loadError(
          RuntimeLoadFailureKind::InvalidDeployment,
          "one programming endpoint aliases multiple SpatialCore transports");
  }
  return llvm::Error::success();
}

ByteVector configurationGroupKey(const RuntimeConfigurationTarget &target) {
  ByteVector key;
  appendFramed(key, target.transportLayoutKey);
  appendFramed(key, target.activationEventKey);
  appendU64(key, target.payloadBitCount);
  appendU32(key, target.commitAddress);
  appendU32(key, target.statusAddress);
  appendU64(key, target.words.size());
  for (const RuntimeConfigurationWord &word : target.words) {
    appendU32(key, word.address);
    appendU32(key, word.value);
    key.push_back(word.byteStrobe);
  }
  return key;
}

std::uint32_t activeWordMask(const RuntimeConfigurationTarget &target,
                             std::size_t wordOrdinal) {
  if (wordOrdinal + 1 != target.words.size())
    return std::numeric_limits<std::uint32_t>::max();
  const unsigned activeBits =
      static_cast<unsigned>(target.payloadBitCount % 32);
  if (activeBits == 0)
    return std::numeric_limits<std::uint32_t>::max();
  return (std::uint32_t{1} << activeBits) - 1;
}

llvm::Error
verifyConfigurationReadback(RuntimeProviderInstance &provider,
                            const RuntimeLeaseHandle &lease,
                            const RuntimeConfigurationTarget &target) {
  for (const auto indexed : llvm::enumerate(target.words)) {
    auto actual = provider.readConfigurationWord(lease, target.endpoint,
                                                 indexed.value().address);
    if (!actual)
      return loadError(RuntimeLoadFailureKind::Programming,
                       "configuration readback failed: " +
                           llvm::toString(actual.takeError()));
    const std::uint32_t mask = activeWordMask(target, indexed.index());
    if ((*actual & mask) != (indexed.value().value & mask))
      return loadError(RuntimeLoadFailureKind::Programming,
                       "active configuration readback mismatch");
  }
  auto status = provider.readConfigurationWord(lease, target.endpoint,
                                               target.statusAddress);
  if (!status)
    return loadError(RuntimeLoadFailureKind::Programming,
                     "configuration status read failed: " +
                         llvm::toString(status.takeError()));
  if (*status != 0)
    return loadError(RuntimeLoadFailureKind::Programming,
                     "configuration status did not clear after commit");
  return llvm::Error::success();
}

llvm::Error
programConfigurations(RuntimeProviderInstance &provider,
                      const RuntimeProviderDescriptor &providerDescriptor,
                      const RuntimeLeaseHandle &lease,
                      llvm::MutableArrayRef<RuntimeConfigurationTarget> targets,
                      bool &deviceStateChanged) {
  std::map<ByteVector, std::vector<std::size_t>> groups;
  for (const auto indexed : llvm::enumerate(targets))
    groups[configurationGroupKey(indexed.value())].push_back(indexed.index());

  for (const auto &[key, ordinals] : groups) {
    (void)key;
    if (ordinals.size() > 1 &&
        providerDescriptor.supportsAtomicProgrammingMulticast) {
      std::vector<RuntimeConfigurationTarget> batch;
      batch.reserve(ordinals.size());
      for (std::size_t ordinal : ordinals)
        batch.push_back(targets[ordinal]);
      deviceStateChanged = true;
      if (llvm::Error error =
              provider.programConfigurationMulticast(lease, batch))
        return loadError(RuntimeLoadFailureKind::Programming,
                         "atomic configuration multicast failed: " +
                             llvm::toString(std::move(error)));
      for (std::size_t ordinal : ordinals)
        if (llvm::Error error =
                verifyConfigurationReadback(provider, lease, targets[ordinal]))
          return error;
      continue;
    }

    for (std::size_t ordinal : ordinals) {
      RuntimeConfigurationTarget &target = targets[ordinal];
      deviceStateChanged = true;
      for (const RuntimeConfigurationWord &word : target.words)
        if (llvm::Error error =
                provider.writeConfigurationWord(lease, target.endpoint, word))
          return loadError(RuntimeLoadFailureKind::Programming,
                           "configuration write failed: " +
                               llvm::toString(std::move(error)));
      if (llvm::Error error = provider.commitConfiguration(
              lease, target.endpoint, target.commitAddress))
        return loadError(RuntimeLoadFailureKind::Programming,
                         "configuration commit failed: " +
                             llvm::toString(std::move(error)));
      if (llvm::Error error =
              verifyConfigurationReadback(provider, lease, target))
        return error;
    }
  }
  return llvm::Error::success();
}

dataflow::LogicalMemoryRootRef
memoryRoot(dataflow::LogicalMemoryRootOrViewRef reference) {
  return std::visit(
      [](const auto &value) -> dataflow::LogicalMemoryRootRef {
        using Value = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<Value, dataflow::LogicalMemoryRootRef>)
          return value;
        else
          return value.root;
      },
      reference);
}

bool selectionMatchesLaunch(const mapping::ServicePlanSelectionAnchor &anchor,
                            dataflow::RootedGraphLaunchRef launch) {
  return std::visit(
      [&](const auto &typed) {
        using Anchor = std::decay_t<decltype(typed)>;
        if constexpr (std::is_same_v<
                          Anchor, mapping::MemoryExposurePlanSelectionAnchor>) {
          return typed.exposure.launch == launch;
        } else {
          return std::visit(
              [&](const auto &member) {
                using Member = std::decay_t<decltype(member)>;
                if constexpr (std::is_same_v<
                                  Member,
                                  dataflow::AddressedMemoryActorMemberRef> ||
                              std::is_same_v<Member,
                                             dataflow::FenceActorMemberRef>)
                  return member.actor.launch == launch;
                else
                  return false;
              },
              typed.member);
        }
      },
      anchor);
}

bool planCanBeSelected(const mapping::SystemServicePlanSelectionView &selection,
                       std::uint64_t planOrdinal) {
  if (selection.defaultPlanOrdinal == planOrdinal)
    return true;
  return llvm::any_of(selection.clauses, [&](const auto &clause) {
    return clause.target == planOrdinal;
  });
}

ByteVector intervalKey(const mapping::SpatialMemoryIntervalView &interval) {
  ByteVector key;
  if (std::holds_alternative<mapping::SpatialMemoryWholeIntervalView>(
          interval)) {
    key.push_back(0);
    return key;
  }
  key.push_back(1);
  const auto &range = std::get<mapping::SpatialMemoryByteRangeView>(interval);
  appendU64(key, range.offsetBytes);
  appendU64(key, range.sizeBytes);
  return key;
}

llvm::Expected<ByteVector>
staticTargetKey(const dataflow::CanonicalDataflowProgramView &dataflow,
                const RuntimeStaticMemoryTarget &target) {
  ByteVector key;
  if (const auto *local = std::get_if<RuntimeSpatialMemoryTarget>(&target)) {
    key.push_back(0);
    auto context = mapping::encodeExecutionContextKey(local->context);
    if (!context)
      return context.takeError();
    appendFramed(key, *context);
    appendFramed(key, intervalKey(local->interval));
    appendFramed(key,
                 fabric::canonicalFabricBytes(local->region.serviceRegion));
    appendU64(key, local->region.physicalOffsetBytes);
    return key;
  }
  key.push_back(1);
  const auto &system = std::get<mapping::SystemMemoryRegionElementView>(target);
  auto logical = dataflow::encodeDataflowReference(dataflow.identity(),
                                                   system.logicalMemory);
  if (!logical)
    return logical.takeError();
  appendFramed(key, *logical);
  appendFramed(key, intervalKey(system.interval));
  appendFramed(key, fabric::canonicalFabricBytes(system.serviceRegion));
  appendU64(key, system.transformPath.size());
  for (const fabric::SystemServiceTransformRef &transform :
       system.transformPath)
    appendFramed(key, fabric::canonicalFabricBytes(transform));
  return key;
}

llvm::Expected<std::vector<RuntimeStaticMemoryTarget>>
collectStaticTargets(const deployment::StaticMemoryImageLeaf &leaf,
                     const ImportedRuntimeClosure &closure,
                     const ArtifactStore &artifacts) {
  std::vector<RuntimeStaticMemoryTarget> candidates;
  bool usesBoundaryProxy = false;
  for (const mapping::SystemSpatialContextDomain &domain :
       closure.mappingClosure.executionContexts.spatialDomains) {
    if (domain.graph != leaf.rootedGraphLaunch())
      continue;
    auto spatial =
        mapping::importSpatialMapping(domain.spatialMapping, artifacts);
    if (!spatial)
      return spatial.takeError();
    for (const mapping::SpatialMemoryBindingView &binding :
         spatial->view().memoryBindings()) {
      if (memoryRoot(binding.logicalMemory) != leaf.logicalMemoryRoot())
        continue;
      if (const auto *local =
              std::get_if<mapping::SpatialMemoryLocalRegionView>(
                  &binding.target)) {
        candidates.emplace_back(RuntimeSpatialMemoryTarget{
            domain.context, binding.interval, *local});
      } else {
        usesBoundaryProxy = true;
      }
    }
  }

  if (usesBoundaryProxy) {
    for (const mapping::SystemServiceRealizationView &service :
         closure.mappingClosure.serviceRealizations) {
      const auto *operation =
          std::get_if<mapping::OperationServiceObligationFamilyKey>(
              &service.key);
      if (!operation)
        continue;
      const auto *logical =
          std::get_if<dataflow::LogicalMemoryRootOrViewRef>(operation);
      if (!logical || memoryRoot(*logical) != leaf.logicalMemoryRoot())
        continue;
      for (const mapping::SystemServicePlanView &plan : service.plans) {
        const bool selected = llvm::any_of(
            service.selections,
            [&](const mapping::SystemServicePlanSelectionView &selection) {
              return selectionMatchesLaunch(selection.key.anchor,
                                            leaf.rootedGraphLaunch()) &&
                     planCanBeSelected(selection, plan.ordinal);
            });
        if (!selected)
          continue;
        for (const mapping::SystemMemoryRegionTargetView &target :
             plan.memoryTargets)
          if (memoryRoot(target.element.logicalMemory) ==
              leaf.logicalMemoryRoot())
            candidates.emplace_back(target.element);
      }
    }
  }

  std::map<ByteVector, RuntimeStaticMemoryTarget> canonical;
  for (RuntimeStaticMemoryTarget &target : candidates) {
    auto key = staticTargetKey(closure.dataflow, target);
    if (!key)
      return key.takeError();
    canonical.try_emplace(std::move(*key), std::move(target));
  }
  if (canonical.empty())
    return loadError(RuntimeLoadFailureKind::InvalidDeployment,
                     "static logical-memory image has no exact Mapping target");
  std::vector<RuntimeStaticMemoryTarget> result;
  result.reserve(canonical.size());
  for (auto &[key, target] : canonical) {
    (void)key;
    result.push_back(std::move(target));
  }
  return result;
}

llvm::Expected<RuntimeStaticMemoryInstall>
materializeStaticMemory(const deployment::StaticMemoryImageLeaf &leaf,
                        const ImportedRuntimeClosure &closure,
                        const ArtifactStore &artifacts,
                        const BlobStore &blobs) {
  if (leaf.sizeBytes() > std::numeric_limits<std::size_t>::max())
    return loadError(RuntimeLoadFailureKind::InvalidDeployment,
                     "static logical-memory image is too large for this host");
  RuntimeStaticMemoryInstall result{
      leaf.canonicalDataflow(),
      leaf.rootedGraphLaunch(),
      leaf.logicalMemoryRoot(),
      leaf.layoutBinding(),
      leaf.alignmentBytes(),
      leaf.permissions(),
      std::vector<std::uint8_t>(static_cast<std::size_t>(leaf.sizeBytes()), 0),
      {}};
  for (const deployment::StaticMemoryInitializedChunk &chunk :
       leaf.initializedChunks()) {
    auto bytes = blobs.get(chunk.blobDigest);
    if (!bytes)
      return loadError(RuntimeLoadFailureKind::InvalidDeployment,
                       "cannot read static-memory blob: " +
                           llvm::toString(bytes.takeError()));
    if (bytes->size() != chunk.byteCount ||
        chunk.byteOffset > result.bytes.size() ||
        chunk.byteCount > result.bytes.size() - chunk.byteOffset)
      return loadError(RuntimeLoadFailureKind::InvalidDeployment,
                       "static-memory chunk is inconsistent with its leaf");
    std::copy(bytes->begin(), bytes->end(),
              result.bytes.begin() +
                  static_cast<std::size_t>(chunk.byteOffset));
  }
  auto targets = collectStaticTargets(leaf, closure, artifacts);
  if (!targets)
    return loadError(RuntimeLoadFailureKind::InvalidDeployment,
                     "cannot derive static-memory targets: " +
                         llvm::toString(targets.takeError()));
  result.targets = std::move(*targets);
  return result;
}

struct ImportedExecutables final {
  std::vector<std::uint8_t> hostBytes;
  std::vector<FinalizedInstructionCoreBinary> instructionBinaries;
  std::vector<std::vector<std::uint8_t>> instructionBytes;
};

llvm::Expected<ImportedExecutables>
importExecutables(const deployment::FinalizedDeployment &deployment,
                  const ArtifactStore &artifacts, const BlobStore &blobs) {
  ImportedExecutables result;
  auto host = blobs.get(deployment.deployment().hostProgram().programBlob());
  if (!host)
    return loadError(RuntimeLoadFailureKind::InvalidDeployment,
                     "cannot read host program blob: " +
                         llvm::toString(host.takeError()));
  result.hostBytes = std::move(*host);
  for (const ArtifactRootReference &reference :
       deployment.deployment().instructionCoreBinaries()) {
    auto binary = importInstructionCoreBinary(reference, artifacts, blobs);
    if (!binary)
      return loadError(RuntimeLoadFailureKind::InvalidDeployment,
                       "cannot import InstructionCore binary: " +
                           llvm::toString(binary.takeError()));
    auto code = blobs.get(binary->binary().codeBlob());
    if (!code)
      return loadError(RuntimeLoadFailureKind::InvalidDeployment,
                       "cannot read InstructionCore code blob: " +
                           llvm::toString(code.takeError()));
    result.instructionBinaries.push_back(std::move(*binary));
    result.instructionBytes.push_back(std::move(*code));
  }
  return result;
}

std::string appendCleanupDiagnostic(std::string primary, llvm::StringRef role,
                                    llvm::Error error) {
  if (!error)
    return primary;
  primary += "; ";
  primary += role;
  primary += ": ";
  primary += llvm::toString(std::move(error));
  return primary;
}

llvm::Error recoverAndRelease(RuntimeLoadFailureKind kind,
                              std::string diagnostic,
                              RuntimeProviderInstance &provider,
                              const RuntimeDeviceHandle &device,
                              const RuntimeLeaseHandle &lease,
                              llvm::ArrayRef<RuntimeIdentityClaim> identities,
                              bool restoreCleanState) {
  bool quarantined = false;
  if (restoreCleanState) {
    llvm::Error reset = provider.quiesceAndReset(lease);
    if (reset) {
      diagnostic = appendCleanupDiagnostic(
          std::move(diagnostic), "recovery reset failed", std::move(reset));
      quarantined = true;
    } else if (llvm::Error identity =
                   verifyProviderIdentities(provider, device, identities)) {
      diagnostic = appendCleanupDiagnostic(std::move(diagnostic),
                                           "recovery identity check failed",
                                           std::move(identity));
      quarantined = true;
    }
  }
  if (llvm::Error release = provider.releaseExclusiveLease(lease)) {
    diagnostic = appendCleanupDiagnostic(
        std::move(diagnostic), "lease release failed", std::move(release));
    quarantined = true;
  }
  if (quarantined)
    provider.quarantineDevice(device);
  return loadError(kind, diagnostic, quarantined);
}

llvm::Error
releaseLoadedState(RuntimeProviderInstance &provider,
                   const RuntimeDeviceHandle &device,
                   const RuntimeLeaseHandle &lease,
                   llvm::ArrayRef<RuntimeIdentityClaim> identities) {
  bool quarantine = false;
  if (llvm::Error reset = provider.quiesceAndReset(lease)) {
    llvm::consumeError(std::move(reset));
    quarantine = true;
  } else if (llvm::Error identity =
                 verifyProviderIdentities(provider, device, identities)) {
    llvm::consumeError(std::move(identity));
    quarantine = true;
  }
  if (llvm::Error release = provider.releaseExclusiveLease(lease)) {
    llvm::consumeError(std::move(release));
    quarantine = true;
  }
  if (quarantine)
    provider.quarantineDevice(device);
  return llvm::Error::success();
}

} // namespace

llvm::Expected<std::vector<std::uint8_t>>
detail::configurationActivationEventKey(
    const ArtifactIdentity &dataflowIdentity,
    llvm::ArrayRef<mapping::SystemSpatialContextDomain> spatialDomains,
    fabric::SpatialCoreOccurrenceRef spatialCore) {
  std::vector<ByteVector> events;
  for (const mapping::SystemSpatialContextDomain &domain : spatialDomains) {
    if (domain.context.accCore != spatialCore.core)
      continue;
    auto event = dataflow::encodeDataflowReference(
        dataflowIdentity, dataflow::graphLaunchStartEventFamily(domain.graph));
    if (!event)
      return event.takeError();
    events.push_back(std::move(*event));
  }
  llvm::sort(events);
  events.erase(std::unique(events.begin(), events.end()), events.end());
  ByteVector key;
  appendU64(key, events.size());
  for (const ByteVector &event : events)
    appendFramed(key, event);
  return key;
}

char RuntimeLoadError::ID = 0;

void RuntimeLoadError::log(llvm::raw_ostream &output) const {
  output << "runtime_load_failed: " << diagnostic_;
  if (deviceQuarantined_)
    output << " (device quarantined)";
}

std::error_code RuntimeLoadError::convertToErrorCode() const {
  return std::make_error_code(std::errc::io_error);
}

char RuntimeActivationReplacementError::ID = 0;

void RuntimeActivationReplacementError::log(llvm::raw_ostream &output) const {
  output << "runtime_activation_replacement_failed: " << diagnostic_;
}

std::error_code RuntimeActivationReplacementError::convertToErrorCode() const {
  return std::make_error_code(std::errc::io_error);
}

llvm::Error RuntimeProviderInstance::programConfigurationMulticast(
    const RuntimeLeaseHandle &, llvm::ArrayRef<RuntimeConfigurationTarget>) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "atomic programming multicast is unsupported");
}

llvm::Expected<RuntimePreparedActivationHandle>
RuntimeProviderInstance::prepareActivation(
    const RuntimeLeaseHandle &, const RuntimeExecutableRegistrationView &,
    const RuntimeActivationView &) {
  return llvm::createStringError(
      std::make_error_code(std::errc::operation_not_supported),
      "prepared activation replacement is unsupported");
}

llvm::Error RuntimeProviderInstance::replaceActivationAtomically(
    const RuntimeLeaseHandle &, const RuntimePreparedActivationHandle &) {
  return llvm::createStringError(
      std::make_error_code(std::errc::operation_not_supported),
      "prepared activation replacement is unsupported");
}

llvm::Error RuntimeProviderInstance::discardPreparedActivation(
    const RuntimeLeaseHandle &, const RuntimePreparedActivationHandle &) {
  return llvm::createStringError(
      std::make_error_code(std::errc::operation_not_supported),
      "prepared activation replacement is unsupported");
}

namespace {

bool sameRootSet(llvm::ArrayRef<dataflow::RootThreadLaunchRef> lhs,
                 llvm::ArrayRef<dataflow::RootThreadLaunchRef> rhs) {
  return lhs.size() == rhs.size() && llvm::all_of(lhs, [&](auto root) {
           return llvm::is_contained(rhs, root);
         });
}

bool sameSafePoint(
    const std::optional<pnr::ResourceTimeSafePointReference> &lhs,
    const std::optional<pnr::ResourceTimeSafePointReference> &rhs) {
  if (lhs.has_value() != rhs.has_value())
    return false;
  return !lhs || (lhs->artifact == rhs->artifact && lhs->kind == rhs->kind);
}

bool sameSelectionEdge(const pnr::ResourceTimeTransition &lhs,
                       const pnr::ResourceTimeTransition &rhs) {
  return lhs.trigger == rhs.trigger &&
         sameSafePoint(lhs.safePoint, rhs.safePoint) &&
         lhs.parent == rhs.parent && lhs.child == rhs.child &&
         sameRootSet(lhs.completedBefore, rhs.completedBefore);
}

} // namespace

namespace detail {

struct ResourceTimeActivationToken final {};

struct LoadedDeploymentState final {
  struct PreparedActivation final {
    pnr::ResourceTimeTransitionEndpointReference endpoint;
    deployment::FinalizedDeployment deployment;
    RuntimePreparedActivationHandle handle;
  };

  LoadedDeploymentState(deployment::FinalizedDeployment deployment,
                        std::shared_ptr<RuntimeProviderInstance> provider,
                        RuntimeDeviceHandle device, RuntimeLeaseHandle lease,
                        std::vector<RuntimeIdentityClaim> identities)
      : deployment(std::move(deployment)), provider(std::move(provider)),
        device(std::move(device)), lease(std::move(lease)),
        identities(std::move(identities)) {}

  LoadedDeploymentState(const LoadedDeploymentState &) = delete;
  LoadedDeploymentState &operator=(const LoadedDeploymentState &) = delete;

  deployment::FinalizedDeployment deployment;
  std::shared_ptr<RuntimeProviderInstance> provider;
  RuntimeDeviceHandle device;
  RuntimeLeaseHandle lease;
  std::vector<RuntimeIdentityClaim> identities;
  std::optional<pnr::ResourceTimeTransitionGraph> preparedActivationGraph;
  std::vector<PreparedActivation> preparedActivations;
  std::shared_ptr<const ResourceTimeActivationToken> preparedActivationToken;

  ~LoadedDeploymentState() {
    if (!provider)
      return;
    for (const PreparedActivation &activation : preparedActivations)
      llvm::consumeError(
          provider->discardPreparedActivation(lease, activation.handle));
    llvm::consumeError(
        releaseLoadedState(*provider, device, lease, identities));
  }
};

} // namespace detail

LoadedDeployment::LoadedDeployment(
    std::unique_ptr<detail::LoadedDeploymentState> state)
    : state_(std::move(state)) {}

LoadedDeployment::LoadedDeployment(LoadedDeployment &&) noexcept = default;
LoadedDeployment &
LoadedDeployment::operator=(LoadedDeployment &&) noexcept = default;
LoadedDeployment::~LoadedDeployment() = default;

const deployment::FinalizedDeployment &LoadedDeployment::deployment() const {
  assert(state_ && "moved-from LoadedDeployment has no deployment");
  return state_->deployment;
}

const RuntimeDeviceHandle &LoadedDeployment::device() const {
  assert(state_ && "moved-from LoadedDeployment has no device");
  return state_->device;
}

llvm::Expected<std::shared_ptr<const detail::ResourceTimeActivationToken>>
LoadedDeployment::prepareResourceTimeActivations(
    const pnr::ResourceTimeTransitionGraph &graph,
    const ArtifactStore &artifacts, const BlobStore &blobs) {
  assert(state_ && "moved-from LoadedDeployment has no deployment state");
  if (state_->preparedActivationGraph || state_->preparedActivationToken ||
      !state_->preparedActivations.empty())
    return activationReplacementError(
        RuntimeActivationReplacementErrorReason::PreparationFailed,
        "loaded Deployment already retains resource-time preparation state");
  if (llvm::Error error =
          pnr::verifyResourceTimeTransitionGraph(graph, artifacts, blobs))
    return activationReplacementError(
        RuntimeActivationReplacementErrorReason::InvalidDeployment,
        "resource-time graph failed independent closure: " +
            llvm::toString(std::move(error)));
  if (!graph.entry.deployment ||
      *graph.entry.deployment != state_->deployment.reference() ||
      graph.entry.mapping != state_->deployment.deployment().systemMapping())
    return activationReplacementError(
        RuntimeActivationReplacementErrorReason::TransitionMismatch,
        "resource-time graph entry is not the active Deployment");
  if (!state_->deployment.deployment().staticMemoryImages().empty())
    return activationReplacementError(
        RuntimeActivationReplacementErrorReason::TransitionMismatch,
        "prepared activation requires an empty static-memory state");
  if (!state_->provider->descriptor().supportsPreparedActivationReplacement)
    return activationReplacementError(
        RuntimeActivationReplacementErrorReason::ProviderCapabilityUnavailable,
        "runtime provider does not support prepared activation replacement");

  std::vector<detail::LoadedDeploymentState::PreparedActivation> prepared;
  const auto failPreparation =
      [&](RuntimeActivationReplacementErrorReason reason,
          const llvm::Twine &message) -> llvm::Error {
    std::string diagnostic = message.str();
    std::vector<detail::LoadedDeploymentState::PreparedActivation>
        cleanupFailures;
    for (auto &activation : prepared)
      if (llvm::Error error = state_->provider->discardPreparedActivation(
              state_->lease, activation.handle)) {
        diagnostic += "; prepared activation cleanup failed: " +
                      llvm::toString(std::move(error));
        cleanupFailures.push_back(std::move(activation));
      }
    state_->preparedActivations = std::move(cleanupFailures);
    return activationReplacementError(reason, diagnostic);
  };

  prepared.reserve(graph.endpoints.size());
  for (const pnr::ResourceTimeTransitionEndpointReference &endpoint :
       graph.endpoints) {
    if (!llvm::any_of(graph.transitions,
                      [&](const pnr::ResourceTimeTransition &transition) {
                        return transition.child == endpoint;
                      }))
      continue;
    if (!endpoint.deployment)
      return failPreparation(
          RuntimeActivationReplacementErrorReason::InvalidDeployment,
          "verified resource-time endpoint has no Deployment");
    auto reimported =
        deployment::importDeployment(*endpoint.deployment, artifacts, blobs);
    if (!reimported)
      return failPreparation(
          RuntimeActivationReplacementErrorReason::InvalidDeployment,
          "candidate Deployment closure validation failed: " +
              llvm::toString(reimported.takeError()));
    deployment::FinalizedDeployment deployment = std::move(*reimported);
    const auto &candidate = deployment.deployment();
    if (!candidate.staticMemoryImages().empty())
      return failPreparation(
          RuntimeActivationReplacementErrorReason::TransitionMismatch,
          "prepared activation requires empty child static-memory state");

    auto closure = importRuntimeClosure(deployment, artifacts, blobs);
    if (!closure)
      return failPreparation(
          RuntimeActivationReplacementErrorReason::InvalidDeployment,
          "candidate runtime closure import failed: " +
              llvm::toString(closure.takeError()));
    const RuntimeProviderBinding &allowed =
        closure->runtimeBindings.front().binding().providerBinding();
    const RuntimeProviderDescriptor *registered =
        findRuntimeProvider(allowed.descriptor);
    if (!registered || registered != &state_->provider->descriptor() ||
        registered->implementationSemanticIdentity !=
            allowed.implementationSemanticIdentity ||
        registered->runtimeAbiIdentity != allowed.runtimeAbiIdentity)
      return failPreparation(
          RuntimeActivationReplacementErrorReason::ProviderMismatch,
          "candidate Deployment selects another runtime provider");

    auto executables = importExecutables(deployment, artifacts, blobs);
    if (!executables)
      return failPreparation(
          RuntimeActivationReplacementErrorReason::InvalidDeployment,
          "candidate executable closure import failed: " +
              llvm::toString(executables.takeError()));
    const RuntimeExecutableRegistrationView registration{
        candidate.hostProgram(), executables->hostBytes,
        executables->instructionBinaries, executables->instructionBytes,
        candidate.threadDispatchImage()};
    const RuntimeActivationView activation{
        deployment.reference(), closure->runtimeBindings,
        candidate.threadDispatchImage(), candidate.spatialLaunchImage(),
        candidate.admissionImage()};
    auto handle = state_->provider->prepareActivation(state_->lease,
                                                      registration, activation);
    if (!handle)
      return failPreparation(
          RuntimeActivationReplacementErrorReason::PreparationFailed,
          "provider rejected activation preparation: " +
              llvm::toString(handle.takeError()));
    prepared.push_back({endpoint, std::move(deployment), std::move(*handle)});
  }

  state_->preparedActivationGraph = graph;
  state_->preparedActivations = std::move(prepared);
  auto token = std::make_shared<const detail::ResourceTimeActivationToken>();
  state_->preparedActivationToken = token;
  return token;
}

llvm::Error LoadedDeployment::activatePreparedTransition(
    const pnr::ResourceTimeTransition &transition,
    const std::shared_ptr<const detail::ResourceTimeActivationToken> &token) {
  assert(state_ && "moved-from LoadedDeployment has no deployment state");
  if (!token || token != state_->preparedActivationToken ||
      !state_->preparedActivationGraph)
    return activationReplacementError(
        RuntimeActivationReplacementErrorReason::PreparationFailed,
        "selector is not bound to the loaded resource-time graph");
  if (!transition.parent.deployment ||
      *transition.parent.deployment != state_->deployment.reference())
    return activationReplacementError(
        RuntimeActivationReplacementErrorReason::TransitionMismatch,
        "resource-time edge does not leave the active Deployment");
  if (!llvm::any_of(state_->preparedActivationGraph->transitions,
                    [&](const pnr::ResourceTimeTransition &candidate) {
                      return sameSelectionEdge(candidate, transition);
                    }))
    return activationReplacementError(
        RuntimeActivationReplacementErrorReason::TransitionMismatch,
        "resource-time edge is outside the prepared graph");

  auto prepared = llvm::find_if(
      state_->preparedActivations,
      [&](const detail::LoadedDeploymentState::PreparedActivation &candidate) {
        return candidate.endpoint == transition.child;
      });
  if (prepared == state_->preparedActivations.end())
    return activationReplacementError(
        RuntimeActivationReplacementErrorReason::PreparationFailed,
        "selected child activation is not prepared");
  if (llvm::Error error = state_->provider->replaceActivationAtomically(
          state_->lease, prepared->handle))
    return activationReplacementError(
        RuntimeActivationReplacementErrorReason::ActivationFailed,
        "provider rejected atomic activation replacement: " +
            llvm::toString(std::move(error)));

  state_->deployment = prepared->deployment;
  return llvm::Error::success();
}

llvm::Expected<LoadedDeployment>
loadDeployment(deployment::FinalizedDeployment deployment,
               RuntimeProviderSelection selection,
               const ArtifactStore &artifacts, const BlobStore &blobs) {
  auto reimported =
      deployment::importDeployment(deployment.reference(), artifacts, blobs);
  if (!reimported)
    return loadError(RuntimeLoadFailureKind::InvalidDeployment,
                     "package or Deployment closure validation failed: " +
                         llvm::toString(reimported.takeError()));
  if (reimported->canonicalBytes().bytes() !=
      deployment.canonicalBytes().bytes())
    return loadError(RuntimeLoadFailureKind::InvalidDeployment,
                     "provided Deployment bytes do not match the exact root");
  deployment = std::move(*reimported);

  auto closure = importRuntimeClosure(deployment, artifacts, blobs);
  if (!closure)
    return closure.takeError();
  if (!selection.provider)
    return loadError(RuntimeLoadFailureKind::ProviderMismatch,
                     "RuntimeProviderSelection has no provider instance");
  RuntimeProviderInstance &provider = *selection.provider;
  const RuntimeProviderBinding &allowed =
      closure->runtimeBindings.front().binding().providerBinding();
  const RuntimeProviderDescriptor *registeredProvider =
      findRuntimeProvider(allowed.descriptor);
  if (!registeredProvider || &provider.descriptor() != registeredProvider ||
      registeredProvider->implementationSemanticIdentity !=
          allowed.implementationSemanticIdentity ||
      registeredProvider->runtimeAbiIdentity != allowed.runtimeAbiIdentity)
    return loadError(RuntimeLoadFailureKind::ProviderMismatch,
                     "selected provider instance is not the exact provider "
                     "allowed by Deployment");
  if (registeredProvider->runtimeAbiIdentity !=
      hardware::rtl::portableConfigurationRuntimeAbiIdentity)
    return loadError(RuntimeLoadFailureKind::ProviderMismatch,
                     "selected provider does not implement the portable "
                     "configuration runtime ABI");

  auto configurations = configurationTargets(deployment, *closure, artifacts);
  if (!configurations)
    return configurations.takeError();
  if (llvm::Error error =
          validateConfigurationEndpointOwnership(*configurations))
    return std::move(error);

  std::vector<RuntimeStaticMemoryInstall> staticMemory;
  staticMemory.reserve(deployment.deployment().staticMemoryImages().size());
  for (const deployment::StaticMemoryImageLeaf &leaf :
       deployment.deployment().staticMemoryImages()) {
    auto install = materializeStaticMemory(leaf, *closure, artifacts, blobs);
    if (!install)
      return install.takeError();
    staticMemory.push_back(std::move(*install));
  }

  auto executables = importExecutables(deployment, artifacts, blobs);
  if (!executables)
    return executables.takeError();

  auto devices = provider.enumerateDevices();
  if (!devices)
    return loadError(RuntimeLoadFailureKind::Enumeration,
                     "provider enumeration failed: " +
                         llvm::toString(devices.takeError()));
  if (selection.deviceOrdinal >= devices->size())
    return loadError(RuntimeLoadFailureKind::Enumeration,
                     "selected device ordinal is absent from enumeration");
  const RuntimeDeviceHandle device =
      (*devices)[static_cast<std::size_t>(selection.deviceOrdinal)];
  std::vector<RuntimeIdentityClaim> identities;
  identities.reserve(closure->runtimeBindings.size());
  for (const auto indexed : llvm::enumerate(closure->runtimeBindings))
    identities.push_back(
        {indexed.value().binding().identityVerification(),
         closure->implementations[indexed.index()].reference().artifact});
  if (llvm::Error error =
          verifyProviderIdentities(provider, device, identities))
    return std::move(error);

  auto lease = provider.acquireExclusiveLease(device);
  if (!lease)
    return loadError(RuntimeLoadFailureKind::Lease,
                     "exclusive lease acquisition failed: " +
                         llvm::toString(lease.takeError()));
  if (llvm::Error error = provider.quiesceAndReset(*lease))
    return recoverAndRelease(RuntimeLoadFailureKind::Reset,
                             "cannot establish declared reset state: " +
                                 llvm::toString(std::move(error)),
                             provider, device, *lease, identities,
                             /*restoreCleanState=*/true);

  bool deviceStateChanged = false;
  if (llvm::Error error =
          programConfigurations(provider, *registeredProvider, *lease,
                                *configurations, deviceStateChanged))
    return recoverAndRelease(RuntimeLoadFailureKind::Programming,
                             llvm::toString(std::move(error)), provider, device,
                             *lease, identities,
                             /*restoreCleanState=*/true);

  std::vector<RuntimeInterfaceBinding> memoryBindings;
  for (const FinalizedRuntimePlatformBinding &binding :
       closure->runtimeBindings)
    memoryBindings.insert(memoryBindings.end(),
                          binding.binding().memoryInterfaceBindings().begin(),
                          binding.binding().memoryInterfaceBindings().end());

  for (const RuntimeStaticMemoryInstall &install : staticMemory) {
    deviceStateChanged = true;
    if (llvm::Error error =
            provider.installStaticMemory(*lease, install, memoryBindings))
      return recoverAndRelease(RuntimeLoadFailureKind::StaticMemory,
                               "static-memory installation failed: " +
                                   llvm::toString(std::move(error)),
                               provider, device, *lease, identities,
                               /*restoreCleanState=*/true);
  }

  RuntimeExecutableRegistrationView registration{
      deployment.deployment().hostProgram(), executables->hostBytes,
      executables->instructionBinaries, executables->instructionBytes,
      deployment.deployment().threadDispatchImage()};
  deviceStateChanged = true;
  if (llvm::Error error = provider.registerExecutables(*lease, registration))
    return recoverAndRelease(RuntimeLoadFailureKind::Registration,
                             "executable registration failed: " +
                                 llvm::toString(std::move(error)),
                             provider, device, *lease, identities,
                             /*restoreCleanState=*/deviceStateChanged);

  const RuntimeActivationView activation{
      deployment.reference(), closure->runtimeBindings,
      deployment.deployment().threadDispatchImage(),
      deployment.deployment().spatialLaunchImage(),
      deployment.deployment().admissionImage()};
  if (llvm::Error error = provider.activate(*lease, activation))
    return recoverAndRelease(RuntimeLoadFailureKind::Activation,
                             "activation failed: " +
                                 llvm::toString(std::move(error)),
                             provider, device, *lease, identities,
                             /*restoreCleanState=*/deviceStateChanged);

  return LoadedDeployment(std::make_unique<detail::LoadedDeploymentState>(
      std::move(deployment), std::move(selection.provider), device,
      std::move(*lease), std::move(identities)));
}

} // namespace loom::runtime
