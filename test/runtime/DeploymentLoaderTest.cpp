#include "DeploymentTestSupport.h"

#include "Common/ArtifactStore.h"
#include "Common/BlobStore.h"
#include "Mapping/IR/MappingSchema.h"
#include "Runtime/DeploymentLoader.h"
#include "Runtime/InProcessPlatform.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

using namespace loom;
using namespace loom::runtime;

namespace {

template <typename T> T take(llvm::StringRef test, llvm::Expected<T> value) {
  if (!value)
    deployment::test::fail(test, llvm::toString(value.takeError()));
  return std::move(*value);
}

ArtifactIdentity identity(llvm::StringRef test, std::uint8_t seed) {
  ArtifactIdentity::Storage bytes{};
  for (std::size_t index = 0; index != bytes.size(); ++index)
    bytes[index] = static_cast<std::uint8_t>(seed + index);
  return take(test, ArtifactIdentity::fromBytes(bytes));
}

struct ObservedLoadError final {
  RuntimeLoadFailureKind kind = RuntimeLoadFailureKind::InvalidDeployment;
  bool quarantined = false;
  std::string diagnostic;
};

class ForwardingRuntimeProvider final : public RuntimeProviderInstance {
public:
  ForwardingRuntimeProvider(std::shared_ptr<InProcessRuntimeProvider> delegate,
                            const RuntimeProviderDescriptor &descriptor)
      : delegate_(std::move(delegate)), descriptor_(&descriptor) {}

  const RuntimeProviderDescriptor &descriptor() const override {
    return *descriptor_;
  }

  llvm::Expected<std::vector<RuntimeDeviceHandle>> enumerateDevices() override {
    return delegate_->enumerateDevices();
  }

  llvm::Expected<ArtifactIdentity> readImplementationIdentity(
      const RuntimeDeviceHandle &device,
      const RuntimeProviderEndpointRef &endpoint) override {
    return delegate_->readImplementationIdentity(device, endpoint);
  }

  llvm::Expected<BlobDigest>
  readTrustedAttestation(const RuntimeDeviceHandle &device) override {
    return delegate_->readTrustedAttestation(device);
  }

  llvm::Expected<RuntimeLeaseHandle>
  acquireExclusiveLease(const RuntimeDeviceHandle &device) override {
    return delegate_->acquireExclusiveLease(device);
  }

  llvm::Error quiesceAndReset(const RuntimeLeaseHandle &lease) override {
    return delegate_->quiesceAndReset(lease);
  }

  llvm::Error
  writeConfigurationWord(const RuntimeLeaseHandle &lease,
                         const RuntimeProviderEndpointRef &endpoint,
                         const RuntimeConfigurationWord &word) override {
    return delegate_->writeConfigurationWord(lease, endpoint, word);
  }

  llvm::Error commitConfiguration(const RuntimeLeaseHandle &lease,
                                  const RuntimeProviderEndpointRef &endpoint,
                                  std::uint32_t commitAddress) override {
    return delegate_->commitConfiguration(lease, endpoint, commitAddress);
  }

  llvm::Expected<std::uint32_t>
  readConfigurationWord(const RuntimeLeaseHandle &lease,
                        const RuntimeProviderEndpointRef &endpoint,
                        std::uint32_t address) override {
    return delegate_->readConfigurationWord(lease, endpoint, address);
  }

  llvm::Error programConfigurationMulticast(
      const RuntimeLeaseHandle &lease,
      llvm::ArrayRef<RuntimeConfigurationTarget> targets) override {
    return delegate_->programConfigurationMulticast(lease, targets);
  }

  llvm::Error installStaticMemory(
      const RuntimeLeaseHandle &lease,
      const RuntimeStaticMemoryInstall &install,
      llvm::ArrayRef<RuntimeInterfaceBinding> memoryBindings) override {
    return delegate_->installStaticMemory(lease, install, memoryBindings);
  }

  llvm::Error registerExecutables(
      const RuntimeLeaseHandle &lease,
      const RuntimeExecutableRegistrationView &registration) override {
    return delegate_->registerExecutables(lease, registration);
  }

  llvm::Error activate(const RuntimeLeaseHandle &lease,
                       const RuntimeActivationView &activation) override {
    return delegate_->activate(lease, activation);
  }

  llvm::Error releaseExclusiveLease(const RuntimeLeaseHandle &lease) override {
    return delegate_->releaseExclusiveLease(lease);
  }

  void quarantineDevice(const RuntimeDeviceHandle &device) override {
    delegate_->quarantineDevice(device);
  }

private:
  std::shared_ptr<InProcessRuntimeProvider> delegate_;
  const RuntimeProviderDescriptor *descriptor_;
};

const RuntimeProviderDescriptor &nonPortableRuntimeProviderDescriptor() {
  const RuntimeProviderDescriptor &portable =
      inProcessRuntimeProviderDescriptor();
  static const RuntimeProviderDescriptor descriptor{
      {"loom.runtime.non_portable_test", SchemaVersion{1, 0}},
      portable.implementationSemanticIdentity,
      "loom.runtime.non_portable_test.v1",
      portable.endpointKinds,
      portable.supportsHardwareReportedIdentity,
      portable.supportsTrustedImmutableIdentity,
      portable.supportsAtomicProgrammingMulticast};
  return descriptor;
}

const RuntimeProviderDescriptor &unicastRuntimeProviderDescriptor() {
  const RuntimeProviderDescriptor &multicast =
      inProcessRuntimeProviderDescriptor();
  static const RuntimeProviderDescriptor descriptor{
      {"loom.runtime.unicast_test", SchemaVersion{1, 0}},
      multicast.implementationSemanticIdentity,
      multicast.runtimeAbiIdentity,
      multicast.endpointKinds,
      multicast.supportsHardwareReportedIdentity,
      multicast.supportsTrustedImmutableIdentity,
      false};
  return descriptor;
}

ObservedLoadError expectLoadError(llvm::StringRef test,
                                  llvm::Expected<LoadedDeployment> loaded) {
  if (loaded)
    deployment::test::fail(test, "invalid load unexpectedly succeeded");
  std::optional<ObservedLoadError> observed;
  llvm::handleAllErrors(
      loaded.takeError(),
      [&](const RuntimeLoadError &error) {
        observed = ObservedLoadError{error.kind(), error.deviceQuarantined(),
                                     error.diagnostic().str()};
      },
      [&](const llvm::ErrorInfoBase &error) {
        deployment::test::fail(test, error.message());
      });
  if (!observed)
    deployment::test::fail(test, "load failure had no typed diagnostic");
  return std::move(*observed);
}

std::vector<ArtifactIdentity> implementationIdentities(
    llvm::StringRef test, const deployment::FinalizedDeployment &deployment,
    const ArtifactStore &artifacts, const BlobStore &blobs) {
  const auto bindings = deployment.deployment().hardwareBindings();
  std::vector<std::optional<ArtifactIdentity>> indexed(bindings.size());
  std::vector<ArtifactIdentity> trusted;
  for (const deployment::DeploymentHardwareBinding &binding : bindings) {
    const FinalizedRuntimePlatformBinding runtime =
        take(test, importRuntimePlatformBinding(binding.runtimePlatformBinding,
                                                artifacts, blobs));
    const auto *reported = std::get_if<HardwareReportedIdentity>(
        &runtime.binding().identityVerification());
    if (!reported) {
      trusted.push_back(binding.hardwareImplementation.artifact);
      continue;
    }
    bool found = false;
    for (std::size_t ordinal = 0; ordinal != bindings.size(); ++ordinal)
      if (reported->implementationIdentityEndpoint == inProcessRuntimeEndpoint(
              RuntimeEndpointClass::Identity, ordinal)) {
        deployment::test::require(test, !indexed[ordinal].has_value(),
                                  "identity endpoint is duplicated");
        indexed[ordinal] = binding.hardwareImplementation.artifact;
        found = true;
        break;
      }
    deployment::test::require(test, found,
                              "identity endpoint is outside the fixture");
  }
  if (!trusted.empty())
    return trusted;
  std::vector<ArtifactIdentity> result;
  result.reserve(indexed.size());
  for (const std::optional<ArtifactIdentity> &identity : indexed) {
    deployment::test::require(test, identity.has_value(),
                              "identity endpoint coverage is incomplete");
    result.push_back(*identity);
  }
  return result;
}

void loadsOneImmutableDeploymentWithAtomicConfigurationMulticast() {
  const llvm::StringRef test = __func__;
  deployment::test::TemporaryTree tree(test);
  ArtifactStore artifacts(tree.path("artifacts"));
  BlobStore blobs(tree.path("blobs"));
  const deployment::FinalizedDeployment deployment =
      deployment::test::buildMinimalDeployment(test, artifacts, blobs, tree);
  deployment::test::require(
      test, deployment.deployment().configurationImages().size() == 2,
      "fixture must contain one configuration image per SpatialCore");
  const auto implementations =
      implementationIdentities(test, deployment, artifacts, blobs);
  auto provider = take(test, createInProcessRuntimeProvider(
                                 {{implementations, std::nullopt, {}},
                                  {implementations, std::nullopt, {}}}));

  {
    auto loaded =
        take(test, loadDeployment(deployment, {provider, 0}, artifacts, blobs));
    deployment::test::require(
        test, loaded.deployment().reference() == deployment.reference(),
        "transient device selection changed Deployment identity");
  }
  {
    auto loaded =
        take(test, loadDeployment(deployment, {provider, 1}, artifacts, blobs));
    deployment::test::require(
        test, loaded.deployment().reference() == deployment.reference(),
        "second transient device changed Deployment identity");
  }

  const InProcessRuntimeStatistics statistics = provider->statistics();
  deployment::test::require(
      test, statistics.multicastTransactionCount == 2,
      "each homogeneous load was not one atomic multicast transaction");
  deployment::test::require(
      test,
      statistics.configurationReadCount ==
          statistics.configurationWriteCount +
              2 * deployment.deployment().configurationImages().size(),
      "configuration payloads and status were not read back independently");
  deployment::test::require(
      test,
      statistics.executableRegistrationCount == 2 &&
          statistics.activationCount == 2 && statistics.resetCount == 4 &&
          statistics.leaseReleaseCount == 2 && statistics.quarantineCount == 0,
      "successful load lifecycle counts: registration=" +
          std::to_string(statistics.executableRegistrationCount) +
          " activation=" + std::to_string(statistics.activationCount) +
          " reset=" + std::to_string(statistics.resetCount) +
          " release=" + std::to_string(statistics.leaseReleaseCount) +
          " quarantine=" + std::to_string(statistics.quarantineCount));
}

void rejectsReadbackMismatchAndRestoresCleanState() {
  const llvm::StringRef test = __func__;
  deployment::test::TemporaryTree tree(test);
  ArtifactStore artifacts(tree.path("artifacts"));
  BlobStore blobs(tree.path("blobs"));
  const deployment::FinalizedDeployment deployment =
      deployment::test::buildMinimalDeployment(test, artifacts, blobs, tree);
  const auto implementations =
      implementationIdentities(test, deployment, artifacts, blobs);
  auto provider = take(
      test,
      createInProcessRuntimeProvider(
          {{implementations,
            std::nullopt,
            {std::nullopt, InProcessRuntimeReadbackCorruption{0, 1}, false}}}));

  const ObservedLoadError error = expectLoadError(
      test, loadDeployment(deployment, {provider, 0}, artifacts, blobs));
  const InProcessRuntimeStatistics statistics = provider->statistics();
  deployment::test::require(
      test,
      error.kind == RuntimeLoadFailureKind::Programming && !error.quarantined &&
          llvm::StringRef(error.diagnostic).contains("readback mismatch"),
      "readback mismatch did not remain a typed programming failure");
  deployment::test::require(
      test,
      statistics.resetCount == 2 && statistics.leaseReleaseCount == 1 &&
          statistics.activationCount == 0 && statistics.quarantineCount == 0,
      "recoverable readback mismatch did not restore a clean device");
}

void rejectsInterruptedAtomicProgrammingAndRestoresCleanState() {
  const llvm::StringRef test = __func__;
  deployment::test::TemporaryTree tree(test);
  ArtifactStore artifacts(tree.path("artifacts"));
  BlobStore blobs(tree.path("blobs"));
  const deployment::FinalizedDeployment deployment =
      deployment::test::buildMinimalDeployment(test, artifacts, blobs, tree);
  const auto implementations =
      implementationIdentities(test, deployment, artifacts, blobs);
  auto provider = take(
      test, createInProcessRuntimeProvider(
                {{implementations, std::nullopt,
                  {1, std::nullopt, false}}}));

  const ObservedLoadError error = expectLoadError(
      test, loadDeployment(deployment, {provider, 0}, artifacts, blobs));
  const InProcessRuntimeStatistics statistics = provider->statistics();
  deployment::test::require(
      test,
      error.kind == RuntimeLoadFailureKind::Programming && !error.quarantined &&
          llvm::StringRef(error.diagnostic)
              .contains("atomic configuration multicast failed"),
      "interrupted programming did not remain a typed programming failure");
  deployment::test::require(
      test,
      statistics.multicastTransactionCount == 1 && statistics.resetCount == 2 &&
          statistics.leaseReleaseCount == 1 &&
          statistics.activationCount == 0 && statistics.quarantineCount == 0,
      "interrupted programming did not restore a clean device");
}

void quarantinesDeviceWhenRecoveryIdentityCannotBeProven() {
  const llvm::StringRef test = __func__;
  deployment::test::TemporaryTree tree(test);
  ArtifactStore artifacts(tree.path("artifacts"));
  BlobStore blobs(tree.path("blobs"));
  const deployment::FinalizedDeployment deployment =
      deployment::test::buildMinimalDeployment(test, artifacts, blobs, tree);
  const auto implementations =
      implementationIdentities(test, deployment, artifacts, blobs);
  auto provider = take(
      test,
      createInProcessRuntimeProvider(
          {{implementations,
            std::nullopt,
            {std::nullopt, InProcessRuntimeReadbackCorruption{0, 1}, true}}}));

  const ObservedLoadError error = expectLoadError(
      test, loadDeployment(deployment, {provider, 0}, artifacts, blobs));
  const InProcessRuntimeStatistics statistics = provider->statistics();
  deployment::test::require(
      test,
      error.kind == RuntimeLoadFailureKind::Programming && error.quarantined &&
          provider->isQuarantined(0) &&
          llvm::StringRef(error.diagnostic)
              .contains("recovery identity check failed"),
      "unproven recovery did not quarantine the selected device");
  deployment::test::require(test,
                            statistics.resetCount == 2 &&
                                statistics.leaseReleaseCount == 1 &&
                                statistics.quarantineCount == 1,
                            "quarantine lifecycle counts are inconsistent");
}

void rejectsSelectedForeignImplementationWithoutDeviceFallback() {
  const llvm::StringRef test = __func__;
  deployment::test::TemporaryTree tree(test);
  ArtifactStore artifacts(tree.path("artifacts"));
  BlobStore blobs(tree.path("blobs"));
  const deployment::FinalizedDeployment deployment =
      deployment::test::buildMinimalDeployment(test, artifacts, blobs, tree);
  const auto implementations =
      implementationIdentities(test, deployment, artifacts, blobs);
  std::vector<ArtifactIdentity> foreignImplementations;
  foreignImplementations.reserve(implementations.size());
  for (const ArtifactIdentity &implementation : implementations) {
    ArtifactIdentity::Storage foreignBytes = implementation.bytes();
    foreignBytes.front() ^= 1;
    foreignImplementations.push_back(
        llvm::cantFail(ArtifactIdentity::fromBytes(foreignBytes)));
  }
  auto provider = take(test, createInProcessRuntimeProvider(
                                 {{foreignImplementations, std::nullopt, {}},
                                  {implementations, std::nullopt, {}}}));

  const ObservedLoadError error = expectLoadError(
      test, loadDeployment(deployment, {provider, 0}, artifacts, blobs));
  const InProcessRuntimeStatistics statistics = provider->statistics();
  deployment::test::require(
      test,
      error.kind == RuntimeLoadFailureKind::IdentityVerification &&
          !error.quarantined && statistics.enumerationCount == 1 &&
          statistics.identityReadCount == 1 &&
          statistics.leaseAcquisitionCount == 0 &&
          statistics.activationCount == 0,
      "foreign selected device triggered fallback or reached activation");
}

void rejectsStaleTrustedAttestationBeforeLease() {
  const llvm::StringRef test = __func__;
  deployment::test::TemporaryTree tree(test);
  ArtifactStore artifacts(tree.path("artifacts"));
  BlobStore blobs(tree.path("blobs"));
  const deployment::FinalizedDeployment deployment =
      deployment::test::buildTrustedIdentityDeployment(test, artifacts, blobs,
                                                       tree);
  const auto implementations =
      implementationIdentities(test, deployment, artifacts, blobs);
  constexpr llvm::StringLiteral stale = "stale implementation attestation";
  const BlobDigest staleDigest = computeBlobDigest(llvm::ArrayRef<std::uint8_t>(
      reinterpret_cast<const std::uint8_t *>(stale.data()), stale.size()));
  auto provider =
      take(test,
           createInProcessRuntimeProvider(
               {{implementations, staleDigest, {}}}));

  const ObservedLoadError error = expectLoadError(
      test, loadDeployment(deployment, {provider, 0}, artifacts, blobs));
  const InProcessRuntimeStatistics statistics = provider->statistics();
  deployment::test::require(
      test,
      error.kind == RuntimeLoadFailureKind::IdentityVerification &&
          !error.quarantined &&
          llvm::StringRef(error.diagnostic)
              .contains("stale trusted attestation") &&
          statistics.leaseAcquisitionCount == 0 &&
          statistics.activationCount == 0,
      "stale attestation reached the leased or active device state");
}

void rejectsProgrammingEndpointAliasedAcrossSpatialCores() {
  const llvm::StringRef test = __func__;
  deployment::test::TemporaryTree tree(test);
  ArtifactStore artifacts(tree.path("artifacts"));
  BlobStore blobs(tree.path("blobs"));
  const deployment::FinalizedDeployment deployment =
      deployment::test::buildSharedProgrammingEndpointDeployment(
          test, artifacts, blobs, tree);
  const auto implementations =
      implementationIdentities(test, deployment, artifacts, blobs);
  auto provider = take(test, createInProcessRuntimeProvider(
                                 {{implementations, std::nullopt, {}}}));

  const ObservedLoadError error = expectLoadError(
      test, loadDeployment(deployment, {provider, 0}, artifacts, blobs));
  deployment::test::require(
      test,
      error.kind == RuntimeLoadFailureKind::InvalidDeployment &&
          llvm::StringRef(error.diagnostic)
              .contains("aliases multiple SpatialCore transports") &&
          provider->statistics().enumerationCount == 0,
      "cross-SpatialCore endpoint alias reached provider enumeration");
}

void rejectsUnregisteredDescriptorAliasBeforeEnumeration() {
  const llvm::StringRef test = __func__;
  deployment::test::TemporaryTree tree(test);
  ArtifactStore artifacts(tree.path("artifacts"));
  BlobStore blobs(tree.path("blobs"));
  const deployment::FinalizedDeployment deployment =
      deployment::test::buildMinimalDeployment(test, artifacts, blobs, tree);
  const auto implementations =
      implementationIdentities(test, deployment, artifacts, blobs);
  auto delegate = take(test, createInProcessRuntimeProvider(
                                 {{implementations, std::nullopt, {}}}));
  RuntimeProviderDescriptor alias = inProcessRuntimeProviderDescriptor();
  auto provider = std::make_shared<ForwardingRuntimeProvider>(delegate, alias);

  const ObservedLoadError error = expectLoadError(
      test, loadDeployment(deployment, {provider, 0}, artifacts, blobs));
  deployment::test::require(
      test,
      error.kind == RuntimeLoadFailureKind::ProviderMismatch &&
          delegate->statistics().enumerationCount == 0,
      "descriptor alias was accepted as the registered provider owner");
}

void rejectsNonPortableRuntimeAbiBeforeEnumeration() {
  const llvm::StringRef test = __func__;
  deployment::test::TemporaryTree tree(test);
  ArtifactStore artifacts(tree.path("artifacts"));
  BlobStore blobs(tree.path("blobs"));
  const RuntimeProviderDescriptor &descriptor =
      nonPortableRuntimeProviderDescriptor();
  const deployment::FinalizedDeployment deployment =
      deployment::test::buildRuntimeProviderDeployment(test, artifacts, blobs,
                                                       tree, descriptor);
  const auto implementations =
      implementationIdentities(test, deployment, artifacts, blobs);
  auto delegate = take(test, createInProcessRuntimeProvider(
                                 {{implementations, std::nullopt, {}}}));
  auto provider =
      std::make_shared<ForwardingRuntimeProvider>(delegate, descriptor);

  const ObservedLoadError error = expectLoadError(
      test, loadDeployment(deployment, {provider, 0}, artifacts, blobs));
  deployment::test::require(
      test,
      error.kind == RuntimeLoadFailureKind::ProviderMismatch &&
          llvm::StringRef(error.diagnostic)
              .contains("portable configuration runtime ABI") &&
          delegate->statistics().enumerationCount == 0,
      "non-portable runtime ABI reached provider enumeration");
}

void loadsThroughCanonicalUnicastProvider() {
  const llvm::StringRef test = __func__;
  deployment::test::TemporaryTree tree(test);
  ArtifactStore artifacts(tree.path("artifacts"));
  BlobStore blobs(tree.path("blobs"));
  const RuntimeProviderDescriptor &descriptor =
      unicastRuntimeProviderDescriptor();
  const deployment::FinalizedDeployment deployment =
      deployment::test::buildRuntimeProviderDeployment(test, artifacts, blobs,
                                                       tree, descriptor);
  const auto implementations =
      implementationIdentities(test, deployment, artifacts, blobs);
  auto delegate = take(test, createInProcessRuntimeProvider(
                                 {{implementations, std::nullopt, {}}}));
  auto provider =
      std::make_shared<ForwardingRuntimeProvider>(delegate, descriptor);

  {
    auto loaded =
        take(test, loadDeployment(deployment, {provider, 0}, artifacts, blobs));
    deployment::test::require(
        test, loaded.deployment().reference() == deployment.reference(),
        "unicast provider changed Deployment identity");
  }
  const InProcessRuntimeStatistics statistics = delegate->statistics();
  deployment::test::require(
      test,
      statistics.multicastTransactionCount == 0 &&
          statistics.configurationWriteCount > 0 &&
          statistics.configurationCommitCount ==
              deployment.deployment().configurationImages().size() &&
          statistics.activationCount == 1 && statistics.quarantineCount == 0,
      "canonical unicast provider did not complete independent programming");
}

void ignoresAbiUnusedHighReadbackBits() {
  const llvm::StringRef test = __func__;
  deployment::test::TemporaryTree tree(test);
  ArtifactStore artifacts(tree.path("artifacts"));
  BlobStore blobs(tree.path("blobs"));
  const deployment::FinalizedDeployment deployment =
      deployment::test::buildMinimalDeployment(test, artifacts, blobs, tree);
  const auto implementations =
      implementationIdentities(test, deployment, artifacts, blobs);
  auto provider = take(
      test, createInProcessRuntimeProvider(
                {{implementations,
                  std::nullopt,
                  {std::nullopt,
                   InProcessRuntimeReadbackCorruption{0, UINT32_C(0x80000000)},
                   false}}}));

  {
    auto loaded =
        take(test, loadDeployment(deployment, {provider, 0}, artifacts, blobs));
    deployment::test::require(
        test, loaded.deployment().reference() == deployment.reference(),
        "unused readback bits changed Deployment identity");
  }
  deployment::test::require(
      test,
      provider->statistics().activationCount == 1 &&
          provider->statistics().quarantineCount == 0,
      "ABI-unused readback bits were treated as active configuration");
}

void activationKeysIgnorePartitionsAndDistinguishRootedLaunches() {
  const llvm::StringRef test = __func__;
  const ArtifactIdentity dataflowIdentity = identity(test, 11);
  const dataflow::RootThreadLaunchRef root{dataflowIdentity,
                                           dataflow::RootThreadLaunchId(3)};
  const dataflow::RootedGraphLaunchRef first{
      root, dataflow::StaticGraphLaunchRef{dataflowIdentity,
                                           dataflow::StaticGraphLaunchId(5)}};
  const dataflow::RootedGraphLaunchRef second{
      root, dataflow::StaticGraphLaunchRef{dataflowIdentity,
                                           dataflow::StaticGraphLaunchId(7)}};
  const ArtifactIdentity firstMapping = identity(test, 31);
  const ArtifactIdentity secondMapping = identity(test, 51);
  const loom::fabric::AccCoreOccurrenceRef firstCore(0);
  const loom::fabric::AccCoreOccurrenceRef secondCore(1);
  const loom::mapping::SystemPresburgerCell even{1, 0, 1, {{1, -2, 0}}, {}};
  const loom::mapping::SystemPresburgerCell odd{1, 0, 1, {{1, -2, -1}}, {}};
  const std::vector<loom::mapping::SystemSpatialContextDomain> sameEvent{
      {first,
       {loom::mapping::mappingArtifactSchema.identity.str(),
        loom::mapping::mappingArtifactSchema.version, firstMapping},
       {firstCore, firstMapping},
       {even}},
      {first,
       {loom::mapping::mappingArtifactSchema.identity.str(),
        loom::mapping::mappingArtifactSchema.version, secondMapping},
       {secondCore, secondMapping},
       {odd}}};
  const auto firstKey =
      take(test, loom::runtime::detail::configurationActivationEventKey(
                     dataflowIdentity, sameEvent,
                     loom::fabric::SpatialCoreOccurrenceRef{firstCore}));
  const auto secondKey =
      take(test, loom::runtime::detail::configurationActivationEventKey(
                     dataflowIdentity, sameEvent,
                     loom::fabric::SpatialCoreOccurrenceRef{secondCore}));
  deployment::test::require(
      test, firstKey == secondKey,
      "partition or SpatialMapping identity changed graph-start event key");

  auto differentEvent = sameEvent;
  differentEvent.back().graph = second;
  const auto differentKey =
      take(test, loom::runtime::detail::configurationActivationEventKey(
                     dataflowIdentity, differentEvent,
                     loom::fabric::SpatialCoreOccurrenceRef{secondCore}));
  deployment::test::require(
      test, firstKey != differentKey,
      "distinct rooted graph starts collapsed to one activation event key");
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 2)
    deployment::test::fail("deployment_loader", "expected one scenario name");
  const llvm::StringRef scenario = argv[1];
  if (scenario == "load")
    loadsOneImmutableDeploymentWithAtomicConfigurationMulticast();
  else if (scenario == "readback")
    rejectsReadbackMismatchAndRestoresCleanState();
  else if (scenario == "interrupted")
    rejectsInterruptedAtomicProgrammingAndRestoresCleanState();
  else if (scenario == "quarantine")
    quarantinesDeviceWhenRecoveryIdentityCannotBeProven();
  else if (scenario == "foreign")
    rejectsSelectedForeignImplementationWithoutDeviceFallback();
  else if (scenario == "attestation")
    rejectsStaleTrustedAttestationBeforeLease();
  else if (scenario == "endpoint-alias")
    rejectsProgrammingEndpointAliasedAcrossSpatialCores();
  else if (scenario == "descriptor-alias")
    rejectsUnregisteredDescriptorAliasBeforeEnumeration();
  else if (scenario == "non-portable")
    rejectsNonPortableRuntimeAbiBeforeEnumeration();
  else if (scenario == "unicast")
    loadsThroughCanonicalUnicastProvider();
  else if (scenario == "unused-high-bits")
    ignoresAbiUnusedHighReadbackBits();
  else if (scenario == "activation-key")
    activationKeysIgnorePartitionsAndDistinguishRootedLaunches();
  else
    deployment::test::fail("deployment_loader", "unknown scenario");
  return 0;
}
