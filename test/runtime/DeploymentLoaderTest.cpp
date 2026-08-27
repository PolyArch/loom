#include "DeploymentTestSupport.h"

#include "Common/ArtifactStore.h"
#include "Common/BlobStore.h"
#include "Deployment/HardwareConfigurationImage.h"
#include "Hardware/Configuration/ConfigurationABI.h"
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
  RuntimeLoadTerminalDisposition terminalDisposition =
      RuntimeLoadTerminalDisposition::NoLeaseAcquired;
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

  llvm::Expected<RuntimeLeaseHandle>
  acquireExclusiveLease(const RuntimeDeviceHandle &device) override {
    return delegate_->acquireExclusiveLease(device);
  }

  llvm::Expected<ArtifactIdentity> readImplementationIdentity(
      const RuntimeLeaseHandle &lease,
      const RuntimeProviderEndpointRef &endpoint) override {
    return delegate_->readImplementationIdentity(lease, endpoint);
  }

  llvm::Expected<BlobDigest>
  readTrustedAttestation(const RuntimeLeaseHandle &lease) override {
    return delegate_->readTrustedAttestation(lease);
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

  llvm::Expected<RuntimePreparedActivationHandle>
  prepareActivation(const RuntimeLeaseHandle &lease,
                    const RuntimeExecutableRegistrationView &registration,
                    const RuntimeActivationView &activation) override {
    return delegate_->prepareActivation(lease, registration, activation);
  }

  llvm::Error replaceActivationAtomically(
      const RuntimeLeaseHandle &lease,
      const RuntimePreparedActivationHandle &prepared) override {
    return delegate_->replaceActivationAtomically(lease, prepared);
  }

  llvm::Error discardPreparedActivation(
      const RuntimeLeaseHandle &lease,
      const RuntimePreparedActivationHandle &prepared) override {
    return delegate_->discardPreparedActivation(lease, prepared);
  }

  RuntimeLeaseFinalizationResult
  finalizeExclusiveLease(const RuntimeLeaseHandle &lease,
                         RuntimeLeaseFinalizationRequest request) override {
    return delegate_->finalizeExclusiveLease(lease, request);
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
      portable.supportsAtomicProgrammingMulticast,
      portable.supportsPreparedActivationReplacement};
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
      false,
      multicast.supportsPreparedActivationReplacement};
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
        observed = ObservedLoadError{error.kind(), error.terminalDisposition(),
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
      if (reported->implementationIdentityEndpoint ==
          inProcessRuntimeEndpoint(RuntimeEndpointClass::Identity, ordinal)) {
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

BlobDigest trustedAttestation(llvm::StringRef test,
                              const deployment::FinalizedDeployment &deployment,
                              const ArtifactStore &artifacts,
                              const BlobStore &blobs) {
  for (const deployment::DeploymentHardwareBinding &binding :
       deployment.deployment().hardwareBindings()) {
    const FinalizedRuntimePlatformBinding runtime =
        take(test, importRuntimePlatformBinding(binding.runtimePlatformBinding,
                                                artifacts, blobs));
    if (const auto *trusted = std::get_if<TrustedImmutableIdentity>(
            &runtime.binding().identityVerification()))
      return trusted->attestationBlob;
  }
  deployment::test::fail(test,
                         "fixture has no trusted immutable identity binding");
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
  auto provider =
      take(test, createInProcessRuntimeProvider(
                     {{implementations,
                       std::nullopt,
                       {std::nullopt, InProcessRuntimeReadbackCorruption{0, 1},
                        std::nullopt, std::nullopt, 0, 0}}}));

  const ObservedLoadError error = expectLoadError(
      test, loadDeployment(deployment, {provider, 0}, artifacts, blobs));
  const InProcessRuntimeStatistics statistics = provider->statistics();
  deployment::test::require(
      test,
      error.kind == RuntimeLoadFailureKind::Programming &&
          error.terminalDisposition ==
              RuntimeLoadTerminalDisposition::LeaseReleased &&
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
  auto provider =
      take(test, createInProcessRuntimeProvider(
                     {{implementations,
                       std::nullopt,
                       {1, std::nullopt, std::nullopt, std::nullopt, 0, 0}}}));

  const ObservedLoadError error = expectLoadError(
      test, loadDeployment(deployment, {provider, 0}, artifacts, blobs));
  const InProcessRuntimeStatistics statistics = provider->statistics();
  deployment::test::require(
      test,
      error.kind == RuntimeLoadFailureKind::Programming &&
          error.terminalDisposition ==
              RuntimeLoadTerminalDisposition::LeaseReleased &&
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
      test, createInProcessRuntimeProvider(
                {{implementations,
                  std::nullopt,
                  {std::nullopt, InProcessRuntimeReadbackCorruption{0, 1},
                   InProcessRuntimeVerificationMismatchBoundary::RecoveryReset,
                   std::nullopt, 0, 0}}}));

  const ObservedLoadError error = expectLoadError(
      test, loadDeployment(deployment, {provider, 0}, artifacts, blobs));
  const InProcessRuntimeStatistics statistics = provider->statistics();
  deployment::test::require(
      test,
      error.kind == RuntimeLoadFailureKind::Programming &&
          error.terminalDisposition ==
              RuntimeLoadTerminalDisposition::DeviceQuarantined &&
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
          error.terminalDisposition ==
              RuntimeLoadTerminalDisposition::LeaseReleased &&
          statistics.enumerationCount == 1 &&
          statistics.identityReadCount == 1 &&
          statistics.leaseAcquisitionCount == 1 &&
          statistics.leaseReleaseCount == 1 && statistics.resetCount == 0 &&
          statistics.activationCount == 0,
      "foreign selected device triggered fallback or reached activation");
}

void rejectsStaleTrustedAttestationUnderLease() {
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
  auto provider = take(test, createInProcessRuntimeProvider(
                                 {{implementations, staleDigest, {}}}));

  const ObservedLoadError error = expectLoadError(
      test, loadDeployment(deployment, {provider, 0}, artifacts, blobs));
  const InProcessRuntimeStatistics statistics = provider->statistics();
  deployment::test::require(
      test,
      error.kind == RuntimeLoadFailureKind::IdentityVerification &&
          error.terminalDisposition ==
              RuntimeLoadTerminalDisposition::LeaseReleased &&
          llvm::StringRef(error.diagnostic)
              .contains("stale trusted attestation") &&
          statistics.leaseAcquisitionCount == 1 &&
          statistics.leaseReleaseCount == 1 && statistics.resetCount == 0 &&
          statistics.activationCount == 0,
      "stale attestation reached reset or the active device state");
}

void rejectsTrustedAttestationChangeAcrossInitialReset() {
  const llvm::StringRef test = __func__;
  deployment::test::TemporaryTree tree(test);
  ArtifactStore artifacts(tree.path("artifacts"));
  BlobStore blobs(tree.path("blobs"));
  const deployment::FinalizedDeployment deployment =
      deployment::test::buildTrustedIdentityDeployment(test, artifacts, blobs,
                                                       tree);
  const auto implementations =
      implementationIdentities(test, deployment, artifacts, blobs);
  InProcessRuntimeFailurePlan failure;
  failure.verificationMismatchBoundary =
      InProcessRuntimeVerificationMismatchBoundary::InitialReset;
  auto provider =
      take(test, createInProcessRuntimeProvider(
                     {{implementations,
                       trustedAttestation(test, deployment, artifacts, blobs),
                       std::move(failure)}}));

  const ObservedLoadError error = expectLoadError(
      test, loadDeployment(deployment, {provider, 0}, artifacts, blobs));
  const InProcessRuntimeStatistics statistics = provider->statistics();
  deployment::test::require(
      test,
      error.kind == RuntimeLoadFailureKind::IdentityVerification &&
          error.terminalDisposition ==
              RuntimeLoadTerminalDisposition::DeviceQuarantined &&
          provider->isQuarantined(0) &&
          llvm::StringRef(error.diagnostic)
              .contains("post-reset identity verification failed") &&
          statistics.attestationReadCount == implementations.size() + 2 &&
          statistics.leaseAcquisitionCount == 1 &&
          statistics.leaseReleaseCount == 1 && statistics.resetCount == 2 &&
          statistics.configurationWriteCount == 0 &&
          statistics.quarantineCount == 1,
      "reset attestation change reached package installation or reuse");
}

void rejectsIdentityChangeAtExclusiveLeaseBoundary() {
  const llvm::StringRef test = __func__;
  deployment::test::TemporaryTree tree(test);
  ArtifactStore artifacts(tree.path("artifacts"));
  BlobStore blobs(tree.path("blobs"));
  const deployment::FinalizedDeployment deployment =
      deployment::test::buildMinimalDeployment(test, artifacts, blobs, tree);
  const auto implementations =
      implementationIdentities(test, deployment, artifacts, blobs);
  InProcessRuntimeFailurePlan failure;
  failure.verificationMismatchBoundary =
      InProcessRuntimeVerificationMismatchBoundary::ExclusiveLease;
  auto provider =
      take(test, createInProcessRuntimeProvider(
                     {{implementations, std::nullopt, std::move(failure)}}));

  const ObservedLoadError error = expectLoadError(
      test, loadDeployment(deployment, {provider, 0}, artifacts, blobs));
  const InProcessRuntimeStatistics statistics = provider->statistics();
  deployment::test::require(
      test,
      error.kind == RuntimeLoadFailureKind::IdentityVerification &&
          error.terminalDisposition ==
              RuntimeLoadTerminalDisposition::LeaseReleased &&
          llvm::StringRef(error.diagnostic)
              .contains("leased device identity verification failed") &&
          statistics.identityReadCount == 1 &&
          statistics.leaseAcquisitionCount == 1 &&
          statistics.leaseReleaseCount == 1 && statistics.resetCount == 0 &&
          statistics.configurationWriteCount == 0 &&
          statistics.activationCount == 0,
      "identity changed between enumeration and lease without typed refusal");
}

void rejectsIdentityChangeAcrossInitialReset() {
  const llvm::StringRef test = __func__;
  deployment::test::TemporaryTree tree(test);
  ArtifactStore artifacts(tree.path("artifacts"));
  BlobStore blobs(tree.path("blobs"));
  const deployment::FinalizedDeployment deployment =
      deployment::test::buildMinimalDeployment(test, artifacts, blobs, tree);
  const auto implementations =
      implementationIdentities(test, deployment, artifacts, blobs);
  InProcessRuntimeFailurePlan failure;
  failure.verificationMismatchBoundary =
      InProcessRuntimeVerificationMismatchBoundary::InitialReset;
  auto provider =
      take(test, createInProcessRuntimeProvider(
                     {{implementations, std::nullopt, std::move(failure)}}));

  const ObservedLoadError error = expectLoadError(
      test, loadDeployment(deployment, {provider, 0}, artifacts, blobs));
  const InProcessRuntimeStatistics statistics = provider->statistics();
  deployment::test::require(
      test,
      error.kind == RuntimeLoadFailureKind::IdentityVerification &&
          error.terminalDisposition ==
              RuntimeLoadTerminalDisposition::DeviceQuarantined &&
          provider->isQuarantined(0) &&
          llvm::StringRef(error.diagnostic)
              .contains("post-reset identity verification failed") &&
          llvm::StringRef(error.diagnostic)
              .contains("recovery identity check failed") &&
          statistics.identityReadCount == implementations.size() + 2 &&
          statistics.leaseAcquisitionCount == 1 &&
          statistics.leaseReleaseCount == 1 && statistics.resetCount == 2 &&
          statistics.configurationWriteCount == 0 &&
          statistics.activationCount == 0 && statistics.quarantineCount == 1,
      "reset identity change reached package installation or escaped recovery");
}

void enforcesLeaseIdentityAndTerminalDispositionContract() {
  const llvm::StringRef test = __func__;
  const ArtifactIdentity firstIdentity = identity(test, 3);
  const ArtifactIdentity secondIdentity = identity(test, 79);
  auto provider = take(test, createInProcessRuntimeProvider(
                                 {{{firstIdentity}, std::nullopt, {}},
                                  {{secondIdentity}, std::nullopt, {}}}));
  const auto devices = take(test, provider->enumerateDevices());
  deployment::test::require(test, devices.size() == 2,
                            "provider did not enumerate both devices");
  const RuntimeProviderEndpointRef endpoint =
      inProcessRuntimeEndpoint(RuntimeEndpointClass::Identity, 0);

  const RuntimeLeaseHandle firstLease =
      take(test, provider->acquireExclusiveLease(devices[0]));
  auto duplicate = provider->acquireExclusiveLease(devices[0]);
  deployment::test::require(test, !duplicate,
                            "one device admitted two exclusive leases");
  llvm::consumeError(duplicate.takeError());
  deployment::test::require(
      test,
      take(test, provider->readImplementationIdentity(firstLease, endpoint)) ==
          firstIdentity,
      "lease identity read observed another device");
  const RuntimeLeaseFinalizationResult firstFinalization =
      provider->finalizeExclusiveLease(
          firstLease, RuntimeLeaseFinalizationRequest::Release);
  deployment::test::require(
      test,
      firstFinalization.state == RuntimeLeaseFinalState::Released &&
          firstFinalization.diagnostic.empty(),
      "clean lease release did not produce its exact terminal state");

  auto stale = provider->readImplementationIdentity(firstLease, endpoint);
  deployment::test::require(test, !stale,
                            "released lease remained valid for identity read");
  llvm::consumeError(stale.takeError());
  const RuntimeLeaseHandle replacementLease =
      take(test, provider->acquireExclusiveLease(devices[0]));
  deployment::test::require(test, !(replacementLease == firstLease),
                            "lease generation did not change on reacquisition");
  auto reboundStale =
      provider->readImplementationIdentity(firstLease, endpoint);
  deployment::test::require(
      test, !reboundStale,
      "old lease generation became valid after device reacquisition");
  llvm::consumeError(reboundStale.takeError());
  const RuntimeLeaseFinalizationResult quarantineFinalization =
      provider->finalizeExclusiveLease(
          replacementLease, RuntimeLeaseFinalizationRequest::Quarantine);
  deployment::test::require(
      test,
      quarantineFinalization.state == RuntimeLeaseFinalState::Quarantined &&
          quarantineFinalization.diagnostic.empty(),
      "explicit quarantine did not produce its exact terminal state");
  auto quarantined = provider->acquireExclusiveLease(devices[0]);
  deployment::test::require(test, !quarantined,
                            "quarantined device admitted another lease");
  llvm::consumeError(quarantined.takeError());

  const RuntimeLeaseHandle independentLease =
      take(test, provider->acquireExclusiveLease(devices[1]));
  deployment::test::require(test,
                            take(test, provider->readImplementationIdentity(
                                           independentLease, endpoint)) ==
                                secondIdentity,
                            "lease identity was not bound to its exact device");
  const RuntimeLeaseFinalizationResult independentFinalization =
      provider->finalizeExclusiveLease(
          independentLease, RuntimeLeaseFinalizationRequest::Release);
  deployment::test::require(test,
                            independentFinalization.state ==
                                    RuntimeLeaseFinalState::Released &&
                                independentFinalization.diagnostic.empty(),
                            "independent lease did not release cleanly");

  const InProcessRuntimeStatistics statistics = provider->statistics();
  deployment::test::require(
      test,
      statistics.leaseAcquisitionCount == 3 &&
          statistics.leaseReleaseCount == 3 && statistics.quarantineCount == 1,
      "lease terminal disposition counters are inconsistent");
}

void quarantinesAtomicallyWhenOrdinaryReleaseFails() {
  const llvm::StringRef test = __func__;
  deployment::test::TemporaryTree tree(test);
  ArtifactStore artifacts(tree.path("artifacts"));
  BlobStore blobs(tree.path("blobs"));
  const deployment::FinalizedDeployment deployment =
      deployment::test::buildMinimalDeployment(test, artifacts, blobs, tree);
  const auto implementations =
      implementationIdentities(test, deployment, artifacts, blobs);
  InProcessRuntimeFailurePlan failure;
  failure.verificationMismatchBoundary =
      InProcessRuntimeVerificationMismatchBoundary::ExclusiveLease;
  failure.leaseReleaseFailures = 1;
  auto provider =
      take(test, createInProcessRuntimeProvider(
                     {{implementations, std::nullopt, std::move(failure)}}));

  const ObservedLoadError error = expectLoadError(
      test, loadDeployment(deployment, {provider, 0}, artifacts, blobs));
  const InProcessRuntimeStatistics statistics = provider->statistics();
  deployment::test::require(
      test,
      error.kind == RuntimeLoadFailureKind::IdentityVerification &&
          error.terminalDisposition ==
              RuntimeLoadTerminalDisposition::DeviceQuarantined &&
          provider->isQuarantined(0) &&
          llvm::StringRef(error.diagnostic).contains("lease release failed") &&
          statistics.leaseAcquisitionCount == 1 &&
          statistics.leaseReleaseCount == 1 &&
          statistics.quarantineCount == 1 && statistics.resetCount == 0,
      "release failure did not establish an atomic quarantine disposition");
}

void quarantinesWhenLeaseReleaseCannotComplete() {
  const llvm::StringRef test = __func__;
  deployment::test::TemporaryTree tree(test);
  ArtifactStore artifacts(tree.path("artifacts"));
  BlobStore blobs(tree.path("blobs"));
  const deployment::FinalizedDeployment deployment =
      deployment::test::buildMinimalDeployment(test, artifacts, blobs, tree);
  const auto implementations =
      implementationIdentities(test, deployment, artifacts, blobs);
  const ArtifactIdentity independentIdentity = identity(test, 131);
  constexpr llvm::StringLiteral targetMachine = "quarantine-failclosed-target";
  constexpr llvm::StringLiteral independentMachine =
      "quarantine-failclosed-independent";
  InProcessRuntimeFailurePlan failure;
  failure.verificationMismatchBoundary =
      InProcessRuntimeVerificationMismatchBoundary::InitialReset;
  failure.quarantineLeaseReleaseFailures = 1;
  auto provider = take(test, createInProcessRuntimeProvider(
                                 {{implementations, std::nullopt,
                                   std::move(failure), targetMachine.str()},
                                  {{independentIdentity},
                                   std::nullopt,
                                   {},
                                   independentMachine.str()}}));

  const ObservedLoadError first = expectLoadError(
      test, loadDeployment(deployment, {provider, 0}, artifacts, blobs));
  const InProcessRuntimeStatistics statistics = provider->statistics();
  deployment::test::require(
      test,
      first.kind == RuntimeLoadFailureKind::IdentityVerification &&
          first.terminalDisposition ==
              RuntimeLoadTerminalDisposition::DeviceQuarantined &&
          llvm::StringRef(first.diagnostic)
              .contains("quarantined lease remains provider-owned") &&
          provider->isQuarantined(0) && statistics.leaseAcquisitionCount == 1 &&
          statistics.leaseReleaseCount == 0 && statistics.quarantineCount == 1,
      "unreleased lease did not establish its quarantine owner");
  provider.reset();

  auto replacement =
      take(test, createInProcessRuntimeProvider(
                     {{implementations, std::nullopt, {}, targetMachine.str()},
                      {{independentIdentity},
                       std::nullopt,
                       {},
                       independentMachine.str()}}));
  const auto available = take(test, replacement->enumerateDevices());
  deployment::test::require(
      test,
      available.size() == 1 && replacement->isQuarantined(0) &&
          !replacement->isQuarantined(1),
      "provider teardown lost quarantine or isolated an independent device");
  const RuntimeLeaseHandle independentLease =
      take(test, replacement->acquireExclusiveLease(available.front()));
  deployment::test::require(
      test,
      take(test, replacement->readImplementationIdentity(
                     independentLease,
                     inProcessRuntimeEndpoint(RuntimeEndpointClass::Identity,
                                              0))) == independentIdentity,
      "post-quarantine enumeration selected the quarantined machine device");
  const RuntimeLeaseFinalizationResult finalization =
      replacement->finalizeExclusiveLease(
          independentLease, RuntimeLeaseFinalizationRequest::Release);
  deployment::test::require(test,
                            finalization.state ==
                                    RuntimeLeaseFinalState::Released &&
                                finalization.diagnostic.empty(),
                            "independent machine device did not remain usable");

  InProcessRuntimeFailurePlan ownershipFailure;
  ownershipFailure.quarantineLeaseReleaseFailures = 1;
  auto ownershipProvider =
      take(test,
           createInProcessRuntimeProvider(
               {{implementations, std::nullopt, std::move(ownershipFailure)}}));
  const auto ownershipDevices =
      take(test, ownershipProvider->enumerateDevices());
  const RuntimeLeaseHandle callerLease = take(
      test, ownershipProvider->acquireExclusiveLease(ownershipDevices.front()));
  const RuntimeLeaseFinalizationResult ownershipTransfer =
      ownershipProvider->finalizeExclusiveLease(
          callerLease, RuntimeLeaseFinalizationRequest::Quarantine);
  deployment::test::require(
      test,
      ownershipTransfer.state == RuntimeLeaseFinalState::Quarantined &&
          llvm::StringRef(ownershipTransfer.diagnostic)
              .contains("quarantined lease remains provider-owned"),
      "quarantine did not transfer the unresolved lease to the provider");

  auto identityAfterTransfer = ownershipProvider->readImplementationIdentity(
      callerLease, inProcessRuntimeEndpoint(RuntimeEndpointClass::Identity, 0));
  deployment::test::require(
      test, !identityAfterTransfer,
      "caller retained identity access after quarantine ownership transfer");
  llvm::consumeError(identityAfterTransfer.takeError());
  llvm::Error resetAfterTransfer =
      ownershipProvider->quiesceAndReset(callerLease);
  deployment::test::require(
      test, static_cast<bool>(resetAfterTransfer),
      "caller retained reset access after quarantine ownership transfer");
  llvm::consumeError(std::move(resetAfterTransfer));
  llvm::Error programmingAfterTransfer =
      ownershipProvider->writeConfigurationWord(
          callerLease,
          inProcessRuntimeEndpoint(RuntimeEndpointClass::Programming, 0),
          RuntimeConfigurationWord{0, 0, UINT8_C(0xf)});
  deployment::test::require(
      test, static_cast<bool>(programmingAfterTransfer),
      "caller retained programming access after quarantine ownership transfer");
  llvm::consumeError(std::move(programmingAfterTransfer));
  const RuntimeLeaseFinalizationResult repeatedFinalization =
      ownershipProvider->finalizeExclusiveLease(
          callerLease, RuntimeLeaseFinalizationRequest::Release);
  deployment::test::require(
      test,
      repeatedFinalization.state == RuntimeLeaseFinalState::Quarantined &&
          llvm::StringRef(repeatedFinalization.diagnostic)
              .contains("lease is stale or inactive"),
      "caller released a lease after quarantine ownership transfer");

  constexpr llvm::StringLiteral abandonedMachine =
      "quarantine-failclosed-abandoned";
  auto abandonedProvider = take(
      test,
      createInProcessRuntimeProvider(
          {{{independentIdentity}, std::nullopt, {}, abandonedMachine.str()}}));
  const auto abandonedDevices =
      take(test, abandonedProvider->enumerateDevices());
  (void)take(
      test, abandonedProvider->acquireExclusiveLease(abandonedDevices.front()));
  abandonedProvider.reset();

  auto abandonedReplacement = take(
      test,
      createInProcessRuntimeProvider(
          {{{independentIdentity}, std::nullopt, {}, abandonedMachine.str()}}));
  const auto abandonedAvailable =
      take(test, abandonedReplacement->enumerateDevices());
  deployment::test::require(
      test,
      abandonedAvailable.empty() && abandonedReplacement->isQuarantined(0),
      "provider teardown left an unowned live lease outside quarantine");
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

void rejectsDirectSystemConfigurationWithoutSystemProvider() {
  const llvm::StringRef test = __func__;
  deployment::test::TemporaryTree tree(test);
  ArtifactStore artifacts(tree.path("artifacts"));
  BlobStore blobs(tree.path("blobs"));
  const deployment::FinalizedDeployment finalized =
      deployment::test::buildDirectSystemConfigurationDeployment(
          test, artifacts, blobs, tree);
  const auto imported =
      take(test, deployment::importDeployment(finalized.reference(), artifacts,
                                              blobs));
  deployment::test::require(
      test, imported.reference() == finalized.reference(),
      "direct System Deployment changed during strict import");

  std::uint64_t directImageCount = 0;
  std::uint64_t localImageCount = 0;
  for (const ArtifactRootReference &reference :
       imported.deployment().configurationImages()) {
    const auto image = take(test, deployment::importHardwareConfigurationImage(
                                      reference, artifacts));
    const auto abi =
        take(test, hardware::importConfigurationABI(
                       image.image().configurationAbi(), artifacts));
    const hardware::ProgrammingUnit *unit =
        abi.abi().findProgrammingUnit(image.image().programmingUnitId());
    deployment::test::require(test, unit != nullptr,
                              "configuration image lost its ABI unit");
    const hardware::ProgrammingUnitOccurrenceScope scope =
        hardware::deriveProgrammingUnitOccurrenceScope(*unit);
    if (scope.includesDirectSystemResources && scope.spatialCores.empty())
      ++directImageCount;
    else if (!scope.includesDirectSystemResources &&
             scope.spatialCores.size() == 1)
      ++localImageCount;
    else
      deployment::test::fail(test,
                             "configuration image has a mixed owner scope");
  }
  deployment::test::require(
      test,
      directImageCount != 0 &&
          localImageCount == finalized.deployment().hardwareBindings().size(),
      "Deployment did not retain global and subject-local images");

  const auto implementations =
      implementationIdentities(test, finalized, artifacts, blobs);
  auto provider = take(test, createInProcessRuntimeProvider(
                                 {{implementations, std::nullopt, {}}}));
  const ObservedLoadError error = expectLoadError(
      test, loadDeployment(finalized, {provider, 0}, artifacts, blobs));
  deployment::test::require(
      test,
      error.kind == RuntimeLoadFailureKind::ProviderMismatch &&
          llvm::StringRef(error.diagnostic)
              .contains("direct System configuration binding") &&
          provider->statistics().enumerationCount == 0,
      "missing System provider did not fail before device enumeration");
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
                   std::nullopt, std::nullopt, 0, 0}}}));

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
       {even},
       ::mapping::SystemBindingRelationKind::PresburgerPartition,
       {}},
      {first,
       {loom::mapping::mappingArtifactSchema.identity.str(),
        loom::mapping::mappingArtifactSchema.version, secondMapping},
       {secondCore, secondMapping},
       {odd},
       ::mapping::SystemBindingRelationKind::PresburgerPartition,
       {}}};
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
    rejectsStaleTrustedAttestationUnderLease();
  else if (scenario == "attestation-reset")
    rejectsTrustedAttestationChangeAcrossInitialReset();
  else if (scenario == "lease-identity")
    rejectsIdentityChangeAtExclusiveLeaseBoundary();
  else if (scenario == "reset-identity")
    rejectsIdentityChangeAcrossInitialReset();
  else if (scenario == "lease-contract")
    enforcesLeaseIdentityAndTerminalDispositionContract();
  else if (scenario == "release-fallback")
    quarantinesAtomicallyWhenOrdinaryReleaseFails();
  else if (scenario == "quarantine-failclosed")
    quarantinesWhenLeaseReleaseCannotComplete();
  else if (scenario == "endpoint-alias")
    rejectsProgrammingEndpointAliasedAcrossSpatialCores();
  else if (scenario == "descriptor-alias")
    rejectsUnregisteredDescriptorAliasBeforeEnumeration();
  else if (scenario == "non-portable")
    rejectsNonPortableRuntimeAbiBeforeEnumeration();
  else if (scenario == "direct-system")
    rejectsDirectSystemConfigurationWithoutSystemProvider();
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
