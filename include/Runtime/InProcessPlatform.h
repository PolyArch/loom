#ifndef LOOM_RUNTIME_INPROCESSPLATFORM_H
#define LOOM_RUNTIME_INPROCESSPLATFORM_H

#include "Runtime/DeploymentLoader.h"

#include <cstddef>
#include <memory>
#include <optional>
#include <vector>

namespace loom::runtime {

struct InProcessRuntimeReadbackCorruption final {
  std::uint64_t readOrdinal = 0;
  std::uint32_t xorMask = 0;
};

/// Deterministic controls for the test-oriented in-process platform. They are
/// local provider behavior and never enter a RuntimePlatformBinding.
struct InProcessRuntimeFailurePlan final {
  std::optional<std::uint64_t> configurationWriteOrdinal;
  std::optional<InProcessRuntimeReadbackCorruption> readbackCorruption;
  bool identityMismatchAfterRecoveryReset = false;
  std::optional<std::uint64_t> activationPreparationOrdinal;
  std::uint64_t activationReplacementFailures = 0;
  std::uint64_t activationDiscardFailures = 0;
};

struct InProcessRuntimeDeviceConfig final {
  std::vector<ArtifactIdentity> hardwareImplementations;
  std::optional<BlobDigest> trustedAttestation;
  InProcessRuntimeFailurePlan failures;
};

struct InProcessRuntimeStatistics final {
  std::uint64_t enumerationCount = 0;
  std::uint64_t identityReadCount = 0;
  std::uint64_t leaseAcquisitionCount = 0;
  std::uint64_t resetCount = 0;
  std::uint64_t configurationWriteCount = 0;
  std::uint64_t configurationCommitCount = 0;
  std::uint64_t configurationReadCount = 0;
  std::uint64_t multicastTransactionCount = 0;
  std::uint64_t staticMemoryInstallCount = 0;
  std::uint64_t executableRegistrationCount = 0;
  std::uint64_t activationCount = 0;
  std::uint64_t activationPreparationCount = 0;
  std::uint64_t activationDiscardCount = 0;
  std::uint64_t activationReplacementCount = 0;
  std::uint64_t leaseReleaseCount = 0;
  std::uint64_t quarantineCount = 0;
};

class InProcessRuntimeProvider final : public RuntimeProviderInstance {
public:
  ~InProcessRuntimeProvider() override;

  const RuntimeProviderDescriptor &descriptor() const override;
  llvm::Expected<std::vector<RuntimeDeviceHandle>> enumerateDevices() override;
  llvm::Expected<ArtifactIdentity> readImplementationIdentity(
      const RuntimeDeviceHandle &device,
      const RuntimeProviderEndpointRef &endpoint) override;
  llvm::Expected<BlobDigest>
  readTrustedAttestation(const RuntimeDeviceHandle &device) override;
  llvm::Expected<RuntimeLeaseHandle>
  acquireExclusiveLease(const RuntimeDeviceHandle &device) override;
  llvm::Error quiesceAndReset(const RuntimeLeaseHandle &lease) override;
  llvm::Error
  writeConfigurationWord(const RuntimeLeaseHandle &lease,
                         const RuntimeProviderEndpointRef &endpoint,
                         const RuntimeConfigurationWord &word) override;
  llvm::Error commitConfiguration(const RuntimeLeaseHandle &lease,
                                  const RuntimeProviderEndpointRef &endpoint,
                                  std::uint32_t commitAddress) override;
  llvm::Expected<std::uint32_t>
  readConfigurationWord(const RuntimeLeaseHandle &lease,
                        const RuntimeProviderEndpointRef &endpoint,
                        std::uint32_t address) override;
  llvm::Error programConfigurationMulticast(
      const RuntimeLeaseHandle &lease,
      llvm::ArrayRef<RuntimeConfigurationTarget> targets) override;
  llvm::Error installStaticMemory(
      const RuntimeLeaseHandle &lease,
      const RuntimeStaticMemoryInstall &install,
      llvm::ArrayRef<RuntimeInterfaceBinding> memoryBindings) override;
  llvm::Error registerExecutables(
      const RuntimeLeaseHandle &lease,
      const RuntimeExecutableRegistrationView &registration) override;
  llvm::Error activate(const RuntimeLeaseHandle &lease,
                       const RuntimeActivationView &activation) override;
  llvm::Expected<RuntimePreparedActivationHandle>
  prepareActivation(const RuntimeLeaseHandle &lease,
                    const RuntimeExecutableRegistrationView &registration,
                    const RuntimeActivationView &activation) override;
  llvm::Error replaceActivationAtomically(
      const RuntimeLeaseHandle &lease,
      const RuntimePreparedActivationHandle &prepared) override;
  llvm::Error discardPreparedActivation(
      const RuntimeLeaseHandle &lease,
      const RuntimePreparedActivationHandle &prepared) override;
  llvm::Error releaseExclusiveLease(const RuntimeLeaseHandle &lease) override;
  void quarantineDevice(const RuntimeDeviceHandle &device) override;

  InProcessRuntimeStatistics statistics() const;
  bool isQuarantined(std::uint64_t deviceOrdinal) const;
  std::optional<ArtifactRootReference>
  activeDeployment(std::uint64_t deviceOrdinal) const;
  std::size_t preparedActivationCount(std::uint64_t deviceOrdinal) const;

private:
  struct State;
  explicit InProcessRuntimeProvider(std::unique_ptr<State> state);

  std::unique_ptr<State> state_;

  friend llvm::Expected<std::shared_ptr<InProcessRuntimeProvider>>
  createInProcessRuntimeProvider(
      std::vector<InProcessRuntimeDeviceConfig> devices);
};

const RuntimeProviderDescriptor &inProcessRuntimeProviderDescriptor();

llvm::Expected<std::shared_ptr<InProcessRuntimeProvider>>
createInProcessRuntimeProvider(
    std::vector<InProcessRuntimeDeviceConfig> devices);

RuntimeProviderEndpointRef
inProcessRuntimeEndpoint(RuntimeEndpointClass endpointClass,
                         std::uint64_t endpointOrdinal);

} // namespace loom::runtime

#endif // LOOM_RUNTIME_INPROCESSPLATFORM_H
