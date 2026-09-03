#ifndef LOOM_RUNTIME_INPROCESSPLATFORM_H
#define LOOM_RUNTIME_INPROCESSPLATFORM_H

#include "Runtime/DeploymentLoader.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace loom::runtime {

struct InProcessRuntimeReadbackCorruption final {
  std::uint64_t readOrdinal = 0;
  std::uint32_t xorMask = 0;
};

enum class InProcessRuntimeVerificationMismatchBoundary : std::uint8_t {
  ExclusiveLease,
  InitialReset,
  RecoveryReset,
};

/// Deterministic controls for the test-oriented in-process platform. They are
/// local provider behavior and never enter a RuntimePlatformBinding.
struct InProcessRuntimeFailurePlan final {
  std::optional<std::uint64_t> configurationWriteOrdinal;
  std::optional<InProcessRuntimeReadbackCorruption> readbackCorruption;
  std::optional<InProcessRuntimeVerificationMismatchBoundary>
      verificationMismatchBoundary;
  std::optional<std::uint64_t> activationPreparationOrdinal;
  std::uint64_t activationReplacementFailures = 0;
  std::uint64_t activationDiscardFailures = 0;
  std::uint64_t leaseReleaseFailures = 0;
  std::uint64_t quarantineLeaseReleaseFailures = 0;
};

struct InProcessRuntimeDeviceConfig final {
  InProcessRuntimeDeviceConfig() = default;
  InProcessRuntimeDeviceConfig(
      std::vector<ArtifactIdentity> hardwareImplementations,
      std::optional<BlobDigest> trustedAttestation,
      InProcessRuntimeFailurePlan failures,
      std::optional<std::string> machineDeviceIdentity = std::nullopt)
      : hardwareImplementations(std::move(hardwareImplementations)),
        trustedAttestation(std::move(trustedAttestation)),
        failures(std::move(failures)),
        machineDeviceIdentity(std::move(machineDeviceIdentity)) {}

  std::vector<ArtifactIdentity> hardwareImplementations;
  std::optional<BlobDigest> trustedAttestation;
  InProcessRuntimeFailurePlan failures;
  /// Equal non-empty values in separate provider instances denote the same
  /// simulated machine-local device. An absent value creates a distinct device.
  std::optional<std::string> machineDeviceIdentity;
};

struct InProcessRuntimeStatistics final {
  std::uint64_t enumerationCount = 0;
  std::uint64_t identityReadCount = 0;
  std::uint64_t attestationReadCount = 0;
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
  std::uint64_t preparedConfigurationWordCount = 0;
  std::uint64_t preparedLogicalMemoryCopyCount = 0;
  std::uint64_t copiedLogicalMemoryByteCount = 0;
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
  llvm::Expected<RuntimeLeaseHandle>
  acquireExclusiveLease(const RuntimeDeviceHandle &device) override;
  llvm::Expected<ArtifactIdentity> readImplementationIdentity(
      const RuntimeLeaseHandle &lease,
      const RuntimeProviderEndpointRef &endpoint) override;
  llvm::Expected<BlobDigest>
  readTrustedAttestation(const RuntimeLeaseHandle &lease) override;
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
  llvm::Expected<RuntimePreparedActivationHandle> prepareResourceTimeTransition(
      const RuntimeLeaseHandle &lease,
      const RuntimeExecutableRegistrationView &registration,
      const RuntimeActivationView &activation,
      llvm::ArrayRef<RuntimeConfigurationDeltaTarget> configuration,
      llvm::ArrayRef<pnr::ResourceTimeLogicalMemoryCopyPlan> logicalMemories)
      override;
  llvm::Error replaceActivationAtomically(
      const RuntimeLeaseHandle &lease,
      const RuntimePreparedActivationHandle &prepared) override;
  llvm::Error discardPreparedActivation(
      const RuntimeLeaseHandle &lease,
      const RuntimePreparedActivationHandle &prepared) override;
  RuntimeLeaseFinalizationResult
  finalizeExclusiveLease(const RuntimeLeaseHandle &lease,
                         RuntimeLeaseFinalizationRequest request) override;

  InProcessRuntimeStatistics statistics() const;
  bool isQuarantined(std::uint64_t deviceOrdinal) const;
  std::optional<ArtifactRootReference>
  activeDeployment(std::uint64_t deviceOrdinal) const;
  std::size_t preparedActivationCount(std::uint64_t deviceOrdinal) const;
  /// Test-oriented live target access. The target identity is the same
  /// canonical projection consumed by prepared transition execution.
  llvm::Error setLiveMemoryTarget(std::uint64_t deviceOrdinal,
                                  const pnr::ResourceTimeMemoryTarget &target,
                                  llvm::ArrayRef<std::uint8_t> bytes);
  llvm::Expected<std::vector<std::uint8_t>>
  readLiveMemoryTarget(std::uint64_t deviceOrdinal,
                       const pnr::ResourceTimeMemoryTarget &target) const;

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
