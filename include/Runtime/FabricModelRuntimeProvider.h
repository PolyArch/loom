#ifndef LOOM_RUNTIME_FABRICMODELRUNTIMEPROVIDER_H
#define LOOM_RUNTIME_FABRICMODELRUNTIMEPROVIDER_H

#include "Runtime/DeploymentLoader.h"

#include <memory>
#include <optional>
#include <vector>

namespace loom::runtime {

struct FabricModelRuntimeDeviceConfig final {
  std::vector<ArtifactIdentity> hardwareImplementations;
};

struct FabricModelRuntimeStatistics final {
  std::uint64_t enumerationCount = 0;
  std::uint64_t leaseAcquisitionCount = 0;
  std::uint64_t implementationIdentityReadCount = 0;
  std::uint64_t resetCount = 0;
  std::uint64_t configurationWriteCount = 0;
  std::uint64_t configurationCommitCount = 0;
  std::uint64_t configurationReadCount = 0;
  std::uint64_t staticMemoryInstallCount = 0;
  std::uint64_t executableRegistrationCount = 0;
  std::uint64_t activationCount = 0;
  std::uint64_t activationPreparationCount = 0;
  std::uint64_t activationReplacementCount = 0;
  std::uint64_t activationDiscardCount = 0;
  std::uint64_t leaseReleaseCount = 0;
  std::uint64_t quarantineCount = 0;
};

/// Deterministic operational owner for a finalized FabricModel. It models the
/// portable configuration and activation boundary only; computation remains
/// owned by the selected Simulation provider.
class FabricModelRuntimeProvider final : public RuntimeProviderInstance {
public:
  struct State;

  ~FabricModelRuntimeProvider() override;

  const RuntimeProviderDescriptor &descriptor() const override;
  llvm::Expected<std::vector<RuntimeDeviceHandle>> enumerateDevices() override;
  llvm::Expected<ArtifactIdentity> readImplementationIdentity(
      const RuntimeLeaseHandle &lease,
      const RuntimeProviderEndpointRef &endpoint) override;
  llvm::Expected<BlobDigest>
  readTrustedAttestation(const RuntimeLeaseHandle &lease) override;
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
  RuntimeLeaseFinalizationResult
  finalizeExclusiveLease(const RuntimeLeaseHandle &lease,
                         RuntimeLeaseFinalizationRequest request) override;

  FabricModelRuntimeStatistics statistics() const;
  std::optional<ArtifactRootReference>
  activeDeployment(std::uint64_t deviceOrdinal) const;
  std::size_t preparedActivationCount(std::uint64_t deviceOrdinal) const;
  bool isQuarantined(std::uint64_t deviceOrdinal) const;

private:
  explicit FabricModelRuntimeProvider(std::unique_ptr<State> state);

  std::unique_ptr<State> state_;

  friend llvm::Expected<std::shared_ptr<FabricModelRuntimeProvider>>
  createFabricModelRuntimeProvider(
      std::vector<FabricModelRuntimeDeviceConfig> devices);
};

llvm::Expected<std::shared_ptr<FabricModelRuntimeProvider>>
createFabricModelRuntimeProvider(
    std::vector<FabricModelRuntimeDeviceConfig> devices);

} // namespace loom::runtime

#endif // LOOM_RUNTIME_FABRICMODELRUNTIMEPROVIDER_H
