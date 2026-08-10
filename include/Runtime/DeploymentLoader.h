#ifndef LOOM_RUNTIME_DEPLOYMENTLOADER_H
#define LOOM_RUNTIME_DEPLOYMENTLOADER_H

#include "Common/Artifact.h"
#include "Common/BlobDigest.h"
#include "Deployment/Deployment.h"
#include "Fabric/Identity/FabricRefs.h"
#include "Frontend/Executable/InstructionCoreBinary.h"
#include "Mapping/Artifact/SystemMappingArtifact.h"
#include "Mapping/Artifact/SystemMappingExecutionProjection.h"
#include "Runtime/RuntimePlatformBinding.h"
#include "Runtime/RuntimeProvider.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <system_error>
#include <variant>
#include <vector>

namespace loom {
class ArtifactStore;
class BlobStore;
} // namespace loom

namespace loom::runtime {

enum class RuntimeLoadFailureKind : std::uint32_t {
  InvalidDeployment = 0,
  ProviderMismatch = 1,
  Enumeration = 2,
  IdentityVerification = 3,
  Lease = 4,
  Reset = 5,
  Programming = 6,
  StaticMemory = 7,
  Registration = 8,
  Activation = 9,
};

class RuntimeLoadError final : public llvm::ErrorInfo<RuntimeLoadError> {
public:
  static char ID;

  RuntimeLoadError(RuntimeLoadFailureKind kind, std::string diagnostic,
                   bool deviceQuarantined = false)
      : kind_(kind), diagnostic_(std::move(diagnostic)),
        deviceQuarantined_(deviceQuarantined) {}

  RuntimeLoadFailureKind kind() const { return kind_; }
  llvm::StringRef diagnostic() const { return diagnostic_; }
  bool deviceQuarantined() const { return deviceQuarantined_; }

  void log(llvm::raw_ostream &output) const override;
  std::error_code convertToErrorCode() const override;

private:
  RuntimeLoadFailureKind kind_;
  std::string diagnostic_;
  bool deviceQuarantined_;
};

/// Provider-owned transient token. Its bytes are meaningful only to the
/// selected provider instance and never enter an Artifact identity.
struct RuntimeDeviceHandle final {
  std::vector<std::uint8_t> opaque;

  friend bool operator==(const RuntimeDeviceHandle &lhs,
                         const RuntimeDeviceHandle &rhs) {
    return lhs.opaque == rhs.opaque;
  }
};

/// Provider-owned transient exclusive lease token.
struct RuntimeLeaseHandle final {
  std::vector<std::uint8_t> opaque;

  friend bool operator==(const RuntimeLeaseHandle &lhs,
                         const RuntimeLeaseHandle &rhs) {
    return lhs.opaque == rhs.opaque;
  }
};

struct RuntimeConfigurationWord final {
  std::uint32_t address = 0;
  std::uint32_t value = 0;
  std::uint8_t byteStrobe = 0;
};

/// One exact ConfigurationABI unit bound to one provider endpoint. The
/// transport fields are the removable shared AXI4-Lite projection.
struct RuntimeConfigurationTarget final {
  ArtifactRootReference image;
  fabric::SpatialCoreOccurrenceRef spatialCore;
  RuntimeProviderEndpointRef endpoint;
  std::uint64_t payloadBitCount = 0;
  std::uint32_t commitAddress = 0;
  std::uint32_t statusAddress = 0;
  std::vector<RuntimeConfigurationWord> words;
  std::vector<std::uint8_t> transportLayoutKey;
  std::vector<std::uint8_t> activationEventKey;
};

struct RuntimeSpatialMemoryTarget final {
  mapping::SpatialExecutionContextKey context;
  mapping::SpatialMemoryIntervalView interval;
  mapping::SpatialMemoryLocalRegionView region;
};

using RuntimeStaticMemoryTarget =
    std::variant<RuntimeSpatialMemoryTarget,
                 mapping::SystemMemoryRegionElementView>;

struct RuntimeStaticMemoryInstall final {
  ArtifactRootReference canonicalDataflow;
  dataflow::RootedGraphLaunchRef rootedGraphLaunch;
  dataflow::LogicalMemoryRootRef logicalMemoryRoot;
  ArtifactRootReference layoutBinding;
  std::uint64_t alignmentBytes = 0;
  frontend::StaticMemoryPermissions permissions =
      frontend::StaticMemoryPermissions::ReadOnly;
  std::vector<std::uint8_t> bytes;
  std::vector<RuntimeStaticMemoryTarget> targets;
};

/// Synchronous view over the exact executable closure. A provider may copy the
/// material it needs, but cannot retain these references after the call.
struct RuntimeExecutableRegistrationView final {
  const deployment::HostProgramLeaf &hostProgram;
  llvm::ArrayRef<std::uint8_t> hostProgramBytes;
  llvm::ArrayRef<FinalizedInstructionCoreBinary> instructionCoreBinaries;
  llvm::ArrayRef<std::vector<std::uint8_t>> instructionCoreProgramBytes;
  const deployment::InlineRuntimeImage &threadDispatchImage;
};

/// Synchronous view over the exact runtime binding and derived activation
/// images. A provider must copy any state needed after activate returns.
struct RuntimeActivationView final {
  ArtifactRootReference deployment;
  const RuntimePlatformBinding &runtimePlatformBinding;
  const deployment::InlineRuntimeImage &threadDispatchImage;
  const std::optional<deployment::InlineRuntimeImage> &spatialLaunchImage;
  const deployment::InlineRuntimeImage &admissionImage;
};

/// Transient operational provider. The immutable descriptor remains the only
/// provider schema owner; this object owns machine-local enumeration and I/O.
class RuntimeProviderInstance {
public:
  virtual ~RuntimeProviderInstance() = default;

  virtual const RuntimeProviderDescriptor &descriptor() const = 0;
  virtual llvm::Expected<std::vector<RuntimeDeviceHandle>>
  enumerateDevices() = 0;
  virtual llvm::Expected<ArtifactIdentity>
  readImplementationIdentity(const RuntimeDeviceHandle &device,
                             const RuntimeProviderEndpointRef &endpoint) = 0;
  virtual llvm::Expected<BlobDigest>
  readTrustedAttestation(const RuntimeDeviceHandle &device) = 0;
  virtual llvm::Expected<RuntimeLeaseHandle>
  acquireExclusiveLease(const RuntimeDeviceHandle &device) = 0;
  virtual llvm::Error quiesceAndReset(const RuntimeLeaseHandle &lease) = 0;
  virtual llvm::Error
  writeConfigurationWord(const RuntimeLeaseHandle &lease,
                         const RuntimeProviderEndpointRef &endpoint,
                         const RuntimeConfigurationWord &word) = 0;
  virtual llvm::Error
  commitConfiguration(const RuntimeLeaseHandle &lease,
                      const RuntimeProviderEndpointRef &endpoint,
                      std::uint32_t commitAddress) = 0;
  virtual llvm::Expected<std::uint32_t>
  readConfigurationWord(const RuntimeLeaseHandle &lease,
                        const RuntimeProviderEndpointRef &endpoint,
                        std::uint32_t address) = 0;
  virtual llvm::Error programConfigurationMulticast(
      const RuntimeLeaseHandle &lease,
      llvm::ArrayRef<RuntimeConfigurationTarget> targets);
  virtual llvm::Error installStaticMemory(
      const RuntimeLeaseHandle &lease,
      const RuntimeStaticMemoryInstall &install,
      llvm::ArrayRef<RuntimeInterfaceBinding> memoryBindings) = 0;
  virtual llvm::Error registerExecutables(
      const RuntimeLeaseHandle &lease,
      const RuntimeExecutableRegistrationView &registration) = 0;
  virtual llvm::Error activate(const RuntimeLeaseHandle &lease,
                               const RuntimeActivationView &activation) = 0;
  virtual llvm::Error
  releaseExclusiveLease(const RuntimeLeaseHandle &lease) = 0;
  virtual void quarantineDevice(const RuntimeDeviceHandle &device) = 0;
};

/// Local invocation binding. deviceOrdinal selects one result from this exact
/// instance's enumeration and has no persistent identity.
struct RuntimeProviderSelection final {
  std::shared_ptr<RuntimeProviderInstance> provider;
  std::uint64_t deviceOrdinal = 0;
};

namespace detail {
struct LoadedDeploymentState;

llvm::Expected<std::vector<std::uint8_t>> configurationActivationEventKey(
    const ArtifactIdentity &dataflowIdentity,
    llvm::ArrayRef<mapping::SystemSpatialContextDomain> spatialDomains,
    fabric::SpatialCoreOccurrenceRef spatialCore);
} // namespace detail

class LoadedDeployment final {
public:
  LoadedDeployment(LoadedDeployment &&) noexcept;
  LoadedDeployment &operator=(LoadedDeployment &&) noexcept;
  ~LoadedDeployment();

  LoadedDeployment(const LoadedDeployment &) = delete;
  LoadedDeployment &operator=(const LoadedDeployment &) = delete;

  const deployment::FinalizedDeployment &deployment() const;
  const RuntimeDeviceHandle &device() const;

private:
  explicit LoadedDeployment(
      std::unique_ptr<detail::LoadedDeploymentState> state);

  std::unique_ptr<detail::LoadedDeploymentState> state_;

  friend llvm::Expected<LoadedDeployment>
  loadDeployment(deployment::FinalizedDeployment, RuntimeProviderSelection,
                 const ArtifactStore &, const BlobStore &);
};

llvm::Expected<LoadedDeployment>
loadDeployment(deployment::FinalizedDeployment deployment,
               RuntimeProviderSelection selection,
               const ArtifactStore &artifacts, const BlobStore &blobs);

} // namespace loom::runtime

#endif // LOOM_RUNTIME_DEPLOYMENTLOADER_H
