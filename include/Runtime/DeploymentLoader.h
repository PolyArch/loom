#ifndef LOOM_RUNTIME_DEPLOYMENTLOADER_H
#define LOOM_RUNTIME_DEPLOYMENTLOADER_H

#include "Common/Artifact.h"
#include "Common/BlobDigest.h"
#include "Deployment/Deployment.h"
#include "Fabric/Identity/FabricRefs.h"
#include "Frontend/Executable/InstructionCoreBinary.h"
#include "Mapping/Artifact/SystemMappingArtifact.h"
#include "Mapping/Artifact/SystemMappingExecutionProjection.h"
#include "PnR/System/SystemMappingMigration.h"
#include "Runtime/RuntimePlatformBinding.h"
#include "Runtime/RuntimeProvider.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <system_error>
#include <utility>
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

enum class RuntimeLoadTerminalDisposition : std::uint8_t {
  NoLeaseAcquired,
  LeaseReleased,
  DeviceQuarantined,
};

class RuntimeLoadError final : public llvm::ErrorInfo<RuntimeLoadError> {
public:
  static char ID;

  RuntimeLoadError(RuntimeLoadFailureKind kind, std::string diagnostic,
                   RuntimeLoadTerminalDisposition terminalDisposition =
                       RuntimeLoadTerminalDisposition::NoLeaseAcquired)
      : kind_(kind), diagnostic_(std::move(diagnostic)),
        terminalDisposition_(terminalDisposition) {}

  RuntimeLoadFailureKind kind() const { return kind_; }
  llvm::StringRef diagnostic() const { return diagnostic_; }
  RuntimeLoadTerminalDisposition terminalDisposition() const {
    return terminalDisposition_;
  }
  bool deviceQuarantined() const {
    return terminalDisposition_ ==
           RuntimeLoadTerminalDisposition::DeviceQuarantined;
  }

  void log(llvm::raw_ostream &output) const override;
  std::error_code convertToErrorCode() const override;

private:
  RuntimeLoadFailureKind kind_;
  std::string diagnostic_;
  RuntimeLoadTerminalDisposition terminalDisposition_;
};

enum class RuntimeActivationReplacementErrorReason : std::uint8_t {
  InvalidDeployment,
  TransitionMismatch,
  ProviderMismatch,
  ProviderCapabilityUnavailable,
  PreparationFailed,
  ActivationFailed,
};

class RuntimeActivationReplacementError final
    : public llvm::ErrorInfo<RuntimeActivationReplacementError> {
public:
  static char ID;

  RuntimeActivationReplacementError(
      RuntimeActivationReplacementErrorReason reason, std::string diagnostic)
      : reason_(reason), diagnostic_(std::move(diagnostic)) {}

  RuntimeActivationReplacementErrorReason reason() const { return reason_; }
  llvm::StringRef diagnostic() const { return diagnostic_; }
  void log(llvm::raw_ostream &output) const override;
  std::error_code convertToErrorCode() const override;

private:
  RuntimeActivationReplacementErrorReason reason_;
  std::string diagnostic_;
};

/// Provider-owned transient enumeration token. Its bytes are meaningful only
/// to the selected provider instance and never enter an Artifact identity.
/// Runtime uses it to acquire a lease and retain the invocation-local
/// selection; identity observations are bound to the resulting live lease.
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

enum class RuntimeLeaseFinalizationRequest : std::uint8_t {
  Release,
  Quarantine,
};

enum class RuntimeLeaseFinalState : std::uint8_t {
  Released,
  Quarantined,
};

struct RuntimeLeaseFinalizationResult final {
  RuntimeLeaseFinalState state = RuntimeLeaseFinalState::Quarantined;
  std::string diagnostic;
};

/// Provider-owned token for one executable and activation image prepared
/// before resource-time execution begins. It has no persistent identity.
struct RuntimePreparedActivationHandle final {
  std::vector<std::uint8_t> opaque;
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
  llvm::ArrayRef<FinalizedRuntimePlatformBinding> runtimePlatformBindings;
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
  /// Atomically binds the returned lease to this exact enumerated device and
  /// excludes replacement or rebinding until the lease is released. An error
  /// acquires no lease.
  virtual llvm::Expected<RuntimeLeaseHandle>
  acquireExclusiveLease(const RuntimeDeviceHandle &device) = 0;
  /// Reads identity from the exact device bound to this live lease. Providers
  /// must reject stale or inactive leases.
  virtual llvm::Expected<ArtifactIdentity>
  readImplementationIdentity(const RuntimeLeaseHandle &lease,
                             const RuntimeProviderEndpointRef &endpoint) = 0;
  virtual llvm::Expected<BlobDigest>
  readTrustedAttestation(const RuntimeLeaseHandle &lease) = 0;
  /// Restores the provider's declared clean state and invalidates every
  /// provider-owned prepared activation handle under this lease.
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
  /// Copies all state needed for a later activation replacement. This call is
  /// part of invocation setup and cannot retain either synchronous view. An
  /// error retains no prepared state from this call.
  virtual llvm::Expected<RuntimePreparedActivationHandle>
  prepareActivation(const RuntimeLeaseHandle &lease,
                    const RuntimeExecutableRegistrationView &registration,
                    const RuntimeActivationView &activation);
  /// Atomically switches to one prepared activation without loading artifacts,
  /// executable bytes, or runtime images. On error, the previous Deployment
  /// remains active. The prepared handle remains reusable after either result.
  virtual llvm::Error
  replaceActivationAtomically(const RuntimeLeaseHandle &lease,
                              const RuntimePreparedActivationHandle &prepared);
  /// Invalidates one reusable prepared copy without changing the currently
  /// active Deployment. Quiesce/reset remains the bounded cleanup fallback.
  virtual llvm::Error
  discardPreparedActivation(const RuntimeLeaseHandle &lease,
                            const RuntimePreparedActivationHandle &prepared);
  /// Atomically establishes one terminal state for this live lease. Release
  /// failure falls back to a process-persistent provider quarantine that owns
  /// any unresolved lease state. The provider must return only after later
  /// acquisition through any instance of this descriptor is excluded.
  virtual RuntimeLeaseFinalizationResult
  finalizeExclusiveLease(const RuntimeLeaseHandle &lease,
                         RuntimeLeaseFinalizationRequest request) = 0;
};

/// Local invocation binding. deviceOrdinal selects one result from this exact
/// instance's enumeration and has no persistent identity.
struct RuntimeProviderSelection final {
  std::shared_ptr<RuntimeProviderInstance> provider;
  std::uint64_t deviceOrdinal = 0;
};

namespace detail {
struct LoadedDeploymentState;
struct ResourceTimeActivationToken;

llvm::Expected<std::vector<std::uint8_t>> configurationActivationEventKey(
    const ArtifactIdentity &dataflowIdentity,
    llvm::ArrayRef<mapping::SystemSpatialContextDomain> spatialDomains,
    fabric::SpatialCoreOccurrenceRef spatialCore);
} // namespace detail

class ResourceTimeTransitionSelectionSession;

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

  llvm::Expected<std::shared_ptr<const detail::ResourceTimeActivationToken>>
  prepareResourceTimeActivations(const pnr::ResourceTimeTransitionGraph &graph,
                                 const ArtifactStore &artifacts,
                                 const BlobStore &blobs);
  llvm::Error activatePreparedTransition(
      const pnr::ResourceTimeTransition &transition,
      const std::shared_ptr<const detail::ResourceTimeActivationToken> &token);

  std::unique_ptr<detail::LoadedDeploymentState> state_;

  friend class ResourceTimeTransitionSelectionSession;

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
