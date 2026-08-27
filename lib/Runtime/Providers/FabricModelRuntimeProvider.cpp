#include "Runtime/FabricModelRuntimeProvider.h"

#include "Runtime/FabricModelPlatform.h"

#include "llvm/ADT/STLExtras.h"

#include <cstdint>
#include <limits>
#include <map>
#include <mutex>
#include <optional>
#include <system_error>
#include <utility>

namespace loom::runtime {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      "fabric_model_runtime_invalid: " + message);
}

llvm::Error unavailable(const llvm::Twine &message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::operation_not_supported),
      "fabric_model_runtime_unavailable: " + message);
}

void appendU64(std::vector<std::uint8_t> &bytes, std::uint64_t value) {
  for (unsigned shift = 56;; shift -= 8) {
    bytes.push_back(static_cast<std::uint8_t>(value >> shift));
    if (shift == 0)
      break;
  }
}

llvm::Expected<std::uint64_t> readU64(llvm::ArrayRef<std::uint8_t> bytes,
                                      std::size_t offset,
                                      const llvm::Twine &context) {
  if (offset > bytes.size() || bytes.size() - offset < sizeof(std::uint64_t))
    return invalid(context + " is truncated");
  std::uint64_t value = 0;
  for (std::size_t index = 0; index != sizeof(value); ++index)
    value = (value << 8) | bytes[offset + index];
  return value;
}

llvm::Expected<std::uint64_t> allocateProviderInstance() {
  static std::mutex mutex;
  static std::uint64_t next = 1;
  std::lock_guard<std::mutex> lock(mutex);
  if (next == 0)
    return invalid("provider instance identity is exhausted");
  return next++;
}

RuntimeDeviceHandle deviceHandle(std::uint64_t instance,
                                 std::uint64_t ordinal) {
  RuntimeDeviceHandle handle;
  appendU64(handle.opaque, instance);
  appendU64(handle.opaque, ordinal);
  return handle;
}

RuntimeLeaseHandle leaseHandle(std::uint64_t instance, std::uint64_t ordinal,
                               std::uint64_t generation) {
  RuntimeLeaseHandle handle;
  appendU64(handle.opaque, instance);
  appendU64(handle.opaque, ordinal);
  appendU64(handle.opaque, generation);
  return handle;
}

RuntimePreparedActivationHandle preparedHandle(std::uint64_t instance,
                                               std::uint64_t ordinal,
                                               std::uint64_t generation,
                                               std::uint64_t identity) {
  RuntimePreparedActivationHandle handle;
  appendU64(handle.opaque, instance);
  appendU64(handle.opaque, ordinal);
  appendU64(handle.opaque, generation);
  appendU64(handle.opaque, identity);
  return handle;
}

struct DeviceIdentity final {
  std::uint64_t instance = 0;
  std::uint64_t device = 0;
};

struct LeaseIdentity final {
  DeviceIdentity device;
  std::uint64_t generation = 0;
};

llvm::Expected<DeviceIdentity> parseDevice(const RuntimeDeviceHandle &handle) {
  if (handle.opaque.size() != 2 * sizeof(std::uint64_t))
    return invalid("device handle has the wrong size");
  auto instance = readU64(handle.opaque, 0, "device handle");
  if (!instance)
    return instance.takeError();
  auto device = readU64(handle.opaque, sizeof(std::uint64_t), "device handle");
  if (!device)
    return device.takeError();
  return DeviceIdentity{*instance, *device};
}

llvm::Expected<LeaseIdentity> parseLease(const RuntimeLeaseHandle &handle) {
  if (handle.opaque.size() != 3 * sizeof(std::uint64_t))
    return invalid("lease handle has the wrong size");
  auto instance = readU64(handle.opaque, 0, "lease handle");
  if (!instance)
    return instance.takeError();
  auto device = readU64(handle.opaque, sizeof(std::uint64_t), "lease handle");
  if (!device)
    return device.takeError();
  auto generation =
      readU64(handle.opaque, 2 * sizeof(std::uint64_t), "lease handle");
  if (!generation)
    return generation.takeError();
  return LeaseIdentity{{*instance, *device}, *generation};
}

struct PreparedIdentity final {
  LeaseIdentity lease;
  std::uint64_t identity = 0;
};

llvm::Expected<PreparedIdentity>
parsePrepared(const RuntimePreparedActivationHandle &handle) {
  if (handle.opaque.size() != 4 * sizeof(std::uint64_t))
    return invalid("prepared activation handle has the wrong size");
  auto instance = readU64(handle.opaque, 0, "prepared activation handle");
  if (!instance)
    return instance.takeError();
  auto device = readU64(handle.opaque, sizeof(std::uint64_t),
                        "prepared activation handle");
  if (!device)
    return device.takeError();
  auto generation = readU64(handle.opaque, 2 * sizeof(std::uint64_t),
                            "prepared activation handle");
  if (!generation)
    return generation.takeError();
  auto identity = readU64(handle.opaque, 3 * sizeof(std::uint64_t),
                          "prepared activation handle");
  if (!identity)
    return identity.takeError();
  return PreparedIdentity{{{*instance, *device}, *generation}, *identity};
}

std::vector<std::uint8_t>
endpointKey(const RuntimeProviderEndpointRef &endpoint) {
  std::vector<std::uint8_t> key;
  appendU64(key, endpoint.kind);
  appendU64(key, endpoint.payload.size());
  key.insert(key.end(), endpoint.payload.begin(), endpoint.payload.end());
  return key;
}

llvm::Error validateExecutableRegistration(
    const RuntimeExecutableRegistrationView &registration) {
  if (registration.hostProgramBytes.empty())
    return invalid("host program is empty");
  if (registration.instructionCoreBinaries.size() !=
      registration.instructionCoreProgramBytes.size())
    return invalid("InstructionCore binary and byte catalogs disagree");
  return llvm::Error::success();
}

} // namespace

struct FabricModelRuntimeProvider::State final {
  struct Device final {
    explicit Device(FabricModelRuntimeDeviceConfig config)
        : config(std::move(config)) {}

    FabricModelRuntimeDeviceConfig config;
    bool leased = false;
    bool quarantined = false;
    std::uint64_t leaseGeneration = 0;
    std::uint64_t nextPreparedIdentity = 1;
    bool executablesRegistered = false;
    std::map<std::vector<std::uint8_t>, std::map<std::uint32_t, std::uint32_t>>
        configurationShadow;
    std::map<std::vector<std::uint8_t>, std::map<std::uint32_t, std::uint32_t>>
        activeConfiguration;
    std::vector<RuntimeStaticMemoryInstall> staticMemory;
    std::optional<ArtifactRootReference> activeDeployment;
    std::map<std::uint64_t, ArtifactRootReference> prepared;
  };

  State(std::uint64_t providerInstance,
        std::vector<FabricModelRuntimeDeviceConfig> configurations)
      : providerInstance(providerInstance) {
    devices.reserve(configurations.size());
    for (auto &configuration : configurations)
      devices.emplace_back(std::move(configuration));
  }

  mutable std::mutex mutex;
  std::uint64_t providerInstance = 0;
  std::vector<Device> devices;
  FabricModelRuntimeStatistics statistics;
};

namespace {

llvm::Expected<FabricModelRuntimeProvider::State::Device *>
requireDevice(FabricModelRuntimeProvider::State &state,
              const RuntimeDeviceHandle &handle) {
  auto ordinal = parseDevice(handle);
  if (!ordinal)
    return ordinal.takeError();
  if (ordinal->instance != state.providerInstance)
    return invalid("device handle belongs to another provider instance");
  if (ordinal->device >= state.devices.size())
    return invalid("device handle names an absent device");
  return &state.devices[static_cast<std::size_t>(ordinal->device)];
}

llvm::Expected<
    std::pair<FabricModelRuntimeProvider::State::Device *, LeaseIdentity>>
requireLease(FabricModelRuntimeProvider::State &state,
             const RuntimeLeaseHandle &handle) {
  auto identity = parseLease(handle);
  if (!identity)
    return identity.takeError();
  if (identity->device.instance != state.providerInstance)
    return invalid("lease handle belongs to another provider instance");
  if (identity->device.device >= state.devices.size())
    return invalid("lease handle names an absent device");
  auto &device =
      state.devices[static_cast<std::size_t>(identity->device.device)];
  if (!device.leased || device.leaseGeneration != identity->generation)
    return invalid("lease handle is stale or inactive");
  return std::make_pair(&device, *identity);
}

} // namespace

FabricModelRuntimeProvider::FabricModelRuntimeProvider(
    std::unique_ptr<State> state)
    : state_(std::move(state)) {}

FabricModelRuntimeProvider::~FabricModelRuntimeProvider() = default;

const RuntimeProviderDescriptor &
FabricModelRuntimeProvider::descriptor() const {
  return fabricModelRuntimeProviderDescriptor();
}

llvm::Expected<std::vector<RuntimeDeviceHandle>>
FabricModelRuntimeProvider::enumerateDevices() {
  std::lock_guard<std::mutex> lock(state_->mutex);
  std::vector<RuntimeDeviceHandle> handles;
  handles.reserve(state_->devices.size());
  for (std::uint64_t ordinal = 0; ordinal != state_->devices.size(); ++ordinal)
    if (!state_->devices[static_cast<std::size_t>(ordinal)].quarantined)
      handles.push_back(deviceHandle(state_->providerInstance, ordinal));
  ++state_->statistics.enumerationCount;
  return handles;
}

llvm::Expected<ArtifactIdentity>
FabricModelRuntimeProvider::readImplementationIdentity(
    const RuntimeLeaseHandle &lease,
    const RuntimeProviderEndpointRef &endpoint) {
  if (llvm::Error error = validateRuntimeProviderEndpoint(
          descriptor(), endpoint, RuntimeEndpointClass::Identity,
          RuntimeEndpointFlow::ImplementationToRuntime))
    return std::move(error);
  std::lock_guard<std::mutex> lock(state_->mutex);
  auto selected = requireLease(*state_, lease);
  if (!selected)
    return selected.takeError();
  auto identity = ArtifactIdentity::fromBytes(endpoint.payload);
  if (!identity)
    return invalid("identity endpoint payload is not an ArtifactIdentity");
  if (!llvm::is_contained(selected->first->config.hardwareImplementations,
                          *identity))
    return invalid("device does not contain the requested implementation");
  ++state_->statistics.implementationIdentityReadCount;
  return *identity;
}

llvm::Expected<BlobDigest> FabricModelRuntimeProvider::readTrustedAttestation(
    const RuntimeLeaseHandle &lease) {
  std::lock_guard<std::mutex> lock(state_->mutex);
  auto selected = requireLease(*state_, lease);
  if (!selected)
    return selected.takeError();
  return unavailable("FabricModel has no trusted attestation");
}

llvm::Expected<RuntimeLeaseHandle>
FabricModelRuntimeProvider::acquireExclusiveLease(
    const RuntimeDeviceHandle &handle) {
  std::lock_guard<std::mutex> lock(state_->mutex);
  auto device = requireDevice(*state_, handle);
  if (!device)
    return device.takeError();
  if ((*device)->quarantined)
    return invalid("device is quarantined");
  if ((*device)->leased)
    return invalid("device is already leased");
  if ((*device)->leaseGeneration == std::numeric_limits<std::uint64_t>::max()) {
    (*device)->quarantined = true;
    ++state_->statistics.quarantineCount;
    return invalid("device lease generation is exhausted; device quarantined");
  }
  (*device)->leased = true;
  ++(*device)->leaseGeneration;
  ++state_->statistics.leaseAcquisitionCount;
  auto identity = parseDevice(handle);
  if (!identity)
    return identity.takeError();
  return leaseHandle(identity->instance, identity->device,
                     (*device)->leaseGeneration);
}

llvm::Error
FabricModelRuntimeProvider::quiesceAndReset(const RuntimeLeaseHandle &lease) {
  std::lock_guard<std::mutex> lock(state_->mutex);
  auto selected = requireLease(*state_, lease);
  if (!selected)
    return selected.takeError();
  selected->first->executablesRegistered = false;
  selected->first->configurationShadow.clear();
  selected->first->activeConfiguration.clear();
  selected->first->staticMemory.clear();
  selected->first->activeDeployment.reset();
  selected->first->prepared.clear();
  ++state_->statistics.resetCount;
  return llvm::Error::success();
}

llvm::Error FabricModelRuntimeProvider::writeConfigurationWord(
    const RuntimeLeaseHandle &lease, const RuntimeProviderEndpointRef &endpoint,
    const RuntimeConfigurationWord &word) {
  if (llvm::Error error = validateRuntimeProviderEndpoint(
          descriptor(), endpoint, RuntimeEndpointClass::Programming,
          RuntimeEndpointFlow::RuntimeToImplementation))
    return error;
  std::lock_guard<std::mutex> lock(state_->mutex);
  auto selected = requireLease(*state_, lease);
  if (!selected)
    return selected.takeError();
  std::uint32_t &stored =
      selected->first->configurationShadow[endpointKey(endpoint)][word.address];
  for (unsigned byte = 0; byte != 4; ++byte)
    if ((word.byteStrobe & (1U << byte)) != 0) {
      const std::uint32_t mask = std::uint32_t{0xff} << (byte * 8);
      stored = (stored & ~mask) | (word.value & mask);
    }
  ++state_->statistics.configurationWriteCount;
  return llvm::Error::success();
}

llvm::Error FabricModelRuntimeProvider::commitConfiguration(
    const RuntimeLeaseHandle &lease, const RuntimeProviderEndpointRef &endpoint,
    std::uint32_t) {
  if (llvm::Error error = validateRuntimeProviderEndpoint(
          descriptor(), endpoint, RuntimeEndpointClass::Programming,
          RuntimeEndpointFlow::RuntimeToImplementation))
    return error;
  std::lock_guard<std::mutex> lock(state_->mutex);
  auto selected = requireLease(*state_, lease);
  if (!selected)
    return selected.takeError();
  const std::vector<std::uint8_t> key = endpointKey(endpoint);
  auto &shadow = selected->first->configurationShadow[key];
  auto &active = selected->first->activeConfiguration[key];
  for (const auto &[address, value] : shadow)
    active[address] = value;
  shadow.clear();
  ++state_->statistics.configurationCommitCount;
  return llvm::Error::success();
}

llvm::Expected<std::uint32_t> FabricModelRuntimeProvider::readConfigurationWord(
    const RuntimeLeaseHandle &lease, const RuntimeProviderEndpointRef &endpoint,
    std::uint32_t address) {
  if (llvm::Error error = validateRuntimeProviderEndpoint(
          descriptor(), endpoint, RuntimeEndpointClass::Programming,
          RuntimeEndpointFlow::ImplementationToRuntime))
    return std::move(error);
  std::lock_guard<std::mutex> lock(state_->mutex);
  auto selected = requireLease(*state_, lease);
  if (!selected)
    return selected.takeError();
  ++state_->statistics.configurationReadCount;
  const auto endpointState =
      selected->first->activeConfiguration.find(endpointKey(endpoint));
  if (endpointState == selected->first->activeConfiguration.end())
    return std::uint32_t{0};
  const auto value = endpointState->second.find(address);
  return value == endpointState->second.end() ? std::uint32_t{0}
                                              : value->second;
}

llvm::Error FabricModelRuntimeProvider::installStaticMemory(
    const RuntimeLeaseHandle &lease, const RuntimeStaticMemoryInstall &install,
    llvm::ArrayRef<RuntimeInterfaceBinding>) {
  std::lock_guard<std::mutex> lock(state_->mutex);
  auto selected = requireLease(*state_, lease);
  if (!selected)
    return selected.takeError();
  if (install.targets.empty())
    return invalid("static memory install has no exact target");
  selected->first->staticMemory.push_back(install);
  ++state_->statistics.staticMemoryInstallCount;
  return llvm::Error::success();
}

llvm::Error FabricModelRuntimeProvider::registerExecutables(
    const RuntimeLeaseHandle &lease,
    const RuntimeExecutableRegistrationView &registration) {
  if (llvm::Error error = validateExecutableRegistration(registration))
    return error;
  std::lock_guard<std::mutex> lock(state_->mutex);
  auto selected = requireLease(*state_, lease);
  if (!selected)
    return selected.takeError();
  selected->first->executablesRegistered = true;
  ++state_->statistics.executableRegistrationCount;
  return llvm::Error::success();
}

llvm::Error
FabricModelRuntimeProvider::activate(const RuntimeLeaseHandle &lease,
                                     const RuntimeActivationView &activation) {
  std::lock_guard<std::mutex> lock(state_->mutex);
  auto selected = requireLease(*state_, lease);
  if (!selected)
    return selected.takeError();
  if (!selected->first->executablesRegistered)
    return invalid("executables are not registered");
  selected->first->activeDeployment = activation.deployment;
  ++state_->statistics.activationCount;
  return llvm::Error::success();
}

llvm::Expected<RuntimePreparedActivationHandle>
FabricModelRuntimeProvider::prepareActivation(
    const RuntimeLeaseHandle &lease,
    const RuntimeExecutableRegistrationView &registration,
    const RuntimeActivationView &activation) {
  if (llvm::Error error = validateExecutableRegistration(registration))
    return std::move(error);
  std::lock_guard<std::mutex> lock(state_->mutex);
  auto selected = requireLease(*state_, lease);
  if (!selected)
    return selected.takeError();
  auto &device = *selected->first;
  if (!device.activeDeployment)
    return invalid("device has no active Deployment during preparation");
  if (device.nextPreparedIdentity == 0)
    return invalid("prepared activation identity is exhausted");
  const std::uint64_t identity = device.nextPreparedIdentity++;
  device.prepared.emplace(identity, activation.deployment);
  ++state_->statistics.activationPreparationCount;
  return preparedHandle(selected->second.device.instance,
                        selected->second.device.device,
                        selected->second.generation, identity);
}

llvm::Error FabricModelRuntimeProvider::replaceActivationAtomically(
    const RuntimeLeaseHandle &lease,
    const RuntimePreparedActivationHandle &prepared) {
  std::lock_guard<std::mutex> lock(state_->mutex);
  auto selected = requireLease(*state_, lease);
  if (!selected)
    return selected.takeError();
  auto identity = parsePrepared(prepared);
  if (!identity)
    return identity.takeError();
  if (identity->lease.device.instance != selected->second.device.instance ||
      identity->lease.device.device != selected->second.device.device ||
      identity->lease.generation != selected->second.generation)
    return invalid("prepared activation belongs to another lease");
  const auto activation = selected->first->prepared.find(identity->identity);
  if (activation == selected->first->prepared.end())
    return invalid("prepared activation handle is stale");
  selected->first->activeDeployment = activation->second;
  ++state_->statistics.activationReplacementCount;
  return llvm::Error::success();
}

llvm::Error FabricModelRuntimeProvider::discardPreparedActivation(
    const RuntimeLeaseHandle &lease,
    const RuntimePreparedActivationHandle &prepared) {
  std::lock_guard<std::mutex> lock(state_->mutex);
  auto selected = requireLease(*state_, lease);
  if (!selected)
    return selected.takeError();
  auto identity = parsePrepared(prepared);
  if (!identity)
    return identity.takeError();
  if (identity->lease.device.instance != selected->second.device.instance ||
      identity->lease.device.device != selected->second.device.device ||
      identity->lease.generation != selected->second.generation)
    return invalid("prepared activation belongs to another lease");
  if (selected->first->prepared.erase(identity->identity) != 1)
    return invalid("prepared activation handle is stale");
  ++state_->statistics.activationDiscardCount;
  return llvm::Error::success();
}

RuntimeLeaseFinalizationResult
FabricModelRuntimeProvider::finalizeExclusiveLease(
    const RuntimeLeaseHandle &lease, RuntimeLeaseFinalizationRequest request) {
  std::lock_guard<std::mutex> lock(state_->mutex);
  auto selected = requireLease(*state_, lease);
  if (!selected) {
    std::string diagnostic = llvm::toString(selected.takeError());
    for (State::Device &device : state_->devices) {
      if (!device.quarantined) {
        device.quarantined = true;
        ++state_->statistics.quarantineCount;
      }
      device.leased = false;
      device.prepared.clear();
    }
    return {RuntimeLeaseFinalState::Quarantined,
            "invalid lease forced provider-wide quarantine: " + diagnostic};
  }
  selected->first->prepared.clear();
  selected->first->leased = false;
  ++state_->statistics.leaseReleaseCount;
  if (request == RuntimeLeaseFinalizationRequest::Release)
    return {RuntimeLeaseFinalState::Released, {}};
  if (!selected->first->quarantined) {
    selected->first->quarantined = true;
    ++state_->statistics.quarantineCount;
  }
  return {RuntimeLeaseFinalState::Quarantined, {}};
}

FabricModelRuntimeStatistics FabricModelRuntimeProvider::statistics() const {
  std::lock_guard<std::mutex> lock(state_->mutex);
  return state_->statistics;
}

std::optional<ArtifactRootReference>
FabricModelRuntimeProvider::activeDeployment(std::uint64_t ordinal) const {
  std::lock_guard<std::mutex> lock(state_->mutex);
  if (ordinal >= state_->devices.size())
    return std::nullopt;
  return state_->devices[static_cast<std::size_t>(ordinal)].activeDeployment;
}

std::size_t FabricModelRuntimeProvider::preparedActivationCount(
    std::uint64_t ordinal) const {
  std::lock_guard<std::mutex> lock(state_->mutex);
  if (ordinal >= state_->devices.size())
    return 0;
  return state_->devices[static_cast<std::size_t>(ordinal)].prepared.size();
}

bool FabricModelRuntimeProvider::isQuarantined(std::uint64_t ordinal) const {
  std::lock_guard<std::mutex> lock(state_->mutex);
  return ordinal < state_->devices.size() &&
         state_->devices[static_cast<std::size_t>(ordinal)].quarantined;
}

llvm::Expected<std::shared_ptr<FabricModelRuntimeProvider>>
createFabricModelRuntimeProvider(
    std::vector<FabricModelRuntimeDeviceConfig> devices) {
  if (devices.empty())
    return invalid("provider requires at least one device");
  for (const auto &device : devices)
    if (device.hardwareImplementations.empty())
      return invalid("device requires at least one HardwareImplementation");
  if (llvm::Error error =
          registerRuntimeProvider(fabricModelRuntimeProviderDescriptor()))
    return std::move(error);
  auto providerInstance = allocateProviderInstance();
  if (!providerInstance)
    return providerInstance.takeError();
  return std::shared_ptr<FabricModelRuntimeProvider>(
      new FabricModelRuntimeProvider(
          std::make_unique<FabricModelRuntimeProvider::State>(
              *providerInstance, std::move(devices))));
}

} // namespace loom::runtime
