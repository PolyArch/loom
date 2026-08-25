#include "Runtime/InProcessPlatform.h"

#include "Hardware/RTL/ConfigurationTransport.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <system_error>
#include <utility>

namespace loom::runtime {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      llvm::Twine("in_process_runtime_invalid: ") + message);
}

llvm::Error failed(const llvm::Twine &message) {
  return llvm::createStringError(std::make_error_code(std::errc::io_error),
                                 llvm::Twine("in_process_runtime_failed: ") +
                                     message);
}

llvm::Error validateEndpointPayload(llvm::ArrayRef<std::uint8_t> payload) {
  if (payload.size() != 8)
    return invalid("endpoint payload must be one u64be ordinal");
  return llvm::Error::success();
}

std::vector<std::uint8_t> encodeU64(std::uint64_t value) {
  std::vector<std::uint8_t> bytes;
  bytes.reserve(8);
  for (unsigned shift = 56;; shift -= 8) {
    bytes.push_back(static_cast<std::uint8_t>(value >> shift));
    if (shift == 0)
      break;
  }
  return bytes;
}

std::vector<std::uint8_t> encodeLease(std::uint64_t device,
                                      std::uint64_t generation) {
  std::vector<std::uint8_t> bytes = encodeU64(device);
  const std::vector<std::uint8_t> suffix = encodeU64(generation);
  bytes.insert(bytes.end(), suffix.begin(), suffix.end());
  return bytes;
}

llvm::Expected<std::uint64_t> decodeU64(llvm::ArrayRef<std::uint8_t> bytes) {
  if (bytes.size() != 8)
    return invalid("transient device handle is malformed");
  std::uint64_t value = 0;
  for (std::uint8_t byte : bytes)
    value = (value << 8) | byte;
  return value;
}

llvm::Expected<std::pair<std::uint64_t, std::uint64_t>>
decodeLease(llvm::ArrayRef<std::uint8_t> bytes) {
  if (bytes.size() != 16)
    return invalid("transient lease handle is malformed");
  auto device = decodeU64(bytes.take_front(8));
  auto generation = decodeU64(bytes.drop_front(8));
  if (!device)
    return device.takeError();
  if (!generation)
    return generation.takeError();
  return std::make_pair(*device, *generation);
}

std::string endpointKey(const RuntimeProviderEndpointRef &endpoint) {
  std::string key;
  key.reserve(4 + endpoint.payload.size());
  for (unsigned shift = 24;; shift -= 8) {
    key.push_back(static_cast<char>(endpoint.kind >> shift));
    if (shift == 0)
      break;
  }
  key.append(reinterpret_cast<const char *>(endpoint.payload.data()),
             endpoint.payload.size());
  return key;
}

ArtifactIdentity foreignIdentity(const ArtifactIdentity &identity) {
  ArtifactIdentity::Storage bytes = identity.bytes();
  bytes.front() ^= 1;
  return llvm::cantFail(ArtifactIdentity::fromBytes(bytes));
}

llvm::Error validateExecutableRegistration(
    const RuntimeExecutableRegistrationView &registration) {
  if (registration.hostProgramBytes.empty())
    return invalid("host program is empty");
  if (registration.instructionCoreBinaries.size() !=
      registration.instructionCoreProgramBytes.size())
    return invalid("InstructionCore binary and byte catalogs disagree");
  if (llvm::any_of(
          registration.instructionCoreProgramBytes,
          [](llvm::ArrayRef<std::uint8_t> bytes) { return bytes.empty(); }))
    return invalid("InstructionCore program is empty");
  return llvm::Error::success();
}

const RuntimeProviderEndpointKindDescriptor endpointKinds[] = {
    {0, "identity", RuntimeEndpointClass::Identity,
     RuntimeEndpointFlow::ImplementationToRuntime, false,
     validateEndpointPayload},
    {1, "programming", RuntimeEndpointClass::Programming,
     RuntimeEndpointFlow::Bidirectional, true, validateEndpointPayload},
    {2, "memory", RuntimeEndpointClass::Memory,
     RuntimeEndpointFlow::Bidirectional, true, validateEndpointPayload},
    {3, "completion", RuntimeEndpointClass::Completion,
     RuntimeEndpointFlow::Bidirectional, true, validateEndpointPayload},
};

const RuntimeProviderDescriptor descriptor{
    {"loom.runtime.in_process", SchemaVersion{1, 0}},
    "loom.runtime.in_process.implementation.v1",
    hardware::rtl::portableConfigurationRuntimeAbiIdentity,
    endpointKinds,
    true,
    true,
    true,
    true};

} // namespace

struct InProcessRuntimeProvider::State final {
  struct Device final {
    explicit Device(InProcessRuntimeDeviceConfig configuration)
        : config(std::move(configuration)) {}

    InProcessRuntimeDeviceConfig config;
    bool leased = false;
    bool quarantined = false;
    bool activated = false;
    std::optional<ArtifactRootReference> activeDeployment;
    std::uint64_t nextPreparedActivation = 1;
    std::uint64_t preparationCount = 0;
    std::map<std::uint64_t, ArtifactRootReference> preparedActivations;
    std::uint64_t leaseGeneration = 0;
    std::uint64_t resetCount = 0;
    std::uint64_t writeCount = 0;
    std::uint64_t readCount = 0;
    std::map<std::string, std::map<std::uint32_t, std::uint32_t>> shadow;
    std::map<std::string, std::map<std::uint32_t, std::uint32_t>> active;
    std::vector<RuntimeStaticMemoryInstall> staticMemory;
  };

  explicit State(std::vector<InProcessRuntimeDeviceConfig> configurations) {
    devices.reserve(configurations.size());
    for (InProcessRuntimeDeviceConfig &configuration : configurations)
      devices.emplace_back(std::move(configuration));
  }

  llvm::Expected<Device *> device(const RuntimeDeviceHandle &handle) {
    auto ordinal = decodeU64(handle.opaque);
    if (!ordinal)
      return ordinal.takeError();
    if (*ordinal >= devices.size())
      return invalid("device handle is outside enumeration");
    return &devices[static_cast<std::size_t>(*ordinal)];
  }

  llvm::Expected<std::pair<std::uint64_t, Device *>>
  leasedDevice(const RuntimeLeaseHandle &lease) {
    auto decoded = decodeLease(lease.opaque);
    if (!decoded)
      return decoded.takeError();
    if (decoded->first >= devices.size())
      return invalid("lease names an absent device");
    Device &device = devices[static_cast<std::size_t>(decoded->first)];
    if (!device.leased || device.leaseGeneration != decoded->second)
      return invalid("lease is stale or inactive");
    return std::make_pair(decoded->first, &device);
  }

  mutable std::mutex mutex;
  std::vector<Device> devices;
  InProcessRuntimeStatistics statistics;
};

InProcessRuntimeProvider::InProcessRuntimeProvider(std::unique_ptr<State> state)
    : state_(std::move(state)) {}

InProcessRuntimeProvider::~InProcessRuntimeProvider() = default;

const RuntimeProviderDescriptor &InProcessRuntimeProvider::descriptor() const {
  return inProcessRuntimeProviderDescriptor();
}

llvm::Expected<std::vector<RuntimeDeviceHandle>>
InProcessRuntimeProvider::enumerateDevices() {
  std::lock_guard<std::mutex> lock(state_->mutex);
  ++state_->statistics.enumerationCount;
  std::vector<RuntimeDeviceHandle> result;
  result.reserve(state_->devices.size());
  for (std::uint64_t ordinal = 0; ordinal != state_->devices.size(); ++ordinal)
    if (!state_->devices[static_cast<std::size_t>(ordinal)].quarantined)
      result.push_back(RuntimeDeviceHandle{encodeU64(ordinal)});
  return result;
}

llvm::Expected<ArtifactIdentity>
InProcessRuntimeProvider::readImplementationIdentity(
    const RuntimeDeviceHandle &handle,
    const RuntimeProviderEndpointRef &endpoint) {
  if (llvm::Error error = validateRuntimeProviderEndpoint(
          descriptor(), endpoint, RuntimeEndpointClass::Identity,
          RuntimeEndpointFlow::ImplementationToRuntime))
    return std::move(error);
  std::lock_guard<std::mutex> lock(state_->mutex);
  auto device = state_->device(handle);
  if (!device)
    return device.takeError();
  auto ordinal = decodeU64(endpoint.payload);
  if (!ordinal)
    return ordinal.takeError();
  if (*ordinal >= (*device)->config.hardwareImplementations.size())
    return invalid("identity endpoint is outside the device implementation "
                   "set");
  ++state_->statistics.identityReadCount;
  const ArtifactIdentity &implementation =
      (*device)
          ->config.hardwareImplementations[static_cast<std::size_t>(*ordinal)];
  if ((*device)->config.failures.identityMismatchAfterRecoveryReset &&
      (*device)->resetCount >= 2)
    return foreignIdentity(implementation);
  return implementation;
}

llvm::Expected<BlobDigest> InProcessRuntimeProvider::readTrustedAttestation(
    const RuntimeDeviceHandle &handle) {
  std::lock_guard<std::mutex> lock(state_->mutex);
  auto device = state_->device(handle);
  if (!device)
    return device.takeError();
  if (!(*device)->config.trustedAttestation)
    return failed("device has no trusted attestation");
  return *(*device)->config.trustedAttestation;
}

llvm::Expected<RuntimeLeaseHandle>
InProcessRuntimeProvider::acquireExclusiveLease(
    const RuntimeDeviceHandle &handle) {
  std::lock_guard<std::mutex> lock(state_->mutex);
  auto ordinal = decodeU64(handle.opaque);
  if (!ordinal)
    return ordinal.takeError();
  auto device = state_->device(handle);
  if (!device)
    return device.takeError();
  if ((*device)->quarantined)
    return failed("device is quarantined");
  if ((*device)->leased)
    return failed("device already has an exclusive lease");
  (*device)->leased = true;
  ++(*device)->leaseGeneration;
  ++state_->statistics.leaseAcquisitionCount;
  return RuntimeLeaseHandle{encodeLease(*ordinal, (*device)->leaseGeneration)};
}

llvm::Error
InProcessRuntimeProvider::quiesceAndReset(const RuntimeLeaseHandle &lease) {
  std::lock_guard<std::mutex> lock(state_->mutex);
  auto device = state_->leasedDevice(lease);
  if (!device)
    return device.takeError();
  device->second->activated = false;
  device->second->activeDeployment.reset();
  device->second->preparedActivations.clear();
  device->second->shadow.clear();
  device->second->active.clear();
  device->second->staticMemory.clear();
  ++device->second->resetCount;
  ++state_->statistics.resetCount;
  return llvm::Error::success();
}

llvm::Error InProcessRuntimeProvider::writeConfigurationWord(
    const RuntimeLeaseHandle &lease, const RuntimeProviderEndpointRef &endpoint,
    const RuntimeConfigurationWord &word) {
  if (llvm::Error error = validateRuntimeProviderEndpoint(
          descriptor(), endpoint, RuntimeEndpointClass::Programming,
          RuntimeEndpointFlow::RuntimeToImplementation))
    return error;
  std::lock_guard<std::mutex> lock(state_->mutex);
  auto device = state_->leasedDevice(lease);
  if (!device)
    return device.takeError();
  State::Device &target = *device->second;
  const std::uint64_t ordinal = target.writeCount++;
  ++state_->statistics.configurationWriteCount;
  if (target.config.failures.configurationWriteOrdinal == ordinal)
    return failed("injected configuration write failure");
  std::uint32_t &stored = target.shadow[endpointKey(endpoint)][word.address];
  for (unsigned byte = 0; byte != 4; ++byte)
    if ((word.byteStrobe & (1U << byte)) != 0) {
      const std::uint32_t mask = std::uint32_t{0xff} << (byte * 8);
      stored = (stored & ~mask) | (word.value & mask);
    }
  return llvm::Error::success();
}

llvm::Error InProcessRuntimeProvider::commitConfiguration(
    const RuntimeLeaseHandle &lease, const RuntimeProviderEndpointRef &endpoint,
    std::uint32_t commitAddress) {
  (void)commitAddress;
  if (llvm::Error error = validateRuntimeProviderEndpoint(
          descriptor(), endpoint, RuntimeEndpointClass::Programming,
          RuntimeEndpointFlow::RuntimeToImplementation))
    return error;
  std::lock_guard<std::mutex> lock(state_->mutex);
  auto device = state_->leasedDevice(lease);
  if (!device)
    return device.takeError();
  const std::string key = endpointKey(endpoint);
  auto &shadow = device->second->shadow[key];
  auto &active = device->second->active[key];
  for (const auto &[address, value] : shadow)
    active[address] = value;
  shadow.clear();
  ++state_->statistics.configurationCommitCount;
  return llvm::Error::success();
}

llvm::Expected<std::uint32_t> InProcessRuntimeProvider::readConfigurationWord(
    const RuntimeLeaseHandle &lease, const RuntimeProviderEndpointRef &endpoint,
    std::uint32_t address) {
  if (llvm::Error error = validateRuntimeProviderEndpoint(
          descriptor(), endpoint, RuntimeEndpointClass::Programming,
          RuntimeEndpointFlow::ImplementationToRuntime))
    return std::move(error);
  std::lock_guard<std::mutex> lock(state_->mutex);
  auto device = state_->leasedDevice(lease);
  if (!device)
    return device.takeError();
  State::Device &source = *device->second;
  const std::uint64_t ordinal = source.readCount++;
  ++state_->statistics.configurationReadCount;
  std::uint32_t value = 0;
  const auto endpointState = source.active.find(endpointKey(endpoint));
  if (endpointState != source.active.end()) {
    const auto word = endpointState->second.find(address);
    if (word != endpointState->second.end())
      value = word->second;
  }
  if (source.config.failures.readbackCorruption &&
      source.config.failures.readbackCorruption->readOrdinal == ordinal)
    value ^= source.config.failures.readbackCorruption->xorMask;
  return value;
}

llvm::Error InProcessRuntimeProvider::programConfigurationMulticast(
    const RuntimeLeaseHandle &lease,
    llvm::ArrayRef<RuntimeConfigurationTarget> targets) {
  if (targets.size() < 2)
    return invalid("multicast requires at least two targets");
  std::lock_guard<std::mutex> lock(state_->mutex);
  auto device = state_->leasedDevice(lease);
  if (!device)
    return device.takeError();
  State::Device &targetDevice = *device->second;
  ++state_->statistics.multicastTransactionCount;

  for (const RuntimeConfigurationTarget &target : targets) {
    if (llvm::Error error = validateRuntimeProviderEndpoint(
            descriptor(), target.endpoint, RuntimeEndpointClass::Programming,
            RuntimeEndpointFlow::RuntimeToImplementation))
      return error;
    for (const RuntimeConfigurationWord &word : target.words) {
      const std::uint64_t ordinal = targetDevice.writeCount++;
      ++state_->statistics.configurationWriteCount;
      if (targetDevice.config.failures.configurationWriteOrdinal == ordinal)
        return failed("injected atomic multicast write failure");
      (void)word;
    }
  }

  for (const RuntimeConfigurationTarget &target : targets) {
    std::map<std::uint32_t, std::uint32_t> active;
    for (const RuntimeConfigurationWord &word : target.words) {
      std::uint32_t &stored = active[word.address];
      for (unsigned byte = 0; byte != 4; ++byte)
        if ((word.byteStrobe & (1U << byte)) != 0) {
          const std::uint32_t mask = std::uint32_t{0xff} << (byte * 8);
          stored = (stored & ~mask) | (word.value & mask);
        }
    }
    targetDevice.active[endpointKey(target.endpoint)] = std::move(active);
  }
  return llvm::Error::success();
}

llvm::Error InProcessRuntimeProvider::installStaticMemory(
    const RuntimeLeaseHandle &lease, const RuntimeStaticMemoryInstall &install,
    llvm::ArrayRef<RuntimeInterfaceBinding> memoryBindings) {
  (void)memoryBindings;
  std::lock_guard<std::mutex> lock(state_->mutex);
  auto device = state_->leasedDevice(lease);
  if (!device)
    return device.takeError();
  if (install.targets.empty())
    return invalid("static memory install has no exact target");
  device->second->staticMemory.push_back(install);
  ++state_->statistics.staticMemoryInstallCount;
  return llvm::Error::success();
}

llvm::Error InProcessRuntimeProvider::registerExecutables(
    const RuntimeLeaseHandle &lease,
    const RuntimeExecutableRegistrationView &registration) {
  if (llvm::Error error = validateExecutableRegistration(registration))
    return error;
  std::lock_guard<std::mutex> lock(state_->mutex);
  auto device = state_->leasedDevice(lease);
  if (!device)
    return device.takeError();
  ++state_->statistics.executableRegistrationCount;
  return llvm::Error::success();
}

llvm::Error
InProcessRuntimeProvider::activate(const RuntimeLeaseHandle &lease,
                                   const RuntimeActivationView &activation) {
  std::lock_guard<std::mutex> lock(state_->mutex);
  auto device = state_->leasedDevice(lease);
  if (!device)
    return device.takeError();
  device->second->activated = true;
  device->second->activeDeployment = activation.deployment;
  ++state_->statistics.activationCount;
  return llvm::Error::success();
}

llvm::Expected<RuntimePreparedActivationHandle>
InProcessRuntimeProvider::prepareActivation(
    const RuntimeLeaseHandle &lease,
    const RuntimeExecutableRegistrationView &registration,
    const RuntimeActivationView &activation) {
  if (llvm::Error error = validateExecutableRegistration(registration))
    return std::move(error);
  std::lock_guard<std::mutex> lock(state_->mutex);
  auto device = state_->leasedDevice(lease);
  if (!device)
    return device.takeError();
  if (!device->second->activated || !device->second->activeDeployment)
    return invalid("device has no active Deployment during preparation");
  const std::uint64_t preparationOrdinal = device->second->preparationCount++;
  if (device->second->config.failures.activationPreparationOrdinal ==
      preparationOrdinal)
    return failed("injected activation preparation failure");
  if (device->second->nextPreparedActivation == 0)
    return invalid("prepared activation handle space is exhausted");
  const std::uint64_t ordinal = device->second->nextPreparedActivation++;
  device->second->preparedActivations.emplace(ordinal, activation.deployment);
  ++state_->statistics.activationPreparationCount;
  return RuntimePreparedActivationHandle{encodeU64(ordinal)};
}

llvm::Error InProcessRuntimeProvider::replaceActivationAtomically(
    const RuntimeLeaseHandle &lease,
    const RuntimePreparedActivationHandle &prepared) {
  auto ordinal = decodeU64(prepared.opaque);
  if (!ordinal)
    return ordinal.takeError();
  std::lock_guard<std::mutex> lock(state_->mutex);
  auto device = state_->leasedDevice(lease);
  if (!device)
    return device.takeError();
  if (!device->second->activated || !device->second->activeDeployment)
    return invalid("device has no active Deployment to replace");
  auto activation = device->second->preparedActivations.find(*ordinal);
  if (activation == device->second->preparedActivations.end())
    return invalid("prepared activation handle is unknown");
  if (device->second->config.failures.activationReplacementFailures != 0) {
    --device->second->config.failures.activationReplacementFailures;
    return failed("injected atomic activation replacement failure");
  }
  device->second->activeDeployment = activation->second;
  ++state_->statistics.activationReplacementCount;
  return llvm::Error::success();
}

llvm::Error InProcessRuntimeProvider::discardPreparedActivation(
    const RuntimeLeaseHandle &lease,
    const RuntimePreparedActivationHandle &prepared) {
  auto ordinal = decodeU64(prepared.opaque);
  if (!ordinal)
    return ordinal.takeError();
  std::lock_guard<std::mutex> lock(state_->mutex);
  auto device = state_->leasedDevice(lease);
  if (!device)
    return device.takeError();
  if (device->second->config.failures.activationDiscardFailures != 0) {
    --device->second->config.failures.activationDiscardFailures;
    return failed("injected prepared activation discard failure");
  }
  if (device->second->preparedActivations.erase(*ordinal) != 1)
    return invalid("prepared activation handle is unknown");
  ++state_->statistics.activationDiscardCount;
  return llvm::Error::success();
}

llvm::Error InProcessRuntimeProvider::releaseExclusiveLease(
    const RuntimeLeaseHandle &lease) {
  std::lock_guard<std::mutex> lock(state_->mutex);
  auto device = state_->leasedDevice(lease);
  if (!device)
    return device.takeError();
  device->second->leased = false;
  ++state_->statistics.leaseReleaseCount;
  return llvm::Error::success();
}

void InProcessRuntimeProvider::quarantineDevice(
    const RuntimeDeviceHandle &handle) {
  std::lock_guard<std::mutex> lock(state_->mutex);
  auto device = state_->device(handle);
  if (!device) {
    llvm::consumeError(device.takeError());
    return;
  }
  if (!(*device)->quarantined) {
    (*device)->quarantined = true;
    ++state_->statistics.quarantineCount;
  }
}

InProcessRuntimeStatistics InProcessRuntimeProvider::statistics() const {
  std::lock_guard<std::mutex> lock(state_->mutex);
  return state_->statistics;
}

bool InProcessRuntimeProvider::isQuarantined(
    std::uint64_t deviceOrdinal) const {
  std::lock_guard<std::mutex> lock(state_->mutex);
  return deviceOrdinal < state_->devices.size() &&
         state_->devices[static_cast<std::size_t>(deviceOrdinal)].quarantined;
}

std::optional<ArtifactRootReference>
InProcessRuntimeProvider::activeDeployment(std::uint64_t deviceOrdinal) const {
  std::lock_guard<std::mutex> lock(state_->mutex);
  if (deviceOrdinal >= state_->devices.size())
    return std::nullopt;
  return state_->devices[static_cast<std::size_t>(deviceOrdinal)]
      .activeDeployment;
}

std::size_t InProcessRuntimeProvider::preparedActivationCount(
    std::uint64_t deviceOrdinal) const {
  std::lock_guard<std::mutex> lock(state_->mutex);
  if (deviceOrdinal >= state_->devices.size())
    return 0;
  return state_->devices[static_cast<std::size_t>(deviceOrdinal)]
      .preparedActivations.size();
}

const RuntimeProviderDescriptor &inProcessRuntimeProviderDescriptor() {
  return descriptor;
}

llvm::Expected<std::shared_ptr<InProcessRuntimeProvider>>
createInProcessRuntimeProvider(
    std::vector<InProcessRuntimeDeviceConfig> devices) {
  if (llvm::Error error = registerRuntimeProvider(descriptor))
    return std::move(error);
  return std::shared_ptr<InProcessRuntimeProvider>(new InProcessRuntimeProvider(
      std::make_unique<InProcessRuntimeProvider::State>(std::move(devices))));
}

RuntimeProviderEndpointRef
inProcessRuntimeEndpoint(RuntimeEndpointClass endpointClass,
                         std::uint64_t endpointOrdinal) {
  std::uint32_t kind = 0;
  switch (endpointClass) {
  case RuntimeEndpointClass::Identity:
    kind = 0;
    break;
  case RuntimeEndpointClass::Programming:
    kind = 1;
    break;
  case RuntimeEndpointClass::Memory:
    kind = 2;
    break;
  case RuntimeEndpointClass::Completion:
    kind = 3;
    break;
  }
  return RuntimeProviderEndpointRef{kind, encodeU64(endpointOrdinal)};
}

} // namespace loom::runtime
