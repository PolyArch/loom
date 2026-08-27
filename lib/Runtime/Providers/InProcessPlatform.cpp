#include "Runtime/InProcessPlatform.h"

#include "Hardware/RTL/ConfigurationTransport.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
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

std::vector<std::uint8_t> encodeDevice(std::uint64_t instance,
                                       std::uint64_t device) {
  std::vector<std::uint8_t> bytes = encodeU64(instance);
  const std::vector<std::uint8_t> suffix = encodeU64(device);
  bytes.insert(bytes.end(), suffix.begin(), suffix.end());
  return bytes;
}

std::vector<std::uint8_t> encodeLease(std::uint64_t instance,
                                      std::uint64_t device,
                                      std::uint64_t generation) {
  std::vector<std::uint8_t> bytes = encodeDevice(instance, device);
  const std::vector<std::uint8_t> generationBytes = encodeU64(generation);
  bytes.insert(bytes.end(), generationBytes.begin(), generationBytes.end());
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

struct InProcessDeviceToken final {
  std::uint64_t instance;
  std::uint64_t device;
};

struct InProcessLeaseToken final {
  std::uint64_t instance;
  std::uint64_t device;
  std::uint64_t generation;
};

llvm::Expected<InProcessDeviceToken>
decodeDevice(llvm::ArrayRef<std::uint8_t> bytes) {
  if (bytes.size() != 16)
    return invalid("transient device handle is malformed");
  auto instance = decodeU64(bytes.take_front(8));
  auto device = decodeU64(bytes.drop_front(8));
  if (!instance)
    return instance.takeError();
  if (!device)
    return device.takeError();
  return InProcessDeviceToken{*instance, *device};
}

llvm::Expected<InProcessLeaseToken>
decodeLease(llvm::ArrayRef<std::uint8_t> bytes) {
  if (bytes.size() != 24)
    return invalid("transient lease handle is malformed");
  auto instance = decodeU64(bytes.take_front(8));
  auto device = decodeU64(bytes.slice(8, 8));
  auto generation = decodeU64(bytes.drop_front(16));
  if (!instance)
    return instance.takeError();
  if (!device)
    return device.takeError();
  if (!generation)
    return generation.takeError();
  return InProcessLeaseToken{*instance, *device, *generation};
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

BlobDigest foreignDigest(const BlobDigest &digest) {
  BlobDigest::Storage bytes = digest.bytes();
  bytes.front() ^= 1;
  return llvm::cantFail(BlobDigest::fromBytes(bytes));
}

enum class InProcessMachineLeaseOwnership : std::uint8_t {
  Unleased,
  Caller,
  Quarantine,
};

struct InProcessMachineDevice final {
  InProcessMachineDevice(std::vector<ArtifactIdentity> implementations,
                         std::optional<BlobDigest> attestation)
      : hardwareImplementations(std::move(implementations)),
        trustedAttestation(std::move(attestation)) {}

  std::vector<ArtifactIdentity> hardwareImplementations;
  std::optional<BlobDigest> trustedAttestation;
  InProcessMachineLeaseOwnership leaseOwnership =
      InProcessMachineLeaseOwnership::Unleased;
  bool quarantined = false;
  std::uint64_t leaseOwnerInstance = 0;
  std::uint64_t leaseGeneration = 0;
  bool activated = false;
  std::optional<ArtifactRootReference> activeDeployment;
  std::uint64_t nextPreparedActivation = 1;
  std::map<std::uint64_t, ArtifactRootReference> preparedActivations;
  std::map<std::string, std::map<std::uint32_t, std::uint32_t>> shadow;
  std::map<std::string, std::map<std::uint32_t, std::uint32_t>> active;
  std::vector<RuntimeStaticMemoryInstall> staticMemory;
};

struct InProcessMachineRegistry final {
  std::mutex mutex;
  std::uint64_t nextProviderInstance = 1;
  std::map<std::string, std::shared_ptr<InProcessMachineDevice>> namedDevices;
};

InProcessMachineRegistry &machineRegistry() {
  static InProcessMachineRegistry registry;
  return registry;
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
    {"loom.runtime.in_process", SchemaVersion{2, 0}},
    "loom.runtime.in_process.implementation.v2",
    hardware::rtl::portableConfigurationRuntimeAbiIdentity,
    endpointKinds,
    true,
    true,
    true,
    true};

} // namespace

struct InProcessRuntimeProvider::State final {
  struct Device final {
    Device(std::shared_ptr<InProcessMachineDevice> machine,
           InProcessRuntimeFailurePlan failures)
        : machine(std::move(machine)), failures(std::move(failures)) {}

    bool reportsForeignVerification() const {
      if (!failures.verificationMismatchBoundary)
        return false;
      switch (*failures.verificationMismatchBoundary) {
      case InProcessRuntimeVerificationMismatchBoundary::ExclusiveLease:
        return true;
      case InProcessRuntimeVerificationMismatchBoundary::InitialReset:
        return resetCount >= 1;
      case InProcessRuntimeVerificationMismatchBoundary::RecoveryReset:
        return resetCount >= 2;
      }
      llvm_unreachable("unknown identity mismatch boundary");
    }

    std::shared_ptr<InProcessMachineDevice> machine;
    InProcessRuntimeFailurePlan failures;
    std::uint64_t resetCount = 0;
    std::uint64_t writeCount = 0;
    std::uint64_t readCount = 0;
    std::uint64_t preparationCount = 0;
  };

  State(InProcessMachineRegistry &registry, std::uint64_t providerInstance,
        std::vector<Device> devices)
      : registry(registry), providerInstance(providerInstance),
        devices(std::move(devices)) {}

  llvm::Expected<Device *> device(const RuntimeDeviceHandle &handle) {
    auto token = decodeDevice(handle.opaque);
    if (!token)
      return token.takeError();
    if (token->instance != providerInstance)
      return invalid("device handle belongs to another provider instance");
    if (token->device >= devices.size())
      return invalid("device handle is outside enumeration");
    return &devices[static_cast<std::size_t>(token->device)];
  }

  llvm::Expected<std::pair<std::uint64_t, Device *>>
  leasedDevice(const RuntimeLeaseHandle &lease) {
    auto token = decodeLease(lease.opaque);
    if (!token)
      return token.takeError();
    if (token->instance != providerInstance)
      return invalid("lease belongs to another provider instance");
    if (token->device >= devices.size())
      return invalid("lease names an absent device");
    Device &device = devices[static_cast<std::size_t>(token->device)];
    if (device.machine->leaseOwnership !=
            InProcessMachineLeaseOwnership::Caller ||
        device.machine->quarantined ||
        device.machine->leaseOwnerInstance != providerInstance ||
        device.machine->leaseGeneration != token->generation)
      return invalid("lease is stale or inactive");
    return std::make_pair(token->device, &device);
  }

  InProcessMachineRegistry &registry;
  std::uint64_t providerInstance;
  std::vector<Device> devices;
  InProcessRuntimeStatistics statistics;
};

InProcessRuntimeProvider::InProcessRuntimeProvider(std::unique_ptr<State> state)
    : state_(std::move(state)) {}

InProcessRuntimeProvider::~InProcessRuntimeProvider() {
  std::lock_guard<std::mutex> lock(state_->registry.mutex);
  for (State::Device &binding : state_->devices) {
    InProcessMachineDevice &machine = *binding.machine;
    if (machine.leaseOwnership != InProcessMachineLeaseOwnership::Caller ||
        machine.leaseOwnerInstance != state_->providerInstance)
      continue;
    machine.quarantined = true;
    machine.leaseOwnership = InProcessMachineLeaseOwnership::Quarantine;
    machine.leaseOwnerInstance = 0;
  }
}

const RuntimeProviderDescriptor &InProcessRuntimeProvider::descriptor() const {
  return inProcessRuntimeProviderDescriptor();
}

llvm::Expected<std::vector<RuntimeDeviceHandle>>
InProcessRuntimeProvider::enumerateDevices() {
  std::lock_guard<std::mutex> lock(state_->registry.mutex);
  ++state_->statistics.enumerationCount;
  std::vector<RuntimeDeviceHandle> result;
  result.reserve(state_->devices.size());
  for (std::uint64_t ordinal = 0; ordinal != state_->devices.size(); ++ordinal)
    if (!state_->devices[static_cast<std::size_t>(ordinal)]
             .machine->quarantined)
      result.push_back(
          RuntimeDeviceHandle{encodeDevice(state_->providerInstance, ordinal)});
  return result;
}

llvm::Expected<ArtifactIdentity>
InProcessRuntimeProvider::readImplementationIdentity(
    const RuntimeLeaseHandle &lease,
    const RuntimeProviderEndpointRef &endpoint) {
  if (llvm::Error error = validateRuntimeProviderEndpoint(
          descriptor(), endpoint, RuntimeEndpointClass::Identity,
          RuntimeEndpointFlow::ImplementationToRuntime))
    return std::move(error);
  std::lock_guard<std::mutex> lock(state_->registry.mutex);
  auto leased = state_->leasedDevice(lease);
  if (!leased)
    return leased.takeError();
  State::Device *device = leased->second;
  auto ordinal = decodeU64(endpoint.payload);
  if (!ordinal)
    return ordinal.takeError();
  if (*ordinal >= device->machine->hardwareImplementations.size())
    return invalid("identity endpoint is outside the device implementation "
                   "set");
  ++state_->statistics.identityReadCount;
  const ArtifactIdentity &implementation =
      device->machine
          ->hardwareImplementations[static_cast<std::size_t>(*ordinal)];
  if (device->reportsForeignVerification())
    return foreignIdentity(implementation);
  return implementation;
}

llvm::Expected<BlobDigest> InProcessRuntimeProvider::readTrustedAttestation(
    const RuntimeLeaseHandle &lease) {
  std::lock_guard<std::mutex> lock(state_->registry.mutex);
  auto leased = state_->leasedDevice(lease);
  if (!leased)
    return leased.takeError();
  State::Device *device = leased->second;
  if (!device->machine->trustedAttestation)
    return failed("device has no trusted attestation");
  ++state_->statistics.attestationReadCount;
  if (device->reportsForeignVerification())
    return foreignDigest(*device->machine->trustedAttestation);
  return *device->machine->trustedAttestation;
}

llvm::Expected<RuntimeLeaseHandle>
InProcessRuntimeProvider::acquireExclusiveLease(
    const RuntimeDeviceHandle &handle) {
  std::lock_guard<std::mutex> lock(state_->registry.mutex);
  auto token = decodeDevice(handle.opaque);
  if (!token)
    return token.takeError();
  auto device = state_->device(handle);
  if (!device)
    return device.takeError();
  InProcessMachineDevice &machine = *(*device)->machine;
  if (machine.quarantined)
    return failed("device is quarantined");
  if (machine.leaseOwnership != InProcessMachineLeaseOwnership::Unleased)
    return failed("device already has an exclusive lease");
  if (machine.leaseGeneration == std::numeric_limits<std::uint64_t>::max()) {
    machine.quarantined = true;
    ++state_->statistics.quarantineCount;
    return failed("lease generation space is exhausted; device quarantined");
  }
  machine.leaseOwnership = InProcessMachineLeaseOwnership::Caller;
  machine.leaseOwnerInstance = state_->providerInstance;
  ++machine.leaseGeneration;
  ++state_->statistics.leaseAcquisitionCount;
  return RuntimeLeaseHandle{encodeLease(state_->providerInstance, token->device,
                                        machine.leaseGeneration)};
}

llvm::Error
InProcessRuntimeProvider::quiesceAndReset(const RuntimeLeaseHandle &lease) {
  std::lock_guard<std::mutex> lock(state_->registry.mutex);
  auto device = state_->leasedDevice(lease);
  if (!device)
    return device.takeError();
  InProcessMachineDevice &machine = *device->second->machine;
  machine.activated = false;
  machine.activeDeployment.reset();
  machine.preparedActivations.clear();
  machine.shadow.clear();
  machine.active.clear();
  machine.staticMemory.clear();
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
  std::lock_guard<std::mutex> lock(state_->registry.mutex);
  auto device = state_->leasedDevice(lease);
  if (!device)
    return device.takeError();
  State::Device &binding = *device->second;
  InProcessMachineDevice &target = *binding.machine;
  const std::uint64_t ordinal = binding.writeCount++;
  ++state_->statistics.configurationWriteCount;
  if (binding.failures.configurationWriteOrdinal == ordinal)
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
  std::lock_guard<std::mutex> lock(state_->registry.mutex);
  auto device = state_->leasedDevice(lease);
  if (!device)
    return device.takeError();
  InProcessMachineDevice &machine = *device->second->machine;
  const std::string key = endpointKey(endpoint);
  auto &shadow = machine.shadow[key];
  auto &active = machine.active[key];
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
  std::lock_guard<std::mutex> lock(state_->registry.mutex);
  auto device = state_->leasedDevice(lease);
  if (!device)
    return device.takeError();
  State::Device &binding = *device->second;
  InProcessMachineDevice &source = *binding.machine;
  const std::uint64_t ordinal = binding.readCount++;
  ++state_->statistics.configurationReadCount;
  std::uint32_t value = 0;
  const auto endpointState = source.active.find(endpointKey(endpoint));
  if (endpointState != source.active.end()) {
    const auto word = endpointState->second.find(address);
    if (word != endpointState->second.end())
      value = word->second;
  }
  if (binding.failures.readbackCorruption &&
      binding.failures.readbackCorruption->readOrdinal == ordinal)
    value ^= binding.failures.readbackCorruption->xorMask;
  return value;
}

llvm::Error InProcessRuntimeProvider::programConfigurationMulticast(
    const RuntimeLeaseHandle &lease,
    llvm::ArrayRef<RuntimeConfigurationTarget> targets) {
  if (targets.size() < 2)
    return invalid("multicast requires at least two targets");
  std::lock_guard<std::mutex> lock(state_->registry.mutex);
  auto device = state_->leasedDevice(lease);
  if (!device)
    return device.takeError();
  State::Device &binding = *device->second;
  InProcessMachineDevice &targetDevice = *binding.machine;
  ++state_->statistics.multicastTransactionCount;

  for (const RuntimeConfigurationTarget &target : targets) {
    if (llvm::Error error = validateRuntimeProviderEndpoint(
            descriptor(), target.endpoint, RuntimeEndpointClass::Programming,
            RuntimeEndpointFlow::RuntimeToImplementation))
      return error;
    for (const RuntimeConfigurationWord &word : target.words) {
      const std::uint64_t ordinal = binding.writeCount++;
      ++state_->statistics.configurationWriteCount;
      if (binding.failures.configurationWriteOrdinal == ordinal)
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
  std::lock_guard<std::mutex> lock(state_->registry.mutex);
  auto device = state_->leasedDevice(lease);
  if (!device)
    return device.takeError();
  if (install.targets.empty())
    return invalid("static memory install has no exact target");
  device->second->machine->staticMemory.push_back(install);
  ++state_->statistics.staticMemoryInstallCount;
  return llvm::Error::success();
}

llvm::Error InProcessRuntimeProvider::registerExecutables(
    const RuntimeLeaseHandle &lease,
    const RuntimeExecutableRegistrationView &registration) {
  if (llvm::Error error = validateExecutableRegistration(registration))
    return error;
  std::lock_guard<std::mutex> lock(state_->registry.mutex);
  auto device = state_->leasedDevice(lease);
  if (!device)
    return device.takeError();
  ++state_->statistics.executableRegistrationCount;
  return llvm::Error::success();
}

llvm::Error
InProcessRuntimeProvider::activate(const RuntimeLeaseHandle &lease,
                                   const RuntimeActivationView &activation) {
  std::lock_guard<std::mutex> lock(state_->registry.mutex);
  auto device = state_->leasedDevice(lease);
  if (!device)
    return device.takeError();
  device->second->machine->activated = true;
  device->second->machine->activeDeployment = activation.deployment;
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
  std::lock_guard<std::mutex> lock(state_->registry.mutex);
  auto device = state_->leasedDevice(lease);
  if (!device)
    return device.takeError();
  State::Device &binding = *device->second;
  InProcessMachineDevice &machine = *binding.machine;
  if (!machine.activated || !machine.activeDeployment)
    return invalid("device has no active Deployment during preparation");
  const std::uint64_t preparationOrdinal = binding.preparationCount++;
  if (binding.failures.activationPreparationOrdinal == preparationOrdinal)
    return failed("injected activation preparation failure");
  if (machine.nextPreparedActivation == 0)
    return invalid("prepared activation handle space is exhausted");
  const std::uint64_t ordinal = machine.nextPreparedActivation++;
  machine.preparedActivations.emplace(ordinal, activation.deployment);
  ++state_->statistics.activationPreparationCount;
  return RuntimePreparedActivationHandle{encodeU64(ordinal)};
}

llvm::Error InProcessRuntimeProvider::replaceActivationAtomically(
    const RuntimeLeaseHandle &lease,
    const RuntimePreparedActivationHandle &prepared) {
  auto ordinal = decodeU64(prepared.opaque);
  if (!ordinal)
    return ordinal.takeError();
  std::lock_guard<std::mutex> lock(state_->registry.mutex);
  auto device = state_->leasedDevice(lease);
  if (!device)
    return device.takeError();
  State::Device &binding = *device->second;
  InProcessMachineDevice &machine = *binding.machine;
  if (!machine.activated || !machine.activeDeployment)
    return invalid("device has no active Deployment to replace");
  auto activation = machine.preparedActivations.find(*ordinal);
  if (activation == machine.preparedActivations.end())
    return invalid("prepared activation handle is unknown");
  if (binding.failures.activationReplacementFailures != 0) {
    --binding.failures.activationReplacementFailures;
    return failed("injected atomic activation replacement failure");
  }
  machine.activeDeployment = activation->second;
  ++state_->statistics.activationReplacementCount;
  return llvm::Error::success();
}

llvm::Error InProcessRuntimeProvider::discardPreparedActivation(
    const RuntimeLeaseHandle &lease,
    const RuntimePreparedActivationHandle &prepared) {
  auto ordinal = decodeU64(prepared.opaque);
  if (!ordinal)
    return ordinal.takeError();
  std::lock_guard<std::mutex> lock(state_->registry.mutex);
  auto device = state_->leasedDevice(lease);
  if (!device)
    return device.takeError();
  State::Device &binding = *device->second;
  if (binding.failures.activationDiscardFailures != 0) {
    --binding.failures.activationDiscardFailures;
    return failed("injected prepared activation discard failure");
  }
  if (binding.machine->preparedActivations.erase(*ordinal) != 1)
    return invalid("prepared activation handle is unknown");
  ++state_->statistics.activationDiscardCount;
  return llvm::Error::success();
}

RuntimeLeaseFinalizationResult InProcessRuntimeProvider::finalizeExclusiveLease(
    const RuntimeLeaseHandle &lease, RuntimeLeaseFinalizationRequest request) {
  std::lock_guard<std::mutex> lock(state_->registry.mutex);
  auto device = state_->leasedDevice(lease);
  if (!device) {
    std::string diagnostic = llvm::toString(device.takeError());
    for (State::Device &candidate : state_->devices)
      if (!candidate.machine->quarantined) {
        candidate.machine->quarantined = true;
        ++state_->statistics.quarantineCount;
      }
    for (State::Device &candidate : state_->devices)
      if (candidate.machine->leaseOwnership ==
          InProcessMachineLeaseOwnership::Caller) {
        candidate.machine->leaseOwnership =
            InProcessMachineLeaseOwnership::Quarantine;
        candidate.machine->leaseOwnerInstance = 0;
      }
    return {RuntimeLeaseFinalState::Quarantined,
            "invalid lease forced provider-wide quarantine: " + diagnostic};
  }

  State::Device &binding = *device->second;
  InProcessMachineDevice &target = *binding.machine;
  std::string diagnostic;
  if (request == RuntimeLeaseFinalizationRequest::Release &&
      binding.failures.leaseReleaseFailures == 0) {
    target.leaseOwnership = InProcessMachineLeaseOwnership::Unleased;
    target.leaseOwnerInstance = 0;
    ++state_->statistics.leaseReleaseCount;
    return {RuntimeLeaseFinalState::Released, {}};
  }
  if (request == RuntimeLeaseFinalizationRequest::Release) {
    --binding.failures.leaseReleaseFailures;
    diagnostic = "exclusive lease release failed; device quarantined";
  }

  if (!target.quarantined) {
    target.quarantined = true;
    ++state_->statistics.quarantineCount;
  }
  if (binding.failures.quarantineLeaseReleaseFailures != 0) {
    --binding.failures.quarantineLeaseReleaseFailures;
    target.leaseOwnership = InProcessMachineLeaseOwnership::Quarantine;
    target.leaseOwnerInstance = 0;
    if (!diagnostic.empty())
      diagnostic += "; ";
    diagnostic += "quarantined lease remains provider-owned";
  } else {
    target.leaseOwnership = InProcessMachineLeaseOwnership::Unleased;
    target.leaseOwnerInstance = 0;
    ++state_->statistics.leaseReleaseCount;
  }
  return {RuntimeLeaseFinalState::Quarantined, std::move(diagnostic)};
}

InProcessRuntimeStatistics InProcessRuntimeProvider::statistics() const {
  std::lock_guard<std::mutex> lock(state_->registry.mutex);
  return state_->statistics;
}

bool InProcessRuntimeProvider::isQuarantined(
    std::uint64_t deviceOrdinal) const {
  std::lock_guard<std::mutex> lock(state_->registry.mutex);
  return deviceOrdinal < state_->devices.size() &&
         state_->devices[static_cast<std::size_t>(deviceOrdinal)]
             .machine->quarantined;
}

std::optional<ArtifactRootReference>
InProcessRuntimeProvider::activeDeployment(std::uint64_t deviceOrdinal) const {
  std::lock_guard<std::mutex> lock(state_->registry.mutex);
  if (deviceOrdinal >= state_->devices.size())
    return std::nullopt;
  return state_->devices[static_cast<std::size_t>(deviceOrdinal)]
      .machine->activeDeployment;
}

std::size_t InProcessRuntimeProvider::preparedActivationCount(
    std::uint64_t deviceOrdinal) const {
  std::lock_guard<std::mutex> lock(state_->registry.mutex);
  if (deviceOrdinal >= state_->devices.size())
    return 0;
  return state_->devices[static_cast<std::size_t>(deviceOrdinal)]
      .machine->preparedActivations.size();
}

const RuntimeProviderDescriptor &inProcessRuntimeProviderDescriptor() {
  return descriptor;
}

llvm::Expected<std::shared_ptr<InProcessRuntimeProvider>>
createInProcessRuntimeProvider(
    std::vector<InProcessRuntimeDeviceConfig> devices) {
  if (llvm::Error error = registerRuntimeProvider(descriptor))
    return std::move(error);

  InProcessMachineRegistry &registry = machineRegistry();
  std::lock_guard<std::mutex> lock(registry.mutex);
  std::set<std::string> localMachineIdentities;
  for (const InProcessRuntimeDeviceConfig &device : devices) {
    if (!device.machineDeviceIdentity)
      continue;
    if (device.machineDeviceIdentity->empty())
      return invalid("machine device identity must be non-empty");
    if (!localMachineIdentities.insert(*device.machineDeviceIdentity).second)
      return invalid("one provider instance names a machine device twice");
    const auto known =
        registry.namedDevices.find(*device.machineDeviceIdentity);
    if (known != registry.namedDevices.end() &&
        (known->second->hardwareImplementations !=
             device.hardwareImplementations ||
         known->second->trustedAttestation != device.trustedAttestation))
      return invalid("machine device identity changed its physical identity");
  }
  if (registry.nextProviderInstance == 0)
    return invalid("provider instance token space is exhausted");
  const std::uint64_t providerInstance = registry.nextProviderInstance++;

  std::vector<InProcessRuntimeProvider::State::Device> bindings;
  bindings.reserve(devices.size());
  for (InProcessRuntimeDeviceConfig &device : devices) {
    std::shared_ptr<InProcessMachineDevice> machine;
    if (device.machineDeviceIdentity) {
      auto [known, inserted] = registry.namedDevices.try_emplace(
          *device.machineDeviceIdentity, nullptr);
      if (inserted)
        known->second = std::make_shared<InProcessMachineDevice>(
            std::move(device.hardwareImplementations),
            std::move(device.trustedAttestation));
      machine = known->second;
    } else {
      machine = std::make_shared<InProcessMachineDevice>(
          std::move(device.hardwareImplementations),
          std::move(device.trustedAttestation));
    }
    bindings.emplace_back(std::move(machine), std::move(device.failures));
  }

  return std::shared_ptr<InProcessRuntimeProvider>(new InProcessRuntimeProvider(
      std::make_unique<InProcessRuntimeProvider::State>(
          registry, providerInstance, std::move(bindings))));
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
