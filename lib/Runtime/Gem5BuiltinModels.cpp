#include "Runtime/Gem5BuiltinModels.h"

#include "Runtime/Gem5BridgeWire.h"

#include "llvm/ADT/STLExtras.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <system_error>

namespace loom::runtime {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      "gem5_builtin_model_invalid: " + message);
}

void appendU64(std::vector<std::uint8_t> &bytes, std::uint64_t value) {
  for (unsigned shift = 56; shift != 0; shift -= 8)
    bytes.push_back(static_cast<std::uint8_t>(value >> shift));
  bytes.push_back(static_cast<std::uint8_t>(value));
}

std::uint64_t readU64(llvm::ArrayRef<std::uint8_t> bytes,
                      std::size_t offset) {
  std::uint64_t value = 0;
  for (unsigned index = 0; index < 8; ++index)
    value = (value << 8) | bytes[offset + index];
  return value;
}

llvm::Error validateEmpty(llvm::ArrayRef<std::uint8_t> bytes) {
  return bytes.empty() ? llvm::Error::success()
                       : invalid("payload must be empty");
}

llvm::Error validateCpu(llvm::ArrayRef<std::uint8_t> bytes) {
  auto parameters = decodeGem5RiscvTimingCpuParameters(bytes);
  return parameters ? llvm::Error::success() : parameters.takeError();
}

llvm::Error validateBridge(llvm::ArrayRef<std::uint8_t> bytes) {
  auto parameters = decodeGem5SpatialBridgeParameters(bytes);
  return parameters ? llvm::Error::success() : parameters.takeError();
}

llvm::Error validateMemory(llvm::ArrayRef<std::uint8_t> bytes) {
  auto parameters = decodeGem5SimpleMemoryParameters(bytes);
  return parameters ? llvm::Error::success() : parameters.takeError();
}

llvm::Error validateCpuCompatibility(
    llvm::ArrayRef<std::uint8_t> payload,
    const fabric::InstructionCoreArchitecturalContract &architecture,
    const fabric::InstructionCoreMicroarchitecturalRealization
        &microarchitecture) {
  if (llvm::Error error = validateCpu(payload))
    return error;
  if (architecture.xlen() != fabric::RiscVXLen::X64 ||
      architecture.endianness() != fabric::InstructionEndianness::Little ||
      !llvm::is_contained(architecture.privilegeModes(),
                          fabric::PrivilegeMode::Machine) ||
      microarchitecture.hardwareThreadCount() != 1)
    return invalid("TimingSimpleCPU requires one little-endian RV64 machine "
                   "hardware thread");
  return llvm::Error::success();
}

const Gem5ModelPortKindDescriptor kBridgePorts[] = {
    {0, "spatial_boundary", Gem5ModelPortClass::SpatialBoundary, false,
     &validateEmpty}};
const Gem5ModelPortKindDescriptor kMemoryPorts[] = {
    {0, "memory_or_service", Gem5ModelPortClass::MemoryOrService, true,
     &validateEmpty}};
const Gem5ModelPortKindDescriptor kTransportPorts[] = {
    {0, "transport", Gem5ModelPortClass::Transport, true, &validateEmpty}};
const Gem5ModelPortKindDescriptor kExternalPorts[] = {
    {0, "external_endpoint", Gem5ModelPortClass::ExternalEndpoint, true,
     &validateEmpty}};

} // namespace

const Gem5ModelContractDescriptor &gem5RiscvTimingCpuModel() {
  static const Gem5ModelContractDescriptor descriptor{
      {"loom.gem5.riscv_timing_cpu", {1, 0}},
      "loom.gem5.riscv_timing_cpu.v1",
      "RiscvTimingSimpleCPU",
      Gem5ModelObjectClass::Processor,
      false,
      &validateCpu,
      &validateCpuCompatibility,
      {}};
  return descriptor;
}

const Gem5ModelContractDescriptor &gem5SpatialBridgeModel() {
  static const Gem5ModelContractDescriptor descriptor{
      {"loom.gem5.spatial_bridge", {1, 0}},
      "loom.gem5.spatial_bridge.v1",
      "LoomSpatialBridge",
      Gem5ModelObjectClass::SpatialBridge,
      false,
      &validateBridge,
      nullptr,
      kBridgePorts};
  return descriptor;
}

const Gem5ModelContractDescriptor &gem5SimpleMemoryModel() {
  static const Gem5ModelContractDescriptor descriptor{
      {"loom.gem5.simple_memory", {1, 0}},
      "loom.gem5.simple_memory.v1",
      "SimpleMemory",
      Gem5ModelObjectClass::MemoryOrService,
      true,
      &validateMemory,
      nullptr,
      kMemoryPorts};
  return descriptor;
}

const Gem5ModelContractDescriptor &gem5SystemXBarModel() {
  static const Gem5ModelContractDescriptor descriptor{
      {"loom.gem5.system_xbar", {1, 0}},
      "loom.gem5.system_xbar.v1",
      "SystemXBar",
      Gem5ModelObjectClass::Transport,
      true,
      &validateEmpty,
      nullptr,
      kTransportPorts};
  return descriptor;
}

const Gem5ModelContractDescriptor &gem5ExternalEndpointModel() {
  static const Gem5ModelContractDescriptor descriptor{
      {"loom.gem5.external_endpoint", {1, 0}},
      "loom.gem5.external_endpoint.v1",
      "LoomExternalEndpoint",
      Gem5ModelObjectClass::ExternalEndpoint,
      true,
      &validateEmpty,
      nullptr,
      kExternalPorts};
  return descriptor;
}

llvm::Error registerBuiltinGem5ModelContracts() {
  const std::array<const Gem5ModelContractDescriptor *, 5> descriptors{
      &gem5RiscvTimingCpuModel(), &gem5SpatialBridgeModel(),
      &gem5SimpleMemoryModel(), &gem5SystemXBarModel(),
      &gem5ExternalEndpointModel()};
  for (const Gem5ModelContractDescriptor *descriptor : descriptors)
    if (llvm::Error error = registerGem5ModelContract(*descriptor))
      return error;
  return llvm::Error::success();
}

std::vector<std::uint8_t>
encodeGem5RiscvTimingCpuParameters(Gem5RiscvTimingCpuParameters parameters) {
  std::vector<std::uint8_t> bytes;
  bytes.reserve(16);
  appendU64(bytes, parameters.cpuId);
  appendU64(bytes, parameters.clockPeriodTicks);
  return bytes;
}

llvm::Expected<Gem5RiscvTimingCpuParameters>
decodeGem5RiscvTimingCpuParameters(llvm::ArrayRef<std::uint8_t> bytes) {
  if (bytes.size() != 16)
    return invalid("TimingSimpleCPU payload must contain two u64 fields");
  Gem5RiscvTimingCpuParameters result{readU64(bytes, 0), readU64(bytes, 8)};
  if (result.clockPeriodTicks == 0)
    return invalid("TimingSimpleCPU clock period must be positive");
  return result;
}

std::vector<std::uint8_t>
encodeGem5SpatialBridgeParameters(Gem5SpatialBridgeParameters parameters) {
  std::vector<std::uint8_t> bytes;
  bytes.reserve(32);
  appendU64(bytes, parameters.pioAddress);
  appendU64(bytes, parameters.pioSize);
  appendU64(bytes, parameters.pioLatencyTicks);
  appendU64(bytes, parameters.maximumMessageBytes);
  return bytes;
}

llvm::Expected<Gem5SpatialBridgeParameters>
decodeGem5SpatialBridgeParameters(llvm::ArrayRef<std::uint8_t> bytes) {
  if (bytes.size() != 32)
    return invalid("SpatialBridge payload must contain four u64 fields");
  Gem5SpatialBridgeParameters result{readU64(bytes, 0), readU64(bytes, 8),
                                     readU64(bytes, 16), readU64(bytes, 24)};
  if (result.pioSize < 0x28 || result.pioLatencyTicks == 0 ||
      result.maximumMessageBytes < gem5BridgeWireHeaderBytes ||
      result.maximumMessageBytes >
          static_cast<std::uint64_t>(std::numeric_limits<int>::max()) ||
      result.pioAddress >
          std::numeric_limits<std::uint64_t>::max() - result.pioSize)
    return invalid("SpatialBridge parameters are outside the supported domain");
  return result;
}

std::vector<std::uint8_t>
encodeGem5SimpleMemoryParameters(Gem5SimpleMemoryParameters parameters) {
  std::vector<std::uint8_t> bytes;
  bytes.reserve(24);
  appendU64(bytes, parameters.baseAddress);
  appendU64(bytes, parameters.sizeBytes);
  appendU64(bytes, parameters.latencyTicks);
  return bytes;
}

llvm::Expected<Gem5SimpleMemoryParameters>
decodeGem5SimpleMemoryParameters(llvm::ArrayRef<std::uint8_t> bytes) {
  if (bytes.size() != 24)
    return invalid("SimpleMemory payload must contain three u64 fields");
  Gem5SimpleMemoryParameters result{readU64(bytes, 0), readU64(bytes, 8),
                                    readU64(bytes, 16)};
  if (result.sizeBytes == 0 || result.latencyTicks == 0 ||
      result.baseAddress >
          std::numeric_limits<std::uint64_t>::max() - result.sizeBytes)
    return invalid("SimpleMemory parameters are outside the supported domain");
  return result;
}

} // namespace loom::runtime
