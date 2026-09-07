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

std::uint64_t readU64(llvm::ArrayRef<std::uint8_t> bytes, std::size_t offset) {
  std::uint64_t value = 0;
  for (unsigned index = 0; index < 8; ++index)
    value = (value << 8) | bytes[offset + index];
  return value;
}

constexpr std::size_t kCacheParameterFields = 5;
constexpr std::size_t kCacheParameterBytes = kCacheParameterFields * 8;

void appendCache(std::vector<std::uint8_t> &bytes,
                 Gem5CacheParameters cache) {
  appendU64(bytes, cache.capacityBytes);
  appendU64(bytes, cache.lineBytes);
  appendU64(bytes, cache.associativity);
  appendU64(bytes, cache.hitLatencyCycles);
  appendU64(bytes, cache.missStatusEntries);
}

Gem5CacheParameters readCache(llvm::ArrayRef<std::uint8_t> bytes,
                              std::size_t offset) {
  return Gem5CacheParameters{
      readU64(bytes, offset), readU64(bytes, offset + 8),
      readU64(bytes, offset + 16), readU64(bytes, offset + 24),
      readU64(bytes, offset + 32)};
}

/// The gem5 payload repeats the Fabric geometry, so it repeats the Fabric
/// admission rule instead of trusting the producer.
llvm::Error validateCache(Gem5CacheParameters cache) {
  if (cache.capacityBytes == 0 || cache.lineBytes == 0 ||
      cache.associativity == 0 || cache.hitLatencyCycles == 0 ||
      cache.missStatusEntries == 0)
    return invalid("cache parameters must be positive");
  if ((cache.lineBytes & (cache.lineBytes - 1)) != 0)
    return invalid("cache line size must be a power of two");
  if (cache.capacityBytes % (cache.lineBytes * cache.associativity) != 0)
    return invalid("cache capacity must be a multiple of line size times "
                   "associativity");
  return llvm::Error::success();
}

llvm::Error requireFabricCache(Gem5CacheParameters projected,
                               const fabric::CacheRealizationRecord &declared,
                               llvm::StringRef role) {
  if (projected != projectGem5Cache(declared))
    return invalid(role + " cache parameters differ from the Fabric "
                          "realization");
  return llvm::Error::success();
}

llvm::Error validateEmpty(llvm::ArrayRef<std::uint8_t> bytes) {
  return bytes.empty() ? llvm::Error::success()
                       : invalid("payload must be empty");
}

llvm::Error validateCpu(llvm::ArrayRef<std::uint8_t> bytes) {
  auto parameters = decodeGem5RiscvCpuParameters(bytes);
  return parameters ? llvm::Error::success() : parameters.takeError();
}

llvm::Error validateBridge(llvm::ArrayRef<std::uint8_t> bytes) {
  auto parameters = decodeGem5SpatialBridgeParameters(bytes);
  return parameters ? llvm::Error::success() : parameters.takeError();
}

llvm::Error validateSpatialBridgeCompatibility(
    llvm::ArrayRef<std::uint8_t> payload,
    const fabric::SpatialMemoryAccessRealization &spatialMemoryAccess) {
  auto parameters = decodeGem5SpatialBridgeParameters(payload);
  if (!parameters)
    return parameters.takeError();
  return requireFabricCache(parameters->cache, spatialMemoryAccess.cache(),
                            "SpatialBridge");
}

llvm::Error validateMemory(llvm::ArrayRef<std::uint8_t> bytes) {
  auto parameters = decodeGem5SimpleMemoryParameters(bytes);
  return parameters ? llvm::Error::success() : parameters.takeError();
}

llvm::Error validateRiscvMachineCompatibility(
    llvm::ArrayRef<std::uint8_t> payload,
    const fabric::InstructionCoreArchitecturalContract &architecture,
    const fabric::InstructionCoreMicroarchitecturalRealization
        &microarchitecture) {
  if (architecture.xlen() != fabric::RiscVXLen::X64 ||
      architecture.endianness() != fabric::InstructionEndianness::Little ||
      !llvm::is_contained(architecture.privilegeModes(),
                          fabric::PrivilegeMode::Machine) ||
      microarchitecture.hardwareThreadCount() == 0)
    return invalid(
        "processor requires little-endian RV64 machine hardware threads");
  auto parameters = decodeGem5RiscvCpuParameters(payload);
  if (!parameters)
    return parameters.takeError();
  const fabric::PrivateCacheRealization &caches =
      microarchitecture.privateCaches();
  if (llvm::Error error = requireFabricCache(parameters->instructionCache,
                                             caches.instruction, "instruction"))
    return error;
  if (llvm::Error error =
          requireFabricCache(parameters->dataCache, caches.data, "data"))
    return error;
  if (parameters->instructionCache.lineBytes != parameters->dataCache.lineBytes)
    return invalid("processor caches disagree on the System line size");
  return llvm::Error::success();
}

llvm::Error validateTimingCpuCompatibility(
    llvm::ArrayRef<std::uint8_t> payload,
    const fabric::InstructionCoreArchitecturalContract &architecture,
    const fabric::InstructionCoreMicroarchitecturalRealization
        &microarchitecture) {
  if (llvm::Error error = validateRiscvMachineCompatibility(
          payload, architecture, microarchitecture))
    return error;
  if (!microarchitecture.inOrder() ||
      microarchitecture.hardwareThreadCount() != 1)
    return invalid("TimingSimpleCPU requires one in-order hardware thread");
  return llvm::Error::success();
}

llvm::Error validateO3CpuCompatibility(
    llvm::ArrayRef<std::uint8_t> payload,
    const fabric::InstructionCoreArchitecturalContract &architecture,
    const fabric::InstructionCoreMicroarchitecturalRealization
        &microarchitecture) {
  if (llvm::Error error = validateRiscvMachineCompatibility(
          payload, architecture, microarchitecture))
    return error;
  if (!microarchitecture.outOfOrder() ||
      microarchitecture.hardwareThreadCount() != 1)
    return invalid("O3CPU requires one out-of-order hardware thread");
  constexpr std::uint32_t architecturalRegisterCount = 32;
  const auto &pipeline = *microarchitecture.outOfOrder();
  if (pipeline.physicalIntegerRegisters <= architecturalRegisterCount ||
      pipeline.physicalFloatRegisters <= architecturalRegisterCount ||
      pipeline.physicalVectorRegisters <= architecturalRegisterCount)
    return invalid("O3CPU requires physical register files larger than the "
                   "architectural register files");
  for (const fabric::ExecutionUnitRecord &unit :
       microarchitecture.executionUnits()) {
    auto operationClasses = projectGem5O3OperationClasses(unit.operationClass);
    if (!operationClasses)
      return operationClasses.takeError();
    if (unit.initiationInterval != 1 &&
        unit.initiationInterval != unit.latencyCycles)
      return invalid("O3CPU cannot represent the execution-unit initiation "
                     "interval");
  }
  return llvm::Error::success();
}

const Gem5ModelPortKindDescriptor kBridgePorts[] = {
    {0, "spatial_boundary", Gem5ModelPortClass::SpatialBoundary, true,
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

llvm::Expected<llvm::ArrayRef<llvm::StringLiteral>>
projectGem5O3OperationClasses(
    fabric::InstructionOperationClass operationClass) {
  static constexpr llvm::StringLiteral integerAlu[] = {"IntAlu"};
  static constexpr llvm::StringLiteral integerMultiply[] = {"IntMult"};
  static constexpr llvm::StringLiteral integerDivide[] = {"IntDiv"};
  static constexpr llvm::StringLiteral loadStore[] = {
      "MemRead", "MemWrite", "FloatMemRead", "FloatMemWrite"};
  static constexpr llvm::StringLiteral floatingPointAlu[] = {
      "FloatAdd", "FloatCmp", "FloatCvt", "FloatMisc", "Bf16Cvt"};
  static constexpr llvm::StringLiteral floatingPointMultiply[] = {
      "FloatMult", "FloatMultAcc"};
  static constexpr llvm::StringLiteral floatingPointDivide[] = {
      "FloatDiv", "FloatSqrt"};

  switch (operationClass) {
  case fabric::InstructionOperationClass::IntegerAlu:
    return integerAlu;
  case fabric::InstructionOperationClass::IntegerMultiply:
    return integerMultiply;
  case fabric::InstructionOperationClass::IntegerDivide:
    return integerDivide;
  case fabric::InstructionOperationClass::LoadStore:
    return loadStore;
  case fabric::InstructionOperationClass::FloatingPointAlu:
    return floatingPointAlu;
  case fabric::InstructionOperationClass::FloatingPointMultiply:
    return floatingPointMultiply;
  case fabric::InstructionOperationClass::FloatingPointDivide:
    return floatingPointDivide;
  case fabric::InstructionOperationClass::Branch:
  case fabric::InstructionOperationClass::VectorAlu:
  case fabric::InstructionOperationClass::VectorMultiply:
  case fabric::InstructionOperationClass::System:
    return invalid("O3CPU does not model one Fabric execution-unit class");
  }
  llvm_unreachable("unknown instruction operation class");
}

const Gem5ModelContractDescriptor &gem5RiscvTimingCpuModel() {
  static const Gem5ModelContractDescriptor descriptor{
      {"loom.gem5.riscv_timing_cpu", {1, 1}},
      "loom.gem5.riscv_timing_cpu.v1.1",
      "RiscvTimingSimpleCPU",
      Gem5ModelObjectClass::Processor,
      false,
      &validateCpu,
      &validateTimingCpuCompatibility,
      nullptr,
      {}};
  return descriptor;
}

const Gem5ModelContractDescriptor &gem5RiscvO3CpuModel() {
  static const Gem5ModelContractDescriptor descriptor{
      {"loom.gem5.riscv_o3_cpu", {1, 2}},
      "loom.gem5.riscv_o3_cpu.v1.2",
      "RiscvO3CPU",
      Gem5ModelObjectClass::Processor,
      false,
      &validateCpu,
      &validateO3CpuCompatibility,
      nullptr,
      {}};
  return descriptor;
}

const Gem5ModelContractDescriptor &gem5SpatialBridgeModel() {
  static const Gem5ModelContractDescriptor descriptor{
      {"loom.gem5.spatial_bridge", {1, 1}},
      "loom.gem5.spatial_bridge.v1.1",
      "LoomSpatialBridge",
      Gem5ModelObjectClass::SpatialBridge,
      true,
      &validateBridge,
      nullptr,
      &validateSpatialBridgeCompatibility,
      kBridgePorts};
  return descriptor;
}

const Gem5ModelContractDescriptor &gem5SimpleMemoryModel() {
  static const Gem5ModelContractDescriptor descriptor{
      {"loom.gem5.simple_memory", {2, 0}},
      "loom.gem5.simple_memory.v2",
      "SimpleMemory",
      Gem5ModelObjectClass::MemoryOrService,
      true,
      &validateMemory,
      nullptr,
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
      nullptr,
      kExternalPorts};
  return descriptor;
}

llvm::Error registerBuiltinGem5ModelContracts() {
  const std::array<const Gem5ModelContractDescriptor *, 6> descriptors{
      &gem5RiscvTimingCpuModel(), &gem5RiscvO3CpuModel(),
      &gem5SpatialBridgeModel(),  &gem5SimpleMemoryModel(),
      &gem5SystemXBarModel(),     &gem5ExternalEndpointModel()};
  for (const Gem5ModelContractDescriptor *descriptor : descriptors)
    if (llvm::Error error = registerGem5ModelContract(*descriptor))
      return error;
  return llvm::Error::success();
}

Gem5CacheParameters
projectGem5Cache(const fabric::CacheRealizationRecord &cache) {
  return Gem5CacheParameters{cache.capacityBytes(), cache.lineBytes(),
                             cache.associativity(), cache.hitLatencyCycles(),
                             cache.missStatusEntries()};
}

std::vector<std::uint8_t>
encodeGem5RiscvCpuParameters(Gem5RiscvCpuParameters parameters) {
  std::vector<std::uint8_t> bytes;
  bytes.reserve(16 + 2 * kCacheParameterBytes);
  appendU64(bytes, parameters.cpuId);
  appendU64(bytes, parameters.clockPeriodTicks);
  appendCache(bytes, parameters.instructionCache);
  appendCache(bytes, parameters.dataCache);
  return bytes;
}

llvm::Expected<Gem5RiscvCpuParameters>
decodeGem5RiscvCpuParameters(llvm::ArrayRef<std::uint8_t> bytes) {
  if (bytes.size() != 16 + 2 * kCacheParameterBytes)
    return invalid("RISC-V CPU payload must contain two u64 fields and two "
                   "cache records");
  Gem5RiscvCpuParameters result{readU64(bytes, 0), readU64(bytes, 8),
                                readCache(bytes, 16),
                                readCache(bytes, 16 + kCacheParameterBytes)};
  if (result.clockPeriodTicks == 0)
    return invalid("RISC-V CPU clock period must be positive");
  if (llvm::Error error = validateCache(result.instructionCache))
    return std::move(error);
  if (llvm::Error error = validateCache(result.dataCache))
    return std::move(error);
  return result;
}

std::vector<std::uint8_t>
encodeGem5SpatialBridgeParameters(Gem5SpatialBridgeParameters parameters) {
  std::vector<std::uint8_t> bytes;
  bytes.reserve(32 + kCacheParameterBytes);
  appendU64(bytes, parameters.pioAddress);
  appendU64(bytes, parameters.pioSize);
  appendU64(bytes, parameters.pioLatencyTicks);
  appendU64(bytes, parameters.maximumMessageBytes);
  appendCache(bytes, parameters.cache);
  return bytes;
}

llvm::Expected<Gem5SpatialBridgeParameters>
decodeGem5SpatialBridgeParameters(llvm::ArrayRef<std::uint8_t> bytes) {
  if (bytes.size() != 32 + kCacheParameterBytes)
    return invalid("SpatialBridge payload must contain four u64 fields and "
                   "one cache record");
  Gem5SpatialBridgeParameters result{readU64(bytes, 0), readU64(bytes, 8),
                                     readU64(bytes, 16), readU64(bytes, 24),
                                     readCache(bytes, 32)};
  if (llvm::Error error = validateCache(result.cache))
    return std::move(error);
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

llvm::Expected<std::optional<Gem5SimpleMemoryParameters>>
projectGem5SharedMemory(const Gem5SimulationBinding &binding) {
  std::optional<Gem5SimpleMemoryParameters> memory;
  for (const auto &row : binding.correspondences()) {
    const auto *service = std::get_if<Gem5MemoryOrServiceCorrespondence>(&row);
    if (!service)
      continue;
    if (service->simObject.contract != gem5ModelContractDescriptorRef(gem5SimpleMemoryModel()))
      return std::nullopt;
    auto parameters = decodeGem5SimpleMemoryParameters(service->simObject.payload);
    if (!parameters)
      return parameters.takeError();
    if (memory && !(*memory == *parameters))
      return std::nullopt;
    memory = *parameters;
  }
  return memory;
}

} // namespace loom::runtime
