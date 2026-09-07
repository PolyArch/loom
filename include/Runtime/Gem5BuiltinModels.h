#ifndef LOOM_RUNTIME_GEM5BUILTINMODELS_H
#define LOOM_RUNTIME_GEM5BUILTINMODELS_H

#include "Runtime/Gem5SimulationBinding.h"

#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>
#include <vector>

namespace loom::runtime {

inline constexpr int gem5TickSecondsExponent = -12;

struct Gem5RiscvCpuParameters final {
  std::uint64_t cpuId = 0;
  std::uint64_t clockPeriodTicks = 0;

  friend bool operator==(Gem5RiscvCpuParameters lhs,
                         Gem5RiscvCpuParameters rhs) {
    return lhs.cpuId == rhs.cpuId &&
           lhs.clockPeriodTicks == rhs.clockPeriodTicks;
  }
};

struct Gem5SpatialBridgeParameters final {
  std::uint64_t pioAddress = 0;
  std::uint64_t pioSize = 0;
  std::uint64_t pioLatencyTicks = 0;
  std::uint64_t maximumMessageBytes = 0;

  friend bool operator==(Gem5SpatialBridgeParameters lhs,
                         Gem5SpatialBridgeParameters rhs) {
    return lhs.pioAddress == rhs.pioAddress && lhs.pioSize == rhs.pioSize &&
           lhs.pioLatencyTicks == rhs.pioLatencyTicks &&
           lhs.maximumMessageBytes == rhs.maximumMessageBytes;
  }
};

/// Native 12.8 GiB/s SimpleMemory service cost after gem5 rounds to integer
/// ticks at 1 ps per tick. Every accepted request occupies size * cost ticks.
inline constexpr std::uint64_t gem5SimpleMemoryServiceTicksPerByte = 73;

struct Gem5SimpleMemoryParameters final {
  std::uint64_t baseAddress = 0;
  std::uint64_t sizeBytes = 0;
  std::uint64_t latencyTicks = 0;

  friend bool operator==(Gem5SimpleMemoryParameters lhs,
                         Gem5SimpleMemoryParameters rhs) {
    return lhs.baseAddress == rhs.baseAddress &&
           lhs.sizeBytes == rhs.sizeBytes &&
           lhs.latencyTicks == rhs.latencyTicks;
  }
};

/// One shared physical SimpleMemory for all memory/service correspondences.
/// Missing, unsupported, or distinct memory objects do not establish this domain.
llvm::Expected<std::optional<Gem5SimpleMemoryParameters>>
projectGem5SharedMemory(const Gem5SimulationBinding &binding);

const Gem5ModelContractDescriptor &gem5RiscvTimingCpuModel();
const Gem5ModelContractDescriptor &gem5RiscvO3CpuModel();
const Gem5ModelContractDescriptor &gem5SpatialBridgeModel();
const Gem5ModelContractDescriptor &gem5SimpleMemoryModel();
const Gem5ModelContractDescriptor &gem5SystemXBarModel();
const Gem5ModelContractDescriptor &gem5ExternalEndpointModel();

llvm::Expected<llvm::ArrayRef<llvm::StringLiteral>>
projectGem5O3OperationClasses(fabric::InstructionOperationClass operationClass);

llvm::Error registerBuiltinGem5ModelContracts();

std::vector<std::uint8_t>
encodeGem5RiscvCpuParameters(Gem5RiscvCpuParameters parameters);
llvm::Expected<Gem5RiscvCpuParameters>
decodeGem5RiscvCpuParameters(llvm::ArrayRef<std::uint8_t> bytes);

std::vector<std::uint8_t>
encodeGem5SpatialBridgeParameters(Gem5SpatialBridgeParameters parameters);
llvm::Expected<Gem5SpatialBridgeParameters>
decodeGem5SpatialBridgeParameters(llvm::ArrayRef<std::uint8_t> bytes);

std::vector<std::uint8_t>
encodeGem5SimpleMemoryParameters(Gem5SimpleMemoryParameters parameters);
llvm::Expected<Gem5SimpleMemoryParameters>
decodeGem5SimpleMemoryParameters(llvm::ArrayRef<std::uint8_t> bytes);

} // namespace loom::runtime

#endif // LOOM_RUNTIME_GEM5BUILTINMODELS_H
