#ifndef LOOM_SIMULATOR_SPATIALINVOCATION_H
#define LOOM_SIMULATOR_SPATIALINVOCATION_H

#include "Runtime/SpatialInvocationWire.h"
#include "Simulator/SimulationArtifacts.h"

#include "llvm/ADT/APInt.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <cstdint>
#include <vector>

namespace loom::sim {

struct SpatialInvocationMemoryWrite final {
  std::uint64_t address = 0;
  std::vector<std::uint8_t> bytes;
};

/// The invocation wire's one little-endian value-payload convention. The host
/// projection that bakes a payload and the engine that decodes it share this
/// mapping; neither reimplements it.
inline std::vector<std::uint8_t>
packSpatialInvocationValueBits(const llvm::APInt &bits) {
  const std::size_t byteCount = (bits.getBitWidth() + 7) / 8;
  std::vector<std::uint8_t> bytes;
  bytes.reserve(byteCount);
  for (std::size_t byte = 0; byte != byteCount; ++byte) {
    const unsigned offset = static_cast<unsigned>(byte * 8);
    const unsigned width = std::min<unsigned>(8, bits.getBitWidth() - offset);
    bytes.push_back(
        static_cast<std::uint8_t>(bits.extractBitsAsZExtValue(width, offset)));
  }
  return bytes;
}

inline llvm::APInt
unpackSpatialInvocationValueBits(const runtime::SpatialInvocationValue &value) {
  llvm::APInt bits(value.bitCount, 0);
  for (std::size_t byte = 0; byte != value.littleEndianBits.size(); ++byte) {
    const std::uint64_t base = byte * 8;
    for (unsigned bit = 0; bit != 8 && base + bit < value.bitCount; ++bit)
      if ((value.littleEndianBits[byte] & (1U << bit)) != 0)
        bits.setBit(static_cast<unsigned>(base + bit));
  }
  return bits;
}

/// Derives the sorted logical roots whose canonical actors can modify memory
/// during one exact rooted graph invocation. Dataflow actor semantics remain
/// the owner; invocation producers and consumers share this projection.
llvm::Expected<std::vector<dataflow::LogicalMemoryRootRef>>
projectSpatialInvocationWritableMemoryRoots(
    const dataflow::CanonicalDataflowProgramView &dataflow,
    dataflow::RootedGraphLaunchRef launch);

/// Rebuilds the transient runtime input named by one guest invocation. The
/// wire passes memory objects by reference; `memorySnapshot` is the Bridge's
/// untimed capture of those objects' guest bytes, concatenated in object
/// ordinal order.
llvm::Expected<CanonicalSimulationRuntimeInput>
materializeSpatialInvocationRuntimeInput(
    const ImportedSpatialSimulationWorkload &workload,
    const runtime::SpatialInvocationWire &wire,
    const std::vector<std::uint8_t> &memorySnapshot);

/// Proves that an effective runtime input retains the exact value, memory,
/// and result-destination semantics carried by the guest invocation. Stream
/// inputs may differ because System channel binding supplies them after the
/// guest launch has been decoded.
llvm::Error validateEffectiveSpatialInvocationRuntimeInput(
    const ImportedSpatialSimulationWorkload &workload,
    const runtime::SpatialInvocationWire &wire,
    const std::vector<std::uint8_t> &memorySnapshot,
    const CanonicalSimulationRuntimeInput &runtimeInput);

llvm::Expected<ImportedSpatialSimulationInputs>
materializeSpatialInvocationInputs(
    ImportedSpatialSimulationWorkload workload,
    const runtime::SpatialInvocationWire &wire,
    const std::vector<std::uint8_t> &memorySnapshot);

llvm::Expected<std::vector<SpatialInvocationMemoryWrite>>
projectSpatialInvocationResultWrites(
    const runtime::SpatialInvocationWire &wire,
    const ImportedSpatialSimulationInputs &inputs,
    const SpatialFunctionalObservations &observations);

llvm::Expected<std::vector<SpatialInvocationMemoryWrite>>
projectSpatialInvocationResultWrites(
    const runtime::SpatialInvocationWire &wire,
    const ImportedSpatialSimulationWorkload &workload,
    const SpatialFunctionalObservations &observations);

} // namespace loom::sim

#endif // LOOM_SIMULATOR_SPATIALINVOCATION_H
