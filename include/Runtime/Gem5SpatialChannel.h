#ifndef LOOM_RUNTIME_GEM5SPATIALCHANNEL_H
#define LOOM_RUNTIME_GEM5SPATIALCHANNEL_H

#include "Common/Artifact.h"
#include "Runtime/Gem5SpatialChannelPlan.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace loom::runtime {

struct Gem5SpatialChannelInput final {
  ArtifactRootReference producerWorkload;
  ArtifactRootReference producerRuntimeInput;
  std::uint64_t producerStreamOutputOrdinal = 0;
  std::uint64_t consumerStreamInputOrdinal = 0;
  std::uint64_t address = 0;
  std::uint64_t capacityBytes = 0;
};

struct Gem5SpatialChannelOutput final {
  std::uint64_t producerStreamOutputOrdinal = 0;
  std::uint64_t address = 0;
  std::uint64_t capacityBytes = 0;
};

struct Gem5SpatialChannelProjection final {
  std::vector<Gem5SpatialChannelInput> inputs;
  std::vector<Gem5SpatialChannelOutput> outputs;
};

llvm::Expected<std::vector<std::uint8_t>>
encodeGem5SpatialChannelProjection(Gem5SpatialChannelProjection projection);

llvm::Expected<Gem5SpatialChannelProjection>
decodeGem5SpatialChannelProjection(llvm::ArrayRef<std::uint8_t> bytes);

std::array<std::uint8_t, gem5SpatialChannelBufferHeaderBytes>
encodeGem5SpatialChannelBufferHeader(std::uint64_t payloadBytes);

llvm::Expected<std::uint64_t>
decodeGem5SpatialChannelBufferHeader(llvm::ArrayRef<std::uint8_t> bytes);

} // namespace loom::runtime

#endif // LOOM_RUNTIME_GEM5SPATIALCHANNEL_H
