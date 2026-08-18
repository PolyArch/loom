#ifndef LOOM_RUNTIME_SPATIALINVOCATIONWIRE_H
#define LOOM_RUNTIME_SPATIALINVOCATIONWIRE_H

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

namespace loom::runtime {

inline constexpr char spatialInvocationAbiIdentity[] =
    "loom.spatial_invocation_abi.v2";
inline constexpr std::array<std::uint8_t, 4> spatialInvocationWireMagic{
    'L', 'G', 'I', '2'};
inline constexpr std::size_t spatialInvocationWireHeaderBytes = 76;

struct SpatialInvocationPointerTarget final {
  std::uint32_t objectOrdinal = 0;
  std::uint64_t byteOffset = 0;
};

struct SpatialInvocationValue final {
  std::uint32_t ordinal = 0;
  std::uint32_t bitCount = 0;
  std::optional<SpatialInvocationPointerTarget> pointerTarget;
  std::vector<std::uint8_t> littleEndianBits;
};

struct SpatialInvocationMemoryObject final {
  std::uint32_t ordinal = 0;
  std::uint64_t address = 0;
  std::vector<std::uint8_t> initialBytes;
};

struct SpatialInvocationMemoryRootBinding final {
  std::uint64_t logicalMemoryRootEntity = 0;
  std::uint32_t objectOrdinal = 0;
  std::uint64_t byteOffset = 0;
};

struct SpatialInvocationResultDestination final {
  std::uint32_t ordinal = 0;
  std::uint32_t bitCount = 0;
  std::uint64_t address = 0;
};

struct SpatialInvocationWire final {
  std::array<std::uint8_t, 32> canonicalDataflowIdentity{};
  std::uint64_t rootThreadLaunchEntity = 0;
  std::uint64_t graphLaunchEntity = 0;
  std::vector<std::uint64_t> denseCoordinates;
  std::vector<SpatialInvocationValue> values;
  std::vector<SpatialInvocationMemoryObject> memoryObjects;
  std::vector<SpatialInvocationMemoryRootBinding> memoryRootBindings;
  std::vector<SpatialInvocationResultDestination> results;
};

struct SpatialInvocationValueLayout final {
  std::uint32_t bitCount = 0;
  std::optional<SpatialInvocationPointerTarget> pointerTarget;
};

struct SpatialInvocationMemoryObjectLayout final {
  std::uint64_t byteCount = 0;
};

struct SpatialInvocationWireLayout final {
  std::vector<std::uint8_t> templateBytes;
  std::vector<std::size_t> valuePayloadOffsets;
  std::vector<std::optional<std::size_t>> valuePointerTargetOffsetOffsets;
  std::vector<std::size_t> memoryAddressOffsets;
  std::vector<std::size_t> memoryPayloadOffsets;
  std::vector<std::size_t> memoryRootByteOffsetOffsets;
  std::vector<std::size_t> resultAddressOffsets;
};

inline constexpr std::array<std::uint8_t, 4> spatialInvocationResultMagic{
    'L', 'G', 'X', '3'};
inline constexpr std::size_t spatialInvocationResultHeaderBytes = 68;

struct SpatialInvocationRuntimeInputSnapshot final {
  std::array<std::uint8_t, 32> identity{};
  std::vector<std::uint8_t> canonicalBytes;
};

struct SpatialInvocationResultWire final {
  std::uint64_t sessionEntryOrdinal = 0;
  std::vector<std::uint8_t> invocation;
  std::optional<SpatialInvocationRuntimeInputSnapshot> runtimeInput;
  std::vector<std::uint8_t> spatialBoundaryResult;
};

namespace spatial_invocation_detail {

inline void appendU32(std::vector<std::uint8_t> &bytes, std::uint32_t value) {
  for (unsigned byte = 0; byte != 4; ++byte)
    bytes.push_back(static_cast<std::uint8_t>(value >> (byte * 8)));
}

inline void appendU64(std::vector<std::uint8_t> &bytes, std::uint64_t value) {
  for (unsigned byte = 0; byte != 8; ++byte)
    bytes.push_back(static_cast<std::uint8_t>(value >> (byte * 8)));
}

inline std::uint32_t readU32(const std::uint8_t *bytes) {
  std::uint32_t value = 0;
  for (unsigned byte = 0; byte != 4; ++byte)
    value |= static_cast<std::uint32_t>(bytes[byte]) << (byte * 8);
  return value;
}

inline std::uint64_t readU64(const std::uint8_t *bytes) {
  std::uint64_t value = 0;
  for (unsigned byte = 0; byte != 8; ++byte)
    value |= static_cast<std::uint64_t>(bytes[byte]) << (byte * 8);
  return value;
}

inline bool encodedByteCount(std::uint32_t bitCount, std::size_t &byteCount) {
  if (bitCount == 0)
    return false;
  byteCount = (static_cast<std::size_t>(bitCount) + 7) / 8;
  return true;
}

inline bool hasCanonicalPadding(const SpatialInvocationValue &value) {
  if (value.bitCount % 8 == 0)
    return true;
  const std::uint8_t admitted =
      static_cast<std::uint8_t>((1U << (value.bitCount % 8)) - 1U);
  return (value.littleEndianBits.back() & ~admitted) == 0;
}

inline bool checkedAdd(std::size_t lhs, std::size_t rhs, std::size_t &sum) {
  if (rhs > std::numeric_limits<std::size_t>::max() - lhs)
    return false;
  sum = lhs + rhs;
  return true;
}

inline bool checkedMultiply(std::size_t lhs, std::size_t rhs,
                            std::size_t &product) {
  if (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs)
    return false;
  product = lhs * rhs;
  return true;
}

} // namespace spatial_invocation_detail

inline bool validateSpatialInvocationWire(const SpatialInvocationWire &wire,
                                          std::string &error) {
  if (wire.denseCoordinates.size() >
          std::numeric_limits<std::uint32_t>::max() ||
      wire.values.size() > std::numeric_limits<std::uint32_t>::max() ||
      wire.memoryObjects.size() > std::numeric_limits<std::uint32_t>::max() ||
      wire.memoryRootBindings.size() >
          std::numeric_limits<std::uint32_t>::max() ||
      wire.results.size() > std::numeric_limits<std::uint32_t>::max()) {
    error = "invocation collection exceeds its wire count domain";
    return false;
  }
  for (std::size_t ordinal = 0; ordinal != wire.values.size(); ++ordinal) {
    const SpatialInvocationValue &value = wire.values[ordinal];
    std::size_t byteCount = 0;
    if (value.ordinal != ordinal ||
        !spatial_invocation_detail::encodedByteCount(value.bitCount,
                                                     byteCount) ||
        value.littleEndianBits.size() != byteCount ||
        !spatial_invocation_detail::hasCanonicalPadding(value)) {
      error = "invocation value table is not dense and canonical";
      return false;
    }
    if (value.pointerTarget &&
        (value.pointerTarget->objectOrdinal >= wire.memoryObjects.size() ||
         value.pointerTarget->byteOffset >=
             wire.memoryObjects[value.pointerTarget->objectOrdinal]
                 .initialBytes.size())) {
      error = "invocation pointer target is outside its memory object";
      return false;
    }
  }
  struct AddressInterval final {
    std::size_t ordinal = 0;
    std::uint64_t begin = 0;
    std::uint64_t end = 0;
  };
  std::vector<AddressInterval> intervals;
  intervals.reserve(wire.memoryObjects.size());
  for (std::size_t ordinal = 0; ordinal != wire.memoryObjects.size();
       ++ordinal) {
    const SpatialInvocationMemoryObject &object = wire.memoryObjects[ordinal];
    if (object.ordinal != ordinal || object.address == 0 ||
        object.initialBytes.empty() ||
        object.initialBytes.size() >
            std::numeric_limits<std::uint64_t>::max() - object.address) {
      error = "invocation memory object table is not dense and finite";
      return false;
    }
    intervals.push_back(
        {ordinal, object.address, object.address + object.initialBytes.size()});
  }
  std::sort(intervals.begin(), intervals.end(),
            [](const AddressInterval &lhs, const AddressInterval &rhs) {
              return std::tie(lhs.begin, lhs.end, lhs.ordinal) <
                     std::tie(rhs.begin, rhs.end, rhs.ordinal);
            });
  for (std::size_t ordinal = 1; ordinal < intervals.size(); ++ordinal) {
    if (intervals[ordinal - 1].end > intervals[ordinal].begin) {
      const AddressInterval &prior = intervals[ordinal - 1];
      const AddressInterval &current = intervals[ordinal];
      error = "invocation memory object address ranges overlap: object " +
              std::to_string(prior.ordinal) + " [" +
              std::to_string(prior.begin) + ", " + std::to_string(prior.end) +
              ") and object " + std::to_string(current.ordinal) + " [" +
              std::to_string(current.begin) + ", " +
              std::to_string(current.end) + ")";
      return false;
    }
  }
  std::vector<bool> referencedObjects(wire.memoryObjects.size(), false);
  for (std::size_t ordinal = 0; ordinal != wire.memoryRootBindings.size();
       ++ordinal) {
    const SpatialInvocationMemoryRootBinding &binding =
        wire.memoryRootBindings[ordinal];
    if ((ordinal != 0 &&
         binding.logicalMemoryRootEntity <=
             wire.memoryRootBindings[ordinal - 1].logicalMemoryRootEntity) ||
        binding.objectOrdinal >= wire.memoryObjects.size() ||
        binding.byteOffset >=
            wire.memoryObjects[binding.objectOrdinal].initialBytes.size()) {
      error = "invocation memory-root binding table is not canonical";
      return false;
    }
    referencedObjects[binding.objectOrdinal] = true;
  }
  if (std::find(referencedObjects.begin(), referencedObjects.end(), false) !=
      referencedObjects.end()) {
    error = "invocation memory object has no logical-root binding";
    return false;
  }
  for (std::size_t ordinal = 0; ordinal != wire.results.size(); ++ordinal) {
    const SpatialInvocationResultDestination &result = wire.results[ordinal];
    std::size_t byteCount = 0;
    if (result.ordinal != ordinal ||
        !spatial_invocation_detail::encodedByteCount(result.bitCount,
                                                     byteCount)) {
      error = "invocation result table is not dense and canonical";
      return false;
    }
  }
  return true;
}

inline std::vector<std::uint8_t>
encodeSpatialInvocationWire(const SpatialInvocationWire &wire) {
  std::string error;
  if (!validateSpatialInvocationWire(wire, error))
    return {};
  std::vector<std::uint8_t> bytes;
  bytes.reserve(spatialInvocationWireHeaderBytes +
                wire.denseCoordinates.size() * 8 + wire.values.size() * 24 +
                wire.memoryObjects.size() * 24 +
                wire.memoryRootBindings.size() * 24 + wire.results.size() * 16);
  bytes.insert(bytes.end(), spatialInvocationWireMagic.begin(),
               spatialInvocationWireMagic.end());
  bytes.insert(bytes.end(), wire.canonicalDataflowIdentity.begin(),
               wire.canonicalDataflowIdentity.end());
  spatial_invocation_detail::appendU64(bytes, wire.rootThreadLaunchEntity);
  spatial_invocation_detail::appendU64(bytes, wire.graphLaunchEntity);
  spatial_invocation_detail::appendU32(
      bytes, static_cast<std::uint32_t>(wire.denseCoordinates.size()));
  spatial_invocation_detail::appendU32(
      bytes, static_cast<std::uint32_t>(wire.values.size()));
  spatial_invocation_detail::appendU32(
      bytes, static_cast<std::uint32_t>(wire.memoryObjects.size()));
  spatial_invocation_detail::appendU32(
      bytes, static_cast<std::uint32_t>(wire.memoryRootBindings.size()));
  spatial_invocation_detail::appendU32(
      bytes, static_cast<std::uint32_t>(wire.results.size()));
  spatial_invocation_detail::appendU32(bytes, 0);
  for (std::uint64_t coordinate : wire.denseCoordinates)
    spatial_invocation_detail::appendU64(bytes, coordinate);
  for (const SpatialInvocationValue &value : wire.values) {
    spatial_invocation_detail::appendU32(bytes, value.ordinal);
    spatial_invocation_detail::appendU32(bytes, value.bitCount);
    spatial_invocation_detail::appendU32(bytes, value.pointerTarget ? 1U : 0U);
    spatial_invocation_detail::appendU32(
        bytes, value.pointerTarget ? value.pointerTarget->objectOrdinal : 0U);
    spatial_invocation_detail::appendU64(
        bytes, value.pointerTarget ? value.pointerTarget->byteOffset : 0U);
    bytes.insert(bytes.end(), value.littleEndianBits.begin(),
                 value.littleEndianBits.end());
  }
  for (const SpatialInvocationMemoryObject &object : wire.memoryObjects) {
    spatial_invocation_detail::appendU32(bytes, object.ordinal);
    spatial_invocation_detail::appendU32(bytes, 0);
    spatial_invocation_detail::appendU64(bytes, object.address);
    spatial_invocation_detail::appendU64(bytes, object.initialBytes.size());
    bytes.insert(bytes.end(), object.initialBytes.begin(),
                 object.initialBytes.end());
  }
  for (const SpatialInvocationMemoryRootBinding &binding :
       wire.memoryRootBindings) {
    spatial_invocation_detail::appendU64(bytes,
                                         binding.logicalMemoryRootEntity);
    spatial_invocation_detail::appendU32(bytes, binding.objectOrdinal);
    spatial_invocation_detail::appendU32(bytes, 0);
    spatial_invocation_detail::appendU64(bytes, binding.byteOffset);
  }
  for (const SpatialInvocationResultDestination &result : wire.results) {
    spatial_invocation_detail::appendU32(bytes, result.ordinal);
    spatial_invocation_detail::appendU32(bytes, result.bitCount);
    spatial_invocation_detail::appendU64(bytes, result.address);
  }
  return bytes;
}

inline bool decodeSpatialInvocationWire(const std::vector<std::uint8_t> &bytes,
                                        SpatialInvocationWire &wire,
                                        std::string &error) {
  if (bytes.size() < spatialInvocationWireHeaderBytes) {
    error = "truncated invocation header";
    return false;
  }
  if (!std::equal(spatialInvocationWireMagic.begin(),
                  spatialInvocationWireMagic.end(), bytes.begin())) {
    error = "wrong invocation ABI magic";
    return false;
  }
  std::copy_n(bytes.begin() + 4, wire.canonicalDataflowIdentity.size(),
              wire.canonicalDataflowIdentity.begin());
  wire.rootThreadLaunchEntity =
      spatial_invocation_detail::readU64(bytes.data() + 36);
  wire.graphLaunchEntity =
      spatial_invocation_detail::readU64(bytes.data() + 44);
  const std::uint32_t coordinateCount =
      spatial_invocation_detail::readU32(bytes.data() + 52);
  const std::uint32_t valueCount =
      spatial_invocation_detail::readU32(bytes.data() + 56);
  const std::uint32_t memoryObjectCount =
      spatial_invocation_detail::readU32(bytes.data() + 60);
  const std::uint32_t memoryRootBindingCount =
      spatial_invocation_detail::readU32(bytes.data() + 64);
  const std::uint32_t resultCount =
      spatial_invocation_detail::readU32(bytes.data() + 68);
  if (spatial_invocation_detail::readU32(bytes.data() + 72) != 0) {
    error = "invocation reserved header field is nonzero";
    return false;
  }
  std::size_t cursor = spatialInvocationWireHeaderBytes;
  std::size_t coordinateTableBytes = 0;
  std::size_t coordinateBytes = 0;
  if (!spatial_invocation_detail::checkedMultiply(
          static_cast<std::size_t>(coordinateCount), sizeof(std::uint64_t),
          coordinateTableBytes) ||
      !spatial_invocation_detail::checkedAdd(cursor, coordinateTableBytes,
                                             coordinateBytes) ||
      coordinateBytes > bytes.size()) {
    error = "invocation coordinate table is truncated";
    return false;
  }
  wire.denseCoordinates.clear();
  wire.denseCoordinates.reserve(coordinateCount);
  for (std::uint32_t ordinal = 0; ordinal != coordinateCount; ++ordinal) {
    wire.denseCoordinates.push_back(
        spatial_invocation_detail::readU64(bytes.data() + cursor));
    cursor += 8;
  }
  wire.values.clear();
  wire.values.reserve(valueCount);
  for (std::uint32_t ordinal = 0; ordinal != valueCount; ++ordinal) {
    std::size_t headerEnd = 0;
    if (!spatial_invocation_detail::checkedAdd(cursor, 24, headerEnd) ||
        headerEnd > bytes.size()) {
      error = "invocation value header is truncated";
      return false;
    }
    const std::uint32_t encodedOrdinal =
        spatial_invocation_detail::readU32(bytes.data() + cursor);
    const std::uint32_t bitCount =
        spatial_invocation_detail::readU32(bytes.data() + cursor + 4);
    const std::uint32_t pointerPresent =
        spatial_invocation_detail::readU32(bytes.data() + cursor + 8);
    if (pointerPresent > 1) {
      error = "invocation pointer-target discriminant is invalid";
      return false;
    }
    std::optional<SpatialInvocationPointerTarget> pointerTarget;
    const std::uint32_t pointerObject =
        spatial_invocation_detail::readU32(bytes.data() + cursor + 12);
    const std::uint64_t pointerOffset =
        spatial_invocation_detail::readU64(bytes.data() + cursor + 16);
    if (pointerPresent)
      pointerTarget =
          SpatialInvocationPointerTarget{pointerObject, pointerOffset};
    else if (pointerObject != 0 || pointerOffset != 0) {
      error = "invocation absent pointer target has a nonzero payload";
      return false;
    }
    std::size_t byteCount = 0;
    std::size_t valueEnd = 0;
    if (!spatial_invocation_detail::encodedByteCount(bitCount, byteCount) ||
        !spatial_invocation_detail::checkedAdd(headerEnd, byteCount,
                                               valueEnd) ||
        valueEnd > bytes.size()) {
      error = "invocation value payload is truncated or has zero width";
      return false;
    }
    wire.values.push_back(
        {encodedOrdinal, bitCount, std::move(pointerTarget),
         std::vector<std::uint8_t>(bytes.begin() + headerEnd,
                                   bytes.begin() + valueEnd)});
    cursor = valueEnd;
  }
  wire.memoryObjects.clear();
  wire.memoryObjects.reserve(memoryObjectCount);
  for (std::uint32_t ordinal = 0; ordinal != memoryObjectCount; ++ordinal) {
    std::size_t headerEnd = 0;
    if (!spatial_invocation_detail::checkedAdd(cursor, 24, headerEnd) ||
        headerEnd > bytes.size()) {
      error = "invocation memory object header is truncated";
      return false;
    }
    const std::uint32_t encodedOrdinal =
        spatial_invocation_detail::readU32(bytes.data() + cursor);
    if (spatial_invocation_detail::readU32(bytes.data() + cursor + 4) != 0) {
      error = "invocation memory object reserved field is nonzero";
      return false;
    }
    const std::uint64_t address =
        spatial_invocation_detail::readU64(bytes.data() + cursor + 8);
    const std::uint64_t byteCount =
        spatial_invocation_detail::readU64(bytes.data() + cursor + 16);
    if (byteCount > std::numeric_limits<std::size_t>::max()) {
      error = "invocation memory object exceeds the host size domain";
      return false;
    }
    std::size_t objectEnd = 0;
    if (!spatial_invocation_detail::checkedAdd(
            headerEnd, static_cast<std::size_t>(byteCount), objectEnd) ||
        objectEnd > bytes.size()) {
      error = "invocation memory object payload is truncated";
      return false;
    }
    wire.memoryObjects.push_back(
        {encodedOrdinal, address,
         std::vector<std::uint8_t>(bytes.begin() + headerEnd,
                                   bytes.begin() + objectEnd)});
    cursor = objectEnd;
  }
  wire.memoryRootBindings.clear();
  wire.memoryRootBindings.reserve(memoryRootBindingCount);
  for (std::uint32_t ordinal = 0; ordinal != memoryRootBindingCount;
       ++ordinal) {
    std::size_t bindingEnd = 0;
    if (!spatial_invocation_detail::checkedAdd(cursor, 24, bindingEnd) ||
        bindingEnd > bytes.size()) {
      error = "invocation memory-root binding table is truncated";
      return false;
    }
    const std::uint64_t rootEntity =
        spatial_invocation_detail::readU64(bytes.data() + cursor);
    const std::uint32_t objectOrdinal =
        spatial_invocation_detail::readU32(bytes.data() + cursor + 8);
    if (spatial_invocation_detail::readU32(bytes.data() + cursor + 12) != 0) {
      error = "invocation memory-root binding reserved field is nonzero";
      return false;
    }
    const std::uint64_t byteOffset =
        spatial_invocation_detail::readU64(bytes.data() + cursor + 16);
    wire.memoryRootBindings.push_back({rootEntity, objectOrdinal, byteOffset});
    cursor = bindingEnd;
  }
  wire.results.clear();
  wire.results.reserve(resultCount);
  for (std::uint32_t ordinal = 0; ordinal != resultCount; ++ordinal) {
    std::size_t resultEnd = 0;
    if (!spatial_invocation_detail::checkedAdd(cursor, 16, resultEnd) ||
        resultEnd > bytes.size()) {
      error = "invocation result table is truncated";
      return false;
    }
    wire.results.push_back(
        {spatial_invocation_detail::readU32(bytes.data() + cursor),
         spatial_invocation_detail::readU32(bytes.data() + cursor + 4),
         spatial_invocation_detail::readU64(bytes.data() + cursor + 8)});
    cursor = resultEnd;
  }
  if (cursor != bytes.size()) {
    error = "invocation wire has trailing bytes";
    return false;
  }
  return validateSpatialInvocationWire(wire, error);
}

inline bool projectSpatialInvocationWireLayout(
    const std::array<std::uint8_t, 32> &canonicalDataflowIdentity,
    std::uint64_t rootThreadLaunchEntity, std::uint64_t graphLaunchEntity,
    const std::vector<std::uint64_t> &denseCoordinates,
    const std::vector<SpatialInvocationValueLayout> &valueLayouts,
    const std::vector<SpatialInvocationMemoryObjectLayout> &memoryObjects,
    const std::vector<SpatialInvocationMemoryRootBinding> &memoryRootBindings,
    const std::vector<std::uint32_t> &resultBitCounts,
    SpatialInvocationWireLayout &layout, std::string &error) {
  if (denseCoordinates.size() > std::numeric_limits<std::uint32_t>::max() ||
      valueLayouts.size() > std::numeric_limits<std::uint32_t>::max() ||
      memoryObjects.size() > std::numeric_limits<std::uint32_t>::max() ||
      memoryRootBindings.size() > std::numeric_limits<std::uint32_t>::max() ||
      resultBitCounts.size() > std::numeric_limits<std::uint32_t>::max()) {
    error = "invocation layout collection exceeds its wire count domain";
    return false;
  }
  SpatialInvocationWire wire;
  wire.canonicalDataflowIdentity = canonicalDataflowIdentity;
  wire.rootThreadLaunchEntity = rootThreadLaunchEntity;
  wire.graphLaunchEntity = graphLaunchEntity;
  wire.denseCoordinates = denseCoordinates;
  std::size_t cursor = spatialInvocationWireHeaderBytes +
                       denseCoordinates.size() * sizeof(std::uint64_t);
  layout.valuePayloadOffsets.clear();
  layout.valuePointerTargetOffsetOffsets.clear();
  for (std::size_t ordinal = 0; ordinal != valueLayouts.size(); ++ordinal) {
    std::size_t byteCount = 0;
    if (!spatial_invocation_detail::encodedByteCount(
            valueLayouts[ordinal].bitCount, byteCount)) {
      error = "invocation layout has a zero-width value";
      return false;
    }
    wire.values.push_back({static_cast<std::uint32_t>(ordinal),
                           valueLayouts[ordinal].bitCount,
                           valueLayouts[ordinal].pointerTarget,
                           std::vector<std::uint8_t>(byteCount, 0)});
    layout.valuePointerTargetOffsetOffsets.push_back(
        valueLayouts[ordinal].pointerTarget
            ? std::optional<std::size_t>(cursor + 16)
            : std::nullopt);
    cursor += 24;
    layout.valuePayloadOffsets.push_back(cursor);
    cursor += byteCount;
  }
  layout.memoryAddressOffsets.clear();
  layout.memoryPayloadOffsets.clear();
  std::uint64_t templateAddress = 1;
  for (std::size_t ordinal = 0; ordinal != memoryObjects.size(); ++ordinal) {
    if (memoryObjects[ordinal].byteCount == 0 ||
        memoryObjects[ordinal].byteCount >
            std::numeric_limits<std::size_t>::max()) {
      error = "invocation layout has a non-finite memory object";
      return false;
    }
    wire.memoryObjects.push_back(
        {static_cast<std::uint32_t>(ordinal), templateAddress,
         std::vector<std::uint8_t>(
             static_cast<std::size_t>(memoryObjects[ordinal].byteCount), 0)});
    if (memoryObjects[ordinal].byteCount >
        std::numeric_limits<std::uint64_t>::max() - templateAddress - 1) {
      error = "invocation layout memory address domain is exhausted";
      return false;
    }
    templateAddress += memoryObjects[ordinal].byteCount + 1;
    layout.memoryAddressOffsets.push_back(cursor + 8);
    cursor += 24;
    layout.memoryPayloadOffsets.push_back(cursor);
    cursor += static_cast<std::size_t>(memoryObjects[ordinal].byteCount);
  }
  wire.memoryRootBindings = memoryRootBindings;
  layout.memoryRootByteOffsetOffsets.clear();
  layout.memoryRootByteOffsetOffsets.reserve(memoryRootBindings.size());
  for (std::size_t ordinal = 0; ordinal != memoryRootBindings.size();
       ++ordinal) {
    layout.memoryRootByteOffsetOffsets.push_back(cursor + 16);
    cursor += 24;
  }
  layout.resultAddressOffsets.clear();
  for (std::size_t ordinal = 0; ordinal != resultBitCounts.size(); ++ordinal) {
    wire.results.push_back(
        {static_cast<std::uint32_t>(ordinal), resultBitCounts[ordinal], 0});
    layout.resultAddressOffsets.push_back(cursor + 8);
    cursor += 16;
  }
  layout.templateBytes = encodeSpatialInvocationWire(wire);
  if (layout.templateBytes.empty()) {
    error = "invocation layout could not be encoded";
    return false;
  }
  return true;
}

inline std::vector<std::uint8_t>
encodeSpatialInvocationResultWire(const SpatialInvocationResultWire &result) {
  if (result.runtimeInput && result.runtimeInput->canonicalBytes.empty())
    return {};
  std::vector<std::uint8_t> bytes;
  bytes.reserve(
      spatialInvocationResultHeaderBytes + result.invocation.size() +
      (result.runtimeInput ? result.runtimeInput->canonicalBytes.size() : 0) +
      result.spatialBoundaryResult.size());
  bytes.insert(bytes.end(), spatialInvocationResultMagic.begin(),
               spatialInvocationResultMagic.end());
  spatial_invocation_detail::appendU64(bytes, result.sessionEntryOrdinal);
  spatial_invocation_detail::appendU64(bytes, result.invocation.size());
  spatial_invocation_detail::appendU64(
      bytes,
      result.runtimeInput ? result.runtimeInput->canonicalBytes.size() : 0);
  spatial_invocation_detail::appendU64(bytes,
                                       result.spatialBoundaryResult.size());
  if (result.runtimeInput)
    bytes.insert(bytes.end(), result.runtimeInput->identity.begin(),
                 result.runtimeInput->identity.end());
  else
    bytes.insert(bytes.end(), 32, 0);
  bytes.insert(bytes.end(), result.invocation.begin(), result.invocation.end());
  if (result.runtimeInput)
    bytes.insert(bytes.end(), result.runtimeInput->canonicalBytes.begin(),
                 result.runtimeInput->canonicalBytes.end());
  bytes.insert(bytes.end(), result.spatialBoundaryResult.begin(),
               result.spatialBoundaryResult.end());
  return bytes;
}

inline bool
decodeSpatialInvocationResultWire(const std::vector<std::uint8_t> &bytes,
                                  SpatialInvocationResultWire &result,
                                  std::string &error) {
  if (bytes.size() < spatialInvocationResultHeaderBytes ||
      !std::equal(spatialInvocationResultMagic.begin(),
                  spatialInvocationResultMagic.end(), bytes.begin())) {
    error = "wrong or truncated invocation result header";
    return false;
  }
  const std::uint64_t sessionEntryOrdinal =
      spatial_invocation_detail::readU64(bytes.data() + 4);
  const std::uint64_t invocationSize =
      spatial_invocation_detail::readU64(bytes.data() + 12);
  const std::uint64_t runtimeInputSize =
      spatial_invocation_detail::readU64(bytes.data() + 20);
  const std::uint64_t boundarySize =
      spatial_invocation_detail::readU64(bytes.data() + 28);
  if (invocationSize > std::numeric_limits<std::size_t>::max() ||
      runtimeInputSize > std::numeric_limits<std::size_t>::max() ||
      boundarySize > std::numeric_limits<std::size_t>::max()) {
    error = "invocation result length exceeds the host size domain";
    return false;
  }
  std::size_t payloadSize = 0;
  if (!spatial_invocation_detail::checkedAdd(
          static_cast<std::size_t>(invocationSize),
          static_cast<std::size_t>(runtimeInputSize), payloadSize) ||
      !spatial_invocation_detail::checkedAdd(
          payloadSize, static_cast<std::size_t>(boundarySize), payloadSize) ||
      payloadSize != bytes.size() - spatialInvocationResultHeaderBytes) {
    error = "invocation result lengths do not match the envelope";
    return false;
  }
  const auto identityBegin = bytes.begin() + 36;
  const bool zeroIdentity =
      std::all_of(identityBegin, identityBegin + 32,
                  [](std::uint8_t byte) { return byte == 0; });
  if ((runtimeInputSize == 0) != zeroIdentity) {
    error = "invocation result runtime identity is not canonical";
    return false;
  }
  const auto invocationEnd = bytes.begin() +
                             spatialInvocationResultHeaderBytes +
                             static_cast<std::size_t>(invocationSize);
  result.sessionEntryOrdinal = sessionEntryOrdinal;
  result.invocation.assign(bytes.begin() + spatialInvocationResultHeaderBytes,
                           invocationEnd);
  const auto runtimeInputEnd =
      invocationEnd + static_cast<std::size_t>(runtimeInputSize);
  result.runtimeInput.reset();
  if (runtimeInputSize != 0) {
    SpatialInvocationRuntimeInputSnapshot snapshot;
    std::copy(identityBegin, identityBegin + 32, snapshot.identity.begin());
    snapshot.canonicalBytes.assign(invocationEnd, runtimeInputEnd);
    result.runtimeInput.emplace(std::move(snapshot));
  }
  result.spatialBoundaryResult.assign(runtimeInputEnd, bytes.end());
  return true;
}

} // namespace loom::runtime

#endif // LOOM_RUNTIME_SPATIALINVOCATIONWIRE_H
