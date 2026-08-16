#ifndef LOOM_RUNTIME_SPATIALINVOCATIONWIRE_H
#define LOOM_RUNTIME_SPATIALINVOCATIONWIRE_H

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace loom::runtime {

inline constexpr char spatialInvocationAbiIdentity[] =
    "loom.spatial_invocation_abi.v1";
inline constexpr std::array<std::uint8_t, 4> spatialInvocationWireMagic{
    'L', 'G', 'I', '1'};
inline constexpr std::size_t spatialInvocationWireHeaderBytes = 68;

struct SpatialInvocationValue final {
  std::uint32_t ordinal = 0;
  std::uint32_t bitCount = 0;
  std::vector<std::uint8_t> littleEndianBits;
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
  std::vector<SpatialInvocationResultDestination> results;
};

struct SpatialInvocationWireLayout final {
  std::vector<std::uint8_t> templateBytes;
  std::vector<std::size_t> valuePayloadOffsets;
  std::vector<std::size_t> resultAddressOffsets;
};

inline constexpr std::array<std::uint8_t, 4> spatialInvocationResultMagic{
    'L', 'G', 'X', '1'};
inline constexpr std::size_t spatialInvocationResultHeaderBytes = 20;

struct SpatialInvocationResultWire final {
  std::vector<std::uint8_t> invocation;
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
                wire.denseCoordinates.size() * 8 + wire.values.size() * 8 +
                wire.results.size() * 16);
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
      bytes, static_cast<std::uint32_t>(wire.results.size()));
  spatial_invocation_detail::appendU32(bytes, 0);
  for (std::uint64_t coordinate : wire.denseCoordinates)
    spatial_invocation_detail::appendU64(bytes, coordinate);
  for (const SpatialInvocationValue &value : wire.values) {
    spatial_invocation_detail::appendU32(bytes, value.ordinal);
    spatial_invocation_detail::appendU32(bytes, value.bitCount);
    bytes.insert(bytes.end(), value.littleEndianBits.begin(),
                 value.littleEndianBits.end());
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
  const std::uint32_t resultCount =
      spatial_invocation_detail::readU32(bytes.data() + 60);
  if (spatial_invocation_detail::readU32(bytes.data() + 64) != 0) {
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
    if (!spatial_invocation_detail::checkedAdd(cursor, 8, headerEnd) ||
        headerEnd > bytes.size()) {
      error = "invocation value header is truncated";
      return false;
    }
    const std::uint32_t encodedOrdinal =
        spatial_invocation_detail::readU32(bytes.data() + cursor);
    const std::uint32_t bitCount =
        spatial_invocation_detail::readU32(bytes.data() + cursor + 4);
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
        {encodedOrdinal, bitCount,
         std::vector<std::uint8_t>(bytes.begin() + headerEnd,
                                   bytes.begin() + valueEnd)});
    cursor = valueEnd;
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
    const std::vector<std::uint32_t> &valueBitCounts,
    const std::vector<std::uint32_t> &resultBitCounts,
    SpatialInvocationWireLayout &layout, std::string &error) {
  if (denseCoordinates.size() > std::numeric_limits<std::uint32_t>::max() ||
      valueBitCounts.size() > std::numeric_limits<std::uint32_t>::max() ||
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
  for (std::size_t ordinal = 0; ordinal != valueBitCounts.size(); ++ordinal) {
    std::size_t byteCount = 0;
    if (!spatial_invocation_detail::encodedByteCount(valueBitCounts[ordinal],
                                                     byteCount)) {
      error = "invocation layout has a zero-width value";
      return false;
    }
    wire.values.push_back({static_cast<std::uint32_t>(ordinal),
                           valueBitCounts[ordinal],
                           std::vector<std::uint8_t>(byteCount, 0)});
    cursor += 8;
    layout.valuePayloadOffsets.push_back(cursor);
    cursor += byteCount;
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
  std::vector<std::uint8_t> bytes;
  bytes.reserve(spatialInvocationResultHeaderBytes + result.invocation.size() +
                result.spatialBoundaryResult.size());
  bytes.insert(bytes.end(), spatialInvocationResultMagic.begin(),
               spatialInvocationResultMagic.end());
  spatial_invocation_detail::appendU64(bytes, result.invocation.size());
  spatial_invocation_detail::appendU64(bytes,
                                       result.spatialBoundaryResult.size());
  bytes.insert(bytes.end(), result.invocation.begin(), result.invocation.end());
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
  const std::uint64_t invocationSize =
      spatial_invocation_detail::readU64(bytes.data() + 4);
  const std::uint64_t boundarySize =
      spatial_invocation_detail::readU64(bytes.data() + 12);
  if (invocationSize > std::numeric_limits<std::size_t>::max() ||
      boundarySize > std::numeric_limits<std::size_t>::max()) {
    error = "invocation result length exceeds the host size domain";
    return false;
  }
  std::size_t payloadSize = 0;
  if (!spatial_invocation_detail::checkedAdd(
          static_cast<std::size_t>(invocationSize),
          static_cast<std::size_t>(boundarySize), payloadSize) ||
      payloadSize != bytes.size() - spatialInvocationResultHeaderBytes) {
    error = "invocation result lengths do not match the envelope";
    return false;
  }
  const auto invocationEnd = bytes.begin() +
                             spatialInvocationResultHeaderBytes +
                             static_cast<std::size_t>(invocationSize);
  result.invocation.assign(bytes.begin() + spatialInvocationResultHeaderBytes,
                           invocationEnd);
  result.spatialBoundaryResult.assign(invocationEnd, bytes.end());
  return true;
}

} // namespace loom::runtime

#endif // LOOM_RUNTIME_SPATIALINVOCATIONWIRE_H
