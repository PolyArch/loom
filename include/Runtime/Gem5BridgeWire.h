#ifndef LOOM_RUNTIME_GEM5BRIDGEWIRE_H
#define LOOM_RUNTIME_GEM5BRIDGEWIRE_H

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace loom::runtime {

inline constexpr char gem5BridgeAbiIdentity[] =
    "loom.gem5_spatial_bridge_abi.v1";

enum class Gem5BridgeMessageKind : std::uint32_t {
  SpatialLaunch = 0,
  MemoryRequest = 1,
  MemoryResponse = 2,
  ChannelTransfer = 3,
  Completion = 4,
};

struct Gem5BridgeMessage final {
  Gem5BridgeMessageKind kind = Gem5BridgeMessageKind::SpatialLaunch;
  std::uint64_t sequence = 0;
  std::vector<std::uint8_t> payload;
};

enum class Gem5BridgeMemoryOperation : std::uint32_t {
  Read = 0,
  Write = 1,
};

struct Gem5BridgeMemoryRequest final {
  Gem5BridgeMemoryOperation operation = Gem5BridgeMemoryOperation::Read;
  std::uint64_t readyAfterTicks = 0;
  std::uint64_t requestId = 0;
  std::uint64_t address = 0;
  std::uint64_t size = 0;
  std::vector<std::uint8_t> data;
};

struct Gem5BridgeMemoryResponse final {
  std::uint64_t requestId = 0;
  bool success = false;
  std::vector<std::uint8_t> data;
};

struct Gem5BridgeCompletion final {
  std::uint64_t readyAfterTicks = 0;
  std::uint32_t status = 0;
  std::vector<std::uint8_t> result;
};

inline constexpr std::array<std::uint8_t, 4> gem5BridgeWireMagic{
    'L', 'G', 'B', '1'};
inline constexpr std::size_t gem5BridgeWireHeaderBytes = 24;

namespace detail {

inline void appendGem5BridgeU32(std::vector<std::uint8_t> &bytes,
                               std::uint32_t value) {
  for (int shift = 24; shift >= 0; shift -= 8)
    bytes.push_back(static_cast<std::uint8_t>(value >> shift));
}

inline void appendGem5BridgeU64(std::vector<std::uint8_t> &bytes,
                               std::uint64_t value) {
  for (int shift = 56; shift >= 0; shift -= 8)
    bytes.push_back(static_cast<std::uint8_t>(value >> shift));
}

inline std::uint32_t readGem5BridgeU32(const std::uint8_t *bytes) {
  std::uint32_t value = 0;
  for (unsigned index = 0; index < 4; ++index)
    value = (value << 8) | bytes[index];
  return value;
}

inline std::uint64_t readGem5BridgeU64(const std::uint8_t *bytes) {
  std::uint64_t value = 0;
  for (unsigned index = 0; index < 8; ++index)
    value = (value << 8) | bytes[index];
  return value;
}

} // namespace detail

inline std::vector<std::uint8_t>
encodeGem5BridgeWireMessage(const Gem5BridgeMessage &message) {
  std::vector<std::uint8_t> bytes;
  bytes.reserve(gem5BridgeWireHeaderBytes + message.payload.size());
  bytes.insert(bytes.end(), gem5BridgeWireMagic.begin(),
               gem5BridgeWireMagic.end());
  detail::appendGem5BridgeU32(bytes,
                             static_cast<std::uint32_t>(message.kind));
  detail::appendGem5BridgeU64(bytes, message.sequence);
  detail::appendGem5BridgeU64(bytes, message.payload.size());
  bytes.insert(bytes.end(), message.payload.begin(), message.payload.end());
  return bytes;
}

inline bool decodeGem5BridgeWireMessage(const std::vector<std::uint8_t> &bytes,
                                        Gem5BridgeMessage &message,
                                        std::string &error) {
  if (bytes.size() < gem5BridgeWireHeaderBytes) {
    error = "truncated header";
    return false;
  }
  if (!std::equal(gem5BridgeWireMagic.begin(), gem5BridgeWireMagic.end(),
                  bytes.begin())) {
    error = "wrong ABI magic";
    return false;
  }
  const std::uint32_t kind = detail::readGem5BridgeU32(bytes.data() + 4);
  if (kind > static_cast<std::uint32_t>(Gem5BridgeMessageKind::Completion)) {
    error = "unknown message kind";
    return false;
  }
  const std::uint64_t payloadSize =
      detail::readGem5BridgeU64(bytes.data() + 16);
  if (payloadSize > std::numeric_limits<std::size_t>::max() ||
      payloadSize != bytes.size() - gem5BridgeWireHeaderBytes) {
    error = "payload length does not match the envelope";
    return false;
  }
  message.kind = static_cast<Gem5BridgeMessageKind>(kind);
  message.sequence = detail::readGem5BridgeU64(bytes.data() + 8);
  message.payload.assign(bytes.begin() + gem5BridgeWireHeaderBytes,
                         bytes.end());
  return true;
}

inline std::vector<std::uint8_t>
encodeGem5BridgeMemoryRequest(const Gem5BridgeMemoryRequest &request) {
  std::vector<std::uint8_t> bytes;
  bytes.reserve(36 + request.data.size());
  detail::appendGem5BridgeU32(bytes,
                             static_cast<std::uint32_t>(request.operation));
  detail::appendGem5BridgeU64(bytes, request.readyAfterTicks);
  detail::appendGem5BridgeU64(bytes, request.requestId);
  detail::appendGem5BridgeU64(bytes, request.address);
  detail::appendGem5BridgeU64(bytes, request.size);
  bytes.insert(bytes.end(), request.data.begin(), request.data.end());
  return bytes;
}

inline bool decodeGem5BridgeMemoryRequest(
    const std::vector<std::uint8_t> &bytes, Gem5BridgeMemoryRequest &request,
    std::string &error) {
  constexpr std::size_t headerBytes = 36;
  if (bytes.size() < headerBytes) {
    error = "truncated memory request";
    return false;
  }
  const std::uint32_t operation =
      detail::readGem5BridgeU32(bytes.data());
  if (operation >
      static_cast<std::uint32_t>(Gem5BridgeMemoryOperation::Write)) {
    error = "unknown memory operation";
    return false;
  }
  request.operation = static_cast<Gem5BridgeMemoryOperation>(operation);
  request.readyAfterTicks = detail::readGem5BridgeU64(bytes.data() + 4);
  request.requestId = detail::readGem5BridgeU64(bytes.data() + 12);
  request.address = detail::readGem5BridgeU64(bytes.data() + 20);
  request.size = detail::readGem5BridgeU64(bytes.data() + 28);
  request.data.assign(bytes.begin() + headerBytes, bytes.end());
  if (request.size == 0 ||
      (request.operation == Gem5BridgeMemoryOperation::Read &&
       !request.data.empty()) ||
      (request.operation == Gem5BridgeMemoryOperation::Write &&
       request.data.size() != request.size)) {
    error = "memory request data does not match its operation and size";
    return false;
  }
  return true;
}

inline std::vector<std::uint8_t>
encodeGem5BridgeMemoryResponse(const Gem5BridgeMemoryResponse &response) {
  std::vector<std::uint8_t> bytes;
  bytes.reserve(20 + response.data.size());
  detail::appendGem5BridgeU64(bytes, response.requestId);
  detail::appendGem5BridgeU32(bytes, response.success ? 1 : 0);
  detail::appendGem5BridgeU64(bytes, response.data.size());
  bytes.insert(bytes.end(), response.data.begin(), response.data.end());
  return bytes;
}

inline bool decodeGem5BridgeMemoryResponse(
    const std::vector<std::uint8_t> &bytes, Gem5BridgeMemoryResponse &response,
    std::string &error) {
  constexpr std::size_t headerBytes = 20;
  if (bytes.size() < headerBytes) {
    error = "truncated memory response";
    return false;
  }
  const std::uint32_t success =
      detail::readGem5BridgeU32(bytes.data() + 8);
  const std::uint64_t size =
      detail::readGem5BridgeU64(bytes.data() + 12);
  if (success > 1 || size != bytes.size() - headerBytes) {
    error = "memory response fields are not canonical";
    return false;
  }
  response.requestId = detail::readGem5BridgeU64(bytes.data());
  response.success = success == 1;
  response.data.assign(bytes.begin() + headerBytes, bytes.end());
  return true;
}

inline std::vector<std::uint8_t>
encodeGem5BridgeCompletion(const Gem5BridgeCompletion &completion) {
  std::vector<std::uint8_t> bytes;
  bytes.reserve(20 + completion.result.size());
  detail::appendGem5BridgeU64(bytes, completion.readyAfterTicks);
  detail::appendGem5BridgeU32(bytes, completion.status);
  detail::appendGem5BridgeU64(bytes, completion.result.size());
  bytes.insert(bytes.end(), completion.result.begin(), completion.result.end());
  return bytes;
}

inline bool decodeGem5BridgeCompletion(
    const std::vector<std::uint8_t> &bytes, Gem5BridgeCompletion &completion,
    std::string &error) {
  constexpr std::size_t headerBytes = 20;
  if (bytes.size() < headerBytes) {
    error = "truncated completion";
    return false;
  }
  const std::uint64_t size =
      detail::readGem5BridgeU64(bytes.data() + 12);
  if (size != bytes.size() - headerBytes) {
    error = "completion result size does not match its payload";
    return false;
  }
  completion.readyAfterTicks = detail::readGem5BridgeU64(bytes.data());
  completion.status = detail::readGem5BridgeU32(bytes.data() + 8);
  completion.result.assign(bytes.begin() + headerBytes, bytes.end());
  return true;
}

} // namespace loom::runtime

#endif // LOOM_RUNTIME_GEM5BRIDGEWIRE_H
