#ifndef LOOM_RUNTIME_GEM5BRIDGESOCKET_H
#define LOOM_RUNTIME_GEM5BRIDGESOCKET_H

#include "Gem5BridgeWire.h"

#include <cerrno>
#include <sys/socket.h>
#include <unistd.h>

namespace loom::runtime {

namespace detail {
inline bool readGem5BridgeBytes(int descriptor, std::uint8_t *bytes,
                                std::size_t size) {
  while (size != 0) {
    const ssize_t count = ::read(descriptor, bytes, size);
    if (count <= 0) {
      if (count < 0 && errno == EINTR)
        continue;
      return false;
    }
    bytes += count;
    size -= static_cast<std::size_t>(count);
  }
  return true;
}

inline bool writeGem5BridgeBytes(int descriptor,
                                 const std::vector<std::uint8_t> &bytes) {
  std::size_t offset = 0;
  while (offset != bytes.size()) {
    const ssize_t count = ::send(descriptor, bytes.data() + offset,
                                 bytes.size() - offset, MSG_NOSIGNAL);
    if (count <= 0) {
      if (count < 0 && errno == EINTR)
        continue;
      return false;
    }
    offset += static_cast<std::size_t>(count);
  }
  return true;
}
} // namespace detail

// The caller waits outside gem5 event callbacks. Framing is owned by the
// canonical wire codec; both engine and simulator use the same transport.
inline bool readGem5BridgeAdvance(int descriptor, std::uint64_t maximumMessages,
                                  std::uint64_t maximumMessageBytes,
                                  Gem5BridgeAdvance &advance,
                                  std::string &error) {
  std::vector<std::uint8_t> bytes(gem5BridgeAdvanceHeaderBytes);
  if (!detail::readGem5BridgeBytes(descriptor, bytes.data(), bytes.size())) {
    error = "engine disconnected before a causal advance header";
    return false;
  }
  Gem5BridgeAdvanceHeader header;
  if (!decodeGem5BridgeAdvanceHeader(bytes, header, error))
    return false;
  if (header.messageCount > maximumMessages ||
      header.messageCount > std::numeric_limits<std::size_t>::max()) {
    error = "causal advance exceeds the registered bridge count";
    return false;
  }
  Gem5BridgeAdvance decoded{header.generation, header.causalTick, {}};
  decoded.messages.reserve(static_cast<std::size_t>(header.messageCount));
  for (std::uint64_t index = 0; index != header.messageCount; ++index) {
    bytes.resize(gem5BridgeWireHeaderBytes);
    if (!detail::readGem5BridgeBytes(descriptor, bytes.data(), bytes.size())) {
      error = "engine disconnected before a boundary message header";
      return false;
    }
    Gem5BridgeWireHeader messageHeader;
    if (!decodeGem5BridgeWireHeader(bytes, messageHeader, error))
      return false;
    if (maximumMessageBytes < gem5BridgeWireHeaderBytes ||
        messageHeader.payloadSize >
            maximumMessageBytes - gem5BridgeWireHeaderBytes ||
        messageHeader.payloadSize > std::numeric_limits<std::size_t>::max() -
                                        gem5BridgeWireHeaderBytes) {
      error = "causal boundary exceeds the bridge message limit";
      return false;
    }
    bytes.resize(gem5BridgeWireHeaderBytes + messageHeader.payloadSize);
    if (!detail::readGem5BridgeBytes(descriptor,
                                     bytes.data() + gem5BridgeWireHeaderBytes,
                                     messageHeader.payloadSize)) {
      error = "engine disconnected before a boundary message payload";
      return false;
    }
    Gem5BridgeMessage message;
    if (!decodeGem5BridgeWireMessage(bytes, message, error))
      return false;
    decoded.messages.push_back(std::move(message));
  }
  advance = std::move(decoded);
  return true;
}

inline bool writeGem5BridgeAdvance(int descriptor,
                                   const Gem5BridgeAdvance &advance,
                                   std::string &error) {
  if (!detail::writeGem5BridgeBytes(descriptor,
                                    encodeGem5BridgeAdvanceHeader(advance))) {
    error = "cannot write causal advance header";
    return false;
  }
  for (const Gem5BridgeMessage &message : advance.messages)
    if (!detail::writeGem5BridgeBytes(descriptor,
                                      encodeGem5BridgeWireMessage(message))) {
      error = "cannot write causal boundary message";
      return false;
    }
  return true;
}

} // namespace loom::runtime

#endif // LOOM_RUNTIME_GEM5BRIDGESOCKET_H
