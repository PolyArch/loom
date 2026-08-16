#include "runtime/gem5/loom_spatial_bridge.hh"

#include "Runtime/Gem5SpatialBridgeABI.h"
#include "Runtime/SpatialInvocationWire.h"

#include "base/addr_range.hh"
#include "base/logging.hh"
#include "mem/packet.hh"

#include <cerrno>
#include <cstring>
#include <fstream>
#include <limits>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

namespace gem5 {
namespace {

using namespace loom::runtime;

bool launchFitsMessageLimit(std::uint64_t staticBytes,
                            std::uint64_t invocationBytes,
                            std::uint64_t limit) {
  constexpr std::uint64_t envelopeBytes =
      loom::runtime::gem5BridgeWireHeaderBytes +
      loom::runtime::gem5SpatialLaunchHeaderBytes;
  return envelopeBytes <= limit && staticBytes != 0 &&
         staticBytes <= limit - envelopeBytes &&
         invocationBytes <= limit - envelopeBytes - staticBytes;
}

bool readAll(int descriptor, std::uint8_t *bytes, std::size_t size) {
  while (size != 0) {
    const ssize_t count = ::read(descriptor, bytes, size);
    if (count == 0)
      return false;
    if (count < 0) {
      if (errno == EINTR)
        continue;
      return false;
    }
    bytes += count;
    size -= static_cast<std::size_t>(count);
  }
  return true;
}

bool writeAll(int descriptor, const std::uint8_t *bytes, std::size_t size) {
  while (size != 0) {
    const ssize_t count = ::write(descriptor, bytes, size);
    if (count < 0) {
      if (errno == EINTR)
        continue;
      return false;
    }
    bytes += count;
    size -= static_cast<std::size_t>(count);
  }
  return true;
}

} // namespace

LoomSpatialBridge::LoomSpatialBridge(const Params &params)
    : DmaDevice(params), pioAddress(params.pio_addr), pioSize(params.pio_size),
      pioDelay(params.pio_latency), engineSocketPath(params.engine_socket),
      resultPath(params.result_path),
      maximumMessageBytes(params.max_message_bytes),
      maximumInvocations(params.max_invocations),
      launchEvent([this] { fetchStaticLaunch(); }, name() + ".launch"),
      staticLaunchCompletionEvent([this] { fetchInvocation(); },
                                  name() + ".static_launch_completion"),
      invocationCompletionEvent([this] { startLaunch(); },
                                name() + ".invocation_completion"),
      dmaCompletionEvent([this] { completeMemoryRequest(); },
                         name() + ".dma_completion"),
      completionEvent([this] { completeInvocation(); },
                      name() + ".completion") {
  panic_if(engineSocketPath.empty(), "LoomSpatialBridge socket is empty");
  panic_if(resultPath.empty(), "LoomSpatialBridge result path is empty");
  panic_if(maximumMessageBytes < loom::runtime::gem5BridgeWireHeaderBytes,
           "LoomSpatialBridge message limit is too small");
  panic_if(maximumMessageBytes >
               static_cast<std::uint64_t>(std::numeric_limits<int>::max()),
           "LoomSpatialBridge message limit exceeds the DMA size domain");
  panic_if(maximumInvocations == 0,
           "LoomSpatialBridge invocation limit must be positive");
}

LoomSpatialBridge::~LoomSpatialBridge() {
  if (engineSocket >= 0)
    ::close(engineSocket);
}

AddrRangeList LoomSpatialBridge::getAddrRanges() const {
  return {AddrRange(pioAddress, pioAddress + pioSize - 1)};
}

std::uint32_t LoomSpatialBridge::status() const {
  switch (state) {
  case State::Idle:
    return 0;
  case State::Running:
  case State::WaitingForMemory:
    return gem5SpatialBridgeBusy;
  case State::Complete:
    return gem5SpatialBridgeDone;
  case State::Failed:
    return gem5SpatialBridgeFailed;
  }
  panic("unknown LoomSpatialBridge state");
}

Tick LoomSpatialBridge::read(PacketPtr packet) {
  const Addr offset = packet->getAddr() - pioAddress;
  panic_if(packet->getSize() != 4,
           "LoomSpatialBridge requires 32-bit MMIO accesses");
  std::uint32_t value = 0;
  switch (offset) {
  case gem5SpatialBridgeStatus:
    value = status();
    break;
  case gem5SpatialBridgeError:
    value = errorCode;
    break;
  case gem5SpatialBridgeSequenceLow:
    value = static_cast<std::uint32_t>(nextSequence);
    break;
  case gem5SpatialBridgeSequenceHigh:
    value = static_cast<std::uint32_t>(nextSequence >> 32);
    break;
  case gem5SpatialBridgeStaticLaunchLow:
    value = static_cast<std::uint32_t>(staticLaunchAddress);
    break;
  case gem5SpatialBridgeStaticLaunchHigh:
    value = static_cast<std::uint32_t>(staticLaunchAddress >> 32);
    break;
  case gem5SpatialBridgeStaticLaunchSize:
    value = staticLaunchSize;
    break;
  case gem5SpatialBridgeInvocationLow:
    value = static_cast<std::uint32_t>(invocationAddress);
    break;
  case gem5SpatialBridgeInvocationHigh:
    value = static_cast<std::uint32_t>(invocationAddress >> 32);
    break;
  case gem5SpatialBridgeInvocationSize:
    value = invocationSize;
    break;
  case gem5SpatialBridgeCompletionTickLow:
    value = static_cast<std::uint32_t>(lastCompletionTick);
    break;
  case gem5SpatialBridgeCompletionTickHigh:
    value = static_cast<std::uint32_t>(lastCompletionTick >> 32);
    break;
  default:
    fail(1, "read from an unknown MMIO register");
    break;
  }
  packet->setUintX(value, ByteOrder::little);
  packet->makeAtomicResponse();
  return pioDelay;
}

Tick LoomSpatialBridge::write(PacketPtr packet) {
  const Addr offset = packet->getAddr() - pioAddress;
  panic_if(packet->getSize() != 4,
           "LoomSpatialBridge requires 32-bit MMIO accesses");
  const std::uint32_t value =
      static_cast<std::uint32_t>(packet->getUintX(ByteOrder::little));
  const bool descriptorWrite = offset == gem5SpatialBridgeStaticLaunchLow ||
                               offset == gem5SpatialBridgeStaticLaunchHigh ||
                               offset == gem5SpatialBridgeStaticLaunchSize ||
                               offset == gem5SpatialBridgeInvocationLow ||
                               offset == gem5SpatialBridgeInvocationHigh ||
                               offset == gem5SpatialBridgeInvocationSize;
  if (descriptorWrite && state != State::Idle && state != State::Complete) {
    fail(2, "launch descriptor changed while the bridge is active");
  } else if (offset == gem5SpatialBridgeStaticLaunchLow) {
    staticLaunchAddress = (staticLaunchAddress & 0xffffffff00000000ULL) |
                          static_cast<std::uint64_t>(value);
  } else if (offset == gem5SpatialBridgeStaticLaunchHigh) {
    staticLaunchAddress = (staticLaunchAddress & 0x00000000ffffffffULL) |
                          (static_cast<std::uint64_t>(value) << 32);
  } else if (offset == gem5SpatialBridgeStaticLaunchSize) {
    staticLaunchSize = value;
  } else if (offset == gem5SpatialBridgeInvocationLow) {
    invocationAddress = (invocationAddress & 0xffffffff00000000ULL) |
                        static_cast<std::uint64_t>(value);
  } else if (offset == gem5SpatialBridgeInvocationHigh) {
    invocationAddress = (invocationAddress & 0x00000000ffffffffULL) |
                        (static_cast<std::uint64_t>(value) << 32);
  } else if (offset == gem5SpatialBridgeInvocationSize) {
    invocationSize = value;
  } else if (offset != gem5SpatialBridgeControl) {
    fail(2, "write to an unknown MMIO register");
  } else if (value & gem5SpatialBridgeReset) {
    resetBridge();
  } else if (value & gem5SpatialBridgeStart) {
    if (state != State::Idle && state != State::Complete)
      fail(3, "launch requested while the bridge is not idle");
    else if (nextSequence >= maximumInvocations)
      fail(20, "launch count exceeds the bridge session limit");
    else if (!launchFitsMessageLimit(staticLaunchSize, invocationSize,
                                     maximumMessageBytes)) {
      fail(17, "launch payload size is outside the bridge limit");
    } else {
      activeStaticLaunchAddress = staticLaunchAddress;
      activeStaticLaunchSize = staticLaunchSize;
      activeInvocationAddress = invocationAddress;
      activeInvocationSize = invocationSize;
      state = State::Running;
      errorCode = 0;
      schedule(&launchEvent, clockEdge());
    }
  }
  packet->makeAtomicResponse();
  return pioDelay;
}

void LoomSpatialBridge::fetchStaticLaunch() {
  if (activeStaticLaunchSize == 0 ||
      activeStaticLaunchSize > maximumMessageBytes) {
    fail(18, "active static launch descriptor is invalid");
    return;
  }
  staticLaunchPayload.assign(activeStaticLaunchSize, 0);
  dmaRead(activeStaticLaunchAddress, static_cast<int>(activeStaticLaunchSize),
          &staticLaunchCompletionEvent, staticLaunchPayload.data());
}

void LoomSpatialBridge::fetchInvocation() {
  invocationPayload.assign(activeInvocationSize, 0);
  if (activeInvocationSize == 0) {
    startLaunch();
    return;
  }
  dmaRead(activeInvocationAddress, static_cast<int>(activeInvocationSize),
          &invocationCompletionEvent, invocationPayload.data());
}

bool LoomSpatialBridge::connectEngine() {
  if (engineSocket >= 0)
    return true;
  engineSocket = ::socket(AF_UNIX, SOCK_STREAM, 0);
  if (engineSocket < 0)
    return false;
  sockaddr_un address{};
  if (engineSocketPath.size() >= sizeof(address.sun_path)) {
    ::close(engineSocket);
    engineSocket = -1;
    return false;
  }
  address.sun_family = AF_UNIX;
  std::memcpy(address.sun_path, engineSocketPath.c_str(),
              engineSocketPath.size() + 1);
  if (::connect(engineSocket, reinterpret_cast<sockaddr *>(&address),
                sizeof(address)) != 0) {
    ::close(engineSocket);
    engineSocket = -1;
    return false;
  }
  return true;
}

bool LoomSpatialBridge::sendMessage(
    const loom::runtime::Gem5BridgeMessage &message) {
  const std::vector<std::uint8_t> bytes =
      loom::runtime::encodeGem5BridgeWireMessage(message);
  return bytes.size() <= maximumMessageBytes &&
         writeAll(engineSocket, bytes.data(), bytes.size());
}

bool LoomSpatialBridge::receiveMessage(
    loom::runtime::Gem5BridgeMessage &message) {
  std::vector<std::uint8_t> bytes(loom::runtime::gem5BridgeWireHeaderBytes);
  if (!readAll(engineSocket, bytes.data(), bytes.size()))
    return false;
  const std::uint64_t payloadSize =
      loom::runtime::detail::readGem5BridgeU64(bytes.data() + 16);
  if (payloadSize > maximumMessageBytes - bytes.size() ||
      payloadSize > std::numeric_limits<std::size_t>::max())
    return false;
  const std::size_t headerSize = bytes.size();
  bytes.resize(headerSize + static_cast<std::size_t>(payloadSize));
  if (payloadSize != 0 && !readAll(engineSocket, bytes.data() + headerSize,
                                   static_cast<std::size_t>(payloadSize)))
    return false;
  std::string diagnostic;
  return loom::runtime::decodeGem5BridgeWireMessage(bytes, message, diagnostic);
}

void LoomSpatialBridge::startLaunch() {
  if (!connectEngine()) {
    fail(4, "could not connect to the Spatial engine");
    return;
  }
  std::vector<std::uint8_t> launchPayload =
      loom::runtime::encodeGem5SpatialLaunchEnvelope(
          {staticLaunchPayload, invocationPayload});
  const loom::runtime::Gem5BridgeMessage launch{
      loom::runtime::Gem5BridgeMessageKind::SpatialLaunch, nextSequence,
      launchPayload};
  if (!sendMessage(launch)) {
    fail(5, "could not send the Spatial launch");
    return;
  }
  consumeEngineMessage();
}

void LoomSpatialBridge::consumeEngineMessage() {
  loom::runtime::Gem5BridgeMessage message;
  if (!receiveMessage(message)) {
    fail(6, "could not receive a canonical Spatial engine message");
    return;
  }
  if (message.sequence != nextSequence) {
    fail(7, "Spatial engine response has the wrong sequence");
    return;
  }
  if (message.kind == loom::runtime::Gem5BridgeMessageKind::MemoryRequest ||
      message.kind == loom::runtime::Gem5BridgeMessageKind::ChannelTransfer) {
    std::string diagnostic;
    if (!loom::runtime::decodeGem5BridgeMemoryRequest(
            message.payload, pendingMemory, diagnostic)) {
      fail(8, diagnostic);
      return;
    }
    if (message.kind == loom::runtime::Gem5BridgeMessageKind::ChannelTransfer &&
        pendingMemory.operation !=
            loom::runtime::Gem5BridgeMemoryOperation::Write) {
      fail(9, "channel transfer is not a write transaction");
      return;
    }
    if (pendingMemory.size > std::numeric_limits<int>::max()) {
      fail(10, "memory transaction is too large");
      return;
    }
    memoryBuffer = pendingMemory.data;
    if (pendingMemory.operation ==
        loom::runtime::Gem5BridgeMemoryOperation::Read)
      memoryBuffer.assign(static_cast<std::size_t>(pendingMemory.size), 0);
    state = State::WaitingForMemory;
    if (pendingMemory.operation ==
        loom::runtime::Gem5BridgeMemoryOperation::Read)
      dmaRead(pendingMemory.address, static_cast<int>(pendingMemory.size),
              &dmaCompletionEvent, memoryBuffer.data(),
              pendingMemory.readyAfterTicks);
    else
      dmaWrite(pendingMemory.address, static_cast<int>(pendingMemory.size),
               &dmaCompletionEvent, memoryBuffer.data(),
               pendingMemory.readyAfterTicks);
    return;
  }
  if (message.kind != loom::runtime::Gem5BridgeMessageKind::Completion) {
    fail(11, "Spatial engine emitted an unexpected message kind");
    return;
  }
  std::string diagnostic;
  if (!loom::runtime::decodeGem5BridgeCompletion(
          message.payload, pendingCompletion, diagnostic)) {
    fail(12, diagnostic);
    return;
  }
  loom::runtime::SpatialInvocationResultWire invocationResult;
  if (!loom::runtime::decodeSpatialInvocationResultWire(
          pendingCompletion.result, invocationResult, diagnostic) ||
      invocationResult.invocation != invocationPayload) {
    fail(19, "Spatial completion names a foreign invocation");
    return;
  }
  schedule(&completionEvent, curTick() + pendingCompletion.readyAfterTicks);
}

void LoomSpatialBridge::completeMemoryRequest() {
  if (state != State::WaitingForMemory) {
    fail(13, "memory completion arrived in the wrong bridge state");
    return;
  }
  const loom::runtime::Gem5BridgeMemoryResponse response{
      pendingMemory.requestId, true,
      pendingMemory.operation == loom::runtime::Gem5BridgeMemoryOperation::Read
          ? memoryBuffer
          : std::vector<std::uint8_t>{}};
  const loom::runtime::Gem5BridgeMessage message{
      loom::runtime::Gem5BridgeMessageKind::MemoryResponse, nextSequence,
      loom::runtime::encodeGem5BridgeMemoryResponse(response)};
  if (!sendMessage(message)) {
    fail(14, "could not send the memory response");
    return;
  }
  state = State::Running;
  consumeEngineMessage();
}

void LoomSpatialBridge::completeInvocation() {
  lastCompletionTick = curTick();
  panic_if(completedResults.results.size() != nextSequence,
           "LoomSpatialBridge result sequence is not dense");
  completedResults.results.push_back({pendingCompletion.status,
                                      lastCompletionTick, nextSequence,
                                      pendingCompletion.result});
  const std::vector<std::uint8_t> normalized =
      loom::runtime::encodeGem5BridgeResultCollection(completedResults);
  if (normalized.size() > maximumMessageBytes) {
    completedResults.results.pop_back();
    fail(21, "normalized result collection exceeds the bridge limit");
    return;
  }
  std::ofstream output(resultPath, std::ios::binary | std::ios::trunc);
  if (!output) {
    fail(15, "could not create the normalized result");
    return;
  }
  output.write(reinterpret_cast<const char *>(normalized.data()),
               static_cast<std::streamsize>(normalized.size()));
  if (!output) {
    fail(16, "could not write the normalized result");
    return;
  }
  errorCode = pendingCompletion.status;
  state = pendingCompletion.status == 0 ? State::Complete : State::Failed;
  ++nextSequence;
}

void LoomSpatialBridge::fail(std::uint32_t code, const std::string &message) {
  warn("LoomSpatialBridge failed: %s", message.c_str());
  errorCode = code;
  state = State::Failed;
}

void LoomSpatialBridge::resetBridge() {
  panic_if(dmaPending(), "cannot reset LoomSpatialBridge with pending DMA");
  if (launchEvent.scheduled())
    deschedule(&launchEvent);
  if (staticLaunchCompletionEvent.scheduled())
    deschedule(&staticLaunchCompletionEvent);
  if (invocationCompletionEvent.scheduled())
    deschedule(&invocationCompletionEvent);
  if (completionEvent.scheduled())
    deschedule(&completionEvent);
  memoryBuffer.clear();
  staticLaunchPayload.clear();
  invocationPayload.clear();
  pendingMemory = {};
  pendingCompletion = {};
  errorCode = 0;
  staticLaunchAddress = 0;
  staticLaunchSize = 0;
  invocationAddress = 0;
  invocationSize = 0;
  activeStaticLaunchAddress = 0;
  activeStaticLaunchSize = 0;
  activeInvocationAddress = 0;
  activeInvocationSize = 0;
  lastCompletionTick = 0;
  state = State::Idle;
}

} // namespace gem5
