#include "runtime/gem5/loom_spatial_bridge.hh"

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

constexpr Addr statusRegister = 0x00;
constexpr Addr controlRegister = 0x04;
constexpr Addr errorRegister = 0x08;
constexpr Addr sequenceLowRegister = 0x0c;
constexpr Addr sequenceHighRegister = 0x10;
constexpr Addr payloadAddressLowRegister = 0x14;
constexpr Addr payloadAddressHighRegister = 0x18;
constexpr Addr payloadSizeRegister = 0x1c;
constexpr Addr completionTickLowRegister = 0x20;
constexpr Addr completionTickHighRegister = 0x24;
constexpr std::uint32_t controlStart = 1u << 0;
constexpr std::uint32_t controlReset = 1u << 1;

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
      launchEvent([this] { fetchLaunchPayload(); }, name() + ".launch"),
      launchPayloadCompletionEvent([this] { startLaunch(); },
                                   name() + ".launch_payload_completion"),
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
    return statusBusy;
  case State::Complete:
    return statusDone;
  case State::Failed:
    return statusError;
  }
  panic("unknown LoomSpatialBridge state");
}

Tick LoomSpatialBridge::read(PacketPtr packet) {
  const Addr offset = packet->getAddr() - pioAddress;
  panic_if(packet->getSize() != 4,
           "LoomSpatialBridge requires 32-bit MMIO accesses");
  std::uint32_t value = 0;
  switch (offset) {
  case statusRegister:
    value = status();
    break;
  case errorRegister:
    value = errorCode;
    break;
  case sequenceLowRegister:
    value = static_cast<std::uint32_t>(nextSequence);
    break;
  case sequenceHighRegister:
    value = static_cast<std::uint32_t>(nextSequence >> 32);
    break;
  case payloadAddressLowRegister:
    value = static_cast<std::uint32_t>(launchPayloadAddress);
    break;
  case payloadAddressHighRegister:
    value = static_cast<std::uint32_t>(launchPayloadAddress >> 32);
    break;
  case payloadSizeRegister:
    value = launchPayloadSize;
    break;
  case completionTickLowRegister:
    value = static_cast<std::uint32_t>(lastCompletionTick);
    break;
  case completionTickHighRegister:
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
  if (offset == payloadAddressLowRegister) {
    launchPayloadAddress = (launchPayloadAddress & 0xffffffff00000000ULL) |
                           static_cast<std::uint64_t>(value);
  } else if (offset == payloadAddressHighRegister) {
    launchPayloadAddress =
        (launchPayloadAddress & 0x00000000ffffffffULL) |
        (static_cast<std::uint64_t>(value) << 32);
  } else if (offset == payloadSizeRegister) {
    launchPayloadSize = value;
  } else if (offset != controlRegister) {
    fail(2, "write to an unknown MMIO register");
  } else if (value & controlReset) {
    resetBridge();
  } else if (value & controlStart) {
    if (state != State::Idle && state != State::Complete)
      fail(3, "launch requested while the bridge is not idle");
    else if (launchPayloadSize == 0 ||
             launchPayloadSize > maximumMessageBytes) {
      fail(17, "launch payload size is outside the bridge limit");
    } else {
      state = State::Running;
      errorCode = 0;
      schedule(&launchEvent, clockEdge());
    }
  }
  packet->makeAtomicResponse();
  return pioDelay;
}

void LoomSpatialBridge::fetchLaunchPayload() {
  if (launchPayloadSize == 0 || launchPayloadSize > maximumMessageBytes) {
    fail(18, "launch payload descriptor changed before DMA");
    return;
  }
  launchPayload.assign(launchPayloadSize, 0);
  dmaRead(launchPayloadAddress, static_cast<int>(launchPayloadSize),
          &launchPayloadCompletionEvent, launchPayload.data());
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
  const std::uint64_t payloadSize = loom::runtime::detail::readGem5BridgeU64(
      bytes.data() + 16);
  if (payloadSize > maximumMessageBytes - bytes.size() ||
      payloadSize > std::numeric_limits<std::size_t>::max())
    return false;
  const std::size_t headerSize = bytes.size();
  bytes.resize(headerSize + static_cast<std::size_t>(payloadSize));
  if (payloadSize != 0 &&
      !readAll(engineSocket, bytes.data() + headerSize,
               static_cast<std::size_t>(payloadSize)))
    return false;
  std::string diagnostic;
  return loom::runtime::decodeGem5BridgeWireMessage(bytes, message,
                                                    diagnostic);
}

void LoomSpatialBridge::startLaunch() {
  if (!connectEngine()) {
    fail(4, "could not connect to the Spatial engine");
    return;
  }
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
  const std::vector<std::uint8_t> normalized =
      loom::runtime::encodeGem5BridgeResult(
          {pendingCompletion.status, lastCompletionTick, nextSequence,
           pendingCompletion.result});
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
  if (launchPayloadCompletionEvent.scheduled())
    deschedule(&launchPayloadCompletionEvent);
  if (completionEvent.scheduled())
    deschedule(&completionEvent);
  memoryBuffer.clear();
  launchPayload.clear();
  pendingMemory = {};
  pendingCompletion = {};
  errorCode = 0;
  launchPayloadAddress = 0;
  launchPayloadSize = 0;
  lastCompletionTick = 0;
  state = State::Idle;
}

} // namespace gem5
