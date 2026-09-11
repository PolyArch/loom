#include "runtime/gem5/loom_spatial_bridge.hh"

#include "Runtime/Gem5SpatialBridgeABI.h"
#include "Runtime/SpatialInvocationWire.h"
#include "runtime/gem5/loom_spatial_engine_session.hh"

#include "base/addr_range.hh"
#include "base/logging.hh"
#include "mem/packet.hh"
#include "sim/system.hh"

#include <chrono>
#include <ctime>
#include <fstream>
#include <limits>
#include <optional>

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

std::optional<std::uint64_t> threadCpuNanoseconds() {
  timespec value{};
  if (::clock_gettime(CLOCK_THREAD_CPUTIME_ID, &value) != 0)
    return std::nullopt;
  constexpr std::uint64_t nanosecondsPerSecond = 1'000'000'000;
  if (value.tv_sec < 0 || value.tv_nsec < 0 ||
      static_cast<std::uint64_t>(value.tv_nsec) >= nanosecondsPerSecond ||
      static_cast<std::uint64_t>(value.tv_sec) >
          (std::numeric_limits<std::uint64_t>::max() -
           static_cast<std::uint64_t>(value.tv_nsec)) /
              nanosecondsPerSecond)
    return std::nullopt;
  return static_cast<std::uint64_t>(value.tv_sec) * nanosecondsPerSecond +
         static_cast<std::uint64_t>(value.tv_nsec);
}

} // namespace

LoomSpatialBridge::PerformanceStatistics::PerformanceStatistics(
    statistics::Group *parent)
    : statistics::Group(parent, "loomPerformance"),
      ADD_STAT(callbackCpuNanoseconds, statistics::units::Count::get(),
               "Host thread CPU nanoseconds spent inside the bridge"),
      ADD_STAT(engineWaitNanoseconds, statistics::units::Count::get(),
               "Host wall nanoseconds awaiting Spatial engine messages"),
      ADD_STAT(messageCount, statistics::units::Count::get(),
               "Canonical Spatial engine messages consumed"),
      ADD_STAT(invocationCount, statistics::units::Count::get(),
               "Spatial invocations completed"),
      ADD_STAT(clockFailureCount, statistics::units::Count::get(),
               "Host performance clock samples that failed"),
      ADD_STAT(staticLaunchFetchCount, statistics::units::Count::get(),
               "Immutable Spatial launch images fetched from guest memory") {}

void LoomSpatialBridge::PhaseObservation::openAt(std::uint64_t tick,
                                                 std::uint64_t occupied) {
  if (observed)
    return;
  observed = true;
  beginTick = tick;
  beginOccupiedTicks = occupied;
  endTick = tick;
  endOccupiedTicks = occupied;
}

void LoomSpatialBridge::PhaseObservation::closeAt(std::uint64_t tick,
                                                  std::uint64_t occupied) {
  panic_if(!observed, "Spatial accelerated phase closed before it opened");
  endTick = tick;
  endOccupiedTicks = occupied;
}

LoomSpatialBridge::MemoryTransaction::MemoryTransaction(
    LoomSpatialBridge &bridge,
    loom::runtime::Gem5BridgeMemoryRequest request)
    : request(std::move(request)),
      issueEvent(
          [this, &bridge] {
            bridge.runAccounted([&] { bridge.issueMemoryRequest(*this); });
          },
          bridge.name() + ".memory_request"),
      completionEvent(
          [this, &bridge] {
            bridge.runAccounted([&] { bridge.completeMemoryRequest(*this); });
          },
          bridge.name() + ".dma_completion") {
  buffer = this->request.data;
  if (this->request.operation ==
      loom::runtime::Gem5BridgeMemoryOperation::Read)
    buffer.assign(static_cast<std::size_t>(this->request.size), 0);
}

LoomSpatialBridge::LoomSpatialBridge(const Params &params)
    : DmaDevice(params), performanceStatistics(this),
      pioAddress(params.pio_addr), pioSize(params.pio_size),
      pioDelay(params.pio_latency),
      bridgeSessionOrdinal(params.session_ordinal),
      engineSession(params.engine_session),
      threadDispatch(params.thread_dispatch),
      memoryService(params.memory_service),
      configurationImageBytes(params.configuration_image_bytes),
      resultPath(params.result_path),
      maximumMessageBytes(params.max_message_bytes),
      collectPerformance(params.collect_performance),
      launchEvent(
          [this] { runAccounted([this] { fetchStaticLaunch(); }); },
          name() + ".launch"),
      staticLaunchCompletionEvent(
          [this] { runAccounted([this] { completeConfigurationTransport(); }); },
          name() + ".static_launch_completion"),
      invocationCompletionEvent(
          [this] { runAccounted([this] { startLaunch(); }); },
          name() + ".invocation_completion"),
      completionEvent(
          [this] { runAccounted([this] { completeInvocation(); }); },
          name() + ".completion"),
      channelCommitEvent(
          [this] { runAccounted([this] { completeChannelCommit(); }); },
          name() + ".channel_commit") {
  if (engineSession)
    engineSession->registerBridge(bridgeSessionOrdinal, *this);
  panic_if(!threadDispatch || !memoryService,
           "LoomSpatialBridge has no computation boundary or service observer");
  panic_if(configurationImageBytes >
               static_cast<std::uint64_t>(std::numeric_limits<int>::max()),
           "LoomSpatialBridge configuration image exceeds the DMA size domain");
  panic_if(resultPath.empty(), "LoomSpatialBridge result path is empty");
  panic_if(maximumMessageBytes < loom::runtime::gem5BridgeWireHeaderBytes,
           "LoomSpatialBridge message limit is too small");
  panic_if(maximumMessageBytes >
               static_cast<std::uint64_t>(std::numeric_limits<int>::max()),
           "LoomSpatialBridge message limit exceeds the DMA size domain");
  panic_if(publishResults() != ResultPublication::Published,
           "LoomSpatialBridge could not publish its empty result");
}

AddrRangeList LoomSpatialBridge::getAddrRanges() const {
  return {AddrRange(pioAddress, pioAddress + pioSize - 1)};
}

std::uint32_t LoomSpatialBridge::status() const {
  switch (state) {
  case State::Idle:
    return 0;
  case State::Running:
  case State::WaitingForChannelCommit:
  case State::WaitingForCompletion:
    return gem5SpatialBridgeBusy;
  case State::Complete:
    return gem5SpatialBridgeDone;
  case State::Failed:
    return gem5SpatialBridgeFailed;
  }
  panic("unknown LoomSpatialBridge state");
}

Tick LoomSpatialBridge::read(PacketPtr packet) {
  const CallbackAccounting accounting = beginCallbackAccounting();
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
  finishCallbackAccounting(accounting);
  return pioDelay;
}

Tick LoomSpatialBridge::write(PacketPtr packet) {
  const CallbackAccounting accounting = beginCallbackAccounting();
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
    if (!engineSession)
      fail(3, "launch requested on a bridge without an executable session");
    else if (state != State::Idle && state != State::Complete)
      fail(3, "launch requested while the bridge is not idle");
    else if (nextSequence == std::numeric_limits<std::uint64_t>::max())
      fail(20, "bridge invocation sequence is exhausted");
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
  finishCallbackAccounting(accounting);
  return pioDelay;
}

bool LoomSpatialBridge::measuring() const {
  return threadDispatch->computationOpen();
}

std::uint64_t LoomSpatialBridge::serviceOccupancy() const {
  return memoryService->occupiedTicks();
}

std::vector<std::uint64_t> LoomSpatialBridge::acceleratedPhases() const {
  if (!invocationPhase.observed)
    return {};
  // A Bridge whose configuration was already resident when the measured
  // computation began establishes no residency inside the window, so its
  // residency phase is the empty span at the start of its invocation phase.
  const PhaseObservation residency =
      configurationResidencyPhase.observed
          ? configurationResidencyPhase
          : PhaseObservation{true, invocationPhase.beginTick,
                             invocationPhase.beginOccupiedTicks,
                             invocationPhase.beginTick,
                             invocationPhase.beginOccupiedTicks};
  return {residency.beginTick,        residency.beginOccupiedTicks,
          residency.endTick,          residency.endOccupiedTicks,
          invocationPhase.beginTick,  invocationPhase.beginOccupiedTicks,
          invocationPhase.endTick,    invocationPhase.endOccupiedTicks};
}

void LoomSpatialBridge::fetchStaticLaunch() {
  if (activeStaticLaunchSize == 0 ||
      activeStaticLaunchSize > maximumMessageBytes) {
    fail(18, "active static launch descriptor is invalid");
    return;
  }
  // The immutable plane is loaded once per configuration residency. A launch
  // naming the resident descriptor reuses those bytes and costs no residency;
  // the engine still compares them against its Deployment projection.
  if (activeStaticLaunchAddress == residentStaticLaunchAddress &&
      activeStaticLaunchSize == residentStaticLaunchSize &&
      staticLaunchPayload.size() == activeStaticLaunchSize) {
    fetchInvocation();
    return;
  }
  ++performanceStatistics.staticLaunchFetchCount;
  residentStaticLaunchAddress = activeStaticLaunchAddress;
  residentStaticLaunchSize = activeStaticLaunchSize;
  if (measuring())
    configurationResidencyPhase.openAt(curTick(), serviceOccupancy());
  // The staged launch image is the functional transport of the immutable
  // plane: a portable document whose size is an encoding choice, not a
  // hardware fact. Configuration residency costs what the Fabric-derived
  // binary configuration image costs, so exactly those bytes cross the modeled
  // memory system while the plane itself is materialized functionally.
  staticLaunchPayload.assign(activeStaticLaunchSize, 0);
  sys->physProxy.readBlob(activeStaticLaunchAddress,
                          staticLaunchPayload.data(), activeStaticLaunchSize);
  if (configurationImageBytes == 0) {
    completeConfigurationTransport();
    return;
  }
  configurationTransportBuffer.assign(configurationImageBytes, 0);
  dmaRead(activeStaticLaunchAddress, static_cast<int>(configurationImageBytes),
          &staticLaunchCompletionEvent, configurationTransportBuffer.data());
}

void LoomSpatialBridge::completeConfigurationTransport() {
  if (measuring() && configurationResidencyPhase.observed)
    configurationResidencyPhase.closeAt(curTick(), serviceOccupancy());
  fetchInvocation();
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

LoomSpatialBridge::CallbackAccounting
LoomSpatialBridge::beginCallbackAccounting() {
  if (!collectPerformance)
    return {};
  const std::optional<std::uint64_t> started = threadCpuNanoseconds();
  if (!started) {
    ++performanceStatistics.clockFailureCount;
    return {};
  }
  return CallbackAccounting{*started, true};
}

void LoomSpatialBridge::finishCallbackAccounting(
    CallbackAccounting accounting) {
  if (!accounting.valid)
    return;
  const std::optional<std::uint64_t> finished = threadCpuNanoseconds();
  if (!finished || *finished < accounting.started) {
    ++performanceStatistics.clockFailureCount;
    return;
  }
  performanceStatistics.callbackCpuNanoseconds +=
      *finished - accounting.started;
}

void LoomSpatialBridge::startEngineWait() {
  if (collectPerformance)
    engineWaitStarted = std::chrono::steady_clock::now();
}

void LoomSpatialBridge::finishEngineWait() {
  if (!engineWaitStarted)
    return;
  const auto elapsed = std::chrono::steady_clock::now() - *engineWaitStarted;
  const auto nanoseconds =
      std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count();
  if (nanoseconds > 0)
    performanceStatistics.engineWaitNanoseconds +=
        static_cast<std::uint64_t>(nanoseconds);
  engineWaitStarted.reset();
}

void LoomSpatialBridge::startLaunch() {
  if (measuring())
    invocationPhase.openAt(curTick(), serviceOccupancy());
  memorySnapshotPayload.clear();
  if (!invocationPayload.empty()) {
    loom::runtime::SpatialInvocationWire wire;
    std::string diagnostic;
    if (!loom::runtime::decodeSpatialInvocationWire(invocationPayload, wire,
                                                    diagnostic)) {
      fail(22, "dynamic invocation wire is invalid: " + diagnostic);
      return;
    }
    std::size_t snapshotBytes = 0;
    if (!loom::runtime::spatialInvocationMemorySnapshotByteCount(
            wire, snapshotBytes) ||
        snapshotBytes > maximumMessageBytes - invocationPayload.size() ||
        !launchFitsMessageLimit(activeStaticLaunchSize,
                                invocationPayload.size() + snapshotBytes,
                                maximumMessageBytes)) {
      fail(23, "invocation memory snapshot exceeds the bridge limit");
      return;
    }
    // The runtime-input snapshot seeds the engine's input identity and its
    // result diff; it is simulation bookkeeping rather than accelerator
    // traffic. Read it functionally so no simulated time is charged here.
    // Timed traffic remains the engine's own memory requests.
    memorySnapshotPayload.assign(snapshotBytes, 0);
    std::size_t cursor = 0;
    for (const loom::runtime::SpatialInvocationMemoryObject &object :
         wire.memoryObjects) {
      const std::size_t byteCount = static_cast<std::size_t>(object.byteCount);
      sys->physProxy.readBlob(object.address,
                              memorySnapshotPayload.data() + cursor, byteCount);
      cursor += byteCount;
    }
  }
  engineSession->submit(
      {loom::runtime::Gem5BridgeMessageKind::SpatialLaunch,
       bridgeSessionOrdinal, nextSequence,
       loom::runtime::encodeGem5SpatialLaunchEnvelope(
           {staticLaunchPayload, invocationPayload, memorySnapshotPayload})});
}

void LoomSpatialBridge::acceptBoundary(
    const loom::runtime::Gem5BridgeMessage &message, Tick causalTick) {
  reclaimRetiredMemory();
  ++performanceStatistics.messageCount;
  if (state != State::Running || message.sequence != nextSequence ||
      message.bridgeSessionOrdinal != bridgeSessionOrdinal ||
      causalTick != curTick()) {
    fail(7, "Spatial engine boundary has the wrong invocation or causal state");
    return;
  }
  if (message.kind == loom::runtime::Gem5BridgeMessageKind::ChannelCommit) {
    loom::runtime::Gem5BridgeChannelCommit commit;
    std::string diagnostic;
    if (!loom::runtime::decodeGem5BridgeChannelCommit(message.payload, commit,
                                                      diagnostic) ||
        commit.readyAfterTicks > MaxTick - causalTick) {
      fail(8, "Spatial channel commit has an invalid causal delay");
      return;
    }
    state = State::WaitingForChannelCommit;
    schedule(&channelCommitEvent, causalTick + commit.readyAfterTicks);
    return;
  }
  if (message.kind == loom::runtime::Gem5BridgeMessageKind::MemoryRequest ||
      message.kind == loom::runtime::Gem5BridgeMessageKind::ChannelTransfer) {
    acceptMemoryRequest(message, causalTick);
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
      invocationResult.invocation != invocationPayload ||
      invocationResult.memorySnapshot != memorySnapshotPayload) {
    fail(19, "Spatial completion names a foreign invocation");
    return;
  }
  if (pendingCompletion.readyAfterTicks > MaxTick - causalTick) {
    fail(12, "Spatial completion delay exceeds the tick domain");
    return;
  }
  state = State::WaitingForCompletion;
  schedule(&completionEvent, causalTick + pendingCompletion.readyAfterTicks);
}

void LoomSpatialBridge::reclaimRetiredMemory() {
  retiredMemoryTransactions.clear();
}

void LoomSpatialBridge::acceptMemoryRequest(
    const loom::runtime::Gem5BridgeMessage &message, Tick causalTick) {
  loom::runtime::Gem5BridgeMemoryRequest request;
  std::string diagnostic;
  if (!loom::runtime::decodeGem5BridgeMemoryRequest(message.payload, request,
                                                   diagnostic)) {
    fail(8, diagnostic);
    return;
  }
  if (message.kind == loom::runtime::Gem5BridgeMessageKind::ChannelTransfer &&
      request.operation != loom::runtime::Gem5BridgeMemoryOperation::Write) {
    fail(9, "channel transfer is not a write transaction");
    return;
  }
  if (request.size > std::numeric_limits<int>::max() ||
      request.readyAfterTicks > MaxTick - causalTick) {
    fail(10, "memory transaction is too large");
    return;
  }
  const std::uint64_t requestId = request.requestId;
  const Tick readyTick = causalTick + request.readyAfterTicks;
  auto transaction =
      std::make_unique<MemoryTransaction>(*this, std::move(request));
  auto *scheduled = transaction.get();
  if (!memoryTransactions.emplace(requestId, std::move(transaction)).second) {
    fail(14, "memory transaction identity is already in flight");
    return;
  }
  schedule(&scheduled->issueEvent, readyTick);
}

void LoomSpatialBridge::issueMemoryRequest(MemoryTransaction &transaction) {
  // DmaDevice's delay argument postpones only its completion callback. Issue
  // the request from this event so memory cannot observe it before readiness.
  if (transaction.request.operation ==
      loom::runtime::Gem5BridgeMemoryOperation::Read)
    dmaRead(transaction.request.address,
            static_cast<int>(transaction.request.size),
            &transaction.completionEvent, transaction.buffer.data());
  else
    dmaWrite(transaction.request.address,
             static_cast<int>(transaction.request.size),
             &transaction.completionEvent, transaction.buffer.data());
}

void LoomSpatialBridge::completeMemoryRequest(MemoryTransaction &transaction) {
  const loom::runtime::Gem5BridgeMemoryResponse response{
      transaction.request.requestId, true,
      transaction.request.operation ==
              loom::runtime::Gem5BridgeMemoryOperation::Read
          ? transaction.buffer
          : std::vector<std::uint8_t>{}};
  const loom::runtime::Gem5BridgeMessage message{
      loom::runtime::Gem5BridgeMessageKind::MemoryResponse,
      bridgeSessionOrdinal, nextSequence,
      loom::runtime::encodeGem5BridgeMemoryResponse(response)};
  // The event queue still owns this transaction's completion event, so its
  // storage is released at the next boundary or completion.
  reclaimRetiredMemory();
  const auto retired = memoryTransactions.find(transaction.request.requestId);
  retiredMemoryTransactions.push_back(std::move(retired->second));
  memoryTransactions.erase(retired);
  engineSession->submit(message);
}

void LoomSpatialBridge::completeChannelCommit() {
  if (state != State::WaitingForChannelCommit) {
    fail(13, "channel commit arrived in the wrong bridge state");
    return;
  }
  state = State::Running;
  engineSession->submit({loom::runtime::Gem5BridgeMessageKind::ChannelCommitted,
                        bridgeSessionOrdinal,
                        nextSequence,
                        {}});
}

LoomSpatialBridge::ResultPublication LoomSpatialBridge::publishResults() {
  if (publishedResultBytes == 0) {
    const std::vector<std::uint8_t> header =
        loom::runtime::encodeGem5BridgeResultCollection({});
    if (header.size() > maximumMessageBytes)
      return ResultPublication::TooLarge;
    std::ofstream output(resultPath, std::ios::binary | std::ios::trunc);
    if (!output)
      return ResultPublication::OpenFailed;
    output.write(reinterpret_cast<const char *>(header.data()),
                 static_cast<std::streamsize>(header.size()));
    if (!output)
      return ResultPublication::WriteFailed;
    publishedResultBytes = header.size();
    return ResultPublication::Published;
  }

  // The message limit bounds one staged result, never the collection: every
  // launch appends its own member to the result file, so the collection
  // grows with the launch count while the staging memory does not.
  const std::vector<std::uint8_t> member =
      loom::runtime::encodeGem5BridgeResult(
          {pendingCompletion.status, lastCompletionTick, nextSequence,
           std::move(pendingCompletion.result)});
  if (member.size() > maximumMessageBytes)
    return ResultPublication::TooLarge;
  std::fstream output(resultPath,
                      std::ios::binary | std::ios::in | std::ios::out);
  if (!output)
    return ResultPublication::OpenFailed;
  output.seekp(0, std::ios::end);
  if (output.tellp() != static_cast<std::streamoff>(publishedResultBytes))
    return ResultPublication::WriteFailed;
  output.write(reinterpret_cast<const char *>(member.data()),
               static_cast<std::streamsize>(member.size()));
  std::vector<std::uint8_t> count;
  count.reserve(sizeof(std::uint64_t));
  loom::runtime::detail::appendGem5BridgeU64(count,
                                             nextSequence + 1);
  output.seekp(loom::runtime::gem5BridgeResultCollectionMagic.size(),
               std::ios::beg);
  output.write(reinterpret_cast<const char *>(count.data()),
               static_cast<std::streamsize>(count.size()));
  output.flush();
  if (!output)
    return ResultPublication::WriteFailed;
  publishedResultBytes += member.size();
  return ResultPublication::Published;
}

void LoomSpatialBridge::completeInvocation() {
  lastCompletionTick = curTick();
  if (measuring() && invocationPhase.observed)
    invocationPhase.closeAt(curTick(), serviceOccupancy());
  const ResultPublication publication = publishResults();
  if (publication != ResultPublication::Published) {
    switch (publication) {
    case ResultPublication::TooLarge:
      fail(21, "normalized result collection exceeds the bridge limit");
      break;
    case ResultPublication::OpenFailed:
      fail(15, "could not create the normalized result");
      break;
    case ResultPublication::WriteFailed:
      fail(16, "could not write the normalized result");
      break;
    case ResultPublication::Published:
      panic("published result classified as a publication failure");
    }
    return;
  }
  errorCode = pendingCompletion.status;
  state = pendingCompletion.status == 0 ? State::Complete : State::Failed;
  ++performanceStatistics.invocationCount;
  ++nextSequence;
}

void LoomSpatialBridge::fail(std::uint32_t code, const std::string &message) {
  finishEngineWait();
  warn("LoomSpatialBridge failed: %s", message.c_str());
  errorCode = code;
  state = State::Failed;
}

void LoomSpatialBridge::resetBridge() {
  panic_if(dmaPending() || (state != State::Idle && state != State::Complete &&
                            state != State::Failed),
           "cannot reset LoomSpatialBridge with an active invocation");
  panic_if(!memoryTransactions.empty() && state != State::Failed,
           "cannot reset LoomSpatialBridge with in-flight memory");
  engineWaitStarted.reset();
  if (launchEvent.scheduled())
    deschedule(&launchEvent);
  if (staticLaunchCompletionEvent.scheduled())
    deschedule(&staticLaunchCompletionEvent);
  if (invocationCompletionEvent.scheduled())
    deschedule(&invocationCompletionEvent);
  if (completionEvent.scheduled())
    deschedule(&completionEvent);
  if (channelCommitEvent.scheduled())
    deschedule(&channelCommitEvent);
  for (auto &[requestId, transaction] : memoryTransactions)
    if (transaction->issueEvent.scheduled())
      deschedule(&transaction->issueEvent);
  memoryTransactions.clear();
  retiredMemoryTransactions.clear();
  // The resident immutable plane is configuration state, not launch state:
  // the AccCore resets the Bridge before every launch, and the next launch
  // reuses the image while its descriptor is unchanged.
  invocationPayload.clear();
  memorySnapshotPayload.clear();
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
