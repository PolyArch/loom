#include "runtime/gem5/loom_thread_dispatch.hh"

#include "Runtime/Gem5DispatchABI.h"

#include "base/logging.hh"
#include "mem/packet.hh"
#include "runtime/gem5/loom_riscv_deployment_workload.hh"
#include "sim/core.hh"

#include <array>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <poll.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/un.h>
#include <unistd.h>

namespace gem5 {
namespace {

using namespace loom::runtime;

void writeU32(std::ostream &output, std::uint32_t value) {
  for (int shift = 24; shift >= 0; shift -= 8)
    output.put(static_cast<char>(value >> shift));
}

void writeU64(std::ostream &output, std::uint64_t value) {
  for (int shift = 56; shift >= 0; shift -= 8)
    output.put(static_cast<char>(value >> shift));
}

void writeU32(std::uint8_t *output, std::uint32_t value) {
  for (int shift = 24; shift >= 0; shift -= 8)
    *output++ = static_cast<std::uint8_t>(value >> shift);
}

void writeU64(std::uint8_t *output, std::uint64_t value) {
  for (int shift = 56; shift >= 0; shift -= 8)
    *output++ = static_cast<std::uint8_t>(value >> shift);
}

std::uint32_t readU32(const std::uint8_t *input) {
  std::uint32_t value = 0;
  for (std::size_t index = 0; index != 4; ++index)
    value = (value << 8) | input[index];
  return value;
}

std::uint64_t readU64(const std::uint8_t *input) {
  std::uint64_t value = 0;
  for (std::size_t index = 0; index != 8; ++index)
    value = (value << 8) | input[index];
  return value;
}

using ControlDeadline = std::chrono::steady_clock::time_point;

int remainingMilliseconds(ControlDeadline deadline) {
  const auto remaining = deadline - std::chrono::steady_clock::now();
  if (remaining <= std::chrono::steady_clock::duration::zero())
    return 0;
  const auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(
      remaining);
  if (milliseconds.count() <= 0)
    return 1;
  if (milliseconds.count() > std::numeric_limits<int>::max())
    return std::numeric_limits<int>::max();
  return static_cast<int>(milliseconds.count());
}

bool waitForControlIo(int socket, short events, ControlDeadline deadline) {
  pollfd descriptor{socket, events, 0};
  while (true) {
    const int timeout = remainingMilliseconds(deadline);
    if (timeout == 0) {
      errno = ETIMEDOUT;
      return false;
    }
    const int result = ::poll(&descriptor, 1, timeout);
    if (result > 0) {
      if (descriptor.revents & (POLLERR | POLLHUP | POLLNVAL)) {
        errno = ECONNRESET;
        return false;
      }
      if (descriptor.revents & events)
        return true;
      continue;
    }
    if (result == 0) {
      errno = ETIMEDOUT;
      return false;
    }
    if (errno == EINTR)
      continue;
    return false;
  }
}

bool connectWithDeadline(int socket, const sockaddr *address,
                         socklen_t addressLength, ControlDeadline deadline) {
  const int originalFlags = ::fcntl(socket, F_GETFL, 0);
  if (originalFlags < 0)
    return false;
  if (::fcntl(socket, F_SETFL, originalFlags | O_NONBLOCK) != 0)
    return false;
  const auto restoreFlags = [&] {
    return ::fcntl(socket, F_SETFL, originalFlags) == 0;
  };
  if (::connect(socket, address, addressLength) == 0)
    return restoreFlags();
  if (errno != EINPROGRESS)
    return false;
  if (!waitForControlIo(socket, POLLOUT, deadline)) {
    (void)restoreFlags();
    return false;
  }
  int error = 0;
  socklen_t errorLength = sizeof(error);
  if (::getsockopt(socket, SOL_SOCKET, SO_ERROR, &error, &errorLength) != 0) {
    (void)restoreFlags();
    return false;
  }
  if (!restoreFlags())
    return false;
  if (error != 0) {
    errno = error;
    return false;
  }
  return true;
}

bool sendAll(int socket, const std::uint8_t *bytes, std::size_t size,
             ControlDeadline deadline) {
  while (size != 0) {
    if (!waitForControlIo(socket, POLLOUT, deadline))
      return false;
    const ssize_t sent =
        ::send(socket, bytes, size, MSG_NOSIGNAL | MSG_DONTWAIT);
    if (sent < 0 && errno == EINTR)
      continue;
    if (sent < 0 && (errno == EAGAIN || errno == EWOULDBLOCK))
      continue;
    if (sent <= 0)
      return false;
    bytes += sent;
    size -= static_cast<std::size_t>(sent);
  }
  return true;
}

bool receiveAll(int socket, std::uint8_t *bytes, std::size_t size,
                ControlDeadline deadline) {
  while (size != 0) {
    if (!waitForControlIo(socket, POLLIN, deadline))
      return false;
    const ssize_t received = ::recv(socket, bytes, size, MSG_DONTWAIT);
    if (received < 0 && errno == EINTR)
      continue;
    if (received < 0 && (errno == EAGAIN || errno == EWOULDBLOCK))
      continue;
    if (received <= 0)
      return false;
    bytes += received;
    size -= static_cast<std::size_t>(received);
  }
  return true;
}

} // namespace

LoomThreadDispatch::LoomThreadDispatch(const Params &params)
    : BasicPioDevice(params, gem5ThreadDispatchApertureBytes),
      workload(params.workload),
      logicalTargetCount(params.logical_target_count),
      endpointTargetOffsets(params.endpoint_target_offsets),
      endpointDispatchEnabled(params.endpoint_dispatch_enabled),
      records(workload ? workload->targetCount() : 0),
      rootEventTrace(params.root_event_trace_path,
                     std::ios::binary | std::ios::trunc),
      serviceEvent([this] { service(); }, name() + ".service") {
  panic_if(!workload, "LoomThreadDispatch workload is absent");
  panic_if(records.empty(), "LoomThreadDispatch has no target records");
  panic_if(logicalTargetCount == 0,
           "LoomThreadDispatch has no logical targets");
  panic_if(endpointTargetOffsets.empty(),
           "LoomThreadDispatch has no runtime endpoint");
  panic_if(endpointDispatchEnabled.size() != endpointTargetOffsets.size(),
           "LoomThreadDispatch endpoint tables differ in cardinality");
  for (std::size_t endpoint = 0; endpoint != endpointTargetOffsets.size();
       ++endpoint) {
    const std::uint64_t offset = endpointTargetOffsets[endpoint];
    panic_if(offset > records.size() ||
                 logicalTargetCount > records.size() - offset,
             "LoomThreadDispatch endpoint target range is invalid");
    panic_if(endpointDispatchEnabled[endpoint] > 1,
             "LoomThreadDispatch endpoint dispatch flag is invalid");
  }
  panic_if(records.size() > gem5MaximumDynamicSpatialInvocations,
           "LoomThreadDispatch target count exceeds the Runtime ABI bound");
  fatal_if(params.root_event_trace_path.empty(),
           "LoomThreadDispatch root event trace path is empty");
  fatal_if(!rootEventTrace,
           "LoomThreadDispatch cannot create root event trace %s",
           params.root_event_trace_path);
  writeU32(rootEventTrace, gem5RootLifecycleTraceMagic);
  rootEventTrace.flush();
  fatal_if(!rootEventTrace,
           "LoomThreadDispatch cannot initialize root event trace %s",
           params.root_event_trace_path);
  if (!params.root_event_control_path.empty()) {
    sockaddr_un address{};
    address.sun_family = AF_UNIX;
    fatal_if(params.root_event_control_path.size() >= sizeof(address.sun_path),
             "LoomThreadDispatch root event control path is too long");
    std::memcpy(address.sun_path, params.root_event_control_path.c_str(),
                params.root_event_control_path.size() + 1);
    rootEventControlSocket =
        ::socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
    fatal_if(rootEventControlSocket < 0,
             "LoomThreadDispatch cannot create its root event control socket: %s",
             std::strerror(errno));
    const timeval timeout{
        static_cast<time_t>(gem5RootEventControlTimeoutMilliseconds / 1000),
        static_cast<suseconds_t>(
            (gem5RootEventControlTimeoutMilliseconds % 1000) * 1000)};
    fatal_if(::setsockopt(rootEventControlSocket, SOL_SOCKET, SO_SNDTIMEO,
                         &timeout, sizeof(timeout)) != 0 ||
                 ::setsockopt(rootEventControlSocket, SOL_SOCKET, SO_RCVTIMEO,
                              &timeout, sizeof(timeout)) != 0,
             "LoomThreadDispatch cannot bound root event control I/O: %s",
             std::strerror(errno));
    const ControlDeadline deadline =
        std::chrono::steady_clock::now() +
        std::chrono::milliseconds(gem5RootEventControlTimeoutMilliseconds);
    fatal_if(!connectWithDeadline(
                 rootEventControlSocket,
                 reinterpret_cast<const sockaddr *>(&address),
                 sizeof(address), deadline),
             "LoomThreadDispatch cannot connect root event control %s: %s",
             params.root_event_control_path, std::strerror(errno));
  }
}

LoomThreadDispatch::~LoomThreadDispatch() {
  if (rootEventControlSocket >= 0)
    ::close(rootEventControlSocket);
}

std::optional<std::size_t> LoomThreadDispatch::selectedRecordOrdinal() const {
  if (activeEndpoint >= endpointTargetOffsets.size() ||
      endpointDispatchEnabled[activeEndpoint] == 0 ||
      selectedTarget >= logicalTargetCount)
    return std::nullopt;
  const std::uint64_t ordinal =
      endpointTargetOffsets[activeEndpoint] + selectedTarget;
  if (ordinal >= records.size())
    return std::nullopt;
  return static_cast<std::size_t>(ordinal);
}

LoomThreadDispatch::DispatchRecord *LoomThreadDispatch::selectedRecord() {
  const auto ordinal = selectedRecordOrdinal();
  return ordinal ? &records[*ordinal] : nullptr;
}

const LoomThreadDispatch::DispatchRecord *
LoomThreadDispatch::selectedRecord() const {
  const auto ordinal = selectedRecordOrdinal();
  return ordinal ? &records[*ordinal] : nullptr;
}

std::uint32_t LoomThreadDispatch::status(const DispatchRecord &record) const {
  switch (record.state) {
  case State::Idle:
    return 0;
  case State::Queued:
  case State::Running:
  case State::Finishing:
    return gem5ThreadDispatchBusy;
  case State::Complete:
    return gem5ThreadDispatchDone;
  case State::Failed:
    return gem5ThreadDispatchFailed;
  }
  panic("unknown LoomThreadDispatch record state");
}

std::uint32_t LoomThreadDispatch::status() const {
  const DispatchRecord *record = selectedRecord();
  return record ? status(*record) : gem5ThreadDispatchFailed;
}

Tick LoomThreadDispatch::read(PacketPtr packet) {
  panic_if(packet->getSize() != 4,
           "LoomThreadDispatch requires 32-bit MMIO accesses");
  const Addr offset = packet->getAddr() - pioAddr;
  const DispatchRecord *record = selectedRecord();
  std::uint32_t value = 0;
  if (offset == gem5ThreadDispatchTargetLow)
    value = static_cast<std::uint32_t>(selectedTarget);
  else if (offset == gem5ThreadDispatchTargetHigh)
    value = static_cast<std::uint32_t>(selectedTarget >> 32);
  else if (offset == gem5ThreadDispatchStatus)
    value = status();
  else if (offset == gem5ThreadDispatchOccurrenceLow)
    value = record ? static_cast<std::uint32_t>(record->occurrence) : 0;
  else if (offset == gem5ThreadDispatchError)
    value = record ? record->errorCode : commandError;
  else if (offset == gem5ThreadDispatchInvocationLow)
    value = static_cast<std::uint32_t>(invocationAddress);
  else if (offset == gem5ThreadDispatchInvocationHigh)
    value = static_cast<std::uint32_t>(invocationAddress >> 32);
  else if (offset == gem5ThreadDispatchInvocationSize)
    value = static_cast<std::uint32_t>(invocationSize);
  else if (offset == gem5ThreadDispatchOccurrenceHigh)
    value = record ? static_cast<std::uint32_t>(record->occurrence >> 32) : 0;
  else if (offset == gem5ThreadDispatchRootOccurrenceLow)
    value = static_cast<std::uint32_t>(rootEventOccurrence);
  else if (offset == gem5ThreadDispatchRootOccurrenceHigh)
    value = static_cast<std::uint32_t>(rootEventOccurrence >> 32);
  else if (offset == gem5ThreadDispatchRootEventStatus)
    value = static_cast<std::uint32_t>(rootEventStatus);
  else
    commandError = 1;
  packet->setUintX(value, ByteOrder::little);
  packet->makeAtomicResponse();
  return pioDelay;
}

Tick LoomThreadDispatch::write(PacketPtr packet) {
  panic_if(packet->getSize() != 4,
           "LoomThreadDispatch requires 32-bit MMIO accesses");
  const Addr offset = packet->getAddr() - pioAddr;
  const std::uint32_t value =
      static_cast<std::uint32_t>(packet->getUintX(ByteOrder::little));

  if (offset >= gem5ThreadDispatchWorkerSlotBase) {
    const Addr workerOffset = offset - gem5ThreadDispatchWorkerSlotBase;
    const std::uint64_t target =
        workerOffset / gem5ThreadDispatchWorkerSlotStride;
    const Addr slotOffset = workerOffset % gem5ThreadDispatchWorkerSlotStride;
    if (target >= records.size()) {
      commandError = 9;
    } else {
      DispatchRecord &record = records[target];
      if (record.state != State::Running) {
        record.errorCode = 10;
        record.state = State::Failed;
      } else if (slotOffset == gem5ThreadDispatchWorkerCompletion &&
                 value == 1) {
        record.state = State::Finishing;
        scheduleService();
      } else if (slotOffset == gem5ThreadDispatchWorkerFailure && value != 0) {
        record.workerFailed = true;
        record.errorCode = value;
        record.state = State::Finishing;
        scheduleService();
      } else {
        record.errorCode = 11;
        record.state = State::Failed;
      }
    }
  } else if (offset == gem5ThreadDispatchTargetLow) {
    selectedTarget = (selectedTarget & 0xffffffff00000000ULL) | value;
    commandError = 0;
  } else if (offset == gem5ThreadDispatchTargetHigh) {
    selectedTarget = (selectedTarget & 0x00000000ffffffffULL) |
                     (static_cast<std::uint64_t>(value) << 32);
    commandError = 0;
  } else if (offset == gem5ThreadDispatchInvocationLow) {
    invocationAddress = (invocationAddress & 0xffffffff00000000ULL) | value;
  } else if (offset == gem5ThreadDispatchInvocationHigh) {
    invocationAddress = (invocationAddress & 0x00000000ffffffffULL) |
                        (static_cast<std::uint64_t>(value) << 32);
  } else if (offset == gem5ThreadDispatchInvocationSize) {
    invocationSize = value;
  } else if (offset == gem5ThreadDispatchRootEntityLow) {
    rootEventEntity = (rootEventEntity & 0xffffffff00000000ULL) | value;
  } else if (offset == gem5ThreadDispatchRootEntityHigh) {
    rootEventEntity = (rootEventEntity & 0x00000000ffffffffULL) |
                      (static_cast<std::uint64_t>(value) << 32);
  } else if (offset == gem5ThreadDispatchRootOccurrenceLow) {
    rootEventOccurrence = (rootEventOccurrence & 0xffffffff00000000ULL) | value;
  } else if (offset == gem5ThreadDispatchRootOccurrenceHigh) {
    rootEventOccurrence = (rootEventOccurrence & 0x00000000ffffffffULL) |
                          (static_cast<std::uint64_t>(value) << 32);
  } else if (offset == gem5ThreadDispatchRootEvent) {
    if (value >
        static_cast<std::uint32_t>(Gem5RootLifecycleAction::Completion)) {
      commandError = 12;
      rootEventStatus = Gem5RootEventStatus::InvalidEvent;
    } else {
      rootEventStatus =
          recordRootEvent(static_cast<Gem5RootLifecycleAction>(value));
      fatal_if(rootEventStatus == Gem5RootEventStatus::ProtocolFailure,
               "LoomThreadDispatch root event control failed");
      commandError =
          rootEventStatus == Gem5RootEventStatus::Acknowledged ? 0 : 13;
    }
  } else if (offset == gem5ThreadDispatchControl &&
             (value & gem5ThreadDispatchReset)) {
    DispatchRecord *record = selectedRecord();
    if (!record) {
      commandError = 3;
    } else if (record->state == State::Queued ||
               record->state == State::Running ||
               record->state == State::Finishing) {
      failSelected(3);
    } else {
      *record = DispatchRecord{};
      commandError = 0;
    }
  } else if (offset == gem5ThreadDispatchControl &&
             (value & gem5ThreadDispatchStart)) {
    DispatchRecord *record = selectedRecord();
    const bool incompleteInvocation =
        (invocationAddress == 0) != (invocationSize == 0);
    if (!record || record->state != State::Idle || incompleteInvocation ||
        nextOccurrence == 0) {
      failSelected(4);
    } else {
      record->state = State::Queued;
      record->occurrence = nextOccurrence++;
      record->invocationAddress = invocationAddress;
      record->invocationSize = invocationSize;
      record->errorCode = 0;
      record->workerFailed = false;
      commandError = 0;
      scheduleService();
    }
  } else {
    failSelected(6);
  }
  packet->makeAtomicResponse();
  return pioDelay;
}

Gem5RootEventStatus
LoomThreadDispatch::acknowledgeRootEvent(Gem5RootLifecycleAction action,
                                        std::uint64_t occurrence, Tick tick,
                                        std::uint64_t delta,
                                        std::uint64_t &generation,
                                        Gem5RootEventControlDecision &decision,
                                        std::uint64_t &endpoint) {
  generation = 0;
  endpoint = activeEndpoint;
  decision = action == Gem5RootLifecycleAction::Start
                 ? Gem5RootEventControlDecision::Continue
                 : Gem5RootEventControlDecision::Stay;
  if (rootEventControlSocket < 0)
    return Gem5RootEventStatus::Acknowledged;
  if (nextRootEventControlGeneration == 0)
    return Gem5RootEventStatus::ProtocolFailure;

  generation = nextRootEventControlGeneration++;
  const ControlDeadline deadline =
      std::chrono::steady_clock::now() +
      std::chrono::milliseconds(gem5RootEventControlTimeoutMilliseconds);
  std::array<std::uint8_t, gem5RootEventControlRequestBytes> request{};
  writeU32(request.data(), gem5RootEventControlRequestMagic);
  writeU64(request.data() + 4, generation);
  writeU64(request.data() + 12, rootEventEntity);
  writeU64(request.data() + 20, occurrence);
  writeU32(request.data() + 28, static_cast<std::uint32_t>(action));
  writeU64(request.data() + 32, tick);
  writeU64(request.data() + 40, delta);
  std::array<std::uint8_t, gem5RootEventControlAckBytes> response{};
  if (!sendAll(rootEventControlSocket, request.data(), request.size(),
               deadline) ||
      !receiveAll(rootEventControlSocket, response.data(), response.size(),
                  deadline) ||
      readU32(response.data()) != gem5RootEventControlAckMagic ||
      readU64(response.data() + 4) != generation)
    return Gem5RootEventStatus::ProtocolFailure;

  const std::uint32_t encodedDecision = readU32(response.data() + 12);
  if (encodedDecision >
      static_cast<std::uint32_t>(Gem5RootEventControlDecision::Reject))
    return Gem5RootEventStatus::ProtocolFailure;
  decision = static_cast<Gem5RootEventControlDecision>(encodedDecision);
  endpoint = readU64(response.data() + 16);
  if (endpoint >= endpointTargetOffsets.size() ||
      decision == Gem5RootEventControlDecision::Reject)
    return Gem5RootEventStatus::ProtocolFailure;
  if (action == Gem5RootLifecycleAction::Start &&
      decision != Gem5RootEventControlDecision::Continue)
    return Gem5RootEventStatus::ProtocolFailure;
  if (action == Gem5RootLifecycleAction::Completion &&
      decision != Gem5RootEventControlDecision::Stay &&
      decision != Gem5RootEventControlDecision::ActivateEndpoint)
    return Gem5RootEventStatus::ProtocolFailure;
  if (decision != Gem5RootEventControlDecision::ActivateEndpoint &&
      endpoint != activeEndpoint)
    return Gem5RootEventStatus::ProtocolFailure;
  if (decision == Gem5RootEventControlDecision::ActivateEndpoint) {
    for (const DispatchRecord &record : records)
      if (record.state == State::Queued || record.state == State::Running ||
          record.state == State::Finishing)
        return Gem5RootEventStatus::ProtocolFailure;
    activeEndpoint = endpoint;
  }
  return Gem5RootEventStatus::Acknowledged;
}

Gem5RootEventStatus
LoomThreadDispatch::recordRootEvent(Gem5RootLifecycleAction action) {
  std::uint64_t occurrence = rootEventOccurrence;
  if (action == Gem5RootLifecycleAction::Start) {
    if (rootEventOccurrence != 0 || nextRootEventOccurrence == 0)
      return Gem5RootEventStatus::InvalidEvent;
    occurrence = nextRootEventOccurrence;
  } else if (rootEventOccurrence == 0) {
    return Gem5RootEventStatus::InvalidEvent;
  }
  const Tick tick = curTick();
  panic_if(hasRootEvent && tick < lastRootEventTick,
           "LoomThreadDispatch root event time moved backwards");
  std::uint64_t delta = 0;
  if (hasRootEvent && tick == lastRootEventTick) {
    panic_if(lastRootEventDelta == std::numeric_limits<std::uint64_t>::max(),
             "LoomThreadDispatch root event delta overflow");
    delta = lastRootEventDelta + 1;
  }
  std::uint64_t acknowledgementGeneration = 0;
  Gem5RootEventControlDecision decision =
      Gem5RootEventControlDecision::Reject;
  std::uint64_t endpoint = 0;
  const Gem5RootEventStatus acknowledgement = acknowledgeRootEvent(
      action, occurrence, tick, delta, acknowledgementGeneration, decision,
      endpoint);
  if (acknowledgement != Gem5RootEventStatus::Acknowledged)
    return acknowledgement;
  if (action == Gem5RootLifecycleAction::Start) {
    rootEventOccurrence = occurrence;
    ++nextRootEventOccurrence;
  }
  writeU64(rootEventTrace, rootEventEntity);
  writeU64(rootEventTrace, occurrence);
  writeU32(rootEventTrace, static_cast<std::uint32_t>(action));
  writeU64(rootEventTrace, tick);
  writeU64(rootEventTrace, delta);
  writeU64(rootEventTrace, acknowledgementGeneration);
  writeU32(rootEventTrace, static_cast<std::uint32_t>(decision));
  writeU64(rootEventTrace, endpoint);
  rootEventTrace.flush();
  fatal_if(!rootEventTrace,
           "LoomThreadDispatch cannot append its root event trace");
  lastRootEventTick = tick;
  lastRootEventDelta = delta;
  hasRootEvent = true;
  return Gem5RootEventStatus::Acknowledged;
}

void LoomThreadDispatch::scheduleService() {
  if (!serviceEvent.scheduled())
    schedule(serviceEvent, clockEdge(Cycles(1)) + pioDelay);
}

void LoomThreadDispatch::service() {
  bool pending = false;
  for (std::size_t target = 0; target < records.size(); ++target) {
    DispatchRecord &record = records[target];
    if (record.state != State::Finishing)
      continue;
    switch (workload->complete(target)) {
    case LoomRiscvDeploymentWorkload::CompletionState::Pending:
      pending = true;
      break;
    case LoomRiscvDeploymentWorkload::CompletionState::Complete:
      record.state = record.workerFailed ? State::Failed : State::Complete;
      record.invocationAddress = 0;
      record.invocationSize = 0;
      break;
    case LoomRiscvDeploymentWorkload::CompletionState::Invalid:
      record.errorCode = 8;
      record.state = State::Failed;
      break;
    }
  }
  for (std::size_t target = 0; target < records.size(); ++target) {
    DispatchRecord &record = records[target];
    if (record.state != State::Queued)
      continue;
    const Addr completionAddress = pioAddr + gem5ThreadDispatchWorkerSlotBase +
                                   target * gem5ThreadDispatchWorkerSlotStride;
    switch (workload->dispatch(target, completionAddress,
                               record.invocationAddress,
                               record.invocationSize)) {
    case LoomRiscvDeploymentWorkload::DispatchState::Started:
      record.state = State::Running;
      break;
    case LoomRiscvDeploymentWorkload::DispatchState::Busy:
      break;
    case LoomRiscvDeploymentWorkload::DispatchState::Invalid:
      record.errorCode = 7;
      record.state = State::Failed;
      break;
    }
  }
  if (pending)
    scheduleService();
}

void LoomThreadDispatch::failSelected(std::uint32_t code) {
  DispatchRecord *record = selectedRecord();
  if (!record) {
    commandError = code;
    return;
  }
  record->errorCode = code;
  record->state = State::Failed;
}

} // namespace gem5
