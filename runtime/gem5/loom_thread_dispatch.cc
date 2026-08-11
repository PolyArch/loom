#include "runtime/gem5/loom_thread_dispatch.hh"

#include "base/logging.hh"
#include "mem/packet.hh"
#include "runtime/gem5/loom_riscv_deployment_workload.hh"

namespace gem5 {
namespace {

constexpr Addr targetLowRegister = 0x00;
constexpr Addr targetHighRegister = 0x04;
constexpr Addr controlRegister = 0x08;
constexpr Addr statusRegister = 0x0c;
constexpr Addr completionRegister = 0x10;
constexpr Addr errorRegister = 0x14;
constexpr std::uint32_t controlStart = 1u << 0;
constexpr std::uint32_t controlReset = 1u << 1;
constexpr std::uint32_t statusBusy = 1u << 0;
constexpr std::uint32_t statusDone = 1u << 1;
constexpr std::uint32_t statusError = 1u << 2;
constexpr Addr dispatchApertureBytes = 0x1000;

} // namespace

LoomThreadDispatch::LoomThreadDispatch(const Params &params)
    : BasicPioDevice(params, dispatchApertureBytes), workload(params.workload),
      dispatchEvent([this] { beginDispatch(); }, name() + ".dispatch"),
      completionEvent([this] { finishDispatch(); }, name() + ".completion") {
  panic_if(!workload, "LoomThreadDispatch workload is absent");
}

std::uint32_t LoomThreadDispatch::status() const {
  switch (state) {
  case State::Idle:
    return 0;
  case State::Running:
    return statusBusy;
  case State::Complete:
    return statusDone;
  case State::Failed:
    return statusError;
  }
  panic("unknown LoomThreadDispatch state");
}

Tick LoomThreadDispatch::read(PacketPtr packet) {
  panic_if(packet->getSize() != 4,
           "LoomThreadDispatch requires 32-bit MMIO accesses");
  const Addr offset = packet->getAddr() - pioAddr;
  std::uint32_t value = 0;
  if (offset == targetLowRegister)
    value = static_cast<std::uint32_t>(selectedTarget);
  else if (offset == targetHighRegister)
    value = static_cast<std::uint32_t>(selectedTarget >> 32);
  else if (offset == statusRegister)
    value = status();
  else if (offset == errorRegister)
    value = errorCode;
  else
    fail(1);
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
  if (offset == targetLowRegister) {
    if (state == State::Running)
      fail(2);
    else
      selectedTarget = (selectedTarget & 0xffffffff00000000ULL) | value;
  } else if (offset == targetHighRegister) {
    if (state == State::Running)
      fail(2);
    else
      selectedTarget = (selectedTarget & 0x00000000ffffffffULL) |
                       (static_cast<std::uint64_t>(value) << 32);
  } else if (offset == controlRegister && (value & controlReset)) {
    if (state == State::Running)
      fail(3);
    else {
      state = State::Idle;
      errorCode = 0;
    }
  } else if (offset == controlRegister && (value & controlStart)) {
    if (state == State::Running || selectedTarget >= workload->targetCount())
      fail(4);
    else {
      state = State::Running;
      errorCode = 0;
      schedule(dispatchEvent, clockEdge());
    }
  } else if (offset == completionRegister && value == 1) {
    if (state != State::Running)
      fail(5);
    else
      schedule(completionEvent, clockEdge(Cycles(1)) + pioDelay);
  } else {
    fail(6);
  }
  packet->makeAtomicResponse();
  return pioDelay;
}

void LoomThreadDispatch::beginDispatch() {
  if (state != State::Running ||
      !workload->dispatch(selectedTarget, pioAddr))
    fail(7);
}

void LoomThreadDispatch::finishDispatch() {
  if (state != State::Running) {
    fail(8);
    return;
  }
  switch (workload->complete(selectedTarget)) {
  case LoomRiscvDeploymentWorkload::CompletionState::Pending:
    schedule(completionEvent, clockEdge(Cycles(1)));
    return;
  case LoomRiscvDeploymentWorkload::CompletionState::Complete:
    state = State::Complete;
    return;
  case LoomRiscvDeploymentWorkload::CompletionState::Invalid:
    fail(8);
    return;
  }
}

void LoomThreadDispatch::fail(std::uint32_t code) {
  errorCode = code;
  state = State::Failed;
}

} // namespace gem5
