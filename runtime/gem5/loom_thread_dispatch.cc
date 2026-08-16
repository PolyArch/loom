#include "runtime/gem5/loom_thread_dispatch.hh"

#include "Runtime/Gem5DispatchABI.h"

#include "base/logging.hh"
#include "mem/packet.hh"
#include "runtime/gem5/loom_riscv_deployment_workload.hh"

namespace gem5 {
namespace {

using namespace loom::runtime;

} // namespace

LoomThreadDispatch::LoomThreadDispatch(const Params &params)
    : BasicPioDevice(params, gem5ThreadDispatchApertureBytes),
      workload(params.workload),
      dispatchEvent([this] { beginDispatch(); }, name() + ".dispatch"),
      completionEvent([this] { finishDispatch(); }, name() + ".completion") {
  panic_if(!workload, "LoomThreadDispatch workload is absent");
}

std::uint32_t LoomThreadDispatch::status() const {
  switch (state) {
  case State::Idle:
    return 0;
  case State::Running:
    return gem5ThreadDispatchBusy;
  case State::Complete:
    return gem5ThreadDispatchDone;
  case State::Failed:
    return gem5ThreadDispatchFailed;
  }
  panic("unknown LoomThreadDispatch state");
}

Tick LoomThreadDispatch::read(PacketPtr packet) {
  panic_if(packet->getSize() != 4,
           "LoomThreadDispatch requires 32-bit MMIO accesses");
  const Addr offset = packet->getAddr() - pioAddr;
  std::uint32_t value = 0;
  if (offset == gem5ThreadDispatchTargetLow)
    value = static_cast<std::uint32_t>(selectedTarget);
  else if (offset == gem5ThreadDispatchTargetHigh)
    value = static_cast<std::uint32_t>(selectedTarget >> 32);
  else if (offset == gem5ThreadDispatchStatus)
    value = status();
  else if (offset == gem5ThreadDispatchError)
    value = errorCode;
  else if (offset == gem5ThreadDispatchInvocationLow)
    value = static_cast<std::uint32_t>(invocationAddress);
  else if (offset == gem5ThreadDispatchInvocationHigh)
    value = static_cast<std::uint32_t>(invocationAddress >> 32);
  else if (offset == gem5ThreadDispatchInvocationSize)
    value = static_cast<std::uint32_t>(invocationSize);
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
  const bool descriptorWrite = offset == gem5ThreadDispatchTargetLow ||
                               offset == gem5ThreadDispatchTargetHigh ||
                               offset == gem5ThreadDispatchInvocationLow ||
                               offset == gem5ThreadDispatchInvocationHigh ||
                               offset == gem5ThreadDispatchInvocationSize;
  if (descriptorWrite && state == State::Running) {
    fail(2);
  } else if (offset == gem5ThreadDispatchTargetLow) {
    selectedTarget = (selectedTarget & 0xffffffff00000000ULL) | value;
  } else if (offset == gem5ThreadDispatchTargetHigh) {
    selectedTarget = (selectedTarget & 0x00000000ffffffffULL) |
                     (static_cast<std::uint64_t>(value) << 32);
  } else if (offset == gem5ThreadDispatchInvocationLow) {
    invocationAddress = (invocationAddress & 0xffffffff00000000ULL) | value;
  } else if (offset == gem5ThreadDispatchInvocationHigh) {
    invocationAddress = (invocationAddress & 0x00000000ffffffffULL) |
                        (static_cast<std::uint64_t>(value) << 32);
  } else if (offset == gem5ThreadDispatchInvocationSize) {
    invocationSize = value;
  } else if (offset == gem5ThreadDispatchControl &&
             (value & gem5ThreadDispatchReset)) {
    if (state == State::Running)
      fail(3);
    else {
      state = State::Idle;
      errorCode = 0;
      activeInvocationAddress = 0;
      activeInvocationSize = 0;
    }
  } else if (offset == gem5ThreadDispatchControl &&
             (value & gem5ThreadDispatchStart)) {
    const bool incompleteInvocation =
        (invocationAddress == 0) != (invocationSize == 0);
    if (state == State::Running || selectedTarget >= workload->targetCount() ||
        incompleteInvocation)
      fail(4);
    else {
      state = State::Running;
      errorCode = 0;
      activeInvocationAddress = invocationAddress;
      activeInvocationSize = invocationSize;
      schedule(dispatchEvent, clockEdge());
    }
  } else if (offset == gem5ThreadDispatchCompletion && value == 1) {
    if (state != State::Running)
      fail(5);
    else
      schedule(completionEvent, clockEdge(Cycles(1)) + pioDelay);
  } else if (offset == gem5ThreadDispatchWorkerFailure && value != 0) {
    if (state != State::Running)
      fail(5);
    else
      fail(value);
  } else {
    fail(6);
  }
  packet->makeAtomicResponse();
  return pioDelay;
}

void LoomThreadDispatch::beginDispatch() {
  if (state != State::Running ||
      !workload->dispatch(selectedTarget, pioAddr, activeInvocationAddress,
                          activeInvocationSize))
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
    activeInvocationAddress = 0;
    activeInvocationSize = 0;
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
