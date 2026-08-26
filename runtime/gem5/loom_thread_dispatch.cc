#include "runtime/gem5/loom_thread_dispatch.hh"

#include "Runtime/Gem5DispatchABI.h"

#include "base/logging.hh"
#include "mem/packet.hh"
#include "runtime/gem5/loom_riscv_deployment_workload.hh"
#include "sim/core.hh"

#include <limits>

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

} // namespace

LoomThreadDispatch::LoomThreadDispatch(const Params &params)
    : BasicPioDevice(params, gem5ThreadDispatchApertureBytes),
      workload(params.workload),
      records(workload ? workload->targetCount() : 0),
      rootEventTrace(params.root_event_trace_path,
                     std::ios::binary | std::ios::trunc),
      serviceEvent([this] { service(); }, name() + ".service") {
  panic_if(!workload, "LoomThreadDispatch workload is absent");
  panic_if(records.empty(), "LoomThreadDispatch has no target records");
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
}

LoomThreadDispatch::DispatchRecord *LoomThreadDispatch::selectedRecord() {
  return selectedTarget < records.size() ? &records[selectedTarget] : nullptr;
}

const LoomThreadDispatch::DispatchRecord *
LoomThreadDispatch::selectedRecord() const {
  return selectedTarget < records.size() ? &records[selectedTarget] : nullptr;
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
    } else {
      commandError =
          recordRootEvent(static_cast<Gem5RootLifecycleAction>(value)) ? 0 : 13;
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

bool LoomThreadDispatch::recordRootEvent(Gem5RootLifecycleAction action) {
  if (action == Gem5RootLifecycleAction::Start) {
    if (rootEventOccurrence != 0 || nextRootEventOccurrence == 0)
      return false;
    rootEventOccurrence = nextRootEventOccurrence++;
  } else if (rootEventOccurrence == 0) {
    return false;
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
  writeU64(rootEventTrace, rootEventEntity);
  writeU64(rootEventTrace, rootEventOccurrence);
  writeU32(rootEventTrace, static_cast<std::uint32_t>(action));
  writeU64(rootEventTrace, tick);
  writeU64(rootEventTrace, delta);
  if (action == Gem5RootLifecycleAction::Completion)
    rootEventTrace.flush();
  fatal_if(!rootEventTrace,
           "LoomThreadDispatch cannot append its root event trace");
  lastRootEventTick = tick;
  lastRootEventDelta = delta;
  hasRootEvent = true;
  return true;
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
