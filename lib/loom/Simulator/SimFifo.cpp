//===-- SimFifo.cpp - Simulated fabric.fifo ------------------------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include "loom/Simulator/SimFifo.h"

namespace loom {
namespace sim {

SimFifo::SimFifo(unsigned depth, bool bypassable)
    : depth_(depth), bypassable_(bypassable), bypassed_(false) {}

void SimFifo::reset() {
  // Clear runtime state (buffer contents), NOT config (bypassed_).
  buffer_.clear();
  errorValid_ = false;
  errorCode_ = RtError::OK;
  pendingError_ = false;
  pendingErrorCode_ = RtError::OK;
  perf_ = PerfSnapshot();
}

void SimFifo::configure(const std::vector<uint32_t> &configWords) {
  // Bypassable FIFOs have 1-bit config: bypass enable.
  if (bypassable_ && !configWords.empty()) {
    bypassed_ = (configWords[0] & 1) != 0;
  }
}

void SimFifo::evaluateCombinational() {
  if (inputs.empty() || outputs.empty())
    return;

  auto *in = inputs[0];
  auto *out = outputs[0];

  if (bypassed_) {
    // Bypass mode: act as combinational wire.
    out->valid = in->valid;
    out->data = in->data;
    out->tag = in->tag;
    out->hasTag = in->hasTag;
    in->ready = out->ready;
  } else {
    // Non-bypass mode: output from head of queue.
    if (!buffer_.empty()) {
      out->valid = true;
      out->data = buffer_.front().data;
      out->tag = buffer_.front().tag;
      out->hasTag = buffer_.front().hasTag;
    } else {
      out->valid = false;
    }

    // Input ready when buffer is not full.
    in->ready = (buffer_.size() < depth_);
  }
}

void SimFifo::advanceClock() {
  if (bypassed_ || inputs.empty() || outputs.empty())
    return;

  auto *in = inputs[0];
  auto *out = outputs[0];

  // Dequeue: output consumed.
  if (out->transferred() && !buffer_.empty()) {
    buffer_.pop_front();
    perf_.tokensOut++;
  }

  // Enqueue: input accepted.
  if (in->transferred()) {
    FifoEntry entry;
    entry.data = in->data;
    entry.tag = in->tag;
    entry.hasTag = in->hasTag;
    buffer_.push_back(entry);
    perf_.tokensIn++;
  }
}

void SimFifo::collectTraceEvents(std::vector<TraceEvent> &events,
                                 uint64_t cycle) {
  if (outputs.empty())
    return;

  bool fired = outputs[0]->transferred();

  if (fired) {
    perf_.activeCycles++;
    TraceEvent ev;
    ev.cycle = cycle;
    ev.hwNodeId = hwNodeId;
    ev.eventKind = EV_ROUTE_USE;
    events.push_back(ev);
  } else {
    // Stall analysis.
    if (!inputs.empty() && inputs[0]->valid && !inputs[0]->ready)
      perf_.stallCyclesOut++; // Full, backpressuring
    else if (outputs[0]->valid && !outputs[0]->ready)
      perf_.stallCyclesOut++;
    else
      perf_.stallCyclesIn++;
  }
}

} // namespace sim
} // namespace loom
