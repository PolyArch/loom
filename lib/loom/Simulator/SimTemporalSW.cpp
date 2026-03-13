//===-- SimTemporalSW.cpp - Simulated fabric.temporal_sw -----------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include "loom/Simulator/SimTemporalSW.h"

#include <algorithm>

namespace loom {
namespace sim {

SimTemporalSW::SimTemporalSW(unsigned numInputs, unsigned numOutputs,
                               unsigned tagWidth, unsigned numRouteTable,
                               const std::vector<bool> &connectivityTable)
    : numInputs_(numInputs), numOutputs_(numOutputs), tagWidth_(tagWidth),
      numRouteTable_(numRouteTable), connectivity_(connectivityTable) {
  routeSlots_.resize(numRouteTable);
  rrPointer_.resize(numOutputs, 0);
  inputSlotMatch_.resize(numInputs, -1);
  arbWinner_.resize(numOutputs, -1);
  broadcastOk_.resize(numInputs, false);
}

void SimTemporalSW::reset() {
  // Only clear runtime state, NOT configured route slots/routings.
  // routeSlots_ and slotRoutings_ are set by configure().
  rrPointer_.assign(numOutputs_, 0);
  inputSlotMatch_.assign(numInputs_, -1);
  arbWinner_.assign(numOutputs_, -1);
  broadcastOk_.assign(numInputs_, false);
  errorValid_ = false;
  errorCode_ = RtError::OK;
  perf_ = PerfSnapshot();
}

void SimTemporalSW::configure(const std::vector<uint32_t> &configWords) {
  // Each slot: valid(1) + tag(tagWidth_) + routes(K) bits.
  // K = popcount(connectivity_table) for this numOutputs*numInputs.
  unsigned numConnected = 0;
  for (bool c : connectivity_)
    if (c)
      ++numConnected;

  unsigned slotWidth = 1 + tagWidth_ + numConnected;
  unsigned bitPos = 0;

  auto extractBits = [&](unsigned width) -> uint64_t {
    uint64_t val = 0;
    for (unsigned b = 0; b < width; ++b) {
      unsigned wordIdx = bitPos / 32;
      unsigned bit = bitPos % 32;
      if (wordIdx < configWords.size()) {
        if (configWords[wordIdx] & (1u << bit))
          val |= (1ULL << b);
      }
      ++bitPos;
    }
    return val;
  };

  for (unsigned s = 0; s < numRouteTable_; ++s) {
    routeSlots_[s].valid = (extractBits(1) != 0);
    routeSlots_[s].tag = static_cast<uint16_t>(extractBits(tagWidth_));
    routeSlots_[s].routes.resize(numConnected);
    for (unsigned r = 0; r < numConnected; ++r)
      routeSlots_[s].routes[r] = (extractBits(1) != 0);
  }

  // Validate: check for duplicate tags.
  for (unsigned i = 0; i < numRouteTable_; ++i) {
    if (!routeSlots_[i].valid)
      continue;
    for (unsigned j = i + 1; j < numRouteTable_; ++j) {
      if (routeSlots_[j].valid &&
          routeSlots_[j].tag == routeSlots_[i].tag) {
        latchError(RtError::CFG_TEMPORAL_SW_DUP_TAG);
      }
    }
  }

  // Validate: per-slot, check for multiple inputs to same output.
  rebuildSlotRouting();
  for (unsigned s = 0; s < numRouteTable_; ++s) {
    if (!routeSlots_[s].valid)
      continue;
    for (unsigned o = 0; o < numOutputs_; ++o) {
      int count = 0;
      for (unsigned i = 0; i < numInputs_; ++i) {
        // Check if input i routes to output o in this slot.
        for (unsigned target : slotRoutings_[s].inputTargets[i]) {
          if (target == o)
            count++;
        }
      }
      if (count > 1)
        latchError(RtError::CFG_TEMPORAL_SW_SAME_TAG_INPUTS);
    }
  }
}

void SimTemporalSW::rebuildSlotRouting() {
  slotRoutings_.resize(numRouteTable_);
  for (unsigned s = 0; s < numRouteTable_; ++s) {
    auto &sr = slotRoutings_[s];
    sr.outputSource.assign(numOutputs_, -1);
    sr.inputTargets.resize(numInputs_);
    for (auto &t : sr.inputTargets)
      t.clear();

    if (!routeSlots_[s].valid)
      continue;

    unsigned connIdx = 0;
    for (unsigned o = 0; o < numOutputs_; ++o) {
      for (unsigned i = 0; i < numInputs_; ++i) {
        if (!connectivity_[o * numInputs_ + i])
          continue;
        if (connIdx < routeSlots_[s].routes.size() &&
            routeSlots_[s].routes[connIdx]) {
          sr.outputSource[o] = static_cast<int>(i);
          sr.inputTargets[i].push_back(o);
        }
        ++connIdx;
      }
    }
  }
}

void SimTemporalSW::performArbitration() {
  arbWinner_.assign(numOutputs_, -1);
  broadcastOk_.assign(numInputs_, false);

  // For each output, perform round-robin arbitration among valid contenders.
  for (unsigned o = 0; o < numOutputs_; ++o) {
    unsigned start = rrPointer_[o];
    int winner = -1;

    // Scan inputs starting from RR pointer.
    for (unsigned k = 0; k < numInputs_; ++k) {
      unsigned i = (start + k) % numInputs_;
      if (i >= inputs.size() || !inputs[i]->valid)
        continue;

      // Check if this input's matched slot routes to this output.
      int slotIdx = inputSlotMatch_[i];
      if (slotIdx < 0)
        continue;

      auto &sr = slotRoutings_[static_cast<unsigned>(slotIdx)];
      bool routesToOutput = false;
      for (unsigned t : sr.inputTargets[i]) {
        if (t == o) {
          routesToOutput = true;
          break;
        }
      }
      if (routesToOutput) {
        winner = static_cast<int>(i);
        break;
      }
    }

    arbWinner_[o] = winner;
  }

  // Determine broadcast_ok for each input:
  // Input i has broadcast_ok if it won arbitration for ALL its broadcast
  // targets AND all those targets have out_ready.
  for (unsigned i = 0; i < numInputs_; ++i) {
    int slotIdx = inputSlotMatch_[i];
    if (slotIdx < 0 || !inputs[i]->valid) {
      broadcastOk_[i] = false;
      continue;
    }

    auto &sr = slotRoutings_[static_cast<unsigned>(slotIdx)];
    if (sr.inputTargets[i].empty()) {
      broadcastOk_[i] = false;
      continue;
    }

    bool ok = true;
    for (unsigned targetOut : sr.inputTargets[i]) {
      if (arbWinner_[targetOut] != static_cast<int>(i)) {
        ok = false;
        break;
      }
      if (targetOut < outputs.size() && !outputs[targetOut]->ready) {
        ok = false;
        break;
      }
    }
    broadcastOk_[i] = ok;
  }
}

void SimTemporalSW::evaluateCombinational() {
  // Tag matching: each input's tag selects a route_table slot.
  inputSlotMatch_.assign(numInputs_, -1);
  for (unsigned i = 0; i < numInputs_ && i < inputs.size(); ++i) {
    if (!inputs[i]->valid)
      continue;

    bool matched = false;
    for (unsigned s = 0; s < numRouteTable_; ++s) {
      if (routeSlots_[s].valid && routeSlots_[s].tag == inputs[i]->tag) {
        inputSlotMatch_[i] = static_cast<int>(s);
        matched = true;
        break;
      }
    }

    if (!matched)
      latchError(RtError::RT_TEMPORAL_SW_NO_MATCH);
  }

  // Perform per-output round-robin arbitration.
  performArbitration();

  // Drive output valid/data.
  for (unsigned o = 0; o < numOutputs_ && o < outputs.size(); ++o) {
    int winner = arbWinner_[o];
    if (winner >= 0 && broadcastOk_[static_cast<unsigned>(winner)]) {
      outputs[o]->valid = true;
      outputs[o]->data = inputs[static_cast<unsigned>(winner)]->data;
      outputs[o]->tag = inputs[static_cast<unsigned>(winner)]->tag;
      outputs[o]->hasTag = inputs[static_cast<unsigned>(winner)]->hasTag;
    } else {
      outputs[o]->valid = false;
    }
  }

  // Drive input ready = broadcast_ok.
  for (unsigned i = 0; i < numInputs_ && i < inputs.size(); ++i) {
    inputs[i]->ready = broadcastOk_[i];
  }

  // Check for unrouted input errors.
  for (unsigned i = 0; i < numInputs_ && i < inputs.size(); ++i) {
    if (!inputs[i]->valid)
      continue;
    int slotIdx = inputSlotMatch_[i];
    if (slotIdx < 0)
      continue; // Already raised NO_MATCH.

    auto &sr = slotRoutings_[static_cast<unsigned>(slotIdx)];
    if (sr.inputTargets[i].empty())
      latchError(RtError::RT_TEMPORAL_SW_UNROUTED_INPUT);
  }
}

void SimTemporalSW::advanceClock() {
  // Advance round-robin pointers for outputs that had successful handshakes.
  for (unsigned o = 0; o < numOutputs_ && o < outputs.size(); ++o) {
    if (outputs[o]->transferred()) {
      int winner = arbWinner_[o];
      if (winner >= 0)
        rrPointer_[o] = (static_cast<unsigned>(winner) + 1) % numInputs_;
    }
  }
}

void SimTemporalSW::collectTraceEvents(std::vector<TraceEvent> &events,
                                         uint64_t cycle) {
  bool anyFired = false;
  for (unsigned o = 0; o < outputs.size(); ++o) {
    if (outputs[o]->transferred()) {
      anyFired = true;
      perf_.tokensOut++;
      TraceEvent ev;
      ev.cycle = cycle;
      ev.hwNodeId = hwNodeId;
      ev.eventKind = EV_ROUTE_USE;
      ev.lane = static_cast<uint8_t>(o);
      ev.arg0 = static_cast<uint32_t>(arbWinner_[o]);
      events.push_back(ev);
    }
  }

  if (anyFired)
    perf_.activeCycles++;
  else
    perf_.stallCyclesIn++;

  if (errorValid_) {
    TraceEvent ev;
    ev.cycle = cycle;
    ev.hwNodeId = hwNodeId;
    ev.eventKind = EV_DEVICE_ERROR;
    ev.arg0 = errorCode_;
    events.push_back(ev);
  }
}

} // namespace sim
} // namespace loom
