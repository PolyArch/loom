//===-- SimSwitch.cpp - Simulated fabric.switch --------------------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include "loom/Simulator/SimSwitch.h"

#include <cassert>

namespace loom {
namespace sim {

SimSwitch::SimSwitch(unsigned numInputs, unsigned numOutputs,
                     const std::vector<bool> &connectivityTable)
    : numInputs_(numInputs), numOutputs_(numOutputs),
      connectivity_(connectivityTable) {
  assert(connectivity_.size() == numOutputs_ * numInputs_);
  outputSource_.resize(numOutputs_, -1);
  inputTargets_.resize(numInputs_);
}

void SimSwitch::reset() {
  errorValid_ = false;
  errorCode_ = RtError::OK;
  perf_ = PerfSnapshot();
  outputSource_.assign(numOutputs_, -1);
  for (auto &t : inputTargets_)
    t.clear();
  routeTable_.clear();
}

void SimSwitch::configure(const std::vector<uint32_t> &configWords) {
  // Route table bits: one bit per connected position in connectivity table.
  // Extract route enables from config words.
  unsigned numConnected = 0;
  for (bool c : connectivity_)
    if (c)
      ++numConnected;

  routeTable_.resize(numConnected, false);
  for (unsigned i = 0; i < numConnected && i < configWords.size() * 32; ++i) {
    unsigned wordIdx = i / 32;
    unsigned bit = i % 32;
    if (wordIdx < configWords.size())
      routeTable_[i] = (configWords[wordIdx] >> bit) & 1;
  }

  rebuildRouting();

  // Validate: check for multiple inputs routed to same output.
  for (unsigned o = 0; o < numOutputs_; ++o) {
    int foundInput = -1;
    unsigned connIdx = 0;
    for (unsigned i = 0; i < numInputs_; ++i) {
      if (!connectivity_[o * numInputs_ + i])
        continue;
      if (routeTable_[connIdx]) {
        if (foundInput >= 0) {
          latchError(RtError::CFG_SWITCH_ROUTE_MIX);
          break;
        }
        foundInput = static_cast<int>(i);
      }
      ++connIdx;
    }
  }
}

void SimSwitch::rebuildRouting() {
  outputSource_.assign(numOutputs_, -1);
  for (auto &t : inputTargets_)
    t.clear();

  unsigned connIdx = 0;
  for (unsigned o = 0; o < numOutputs_; ++o) {
    for (unsigned i = 0; i < numInputs_; ++i) {
      if (!connectivity_[o * numInputs_ + i])
        continue;
      if (connIdx < routeTable_.size() && routeTable_[connIdx]) {
        outputSource_[o] = static_cast<int>(i);
        if (i < inputTargets_.size())
          inputTargets_[i].push_back(o);
      }
      ++connIdx;
    }
  }
}

void SimSwitch::evaluateCombinational() {
  // Per spec-fabric-switch.md:
  // out_valid[j] = in_valid[source] (no out_ready dependency)
  // in_ready[i] = AND(out_ready[k]) for all broadcast targets k of input i

  // Drive output valid/data from routed input.
  for (unsigned o = 0; o < numOutputs_ && o < outputs.size(); ++o) {
    int src = outputSource_[o];
    if (src >= 0 && static_cast<unsigned>(src) < inputs.size()) {
      outputs[o]->valid = inputs[src]->valid;
      outputs[o]->data = inputs[src]->data;
      outputs[o]->tag = inputs[src]->tag;
      outputs[o]->hasTag = inputs[src]->hasTag;
    } else {
      outputs[o]->valid = false;
    }
  }

  // Drive input ready: AND of all broadcast targets' ready signals.
  for (unsigned i = 0; i < numInputs_ && i < inputs.size(); ++i) {
    if (inputTargets_[i].empty()) {
      // No route for this input.
      inputs[i]->ready = false;

      // Check for unrouted input error: valid token on unrouted input.
      if (inputs[i]->valid) {
        // Only raise error if there is physical connectivity but no route.
        bool hasConnectivity = false;
        for (unsigned o = 0; o < numOutputs_; ++o) {
          if (connectivity_[o * numInputs_ + i]) {
            hasConnectivity = true;
            break;
          }
        }
        if (hasConnectivity)
          latchError(RtError::RT_SWITCH_UNROUTED_INPUT);
      }
    } else {
      bool allReady = true;
      for (unsigned targetOut : inputTargets_[i]) {
        if (targetOut < outputs.size() && !outputs[targetOut]->ready)
          allReady = false;
      }
      inputs[i]->ready = allReady;
    }
  }
}

void SimSwitch::collectTraceEvents(std::vector<TraceEvent> &events,
                                   uint64_t cycle) {
  // Emit EV_ROUTE_USE for each active routing path.
  for (unsigned o = 0; o < numOutputs_ && o < outputs.size(); ++o) {
    if (outputs[o]->transferred()) {
      TraceEvent ev;
      ev.cycle = cycle;
      ev.hwNodeId = hwNodeId;
      ev.eventKind = EV_ROUTE_USE;
      ev.lane = static_cast<uint8_t>(o);
      ev.arg0 = static_cast<uint32_t>(outputSource_[o]);
      events.push_back(ev);
      perf_.tokensOut++;
    }
  }

  // Count active/stall cycles.
  bool anyActive = false;
  for (unsigned o = 0; o < outputs.size(); ++o) {
    if (outputs[o]->transferred())
      anyActive = true;
  }
  if (anyActive)
    perf_.activeCycles++;

  // Check error condition.
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
