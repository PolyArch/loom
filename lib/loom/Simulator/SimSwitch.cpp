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

  // Precompute per-input physical connectivity.
  hasPhysicalConnectivity_.resize(numInputs_, false);
  for (unsigned i = 0; i < numInputs_; ++i) {
    for (unsigned o = 0; o < numOutputs_; ++o) {
      if (connectivity_[o * numInputs_ + i]) {
        hasPhysicalConnectivity_[i] = true;
        break;
      }
    }
  }
}

void SimSwitch::reset() {
  // Only clear runtime state, NOT configured routing.
  // routeTable_, outputSource_, inputTargets_ are set by configure().
  errorValid_ = false;
  errorCode_ = RtError::OK;
  pendingError_ = false;
  pendingErrorCode_ = RtError::OK;
  perf_ = PerfSnapshot();
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
  // Note: switch does NOT implement atomic broadcast at the routing level.
  // Atomic broadcast semantics are the responsibility of the producer PE
  // (stream, gate, etc.) via registered outputs. The switch simply
  // propagates valid from input to output combinationally.
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
      // Per spec-fabric-switch.md: an input with physical connectivity but
      // no enabled route that receives a valid token raises
      // RT_SWITCH_UNROUTED_INPUT.  Do NOT consume (ready=false).
      inputs[i]->ready = false;
      if (inputs[i]->valid && hasPhysicalConnectivity_[i])
        latchError(RtError::RT_SWITCH_UNROUTED_INPUT);
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

bool SimSwitch::inputHasRoute(unsigned portIdx) const {
  if (portIdx >= inputTargets_.size())
    return false;
  return !inputTargets_[portIdx].empty();
}

void SimSwitch::auditRoutes(const std::vector<bool> &connectedInputs,
                            std::vector<AuditDiagnostic> &diags) const {
  unsigned unroutedCount = 0;
  unsigned routedCount = 0;

  for (unsigned i = 0; i < numInputs_ && i < connectedInputs.size(); ++i) {
    if (!connectedInputs[i])
      continue; // Not connected by an ADG edge -- skip.
    if (i < inputTargets_.size() && !inputTargets_[i].empty()) {
      ++routedCount;
      continue;
    }
    ++unroutedCount;
  }

  // Report a summary diagnostic if any connected inputs lack routes.
  // Physical mesh edges create many connections that the mapper may not use
  // for a given mapping, so this is a warning rather than an error. The
  // truly problematic case (DFG-mapped edge with no route) requires
  // mapper-level audit info which is not available at the simulator level.
  if (unroutedCount > 0) {
    AuditDiagnostic d;
    d.level = AuditDiagnostic::Warning;
    d.hwNodeId = hwNodeId;
    d.moduleName = name;
    d.message = std::to_string(unroutedCount) +
                " of " + std::to_string(unroutedCount + routedCount) +
                " connected input(s) have no configured route. Routed: " +
                std::to_string(routedCount) + ".";
    diags.push_back(std::move(d));
  }
}

} // namespace sim
} // namespace loom
