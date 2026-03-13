//===-- SimTagOps.cpp - Simulated tag operations ------------------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include "loom/Simulator/SimTagOps.h"

namespace loom {
namespace sim {

//===----------------------------------------------------------------------===//
// SimAddTag
//===----------------------------------------------------------------------===//

SimAddTag::SimAddTag(unsigned tagWidth) : tagWidth_(tagWidth) {}

void SimAddTag::reset() {
  configuredTag_ = 0;
  errorValid_ = false;
  errorCode_ = RtError::OK;
  perf_ = PerfSnapshot();
}

void SimAddTag::configure(const std::vector<uint32_t> &configWords) {
  // CONFIG_WIDTH = TAG_WIDTH bits. Extract tag value.
  if (!configWords.empty()) {
    uint16_t mask = (1u << tagWidth_) - 1;
    configuredTag_ = static_cast<uint16_t>(configWords[0]) & mask;
  }
}

void SimAddTag::evaluateCombinational() {
  if (inputs.empty() || outputs.empty())
    return;

  auto *in = inputs[0];
  auto *out = outputs[0];

  // Combinational: pass through data, append tag.
  out->valid = in->valid;
  out->data = in->data;
  out->tag = configuredTag_;
  out->hasTag = true;
  in->ready = out->ready;
}

void SimAddTag::collectTraceEvents(std::vector<TraceEvent> &events,
                                   uint64_t cycle) {
  if (!outputs.empty() && outputs[0]->transferred()) {
    perf_.activeCycles++;
    perf_.tokensOut++;
    TraceEvent ev;
    ev.cycle = cycle;
    ev.hwNodeId = hwNodeId;
    ev.eventKind = EV_ROUTE_USE;
    events.push_back(ev);
  }
}

//===----------------------------------------------------------------------===//
// SimMapTag
//===----------------------------------------------------------------------===//

SimMapTag::SimMapTag(unsigned inTagWidth, unsigned outTagWidth,
                     unsigned tableSize)
    : inTagWidth_(inTagWidth), outTagWidth_(outTagWidth),
      tableSize_(tableSize) {
  table_.resize(tableSize);
}

void SimMapTag::reset() {
  for (auto &e : table_)
    e = MapEntry();
  errorValid_ = false;
  errorCode_ = RtError::OK;
  perf_ = PerfSnapshot();
}

void SimMapTag::configure(const std::vector<uint32_t> &configWords) {
  // CONFIG_WIDTH = TABLE_SIZE * (1 + IN_TAG_WIDTH + OUT_TAG_WIDTH) bits.
  // Each entry: valid(1) + srcTag(IN_TAG_WIDTH) + dstTag(OUT_TAG_WIDTH).
  unsigned entryWidth = 1 + inTagWidth_ + outTagWidth_;
  unsigned bitPos = 0;

  for (unsigned i = 0; i < tableSize_; ++i) {
    unsigned wordIdx = bitPos / 32;
    unsigned bitIdx = bitPos % 32;

    // Extract entry bits across word boundaries.
    uint64_t raw = 0;
    if (wordIdx < configWords.size())
      raw = configWords[wordIdx] >> bitIdx;
    if (bitIdx + entryWidth > 32 && wordIdx + 1 < configWords.size())
      raw |= static_cast<uint64_t>(configWords[wordIdx + 1])
             << (32 - bitIdx);

    table_[i].valid = (raw & 1) != 0;
    table_[i].srcTag =
        static_cast<uint16_t>((raw >> 1) & ((1u << inTagWidth_) - 1));
    table_[i].dstTag = static_cast<uint16_t>(
        (raw >> (1 + inTagWidth_)) & ((1u << outTagWidth_) - 1));

    bitPos += entryWidth;
  }

  // Validate: check for duplicate src_tag entries.
  for (unsigned i = 0; i < tableSize_; ++i) {
    if (!table_[i].valid)
      continue;
    for (unsigned j = i + 1; j < tableSize_; ++j) {
      if (table_[j].valid && table_[j].srcTag == table_[i].srcTag) {
        latchError(RtError::CFG_MAP_TAG_DUP_TAG);
        break;
      }
    }
  }
}

void SimMapTag::evaluateCombinational() {
  if (inputs.empty() || outputs.empty())
    return;

  auto *in = inputs[0];
  auto *out = outputs[0];

  if (!in->valid) {
    out->valid = false;
    in->ready = out->ready;
    return;
  }

  // Lookup input tag in table.
  bool found = false;
  for (unsigned i = 0; i < tableSize_; ++i) {
    if (table_[i].valid && table_[i].srcTag == in->tag) {
      out->valid = true;
      out->data = in->data;
      out->tag = table_[i].dstTag;
      out->hasTag = true;
      in->ready = out->ready;
      found = true;
      break;
    }
  }

  if (!found) {
    latchError(RtError::RT_MAP_TAG_NO_MATCH);
    out->valid = false;
    in->ready = false;
  }
}

void SimMapTag::collectTraceEvents(std::vector<TraceEvent> &events,
                                   uint64_t cycle) {
  if (!outputs.empty() && outputs[0]->transferred()) {
    perf_.activeCycles++;
    perf_.tokensOut++;
    TraceEvent ev;
    ev.cycle = cycle;
    ev.hwNodeId = hwNodeId;
    ev.eventKind = EV_ROUTE_USE;
    events.push_back(ev);
  }
  if (errorValid_) {
    TraceEvent ev;
    ev.cycle = cycle;
    ev.hwNodeId = hwNodeId;
    ev.eventKind = EV_DEVICE_ERROR;
    ev.arg0 = errorCode_;
    events.push_back(ev);
  }
}

//===----------------------------------------------------------------------===//
// SimDelTag
//===----------------------------------------------------------------------===//

SimDelTag::SimDelTag() = default;

void SimDelTag::reset() {
  errorValid_ = false;
  errorCode_ = RtError::OK;
  perf_ = PerfSnapshot();
}

void SimDelTag::configure(const std::vector<uint32_t> & /*configWords*/) {
  // CONFIG_WIDTH = 0. No configuration needed.
}

void SimDelTag::evaluateCombinational() {
  if (inputs.empty() || outputs.empty())
    return;

  auto *in = inputs[0];
  auto *out = outputs[0];

  // Combinational: pass through data, strip tag.
  out->valid = in->valid;
  out->data = in->data;
  out->tag = 0;
  out->hasTag = false;
  in->ready = out->ready;
}

void SimDelTag::collectTraceEvents(std::vector<TraceEvent> &events,
                                   uint64_t cycle) {
  if (!outputs.empty() && outputs[0]->transferred()) {
    perf_.activeCycles++;
    perf_.tokensOut++;
    TraceEvent ev;
    ev.cycle = cycle;
    ev.hwNodeId = hwNodeId;
    ev.eventKind = EV_ROUTE_USE;
    events.push_back(ev);
  }
}

} // namespace sim
} // namespace loom
