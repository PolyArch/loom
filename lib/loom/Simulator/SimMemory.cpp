//===-- SimMemory.cpp - Simulated fabric.memory/extmemory ----------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include "loom/Simulator/SimMemory.h"

#include <algorithm>
#include <cstring>

namespace loom {
namespace sim {

SimMemory::SimMemory(bool isExternal, unsigned ldCount, unsigned stCount,
                     unsigned dataWidth, unsigned tagWidth, unsigned addrWidth,
                     uint32_t extLatency)
    : isExternal_(isExternal), ldCount_(ldCount), stCount_(stCount),
      dataWidth_(dataWidth), tagWidth_(tagWidth), addrWidth_(addrWidth),
      extLatency_(extLatency) {
  if (!isExternal_)
    onChipMem_.resize(kDefaultOnChipSize, 0);
}

void SimMemory::reset() {
  if (!isExternal_)
    std::fill(onChipMem_.begin(), onChipMem_.end(), 0);
  pendingLoads_.clear();
  firedThisCycle_ = false;
  currentSimCycle_ = 0;
  errorValid_ = false;
  errorCode_ = RtError::OK;
  perf_ = PerfSnapshot();
}

void SimMemory::configure(const std::vector<uint32_t> & /*configWords*/) {
  // CONFIG_WIDTH = 0 per spec-fabric-config_mem.md.
  // addr_offset_table is handled separately if needed.
}

uint64_t SimMemory::memRead(uint64_t addr) const {
  unsigned byteWidth = (dataWidth_ + 7) / 8;
  uint64_t result = 0;

  if (!isExternal_) {
    if (addr + byteWidth <= onChipMem_.size()) {
      std::memcpy(&result, onChipMem_.data() + addr, byteWidth);
    }
  } else if (backingMem_) {
    if (addr + byteWidth <= backingMemSize_) {
      std::memcpy(&result, backingMem_ + addr, byteWidth);
    }
  }
  return result;
}

void SimMemory::memWrite(uint64_t addr, uint64_t data) {
  unsigned byteWidth = (dataWidth_ + 7) / 8;

  if (!isExternal_) {
    if (addr + byteWidth <= onChipMem_.size()) {
      std::memcpy(onChipMem_.data() + addr, &data, byteWidth);
    }
  } else if (backingMem_) {
    if (addr + byteWidth <= backingMemSize_) {
      std::memcpy(backingMem_ + addr, &data, byteWidth);
    }
  }
}

void SimMemory::evaluateCombinational() {
  firedThisCycle_ = false;

  // Simple single-port model for now.
  // Port layout for single-port (ldCount=1, stCount=1):
  //   Inputs:  [ld_addr(0), st_data(1), st_addr(2)]
  //   Outputs: [ld_data(0), ld_done(1), st_done(2)]
  //
  // Multi-port layout for ldCount=L, stCount=S:
  //   Inputs:  [ld_addr_0..L-1, st_data_0, st_addr_0, ..., st_data_{S-1}, st_addr_{S-1}]
  //   Outputs: [ld_data_0..L-1, ld_done_0..L-1, st_done_0..S-1]

  // Handle loads.
  for (unsigned l = 0; l < ldCount_ && l < inputs.size(); ++l) {
    auto *ldAddr = inputs[l];
    unsigned ldDataIdx = l;
    unsigned ldDoneIdx = ldCount_ + l;

    if (ldDataIdx >= outputs.size() || ldDoneIdx >= outputs.size())
      continue;

    auto *ldData = outputs[ldDataIdx];
    auto *ldDone = outputs[ldDoneIdx];

    // Check for completed pending loads (latency model).
    bool hasPendingResult = false;
    uint64_t pendingData = 0;
    for (auto it = pendingLoads_.begin(); it != pendingLoads_.end(); ++it) {
      if (it->laneIdx == l && it->readyCycle <= currentSimCycle_) {
        hasPendingResult = true;
        pendingData = it->data;
        // Will remove in advanceClock if consumed.
        break;
      }
    }

    if (hasPendingResult) {
      ldData->valid = true;
      ldData->data = pendingData;
      ldDone->valid = true;
      ldDone->data = 1;
      ldAddr->ready = false; // Don't accept new addr while serving result.
    } else if (ldAddr->valid && extLatency_ == 0) {
      // Zero-latency read (on-chip or extmemory with 0 latency).
      uint64_t val = memRead(ldAddr->data);
      ldData->valid = true;
      ldData->data = val;
      ldDone->valid = true;
      ldDone->data = 1;
      // Accept address when all outputs ready.
      ldAddr->ready = ldData->ready && ldDone->ready;
      firedThisCycle_ = true;
    } else if (ldAddr->valid && extLatency_ > 0) {
      // Will issue in advanceClock.
      ldData->valid = false;
      ldDone->valid = false;
      ldAddr->ready = true; // Accept the address.
    } else {
      ldData->valid = false;
      ldDone->valid = false;
      ldAddr->ready = false;
    }
  }

  // Handle stores.
  for (unsigned s = 0; s < stCount_; ++s) {
    unsigned stDataIdx = ldCount_ + s * 2;
    unsigned stAddrIdx = ldCount_ + s * 2 + 1;
    unsigned stDoneIdx = ldCount_ * 2 + s;

    if (stDataIdx >= inputs.size() || stAddrIdx >= inputs.size())
      continue;
    if (stDoneIdx >= outputs.size())
      continue;

    auto *stData = inputs[stDataIdx];
    auto *stAddr = inputs[stAddrIdx];
    auto *stDone = outputs[stDoneIdx];

    if (stData->valid && stAddr->valid) {
      stDone->valid = true;
      stDone->data = 1;
      // Accept store when done channel is ready.
      bool accept = stDone->ready;
      stData->ready = accept;
      stAddr->ready = accept;
      firedThisCycle_ = true;
    } else {
      stDone->valid = false;
      stData->ready = false;
      stAddr->ready = false;
    }
  }
}

void SimMemory::advanceClock() {
  currentSimCycle_++;

  // Process stores: write to memory.
  for (unsigned s = 0; s < stCount_; ++s) {
    unsigned stDataIdx = ldCount_ + s * 2;
    unsigned stAddrIdx = ldCount_ + s * 2 + 1;

    if (stDataIdx >= inputs.size() || stAddrIdx >= inputs.size())
      continue;

    if (inputs[stDataIdx]->transferred() && inputs[stAddrIdx]->transferred()) {
      memWrite(inputs[stAddrIdx]->data, inputs[stDataIdx]->data);
      perf_.tokensIn += 2;
    }
  }

  // Issue new loads with latency.
  for (unsigned l = 0; l < ldCount_ && l < inputs.size(); ++l) {
    if (inputs[l]->transferred() && extLatency_ > 0) {
      PendingLoad pl;
      pl.data = memRead(inputs[l]->data);
      pl.tag = inputs[l]->tag;
      pl.readyCycle = currentSimCycle_ + extLatency_;
      pl.laneIdx = l;
      pendingLoads_.push_back(pl);
      perf_.tokensIn++;
    }
  }

  // Remove consumed pending loads.
  for (unsigned l = 0; l < ldCount_; ++l) {
    unsigned ldDataIdx = l;
    if (ldDataIdx < outputs.size() && outputs[ldDataIdx]->transferred()) {
      for (auto it = pendingLoads_.begin(); it != pendingLoads_.end(); ++it) {
        if (it->laneIdx == l && it->readyCycle <= currentSimCycle_) {
          pendingLoads_.erase(it);
          perf_.tokensOut++;
          break;
        }
      }
    }
  }

  // Count zero-latency load outputs.
  if (extLatency_ == 0) {
    for (unsigned l = 0; l < ldCount_; ++l) {
      if (l < outputs.size() && outputs[l]->transferred()) {
        perf_.tokensIn++;
        perf_.tokensOut++;
      }
    }
  }
}

void SimMemory::collectTraceEvents(std::vector<TraceEvent> &events,
                                   uint64_t cycle) {
  if (firedThisCycle_) {
    perf_.activeCycles++;
    TraceEvent ev;
    ev.cycle = cycle;
    ev.hwNodeId = hwNodeId;
    ev.eventKind = EV_NODE_FIRE;
    events.push_back(ev);
  } else {
    bool anyInputValid = false;
    for (auto *in : inputs) {
      if (in && in->valid) {
        anyInputValid = true;
        break;
      }
    }
    if (anyInputValid)
      perf_.stallCyclesOut++;
    else
      perf_.stallCyclesIn++;
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

} // namespace sim
} // namespace loom
