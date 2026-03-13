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
                     unsigned numRegion, uint32_t extLatency)
    : isExternal_(isExternal), ldCount_(ldCount), stCount_(stCount),
      dataWidth_(dataWidth), tagWidth_(tagWidth), addrWidth_(addrWidth),
      numRegion_(numRegion), extLatency_(extLatency),
      portOffset_(isExternal ? 1 : 0) {
  if (!isExternal_)
    onChipMem_.resize(kDefaultOnChipSize, 0);
  addrOffsetTable_.resize(numRegion_);
  pendingStDone_.resize(stCount_);
}

void SimMemory::reset() {
  if (!isExternal_)
    std::fill(onChipMem_.begin(), onChipMem_.end(), 0);
  pendingLoads_.clear();
  storePairQueues_.clear();
  storeDeadlockCounters_.clear();
  for (auto &q : pendingStDone_)
    q.clear();
  firedThisCycle_ = false;
  currentSimCycle_ = 0;
  errorValid_ = false;
  errorCode_ = RtError::OK;
  pendingError_ = false;
  pendingErrorCode_ = RtError::OK;
  perf_ = PerfSnapshot();
}

void SimMemory::configure(const std::vector<uint32_t> &configWords) {
  // CONFIG_WIDTH = numRegion * (1 + 2*TAG_WIDTH + ADDR_WIDTH) per
  // spec-fabric-mem.md. Each entry: valid(1) + start_tag(TW) + end_tag(TW)
  // + addr_offset(AW).
  unsigned entryWidth = 1 + 2 * tagWidth_ + addrWidth_;
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

  for (unsigned r = 0; r < numRegion_; ++r) {
    addrOffsetTable_[r].valid = (extractBits(1) != 0);
    addrOffsetTable_[r].startTag =
        static_cast<uint16_t>(extractBits(tagWidth_));
    addrOffsetTable_[r].endTag =
        static_cast<uint16_t>(extractBits(tagWidth_));
    addrOffsetTable_[r].addrOffset = extractBits(addrWidth_);
  }

  // Validate: empty tag ranges (end_tag <= start_tag).
  uint16_t emptyErr = isExternal_ ? RtError::CFG_EXTMEMORY_EMPTY_TAG_RANGE
                                  : RtError::CFG_MEMORY_EMPTY_TAG_RANGE;
  for (unsigned r = 0; r < numRegion_; ++r) {
    if (addrOffsetTable_[r].valid &&
        addrOffsetTable_[r].endTag <= addrOffsetTable_[r].startTag) {
      latchError(emptyErr);
    }
  }

  // Validate: overlapping tag ranges between valid regions.
  uint16_t overlapErr = isExternal_
                            ? RtError::CFG_EXTMEMORY_OVERLAP_TAG_REGION
                            : RtError::CFG_MEMORY_OVERLAP_TAG_REGION;
  for (unsigned i = 0; i < numRegion_; ++i) {
    if (!addrOffsetTable_[i].valid)
      continue;
    for (unsigned j = i + 1; j < numRegion_; ++j) {
      if (!addrOffsetTable_[j].valid)
        continue;
      // Half-open ranges [s, e) overlap if s_i < e_j && s_j < e_i.
      if (addrOffsetTable_[i].startTag < addrOffsetTable_[j].endTag &&
          addrOffsetTable_[j].startTag < addrOffsetTable_[i].endTag) {
        latchError(overlapErr);
      }
    }
  }

  (void)entryWidth;
}

uint64_t SimMemory::resolveAddr(uint64_t addr, uint16_t tag,
                                bool &matched) const {
  matched = false;
  for (unsigned r = 0; r < numRegion_; ++r) {
    if (!addrOffsetTable_[r].valid)
      continue;
    if (tag >= addrOffsetTable_[r].startTag &&
        tag < addrOffsetTable_[r].endTag) {
      matched = true;
      return addr + addrOffsetTable_[r].addrOffset;
    }
  }
  return addr;
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

  // Maximum valid tag per spec-fabric-mem.md tagging rules.
  uint16_t maxTag = static_cast<uint16_t>(
      ldCount_ > stCount_ ? ldCount_ : stCount_);

  // Port layout per spec-fabric-mem.md:
  //   Inputs:  [memref?] [ld_addr_0..L-1, st_addr_0..S-1, st_data_0..S-1]
  //   Outputs: [ld_data_0..L-1, ld_done_0..L-1, st_done_0..S-1]
  // extmemory has memref at input 0; portOffset_ skips it.

  // For extmemory, accept the memref reference input unconditionally.
  if (portOffset_ > 0 && !inputs.empty())
    inputs[0]->ready = true;

  // Handle loads.
  for (unsigned l = 0; l < ldCount_; ++l) {
    unsigned ldAddrIdx = portOffset_ + l;
    if (ldAddrIdx >= inputs.size())
      continue;
    auto *ldAddr = inputs[ldAddrIdx];
    unsigned ldDataIdx = l;
    unsigned ldDoneIdx = ldCount_ + l;

    if (ldDataIdx >= outputs.size() || ldDoneIdx >= outputs.size())
      continue;

    auto *ldData = outputs[ldDataIdx];
    auto *ldDone = outputs[ldDoneIdx];

    // Check for completed pending loads (latency model).
    bool hasPendingResult = false;
    uint64_t pendingData = 0;
    uint16_t pendingTag = 0;
    for (auto it = pendingLoads_.begin(); it != pendingLoads_.end(); ++it) {
      if (it->laneIdx == l && it->readyCycle <= currentSimCycle_) {
        hasPendingResult = true;
        pendingData = it->data;
        pendingTag = it->tag;
        break;
      }
    }

    if (hasPendingResult) {
      ldData->valid = true;
      ldData->data = pendingData;
      if (tagWidth_ > 0) {
        ldData->tag = pendingTag;
        ldData->hasTag = true;
      }
      ldDone->valid = true;
      ldDone->data = 0; // Done token carries no data payload.
      if (tagWidth_ > 0) {
        ldDone->tag = pendingTag;
        ldDone->hasTag = true;
      }
      ldAddr->ready = false; // Don't accept new addr while serving result.
    } else if (ldAddr->valid && extLatency_ == 0) {
      // Tag OOB check per spec-fabric-mem.md (only when address carries a tag).
      if (ldAddr->hasTag && tagWidth_ > 0 && ldAddr->tag >= maxTag) {
        latchError(RtError::RT_MEMORY_TAG_OOB);
        ldData->valid = false;
        ldDone->valid = false;
        ldAddr->ready = false;
        continue;
      }
      // Resolve address: use tag-based addr_offset_table when tagged,
      // otherwise use raw address directly.
      uint64_t resolvedAddr = ldAddr->data;
      if (ldAddr->hasTag && tagWidth_ > 0) {
        bool matched = false;
        resolvedAddr = resolveAddr(ldAddr->data, ldAddr->tag, matched);
        if (!matched) {
          uint16_t noMatchErr = isExternal_ ? RtError::RT_EXTMEMORY_NO_MATCH
                                            : RtError::RT_MEMORY_NO_MATCH;
          latchError(noMatchErr);
          ldData->valid = false;
          ldDone->valid = false;
          ldAddr->ready = false;
          continue;
        }
      }
      {
        uint64_t val = memRead(resolvedAddr);
        ldData->valid = true;
        ldData->data = val;
        if (ldAddr->hasTag && tagWidth_ > 0) {
          ldData->tag = ldAddr->tag;
          ldData->hasTag = true;
        }
        ldDone->valid = true;
        ldDone->data = 0;
        if (ldAddr->hasTag && tagWidth_ > 0) {
          ldDone->tag = ldAddr->tag;
          ldDone->hasTag = true;
        }
        ldAddr->ready = ldData->ready && ldDone->ready;
        firedThisCycle_ = true;
      }
    } else if (ldAddr->valid && extLatency_ > 0) {
      // Tag OOB check (only when address carries a tag).
      if (ldAddr->hasTag && tagWidth_ > 0 && ldAddr->tag >= maxTag) {
        latchError(RtError::RT_MEMORY_TAG_OOB);
        ldAddr->ready = false;
        ldData->valid = false;
        ldDone->valid = false;
        continue;
      }
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

  // Handle stores: accept staddr and stdata independently per
  // spec-fabric-mem.md store-queue pairing semantics. Each input is accepted
  // on its own; pairing happens in advanceClock via storePairQueues_.
  for (unsigned s = 0; s < stCount_; ++s) {
    unsigned stAddrIdx = portOffset_ + ldCount_ + s;
    unsigned stDataIdx = portOffset_ + ldCount_ + stCount_ + s;
    unsigned stDoneIdx = ldCount_ * 2 + s;

    if (stAddrIdx >= inputs.size() || stDataIdx >= inputs.size())
      continue;
    if (stDoneIdx >= outputs.size())
      continue;

    auto *stAddr = inputs[stAddrIdx];
    auto *stData = inputs[stDataIdx];
    auto *stDone = outputs[stDoneIdx];

    // Tag OOB check on stAddr when valid (only when tagged).
    if (stAddr->valid && stAddr->hasTag && tagWidth_ > 0 &&
        stAddr->tag >= maxTag) {
      latchError(RtError::RT_MEMORY_TAG_OOB);
      stAddr->ready = false;
    } else {
      // Accept stAddr independently if valid (will be queued in advanceClock).
      stAddr->ready = stAddr->valid;
    }

    // Accept stData independently if valid.
    if (stData->valid && stData->hasTag && tagWidth_ > 0 &&
        stData->tag >= maxTag) {
      latchError(RtError::RT_MEMORY_TAG_OOB);
      stData->ready = false;
    } else {
      stData->ready = stData->valid;
    }

    // stDone fires when a paired store completes in advanceClock. Drive it
    // from the per-lane pending-done queue.
    if (s < pendingStDone_.size() && !pendingStDone_[s].empty()) {
      stDone->valid = true;
      stDone->data = 0;
      if (tagWidth_ > 0) {
        stDone->tag = pendingStDone_[s].front().tag;
        stDone->hasTag = true;
      }
    } else {
      stDone->valid = false;
    }
  }
}

void SimMemory::advanceClock() {
  currentSimCycle_++;

  // Consume stDone tokens that were transferred by downstream (per-lane).
  for (unsigned s = 0; s < stCount_; ++s) {
    unsigned doneIdx = ldCount_ * 2 + s;
    if (doneIdx < outputs.size() && s < pendingStDone_.size() &&
        !pendingStDone_[s].empty() && outputs[doneIdx]->transferred()) {
      pendingStDone_[s].pop_front();
      perf_.tokensOut++;
    }
  }

  // Read-before-write: issue new delayed loads BEFORE processing stores,
  // so loads read the value at the beginning of the cycle per spec.
  for (unsigned l = 0; l < ldCount_; ++l) {
    unsigned ldAddrIdx = portOffset_ + l;
    if (ldAddrIdx >= inputs.size())
      continue;
    if (inputs[ldAddrIdx]->transferred() && extLatency_ > 0) {
      auto *ldAddr = inputs[ldAddrIdx];
      uint64_t resolvedAddr = ldAddr->data;
      bool resolveOk = true;
      if (ldAddr->hasTag && tagWidth_ > 0) {
        bool matched = false;
        resolvedAddr = resolveAddr(ldAddr->data, ldAddr->tag, matched);
        if (!matched) {
          uint16_t noMatchErr = isExternal_ ? RtError::RT_EXTMEMORY_NO_MATCH
                                            : RtError::RT_MEMORY_NO_MATCH;
          latchError(noMatchErr);
          resolveOk = false;
        }
      }
      if (resolveOk) {
        PendingLoad pl;
        pl.data = memRead(resolvedAddr);
        pl.tag = ldAddr->tag;
        pl.readyCycle = currentSimCycle_ + extLatency_;
        pl.laneIdx = l;
        pendingLoads_.push_back(pl);
      }
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

  // Process stores: pair staddr/stdata per tag in FIFO order, then write.
  // This comes AFTER loads to preserve read-before-write semantics.
  for (unsigned s = 0; s < stCount_; ++s) {
    unsigned stAddrIdx = portOffset_ + ldCount_ + s;
    unsigned stDataIdx = portOffset_ + ldCount_ + stCount_ + s;

    if (stAddrIdx >= inputs.size() || stDataIdx >= inputs.size())
      continue;

    bool addrIsTagged = inputs[stAddrIdx]->hasTag && tagWidth_ > 0;
    bool dataIsTagged = inputs[stDataIdx]->hasTag && tagWidth_ > 0;
    uint16_t addrTag = addrIsTagged ? inputs[stAddrIdx]->tag : 0;
    uint16_t dataTag = dataIsTagged ? inputs[stDataIdx]->tag : 0;

    // Queue arriving addr into the store pair queue.
    if (inputs[stAddrIdx]->transferred()) {
      auto &q = storePairQueues_[addrTag];
      bool paired = false;
      for (auto &entry : q) {
        if (!entry.hasAddr) {
          entry.hasAddr = true;
          entry.addr = inputs[stAddrIdx]->data;
          entry.addrHasTag = addrIsTagged;
          paired = true;
          break;
        }
      }
      if (!paired) {
        StorePairEntry e;
        e.hasAddr = true;
        e.addr = inputs[stAddrIdx]->data;
        e.addrHasTag = addrIsTagged;
        e.laneIdx = s;
        q.push_back(e);
      }
      perf_.tokensIn++;
      firedThisCycle_ = true;
    }

    // Queue arriving data into the store pair queue.
    if (inputs[stDataIdx]->transferred()) {
      auto &q = storePairQueues_[dataTag];
      bool paired = false;
      for (auto &entry : q) {
        if (!entry.hasData) {
          entry.hasData = true;
          entry.data = inputs[stDataIdx]->data;
          paired = true;
          break;
        }
      }
      if (!paired) {
        StorePairEntry e;
        e.hasData = true;
        e.data = inputs[stDataIdx]->data;
        e.laneIdx = s;
        q.push_back(e);
      }
      perf_.tokensIn++;
    }
  }

  // Commit fully paired entries from ALL store queues (not just current input
  // tag). This ensures data-only arrivals that complete a pair are committed
  // even when no stAddr transfers this cycle.
  for (auto &[tag, q] : storePairQueues_) {
    while (!q.empty() && q.front().hasAddr && q.front().hasData) {
      auto &front = q.front();
      uint64_t resolvedAddr = front.addr;
      bool writeOk = true;
      if (front.addrHasTag && tagWidth_ > 0) {
        bool matched = false;
        resolvedAddr = resolveAddr(front.addr, tag, matched);
        if (!matched) {
          uint16_t noMatchErr = isExternal_ ? RtError::RT_EXTMEMORY_NO_MATCH
                                            : RtError::RT_MEMORY_NO_MATCH;
          latchError(noMatchErr);
          writeOk = false;
        }
      }
      if (writeOk) {
        memWrite(resolvedAddr, front.data);
      }
      // Enqueue stDone token for the originating lane.
      unsigned lane = front.laneIdx;
      if (lane < pendingStDone_.size()) {
        PendingStDone done;
        done.tag = tag;
        done.laneIdx = lane;
        pendingStDone_[lane].push_back(done);
      }
      q.pop_front();
    }
  }

  // Store deadlock detection per spec-fabric-mem.md: detect when a
  // store queue front has addr but not data (or vice versa) for too long.
  for (auto &[tag, q] : storePairQueues_) {
    if (q.empty()) {
      storeDeadlockCounters_[tag] = 0;
      continue;
    }
    auto &front = q.front();
    bool imbalance = (front.hasAddr != front.hasData);
    if (imbalance) {
      storeDeadlockCounters_[tag]++;
      if (storeDeadlockCounters_[tag] >= kDeadlockTimeout)
        latchError(RtError::RT_MEMORY_STORE_DEADLOCK);
    } else {
      storeDeadlockCounters_[tag] = 0;
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
