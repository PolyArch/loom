#include "loom/MultiCoreSim/DMAEngine.h"
#include "loom/MultiCoreSim/DRAMModel.h"

#include <algorithm>
#include <cassert>
#include <stdexcept>

namespace loom {
namespace mcsim {

DMAEngine::DMAEngine(unsigned coreId, unsigned maxInflight,
                     unsigned setupLatency,
                     unsigned transferBandwidthBytesPerCycle)
    : coreId_(coreId), maxInflight_(maxInflight), setupLatency_(setupLatency),
      transferBandwidthBytesPerCycle_(transferBandwidthBytesPerCycle) {}

uint64_t DMAEngine::submit(const DMAOpDescriptor &op, uint64_t globalCycle) {
  if (!hasCapacity()) {
    return 0;
  }

  uint64_t id = nextDMAId_++;
  PendingDMAOp pending;
  pending.id = id;
  pending.op = op;
  pending.startCycle = globalCycle;
  pending.state = PendingDMAOp::SETUP;
  pending.bytesTransferred = 0;

  // Completion of setup phase.
  pending.completionCycle = globalCycle + setupLatency_;

  inflightOps_.push_back(std::move(pending));
  opsSubmitted_++;
  return id;
}

void DMAEngine::tick(uint64_t globalCycle, std::vector<SPMModel> &spms,
                     std::vector<L2BankModel> &l2Banks, DRAMModel &dram,
                     unsigned l2NumBanks, uint64_t l2BankSizeBytes) {
  auto it = inflightOps_.begin();
  while (it != inflightOps_.end()) {
    PendingDMAOp &op = *it;

    switch (op.state) {
    case PendingDMAOp::SETUP:
      if (globalCycle >= op.completionCycle) {
        // Transition to TRANSFERRING.
        op.state = PendingDMAOp::TRANSFERRING;
        uint64_t transferCycles = computeTransferCycles(op.op.sizeBytes);
        op.completionCycle = globalCycle + transferCycles;
      }
      break;

    case PendingDMAOp::TRANSFERRING:
      if (globalCycle >= op.completionCycle) {
        // Transfer complete: execute the actual data movement.
        executeTransfer(op, spms, l2Banks, dram, l2NumBanks, l2BankSizeBytes);
        op.state = PendingDMAOp::COMPLETE;
        op.bytesTransferred = op.op.sizeBytes;

        // Update statistics.
        totalBytesTransferred_ += op.op.sizeBytes;
        totalTransferCycles_ += (globalCycle - op.startCycle);
        opsCompleted_++;

        // Invoke completion callback if provided.
        if (op.op.completionCallback) {
          op.op.completionCallback();
        }
      }
      break;

    case PendingDMAOp::COMPLETE:
      // Will be removed below.
      break;
    }

    ++it;
  }

  // Move completed ops to the completed ID list and remove from inflight.
  auto removeIt = inflightOps_.begin();
  while (removeIt != inflightOps_.end()) {
    if (removeIt->state == PendingDMAOp::COMPLETE) {
      completedIds_.push_back(removeIt->id);
      removeIt = inflightOps_.erase(removeIt);
    } else {
      ++removeIt;
    }
  }

  // Limit completed ID history.
  while (completedIds_.size() > 1024) {
    completedIds_.pop_front();
  }
}

bool DMAEngine::isComplete(uint64_t dmaId) const {
  for (uint64_t id : completedIds_) {
    if (id == dmaId) {
      return true;
    }
  }
  return false;
}

bool DMAEngine::hasCapacity() const {
  return inflightOps_.size() < maxInflight_;
}

unsigned DMAEngine::inflightCount() const {
  return static_cast<unsigned>(inflightOps_.size());
}

uint64_t DMAEngine::computeTransferCycles(uint64_t sizeBytes) const {
  if (transferBandwidthBytesPerCycle_ == 0) {
    return 1;
  }
  return (sizeBytes + transferBandwidthBytesPerCycle_ - 1) /
         transferBandwidthBytesPerCycle_;
}

void DMAEngine::executeTransfer(PendingDMAOp &pendingOp,
                                std::vector<SPMModel> &spms,
                                std::vector<L2BankModel> &l2Banks,
                                DRAMModel &dram, unsigned l2NumBanks,
                                uint64_t l2BankSizeBytes) {
  const DMAOpDescriptor &op = pendingOp.op;

  switch (op.type) {
  case DMAOpDescriptor::SPM_TO_L2: {
    // Read block from source SPM, write to L2 banks.
    auto data = spms[op.srcCoreId].readBlock(op.srcOffset, op.sizeBytes);
    // Distribute across L2 banks based on address interleaving.
    uint64_t remaining = op.sizeBytes;
    uint64_t srcPos = 0;
    uint64_t dstAddr = op.dstOffset;
    while (remaining > 0) {
      unsigned bankIdx =
          static_cast<unsigned>((dstAddr / l2BankSizeBytes) % l2NumBanks);
      uint64_t bankOffset = dstAddr % l2BankSizeBytes;

      // Determine how many bytes go to this bank in this chunk.
      uint64_t chunkEnd =
          ((dstAddr / l2BankSizeBytes) + 1) * l2BankSizeBytes;
      uint64_t chunkSize = std::min(remaining, chunkEnd - dstAddr);

      std::vector<uint8_t> chunk(data.begin() + srcPos,
                                 data.begin() + srcPos + chunkSize);
      l2Banks[bankIdx].writeBlock(bankOffset, chunk);

      srcPos += chunkSize;
      dstAddr += chunkSize;
      remaining -= chunkSize;
    }
    break;
  }

  case DMAOpDescriptor::L2_TO_SPM: {
    // Read from L2 banks, write to destination SPM.
    std::vector<uint8_t> assembled;
    assembled.reserve(op.sizeBytes);
    uint64_t remaining = op.sizeBytes;
    uint64_t srcAddr = op.srcOffset;
    while (remaining > 0) {
      unsigned bankIdx =
          static_cast<unsigned>((srcAddr / l2BankSizeBytes) % l2NumBanks);
      uint64_t bankOffset = srcAddr % l2BankSizeBytes;

      uint64_t chunkEnd =
          ((srcAddr / l2BankSizeBytes) + 1) * l2BankSizeBytes;
      uint64_t chunkSize = std::min(remaining, chunkEnd - srcAddr);

      auto chunk = l2Banks[bankIdx].readBlock(bankOffset, chunkSize);
      assembled.insert(assembled.end(), chunk.begin(), chunk.end());

      srcAddr += chunkSize;
      remaining -= chunkSize;
    }
    spms[op.dstCoreId].writeBlock(op.dstOffset, assembled);
    break;
  }

  case DMAOpDescriptor::SPM_TO_DRAM: {
    auto data = spms[op.srcCoreId].readBlock(op.srcOffset, op.sizeBytes);
    dram.directWrite(op.dstOffset, data);
    break;
  }

  case DMAOpDescriptor::DRAM_TO_SPM: {
    auto data = dram.directRead(op.srcOffset, op.sizeBytes);
    spms[op.dstCoreId].writeBlock(op.dstOffset, data);
    break;
  }

  case DMAOpDescriptor::SPM_TO_SPM_VIA_L2: {
    // Two-phase transfer: SPM -> L2 (temporary) -> SPM.
    auto data = spms[op.srcCoreId].readBlock(op.srcOffset, op.sizeBytes);
    spms[op.dstCoreId].writeBlock(op.dstOffset, data);
    break;
  }
  }
}

} // namespace mcsim
} // namespace loom
