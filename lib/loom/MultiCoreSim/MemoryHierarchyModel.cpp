#include "loom/MultiCoreSim/MemoryHierarchyModel.h"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <stdexcept>

namespace loom {
namespace mcsim {

MemoryHierarchyModel::MemoryHierarchyModel(const MemoryHierarchyConfig &config)
    : config_(config),
      dram_(config.dramSizeBytes, config.dramAccessLatency,
            config.dramBandwidthBytesPerCycle) {
  // Create per-core SPMs.
  spms_.reserve(config.numCores);
  for (unsigned i = 0; i < config.numCores; ++i) {
    spms_.emplace_back(i, config.spmSizeBytes);
  }

  // Create L2 banks with evenly divided capacity.
  uint64_t perBankSize = config.l2SizeBytes / config.l2NumBanks;
  l2Banks_.reserve(config.l2NumBanks);
  for (unsigned i = 0; i < config.l2NumBanks; ++i) {
    l2Banks_.emplace_back(i, perBankSize, config.l2BankAccessLatency,
                          config.l2BankWidthBytes);
  }

  // Create per-core DMA engines.
  dmaEngines_.reserve(config.numCores);
  for (unsigned i = 0; i < config.numCores; ++i) {
    dmaEngines_.emplace_back(i, config.dmaMaxInflight, config.dmaSetupLatency,
                             config.dmaTransferBandwidthBytesPerCycle);
  }
}

MemoryHierarchyModel::~MemoryHierarchyModel() = default;

//===----------------------------------------------------------------------===//
// Initialization
//===----------------------------------------------------------------------===//

void MemoryHierarchyModel::initializeSPM(unsigned coreId,
                                         const std::vector<uint8_t> &data) {
  if (coreId >= spms_.size()) {
    throw std::out_of_range("Invalid coreId for SPM initialization: " +
                            std::to_string(coreId));
  }
  spms_[coreId].initialize(data);
}

void MemoryHierarchyModel::initializeL2(const std::vector<uint8_t> &data) {
  // Distribute data across banks using the same interleaving scheme.
  uint64_t perBankSize = config_.l2SizeBytes / config_.l2NumBanks;
  uint64_t remaining = data.size();
  uint64_t srcPos = 0;
  uint64_t addr = 0;

  while (remaining > 0 && addr < config_.l2SizeBytes) {
    auto [bankIdx, bankOffset] = decodeL2Address(addr);

    uint64_t chunkEnd = ((addr / perBankSize) + 1) * perBankSize;
    uint64_t chunkSize = std::min(remaining, chunkEnd - addr);
    chunkSize = std::min(chunkSize, perBankSize - bankOffset);

    std::vector<uint8_t> chunk(data.begin() + srcPos,
                               data.begin() + srcPos + chunkSize);
    l2Banks_[bankIdx].writeBlock(bankOffset, chunk);

    srcPos += chunkSize;
    addr += chunkSize;
    remaining -= chunkSize;
  }
}

void MemoryHierarchyModel::initializeDRAM(uint64_t baseAddr,
                                          const std::vector<uint8_t> &data) {
  dram_.initialize(baseAddr, data);
}

//===----------------------------------------------------------------------===//
// SPM Access
//===----------------------------------------------------------------------===//

uint64_t MemoryHierarchyModel::spmRead(unsigned coreId, uint64_t offset,
                                       unsigned sizeBytes) {
  if (coreId >= spms_.size()) {
    throw std::out_of_range("Invalid coreId for SPM read: " +
                            std::to_string(coreId));
  }
  spmReadCount_++;
  return spms_[coreId].read(offset, sizeBytes);
}

void MemoryHierarchyModel::spmWrite(unsigned coreId, uint64_t offset,
                                    uint64_t data, unsigned sizeBytes) {
  if (coreId >= spms_.size()) {
    throw std::out_of_range("Invalid coreId for SPM write: " +
                            std::to_string(coreId));
  }
  spmWriteCount_++;
  spms_[coreId].write(offset, data, sizeBytes);
}

//===----------------------------------------------------------------------===//
// L2 Access
//===----------------------------------------------------------------------===//

MemoryAccessResult MemoryHierarchyModel::l2Read(unsigned requestingCoreId,
                                                uint64_t offset,
                                                unsigned sizeBytes) {
  auto [bankIdx, bankOffset] = decodeL2Address(offset);

  unsigned latency = l2Banks_[bankIdx].read(bankOffset, sizeBytes, currentCycle_);
  l2ReadCount_++;

  MemoryAccessResult result;
  result.hit = true;
  result.latency = latency;
  result.requestId = 0;
  return result;
}

MemoryAccessResult MemoryHierarchyModel::l2Write(unsigned requestingCoreId,
                                                 uint64_t offset,
                                                 uint64_t data,
                                                 unsigned sizeBytes) {
  auto [bankIdx, bankOffset] = decodeL2Address(offset);

  unsigned latency =
      l2Banks_[bankIdx].write(bankOffset, data, sizeBytes, currentCycle_);
  l2WriteCount_++;

  MemoryAccessResult result;
  result.hit = true;
  result.latency = latency;
  result.requestId = 0;
  return result;
}

//===----------------------------------------------------------------------===//
// DMA Operations
//===----------------------------------------------------------------------===//

uint64_t MemoryHierarchyModel::submitDMA(unsigned coreId,
                                         const DMAOpDescriptor &op) {
  if (coreId >= dmaEngines_.size()) {
    throw std::out_of_range("Invalid coreId for DMA submit: " +
                            std::to_string(coreId));
  }
  return dmaEngines_[coreId].submit(op, currentCycle_);
}

bool MemoryHierarchyModel::isDMAComplete(unsigned coreId,
                                         uint64_t dmaId) const {
  if (coreId >= dmaEngines_.size()) {
    return false;
  }
  return dmaEngines_[coreId].isComplete(dmaId);
}

//===----------------------------------------------------------------------===//
// Per-Cycle Update
//===----------------------------------------------------------------------===//

void MemoryHierarchyModel::tick(uint64_t globalCycle) {
  currentCycle_ = globalCycle;

  // Advance L2 banks.
  for (auto &bank : l2Banks_) {
    bank.tick(globalCycle);
  }

  // Advance DRAM.
  dram_.tick(globalCycle);

  // Advance all DMA engines.
  uint64_t perBankSize = config_.l2SizeBytes / config_.l2NumBanks;
  for (auto &engine : dmaEngines_) {
    engine.tick(globalCycle, spms_, l2Banks_, dram_, config_.l2NumBanks,
                perBankSize);
  }
}

//===----------------------------------------------------------------------===//
// Statistics
//===----------------------------------------------------------------------===//

MemStats MemoryHierarchyModel::getStats() const {
  MemStats stats;
  stats.spmReads = spmReadCount_;
  stats.spmWrites = spmWriteCount_;
  stats.l2Reads = l2ReadCount_;
  stats.l2Writes = l2WriteCount_;

  // Aggregate L2 bank conflicts.
  for (const auto &bank : l2Banks_) {
    stats.l2BankConflicts += bank.getConflictCount();
  }

  stats.dramReads = dram_.getReadCount();
  stats.dramWrites = dram_.getWriteCount();

  // Aggregate DMA statistics across all engines.
  for (const auto &engine : dmaEngines_) {
    stats.dmaOpsSubmitted += engine.getOpsSubmitted();
    stats.dmaOpsCompleted += engine.getOpsCompleted();
    stats.dmaTotalBytesTransferred += engine.getTotalBytesTransferred();
    stats.dmaTotalCycles += engine.getTotalTransferCycles();
  }

  // Overlap ratio: fraction of DMA cycles that overlap with compute.
  // Approximation: if multiple DMA ops ran concurrently, overlap is higher.
  if (stats.dmaTotalCycles > 0 && currentCycle_ > 0) {
    stats.dmaOverlapRatio =
        1.0 -
        (static_cast<double>(stats.dmaTotalCycles) /
         static_cast<double>(currentCycle_ * config_.numCores));
    if (stats.dmaOverlapRatio < 0.0) {
      stats.dmaOverlapRatio = 0.0;
    }
  }

  return stats;
}

//===----------------------------------------------------------------------===//
// L2 Address Decoding
//===----------------------------------------------------------------------===//

std::pair<unsigned, uint64_t>
MemoryHierarchyModel::decodeL2Address(uint64_t offset) const {
  // Bank-interleaved addressing: bank = (offset / bankSize) % numBanks
  uint64_t perBankSize = config_.l2SizeBytes / config_.l2NumBanks;
  unsigned bankIdx =
      static_cast<unsigned>((offset / perBankSize) % config_.l2NumBanks);
  uint64_t bankOffset = offset % perBankSize;
  return {bankIdx, bankOffset};
}

} // namespace mcsim
} // namespace loom
