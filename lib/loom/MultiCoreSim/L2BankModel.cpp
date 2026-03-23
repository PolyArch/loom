#include "loom/MultiCoreSim/L2BankModel.h"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <stdexcept>

namespace loom {
namespace mcsim {

L2BankModel::L2BankModel(unsigned bankId, uint64_t sizeBytes,
                         unsigned accessLatency, unsigned widthBytes)
    : bankId_(bankId), sizeBytes_(sizeBytes), accessLatency_(accessLatency),
      widthBytes_(widthBytes), storage_(sizeBytes, 0) {}

unsigned L2BankModel::read(uint64_t bankOffset, unsigned sizeBytes,
                           uint64_t globalCycle) {
  if (bankOffset + sizeBytes > sizeBytes_) {
    throw std::out_of_range("L2 bank read out of bounds: bank=" +
                            std::to_string(bankId_) +
                            " offset=" + std::to_string(bankOffset) +
                            " size=" + std::to_string(sizeBytes));
  }

  accessCount_++;
  return computeAccessLatency(globalCycle);
}

unsigned L2BankModel::write(uint64_t bankOffset, uint64_t data,
                            unsigned sizeBytes, uint64_t globalCycle) {
  if (bankOffset + sizeBytes > sizeBytes_) {
    throw std::out_of_range("L2 bank write out of bounds: bank=" +
                            std::to_string(bankId_) +
                            " offset=" + std::to_string(bankOffset) +
                            " size=" + std::to_string(sizeBytes));
  }

  std::memcpy(storage_.data() + bankOffset, &data, sizeBytes);
  accessCount_++;
  return computeAccessLatency(globalCycle);
}

std::vector<uint8_t> L2BankModel::readBlock(uint64_t bankOffset,
                                            uint64_t sizeBytes) const {
  if (bankOffset + sizeBytes > sizeBytes_) {
    throw std::out_of_range("L2 bank readBlock out of bounds: bank=" +
                            std::to_string(bankId_) +
                            " offset=" + std::to_string(bankOffset) +
                            " size=" + std::to_string(sizeBytes));
  }

  return std::vector<uint8_t>(storage_.begin() + bankOffset,
                              storage_.begin() + bankOffset + sizeBytes);
}

void L2BankModel::writeBlock(uint64_t bankOffset,
                             const std::vector<uint8_t> &data) {
  if (bankOffset + data.size() > sizeBytes_) {
    throw std::out_of_range(
        "L2 bank writeBlock out of bounds: bank=" + std::to_string(bankId_) +
        " offset=" + std::to_string(bankOffset) +
        " size=" + std::to_string(data.size()));
  }

  std::memcpy(storage_.data() + bankOffset, data.data(), data.size());
}

bool L2BankModel::isBusy(uint64_t globalCycle) const {
  return globalCycle < busyUntilCycle_;
}

void L2BankModel::tick(uint64_t /*globalCycle*/) {
  // Bank state is managed via busyUntilCycle_; no per-cycle work needed.
}

void L2BankModel::initialize(const std::vector<uint8_t> &data) {
  uint64_t copySize = std::min(static_cast<uint64_t>(data.size()), sizeBytes_);
  std::memcpy(storage_.data(), data.data(), copySize);
}

unsigned L2BankModel::computeAccessLatency(uint64_t globalCycle) {
  unsigned latency = accessLatency_;

  if (globalCycle < busyUntilCycle_) {
    // Bank conflict: must wait until the bank is free, then add access latency.
    unsigned conflictDelay =
        static_cast<unsigned>(busyUntilCycle_ - globalCycle);
    latency += conflictDelay;
    conflictCount_++;
  }

  // Mark the bank as busy from when the access starts until it completes.
  uint64_t accessStartCycle =
      (globalCycle >= busyUntilCycle_) ? globalCycle : busyUntilCycle_;
  busyUntilCycle_ = accessStartCycle + accessLatency_;

  return latency;
}

} // namespace mcsim
} // namespace loom
