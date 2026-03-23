#include "loom/MultiCoreSim/DRAMModel.h"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <stdexcept>

namespace loom {
namespace mcsim {

DRAMModel::DRAMModel(uint64_t sizeBytes, unsigned accessLatency,
                     unsigned bandwidthBytesPerCycle)
    : sizeBytes_(sizeBytes), accessLatency_(accessLatency),
      bandwidthBytesPerCycle_(bandwidthBytesPerCycle), storage_(sizeBytes, 0) {}

uint64_t DRAMModel::submitRead(uint64_t address, uint64_t sizeBytes,
                               uint64_t globalCycle) {
  if (address + sizeBytes > sizeBytes_) {
    throw std::out_of_range("DRAM read out of bounds: address=" +
                            std::to_string(address) +
                            " size=" + std::to_string(sizeBytes));
  }

  uint64_t id = nextRequestId_++;
  uint64_t completionCycle = computeCompletionCycle(sizeBytes, globalCycle);

  PendingRequest req;
  req.requestId = id;
  req.address = address;
  req.sizeBytes = sizeBytes;
  req.isWrite = false;
  req.submitCycle = globalCycle;
  req.completionCycle = completionCycle;
  pendingRequests_.push_back(std::move(req));

  readCount_++;
  totalBytesTransferred_ += sizeBytes;
  return id;
}

uint64_t DRAMModel::submitWrite(uint64_t address,
                                const std::vector<uint8_t> &data,
                                uint64_t globalCycle) {
  if (address + data.size() > sizeBytes_) {
    throw std::out_of_range("DRAM write out of bounds: address=" +
                            std::to_string(address) +
                            " size=" + std::to_string(data.size()));
  }

  uint64_t id = nextRequestId_++;
  uint64_t completionCycle =
      computeCompletionCycle(data.size(), globalCycle);

  PendingRequest req;
  req.requestId = id;
  req.address = address;
  req.sizeBytes = data.size();
  req.isWrite = true;
  req.writeData = data;
  req.submitCycle = globalCycle;
  req.completionCycle = completionCycle;
  pendingRequests_.push_back(std::move(req));

  writeCount_++;
  totalBytesTransferred_ += data.size();
  return id;
}

bool DRAMModel::isComplete(uint64_t requestId) const {
  for (const auto &req : completedRequests_) {
    if (req.requestId == requestId) {
      return true;
    }
  }
  return false;
}

std::vector<uint8_t> DRAMModel::getReadData(uint64_t requestId) const {
  for (const auto &req : completedRequests_) {
    if (req.requestId == requestId && !req.isWrite) {
      return std::vector<uint8_t>(storage_.begin() + req.address,
                                  storage_.begin() + req.address +
                                      req.sizeBytes);
    }
  }
  return {};
}

void DRAMModel::tick(uint64_t globalCycle) {
  // Move completed requests from pending to completed.
  auto it = pendingRequests_.begin();
  while (it != pendingRequests_.end()) {
    if (globalCycle >= it->completionCycle) {
      // For writes, commit the data to storage upon completion.
      if (it->isWrite) {
        std::memcpy(storage_.data() + it->address, it->writeData.data(),
                    it->writeData.size());
      }
      completedRequests_.push_back(std::move(*it));
      it = pendingRequests_.erase(it);
    } else {
      ++it;
    }
  }

  // Prune old completed requests (keep the most recent 256).
  while (completedRequests_.size() > 256) {
    completedRequests_.pop_front();
  }
}

void DRAMModel::initialize(uint64_t baseAddr,
                            const std::vector<uint8_t> &data) {
  if (baseAddr + data.size() > sizeBytes_) {
    uint64_t copySize = sizeBytes_ - baseAddr;
    std::memcpy(storage_.data() + baseAddr, data.data(), copySize);
  } else {
    std::memcpy(storage_.data() + baseAddr, data.data(), data.size());
  }
}

std::vector<uint8_t> DRAMModel::directRead(uint64_t address,
                                           uint64_t sizeBytes) const {
  if (address + sizeBytes > sizeBytes_) {
    throw std::out_of_range("DRAM directRead out of bounds");
  }
  return std::vector<uint8_t>(storage_.begin() + address,
                              storage_.begin() + address + sizeBytes);
}

void DRAMModel::directWrite(uint64_t address,
                             const std::vector<uint8_t> &data) {
  if (address + data.size() > sizeBytes_) {
    throw std::out_of_range("DRAM directWrite out of bounds");
  }
  std::memcpy(storage_.data() + address, data.data(), data.size());
}

uint64_t DRAMModel::computeCompletionCycle(uint64_t sizeBytes,
                                           uint64_t globalCycle) {
  // Bus availability: serialization if bus is busy.
  uint64_t busAvailable =
      (globalCycle >= busBusyUntilCycle_) ? globalCycle : busBusyUntilCycle_;

  // Transfer time based on bandwidth.
  uint64_t transferCycles = (sizeBytes + bandwidthBytesPerCycle_ - 1) /
                            bandwidthBytesPerCycle_;

  // Completion = bus available + access latency + transfer time.
  uint64_t completion = busAvailable + accessLatency_ + transferCycles;

  // Update bus busy tracking.
  busBusyUntilCycle_ = busAvailable + transferCycles;

  return completion;
}

} // namespace mcsim
} // namespace loom
