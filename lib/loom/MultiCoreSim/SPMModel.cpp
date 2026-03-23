#include "loom/MultiCoreSim/SPMModel.h"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <stdexcept>

namespace loom {
namespace mcsim {

SPMModel::SPMModel(unsigned coreId, uint64_t sizeBytes)
    : coreId_(coreId), sizeBytes_(sizeBytes), storage_(sizeBytes, 0) {}

uint64_t SPMModel::read(uint64_t offset, unsigned sizeBytes) const {
  if (offset + sizeBytes > sizeBytes_) {
    throw std::out_of_range("SPM read out of bounds: core=" +
                            std::to_string(coreId_) +
                            " offset=" + std::to_string(offset) +
                            " size=" + std::to_string(sizeBytes) +
                            " capacity=" + std::to_string(sizeBytes_));
  }

  uint64_t value = 0;
  std::memcpy(&value, storage_.data() + offset, sizeBytes);
  return value;
}

void SPMModel::write(uint64_t offset, uint64_t data, unsigned sizeBytes) {
  if (offset + sizeBytes > sizeBytes_) {
    throw std::out_of_range("SPM write out of bounds: core=" +
                            std::to_string(coreId_) +
                            " offset=" + std::to_string(offset) +
                            " size=" + std::to_string(sizeBytes) +
                            " capacity=" + std::to_string(sizeBytes_));
  }

  std::memcpy(storage_.data() + offset, &data, sizeBytes);

  uint64_t endAddr = offset + sizeBytes;
  if (endAddr > highWaterMark_) {
    highWaterMark_ = endAddr;
  }
}

std::vector<uint8_t> SPMModel::readBlock(uint64_t offset,
                                         uint64_t sizeBytes) const {
  if (offset + sizeBytes > sizeBytes_) {
    throw std::out_of_range("SPM readBlock out of bounds: core=" +
                            std::to_string(coreId_) +
                            " offset=" + std::to_string(offset) +
                            " size=" + std::to_string(sizeBytes) +
                            " capacity=" + std::to_string(sizeBytes_));
  }

  return std::vector<uint8_t>(storage_.begin() + offset,
                              storage_.begin() + offset + sizeBytes);
}

void SPMModel::writeBlock(uint64_t offset, const std::vector<uint8_t> &data) {
  if (offset + data.size() > sizeBytes_) {
    throw std::out_of_range(
        "SPM writeBlock out of bounds: core=" + std::to_string(coreId_) +
        " offset=" + std::to_string(offset) +
        " size=" + std::to_string(data.size()) +
        " capacity=" + std::to_string(sizeBytes_));
  }

  std::memcpy(storage_.data() + offset, data.data(), data.size());

  uint64_t endAddr = offset + data.size();
  if (endAddr > highWaterMark_) {
    highWaterMark_ = endAddr;
  }
}

void SPMModel::initialize(const std::vector<uint8_t> &data) {
  uint64_t copySize = std::min(static_cast<uint64_t>(data.size()), sizeBytes_);
  std::memcpy(storage_.data(), data.data(), copySize);
  if (copySize > highWaterMark_) {
    highWaterMark_ = copySize;
  }
}

} // namespace mcsim
} // namespace loom
