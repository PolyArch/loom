#include "fcc/Simulator/CycleKernel.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <vector>

namespace fcc {
namespace sim {

namespace {

bool simDebugEnabled() {
  const char *env = std::getenv("FCC_SIM_DEBUG");
  return env && env[0] != '\0' && env[0] != '0';
}

} // namespace

void CycleKernel::setUseExternalMemoryService(bool enable) {
  externalMemoryMode_ = enable;
  outgoingMemoryRequests_.clear();
}

std::vector<MemoryRequestRecord> CycleKernel::drainOutgoingMemoryRequests() {
  std::vector<MemoryRequestRecord> drained;
  drained.swap(outgoingMemoryRequests_);
  return drained;
}

void CycleKernel::pushMemoryCompletion(const MemoryCompletion &completion) {
  auto it = std::find_if(
      outstandingMemoryRequests_.begin(), outstandingMemoryRequests_.end(),
      [&](const OutstandingMemoryRequest &request) {
        return request.requestId == completion.requestId;
      });
  if (it != outstandingMemoryRequests_.end())
    outstandingMemoryRequests_.erase(it);
  completedMemoryRequests_[completion.requestId] = completion;
  MemoryRegionPerfState &regionPerf = memoryRegionPerf_[completion.regionId];
  regionPerf.hasLastCompletionCycle = true;
  regionPerf.lastCompletionCycle = currentCycle_;
  if (completion.kind == MemoryRequestKind::Store &&
      completion.regionId < completedStoreRegions_.size())
    completedStoreRegions_[completion.regionId] = 1;
}

unsigned CycleKernel::getMemoryLatencyCycles(bool isExtMemory) const {
  return isExtMemory ? config_.extMemLatency : 1u;
}

bool CycleKernel::checkMemoryAccess(unsigned regionId, uint64_t byteAddr,
                                    unsigned byteWidth,
                                    std::string &error) const {
  if (regionId >= boundMemoryRegions_.size()) {
    error = "memory region out of range";
    return false;
  }
  const BoundMemoryRegion &region = boundMemoryRegions_[regionId];
  if (!region.data) {
    if (region.external && externalMemoryMode_) {
      if (byteAddr < region.baseByteAddr) {
        std::ostringstream oss;
        oss << "memory region " << regionId << " OOB at byte " << byteAddr
            << " below base " << region.baseByteAddr;
        error = oss.str();
        return false;
      }
      uint64_t relative = byteAddr - region.baseByteAddr;
      if (relative + byteWidth > region.sizeBytes) {
        std::ostringstream oss;
        oss << "memory region " << regionId << " OOB at byte " << byteAddr
            << " outside [" << region.baseByteAddr << ", "
            << (region.baseByteAddr + region.sizeBytes) << ")";
        error = oss.str();
        return false;
      }
      return true;
    }
    std::ostringstream oss;
    oss << "memory region " << regionId << " is not bound";
    error = oss.str();
    return false;
  }
  if (byteAddr + byteWidth > region.sizeBytes) {
    std::ostringstream oss;
    oss << "memory region " << regionId << " OOB at byte " << byteAddr;
    error = oss.str();
    return false;
  }
  return true;
}

bool CycleKernel::issueMemoryLoad(uint32_t ownerNodeId, unsigned regionId,
                                  uint64_t byteAddr, unsigned byteWidth,
                                  uint16_t tag, bool hasTag,
                                  uint64_t &requestId, std::string &error) {
  if (!checkMemoryAccess(regionId, byteAddr, byteWidth, error))
    return false;
  requestId = nextMemoryRequestId_++;
  outstandingMemoryRequests_.push_back({requestId,
                                        currentCycle_ + config_.extMemLatency,
                                        MemoryRequestKind::Load,
                                        regionId,
                                        ownerNodeId,
                                        byteAddr,
                                        0,
                                        byteWidth,
                                        tag,
                                        hasTag});
  if (externalMemoryMode_) {
    outgoingMemoryRequests_.push_back({requestId, MemoryRequestKind::Load,
                                       regionId, ownerNodeId, byteAddr, 0,
                                       byteWidth, tag, hasTag});
  }
  ++loadRequestCount_;
  loadBytes_ += byteWidth;
  MemoryRegionPerfState &regionPerf = memoryRegionPerf_[regionId];
  ++regionPerf.loadRequestCount;
  regionPerf.loadBytes += byteWidth;
  if (!regionPerf.hasFirstRequestCycle) {
    regionPerf.hasFirstRequestCycle = true;
    regionPerf.firstRequestCycle = currentCycle_;
  }
  maxInflightMemoryRequests_ =
      std::max<uint64_t>(maxInflightMemoryRequests_,
                         outstandingMemoryRequests_.size());
  return true;
}

bool CycleKernel::issueMemoryStore(uint32_t ownerNodeId, unsigned regionId,
                                   uint64_t byteAddr, uint64_t data,
                                   unsigned byteWidth, uint16_t tag,
                                   bool hasTag, uint64_t &requestId,
                                   std::string &error) {
  if (!checkMemoryAccess(regionId, byteAddr, byteWidth, error))
    return false;
  requestId = nextMemoryRequestId_++;
  outstandingMemoryRequests_.push_back({requestId,
                                        currentCycle_ + config_.extMemLatency,
                                        MemoryRequestKind::Store,
                                        regionId,
                                        ownerNodeId,
                                        byteAddr,
                                        data,
                                        byteWidth,
                                        tag,
                                        hasTag});
  if (externalMemoryMode_) {
    outgoingMemoryRequests_.push_back({requestId, MemoryRequestKind::Store,
                                       regionId, ownerNodeId, byteAddr, data,
                                       byteWidth, tag, hasTag});
  }
  ++storeRequestCount_;
  storeBytes_ += byteWidth;
  MemoryRegionPerfState &regionPerf = memoryRegionPerf_[regionId];
  ++regionPerf.storeRequestCount;
  regionPerf.storeBytes += byteWidth;
  if (!regionPerf.hasFirstRequestCycle) {
    regionPerf.hasFirstRequestCycle = true;
    regionPerf.firstRequestCycle = currentCycle_;
  }
  maxInflightMemoryRequests_ =
      std::max<uint64_t>(maxInflightMemoryRequests_,
                         outstandingMemoryRequests_.size());
  return true;
}

bool CycleKernel::takeMemoryCompletion(uint64_t requestId,
                                       MemoryCompletion &completion) {
  auto it = completedMemoryRequests_.find(requestId);
  if (it == completedMemoryRequests_.end())
    return false;
  completion = it->second;
  completedMemoryRequests_.erase(it);
  return true;
}

bool CycleKernel::hasOutstandingMemoryRequest(uint64_t requestId) const {
  return std::any_of(outstandingMemoryRequests_.begin(),
                     outstandingMemoryRequests_.end(),
                     [&](const OutstandingMemoryRequest &request) {
                       return request.requestId == requestId;
                     });
}

bool CycleKernel::regionHasOutstandingRequests(unsigned regionId) const {
  return std::any_of(outstandingMemoryRequests_.begin(),
                     outstandingMemoryRequests_.end(),
                     [&](const OutstandingMemoryRequest &request) {
                       return request.regionId == regionId;
                     });
}

bool CycleKernel::bindMemoryRegion(unsigned regionId, uint8_t *data,
                                   size_t sizeBytes, std::string &error) {
  if (boundMemoryRegions_.size() <= regionId)
    boundMemoryRegions_.resize(regionId + 1);
  boundMemoryRegions_[regionId].data = data;
  boundMemoryRegions_[regionId].baseByteAddr = 0;
  boundMemoryRegions_[regionId].sizeBytes = sizeBytes;
  boundMemoryRegions_[regionId].external = false;
  (void)error;
  return true;
}

void CycleKernel::retireReadyMemoryRequests() {
  if (externalMemoryMode_)
    return;
  std::vector<size_t> readyIndices;
  readyIndices.reserve(outstandingMemoryRequests_.size());
  for (size_t idx = 0; idx < outstandingMemoryRequests_.size(); ++idx) {
    if (outstandingMemoryRequests_[idx].readyCycle <= currentCycle_)
      readyIndices.push_back(idx);
  }
  for (size_t reverse = readyIndices.size(); reverse > 0; --reverse) {
    OutstandingMemoryRequest request =
        outstandingMemoryRequests_[readyIndices[reverse - 1]];
    outstandingMemoryRequests_.erase(outstandingMemoryRequests_.begin() +
                                     static_cast<std::ptrdiff_t>(
                                         readyIndices[reverse - 1]));

    MemoryCompletion completion;
    completion.requestId = request.requestId;
    completion.kind = request.kind;
    completion.regionId = request.regionId;
    completion.ownerNodeId = request.ownerNodeId;
    completion.tag = request.tag;
    completion.hasTag = request.hasTag;

    BoundMemoryRegion &region = boundMemoryRegions_[request.regionId];
    if (request.kind == MemoryRequestKind::Load) {
      uint64_t value = 0;
      for (unsigned byte = 0; byte < request.byteWidth; ++byte) {
        value |= uint64_t(region.data[request.byteAddr + byte]) << (byte * 8);
      }
      completion.data = value;
    } else {
      for (unsigned byte = 0; byte < request.byteWidth; ++byte) {
        region.data[request.byteAddr + byte] = static_cast<uint8_t>(
            (request.data >> (byte * 8)) & 0xffu);
      }
      completion.data = 0;
      if (request.regionId < completedStoreRegions_.size())
        completedStoreRegions_[request.regionId] = 1;
    }
    if (completion.ownerNodeId == 735 || simDebugEnabled()) {
      std::cerr << "CycleKernel retire mem req=" << completion.requestId
                   << " owner=" << completion.ownerNodeId
                   << " kind="
                   << (completion.kind == MemoryRequestKind::Load ? "load"
                                                                  : "store")
                   << " region=" << completion.regionId
                   << " byteAddr=" << request.byteAddr
                   << " bytes=" << request.byteWidth
                   << " data=" << completion.data
                   << " cycle=" << currentCycle_ << "\n";
    }
    completedMemoryRequests_[completion.requestId] = completion;
    MemoryRegionPerfState &regionPerf = memoryRegionPerf_[completion.regionId];
    regionPerf.hasLastCompletionCycle = true;
    regionPerf.lastCompletionCycle = currentCycle_;
  }
}

} // namespace sim
} // namespace fcc
