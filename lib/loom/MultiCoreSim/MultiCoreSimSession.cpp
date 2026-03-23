#include "loom/MultiCoreSim/MultiCoreSimSession.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <sstream>

namespace loom {
namespace mcsim {

//===----------------------------------------------------------------------===//
// Construction
//===----------------------------------------------------------------------===//

MultiCoreSimSession::MultiCoreSimSession(const MultiCoreSimConfig &config)
    : config_(config) {}

MultiCoreSimSession::~MultiCoreSimSession() = default;

//===----------------------------------------------------------------------===//
// Build phase
//===----------------------------------------------------------------------===//

std::string MultiCoreSimSession::addCore(const CoreSpec &spec) {
  if (coreNameToIdx_.count(spec.name))
    return "duplicate core name: '" + spec.name + "'";

  unsigned idx = static_cast<unsigned>(cores_.size());
  auto wrapper =
      std::make_unique<CoreSimWrapper>(spec.name, idx, spec.coreType);

  std::string err = wrapper->build(spec.model, spec.configBlob);
  if (!err.empty())
    return err;

  coreNameToIdx_[spec.name] = idx;
  cores_.push_back(std::move(wrapper));
  return {};
}

std::string MultiCoreSimSession::addInterCoreRoute(const InterCoreRoute &route) {
  // Validate that producer and consumer cores exist.
  if (!coreNameToIdx_.count(route.producerCoreName))
    return "unknown producer core: '" + route.producerCoreName + "'";
  if (!coreNameToIdx_.count(route.consumerCoreName))
    return "unknown consumer core: '" + route.consumerCoreName + "'";

  routes_.push_back(route);

  // Register NoC port mappings on the producer and consumer cores.
  unsigned prodIdx = coreNameToIdx_[route.producerCoreName];
  unsigned consIdx = coreNameToIdx_[route.consumerCoreName];

  cores_[prodIdx]->mapBoundaryToNoCPort(route.producerNoCPortIdx,
                                        route.producerNoCPortIdx,
                                        /*isOutput=*/true);
  cores_[consIdx]->mapBoundaryToNoCPort(route.consumerNoCPortIdx,
                                        route.consumerNoCPortIdx,
                                        /*isOutput=*/false);

  return {};
}

//===----------------------------------------------------------------------===//
// Input
//===----------------------------------------------------------------------===//

std::string MultiCoreSimSession::setCoreInput(const std::string &coreName,
                                              unsigned portIdx,
                                              const std::vector<uint64_t> &data) {
  CoreSimWrapper *core = findCore(coreName);
  if (!core)
    return "unknown core: '" + coreName + "'";
  return core->setInput(portIdx, data);
}

std::string MultiCoreSimSession::setCoreExtMemory(const std::string &coreName,
                                                  unsigned regionId,
                                                  uint8_t *data,
                                                  size_t sizeBytes) {
  CoreSimWrapper *core = findCore(coreName);
  if (!core)
    return "unknown core: '" + coreName + "'";
  return core->setExtMemoryBacking(regionId, data, sizeBytes);
}

//===----------------------------------------------------------------------===//
// Execution
//===----------------------------------------------------------------------===//

MultiCoreSimResult MultiCoreSimSession::run() {
  if (cores_.empty()) {
    MultiCoreSimResult result;
    result.success = false;
    result.errorMessage = "no cores added to simulation";
    return result;
  }

  uint64_t globalCycle = 0;
  bool allComplete = false;

  while (!allComplete && globalCycle < config_.maxGlobalCycles) {
    // Advance all cores by one cycle.
    stepAllCores();

    // Process inter-core data transfers through the NoC model.
    processInterCoreTransfers(globalCycle);

    // Process DMA requests through the memory hierarchy.
    if (config_.enableMemoryHierarchy) {
      processMemoryHierarchy(globalCycle);
    }

    // Check if all cores have completed.
    allComplete = allCoresComplete();

    // Invoke the global cycle callback if registered.
    if (globalCycleCallback_) {
      auto states = collectCoreStates(globalCycle);
      globalCycleCallback_(globalCycle, states);
    }

    ++globalCycle;
  }

  // Compute final link utilization.
  if (globalCycle > 0) {
    computeLinkUtilization(globalCycle);
  }

  return assembleResult(globalCycle);
}

//===----------------------------------------------------------------------===//
// Output
//===----------------------------------------------------------------------===//

std::vector<uint64_t>
MultiCoreSimSession::getCoreOutput(const std::string &coreName,
                                   unsigned portIdx) const {
  const CoreSimWrapper *core = findCore(coreName);
  if (!core)
    return {};
  return core->getOutput(portIdx);
}

//===----------------------------------------------------------------------===//
// Callback
//===----------------------------------------------------------------------===//

void MultiCoreSimSession::setGlobalCycleCallback(GlobalCycleCallback cb) {
  globalCycleCallback_ = std::move(cb);
}

//===----------------------------------------------------------------------===//
// Private: core lookup
//===----------------------------------------------------------------------===//

CoreSimWrapper *MultiCoreSimSession::findCore(const std::string &name) {
  auto it = coreNameToIdx_.find(name);
  if (it == coreNameToIdx_.end())
    return nullptr;
  return cores_[it->second].get();
}

const CoreSimWrapper *
MultiCoreSimSession::findCore(const std::string &name) const {
  auto it = coreNameToIdx_.find(name);
  if (it == coreNameToIdx_.end())
    return nullptr;
  return cores_[it->second].get();
}

//===----------------------------------------------------------------------===//
// Private: per-cycle stepping
//===----------------------------------------------------------------------===//

void MultiCoreSimSession::stepAllCores() {
  for (auto &core : cores_) {
    if (core->getState() != CoreState::COMPLETED &&
        core->getState() != CoreState::ERROR) {
      core->stepCycle();
    }
  }
}

//===----------------------------------------------------------------------===//
// Private: inter-core transfer processing
//===----------------------------------------------------------------------===//

void MultiCoreSimSession::processInterCoreTransfers(uint64_t globalCycle) {
  // Detect new outgoing data from producer cores and create transfers.
  for (const auto &route : routes_) {
    unsigned prodIdx = coreNameToIdx_[route.producerCoreName];
    CoreSimWrapper *producer = cores_[prodIdx].get();

    if (producer->hasOutgoingData(route.producerNoCPortIdx)) {
      auto data = producer->consumeOutgoingData(route.producerNoCPortIdx);
      if (!data.empty()) {
        PendingTransfer transfer;
        transfer.contractEdgeName = route.contractEdgeName;
        transfer.srcCoreName = route.producerCoreName;
        transfer.dstCoreName = route.consumerCoreName;
        transfer.srcCoreIdx = route.producerCoreIdx;
        transfer.dstCoreIdx = route.consumerCoreIdx;
        transfer.data = std::move(data);
        transfer.startCycle = globalCycle;
        transfer.totalLatencyCycles = route.transferLatencyCycles;
        transfer.remainingCycles = route.transferLatencyCycles;
        transfer.dstNoCPortIdx = route.consumerNoCPortIdx;

        transferStats_.totalTransfersInitiated++;
        transferStats_.totalFlitsTransferred +=
            static_cast<uint64_t>(transfer.data.size());

        inFlightTransfers_.push_back(std::move(transfer));
      }
    }
  }

  // Advance in-flight transfers and deliver completed ones.
  auto it = inFlightTransfers_.begin();
  while (it != inFlightTransfers_.end()) {
    if (it->remainingCycles > 0) {
      --it->remainingCycles;
      transferStats_.totalTransferCycles++;
    }

    if (it->remainingCycles == 0) {
      // Deliver data to the destination core.
      unsigned dstIdx = coreNameToIdx_[it->dstCoreName];
      CoreSimWrapper *consumer = cores_[dstIdx].get();
      consumer->injectIncomingData(it->dstNoCPortIdx, it->data);
      consumer->clearStall();

      transferStats_.totalTransfersCompleted++;
      it = inFlightTransfers_.erase(it);
    } else {
      ++it;
    }
  }
}

//===----------------------------------------------------------------------===//
// Private: memory hierarchy processing
//===----------------------------------------------------------------------===//

void MultiCoreSimSession::processMemoryHierarchy(uint64_t globalCycle) {
  for (auto &core : cores_) {
    while (core->hasPendingDMA()) {
      DMARequest req = core->dequeueDMA();
      memStats_.dmaTotalBytes += req.sizeBytes;

      // Model memory access latency based on DMA type.
      DMAResponse response;
      response.request = req;
      response.completionCycle =
          globalCycle + config_.dmaLatencyOverheadCycles;

      if (req.type == DMARequest::READ) {
        memStats_.dramReads++;
        response.readData.resize(req.sizeBytes, 0);
      } else {
        memStats_.dramWrites++;
      }

      memStats_.dmaTotalCycles += config_.dmaLatencyOverheadCycles;
      core->completeDMA(response);
    }
  }
}

//===----------------------------------------------------------------------===//
// Private: completion check
//===----------------------------------------------------------------------===//

bool MultiCoreSimSession::allCoresComplete() const {
  for (const auto &core : cores_) {
    if (!core->isComplete())
      return false;
  }
  return true;
}

//===----------------------------------------------------------------------===//
// Private: state collection
//===----------------------------------------------------------------------===//

std::vector<CoreSimResultEntry>
MultiCoreSimSession::collectCoreStates(uint64_t globalCycle) const {
  std::vector<CoreSimResultEntry> states;
  states.reserve(cores_.size());

  for (const auto &core : cores_) {
    CoreSimResultEntry entry;
    entry.coreName = core->getCoreName();
    entry.coreType = core->getCoreType();
    entry.coreIdx = core->getCoreIdx();
    entry.activeCycles = core->getActiveCycles();
    entry.stallCycles = core->getStallCycles();
    entry.idleCycles = core->getIdleCycles();
    entry.utilization = (globalCycle > 0)
                            ? static_cast<double>(core->getActiveCycles()) /
                                  static_cast<double>(globalCycle)
                            : 0.0;
    states.push_back(std::move(entry));
  }

  return states;
}

//===----------------------------------------------------------------------===//
// Private: result assembly
//===----------------------------------------------------------------------===//

MultiCoreSimResult
MultiCoreSimSession::assembleResult(uint64_t totalCycles) const {
  MultiCoreSimResult result;
  result.totalGlobalCycles = totalCycles;

  // Determine success: all cores must have completed.
  result.success = allCoresComplete();
  if (!result.success && totalCycles >= config_.maxGlobalCycles) {
    result.errorMessage = "simulation timed out after " +
                          std::to_string(config_.maxGlobalCycles) +
                          " global cycles";
  }

  // Collect per-core results.
  result.coreResults.reserve(cores_.size());
  for (const auto &core : cores_) {
    CoreSimResultEntry entry;
    entry.coreName = core->getCoreName();
    entry.coreType = core->getCoreType();
    entry.coreIdx = core->getCoreIdx();
    entry.activeCycles = core->getActiveCycles();
    entry.stallCycles = core->getStallCycles();
    entry.idleCycles = core->getIdleCycles();
    entry.utilization = (totalCycles > 0)
                            ? static_cast<double>(core->getActiveCycles()) /
                                  static_cast<double>(totalCycles)
                            : 0.0;
    entry.perCoreResult = core->getSimResult();
    result.coreResults.push_back(std::move(entry));
  }

  // NoC statistics.
  result.nocStats.totalFlitsTransferred = transferStats_.totalFlitsTransferred;
  result.nocStats.totalTransferCycles = transferStats_.totalTransferCycles;
  result.nocStats.avgLinkUtilization = transferStats_.avgLinkUtilization;
  result.nocStats.maxLinkUtilization = transferStats_.maxLinkUtilization;
  result.nocStats.contentionStallCycles = transferStats_.contentionStallCycles;

  // Memory statistics.
  result.memStats = memStats_;

  return result;
}

//===----------------------------------------------------------------------===//
// Private: link utilization
//===----------------------------------------------------------------------===//

void MultiCoreSimSession::computeLinkUtilization(uint64_t totalCycles) {
  if (linkStates_.empty())
    return;

  double maxUtil = 0.0;
  double sumUtil = 0.0;

  for (const auto &link : linkStates_) {
    double util = (totalCycles > 0)
                      ? static_cast<double>(link.activeCycles) /
                            static_cast<double>(totalCycles)
                      : 0.0;
    sumUtil += util;
    maxUtil = std::max(maxUtil, util);
  }

  transferStats_.avgLinkUtilization =
      linkStates_.empty() ? 0.0 : sumUtil / static_cast<double>(linkStates_.size());
  transferStats_.maxLinkUtilization = maxUtil;
}

} // namespace mcsim
} // namespace loom
