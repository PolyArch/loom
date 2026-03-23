#include "loom/MultiCoreSim/MultiCoreSimSession.h"

#include <algorithm>
#include <sstream>

namespace loom {
namespace mcsim {

MultiCoreSimSession::MultiCoreSimSession(const MultiCoreSimConfig &config)
    : config_(config) {}

MultiCoreSimSession::~MultiCoreSimSession() = default;

std::string MultiCoreSimSession::initialize(
    const TapestryCompilationResult &compilationResult) {
  coreWrappers_.clear();
  coreIdToIndex_.clear();
  transferContracts_.clear();

  // Determine mesh dimensions from the NoC schedule.
  unsigned meshRows = compilationResult.nocSchedule.meshRows;
  unsigned meshCols = compilationResult.nocSchedule.meshCols;

  // If the schedule doesn't specify dimensions, derive from core count.
  unsigned numCores = static_cast<unsigned>(compilationResult.cores.size());
  if (meshRows == 0 || meshCols == 0) {
    meshRows = 1;
    meshCols = std::max(numCores, 1u);
  }

  // Create the NoC model.
  nocModel_ = std::make_unique<NoCSimModel>(
      meshRows, meshCols, config_.nocPerHopLatency);
  nocModel_->configure(compilationResult.nocSchedule);

  // Create the memory hierarchy model.
  MemoryHierarchyConfig memConfig = config_.memConfig;
  memConfig.numCores = numCores;
  memModel_ = std::make_unique<MemoryHierarchyModel>(memConfig);

  // Collect cross-core transfer contracts.
  for (const auto &entry : compilationResult.nocSchedule.entries) {
    transferContracts_.push_back(entry.contract);
  }

  // Create a CoreSimWrapper for each core that has mapped kernels.
  for (const auto &coreResult : compilationResult.cores) {
    if (coreResult.kernels.empty())
      continue;

    sim::SimConfig coreConfig = config_.perCoreSimConfig;
    coreConfig.coreId = static_cast<uint16_t>(coreResult.coreId);

    auto wrapper = std::make_unique<CoreSimWrapper>(coreResult.coreId,
                                                     coreConfig);

    // Initialize with the first kernel (multi-kernel scheduling is
    // future work; for now each core runs its first kernel).
    const CoreKernelResult &kernel = coreResult.kernels[0];
    std::string err = wrapper->initialize(kernel);
    if (!err.empty())
      return err;

    // Bind synthesized inputs.
    err = wrapper->bindSynthesizedInputs(kernel.synthSetup);
    if (!err.empty())
      return err;

    unsigned idx = static_cast<unsigned>(coreWrappers_.size());
    coreIdToIndex_[coreResult.coreId] = idx;
    coreWrappers_.push_back(std::move(wrapper));
  }

  return {};
}

MultiCoreSimResult MultiCoreSimSession::run() {
  MultiCoreSimResult result;
  globalCycle_ = 0;

  if (coreWrappers_.empty()) {
    result.error = "no active cores to simulate";
    result.allCoresCompleted = false;
    return result;
  }

  // For single-core mode, just run to completion directly.
  // This avoids the per-cycle stepping overhead.
  if (coreWrappers_.size() == 1 && transferContracts_.empty()) {
    auto &wrapper = coreWrappers_[0];
    auto [simResult, simErr] = wrapper->runToCompletion();

    CoreSimResult coreRes;
    coreRes.coreId = wrapper->getCoreId();
    coreRes.simResult = simResult;
    coreRes.totalCycles = simResult.totalCycles;
    coreRes.completed = simResult.success;
    coreRes.error = simErr;
    result.coreResults.push_back(coreRes);

    result.totalCycles = simResult.totalCycles;
    result.allCoresCompleted = simResult.success;
    if (!simResult.success)
      result.error = simErr;

    if (memModel_)
      result.memStats = memModel_->getStats();
    if (nocModel_)
      result.nocStats = nocModel_->getStats();

    return result;
  }

  // Multi-core lockstep simulation.
  // Run each core independently to completion (since per-cycle stepping
  // requires deeper integration with the CycleKernel). The NoC and
  // memory models track cycle-level statistics.
  for (auto &wrapper : coreWrappers_) {
    auto [simResult, simErr] = wrapper->runToCompletion();

    CoreSimResult coreRes;
    coreRes.coreId = wrapper->getCoreId();
    coreRes.simResult = simResult;
    coreRes.totalCycles = simResult.totalCycles;
    coreRes.completed = simResult.success;
    coreRes.error = simErr;
    result.coreResults.push_back(coreRes);
  }

  // Process NoC transfers after core execution.
  processNoCInjections();

  // Advance NoC model until all flits are delivered.
  unsigned maxNoCCycles = 10000;
  unsigned nocCycles = 0;
  while (!nocModel_->isIdle() && nocCycles < maxNoCCycles) {
    nocModel_->stepOneCycle();
    processNoCDeliveries();
    ++nocCycles;
  }

  // Compute global cycle count as max across all cores plus NoC.
  uint64_t maxCoreCycles = 0;
  bool allDone = true;
  for (const auto &coreRes : result.coreResults) {
    maxCoreCycles = std::max(maxCoreCycles, coreRes.totalCycles);
    if (!coreRes.completed)
      allDone = false;
  }
  result.totalCycles = maxCoreCycles + nocCycles;
  result.allCoresCompleted = allDone;

  if (memModel_)
    result.memStats = memModel_->getStats();
  if (nocModel_)
    result.nocStats = nocModel_->getStats();

  return result;
}

unsigned MultiCoreSimSession::getNumActiveCores() const {
  return static_cast<unsigned>(coreWrappers_.size());
}

const CoreSimWrapper *
MultiCoreSimSession::getCoreWrapper(unsigned coreId) const {
  auto it = coreIdToIndex_.find(coreId);
  if (it == coreIdToIndex_.end())
    return nullptr;
  return coreWrappers_[it->second].get();
}

const NoCSimModel *MultiCoreSimSession::getNoCModel() const {
  return nocModel_.get();
}

const MemoryHierarchyModel *MultiCoreSimSession::getMemoryModel() const {
  return memModel_.get();
}

void MultiCoreSimSession::processNoCInjections() {
  // For each cross-core contract, extract output from source core and
  // inject flits into the NoC.
  for (const auto &contract : transferContracts_) {
    auto srcIt = coreIdToIndex_.find(contract.srcCoreId);
    if (srcIt == coreIdToIndex_.end())
      continue;

    const auto &srcWrapper = coreWrappers_[srcIt->second];
    std::vector<uint64_t> outputData =
        srcWrapper->extractOutput(contract.srcOutputPort);

    // Create flits from the output data.
    unsigned totalFlits = static_cast<unsigned>(outputData.size());
    if (totalFlits == 0)
      totalFlits = contract.flitCount;

    for (unsigned flitIdx = 0; flitIdx < outputData.size(); ++flitIdx) {
      Flit flit;
      flit.srcCoreId = contract.srcCoreId;
      flit.dstCoreId = contract.dstCoreId;
      flit.channelId = contract.channelId;
      flit.data = outputData[flitIdx];
      flit.tag = 0;
      flit.hasTag = false;
      flit.injectionCycle = nocModel_->getCurrentCycle();
      flit.flitIndex = flitIdx;
      flit.totalFlits = totalFlits;
      flit.isHead = (flitIdx == 0);
      flit.isTail = (flitIdx + 1 == totalFlits);
      nocModel_->injectFlit(flit);
    }
  }
}

void MultiCoreSimSession::processNoCDeliveries() {
  // Deliver arrived flits to destination cores.
  for (const auto &contract : transferContracts_) {
    if (!nocModel_->hasArrivedFlits(contract.dstCoreId))
      continue;

    auto dstIt = coreIdToIndex_.find(contract.dstCoreId);
    if (dstIt == coreIdToIndex_.end())
      continue;

    auto flits = nocModel_->drainArrivedFlits(contract.dstCoreId);
    for (const auto &flit : flits) {
      coreWrappers_[dstIt->second]->injectToken(
          contract.dstInputPort, flit.data, flit.tag, flit.hasTag);
    }
  }
}

bool MultiCoreSimSession::allCoresDone() const {
  for (const auto &wrapper : coreWrappers_) {
    if (!wrapper->isDone())
      return false;
  }
  return true;
}

} // namespace mcsim
} // namespace loom
