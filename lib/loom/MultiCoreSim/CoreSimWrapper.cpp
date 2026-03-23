#include "loom/MultiCoreSim/CoreSimWrapper.h"
#include "loom/Simulator/CycleBackend.h"

#include <sstream>
#include <utility>

namespace loom {
namespace mcsim {

CoreSimWrapper::CoreSimWrapper(unsigned coreId, const sim::SimConfig &config)
    : coreId_(coreId), config_(config),
      session_(std::make_unique<sim::CycleSimulationBackend>(config), config) {}

CoreSimWrapper::~CoreSimWrapper() = default;

CoreSimWrapper::CoreSimWrapper(CoreSimWrapper &&other) noexcept
    : coreId_(other.coreId_), config_(other.config_),
      session_(std::move(other.session_)), result_(std::move(other.result_)),
      error_(std::move(other.error_)), initialized_(other.initialized_),
      done_(other.done_), hasError_(other.hasError_),
      cycleCount_(other.cycleCount_),
      memoryStorage_(std::move(other.memoryStorage_)) {
  other.initialized_ = false;
  other.done_ = false;
  other.hasError_ = false;
  other.cycleCount_ = 0;
}

CoreSimWrapper &CoreSimWrapper::operator=(CoreSimWrapper &&other) noexcept {
  if (this == &other)
    return *this;
  coreId_ = other.coreId_;
  config_ = other.config_;
  session_ = std::move(other.session_);
  result_ = std::move(other.result_);
  error_ = std::move(other.error_);
  initialized_ = other.initialized_;
  done_ = other.done_;
  hasError_ = other.hasError_;
  cycleCount_ = other.cycleCount_;
  memoryStorage_ = std::move(other.memoryStorage_);
  other.initialized_ = false;
  other.done_ = false;
  other.hasError_ = false;
  other.cycleCount_ = 0;
  return *this;
}

std::string CoreSimWrapper::initialize(const CoreKernelResult &kernelResult) {
  // Connect the session (transitions Created -> Connected).
  std::string err = session_.connect();
  if (!err.empty()) {
    error_ = "core " + std::to_string(coreId_) + " connect failed: " + err;
    hasError_ = true;
    return error_;
  }

  // Build the static model (transitions Connected -> Ready).
  err = session_.buildFromStaticModel(kernelResult.staticModel);
  if (!err.empty()) {
    error_ = "core " + std::to_string(coreId_) +
             " static model build failed: " + err;
    hasError_ = true;
    return error_;
  }

  // Load the config blob (transitions Ready -> Configured).
  if (!kernelResult.configSlices.empty()) {
    err = session_.loadConfig(kernelResult.configBlob,
                              kernelResult.configSlices);
  } else {
    err = session_.loadConfig(kernelResult.configBlob);
  }
  if (!err.empty()) {
    error_ = "core " + std::to_string(coreId_) +
             " config load failed: " + err;
    hasError_ = true;
    return error_;
  }

  initialized_ = true;
  return {};
}

std::string
CoreSimWrapper::bindSynthesizedInputs(const sim::SynthesizedSetup &setup) {
  if (!initialized_) {
    return "core " + std::to_string(coreId_) +
           " not initialized before binding inputs";
  }

  // Bind scalar input ports.
  for (const auto &input : setup.inputs) {
    std::string err =
        session_.setInput(input.portIdx, input.data, input.tags);
    if (!err.empty()) {
      error_ = "core " + std::to_string(coreId_) +
               " input bind failed for port " +
               std::to_string(input.portIdx) + ": " + err;
      hasError_ = true;
      return error_;
    }
  }

  // Bind memory regions.
  memoryStorage_.clear();
  memoryStorage_.reserve(setup.memoryRegions.size());
  for (const auto &region : setup.memoryRegions) {
    memoryStorage_.push_back(region.data);
    auto &bytes = memoryStorage_.back();
    std::string err = session_.setExtMemoryBacking(region.regionId,
                                                    bytes.data(), bytes.size());
    if (!err.empty()) {
      error_ = "core " + std::to_string(coreId_) +
               " memory bind failed for region " +
               std::to_string(region.regionId) + ": " + err;
      hasError_ = true;
      return error_;
    }
  }

  return {};
}

std::pair<sim::SimResult, std::string> CoreSimWrapper::runToCompletion() {
  if (!initialized_) {
    sim::SimResult failResult;
    failResult.success = false;
    failResult.termination = sim::RunTermination::ContractError;
    failResult.errorMessage = "core not initialized";
    return {failResult, failResult.errorMessage};
  }

  auto [simResult, invokeErr] = session_.invoke();
  result_ = simResult;
  cycleCount_ = simResult.totalCycles;

  if (!invokeErr.empty()) {
    error_ = "core " + std::to_string(coreId_) +
             " invocation failed: " + invokeErr;
    hasError_ = true;
    done_ = true;
    return {simResult, error_};
  }

  done_ = true;
  if (!simResult.success) {
    error_ = "core " + std::to_string(coreId_) +
             " simulation failed: " + simResult.errorMessage;
    hasError_ = true;
  }
  return {simResult, error_};
}

bool CoreSimWrapper::stepOneCycle() {
  // The SimSession does not expose per-cycle stepping directly.
  // For lockstep mode, we rely on the CycleCallback to observe
  // each cycle. For now, run the full invocation on the first call.
  if (done_ || hasError_)
    return false;

  if (!initialized_) {
    error_ = "core " + std::to_string(coreId_) + " not initialized";
    hasError_ = true;
    return false;
  }

  // Run the full simulation. The cycle callback (if set) will be
  // invoked for each cycle.
  runToCompletion();
  return false;
}

bool CoreSimWrapper::isDone() const { return done_; }

bool CoreSimWrapper::hasError() const { return hasError_; }

uint64_t CoreSimWrapper::getCurrentCycle() const { return cycleCount_; }

const sim::SimResult &CoreSimWrapper::getResult() const { return result_; }

const std::string &CoreSimWrapper::getError() const { return error_; }

std::string CoreSimWrapper::injectToken(unsigned portIdx, uint64_t data,
                                        uint16_t tag, bool hasTag) {
  if (!initialized_) {
    return "core " + std::to_string(coreId_) +
           " not initialized for token injection";
  }
  std::vector<uint64_t> dataVec = {data};
  std::vector<uint16_t> tagVec;
  if (hasTag)
    tagVec.push_back(tag);
  return session_.setInput(portIdx, dataVec, tagVec);
}

std::vector<uint64_t> CoreSimWrapper::extractOutput(unsigned portIdx) const {
  return session_.getOutput(portIdx);
}

} // namespace mcsim
} // namespace loom
