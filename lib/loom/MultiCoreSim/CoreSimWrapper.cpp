#include "loom/MultiCoreSim/CoreSimWrapper.h"

#include <cassert>
#include <sstream>
#include <utility>

namespace loom {
namespace mcsim {

const char *coreStateName(CoreState state) {
  switch (state) {
  case CoreState::IDLE:
    return "IDLE";
  case CoreState::RUNNING:
    return "RUNNING";
  case CoreState::STALLED:
    return "STALLED";
  case CoreState::COMPLETED:
    return "COMPLETED";
  case CoreState::ERROR:
    return "ERROR";
  }
  return "UNKNOWN";
}

//===----------------------------------------------------------------------===//
// Construction / move
//===----------------------------------------------------------------------===//

CoreSimWrapper::CoreSimWrapper(const std::string &name, unsigned coreIdx,
                               const std::string &coreType)
    : name_(name), coreIdx_(coreIdx), coreType_(coreType), session_() {}

CoreSimWrapper::~CoreSimWrapper() = default;

CoreSimWrapper::CoreSimWrapper(CoreSimWrapper &&other) noexcept
    : name_(std::move(other.name_)), coreIdx_(other.coreIdx_),
      coreType_(std::move(other.coreType_)), state_(other.state_),
      session_(std::move(other.session_)), built_(other.built_),
      configured_(other.configured_),
      invocationStarted_(other.invocationStarted_),
      localCycleCount_(other.localCycleCount_),
      activeCycles_(other.activeCycles_), stallCycles_(other.stallCycles_),
      idleCycles_(other.idleCycles_),
      outputBoundaryToNoC_(std::move(other.outputBoundaryToNoC_)),
      inputNoCToBoundary_(std::move(other.inputNoCToBoundary_)),
      outgoingBuffers_(std::move(other.outgoingBuffers_)),
      incomingBuffers_(std::move(other.incomingBuffers_)),
      pendingDMAs_(std::move(other.pendingDMAs_)),
      lastResult_(std::move(other.lastResult_)) {
  other.state_ = CoreState::IDLE;
}

CoreSimWrapper &CoreSimWrapper::operator=(CoreSimWrapper &&other) noexcept {
  if (this == &other)
    return *this;
  name_ = std::move(other.name_);
  coreIdx_ = other.coreIdx_;
  coreType_ = std::move(other.coreType_);
  state_ = other.state_;
  session_ = std::move(other.session_);
  built_ = other.built_;
  configured_ = other.configured_;
  invocationStarted_ = other.invocationStarted_;
  localCycleCount_ = other.localCycleCount_;
  activeCycles_ = other.activeCycles_;
  stallCycles_ = other.stallCycles_;
  idleCycles_ = other.idleCycles_;
  outputBoundaryToNoC_ = std::move(other.outputBoundaryToNoC_);
  inputNoCToBoundary_ = std::move(other.inputNoCToBoundary_);
  outgoingBuffers_ = std::move(other.outgoingBuffers_);
  incomingBuffers_ = std::move(other.incomingBuffers_);
  pendingDMAs_ = std::move(other.pendingDMAs_);
  lastResult_ = std::move(other.lastResult_);
  other.state_ = CoreState::IDLE;
  return *this;
}

//===----------------------------------------------------------------------===//
// Build phase
//===----------------------------------------------------------------------===//

std::string CoreSimWrapper::build(const sim::StaticMappedModel &model,
                                  const std::vector<uint8_t> &configBlob) {
  // Connect the session (transition Created -> Connected).
  std::string err = session_.connect();
  if (!err.empty())
    return "core '" + name_ + "' connect failed: " + err;

  // Build the model (transition Connected -> Ready).
  err = session_.buildFromStaticModel(model);
  if (!err.empty())
    return "core '" + name_ + "' buildFromStaticModel failed: " + err;
  built_ = true;

  // Load configuration (transition Ready -> Configured).
  err = session_.loadConfig(configBlob);
  if (!err.empty())
    return "core '" + name_ + "' loadConfig failed: " + err;
  configured_ = true;

  return {};
}

std::string CoreSimWrapper::setInput(unsigned portIdx,
                                     const std::vector<uint64_t> &data) {
  if (!configured_)
    return "core '" + name_ + "' not configured; cannot setInput";
  return session_.setInput(portIdx, data);
}

std::string CoreSimWrapper::setExtMemoryBacking(unsigned regionId,
                                                uint8_t *data,
                                                size_t sizeBytes) {
  if (!configured_)
    return "core '" + name_ + "' not configured; cannot set memory backing";
  return session_.setExtMemoryBacking(regionId, data, sizeBytes);
}

//===----------------------------------------------------------------------===//
// Per-cycle execution
//===----------------------------------------------------------------------===//

CoreState CoreSimWrapper::stepCycle() {
  if (state_ == CoreState::COMPLETED || state_ == CoreState::ERROR)
    return state_;

  // On first step, invoke the session to start the simulation.
  // The CycleSimulationBackend will run until a boundary event or completion.
  if (!invocationStarted_) {
    auto [result, err] = session_.invoke();
    if (!err.empty()) {
      state_ = CoreState::ERROR;
      lastResult_ = result;
      return state_;
    }
    lastResult_ = result;
    invocationStarted_ = true;

    // After invoke, check if the core completed immediately.
    if (result.termination == sim::RunTermination::Completed) {
      state_ = CoreState::COMPLETED;
      return state_;
    }

    state_ = CoreState::RUNNING;
    localCycleCount_ = result.totalCycles;
    activeCycles_ = localCycleCount_;
    return state_;
  }

  // For subsequent cycles, the single-core SimSession runs to completion
  // during invoke(). In the multi-core model, we track global cycles.
  ++localCycleCount_;

  if (state_ == CoreState::STALLED) {
    ++stallCycles_;
  } else if (state_ == CoreState::RUNNING) {
    ++activeCycles_;
  } else {
    ++idleCycles_;
  }

  return state_;
}

void CoreSimWrapper::clearStall() {
  if (state_ == CoreState::STALLED)
    state_ = CoreState::RUNNING;
}

void CoreSimWrapper::updateState() {
  // State transitions are managed by the orchestrator and explicit calls
  // (markStalled, clearStall) rather than by polling the kernel.
}

//===----------------------------------------------------------------------===//
// Inter-core I/O
//===----------------------------------------------------------------------===//

void CoreSimWrapper::mapBoundaryToNoCPort(unsigned boundaryOrdinal,
                                          unsigned nocPortIdx, bool isOutput) {
  if (isOutput) {
    outputBoundaryToNoC_[boundaryOrdinal] = nocPortIdx;
  } else {
    inputNoCToBoundary_[nocPortIdx] = boundaryOrdinal;
  }
}

bool CoreSimWrapper::hasOutgoingData(unsigned nocPortIdx) const {
  auto it = outgoingBuffers_.find(nocPortIdx);
  return it != outgoingBuffers_.end() && !it->second.empty();
}

std::vector<uint64_t>
CoreSimWrapper::consumeOutgoingData(unsigned nocPortIdx) {
  auto it = outgoingBuffers_.find(nocPortIdx);
  if (it == outgoingBuffers_.end() || it->second.empty())
    return {};
  std::vector<uint64_t> data = std::move(it->second.front());
  it->second.pop_front();
  return data;
}

void CoreSimWrapper::injectIncomingData(unsigned nocPortIdx,
                                        const std::vector<uint64_t> &data) {
  incomingBuffers_[nocPortIdx].push_back(data);
}

//===----------------------------------------------------------------------===//
// DMA interface
//===----------------------------------------------------------------------===//

DMARequest CoreSimWrapper::dequeueDMA() {
  assert(!pendingDMAs_.empty() && "No pending DMA to dequeue");
  DMARequest req = std::move(pendingDMAs_.front());
  pendingDMAs_.pop_front();
  return req;
}

void CoreSimWrapper::completeDMA(const DMAResponse &response) {
  // For read DMAs, the data is available to the core.
  // In a full implementation, this would feed data back into the
  // CycleKernel's memory system. For now, we track completion.
  (void)response;
}

//===----------------------------------------------------------------------===//
// Status
//===----------------------------------------------------------------------===//

const sim::SimResult &CoreSimWrapper::getSimResult() const {
  return lastResult_;
}

std::vector<uint64_t> CoreSimWrapper::getOutput(unsigned portIdx) const {
  return session_.getOutput(portIdx);
}

} // namespace mcsim
} // namespace loom
