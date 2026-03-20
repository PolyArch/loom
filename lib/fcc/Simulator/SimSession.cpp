#include "fcc/Simulator/CycleBackend.h"
#include "fcc/Simulator/FunctionalBackend.h"
#include "fcc/Simulator/SimSession.h"

#include <algorithm>
#include <cstring>
#include <sstream>

namespace fcc {
namespace sim {

namespace {

} // namespace

const char *runTerminationName(RunTermination termination) {
  switch (termination) {
  case RunTermination::Completed:
    return "completed";
  case RunTermination::Timeout:
    return "timeout";
  case RunTermination::DeviceError:
    return "device_error";
  case RunTermination::ContractError:
    return "contract_error";
  }
  return "unknown";
}

const char *sessionStateName(SessionState state) {
  switch (state) {
  case SessionState::Created:
    return "Created";
  case SessionState::Connected:
    return "Connected";
  case SessionState::Ready:
    return "Ready";
  case SessionState::Configured:
    return "Configured";
  case SessionState::Running:
    return "Running";
  case SessionState::Draining:
    return "Draining";
  case SessionState::Verified:
    return "Verified";
  case SessionState::Closed:
    return "Closed";
  }
  return "Unknown";
}

std::unique_ptr<SimulationBackend>
SimSession::createDefaultBackend(const SimConfig &config) {
  return std::make_unique<CycleSimulationBackend>(config);
}

SimSession::SimSession(std::unique_ptr<SimulationBackend> backend,
                       const SimConfig &config)
    : config_(config),
      backend_(backend ? std::move(backend) : createDefaultBackend(config)) {}

SimSession::~SimSession() {
  std::lock_guard<std::mutex> lock(mu_);
  state_ = SessionState::Closed;
  backend_.reset();
}

SimSession::SimSession(SimSession &&other) noexcept {
  std::lock_guard<std::mutex> lock(other.mu_);
  state_ = other.state_;
  epochId_ = other.epochId_;
  invocationId_ = other.invocationId_;
  config_ = other.config_;
  backend_ = std::move(other.backend_);
  lastResult_ = std::move(other.lastResult_);
  configBlob_ = std::move(other.configBlob_);
  configSlices_ = std::move(other.configSlices_);
  memoryRegions_ = std::move(other.memoryRegions_);
  other.state_ = SessionState::Closed;
}

SimSession &SimSession::operator=(SimSession &&other) noexcept {
  if (this == &other)
    return *this;

  std::scoped_lock lock(mu_, other.mu_);
  state_ = other.state_;
  epochId_ = other.epochId_;
  invocationId_ = other.invocationId_;
  config_ = other.config_;
  backend_ = std::move(other.backend_);
  lastResult_ = std::move(other.lastResult_);
  configBlob_ = std::move(other.configBlob_);
  configSlices_ = std::move(other.configSlices_);
  memoryRegions_ = std::move(other.memoryRegions_);
  other.state_ = SessionState::Closed;
  return *this;
}

SessionState SimSession::getState() const {
  std::lock_guard<std::mutex> lock(mu_);
  return state_;
}

std::string SimSession::validateTransition(SessionState from,
                                           SessionState to) const {
  bool valid = false;
  switch (from) {
  case SessionState::Created:
    valid = (to == SessionState::Connected || to == SessionState::Closed);
    break;
  case SessionState::Connected:
    valid = (to == SessionState::Ready || to == SessionState::Closed);
    break;
  case SessionState::Ready:
    valid = (to == SessionState::Configured || to == SessionState::Closed);
    break;
  case SessionState::Configured:
    valid = (to == SessionState::Running || to == SessionState::Connected ||
             to == SessionState::Closed);
    break;
  case SessionState::Running:
    valid = (to == SessionState::Draining || to == SessionState::Closed);
    break;
  case SessionState::Draining:
    valid = (to == SessionState::Verified || to == SessionState::Configured ||
             to == SessionState::Closed);
    break;
  case SessionState::Verified:
    valid = (to == SessionState::Configured || to == SessionState::Connected ||
             to == SessionState::Closed);
    break;
  case SessionState::Closed:
    valid = false;
    break;
  }

  if (valid)
    return {};

  std::ostringstream oss;
  oss << "invalid state transition: " << sessionStateName(from) << " -> "
      << sessionStateName(to);
  return oss.str();
}

std::string SimSession::connect() {
  std::lock_guard<std::mutex> lock(mu_);
  std::string err = validateTransition(state_, SessionState::Connected);
  if (!err.empty())
    return err;
  err = backend_->connect();
  if (!err.empty())
    return err;
  state_ = SessionState::Connected;
  return {};
}

std::string SimSession::buildFromMappedState(const Graph &dfg, const Graph &adg,
                                             const MappingState &mapping) {
  std::lock_guard<std::mutex> lock(mu_);
  std::string err = validateTransition(state_, SessionState::Ready);
  if (!err.empty())
    return err;
  err = backend_->buildFromMappedState(dfg, adg, mapping);
  if (!err.empty())
    return err;
  state_ = SessionState::Ready;
  return {};
}

std::string SimSession::buildFromMappedState(
    const Graph &dfg, const Graph &adg, const MappingState &mapping,
    llvm::ArrayRef<PEContainment> peContainment) {
  std::lock_guard<std::mutex> lock(mu_);
  std::string err = validateTransition(state_, SessionState::Ready);
  if (!err.empty())
    return err;
  err = backend_->buildFromMappedState(dfg, adg, mapping, peContainment);
  if (!err.empty())
    return err;
  state_ = SessionState::Ready;
  return {};
}

std::string SimSession::buildFromStaticModel(const StaticMappedModel &model) {
  std::lock_guard<std::mutex> lock(mu_);
  std::string err = validateTransition(state_, SessionState::Ready);
  if (!err.empty())
    return err;
  err = backend_->buildFromStaticModel(model);
  if (!err.empty())
    return err;
  state_ = SessionState::Ready;
  return {};
}

std::string SimSession::loadConfig(const std::vector<uint8_t> &configBlob) {
  return loadConfig(configBlob, {});
}

std::string SimSession::loadConfig(
    const std::vector<uint8_t> &configBlob,
    llvm::ArrayRef<fcc::ConfigGen::ConfigSlice> configSlices) {
  std::lock_guard<std::mutex> lock(mu_);
  if (state_ != SessionState::Ready && state_ != SessionState::Verified) {
    std::string err = validateTransition(state_, SessionState::Configured);
    return err.empty() ? "unexpected state for loadConfig" : err;
  }
  configBlob_ = configBlob;
  configSlices_.assign(configSlices.begin(), configSlices.end());
  ++epochId_;
  std::string err = backend_->loadConfig(configBlob_, configSlices_);
  if (!err.empty())
    return err;
  state_ = SessionState::Configured;
  return {};
}

std::string SimSession::setInput(unsigned portIdx,
                                 const std::vector<uint64_t> &data,
                                 const std::vector<uint16_t> &tags) {
  std::lock_guard<std::mutex> lock(mu_);
  if (state_ != SessionState::Configured)
    return "setInput requires Configured state";
  return backend_->setInput(portIdx, data, tags);
}

std::string SimSession::setExtMemoryBacking(unsigned regionId, uint8_t *data,
                                            size_t sizeBytes) {
  std::lock_guard<std::mutex> lock(mu_);
  if (state_ != SessionState::Configured)
    return "setExtMemoryBacking requires Configured state";
  if (memoryRegions_.size() <= regionId)
    memoryRegions_.resize(regionId + 1);
  memoryRegions_[regionId].data = data;
  memoryRegions_[regionId].sizeBytes = sizeBytes;
  return backend_->setExtMemoryBacking(regionId, data, sizeBytes);
}

std::pair<SimResult, std::string> SimSession::invoke() {
  std::lock_guard<std::mutex> lock(mu_);
  std::string err = validateTransition(state_, SessionState::Running);
  if (!err.empty())
    return {SimResult{}, err};

  state_ = SessionState::Running;
  ++invocationId_;
  lastResult_ = backend_->invoke(epochId_, invocationId_);
  state_ = SessionState::Draining;
  return {lastResult_, {}};
}

std::vector<uint64_t> SimSession::getOutput(unsigned portIdx) const {
  std::lock_guard<std::mutex> lock(mu_);
  if (state_ != SessionState::Draining && state_ != SessionState::Verified)
    return {};
  return backend_->getOutput(portIdx);
}

std::vector<uint16_t> SimSession::getOutputTags(unsigned portIdx) const {
  std::lock_guard<std::mutex> lock(mu_);
  if (state_ != SessionState::Draining && state_ != SessionState::Verified)
    return {};
  return backend_->getOutputTags(portIdx);
}

CompareResult SimSession::compareOutputPorts(
    const std::vector<std::vector<uint64_t>> &reference) const {
  std::lock_guard<std::mutex> lock(mu_);
  CompareResult result;
  if (state_ != SessionState::Draining && state_ != SessionState::Verified) {
    result.details = "compareOutputPorts requires Draining or Verified state";
    return result;
  }

  result.pass = true;
  unsigned actualPorts = backend_->getNumOutputPorts();
  unsigned numPorts = std::max(actualPorts, static_cast<unsigned>(reference.size()));
  std::ostringstream details;

  for (unsigned port = 0; port < numPorts; ++port) {
    std::vector<uint64_t> actual = backend_->getOutput(port);
    std::vector<uint64_t> expected =
        port < reference.size() ? reference[port] : std::vector<uint64_t>{};
    size_t count = std::max(actual.size(), expected.size());
    result.totalOutputs += static_cast<unsigned>(count);
    for (size_t idx = 0; idx < count; ++idx) {
      uint64_t actualValue = idx < actual.size() ? actual[idx] : 0;
      uint64_t expectedValue = idx < expected.size() ? expected[idx] : 0;
      if (actualValue == expectedValue)
        continue;
      result.pass = false;
      ++result.mismatches;
      if (result.mismatches <= 10) {
        details << "port " << port << " elem " << idx << ": expected "
                << expectedValue << " got " << actualValue << "\n";
      }
    }
  }

  if (result.mismatches > 10)
    details << "... and " << (result.mismatches - 10) << " more mismatches\n";

  result.details = details.str();
  return result;
}

CompareResult SimSession::compareMemoryRegion(unsigned regionId,
                                              llvm::ArrayRef<uint8_t> expected) const {
  std::lock_guard<std::mutex> lock(mu_);
  CompareResult result;
  if (state_ != SessionState::Draining && state_ != SessionState::Verified) {
    result.details = "compareMemoryRegion requires Draining or Verified state";
    return result;
  }
  if (regionId >= memoryRegions_.size() || !memoryRegions_[regionId].data) {
    result.details = "memory region not bound";
    return result;
  }

  const MemoryRegionBinding &binding = memoryRegions_[regionId];
  result.pass = true;
  result.totalOutputs = static_cast<unsigned>(expected.size());
  std::ostringstream details;
  for (size_t idx = 0; idx < expected.size(); ++idx) {
    uint8_t actualValue =
        idx < binding.sizeBytes ? binding.data[idx] : static_cast<uint8_t>(0);
    uint8_t expectedValue = expected[idx];
    if (actualValue == expectedValue)
      continue;
    result.pass = false;
    ++result.mismatches;
    if (result.mismatches <= 10) {
      details << "region " << regionId << " byte " << idx << ": expected "
              << static_cast<unsigned>(expectedValue) << " got "
              << static_cast<unsigned>(actualValue) << "\n";
    }
  }
  if (binding.sizeBytes < expected.size()) {
    result.pass = false;
    ++result.mismatches;
    details << "region " << regionId << " shorter than expected: "
            << binding.sizeBytes << " < " << expected.size() << "\n";
  }
  result.details = details.str();
  return result;
}

std::string SimSession::resetExecution() {
  std::lock_guard<std::mutex> lock(mu_);
  if (state_ != SessionState::Running && state_ != SessionState::Draining &&
      state_ != SessionState::Verified && state_ != SessionState::Configured) {
    return "resetExecution requires Running, Draining, Verified, or Configured state";
  }
  backend_->resetExecution();
  lastResult_ = SimResult();
  state_ = SessionState::Configured;
  return {};
}

std::string SimSession::resetAll() {
  std::lock_guard<std::mutex> lock(mu_);
  if (state_ == SessionState::Closed)
    return "session is closed";
  backend_->resetAll();
  epochId_ = 0;
  invocationId_ = 0;
  configBlob_.clear();
  memoryRegions_.clear();
  lastResult_ = SimResult();
  state_ = SessionState::Connected;
  return {};
}

std::string SimSession::disconnect() {
  std::lock_guard<std::mutex> lock(mu_);
  if (state_ == SessionState::Closed)
    return {};
  backend_.reset();
  state_ = SessionState::Closed;
  return {};
}

const SimResult &SimSession::getLastResult() const {
  std::lock_guard<std::mutex> lock(mu_);
  return lastResult_;
}

uint32_t SimSession::getEpochId() const {
  std::lock_guard<std::mutex> lock(mu_);
  return epochId_;
}

uint64_t SimSession::getInvocationId() const {
  std::lock_guard<std::mutex> lock(mu_);
  return invocationId_;
}

unsigned SimSession::getNumInputPorts() const {
  std::lock_guard<std::mutex> lock(mu_);
  return backend_ ? backend_->getNumInputPorts() : 0;
}

unsigned SimSession::getNumOutputPorts() const {
  std::lock_guard<std::mutex> lock(mu_);
  return backend_ ? backend_->getNumOutputPorts() : 0;
}

size_t SimSession::getNumBoundMemoryRegions() const {
  std::lock_guard<std::mutex> lock(mu_);
  return memoryRegions_.size();
}

} // namespace sim
} // namespace fcc
