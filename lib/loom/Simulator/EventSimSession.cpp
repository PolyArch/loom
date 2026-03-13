//===-- EventSimSession.cpp - Cosim session implementation --------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include "loom/Simulator/EventSimSession.h"
#include "loom/Mapper/Graph.h"

#include <sstream>

namespace loom {
namespace sim {

const char *sessionStateName(SessionState s) {
  switch (s) {
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

EventSimSession::EventSimSession(const SimConfig &config)
    : config_(config), engine_(std::make_unique<SimEngine>(config)) {}

EventSimSession::~EventSimSession() {
  // Ensure cleanup regardless of state.
  std::lock_guard<std::mutex> lock(mu_);
  state_ = SessionState::Closed;
  engine_.reset();
}

EventSimSession::EventSimSession(EventSimSession &&other) noexcept {
  std::lock_guard<std::mutex> lock(other.mu_);
  state_ = other.state_;
  epochId_ = other.epochId_;
  config_ = other.config_;
  engine_ = std::move(other.engine_);
  lastResult_ = std::move(other.lastResult_);
  other.state_ = SessionState::Closed;
}

EventSimSession &
EventSimSession::operator=(EventSimSession &&other) noexcept {
  if (this != &other) {
    std::scoped_lock lock(mu_, other.mu_);
    state_ = other.state_;
    epochId_ = other.epochId_;
    config_ = other.config_;
    engine_ = std::move(other.engine_);
    lastResult_ = std::move(other.lastResult_);
    other.state_ = SessionState::Closed;
  }
  return *this;
}

SessionState EventSimSession::getState() const {
  std::lock_guard<std::mutex> lock(mu_);
  return state_;
}

std::string EventSimSession::validateTransition(SessionState from,
                                                 SessionState to) const {
  // Define valid transitions per spec-cosim-architecture.md.
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
    valid = (to == SessionState::Configured ||
             to == SessionState::Connected || to == SessionState::Closed);
    break;
  case SessionState::Closed:
    valid = false; // Terminal state.
    break;
  }

  if (!valid) {
    std::ostringstream oss;
    oss << "invalid state transition: " << sessionStateName(from) << " -> "
        << sessionStateName(to);
    return oss.str();
  }
  return {};
}

std::string EventSimSession::connect() {
  std::lock_guard<std::mutex> lock(mu_);
  std::string err = validateTransition(state_, SessionState::Connected);
  if (!err.empty())
    return err;

  if (!engine_)
    engine_ = std::make_unique<SimEngine>(config_);

  state_ = SessionState::Connected;
  return {};
}

std::string EventSimSession::buildFromGraph(const Graph &adg) {
  std::lock_guard<std::mutex> lock(mu_);
  std::string err = validateTransition(state_, SessionState::Ready);
  if (!err.empty())
    return err;

  if (!engine_->buildFromGraph(adg))
    return "failed to build simulation model from ADG graph";

  state_ = SessionState::Ready;
  return {};
}

std::string EventSimSession::loadConfig(const std::string &configBinPath) {
  std::lock_guard<std::mutex> lock(mu_);

  // loadConfig is valid from Ready or Verified (reconfiguration).
  if (state_ != SessionState::Ready && state_ != SessionState::Verified) {
    std::string err =
        validateTransition(state_, SessionState::Configured);
    return err.empty() ? "unexpected state for loadConfig" : err;
  }

  ++epochId_;
  engine_->setEpochId(epochId_);

  if (!engine_->loadConfig(configBinPath))
    return "failed to load config from: " + configBinPath;

  state_ = SessionState::Configured;
  return {};
}

std::string EventSimSession::loadConfig(const std::vector<uint8_t> &configBlob) {
  std::lock_guard<std::mutex> lock(mu_);

  if (state_ != SessionState::Ready && state_ != SessionState::Verified) {
    std::string err =
        validateTransition(state_, SessionState::Configured);
    return err.empty() ? "unexpected state for loadConfig" : err;
  }

  ++epochId_;
  engine_->setEpochId(epochId_);

  if (!engine_->loadConfig(configBlob))
    return "failed to load config from blob";

  state_ = SessionState::Configured;
  return {};
}

std::string EventSimSession::loadConfig(
    const std::vector<uint8_t> &configBlob,
    const std::vector<SimEngine::ExternalConfigSlice> &slices) {
  std::lock_guard<std::mutex> lock(mu_);

  if (state_ != SessionState::Ready && state_ != SessionState::Verified) {
    std::string err =
        validateTransition(state_, SessionState::Configured);
    return err.empty() ? "unexpected state for loadConfig" : err;
  }

  ++epochId_;
  engine_->setEpochId(epochId_);

  if (!engine_->loadConfig(configBlob, slices))
    return "failed to load config from blob with slices";

  state_ = SessionState::Configured;
  return {};
}

std::string EventSimSession::setInput(unsigned portIdx,
                                       const std::vector<uint64_t> &data,
                                       const std::vector<uint16_t> &tags) {
  std::lock_guard<std::mutex> lock(mu_);
  if (state_ != SessionState::Configured)
    return "setInput requires Configured state, current: " +
           std::string(sessionStateName(state_));

  engine_->setInput(portIdx, data, tags);
  return {};
}

std::pair<SimResult, std::string> EventSimSession::invoke() {
  std::lock_guard<std::mutex> lock(mu_);
  std::string err = validateTransition(state_, SessionState::Running);
  if (!err.empty())
    return {SimResult{}, err};

  state_ = SessionState::Running;

  lastResult_ = engine_->run();

  // Transition to Draining after run completes.
  state_ = SessionState::Draining;
  return {lastResult_, {}};
}

std::vector<uint64_t> EventSimSession::getOutput(unsigned portIdx) const {
  std::lock_guard<std::mutex> lock(mu_);
  if (state_ != SessionState::Draining && state_ != SessionState::Verified)
    return {};
  return engine_->getOutput(portIdx);
}

std::vector<uint16_t> EventSimSession::getOutputTags(unsigned portIdx) const {
  std::lock_guard<std::mutex> lock(mu_);
  if (state_ != SessionState::Draining && state_ != SessionState::Verified)
    return {};
  return engine_->getOutputTags(portIdx);
}

CompareResult EventSimSession::compare(
    const std::vector<std::vector<uint64_t>> &referenceOutputs) {
  std::lock_guard<std::mutex> lock(mu_);
  if (state_ != SessionState::Draining) {
    CompareResult r;
    r.details = "compare requires Draining state, current: " +
                std::string(sessionStateName(state_));
    return r;
  }

  CompareResult result;
  result.pass = true;
  result.totalOutputs = 0;
  result.mismatches = 0;

  std::ostringstream details;

  // Compare against all ports: max of reference count and actual port count.
  unsigned numActualPorts = engine_->getNumOutputPorts();
  unsigned numPorts = std::max(static_cast<unsigned>(referenceOutputs.size()),
                               numActualPorts);

  for (unsigned p = 0; p < numPorts; ++p) {
    auto actual = engine_->getOutput(p);
    const auto &expected =
        (p < referenceOutputs.size()) ? referenceOutputs[p]
                                      : std::vector<uint64_t>{};

    // Port count mismatch: extra accelerator ports with data are failures.
    if (p >= referenceOutputs.size() && !actual.empty()) {
      result.pass = false;
      ++result.mismatches;
      if (result.mismatches <= 10) {
        details << "port " << p << ": unexpected accelerator output ("
                << actual.size() << " elements, no reference)\n";
      }
      result.totalOutputs += static_cast<unsigned>(actual.size());
      continue;
    }

    size_t len = std::max(actual.size(), expected.size());
    result.totalOutputs += static_cast<unsigned>(len);

    for (size_t i = 0; i < len; ++i) {
      uint64_t act = (i < actual.size()) ? actual[i] : 0;
      uint64_t exp = (i < expected.size()) ? expected[i] : 0;
      if (act != exp) {
        ++result.mismatches;
        result.pass = false;
        if (result.mismatches <= 10) {
          details << "port " << p << " elem " << i << ": expected " << exp
                  << " got " << act << "\n";
        }
      }
    }
  }

  if (result.mismatches > 10)
    details << "... and " << (result.mismatches - 10) << " more mismatches\n";

  result.details = details.str();
  state_ = SessionState::Verified;
  return result;
}

std::string EventSimSession::resetExecution() {
  std::lock_guard<std::mutex> lock(mu_);

  // resetExecution returns to Configured state, preserving config.
  if (state_ != SessionState::Running && state_ != SessionState::Draining &&
      state_ != SessionState::Verified && state_ != SessionState::Configured) {
    return "resetExecution requires Running/Draining/Verified/Configured state, "
           "current: " +
           std::string(sessionStateName(state_));
  }

  engine_->resetExecution();
  state_ = SessionState::Configured;
  return {};
}

std::string EventSimSession::resetAll() {
  std::lock_guard<std::mutex> lock(mu_);

  if (state_ == SessionState::Closed)
    return "session is closed";

  engine_->resetAll();
  epochId_ = 0;
  state_ = SessionState::Connected;
  return {};
}

std::string EventSimSession::disconnect() {
  std::lock_guard<std::mutex> lock(mu_);
  if (state_ == SessionState::Closed)
    return {}; // Already closed, idempotent.

  engine_.reset();
  state_ = SessionState::Closed;
  return {};
}

uint32_t EventSimSession::getEpochId() const {
  std::lock_guard<std::mutex> lock(mu_);
  return epochId_;
}

const SimResult &EventSimSession::getLastResult() const {
  std::lock_guard<std::mutex> lock(mu_);
  return lastResult_;
}

unsigned EventSimSession::getNumInputPorts() const {
  std::lock_guard<std::mutex> lock(mu_);
  return engine_ ? engine_->getNumInputPorts() : 0;
}

unsigned EventSimSession::getNumOutputPorts() const {
  std::lock_guard<std::mutex> lock(mu_);
  return engine_ ? engine_->getNumOutputPorts() : 0;
}

AuditResult EventSimSession::auditRoutes() const {
  std::lock_guard<std::mutex> lock(mu_);
  if (!engine_) {
    AuditResult r;
    r.pass = false;
    AuditDiagnostic d;
    d.level = AuditDiagnostic::Error;
    d.message = "no engine built";
    r.diagnostics.push_back(d);
    return r;
  }
  return engine_->auditRoutes();
}

} // namespace sim
} // namespace loom
