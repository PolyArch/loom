//===-- EventSimSession.h - Cosim session for event-driven backend -*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Session lifecycle wrapper around SimEngine per spec-cosim-backend-eventsim.md.
// Implements the cosim session state machine (Created -> Connected -> Ready ->
// Configured -> Running -> Draining -> Verified -> Closed) with state
// transition validation.
//
//===----------------------------------------------------------------------===//

#ifndef LOOM_SIMULATOR_EVENTSIMSESSION_H
#define LOOM_SIMULATOR_EVENTSIMSESSION_H

#include "loom/Simulator/SimEngine.h"
#include "loom/Simulator/SimTypes.h"

#include <mutex>
#include <string>
#include <vector>

namespace loom {

class Graph; // Forward declaration.

namespace sim {

/// Session lifecycle states per spec-cosim-architecture.md.
enum class SessionState : uint8_t {
  Created = 0,
  Connected = 1,
  Ready = 2,
  Configured = 3,
  Running = 4,
  Draining = 5,
  Verified = 6,
  Closed = 7,
};

/// Return a human-readable name for a session state.
const char *sessionStateName(SessionState s);

/// Result of a CPU oracle comparison.
struct CompareResult {
  bool pass = false;
  unsigned totalOutputs = 0;
  unsigned mismatches = 0;
  std::string details;
};

/// Event-driven simulator cosim session.
///
/// Thread-safe at the API boundary: all public methods acquire an internal
/// mutex. Internally, execution is single-threaded.
class EventSimSession {
public:
  explicit EventSimSession(const SimConfig &config = SimConfig());
  ~EventSimSession();

  // Non-copyable, movable.
  EventSimSession(const EventSimSession &) = delete;
  EventSimSession &operator=(const EventSimSession &) = delete;
  EventSimSession(EventSimSession &&) noexcept;
  EventSimSession &operator=(EventSimSession &&) noexcept;

  /// Get the current session state.
  SessionState getState() const;

  /// Connect the session (Created -> Connected).
  /// Returns error message on failure (empty on success).
  std::string connect();

  /// Build the simulation model from an ADG graph (Connected -> Ready).
  std::string buildFromGraph(const Graph &adg);

  /// Load configuration from a config.bin file (Ready|Verified -> Configured).
  std::string loadConfig(const std::string &configBinPath);

  /// Load configuration from raw bytes (Ready|Verified -> Configured).
  std::string loadConfig(const std::vector<uint8_t> &configBlob);

  /// Load configuration from raw bytes with mapper-authored config slices.
  std::string loadConfig(const std::vector<uint8_t> &configBlob,
                         const std::vector<SimEngine::ExternalConfigSlice> &slices);

  /// Set input data for a boundary input port.
  /// Only valid in Configured state.
  std::string setInput(unsigned portIdx, const std::vector<uint64_t> &data,
                       const std::vector<uint16_t> &tags = {});

  /// Run the simulation (Configured -> Running -> Draining).
  /// Returns the simulation result.
  std::pair<SimResult, std::string> invoke();

  /// Get output data from a boundary output port (valid after Draining).
  std::vector<uint64_t> getOutput(unsigned portIdx) const;

  /// Get output tags from a boundary output port (valid after Draining).
  std::vector<uint16_t> getOutputTags(unsigned portIdx) const;

  /// Run CPU oracle comparison (Draining -> Verified).
  /// referenceOutputs: per-port expected output vectors.
  CompareResult compare(
      const std::vector<std::vector<uint64_t>> &referenceOutputs);

  /// Reset execution state, keep configuration (Running/Draining/Verified ->
  /// Configured).
  std::string resetExecution();

  /// Full reset, clear configuration (any non-Closed state -> Connected).
  std::string resetAll();

  /// Disconnect and release resources (any state -> Closed).
  std::string disconnect();

  /// Get the current epoch ID.
  uint32_t getEpochId() const;

  /// Get last simulation result (valid after invoke()).
  const SimResult &getLastResult() const;

  /// Get the number of boundary input ports (valid after buildFromGraph).
  unsigned getNumInputPorts() const;

  /// Get the number of boundary output ports (valid after buildFromGraph).
  unsigned getNumOutputPorts() const;

  /// Audit route completeness (valid after loadConfig).
  AuditResult auditRoutes() const;

  /// Set backing memory for all extmemory modules.
  void setExtMemoryBacking(uint8_t *data, size_t sizeBytes);

  /// Mark specific dead output channels as sinks (valid after loadConfig).
  unsigned markDeadOutputSinks(
      const std::vector<std::pair<uint32_t, unsigned>> &deadPorts);

private:
  mutable std::mutex mu_;
  SessionState state_ = SessionState::Created;
  uint32_t epochId_ = 0;
  SimConfig config_;
  std::unique_ptr<SimEngine> engine_;
  SimResult lastResult_;

  /// Validate a state transition and return error message if invalid.
  std::string validateTransition(SessionState from, SessionState to) const;
};

} // namespace sim
} // namespace loom

#endif // LOOM_SIMULATOR_EVENTSIMSESSION_H
