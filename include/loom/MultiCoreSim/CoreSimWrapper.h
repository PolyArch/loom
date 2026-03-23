#ifndef LOOM_MULTICORESIM_CORESIMWRAPPER_H
#define LOOM_MULTICORESIM_CORESIMWRAPPER_H

#include "loom/Simulator/CycleBackend.h"
#include "loom/Simulator/SimSession.h"
#include "loom/Simulator/SimTypes.h"
#include "loom/Simulator/StaticModel.h"

#include <cstdint>
#include <deque>
#include <string>
#include <unordered_map>
#include <vector>

namespace loom {
namespace mcsim {

//===----------------------------------------------------------------------===//
// DMA Request
//===----------------------------------------------------------------------===//

/// Represents a DMA request issued by a core for external memory access.
struct DMARequest {
  enum Type : uint8_t { READ = 0, WRITE = 1 };

  Type type = READ;
  uint64_t address = 0;
  uint64_t sizeBytes = 0;
  std::vector<uint8_t> writeData;
  unsigned coreIdx = 0;
  uint64_t requestId = 0;
};

/// Response to a completed DMA request.
struct DMAResponse {
  DMARequest request;
  std::vector<uint8_t> readData;
  uint64_t completionCycle = 0;
};

//===----------------------------------------------------------------------===//
// Core State
//===----------------------------------------------------------------------===//

/// Execution state of a single core within the multi-core simulation.
enum class CoreState : uint8_t {
  IDLE = 0,
  RUNNING = 1,
  STALLED = 2,
  COMPLETED = 3,
  ERROR = 4,
};

const char *coreStateName(CoreState state);

//===----------------------------------------------------------------------===//
// CoreSimWrapper
//===----------------------------------------------------------------------===//

/// Wraps an existing SimSession for use as one core in a multi-core simulation.
///
/// The wrapper bridges the single-core CycleKernel interface with the
/// multi-core orchestrator by adding:
///   - NoC send/receive ports for inter-core data transfer
///   - Per-core cycle counting and stall tracking
///   - DMA request queuing for memory hierarchy access
///   - State machine tracking (IDLE -> RUNNING -> COMPLETED)
class CoreSimWrapper {
public:
  /// Construct a wrapper around an existing SimSession.
  /// The SimSession must already be in Connected state (backend created).
  CoreSimWrapper(const std::string &name, unsigned coreIdx,
                 const std::string &coreType);
  ~CoreSimWrapper();

  CoreSimWrapper(const CoreSimWrapper &) = delete;
  CoreSimWrapper &operator=(const CoreSimWrapper &) = delete;
  CoreSimWrapper(CoreSimWrapper &&) noexcept;
  CoreSimWrapper &operator=(CoreSimWrapper &&) noexcept;

  // --- Build phase ---

  /// Build the core from a static model and configuration blob.
  std::string build(const sim::StaticMappedModel &model,
                    const std::vector<uint8_t> &configBlob);

  /// Set input data on a boundary input port.
  std::string setInput(unsigned portIdx, const std::vector<uint64_t> &data);

  /// Set external memory backing for a memory region.
  std::string setExtMemoryBacking(unsigned regionId, uint8_t *data,
                                  size_t sizeBytes);

  // --- Per-cycle execution (called by orchestrator) ---

  /// Get the current state of this core.
  CoreState getState() const { return state_; }

  /// Advance this core by one cycle.
  /// Returns the new state after stepping.
  CoreState stepCycle();

  /// Mark this core as stalled (waiting for inter-core data).
  void markStalled() { state_ = CoreState::STALLED; }

  /// Clear stall condition (data has arrived).
  void clearStall();

  // --- Inter-core I/O ---

  /// Check if the core has outgoing data on a NoC port.
  bool hasOutgoingData(unsigned nocPortIdx) const;

  /// Consume outgoing data from a NoC port (moves data out of the core).
  std::vector<uint64_t> consumeOutgoingData(unsigned nocPortIdx);

  /// Inject incoming data into a NoC port (delivers data to the core).
  void injectIncomingData(unsigned nocPortIdx,
                          const std::vector<uint64_t> &data);

  /// Register NoC port mappings (boundary ordinal -> NoC port index).
  void mapBoundaryToNoCPort(unsigned boundaryOrdinal, unsigned nocPortIdx,
                            bool isOutput);

  // --- DMA interface ---

  /// Check if the core has pending DMA requests.
  bool hasPendingDMA() const { return !pendingDMAs_.empty(); }

  /// Dequeue the next pending DMA request.
  DMARequest dequeueDMA();

  /// Complete a DMA request and deliver read data back to the core.
  void completeDMA(const DMAResponse &response);

  // --- Status ---

  uint64_t getCycleCount() const { return localCycleCount_; }
  uint64_t getActiveCycles() const { return activeCycles_; }
  uint64_t getStallCycles() const { return stallCycles_; }
  uint64_t getIdleCycles() const { return idleCycles_; }
  bool isComplete() const { return state_ == CoreState::COMPLETED; }
  const std::string &getCoreName() const { return name_; }
  const std::string &getCoreType() const { return coreType_; }
  unsigned getCoreIdx() const { return coreIdx_; }

  /// Get the underlying SimSession's last result (after completion).
  const sim::SimResult &getSimResult() const;

  /// Get output data from a boundary output port.
  std::vector<uint64_t> getOutput(unsigned portIdx) const;

private:
  /// Transition state based on the CycleKernel's reported condition.
  void updateState();

  std::string name_;
  unsigned coreIdx_ = 0;
  std::string coreType_;
  CoreState state_ = CoreState::IDLE;

  /// The underlying single-core simulation session.
  sim::SimSession session_;

  /// Whether the session has been built and configured.
  bool built_ = false;
  bool configured_ = false;
  bool invocationStarted_ = false;

  /// Cycle counters.
  uint64_t localCycleCount_ = 0;
  uint64_t activeCycles_ = 0;
  uint64_t stallCycles_ = 0;
  uint64_t idleCycles_ = 0;

  /// NoC port mappings: boundary ordinal <-> NoC port index.
  std::unordered_map<unsigned, unsigned> outputBoundaryToNoC_;
  std::unordered_map<unsigned, unsigned> inputNoCToBoundary_;

  /// Outgoing data buffers indexed by NoC port.
  std::unordered_map<unsigned, std::deque<std::vector<uint64_t>>>
      outgoingBuffers_;

  /// Incoming data buffers indexed by NoC port.
  std::unordered_map<unsigned, std::deque<std::vector<uint64_t>>>
      incomingBuffers_;

  /// Pending DMA requests from this core.
  std::deque<DMARequest> pendingDMAs_;

  /// Last simulation result from the underlying session.
  sim::SimResult lastResult_;
};

} // namespace mcsim
} // namespace loom

#endif // LOOM_MULTICORESIM_CORESIMWRAPPER_H
