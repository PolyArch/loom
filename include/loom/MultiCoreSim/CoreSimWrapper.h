#ifndef LOOM_MULTICORESIM_CORESIMWRAPPER_H
#define LOOM_MULTICORESIM_CORESIMWRAPPER_H

#include "loom/MultiCoreSim/TapestryTypes.h"
#include "loom/Simulator/CycleBackend.h"
#include "loom/Simulator/SimSession.h"
#include "loom/Simulator/SimTypes.h"

#include <cstdint>
#include <deque>
#include <memory>
#include <string>
#include <vector>

namespace loom {
namespace mcsim {

// Wraps a real SimSession for one core in the multi-core simulation.
// Each CoreSimWrapper manages its own SimSession lifecycle and
// provides per-cycle stepping for lockstep orchestration.
class CoreSimWrapper {
public:
  CoreSimWrapper(unsigned coreId, const sim::SimConfig &config);
  ~CoreSimWrapper();

  CoreSimWrapper(const CoreSimWrapper &) = delete;
  CoreSimWrapper &operator=(const CoreSimWrapper &) = delete;
  CoreSimWrapper(CoreSimWrapper &&) noexcept;
  CoreSimWrapper &operator=(CoreSimWrapper &&) noexcept;

  // Initialize from a CoreKernelResult: builds the static model and
  // loads the config. Returns empty string on success, error otherwise.
  std::string initialize(const CoreKernelResult &kernelResult);

  // Bind synthesized inputs from the kernel result.
  std::string bindSynthesizedInputs(const sim::SynthesizedSetup &setup);

  // Run the full simulation to completion (non-stepping mode).
  // Returns the SimResult including cycle counts.
  std::pair<sim::SimResult, std::string> runToCompletion();

  // Per-cycle stepping support for lockstep mode.
  // Returns true if the core is still running (not yet done/error).
  bool stepOneCycle();

  // Check if this core has finished execution.
  bool isDone() const;

  // Check if this core encountered an error.
  bool hasError() const;

  // Get the accumulated cycle count.
  uint64_t getCurrentCycle() const;

  // Get the final SimResult (valid after runToCompletion or done).
  const sim::SimResult &getResult() const;

  // Get error message if any.
  const std::string &getError() const;

  // Get the core ID.
  unsigned getCoreId() const { return coreId_; }

  // Inject a cross-core token into the specified input port.
  std::string injectToken(unsigned portIdx, uint64_t data, uint16_t tag,
                          bool hasTag);

  // Extract produced output data from the specified output port.
  std::vector<uint64_t> extractOutput(unsigned portIdx) const;

  // Get the underlying SimSession (for advanced queries).
  sim::SimSession &getSession() { return session_; }
  const sim::SimSession &getSession() const { return session_; }

private:
  unsigned coreId_ = 0;
  sim::SimConfig config_;
  sim::SimSession session_;
  sim::SimResult result_;
  std::string error_;
  bool initialized_ = false;
  bool done_ = false;
  bool hasError_ = false;
  uint64_t cycleCount_ = 0;

  // Memory storage owned by this wrapper for synthesized regions.
  std::vector<std::vector<uint8_t>> memoryStorage_;
};

} // namespace mcsim
} // namespace loom

#endif // LOOM_MULTICORESIM_CORESIMWRAPPER_H
