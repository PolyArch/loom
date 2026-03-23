#ifndef LOOM_MULTICORESIM_MULTICORESIMSESSION_H
#define LOOM_MULTICORESIM_MULTICORESIMSESSION_H

#include "loom/MultiCoreSim/CoreSimWrapper.h"
#include "loom/MultiCoreSim/MemoryHierarchyModel.h"
#include "loom/MultiCoreSim/NoCSimModel.h"
#include "loom/MultiCoreSim/TapestryTypes.h"
#include "loom/Simulator/SimTypes.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace loom {
namespace mcsim {

// Configuration for the multi-core simulation session.
struct MultiCoreSimConfig {
  sim::SimConfig perCoreSimConfig;
  MemoryHierarchyConfig memConfig;
  unsigned nocPerHopLatency = 1;
  uint64_t maxGlobalCycles = 10000000;
};

// Orchestrates lockstep simulation of multiple cores, coordinating
// the NoC and memory hierarchy models.
class MultiCoreSimSession {
public:
  explicit MultiCoreSimSession(const MultiCoreSimConfig &config);
  ~MultiCoreSimSession();

  MultiCoreSimSession(const MultiCoreSimSession &) = delete;
  MultiCoreSimSession &operator=(const MultiCoreSimSession &) = delete;

  // Initialize from a complete TapestryCompilationResult.
  // Creates CoreSimWrappers for each core with mapped kernels,
  // configures the NoC model, and sets up the memory hierarchy.
  std::string initialize(const TapestryCompilationResult &compilationResult);

  // Run the entire multi-core simulation to completion.
  // All cores run in lockstep, with the NoC and memory hierarchy
  // models advanced each cycle.
  MultiCoreSimResult run();

  // Get the number of active cores.
  unsigned getNumActiveCores() const;

  // Get a reference to a specific core wrapper (for inspection).
  const CoreSimWrapper *getCoreWrapper(unsigned coreId) const;

  // Get the NoC model (for inspection).
  const NoCSimModel *getNoCModel() const;

  // Get the memory hierarchy model (for inspection).
  const MemoryHierarchyModel *getMemoryModel() const;

private:
  // Process cross-core transfers: check if any core has produced
  // data for another core, inject it into the NoC.
  void processNoCInjections();

  // Deliver arrived NoC flits to destination cores.
  void processNoCDeliveries();

  // Check if all cores are done.
  bool allCoresDone() const;

  MultiCoreSimConfig config_;
  std::vector<std::unique_ptr<CoreSimWrapper>> coreWrappers_;
  std::unique_ptr<NoCSimModel> nocModel_;
  std::unique_ptr<MemoryHierarchyModel> memModel_;

  // Mapping from core ID to coreWrappers_ index.
  std::unordered_map<unsigned, unsigned> coreIdToIndex_;

  // Cross-core transfer contracts.
  std::vector<NoCTransferContract> transferContracts_;

  uint64_t globalCycle_ = 0;
};

} // namespace mcsim
} // namespace loom

#endif // LOOM_MULTICORESIM_MULTICORESIMSESSION_H
