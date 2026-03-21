#ifndef FCC_SIMULATOR_SIMSESSION_H
#define FCC_SIMULATOR_SIMSESSION_H

#include "fcc/Mapper/Graph.h"
#include "fcc/Mapper/ConfigGen.h"
#include "fcc/Mapper/ADGFlattener.h"
#include "fcc/Mapper/MappingState.h"
#include "fcc/Simulator/SimTypes.h"
#include "fcc/Simulator/StaticModel.h"

#include "llvm/ADT/ArrayRef.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace fcc {
namespace sim {

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

struct CompareResult {
  bool pass = false;
  unsigned totalOutputs = 0;
  unsigned mismatches = 0;
  std::string details;
};

const char *sessionStateName(SessionState state);

class SimulationBackend {
public:
  virtual ~SimulationBackend() = default;

  virtual std::string connect() { return {}; }

  virtual std::string buildFromMappedState(const Graph &dfg, const Graph &adg,
                                           const MappingState &mapping) = 0;
  virtual std::string
  buildFromMappedState(const Graph &dfg, const Graph &adg,
                       const MappingState &mapping,
                       llvm::ArrayRef<PEContainment> peContainment) {
    (void)peContainment;
    return buildFromMappedState(dfg, adg, mapping);
  }
  virtual std::string buildFromStaticModel(const StaticMappedModel &model) {
    (void)model;
    return "backend does not support buildFromStaticModel";
  }
  virtual std::string buildFromStaticModel(StaticMappedModel &&model) {
    return buildFromStaticModel(model);
  }

  virtual std::string loadConfig(const std::vector<uint8_t> &configBlob) = 0;
  virtual std::string loadConfig(
      const std::vector<uint8_t> &configBlob,
      llvm::ArrayRef<fcc::ConfigGen::ConfigSlice> configSlices) {
    (void)configSlices;
    return loadConfig(configBlob);
  }

  virtual std::string setInput(unsigned portIdx,
                               const std::vector<uint64_t> &data,
                               const std::vector<uint16_t> &tags) = 0;

  virtual std::string setExtMemoryBacking(unsigned regionId, uint8_t *data,
                                          size_t sizeBytes) = 0;

  virtual SimResult invoke(uint32_t epochId, uint64_t invocationId) = 0;

  virtual std::vector<uint64_t> getOutput(unsigned portIdx) const { return {}; }
  virtual std::vector<uint16_t> getOutputTags(unsigned portIdx) const {
    return {};
  }

  virtual void resetExecution() {}
  virtual void resetAll() {}

  virtual unsigned getNumInputPorts() const { return 0; }
  virtual unsigned getNumOutputPorts() const { return 0; }
};

class SimSession {
public:
  explicit SimSession(std::unique_ptr<SimulationBackend> backend = nullptr,
                      const SimConfig &config = SimConfig());
  ~SimSession();

  SimSession(const SimSession &) = delete;
  SimSession &operator=(const SimSession &) = delete;
  SimSession(SimSession &&other) noexcept;
  SimSession &operator=(SimSession &&other) noexcept;

  SessionState getState() const;
  std::string connect();
  std::string buildFromMappedState(const Graph &dfg, const Graph &adg,
                                   const MappingState &mapping);
  std::string buildFromMappedState(const Graph &dfg, const Graph &adg,
                                   const MappingState &mapping,
                                   llvm::ArrayRef<PEContainment> peContainment);
  std::string buildFromStaticModel(const StaticMappedModel &model);
  std::string buildFromStaticModel(StaticMappedModel &&model);
  std::string loadConfig(const std::vector<uint8_t> &configBlob);
  std::string loadConfig(
      const std::vector<uint8_t> &configBlob,
      llvm::ArrayRef<fcc::ConfigGen::ConfigSlice> configSlices);
  std::string setInput(unsigned portIdx, const std::vector<uint64_t> &data,
                       const std::vector<uint16_t> &tags = {});
  std::string setExtMemoryBacking(unsigned regionId, uint8_t *data,
                                  size_t sizeBytes);
  std::pair<SimResult, std::string> invoke();
  std::vector<uint64_t> getOutput(unsigned portIdx) const;
  std::vector<uint16_t> getOutputTags(unsigned portIdx) const;
  CompareResult
  compareOutputPorts(const std::vector<std::vector<uint64_t>> &reference) const;
  CompareResult compareMemoryRegion(unsigned regionId,
                                    llvm::ArrayRef<uint8_t> expected) const;
  std::string resetExecution();
  std::string resetAll();
  std::string disconnect();
  const SimResult &getLastResult() const;
  uint32_t getEpochId() const;
  uint64_t getInvocationId() const;
  unsigned getNumInputPorts() const;
  unsigned getNumOutputPorts() const;
  size_t getNumBoundMemoryRegions() const;

private:
  struct MemoryRegionBinding {
    uint8_t *data = nullptr;
    size_t sizeBytes = 0;
  };

  std::string validateTransition(SessionState from, SessionState to) const;
  static std::unique_ptr<SimulationBackend>
  createDefaultBackend(const SimConfig &config);

  mutable std::mutex mu_;
  SessionState state_ = SessionState::Created;
  uint32_t epochId_ = 0;
  uint64_t invocationId_ = 0;
  SimConfig config_;
  std::unique_ptr<SimulationBackend> backend_;
  SimResult lastResult_;
  std::vector<uint8_t> configBlob_;
  std::vector<fcc::ConfigGen::ConfigSlice> configSlices_;
  std::vector<MemoryRegionBinding> memoryRegions_;
};

} // namespace sim
} // namespace fcc

#endif // FCC_SIMULATOR_SIMSESSION_H
