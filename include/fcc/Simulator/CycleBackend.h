#ifndef FCC_SIMULATOR_CYCLEBACKEND_H
#define FCC_SIMULATOR_CYCLEBACKEND_H

#include "fcc/Simulator/CycleKernel.h"
#include "fcc/Simulator/SimSession.h"
#include "fcc/Simulator/StaticModel.h"

#include <memory>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace fcc {
namespace sim {

class CycleSimulationBackend final : public SimulationBackend {
public:
  explicit CycleSimulationBackend(const SimConfig &config);
  ~CycleSimulationBackend() override;

  std::string connect() override;
  std::string buildFromMappedState(const Graph &dfg, const Graph &adg,
                                   const MappingState &mapping) override;
  std::string buildFromMappedState(const Graph &dfg, const Graph &adg,
                                   const MappingState &mapping,
                                   llvm::ArrayRef<PEContainment> peContainment) override;
  std::string buildFromStaticModel(const StaticMappedModel &model) override;
  std::string buildFromStaticModel(StaticMappedModel &&model) override;
  std::string loadConfig(const std::vector<uint8_t> &configBlob) override;
  std::string loadConfig(
      const std::vector<uint8_t> &configBlob,
      llvm::ArrayRef<fcc::ConfigGen::ConfigSlice> configSlices) override;
  std::string setInput(unsigned portIdx, const std::vector<uint64_t> &data,
                       const std::vector<uint16_t> &tags) override;
  std::string setExtMemoryBacking(unsigned regionId, uint8_t *data,
                                  size_t sizeBytes) override;
  SimResult invoke(uint32_t epochId, uint64_t invocationId) override;

  std::vector<uint64_t> getOutput(unsigned portIdx) const override;
  std::vector<uint16_t> getOutputTags(unsigned portIdx) const override;

  void resetExecution() override;
  void resetAll() override;

  unsigned getNumInputPorts() const override;
  unsigned getNumOutputPorts() const override;

  const StaticMappedModel &getStaticModel() const { return staticModel_; }
  const StaticConfigImage &getConfigImage() const { return configImage_; }

private:
  struct MemoryRegionBinding {
    uint8_t *data = nullptr;
    size_t sizeBytes = 0;
  };

  bool modelSupportsKernelExecution() const;
  bool serviceOutgoingMemoryRequests(std::string &error);

  SimConfig config_;
  StaticMappedModel staticModel_;
  StaticConfigImage configImage_;
  CycleKernel kernel_;
  std::unique_ptr<SimulationBackend> fallback_;
  bool useKernelExecution_ = false;
  bool hasFallbackGraph_ = false;
  std::vector<std::vector<SimToken>> pendingInputs_;
  std::vector<std::vector<SimToken>> collectedOutputs_;
  std::vector<MemoryRegionBinding> memoryBindings_;
};

} // namespace sim
} // namespace fcc

#endif // FCC_SIMULATOR_CYCLEBACKEND_H
