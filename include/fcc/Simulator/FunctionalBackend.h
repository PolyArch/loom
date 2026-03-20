#ifndef FCC_SIMULATOR_FUNCTIONALBACKEND_H
#define FCC_SIMULATOR_FUNCTIONALBACKEND_H

#include "fcc/Simulator/SimSession.h"

namespace fcc {
namespace sim {

class FunctionalSimulationBackend final : public SimulationBackend {
public:
  explicit FunctionalSimulationBackend(const SimConfig &config);
  ~FunctionalSimulationBackend() override;

  std::string connect() override;
  std::string buildFromMappedState(const Graph &dfg, const Graph &adg,
                                   const MappingState &mapping) override;
  std::string buildFromMappedState(const Graph &dfg, const Graph &adg,
                                   const MappingState &mapping,
                                   llvm::ArrayRef<PEContainment> peContainment) override;
  std::string loadConfig(const std::vector<uint8_t> &configBlob) override;
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

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace sim
} // namespace fcc

#endif // FCC_SIMULATOR_FUNCTIONALBACKEND_H
