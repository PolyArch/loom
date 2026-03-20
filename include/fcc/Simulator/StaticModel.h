#ifndef FCC_SIMULATOR_STATICMODEL_H
#define FCC_SIMULATOR_STATICMODEL_H

#include "fcc/Simulator/StaticModelTypes.h"

#include <optional>
#include <vector>

namespace fcc {

namespace sim {

class StaticMappedModel {
public:
  const std::vector<StaticModuleDesc> &getModules() const { return modules_; }
  const std::vector<StaticChannelDesc> &getChannels() const { return channels_; }
  const std::vector<StaticPortDesc> &getPorts() const { return ports_; }
  const std::vector<StaticPEDesc> &getPEs() const { return pes_; }
  const std::vector<StaticInputBinding> &getInputBindings() const {
    return inputBindings_;
  }
  const std::vector<StaticOutputBinding> &getOutputBindings() const {
    return outputBindings_;
  }
  const std::vector<StaticMemoryBinding> &getMemoryBindings() const {
    return memoryBindings_;
  }
  const std::vector<CompletionObligation> &getCompletionObligations() const {
    return obligations_;
  }
  const std::vector<unsigned> &getBoundaryInputOrdinals() const {
    return boundaryInputOrdinals_;
  }
  const std::vector<unsigned> &getBoundaryOutputOrdinals() const {
    return boundaryOutputOrdinals_;
  }

  std::optional<unsigned> getBoundaryInputOrdinal(IdIndex hwNodeId) const;
  std::optional<unsigned> getBoundaryOutputOrdinal(IdIndex hwNodeId) const;
  const StaticModuleDesc *findModule(IdIndex hwNodeId) const;
  const StaticPortDesc *findPort(IdIndex portId) const;

  std::vector<StaticModuleDesc> &mutableModules() { return modules_; }
  std::vector<StaticChannelDesc> &mutableChannels() { return channels_; }
  std::vector<StaticPortDesc> &mutablePorts() { return ports_; }
  std::vector<StaticPEDesc> &mutablePEs() { return pes_; }
  std::vector<StaticInputBinding> &mutableInputBindings() { return inputBindings_; }
  std::vector<StaticOutputBinding> &mutableOutputBindings() { return outputBindings_; }
  std::vector<StaticMemoryBinding> &mutableMemoryBindings() { return memoryBindings_; }
  std::vector<CompletionObligation> &mutableCompletionObligations() {
    return obligations_;
  }
  std::vector<unsigned> &mutableBoundaryInputOrdinals() {
    return boundaryInputOrdinals_;
  }
  std::vector<unsigned> &mutableBoundaryOutputOrdinals() {
    return boundaryOutputOrdinals_;
  }

private:
  std::vector<StaticModuleDesc> modules_;
  std::vector<StaticChannelDesc> channels_;
  std::vector<StaticPortDesc> ports_;
  std::vector<StaticPEDesc> pes_;
  std::vector<StaticInputBinding> inputBindings_;
  std::vector<StaticOutputBinding> outputBindings_;
  std::vector<StaticMemoryBinding> memoryBindings_;
  std::vector<CompletionObligation> obligations_;
  std::vector<unsigned> boundaryInputOrdinals_;
  std::vector<unsigned> boundaryOutputOrdinals_;
};

} // namespace sim
} // namespace fcc

#endif // FCC_SIMULATOR_STATICMODEL_H
