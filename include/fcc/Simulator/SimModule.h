#ifndef FCC_SIMULATOR_SIMMODULE_H
#define FCC_SIMULATOR_SIMMODULE_H

#include "fcc/Simulator/SimRuntime.h"
#include "fcc/Simulator/SimTypes.h"
#include "fcc/Simulator/StaticModel.h"

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <memory>
#include <string>
#include <vector>

namespace fcc {
namespace sim {

struct SimToken {
  uint64_t data = 0;
  uint16_t tag = 0;
  bool hasTag = false;
  uint64_t generation = 0;
};

inline uint64_t composeTokenGeneration(uint32_t hwNodeId, uint64_t localId) {
  return (uint64_t{hwNodeId} << 32) | (localId & 0xffffffffULL);
}

inline SimToken tokenFromChannel(const SimChannel &channel) {
  SimToken token;
  token.data = channel.data;
  token.tag = channel.tag;
  token.hasTag = channel.hasTag;
  token.generation = channel.generation;
  return token;
}

inline void driveChannelFromToken(SimChannel &channel, const SimToken &token) {
  channel.valid = true;
  channel.data = token.data;
  channel.tag = token.tag;
  channel.hasTag = token.hasTag;
  channel.generation = token.generation;
}

class SimModule {
public:
  virtual ~SimModule() = default;

  virtual bool isCombinational() const = 0;
  virtual void reset() = 0;
  virtual void configure(const std::vector<uint32_t> &configWords) = 0;
  virtual void evaluate() = 0;
  virtual void commit() {}
  virtual bool hasPendingWork() const { return false; }
  virtual void collectTraceEvents(std::vector<TraceEvent> &events,
                                  uint64_t cycle) = 0;
  virtual PerfSnapshot getPerfSnapshot() const = 0;
  virtual void bindRuntimeServices(SimRuntimeServices *services) {
    (void)services;
  }

  virtual void setInputTokens(const std::vector<SimToken> &tokens) {
    (void)tokens;
  }
  virtual void debugDump(std::ostream &os) const { (void)os; }
  virtual const std::vector<SimToken> &getCollectedTokens() const {
    static const std::vector<SimToken> empty;
    return empty;
  }
  virtual uint64_t getLogicalFireCount() const { return 0; }
  virtual uint64_t getInputCaptureCount() const { return 0; }
  virtual uint64_t getOutputTransferCount() const { return 0; }
  virtual std::vector<NamedCounter> getDebugCounters() const { return {}; }
  virtual std::string getDebugStateSummary() const { return {}; }

  uint32_t hwNodeId = 0;
  std::string name;
  StaticModuleKind kind = StaticModuleKind::Unknown;
  std::vector<SimChannel *> inputs;
  std::vector<SimChannel *> outputs;

protected:
  PerfSnapshot perf_;
};

std::unique_ptr<SimModule> createSimModule(const StaticModuleDesc &module,
                                           const StaticMappedModel &model);

} // namespace sim
} // namespace fcc

#endif // FCC_SIMULATOR_SIMMODULE_H
