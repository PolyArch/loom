#ifndef FCC_SIMULATOR_CYCLEKERNEL_H
#define FCC_SIMULATOR_CYCLEKERNEL_H

#include "fcc/Simulator/SimRuntime.h"
#include "fcc/Simulator/SimTypes.h"
#include "fcc/Simulator/SimModule.h"
#include "fcc/Simulator/StaticModel.h"

#include <deque>
#include <cstdint>
#include <optional>
#include <memory>
#include <string>
#include <unordered_set>
#include <unordered_map>
#include <vector>

namespace fcc {
namespace sim {

class CycleKernel final : public SimRuntimeServices {
public:
  explicit CycleKernel(const SimConfig &config = SimConfig());

  bool build(const StaticMappedModel &staticModel);
  bool configure(const StaticConfigImage &configImage);

  void resetExecution();
  void resetAll();
  void setInvocationContext(uint32_t epochId, uint64_t invocationId);

  BoundaryReason getLastBoundaryReason() const { return lastBoundaryReason_; }
  FinalStateSummary getFinalStateSummary() const;
  bool validateSuccessfulTermination(std::string &error) const;

  void stepCycle();
  BoundaryReason runUntilBoundary(uint64_t maxCycles);

  bool isDone() const { return done_; }
  bool isQuiescent() const { return quiescent_; }
  bool isDeadlocked() const { return deadlocked_; }

  const TraceDocument &getTraceDocument() const { return traceDocument_; }
  void setInputTokens(unsigned boundaryOrdinal,
                      const std::vector<SimToken> &tokens);
  const std::vector<SimToken> &getOutputTokens(unsigned boundaryOrdinal) const;
  std::string setMemoryRegionBacking(unsigned regionId, uint8_t *data,
                                     size_t sizeBytes);

  uint64_t getCurrentCycle() const override { return currentCycle_; }
  unsigned getMemoryLatencyCycles(bool isExtMemory) const override;
  bool issueMemoryLoad(uint32_t ownerNodeId, unsigned regionId,
                       uint64_t byteAddr, unsigned byteWidth, uint16_t tag,
                       bool hasTag, uint64_t &requestId,
                       std::string &error) override;
  bool issueMemoryStore(uint32_t ownerNodeId, unsigned regionId,
                        uint64_t byteAddr, uint64_t data,
                        unsigned byteWidth, uint16_t tag, bool hasTag,
                        uint64_t &requestId, std::string &error) override;
  bool takeMemoryCompletion(uint64_t requestId,
                            MemoryCompletion &completion) override;
  bool hasOutstandingMemoryRequest(uint64_t requestId) const override;
  bool regionHasOutstandingRequests(unsigned regionId) const override;
  bool bindMemoryRegion(unsigned regionId, uint8_t *data, size_t sizeBytes,
                        std::string &error) override;

private:
  struct BoundMemoryRegion {
    uint8_t *data = nullptr;
    size_t sizeBytes = 0;
  };

  struct OutstandingMemoryRequest {
    uint64_t requestId = 0;
    uint64_t readyCycle = 0;
    MemoryRequestKind kind = MemoryRequestKind::Load;
    unsigned regionId = 0;
    uint32_t ownerNodeId = 0;
    uint64_t byteAddr = 0;
    uint64_t data = 0;
    unsigned byteWidth = 0;
    uint16_t tag = 0;
    bool hasTag = false;
  };

  struct OutputFanoutState {
    uint64_t generation = 0;
    std::vector<uint8_t> captured;
    bool completionEmitted = false;
  };

  void rebuildPortSignalsFromSnapshot(const std::vector<SimChannel> &snapshot);
  void finalizePortSignals();
  void rebuildVisibleEdgeSignals();
  void evaluateBoundaryState();
  void appendKernelEvent(uint64_t cycle, SimPhase phase, uint32_t hwNodeId,
                         EventKind kind, uint32_t arg0 = 0,
                         uint32_t arg1 = 0);
  bool edgeCanAcceptNow(unsigned edgeIdx,
                        const std::vector<SimChannel> &snapshot) const;
  void syncOutputFanoutState(IdIndex outputPortId, uint64_t generation);
  bool completionObligationsSatisfied() const;
  bool hardwareEmpty(std::string *details = nullptr) const;
  bool hasPendingInternalWork() const;
  bool hasPendingModuleOrMemoryWork() const;
  bool memoryRegionCompletionObserved(unsigned regionId) const;
  bool checkMemoryAccess(unsigned regionId, uint64_t byteAddr,
                         unsigned byteWidth, std::string &error) const;
  void retireReadyMemoryRequests();

  SimConfig config_;
  bool built_ = false;
  bool configured_ = false;
  bool done_ = false;
  bool quiescent_ = false;
  bool deadlocked_ = false;
  uint64_t currentCycle_ = 0;
  BoundaryReason lastBoundaryReason_ = BoundaryReason::None;
  StaticMappedModel staticModel_;
  StaticConfigImage configImage_;
  TraceDocument traceDocument_;
  std::vector<SimChannel> portState_;
  std::vector<SimChannel> edgeState_;
  std::vector<IdIndex> inputSourcePort_;
  std::vector<std::vector<IdIndex>> outputDestPorts_;
  std::vector<int> inputChannelIndex_;
  std::vector<std::vector<unsigned>> outputChannelIndices_;
  std::vector<OutputFanoutState> outputFanoutState_;
  std::vector<uint8_t> forcedReadyOutputPort_;
  std::vector<std::unique_ptr<SimModule>> modules_;
  std::vector<int> boundaryInputModuleIndex_;
  std::vector<int> boundaryOutputModuleIndex_;
  std::unordered_set<IdIndex> debugInterestingNodes_;
  uint64_t lastTransferCount_ = 0;
  uint64_t lastActivityCount_ = 0;
  uint64_t idleCycleStreak_ = 0;
  std::vector<BoundMemoryRegion> boundMemoryRegions_;
  std::vector<OutstandingMemoryRequest> outstandingMemoryRequests_;
  std::unordered_map<uint64_t, MemoryCompletion> completedMemoryRequests_;
  std::vector<uint8_t> completedStoreRegions_;
  uint64_t nextMemoryRequestId_ = 1;
};

} // namespace sim
} // namespace fcc

#endif // FCC_SIMULATOR_CYCLEKERNEL_H
