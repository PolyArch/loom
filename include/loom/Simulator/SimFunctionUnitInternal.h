#ifndef LOOM_SIMULATOR_SIMFUNCTIONUNITINTERNAL_H
#define LOOM_SIMULATOR_SIMFUNCTIONUNITINTERNAL_H

#include "loom/Simulator/SimFunctionUnit.h"

#include <cstdint>
#include <deque>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace loom {
namespace sim {

unsigned log2Ceil32(unsigned value);

int64_t getIntAttr(const StaticModuleDesc &module, std::string_view name,
                   int64_t defaultValue = 0);

const std::vector<std::string> *
getStringArrayAttr(const StaticModuleDesc &module, std::string_view name);

const StaticPortDesc *findPort(const StaticMappedModel &model, IdIndex portId);

enum class BodyType : uint8_t {
  Unsupported = 0,
  Compute,
  Constant,
  Select,
  CondBranch,
  HandshakeMux,
  Join,
  Load,
  Store,
  Stream,
  Carry,
  Invariant,
  Gate,
};

bool isBinaryComputeOp(std::string_view op);
bool isUnaryComputeOp(std::string_view op);
std::string modulePrimaryOp(const StaticModuleDesc &module);
BodyType classifyFunctionUnitBody(const StaticModuleDesc &module);

struct DecodedMuxField {
  uint64_t sel = 0;
  bool discard = false;
  bool disconnect = true;
};

class FunctionUnitModule final : public SimModule {
public:
  FunctionUnitModule(const StaticModuleDesc &module,
                     const StaticMappedModel &model);

  bool isCombinational() const override;
  void reset() override;
  void configure(const std::vector<uint32_t> &configWords) override;
  void evaluate() override;
  void commit() override;
  void collectTraceEvents(std::vector<TraceEvent> &events,
                          uint64_t cycle) override;
  PerfSnapshot getPerfSnapshot() const override;
  void debugDump(std::ostream &os) const override;
  bool hasPendingWork() const override;
  uint64_t getLogicalFireCount() const override;
  uint64_t getInputCaptureCount() const override;
  uint64_t getOutputTransferCount() const override;
  std::vector<NamedCounter> getDebugCounters() const override;
  std::string getDebugStateSummary() const override;

private:
  bool usesElasticOutputQueues() const;
  void enqueueOutputToken(size_t outputIdx, const SimToken &token);
  void clearOutputBuffer(size_t outputIdx);

  struct InflightResult {
    unsigned remainingCycles = 0;
    std::vector<std::optional<SimToken>> outputs;
  };

  static std::string resolvePrimaryOp(const StaticModuleDesc &module);
  static unsigned inferDataWidth(const StaticModuleDesc &module,
                                 const StaticMappedModel &model);
  static std::vector<unsigned> inferOutputWidths(const StaticModuleDesc &module,
                                                 const StaticMappedModel &model);
  static std::vector<unsigned> inferInputWidths(const StaticModuleDesc &module,
                                                const StaticMappedModel &model);
  static unsigned countConfigMuxFields(const StaticModuleDesc &module);
  static unsigned inferConstantValueWidth(const StaticModuleDesc &module,
                                          const StaticMappedModel &model);

  uint64_t maskToWidth(uint64_t value, unsigned width) const;
  int64_t signExtend(uint64_t value, unsigned width) const;
  float toFloat(uint64_t value) const;
  uint64_t fromFloat(float value) const;
  double toDouble(uint64_t value) const;
  uint64_t fromDouble(double value) const;
  bool cmpi(uint64_t a, uint64_t b, unsigned width) const;
  bool cmpf(uint64_t a, uint64_t b, unsigned width) const;

  void setAllInputReady(bool ready);
  bool inputFresh(size_t idx) const;
  bool inputAlreadyConsumed(size_t idx) const;
  void setInputReadyFreshAware(size_t idx, bool allowFresh);
  void markInputConsumed(size_t idx);
  uint64_t allocateOutputGeneration();
  uint64_t reserveDirectGeneration();
  void finalizeDirectGeneration();
  SimToken makeGeneratedToken(size_t outputIdx, uint64_t data,
                              uint16_t tag = 0, bool hasTag = false,
                              uint64_t generation = 0) const;
  bool anyOutputRegisterBusy() const;
  bool canDrainIntoOutputRegisters(
      const std::vector<std::optional<SimToken>> &outputs) const;
  bool outputRegisterFree(size_t outputIdx) const;
  bool outputRegistersFree(std::initializer_list<size_t> outputIdxs) const;
  void drainToOutputRegisters(
      const std::vector<std::optional<SimToken>> &outputs);
  void driveBufferedOutputs();
  void clearDirectOutputs();
  void driveDirectOutputs();
  void commitOutputTransfers();
  std::optional<std::vector<std::optional<SimToken>>>
  computeOutputs(uint64_t generation) const;

  void evaluateFireReadinessForStrictInputs();
  void evaluateMuxReadiness();
  void evaluateLoadReadiness();
  void evaluateStoreReadiness();
  void evaluateCombinationalBody(bool readyForNewFire);
  void commitZeroLatencyBody();
  bool canFireCurrentBody() const;
  void consumeInputsForCurrentBody();
  unsigned countConsumedInputsForCurrentBody() const;
  void evaluateStatefulBody();
  void commitStatefulBody();
  bool bodyUsesGenericOperandLatches() const;
  bool inputRequiredForCurrentBody(size_t idx) const;
  bool inputAvailableForCurrentBody(size_t idx) const;
  unsigned countFreshInputsForCurrentBody() const;
  unsigned countLatchedOperands() const;
  void captureGenericOperands();

  void evaluateCarry();
  void commitLoad();
  void commitStore();
  void commitCarry();
  void captureMuxOperands();
  void evaluateInvariant();
  void commitInvariant();
  void evaluateGate();
  void commitGate();
  bool evalStreamCond(int64_t idx, int64_t bound) const;
  uint64_t nextStreamValue(uint64_t idx, uint64_t step) const;
  void evaluateStream();
  void commitStream();

  std::string opName_;
  BodyType bodyType_ = BodyType::Unsupported;
  int64_t latency_ = 1;
  int64_t interval_ = 1;
  unsigned dataWidth_ = 32;
  std::vector<unsigned> outputWidths_;
  std::vector<unsigned> inputWidths_;
  unsigned maxOutputs_ = 0;
  bool hasStatefulBody_ = false;
  bool directFireArmed_ = false;

  uint64_t cyclesSinceLastFire_ = 0;
  std::deque<InflightResult> inflight_;
  std::vector<std::optional<SimToken>> outputRegisters_;
  std::vector<std::deque<SimToken>> outputQueues_;
  std::vector<std::optional<SimToken>> directOutputs_;
  std::vector<uint64_t> consumedInputGeneration_;
  uint64_t nextOutputGeneration_ = 1;
  uint64_t pendingDirectGeneration_ = 0;
  unsigned countdown_ = 0;

  uint64_t configuredConstantValue_ = 0;
  unsigned constantValueWidth_ = 0;
  uint8_t cmpPredicate_ = 0;
  uint8_t streamContCond_ = 1;
  uint64_t joinMask_ = 0;
  uint64_t muxSel_ = 0;
  std::vector<DecodedMuxField> fabricMuxFields_;
  bool emittedFireThisCycle_ = false;
  uint64_t logicalFireCount_ = 0;
  uint64_t inputCaptureCount_ = 0;
  uint64_t outputTransferCount_ = 0;
  uint64_t loadIssueCount_ = 0;
  uint64_t loadReturnCount_ = 0;
  uint64_t storeIssueCount_ = 0;
  uint64_t condTrueCount_ = 0;
  uint64_t condFalseCount_ = 0;
  uint64_t streamEmitCount_ = 0;
  uint64_t streamTerminalCount_ = 0;
  uint64_t gateHeadCount_ = 0;
  uint64_t gateTrueCount_ = 0;
  uint64_t gateFalseCount_ = 0;
  uint64_t carryInitCount_ = 0;
  uint64_t carryLoopCount_ = 0;
  uint64_t carryResetCount_ = 0;
  uint64_t invariantInitCount_ = 0;
  uint64_t invariantLoopCount_ = 0;
  uint64_t invariantResetCount_ = 0;
  bool loadIssueSelected_ = false;
  bool loadReturnSelected_ = false;
  bool storeIssueSelected_ = false;
  std::optional<SimToken> muxSelectorToken_;
  std::optional<size_t> muxSelectedInputIdx_;
  std::optional<SimToken> muxSelectedDataToken_;
  std::optional<SimToken> loadAddrToken_;
  std::optional<SimToken> loadDataToken_;
  std::optional<SimToken> loadCtrlToken_;
  std::optional<SimToken> storeAddrToken_;
  std::optional<SimToken> storeDataToken_;
  std::optional<SimToken> storeCtrlToken_;
  std::optional<SimToken> streamStartToken_;
  std::optional<SimToken> streamStepToken_;
  std::optional<SimToken> streamBoundToken_;
  std::vector<std::optional<SimToken>> operandTokens_;

  bool dataflowInitialStage_ = true;
  uint64_t invariantStoredValue_ = 0;
  bool initLatched_ = false;
  uint64_t initLatchedValue_ = 0;
  bool carryDLatched_ = false;
  bool carryDValue_ = false;
  bool streamInitialPhase_ = true;
  bool streamTerminalPending_ = false;
  uint64_t streamNextIdx_ = 0;
  uint64_t streamBoundReg_ = 0;
  uint64_t streamStepReg_ = 0;
  unsigned gateState_ = 0;
  uint64_t gateLatchedValue_ = 0;
  bool gateLatchedCond_ = false;
  std::optional<SimToken> gateValueToken_;
  std::optional<SimToken> gateCondToken_;
  std::vector<bool> gateOutputsAccepted_;
  std::vector<bool> streamOutputsAccepted_;
};

} // namespace sim
} // namespace loom

#endif // LOOM_SIMULATOR_SIMFUNCTIONUNITINTERNAL_H
