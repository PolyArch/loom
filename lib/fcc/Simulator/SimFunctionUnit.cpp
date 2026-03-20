#include "fcc/Simulator/SimFunctionUnit.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <deque>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace fcc {
namespace sim {

namespace {

unsigned log2Ceil32(unsigned value) {
  if (value <= 1)
    return 0;
  unsigned bits = 0;
  value -= 1u;
  while (value != 0) {
    ++bits;
    value >>= 1;
  }
  return bits;
}

int64_t getIntAttr(const StaticModuleDesc &module, std::string_view name,
                   int64_t defaultValue = 0) {
  for (const auto &attr : module.intAttrs) {
    if (attr.name == name)
      return attr.value;
  }
  return defaultValue;
}

const std::vector<std::string> *
getStringArrayAttr(const StaticModuleDesc &module, std::string_view name) {
  for (const auto &attr : module.stringArrayAttrs) {
    if (attr.name == name)
      return &attr.value;
  }
  return nullptr;
}

const StaticPortDesc *findPort(const StaticMappedModel &model, IdIndex portId) {
  return model.findPort(portId);
}

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

std::string modulePrimaryOp(const StaticModuleDesc &module) {
  const auto *ops = getStringArrayAttr(module, "ops");
  if (!ops || ops->empty())
    return {};
  return ops->front();
}

BodyType classifyFunctionUnitBody(const StaticModuleDesc &module) {
  std::string opName = modulePrimaryOp(module);
  if (opName.empty())
    return BodyType::Unsupported;
  if (opName == "handshake.constant")
    return BodyType::Constant;
  if (opName == "arith.select" || opName == "fabric.mux")
    return BodyType::Select;
  if (opName == "handshake.cond_br")
    return BodyType::CondBranch;
  if (opName == "handshake.mux")
    return BodyType::HandshakeMux;
  if (opName == "handshake.join")
    return BodyType::Join;
  if (opName == "handshake.load")
    return BodyType::Load;
  if (opName == "handshake.store")
    return BodyType::Store;
  if (opName == "dataflow.stream")
    return BodyType::Stream;
  if (opName == "dataflow.carry")
    return BodyType::Carry;
  if (opName == "dataflow.invariant")
    return BodyType::Invariant;
  if (opName == "dataflow.gate")
    return BodyType::Gate;
  if (isBinaryComputeOp(opName) || isUnaryComputeOp(opName))
    return BodyType::Compute;
  return BodyType::Unsupported;
}

bool isBinaryComputeOp(std::string_view op) {
  return op == "arith.addi" || op == "arith.subi" || op == "arith.muli" ||
         op == "arith.divsi" || op == "arith.divui" || op == "arith.remsi" ||
         op == "arith.remui" || op == "arith.andi" || op == "arith.ori" ||
         op == "arith.xori" || op == "arith.shli" || op == "arith.shrsi" ||
         op == "arith.shrui" || op == "arith.cmpi" || op == "arith.cmpf" ||
         op == "arith.addf" || op == "arith.subf" || op == "arith.mulf" ||
         op == "arith.divf" || op == "math.fma";
}

bool isUnaryComputeOp(std::string_view op) {
  return op == "arith.extsi" || op == "arith.extui" ||
         op == "arith.trunci" || op == "arith.index_cast" ||
         op == "arith.index_castui" || op == "arith.negf" ||
         op == "arith.fptosi" || op == "arith.fptoui" ||
         op == "arith.sitofp" || op == "arith.uitofp" ||
         op == "math.absf" || op == "math.cos" || op == "math.exp" ||
         op == "math.log2" || op == "math.sin" || op == "math.sqrt" ||
         op == "llvm.intr.bitreverse";
}

struct DecodedMuxField {
  uint64_t sel = 0;
  bool discard = false;
  bool disconnect = true;
};

class FunctionUnitModule final : public SimModule {
public:
  FunctionUnitModule(const StaticModuleDesc &module, const StaticMappedModel &model)
      : opName_(modulePrimaryOp(module)),
        bodyType_(classifyFunctionUnitBody(module)),
        latency_(std::max<int64_t>(0, getIntAttr(module, "latency", 1))),
        interval_(std::max<int64_t>(1, getIntAttr(module, "interval", 1))),
        dataWidth_(inferDataWidth(module, model)),
        outputWidths_(inferOutputWidths(module, model)),
        inputWidths_(inferInputWidths(module, model)),
        constantValueWidth_(inferConstantValueWidth(module, model)),
        joinMask_(module.inputPorts.empty()
                      ? 0
                      : ((module.inputPorts.size() >= 64)
                             ? std::numeric_limits<uint64_t>::max()
                             : ((uint64_t{1} << module.inputPorts.size()) - 1))),
        fabricMuxFields_(countConfigMuxFields(module)) {
    hwNodeId = module.hwNodeId;
    name = module.name;
    kind = module.kind;
    maxOutputs_ = module.outputPorts.size();
    hasStatefulBody_ = bodyType_ == BodyType::Stream ||
                       bodyType_ == BodyType::Carry ||
                       bodyType_ == BodyType::Invariant ||
                       bodyType_ == BodyType::Gate ||
                       bodyType_ == BodyType::Load ||
                       bodyType_ == BodyType::Store ||
                       bodyType_ == BodyType::HandshakeMux;
    outputRegisters_.assign(maxOutputs_, std::nullopt);
    outputQueues_.assign(maxOutputs_, {});
    directOutputs_.assign(maxOutputs_, std::nullopt);
    gateOutputsAccepted_.assign(maxOutputs_, false);
    streamOutputsAccepted_.assign(maxOutputs_, false);
    consumedInputGeneration_.assign(module.inputPorts.size(), 0);
    operandTokens_.assign(module.inputPorts.size(), std::nullopt);
  }

  bool isCombinational() const override {
    return false;
  }

  void reset() override {
    perf_ = PerfSnapshot();
    countdown_ = 0;
    cyclesSinceLastFire_ = interval_;
    inflight_.clear();
    for (auto &reg : outputRegisters_)
      reg.reset();
    for (auto &queue : outputQueues_)
      queue.clear();
    for (auto &token : directOutputs_)
      token.reset();
    dataflowInitialStage_ = true;
    invariantStoredValue_ = 0;
    initLatched_ = false;
    initLatchedValue_ = 0;
    carryDLatched_ = false;
    carryDValue_ = false;
    streamInitialPhase_ = true;
    streamTerminalPending_ = false;
    streamNextIdx_ = 0;
    streamBoundReg_ = 0;
    streamStepReg_ = 0;
    gateState_ = 0;
    gateLatchedValue_ = 0;
    gateLatchedCond_ = false;
    gateValueToken_.reset();
    gateCondToken_.reset();
    muxSelectorToken_.reset();
    muxSelectedDataToken_.reset();
    muxSelectedInputIdx_.reset();
    loadAddrToken_.reset();
    loadDataToken_.reset();
    loadCtrlToken_.reset();
    storeAddrToken_.reset();
    storeDataToken_.reset();
    storeCtrlToken_.reset();
    streamStartToken_.reset();
    streamStepToken_.reset();
    streamBoundToken_.reset();
    std::fill(gateOutputsAccepted_.begin(), gateOutputsAccepted_.end(), false);
    std::fill(streamOutputsAccepted_.begin(), streamOutputsAccepted_.end(), false);
    std::fill(consumedInputGeneration_.begin(), consumedInputGeneration_.end(), 0);
    for (auto &token : operandTokens_)
      token.reset();
    nextOutputGeneration_ = 1;
    pendingDirectGeneration_ = 0;
    emittedFireThisCycle_ = false;
    logicalFireCount_ = 0;
    inputCaptureCount_ = 0;
    outputTransferCount_ = 0;
    loadIssueCount_ = 0;
    loadReturnCount_ = 0;
    storeIssueCount_ = 0;
    condTrueCount_ = 0;
    condFalseCount_ = 0;
    streamEmitCount_ = 0;
    streamTerminalCount_ = 0;
    gateHeadCount_ = 0;
    gateTrueCount_ = 0;
    gateFalseCount_ = 0;
    carryInitCount_ = 0;
    carryLoopCount_ = 0;
    carryResetCount_ = 0;
    invariantInitCount_ = 0;
    invariantLoopCount_ = 0;
    invariantResetCount_ = 0;
  }

  void configure(const std::vector<uint32_t> &configWords) override {
    muxSel_ = 0;
    cmpPredicate_ = 0;
    streamContCond_ = 1;
    joinMask_ = inputs.size() >= 64 ? std::numeric_limits<uint64_t>::max()
                                    : ((uint64_t{1} << inputs.size()) - 1);
    configuredConstantValue_ = 0;
    fabricMuxFields_.assign(fabricMuxFields_.size(), {});
    unsigned bitPos = 0;
    auto readBits = [&](unsigned width) -> uint64_t {
      uint64_t value = 0;
      for (unsigned bit = 0; bit < width; ++bit) {
        unsigned wordIdx = bitPos / 32;
        unsigned wordBit = bitPos % 32;
        if (wordIdx < configWords.size() &&
            ((configWords[wordIdx] >> wordBit) & 1u) != 0)
          value |= (uint64_t{1} << bit);
        ++bitPos;
      }
      return value;
    };

    if (opName_ == "fabric.mux") {
      for (auto &field : fabricMuxFields_) {
        unsigned branchCount = static_cast<unsigned>(
            std::max<size_t>(inputs.size(), outputs.size()));
        unsigned selBits = log2Ceil32(branchCount);
        field.sel = readBits(selBits);
        field.discard = readBits(1) != 0;
        field.disconnect = readBits(1) != 0;
      }
      return;
    }
    if (opName_ == "handshake.constant") {
      configuredConstantValue_ = readBits(constantValueWidth_);
      return;
    }
    if (opName_ == "arith.cmpi" || opName_ == "arith.cmpf") {
      cmpPredicate_ = static_cast<uint8_t>(readBits(4));
      return;
    }
    if (opName_ == "dataflow.stream") {
      streamContCond_ = static_cast<uint8_t>(readBits(5));
      return;
    }
    if (opName_ == "handshake.join") {
      joinMask_ = readBits(static_cast<unsigned>(inputs.size()));
      return;
    }
  }

  void evaluate() override {
    emittedFireThisCycle_ = false;
    clearDirectOutputs();
    driveBufferedOutputs();

    if (hasStatefulBody_) {
      evaluateStatefulBody();
      driveBufferedOutputs();
      return;
    }

    bool outputBufferBusy = anyOutputRegisterBusy();
    bool intervalReady = cyclesSinceLastFire_ >= interval_;
    bool readyForNewFire = !outputBufferBusy && intervalReady && inflight_.empty();

    if (latency_ == 0) {
      evaluateCombinationalBody(readyForNewFire);
      return;
    }

    if (!readyForNewFire) {
      setAllInputReady(false);
      driveBufferedOutputs();
      return;
    }

    switch (bodyType_) {
    case BodyType::Constant: {
      bool trigger = inputs.empty() || inputs.front()->valid;
      if (!trigger) {
        setAllInputReady(false);
        return;
      }
      if (!inputs.empty())
        inputs.front()->ready = true;
      break;
    }
    case BodyType::Select:
    case BodyType::Compute:
    case BodyType::CondBranch:
    case BodyType::Join:
      evaluateFireReadinessForStrictInputs();
      break;
    case BodyType::Load:
      evaluateLoadReadiness();
      break;
    case BodyType::Store:
      evaluateStoreReadiness();
      break;
    case BodyType::HandshakeMux:
      evaluateMuxReadiness();
      break;
    case BodyType::Unsupported:
      setAllInputReady(false);
      break;
    default:
      setAllInputReady(false);
      break;
    }
    driveBufferedOutputs();
  }

  void commit() override {
    if (cyclesSinceLastFire_ < std::numeric_limits<uint64_t>::max())
      ++cyclesSinceLastFire_;

    if (hasStatefulBody_) {
      commitStatefulBody();
      return;
    }

    for (auto &flight : inflight_) {
      if (flight.remainingCycles > 0)
        --flight.remainingCycles;
    }
    while (!inflight_.empty() && inflight_.front().remainingCycles == 0 &&
           canDrainIntoOutputRegisters(inflight_.front().outputs)) {
      drainToOutputRegisters(inflight_.front().outputs);
      inflight_.pop_front();
    }
    commitOutputTransfers();

    if (bodyType_ == BodyType::HandshakeMux)
      captureMuxOperands();

    if (latency_ == 0) {
      commitZeroLatencyBody();
      if (!directFireArmed_)
        captureGenericOperands();
      return;
    }

    if (!canFireCurrentBody()) {
      captureGenericOperands();
      return;
    }

    auto outputs = computeOutputs(allocateOutputGeneration());
    if (!outputs.has_value())
      return;

    inflight_.push_back({static_cast<unsigned>(latency_), *outputs});
    consumeInputsForCurrentBody();
    ++logicalFireCount_;
    if (bodyType_ == BodyType::CondBranch) {
      if (!outputs->empty() && (*outputs)[0].has_value())
        ++condTrueCount_;
      else
        ++condFalseCount_;
    }
    if (bodyType_ == BodyType::HandshakeMux) {
      muxSelectorToken_.reset();
      muxSelectedInputIdx_.reset();
      muxSelectedDataToken_.reset();
    }
    cyclesSinceLastFire_ = 0;
    emittedFireThisCycle_ = true;
    perf_.activeCycles++;
    perf_.tokensIn += countConsumedInputsForCurrentBody();
  }

  void collectTraceEvents(std::vector<TraceEvent> &events,
                          uint64_t cycle) override {
    bool emitted = emittedFireThisCycle_;
    if (!emitted) {
      for (const auto &out : outputs) {
        if (out && out->transferred()) {
          emitted = true;
          break;
        }
      }
    }
    if (!emitted)
      return;
    TraceEvent ev;
    ev.cycle = cycle;
    ev.phase = SimPhase::Commit;
    ev.hwNodeId = hwNodeId;
    ev.eventKind = EventKind::NodeFire;
    events.push_back(ev);
  }

  PerfSnapshot getPerfSnapshot() const override { return perf_; }

  void debugDump(std::ostream &os) const override {
    os << "      fu body=" << opName_ << " type="
       << static_cast<unsigned>(bodyType_) << " latency=" << latency_
       << " interval=" << interval_
       << " emitted=" << emittedFireThisCycle_
       << " inflight=" << inflight_.size()
       << " directArmed=" << directFireArmed_
       << " initStage=" << dataflowInitialStage_
       << " carryDLatched=" << carryDLatched_
       << " carryDValue=" << carryDValue_
       << " streamInitial=" << streamInitialPhase_
       << " streamTerminalPending=" << streamTerminalPending_
       << " streamNextIdx=" << streamNextIdx_
       << " streamBound=" << streamBoundReg_
       << " streamStep=" << streamStepReg_
       << " gateState=" << gateState_
       << " gateLatchedCond=" << gateLatchedCond_
       << " gateLatchedValue=" << gateLatchedValue_
       << " gateValueLatched=" << gateValueToken_.has_value()
       << " gateCondLatched=" << gateCondToken_.has_value()
       << " muxSelLatched=" << muxSelectorToken_.has_value()
       << " muxDataLatched=" << muxSelectedDataToken_.has_value()
       << " loadAddrLatched=" << loadAddrToken_.has_value()
       << " loadDataLatched=" << loadDataToken_.has_value()
       << " loadCtrlLatched=" << loadCtrlToken_.has_value()
       << " storeAddrLatched=" << storeAddrToken_.has_value()
       << " storeDataLatched=" << storeDataToken_.has_value()
       << " storeCtrlLatched=" << storeCtrlToken_.has_value()
       << " streamStartLatched=" << streamStartToken_.has_value()
       << " streamStepLatched=" << streamStepToken_.has_value()
       << " streamBoundLatched=" << streamBoundToken_.has_value()
       << " genericOperandLatched=" << countLatchedOperands();
    if (muxSelectedInputIdx_.has_value())
      os << " muxSelIdx=" << *muxSelectedInputIdx_;
    os << "\n";
    os << "      consumed_generations=[";
    for (size_t idx = 0; idx < consumedInputGeneration_.size(); ++idx) {
      if (idx != 0)
        os << ", ";
      os << consumedInputGeneration_[idx];
    }
    os << "]\n";
    if (usesElasticOutputQueues()) {
      os << "      output_queues=[";
      for (size_t idx = 0; idx < outputQueues_.size(); ++idx) {
        if (idx != 0)
          os << ", ";
        if (outputQueues_[idx].empty()) {
          os << "-";
          continue;
        }
        os << "g" << outputQueues_[idx].front().generation << ":"
           << outputQueues_[idx].size();
      }
      os << "]\n";
    } else {
      os << "      output_registers=[";
      for (size_t idx = 0; idx < outputRegisters_.size(); ++idx) {
        if (idx != 0)
          os << ", ";
        if (!outputRegisters_[idx].has_value()) {
          os << "-";
          continue;
        }
        os << "g" << outputRegisters_[idx]->generation;
      }
      os << "]\n";
    }
  }

  bool hasPendingWork() const override {
    if (!inflight_.empty())
      return true;
    if (usesElasticOutputQueues()) {
      if (std::any_of(outputQueues_.begin(), outputQueues_.end(),
                      [](const auto &queue) { return !queue.empty(); }))
        return true;
    } else if (std::any_of(outputRegisters_.begin(), outputRegisters_.end(),
                           [](const auto &value) { return value.has_value(); })) {
      return true;
    }
    if (!hasStatefulBody_)
      return std::any_of(operandTokens_.begin(), operandTokens_.end(),
                         [](const auto &token) { return token.has_value(); });

    switch (bodyType_) {
    case BodyType::Carry:
      return carryDLatched_;
    case BodyType::Invariant:
      return false;
    case BodyType::Gate:
      return gateState_ == 1 || gateState_ == 3 || gateState_ == 4 ||
             gateValueToken_.has_value() || gateCondToken_.has_value();
    case BodyType::Stream:
      return streamStartToken_.has_value() || streamStepToken_.has_value() ||
             streamBoundToken_.has_value();
    case BodyType::Load:
      return loadAddrToken_.has_value() || loadDataToken_.has_value() ||
             loadCtrlToken_.has_value();
    case BodyType::Store:
      return storeAddrToken_.has_value() || storeDataToken_.has_value() ||
             storeCtrlToken_.has_value();
    case BodyType::HandshakeMux:
      return muxSelectorToken_.has_value() || muxSelectedDataToken_.has_value();
    default:
      return false;
    }
  }

  uint64_t getLogicalFireCount() const override { return logicalFireCount_; }

  uint64_t getInputCaptureCount() const override { return inputCaptureCount_; }

  uint64_t getOutputTransferCount() const override {
    return outputTransferCount_;
  }

  std::vector<NamedCounter> getDebugCounters() const override {
    std::vector<NamedCounter> counters;
    auto pushIfNonZero = [&](const char *name, uint64_t value) {
      if (value != 0)
        counters.push_back({name, value});
    };
    pushIfNonZero("load_issue_count", loadIssueCount_);
    pushIfNonZero("load_return_count", loadReturnCount_);
    pushIfNonZero("store_issue_count", storeIssueCount_);
    pushIfNonZero("cond_true_count", condTrueCount_);
    pushIfNonZero("cond_false_count", condFalseCount_);
    pushIfNonZero("stream_emit_count", streamEmitCount_);
    pushIfNonZero("stream_terminal_count", streamTerminalCount_);
    pushIfNonZero("gate_head_count", gateHeadCount_);
    pushIfNonZero("gate_true_count", gateTrueCount_);
    pushIfNonZero("gate_false_count", gateFalseCount_);
    pushIfNonZero("carry_init_count", carryInitCount_);
    pushIfNonZero("carry_loop_count", carryLoopCount_);
    pushIfNonZero("carry_reset_count", carryResetCount_);
    pushIfNonZero("invariant_init_count", invariantInitCount_);
    pushIfNonZero("invariant_loop_count", invariantLoopCount_);
    pushIfNonZero("invariant_reset_count", invariantResetCount_);
    return counters;
  }

  std::string getDebugStateSummary() const override {
    std::ostringstream os;
    os << "body=" << opName_ << " inflight=" << inflight_.size()
       << " out_busy=" << (anyOutputRegisterBusy() ? 1 : 0);
    switch (bodyType_) {
    case BodyType::Load:
      os << " load(addr=" << (loadAddrToken_.has_value() ? 1 : 0)
         << ",data=" << (loadDataToken_.has_value() ? 1 : 0)
         << ",ctrl=" << (loadCtrlToken_.has_value() ? 1 : 0) << ")";
      break;
    case BodyType::Store:
      os << " store(addr=" << (storeAddrToken_.has_value() ? 1 : 0)
         << ",data=" << (storeDataToken_.has_value() ? 1 : 0)
         << ",ctrl=" << (storeCtrlToken_.has_value() ? 1 : 0) << ")";
      break;
    case BodyType::CondBranch:
      os << " cond_latched=" << countLatchedOperands();
      break;
    case BodyType::Gate:
      os << " gate(state=" << gateState_
         << ",value=" << (gateValueToken_.has_value() ? 1 : 0)
         << ",cond=" << (gateCondToken_.has_value() ? 1 : 0) << ")";
      break;
    case BodyType::Stream:
      os << " stream(init=" << (streamInitialPhase_ ? 1 : 0)
         << ",start=" << (streamStartToken_.has_value() ? 1 : 0)
         << ",step=" << (streamStepToken_.has_value() ? 1 : 0)
         << ",bound=" << (streamBoundToken_.has_value() ? 1 : 0)
         << ",terminal=" << (streamTerminalPending_ ? 1 : 0) << ")";
      break;
    case BodyType::Carry:
      os << " carry(initStage=" << (dataflowInitialStage_ ? 1 : 0)
         << ",initLatched=" << (initLatched_ ? 1 : 0)
         << ",dLatched=" << (carryDLatched_ ? 1 : 0)
         << ",dValue=" << (carryDValue_ ? 1 : 0) << ")";
      break;
    case BodyType::Invariant:
      os << " invariant(initStage=" << (dataflowInitialStage_ ? 1 : 0)
         << ",initLatched=" << (initLatched_ ? 1 : 0) << ")";
      break;
    default:
      break;
    }
    return os.str();
  }

private:
  bool usesElasticOutputQueues() const {
    return false;
  }

  void enqueueOutputToken(size_t outputIdx, const SimToken &token) {
    if (usesElasticOutputQueues()) {
      if (outputIdx < outputQueues_.size())
        outputQueues_[outputIdx].push_back(token);
      return;
    }
    if (outputIdx < outputRegisters_.size())
      outputRegisters_[outputIdx] = token;
  }

  void clearOutputBuffer(size_t outputIdx) {
    if (usesElasticOutputQueues()) {
      if (outputIdx < outputQueues_.size() && !outputQueues_[outputIdx].empty())
        outputQueues_[outputIdx].pop_front();
      return;
    }
    if (outputIdx < outputRegisters_.size())
      outputRegisters_[outputIdx].reset();
  }

  struct InflightResult {
    unsigned remainingCycles = 0;
    std::vector<std::optional<SimToken>> outputs;
  };

  static std::string resolvePrimaryOp(const StaticModuleDesc &module) {
    return modulePrimaryOp(module);
  }

  static unsigned inferDataWidth(const StaticModuleDesc &module,
                                 const StaticMappedModel &model) {
    for (IdIndex portId : module.outputPorts) {
      if (const StaticPortDesc *port = findPort(model, portId)) {
        if (!port->isNone && !port->isMemRef)
          return std::max(1u, port->valueWidth);
      }
    }
    for (IdIndex portId : module.inputPorts) {
      if (const StaticPortDesc *port = findPort(model, portId)) {
        if (!port->isNone && !port->isMemRef)
          return std::max(1u, port->valueWidth);
      }
    }
    return 32;
  }

  static std::vector<unsigned> inferOutputWidths(const StaticModuleDesc &module,
                                                 const StaticMappedModel &model) {
    std::vector<unsigned> widths;
    widths.reserve(module.outputPorts.size());
    for (IdIndex portId : module.outputPorts) {
      if (const StaticPortDesc *port = findPort(model, portId))
        widths.push_back(std::max(1u, port->valueWidth));
      else
        widths.push_back(32);
    }
    return widths;
  }

  static std::vector<unsigned> inferInputWidths(const StaticModuleDesc &module,
                                                const StaticMappedModel &model) {
    std::vector<unsigned> widths;
    widths.reserve(module.inputPorts.size());
    for (IdIndex portId : module.inputPorts) {
      if (const StaticPortDesc *port = findPort(model, portId))
        widths.push_back(std::max(1u, port->valueWidth));
      else
        widths.push_back(32);
    }
    return widths;
  }

  static unsigned countConfigMuxFields(const StaticModuleDesc &module) {
    return resolvePrimaryOp(module) == "fabric.mux" ? 1u : 0u;
  }

  static unsigned inferConstantValueWidth(const StaticModuleDesc &module,
                                          const StaticMappedModel &model) {
    if (resolvePrimaryOp(module) != "handshake.constant")
      return 0;
    for (IdIndex portId : module.outputPorts) {
      if (const StaticPortDesc *port = findPort(model, portId))
        return std::max(1u, port->valueWidth);
    }
    return 32;
  }

  uint64_t maskToWidth(uint64_t value, unsigned width) const {
    if (width >= 64)
      return value;
    return value & ((uint64_t{1} << width) - 1);
  }

  int64_t signExtend(uint64_t value, unsigned width) const {
    if (width >= 64)
      return static_cast<int64_t>(value);
    uint64_t mask = uint64_t{1} << (width - 1);
    if (value & mask)
      return static_cast<int64_t>(value | (~uint64_t{0} << width));
    return static_cast<int64_t>(value);
  }

  float toFloat(uint64_t value) const {
    float out = 0.0f;
    uint32_t bits = static_cast<uint32_t>(value);
    std::memcpy(&out, &bits, sizeof(out));
    return out;
  }

  uint64_t fromFloat(float value) const {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
  }

  double toDouble(uint64_t value) const {
    double out = 0.0;
    std::memcpy(&out, &value, sizeof(out));
    return out;
  }

  uint64_t fromDouble(double value) const {
    uint64_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
  }

  bool cmpi(uint64_t a, uint64_t b, unsigned width) const {
    switch (cmpPredicate_) {
    case 0: return a == b;
    case 1: return a != b;
    case 2: return signExtend(a, width) < signExtend(b, width);
    case 3: return signExtend(a, width) <= signExtend(b, width);
    case 4: return signExtend(a, width) > signExtend(b, width);
    case 5: return signExtend(a, width) >= signExtend(b, width);
    case 6: return a < b;
    case 7: return a <= b;
    case 8: return a > b;
    case 9: return a >= b;
    default: return false;
    }
  }

  bool cmpf(uint64_t a, uint64_t b, unsigned width) const {
    double lhs = width <= 32 ? static_cast<double>(toFloat(a)) : toDouble(a);
    double rhs = width <= 32 ? static_cast<double>(toFloat(b)) : toDouble(b);
    bool lhsNaN = std::isnan(lhs);
    bool rhsNaN = std::isnan(rhs);
    switch (cmpPredicate_) {
    case 0: return false;
    case 1: return lhs == rhs;
    case 2: return lhs > rhs;
    case 3: return lhs >= rhs;
    case 4: return lhs < rhs;
    case 5: return lhs <= rhs;
    case 6: return lhs != rhs;
    case 7: return !lhsNaN && !rhsNaN;
    case 8: return lhsNaN || rhsNaN;
    case 9: return lhsNaN || rhsNaN || lhs == rhs;
    case 10: return lhsNaN || rhsNaN || lhs > rhs;
    case 11: return lhsNaN || rhsNaN || lhs >= rhs;
    case 12: return lhsNaN || rhsNaN || lhs < rhs;
    case 13: return lhsNaN || rhsNaN || lhs <= rhs;
    case 14: return lhsNaN || rhsNaN || lhs != rhs;
    case 15: return true;
    default: return false;
    }
  }

  void setAllInputReady(bool ready) {
    for (size_t idx = 0; idx < inputs.size(); ++idx) {
      if (ready) {
        inputs[idx]->ready = true;
        continue;
      }
      inputs[idx]->ready = inputAlreadyConsumed(idx);
    }
  }

  bool inputFresh(size_t idx) const {
    return idx < inputs.size() && inputs[idx]->valid &&
           inputs[idx]->generation != 0 &&
           consumedInputGeneration_[idx] != inputs[idx]->generation;
  }

  bool inputAlreadyConsumed(size_t idx) const {
    return idx < inputs.size() && inputs[idx]->valid &&
           inputs[idx]->generation != 0 &&
           consumedInputGeneration_[idx] == inputs[idx]->generation;
  }

  void setInputReadyFreshAware(size_t idx, bool allowFresh) {
    if (idx >= inputs.size())
      return;
    if (inputAlreadyConsumed(idx)) {
      inputs[idx]->ready = true;
      return;
    }
    inputs[idx]->ready = allowFresh;
  }

  void markInputConsumed(size_t idx) {
    if (idx >= inputs.size())
      return;
    consumedInputGeneration_[idx] = inputs[idx]->generation;
  }

  uint64_t allocateOutputGeneration() {
    return composeTokenGeneration(hwNodeId, nextOutputGeneration_++);
  }

  uint64_t reserveDirectGeneration() {
    if (pendingDirectGeneration_ == 0)
      pendingDirectGeneration_ = nextOutputGeneration_;
    return composeTokenGeneration(hwNodeId, pendingDirectGeneration_);
  }

  void finalizeDirectGeneration() {
    if (pendingDirectGeneration_ == 0)
      return;
    nextOutputGeneration_ =
        std::max<uint64_t>(nextOutputGeneration_, pendingDirectGeneration_ + 1);
    pendingDirectGeneration_ = 0;
  }

  SimToken makeGeneratedToken(size_t outputIdx, uint64_t data,
                              uint16_t tag = 0, bool hasTag = false,
                              uint64_t generation = 0) const {
    SimToken token;
    unsigned width = outputIdx < outputWidths_.size() ? outputWidths_[outputIdx]
                                                      : dataWidth_;
    token.data = maskToWidth(data, width);
    token.tag = tag;
    token.hasTag = hasTag;
    token.generation = generation;
    return token;
  }

  bool anyOutputRegisterBusy() const {
    if (usesElasticOutputQueues()) {
      return std::any_of(outputQueues_.begin(), outputQueues_.end(),
                         [](const auto &queue) { return !queue.empty(); });
    }
    return std::any_of(outputRegisters_.begin(), outputRegisters_.end(),
                       [](const auto &value) { return value.has_value(); });
  }

  bool canDrainIntoOutputRegisters(
      const std::vector<std::optional<SimToken>> &outputs) const {
    for (size_t idx = 0; idx < outputs.size() && idx < outputRegisters_.size(); ++idx) {
      if (outputs[idx].has_value() && outputRegisters_[idx].has_value())
        return false;
    }
    return true;
  }

  bool outputRegisterFree(size_t outputIdx) const {
    if (usesElasticOutputQueues())
      return outputIdx < outputQueues_.size();
    return outputIdx < outputRegisters_.size() &&
           !outputRegisters_[outputIdx].has_value();
  }

  bool outputRegistersFree(std::initializer_list<size_t> outputIdxs) const {
    for (size_t outputIdx : outputIdxs) {
      if (!outputRegisterFree(outputIdx))
        return false;
    }
    return true;
  }

  void drainToOutputRegisters(
      const std::vector<std::optional<SimToken>> &outputs) {
    for (size_t idx = 0; idx < outputs.size() && idx < outputRegisters_.size(); ++idx) {
      if (outputs[idx].has_value())
        outputRegisters_[idx] = outputs[idx];
    }
  }

  void driveBufferedOutputs() {
    for (size_t idx = 0; idx < outputs.size(); ++idx) {
      if (usesElasticOutputQueues()) {
        if (idx < outputQueues_.size() && !outputQueues_[idx].empty()) {
          driveChannelFromToken(*outputs[idx], outputQueues_[idx].front());
        } else {
          outputs[idx]->valid = false;
        }
      } else if (idx < outputRegisters_.size() &&
                 outputRegisters_[idx].has_value()) {
        driveChannelFromToken(*outputs[idx], *outputRegisters_[idx]);
      } else {
        outputs[idx]->valid = false;
      }
    }
  }

  void clearDirectOutputs() {
    directFireArmed_ = false;
    for (auto &token : directOutputs_)
      token.reset();
  }

  void driveDirectOutputs() {
    for (size_t idx = 0; idx < outputs.size() && idx < directOutputs_.size();
         ++idx) {
      if (!directOutputs_[idx].has_value())
        continue;
      driveChannelFromToken(*outputs[idx], *directOutputs_[idx]);
    }
  }

  void commitOutputTransfers() {
    if (bodyType_ == BodyType::Gate || bodyType_ == BodyType::Stream)
      return;
    bool anyTransfer = false;
    if (usesElasticOutputQueues()) {
      for (size_t idx = 0; idx < outputs.size() && idx < outputQueues_.size();
           ++idx) {
        if (outputQueues_[idx].empty() || !outputs[idx]->transferred())
          continue;
        outputQueues_[idx].pop_front();
        perf_.tokensOut++;
        ++outputTransferCount_;
        anyTransfer = true;
      }
      if (anyTransfer)
        perf_.activeCycles++;
      return;
    }
    for (size_t idx = 0; idx < outputs.size() && idx < outputRegisters_.size(); ++idx) {
      if (!outputRegisters_[idx].has_value() || !outputs[idx]->transferred())
        continue;
      outputRegisters_[idx].reset();
      perf_.tokensOut++;
      ++outputTransferCount_;
      anyTransfer = true;
    }
    if (anyTransfer)
      perf_.activeCycles++;
  }

  std::optional<std::vector<std::optional<SimToken>>>
  computeOutputs(uint64_t generation) const {
    std::vector<std::optional<SimToken>> produced(maxOutputs_);
    if (bodyType_ == BodyType::Unsupported)
      return std::nullopt;

    auto makeToken = [&](size_t outputIdx, uint64_t data, uint16_t tag = 0,
                         bool hasTag = false) {
      produced[outputIdx] =
          makeGeneratedToken(outputIdx, data, tag, hasTag, generation);
    };

    auto getInputToken = [&](size_t idx) -> std::optional<SimToken> {
      if (idx >= inputs.size())
        return std::nullopt;
      if (idx < operandTokens_.size() && operandTokens_[idx].has_value())
        return operandTokens_[idx];
      if (inputFresh(idx) && inputs[idx]->ready)
        return tokenFromChannel(*inputs[idx]);
      return std::nullopt;
    };

    auto getInput = [&](size_t idx) -> uint64_t {
      auto token = getInputToken(idx);
      return token.has_value() ? token->data : 0;
    };

    auto getInputTag = [&](size_t idx) -> uint16_t {
      auto token = getInputToken(idx);
      return token.has_value() ? token->tag : 0;
    };

    auto getInputHasTag = [&](size_t idx) -> bool {
      auto token = getInputToken(idx);
      return token.has_value() ? token->hasTag : false;
    };

    if (bodyType_ == BodyType::Constant) {
      makeToken(0, configuredConstantValue_);
      return produced;
    }
    if (bodyType_ == BodyType::Select) {
      if (opName_ == "arith.select")
        makeToken(0, (getInput(0) & 1u) ? getInput(1) : getInput(2));
      else if (!fabricMuxFields_.empty() && !fabricMuxFields_.front().disconnect &&
               !fabricMuxFields_.front().discard) {
        makeToken(0, getInput(fabricMuxFields_.front().sel));
      }
      return produced;
    }
    if (bodyType_ == BodyType::HandshakeMux) {
      if (!muxSelectorToken_.has_value() || !muxSelectedDataToken_.has_value())
        return std::nullopt;
      produced[0] = makeGeneratedToken(0, muxSelectedDataToken_->data,
                                       muxSelectedDataToken_->tag,
                                       muxSelectedDataToken_->hasTag,
                                       generation);
      return produced;
    }
    if (bodyType_ == BodyType::CondBranch) {
      size_t selected = (getInput(0) & 1u) ? 0u : 1u;
      if (selected < produced.size() && inputs.size() >= 2) {
        produced[selected] = makeGeneratedToken(
            selected, getInput(1), getInputTag(1), getInputHasTag(1),
            generation);
      }
      return produced;
    }
    if (bodyType_ == BodyType::Load) {
      if (produced.size() >= 2 && inputs.size() >= 2) {
        produced[0] = makeGeneratedToken(0, inputs[1]->data, inputs[1]->tag,
                                         inputs[1]->hasTag, generation);
        produced[1] = makeGeneratedToken(1, inputs[0]->data, inputs[0]->tag,
                                         inputs[0]->hasTag, generation);
      }
      return produced;
    }
    if (bodyType_ == BodyType::Store) {
      if (produced.size() >= 2 && inputs.size() >= 2) {
        produced[0] = makeGeneratedToken(0, inputs[1]->data, inputs[1]->tag,
                                         inputs[1]->hasTag, generation);
        produced[1] = makeGeneratedToken(1, inputs[0]->data, inputs[0]->tag,
                                         inputs[0]->hasTag, generation);
      }
      return produced;
    }
    if (bodyType_ == BodyType::Join) {
      makeToken(0, 0);
      return produced;
    }
    if (bodyType_ == BodyType::Compute) {
      uint64_t a = getInput(0);
      uint64_t b = getInput(1);
      unsigned width = dataWidth_;
      uint64_t result = 0;
      if (opName_ == "arith.addi") result = a + b;
      else if (opName_ == "arith.subi") result = a - b;
      else if (opName_ == "arith.muli") result = a * b;
      else if (opName_ == "arith.divsi") result = b ? static_cast<uint64_t>(signExtend(a, width) / signExtend(b, width)) : 0;
      else if (opName_ == "arith.divui") result = b ? (a / b) : 0;
      else if (opName_ == "arith.remsi") result = b ? static_cast<uint64_t>(signExtend(a, width) % signExtend(b, width)) : 0;
      else if (opName_ == "arith.remui") result = b ? (a % b) : 0;
      else if (opName_ == "arith.andi") result = a & b;
      else if (opName_ == "arith.ori") result = a | b;
      else if (opName_ == "arith.xori") result = a ^ b;
      else if (opName_ == "arith.shli") result = a << (b & 63);
      else if (opName_ == "arith.shrsi") result = static_cast<uint64_t>(signExtend(a, width) >> (b & 63));
      else if (opName_ == "arith.shrui") result = a >> (b & 63);
      else if (opName_ == "arith.extsi") result = static_cast<uint64_t>(signExtend(a, inputWidths_.empty() ? width : inputWidths_[0]));
      else if (opName_ == "arith.extui" || opName_ == "arith.trunci" ||
               opName_ == "arith.index_cast" || opName_ == "arith.index_castui") result = a;
      else if (opName_ == "arith.negf") result = width <= 32 ? fromFloat(-toFloat(a)) : fromDouble(-toDouble(a));
      else if (opName_ == "arith.addf") result = width <= 32 ? fromFloat(toFloat(a) + toFloat(b)) : fromDouble(toDouble(a) + toDouble(b));
      else if (opName_ == "arith.subf") result = width <= 32 ? fromFloat(toFloat(a) - toFloat(b)) : fromDouble(toDouble(a) - toDouble(b));
      else if (opName_ == "arith.mulf") result = width <= 32 ? fromFloat(toFloat(a) * toFloat(b)) : fromDouble(toDouble(a) * toDouble(b));
      else if (opName_ == "arith.divf") result = width <= 32 ? fromFloat(toFloat(a) / toFloat(b)) : fromDouble(toDouble(a) / toDouble(b));
      else if (opName_ == "arith.fptosi") result = width <= 32 ? static_cast<uint64_t>(static_cast<int64_t>(toFloat(a))) : static_cast<uint64_t>(static_cast<int64_t>(toDouble(a)));
      else if (opName_ == "arith.fptoui") result = width <= 32 ? static_cast<uint64_t>(toFloat(a)) : static_cast<uint64_t>(toDouble(a));
      else if (opName_ == "arith.sitofp") result = width <= 32 ? fromFloat(static_cast<float>(signExtend(a, inputWidths_.empty() ? width : inputWidths_[0]))) : fromDouble(static_cast<double>(signExtend(a, inputWidths_.empty() ? width : inputWidths_[0])));
      else if (opName_ == "arith.uitofp") result = width <= 32 ? fromFloat(static_cast<float>(a)) : fromDouble(static_cast<double>(a));
      else if (opName_ == "arith.cmpi") result = cmpi(a, b, inputWidths_.empty() ? width : inputWidths_[0]) ? 1 : 0;
      else if (opName_ == "arith.cmpf") result = cmpf(a, b, width) ? 1 : 0;
      else if (opName_ == "math.absf") result = width <= 32 ? fromFloat(std::fabs(toFloat(a))) : fromDouble(std::fabs(toDouble(a)));
      else if (opName_ == "math.cos") result = width <= 32 ? fromFloat(std::cos(toFloat(a))) : fromDouble(std::cos(toDouble(a)));
      else if (opName_ == "math.exp") result = width <= 32 ? fromFloat(std::exp(toFloat(a))) : fromDouble(std::exp(toDouble(a)));
      else if (opName_ == "math.log2") result = width <= 32 ? fromFloat(std::log2(toFloat(a))) : fromDouble(std::log2(toDouble(a)));
      else if (opName_ == "math.sin") result = width <= 32 ? fromFloat(std::sin(toFloat(a))) : fromDouble(std::sin(toDouble(a)));
      else if (opName_ == "math.sqrt") result = width <= 32 ? fromFloat(std::sqrt(toFloat(a))) : fromDouble(std::sqrt(toDouble(a)));
      else if (opName_ == "math.fma") {
        uint64_t c = getInput(2);
        result = width <= 32 ? fromFloat(std::fma(toFloat(a), toFloat(b), toFloat(c)))
                             : fromDouble(std::fma(toDouble(a), toDouble(b), toDouble(c)));
      } else if (opName_ == "llvm.intr.bitreverse") {
        result = 0;
        for (unsigned bit = 0; bit < width; ++bit) {
          if (a & (uint64_t{1} << bit))
            result |= uint64_t{1} << (width - 1 - bit);
        }
      }
      makeToken(0, result);
      return produced;
    }
    return std::nullopt;
  }

  void evaluateFireReadinessForStrictInputs() {
    if (bodyUsesGenericOperandLatches()) {
      setAllInputReady(false);
      bool anyRequired = false;
      for (size_t idx = 0; idx < inputs.size(); ++idx) {
        if (!inputRequiredForCurrentBody(idx))
          continue;
        anyRequired = true;
        bool canCapture = !operandTokens_[idx].has_value();
        setInputReadyFreshAware(idx, canCapture);
      }
      if (!anyRequired)
        return;
      return;
    }

    bool allFresh = true;
    for (size_t idx = 0; idx < inputs.size(); ++idx)
      allFresh = allFresh && inputFresh(idx);
    for (size_t idx = 0; idx < inputs.size(); ++idx)
      setInputReadyFreshAware(idx, allFresh);
  }

  void evaluateMuxReadiness() {
    setAllInputReady(false);
    if (inputs.size() < 2)
      return;
    if (!muxSelectorToken_.has_value()) {
      setInputReadyFreshAware(0, true);
      if (inputs[0]->valid && !inputs[0]->ready)
        perf_.stallCyclesOut++;
      return;
    }
    if (!muxSelectedInputIdx_.has_value() || *muxSelectedInputIdx_ >= inputs.size() ||
        outputs.empty())
      return;
    if (!muxSelectedDataToken_.has_value()) {
      size_t selectedIdx = *muxSelectedInputIdx_;
      setInputReadyFreshAware(selectedIdx, true);
      if (inputs[selectedIdx]->valid && !inputs[selectedIdx]->ready)
        perf_.stallCyclesOut++;
      return;
    }
    if (!outputRegisterFree(0) || !inflight_.empty() ||
        cyclesSinceLastFire_ < interval_)
      perf_.stallCyclesOut++;
  }

  void evaluateLoadReadiness() {
    setAllInputReady(false);
    if (inputs.size() < 3 || outputs.size() < 2)
      return;

    bool canCaptureAddr = !loadAddrToken_.has_value();
    bool canCaptureData = !loadDataToken_.has_value();
    bool canCaptureCtrl = !loadCtrlToken_.has_value();
    setInputReadyFreshAware(0, canCaptureAddr);
    setInputReadyFreshAware(1, canCaptureData);
    setInputReadyFreshAware(2, canCaptureCtrl);

    bool issueBlocked =
        loadAddrToken_.has_value() && loadCtrlToken_.has_value() &&
        outputRegisters_[1].has_value();
    bool returnBlocked =
        loadDataToken_.has_value() && outputRegisters_[0].has_value();
    if ((inputs[0]->valid && !inputAlreadyConsumed(0) && !canCaptureAddr) ||
        (inputs[1]->valid && !inputAlreadyConsumed(1) && !canCaptureData) ||
        (inputs[2]->valid && !inputAlreadyConsumed(2) && !canCaptureCtrl) ||
        issueBlocked || returnBlocked)
      perf_.stallCyclesOut++;
  }

  void evaluateStoreReadiness() {
    setAllInputReady(false);
    if (inputs.size() < 3 || outputs.size() < 2)
      return;
    bool canCaptureAddr = !storeAddrToken_.has_value();
    bool canCaptureData = !storeDataToken_.has_value();
    bool canCaptureCtrl = !storeCtrlToken_.has_value();

    setInputReadyFreshAware(0, canCaptureAddr);
    setInputReadyFreshAware(1, canCaptureData);
    setInputReadyFreshAware(2, canCaptureCtrl);

    bool issueBlocked = storeAddrToken_.has_value() &&
                        storeDataToken_.has_value() &&
                        storeCtrlToken_.has_value() &&
                        (outputRegisters_[0].has_value() ||
                         outputRegisters_[1].has_value());
    if ((inputs[0]->valid && !inputAlreadyConsumed(0) && !canCaptureAddr) ||
        (inputs[1]->valid && !inputAlreadyConsumed(1) && !canCaptureData) ||
        (inputs[2]->valid && !inputAlreadyConsumed(2) && !canCaptureCtrl) ||
        issueBlocked)
      perf_.stallCyclesOut++;
  }

  void evaluateCombinationalBody(bool readyForNewFire) {
    if (!readyForNewFire) {
      setAllInputReady(false);
      return;
    }
    if (bodyType_ == BodyType::Constant) {
      bool trigger = inputs.empty() || inputFresh(0);
      if (!trigger) {
        if (!inputs.empty())
          setInputReadyFreshAware(0, false);
        else
          setAllInputReady(false);
        return;
      }
      if (!inputs.empty())
        setInputReadyFreshAware(0, true);
      auto outputs = computeOutputs(reserveDirectGeneration());
      if (!outputs.has_value())
        return;
      directOutputs_ = *outputs;
      directFireArmed_ = true;
      driveDirectOutputs();
      return;
    }
    if (bodyType_ == BodyType::HandshakeMux) {
      setAllInputReady(false);
      if (inputs.size() < 2 || outputs.empty())
        return;
      size_t idx = static_cast<size_t>(inputs[0]->data + 1);
      bool operandsValid =
          idx < inputs.size() && inputFresh(0) && inputFresh(idx);
      if (!operandsValid)
        setInputReadyFreshAware(0, false);
      if (!operandsValid)
        return;
      auto outputs = computeOutputs(reserveDirectGeneration());
      if (!outputs.has_value() || !(*outputs)[0].has_value())
        return;
      setInputReadyFreshAware(0, true);
      setInputReadyFreshAware(idx, true);
      directOutputs_ = *outputs;
      directFireArmed_ = true;
      driveDirectOutputs();
      return;
    }

    evaluateFireReadinessForStrictInputs();
    if (!canFireCurrentBody())
      return;

    auto outputs = computeOutputs(reserveDirectGeneration());
    if (!outputs.has_value())
      return;
    directOutputs_ = *outputs;
    directFireArmed_ = true;
    driveDirectOutputs();
  }

  void commitZeroLatencyBody() {
    if (!directFireArmed_)
      return;

    bool anyTransfer = false;
    bool anyEmission = false;
    for (size_t idx = 0; idx < outputs.size() && idx < directOutputs_.size();
         ++idx) {
      if (!directOutputs_[idx].has_value())
        continue;
      if (outputs[idx]->transferred()) {
        perf_.tokensOut++;
        ++outputTransferCount_;
        anyTransfer = true;
        anyEmission = true;
      } else if (idx < outputRegisters_.size() &&
                 !outputRegisters_[idx].has_value()) {
        outputRegisters_[idx] = directOutputs_[idx];
        anyEmission = true;
      }
    }
    if (anyTransfer)
      perf_.activeCycles++;

    if (!anyEmission) {
      pendingDirectGeneration_ = 0;
      clearDirectOutputs();
      return;
    }

    consumeInputsForCurrentBody();
    ++logicalFireCount_;
    if (bodyType_ == BodyType::CondBranch) {
      if (!directOutputs_.empty() && directOutputs_[0].has_value())
        ++condTrueCount_;
      else
        ++condFalseCount_;
    }
    if (bodyType_ == BodyType::HandshakeMux) {
      muxSelectorToken_.reset();
      muxSelectedInputIdx_.reset();
      muxSelectedDataToken_.reset();
    }
    finalizeDirectGeneration();
    cyclesSinceLastFire_ = 0;
    emittedFireThisCycle_ = true;
    perf_.activeCycles++;
    perf_.tokensIn += countConsumedInputsForCurrentBody();
    clearDirectOutputs();
  }

  bool canFireCurrentBody() const {
    switch (bodyType_) {
    case BodyType::Constant:
      return inputs.empty() ? true : inputFresh(0) && inputs.front()->ready;
    case BodyType::Load:
      return loadIssueSelected_ || loadReturnSelected_;
    case BodyType::Store:
      return storeIssueSelected_;
    case BodyType::HandshakeMux: {
      return muxSelectorToken_.has_value() && muxSelectedDataToken_.has_value() &&
             outputRegisterFree(0) && inflight_.empty() &&
             cyclesSinceLastFire_ >= interval_;
    }
    case BodyType::Select:
    case BodyType::Compute:
    case BodyType::CondBranch:
      for (size_t idx = 0; idx < inputs.size(); ++idx) {
        if (!inputRequiredForCurrentBody(idx))
          continue;
        if (!inputAvailableForCurrentBody(idx))
          return false;
      }
      return true;
    case BodyType::Join: {
      bool anyActive = false;
      for (size_t idx = 0; idx < inputs.size(); ++idx) {
        if (!inputRequiredForCurrentBody(idx))
          continue;
        anyActive = true;
        if (!inputAvailableForCurrentBody(idx))
          return false;
      }
      return anyActive;
    }
    default:
      return false;
    }
  }

  void consumeInputsForCurrentBody() {
    switch (bodyType_) {
    case BodyType::Constant:
      if (!inputs.empty())
        markInputConsumed(0);
      return;
    case BodyType::Load:
      if (loadIssueSelected_) {
        markInputConsumed(0);
        markInputConsumed(2);
      }
      if (loadReturnSelected_)
        markInputConsumed(1);
      return;
    case BodyType::Store:
      if (storeIssueSelected_) {
        markInputConsumed(0);
        markInputConsumed(1);
        markInputConsumed(2);
      }
      return;
    case BodyType::HandshakeMux:
      return;
    case BodyType::Join:
      [[fallthrough]];
    default:
      for (size_t idx = 0; idx < inputs.size(); ++idx) {
        if (!inputRequiredForCurrentBody(idx))
          continue;
        if (idx < operandTokens_.size() && operandTokens_[idx].has_value()) {
          operandTokens_[idx].reset();
          continue;
        }
        markInputConsumed(idx);
      }
      return;
    }
  }

  unsigned countConsumedInputsForCurrentBody() const {
    switch (bodyType_) {
    case BodyType::Constant:
      return inputs.empty() ? 0u : 1u;
    case BodyType::HandshakeMux:
      return 2u;
    case BodyType::Load:
      return (loadIssueSelected_ ? 2u : 0u) +
             (loadReturnSelected_ ? 1u : 0u);
    case BodyType::Store:
      return storeIssueSelected_ ? 3u : 0u;
    case BodyType::Join:
      [[fallthrough]];
    default:
      return countFreshInputsForCurrentBody();
    }
  }

  void evaluateStatefulBody() {
    setAllInputReady(false);
    driveBufferedOutputs();
    switch (bodyType_) {
    case BodyType::Load:
      evaluateLoadReadiness();
      break;
    case BodyType::Store:
      evaluateStoreReadiness();
      break;
    case BodyType::HandshakeMux:
      evaluateMuxReadiness();
      break;
    case BodyType::Carry:
      evaluateCarry();
      break;
    case BodyType::Invariant:
      evaluateInvariant();
      break;
    case BodyType::Gate:
      evaluateGate();
      break;
    case BodyType::Stream:
      evaluateStream();
      break;
    default:
      break;
    }
  }

  void commitStatefulBody() {
    for (auto &flight : inflight_) {
      if (flight.remainingCycles > 0)
        --flight.remainingCycles;
    }
    while (!inflight_.empty() && inflight_.front().remainingCycles == 0 &&
           canDrainIntoOutputRegisters(inflight_.front().outputs)) {
      drainToOutputRegisters(inflight_.front().outputs);
      inflight_.pop_front();
    }
    commitOutputTransfers();
    switch (bodyType_) {
    case BodyType::Load:
      commitLoad();
      break;
    case BodyType::Store:
      commitStore();
      break;
    case BodyType::HandshakeMux:
      captureMuxOperands();
      if (!canFireCurrentBody())
        break;
      {
        auto outputs = computeOutputs(allocateOutputGeneration());
        if (!outputs.has_value())
          break;
        inflight_.push_back({static_cast<unsigned>(latency_), *outputs});
        muxSelectorToken_.reset();
        muxSelectedInputIdx_.reset();
        muxSelectedDataToken_.reset();
        cyclesSinceLastFire_ = 0;
        emittedFireThisCycle_ = true;
        perf_.activeCycles++;
        perf_.tokensIn += 2;
      }
      break;
    case BodyType::Carry:
      commitCarry();
      break;
    case BodyType::Invariant:
      commitInvariant();
      break;
    case BodyType::Gate:
      commitGate();
      break;
    case BodyType::Stream:
      commitStream();
      break;
    default:
      break;
    }
  }

  bool bodyUsesGenericOperandLatches() const {
    switch (bodyType_) {
    case BodyType::Select:
    case BodyType::Compute:
    case BodyType::CondBranch:
    case BodyType::Join:
      return true;
    default:
      return false;
    }
  }

  bool inputRequiredForCurrentBody(size_t idx) const {
    switch (bodyType_) {
    case BodyType::Join:
      return idx < 64 && (joinMask_ & (uint64_t{1} << idx)) != 0;
    case BodyType::CondBranch:
      return idx < 2;
    case BodyType::Select:
      if (opName_ == "arith.select")
        return idx < 3;
      if (opName_ == "fabric.mux") {
        if (fabricMuxFields_.empty() || fabricMuxFields_.front().disconnect)
          return false;
        return idx == fabricMuxFields_.front().sel;
      }
      return idx < inputs.size();
    case BodyType::Compute:
      if (opName_ == "math.fma")
        return idx < 3;
      if (isUnaryComputeOp(opName_))
        return idx == 0;
      if (isBinaryComputeOp(opName_))
        return idx < 2;
      return idx < inputs.size();
    default:
      return idx < inputs.size();
    }
  }

  bool inputAvailableForCurrentBody(size_t idx) const {
    if (!inputRequiredForCurrentBody(idx) || idx >= inputs.size())
      return true;
    if (idx < operandTokens_.size() && operandTokens_[idx].has_value())
      return true;
    return inputFresh(idx) && inputs[idx]->ready;
  }

  unsigned countFreshInputsForCurrentBody() const {
    unsigned count = 0;
    for (size_t idx = 0; idx < inputs.size(); ++idx) {
      if (!inputRequiredForCurrentBody(idx))
        continue;
      if (idx < operandTokens_.size() && operandTokens_[idx].has_value())
        continue;
      ++count;
    }
    return count;
  }

  unsigned countLatchedOperands() const {
    unsigned count = 0;
    for (const auto &token : operandTokens_) {
      if (token.has_value())
        ++count;
    }
    return count;
  }

  void captureGenericOperands() {
    if (!bodyUsesGenericOperandLatches())
      return;
    for (size_t idx = 0; idx < inputs.size(); ++idx) {
      if (!inputRequiredForCurrentBody(idx))
        continue;
      if (idx >= operandTokens_.size() || operandTokens_[idx].has_value())
        continue;
      if (!inputFresh(idx) || !inputs[idx]->ready || !inputs[idx]->transferred())
        continue;
      operandTokens_[idx] = tokenFromChannel(*inputs[idx]);
      markInputConsumed(idx);
      ++inputCaptureCount_;
      ++perf_.activeCycles;
      ++perf_.tokensIn;
      emittedFireThisCycle_ = true;
    }
  }

  void evaluateCarry() {
    if (outputs.empty())
      return;
    if (dataflowInitialStage_) {
      bool ready = !initLatched_ && inputs.size() >= 3 && outputRegistersFree({0});
      if (inputs.size() > 0) setInputReadyFreshAware(0, false);
      if (inputs.size() > 1) setInputReadyFreshAware(1, ready);
      if (inputs.size() > 2) setInputReadyFreshAware(2, false);
      return;
    }
    if (initLatched_) {
      if (inputs.size() > 0) setInputReadyFreshAware(0, false);
      if (inputs.size() > 1) setInputReadyFreshAware(1, false);
      if (inputs.size() > 2) setInputReadyFreshAware(2, false);
      return;
    }
    if (!carryDLatched_) {
      bool ready = inputs.size() >= 1;
      if (inputs.size() > 0) setInputReadyFreshAware(0, ready);
      if (inputs.size() > 1) setInputReadyFreshAware(1, false);
      if (inputs.size() > 2) setInputReadyFreshAware(2, false);
    } else {
      bool ready = inputs.size() >= 3 && outputRegistersFree({0});
      if (inputs.size() > 0) setInputReadyFreshAware(0, false);
      if (inputs.size() > 1) setInputReadyFreshAware(1, false);
      if (inputs.size() > 2) setInputReadyFreshAware(2, ready);
    }
  }

  void commitLoad() {
    if (inputs.size() < 3 || outputs.size() < 2)
      return;

    if (!loadAddrToken_.has_value() && inputFresh(0) && inputs[0]->ready &&
        inputs[0]->transferred()) {
      loadAddrToken_ = tokenFromChannel(*inputs[0]);
      markInputConsumed(0);
      ++inputCaptureCount_;
      ++perf_.activeCycles;
      ++perf_.tokensIn;
      emittedFireThisCycle_ = true;
    }
    if (!loadDataToken_.has_value() && inputFresh(1) && inputs[1]->ready &&
        inputs[1]->transferred()) {
      loadDataToken_ = tokenFromChannel(*inputs[1]);
      markInputConsumed(1);
      ++inputCaptureCount_;
      ++perf_.activeCycles;
      ++perf_.tokensIn;
      emittedFireThisCycle_ = true;
    }
    if (!loadCtrlToken_.has_value() && inputFresh(2) && inputs[2]->ready &&
        inputs[2]->transferred()) {
      loadCtrlToken_ = tokenFromChannel(*inputs[2]);
      markInputConsumed(2);
      ++inputCaptureCount_;
      ++perf_.activeCycles;
      ++perf_.tokensIn;
      emittedFireThisCycle_ = true;
    }

    if (loadAddrToken_.has_value() && loadCtrlToken_.has_value() &&
        !outputRegisters_[1].has_value()) {
      uint64_t generation = allocateOutputGeneration();
      outputRegisters_[1] = makeGeneratedToken(
          1, loadAddrToken_->data, loadAddrToken_->tag, loadAddrToken_->hasTag,
          generation);
      loadAddrToken_.reset();
      loadCtrlToken_.reset();
      ++logicalFireCount_;
      ++loadIssueCount_;
      ++perf_.activeCycles;
      emittedFireThisCycle_ = true;
    }
    if (loadDataToken_.has_value() && !outputRegisters_[0].has_value()) {
      uint64_t generation = allocateOutputGeneration();
      outputRegisters_[0] = makeGeneratedToken(
          0, loadDataToken_->data, loadDataToken_->tag, loadDataToken_->hasTag,
          generation);
      loadDataToken_.reset();
      ++loadReturnCount_;
      ++perf_.activeCycles;
      emittedFireThisCycle_ = true;
    }
  }

  void commitStore() {
    if (inputs.size() < 3 || outputs.size() < 2)
      return;
    if (!storeAddrToken_.has_value() && inputFresh(0) && inputs[0]->ready &&
        inputs[0]->transferred()) {
      storeAddrToken_ = tokenFromChannel(*inputs[0]);
      markInputConsumed(0);
      ++inputCaptureCount_;
      ++perf_.activeCycles;
      ++perf_.tokensIn;
      emittedFireThisCycle_ = true;
    }
    if (!storeDataToken_.has_value() && inputFresh(1) && inputs[1]->ready &&
        inputs[1]->transferred()) {
      storeDataToken_ = tokenFromChannel(*inputs[1]);
      markInputConsumed(1);
      ++inputCaptureCount_;
      ++perf_.activeCycles;
      ++perf_.tokensIn;
      emittedFireThisCycle_ = true;
    }
    if (!storeCtrlToken_.has_value() && inputFresh(2) && inputs[2]->ready &&
        inputs[2]->transferred()) {
      storeCtrlToken_ = tokenFromChannel(*inputs[2]);
      markInputConsumed(2);
      ++inputCaptureCount_;
      ++perf_.activeCycles;
      ++perf_.tokensIn;
      emittedFireThisCycle_ = true;
    }
    if (!storeAddrToken_.has_value() || !storeDataToken_.has_value() ||
        !storeCtrlToken_.has_value() || outputRegisters_[0].has_value() ||
        outputRegisters_[1].has_value())
      return;
    uint64_t generation = allocateOutputGeneration();
    outputRegisters_[0] = makeGeneratedToken(
        0, storeDataToken_->data, storeDataToken_->tag, storeDataToken_->hasTag,
        generation);
    outputRegisters_[1] = makeGeneratedToken(
        1, storeAddrToken_->data, storeAddrToken_->tag, storeAddrToken_->hasTag,
        generation);
    storeAddrToken_.reset();
    storeDataToken_.reset();
    storeCtrlToken_.reset();
    ++logicalFireCount_;
    ++storeIssueCount_;
    ++perf_.activeCycles;
    emittedFireThisCycle_ = true;
  }

  void commitCarry() {
    if (initLatched_ && outputRegistersFree({0})) {
      initLatched_ = false;
      if (dataflowInitialStage_)
        dataflowInitialStage_ = false;
      else
        carryDLatched_ = false;
      return;
    }
    if (dataflowInitialStage_) {
      bool fire = !initLatched_ && inputs.size() >= 2 &&
                  inputFresh(1) && inputs[1]->ready &&
                  outputRegistersFree({0});
      if (!fire)
        return;
      SimToken token = makeGeneratedToken(
          0, inputs[1]->data, inputs[1]->tag, inputs[1]->hasTag,
          allocateOutputGeneration());
      outputRegisters_[0] = token;
      initLatched_ = true;
      markInputConsumed(1);
      ++inputCaptureCount_;
      ++logicalFireCount_;
      ++carryInitCount_;
      perf_.activeCycles++;
      perf_.tokensIn += 1;
      emittedFireThisCycle_ = true;
      return;
    }
    if (!carryDLatched_) {
      bool latch = inputs.size() >= 1 && inputFresh(0) && inputs[0]->ready;
      if (!latch)
        return;
      carryDValue_ = (inputs[0]->data & 1u) != 0;
      markInputConsumed(0);
      ++inputCaptureCount_;
      perf_.activeCycles++;
      perf_.tokensIn++;
      emittedFireThisCycle_ = true;
      if (!carryDValue_) {
        dataflowInitialStage_ = true;
        carryDLatched_ = false;
        ++carryResetCount_;
        return;
      }
      carryDLatched_ = true;
      return;
    }
    bool latchLoop = inputs.size() >= 3 && inputFresh(2) && inputs[2]->ready &&
                     carryDLatched_ && !initLatched_ &&
                     outputRegistersFree({0});
    if (!latchLoop)
      return;
    if (carryDValue_) {
      uint64_t generation = allocateOutputGeneration();
      SimToken token = makeGeneratedToken(0, inputs[2]->data, inputs[2]->tag,
                                          inputs[2]->hasTag, generation);
      enqueueOutputToken(0, token);
    }
    initLatched_ = true;
    markInputConsumed(2);
    ++inputCaptureCount_;
    ++logicalFireCount_;
    ++carryLoopCount_;
    perf_.activeCycles++;
    perf_.tokensIn++;
    emittedFireThisCycle_ = true;
  }

  void captureMuxOperands() {
    if (bodyType_ != BodyType::HandshakeMux || inputs.size() < 2)
      return;
    if (!muxSelectorToken_.has_value() && inputFresh(0) && inputs[0]->ready) {
      muxSelectorToken_ = tokenFromChannel(*inputs[0]);
      markInputConsumed(0);
      ++inputCaptureCount_;
      size_t selectedIdx = static_cast<size_t>(muxSelectorToken_->data + 1);
      if (selectedIdx < inputs.size())
        muxSelectedInputIdx_ = selectedIdx;
    }
    if (muxSelectorToken_.has_value() && muxSelectedInputIdx_.has_value() &&
        !muxSelectedDataToken_.has_value() &&
        *muxSelectedInputIdx_ < inputs.size() &&
        inputFresh(*muxSelectedInputIdx_) &&
        inputs[*muxSelectedInputIdx_]->ready) {
      muxSelectedDataToken_ = tokenFromChannel(*inputs[*muxSelectedInputIdx_]);
      markInputConsumed(*muxSelectedInputIdx_);
      ++inputCaptureCount_;
    }
  }

  void evaluateInvariant() {
    if (outputs.empty())
      return;
    if (dataflowInitialStage_) {
      bool ready = !initLatched_ && inputs.size() >= 2 && outputRegistersFree({0});
      if (inputs.size() > 0) setInputReadyFreshAware(0, false);
      if (inputs.size() > 1) setInputReadyFreshAware(1, ready);
      return;
    }
    if (initLatched_) {
      if (inputs.size() > 0)
        setInputReadyFreshAware(0, false);
      return;
    }
    bool cond = inputs.size() >= 1 && inputs[0]->valid &&
                ((inputs[0]->data & 1u) != 0);
    bool ready = inputs.size() >= 1 &&
                 (cond ? outputRegistersFree({0}) : true);
    if (inputs.size() > 0)
      setInputReadyFreshAware(0, ready);
  }

  void commitInvariant() {
    if (initLatched_ && outputRegistersFree({0})) {
      initLatched_ = false;
      if (dataflowInitialStage_)
        dataflowInitialStage_ = false;
      return;
    }
    if (dataflowInitialStage_) {
      bool fire = !initLatched_ && inputs.size() >= 2 &&
                  inputFresh(1) && inputs[1]->ready &&
                  outputRegistersFree({0});
      if (!fire)
        return;
      invariantStoredValue_ = inputs[1]->data;
      enqueueOutputToken(0, makeGeneratedToken(0, invariantStoredValue_,
                                               inputs[1]->tag,
                                               inputs[1]->hasTag,
                                               allocateOutputGeneration()));
      initLatched_ = true;
      markInputConsumed(1);
      ++inputCaptureCount_;
      ++logicalFireCount_;
      ++invariantInitCount_;
      perf_.activeCycles++;
      perf_.tokensIn += 1;
      emittedFireThisCycle_ = true;
      return;
    }
    bool fire = inputs.size() >= 1 && inputFresh(0) && inputs[0]->ready;
    if (!fire)
      return;
    markInputConsumed(0);
    ++inputCaptureCount_;
    perf_.activeCycles++;
    perf_.tokensIn++;
    emittedFireThisCycle_ = true;
    if ((inputs[0]->data & 1u) == 0) {
      dataflowInitialStage_ = true;
      ++invariantResetCount_;
      return;
    }
    SimToken token = makeGeneratedToken(0, invariantStoredValue_, 0, false,
                                        allocateOutputGeneration());
    enqueueOutputToken(0, token);
    initLatched_ = true;
    ++logicalFireCount_;
    ++invariantLoopCount_;
  }

  void evaluateGate() {
    if (outputs.size() < 2 || inputs.size() < 2)
      return;
    if (gateState_ == 0) {
      bool canCaptureValue = !gateValueToken_.has_value();
      bool canCaptureCond = !gateCondToken_.has_value();
      setInputReadyFreshAware(0, canCaptureValue);
      setInputReadyFreshAware(1, canCaptureCond);
      return;
    }
    if (gateState_ == 1) {
      return;
    }
    bool canCaptureValue = !gateValueToken_.has_value();
    bool canCaptureCond = !gateCondToken_.has_value();
    setInputReadyFreshAware(0, canCaptureValue);
    setInputReadyFreshAware(1, canCaptureCond);
    if (gateState_ == 3 || gateState_ == 4) {
      if (usesElasticOutputQueues() && !outputQueues_[0].empty()) {
        driveChannelFromToken(*outputs[0], outputQueues_[0].front());
      } else if (!usesElasticOutputQueues() &&
                 outputRegisters_[0].has_value()) {
        driveChannelFromToken(*outputs[0], *outputRegisters_[0]);
      }
      if (usesElasticOutputQueues() && !outputQueues_[1].empty()) {
        driveChannelFromToken(*outputs[1], outputQueues_[1].front());
      } else if (!usesElasticOutputQueues() &&
                 outputRegisters_[1].has_value()) {
        driveChannelFromToken(*outputs[1], *outputRegisters_[1]);
      }
    }
  }

  void commitGate() {
    if (outputs.size() < 2 || inputs.size() < 2)
      return;
    if (!gateValueToken_.has_value() && inputFresh(0) && inputs[0]->ready &&
        inputs[0]->transferred()) {
      gateValueToken_ = tokenFromChannel(*inputs[0]);
      markInputConsumed(0);
      ++inputCaptureCount_;
      perf_.activeCycles++;
      perf_.tokensIn++;
      emittedFireThisCycle_ = true;
    }
    if (!gateCondToken_.has_value() && inputFresh(1) && inputs[1]->ready &&
        inputs[1]->transferred()) {
      gateCondToken_ = tokenFromChannel(*inputs[1]);
      markInputConsumed(1);
      ++inputCaptureCount_;
      perf_.activeCycles++;
      perf_.tokensIn++;
      emittedFireThisCycle_ = true;
    }
    if (gateState_ == 0) {
      bool fire = gateValueToken_.has_value() && gateCondToken_.has_value();
      if (!fire)
        return;
      if ((gateCondToken_->data & 1u) != 0) {
        uint64_t generation = allocateOutputGeneration();
        enqueueOutputToken(0, makeGeneratedToken(
                                  0, gateValueToken_->data, gateValueToken_->tag,
                                  gateValueToken_->hasTag, generation));
        std::fill(gateOutputsAccepted_.begin(), gateOutputsAccepted_.end(), false);
        gateState_ = 1;
        ++logicalFireCount_;
        ++gateHeadCount_;
      } else {
        gateState_ = 0;
      }
      gateValueToken_.reset();
      gateCondToken_.reset();
      perf_.activeCycles++;
      emittedFireThisCycle_ = true;
      return;
    }
    if (gateState_ == 1) {
      if (!outputs[0]->transferred())
        return;
      perf_.tokensOut++;
      ++outputTransferCount_;
      perf_.activeCycles++;
      clearOutputBuffer(0);
      gateState_ = 2;
      return;
    }
    if (gateState_ == 2) {
      bool fire = gateValueToken_.has_value() && gateCondToken_.has_value();
      if (!fire)
        return;
      gateLatchedCond_ = (gateCondToken_->data & 1u) != 0;
      uint64_t generation = allocateOutputGeneration();
      if (gateLatchedCond_) {
        enqueueOutputToken(0, makeGeneratedToken(
                                  0, gateValueToken_->data, gateValueToken_->tag,
                                  gateValueToken_->hasTag, generation));
        enqueueOutputToken(1,
                           makeGeneratedToken(1, 1u, 0, false, generation));
        std::fill(gateOutputsAccepted_.begin(), gateOutputsAccepted_.end(), false);
        gateState_ = 3;
        ++logicalFireCount_;
        ++gateTrueCount_;
      } else {
        enqueueOutputToken(1,
                           makeGeneratedToken(1, 0u, 0, false, generation));
        std::fill(gateOutputsAccepted_.begin(), gateOutputsAccepted_.end(), false);
        gateState_ = 4;
        ++logicalFireCount_;
        ++gateFalseCount_;
      }
      gateValueToken_.reset();
      gateCondToken_.reset();
      perf_.activeCycles++;
      emittedFireThisCycle_ = true;
      return;
    }
    if (gateState_ == 3) {
      for (size_t idx = 0; idx < 2 && idx < outputs.size() &&
                           idx < gateOutputsAccepted_.size();
           ++idx) {
        if (outputs[idx]->transferred() && !gateOutputsAccepted_[idx]) {
          gateOutputsAccepted_[idx] = true;
          perf_.tokensOut++;
          ++outputTransferCount_;
          perf_.activeCycles++;
        }
      }
      if (!(gateOutputsAccepted_[0] && gateOutputsAccepted_[1]))
        return;
      clearOutputBuffer(0);
      clearOutputBuffer(1);
      gateOutputsAccepted_[0] = false;
      gateOutputsAccepted_[1] = false;
      gateState_ = 2;
      return;
    }
    if (gateState_ == 4) {
      if (!outputs[1]->transferred())
        return;
      perf_.tokensOut++;
      ++outputTransferCount_;
      perf_.activeCycles++;
      clearOutputBuffer(1);
      gateOutputsAccepted_[1] = false;
      gateState_ = 0;
    }
  }

  bool evalStreamCond(int64_t idx, int64_t bound) const {
    if (streamContCond_ & 0x01) return idx < bound;
    if (streamContCond_ & 0x02) return idx <= bound;
    if (streamContCond_ & 0x04) return idx > bound;
    if (streamContCond_ & 0x08) return idx >= bound;
    if (streamContCond_ & 0x10) return idx != bound;
    return false;
  }

  uint64_t nextStreamValue(uint64_t idx, uint64_t step) const {
    return idx + step;
  }

  void evaluateStream() {
    if (outputs.size() < 2)
      return;
    if (streamInitialPhase_) {
      if (inputs.size() > 0)
        setInputReadyFreshAware(0, !streamStartToken_.has_value());
      if (inputs.size() > 1)
        setInputReadyFreshAware(1, !streamStepToken_.has_value());
      if (inputs.size() > 2)
        setInputReadyFreshAware(2, !streamBoundToken_.has_value());
      return;
    }
    if (usesElasticOutputQueues() && !outputQueues_[0].empty()) {
      driveChannelFromToken(*outputs[0], outputQueues_[0].front());
    } else if (!usesElasticOutputQueues() &&
               outputRegisters_[0].has_value()) {
      driveChannelFromToken(*outputs[0], *outputRegisters_[0]);
    }
    if (usesElasticOutputQueues() && !outputQueues_[1].empty()) {
      driveChannelFromToken(*outputs[1], outputQueues_[1].front());
    } else if (!usesElasticOutputQueues() &&
               outputRegisters_[1].has_value()) {
      driveChannelFromToken(*outputs[1], *outputRegisters_[1]);
    }
  }

  void commitStream() {
    if (outputs.size() < 2)
      return;
    if (streamInitialPhase_) {
      if (!streamStartToken_.has_value() && inputs.size() > 0 &&
          inputFresh(0) && inputs[0]->ready && inputs[0]->transferred()) {
        streamStartToken_ = tokenFromChannel(*inputs[0]);
        markInputConsumed(0);
        ++inputCaptureCount_;
        perf_.activeCycles++;
        perf_.tokensIn++;
        emittedFireThisCycle_ = true;
      }
      if (!streamStepToken_.has_value() && inputs.size() > 1 &&
          inputFresh(1) && inputs[1]->ready && inputs[1]->transferred()) {
        streamStepToken_ = tokenFromChannel(*inputs[1]);
        markInputConsumed(1);
        ++inputCaptureCount_;
        perf_.activeCycles++;
        perf_.tokensIn++;
        emittedFireThisCycle_ = true;
      }
      if (!streamBoundToken_.has_value() && inputs.size() > 2 &&
          inputFresh(2) && inputs[2]->ready && inputs[2]->transferred()) {
        streamBoundToken_ = tokenFromChannel(*inputs[2]);
        markInputConsumed(2);
        ++inputCaptureCount_;
        perf_.activeCycles++;
        perf_.tokensIn++;
        emittedFireThisCycle_ = true;
      }
      bool fire = streamStartToken_.has_value() && streamStepToken_.has_value() &&
                  streamBoundToken_.has_value() && outputRegistersFree({0, 1});
      if (!fire)
        return;
      uint64_t idx = streamStartToken_->data;
      uint64_t step = streamStepToken_->data;
      uint64_t bound = streamBoundToken_->data;
      bool cont =
          evalStreamCond(static_cast<int64_t>(idx), static_cast<int64_t>(bound));
      uint64_t generation = allocateOutputGeneration();
      enqueueOutputToken(0, makeGeneratedToken(0, idx, 0, false, generation));
      enqueueOutputToken(1,
                         makeGeneratedToken(1, cont ? 1u : 0u, 0, false,
                                            generation));
      std::fill(streamOutputsAccepted_.begin(), streamOutputsAccepted_.end(),
                false);
      streamNextIdx_ = nextStreamValue(idx, step);
      streamStepReg_ = step;
      streamBoundReg_ = bound;
      streamInitialPhase_ = false;
      streamTerminalPending_ = !cont;
      ++logicalFireCount_;
      ++streamEmitCount_;
      if (!cont)
        ++streamTerminalCount_;
      streamStartToken_.reset();
      streamStepToken_.reset();
      streamBoundToken_.reset();
      perf_.activeCycles++;
      emittedFireThisCycle_ = true;
      return;
    }
    for (size_t idx = 0; idx < 2 && idx < outputs.size() &&
                         idx < streamOutputsAccepted_.size();
         ++idx) {
      if (outputs[idx]->transferred() && !streamOutputsAccepted_[idx]) {
        streamOutputsAccepted_[idx] = true;
        perf_.tokensOut++;
        ++outputTransferCount_;
        perf_.activeCycles++;
      }
    }
    if (!(streamOutputsAccepted_[0] && streamOutputsAccepted_[1]))
      return;
    clearOutputBuffer(0);
    clearOutputBuffer(1);
    streamOutputsAccepted_[0] = false;
    streamOutputsAccepted_[1] = false;
    if (streamTerminalPending_) {
      streamInitialPhase_ = true;
      streamTerminalPending_ = false;
      return;
    }
    if (!outputRegistersFree({0, 1}))
      return;
    uint64_t idx = streamNextIdx_;
    bool cont = evalStreamCond(static_cast<int64_t>(idx),
                               static_cast<int64_t>(streamBoundReg_));
    uint64_t generation = allocateOutputGeneration();
    enqueueOutputToken(0, makeGeneratedToken(0, idx, 0, false, generation));
    enqueueOutputToken(1,
                       makeGeneratedToken(1, cont ? 1u : 0u, 0, false,
                                          generation));
    std::fill(streamOutputsAccepted_.begin(), streamOutputsAccepted_.end(),
              false);
    ++logicalFireCount_;
    ++streamEmitCount_;
    streamTerminalPending_ = !cont;
    if (!cont)
      ++streamTerminalCount_;
    if (cont)
      streamNextIdx_ = nextStreamValue(idx, streamStepReg_);
  }

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

} // namespace

bool functionUnitModuleSupportedByCycleKernel(const StaticModuleDesc &module) {
  if (module.kind != StaticModuleKind::FunctionUnit)
    return false;
  for (const auto &attr : module.strAttrs) {
    if (attr.name == "pe_kind" && attr.value == "temporal_pe")
      return false;
  }
  const auto *ops = getStringArrayAttr(module, "ops");
  if (!ops || ops->size() != 1)
    return false;
  return classifyFunctionUnitBody(module) != BodyType::Unsupported;
}

std::unique_ptr<SimModule> createFunctionUnitModule(
    const StaticModuleDesc &module, const StaticMappedModel &model) {
  if (!functionUnitModuleSupportedByCycleKernel(module))
    return nullptr;
  return std::make_unique<FunctionUnitModule>(module, model);
}

} // namespace sim
} // namespace fcc
