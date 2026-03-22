#include "loom/Simulator/SimFunctionUnitInternal.h"

#include <algorithm>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace loom {
namespace sim {

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
                   int64_t defaultValue) {
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

std::string modulePrimaryOp(const StaticModuleDesc &module) {
  const auto *ops = getStringArrayAttr(module, "ops");
  if (!ops || ops->empty())
    return {};
  return ops->front();
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

FunctionUnitModule::FunctionUnitModule(const StaticModuleDesc &module,
                                       const StaticMappedModel &model)
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

bool FunctionUnitModule::isCombinational() const { return false; }

void FunctionUnitModule::reset() {
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

void FunctionUnitModule::configure(const std::vector<uint32_t> &configWords) {
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
  if (opName_ == "handshake.join")
    joinMask_ = readBits(static_cast<unsigned>(inputs.size()));
}

void FunctionUnitModule::evaluate() {
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

void FunctionUnitModule::commit() {
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

void FunctionUnitModule::collectTraceEvents(std::vector<TraceEvent> &events,
                                            uint64_t cycle) {
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

PerfSnapshot FunctionUnitModule::getPerfSnapshot() const { return perf_; }

void FunctionUnitModule::debugDump(std::ostream &os) const {
  os << "      fu body=" << opName_ << " type="
     << static_cast<unsigned>(bodyType_) << " latency=" << latency_
     << " interval=" << interval_ << " emitted=" << emittedFireThisCycle_
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
    return;
  }
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

bool FunctionUnitModule::hasPendingWork() const {
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
  if (!hasStatefulBody_) {
    return std::any_of(operandTokens_.begin(), operandTokens_.end(),
                       [](const auto &token) { return token.has_value(); });
  }

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

uint64_t FunctionUnitModule::getLogicalFireCount() const {
  return logicalFireCount_;
}

uint64_t FunctionUnitModule::getInputCaptureCount() const {
  return inputCaptureCount_;
}

uint64_t FunctionUnitModule::getOutputTransferCount() const {
  return outputTransferCount_;
}

std::vector<NamedCounter> FunctionUnitModule::getDebugCounters() const {
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

std::string FunctionUnitModule::getDebugStateSummary() const {
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

bool functionUnitModuleSupportedByCycleKernel(const StaticModuleDesc &module,
                                              bool allowTemporalPE) {
  if (module.kind != StaticModuleKind::FunctionUnit)
    return false;
  for (const auto &attr : module.strAttrs) {
    if (!allowTemporalPE && attr.name == "pe_kind" &&
        attr.value == "temporal_pe")
      return false;
  }
  const auto *ops = getStringArrayAttr(module, "ops");
  if (!ops || ops->empty())
    return false;
  if (!allowTemporalPE && ops->size() != 1)
    return false;
  return classifyFunctionUnitBody(module) != BodyType::Unsupported;
}

std::unique_ptr<SimModule>
createFunctionUnitModule(const StaticModuleDesc &module,
                         const StaticMappedModel &model,
                         bool allowTemporalPE) {
  if (!functionUnitModuleSupportedByCycleKernel(module, allowTemporalPE))
    return nullptr;
  return std::make_unique<FunctionUnitModule>(module, model);
}

} // namespace sim
} // namespace loom
