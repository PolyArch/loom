#include "loom/Simulator/SimFunctionUnitInternal.h"

#include <algorithm>

namespace loom {
namespace sim {

void FunctionUnitModule::captureGenericOperands() {
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

void FunctionUnitModule::evaluateCarry() {
  if (outputs.empty())
    return;
  if (dataflowInitialStage_) {
    bool ready =
        !initLatched_ && inputs.size() >= 3 && outputRegistersFree({0});
    if (inputs.size() > 0)
      setInputReadyFreshAware(0, false);
    if (inputs.size() > 1)
      setInputReadyFreshAware(1, ready);
    if (inputs.size() > 2)
      setInputReadyFreshAware(2, false);
    return;
  }
  if (initLatched_) {
    if (inputs.size() > 0)
      setInputReadyFreshAware(0, false);
    if (inputs.size() > 1)
      setInputReadyFreshAware(1, false);
    if (inputs.size() > 2)
      setInputReadyFreshAware(2, false);
    return;
  }
  if (!carryDLatched_) {
    bool ready = inputs.size() >= 1;
    if (inputs.size() > 0)
      setInputReadyFreshAware(0, ready);
    if (inputs.size() > 1)
      setInputReadyFreshAware(1, false);
    if (inputs.size() > 2)
      setInputReadyFreshAware(2, false);
  } else {
    bool ready = inputs.size() >= 3 && outputRegistersFree({0});
    if (inputs.size() > 0)
      setInputReadyFreshAware(0, false);
    if (inputs.size() > 1)
      setInputReadyFreshAware(1, false);
    if (inputs.size() > 2)
      setInputReadyFreshAware(2, ready);
  }
}

void FunctionUnitModule::commitLoad() {
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

void FunctionUnitModule::commitStore() {
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

void FunctionUnitModule::commitCarry() {
  if (initLatched_ && outputRegistersFree({0})) {
    initLatched_ = false;
    if (dataflowInitialStage_)
      dataflowInitialStage_ = false;
    else
      carryDLatched_ = false;
    return;
  }
  if (dataflowInitialStage_) {
    bool fire = !initLatched_ && inputs.size() >= 2 && inputFresh(1) &&
                inputs[1]->ready && outputRegistersFree({0});
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

void FunctionUnitModule::captureMuxOperands() {
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

void FunctionUnitModule::evaluateInvariant() {
  if (outputs.empty())
    return;
  if (dataflowInitialStage_) {
    bool ready =
        !initLatched_ && inputs.size() >= 2 && outputRegistersFree({0});
    if (inputs.size() > 0)
      setInputReadyFreshAware(0, false);
    if (inputs.size() > 1)
      setInputReadyFreshAware(1, ready);
    return;
  }
  if (initLatched_) {
    if (inputs.size() > 0)
      setInputReadyFreshAware(0, false);
    return;
  }
  bool cond = inputs.size() >= 1 && inputs[0]->valid &&
              ((inputs[0]->data & 1u) != 0);
  bool ready = inputs.size() >= 1 && (cond ? outputRegistersFree({0}) : true);
  if (inputs.size() > 0)
    setInputReadyFreshAware(0, ready);
}

void FunctionUnitModule::commitInvariant() {
  if (initLatched_ && outputRegistersFree({0})) {
    initLatched_ = false;
    if (dataflowInitialStage_)
      dataflowInitialStage_ = false;
    return;
  }
  if (dataflowInitialStage_) {
    bool fire = !initLatched_ && inputs.size() >= 2 && inputFresh(1) &&
                inputs[1]->ready && outputRegistersFree({0});
    if (!fire)
      return;
    invariantStoredValue_ = inputs[1]->data;
    enqueueOutputToken(0,
                       makeGeneratedToken(0, invariantStoredValue_,
                                          inputs[1]->tag, inputs[1]->hasTag,
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

void FunctionUnitModule::evaluateGate() {
  if (outputs.size() < 2 || inputs.size() < 2)
    return;
  if (gateState_ == 0) {
    bool canCaptureValue = !gateValueToken_.has_value();
    bool canCaptureCond = !gateCondToken_.has_value();
    setInputReadyFreshAware(0, canCaptureValue);
    setInputReadyFreshAware(1, canCaptureCond);
    return;
  }
  if (gateState_ == 1)
    return;
  bool canCaptureValue = !gateValueToken_.has_value();
  bool canCaptureCond = !gateCondToken_.has_value();
  setInputReadyFreshAware(0, canCaptureValue);
  setInputReadyFreshAware(1, canCaptureCond);
  if (gateState_ == 3 || gateState_ == 4) {
    if (usesElasticOutputQueues() && !outputQueues_[0].empty()) {
      driveChannelFromToken(*outputs[0], outputQueues_[0].front());
    } else if (!usesElasticOutputQueues() && outputRegisters_[0].has_value()) {
      driveChannelFromToken(*outputs[0], *outputRegisters_[0]);
    }
    if (usesElasticOutputQueues() && !outputQueues_[1].empty()) {
      driveChannelFromToken(*outputs[1], outputQueues_[1].front());
    } else if (!usesElasticOutputQueues() && outputRegisters_[1].has_value()) {
      driveChannelFromToken(*outputs[1], *outputRegisters_[1]);
    }
  }
}

void FunctionUnitModule::commitGate() {
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
      enqueueOutputToken(1, makeGeneratedToken(1, 1u, 0, false, generation));
      std::fill(gateOutputsAccepted_.begin(), gateOutputsAccepted_.end(), false);
      gateState_ = 3;
      ++logicalFireCount_;
      ++gateTrueCount_;
    } else {
      enqueueOutputToken(1, makeGeneratedToken(1, 0u, 0, false, generation));
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

bool FunctionUnitModule::evalStreamCond(int64_t idx, int64_t bound) const {
  if (streamContCond_ & 0x01)
    return idx < bound;
  if (streamContCond_ & 0x02)
    return idx <= bound;
  if (streamContCond_ & 0x04)
    return idx > bound;
  if (streamContCond_ & 0x08)
    return idx >= bound;
  if (streamContCond_ & 0x10)
    return idx != bound;
  return false;
}

uint64_t FunctionUnitModule::nextStreamValue(uint64_t idx, uint64_t step) const {
  return idx + step;
}

void FunctionUnitModule::evaluateStream() {
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
  } else if (!usesElasticOutputQueues() && outputRegisters_[0].has_value()) {
    driveChannelFromToken(*outputs[0], *outputRegisters_[0]);
  }
  if (usesElasticOutputQueues() && !outputQueues_[1].empty()) {
    driveChannelFromToken(*outputs[1], outputQueues_[1].front());
  } else if (!usesElasticOutputQueues() && outputRegisters_[1].has_value()) {
    driveChannelFromToken(*outputs[1], *outputRegisters_[1]);
  }
}

void FunctionUnitModule::commitStream() {
  if (outputs.size() < 2)
    return;
  if (streamInitialPhase_) {
    if (!streamStartToken_.has_value() && inputs.size() > 0 && inputFresh(0) &&
        inputs[0]->ready && inputs[0]->transferred()) {
      streamStartToken_ = tokenFromChannel(*inputs[0]);
      markInputConsumed(0);
      ++inputCaptureCount_;
      perf_.activeCycles++;
      perf_.tokensIn++;
      emittedFireThisCycle_ = true;
    }
    if (!streamStepToken_.has_value() && inputs.size() > 1 && inputFresh(1) &&
        inputs[1]->ready && inputs[1]->transferred()) {
      streamStepToken_ = tokenFromChannel(*inputs[1]);
      markInputConsumed(1);
      ++inputCaptureCount_;
      perf_.activeCycles++;
      perf_.tokensIn++;
      emittedFireThisCycle_ = true;
    }
    if (!streamBoundToken_.has_value() && inputs.size() > 2 && inputFresh(2) &&
        inputs[2]->ready && inputs[2]->transferred()) {
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
  bool cont =
      evalStreamCond(static_cast<int64_t>(idx), static_cast<int64_t>(streamBoundReg_));
  uint64_t generation = allocateOutputGeneration();
  enqueueOutputToken(0, makeGeneratedToken(0, idx, 0, false, generation));
  enqueueOutputToken(1,
                     makeGeneratedToken(1, cont ? 1u : 0u, 0, false,
                                        generation));
  std::fill(streamOutputsAccepted_.begin(), streamOutputsAccepted_.end(), false);
  ++logicalFireCount_;
  ++streamEmitCount_;
  streamTerminalPending_ = !cont;
  if (!cont)
    ++streamTerminalCount_;
  if (cont)
    streamNextIdx_ = nextStreamValue(idx, streamStepReg_);
}

} // namespace sim
} // namespace loom
