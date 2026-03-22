#include "loom/Simulator/SimFunctionUnitInternal.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <vector>

namespace loom {
namespace sim {

bool FunctionUnitModule::usesElasticOutputQueues() const { return false; }

void FunctionUnitModule::enqueueOutputToken(size_t outputIdx,
                                            const SimToken &token) {
  if (usesElasticOutputQueues()) {
    if (outputIdx < outputQueues_.size())
      outputQueues_[outputIdx].push_back(token);
    return;
  }
  if (outputIdx < outputRegisters_.size())
    outputRegisters_[outputIdx] = token;
}

void FunctionUnitModule::clearOutputBuffer(size_t outputIdx) {
  if (usesElasticOutputQueues()) {
    if (outputIdx < outputQueues_.size() && !outputQueues_[outputIdx].empty())
      outputQueues_[outputIdx].pop_front();
    return;
  }
  if (outputIdx < outputRegisters_.size())
    outputRegisters_[outputIdx].reset();
}

std::string FunctionUnitModule::resolvePrimaryOp(const StaticModuleDesc &module) {
  return modulePrimaryOp(module);
}

unsigned FunctionUnitModule::inferDataWidth(const StaticModuleDesc &module,
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

std::vector<unsigned>
FunctionUnitModule::inferOutputWidths(const StaticModuleDesc &module,
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

std::vector<unsigned>
FunctionUnitModule::inferInputWidths(const StaticModuleDesc &module,
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

unsigned FunctionUnitModule::countConfigMuxFields(const StaticModuleDesc &module) {
  return resolvePrimaryOp(module) == "fabric.mux" ? 1u : 0u;
}

unsigned
FunctionUnitModule::inferConstantValueWidth(const StaticModuleDesc &module,
                                            const StaticMappedModel &model) {
  if (resolvePrimaryOp(module) != "handshake.constant")
    return 0;
  for (IdIndex portId : module.outputPorts) {
    if (const StaticPortDesc *port = findPort(model, portId))
      return std::max(1u, port->valueWidth);
  }
  return 32;
}

uint64_t FunctionUnitModule::maskToWidth(uint64_t value, unsigned width) const {
  if (width >= 64)
    return value;
  return value & ((uint64_t{1} << width) - 1);
}

int64_t FunctionUnitModule::signExtend(uint64_t value, unsigned width) const {
  if (width >= 64)
    return static_cast<int64_t>(value);
  uint64_t mask = uint64_t{1} << (width - 1);
  if (value & mask)
    return static_cast<int64_t>(value | (~uint64_t{0} << width));
  return static_cast<int64_t>(value);
}

float FunctionUnitModule::toFloat(uint64_t value) const {
  float out = 0.0f;
  uint32_t bits = static_cast<uint32_t>(value);
  std::memcpy(&out, &bits, sizeof(out));
  return out;
}

uint64_t FunctionUnitModule::fromFloat(float value) const {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

double FunctionUnitModule::toDouble(uint64_t value) const {
  double out = 0.0;
  std::memcpy(&out, &value, sizeof(out));
  return out;
}

uint64_t FunctionUnitModule::fromDouble(double value) const {
  uint64_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

bool FunctionUnitModule::cmpi(uint64_t a, uint64_t b, unsigned width) const {
  switch (cmpPredicate_) {
  case 0:
    return a == b;
  case 1:
    return a != b;
  case 2:
    return signExtend(a, width) < signExtend(b, width);
  case 3:
    return signExtend(a, width) <= signExtend(b, width);
  case 4:
    return signExtend(a, width) > signExtend(b, width);
  case 5:
    return signExtend(a, width) >= signExtend(b, width);
  case 6:
    return a < b;
  case 7:
    return a <= b;
  case 8:
    return a > b;
  case 9:
    return a >= b;
  default:
    return false;
  }
}

bool FunctionUnitModule::cmpf(uint64_t a, uint64_t b, unsigned width) const {
  double lhs = width <= 32 ? static_cast<double>(toFloat(a)) : toDouble(a);
  double rhs = width <= 32 ? static_cast<double>(toFloat(b)) : toDouble(b);
  bool lhsNaN = std::isnan(lhs);
  bool rhsNaN = std::isnan(rhs);
  switch (cmpPredicate_) {
  case 0:
    return false;
  case 1:
    return lhs == rhs;
  case 2:
    return lhs > rhs;
  case 3:
    return lhs >= rhs;
  case 4:
    return lhs < rhs;
  case 5:
    return lhs <= rhs;
  case 6:
    return lhs != rhs;
  case 7:
    return !lhsNaN && !rhsNaN;
  case 8:
    return lhsNaN || rhsNaN;
  case 9:
    return lhsNaN || rhsNaN || lhs == rhs;
  case 10:
    return lhsNaN || rhsNaN || lhs > rhs;
  case 11:
    return lhsNaN || rhsNaN || lhs >= rhs;
  case 12:
    return lhsNaN || rhsNaN || lhs < rhs;
  case 13:
    return lhsNaN || rhsNaN || lhs <= rhs;
  case 14:
    return lhsNaN || rhsNaN || lhs != rhs;
  case 15:
    return true;
  default:
    return false;
  }
}

void FunctionUnitModule::setAllInputReady(bool ready) {
  for (size_t idx = 0; idx < inputs.size(); ++idx) {
    if (ready) {
      inputs[idx]->ready = true;
      continue;
    }
    inputs[idx]->ready = inputAlreadyConsumed(idx);
  }
}

bool FunctionUnitModule::inputFresh(size_t idx) const {
  return idx < inputs.size() && inputs[idx]->valid &&
         inputs[idx]->generation != 0 &&
         consumedInputGeneration_[idx] != inputs[idx]->generation;
}

bool FunctionUnitModule::inputAlreadyConsumed(size_t idx) const {
  return idx < inputs.size() && inputs[idx]->valid &&
         inputs[idx]->generation != 0 &&
         consumedInputGeneration_[idx] == inputs[idx]->generation;
}

void FunctionUnitModule::setInputReadyFreshAware(size_t idx, bool allowFresh) {
  if (idx >= inputs.size())
    return;
  if (inputAlreadyConsumed(idx)) {
    inputs[idx]->ready = true;
    return;
  }
  inputs[idx]->ready = allowFresh;
}

void FunctionUnitModule::markInputConsumed(size_t idx) {
  if (idx >= inputs.size())
    return;
  consumedInputGeneration_[idx] = inputs[idx]->generation;
}

uint64_t FunctionUnitModule::allocateOutputGeneration() {
  return composeTokenGeneration(hwNodeId, nextOutputGeneration_++);
}

uint64_t FunctionUnitModule::reserveDirectGeneration() {
  if (pendingDirectGeneration_ == 0)
    pendingDirectGeneration_ = nextOutputGeneration_;
  return composeTokenGeneration(hwNodeId, pendingDirectGeneration_);
}

void FunctionUnitModule::finalizeDirectGeneration() {
  if (pendingDirectGeneration_ == 0)
    return;
  nextOutputGeneration_ =
      std::max<uint64_t>(nextOutputGeneration_, pendingDirectGeneration_ + 1);
  pendingDirectGeneration_ = 0;
}

SimToken FunctionUnitModule::makeGeneratedToken(size_t outputIdx, uint64_t data,
                                                uint16_t tag, bool hasTag,
                                                uint64_t generation) const {
  SimToken token;
  unsigned width =
      outputIdx < outputWidths_.size() ? outputWidths_[outputIdx] : dataWidth_;
  token.data = maskToWidth(data, width);
  token.tag = tag;
  token.hasTag = hasTag;
  token.generation = generation;
  return token;
}

bool FunctionUnitModule::anyOutputRegisterBusy() const {
  if (usesElasticOutputQueues()) {
    return std::any_of(outputQueues_.begin(), outputQueues_.end(),
                       [](const auto &queue) { return !queue.empty(); });
  }
  return std::any_of(outputRegisters_.begin(), outputRegisters_.end(),
                     [](const auto &value) { return value.has_value(); });
}

bool FunctionUnitModule::canDrainIntoOutputRegisters(
    const std::vector<std::optional<SimToken>> &outputs) const {
  for (size_t idx = 0; idx < outputs.size() && idx < outputRegisters_.size();
       ++idx) {
    if (outputs[idx].has_value() && outputRegisters_[idx].has_value())
      return false;
  }
  return true;
}

bool FunctionUnitModule::outputRegisterFree(size_t outputIdx) const {
  if (usesElasticOutputQueues())
    return outputIdx < outputQueues_.size();
  return outputIdx < outputRegisters_.size() &&
         !outputRegisters_[outputIdx].has_value();
}

bool FunctionUnitModule::outputRegistersFree(
    std::initializer_list<size_t> outputIdxs) const {
  for (size_t outputIdx : outputIdxs) {
    if (!outputRegisterFree(outputIdx))
      return false;
  }
  return true;
}

void FunctionUnitModule::drainToOutputRegisters(
    const std::vector<std::optional<SimToken>> &outputs) {
  for (size_t idx = 0; idx < outputs.size() && idx < outputRegisters_.size();
       ++idx) {
    if (outputs[idx].has_value())
      outputRegisters_[idx] = outputs[idx];
  }
}

void FunctionUnitModule::driveBufferedOutputs() {
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

void FunctionUnitModule::clearDirectOutputs() {
  directFireArmed_ = false;
  for (auto &token : directOutputs_)
    token.reset();
}

void FunctionUnitModule::driveDirectOutputs() {
  for (size_t idx = 0; idx < outputs.size() && idx < directOutputs_.size();
       ++idx) {
    if (!directOutputs_[idx].has_value())
      continue;
    driveChannelFromToken(*outputs[idx], *directOutputs_[idx]);
  }
}

void FunctionUnitModule::commitOutputTransfers() {
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
  for (size_t idx = 0; idx < outputs.size() && idx < outputRegisters_.size();
       ++idx) {
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
FunctionUnitModule::computeOutputs(uint64_t generation) const {
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
                                     muxSelectedDataToken_->hasTag, generation);
    return produced;
  }
  if (bodyType_ == BodyType::CondBranch) {
    size_t selected = (getInput(0) & 1u) ? 0u : 1u;
    if (selected < produced.size() && inputs.size() >= 2) {
      produced[selected] = makeGeneratedToken(
          selected, getInput(1), getInputTag(1), getInputHasTag(1), generation);
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
    if (opName_ == "arith.addi")
      result = a + b;
    else if (opName_ == "arith.subi")
      result = a - b;
    else if (opName_ == "arith.muli")
      result = a * b;
    else if (opName_ == "arith.divsi")
      result =
          b ? static_cast<uint64_t>(signExtend(a, width) / signExtend(b, width))
            : 0;
    else if (opName_ == "arith.divui")
      result = b ? (a / b) : 0;
    else if (opName_ == "arith.remsi")
      result =
          b ? static_cast<uint64_t>(signExtend(a, width) % signExtend(b, width))
            : 0;
    else if (opName_ == "arith.remui")
      result = b ? (a % b) : 0;
    else if (opName_ == "arith.andi")
      result = a & b;
    else if (opName_ == "arith.ori")
      result = a | b;
    else if (opName_ == "arith.xori")
      result = a ^ b;
    else if (opName_ == "arith.shli")
      result = a << (b & 63);
    else if (opName_ == "arith.shrsi")
      result = static_cast<uint64_t>(signExtend(a, width) >> (b & 63));
    else if (opName_ == "arith.shrui")
      result = a >> (b & 63);
    else if (opName_ == "arith.extsi")
      result = static_cast<uint64_t>(
          signExtend(a, inputWidths_.empty() ? width : inputWidths_[0]));
    else if (opName_ == "arith.extui" || opName_ == "arith.trunci" ||
             opName_ == "arith.index_cast" || opName_ == "arith.index_castui")
      result = a;
    else if (opName_ == "arith.negf")
      result = width <= 32 ? fromFloat(-toFloat(a)) : fromDouble(-toDouble(a));
    else if (opName_ == "arith.addf")
      result = width <= 32 ? fromFloat(toFloat(a) + toFloat(b))
                           : fromDouble(toDouble(a) + toDouble(b));
    else if (opName_ == "arith.subf")
      result = width <= 32 ? fromFloat(toFloat(a) - toFloat(b))
                           : fromDouble(toDouble(a) - toDouble(b));
    else if (opName_ == "arith.mulf")
      result = width <= 32 ? fromFloat(toFloat(a) * toFloat(b))
                           : fromDouble(toDouble(a) * toDouble(b));
    else if (opName_ == "arith.divf")
      result = width <= 32 ? fromFloat(toFloat(a) / toFloat(b))
                           : fromDouble(toDouble(a) / toDouble(b));
    else if (opName_ == "arith.fptosi")
      result = width <= 32 ? static_cast<uint64_t>(static_cast<int64_t>(toFloat(a)))
                           : static_cast<uint64_t>(static_cast<int64_t>(toDouble(a)));
    else if (opName_ == "arith.fptoui")
      result = width <= 32 ? static_cast<uint64_t>(toFloat(a))
                           : static_cast<uint64_t>(toDouble(a));
    else if (opName_ == "arith.sitofp")
      result =
          width <= 32
              ? fromFloat(static_cast<float>(
                    signExtend(a, inputWidths_.empty() ? width : inputWidths_[0])))
              : fromDouble(static_cast<double>(
                    signExtend(a, inputWidths_.empty() ? width : inputWidths_[0])));
    else if (opName_ == "arith.uitofp")
      result = width <= 32 ? fromFloat(static_cast<float>(a))
                           : fromDouble(static_cast<double>(a));
    else if (opName_ == "arith.cmpi")
      result = cmpi(a, b, inputWidths_.empty() ? width : inputWidths_[0]) ? 1 : 0;
    else if (opName_ == "arith.cmpf")
      result = cmpf(a, b, width) ? 1 : 0;
    else if (opName_ == "math.absf")
      result = width <= 32 ? fromFloat(std::fabs(toFloat(a)))
                           : fromDouble(std::fabs(toDouble(a)));
    else if (opName_ == "math.cos")
      result = width <= 32 ? fromFloat(std::cos(toFloat(a)))
                           : fromDouble(std::cos(toDouble(a)));
    else if (opName_ == "math.exp")
      result = width <= 32 ? fromFloat(std::exp(toFloat(a)))
                           : fromDouble(std::exp(toDouble(a)));
    else if (opName_ == "math.log2")
      result = width <= 32 ? fromFloat(std::log2(toFloat(a)))
                           : fromDouble(std::log2(toDouble(a)));
    else if (opName_ == "math.sin")
      result = width <= 32 ? fromFloat(std::sin(toFloat(a)))
                           : fromDouble(std::sin(toDouble(a)));
    else if (opName_ == "math.sqrt")
      result = width <= 32 ? fromFloat(std::sqrt(toFloat(a)))
                           : fromDouble(std::sqrt(toDouble(a)));
    else if (opName_ == "math.fma") {
      uint64_t c = getInput(2);
      result = width <= 32
                   ? fromFloat(std::fma(toFloat(a), toFloat(b), toFloat(c)))
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

void FunctionUnitModule::evaluateFireReadinessForStrictInputs() {
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

void FunctionUnitModule::evaluateMuxReadiness() {
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

void FunctionUnitModule::evaluateLoadReadiness() {
  setAllInputReady(false);
  if (inputs.size() < 3 || outputs.size() < 2)
    return;

  bool canCaptureAddr = !loadAddrToken_.has_value();
  bool canCaptureData = !loadDataToken_.has_value();
  bool canCaptureCtrl = !loadCtrlToken_.has_value();
  setInputReadyFreshAware(0, canCaptureAddr);
  setInputReadyFreshAware(1, canCaptureData);
  setInputReadyFreshAware(2, canCaptureCtrl);

  bool issueBlocked = loadAddrToken_.has_value() && loadCtrlToken_.has_value() &&
                      outputRegisters_[1].has_value();
  bool returnBlocked =
      loadDataToken_.has_value() && outputRegisters_[0].has_value();
  if ((inputs[0]->valid && !inputAlreadyConsumed(0) && !canCaptureAddr) ||
      (inputs[1]->valid && !inputAlreadyConsumed(1) && !canCaptureData) ||
      (inputs[2]->valid && !inputAlreadyConsumed(2) && !canCaptureCtrl) ||
      issueBlocked || returnBlocked)
    perf_.stallCyclesOut++;
}

void FunctionUnitModule::evaluateStoreReadiness() {
  setAllInputReady(false);
  if (inputs.size() < 3 || outputs.size() < 2)
    return;
  bool canCaptureAddr = !storeAddrToken_.has_value();
  bool canCaptureData = !storeDataToken_.has_value();
  bool canCaptureCtrl = !storeCtrlToken_.has_value();

  setInputReadyFreshAware(0, canCaptureAddr);
  setInputReadyFreshAware(1, canCaptureData);
  setInputReadyFreshAware(2, canCaptureCtrl);

  bool issueBlocked = storeAddrToken_.has_value() && storeDataToken_.has_value() &&
                      storeCtrlToken_.has_value() &&
                      (outputRegisters_[0].has_value() ||
                       outputRegisters_[1].has_value());
  if ((inputs[0]->valid && !inputAlreadyConsumed(0) && !canCaptureAddr) ||
      (inputs[1]->valid && !inputAlreadyConsumed(1) && !canCaptureData) ||
      (inputs[2]->valid && !inputAlreadyConsumed(2) && !canCaptureCtrl) ||
      issueBlocked)
    perf_.stallCyclesOut++;
}

void FunctionUnitModule::evaluateCombinationalBody(bool readyForNewFire) {
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
    bool operandsValid = idx < inputs.size() && inputFresh(0) && inputFresh(idx);
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

void FunctionUnitModule::commitZeroLatencyBody() {
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

bool FunctionUnitModule::canFireCurrentBody() const {
  switch (bodyType_) {
  case BodyType::Constant:
    return inputs.empty() ? true : inputFresh(0) && inputs.front()->ready;
  case BodyType::Load:
    return loadIssueSelected_ || loadReturnSelected_;
  case BodyType::Store:
    return storeIssueSelected_;
  case BodyType::HandshakeMux:
    return muxSelectorToken_.has_value() && muxSelectedDataToken_.has_value() &&
           outputRegisterFree(0) && inflight_.empty() &&
           cyclesSinceLastFire_ >= interval_;
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

void FunctionUnitModule::consumeInputsForCurrentBody() {
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

unsigned FunctionUnitModule::countConsumedInputsForCurrentBody() const {
  switch (bodyType_) {
  case BodyType::Constant:
    return inputs.empty() ? 0u : 1u;
  case BodyType::HandshakeMux:
    return 2u;
  case BodyType::Load:
    return (loadIssueSelected_ ? 2u : 0u) + (loadReturnSelected_ ? 1u : 0u);
  case BodyType::Store:
    return storeIssueSelected_ ? 3u : 0u;
  case BodyType::Join:
    [[fallthrough]];
  default:
    return countFreshInputsForCurrentBody();
  }
}

void FunctionUnitModule::evaluateStatefulBody() {
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

void FunctionUnitModule::commitStatefulBody() {
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

bool FunctionUnitModule::bodyUsesGenericOperandLatches() const {
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

bool FunctionUnitModule::inputRequiredForCurrentBody(size_t idx) const {
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

bool FunctionUnitModule::inputAvailableForCurrentBody(size_t idx) const {
  if (!inputRequiredForCurrentBody(idx) || idx >= inputs.size())
    return true;
  if (idx < operandTokens_.size() && operandTokens_[idx].has_value())
    return true;
  return inputFresh(idx) && inputs[idx]->ready;
}

unsigned FunctionUnitModule::countFreshInputsForCurrentBody() const {
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

unsigned FunctionUnitModule::countLatchedOperands() const {
  unsigned count = 0;
  for (const auto &token : operandTokens_) {
    if (token.has_value())
      ++count;
  }
  return count;
}

} // namespace sim
} // namespace loom
