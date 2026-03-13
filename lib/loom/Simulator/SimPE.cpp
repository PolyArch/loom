//===-- SimPE.cpp - Simulated fabric.pe ----------------------------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include "loom/Simulator/SimPE.h"

#include <cmath>

namespace loom {
namespace sim {

SimPE::SimPE(BodyType bodyType, unsigned numInputs, unsigned numOutputs,
             bool isTagged, unsigned tagWidth, unsigned dataWidth,
             const std::string &opcodeStr, TagMode tagMode)
    : bodyType_(bodyType), isTagged_(isTagged), tagWidth_(tagWidth),
      dataWidth_(dataWidth), opcodeStr_(opcodeStr), tagMode_(tagMode) {
  outputTags_.resize(numOutputs, 0);
}

void SimPE::reset() {
  // Only clear runtime state, NOT configured state.
  // constantValue_, cmpPredicate_, contCondSel_, outputTags_ are set by configure().
  errorValid_ = false;
  errorCode_ = RtError::OK;
  perf_ = PerfSnapshot();
}

void SimPE::configure(const std::vector<uint32_t> &configWords) {
  if (configWords.empty())
    return;

  unsigned bitPos = 0;
  auto extractBits = [&](unsigned width) -> uint64_t {
    uint64_t val = 0;
    for (unsigned b = 0; b < width; ++b) {
      unsigned wordIdx = bitPos / 32;
      unsigned bit = bitPos % 32;
      if (wordIdx < configWords.size()) {
        if (configWords[wordIdx] & (1u << bit))
          val |= (1ULL << b);
      }
      ++bitPos;
    }
    return val;
  };

  if (bodyType_ == BodyType::Constant) {
    // Constant PE: value + optional output tag.
    constantValue_ = extractBits(dataWidth_);
    if (isTagged_ && tagWidth_ > 0)
      outputTags_[0] = static_cast<uint16_t>(extractBits(tagWidth_));
  } else if (bodyType_ == BodyType::StreamCont) {
    // Stream: cont_cond_sel (5 bits) + optional output tags.
    contCondSel_ = static_cast<uint8_t>(extractBits(5));
    // Validate one-hot.
    unsigned popcount = 0;
    for (unsigned b = 0; b < 5; ++b)
      if (contCondSel_ & (1u << b))
        ++popcount;
    if (popcount != 1)
      latchError(RtError::CFG_PE_STREAM_CONT_COND_ONEHOT);
    if (isTagged_) {
      for (unsigned o = 0; o < outputTags_.size(); ++o)
        outputTags_[o] = static_cast<uint16_t>(extractBits(tagWidth_));
    }
  } else {
    // Compute/Load/Store: output tags + optional compare predicates.
    if (isTagged_) {
      for (unsigned o = 0; o < outputTags_.size(); ++o)
        outputTags_[o] = static_cast<uint16_t>(extractBits(tagWidth_));
    }
    // Compare predicates (4 bits each) come after output tags.
    if (opcodeStr_.find("cmpi") != std::string::npos) {
      cmpPredicate_ = static_cast<uint8_t>(extractBits(4));
      if (cmpPredicate_ >= 10)
        latchError(RtError::CFG_PE_CMPI_PREDICATE_INVALID);
    } else if (opcodeStr_.find("cmpf") != std::string::npos) {
      // cmpf allows all 16 predicate encodings (4-bit).
      cmpPredicate_ = static_cast<uint8_t>(extractBits(4));
    }
  }
}

int64_t SimPE::signExt(uint64_t v) const {
  if (dataWidth_ >= 64)
    return static_cast<int64_t>(v);
  if (v & (1ULL << (dataWidth_ - 1)))
    return static_cast<int64_t>(v | (~0ULL << dataWidth_));
  return static_cast<int64_t>(v);
}

uint64_t SimPE::maskToWidth(uint64_t v) const {
  if (dataWidth_ >= 64)
    return v;
  return v & ((1ULL << dataWidth_) - 1);
}

uint64_t SimPE::executeOp(uint64_t a, uint64_t b) const {
  // Arithmetic operations.
  if (opcodeStr_ == "arith.addi" || opcodeStr_ == "arith.addf")
    return maskToWidth(a + b);
  if (opcodeStr_ == "arith.subi" || opcodeStr_ == "arith.subf")
    return maskToWidth(a - b);
  if (opcodeStr_ == "arith.muli" || opcodeStr_ == "arith.mulf")
    return maskToWidth(a * b);
  if (opcodeStr_ == "arith.divsi")
    return b != 0 ? maskToWidth(static_cast<uint64_t>(signExt(a) / signExt(b))) : 0;
  if (opcodeStr_ == "arith.divui")
    return b != 0 ? maskToWidth(a / b) : 0;
  if (opcodeStr_ == "arith.remsi")
    return b != 0 ? maskToWidth(static_cast<uint64_t>(signExt(a) % signExt(b))) : 0;
  if (opcodeStr_ == "arith.remui")
    return b != 0 ? maskToWidth(a % b) : 0;

  // Bitwise operations.
  if (opcodeStr_ == "arith.andi")
    return maskToWidth(a & b);
  if (opcodeStr_ == "arith.ori")
    return maskToWidth(a | b);
  if (opcodeStr_ == "arith.xori")
    return maskToWidth(a ^ b);
  if (opcodeStr_ == "arith.shli")
    return maskToWidth(a << (b & 63));
  if (opcodeStr_ == "arith.shrsi")
    return maskToWidth(static_cast<uint64_t>(signExt(a) >> (b & 63)));
  if (opcodeStr_ == "arith.shrui")
    return maskToWidth(a >> (b & 63));

  // Unary / conversion operations.
  if (opcodeStr_ == "arith.extsi")
    return static_cast<uint64_t>(signExt(a));
  if (opcodeStr_ == "arith.extui" || opcodeStr_ == "arith.trunci")
    return maskToWidth(a);
  if (opcodeStr_ == "arith.index_cast" ||
      opcodeStr_ == "arith.index_castui")
    return maskToWidth(a);

  // Select.
  if (opcodeStr_ == "arith.select")
    return (a & 1) ? b : (inputs.size() > 2 ? inputs[2]->data : 0);

  // Compare.
  if (opcodeStr_ == "arith.cmpi" || opcodeStr_ == "arith.cmpf")
    return evaluateCmp(a, b) ? 1 : 0;

  // Math operations.
  if (opcodeStr_ == "math.absi")
    return maskToWidth(static_cast<uint64_t>(std::abs(signExt(a))));
  if (opcodeStr_ == "arith.maxsi")
    return signExt(a) >= signExt(b) ? maskToWidth(a) : maskToWidth(b);
  if (opcodeStr_ == "arith.minsi")
    return signExt(a) <= signExt(b) ? maskToWidth(a) : maskToWidth(b);
  if (opcodeStr_ == "arith.maxui")
    return a >= b ? maskToWidth(a) : maskToWidth(b);
  if (opcodeStr_ == "arith.minui")
    return a <= b ? maskToWidth(a) : maskToWidth(b);

  // Default: pass through first operand.
  return maskToWidth(a);
}

bool SimPE::evaluateCmp(uint64_t a, uint64_t b) const {
  switch (cmpPredicate_) {
  case 0: return a == b;                    // eq
  case 1: return a != b;                    // ne
  case 2: return signExt(a) < signExt(b);   // slt
  case 3: return signExt(a) <= signExt(b);  // sle
  case 4: return signExt(a) > signExt(b);   // sgt
  case 5: return signExt(a) >= signExt(b);  // sge
  case 6: return a < b;                     // ult
  case 7: return a <= b;                    // ule
  case 8: return a > b;                     // ugt
  case 9: return a >= b;                    // uge
  default: return false;
  }
}

void SimPE::evaluateCombinational() {
  // All PE types are combinational (single-cycle computation).
  // Valid output when all required inputs are valid.

  if (bodyType_ == BodyType::Constant) {
    // Constant PE: always valid, no inputs needed.
    if (!outputs.empty()) {
      outputs[0]->valid = true;
      outputs[0]->data = constantValue_;
      if (isTagged_)
        outputs[0]->tag = outputTags_[0];
      outputs[0]->hasTag = isTagged_;
    }
    // No inputs to set ready for.
    return;
  }

  // Check all inputs are valid.
  bool allValid = true;
  for (auto *in : inputs) {
    if (!in->valid) {
      allValid = false;
      break;
    }
  }

  if (!allValid) {
    for (auto *out : outputs)
      out->valid = false;
    // Don't accept any inputs if not all are ready.
    for (auto *in : inputs)
      in->ready = false;
    return;
  }

  // All inputs valid - compute result.
  uint64_t result = 0;
  uint64_t operandA = !inputs.empty() ? inputs[0]->data : 0;
  uint64_t operandB = inputs.size() > 1 ? inputs[1]->data : 0;

  switch (bodyType_) {
  case BodyType::Compute:
    result = executeOp(operandA, operandB);
    break;

  case BodyType::Gate:
    // Input 0 = data, Input 1 = condition (1-bit).
    if (inputs.size() <= 1 || !(inputs[1]->data & 1)) {
      for (auto *out : outputs)
        out->valid = false;
      for (auto *in : inputs)
        in->ready = false;
      return;
    }
    result = operandA;
    break;

  default:
    // Load, Store, StreamCont, Carry, Invariant: pass through first operand.
    result = operandA;
    break;
  }

  // Drive outputs.
  for (unsigned o = 0; o < outputs.size(); ++o) {
    outputs[o]->valid = true;
    outputs[o]->data = result;
    if (isTagged_) {
      if (tagMode_ == TagMode::TagTransparent && !inputs.empty())
        outputs[o]->tag = inputs[0]->tag;
      else if (tagWidth_ > 0)
        outputs[o]->tag = outputTags_[o];
      outputs[o]->hasTag = true;
    } else {
      outputs[o]->hasTag = false;
    }
  }

  // Set input ready: all outputs must be ready (broadcast semantics).
  bool allOutReady = true;
  for (auto *out : outputs) {
    if (!out->ready)
      allOutReady = false;
  }
  for (auto *in : inputs)
    in->ready = allOutReady;
}

void SimPE::collectTraceEvents(std::vector<TraceEvent> &events,
                               uint64_t cycle) {
  bool fired = false;
  for (auto *out : outputs) {
    if (out->transferred()) {
      fired = true;
      break;
    }
  }

  if (fired) {
    perf_.activeCycles++;
    for (auto *out : outputs) {
      if (out->transferred())
        perf_.tokensOut++;
    }
    for (auto *in : inputs) {
      if (in->transferred())
        perf_.tokensIn++;
    }
    TraceEvent ev;
    ev.cycle = cycle;
    ev.hwNodeId = hwNodeId;
    ev.eventKind = EV_NODE_FIRE;
    events.push_back(ev);
  } else {
    // Stall analysis.
    bool anyInputStall = false;
    bool anyOutputStall = false;
    for (auto *in : inputs) {
      if (!in->valid) {
        anyInputStall = true;
        break;
      }
    }
    for (auto *out : outputs) {
      if (out->valid && !out->ready) {
        anyOutputStall = true;
        break;
      }
    }
    if (anyOutputStall)
      perf_.stallCyclesOut++;
    else if (anyInputStall)
      perf_.stallCyclesIn++;
  }

  if (errorValid_) {
    TraceEvent ev;
    ev.cycle = cycle;
    ev.hwNodeId = hwNodeId;
    ev.eventKind = EV_DEVICE_ERROR;
    ev.arg0 = errorCode_;
    events.push_back(ev);
  }
}

} // namespace sim
} // namespace loom
