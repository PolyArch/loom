//===-- SimPE.cpp - Simulated fabric.pe ----------------------------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include "loom/Simulator/SimPE.h"

#include <cmath>
#include <cstring>

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
  errorValid_ = false;
  errorCode_ = RtError::OK;
  pendingError_ = false;
  pendingErrorCode_ = RtError::OK;
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
    // Stream: output tags first, then cont_cond_sel (5 bits).
    if (isTagged_) {
      for (unsigned o = 0; o < outputTags_.size(); ++o)
        outputTags_[o] = static_cast<uint16_t>(extractBits(tagWidth_));
    }
    contCondSel_ = static_cast<uint8_t>(extractBits(5));
    // Validate one-hot.
    unsigned popcount = 0;
    for (unsigned b = 0; b < 5; ++b)
      if (contCondSel_ & (1u << b))
        ++popcount;
    if (popcount != 1)
      latchError(RtError::CFG_PE_STREAM_CONT_COND_ONEHOT);
  } else {
    // Compute/Load/Store/Carry/Invariant/Gate:
    // Per spec-fabric-config_mem.md: output tags first, then cmp predicates.
    if (isTagged_) {
      if (bodyType_ == BodyType::Load || bodyType_ == BodyType::Store) {
        // Load/Store TagOverwrite: single shared output_tag for all outputs.
        if (tagMode_ == TagMode::TagOverwrite && tagWidth_ > 0) {
          uint16_t sharedTag = static_cast<uint16_t>(extractBits(tagWidth_));
          for (auto &t : outputTags_)
            t = sharedTag;
        }
      } else {
        for (unsigned o = 0; o < outputTags_.size(); ++o)
          outputTags_[o] = static_cast<uint16_t>(extractBits(tagWidth_));
      }
    }
    // Compare predicates (4 bits each) come after output tags.
    if (opcodeStr_ == "arith.cmpi") {
      cmpPredicate_ = static_cast<uint8_t>(extractBits(4));
      if (cmpPredicate_ >= 10)
        latchError(RtError::CFG_PE_CMPI_PREDICATE_INVALID);
    } else if (opcodeStr_ == "arith.cmpf") {
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

float SimPE::toFloat(uint64_t v) const {
  float f;
  uint32_t u = static_cast<uint32_t>(v);
  std::memcpy(&f, &u, sizeof(f));
  return f;
}

uint64_t SimPE::fromFloat(float f) const {
  uint32_t u;
  std::memcpy(&u, &f, sizeof(u));
  return u;
}

double SimPE::toDouble(uint64_t v) const {
  double d;
  std::memcpy(&d, &v, sizeof(d));
  return d;
}

uint64_t SimPE::fromDouble(double d) const {
  uint64_t u;
  std::memcpy(&u, &d, sizeof(u));
  return u;
}

uint64_t SimPE::executeOp(uint64_t a, uint64_t b) const {
  //--- Integer arith ops ---
  if (opcodeStr_ == "arith.addi")
    return maskToWidth(a + b);
  if (opcodeStr_ == "arith.subi")
    return maskToWidth(a - b);
  if (opcodeStr_ == "arith.muli")
    return maskToWidth(a * b);
  if (opcodeStr_ == "arith.divsi")
    return b != 0
               ? maskToWidth(
                     static_cast<uint64_t>(signExt(a) / signExt(b)))
               : 0;
  if (opcodeStr_ == "arith.divui")
    return b != 0 ? maskToWidth(a / b) : 0;
  if (opcodeStr_ == "arith.remsi")
    return b != 0
               ? maskToWidth(
                     static_cast<uint64_t>(signExt(a) % signExt(b)))
               : 0;
  if (opcodeStr_ == "arith.remui")
    return b != 0 ? maskToWidth(a % b) : 0;

  // Bitwise ops.
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

  // Extension/truncation ops.
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

  // Integer compare.
  if (opcodeStr_ == "arith.cmpi")
    return evaluateCmpi(a, b) ? 1 : 0;

  //--- Floating-point arith ops (bit-cast through float/double) ---
  if (opcodeStr_ == "arith.addf") {
    if (dataWidth_ <= 32)
      return fromFloat(toFloat(a) + toFloat(b));
    return fromDouble(toDouble(a) + toDouble(b));
  }
  if (opcodeStr_ == "arith.subf") {
    if (dataWidth_ <= 32)
      return fromFloat(toFloat(a) - toFloat(b));
    return fromDouble(toDouble(a) - toDouble(b));
  }
  if (opcodeStr_ == "arith.mulf") {
    if (dataWidth_ <= 32)
      return fromFloat(toFloat(a) * toFloat(b));
    return fromDouble(toDouble(a) * toDouble(b));
  }
  if (opcodeStr_ == "arith.divf") {
    if (dataWidth_ <= 32)
      return fromFloat(toFloat(a) / toFloat(b));
    return fromDouble(toDouble(a) / toDouble(b));
  }
  if (opcodeStr_ == "arith.negf") {
    if (dataWidth_ <= 32)
      return fromFloat(-toFloat(a));
    return fromDouble(-toDouble(a));
  }

  // Float compare.
  if (opcodeStr_ == "arith.cmpf")
    return evaluateCmpf(a, b) ? 1 : 0;

  // Float-int conversion ops.
  if (opcodeStr_ == "arith.fptosi") {
    if (dataWidth_ <= 32)
      return maskToWidth(
          static_cast<uint64_t>(static_cast<int64_t>(toFloat(a))));
    return static_cast<uint64_t>(static_cast<int64_t>(toDouble(a)));
  }
  if (opcodeStr_ == "arith.fptoui") {
    if (dataWidth_ <= 32)
      return maskToWidth(static_cast<uint64_t>(toFloat(a)));
    return static_cast<uint64_t>(toDouble(a));
  }
  if (opcodeStr_ == "arith.sitofp") {
    if (dataWidth_ <= 32)
      return fromFloat(static_cast<float>(signExt(a)));
    return fromDouble(static_cast<double>(signExt(a)));
  }
  if (opcodeStr_ == "arith.uitofp") {
    if (dataWidth_ <= 32)
      return fromFloat(static_cast<float>(a));
    return fromDouble(static_cast<double>(a));
  }

  //--- Math ops ---
  if (opcodeStr_ == "math.absf") {
    if (dataWidth_ <= 32)
      return fromFloat(std::fabs(toFloat(a)));
    return fromDouble(std::fabs(toDouble(a)));
  }
  if (opcodeStr_ == "math.cos") {
    if (dataWidth_ <= 32)
      return fromFloat(std::cos(toFloat(a)));
    return fromDouble(std::cos(toDouble(a)));
  }
  if (opcodeStr_ == "math.exp") {
    if (dataWidth_ <= 32)
      return fromFloat(std::exp(toFloat(a)));
    return fromDouble(std::exp(toDouble(a)));
  }
  if (opcodeStr_ == "math.fma") {
    // Fused multiply-add: a * b + c. Third operand from input[2].
    uint64_t c = inputs.size() > 2 ? inputs[2]->data : 0;
    if (dataWidth_ <= 32)
      return fromFloat(std::fma(toFloat(a), toFloat(b), toFloat(c)));
    return fromDouble(std::fma(toDouble(a), toDouble(b), toDouble(c)));
  }
  if (opcodeStr_ == "math.log2") {
    if (dataWidth_ <= 32)
      return fromFloat(std::log2(toFloat(a)));
    return fromDouble(std::log2(toDouble(a)));
  }
  if (opcodeStr_ == "math.sin") {
    if (dataWidth_ <= 32)
      return fromFloat(std::sin(toFloat(a)));
    return fromDouble(std::sin(toDouble(a)));
  }
  if (opcodeStr_ == "math.sqrt") {
    if (dataWidth_ <= 32)
      return fromFloat(std::sqrt(toFloat(a)));
    return fromDouble(std::sqrt(toDouble(a)));
  }

  //--- LLVM dialect ---
  if (opcodeStr_ == "llvm.intr.bitreverse") {
    // Bit reversal within dataWidth_ bits.
    uint64_t v = a;
    uint64_t result = 0;
    for (unsigned i = 0; i < dataWidth_; ++i) {
      if (v & (1ULL << i))
        result |= (1ULL << (dataWidth_ - 1 - i));
    }
    return result;
  }

  // Unknown op: pass through first operand. Hardware-level op validation is
  // a compile-time check, not a runtime error per spec-fabric-error.md.
  return maskToWidth(a);
}

bool SimPE::evaluateCmpi(uint64_t a, uint64_t b) const {
  // MLIR arith.cmpi predicate encoding (10 valid predicates).
  switch (cmpPredicate_) {
  case 0:
    return a == b; // eq
  case 1:
    return a != b; // ne
  case 2:
    return signExt(a) < signExt(b); // slt
  case 3:
    return signExt(a) <= signExt(b); // sle
  case 4:
    return signExt(a) > signExt(b); // sgt
  case 5:
    return signExt(a) >= signExt(b); // sge
  case 6:
    return a < b; // ult
  case 7:
    return a <= b; // ule
  case 8:
    return a > b; // ugt
  case 9:
    return a >= b; // uge
  default:
    return false;
  }
}

bool SimPE::evaluateCmpf(uint64_t a, uint64_t b) const {
  // MLIR arith.cmpf IEEE 754 predicate encoding (16 predicates).
  bool aIsNaN, bIsNaN;
  double fa, fb;
  if (dataWidth_ <= 32) {
    fa = static_cast<double>(toFloat(a));
    fb = static_cast<double>(toFloat(b));
    aIsNaN = std::isnan(toFloat(a));
    bIsNaN = std::isnan(toFloat(b));
  } else {
    fa = toDouble(a);
    fb = toDouble(b);
    aIsNaN = std::isnan(fa);
    bIsNaN = std::isnan(fb);
  }

  bool unordered = aIsNaN || bIsNaN;
  bool ordered = !unordered;

  switch (cmpPredicate_) {
  case 0:
    return false; // false (always false)
  case 1:
    return ordered && (fa == fb); // oeq
  case 2:
    return ordered && (fa > fb); // ogt
  case 3:
    return ordered && (fa >= fb); // oge
  case 4:
    return ordered && (fa < fb); // olt
  case 5:
    return ordered && (fa <= fb); // ole
  case 6:
    return ordered && (fa != fb); // one
  case 7:
    return ordered; // ord
  case 8:
    return unordered || (fa == fb); // ueq
  case 9:
    return unordered || (fa > fb); // ugt
  case 10:
    return unordered || (fa >= fb); // uge
  case 11:
    return unordered || (fa < fb); // ult
  case 12:
    return unordered || (fa <= fb); // ule
  case 13:
    return unordered || (fa != fb); // une
  case 14:
    return unordered; // uno
  case 15:
    return true; // true (always true)
  default:
    return false;
  }
}

void SimPE::evaluateCombinational() {
  if (bodyType_ == BodyType::Constant) {
    // Constant PE: always valid, no inputs needed.
    if (!outputs.empty()) {
      outputs[0]->valid = true;
      outputs[0]->data = constantValue_;
      if (isTagged_)
        outputs[0]->tag = outputTags_[0];
      outputs[0]->hasTag = isTagged_;
    }
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

  case BodyType::StreamCont: {
    // dataflow.stream: compare operandA against operandB using cont_cond_sel.
    // cont_cond_sel is 5-bit one-hot: [<, <=, >, >=, !=].
    bool cont = false;
    int64_t sa = signExt(operandA);
    int64_t sb = signExt(operandB);
    if (contCondSel_ & 0x01)
      cont = (sa < sb);
    else if (contCondSel_ & 0x02)
      cont = (sa <= sb);
    else if (contCondSel_ & 0x04)
      cont = (sa > sb);
    else if (contCondSel_ & 0x08)
      cont = (sa >= sb);
    else if (contCondSel_ & 0x10)
      cont = (sa != sb);
    result = cont ? 1 : 0;
    break;
  }

  default:
    // Load, Store, Carry, Invariant: pass through first operand.
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
