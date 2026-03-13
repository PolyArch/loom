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
  // Reset dataflow state machines.
  dataflowInitialStage_ = true;
  invariantStoredValue_ = 0;
  gateFirstElement_ = true;
  gateBufferedValue_ = 0;
  streamInitialPhase_ = true;
  streamNextIdx_ = 0;
  streamBoundReg_ = 0;
  streamStepReg_ = 0;
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
    // Constant PE: always outputs the configured constant value.
    if (!outputs.empty()) {
      outputs[0]->valid = true;
      outputs[0]->data = constantValue_;
      if (isTagged_)
        outputs[0]->tag = outputTags_[0];
      outputs[0]->hasTag = isTagged_;
    }
    // Consume trigger input (if present) when output is ready.
    if (!inputs.empty()) {
      bool outReady = outputs.empty() || outputs[0]->ready;
      inputs[0]->ready = outReady;
    }
    return;
  }

  // Dispatch to specialized state-machine evaluators for dataflow ops.
  if (bodyType_ == BodyType::Carry) {
    evaluateCarry();
    return;
  }
  if (bodyType_ == BodyType::Invariant) {
    evaluateInvariant();
    return;
  }
  if (bodyType_ == BodyType::Gate) {
    evaluateGate();
    return;
  }
  if (bodyType_ == BodyType::StreamCont) {
    evaluateStream();
    return;
  }

  // Dispatch to handshake control evaluators.
  if (bodyType_ == BodyType::CondBranch) {
    evaluateCondBranch();
    return;
  }
  if (bodyType_ == BodyType::Mux) {
    evaluateMux();
    return;
  }
  if (bodyType_ == BodyType::Join) {
    evaluateJoin();
    return;
  }
  if (bodyType_ == BodyType::Sink) {
    evaluateSink();
    return;
  }

  // Load PE: two independent data paths per spec-fabric-pe.md.
  if (bodyType_ == BodyType::Load) {
    evaluateLoad();
    return;
  }
  // Store PE: all-input sync with correct per-output data mapping.
  if (bodyType_ == BodyType::Store) {
    evaluateStore();
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

  default:
    // Load, Store: pass through first operand.
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

//===----------------------------------------------------------------------===//
// Load PE: two independent data paths per spec-fabric-pe.md and
// fabric_pe_load.sv (TagOverwrite mode).
//
// Ports:
//   in0 = addr_from_comp, in1 = data_from_mem, in2 = ctrl
//   out0 = data_to_comp,  out1 = addr_to_mem
//
// Path 1 (addr + ctrl sync → addr_to_mem):
//   out1_valid = in0_valid && in2_valid
//   fire = sync_valid && out1_ready
//   in0_ready = in2_ready = fire
//
// Path 2 (data passthrough, independent):
//   out0_valid = in1_valid
//   in1_ready  = out0_ready
//===----------------------------------------------------------------------===//

void SimPE::evaluateLoad() {
  if (inputs.size() < 3 || outputs.size() < 2)
    return;

  SimChannel *addrIn = inputs[0];  // addr_from_comp
  SimChannel *dataIn = inputs[1];  // data_from_mem
  SimChannel *ctrlIn = inputs[2];  // ctrl
  SimChannel *dataOut = outputs[0]; // data_to_comp
  SimChannel *addrOut = outputs[1]; // addr_to_mem

  // Path 1: addr + ctrl → addr_to_mem
  bool addrCtrlSync = addrIn->valid && ctrlIn->valid;
  addrOut->valid = addrCtrlSync;
  addrOut->data = addrIn->data;
  driveOutputTag(addrOut, 0);

  bool addrFire = addrCtrlSync && addrOut->ready;
  addrIn->ready = addrFire;
  ctrlIn->ready = addrFire;

  // Path 2: data_from_mem → data_to_comp (independent)
  dataOut->valid = dataIn->valid;
  dataOut->data = dataIn->data;
  driveOutputTag(dataOut, 1);

  dataIn->ready = dataOut->ready;
}

//===----------------------------------------------------------------------===//
// Store PE: all three inputs must synchronize before producing outputs.
// Per spec-fabric-pe.md: store fires when addr, data, AND ctrl are all ready.
//
// Ports:
//   in0 = addr_from_comp, in1 = data_from_comp, in2 = ctrl
//   out0 = addr_to_mem,   out1 = data_to_mem
//===----------------------------------------------------------------------===//

void SimPE::evaluateStore() {
  if (inputs.size() < 3 || outputs.size() < 2)
    return;

  SimChannel *addrIn = inputs[0];  // addr_from_comp
  SimChannel *dataIn = inputs[1];  // data_from_comp
  SimChannel *ctrlIn = inputs[2];  // ctrl
  SimChannel *addrOut = outputs[0]; // addr_to_mem
  SimChannel *dataOut = outputs[1]; // data_to_mem

  bool allSync = addrIn->valid && dataIn->valid && ctrlIn->valid;

  addrOut->valid = allSync;
  addrOut->data = addrIn->data;
  driveOutputTag(addrOut, 0);

  dataOut->valid = allSync;
  dataOut->data = dataIn->data;
  driveOutputTag(dataOut, 1);

  bool allOutReady = addrOut->ready && dataOut->ready;
  bool fire = allSync && allOutReady;
  addrIn->ready = fire;
  dataIn->ready = fire;
  ctrlIn->ready = fire;
}

//===----------------------------------------------------------------------===//
// dataflow.carry state machine per spec-dataflow.md
//   Inputs: [0]=%d (i1 ctrl), [1]=%a (initial), [2]=%b (loop-carried)
//   Output: [0]=%o
//===----------------------------------------------------------------------===//

void SimPE::evaluateCarry() {
  if (outputs.empty())
    return;

  auto *out = outputs[0];

  if (dataflowInitialStage_) {
    // Initial stage: consume one element from %a, emit on %o.
    // Do NOT consume %d in this stage.
    if (inputs.size() <= 1 || !inputs[1]->valid) {
      out->valid = false;
      for (auto *in : inputs)
        in->ready = false;
      return;
    }
    out->valid = true;
    out->data = inputs[1]->data;
    driveOutputTag(out, 1);
    // Ready only on %a (input[1]), not on %d or %b.
    inputs[0]->ready = false;
    inputs[1]->ready = out->ready;
    if (inputs.size() > 2)
      inputs[2]->ready = false;
  } else {
    // Block stage: consume %d. If true, also consume %b and emit.
    if (inputs.empty() || !inputs[0]->valid) {
      out->valid = false;
      for (auto *in : inputs)
        in->ready = false;
      return;
    }
    bool dVal = (inputs[0]->data & 1) != 0;
    if (dVal) {
      // d=true: need %b valid. Emit %b on output.
      if (inputs.size() <= 2 || !inputs[2]->valid) {
        out->valid = false;
        for (auto *in : inputs)
          in->ready = false;
        return;
      }
      out->valid = true;
      out->data = inputs[2]->data;
      driveOutputTag(out, 2);
      inputs[0]->ready = out->ready;
      inputs[1]->ready = false;
      inputs[2]->ready = out->ready;
    } else {
      // d=false: consume only %d, no output. Transition handled in
      // advanceClock.
      out->valid = false;
      inputs[0]->ready = true;
      inputs[1]->ready = false;
      if (inputs.size() > 2)
        inputs[2]->ready = false;
    }
  }
}

//===----------------------------------------------------------------------===//
// dataflow.invariant state machine per spec-dataflow.md
//   Inputs: [0]=%d (i1 ctrl), [1]=%a (invariant value)
//   Output: [0]=%o
//===----------------------------------------------------------------------===//

void SimPE::evaluateInvariant() {
  if (outputs.empty())
    return;

  auto *out = outputs[0];

  if (dataflowInitialStage_) {
    // Initial stage: consume one element from %a, emit on %o, store value.
    if (inputs.size() <= 1 || !inputs[1]->valid) {
      out->valid = false;
      for (auto *in : inputs)
        in->ready = false;
      return;
    }
    out->valid = true;
    out->data = inputs[1]->data;
    driveOutputTag(out, 1);
    inputs[0]->ready = false;
    inputs[1]->ready = out->ready;
  } else {
    // Block stage: consume %d. If true, emit stored value.
    if (inputs.empty() || !inputs[0]->valid) {
      out->valid = false;
      for (auto *in : inputs)
        in->ready = false;
      return;
    }
    bool dVal = (inputs[0]->data & 1) != 0;
    if (dVal) {
      out->valid = true;
      out->data = invariantStoredValue_;
      driveOutputTag(out, -1); // Use configured output tag.
      inputs[0]->ready = out->ready;
      if (inputs.size() > 1)
        inputs[1]->ready = false;
    } else {
      // d=false: consume only %d, no output. Transition in advanceClock.
      out->valid = false;
      inputs[0]->ready = true;
      if (inputs.size() > 1)
        inputs[1]->ready = false;
    }
  }
}

//===----------------------------------------------------------------------===//
// dataflow.gate state machine per spec-dataflow.md
//   Inputs: [0]=before_value, [1]=before_cond (i1)
//   Outputs: [0]=after_value, [1]=after_cond (i1)
//   Semantics: after_value[i] = before_value[i],
//              after_cond[i] = before_cond[i+1]
//===----------------------------------------------------------------------===//

void SimPE::evaluateGate() {
  if (outputs.size() < 2 || inputs.size() < 2)
    return;

  auto *outVal = outputs[0];
  auto *outCond = outputs[1];

  if (!inputs[0]->valid || !inputs[1]->valid) {
    outVal->valid = false;
    outCond->valid = false;
    inputs[0]->ready = false;
    inputs[1]->ready = false;
    return;
  }

  // Pass-through mode: emit before_value on output[0], discard afterCond
  // on output[1].  The compiler IR uses willContinue (N+1 tokens) directly
  // for carry/cond_br rather than afterCond, so afterCond is dead in the
  // DFG.  The mapper may not route output[1] through the switch network,
  // leaving its ready signal permanently false.  Discarding it here avoids
  // deadlock on the unrouted output.
  outVal->valid = true;
  outVal->data = inputs[0]->data;
  driveOutputTag(outVal, 0);
  outCond->valid = false; // afterCond discarded -- unused in IR
  bool accept = outVal->ready;
  inputs[0]->ready = accept;
  inputs[1]->ready = accept;
}

//===----------------------------------------------------------------------===//
// dataflow.stream state machine per spec-dataflow.md
//   Inputs: [0]=%start, [1]=%step, [2]=%bound
//   Outputs: [0]=%idx, [1]=%cont (i1)
//
//   Initial phase: wait for all 3 inputs, emit (start, willContinue).
//   Block phase: emit (nextIdxReg, willContinue), update nextIdxReg.
//===----------------------------------------------------------------------===//

void SimPE::evaluateStream() {
  if (outputs.size() < 2 || inputs.size() < 3)
    return;

  auto *outIdx = outputs[0];
  auto *outCont = outputs[1];

  // Helper: evaluate continuation condition using contCondSel_ (5-bit one-hot).
  auto evalCont = [this](int64_t idx, int64_t bound) -> bool {
    if (contCondSel_ & 0x01) return idx < bound;
    if (contCondSel_ & 0x02) return idx <= bound;
    if (contCondSel_ & 0x04) return idx > bound;
    if (contCondSel_ & 0x08) return idx >= bound;
    if (contCondSel_ & 0x10) return idx != bound;
    return false;
  };

  // Helper: compute next index using stepOp_ from fabric.pe definition.
  auto computeNext = [this](uint64_t idx, uint64_t step) -> uint64_t {
    if (stepOp_ == "-=")
      return maskToWidth(idx - step);
    if (stepOp_ == "*=")
      return maskToWidth(idx * step);
    if (stepOp_ == "/=")
      return step != 0 ? maskToWidth(idx / step) : 0;
    if (stepOp_ == "<<=")
      return maskToWidth(idx << (step & 63));
    if (stepOp_ == ">>=")
      return maskToWidth(idx >> (step & 63));
    // Default: += .
    return maskToWidth(idx + step);
  };

  if (streamInitialPhase_) {
    // Wait for all 3 inputs: start, step, bound.
    if (!inputs[0]->valid || !inputs[1]->valid || !inputs[2]->valid) {
      outIdx->valid = false;
      outCont->valid = false;
      for (auto *in : inputs)
        in->ready = false;
      return;
    }

    uint64_t start = inputs[0]->data;
    uint64_t step = inputs[1]->data;
    uint64_t bound = inputs[2]->data;

    // DEBUG: print stream initial phase values (only first call per bound)
    static uint64_t lastDbgBound = ~0ULL;
    static unsigned dbgCount = 0;
    if (bound != lastDbgBound) {
      fprintf(stderr, "[DBG-STREAM] hwNode=%u initial: start=%lu step=%lu "
              "bound=%lu contCondSel=0x%02x outIdx.ready=%d outCont.ready=%d\n",
              hwNodeId, start, step, bound, contCondSel_,
              outIdx->ready, outCont->ready);
      lastDbgBound = bound;
      dbgCount = 0;
    } else if (dbgCount < 3) {
      fprintf(stderr, "[DBG-STREAM] hwNode=%u initial(repeat): "
              "outIdx.ready=%d outCont.ready=%d\n",
              hwNodeId, outIdx->ready, outCont->ready);
      dbgCount++;
    }

    // Per spec-dataflow.md: step must be nonzero.
    if (step == 0) {
      latchError(RtError::RT_DATAFLOW_STREAM_ZERO_STEP);
      outIdx->valid = false;
      outCont->valid = false;
      for (auto *in : inputs)
        in->ready = false;
      return;
    }
    bool willContinue = evalCont(signExt(start), signExt(bound));

    outIdx->valid = true;
    outIdx->data = start;
    driveOutputTag(outIdx, 0);
    outCont->valid = true;
    outCont->data = willContinue ? 1 : 0;
    driveOutputTag(outCont, -1);

    // Accept inputs when both outputs are ready.
    bool accept = outIdx->ready && outCont->ready;
    for (auto *in : inputs)
      in->ready = accept;

    // Latch step/bound for block phase. State transition in advanceClock.
    // (streamNextIdx_, streamBoundReg_, streamStepReg_ are set in advanceClock
    // when the transfer actually occurs.)
  } else {
    // Block phase: emit nextIdxReg and willContinue.
    bool willContinue = evalCont(signExt(streamNextIdx_),
                                 signExt(streamBoundReg_));

    outIdx->valid = true;
    outIdx->data = streamNextIdx_;
    driveOutputTag(outIdx, -1);
    outCont->valid = true;
    outCont->data = willContinue ? 1 : 0;
    driveOutputTag(outCont, -1);

    // DEBUG: report block phase when stuck (output not ready)
    static uint64_t blockDbgCount = 0;
    static uint64_t lastBoundDbg = ~0ULL;
    if (streamBoundReg_ != lastBoundDbg) {
      fprintf(stderr, "[DBG-STREAM] hwNode=%u block: nextIdx=%lu bound=%lu "
              "cont=%d outIdx.r=%d outCont.r=%d\n",
              hwNodeId, streamNextIdx_, streamBoundReg_,
              willContinue ? 1 : 0, outIdx->ready, outCont->ready);
      lastBoundDbg = streamBoundReg_;
      blockDbgCount = 0;
    }
    if (!outIdx->ready || !outCont->ready) {
      if (blockDbgCount < 5) {
        fprintf(stderr, "[DBG-STREAM] hwNode=%u block STUCK: nextIdx=%lu "
                "bound=%lu cont=%d outIdx.r=%d outCont.r=%d\n",
                hwNodeId, streamNextIdx_, streamBoundReg_,
                willContinue ? 1 : 0, outIdx->ready, outCont->ready);
        blockDbgCount++;
      }
    }

    // No inputs consumed in block phase.
    for (auto *in : inputs)
      in->ready = false;
  }
}

//===----------------------------------------------------------------------===//
// handshake.cond_br: inputs[0] = condition (i1), inputs[1] = data
// If condition is true, drive outputs[0] (true branch).
// If condition is false, drive outputs[1] (false branch).
// The other output is invalid. Only consume inputs when the active output is
// ready.
//===----------------------------------------------------------------------===//

void SimPE::evaluateCondBranch() {
  if (inputs.size() < 2 || outputs.size() < 2) {
    for (auto *out : outputs)
      out->valid = false;
    for (auto *in : inputs)
      in->ready = false;
    return;
  }

  // Need both condition and data valid.
  if (!inputs[0]->valid || !inputs[1]->valid) {
    for (auto *out : outputs)
      out->valid = false;
    for (auto *in : inputs)
      in->ready = false;
    return;
  }

  bool cond = (inputs[0]->data & 1) != 0;
  unsigned activeOut = cond ? 0 : 1;
  unsigned inactiveOut = cond ? 1 : 0;

  outputs[activeOut]->valid = true;
  outputs[activeOut]->data = inputs[1]->data;
  driveOutputTag(outputs[activeOut], 1);
  outputs[inactiveOut]->valid = false;

  // Consume inputs when the active output is accepted.
  bool accepted = outputs[activeOut]->ready;
  inputs[0]->ready = accepted;
  inputs[1]->ready = accepted;
}

//===----------------------------------------------------------------------===//
// handshake.mux: inputs[0] = select (index), inputs[1..N] = data alternatives
// Output = data[select]. Only the select and selected data input need to be
// valid. Consume only those two inputs.
//===----------------------------------------------------------------------===//

void SimPE::evaluateMux() {
  if (inputs.empty() || outputs.empty()) {
    for (auto *out : outputs)
      out->valid = false;
    for (auto *in : inputs)
      in->ready = false;
    return;
  }

  // Need select input valid.
  if (!inputs[0]->valid) {
    for (auto *out : outputs)
      out->valid = false;
    for (auto *in : inputs)
      in->ready = false;
    return;
  }

  unsigned sel = static_cast<unsigned>(inputs[0]->data);
  unsigned dataIdx = sel + 1; // data inputs start at index 1

  if (dataIdx >= inputs.size()) {
    // Out-of-range select: no output.
    for (auto *out : outputs)
      out->valid = false;
    for (auto *in : inputs)
      in->ready = false;
    return;
  }

  if (!inputs[dataIdx]->valid) {
    // Selected input not yet valid.
    for (auto *out : outputs)
      out->valid = false;
    for (auto *in : inputs)
      in->ready = false;
    return;
  }

  outputs[0]->valid = true;
  outputs[0]->data = inputs[dataIdx]->data;
  driveOutputTag(outputs[0], static_cast<int>(dataIdx));

  // Consume select and selected data input when output is accepted.
  bool accepted = outputs[0]->ready;
  for (auto *in : inputs)
    in->ready = false;
  inputs[0]->ready = accepted;
  inputs[dataIdx]->ready = accepted;
}

//===----------------------------------------------------------------------===//
// handshake.join: wait for all inputs to be valid, produce a single output
// token. Data output is don't-care (0).
//===----------------------------------------------------------------------===//

void SimPE::evaluateJoin() {
  if (outputs.empty()) {
    for (auto *in : inputs)
      in->ready = false;
    return;
  }

  bool allValid = true;
  for (auto *in : inputs) {
    if (!in->valid) {
      allValid = false;
      break;
    }
  }

  if (!allValid) {
    outputs[0]->valid = false;
    for (auto *in : inputs)
      in->ready = false;
    return;
  }

  outputs[0]->valid = true;
  outputs[0]->data = 0;
  outputs[0]->hasTag = false;

  // Consume all inputs when output is accepted.
  bool accepted = outputs[0]->ready;
  for (auto *in : inputs)
    in->ready = accepted;
}

//===----------------------------------------------------------------------===//
// handshake.sink: always ready on all inputs, never produce output.
//===----------------------------------------------------------------------===//

void SimPE::evaluateSink() {
  for (auto *out : outputs)
    out->valid = false;
  for (auto *in : inputs)
    in->ready = true;
}

void SimPE::advanceClock() {
  // State machine transitions for carry/invariant/gate.
  if (bodyType_ == BodyType::Carry) {
    if (dataflowInitialStage_) {
      // Initial: if %a was transferred, transition to block stage.
      if (inputs.size() > 1 && inputs[1]->transferred())
        dataflowInitialStage_ = false;
    } else {
      // Block: if %d was transferred with d=false, return to initial.
      if (!inputs.empty() && inputs[0]->transferred() &&
          (inputs[0]->data & 1) == 0)
        dataflowInitialStage_ = true;
    }
  } else if (bodyType_ == BodyType::Invariant) {
    if (dataflowInitialStage_) {
      // Initial: if %a was transferred, store value and go to block.
      if (inputs.size() > 1 && inputs[1]->transferred()) {
        invariantStoredValue_ = inputs[1]->data;
        dataflowInitialStage_ = false;
      }
    } else {
      // Block: if %d was transferred with d=false, return to initial.
      if (!inputs.empty() && inputs[0]->transferred() &&
          (inputs[0]->data & 1) == 0)
        dataflowInitialStage_ = true;
    }
  } else if (bodyType_ == BodyType::Gate) {
    // Pass-through gate: no state machine needed (see evaluateGate comment).
  } else if (bodyType_ == BodyType::StreamCont) {
    // Helper: compute next index using stepOp_.
    auto computeNext = [this](uint64_t idx, uint64_t step) -> uint64_t {
      if (stepOp_ == "-=")
        return maskToWidth(idx - step);
      if (stepOp_ == "*=")
        return maskToWidth(idx * step);
      if (stepOp_ == "/=")
        return step != 0 ? maskToWidth(idx / step) : 0;
      if (stepOp_ == "<<=")
        return maskToWidth(idx << (step & 63));
      if (stepOp_ == ">>=")
        return maskToWidth(idx >> (step & 63));
      return maskToWidth(idx + step);
    };

    auto evalCont = [this](int64_t idx, int64_t bound) -> bool {
      if (contCondSel_ & 0x01) return idx < bound;
      if (contCondSel_ & 0x02) return idx <= bound;
      if (contCondSel_ & 0x04) return idx > bound;
      if (contCondSel_ & 0x08) return idx >= bound;
      if (contCondSel_ & 0x10) return idx != bound;
      return false;
    };

    if (streamInitialPhase_) {
      // Initial: if all 3 inputs transferred, latch and transition.
      if (inputs.size() >= 3 && outputs.size() >= 2 &&
          outputs[0]->transferred() && outputs[1]->transferred()) {
        uint64_t start = inputs[0]->data;
        uint64_t step = inputs[1]->data;
        uint64_t bound = inputs[2]->data;
        bool willContinue = evalCont(signExt(start), signExt(bound));
        if (willContinue) {
          streamNextIdx_ = computeNext(start, step);
          streamBoundReg_ = bound;
          streamStepReg_ = step;
          streamInitialPhase_ = false;
        }
        // If !willContinue (zero-trip), remain in initial phase.
      }
    } else {
      // Block: if outputs transferred, update nextIdx or return to initial.
      if (outputs.size() >= 2 &&
          outputs[0]->transferred() && outputs[1]->transferred()) {
        bool willContinue = evalCont(signExt(streamNextIdx_),
                                     signExt(streamBoundReg_));
        if (willContinue) {
          streamNextIdx_ = computeNext(streamNextIdx_, streamStepReg_);
        } else {
          fprintf(stderr, "[DBG-STREAM] hwNode=%u block->initial: "
                  "nextIdx=%lu bound=%lu\n",
                  hwNodeId, streamNextIdx_, streamBoundReg_);
          streamInitialPhase_ = true;
        }
      }
    }
  }
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
