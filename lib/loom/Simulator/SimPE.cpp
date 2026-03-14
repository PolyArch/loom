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
  initLatched_ = false;
  initLatchedValue_ = 0;
  carryDLatched_ = false;
  carryDValue_ = false;
  invariantStoredValue_ = 0;
  gateState_ = 0;
  gateLatchedValue_ = 0;
  gateLatchedCond_ = false;
  gateOutAccepted_[0] = gateOutAccepted_[1] = false;
  streamOutAccepted_[0] = streamOutAccepted_[1] = false;
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
// Store PE: two independent data paths (mirrors load PE design).
// Decoupled to avoid structural deadlock when addr is broadcast with loads.
//
// Ports:
//   in0 = addr_from_comp, in1 = data_from_comp, in2 = ctrl
//   out0 = addr_to_mem,   out1 = data_to_mem
//
// Path 1 (addr + ctrl sync → addr_to_mem):
//   out0_valid = in0_valid && in2_valid
//   fire = sync_valid && out0_ready
//   in0_ready = in2_ready = fire
//
// Path 2 (data passthrough, independent):
//   out1_valid = in1_valid
//   in1_ready  = out1_ready
//
// Pairing of addr and data happens at the memory side (SimMemory store queue).
//===----------------------------------------------------------------------===//

void SimPE::evaluateStore() {
  if (inputs.size() < 3 || outputs.size() < 2)
    return;

  SimChannel *addrIn = inputs[0];  // addr_from_comp
  SimChannel *dataIn = inputs[1];  // data_from_comp
  SimChannel *ctrlIn = inputs[2];  // ctrl
  SimChannel *addrOut = outputs[0]; // addr_to_mem
  SimChannel *dataOut = outputs[1]; // data_to_mem

  // Path 1: addr + ctrl → addr_to_mem
  bool addrCtrlSync = addrIn->valid && ctrlIn->valid;
  addrOut->valid = addrCtrlSync;
  addrOut->data = addrIn->data;
  driveOutputTag(addrOut, 0);

  bool addrFire = addrCtrlSync && addrOut->ready;
  addrIn->ready = addrFire;
  ctrlIn->ready = addrFire;

  // Path 2: data_from_comp → data_to_mem (independent)
  dataOut->valid = dataIn->valid;
  dataOut->data = dataIn->data;
  driveOutputTag(dataOut, 1);

  dataIn->ready = dataOut->ready;
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

  if (dataflowInitialStage_ && !initLatched_) {
    // WAIT: consume %a (input[1]). No output. Latch value.
    out->valid = false;
    if (inputs.size() > 1 && inputs[1]->valid) {
      inputs[0]->ready = false;
      inputs[1]->ready = true;
      if (inputs.size() > 2) inputs[2]->ready = false;
    } else {
      for (auto *in : inputs) in->ready = false;
    }
  } else if (dataflowInitialStage_ && initLatched_) {
    // PRODUCE: output latched initial value. No input consumed.
    out->valid = true;
    out->data = initLatchedValue_;
    driveOutputTag(out, -1);
    for (auto *in : inputs) in->ready = false;
  } else if (!carryDLatched_) {
    // Block stage step 1: accept %d independently. No output.
    // This decouples %d acceptance from %b, preventing broadcast deadlock.
    out->valid = false;
    if (!inputs.empty() && inputs[0]->valid) {
      inputs[0]->ready = true;
      inputs[1]->ready = false;
      if (inputs.size() > 2) inputs[2]->ready = false;
    } else {
      for (auto *in : inputs) in->ready = false;
    }
  } else {
    // Block stage step 2: %d latched. Act on its value.
    if (carryDValue_) {
      // d=true: wait for %b, then emit %b on output.
      if (inputs.size() <= 2 || !inputs[2]->valid) {
        out->valid = false;
        for (auto *in : inputs) in->ready = false;
        return;
      }
      out->valid = true;
      out->data = inputs[2]->data;
      driveOutputTag(out, 2);
      inputs[0]->ready = false;
      inputs[1]->ready = false;
      inputs[2]->ready = out->ready;
    } else {
      // d=false: no output needed. Transition in advanceClock.
      out->valid = false;
      for (auto *in : inputs) in->ready = false;
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

  if (dataflowInitialStage_ && !initLatched_) {
    // WAIT: consume %a (input[1]). No output. Latch value.
    out->valid = false;
    if (inputs.size() > 1 && inputs[1]->valid) {
      inputs[0]->ready = false;
      inputs[1]->ready = true;
      if (inputs.size() > 2) inputs[2]->ready = false;
    } else {
      for (auto *in : inputs) in->ready = false;
    }
  } else if (dataflowInitialStage_ && initLatched_) {
    // PRODUCE: output latched initial value. No input consumed.
    out->valid = true;
    out->data = initLatchedValue_;
    driveOutputTag(out, -1);
    for (auto *in : inputs) in->ready = false;
  } else if (!carryDLatched_) {
    // Block stage step 1: accept %d independently. No output.
    // Reuses carryDLatched_/carryDValue_ (shared with carry).
    out->valid = false;
    if (!inputs.empty() && inputs[0]->valid) {
      inputs[0]->ready = true;
      if (inputs.size() > 1) inputs[1]->ready = false;
    } else {
      for (auto *in : inputs) in->ready = false;
    }
  } else {
    // Block stage step 2: %d latched.
    if (carryDValue_) {
      // d=true: emit stored invariant value.
      out->valid = true;
      out->data = invariantStoredValue_;
      driveOutputTag(out, -1);
      for (auto *in : inputs) in->ready = false;
    } else {
      // d=false: no output. Transition in advanceClock.
      out->valid = false;
      for (auto *in : inputs) in->ready = false;
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
  // dataflow.gate per spec-dataflow.md:
  //   after_value[i] = before_value[i]    (cuts the tail)
  //   after_cond[i]  = before_cond[i+1]   (cuts the head)
  //
  // Hardware timing (one-element shift):
  //   Initial: output after_value[0] alone, discard before_cond[0] (head cut)
  //   Block:   output (after_value, after_cond) together using current inputs
  //   Tail:    when cond=false, output only after_cond=false, discard value
  //            (tail cut), then return to initial.
  if (outputs.size() < 2 || inputs.size() < 2)
    return;

  auto *outVal = outputs[0];   // after_value
  auto *outCond = outputs[1];  // after_cond

  switch (gateState_) {
  case 0: // INIT: consume first (value, cond) pair. No output.
    outVal->valid = false;
    outCond->valid = false;
    if (inputs[0]->valid && inputs[1]->valid) {
      inputs[0]->ready = true;
      inputs[1]->ready = true;
    } else {
      inputs[0]->ready = false;
      inputs[1]->ready = false;
    }
    break;

  case 1: // HEAD: output after_value[0] from latch. No input accepted.
    if (gateLatchedCond_ && !gateOutAccepted_[0]) {
      outVal->valid = true;
      outVal->data = gateLatchedValue_;
      driveOutputTag(outVal, 0);
    } else {
      outVal->valid = false;
    }
    outCond->valid = false;
    inputs[0]->ready = false;
    inputs[1]->ready = false;
    break;

  case 2: // WAIT: consume next (value, cond) pair. No output.
    outVal->valid = false;
    outCond->valid = false;
    if (inputs[0]->valid && inputs[1]->valid) {
      inputs[0]->ready = true;
      inputs[1]->ready = true;
    } else {
      inputs[0]->ready = false;
      inputs[1]->ready = false;
    }
    break;

  case 3: // BLOCK: output latched (value, cond). No input accepted.
    if (gateLatchedCond_) {
      // Continuing: output both (only legs not yet accepted).
      outVal->valid = !gateOutAccepted_[0];
      if (outVal->valid) {
        outVal->data = gateLatchedValue_;
        driveOutputTag(outVal, 0);
      }
      outCond->valid = !gateOutAccepted_[1];
      if (outCond->valid) {
        outCond->data = 1;
        driveOutputTag(outCond, -1);
      }
    } else {
      // Tail cut: only output after_cond=false (if not yet accepted).
      outVal->valid = false;
      outCond->valid = !gateOutAccepted_[1];
      if (outCond->valid) {
        outCond->data = 0;
        driveOutputTag(outCond, -1);
      }
    }
    inputs[0]->ready = false;
    inputs[1]->ready = false;
    break;
  }
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

  if (streamInitialPhase_) {
    // WAIT_INPUT: accept 3 scalar inputs. No output.
    outIdx->valid = false;
    outCont->valid = false;
    if (inputs[0]->valid && inputs[1]->valid && inputs[2]->valid) {
      inputs[0]->ready = true;
      inputs[1]->ready = true;
      inputs[2]->ready = true;
    } else {
      for (auto *in : inputs)
        in->ready = false;
    }
  } else {
    // PRODUCE: drive (idx, cont) from registers. Hold until both accepted.
    // Per-output acceptance: only drive valid on legs not yet accepted.
    bool willContinue = streamEvalCont(signExt(streamNextIdx_),
                                       signExt(streamBoundReg_));
    outIdx->valid = !streamOutAccepted_[0];
    if (outIdx->valid) {
      outIdx->data = streamNextIdx_;
      driveOutputTag(outIdx, -1);
    }
    outCont->valid = !streamOutAccepted_[1];
    if (outCont->valid) {
      outCont->data = willContinue ? 1 : 0;
      driveOutputTag(outCont, -1);
    }

    // No inputs consumed while producing.
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
    if (dataflowInitialStage_ && !initLatched_) {
      // WAIT: %a consumed → latch and mark latched.
      if (inputs.size() > 1 && inputs[1]->transferred()) {
        initLatchedValue_ = inputs[1]->data;
        initLatched_ = true;
      }
    } else if (dataflowInitialStage_ && initLatched_) {
      // PRODUCE: output accepted → transition to block.
      if (!outputs.empty() && outputs[0]->transferred()) {
        initLatched_ = false;
        dataflowInitialStage_ = false;
      }
    } else if (!carryDLatched_) {
      // Block step 1: %d consumed → latch d value.
      if (!inputs.empty() && inputs[0]->transferred()) {
        carryDValue_ = (inputs[0]->data & 1) != 0;
        carryDLatched_ = true;
      }
    } else {
      // Block step 2: d latched.
      if (carryDValue_) {
        // d=true: %b consumed and output accepted → clear d latch.
        if (inputs.size() > 2 && inputs[2]->transferred() &&
            !outputs.empty() && outputs[0]->transferred()) {
          carryDLatched_ = false;
        }
      } else {
        // d=false: no %b needed. Return to initial.
        dataflowInitialStage_ = true;
        initLatched_ = false;
        carryDLatched_ = false;
      }
    }
  } else if (bodyType_ == BodyType::Invariant) {
    if (dataflowInitialStage_ && !initLatched_) {
      // WAIT: %a consumed → latch and mark latched.
      if (inputs.size() > 1 && inputs[1]->transferred()) {
        initLatchedValue_ = inputs[1]->data;
        invariantStoredValue_ = inputs[1]->data; // Also store for block phase.
        initLatched_ = true;
      }
    } else if (dataflowInitialStage_ && initLatched_) {
      // PRODUCE: output accepted → transition to block.
      if (!outputs.empty() && outputs[0]->transferred()) {
        initLatched_ = false;
        dataflowInitialStage_ = false;
      }
    } else if (!carryDLatched_) {
      // Block step 1: %d consumed → latch d value.
      if (!inputs.empty() && inputs[0]->transferred()) {
        carryDValue_ = (inputs[0]->data & 1) != 0;
        carryDLatched_ = true;
      }
    } else {
      // Block step 2: d latched.
      if (carryDValue_) {
        // d=true: output accepted → clear d latch.
        if (!outputs.empty() && outputs[0]->transferred())
          carryDLatched_ = false;
      } else {
        // d=false: return to initial.
        dataflowInitialStage_ = true;
        initLatched_ = false;
        carryDLatched_ = false;
      }
    }
  } else if (bodyType_ == BodyType::Gate) {
    // 4-state gate: INIT(0) → HEAD(1) → WAIT(2) → BLOCK(3) → WAIT → ...
    // Consume and produce NEVER happen in the same cycle.
    switch (gateState_) {
    case 0: // INIT: if consumed → latch → HEAD
      if (inputs.size() >= 2 && inputs[0]->transferred() &&
          inputs[1]->transferred()) {
        gateLatchedValue_ = inputs[0]->data;
        gateLatchedCond_ = (inputs[1]->data & 1) != 0;
        gateState_ = 1;
      }
      break;
    case 1: // HEAD: output from latch (only out0)
      if (!gateLatchedCond_) {
        gateState_ = 0; // Zero-trip → INIT
      } else {
        if (outputs.size() >= 1 && outputs[0]->transferred())
          gateOutAccepted_[0] = true;
        if (gateOutAccepted_[0]) {
          gateOutAccepted_[0] = false;
          gateState_ = 2; // → WAIT
        }
      }
      break;
    case 2: // WAIT: if consumed → latch → BLOCK
      if (inputs.size() >= 2 && inputs[0]->transferred() &&
          inputs[1]->transferred()) {
        gateLatchedValue_ = inputs[0]->data;
        gateLatchedCond_ = (inputs[1]->data & 1) != 0;
        gateOutAccepted_[0] = gateOutAccepted_[1] = false;
        gateState_ = 3;
      }
      break;
    case 3: // BLOCK: per-output acceptance tracking
      if (gateLatchedCond_) {
        // Continuing: track each output independently.
        if (outputs.size() >= 2) {
          if (outputs[0]->transferred()) gateOutAccepted_[0] = true;
          if (outputs[1]->transferred()) gateOutAccepted_[1] = true;
        }
        if (gateOutAccepted_[0] && gateOutAccepted_[1]) {
          gateOutAccepted_[0] = gateOutAccepted_[1] = false;
          gateState_ = 2; // Both accepted → WAIT
        }
      } else {
        // Tail cut: only out1 matters.
        if (outputs.size() >= 2 && outputs[1]->transferred())
          gateOutAccepted_[1] = true;
        if (gateOutAccepted_[1]) {
          gateOutAccepted_[0] = gateOutAccepted_[1] = false;
          gateState_ = 0; // → INIT
        }
      }
      break;
    }
  } else if (bodyType_ == BodyType::StreamCont) {
    if (streamInitialPhase_) {
      // WAIT_INPUT: if all 3 inputs consumed, latch and → PRODUCE.
      if (inputs.size() >= 3 && inputs[0]->transferred() &&
          inputs[1]->transferred() && inputs[2]->transferred()) {
        uint64_t start = inputs[0]->data;
        uint64_t step = inputs[1]->data;
        uint64_t bound = inputs[2]->data;
        if (step == 0) {
          latchError(RtError::RT_DATAFLOW_STREAM_ZERO_STEP);
          return;
        }
        streamNextIdx_ = start;
        streamBoundReg_ = bound;
        streamStepReg_ = step;
        streamInitialPhase_ = false; // → PRODUCE
      }
    } else {
      // PRODUCE: per-output acceptance tracking.
      if (outputs.size() >= 2) {
        if (outputs[0]->transferred()) streamOutAccepted_[0] = true;
        if (outputs[1]->transferred()) streamOutAccepted_[1] = true;
      }
      if (streamOutAccepted_[0] && streamOutAccepted_[1]) {
        // Both accepted → advance.
        streamOutAccepted_[0] = streamOutAccepted_[1] = false;
        bool willContinue = streamEvalCont(signExt(streamNextIdx_),
                                           signExt(streamBoundReg_));
        if (willContinue) {
          streamNextIdx_ = streamComputeNext(streamNextIdx_, streamStepReg_);
        } else {
          streamInitialPhase_ = true; // → WAIT_INPUT
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
