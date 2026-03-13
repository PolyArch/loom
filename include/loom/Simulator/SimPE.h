//===-- SimPE.h - Simulated fabric.pe ----------------------------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// PE model per spec-fabric-pe.md. Supports all PE body types:
//   - Compute (native and tagged, with optional compare predicates)
//   - Constant (native and tagged)
//   - Load/Store (TagOverwrite and TagTransparent)
//   - dataflow.stream, dataflow.carry, dataflow.invariant, dataflow.gate
//
//===----------------------------------------------------------------------===//

#ifndef LOOM_SIMULATOR_SIMPE_H
#define LOOM_SIMULATOR_SIMPE_H

#include "loom/Simulator/SimModule.h"

namespace loom {
namespace sim {

class SimPE : public SimModule {
public:
  /// PE body type classification.
  enum class BodyType {
    Compute,       // Arithmetic/logic operation
    Constant,      // Constant output
    Load,          // Memory load
    Store,         // Memory store
    StreamCont,    // dataflow.stream continuation condition
    Carry,         // dataflow.carry
    Invariant,     // dataflow.invariant
    Gate,          // dataflow.gate
    CondBranch,    // handshake.cond_br: route data to one of two outputs
    Mux,           // handshake.mux: select one of N data inputs
    Join,          // handshake.join: synchronize all inputs
    Sink,          // handshake.sink: consume input, no output
  };

  /// Tag mode for load/store PEs.
  enum class TagMode {
    Native,        // No tags
    TagOverwrite,  // Output tag from config
    TagTransparent // Tag passthrough
  };

  SimPE(BodyType bodyType, unsigned numInputs, unsigned numOutputs,
        bool isTagged, unsigned tagWidth, unsigned dataWidth,
        const std::string &opcodeStr, TagMode tagMode = TagMode::Native);

  bool isCombinational() const override {
    return bodyType_ != BodyType::Carry && bodyType_ != BodyType::Invariant &&
           bodyType_ != BodyType::Gate && bodyType_ != BodyType::StreamCont;
  }
  // Note: CondBranch, Mux, Join, Sink are all combinational (no state).
  void evaluateCombinational() override;
  void advanceClock() override;
  void reset() override;
  void configure(const std::vector<uint32_t> &configWords) override;
  void collectTraceEvents(std::vector<TraceEvent> &events,
                          uint64_t cycle) override;
  PerfSnapshot getPerfSnapshot() const override { return perf_; }

  /// Set the stream step_op (called from factory after construction).
  void setStepOp(const std::string &op) { stepOp_ = op; }

private:
  BodyType bodyType_;
  bool isTagged_;
  unsigned tagWidth_;
  unsigned dataWidth_;
  std::string opcodeStr_;
  TagMode tagMode_;

  /// Configured output tags (one per output, for tagged PEs).
  std::vector<uint16_t> outputTags_;

  /// Constant value (for constant PEs).
  uint64_t constantValue_ = 0;

  /// Compare predicate (4-bit encoding for cmpi PEs).
  uint8_t cmpPredicate_ = 0;

  /// Stream continuation condition selector (5-bit one-hot).
  uint8_t contCondSel_ = 0;

  /// Stream step_op (from fabric.pe definition): "+=", "-=", "*=", "/=",
  /// "<<=", ">>=". Default "+=".
  std::string stepOp_ = "+=";

  //--- dataflow.stream state machine ---

  /// Stream phase: true = initial (waiting for start/step/bound),
  /// false = block (emitting idx/cont each cycle).
  bool streamInitialPhase_ = true;

  /// Latched registers for the block phase.
  uint64_t streamNextIdx_ = 0;
  uint64_t streamBoundReg_ = 0;
  uint64_t streamStepReg_ = 0;

  //--- Dataflow state machines (carry, invariant, gate) ---

  /// Carry/Invariant stage: true = initial, false = block.
  bool dataflowInitialStage_ = true;

  /// Invariant stored value (latched from %a in initial stage).
  uint64_t invariantStoredValue_ = 0;

  /// Carry/invariant: true when initial value has been latched but not yet output.
  bool initLatched_ = false;
  uint64_t initLatchedValue_ = 0;

  /// Stream helpers (used by both evaluateCombinational and advanceClock).
  bool streamEvalCont(int64_t idx, int64_t bound) const {
    if (contCondSel_ & 0x01) return idx < bound;
    if (contCondSel_ & 0x02) return idx <= bound;
    if (contCondSel_ & 0x04) return idx > bound;
    if (contCondSel_ & 0x08) return idx >= bound;
    if (contCondSel_ & 0x10) return idx != bound;
    return false;
  }
  uint64_t streamComputeNext(uint64_t idx, uint64_t step) const {
    if (stepOp_ == "-=") return maskToWidth(idx - step);
    if (stepOp_ == "*=") return maskToWidth(idx * step);
    if (stepOp_ == "/=") return step != 0 ? maskToWidth(idx / step) : 0;
    if (stepOp_ == "<<=") return maskToWidth(idx << (step & 63));
    if (stepOp_ == ">>=") return maskToWidth(idx >> (step & 63));
    return maskToWidth(idx + step);
  }

  /// Gate state machine (4 states, consume/produce never overlap):
  ///   0 = INIT: consume first (value, cond) pair. No output.
  ///   1 = HEAD: output after_value[0] from latch. No input.
  ///   2 = WAIT: consume next (value, cond) pair. No output.
  ///   3 = BLOCK: output latched (value, cond). No input.
  ///             If latched cond=F: output only cond (tail cut).
  unsigned gateState_ = 0;
  uint64_t gateLatchedValue_ = 0;
  bool gateLatchedCond_ = false;
  /// Per-output acceptance for current logical step.
  /// Once set, evaluateCombinational drives valid=false on that leg.
  /// Cleared when PE advances state (all legs accepted).
  bool gateOutAccepted_[2] = {false, false};

  /// Stream per-output acceptance (same semantics as gate).
  bool streamOutAccepted_[2] = {false, false};

  /// Sign-extend a value from dataWidth_ to 64 bits.
  int64_t signExt(uint64_t v) const;

  /// Mask a value to dataWidth_ bits.
  uint64_t maskToWidth(uint64_t v) const;

  /// Float bit-cast helpers (data stored as uint64_t bit pattern).
  float toFloat(uint64_t v) const;
  uint64_t fromFloat(float f) const;
  double toDouble(uint64_t v) const;
  uint64_t fromDouble(double d) const;

  /// Execute the PE's arithmetic/logic operation.
  uint64_t executeOp(uint64_t a, uint64_t b) const;

  /// Evaluate integer compare predicate (arith.cmpi, 10 predicates).
  bool evaluateCmpi(uint64_t a, uint64_t b) const;

  /// Evaluate float compare predicate (arith.cmpf, 16 IEEE 754 predicates).
  bool evaluateCmpf(uint64_t a, uint64_t b) const;

  /// Dataflow state machine evaluators.
  void evaluateCarry();
  void evaluateInvariant();
  void evaluateGate();
  void evaluateStream();

  /// Load/Store evaluators with spec-correct port semantics.
  void evaluateLoad();
  void evaluateStore();

  /// Handshake control evaluators.
  void evaluateCondBranch();
  void evaluateMux();
  void evaluateJoin();
  void evaluateSink();

  /// Drive output tag based on tag mode. If srcInputIdx >= 0, uses that
  /// input's tag for TagTransparent; otherwise uses configured outputTags_.
  void driveOutputTag(SimChannel *out, int srcInputIdx) {
    if (isTagged_) {
      if (tagMode_ == TagMode::TagTransparent && srcInputIdx >= 0 &&
          static_cast<unsigned>(srcInputIdx) < inputs.size())
        out->tag = inputs[srcInputIdx]->tag;
      else if (tagWidth_ > 0 && !outputTags_.empty())
        out->tag = outputTags_[0];
      out->hasTag = true;
    } else {
      out->hasTag = false;
    }
  }
};

} // namespace sim
} // namespace loom

#endif // LOOM_SIMULATOR_SIMPE_H
