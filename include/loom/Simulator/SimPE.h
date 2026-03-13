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
  void evaluateCombinational() override;
  void advanceClock() override;
  void reset() override;
  void configure(const std::vector<uint32_t> &configWords) override;
  void collectTraceEvents(std::vector<TraceEvent> &events,
                          uint64_t cycle) override;
  PerfSnapshot getPerfSnapshot() const override { return perf_; }

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

  /// Gate: true if the first element hasn't been processed yet.
  bool gateFirstElement_ = true;

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
