//===-- SimTemporalPE.h - Simulated fabric.temporal_pe -----------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Temporal PE model per spec-fabric-temporal_pe.md.
// Tag matching, instruction firing, operand buffers (per-instruction and
// shared modes), register FIFOs with multi-reader fork semantics.
//
//===----------------------------------------------------------------------===//

#ifndef LOOM_SIMULATOR_SIMTEMPORALPE_H
#define LOOM_SIMULATOR_SIMTEMPORALPE_H

#include "loom/Simulator/SimModule.h"

#include <deque>

namespace loom {
namespace sim {

class SimTemporalPE : public SimModule {
public:
  SimTemporalPE(unsigned numInputs, unsigned numOutputs, unsigned tagWidth,
                unsigned numInstructions, unsigned numRegisters,
                unsigned regFifoDepth, unsigned numFuTypes,
                unsigned valueWidth, bool sharedOperandBuffer,
                unsigned operandBufferSize);

  bool isCombinational() const override { return false; }
  void evaluateCombinational() override;
  void advanceClock() override;
  void reset() override;
  void configure(const std::vector<uint32_t> &configWords) override;
  void collectTraceEvents(std::vector<TraceEvent> &events,
                          uint64_t cycle) override;
  PerfSnapshot getPerfSnapshot() const override { return perf_; }

private:
  unsigned numInputs_;
  unsigned numOutputs_;
  unsigned tagWidth_;
  unsigned numInstructions_;
  unsigned numRegisters_;
  unsigned regFifoDepth_;
  unsigned numFuTypes_;
  unsigned valueWidth_;
  bool sharedOperandBuffer_;
  unsigned operandBufferSize_;

  /// Decoded instruction from instruction_mem.
  struct Instruction {
    bool valid = false;
    uint16_t tag = 0;
    uint8_t opcode = 0;
    /// Per-operand: is_reg flag and reg_idx.
    struct OperandConfig {
      bool isReg = false;
      unsigned regIdx = 0;
    };
    std::vector<OperandConfig> operands;
    /// Per-result: is_reg flag, reg_idx, output tag.
    struct ResultConfig {
      bool isReg = false;
      unsigned regIdx = 0;
      uint16_t tag = 0;
    };
    std::vector<ResultConfig> results;
  };
  std::vector<Instruction> instructions_;

  //--- Per-instruction operand buffer mode ---
  struct PerInsnBuffer {
    /// [num_instructions][num_inputs]: valid + value.
    std::vector<std::vector<bool>> opValid;
    std::vector<std::vector<uint64_t>> opValue;
  };
  PerInsnBuffer perInsnBuf_;

  //--- Shared operand buffer mode ---
  struct SharedBufEntry {
    uint16_t tag = 0;
    unsigned position = 0;
    std::vector<bool> opValid;
    std::vector<uint64_t> opValue;
  };
  std::vector<SharedBufEntry> sharedBuf_;

  //--- Register FIFOs ---
  struct RegFifo {
    std::deque<uint64_t> data;
    /// Track how many readers still need to consume the front entry.
    unsigned readersRemaining = 0;
    unsigned totalReaders = 0;
  };
  std::vector<RegFifo> regFifos_;

  /// Which instruction fired this cycle (-1 if none).
  int firedInstruction_ = -1;
  bool firedThisCycle_ = false;

  /// Helper: execute the FU operation.
  uint64_t executeFU(uint8_t opcode, const std::vector<uint64_t> &operands) const;

  /// Helper: check if instruction can fire.
  bool canFire(unsigned insnIdx) const;
};

} // namespace sim
} // namespace loom

#endif // LOOM_SIMULATOR_SIMTEMPORALPE_H
