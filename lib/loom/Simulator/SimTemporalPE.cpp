//===-- SimTemporalPE.cpp - Simulated fabric.temporal_pe -----------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include "loom/Simulator/SimTemporalPE.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace loom {
namespace sim {

namespace {

/// Compute ceil(log2(n)), returning 0 for n <= 1.
unsigned log2Ceil(unsigned n) {
  unsigned result = 0;
  unsigned v = n;
  while (v > 1) {
    result++;
    v = (v + 1) / 2;
  }
  return result;
}

} // namespace

SimTemporalPE::SimTemporalPE(unsigned numInputs, unsigned numOutputs,
                               unsigned tagWidth, unsigned numInstructions,
                               unsigned numRegisters, unsigned regFifoDepth,
                               unsigned numFuTypes, unsigned valueWidth,
                               bool sharedOperandBuffer,
                               unsigned operandBufferSize)
    : numInputs_(numInputs), numOutputs_(numOutputs), tagWidth_(tagWidth),
      numInstructions_(numInstructions), numRegisters_(numRegisters),
      regFifoDepth_(regFifoDepth), numFuTypes_(numFuTypes),
      valueWidth_(valueWidth), sharedOperandBuffer_(sharedOperandBuffer),
      operandBufferSize_(operandBufferSize) {
  instructions_.resize(numInstructions);
  regFifos_.resize(numRegisters);
}

void SimTemporalPE::reset() {
  // NOTE: Do NOT clear instructions_ here. Instructions are configuration
  // state set by configure(), not runtime state. reset() only clears
  // execution state (operand buffers, register FIFOs, errors).

  if (!sharedOperandBuffer_) {
    perInsnBuf_.opValid.assign(numInstructions_,
                               std::vector<bool>(numInputs_, false));
    perInsnBuf_.opValue.assign(numInstructions_,
                               std::vector<uint64_t>(numInputs_, 0));
  } else {
    sharedBuf_.clear();
  }

  for (auto &rf : regFifos_) {
    rf.data.clear();
    rf.readersRemaining = 0;
    rf.totalReaders = 0;
  }

  firedInstruction_ = -1;
  firedThisCycle_ = false;
  errorValid_ = false;
  errorCode_ = RtError::OK;
  pendingError_ = false;
  pendingErrorCode_ = RtError::OK;
  perf_ = PerfSnapshot();
}

void SimTemporalPE::configure(const std::vector<uint32_t> &configWords) {
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

  // Instruction width calculation per spec.
  unsigned opcodeWidth = (numFuTypes_ > 1) ? log2Ceil(numFuTypes_) : 0;
  unsigned R = numRegisters_;
  unsigned logR = (R > 1) ? log2Ceil(R) : 0;

  for (unsigned s = 0; s < numInstructions_; ++s) {
    auto &insn = instructions_[s];
    insn.valid = (extractBits(1) != 0);
    insn.tag = static_cast<uint16_t>(extractBits(tagWidth_));

    if (opcodeWidth > 0)
      insn.opcode = static_cast<uint8_t>(extractBits(opcodeWidth));

    insn.operands.resize(numInputs_);
    for (unsigned i = 0; i < numInputs_; ++i) {
      if (R > 0) {
        insn.operands[i].isReg = (extractBits(1) != 0);
        if (logR > 0)
          insn.operands[i].regIdx = static_cast<unsigned>(extractBits(logR));
      }
    }

    insn.results.resize(numOutputs_);
    for (unsigned o = 0; o < numOutputs_; ++o) {
      if (R > 0) {
        insn.results[o].isReg = (extractBits(1) != 0);
        if (logR > 0)
          insn.results[o].regIdx = static_cast<unsigned>(extractBits(logR));
      }
      insn.results[o].tag = static_cast<uint16_t>(extractBits(tagWidth_));
    }
  }

  // Validate: duplicate tags.
  for (unsigned i = 0; i < numInstructions_; ++i) {
    if (!instructions_[i].valid)
      continue;
    for (unsigned j = i + 1; j < numInstructions_; ++j) {
      if (instructions_[j].valid &&
          instructions_[j].tag == instructions_[i].tag)
        latchError(RtError::CFG_TEMPORAL_PE_DUP_TAG);
    }
  }

  // Validate: register indices.
  for (auto &insn : instructions_) {
    if (!insn.valid)
      continue;
    for (auto &op : insn.operands) {
      if (op.isReg && op.regIdx >= numRegisters_)
        latchError(RtError::CFG_TEMPORAL_PE_ILLEGAL_REG);
    }
    for (auto &res : insn.results) {
      if (res.isReg) {
        if (res.regIdx >= numRegisters_)
          latchError(RtError::CFG_TEMPORAL_PE_ILLEGAL_REG);
        if (res.tag != 0)
          latchError(RtError::CFG_TEMPORAL_PE_REG_TAG_NONZERO);
      }
    }
  }

  // Validate: single-writer constraint per spec. Each register must have
  // at most one instruction that writes to it.
  if (numRegisters_ > 0) {
    std::vector<int> regWriter(numRegisters_, -1);
    for (unsigned s = 0; s < numInstructions_; ++s) {
      if (!instructions_[s].valid)
        continue;
      for (auto &res : instructions_[s].results) {
        if (res.isReg && res.regIdx < numRegisters_) {
          if (regWriter[res.regIdx] >= 0 &&
              regWriter[res.regIdx] != static_cast<int>(s)) {
            // Multiple writers to same register: configuration error.
            // Use ILLEGAL_REG as the closest error code.
            latchError(RtError::CFG_TEMPORAL_PE_ILLEGAL_REG);
          }
          regWriter[res.regIdx] = static_cast<int>(s);
        }
      }
    }
  }

  // Initialize operand buffers and register FIFOs for execution.
  // Preserve configuration errors across reset() per spec: "error_valid is
  // sticky after first assertion" and "held until reset". Config errors from
  // validation above must persist.
  bool savedErrorValid = errorValid_;
  uint16_t savedErrorCode = errorCode_;
  bool savedPendingError = pendingError_;
  uint16_t savedPendingErrorCode = pendingErrorCode_;
  reset();
  errorValid_ = savedErrorValid;
  errorCode_ = savedErrorCode;
  pendingError_ = savedPendingError;
  pendingErrorCode_ = savedPendingErrorCode;

  // Compute totalReaders for each register FIFO by counting how many
  // instruction operands reference it as a register source.
  for (auto &rf : regFifos_) {
    rf.totalReaders = 0;
  }
  for (auto &insn : instructions_) {
    if (!insn.valid)
      continue;
    for (auto &op : insn.operands) {
      if (op.isReg && op.regIdx < regFifos_.size())
        regFifos_[op.regIdx].totalReaders++;
    }
  }
  // Initialize readersRemaining to totalReaders for each FIFO.
  for (auto &rf : regFifos_) {
    rf.readersRemaining = rf.totalReaders;
  }
}

bool SimTemporalPE::canFire(unsigned insnIdx) const {
  auto &insn = instructions_[insnIdx];
  if (!insn.valid)
    return false;

  if (!sharedOperandBuffer_) {
    // Per-instruction mode: all operands must be ready.
    for (unsigned i = 0; i < numInputs_; ++i) {
      if (insn.operands[i].isReg) {
        // Operand comes from register - check register has data.
        if (insn.operands[i].regIdx < regFifos_.size() &&
            regFifos_[insn.operands[i].regIdx].data.empty())
          return false;
      } else {
        if (!perInsnBuf_.opValid[insnIdx][i])
          return false;
      }
    }
  } else {
    // Shared buffer mode: find entry with matching tag, position=0,
    // and all op_valid set.
    bool found = false;
    for (auto &entry : sharedBuf_) {
      if (entry.tag != insn.tag || entry.position != 0)
        continue;
      bool allValid = true;
      for (unsigned i = 0; i < numInputs_; ++i) {
        if (insn.operands[i].isReg)
          continue; // Register operands checked separately.
        if (!entry.opValid[i]) {
          allValid = false;
          break;
        }
      }
      if (allValid) {
        found = true;
        break;
      }
    }
    if (!found)
      return false;

    // Also check register operands.
    for (unsigned i = 0; i < numInputs_; ++i) {
      if (insn.operands[i].isReg &&
          insn.operands[i].regIdx < regFifos_.size() &&
          regFifos_[insn.operands[i].regIdx].data.empty())
        return false;
    }
  }

  // Check all non-register output channels are ready.
  for (unsigned o = 0; o < numOutputs_ && o < outputs.size(); ++o) {
    if (!insn.results[o].isReg && !outputs[o]->ready)
      return false;
  }

  // Check register FIFO capacity for register results.
  for (unsigned o = 0; o < numOutputs_; ++o) {
    if (insn.results[o].isReg &&
        insn.results[o].regIdx < regFifos_.size() &&
        regFifos_[insn.results[o].regIdx].data.size() >= regFifoDepth_)
      return false;
  }

  return true;
}

void SimTemporalPE::setFUDescriptor(unsigned fuIdx,
                                     const std::string &bodyOp,
                                     unsigned dataWidth) {
  if (fuIdx >= fuDescriptors_.size())
    fuDescriptors_.resize(fuIdx + 1);
  fuDescriptors_[fuIdx].bodyOp = bodyOp;
  fuDescriptors_[fuIdx].dataWidth = dataWidth;
}

uint64_t SimTemporalPE::executeFU(
    uint8_t opcode, const std::vector<uint64_t> &operands) const {
  // Dispatch to the FU body identified by opcode. Each FU type is a native
  // fabric.pe body per spec-fabric-temporal_pe.md.
  unsigned dw = valueWidth_;
  std::string bodyOp = "arith.addi"; // Fallback if no descriptor.
  if (opcode < fuDescriptors_.size()) {
    bodyOp = fuDescriptors_[opcode].bodyOp;
    dw = fuDescriptors_[opcode].dataWidth;
  }

  uint64_t a = !operands.empty() ? operands[0] : 0;
  uint64_t b = operands.size() > 1 ? operands[1] : 0;

  auto mask = [dw](uint64_t v) -> uint64_t {
    if (dw >= 64)
      return v;
    return v & ((1ULL << dw) - 1);
  };

  auto signExt = [dw](uint64_t v) -> int64_t {
    if (dw >= 64)
      return static_cast<int64_t>(v);
    if (v & (1ULL << (dw - 1)))
      return static_cast<int64_t>(v | (~0ULL << dw));
    return static_cast<int64_t>(v);
  };

  auto toFloat = [](uint64_t v) -> float {
    float f;
    uint32_t u = static_cast<uint32_t>(v);
    std::memcpy(&f, &u, sizeof(f));
    return f;
  };

  auto fromFloat = [](float f) -> uint64_t {
    uint32_t u;
    std::memcpy(&u, &f, sizeof(u));
    return u;
  };

  auto toDouble = [](uint64_t v) -> double {
    double d;
    std::memcpy(&d, &v, sizeof(d));
    return d;
  };

  auto fromDouble = [](double d) -> uint64_t {
    uint64_t u;
    std::memcpy(&u, &d, sizeof(u));
    return u;
  };

  // Integer arith ops.
  if (bodyOp == "arith.addi") return mask(a + b);
  if (bodyOp == "arith.subi") return mask(a - b);
  if (bodyOp == "arith.muli") return mask(a * b);
  if (bodyOp == "arith.divsi")
    return b != 0 ? mask(static_cast<uint64_t>(signExt(a) / signExt(b))) : 0;
  if (bodyOp == "arith.divui")
    return b != 0 ? mask(a / b) : 0;
  if (bodyOp == "arith.remsi")
    return b != 0 ? mask(static_cast<uint64_t>(signExt(a) % signExt(b))) : 0;
  if (bodyOp == "arith.remui")
    return b != 0 ? mask(a % b) : 0;
  if (bodyOp == "arith.andi") return mask(a & b);
  if (bodyOp == "arith.ori") return mask(a | b);
  if (bodyOp == "arith.xori") return mask(a ^ b);
  if (bodyOp == "arith.shli") return mask(a << (b & 63));
  if (bodyOp == "arith.shrsi")
    return mask(static_cast<uint64_t>(signExt(a) >> (b & 63)));
  if (bodyOp == "arith.shrui") return mask(a >> (b & 63));
  if (bodyOp == "arith.extsi")
    return static_cast<uint64_t>(signExt(a));
  if (bodyOp == "arith.extui" || bodyOp == "arith.trunci")
    return mask(a);
  if (bodyOp == "arith.index_cast" || bodyOp == "arith.index_castui")
    return mask(a);
  if (bodyOp == "arith.select")
    return (a & 1) ? b : (operands.size() > 2 ? operands[2] : 0);

  // Float arith ops.
  if (bodyOp == "arith.addf") {
    if (dw <= 32) return fromFloat(toFloat(a) + toFloat(b));
    return fromDouble(toDouble(a) + toDouble(b));
  }
  if (bodyOp == "arith.subf") {
    if (dw <= 32) return fromFloat(toFloat(a) - toFloat(b));
    return fromDouble(toDouble(a) - toDouble(b));
  }
  if (bodyOp == "arith.mulf") {
    if (dw <= 32) return fromFloat(toFloat(a) * toFloat(b));
    return fromDouble(toDouble(a) * toDouble(b));
  }
  if (bodyOp == "arith.divf") {
    if (dw <= 32) return fromFloat(toFloat(a) / toFloat(b));
    return fromDouble(toDouble(a) / toDouble(b));
  }
  if (bodyOp == "arith.negf") {
    if (dw <= 32) return fromFloat(-toFloat(a));
    return fromDouble(-toDouble(a));
  }

  // Float-int conversion.
  if (bodyOp == "arith.fptosi") {
    if (dw <= 32)
      return mask(static_cast<uint64_t>(static_cast<int64_t>(toFloat(a))));
    return static_cast<uint64_t>(static_cast<int64_t>(toDouble(a)));
  }
  if (bodyOp == "arith.fptoui") {
    if (dw <= 32) return mask(static_cast<uint64_t>(toFloat(a)));
    return static_cast<uint64_t>(toDouble(a));
  }
  if (bodyOp == "arith.sitofp") {
    if (dw <= 32) return fromFloat(static_cast<float>(signExt(a)));
    return fromDouble(static_cast<double>(signExt(a)));
  }
  if (bodyOp == "arith.uitofp") {
    if (dw <= 32) return fromFloat(static_cast<float>(a));
    return fromDouble(static_cast<double>(a));
  }

  // Math ops.
  if (bodyOp == "math.absf") {
    if (dw <= 32) return fromFloat(std::fabs(toFloat(a)));
    return fromDouble(std::fabs(toDouble(a)));
  }
  if (bodyOp == "math.cos") {
    if (dw <= 32) return fromFloat(std::cos(toFloat(a)));
    return fromDouble(std::cos(toDouble(a)));
  }
  if (bodyOp == "math.exp") {
    if (dw <= 32) return fromFloat(std::exp(toFloat(a)));
    return fromDouble(std::exp(toDouble(a)));
  }
  if (bodyOp == "math.fma") {
    uint64_t c = operands.size() > 2 ? operands[2] : 0;
    if (dw <= 32)
      return fromFloat(std::fma(toFloat(a), toFloat(b), toFloat(c)));
    return fromDouble(std::fma(toDouble(a), toDouble(b), toDouble(c)));
  }
  if (bodyOp == "math.log2") {
    if (dw <= 32) return fromFloat(std::log2(toFloat(a)));
    return fromDouble(std::log2(toDouble(a)));
  }
  if (bodyOp == "math.sin") {
    if (dw <= 32) return fromFloat(std::sin(toFloat(a)));
    return fromDouble(std::sin(toDouble(a)));
  }
  if (bodyOp == "math.sqrt") {
    if (dw <= 32) return fromFloat(std::sqrt(toFloat(a)));
    return fromDouble(std::sqrt(toDouble(a)));
  }

  // LLVM ops.
  if (bodyOp == "llvm.intr.bitreverse") {
    uint64_t result = 0;
    for (unsigned i = 0; i < dw; ++i) {
      if (a & (1ULL << i))
        result |= (1ULL << (dw - 1 - i));
    }
    return result;
  }

  // Fallback: pass through.
  return mask(a);
}

void SimTemporalPE::evaluateCombinational() {
  firedThisCycle_ = false;
  firedInstruction_ = -1;

  // Accept incoming operands (tag matching).
  for (unsigned i = 0; i < numInputs_ && i < inputs.size(); ++i) {
    inputs[i]->ready = false; // Default: not ready.

    if (!inputs[i]->valid)
      continue;

    uint16_t inTag = inputs[i]->tag;

    // Find matching instruction.
    int matchIdx = -1;
    for (unsigned s = 0; s < numInstructions_; ++s) {
      if (instructions_[s].valid && instructions_[s].tag == inTag) {
        matchIdx = static_cast<int>(s);
        break;
      }
    }

    if (matchIdx < 0) {
      latchError(RtError::RT_TEMPORAL_PE_NO_MATCH);
      continue;
    }

    // Store operand in buffer.
    if (!sharedOperandBuffer_) {
      if (!perInsnBuf_.opValid[matchIdx][i]) {
        inputs[i]->ready = true; // Accept the operand.
      }
      // Backpressure if slot already occupied.
    } else {
      // Shared buffer: ready check must mirror advanceClock insertion logic.
      // advanceClock targets the entry with LARGEST position for this tag.
      // If that entry's opValid[i]=0, it inserts there.
      // If that entry's opValid[i]=1, it creates a new entry.
      // So: ready if max-pos entry has opValid[i]=0, or buffer has capacity.
      bool canAccept = false;
      SharedBufEntry *maxPosEntry = nullptr;
      unsigned maxPos = 0;
      bool hasExisting = false;
      for (auto &entry : sharedBuf_) {
        if (entry.tag == inTag) {
          if (!hasExisting || entry.position > maxPos) {
            maxPos = entry.position;
            maxPosEntry = &entry;
          }
          hasExisting = true;
        }
      }
      if (hasExisting && maxPosEntry && !maxPosEntry->opValid[i]) {
        canAccept = true; // Can store in existing max-pos entry.
      } else if (sharedBuf_.size() < operandBufferSize_) {
        canAccept = true; // Can create new entry.
      }
      inputs[i]->ready = canAccept;
    }
  }

  // Check which instruction can fire (at most one per cycle).
  for (unsigned s = 0; s < numInstructions_; ++s) {
    if (canFire(s)) {
      firedInstruction_ = static_cast<int>(s);
      firedThisCycle_ = true;
      break;
    }
  }

  // Drive outputs for fired instruction.
  if (firedThisCycle_ && firedInstruction_ >= 0) {
    auto &insn = instructions_[static_cast<unsigned>(firedInstruction_)];

    // Gather operand values.
    std::vector<uint64_t> operandValues(numInputs_);
    if (!sharedOperandBuffer_) {
      for (unsigned i = 0; i < numInputs_; ++i) {
        if (insn.operands[i].isReg && insn.operands[i].regIdx < regFifos_.size())
          operandValues[i] = regFifos_[insn.operands[i].regIdx].data.front();
        else
          operandValues[i] = perInsnBuf_.opValue[firedInstruction_][i];
      }
    } else {
      // Find the position-0 entry with matching tag.
      for (auto &entry : sharedBuf_) {
        if (entry.tag == insn.tag && entry.position == 0) {
          for (unsigned i = 0; i < numInputs_; ++i) {
            if (insn.operands[i].isReg && insn.operands[i].regIdx < regFifos_.size())
              operandValues[i] = regFifos_[insn.operands[i].regIdx].data.front();
            else
              operandValues[i] = entry.opValue[i];
          }
          break;
        }
      }
    }

    // Execute FU.
    uint64_t result = executeFU(insn.opcode, operandValues);

    // Drive output channels.
    for (unsigned o = 0; o < numOutputs_ && o < outputs.size(); ++o) {
      if (!insn.results[o].isReg) {
        outputs[o]->valid = true;
        outputs[o]->data = result;
        outputs[o]->tag = insn.results[o].tag;
        outputs[o]->hasTag = true;
      }
    }
  } else {
    for (unsigned o = 0; o < outputs.size(); ++o)
      outputs[o]->valid = false;
  }
}

void SimTemporalPE::advanceClock() {
  // Accept operands that were handshaked.
  for (unsigned i = 0; i < numInputs_ && i < inputs.size(); ++i) {
    if (!inputs[i]->transferred())
      continue;

    uint16_t inTag = inputs[i]->tag;
    int matchIdx = -1;
    for (unsigned s = 0; s < numInstructions_; ++s) {
      if (instructions_[s].valid && instructions_[s].tag == inTag) {
        matchIdx = static_cast<int>(s);
        break;
      }
    }
    if (matchIdx < 0)
      continue;

    if (!sharedOperandBuffer_) {
      perInsnBuf_.opValid[matchIdx][i] = true;
      perInsnBuf_.opValue[matchIdx][i] = inputs[i]->data;
    } else {
      // Shared buffer: per spec, find entry with LARGEST position for this
      // tag. If that entry's op_valid[i]=0, store there. Otherwise create
      // new entry with position = max_position + 1.
      SharedBufEntry *maxPosEntry = nullptr;
      unsigned maxPos = 0;
      bool hasExisting = false;
      for (auto &entry : sharedBuf_) {
        if (entry.tag == inTag) {
          if (!hasExisting || entry.position > maxPos) {
            maxPos = entry.position;
            maxPosEntry = &entry;
          }
          hasExisting = true;
        }
      }

      SharedBufEntry *target = nullptr;
      if (hasExisting && maxPosEntry && !maxPosEntry->opValid[i]) {
        // Max-position entry has open slot for this operand.
        target = maxPosEntry;
      } else if (sharedBuf_.size() < operandBufferSize_) {
        // Create new entry.
        sharedBuf_.emplace_back();
        target = &sharedBuf_.back();
        target->tag = inTag;
        target->position = hasExisting ? maxPos + 1 : 0;
        target->opValid.resize(numInputs_, false);
        target->opValue.resize(numInputs_, 0);
      }
      if (target) {
        target->opValid[i] = true;
        target->opValue[i] = inputs[i]->data;
      }
    }
    perf_.tokensIn++;
  }

  // Process fired instruction.
  if (firedThisCycle_ && firedInstruction_ >= 0) {
    auto &insn = instructions_[static_cast<unsigned>(firedInstruction_)];
    uint64_t result = 0;

    // Gather operands again for register writes.
    std::vector<uint64_t> operandValues(numInputs_);
    if (!sharedOperandBuffer_) {
      for (unsigned i = 0; i < numInputs_; ++i) {
        if (insn.operands[i].isReg && insn.operands[i].regIdx < regFifos_.size())
          operandValues[i] = regFifos_[insn.operands[i].regIdx].data.front();
        else
          operandValues[i] = perInsnBuf_.opValue[firedInstruction_][i];
      }
      result = executeFU(insn.opcode, operandValues);

      // Clear operand buffer for fired instruction.
      for (unsigned i = 0; i < numInputs_; ++i)
        perInsnBuf_.opValid[firedInstruction_][i] = false;
    } else {
      // Remove position-0 entry.
      for (auto it = sharedBuf_.begin(); it != sharedBuf_.end(); ++it) {
        if (it->tag == insn.tag && it->position == 0) {
          for (unsigned i = 0; i < numInputs_; ++i) {
            if (insn.operands[i].isReg && insn.operands[i].regIdx < regFifos_.size())
              operandValues[i] = regFifos_[insn.operands[i].regIdx].data.front();
            else
              operandValues[i] = it->opValue[i];
          }
          result = executeFU(insn.opcode, operandValues);

          sharedBuf_.erase(it);
          // Decrement positions for remaining entries with same tag.
          for (auto &entry : sharedBuf_) {
            if (entry.tag == insn.tag && entry.position > 0)
              entry.position--;
          }
          break;
        }
      }
    }

    // Write results to registers or outputs.
    for (unsigned o = 0; o < numOutputs_; ++o) {
      if (insn.results[o].isReg && insn.results[o].regIdx < regFifos_.size()) {
        regFifos_[insn.results[o].regIdx].data.push_back(result);
      }
      if (!insn.results[o].isReg && o < outputs.size() &&
          outputs[o]->transferred())
        perf_.tokensOut++;
    }

    // Consume register operands.
    for (unsigned i = 0; i < numInputs_; ++i) {
      if (insn.operands[i].isReg && insn.operands[i].regIdx < regFifos_.size()) {
        auto &rf = regFifos_[insn.operands[i].regIdx];
        if (!rf.data.empty()) {
          rf.readersRemaining--;
          if (rf.readersRemaining == 0) {
            rf.data.pop_front();
            rf.readersRemaining = rf.totalReaders;
          }
        }
      }
    }
  }
}

void SimTemporalPE::collectTraceEvents(std::vector<TraceEvent> &events,
                                         uint64_t cycle) {
  if (firedThisCycle_) {
    perf_.activeCycles++;
    TraceEvent ev;
    ev.cycle = cycle;
    ev.hwNodeId = hwNodeId;
    ev.eventKind = EV_NODE_FIRE;
    ev.arg0 = static_cast<uint32_t>(firedInstruction_);
    events.push_back(ev);
  } else {
    bool anyInputValid = false;
    for (auto *in : inputs) {
      if (in && in->valid) {
        anyInputValid = true;
        break;
      }
    }
    if (anyInputValid)
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
