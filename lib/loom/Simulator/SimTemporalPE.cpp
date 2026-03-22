#include "loom/Simulator/SimTemporalPE.h"

#include "loom/Simulator/SimFunctionUnit.h"

#include <algorithm>
#include <cstdint>
#include <deque>
#include <limits>
#include <optional>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

namespace loom {
namespace sim {

namespace {

int64_t getIntAttr(const StaticModuleDesc &module, const char *name,
                   int64_t defaultValue = 0) {
  for (const auto &attr : module.intAttrs) {
    if (attr.name == name)
      return attr.value;
  }
  return defaultValue;
}

unsigned bitWidthForChoices(unsigned count) {
  if (count <= 1)
    return 0;
  unsigned value = count - 1u;
  unsigned bits = 0;
  while (value != 0) {
    ++bits;
    value >>= 1;
  }
  return bits;
}

uint64_t readBits(const std::vector<uint32_t> &words, unsigned &bitPos,
                  unsigned width) {
  uint64_t value = 0;
  for (unsigned bit = 0; bit < width; ++bit) {
    unsigned wordIdx = bitPos / 32;
    unsigned wordBit = bitPos % 32;
    if (wordIdx < words.size() && ((words[wordIdx] >> wordBit) & 1u) != 0)
      value |= (uint64_t{1} << bit);
    ++bitPos;
  }
  return value;
}

unsigned getPortOrdinal(const StaticModuleDesc &module, IdIndex portId,
                        StaticPortDirection direction) {
  const std::vector<IdIndex> &ports =
      (direction == StaticPortDirection::Input) ? module.inputPorts
                                                : module.outputPorts;
  for (unsigned idx = 0; idx < ports.size(); ++idx) {
    if (ports[idx] == portId)
      return idx;
  }
  return std::numeric_limits<unsigned>::max();
}

struct DecodedMuxField {
  uint64_t sel = 0;
  bool discard = false;
  bool disconnect = true;
};

struct ResultConfig {
  uint64_t tag = 0;
  uint64_t regIdx = 0;
  bool isReg = false;
};

struct SlotConfig {
  bool valid = false;
  uint16_t tag = 0;
  unsigned opcode = 0;
  IdIndex fuNodeId = INVALID_ID;
  std::vector<std::optional<unsigned>> operandRegs;
  std::vector<DecodedMuxField> inputMuxFields;
  std::vector<DecodedMuxField> outputMuxFields;
  std::vector<ResultConfig> resultConfigs;
};

struct InputBinding {
  unsigned ingressIndex = 0;
  IdIndex representativeExternalPort = INVALID_ID;
  uint32_t hwEdgeId = std::numeric_limits<uint32_t>::max();
};

struct OutputCandidate {
  IdIndex fuNodeId = INVALID_ID;
  unsigned fuOrdinal = 0;
  unsigned fuOutputOrdinal = 0;
  unsigned peOutputIndex = 0;
  IdIndex externalSrcPort = INVALID_ID;
};

struct SharedOperandEntry {
  std::vector<std::optional<SimToken>> operands;
};

class TemporalPEModule final : public SimModule {
public:
  TemporalPEModule(const StaticPEDesc &pe, const StaticMappedModel &model)
      : pe_(pe) {
    hwNodeId = pe.fuNodeIds.empty() ? 0 : static_cast<uint32_t>(pe.fuNodeIds.front());
    name = pe.peName;
    kind = StaticModuleKind::TemporalPE;

    buildFUs(model);
    buildExternalBindings(model);

    outputs.reserve(outputCandidates_.size());
    inputs.reserve(inputBindings_.size());
    slotOperandBuffers_.resize(pe_.numInstruction);
    for (auto &row : slotOperandBuffers_)
      row.resize(maxFuInputs_);
    registerFiles_.resize(pe_.numRegister);
    rrPointer_.assign(pe_.numOutputPorts, 0);
    fuActiveSlot_.assign(fus_.size(), std::nullopt);
    fuOutputGranted_.assign(fus_.size(), {});
    for (size_t fuIdx = 0; fuIdx < fus_.size(); ++fuIdx)
      fuOutputGranted_[fuIdx].assign(fuOutputChannels_[fuIdx].size(), false);
  }

  bool isCombinational() const override { return false; }

  void reset() override {
    resetDynamicState();
  }

  void configure(const std::vector<uint32_t> &configWords) override {
    pendingRuntimeError_.clear();
    slots_.clear();
    slots_.resize(pe_.numInstruction);

    unsigned opcodeBits = bitWidthForChoices(static_cast<unsigned>(pe_.fuNodeIds.size()));
    unsigned regIdxBits = bitWidthForChoices(pe_.numRegister);
    unsigned inputSelBits = bitWidthForChoices(pe_.numInputPorts);
    unsigned outputSelBits = bitWidthForChoices(pe_.numOutputPorts);
    unsigned bitPos = 0;

    std::unordered_map<uint16_t, unsigned> tagToSlot;
    for (unsigned slotIdx = 0; slotIdx < pe_.numInstruction; ++slotIdx) {
      SlotConfig slot;
      slot.valid = readBits(configWords, bitPos, 1) != 0;
      slot.tag = static_cast<uint16_t>(readBits(configWords, bitPos, pe_.tagWidth));
      slot.opcode = static_cast<unsigned>(readBits(configWords, bitPos, opcodeBits));
      if (slot.valid && slot.opcode < pe_.fuNodeIds.size())
        slot.fuNodeId = pe_.fuNodeIds[slot.opcode];
      slot.operandRegs.resize(maxFuInputs_);
      slot.inputMuxFields.resize(maxFuInputs_);
      slot.outputMuxFields.resize(maxFuOutputs_);
      slot.resultConfigs.resize(maxFuOutputs_);

      for (unsigned operandIdx = 0; operandIdx < maxFuInputs_; ++operandIdx) {
        if (pe_.numRegister == 0)
          continue;
        uint64_t regIdx = readBits(configWords, bitPos, regIdxBits);
        bool isReg = readBits(configWords, bitPos, 1) != 0;
        if (isReg)
          slot.operandRegs[operandIdx] = static_cast<unsigned>(regIdx);
      }

      for (unsigned inputIdx = 0; inputIdx < maxFuInputs_; ++inputIdx) {
        DecodedMuxField field;
        field.sel = readBits(configWords, bitPos, inputSelBits);
        field.discard = readBits(configWords, bitPos, 1) != 0;
        field.disconnect = readBits(configWords, bitPos, 1) != 0;
        slot.inputMuxFields[inputIdx] = field;
      }

      for (unsigned outputIdx = 0; outputIdx < maxFuOutputs_; ++outputIdx) {
        DecodedMuxField field;
        field.sel = readBits(configWords, bitPos, outputSelBits);
        field.discard = readBits(configWords, bitPos, 1) != 0;
        field.disconnect = readBits(configWords, bitPos, 1) != 0;
        slot.outputMuxFields[outputIdx] = field;
      }

      for (unsigned resultIdx = 0; resultIdx < maxFuOutputs_; ++resultIdx) {
        ResultConfig result;
        result.tag = readBits(configWords, bitPos, pe_.tagWidth);
        if (pe_.numRegister > 0) {
          result.regIdx = readBits(configWords, bitPos, regIdxBits);
          result.isReg = readBits(configWords, bitPos, 1) != 0;
        }
        slot.resultConfigs[resultIdx] = result;
      }

      if (slot.valid) {
        if (tagToSlot.count(slot.tag) != 0)
          pendingRuntimeError_ = "duplicate temporal_pe slot tags";
        else
          tagToSlot[slot.tag] = slotIdx;
      }
      slots_[slotIdx] = std::move(slot);
    }

    for (size_t fuIdx = 0; fuIdx < fus_.size(); ++fuIdx) {
      const StaticModuleDesc &fuDesc = *fuDescs_[fuIdx];
      unsigned fuConfigBits = static_cast<unsigned>(
          std::max<int64_t>(0, getIntAttr(fuDesc, "fu_config_bits", 0)));
      std::vector<uint32_t> fuWords;
      if (fuConfigBits != 0) {
        unsigned startBit = bitPos;
        unsigned outBit = 0;
        fuWords.resize((fuConfigBits + 31) / 32, 0);
        for (unsigned copied = 0; copied < fuConfigBits; ++copied) {
          uint64_t bit = readBits(configWords, startBit, 1);
          unsigned wordIdx = outBit / 32;
          unsigned wordBit = outBit % 32;
          if (bit != 0)
            fuWords[wordIdx] |= (1u << wordBit);
          ++outBit;
        }
        bitPos += fuConfigBits;
      }
      fus_[fuIdx]->configure(fuWords);
    }

    resetDynamicState();
  }

private:
  void resetDynamicState() {
    perf_ = PerfSnapshot();
    rrPointer_.assign(pe_.numOutputPorts, 0);
    for (auto &row : slotOperandBuffers_) {
      for (auto &operand : row)
        operand.reset();
    }
    sharedOperandQueues_.clear();
    for (auto &fifo : registerFiles_)
      fifo.clear();
    for (auto &active : fuActiveSlot_)
      active.reset();
    for (auto &fu : fus_)
      fu->reset();
    for (auto &grants : fuOutputGranted_)
      std::fill(grants.begin(), grants.end(), false);
    for (auto &channels : fuInputChannels_) {
      for (auto &channel : channels)
        channel = SimChannel();
    }
    for (auto &channels : fuOutputChannels_) {
      for (auto &channel : channels)
        channel = SimChannel();
    }
  }

public:
  void bindRuntimeServices(SimRuntimeServices *services) override {
    runtimeServices_ = services;
    for (auto &fu : fus_)
      fu->bindRuntimeServices(services);
  }

  void evaluate() override {
    clearExternalOutputs();
    clearLocalChannels();

    for (unsigned ingressIdx = 0; ingressIdx < inputBindings_.size(); ++ingressIdx) {
      if (ingressIdx >= inputs.size())
        break;
      SimChannel *input = inputs[ingressIdx];
      input->ready = input->valid && canAcceptIngressToken(ingressIdx, tokenFromChannel(*input));
    }

    std::optional<unsigned> selectedSlot = selectReadySlot();
    if (selectedSlot.has_value()) {
      unsigned slotIdx = *selectedSlot;
      const SlotConfig &slot = slots_[slotIdx];
      auto it = fuIndexByNode_.find(slot.fuNodeId);
      if (it != fuIndexByNode_.end()) {
        unsigned fuIdx = it->second;
        fuActiveSlot_[fuIdx] = slotIdx;
        const StaticModuleDesc &fuDesc = *fuDescs_[fuIdx];
        auto regTokens = gatherRegisterOperands(slot, fuDesc);
        for (unsigned operandIdx = 0; operandIdx < fuDesc.inputPorts.size();
             ++operandIdx) {
          std::optional<SimToken> token;
          if (operandIdx < slot.operandRegs.size() &&
              slot.operandRegs[operandIdx].has_value()) {
            auto regIt = regTokens.find(*slot.operandRegs[operandIdx]);
            if (regIt != regTokens.end())
              token = regIt->second;
          } else {
            token = peekOperandToken(slotIdx, operandIdx);
          }
          if (token.has_value()) {
            driveChannelFromToken(fuInputChannels_[fuIdx][operandIdx], *token);
            fuInputChannels_[fuIdx][operandIdx].ready = true;
          }
        }
        selectedSlotThisCycle_ = slotIdx;
        selectedFUThisCycle_ = fuIdx;
      }
    }

    for (size_t fuIdx = 0; fuIdx < fus_.size(); ++fuIdx) {
      if (fuActiveSlot_[fuIdx].has_value()) {
        const SlotConfig &slot = slots_[*fuActiveSlot_[fuIdx]];
        for (unsigned outputIdx = 0; outputIdx < fuOutputChannels_[fuIdx].size();
             ++outputIdx) {
          if (outputIdx >= slot.resultConfigs.size())
            continue;
          const ResultConfig &result = slot.resultConfigs[outputIdx];
          const DecodedMuxField &outField = slot.outputMuxFields[outputIdx];
          if (result.isReg) {
            bool room = result.regIdx < registerFiles_.size() &&
                        registerFiles_[result.regIdx].size() < pe_.regFifoDepth;
            fuOutputChannels_[fuIdx][outputIdx].ready = room;
          } else if (outField.discard && !outField.disconnect) {
            fuOutputChannels_[fuIdx][outputIdx].ready = true;
          }
        }
      }
      fus_[fuIdx]->evaluate();
    }

    arbitrateExternalOutputs();
  }

  void commit() override {
    captureIngressTransfers();

    std::vector<uint64_t> fireCountsBefore;
    fireCountsBefore.reserve(fus_.size());
    for (const auto &fu : fus_)
      fireCountsBefore.push_back(fu->getLogicalFireCount());

    for (size_t fuIdx = 0; fuIdx < fus_.size(); ++fuIdx) {
      for (auto &channel : fuInputChannels_[fuIdx])
        channel.didTransfer = channel.valid && channel.ready;
      for (auto &channel : fuOutputChannels_[fuIdx])
        channel.didTransfer = channel.valid && channel.ready;
    }

    for (auto &fu : fus_)
      fu->commit();

    finalizeFires(fireCountsBefore);
    finalizeOutputDrains();

    selectedSlotThisCycle_.reset();
    selectedFUThisCycle_.reset();
  }

  bool hasPendingWork() const override {
    if (!pendingRuntimeError_.empty())
      return true;
    for (const auto &fu : fus_) {
      if (fu->hasPendingWork())
        return true;
    }
    for (const auto &row : slotOperandBuffers_) {
      for (const auto &operand : row) {
        if (operand.has_value())
          return true;
      }
    }
    for (const auto &entry : sharedOperandQueues_) {
      for (const auto &row : entry.second) {
        for (const auto &operand : row.operands) {
          if (operand.has_value())
            return true;
        }
      }
    }
    for (const auto &fifo : registerFiles_) {
      if (!fifo.empty())
        return true;
    }
    return false;
  }

  void collectTraceEvents(std::vector<TraceEvent> &events,
                          uint64_t cycle) override {
    for (auto &fu : fus_)
      fu->collectTraceEvents(events, cycle);
  }

  PerfSnapshot getPerfSnapshot() const override {
    PerfSnapshot out = perf_;
    for (const auto &fu : fus_) {
      PerfSnapshot perf = fu->getPerfSnapshot();
      out.activeCycles += perf.activeCycles;
      out.stallCyclesIn += perf.stallCyclesIn;
      out.stallCyclesOut += perf.stallCyclesOut;
      out.tokensIn += perf.tokensIn;
      out.tokensOut += perf.tokensOut;
    }
    return out;
  }

  void debugDump(std::ostream &os) const override {
    os << "      temporal_pe slots=" << slots_.size()
       << " pending_error=" << (!pendingRuntimeError_.empty())
       << " shared=" << pe_.enableShareOperandBuffer
       << " fus=" << fus_.size()
       << " max_in=" << maxFuInputs_
       << " max_out=" << maxFuOutputs_ << "\n";
    if (!pendingRuntimeError_.empty())
      os << "      error=" << pendingRuntimeError_ << "\n";
    for (unsigned slotIdx = 0; slotIdx < slots_.size(); ++slotIdx) {
      const SlotConfig &slot = slots_[slotIdx];
      os << "      slot[" << slotIdx << "] valid=" << slot.valid
         << " tag=" << slot.tag << " opcode=" << slot.opcode
         << " fu=" << slot.fuNodeId << "\n";
      for (unsigned operandIdx = 0; operandIdx < slot.inputMuxFields.size();
           ++operandIdx) {
        const DecodedMuxField &field = slot.inputMuxFields[operandIdx];
        os << "        in_mux[" << operandIdx << "] sel=" << field.sel
           << " discard=" << field.discard
           << " disconnect=" << field.disconnect;
        if (operandIdx < slot.operandRegs.size() &&
            slot.operandRegs[operandIdx].has_value()) {
          os << " reg=" << *slot.operandRegs[operandIdx];
        }
        os << "\n";
      }
      for (unsigned outputIdx = 0; outputIdx < slot.outputMuxFields.size();
           ++outputIdx) {
        const DecodedMuxField &field = slot.outputMuxFields[outputIdx];
        os << "        out_mux[" << outputIdx << "] sel=" << field.sel
           << " discard=" << field.discard
           << " disconnect=" << field.disconnect;
        if (outputIdx < slot.resultConfigs.size()) {
          const ResultConfig &result = slot.resultConfigs[outputIdx];
          os << " result_tag=" << result.tag;
          if (result.isReg)
            os << " reg=" << result.regIdx;
        }
        os << "\n";
      }
    }
  }

  std::string getDebugStateSummary() const override {
    std::ostringstream ss;
    unsigned buffered = 0;
    for (const auto &row : slotOperandBuffers_) {
      for (const auto &operand : row)
        buffered += operand.has_value() ? 1u : 0u;
    }
    unsigned sharedRows = 0;
    for (const auto &entry : sharedOperandQueues_)
      sharedRows += static_cast<unsigned>(entry.second.size());
    unsigned regEntries = 0;
    for (const auto &fifo : registerFiles_)
      regEntries += static_cast<unsigned>(fifo.size());
    ss << "temporal_pe(buffers=" << buffered << ",shared_rows=" << sharedRows
       << ",reg_entries=" << regEntries << ")";
    return ss.str();
  }

  std::vector<NamedCounter> getDebugCounters() const override {
    std::vector<NamedCounter> counters;
    counters.push_back({"temporal_slots", static_cast<uint64_t>(slots_.size())});
    counters.push_back(
        {"temporal_register_count", static_cast<uint64_t>(registerFiles_.size())});
    return counters;
  }

private:
  void buildFUs(const StaticMappedModel &model) {
    std::unordered_map<IdIndex, const StaticModuleDesc *> modulesById;
    for (const auto &module : model.getModules())
      modulesById[module.hwNodeId] = &module;

    for (unsigned ordinal = 0; ordinal < pe_.fuNodeIds.size(); ++ordinal) {
      IdIndex fuNodeId = pe_.fuNodeIds[ordinal];
      auto it = modulesById.find(fuNodeId);
      if (it == modulesById.end())
        continue;
      const StaticModuleDesc *fuDesc = it->second;
      auto fu = createFunctionUnitModule(*fuDesc, model, true);
      if (!fu)
        continue;
      fuDescs_.push_back(fuDesc);
      fuIndexByNode_[fuNodeId] = fus_.size();
      fuOrdinalByNode_[fuNodeId] = ordinal;
      maxFuInputs_ = std::max<unsigned>(maxFuInputs_, fuDesc->inputPorts.size());
      maxFuOutputs_ = std::max<unsigned>(maxFuOutputs_, fuDesc->outputPorts.size());
      fu->hwNodeId = fuDesc->hwNodeId;
      fu->name = fuDesc->name;
      fu->kind = fuDesc->kind;
      fuInputChannels_.emplace_back(fuDesc->inputPorts.size());
      fuOutputChannels_.emplace_back(fuDesc->outputPorts.size());
      for (unsigned idx = 0; idx < fuInputChannels_.back().size(); ++idx)
        fu->inputs.push_back(&fuInputChannels_.back()[idx]);
      for (unsigned idx = 0; idx < fuOutputChannels_.back().size(); ++idx)
        fu->outputs.push_back(&fuOutputChannels_.back()[idx]);
      fu->bindRuntimeServices(runtimeServices_);
      fus_.push_back(std::move(fu));
    }
  }

  void buildExternalBindings(const StaticMappedModel &model) {
    std::unordered_map<unsigned, InputBinding> inputByIndex;
    for (const auto &channel : model.getChannels()) {
      if (fuIndexByNode_.count(channel.dstNode) != 0 && channel.peInputIndex >= 0) {
        unsigned ingressIdx = static_cast<unsigned>(channel.peInputIndex);
        auto it = inputByIndex.find(ingressIdx);
        if (it == inputByIndex.end() || channel.hwEdgeId < it->second.hwEdgeId) {
          InputBinding binding;
          binding.ingressIndex = ingressIdx;
          binding.representativeExternalPort = channel.srcPort;
          binding.hwEdgeId = channel.hwEdgeId;
          inputByIndex[ingressIdx] = std::move(binding);
        }
      }
      if (fuIndexByNode_.count(channel.srcNode) != 0 && channel.peOutputIndex >= 0) {
        auto fuIt = fuIndexByNode_.find(channel.srcNode);
        const StaticModuleDesc *fuDesc = fuDescs_[fuIt->second];
        unsigned outOrd =
            getPortOrdinal(*fuDesc, channel.srcPort, StaticPortDirection::Output);
        OutputCandidate candidate;
        candidate.fuNodeId = channel.srcNode;
        candidate.fuOrdinal = fuOrdinalByNode_[channel.srcNode];
        candidate.fuOutputOrdinal = outOrd;
        candidate.peOutputIndex = static_cast<unsigned>(channel.peOutputIndex);
        candidate.externalSrcPort = channel.srcPort;
        outputCandidates_.push_back(std::move(candidate));
      }
    }

    inputBindings_.resize(pe_.numInputPorts);
    for (unsigned idx = 0; idx < pe_.numInputPorts; ++idx) {
      auto it = inputByIndex.find(idx);
      inputBindings_[idx].ingressIndex = idx;
      inputBindings_[idx].representativeExternalPort =
          (it == inputByIndex.end()) ? INVALID_ID
                                     : it->second.representativeExternalPort;
    }
    std::sort(outputCandidates_.begin(), outputCandidates_.end(),
              [](const OutputCandidate &lhs, const OutputCandidate &rhs) {
                if (lhs.peOutputIndex != rhs.peOutputIndex)
                  return lhs.peOutputIndex < rhs.peOutputIndex;
                if (lhs.fuOrdinal != rhs.fuOrdinal)
                  return lhs.fuOrdinal < rhs.fuOrdinal;
                if (lhs.fuOutputOrdinal != rhs.fuOutputOrdinal)
                  return lhs.fuOutputOrdinal < rhs.fuOutputOrdinal;
                return lhs.externalSrcPort < rhs.externalSrcPort;
              });
  }

  void clearExternalOutputs() {
    for (SimChannel *output : outputs) {
      if (!output)
        continue;
      output->valid = false;
      output->data = 0;
      output->tag = 0;
      output->hasTag = false;
      output->generation = 0;
    }
  }

  void clearLocalChannels() {
    for (auto &channels : fuInputChannels_) {
      for (auto &channel : channels)
        channel = SimChannel();
    }
    for (auto &channels : fuOutputChannels_) {
      for (auto &channel : channels) {
        channel.ready = false;
        channel.didTransfer = false;
      }
    }
  }

  bool canAcceptIngressToken(unsigned ingressIdx, const SimToken &token) const {
    auto slotIdx = findSlotByTag(token.tag);
    if (!slotIdx.has_value())
      return false;
    const SlotConfig &slot = slots_[*slotIdx];
    bool hasExternalUse = false;
    if (pe_.enableShareOperandBuffer) {
      std::vector<unsigned> operandPositions = operandPositionsForIngress(slot, ingressIdx);
      if (operandPositions.empty())
        return false;
      hasExternalUse = true;
      return canAcceptSharedOperand(slot.tag, operandPositions);
    }

    for (unsigned operandIdx = 0; operandIdx < slot.inputMuxFields.size(); ++operandIdx) {
      if (operandIdx < slot.operandRegs.size() && slot.operandRegs[operandIdx].has_value())
        continue;
      const DecodedMuxField &field = slot.inputMuxFields[operandIdx];
      if (field.disconnect || field.discard || field.sel != ingressIdx)
        continue;
      hasExternalUse = true;
      if (operandIdx >= slotOperandBuffers_[*slotIdx].size() ||
          slotOperandBuffers_[*slotIdx][operandIdx].has_value())
        return false;
    }
    return hasExternalUse;
  }

  std::optional<unsigned> findSlotByTag(uint16_t tag) const {
    for (unsigned slotIdx = 0; slotIdx < slots_.size(); ++slotIdx) {
      if (slots_[slotIdx].valid && slots_[slotIdx].tag == tag)
        return slotIdx;
    }
    return std::nullopt;
  }

  std::vector<unsigned> operandPositionsForIngress(const SlotConfig &slot,
                                                   unsigned ingressIdx) const {
    std::vector<unsigned> positions;
    for (unsigned operandIdx = 0; operandIdx < slot.inputMuxFields.size(); ++operandIdx) {
      if (operandIdx < slot.operandRegs.size() && slot.operandRegs[operandIdx].has_value())
        continue;
      const DecodedMuxField &field = slot.inputMuxFields[operandIdx];
      if (!field.disconnect && !field.discard && field.sel == ingressIdx)
        positions.push_back(operandIdx);
    }
    return positions;
  }

  bool canAcceptSharedOperand(uint16_t tag,
                              const std::vector<unsigned> &positions) const {
    auto it = sharedOperandQueues_.find(tag);
    if (it == sharedOperandQueues_.end())
      return sharedOperandQueuesSize() < pe_.operandBufferSize;
    if (it->second.empty())
      return true;
    const SharedOperandEntry &back = it->second.back();
    bool allFree = true;
    for (unsigned pos : positions) {
      if (pos >= back.operands.size() || back.operands[pos].has_value()) {
        allFree = false;
        break;
      }
    }
    if (allFree)
      return true;
    return sharedOperandQueuesSize() < pe_.operandBufferSize;
  }

  size_t sharedOperandQueuesSize() const {
    size_t total = 0;
    for (const auto &entry : sharedOperandQueues_)
      total += entry.second.size();
    return total;
  }

  std::optional<unsigned> selectReadySlot() const {
    for (unsigned slotIdx = 0; slotIdx < slots_.size(); ++slotIdx) {
      const SlotConfig &slot = slots_[slotIdx];
      if (!slot.valid || slot.fuNodeId == INVALID_ID)
        continue;
      auto fuIt = fuIndexByNode_.find(slot.fuNodeId);
      if (fuIt == fuIndexByNode_.end())
        continue;
      unsigned fuIdx = fuIt->second;
      if (fuActiveSlot_[fuIdx].has_value() || fus_[fuIdx]->hasPendingWork())
        continue;
      const StaticModuleDesc &fuDesc = *fuDescs_[fuIdx];
      auto regTokens = gatherRegisterOperands(slot, fuDesc);
      bool ready = true;
      for (unsigned operandIdx = 0; operandIdx < fuDesc.inputPorts.size(); ++operandIdx) {
        if (operandIdx < slot.operandRegs.size() && slot.operandRegs[operandIdx].has_value()) {
          if (regTokens.find(*slot.operandRegs[operandIdx]) == regTokens.end()) {
            ready = false;
            break;
          }
          continue;
        }
        if (operandIdx < slot.inputMuxFields.size()) {
          const DecodedMuxField &field = slot.inputMuxFields[operandIdx];
          if (field.disconnect || field.discard)
            continue;
        }
        if (!peekOperandToken(slotIdx, operandIdx).has_value()) {
          ready = false;
          break;
        }
      }
      if (ready)
        return slotIdx;
    }
    return std::nullopt;
  }

  std::unordered_map<unsigned, SimToken>
  gatherRegisterOperands(const SlotConfig &slot,
                         const StaticModuleDesc &fuDesc) const {
    std::unordered_map<unsigned, SimToken> tokens;
    for (unsigned operandIdx = 0; operandIdx < fuDesc.inputPorts.size(); ++operandIdx) {
      if (operandIdx >= slot.operandRegs.size() || !slot.operandRegs[operandIdx].has_value())
        continue;
      unsigned regIdx = *slot.operandRegs[operandIdx];
      if (regIdx >= registerFiles_.size() || registerFiles_[regIdx].empty())
        continue;
      tokens.try_emplace(regIdx, registerFiles_[regIdx].front());
    }
    return tokens;
  }

  std::optional<SimToken> peekOperandToken(unsigned slotIdx,
                                           unsigned operandIdx) const {
    if (slotIdx >= slots_.size())
      return std::nullopt;
    const SlotConfig &slot = slots_[slotIdx];
    if (pe_.enableShareOperandBuffer) {
      auto it = sharedOperandQueues_.find(slot.tag);
      if (it == sharedOperandQueues_.end() || it->second.empty())
        return std::nullopt;
      const SharedOperandEntry &entry = it->second.front();
      if (operandIdx >= entry.operands.size())
        return std::nullopt;
      return entry.operands[operandIdx];
    }
    if (slotIdx >= slotOperandBuffers_.size() ||
        operandIdx >= slotOperandBuffers_[slotIdx].size())
      return std::nullopt;
    return slotOperandBuffers_[slotIdx][operandIdx];
  }

  void arbitrateExternalOutputs() {
    for (auto &grants : fuOutputGranted_)
      std::fill(grants.begin(), grants.end(), false);

    for (unsigned peOutIdx = 0; peOutIdx < pe_.numOutputPorts; ++peOutIdx) {
      std::vector<const OutputCandidate *> candidates;
      for (const auto &candidate : outputCandidates_) {
        if (candidate.peOutputIndex != peOutIdx)
          continue;
        auto fuIt = fuIndexByNode_.find(candidate.fuNodeId);
        if (fuIt == fuIndexByNode_.end())
          continue;
        unsigned fuIdx = fuIt->second;
        if (!fuActiveSlot_[fuIdx].has_value())
          continue;
        if (candidate.fuOutputOrdinal >= fuOutputChannels_[fuIdx].size())
          continue;
        const SimChannel &localOut = fuOutputChannels_[fuIdx][candidate.fuOutputOrdinal];
        if (!localOut.valid)
          continue;
        const SlotConfig &slot = slots_[*fuActiveSlot_[fuIdx]];
        if (candidate.fuOutputOrdinal >= slot.outputMuxFields.size())
          continue;
        const DecodedMuxField &outField =
            slot.outputMuxFields[candidate.fuOutputOrdinal];
        if (outField.disconnect || outField.discard ||
            outField.sel != candidate.peOutputIndex)
          continue;
        candidates.push_back(&candidate);
      }
      if (candidates.empty())
        continue;

      std::sort(candidates.begin(), candidates.end(),
                [](const OutputCandidate *lhs, const OutputCandidate *rhs) {
                  if (lhs->fuOrdinal != rhs->fuOrdinal)
                    return lhs->fuOrdinal < rhs->fuOrdinal;
                  return lhs->fuOutputOrdinal < rhs->fuOutputOrdinal;
                });

      const OutputCandidate *granted = nullptr;
      unsigned rrStart = (peOutIdx < rrPointer_.size()) ? rrPointer_[peOutIdx] : 0;
      for (unsigned step = 0; step < pe_.fuNodeIds.size() && !granted; ++step) {
        unsigned wantedOrdinal = (rrStart + step) % std::max<unsigned>(1, pe_.fuNodeIds.size());
        for (const OutputCandidate *candidate : candidates) {
          if (candidate->fuOrdinal == wantedOrdinal) {
            granted = candidate;
            break;
          }
        }
      }
      if (!granted)
        granted = candidates.front();

      auto fuIt = fuIndexByNode_.find(granted->fuNodeId);
      unsigned fuIdx = fuIt->second;
      const SimChannel &localOut = fuOutputChannels_[fuIdx][granted->fuOutputOrdinal];
      const SlotConfig &slot = slots_[*fuActiveSlot_[fuIdx]];
      const ResultConfig &result = slot.resultConfigs[granted->fuOutputOrdinal];
      size_t outVecIdx = outputVectorIndex(granted->externalSrcPort);
      if (outVecIdx >= outputs.size() || outputs[outVecIdx] == nullptr)
        continue;
      SimChannel *externalOut = outputs[outVecIdx];
      externalOut->valid = true;
      externalOut->data = localOut.data;
      externalOut->tag = static_cast<uint16_t>(result.tag);
      externalOut->hasTag = pe_.tagWidth != 0;
      externalOut->generation = localOut.generation;
      fuOutputChannels_[fuIdx][granted->fuOutputOrdinal].ready = externalOut->ready;
      fuOutputGranted_[fuIdx][granted->fuOutputOrdinal] = true;
    }
  }

  size_t outputVectorIndex(IdIndex portId) const {
    for (size_t idx = 0; idx < outputCandidates_.size(); ++idx) {
      if (outputCandidates_[idx].externalSrcPort == portId)
        return idx;
    }
    return outputs.size();
  }

  void captureIngressTransfers() {
    for (unsigned ingressIdx = 0; ingressIdx < inputBindings_.size(); ++ingressIdx) {
      if (ingressIdx >= inputs.size() || inputs[ingressIdx] == nullptr)
        continue;
      SimChannel *input = inputs[ingressIdx];
      if (!input->transferred())
        continue;
      SimToken token = tokenFromChannel(*input);
      auto slotIdx = findSlotByTag(token.tag);
      if (!slotIdx.has_value())
        continue;
      const SlotConfig &slot = slots_[*slotIdx];
      std::vector<unsigned> positions = operandPositionsForIngress(slot, ingressIdx);
      if (positions.empty())
        continue;
      if (pe_.enableShareOperandBuffer) {
        auto &queue = sharedOperandQueues_[slot.tag];
        bool appendRow = true;
        if (!queue.empty()) {
          appendRow = false;
          for (unsigned pos : positions) {
            if (pos >= queue.back().operands.size() ||
                queue.back().operands[pos].has_value()) {
              appendRow = true;
              break;
            }
          }
        }
        if (appendRow) {
          SharedOperandEntry entry;
          entry.operands.resize(maxFuInputs_);
          queue.push_back(std::move(entry));
        }
        for (unsigned pos : positions)
          queue.back().operands[pos] = token;
      } else {
        for (unsigned pos : positions)
          slotOperandBuffers_[*slotIdx][pos] = token;
      }
    }
  }

  void finalizeFires(const std::vector<uint64_t> &fireCountsBefore) {
    for (size_t fuIdx = 0; fuIdx < fus_.size(); ++fuIdx) {
      if (fus_[fuIdx]->getLogicalFireCount() == fireCountsBefore[fuIdx])
        continue;
      if (!fuActiveSlot_[fuIdx].has_value())
        continue;
      unsigned slotIdx = *fuActiveSlot_[fuIdx];
      const SlotConfig &slot = slots_[slotIdx];
      const StaticModuleDesc &fuDesc = *fuDescs_[fuIdx];
      if (pe_.enableShareOperandBuffer) {
        auto it = sharedOperandQueues_.find(slot.tag);
        if (it != sharedOperandQueues_.end() && !it->second.empty())
          it->second.pop_front();
      } else {
        for (unsigned operandIdx = 0; operandIdx < fuDesc.inputPorts.size(); ++operandIdx) {
          if (operandIdx < slot.operandRegs.size() && slot.operandRegs[operandIdx].has_value())
            continue;
          if (operandIdx < slotOperandBuffers_[slotIdx].size())
            slotOperandBuffers_[slotIdx][operandIdx].reset();
        }
      }
      std::unordered_set<unsigned> consumedRegs;
      for (unsigned operandIdx = 0; operandIdx < fuDesc.inputPorts.size(); ++operandIdx) {
        if (operandIdx >= slot.operandRegs.size() || !slot.operandRegs[operandIdx].has_value())
          continue;
        consumedRegs.insert(*slot.operandRegs[operandIdx]);
      }
      for (unsigned regIdx : consumedRegs) {
        if (regIdx < registerFiles_.size() && !registerFiles_[regIdx].empty())
          registerFiles_[regIdx].pop_front();
      }
    }
  }

  void finalizeOutputDrains() {
    for (size_t fuIdx = 0; fuIdx < fus_.size(); ++fuIdx) {
      if (!fuActiveSlot_[fuIdx].has_value())
        continue;
      unsigned slotIdx = *fuActiveSlot_[fuIdx];
      const SlotConfig &slot = slots_[slotIdx];
      for (unsigned outputIdx = 0; outputIdx < fuOutputChannels_[fuIdx].size();
           ++outputIdx) {
        SimChannel &localOut = fuOutputChannels_[fuIdx][outputIdx];
        if (!localOut.transferred())
          continue;
        const ResultConfig &result = slot.resultConfigs[outputIdx];
        const DecodedMuxField &outField = slot.outputMuxFields[outputIdx];
        if (result.isReg) {
          if (result.regIdx < registerFiles_.size())
            registerFiles_[result.regIdx].push_back(tokenFromChannel(localOut));
        } else if (!outField.disconnect && !outField.discard &&
                   outputIdx < fuOutputGranted_[fuIdx].size() &&
                   fuOutputGranted_[fuIdx][outputIdx]) {
          unsigned peOutIdx = static_cast<unsigned>(outField.sel);
          if (peOutIdx < rrPointer_.size())
            rrPointer_[peOutIdx] = (fuOrdinalByNode_[slot.fuNodeId] + 1) %
                                   std::max<unsigned>(1, pe_.fuNodeIds.size());
        }
      }
      if (!fus_[fuIdx]->hasPendingWork())
        fuActiveSlot_[fuIdx].reset();
    }
  }

  const StaticPEDesc pe_;
  SimRuntimeServices *runtimeServices_ = nullptr;
  unsigned maxFuInputs_ = 0;
  unsigned maxFuOutputs_ = 0;
  std::vector<const StaticModuleDesc *> fuDescs_;
  std::unordered_map<IdIndex, unsigned> fuIndexByNode_;
  std::unordered_map<IdIndex, unsigned> fuOrdinalByNode_;
  std::vector<std::unique_ptr<SimModule>> fus_;
  std::vector<std::vector<SimChannel>> fuInputChannels_;
  std::vector<std::vector<SimChannel>> fuOutputChannels_;
  std::vector<InputBinding> inputBindings_;
  std::vector<OutputCandidate> outputCandidates_;
  std::vector<SlotConfig> slots_;
  std::vector<std::vector<std::optional<SimToken>>> slotOperandBuffers_;
  std::unordered_map<uint16_t, std::deque<SharedOperandEntry>> sharedOperandQueues_;
  std::vector<std::deque<SimToken>> registerFiles_;
  std::vector<unsigned> rrPointer_;
  std::vector<std::optional<unsigned>> fuActiveSlot_;
  std::vector<std::vector<bool>> fuOutputGranted_;
  std::optional<unsigned> selectedSlotThisCycle_;
  std::optional<unsigned> selectedFUThisCycle_;
  std::string pendingRuntimeError_;
};

} // namespace

std::vector<IdIndex> temporalPEInputRepresentativePorts(
    const StaticPEDesc &pe, const StaticMappedModel &model) {
  std::vector<IdIndex> ports(pe.numInputPorts, INVALID_ID);
  std::vector<uint32_t> bestHwEdge(pe.numInputPorts,
                                   std::numeric_limits<uint32_t>::max());
  std::unordered_set<IdIndex> fuSet(pe.fuNodeIds.begin(), pe.fuNodeIds.end());
  for (const auto &channel : model.getChannels()) {
    if (fuSet.count(channel.dstNode) == 0 || channel.peInputIndex < 0)
      continue;
    unsigned ingressIdx = static_cast<unsigned>(channel.peInputIndex);
    if (ingressIdx >= ports.size())
      continue;
    if (channel.hwEdgeId < bestHwEdge[ingressIdx]) {
      bestHwEdge[ingressIdx] = channel.hwEdgeId;
      ports[ingressIdx] = channel.srcPort;
    }
  }
  return ports;
}

std::vector<IdIndex> temporalPEOutputCandidatePorts(const StaticPEDesc &pe,
                                                    const StaticMappedModel &model) {
  std::vector<IdIndex> ports;
  std::unordered_set<IdIndex> fuSet(pe.fuNodeIds.begin(), pe.fuNodeIds.end());
  for (const auto &channel : model.getChannels()) {
    if (fuSet.count(channel.srcNode) == 0 || channel.peOutputIndex < 0)
      continue;
    ports.push_back(channel.srcPort);
  }
  std::sort(ports.begin(), ports.end());
  ports.erase(std::unique(ports.begin(), ports.end()), ports.end());
  return ports;
}

std::unique_ptr<SimModule> createTemporalPEModule(const StaticPEDesc &pe,
                                                  const StaticMappedModel &model) {
  if (pe.peKind != "temporal_pe")
    return nullptr;
  return std::make_unique<TemporalPEModule>(pe, model);
}

} // namespace sim
} // namespace loom
