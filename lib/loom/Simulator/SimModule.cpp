#include "loom/Simulator/SimModule.h"
#include "loom/Simulator/SimFunctionUnit.h"
#include "loom/Simulator/SimMemory.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <limits>
#include <optional>
#include <ostream>
#include <utility>

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

const std::vector<std::string> *
getStringArrayAttr(const StaticModuleDesc &module, const char *name) {
  for (const auto &attr : module.stringArrayAttrs) {
    if (attr.name == name)
      return &attr.value;
  }
  return nullptr;
}

const StaticPortDesc *findPortById(const StaticMappedModel &model, IdIndex portId) {
  return model.findPort(portId);
}

std::vector<bool> decodeConnectivityFlat(const StaticModuleDesc &module,
                                         unsigned numInputs,
                                         unsigned numOutputs) {
  std::vector<bool> connectivity(numInputs * numOutputs, true);
  const auto *rows = getStringArrayAttr(module, "connectivity_table");
  if (!rows || rows->size() != numOutputs)
    return connectivity;

  connectivity.assign(numInputs * numOutputs, false);
  for (unsigned outIdx = 0; outIdx < numOutputs; ++outIdx) {
    const std::string &row = (*rows)[outIdx];
    if (row.size() != numInputs)
      return std::vector<bool>(numInputs * numOutputs, true);
    for (unsigned inIdx = 0; inIdx < numInputs; ++inIdx)
      connectivity[outIdx * numInputs + inIdx] = (row[inIdx] == '1');
  }
  return connectivity;
}

class BoundaryInputModule final : public SimModule {
public:
  explicit BoundaryInputModule(const StaticModuleDesc &module) {
    hwNodeId = module.hwNodeId;
    name = module.name;
    kind = module.kind;
  }

  bool isCombinational() const override { return false; }
  void reset() override {
    pending_.clear();
    nextGeneration_ = 1;
    perf_ = PerfSnapshot();
  }
  void configure(const std::vector<uint32_t> &configWords) override {
    (void)configWords;
  }

  void setInputTokens(const std::vector<SimToken> &tokens) override {
    pending_.assign(tokens.begin(), tokens.end());
    for (SimToken &token : pending_) {
      if (token.generation == 0)
        token.generation = composeTokenGeneration(hwNodeId, nextGeneration_++);
    }
  }

  void evaluate() override {
    if (outputs.empty())
      return;
    SimChannel *out = outputs.front();
    if (pending_.empty()) {
      out->valid = false;
      return;
    }
    driveChannelFromToken(*out, pending_.front());
  }

  void commit() override {
    if (outputs.empty() || pending_.empty())
      return;
    if (outputs.front()->transferred()) {
      pending_.pop_front();
      perf_.activeCycles++;
      perf_.tokensOut++;
    } else if (outputs.front()->valid) {
      perf_.stallCyclesOut++;
    }
  }

  void collectTraceEvents(std::vector<TraceEvent> &events,
                          uint64_t cycle) override {
    if (outputs.empty() || !outputs.front()->transferred())
      return;
    TraceEvent ev;
    ev.cycle = cycle;
    ev.phase = SimPhase::Commit;
    ev.hwNodeId = hwNodeId;
    ev.eventKind = EventKind::NodeFire;
    events.push_back(ev);
  }

  PerfSnapshot getPerfSnapshot() const override { return perf_; }

  bool hasPendingWork() const override { return !pending_.empty(); }

private:
  std::deque<SimToken> pending_;
  uint64_t nextGeneration_ = 1;
};

class BoundaryOutputModule final : public SimModule {
public:
  explicit BoundaryOutputModule(const StaticModuleDesc &module) {
    hwNodeId = module.hwNodeId;
    name = module.name;
    kind = module.kind;
  }

  bool isCombinational() const override { return false; }
  void reset() override {
    collected_.clear();
    lastCollectedGeneration_ = 0;
    perf_ = PerfSnapshot();
  }
  void configure(const std::vector<uint32_t> &configWords) override {
    (void)configWords;
  }

  void evaluate() override {
    if (inputs.empty())
      return;
    inputs.front()->ready = true;
  }

  void commit() override {
    if (inputs.empty())
      return;
    SimChannel *in = inputs.front();
    if (!in->transferred()) {
      if (in->valid)
        perf_.stallCyclesOut++;
      return;
    }
    if (in->generation != 0 && in->generation == lastCollectedGeneration_)
      return;
    collected_.push_back(tokenFromChannel(*in));
    lastCollectedGeneration_ = in->generation;
    perf_.activeCycles++;
    perf_.tokensIn++;
  }

  void collectTraceEvents(std::vector<TraceEvent> &events,
                          uint64_t cycle) override {
    if (inputs.empty() || !inputs.front()->transferred())
      return;
    TraceEvent ev;
    ev.cycle = cycle;
    ev.phase = SimPhase::Commit;
    ev.hwNodeId = hwNodeId;
    ev.eventKind = EventKind::NodeFire;
    events.push_back(ev);
  }

  PerfSnapshot getPerfSnapshot() const override { return perf_; }

  const std::vector<SimToken> &getCollectedTokens() const override {
    return collected_;
  }

private:
  std::vector<SimToken> collected_;
  uint64_t lastCollectedGeneration_ = 0;
};

class AddTagModule final : public SimModule {
public:
  AddTagModule(const StaticModuleDesc &module, unsigned tagWidth)
      : tagMask_(tagWidth >= 16 ? std::numeric_limits<uint16_t>::max()
                                : static_cast<uint16_t>((1u << tagWidth) - 1)) {
    hwNodeId = module.hwNodeId;
    name = module.name;
    kind = module.kind;
  }

  bool isCombinational() const override { return true; }
  void reset() override { perf_ = PerfSnapshot(); }
  void configure(const std::vector<uint32_t> &configWords) override {
    configuredTag_ = configWords.empty() ? 0
                                         : static_cast<uint16_t>(configWords[0]) & tagMask_;
  }

  void evaluate() override {
    if (inputs.empty() || outputs.empty())
      return;
    outputs.front()->valid = inputs.front()->valid;
    outputs.front()->data = inputs.front()->data;
    outputs.front()->tag = configuredTag_;
    outputs.front()->hasTag = true;
    outputs.front()->generation = inputs.front()->generation;
    inputs.front()->ready = outputs.front()->ready;
  }

  void collectTraceEvents(std::vector<TraceEvent> &events,
                          uint64_t cycle) override {
    if (outputs.empty() || !outputs.front()->transferred())
      return;
    perf_.activeCycles++;
    perf_.tokensOut++;
    TraceEvent ev;
    ev.cycle = cycle;
    ev.phase = SimPhase::Commit;
    ev.hwNodeId = hwNodeId;
    ev.eventKind = EventKind::RouteUse;
    events.push_back(ev);
  }

  PerfSnapshot getPerfSnapshot() const override { return perf_; }

private:
  uint16_t configuredTag_ = 0;
  uint16_t tagMask_ = 0xffff;
};

class DelTagModule final : public SimModule {
public:
  explicit DelTagModule(const StaticModuleDesc &module) {
    hwNodeId = module.hwNodeId;
    name = module.name;
    kind = module.kind;
  }

  bool isCombinational() const override { return true; }
  void reset() override { perf_ = PerfSnapshot(); }
  void configure(const std::vector<uint32_t> &configWords) override {
    (void)configWords;
  }

  void evaluate() override {
    if (inputs.empty() || outputs.empty())
      return;
    outputs.front()->valid = inputs.front()->valid;
    outputs.front()->data = inputs.front()->data;
    outputs.front()->tag = 0;
    outputs.front()->hasTag = false;
    outputs.front()->generation = inputs.front()->generation;
    inputs.front()->ready = outputs.front()->ready;
  }

  void collectTraceEvents(std::vector<TraceEvent> &events,
                          uint64_t cycle) override {
    if (outputs.empty() || !outputs.front()->transferred())
      return;
    perf_.activeCycles++;
    perf_.tokensOut++;
    TraceEvent ev;
    ev.cycle = cycle;
    ev.phase = SimPhase::Commit;
    ev.hwNodeId = hwNodeId;
    ev.eventKind = EventKind::RouteUse;
    events.push_back(ev);
  }

  PerfSnapshot getPerfSnapshot() const override { return perf_; }
};

class MapTagModule final : public SimModule {
public:
  MapTagModule(const StaticModuleDesc &module, unsigned inputTagWidth,
               unsigned outputTagWidth, unsigned tableSize)
      : inputTagMask_(inputTagWidth >= 16
                          ? std::numeric_limits<uint16_t>::max()
                          : static_cast<uint16_t>((1u << inputTagWidth) - 1)),
        outputTagMask_(outputTagWidth >= 16
                           ? std::numeric_limits<uint16_t>::max()
                           : static_cast<uint16_t>((1u << outputTagWidth) - 1)),
        inputTagWidthBits_(inputTagWidth),
        outputTagWidthBits_(outputTagWidth),
        tableSize_(tableSize) {
    hwNodeId = module.hwNodeId;
    name = module.name;
    kind = module.kind;
    table_.resize(tableSize_);
  }

  bool isCombinational() const override { return true; }
  void reset() override { perf_ = PerfSnapshot(); }

  void configure(const std::vector<uint32_t> &configWords) override {
    auto readBits = [&](unsigned &bitPos, unsigned width) -> uint32_t {
      uint32_t value = 0;
      for (unsigned bit = 0; bit < width; ++bit) {
        unsigned wordIdx = bitPos / 32;
        unsigned wordBit = bitPos % 32;
        if (wordIdx < configWords.size() &&
            ((configWords[wordIdx] >> wordBit) & 1u) != 0)
          value |= (1u << bit);
        ++bitPos;
      }
      return value;
    };

    unsigned bitPos = 0;
    for (unsigned idx = 0; idx < tableSize_; ++idx) {
      table_[idx].valid = readBits(bitPos, 1) != 0;
      table_[idx].srcTag =
          static_cast<uint16_t>(readBits(bitPos, inputTagWidthBits_)) & inputTagMask_;
      table_[idx].dstTag = static_cast<uint16_t>(readBits(bitPos, outputTagWidthBits_)) &
                           outputTagMask_;
    }
  }

  void evaluate() override {
    if (inputs.empty() || outputs.empty())
      return;

    SimChannel *in = inputs.front();
    SimChannel *out = outputs.front();
    if (!in->valid) {
      out->valid = false;
      in->ready = out->ready;
      return;
    }

    auto it = std::find_if(table_.begin(), table_.end(), [&](const Entry &entry) {
      return entry.valid && entry.srcTag == (in->tag & inputTagMask_);
    });
    if (it == table_.end()) {
      out->valid = false;
      in->ready = false;
      return;
    }

    out->valid = true;
    out->data = in->data;
    out->tag = it->dstTag;
    out->hasTag = true;
    out->generation = in->generation;
    in->ready = out->ready;
  }

  void collectTraceEvents(std::vector<TraceEvent> &events,
                          uint64_t cycle) override {
    if (outputs.empty() || !outputs.front()->transferred())
      return;
    perf_.activeCycles++;
    perf_.tokensOut++;
    TraceEvent ev;
    ev.cycle = cycle;
    ev.phase = SimPhase::Commit;
    ev.hwNodeId = hwNodeId;
    ev.eventKind = EventKind::RouteUse;
    events.push_back(ev);
  }

  PerfSnapshot getPerfSnapshot() const override { return perf_; }

private:
  struct Entry {
    bool valid = false;
    uint16_t srcTag = 0;
    uint16_t dstTag = 0;
  };

  uint16_t inputTagMask_ = 0xffff;
  uint16_t outputTagMask_ = 0xffff;
  unsigned inputTagWidthBits_ = 0;
  unsigned outputTagWidthBits_ = 0;
  unsigned tableSize_ = 0;
  std::vector<Entry> table_;
};

class FifoModule final : public SimModule {
public:
  FifoModule(const StaticModuleDesc &module, unsigned depth, bool bypassable,
             bool bypassed)
      : depth_(depth), bypassable_(bypassable), bypassed_(bypassable && bypassed) {
    hwNodeId = module.hwNodeId;
    name = module.name;
    kind = module.kind;
  }

  bool isCombinational() const override { return bypassed_; }
  void reset() override {
    buffer_.clear();
    perf_ = PerfSnapshot();
  }
  void configure(const std::vector<uint32_t> &configWords) override {
    if (!bypassable_)
      return;
    bypassed_ = !configWords.empty() && (configWords.front() & 1u) != 0;
  }

  void evaluate() override {
    if (inputs.empty() || outputs.empty())
      return;
    SimChannel *in = inputs.front();
    SimChannel *out = outputs.front();

    if (bypassed_) {
      out->valid = in->valid;
      out->data = in->data;
      out->tag = in->tag;
      out->hasTag = in->hasTag;
      out->generation = in->generation;
      in->ready = out->ready;
      return;
    }

    if (buffer_.empty()) {
      out->valid = false;
    } else {
      driveChannelFromToken(*out, buffer_.front());
    }
    in->ready = buffer_.size() < depth_;
  }

  void commit() override {
    if (inputs.empty() || outputs.empty())
      return;
    SimChannel *in = inputs.front();
    SimChannel *out = outputs.front();
    if (bypassed_) {
      if (out->transferred()) {
        perf_.activeCycles++;
        perf_.tokensIn++;
        perf_.tokensOut++;
      }
      return;
    }
    if (out->transferred() && !buffer_.empty()) {
      buffer_.pop_front();
      perf_.tokensOut++;
    }
    if (in->transferred()) {
      buffer_.push_back(tokenFromChannel(*in));
      perf_.tokensIn++;
    }
    if (in->transferred() || out->transferred())
      perf_.activeCycles++;
  }

  void collectTraceEvents(std::vector<TraceEvent> &events,
                          uint64_t cycle) override {
    if (outputs.empty() || !outputs.front()->transferred())
      return;
    TraceEvent ev;
    ev.cycle = cycle;
    ev.phase = SimPhase::Commit;
    ev.hwNodeId = hwNodeId;
    ev.eventKind = EventKind::RouteUse;
    events.push_back(ev);
  }

  PerfSnapshot getPerfSnapshot() const override { return perf_; }

  bool hasPendingWork() const override { return !buffer_.empty(); }

private:
  unsigned depth_ = 0;
  bool bypassable_ = false;
  bool bypassed_ = false;
  std::deque<SimToken> buffer_;
};

class SpatialSwitchModule final : public SimModule {
public:
  SpatialSwitchModule(const StaticModuleDesc &module,
                      std::vector<bool> connectivity,
                      std::vector<bool> outputTagged, unsigned numInputs,
                      unsigned numOutputs)
      : connectivity_(std::move(connectivity)), numInputs_(numInputs),
        numOutputs_(numOutputs), outputTagged_(std::move(outputTagged)) {
    hwNodeId = module.hwNodeId;
    name = module.name;
    kind = module.kind;
    outputSource_.assign(numOutputs_, -1);
    outputInputs_.resize(numOutputs_);
    inputTargets_.resize(numInputs_);
    rrPointer_.assign(numOutputs_, 0);
  }

  bool isCombinational() const override { return false; }
  void reset() override {
    perf_ = PerfSnapshot();
    std::fill(rrPointer_.begin(), rrPointer_.end(), 0);
    std::fill(inputGeneration_.begin(), inputGeneration_.end(), 0);
    for (auto &accepted : inputAccepted_)
      std::fill(accepted.begin(), accepted.end(), 0);
  }

  void configure(const std::vector<uint32_t> &configWords) override {
    routeBits_.assign(countConnected(), false);
    unsigned bitPos = 0;
    for (size_t idx = 0; idx < routeBits_.size(); ++idx) {
      unsigned wordIdx = bitPos / 32;
      unsigned wordBit = bitPos % 32;
      routeBits_[idx] =
          wordIdx < configWords.size() && ((configWords[wordIdx] >> wordBit) & 1u) != 0;
      ++bitPos;
    }
    rebuild();
  }

  void evaluate() override {
    std::vector<bool> inputReady(numInputs_, false);
    std::vector<bool> outputConflict(numOutputs_, false);
    outputSource_.assign(numOutputs_, -1);

    refreshInputFanoutState();

    for (unsigned outIdx = 0; outIdx < numOutputs_ && outIdx < outputs.size();
         ++outIdx) {
      std::vector<unsigned> validInputs;
      for (unsigned inIdx : outputInputs_[outIdx]) {
        if (inIdx < inputs.size() && inputs[inIdx]->valid &&
            inputNeedsTarget(inIdx, outIdx))
          validInputs.push_back(inIdx);
      }

      int chosenSrc = -1;
      if (validInputs.size() == 1) {
        chosenSrc = static_cast<int>(validInputs.front());
      } else if (validInputs.size() > 1) {
        if (outIdx < outputTagged_.size() && outputTagged_[outIdx]) {
          unsigned start = outIdx < rrPointer_.size() ? rrPointer_[outIdx] : 0;
          for (unsigned offset = 0; offset < outputInputs_[outIdx].size();
               ++offset) {
            unsigned candidatePos =
                (start + offset) % outputInputs_[outIdx].size();
            unsigned candidateInput = outputInputs_[outIdx][candidatePos];
            if (std::find(validInputs.begin(), validInputs.end(),
                          candidateInput) != validInputs.end()) {
              chosenSrc = static_cast<int>(candidateInput);
              break;
            }
          }
        } else {
          outputConflict[outIdx] = true;
        }
      }
      outputSource_[outIdx] = chosenSrc;
      if (chosenSrc >= 0) {
        int srcIdx = chosenSrc;
        outputs[outIdx]->valid = inputs[srcIdx]->valid;
        outputs[outIdx]->data = inputs[srcIdx]->data;
        outputs[outIdx]->tag = inputs[srcIdx]->tag;
        outputs[outIdx]->hasTag = inputs[srcIdx]->hasTag;
        outputs[outIdx]->generation = inputs[srcIdx]->generation;
      } else {
        outputs[outIdx]->valid = false;
      }
    }

    for (unsigned inIdx = 0; inIdx < numInputs_ && inIdx < inputs.size(); ++inIdx) {
      if (inputTargets_[inIdx].empty()) {
        inputs[inIdx]->ready = false;
        continue;
      }
      bool allReady = true;
      for (unsigned outIdx : inputTargets_[inIdx]) {
        if (!inputNeedsTarget(inIdx, outIdx))
          continue;
        if (outIdx >= outputs.size())
          continue;
        if (outputConflict[outIdx]) {
          allReady = false;
          continue;
        }
        int chosenSrc = outputSource_[outIdx];
        if (chosenSrc >= 0) {
          if (chosenSrc != static_cast<int>(inIdx)) {
            allReady = false;
            continue;
          }
          if (!outputs[outIdx]->ready)
            allReady = false;
          continue;
        }
        if (!outputs[outIdx]->ready)
          allReady = false;
      }
      inputs[inIdx]->ready = allReady;
    }
  }

  void commit() override {
    for (unsigned inIdx = 0; inIdx < numInputs_ && inIdx < inputs.size(); ++inIdx) {
      if (!inputs[inIdx]->valid || inputTargets_[inIdx].empty())
        continue;
      for (size_t targetPos = 0; targetPos < inputTargets_[inIdx].size(); ++targetPos) {
        unsigned outIdx = inputTargets_[inIdx][targetPos];
        if (outIdx >= outputs.size())
          continue;
        if (outputSource_[outIdx] != static_cast<int>(inIdx))
          continue;
        if (!outputs[outIdx]->transferred())
          continue;
        if (targetPos < inputAccepted_[inIdx].size())
          inputAccepted_[inIdx][targetPos] = 1;
      }
    }

    for (unsigned outIdx = 0; outIdx < numOutputs_ && outIdx < outputs.size();
         ++outIdx) {
      if (!(outIdx < outputTagged_.size() && outputTagged_[outIdx]))
        continue;
      if (!outputs[outIdx]->transferred())
        continue;
      int chosenSrc = outIdx < outputSource_.size() ? outputSource_[outIdx] : -1;
      if (chosenSrc < 0)
        continue;
      for (unsigned pos = 0; pos < outputInputs_[outIdx].size(); ++pos) {
        if (outputInputs_[outIdx][pos] == static_cast<unsigned>(chosenSrc)) {
          rrPointer_[outIdx] = (pos + 1) % outputInputs_[outIdx].size();
          break;
        }
      }
    }
  }

  void collectTraceEvents(std::vector<TraceEvent> &events,
                          uint64_t cycle) override {
    bool any = false;
    for (unsigned outIdx = 0; outIdx < outputs.size(); ++outIdx) {
      if (!outputs[outIdx]->transferred())
        continue;
      any = true;
      perf_.tokensOut++;
      TraceEvent ev;
      ev.cycle = cycle;
      ev.phase = SimPhase::Commit;
      ev.hwNodeId = hwNodeId;
      ev.eventKind = EventKind::RouteUse;
      ev.lane = static_cast<uint8_t>(outIdx);
      ev.arg0 = outputSource_[outIdx] >= 0 ? static_cast<uint32_t>(outputSource_[outIdx]) : 0;
      events.push_back(ev);
    }
    if (any)
      perf_.activeCycles++;
  }

  PerfSnapshot getPerfSnapshot() const override { return perf_; }
  bool hasPendingWork() const override {
    for (unsigned inIdx = 0; inIdx < numInputs_ && inIdx < inputs.size(); ++inIdx) {
      if (!inputs[inIdx]->valid || inputTargets_[inIdx].empty())
        continue;
      for (uint8_t accepted : inputAccepted_[inIdx]) {
        if (accepted == 0)
          return true;
      }
    }
    return false;
  }

  std::string getDebugStateSummary() const override {
    std::string summary;
    bool firstInput = true;
    for (unsigned inIdx = 0; inIdx < numInputs_ && inIdx < inputs.size(); ++inIdx) {
      if (!inputs[inIdx]->valid || inputTargets_[inIdx].empty())
        continue;
      std::vector<unsigned> pendingTargets;
      for (size_t targetPos = 0; targetPos < inputTargets_[inIdx].size(); ++targetPos) {
        if (inIdx < inputAccepted_.size() &&
            targetPos < inputAccepted_[inIdx].size() &&
            inputAccepted_[inIdx][targetPos] == 0) {
          pendingTargets.push_back(inputTargets_[inIdx][targetPos]);
        }
      }
      if (pendingTargets.empty())
        continue;
      if (!firstInput)
        summary += " ";
      firstInput = false;
      summary += "in" + std::to_string(inIdx) + "(gen=" +
                 std::to_string(inputGeneration_[inIdx]) + ",pending=[";
      for (size_t idx = 0; idx < pendingTargets.size(); ++idx) {
        if (idx != 0)
          summary += ",";
        summary += std::to_string(pendingTargets[idx]);
      }
      summary += "])";
    }
    return summary;
  }

private:
  unsigned countConnected() const {
    return static_cast<unsigned>(
        std::count(connectivity_.begin(), connectivity_.end(), true));
  }

  void rebuild() {
    outputSource_.assign(numOutputs_, -1);
    for (auto &inputs : outputInputs_)
      inputs.clear();
    for (auto &targets : inputTargets_)
      targets.clear();

    unsigned routeIdx = 0;
    for (unsigned outIdx = 0; outIdx < numOutputs_; ++outIdx) {
      for (unsigned inIdx = 0; inIdx < numInputs_; ++inIdx) {
        if (!connectivity_[outIdx * numInputs_ + inIdx])
          continue;
        if (routeIdx < routeBits_.size() && routeBits_[routeIdx]) {
          outputInputs_[outIdx].push_back(inIdx);
          inputTargets_[inIdx].push_back(outIdx);
        }
        ++routeIdx;
      }
    }

    inputGeneration_.assign(numInputs_, 0);
    inputAccepted_.assign(numInputs_, {});
    for (unsigned inIdx = 0; inIdx < numInputs_; ++inIdx)
      inputAccepted_[inIdx].assign(inputTargets_[inIdx].size(), 0);
  }

  void refreshInputFanoutState() {
    for (unsigned inIdx = 0; inIdx < numInputs_ && inIdx < inputs.size(); ++inIdx) {
      uint64_t generation = inputs[inIdx]->valid ? inputs[inIdx]->generation : 0;
      if (generation == 0) {
        inputGeneration_[inIdx] = 0;
        if (inIdx < inputAccepted_.size())
          std::fill(inputAccepted_[inIdx].begin(), inputAccepted_[inIdx].end(), 0);
        continue;
      }
      if (inputGeneration_[inIdx] != generation) {
        inputGeneration_[inIdx] = generation;
        if (inIdx < inputAccepted_.size())
          std::fill(inputAccepted_[inIdx].begin(), inputAccepted_[inIdx].end(), 0);
      }
    }
  }

  bool inputNeedsTarget(unsigned inIdx, unsigned outIdx) const {
    if (inIdx >= inputTargets_.size())
      return false;
    const auto &targets = inputTargets_[inIdx];
    for (size_t targetPos = 0; targetPos < targets.size(); ++targetPos) {
      if (targets[targetPos] != outIdx)
        continue;
      return inIdx >= inputAccepted_.size() ||
             targetPos >= inputAccepted_[inIdx].size() ||
             inputAccepted_[inIdx][targetPos] == 0;
    }
    return false;
  }

  std::vector<bool> connectivity_;
  unsigned numInputs_ = 0;
  unsigned numOutputs_ = 0;
  std::vector<bool> outputTagged_;
  std::vector<bool> routeBits_;
  std::vector<int> outputSource_;
  std::vector<std::vector<unsigned>> outputInputs_;
  std::vector<std::vector<unsigned>> inputTargets_;
  std::vector<unsigned> rrPointer_;
  std::vector<uint64_t> inputGeneration_;
  std::vector<std::vector<uint8_t>> inputAccepted_;
};

class TemporalSwitchModule final : public SimModule {
public:
  TemporalSwitchModule(const StaticModuleDesc &module,
                       std::vector<bool> connectivity, unsigned numInputs,
                       unsigned numOutputs, unsigned tagWidth,
                       unsigned slotCount)
      : connectivity_(std::move(connectivity)), numInputs_(numInputs),
        numOutputs_(numOutputs), tagWidth_(tagWidth), slotCount_(slotCount) {
    hwNodeId = module.hwNodeId;
    name = module.name;
    kind = module.kind;
    slots_.resize(slotCount_);
    rrPointer_.assign(numOutputs_, 0);
    inputSlot_.assign(numInputs_, -1);
    winners_.assign(numOutputs_, -1);
    broadcastOk_.assign(numInputs_, false);
    slotRouting_.resize(slotCount_);
  }

  bool isCombinational() const override { return false; }
  void reset() override {
    rrPointer_.assign(numOutputs_, 0);
    perf_ = PerfSnapshot();
    std::fill(inputGeneration_.begin(), inputGeneration_.end(), 0);
    for (auto &accepted : inputAccepted_)
      std::fill(accepted.begin(), accepted.end(), 0);
  }

  void configure(const std::vector<uint32_t> &configWords) override {
    unsigned routeBits = static_cast<unsigned>(
        std::count(connectivity_.begin(), connectivity_.end(), true));
    auto readBits = [&](unsigned &bitPos, unsigned width) -> uint32_t {
      uint32_t value = 0;
      for (unsigned bit = 0; bit < width; ++bit) {
        unsigned wordIdx = bitPos / 32;
        unsigned wordBit = bitPos % 32;
        if (wordIdx < configWords.size() &&
            ((configWords[wordIdx] >> wordBit) & 1u) != 0)
          value |= (1u << bit);
        ++bitPos;
      }
      return value;
    };

    unsigned bitPos = 0;
    for (unsigned idx = 0; idx < slotCount_; ++idx) {
      slots_[idx].valid = readBits(bitPos, 1) != 0;
      slots_[idx].tag = static_cast<uint16_t>(readBits(bitPos, tagWidth_));
      slots_[idx].routes.assign(routeBits, false);
      for (unsigned routeIdx = 0; routeIdx < routeBits; ++routeIdx)
        slots_[idx].routes[routeIdx] = readBits(bitPos, 1) != 0;
    }
    rebuildSlots();
  }

  void evaluate() override {
    std::fill(inputSlot_.begin(), inputSlot_.end(), -1);
    for (unsigned inIdx = 0; inIdx < numInputs_ && inIdx < inputs.size(); ++inIdx) {
      if (!inputs[inIdx]->valid)
        continue;
      for (unsigned slotIdx = 0; slotIdx < slotCount_; ++slotIdx) {
        if (slots_[slotIdx].valid && slots_[slotIdx].tag == inputs[inIdx]->tag) {
          inputSlot_[inIdx] = static_cast<int>(slotIdx);
          break;
        }
      }
    }

    refreshInputFanoutState();
    arbitrate();

    for (unsigned outIdx = 0; outIdx < numOutputs_ && outIdx < outputs.size();
         ++outIdx) {
      int winner = winners_[outIdx];
      if (winner >= 0) {
        outputs[outIdx]->valid = true;
        outputs[outIdx]->data = inputs[winner]->data;
        outputs[outIdx]->tag = inputs[winner]->tag;
        outputs[outIdx]->hasTag = inputs[winner]->hasTag;
        outputs[outIdx]->generation = inputs[winner]->generation;
      } else {
        outputs[outIdx]->valid = false;
      }
    }

    for (unsigned inIdx = 0; inIdx < numInputs_ && inIdx < inputs.size(); ++inIdx)
      inputs[inIdx]->ready = broadcastOk_[inIdx];
  }

  void commit() override {
    for (unsigned inIdx = 0; inIdx < numInputs_ && inIdx < inputs.size(); ++inIdx) {
      int slotIdx = inputSlot_[inIdx];
      if (slotIdx < 0 || !inputs[inIdx]->valid)
        continue;
      const auto &targets = slotRouting_[slotIdx].inputTargets[inIdx];
      for (size_t targetPos = 0; targetPos < targets.size(); ++targetPos) {
        unsigned outIdx = targets[targetPos];
        if (outIdx >= outputs.size())
          continue;
        if (winners_[outIdx] != static_cast<int>(inIdx))
          continue;
        if (!outputs[outIdx]->transferred())
          continue;
        if (targetPos < inputAccepted_[inIdx].size())
          inputAccepted_[inIdx][targetPos] = 1;
      }
    }
    for (unsigned outIdx = 0; outIdx < outputs.size(); ++outIdx) {
      if (!outputs[outIdx]->transferred())
        continue;
      int winner = winners_[outIdx];
      if (winner >= 0)
        rrPointer_[outIdx] = (static_cast<unsigned>(winner) + 1) % numInputs_;
    }
  }

  void collectTraceEvents(std::vector<TraceEvent> &events,
                          uint64_t cycle) override {
    bool any = false;
    for (unsigned outIdx = 0; outIdx < outputs.size(); ++outIdx) {
      if (!outputs[outIdx]->transferred())
        continue;
      any = true;
      perf_.tokensOut++;
      TraceEvent ev;
      ev.cycle = cycle;
      ev.phase = SimPhase::Commit;
      ev.hwNodeId = hwNodeId;
      ev.eventKind = EventKind::RouteUse;
      ev.lane = static_cast<uint8_t>(outIdx);
      ev.arg0 = winners_[outIdx] >= 0 ? static_cast<uint32_t>(winners_[outIdx]) : 0;
      events.push_back(ev);
    }
    if (any)
      perf_.activeCycles++;
  }

  PerfSnapshot getPerfSnapshot() const override { return perf_; }

  bool hasPendingWork() const override {
    for (unsigned inIdx = 0; inIdx < numInputs_ && inIdx < inputs.size(); ++inIdx) {
      int slotIdx = inputSlot_[inIdx];
      if (slotIdx < 0 || !inputs[inIdx]->valid)
        continue;
      const auto &targets = slotRouting_[slotIdx].inputTargets[inIdx];
      for (size_t targetPos = 0; targetPos < targets.size(); ++targetPos) {
        if (targetPos >= inputAccepted_[inIdx].size() ||
            inputAccepted_[inIdx][targetPos] == 0)
          return true;
      }
    }
    return false;
  }

  void debugDump(std::ostream &os) const override {
    os << "      temporal_sw slots=" << slots_.size()
       << " tag_width=" << tagWidth_ << " inputs=" << numInputs_
       << " outputs=" << numOutputs_ << "\n";
    for (unsigned slotIdx = 0; slotIdx < slots_.size(); ++slotIdx) {
      os << "      slot[" << slotIdx << "] valid=" << slots_[slotIdx].valid
         << " tag=" << slots_[slotIdx].tag << " routes=[";
      for (size_t routeIdx = 0; routeIdx < slots_[slotIdx].routes.size(); ++routeIdx) {
        if (routeIdx)
          os << ",";
        os << (slots_[slotIdx].routes[routeIdx] ? "1" : "0");
      }
      os << "]\n";
    }
    os << "      input_slot=[";
    for (size_t idx = 0; idx < inputSlot_.size(); ++idx) {
      if (idx)
        os << ",";
      os << inputSlot_[idx];
    }
    os << "] winners=[";
    for (size_t idx = 0; idx < winners_.size(); ++idx) {
      if (idx)
        os << ",";
      os << winners_[idx];
    }
    os << "] broadcast_ok=[";
    for (size_t idx = 0; idx < broadcastOk_.size(); ++idx) {
      if (idx)
        os << ",";
      os << (broadcastOk_[idx] ? "1" : "0");
    }
    os << "]\n";
  }

private:
  struct Slot {
    bool valid = false;
    uint16_t tag = 0;
    std::vector<bool> routes;
  };

  struct SlotRouting {
    std::vector<int> outputSource;
    std::vector<std::vector<unsigned>> inputTargets;
  };

  void rebuildSlots() {
    for (auto &routing : slotRouting_) {
      routing.outputSource.assign(numOutputs_, -1);
      routing.inputTargets.assign(numInputs_, {});
    }
    for (unsigned slotIdx = 0; slotIdx < slotCount_; ++slotIdx) {
      unsigned routeIdx = 0;
      for (unsigned outIdx = 0; outIdx < numOutputs_; ++outIdx) {
        for (unsigned inIdx = 0; inIdx < numInputs_; ++inIdx) {
          if (!connectivity_[outIdx * numInputs_ + inIdx])
            continue;
          if (routeIdx < slots_[slotIdx].routes.size() &&
              slots_[slotIdx].routes[routeIdx]) {
            slotRouting_[slotIdx].outputSource[outIdx] = static_cast<int>(inIdx);
            slotRouting_[slotIdx].inputTargets[inIdx].push_back(outIdx);
          }
          ++routeIdx;
        }
      }
    }
    inputGeneration_.assign(numInputs_, 0);
    inputAccepted_.assign(numInputs_, {});
    for (unsigned inIdx = 0; inIdx < numInputs_; ++inIdx) {
      unsigned maxTargets = 0;
      for (const auto &routing : slotRouting_)
        maxTargets = std::max<unsigned>(maxTargets, routing.inputTargets[inIdx].size());
      inputAccepted_[inIdx].assign(maxTargets, 0);
    }
  }

  void refreshInputFanoutState() {
    for (unsigned inIdx = 0; inIdx < numInputs_ && inIdx < inputs.size(); ++inIdx) {
      uint64_t generation = inputs[inIdx]->valid ? inputs[inIdx]->generation : 0;
      if (generation == 0) {
        inputGeneration_[inIdx] = 0;
        std::fill(inputAccepted_[inIdx].begin(), inputAccepted_[inIdx].end(), 0);
        continue;
      }
      if (inputGeneration_[inIdx] != generation) {
        inputGeneration_[inIdx] = generation;
        std::fill(inputAccepted_[inIdx].begin(), inputAccepted_[inIdx].end(), 0);
      }
    }
  }

  bool inputNeedsTarget(unsigned inIdx, int slotIdx, unsigned outIdx) const {
    if (slotIdx < 0 || static_cast<size_t>(slotIdx) >= slotRouting_.size() ||
        inIdx >= numInputs_)
      return false;
    const auto &targets = slotRouting_[slotIdx].inputTargets[inIdx];
    for (size_t targetPos = 0; targetPos < targets.size(); ++targetPos) {
      if (targets[targetPos] != outIdx)
        continue;
      return targetPos >= inputAccepted_[inIdx].size() ||
             inputAccepted_[inIdx][targetPos] == 0;
    }
    return false;
  }

  void arbitrate() {
    std::fill(winners_.begin(), winners_.end(), -1);
    std::fill(broadcastOk_.begin(), broadcastOk_.end(), false);

    for (unsigned outIdx = 0; outIdx < numOutputs_; ++outIdx) {
      unsigned start = rrPointer_[outIdx];
      for (unsigned probe = 0; probe < numInputs_; ++probe) {
        unsigned inIdx = (start + probe) % numInputs_;
        if (inIdx >= inputs.size() || !inputs[inIdx]->valid)
          continue;
        int slotIdx = inputSlot_[inIdx];
        if (slotIdx < 0)
          continue;
        if (!inputNeedsTarget(inIdx, slotIdx, outIdx))
          continue;
        if (outIdx < slotRouting_[slotIdx].outputSource.size() &&
            slotRouting_[slotIdx].outputSource[outIdx] == static_cast<int>(inIdx)) {
          winners_[outIdx] = static_cast<int>(inIdx);
          break;
        }
      }
    }

    for (unsigned inIdx = 0; inIdx < numInputs_ && inIdx < inputs.size(); ++inIdx) {
      int slotIdx = inputSlot_[inIdx];
      if (slotIdx < 0 || !inputs[inIdx]->valid)
        continue;
      const auto &targets = slotRouting_[slotIdx].inputTargets[inIdx];
      if (targets.empty())
        continue;
      bool ok = true;
      for (size_t targetPos = 0; targetPos < targets.size(); ++targetPos) {
        if (targetPos < inputAccepted_[inIdx].size() &&
            inputAccepted_[inIdx][targetPos] != 0)
          continue;
        unsigned outIdx = targets[targetPos];
        if (winners_[outIdx] != static_cast<int>(inIdx)) {
          ok = false;
          break;
        }
        if (outIdx < outputs.size() && !outputs[outIdx]->ready) {
          ok = false;
          break;
        }
      }
      broadcastOk_[inIdx] = ok;
    }
  }

  std::vector<bool> connectivity_;
  unsigned numInputs_ = 0;
  unsigned numOutputs_ = 0;
  unsigned tagWidth_ = 0;
  unsigned slotCount_ = 0;
  std::vector<Slot> slots_;
  std::vector<unsigned> rrPointer_;
  std::vector<int> inputSlot_;
  std::vector<int> winners_;
  std::vector<bool> broadcastOk_;
  std::vector<SlotRouting> slotRouting_;
  std::vector<uint64_t> inputGeneration_;
  std::vector<std::vector<uint8_t>> inputAccepted_;
};

class UnsupportedModule final : public SimModule {
public:
  explicit UnsupportedModule(const StaticModuleDesc &module) {
    hwNodeId = module.hwNodeId;
    name = module.name;
    kind = module.kind;
  }

  bool isCombinational() const override { return true; }
  void reset() override { perf_ = PerfSnapshot(); }
  void configure(const std::vector<uint32_t> &configWords) override {
    (void)configWords;
  }
  void evaluate() override {}
  void collectTraceEvents(std::vector<TraceEvent> &events,
                          uint64_t cycle) override {
    (void)events;
    (void)cycle;
  }
  PerfSnapshot getPerfSnapshot() const override { return perf_; }
};

} // namespace

std::unique_ptr<SimModule> createSimModule(const StaticModuleDesc &module,
                                           const StaticMappedModel &model) {
  switch (module.kind) {
  case StaticModuleKind::BoundaryInput:
    return std::make_unique<BoundaryInputModule>(module);
  case StaticModuleKind::BoundaryOutput:
    return std::make_unique<BoundaryOutputModule>(module);
  case StaticModuleKind::AddTag: {
    unsigned tagWidth = 1;
    if (!module.outputPorts.empty()) {
      if (const StaticPortDesc *port = findPortById(model, module.outputPorts.front()))
        tagWidth = std::max(1u, port->tagWidth);
    }
    return std::make_unique<AddTagModule>(module, tagWidth);
  }
  case StaticModuleKind::DelTag:
    return std::make_unique<DelTagModule>(module);
  case StaticModuleKind::MapTag: {
    unsigned inTagWidth = 1;
    unsigned outTagWidth = 1;
    if (!module.inputPorts.empty()) {
      if (const StaticPortDesc *port = findPortById(model, module.inputPorts.front()))
        inTagWidth = std::max(1u, port->tagWidth);
    }
    if (!module.outputPorts.empty()) {
      if (const StaticPortDesc *port = findPortById(model, module.outputPorts.front()))
        outTagWidth = std::max(1u, port->tagWidth);
    }
    unsigned tableSize =
        static_cast<unsigned>(std::max<int64_t>(1, getIntAttr(module, "table_size", 1)));
    auto result = std::make_unique<MapTagModule>(module, inTagWidth, outTagWidth, tableSize);
    return result;
  }
  case StaticModuleKind::Fifo: {
    unsigned depth =
        static_cast<unsigned>(std::max<int64_t>(1, getIntAttr(module, "depth", 1)));
    bool bypassable = getIntAttr(module, "bypassable", 0) != 0;
    bool bypassed = getIntAttr(module, "bypassed", 0) != 0;
    return std::make_unique<FifoModule>(module, depth, bypassable, bypassed);
  }
  case StaticModuleKind::SpatialSwitch: {
    std::vector<bool> outputTagged;
    outputTagged.reserve(module.outputPorts.size());
    for (IdIndex portId : module.outputPorts) {
      bool isTagged = false;
      if (const StaticPortDesc *port = findPortById(model, portId))
        isTagged = port->isTagged;
      outputTagged.push_back(isTagged);
    }
    return std::make_unique<SpatialSwitchModule>(
        module, decodeConnectivityFlat(module, module.inputPorts.size(),
                                       module.outputPorts.size()),
        std::move(outputTagged),
        static_cast<unsigned>(module.inputPorts.size()),
        static_cast<unsigned>(module.outputPorts.size()));
  }
  case StaticModuleKind::TemporalSwitch: {
    unsigned tagWidth = 1;
    if (!module.inputPorts.empty()) {
      if (const StaticPortDesc *port = findPortById(model, module.inputPorts.front()))
        tagWidth = std::max(1u, port->tagWidth);
    }
    unsigned slotCount = static_cast<unsigned>(
        std::max<int64_t>(1, getIntAttr(module, "num_route_table", 1)));
    return std::make_unique<TemporalSwitchModule>(
        module, decodeConnectivityFlat(module, module.inputPorts.size(),
                                       module.outputPorts.size()),
        static_cast<unsigned>(module.inputPorts.size()),
        static_cast<unsigned>(module.outputPorts.size()), tagWidth, slotCount);
  }
  case StaticModuleKind::FunctionUnit:
    if (auto modulePtr = createFunctionUnitModule(module, model))
      return modulePtr;
    break;
  case StaticModuleKind::Memory:
  case StaticModuleKind::ExtMemory:
    if (auto modulePtr = createMemoryModule(module, model))
      return modulePtr;
    break;
  case StaticModuleKind::TemporalPE:
    break;
  case StaticModuleKind::Unknown:
    break;
  }
  return std::make_unique<UnsupportedModule>(module);
}

} // namespace sim
} // namespace loom
