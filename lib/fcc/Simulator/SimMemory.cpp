#include "fcc/Simulator/SimMemory.h"

#include <algorithm>
#include <cstdlib>
#include <deque>
#include <iostream>
#include <optional>
#include <sstream>
#include <unordered_map>
#include <utility>
#include <vector>

namespace fcc {
namespace sim {

namespace {

bool simDebugEnabled() {
  const char *env = std::getenv("FCC_SIM_DEBUG");
  return env && env[0] != '\0' && env[0] != '0';
}

int64_t getIntAttr(const StaticModuleDesc &module, const char *name,
                   int64_t defaultValue = 0) {
  for (const auto &attr : module.intAttrs) {
    if (attr.name == name)
      return attr.value;
  }
  return defaultValue;
}

const StaticPortDesc *findPort(const StaticMappedModel &model, IdIndex portId) {
  return model.findPort(portId);
}

struct ConfiguredRegionSlot {
  bool valid = false;
  unsigned startLane = 0;
  unsigned endLane = 0;
  uint64_t baseByteOffset = 0;
  unsigned elemSizeLog2 = 0;
};

struct BoundRegionInfo {
  unsigned regionId = 0;
  unsigned regionIndex = 0;
  unsigned startLane = 0;
  unsigned endLane = 1;
  unsigned elemSizeLog2 = 0;
  bool supportsLoad = false;
  bool supportsStore = false;
};

struct LatchedStoreHalf {
  SimToken token;
};

struct ResolvedRegion {
  unsigned regionId = 0;
  uint64_t byteAddr = 0;
  unsigned elemSizeLog2 = 0;
};

class MemoryModule final : public SimModule {
public:
  MemoryModule(const StaticModuleDesc &module, const StaticMappedModel &model)
      : isExtMemory_(module.kind == StaticModuleKind::ExtMemory),
        ldCount_(static_cast<unsigned>(std::max<int64_t>(
            0, getIntAttr(module, "ldCount", 0)))),
        stCount_(static_cast<unsigned>(std::max<int64_t>(
            0, getIntAttr(module, "stCount", 0)))),
        numRegion_(static_cast<unsigned>(std::max<int64_t>(
            1, getIntAttr(module, "numRegion", 1)))) {
    hwNodeId = module.hwNodeId;
    name = module.name;
    kind = module.kind;

    hasMemrefInput_ = !module.inputPorts.empty();
    if (hasMemrefInput_) {
      if (const StaticPortDesc *port = findPort(model, module.inputPorts.front()))
        hasMemrefInput_ = port->isMemRef;
    }

    unsigned inputBase = hasMemrefInput_ ? 1u : 0u;
    if (ldCount_ > 0)
      ldAddrInputIdx_ = inputBase;
    if (stCount_ > 0) {
      stAddrInputIdx_ = inputBase + (ldCount_ > 0 ? 1u : 0u);
      stDataInputIdx_ = stAddrInputIdx_ + 1u;
    }
    if (ldCount_ > 0) {
      ldDataOutputIdx_ = 0u;
      ldDoneOutputIdx_ = 1u;
    }
    if (stCount_ > 0) {
      stDoneOutputIdx_ = (ldCount_ > 0) ? 2u : 0u;
    }

    outputRegisters_.assign(module.outputPorts.size(), std::nullopt);

    for (const auto &binding : model.getMemoryBindings()) {
      if (binding.hwNodeId != static_cast<IdIndex>(module.hwNodeId))
        continue;
      boundRegions_.push_back(
          {binding.regionId, binding.regionIndex, binding.startLane, binding.endLane,
           binding.elemSizeLog2, binding.supportsLoad, binding.supportsStore});
    }
  }

  bool isCombinational() const override { return false; }

  void bindRuntimeServices(SimRuntimeServices *services) override {
    runtime_ = services;
  }

  void reset() override {
    perf_ = PerfSnapshot();
    outstandingLoadsByLane_.clear();
    outstandingStoresByLane_.clear();
    latchedStoreAddrByLane_.clear();
    latchedStoreDataByLane_.clear();
    pendingResponses_.clear();
    for (auto &reg : outputRegisters_)
      reg.reset();
    consumedInputTokens_.assign(inputs.size(), std::nullopt);
    nextOutputGeneration_ = 1;
    issuedThisCycle_ = false;
    completedThisCycle_ = false;
    loadIssueSelected_ = false;
    storeIssueSelected_ = false;
    selectedStoreLane_.reset();
    inputCaptureCount_ = 0;
    outputTransferCount_ = 0;
    loadIssueCount_ = 0;
    storeIssueCount_ = 0;
    loadCompletionCount_ = 0;
    storeCompletionCount_ = 0;
    if (runtime_ && (hwNodeId == 735 || simDebugEnabled())) {
      std::cerr << "SimMemory reset hw=" << hwNodeId << " boundRegions=";
      for (const auto &region : boundRegions_) {
        std::cerr << " {rid=" << region.regionId << " idx="
                  << region.regionIndex << " lanes=["
                  << region.startLane << "," << region.endLane
                  << ") ld=" << region.supportsLoad
                  << " st=" << region.supportsStore
                  << " elem=" << region.elemSizeLog2 << "}";
      }
      std::cerr << " configuredSlots=";
      for (const auto &slot : configuredSlots_) {
        std::cerr << " {valid=" << slot.valid << " lanes=[" << slot.startLane
                  << "," << slot.endLane << ") base=" << slot.baseByteOffset
                  << " elem=" << slot.elemSizeLog2 << "}";
      }
      std::cerr << "\n";
    }
  }

  void configure(const std::vector<uint32_t> &configWords) override {
    configuredSlots_.clear();
    configuredSlots_.resize(numRegion_);
    for (unsigned slot = 0; slot < numRegion_; ++slot) {
      size_t base = static_cast<size_t>(slot) * 5;
      if (base + 4 < configWords.size()) {
        configuredSlots_[slot].valid = configWords[base + 0] != 0;
        configuredSlots_[slot].startLane =
            static_cast<unsigned>(configWords[base + 1]);
        configuredSlots_[slot].endLane =
            static_cast<unsigned>(configWords[base + 2]);
        configuredSlots_[slot].baseByteOffset = configWords[base + 3];
        configuredSlots_[slot].elemSizeLog2 =
            static_cast<unsigned>(configWords[base + 4]);
      }
    }

    if (configWords.empty()) {
      configuredSlots_.clear();
      unsigned slotCount = 0;
      for (const auto &region : boundRegions_)
        slotCount = std::max(slotCount, region.regionIndex + 1);
      configuredSlots_.resize(slotCount);
      for (const auto &region : boundRegions_) {
        ConfiguredRegionSlot slot;
        slot.valid = true;
        slot.startLane = region.startLane;
        slot.endLane = region.endLane;
        slot.baseByteOffset = isExtMemory_ ? 0 : 0;
        slot.elemSizeLog2 = region.elemSizeLog2;
        configuredSlots_[region.regionIndex] = slot;
      }
    }

    if (hwNodeId == 735 || simDebugEnabled()) {
      std::cerr << "SimMemory configure hw=" << hwNodeId << " boundRegions=";
      for (const auto &region : boundRegions_) {
        std::cerr << " {rid=" << region.regionId << " idx="
                  << region.regionIndex << " lanes=["
                  << region.startLane << "," << region.endLane
                  << ") ld=" << region.supportsLoad
                  << " st=" << region.supportsStore
                  << " elem=" << region.elemSizeLog2 << "}";
      }
      std::cerr << " configuredSlots=";
      for (const auto &slot : configuredSlots_) {
        std::cerr << " {valid=" << slot.valid << " lanes=[" << slot.startLane
                  << "," << slot.endLane << ") base=" << slot.baseByteOffset
                  << " elem=" << slot.elemSizeLog2 << "}";
      }
      std::cerr << "\n";
    }
  }

  void evaluate() override {
    issuedThisCycle_ = false;
    completedThisCycle_ = false;
    loadIssueSelected_ = false;
    storeIssueSelected_ = false;
    selectedStoreLane_.reset();
    driveBufferedOutputs();
    setAllInputReady(false);

    bool loadCanIssue = false;
    unsigned loadLane = 0;
    if (ldAddrInputIdx_ >= 0 && static_cast<size_t>(ldAddrInputIdx_) < inputs.size() &&
        inputs[ldAddrInputIdx_]->valid) {
      loadLane = inputs[ldAddrInputIdx_]->hasTag ? inputs[ldAddrInputIdx_]->tag : 0;
      loadCanIssue =
          inputFresh(ldAddrInputIdx_) &&
          outstandingLoadsByLane_.find(loadLane) ==
              outstandingLoadsByLane_.end();
      if (!resolveRegion(MemoryRequestKind::Load, loadLane,
                         inputs[ldAddrInputIdx_]->data)
               .has_value())
        loadCanIssue = false;
      inputs[ldAddrInputIdx_]->ready =
          loadCanIssue || inputAlreadyConsumed(ldAddrInputIdx_);
      loadIssueSelected_ = loadCanIssue;
    }

    bool storeCanIssue = false;
    if (stAddrInputIdx_ >= 0 && stDataInputIdx_ >= 0 &&
        static_cast<size_t>(stAddrInputIdx_) < inputs.size() &&
        static_cast<size_t>(stDataInputIdx_) < inputs.size()) {
      if (inputs[stAddrInputIdx_]->valid) {
        unsigned addrLane = inputs[stAddrInputIdx_]->hasTag
                                ? inputs[stAddrInputIdx_]->tag
                                : 0;
        bool canCaptureAddr =
            inputFresh(stAddrInputIdx_) &&
            latchedStoreAddrByLane_.find(addrLane) ==
                latchedStoreAddrByLane_.end() &&
            resolveRegion(MemoryRequestKind::Store, addrLane,
                          inputs[stAddrInputIdx_]->data)
                .has_value();
        inputs[stAddrInputIdx_]->ready =
            canCaptureAddr || inputAlreadyConsumed(stAddrInputIdx_);
      }
      if (inputs[stDataInputIdx_]->valid) {
        unsigned dataLane = inputs[stDataInputIdx_]->hasTag
                                ? inputs[stDataInputIdx_]->tag
                                : 0;
        bool canCaptureData =
            inputFresh(stDataInputIdx_) &&
            latchedStoreDataByLane_.find(dataLane) ==
                latchedStoreDataByLane_.end();
        inputs[stDataInputIdx_]->ready =
            canCaptureData || inputAlreadyConsumed(stDataInputIdx_);
      }
      auto issueLane = selectIssuableStoreLane();
      storeIssueSelected_ = issueLane.has_value();
      selectedStoreLane_ = issueLane;
      storeCanIssue = issueLane.has_value();
    }

    if (!loadCanIssue && ldAddrInputIdx_ >= 0 &&
        static_cast<size_t>(ldAddrInputIdx_) < inputs.size() &&
        inputs[ldAddrInputIdx_]->valid)
      perf_.stallCyclesIn++;
    if (!storeCanIssue && stAddrInputIdx_ >= 0 && stDataInputIdx_ >= 0 &&
        static_cast<size_t>(stAddrInputIdx_) < inputs.size() &&
        static_cast<size_t>(stDataInputIdx_) < inputs.size() &&
        (inputs[stAddrInputIdx_]->valid || inputs[stDataInputIdx_]->valid ||
         !latchedStoreAddrByLane_.empty() || !latchedStoreDataByLane_.empty()))
      perf_.stallCyclesIn++;

    if ((hwNodeId == 735 || simDebugEnabled()) && runtime_) {
      bool anyInterestingInput = false;
      for (const SimChannel *input : inputs)
        anyInterestingInput = anyInterestingInput || input->valid;
      anyInterestingInput = anyInterestingInput || !latchedStoreAddrByLane_.empty() ||
                            !latchedStoreDataByLane_.empty() ||
                            !outstandingLoadsByLane_.empty() ||
                            !outstandingStoresByLane_.empty();
      if (anyInterestingInput) {
        std::cerr << "SimMemory evaluate hw=" << hwNodeId
                     << " cycle=" << runtime_->getCurrentCycle()
                     << " loadCanIssue=" << loadCanIssue
                     << " loadSelected=" << loadIssueSelected_
                     << " storeCanIssue=" << storeCanIssue
                     << " storeSelected=" << storeIssueSelected_
                     << " outstanding_loads=" << outstandingLoadsByLane_.size()
                     << " outstanding_stores="
                     << outstandingStoresByLane_.size() << "\n";
        for (size_t idx = 0; idx < inputs.size(); ++idx) {
          const SimChannel *input = inputs[idx];
          std::cerr << "  in" << idx << " v=" << input->valid
                       << " r=" << input->ready << " d=" << input->data
                       << " t=" << input->tag << " ht=" << input->hasTag
                       << " g=" << input->generation << "\n";
        }
        if (ldAddrInputIdx_ >= 0 &&
            static_cast<size_t>(ldAddrInputIdx_) < inputs.size() &&
            inputs[ldAddrInputIdx_]->valid) {
          unsigned lane =
              inputs[ldAddrInputIdx_]->hasTag ? inputs[ldAddrInputIdx_]->tag : 0;
          auto resolved = resolveRegion(MemoryRequestKind::Load, lane,
                                        inputs[ldAddrInputIdx_]->data);
          std::cerr << "  load lane=" << lane
                       << " fresh=" << inputFresh(ldAddrInputIdx_)
                       << " consumed=" << inputAlreadyConsumed(ldAddrInputIdx_);
          if (resolved) {
            std::cerr << " region=" << resolved->regionId
                         << " byteAddr=" << resolved->byteAddr
                         << " elemLog2=" << resolved->elemSizeLog2;
          } else {
            std::cerr << " region=NONE boundRegions=";
            for (const auto &region : boundRegions_) {
              std::cerr << " {rid=" << region.regionId << " idx="
                           << region.regionIndex << " lanes=["
                           << region.startLane << "," << region.endLane
                           << ") ld=" << region.supportsLoad
                           << " st=" << region.supportsStore << "}";
            }
          }
          std::cerr << "\n";
        }
      }
    }
  }

  void commit() override {
    drainTransferredOutputs();
    reapCompletions();
    materializePendingResponses();

    if (!runtime_)
      return;

    if (loadIssueSelected_ && ldAddrInputIdx_ >= 0 &&
        static_cast<size_t>(ldAddrInputIdx_) < inputs.size() &&
        inputs[ldAddrInputIdx_]->transferred()) {
      SimChannel *addr = inputs[ldAddrInputIdx_];
      unsigned lane = addr->hasTag ? addr->tag : 0;
      auto resolved = resolveRegion(MemoryRequestKind::Load, lane, addr->data);
      if (resolved.has_value()) {
        uint64_t requestId = 0;
        std::string error;
        if (runtime_->issueMemoryLoad(hwNodeId, resolved->regionId,
                                      resolved->byteAddr,
                                      1u << resolved->elemSizeLog2,
                                      addr->tag, addr->hasTag, requestId,
                                      error)) {
          if (hwNodeId == 735 || simDebugEnabled()) {
            std::cerr << "SimMemory issue load hw=" << hwNodeId
                         << " lane=" << lane
                         << " region=" << resolved->regionId
                         << " byteAddr=" << resolved->byteAddr
                         << " bytes=" << (1u << resolved->elemSizeLog2)
                         << " req=" << requestId
                         << " tag=" << addr->tag
                         << " hasTag=" << addr->hasTag << "\n";
          }
          outstandingLoadsByLane_[lane] = requestId;
          markInputConsumed(ldAddrInputIdx_);
          ++loadIssueCount_;
          ++perf_.tokensIn;
          ++perf_.activeCycles;
          issuedThisCycle_ = true;
        }
      }
    }

    if (stAddrInputIdx_ >= 0 && static_cast<size_t>(stAddrInputIdx_) < inputs.size() &&
        inputs[stAddrInputIdx_]->transferred()) {
      unsigned lane =
          inputs[stAddrInputIdx_]->hasTag ? inputs[stAddrInputIdx_]->tag : 0;
      if (latchedStoreAddrByLane_.find(lane) == latchedStoreAddrByLane_.end()) {
        latchedStoreAddrByLane_[lane] = {tokenFromChannel(*inputs[stAddrInputIdx_])};
        markInputConsumed(stAddrInputIdx_);
        ++inputCaptureCount_;
        ++perf_.tokensIn;
        ++perf_.activeCycles;
        issuedThisCycle_ = true;
      }
    }
    if (stDataInputIdx_ >= 0 && static_cast<size_t>(stDataInputIdx_) < inputs.size() &&
        inputs[stDataInputIdx_]->transferred()) {
      unsigned lane =
          inputs[stDataInputIdx_]->hasTag ? inputs[stDataInputIdx_]->tag : 0;
      if (latchedStoreDataByLane_.find(lane) == latchedStoreDataByLane_.end()) {
        latchedStoreDataByLane_[lane] = {tokenFromChannel(*inputs[stDataInputIdx_])};
        markInputConsumed(stDataInputIdx_);
        ++inputCaptureCount_;
        ++perf_.tokensIn;
        ++perf_.activeCycles;
        issuedThisCycle_ = true;
      }
    }

    if (storeIssueSelected_ && selectedStoreLane_.has_value()) {
      unsigned lane = *selectedStoreLane_;
      auto addrIt = latchedStoreAddrByLane_.find(lane);
      auto dataIt = latchedStoreDataByLane_.find(lane);
      if (addrIt != latchedStoreAddrByLane_.end() &&
          dataIt != latchedStoreDataByLane_.end()) {
        auto resolved = resolveRegion(MemoryRequestKind::Store, lane,
                                      addrIt->second.token.data);
        if (resolved.has_value()) {
          uint64_t requestId = 0;
          std::string error;
          bool hasTag = addrIt->second.token.hasTag || dataIt->second.token.hasTag;
          uint16_t tag = addrIt->second.token.hasTag ? addrIt->second.token.tag
                                                     : dataIt->second.token.tag;
          if (runtime_->issueMemoryStore(
                  hwNodeId, resolved->regionId, resolved->byteAddr,
                  dataIt->second.token.data, 1u << resolved->elemSizeLog2, tag,
                  hasTag, requestId, error)) {
            if (hwNodeId == 735 || simDebugEnabled()) {
              std::cerr << "SimMemory issue store hw=" << hwNodeId
                           << " lane=" << lane
                           << " region=" << resolved->regionId
                           << " byteAddr=" << resolved->byteAddr
                           << " bytes=" << (1u << resolved->elemSizeLog2)
                           << " data=" << dataIt->second.token.data
                           << " req=" << requestId << "\n";
            }
            outstandingStoresByLane_[lane] = requestId;
            latchedStoreAddrByLane_.erase(addrIt);
            latchedStoreDataByLane_.erase(dataIt);
            ++storeIssueCount_;
            ++perf_.activeCycles;
            issuedThisCycle_ = true;
          }
        }
      }
    }
  }

  bool hasPendingWork() const override {
    if (!pendingResponses_.empty())
      return true;
    if (!outstandingLoadsByLane_.empty() || !outstandingStoresByLane_.empty())
      return true;
    return std::any_of(outputRegisters_.begin(), outputRegisters_.end(),
                       [](const auto &reg) { return reg.has_value(); });
  }

  uint64_t getLogicalFireCount() const override {
    return loadIssueCount_ + storeIssueCount_;
  }

  uint64_t getInputCaptureCount() const override { return inputCaptureCount_; }

  uint64_t getOutputTransferCount() const override {
    return outputTransferCount_;
  }

  std::vector<NamedCounter> getDebugCounters() const override {
    std::vector<NamedCounter> counters;
    auto pushIfNonZero = [&](const char *name, uint64_t value) {
      if (value != 0)
        counters.push_back({name, value});
    };
    pushIfNonZero("load_issue_count", loadIssueCount_);
    pushIfNonZero("store_issue_count", storeIssueCount_);
    pushIfNonZero("load_completion_count", loadCompletionCount_);
    pushIfNonZero("store_completion_count", storeCompletionCount_);
    return counters;
  }

  std::string getDebugStateSummary() const override {
    std::ostringstream os;
    os << "outstanding(load=" << outstandingLoadsByLane_.size()
       << ",store=" << outstandingStoresByLane_.size() << ")"
       << " latched(addr=" << latchedStoreAddrByLane_.size()
       << ",data=" << latchedStoreDataByLane_.size() << ")"
       << " pendingResponses=" << pendingResponses_.size();
    return os.str();
  }

  void collectTraceEvents(std::vector<TraceEvent> &events,
                          uint64_t cycle) override {
    if (!issuedThisCycle_ && !completedThisCycle_)
      return;
    TraceEvent ev;
    ev.cycle = cycle;
    ev.phase = SimPhase::Commit;
    ev.hwNodeId = hwNodeId;
    ev.eventKind = EventKind::NodeFire;
    events.push_back(ev);
  }

  PerfSnapshot getPerfSnapshot() const override { return perf_; }

private:
  bool inputFresh(int idx) const {
    return idx >= 0 && static_cast<size_t>(idx) < inputs.size() &&
           inputs[idx]->valid && inputs[idx]->generation != 0 &&
           !inputAlreadyConsumed(idx);
  }

  bool inputAlreadyConsumed(int idx) const {
    if (idx < 0 || static_cast<size_t>(idx) >= inputs.size())
      return false;
    if (!inputs[idx]->valid || inputs[idx]->generation == 0)
      return false;
    if (!consumedInputTokens_[idx].has_value())
      return false;
    const SimToken current = tokenFromChannel(*inputs[idx]);
    return current.generation == consumedInputTokens_[idx]->generation &&
           current.hasTag == consumedInputTokens_[idx]->hasTag &&
           current.tag == consumedInputTokens_[idx]->tag &&
           current.data == consumedInputTokens_[idx]->data;
  }

  void markInputConsumed(int idx) {
    if (idx < 0 || static_cast<size_t>(idx) >= inputs.size())
      return;
    consumedInputTokens_[idx] = tokenFromChannel(*inputs[idx]);
  }

  SimToken makeGeneratedToken(uint64_t data, uint16_t tag, bool hasTag,
                              uint64_t generation = 0) {
    SimToken token;
    token.data = data;
    token.tag = tag;
    token.hasTag = hasTag;
    token.generation =
        generation ? generation : composeTokenGeneration(hwNodeId, nextOutputGeneration_++);
    return token;
  }

  struct PendingResponse {
    MemoryRequestKind kind = MemoryRequestKind::Load;
    uint64_t data = 0;
    uint16_t tag = 0;
    bool hasTag = false;
  };

  void setAllInputReady(bool ready) {
    for (int idx = 0; idx < static_cast<int>(inputs.size()); ++idx) {
      if (ready) {
        inputs[idx]->ready = true;
        continue;
      }
      inputs[idx]->ready = inputAlreadyConsumed(idx);
    }
  }

  void driveBufferedOutputs() {
    for (size_t idx = 0; idx < outputs.size(); ++idx) {
      if (idx < outputRegisters_.size() && outputRegisters_[idx].has_value()) {
        driveChannelFromToken(*outputs[idx], *outputRegisters_[idx]);
      } else {
        outputs[idx]->valid = false;
      }
    }
  }

  void drainTransferredOutputs() {
    bool any = false;
    for (size_t idx = 0; idx < outputs.size() && idx < outputRegisters_.size();
         ++idx) {
      if (!outputRegisters_[idx].has_value() || !outputs[idx]->transferred())
        continue;
      outputRegisters_[idx].reset();
      ++perf_.tokensOut;
      ++outputTransferCount_;
      any = true;
    }
    if (any)
      ++perf_.activeCycles;
  }

  std::optional<ResolvedRegion> resolveRegion(MemoryRequestKind kind,
                                              unsigned lane,
                                              uint64_t logicalIndex) const {
    auto kindMatches = [&](const BoundRegionInfo &binding) -> bool {
      return kind == MemoryRequestKind::Load ? binding.supportsLoad
                                             : binding.supportsStore;
    };
    auto makeResolved = [&](const BoundRegionInfo &binding,
                            unsigned elemSizeLog2,
                            uint64_t baseByteOffset) -> ResolvedRegion {
      ResolvedRegion resolved;
      resolved.regionId = binding.regionId;
      resolved.elemSizeLog2 = elemSizeLog2;
      resolved.byteAddr = baseByteOffset + (logicalIndex << resolved.elemSizeLog2);
      return resolved;
    };
    for (size_t slotIdx = 0; slotIdx < configuredSlots_.size(); ++slotIdx) {
      const auto &slot = configuredSlots_[slotIdx];
      if (!slot.valid || lane < slot.startLane || lane >= slot.endLane)
        continue;
      for (const auto &binding : boundRegions_) {
        if (!kindMatches(binding))
          continue;
        if (binding.regionIndex != slotIdx)
          continue;
        if (binding.startLane != slot.startLane || binding.endLane != slot.endLane)
          continue;
        return makeResolved(binding, slot.elemSizeLog2, slot.baseByteOffset);
      }
    }

    for (const auto &binding : boundRegions_) {
      if (lane < binding.startLane || lane >= binding.endLane ||
          !kindMatches(binding))
        continue;
      return makeResolved(binding, binding.elemSizeLog2, 0);
    }
    return std::nullopt;
  }

  std::optional<unsigned> selectIssuableStoreLane() const {
    std::optional<unsigned> selected;
    for (const auto &it : latchedStoreAddrByLane_) {
      unsigned lane = it.first;
      if (latchedStoreDataByLane_.find(lane) == latchedStoreDataByLane_.end())
        continue;
      if (outstandingStoresByLane_.find(lane) != outstandingStoresByLane_.end())
        continue;
      if (!resolveRegion(MemoryRequestKind::Store, lane, it.second.token.data)
               .has_value())
        continue;
      if (!selected.has_value() || lane < *selected)
        selected = lane;
    }
    return selected;
  }

  void reapCompletions() {
    if (!runtime_)
      return;
    std::vector<unsigned> completedLoadLanes;
    std::vector<unsigned> completedStoreLanes;
    completedLoadLanes.reserve(outstandingLoadsByLane_.size());
    completedStoreLanes.reserve(outstandingStoresByLane_.size());
    for (const auto &it : outstandingLoadsByLane_) {
      MemoryCompletion completion;
      if (!runtime_->takeMemoryCompletion(it.second, completion))
        continue;
      PendingResponse response;
      response.kind = completion.kind;
      response.tag = completion.tag;
      response.hasTag = completion.hasTag;
      response.data = completion.data;
      if (hwNodeId == 735 || simDebugEnabled()) {
        std::cerr << "SimMemory completion hw=" << hwNodeId
                     << " lane=" << it.first
                     << " req=" << it.second
                     << " kind="
                     << (completion.kind == MemoryRequestKind::Load ? "load"
                                                                    : "store")
                     << " data=" << completion.data
                     << " tag=" << completion.tag
                     << " hasTag=" << completion.hasTag << "\n";
      }
      pendingResponses_.push_back(std::move(response));
      ++loadCompletionCount_;
      completedLoadLanes.push_back(it.first);
      completedThisCycle_ = true;
    }
    for (const auto &it : outstandingStoresByLane_) {
      MemoryCompletion completion;
      if (!runtime_->takeMemoryCompletion(it.second, completion))
        continue;
      PendingResponse response;
      response.kind = completion.kind;
      response.tag = completion.tag;
      response.hasTag = completion.hasTag;
      response.data = completion.data;
      if (hwNodeId == 735 || simDebugEnabled()) {
        std::cerr << "SimMemory completion hw=" << hwNodeId
                     << " store-lane=" << it.first
                     << " req=" << it.second
                     << " kind="
                     << (completion.kind == MemoryRequestKind::Load ? "load"
                                                                    : "store")
                     << " data=" << completion.data
                     << " tag=" << completion.tag
                     << " hasTag=" << completion.hasTag << "\n";
      }
      pendingResponses_.push_back(std::move(response));
      ++storeCompletionCount_;
      completedStoreLanes.push_back(it.first);
      completedThisCycle_ = true;
    }
    for (unsigned lane : completedLoadLanes)
      outstandingLoadsByLane_.erase(lane);
    for (unsigned lane : completedStoreLanes)
      outstandingStoresByLane_.erase(lane);
  }

  void materializePendingResponses() {
    while (!pendingResponses_.empty()) {
      const PendingResponse &response = pendingResponses_.front();
      if (!canAcceptResponse(response))
        break;
      placeResponse(response);
      pendingResponses_.pop_front();
    }
  }

  bool canAcceptResponse(const PendingResponse &response) const {
    if (response.kind == MemoryRequestKind::Load) {
      return ldDataOutputIdx_ >= 0 && ldDoneOutputIdx_ >= 0 &&
             !outputRegisters_[ldDataOutputIdx_].has_value() &&
             !outputRegisters_[ldDoneOutputIdx_].has_value();
    }
    return stDoneOutputIdx_ >= 0 && !outputRegisters_[stDoneOutputIdx_].has_value();
  }

  void placeResponse(const PendingResponse &response) {
    uint64_t generation = composeTokenGeneration(hwNodeId, nextOutputGeneration_++);
    if (hwNodeId == 735 || simDebugEnabled()) {
      std::cerr << "SimMemory place response hw=" << hwNodeId
                   << " kind="
                   << (response.kind == MemoryRequestKind::Load ? "load"
                                                                : "store")
                   << " data=" << response.data
                   << " tag=" << response.tag
                   << " hasTag=" << response.hasTag
                   << " gen=" << generation << "\n";
    }
    if (response.kind == MemoryRequestKind::Load) {
      outputRegisters_[ldDataOutputIdx_] =
          makeGeneratedToken(response.data, response.tag, response.hasTag,
                             generation);
      outputRegisters_[ldDoneOutputIdx_] =
          makeGeneratedToken(0, response.tag, response.hasTag, generation);
    } else if (stDoneOutputIdx_ >= 0) {
      outputRegisters_[stDoneOutputIdx_] =
          makeGeneratedToken(0, response.tag, response.hasTag, generation);
    }
  }

  SimRuntimeServices *runtime_ = nullptr;
  bool isExtMemory_ = false;
  unsigned ldCount_ = 0;
  unsigned stCount_ = 0;
  unsigned numRegion_ = 1;
  bool hasMemrefInput_ = false;
  int ldAddrInputIdx_ = -1;
  int stAddrInputIdx_ = -1;
  int stDataInputIdx_ = -1;
  int ldDataOutputIdx_ = -1;
  int ldDoneOutputIdx_ = -1;
  int stDoneOutputIdx_ = -1;
  std::vector<BoundRegionInfo> boundRegions_;
  std::vector<ConfiguredRegionSlot> configuredSlots_;
  std::unordered_map<unsigned, uint64_t> outstandingLoadsByLane_;
  std::unordered_map<unsigned, uint64_t> outstandingStoresByLane_;
  std::unordered_map<unsigned, LatchedStoreHalf> latchedStoreAddrByLane_;
  std::unordered_map<unsigned, LatchedStoreHalf> latchedStoreDataByLane_;
  std::deque<PendingResponse> pendingResponses_;
  std::vector<std::optional<SimToken>> outputRegisters_;
  std::vector<std::optional<SimToken>> consumedInputTokens_;
  uint64_t nextOutputGeneration_ = 1;
  bool issuedThisCycle_ = false;
  bool completedThisCycle_ = false;
  bool loadIssueSelected_ = false;
  bool storeIssueSelected_ = false;
  std::optional<unsigned> selectedStoreLane_;
  uint64_t inputCaptureCount_ = 0;
  uint64_t outputTransferCount_ = 0;
  uint64_t loadIssueCount_ = 0;
  uint64_t storeIssueCount_ = 0;
  uint64_t loadCompletionCount_ = 0;
  uint64_t storeCompletionCount_ = 0;
};

} // namespace

std::unique_ptr<SimModule> createMemoryModule(const StaticModuleDesc &module,
                                              const StaticMappedModel &model) {
  if (module.kind != StaticModuleKind::Memory &&
      module.kind != StaticModuleKind::ExtMemory)
    return nullptr;
  return std::make_unique<MemoryModule>(module, model);
}

} // namespace sim
} // namespace fcc
