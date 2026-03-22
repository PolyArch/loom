#include "loom/Simulator/CycleKernel.h"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace loom {
namespace sim {

namespace {

constexpr uint64_t kIdleCyclesForBoundary = 32;
constexpr unsigned kMaxCombIterations = 4;

bool simDebugEnabled() {
  const char *env = std::getenv("LOOM_SIM_DEBUG");
  return env && env[0] != '\0' && env[0] != '0';
}

bool channelEquals(const SimChannel &lhs, const SimChannel &rhs) {
  return lhs.valid == rhs.valid && lhs.ready == rhs.ready &&
         lhs.data == rhs.data && lhs.tag == rhs.tag &&
         lhs.hasTag == rhs.hasTag && lhs.generation == rhs.generation;
}

bool hasOutstandingCompletionObligations(const StaticMappedModel &staticModel) {
  return !staticModel.getCompletionObligations().empty();
}

unsigned computeCombIterationBudget(const StaticMappedModel &model) {
  unsigned structuralBound =
      static_cast<unsigned>(std::max<size_t>(model.getChannels().size(),
                                             model.getModules().size()));
  structuralBound = std::max<unsigned>(structuralBound, 1);
  return std::max(kMaxCombIterations, structuralBound);
}

const char *moduleKindName(StaticModuleKind kind) {
  switch (kind) {
  case StaticModuleKind::BoundaryInput:
    return "boundary_input";
  case StaticModuleKind::BoundaryOutput:
    return "boundary_output";
  case StaticModuleKind::FunctionUnit:
    return "function_unit";
  case StaticModuleKind::SpatialSwitch:
    return "spatial_sw";
  case StaticModuleKind::TemporalSwitch:
    return "temporal_sw";
  case StaticModuleKind::AddTag:
    return "add_tag";
  case StaticModuleKind::MapTag:
    return "map_tag";
  case StaticModuleKind::DelTag:
    return "del_tag";
  case StaticModuleKind::Fifo:
    return "fifo";
  case StaticModuleKind::Memory:
    return "memory";
  case StaticModuleKind::ExtMemory:
    return "extmemory";
  case StaticModuleKind::TemporalPE:
    return "temporal_pe";
  case StaticModuleKind::Unknown:
    return "unknown";
  }
  return "unknown";
}

void dumpBudgetHitDebug(const StaticMappedModel &staticModel,
                        const std::vector<std::unique_ptr<SimModule>> &modules,
                        const std::vector<SimChannel> &ports,
                        const std::vector<SimChannel> &edges,
                        size_t outstandingCount,
                        bool obligationsSatisfied, bool quiescent, bool done,
                        bool deadlocked, uint64_t cycle) {
  if (!simDebugEnabled())
    return;
  std::cerr << "CycleKernel budget-hit summary cycle=" << cycle
            << " obligationsSatisfied=" << obligationsSatisfied
            << " quiescent=" << quiescent << " done=" << done
            << " deadlocked=" << deadlocked
            << " outstanding_mem=" << outstandingCount << "\n";

  unsigned validPorts = 0;
  for (const auto &port : staticModel.getPorts()) {
    if (ports[port.portId].valid)
      ++validPorts;
  }
  std::cerr << "  valid_ports=" << validPorts << "\n";
  for (const auto &port : staticModel.getPorts()) {
    if (!ports[port.portId].valid)
      continue;
    std::cerr << "    port " << port.portId << " node=" << port.parentNodeId
              << " dir="
              << (port.direction == StaticPortDirection::Input ? "in" : "out")
              << " g=" << ports[port.portId].generation
              << " d=" << ports[port.portId].data
              << " t=" << ports[port.portId].tag
              << " ht=" << ports[port.portId].hasTag << "\n";
  }

  unsigned validEdges = 0;
  for (size_t edgeIdx = 0; edgeIdx < edges.size(); ++edgeIdx) {
    if (!edges[edgeIdx].valid)
      continue;
    ++validEdges;
    const StaticChannelDesc &channel = staticModel.getChannels()[edgeIdx];
    std::cerr << "    edge[" << edgeIdx << "] hw=" << channel.hwEdgeId << " "
              << channel.srcPort << "->" << channel.dstPort
              << " g=" << edges[edgeIdx].generation
              << " d=" << edges[edgeIdx].data << " t=" << edges[edgeIdx].tag
              << " ht=" << edges[edgeIdx].hasTag << "\n";
  }
  std::cerr << "  valid_edges=" << validEdges << "\n";

  for (const auto &module : modules) {
    bool interesting =
        module->hasPendingWork() || !module->getCollectedTokens().empty();
    if (!interesting) {
      for (const SimChannel *ch : module->inputs)
        interesting = interesting || ch->valid;
      for (const SimChannel *ch : module->outputs)
        interesting = interesting || ch->valid;
    }
    if (!interesting)
      continue;
    std::cerr << "  module hw=" << module->hwNodeId << " name="
              << module->name << " kind="
              << static_cast<unsigned>(module->kind)
              << " pending=" << module->hasPendingWork()
              << " collected=" << module->getCollectedTokens().size() << "\n";
    module->debugDump(std::cerr);
    const StaticModuleDesc *moduleDesc =
        staticModel.findModule(static_cast<IdIndex>(module->hwNodeId));
    for (size_t i = 0; i < module->inputs.size(); ++i) {
      const SimChannel *ch = module->inputs[i];
      std::cerr << "    in" << i << " v=" << ch->valid << " r=" << ch->ready
                << " d=" << ch->data << " t=" << ch->tag
                << " ht=" << ch->hasTag << " g=" << ch->generation << "\n";
      if (moduleDesc && i < moduleDesc->inputPorts.size())
        std::cerr << "      portId=" << moduleDesc->inputPorts[i] << "\n";
    }
    for (size_t i = 0; i < module->outputs.size(); ++i) {
      const SimChannel *ch = module->outputs[i];
      std::cerr << "    out" << i << " v=" << ch->valid << " r=" << ch->ready
                << " d=" << ch->data << " t=" << ch->tag
                << " ht=" << ch->hasTag << " g=" << ch->generation << "\n";
      if (moduleDesc && i < moduleDesc->outputPorts.size())
        std::cerr << "      portId=" << moduleDesc->outputPorts[i] << "\n";
    }
  }
}

} // namespace

bool CycleKernel::completionObligationsSatisfied() const {
  for (const auto &obligation : staticModel_.getCompletionObligations()) {
    if (obligation.kind == CompletionObligationKind::OutputPort) {
      if (obligation.ordinal >= boundaryOutputModuleIndex_.size())
        return false;
      const auto &tokens = getOutputTokens(obligation.ordinal);
      if (tokens.empty())
        return false;
      continue;
    }
    if (obligation.kind == CompletionObligationKind::MemoryRegion &&
        (regionHasOutstandingRequests(obligation.ordinal) ||
         !memoryRegionCompletionObserved(obligation.ordinal))) {
      return false;
    }
  }
  return true;
}

bool CycleKernel::hardwareEmpty(std::string *details) const {
  std::vector<std::string> parts;

  if (!completedMemoryRequests_.empty() || !outstandingMemoryRequests_.empty()) {
    parts.push_back("memory(outstanding=" +
                    std::to_string(outstandingMemoryRequests_.size()) +
                    ", completed=" +
                    std::to_string(completedMemoryRequests_.size()) + ")");
  }

  size_t liveEdgeCount = 0;
  std::vector<std::string> edgeParts;
  for (size_t edgeIdx = 0; edgeIdx < edgeState_.size(); ++edgeIdx) {
    const SimChannel &state = edgeState_[edgeIdx];
    if (!state.valid)
      continue;
    ++liveEdgeCount;
    if (edgeParts.size() < 4) {
      const auto &edge = staticModel_.getChannels()[edgeIdx];
      edgeParts.push_back("edge#" + std::to_string(edgeIdx) + "(hw=" +
                          std::to_string(edge.hwEdgeId) + ", gen=" +
                          std::to_string(state.generation) + ")");
    }
  }
  if (liveEdgeCount != 0) {
    std::string text = "live_edges=" + std::to_string(liveEdgeCount);
    if (!edgeParts.empty()) {
      text += " [";
      for (size_t idx = 0; idx < edgeParts.size(); ++idx) {
        if (idx)
          text += ", ";
        text += edgeParts[idx];
      }
      text += "]";
    }
    parts.push_back(std::move(text));
  }

  std::vector<std::string> pendingModules;
  for (const auto &module : modules_) {
    if (!module->hasPendingWork())
      continue;
    if (pendingModules.size() < 6) {
      std::string text = module->name;
      std::string state = module->getDebugStateSummary();
      if (!state.empty())
        text += "{" + state + "}";
      pendingModules.push_back(std::move(text));
    }
  }
  if (!pendingModules.empty()) {
    std::string text = "pending_modules=" + std::to_string(pendingModules.size());
    text += " [";
    for (size_t idx = 0; idx < pendingModules.size(); ++idx) {
      if (idx)
        text += ", ";
      text += pendingModules[idx];
    }
    text += "]";
    parts.push_back(std::move(text));
  }

  if (details) {
    details->clear();
    for (size_t idx = 0; idx < parts.size(); ++idx) {
      if (idx)
        *details += "; ";
      *details += parts[idx];
    }
  }
  return parts.empty();
}

bool CycleKernel::validateSuccessfulTermination(std::string &error) const {
  error.clear();
  if (!completionObligationsSatisfied()) {
    error = "termination audit failed: software-visible completion obligations "
            "are not satisfied";
    return false;
  }
  std::string hardwareDetails;
  if (!hardwareEmpty(&hardwareDetails)) {
    error =
        "termination audit failed: invocation completed but hardware is not "
        "empty";
    if (!hardwareDetails.empty())
      error += " (" + hardwareDetails + ")";
    return false;
  }
  return true;
}

bool CycleKernel::hasPendingInternalWork() const {
  return hasPendingModuleOrMemoryWork();
}

void CycleKernel::rebuildVisibleEdgeSignals() {
  std::fill(edgeState_.begin(), edgeState_.end(), SimChannel());
  for (size_t edgeIdx = 0; edgeIdx < staticModel_.getChannels().size();
       ++edgeIdx) {
    const StaticChannelDesc &edge = staticModel_.getChannels()[edgeIdx];
    if (edge.srcPort == INVALID_ID ||
        edge.srcPort >= static_cast<IdIndex>(portState_.size())) {
      continue;
    }
    const SimChannel &src = portState_[edge.srcPort];
    if (!src.valid)
      continue;

    bool visible = true;
    if (edge.srcPort < static_cast<IdIndex>(outputFanoutState_.size())) {
      const OutputFanoutState &fanout = outputFanoutState_[edge.srcPort];
      if (fanout.generation == src.generation &&
          edge.srcPort < static_cast<IdIndex>(outputChannelIndices_.size())) {
        const auto &edgeIndices = outputChannelIndices_[edge.srcPort];
        for (size_t localIdx = 0; localIdx < edgeIndices.size(); ++localIdx) {
          if (edgeIndices[localIdx] != static_cast<unsigned>(edgeIdx))
            continue;
          if (localIdx < fanout.captured.size() && fanout.captured[localIdx] != 0)
            visible = false;
          break;
        }
      }
    }
    if (!visible)
      continue;

    SimChannel &dst = edgeState_[edgeIdx];
    dst.valid = true;
    dst.data = src.data;
    dst.tag = src.tag;
    dst.hasTag = src.hasTag;
    dst.generation = src.generation;
  }
}

bool CycleKernel::hasPendingModuleOrMemoryWork() const {
  if (!completedMemoryRequests_.empty() || !outstandingMemoryRequests_.empty())
    return true;
  for (const auto &module : modules_) {
    if (module->hasPendingWork())
      return true;
  }
  return false;
}

bool CycleKernel::memoryRegionCompletionObserved(unsigned regionId) const {
  return regionId < completedStoreRegions_.size() &&
         completedStoreRegions_[regionId] != 0;
}

FinalStateSummary CycleKernel::getFinalStateSummary() const {
  FinalStateSummary summary;
  summary.obligationsSatisfied = completionObligationsSatisfied();
  summary.hardwareEmpty = hardwareEmpty(&summary.terminationAuditError);
  summary.quiescent = quiescent_;
  summary.done = done_;
  summary.deadlocked = deadlocked_;
  summary.idleCycleStreak = idleCycleStreak_;
  summary.outstandingMemoryRequestCount = outstandingMemoryRequests_.size();
  summary.completedMemoryResponseCount = completedMemoryRequests_.size();

  summary.livePorts.reserve(staticModel_.getPorts().size());
  for (const auto &port : staticModel_.getPorts()) {
    if (port.portId >= portState_.size())
      continue;
    const SimChannel &state = portState_[port.portId];
    if (!state.valid)
      continue;
    FinalStatePortSnapshot snap;
    snap.portId = port.portId;
    snap.parentNodeId = port.parentNodeId;
    snap.isInput = (port.direction == StaticPortDirection::Input);
    snap.valid = state.valid;
    snap.ready = state.ready;
    snap.data = state.data;
    snap.tag = state.tag;
    snap.hasTag = state.hasTag;
    snap.generation = state.generation;
    summary.livePorts.push_back(std::move(snap));
  }

  summary.liveEdges.reserve(staticModel_.getChannels().size());
  for (size_t edgeIdx = 0; edgeIdx < staticModel_.getChannels().size();
       ++edgeIdx) {
    if (edgeIdx >= edgeState_.size())
      continue;
    const SimChannel &state = edgeState_[edgeIdx];
    if (!state.valid)
      continue;
    const StaticChannelDesc &edge = staticModel_.getChannels()[edgeIdx];
    FinalStateEdgeSnapshot snap;
    snap.edgeIndex = static_cast<uint32_t>(edgeIdx);
    snap.hwEdgeId = edge.hwEdgeId;
    snap.srcPort = edge.srcPort;
    snap.dstPort = edge.dstPort;
    snap.valid = state.valid;
    snap.ready = state.ready;
    snap.data = state.data;
    snap.tag = state.tag;
    snap.hasTag = state.hasTag;
    snap.generation = state.generation;
    summary.liveEdges.push_back(std::move(snap));
  }

  summary.pendingModules.reserve(modules_.size());
  summary.moduleSummaries.reserve(modules_.size());
  for (const auto &module : modules_) {
    bool pending = module->hasPendingWork();
    uint64_t collectedCount = module->getCollectedTokens().size();
    FinalStateModuleSnapshot snap;
    snap.hwNodeId = module->hwNodeId;
    snap.name = module->name;
    snap.kind = moduleKindName(module->kind);
    snap.hasPendingWork = pending;
    snap.collectedTokenCount = collectedCount;
    snap.logicalFireCount = module->getLogicalFireCount();
    snap.inputCaptureCount = module->getInputCaptureCount();
    snap.outputTransferCount = module->getOutputTransferCount();
    snap.debugState = module->getDebugStateSummary();
    snap.counters = module->getDebugCounters();
    bool hasLivePort = false;
    for (const SimChannel *ch : module->inputs)
      hasLivePort = hasLivePort || (ch != nullptr && ch->valid);
    for (const SimChannel *ch : module->outputs)
      hasLivePort = hasLivePort || (ch != nullptr && ch->valid);
    bool interesting =
        pending || collectedCount != 0 || snap.logicalFireCount != 0 ||
        snap.inputCaptureCount != 0 || snap.outputTransferCount != 0 ||
        !snap.debugState.empty() || !snap.counters.empty() || hasLivePort;
    if (interesting)
      summary.moduleSummaries.push_back(snap);
    if (pending)
      summary.pendingModules.push_back(std::move(snap));
  }

  return summary;
}

void CycleKernel::evaluateBoundaryState() {
  if (!built_ || !configured_) {
    quiescent_ = false;
    done_ = false;
    deadlocked_ = false;
    lastBoundaryReason_ = BoundaryReason::None;
    return;
  }

  bool idleThisCycle = lastTransferCount_ == 0 && lastActivityCount_ == 0 &&
                       outstandingMemoryRequests_.empty() &&
                       completedMemoryRequests_.empty();
  if (idleThisCycle)
    ++idleCycleStreak_;
  else
    idleCycleStreak_ = 0;

  bool pendingInternalWork = hasPendingInternalWork();
  if (externalMemoryMode_ && !outgoingMemoryRequests_.empty()) {
    quiescent_ = false;
    done_ = false;
    deadlocked_ = false;
    lastBoundaryReason_ = BoundaryReason::NeedMemIssue;
    return;
  }
  if (externalMemoryMode_ && outgoingMemoryRequests_.empty() &&
      !outstandingMemoryRequests_.empty() && completedMemoryRequests_.empty() &&
      idleThisCycle) {
    quiescent_ = false;
    done_ = false;
    deadlocked_ = false;
    lastBoundaryReason_ = BoundaryReason::WaitMemResp;
    return;
  }
  quiescent_ = idleCycleStreak_ >= kIdleCyclesForBoundary;
  bool obligationsSatisfied = completionObligationsSatisfied();
  done_ = obligationsSatisfied && quiescent_;
  deadlocked_ =
      !obligationsSatisfied && quiescent_ &&
      (pendingInternalWork || hasOutstandingCompletionObligations(staticModel_));

  if (done_)
    lastBoundaryReason_ = BoundaryReason::InvocationDone;
  else if (deadlocked_)
    lastBoundaryReason_ = BoundaryReason::Deadlock;
  else
    lastBoundaryReason_ = BoundaryReason::None;
}

void CycleKernel::stepCycle() {
  if (!built_ || !configured_)
    return;

  for (auto &channel : portState_)
    channel.didTransfer = false;

  if (currentCycle_ == 0) {
    appendKernelEvent(currentCycle_, SimPhase::Evaluate, 0,
                      EventKind::InvocationStart);
  }

  retireReadyMemoryRequests();

  std::vector<SimChannel> snapshot = portState_;
  bool converged = false;
  unsigned combIterationBudget = computeCombIterationBudget(staticModel_);
  for (unsigned iter = 0; iter < combIterationBudget; ++iter) {
    rebuildPortSignalsFromSnapshot(snapshot);
    for (auto &module : modules_)
      module->evaluate();
    finalizePortSignals();

    bool stable = true;
    for (size_t idx = 0; idx < portState_.size(); ++idx) {
      if (!channelEquals(snapshot[idx], portState_[idx])) {
        stable = false;
        break;
      }
    }
    if (stable) {
      converged = true;
      break;
    }
    snapshot = portState_;
  }

  if (!converged) {
    if (simDebugEnabled()) {
      std::cerr << "CycleKernel non-convergence cycle=" << currentCycle_
                << " iter_budget=" << combIterationBudget << "\n";
      dumpBudgetHitDebug(staticModel_, modules_, portState_, edgeState_,
                         outstandingMemoryRequests_.size(),
                         completionObligationsSatisfied(), quiescent_, done_,
                         true, currentCycle_);
    }
    appendKernelEvent(currentCycle_, SimPhase::Commit, 0,
                      EventKind::DeviceError);
    quiescent_ = false;
    done_ = false;
    deadlocked_ = true;
    lastBoundaryReason_ = BoundaryReason::Deadlock;
    ++currentCycle_;
    return;
  }

  if (simDebugEnabled()) {
    for (const auto &port : staticModel_.getPorts()) {
      if (port.direction != StaticPortDirection::Output)
        continue;
      if (port.portId == INVALID_ID ||
          port.portId >= static_cast<IdIndex>(outputChannelIndices_.size())) {
        continue;
      }
      const auto &channels = outputChannelIndices_[port.portId];
      if (channels.size() != 1)
        continue;
      const StaticChannelDesc &edge = staticModel_.getChannels()[channels.front()];
      IdIndex dstPort = edge.dstPort;
      if (dstPort == INVALID_ID ||
          dstPort >= static_cast<IdIndex>(portState_.size())) {
        continue;
      }
      if (portState_[port.portId].ready != portState_[dstPort].ready) {
        std::cerr << "CycleKernel single-dst ready mismatch srcPort="
                  << port.portId << " dstPort=" << dstPort
                  << " srcReady=" << portState_[port.portId].ready
                  << " dstReady=" << portState_[dstPort].ready << "\n";
      }
    }
  }

  if (simDebugEnabled() &&
      (currentCycle_ < 2 || (currentCycle_ >= 20 && currentCycle_ < 27) ||
       (currentCycle_ >= 32 && currentCycle_ < 41) ||
       (currentCycle_ >= 130 && currentCycle_ < 151) ||
       (currentCycle_ >= 224 && currentCycle_ < 258))) {
    std::cerr << "CycleKernel debug cycle=" << currentCycle_ << "\n";
    for (size_t moduleIdx = 0; moduleIdx < modules_.size(); ++moduleIdx) {
      const auto &module = modules_[moduleIdx];
      bool interesting = false;
      for (const SimChannel *ch : module->inputs)
        interesting = interesting || ch->valid || ch->ready;
      for (const SimChannel *ch : module->outputs)
        interesting = interesting || ch->valid || ch->ready;
      interesting = interesting ||
                    debugInterestingNodes_.find(module->hwNodeId) !=
                        debugInterestingNodes_.end();
      if (!interesting)
        continue;
      std::cerr << "  module[" << moduleIdx << "] hw=" << module->hwNodeId
                << " " << module->name << " kind="
                << static_cast<unsigned>(module->kind) << "\n";
      module->debugDump(std::cerr);
      const StaticModuleDesc *moduleDesc =
          staticModel_.findModule(static_cast<IdIndex>(module->hwNodeId));
      for (size_t i = 0; i < module->inputs.size(); ++i) {
        const SimChannel *ch = module->inputs[i];
        std::cerr << "    in" << i << " v=" << ch->valid << " r=" << ch->ready
                  << " d=" << ch->data << " t=" << ch->tag
                  << " ht=" << ch->hasTag << " g=" << ch->generation << "\n";
        if (moduleDesc && i < moduleDesc->inputPorts.size()) {
          IdIndex portId = moduleDesc->inputPorts[i];
          std::cerr << "      portId=" << portId;
          if (portId != INVALID_ID &&
              portId < static_cast<IdIndex>(inputSourcePort_.size())) {
            for (size_t srcIdx = 0; srcIdx < inputSourcePort_[portId].size();
                 ++srcIdx) {
              IdIndex srcPort = inputSourcePort_[portId][srcIdx];
              std::cerr << (srcIdx == 0 ? " srcPort=" : ",srcPort=") << srcPort;
              const StaticPortDesc *srcPortDesc = staticModel_.findPort(srcPort);
              const StaticModuleDesc *srcModule =
                  srcPortDesc
                      ? staticModel_.findModule(srcPortDesc->parentNodeId)
                      : nullptr;
              if (srcModule) {
                std::cerr << " srcNode=" << srcModule->hwNodeId << ":"
                          << srcModule->name << ":"
                          << moduleKindName(srcModule->kind);
              }
              if (portId < static_cast<IdIndex>(inputChannelIndex_.size()) &&
                  srcIdx < inputChannelIndex_[portId].size()) {
                std::cerr << " edge=" << inputChannelIndex_[portId][srcIdx];
              }
            }
          }
          std::cerr << "\n";
        }
      }
      for (size_t i = 0; i < module->outputs.size(); ++i) {
        const SimChannel *ch = module->outputs[i];
        std::cerr << "    out" << i << " v=" << ch->valid << " r=" << ch->ready
                  << " d=" << ch->data << " t=" << ch->tag
                  << " ht=" << ch->hasTag << " g=" << ch->generation << "\n";
        if (moduleDesc && i < moduleDesc->outputPorts.size()) {
          IdIndex portId = moduleDesc->outputPorts[i];
          std::cerr << "      portId=" << portId;
          if (portId != INVALID_ID &&
              portId < static_cast<IdIndex>(outputChannelIndices_.size())) {
            std::cerr << " dsts=[";
            bool first = true;
            for (unsigned edgeIdx : outputChannelIndices_[portId]) {
              IdIndex dstPort = staticModel_.getChannels()[edgeIdx].dstPort;
              if (!first)
                std::cerr << ", ";
              first = false;
              std::cerr << dstPort;
              const StaticPortDesc *dstPortDesc = staticModel_.findPort(dstPort);
              const StaticModuleDesc *dstModule =
                  dstPortDesc
                      ? staticModel_.findModule(dstPortDesc->parentNodeId)
                      : nullptr;
              if (dstModule) {
                std::cerr << "->" << dstModule->hwNodeId << ":"
                          << dstModule->name << ":"
                          << moduleKindName(dstModule->kind);
              }
              std::cerr << "(edge=" << edgeIdx << ")";
            }
            std::cerr << "]";
          }
          std::cerr << "\n";
        }
      }
    }
  }

  lastTransferCount_ = 0;
  for (const auto &port : staticModel_.getPorts()) {
    if (port.direction != StaticPortDirection::Output)
      continue;
    if (!portState_[port.portId].valid || portState_[port.portId].generation == 0)
      continue;
    syncOutputFanoutState(port.portId, portState_[port.portId].generation);
    bool allCaptured = true;
    if (port.portId != INVALID_ID &&
        port.portId < static_cast<IdIndex>(outputChannelIndices_.size())) {
      const auto &edgeIndices = outputChannelIndices_[port.portId];
      OutputFanoutState &fanout = outputFanoutState_[port.portId];
      for (size_t localIdx = 0; localIdx < edgeIndices.size(); ++localIdx) {
        unsigned edgeIdx = edgeIndices[localIdx];
        if (edgeIdx >= staticModel_.getChannels().size() ||
            localIdx >= fanout.captured.size()) {
          allCaptured = false;
          continue;
        }
        if (fanout.captured[localIdx] != 0)
          continue;
        if (!edgeCanAcceptNow(edgeIdx, portState_)) {
          allCaptured = false;
          continue;
        }
        const StaticChannelDesc &edge = staticModel_.getChannels()[edgeIdx];
        if (edge.dstPort != INVALID_ID &&
            edge.dstPort < static_cast<IdIndex>(portState_.size())) {
          portState_[edge.dstPort].didTransfer = true;
        }
        fanout.captured[localIdx] = 1;
        ++lastTransferCount_;
      }
      for (uint8_t captured : fanout.captured)
        allCaptured = allCaptured && (captured != 0);
    }
    if (allCaptured && !outputFanoutState_[port.portId].completionEmitted) {
      portState_[port.portId].didTransfer = true;
      if (port.portId != INVALID_ID &&
          port.portId < static_cast<IdIndex>(outputFanoutState_.size())) {
        outputFanoutState_[port.portId].completionEmitted = true;
      }
    }
  }
  rebuildVisibleEdgeSignals();

  size_t traceCountBeforeModules = traceDocument_.events.size();
  for (auto &module : modules_)
    module->commit();
  for (auto &module : modules_)
    module->collectTraceEvents(traceDocument_.events, currentCycle_);
  lastActivityCount_ =
      static_cast<uint64_t>(traceDocument_.events.size() - traceCountBeforeModules);

  evaluateBoundaryState();

  if (done_) {
    appendKernelEvent(currentCycle_, SimPhase::Commit, 0,
                      EventKind::InvocationDone);
  } else if (deadlocked_) {
    dumpBudgetHitDebug(staticModel_, modules_, portState_, edgeState_,
                       outstandingMemoryRequests_.size(),
                       completionObligationsSatisfied(), quiescent_, done_,
                       deadlocked_, currentCycle_);
    appendKernelEvent(currentCycle_, SimPhase::Commit, 0,
                      EventKind::DeviceError);
  }

  if (lastTransferCount_ != 0 || lastActivityCount_ != 0)
    ++activeCycles_;
  else
    ++idleCycles_;
  if (lastActivityCount_ != 0 || lastTransferCount_ != 0)
    ++fabricActiveCycles_;
  if (lastBoundaryReason_ == BoundaryReason::NeedMemIssue)
    ++needMemIssueCycles_;
  else if (lastBoundaryReason_ == BoundaryReason::WaitMemResp)
    ++waitMemRespCycles_;
  else if (lastBoundaryReason_ == BoundaryReason::Deadlock)
    ++deadlockBoundaryCount_;

  ++currentCycle_;
}

BoundaryReason CycleKernel::runUntilBoundary(uint64_t maxCycles) {
  if (!built_ || !configured_) {
    lastBoundaryReason_ = BoundaryReason::Deadlock;
    return lastBoundaryReason_;
  }
  for (uint64_t iter = 0; iter < maxCycles; ++iter) {
    stepCycle();
    if (lastBoundaryReason_ != BoundaryReason::None)
      return lastBoundaryReason_;
  }
  dumpBudgetHitDebug(staticModel_, modules_, portState_, edgeState_,
                     outstandingMemoryRequests_.size(),
                     completionObligationsSatisfied(), quiescent_, done_,
                     deadlocked_, currentCycle_);
  ++budgetBoundaryCount_;
  lastBoundaryReason_ = BoundaryReason::BudgetHit;
  return lastBoundaryReason_;
}

} // namespace sim
} // namespace loom
