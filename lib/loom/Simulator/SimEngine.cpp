//===-- SimEngine.cpp - Two-phase simulation engine ----------------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include "loom/Simulator/SimEngine.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cassert>
#include <fstream>
#include <queue>
#include <unordered_map>
#include <unordered_set>

namespace loom {
namespace sim {

SimEngine::SimEngine(const SimConfig &config) : config_(config) {}

bool SimEngine::buildFromGraph(const Graph &adg) {
  modules_.clear();
  channels_.clear();
  combOrder_.clear();
  seqModules_.clear();
  boundaryInputs_.clear();
  boundaryOutputs_.clear();

  // Maps from ADG port IDs to channels.
  std::unordered_map<uint32_t, SimChannel *> portToChannel;

  // First pass: create modules for each non-null node.
  // Track CONFIG_WIDTH per module for address allocation.
  moduleConfigMap_.clear();
  uint32_t currentWordOffset = 0;

  for (size_t nodeIdx = 0; nodeIdx < adg.nodes.size(); ++nodeIdx) {
    auto *node = adg.getNode(static_cast<IdIndex>(nodeIdx));
    if (!node)
      continue;

    // Extract node attributes.
    std::string nodeName, nodeOpName;
    std::vector<std::pair<std::string, int64_t>> intAttrs;
    std::vector<std::pair<std::string, std::string>> strAttrs;
    std::vector<std::pair<std::string, std::vector<int8_t>>> arrayAttrs;

    for (auto &attr : node->attributes) {
      std::string attrName = attr.getName().str();
      if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(attr.getValue()))
        intAttrs.emplace_back(attrName, intAttr.getInt());
      else if (auto strAttr = mlir::dyn_cast<mlir::StringAttr>(attr.getValue())) {
        strAttrs.emplace_back(attrName, strAttr.getValue().str());
        if (attrName == "sym_name")
          nodeName = strAttr.getValue().str();
        else if (attrName == "op_name")
          nodeOpName = strAttr.getValue().str();
      } else if (auto arrAttr =
                     mlir::dyn_cast<mlir::DenseI8ArrayAttr>(attr.getValue())) {
        std::vector<int8_t> vals(arrAttr.asArrayRef().begin(),
                                 arrAttr.asArrayRef().end());
        arrayAttrs.emplace_back(attrName, std::move(vals));
      } else if (auto strArrAttr =
                     mlir::dyn_cast<mlir::ArrayAttr>(attr.getValue())) {
        // Extract body_ops (array of StringAttr) → synthesize body_op string.
        if (attrName == "body_ops" && !strArrAttr.empty()) {
          if (auto first =
                  mlir::dyn_cast<mlir::StringAttr>(strArrAttr[0]))
            strAttrs.emplace_back("body_op", first.getValue().str());
        }
      }
    }

    // Use the same fallback naming as ConfigGen: node_<id> when sym_name
    // is absent, so external config slices match by name.
    if (nodeName.empty())
      nodeName = "node_" + std::to_string(nodeIdx);

    if (node->kind == Node::ModuleInputNode) {
      // Boundary input: create channels for each output port.
      for (auto portId : node->outputPorts) {
        auto ch = std::make_unique<SimChannel>();
        portToChannel[portId] = ch.get();
        boundaryInputs_.push_back(ch.get());
        channels_.push_back(std::move(ch));
      }
      continue;
    }

    if (node->kind == Node::ModuleOutputNode) {
      // Boundary output: channels will be connected later.
      continue;
    }

    // Create simulation module.
    unsigned nIn = static_cast<unsigned>(node->inputPorts.size());
    unsigned nOut = static_cast<unsigned>(node->outputPorts.size());
    auto mod = createSimModule(
        static_cast<uint32_t>(nodeIdx), nodeName, nodeOpName,
        nIn, nOut, intAttrs, strAttrs, arrayAttrs);

    if (!mod) {
      llvm::errs() << "SimEngine: unsupported module type '" << nodeOpName
                    << "' at node " << nodeIdx << " (" << nodeName << ")\n";
      continue;
    }

    // Compute CONFIG_WIDTH and allocate config_mem words for this module.
    unsigned configBits =
        computeConfigWidth(nodeOpName, nIn, nOut, intAttrs, strAttrs,
                           arrayAttrs);
    uint32_t wordCount = (configBits + 31) / 32;
    ModuleConfigSlice slice;
    slice.wordOffset = currentWordOffset;
    slice.wordCount = wordCount;
    moduleConfigMap_.push_back(slice);
    currentWordOffset += wordCount;

    // Create output channels for this module.
    for (unsigned o = 0; o < node->outputPorts.size(); ++o) {
      auto ch = std::make_unique<SimChannel>();
      portToChannel[node->outputPorts[o]] = ch.get();
      mod->outputs.push_back(ch.get());
      channels_.push_back(std::move(ch));
    }

    modules_.push_back(std::move(mod));
  }

  // Second pass: wire input channels via edges.
  for (size_t edgeIdx = 0; edgeIdx < adg.edges.size(); ++edgeIdx) {
    auto *edge = adg.getEdge(static_cast<IdIndex>(edgeIdx));
    if (!edge)
      continue;

    auto srcIt = portToChannel.find(edge->srcPort);
    if (srcIt == portToChannel.end())
      continue;

    SimChannel *ch = srcIt->second;

    // Find the destination port's parent node and connect.
    auto *dstPort = adg.getPort(edge->dstPort);
    if (!dstPort)
      continue;

    auto *dstNode = adg.getNode(dstPort->parentNode);
    if (!dstNode)
      continue;

    if (dstNode->kind == Node::ModuleOutputNode) {
      // Boundary output.
      boundaryOutputs_.push_back(ch);
      continue;
    }

    // Find the module that owns this destination node.
    for (auto &mod : modules_) {
      if (mod->hwNodeId == dstPort->parentNode) {
        // Find which input port index this is.
        for (unsigned i = 0; i < dstNode->inputPorts.size(); ++i) {
          if (dstNode->inputPorts[i] == edge->dstPort) {
            // Ensure input vector is large enough.
            if (mod->inputs.size() <= i)
              mod->inputs.resize(i + 1, nullptr);
            mod->inputs[i] = ch;
            break;
          }
        }
        break;
      }
    }
  }

  // Fill any missing input channels with dummy channels.
  for (auto &mod : modules_) {
    for (auto &inp : mod->inputs) {
      if (!inp) {
        auto ch = std::make_unique<SimChannel>();
        inp = ch.get();
        channels_.push_back(std::move(ch));
      }
    }
  }

  // Initialize input/output queues.
  inputQueues_.resize(boundaryInputs_.size());
  outputCollectors_.resize(boundaryOutputs_.size());

  // Compute topological order.
  computeTopologicalOrder();

  return true;
}

void SimEngine::computeTopologicalOrder() {
  combOrder_.clear();
  seqModules_.clear();

  // Separate combinational and sequential modules.
  std::vector<SimModule *> combModules;
  for (auto &mod : modules_) {
    if (mod->isCombinational())
      combModules.push_back(mod.get());
    else
      seqModules_.push_back(mod.get());
  }

  // Build adjacency based on channel connectivity.
  // Module A -> Module B if any output channel of A is an input channel of B.
  std::unordered_map<SimModule *, size_t> modIndex;
  for (size_t i = 0; i < combModules.size(); ++i)
    modIndex[combModules[i]] = i;

  // Map channels to their producing module.
  std::unordered_map<SimChannel *, SimModule *> channelProducer;
  for (auto *mod : combModules) {
    for (auto *ch : mod->outputs)
      channelProducer[ch] = mod;
  }

  size_t n = combModules.size();
  std::vector<std::vector<size_t>> adj(n);
  std::vector<size_t> inDegree(n, 0);

  for (size_t i = 0; i < n; ++i) {
    for (auto *inCh : combModules[i]->inputs) {
      auto it = channelProducer.find(inCh);
      if (it != channelProducer.end()) {
        auto srcIt = modIndex.find(it->second);
        if (srcIt != modIndex.end() && srcIt->second != i) {
          adj[srcIt->second].push_back(i);
          inDegree[i]++;
        }
      }
    }
  }

  // Kahn's algorithm for topological sort.
  std::queue<size_t> q;
  for (size_t i = 0; i < n; ++i) {
    if (inDegree[i] == 0)
      q.push(i);
  }

  while (!q.empty()) {
    size_t cur = q.front();
    q.pop();
    combOrder_.push_back(combModules[cur]);
    for (size_t next : adj[cur]) {
      if (--inDegree[next] == 0)
        q.push(next);
    }
  }

  // If topological sort didn't include all modules, there's a combinational
  // SCC. Latch CFG_ADG_COMBINATIONAL_LOOP on each module in the SCC, and
  // append them for fixed-point iteration.
  hasCombLoop_ = (combOrder_.size() < n);
  if (hasCombLoop_) {
    std::unordered_set<SimModule *> sorted(combOrder_.begin(),
                                           combOrder_.end());
    for (auto *mod : combModules) {
      if (sorted.find(mod) == sorted.end()) {
        mod->latchError(RtError::CFG_ADG_COMBINATIONAL_LOOP);
        mod->commitError();
        combOrder_.push_back(mod);
      }
    }
  }
}

bool SimEngine::loadConfig(const std::string &configBinPath) {
  std::ifstream file(configBinPath, std::ios::binary);
  if (!file)
    return false;

  file.seekg(0, std::ios::end);
  auto size = file.tellg();
  file.seekg(0, std::ios::beg);

  configBlob_.resize(static_cast<size_t>(size));
  file.read(reinterpret_cast<char *>(configBlob_.data()),
            static_cast<std::streamsize>(size));

  return loadConfig(configBlob_);
}

bool SimEngine::loadConfig(const std::vector<uint8_t> &configBlob) {
  configBlob_ = configBlob;

  // Parse config blob as 32-bit words.
  size_t numWords = configBlob_.size() / 4;
  std::vector<uint32_t> allWords(numWords);
  for (size_t i = 0; i < numWords; ++i) {
    allWords[i] = static_cast<uint32_t>(configBlob_[i * 4]) |
                  (static_cast<uint32_t>(configBlob_[i * 4 + 1]) << 8) |
                  (static_cast<uint32_t>(configBlob_[i * 4 + 2]) << 16) |
                  (static_cast<uint32_t>(configBlob_[i * 4 + 3]) << 24);
  }

  // Apply per-module config slices based on address map.
  moduleConfigWrites_.assign(modules_.size(), 0);
  for (size_t m = 0; m < modules_.size(); ++m) {
    modules_[m]->reset();

    if (m < moduleConfigMap_.size()) {
      auto &slice = moduleConfigMap_[m];
      if (slice.wordCount > 0 && slice.wordOffset < numWords) {
        size_t end = std::min(static_cast<size_t>(slice.wordOffset + slice.wordCount),
                              numWords);
        std::vector<uint32_t> modWords(allWords.begin() + slice.wordOffset,
                                        allWords.begin() + end);
        modules_[m]->configure(modWords);
        moduleConfigWrites_[m] = modWords.size();
      }
    }
  }

  return true;
}

bool SimEngine::loadConfig(const std::vector<uint8_t> &configBlob,
                           const std::vector<ExternalConfigSlice> &slices) {
  // Reject non-word-aligned config blobs.
  if (configBlob.size() % 4 != 0) {
    llvm::errs() << "SimEngine: config blob size " << configBlob.size()
                  << " is not word-aligned (must be multiple of 4)\n";
    return false;
  }

  configBlob_ = configBlob;

  // Build name -> slice index map.
  std::unordered_map<std::string, const ExternalConfigSlice *> sliceByName;
  for (const auto &s : slices)
    sliceByName[s.name] = &s;

  // Parse config blob as 32-bit words.
  size_t numWords = configBlob_.size() / 4;
  std::vector<uint32_t> allWords(numWords);
  for (size_t i = 0; i < numWords; ++i) {
    allWords[i] = static_cast<uint32_t>(configBlob_[i * 4]) |
                  (static_cast<uint32_t>(configBlob_[i * 4 + 1]) << 8) |
                  (static_cast<uint32_t>(configBlob_[i * 4 + 2]) << 16) |
                  (static_cast<uint32_t>(configBlob_[i * 4 + 3]) << 24);
  }

  // Apply per-module config slices based on mapper-authored address metadata.
  moduleConfigWrites_.assign(modules_.size(), 0);
  for (size_t m = 0; m < modules_.size(); ++m) {
    auto &mod = modules_[m];
    mod->reset();

    auto it = sliceByName.find(mod->name);
    if (it == sliceByName.end())
      continue;

    const auto *slice = it->second;
    if (slice->wordCount == 0)
      continue;
    if (slice->wordOffset + slice->wordCount > numWords) {
      llvm::errs() << "SimEngine: config slice for '" << mod->name
                    << "' [offset=" << slice->wordOffset
                    << " count=" << slice->wordCount
                    << "] exceeds blob size " << numWords << " words\n";
      return false;
    }

    std::vector<uint32_t> modWords(allWords.begin() + slice->wordOffset,
                                    allWords.begin() + slice->wordOffset +
                                        slice->wordCount);
    mod->configure(modWords);
    moduleConfigWrites_[m] = modWords.size();
  }

  return true;
}

void SimEngine::setInput(unsigned portIdx, const std::vector<uint64_t> &data,
                         const std::vector<uint16_t> &tags) {
  if (portIdx >= inputQueues_.size())
    inputQueues_.resize(portIdx + 1);

  inputQueues_[portIdx].data = data;
  inputQueues_[portIdx].tags = tags;
  inputQueues_[portIdx].pos = 0;
  inputQueues_[portIdx].hasTag = !tags.empty();
}

std::vector<uint64_t> SimEngine::getOutput(unsigned portIdx) const {
  if (portIdx >= outputCollectors_.size())
    return {};
  return outputCollectors_[portIdx].data;
}

std::vector<uint16_t> SimEngine::getOutputTags(unsigned portIdx) const {
  if (portIdx >= outputCollectors_.size())
    return {};
  return outputCollectors_[portIdx].tags;
}

void SimEngine::resetExecution() {
  currentCycle_ = 0;
  invocationId_++;

  // Reset all module runtime state while preserving configuration.
  for (auto &mod : modules_)
    mod->reset();

  for (auto &ch : channels_) {
    ch->valid = false;
    ch->ready = false;
    ch->data = 0;
    ch->tag = 0;
    ch->hasTag = false;
  }
  for (auto &q : inputQueues_)
    q.pos = 0;
  for (auto &c : outputCollectors_) {
    c.data.clear();
    c.tags.clear();
  }
  cycleEvents_.clear();
  // Clear trace buffer between invocations for deterministic output.
  allTraceEvents_.clear();
}

void SimEngine::resetAll() {
  resetExecution();
  epochId_ = 0;
  invocationId_ = 0;
  configBlob_.clear();
  allTraceEvents_.clear();
  for (auto &mod : modules_)
    mod->reset();
}

SimResult SimEngine::run() {
  SimResult result;

  // Validate config rate.
  if (config_.configWordsPerCycle == 0) {
    result.success = false;
    result.errorMessage = "configWordsPerCycle must be > 0";
    return result;
  }

  // Combinational loops are flagged per-module via CFG_ADG_COMBINATIONAL_LOOP
  // during computeTopologicalOrder(). Fixed-point iteration handles convergence.

  // Configuration overhead cycles.
  uint64_t configWords = configBlob_.size() / 4;
  uint64_t configCycles =
      (configWords + config_.configWordsPerCycle - 1) /
      config_.configWordsPerCycle;
  configCycles += config_.resetOverheadCycles;
  result.configCycles = configCycles;
  currentCycle_ = configCycles;

  // Helpers to check whether a direct-emission event passes the active
  // trace filters (kind, node, core).  System-level events use hwNodeId=0
  // and coreId=0, so they are excluded when the whitelist is non-empty and
  // does not contain 0.
  auto kindAllowed = [&](EventKind ek) {
    if (config_.traceFilterKinds.empty()) return true;
    for (auto k : config_.traceFilterKinds)
      if (k == ek) return true;
    return false;
  };
  auto nodeAllowed = [&](uint32_t nid) {
    if (config_.traceFilterNodes.empty()) return true;
    for (auto n : config_.traceFilterNodes)
      if (n == nid) return true;
    return false;
  };
  auto coreAllowed = [&](uint16_t cid) {
    if (config_.traceFilterCores.empty()) return true;
    for (auto c : config_.traceFilterCores)
      if (c == cid) return true;
    return false;
  };

  // Emit config write events and count total config writes.
  result.totalConfigWrites = configWords;
  if (config_.traceMode != TraceMode::Off) {
    if (kindAllowed(EV_CONFIG_WRITE) && nodeAllowed(0) && coreAllowed(0)) {
      for (uint64_t w = 0; w < configWords; ++w) {
        uint64_t writeCycle = w / config_.configWordsPerCycle;
        TraceEvent ev;
        ev.cycle = writeCycle;
        ev.epochId = epochId_;
        ev.invocationId = invocationId_;
        ev.eventKind = EV_CONFIG_WRITE;
        ev.arg0 = static_cast<uint32_t>(w);
        allTraceEvents_.push_back(ev);
      }
    }
  }

  // Emit invocation start (subject to kind, node, and core filters).
  if (config_.traceMode == TraceMode::Full) {
    if (kindAllowed(EV_INVOCATION_START) && nodeAllowed(0) && coreAllowed(0)) {
      TraceEvent ev;
      ev.cycle = currentCycle_;
      ev.epochId = epochId_;
      ev.invocationId = invocationId_;
      ev.eventKind = EV_INVOCATION_START;
      allTraceEvents_.push_back(ev);
    }
  }

  // Main simulation loop.
  uint64_t maxCycle = (config_.maxCycles > 0)
                        ? configCycles + config_.maxCycles
                        : UINT64_MAX;
  while (currentCycle_ < maxCycle) {
    stepOneCycle();
    currentCycle_++;

    if (isComplete())
      break;
  }

  // Emit invocation done (subject to kind, node, and core filters).
  if (config_.traceMode == TraceMode::Full) {
    if (kindAllowed(EV_INVOCATION_DONE) && nodeAllowed(0) && coreAllowed(0)) {
      TraceEvent ev;
      ev.cycle = currentCycle_;
      ev.epochId = epochId_;
      ev.invocationId = invocationId_;
      ev.eventKind = EV_INVOCATION_DONE;
      allTraceEvents_.push_back(ev);
    }
  }

  result.success = isComplete();
  result.totalCycles = currentCycle_;
  result.traceEvents = allTraceEvents_;

  // Collect per-node performance and inject config write counts.
  result.nodePerf.resize(modules_.size());
  for (size_t i = 0; i < modules_.size(); ++i) {
    result.nodePerf[i] = modules_[i]->getPerfSnapshot();
    result.nodePerf[i].nodeIndex = modules_[i]->hwNodeId;
    if (i < moduleConfigWrites_.size())
      result.nodePerf[i].configWrites = moduleConfigWrites_[i];
  }

  if (!result.success)
    result.errorMessage = "Simulation did not complete within cycle limit";

  return result;
}

void SimEngine::stepOneCycle() {
  // Pre-phase: Drive boundary signals BEFORE combinational evaluation.
  // This ensures producers see boundary ready signals during phase 1.
  driveBoundaryInputs();
  driveBoundaryOutputReady();

  // Phase 1: Combinational convergence.
  // Iterate until all channel states are stable.
  constexpr unsigned kMaxCombIterations = 100;
  for (unsigned iter = 0; iter < kMaxCombIterations; ++iter) {
    // Save channel state for convergence check.
    std::vector<SimChannel> savedState;
    savedState.reserve(channels_.size());
    for (auto &ch : channels_)
      savedState.push_back(*ch);

    // Evaluate all combinational modules in topological order.
    for (auto *mod : combOrder_)
      mod->evaluateCombinational();

    // Also evaluate combinational logic of sequential modules.
    for (auto *mod : seqModules_)
      mod->evaluateCombinational();

    // Check convergence.
    bool stable = true;
    for (size_t i = 0; i < channels_.size(); ++i) {
      if (*channels_[i] != savedState[i]) {
        stable = false;
        break;
      }
    }
    if (stable)
      break;
  }

  // Commit pending errors (same-cycle min-code precedence).
  for (auto &mod : modules_)
    mod->commitError();

  emitTraceEvents();

  // Phase 2: Sequential state advance + boundary bookkeeping.
  // Advance boundary queues/collectors based on handshake results.
  advanceBoundaryState();

  for (auto *mod : seqModules_)
    mod->advanceClock();
}

void SimEngine::driveBoundaryInputs() {
  // Drive valid/data/tag on boundary input channels from queues.
  // Do NOT advance queue position here - that happens in advanceBoundaryState.
  for (unsigned i = 0; i < boundaryInputs_.size() && i < inputQueues_.size();
       ++i) {
    auto &q = inputQueues_[i];
    auto *ch = boundaryInputs_[i];

    if (q.pos < q.data.size()) {
      ch->valid = true;
      ch->data = q.data[q.pos];
      if (q.hasTag && q.pos < q.tags.size()) {
        ch->tag = q.tags[q.pos];
        ch->hasTag = true;
      } else {
        ch->hasTag = false;
      }
    } else {
      ch->valid = false;
    }
  }
}

void SimEngine::driveBoundaryOutputReady() {
  // Set boundary output channels as ready BEFORE combinational evaluation
  // so that producers can see the ready signal during phase 1.
  for (unsigned i = 0;
       i < boundaryOutputs_.size() && i < outputCollectors_.size(); ++i) {
    boundaryOutputs_[i]->ready = true;
  }
}

void SimEngine::advanceBoundaryState() {
  // After combinational evaluation, check which handshakes completed
  // and advance input queues / collect outputs accordingly.

  // Advance input queues where handshake occurred.
  for (unsigned i = 0; i < boundaryInputs_.size() && i < inputQueues_.size();
       ++i) {
    if (boundaryInputs_[i]->transferred())
      inputQueues_[i].pos++;
  }

  // Collect outputs where handshake occurred.
  for (unsigned i = 0;
       i < boundaryOutputs_.size() && i < outputCollectors_.size(); ++i) {
    auto *ch = boundaryOutputs_[i];
    if (ch->transferred()) {
      outputCollectors_[i].data.push_back(ch->data);
      if (ch->hasTag)
        outputCollectors_[i].tags.push_back(ch->tag);
    }
  }
}

bool SimEngine::isComplete() const {
  // Complete when all input queues are drained.
  for (auto &q : inputQueues_) {
    if (q.pos < q.data.size())
      return false;
  }

  // Check if any module still has pending tokens (output valid).
  for (auto &mod : modules_) {
    for (auto *out : mod->outputs) {
      if (out->valid)
        return false;
    }
  }

  // Require at least one cycle of actual execution to avoid immediate
  // termination when no boundary inputs were provided. Most Loom apps
  // receive data through extmemory, not boundary input ports.
  uint64_t execStart = 0;
  if (config_.configWordsPerCycle > 0) {
    uint64_t configWords = configBlob_.size() / 4;
    execStart = (configWords + config_.configWordsPerCycle - 1) /
                    config_.configWordsPerCycle +
                config_.resetOverheadCycles;
  }
  if (currentCycle_ <= execStart)
    return false;

  return true;
}

void SimEngine::emitTraceEvents() {
  if (config_.traceMode == TraceMode::Off)
    return;

  cycleEvents_.clear();
  for (auto &mod : modules_)
    mod->collectTraceEvents(cycleEvents_, currentCycle_);

  // Tag events with session info.
  for (auto &ev : cycleEvents_) {
    ev.epochId = epochId_;
    ev.invocationId = invocationId_;
  }

  // Apply trace filters: remove events not matching any active whitelist.
  if (!config_.traceFilterKinds.empty() || !config_.traceFilterNodes.empty() ||
      !config_.traceFilterCores.empty()) {
    cycleEvents_.erase(
        std::remove_if(cycleEvents_.begin(), cycleEvents_.end(),
                       [this](const TraceEvent &ev) {
                         if (!config_.traceFilterKinds.empty()) {
                           bool kindMatch = false;
                           for (auto k : config_.traceFilterKinds)
                             if (ev.eventKind == k) { kindMatch = true; break; }
                           if (!kindMatch) return true;
                         }
                         if (!config_.traceFilterNodes.empty()) {
                           bool nodeMatch = false;
                           for (auto n : config_.traceFilterNodes)
                             if (ev.hwNodeId == n) { nodeMatch = true; break; }
                           if (!nodeMatch) return true;
                         }
                         if (!config_.traceFilterCores.empty()) {
                           bool coreMatch = false;
                           for (auto c : config_.traceFilterCores)
                             if (ev.coreId == c) { coreMatch = true; break; }
                           if (!coreMatch) return true;
                         }
                         return false;
                       }),
        cycleEvents_.end());
  }

  if (config_.traceMode == TraceMode::Full) {
    allTraceEvents_.insert(allTraceEvents_.end(), cycleEvents_.begin(),
                           cycleEvents_.end());
  }
}

} // namespace sim
} // namespace loom
