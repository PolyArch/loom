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
  for (size_t nodeIdx = 0; nodeIdx < adg.nodes.size(); ++nodeIdx) {
    auto *node = adg.getNode(static_cast<IdIndex>(nodeIdx));
    if (!node)
      continue;

    // Extract node attributes.
    std::string nodeName, nodeOpName;
    std::vector<std::pair<std::string, int64_t>> intAttrs;
    std::vector<std::pair<std::string, std::string>> strAttrs;

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
      }
    }

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
    auto mod = createSimModule(
        static_cast<uint32_t>(nodeIdx), nodeName, nodeOpName,
        static_cast<unsigned>(node->inputPorts.size()),
        static_cast<unsigned>(node->outputPorts.size()),
        intAttrs, strAttrs);

    if (!mod) {
      llvm::errs() << "SimEngine: unsupported module type '" << nodeOpName
                    << "' at node " << nodeIdx << " (" << nodeName << ")\n";
      continue;
    }

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

  // If topological sort didn't include all modules, there's a cycle.
  // Add remaining modules (they'll be handled by fixed-point iteration).
  if (combOrder_.size() < n) {
    std::unordered_set<SimModule *> sorted(combOrder_.begin(),
                                           combOrder_.end());
    for (auto *mod : combModules) {
      if (sorted.find(mod) == sorted.end())
        combOrder_.push_back(mod);
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

  // For now, apply all config words to all modules.
  // TODO: use address map from configured fabric.mlir to route config words
  // to specific modules based on their config_mem addresses.
  for (auto &mod : modules_) {
    mod->reset();
    mod->configure(allWords);
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
  for (auto &ch : channels_) {
    ch->valid = false;
    ch->ready = false;
    ch->data = 0;
    ch->tag = 0;
  }
  for (auto &q : inputQueues_)
    q.pos = 0;
  for (auto &c : outputCollectors_) {
    c.data.clear();
    c.tags.clear();
  }
  cycleEvents_.clear();
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

  // Configuration overhead cycles.
  uint64_t configWords = configBlob_.size() / 4;
  uint64_t configCycles = 0;
  if (config_.configWordsPerCycle > 0)
    configCycles = (configWords + config_.configWordsPerCycle - 1) /
                   config_.configWordsPerCycle;
  configCycles += config_.resetOverheadCycles;
  result.configCycles = configCycles;
  currentCycle_ = configCycles;

  // Emit config write events.
  if (config_.traceMode != TraceMode::Off) {
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

  // Emit invocation start.
  if (config_.traceMode == TraceMode::Full) {
    TraceEvent ev;
    ev.cycle = currentCycle_;
    ev.epochId = epochId_;
    ev.invocationId = invocationId_;
    ev.eventKind = EV_INVOCATION_START;
    allTraceEvents_.push_back(ev);
  }

  // Main simulation loop.
  while (currentCycle_ < config_.maxCycles + configCycles) {
    stepOneCycle();
    currentCycle_++;

    if (isComplete())
      break;
  }

  // Emit invocation done.
  if (config_.traceMode == TraceMode::Full) {
    TraceEvent ev;
    ev.cycle = currentCycle_;
    ev.epochId = epochId_;
    ev.invocationId = invocationId_;
    ev.eventKind = EV_INVOCATION_DONE;
    allTraceEvents_.push_back(ev);
  }

  result.success = isComplete();
  result.totalCycles = currentCycle_;
  result.traceEvents = allTraceEvents_;

  // Collect per-node performance.
  result.nodePerf.resize(modules_.size());
  for (size_t i = 0; i < modules_.size(); ++i)
    result.nodePerf[i] = modules_[i]->getPerfSnapshot();

  if (!result.success)
    result.errorMessage = "Simulation did not complete within cycle limit";

  return result;
}

void SimEngine::stepOneCycle() {
  feedBoundaryInputs();

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

  drainBoundaryOutputs();
  emitTraceEvents();

  // Phase 2: Sequential state advance.
  for (auto *mod : seqModules_)
    mod->advanceClock();
}

void SimEngine::feedBoundaryInputs() {
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
      }
      // If handshake occurred, advance queue.
      if (ch->transferred())
        q.pos++;
    } else {
      ch->valid = false;
    }
  }
}

void SimEngine::drainBoundaryOutputs() {
  for (unsigned i = 0;
       i < boundaryOutputs_.size() && i < outputCollectors_.size(); ++i) {
    auto *ch = boundaryOutputs_[i];
    ch->ready = true; // Always ready to accept output.

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

  // And all modules have no pending tokens (check if any output is still valid).
  for (auto &mod : modules_) {
    for (auto *out : mod->outputs) {
      if (out->valid)
        return false;
    }
  }

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

  if (config_.traceMode == TraceMode::Full) {
    allTraceEvents_.insert(allTraceEvents_.end(), cycleEvents_.begin(),
                           cycleEvents_.end());
  }
}

} // namespace sim
} // namespace loom
