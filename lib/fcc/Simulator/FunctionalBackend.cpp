#include "FunctionalBackendImpl.h"

#include <memory>
#include <utility>

namespace fcc {
namespace sim {

// ===----------------------------------------------------------------------===
// Impl -- public interface methods
// ===----------------------------------------------------------------------===

std::string FunctionalSimulationBackend::Impl::connect() { return {}; }

std::string FunctionalSimulationBackend::Impl::buildFromMappedState(
    const Graph &dfgGraph, const Graph &adgGraph, const MappingState &state) {
  dfg = dfgGraph.clone();
  adg = adgGraph.clone();
  mapping = state;
  built = true;
  configWords = 0;

  inputPortEdge.assign(dfg.ports.size(), INVALID_ID);
  outputPortEdges.assign(dfg.ports.size(), {});
  edgeQueues.assign(dfg.edges.size(), {});
  for (IdIndex edgeId = 0; edgeId < static_cast<IdIndex>(dfg.edges.size());
       ++edgeId) {
    const Edge *edge = dfg.getEdge(edgeId);
    if (!edge)
      continue;
    if (edge->dstPort >= 0 &&
        edge->dstPort < static_cast<IdIndex>(inputPortEdge.size()))
      inputPortEdge[edge->dstPort] = edgeId;
    if (edge->srcPort >= 0 &&
        edge->srcPort < static_cast<IdIndex>(outputPortEdges.size()))
      outputPortEdges[edge->srcPort].push_back(edgeId);
  }

  hwInputOrdinals =
      buildBoundaryOrdinals(adg, Node::ModuleInputNode, numInputPorts);
  hwOutputOrdinals =
      buildBoundaryOrdinals(adg, Node::ModuleOutputNode, numOutputPorts);

  swInputToHwOrdinal.assign(dfg.nodes.size(), kInvalidOrdinal);
  swOutputToHwOrdinal.assign(dfg.nodes.size(), kInvalidOrdinal);
  hwOrdinalToSwInputNode.assign(numInputPorts, INVALID_ID);
  hwOrdinalToSwOutputNode.assign(numOutputPorts, INVALID_ID);

  for (IdIndex swNodeId = 0;
       swNodeId < static_cast<IdIndex>(dfg.nodes.size()); ++swNodeId) {
    const Node *swNode = dfg.getNode(swNodeId);
    if (!swNode ||
        swNodeId >= static_cast<IdIndex>(mapping.swNodeToHwNode.size()))
      continue;
    IdIndex hwNodeId = mapping.swNodeToHwNode[swNodeId];
    if (hwNodeId == INVALID_ID ||
        hwNodeId >= static_cast<IdIndex>(adg.nodes.size()))
      continue;

    if (swNode->kind == Node::ModuleInputNode) {
      if (!swNode->outputPorts.empty()) {
        const Port *outPort = dfg.getPort(swNode->outputPorts.front());
        if (outPort && mlir::isa<mlir::MemRefType>(outPort->type))
          continue;
      }
      if (hwNodeId < static_cast<IdIndex>(hwInputOrdinals.size()) &&
          hwInputOrdinals[hwNodeId] != kInvalidOrdinal) {
        unsigned ordinal = hwInputOrdinals[hwNodeId];
        swInputToHwOrdinal[swNodeId] = ordinal;
        hwOrdinalToSwInputNode[ordinal] = swNodeId;
      }
    }

    if (swNode->kind == Node::ModuleOutputNode &&
        hwNodeId < static_cast<IdIndex>(hwOutputOrdinals.size()) &&
        hwOutputOrdinals[hwNodeId] != kInvalidOrdinal) {
      unsigned ordinal = hwOutputOrdinals[hwNodeId];
      swOutputToHwOrdinal[swNodeId] = ordinal;
      hwOrdinalToSwOutputNode[ordinal] = swNodeId;
    }
  }

  memoryRegions.clear();
  swMemoryNodeToRegion.clear();
  for (IdIndex hwNodeId = 0;
       hwNodeId < static_cast<IdIndex>(adg.nodes.size()); ++hwNodeId) {
    const Node *hwNode = adg.getNode(hwNodeId);
    if (!hwNode || getNodeAttrStr(hwNode, "resource_class") != "memory")
      continue;
    if (hwNodeId >= static_cast<IdIndex>(mapping.hwNodeToSwNodes.size()))
      continue;
    for (IdIndex swNodeId : mapping.hwNodeToSwNodes[hwNodeId]) {
      const Node *swNode = dfg.getNode(swNodeId);
      if (!swNode)
        continue;
      llvm::StringRef opName = getNodeAttrStr(swNode, "op_name");
      if (!opMatches(opName, "memory") && !opMatches(opName, "extmemory"))
        continue;
      unsigned regionId = static_cast<unsigned>(memoryRegions.size());
      swMemoryNodeToRegion[swNodeId] = regionId;
      MemoryRegionBinding region;
      region.elemSizeLog2 = getMemoryElemSizeLog2(swNode, hwNode, dfg, adg);
      region.hwNode = hwNodeId;
      region.swNode = swNodeId;
      memoryRegions.push_back(region);
    }
  }

  inputBindings.assign(numInputPorts, {});
  outputCollectors.assign(numOutputPorts, {});
  resetExecution();
  return {};
}

std::string FunctionalSimulationBackend::Impl::loadConfig(
    const std::vector<uint8_t> &configBlob) {
  if (!built)
    return "simulation backend has no mapped graph";
  configWords = (configBlob.size() + 3) / 4;
  return {};
}

std::string FunctionalSimulationBackend::Impl::setInput(
    unsigned portIdx, const std::vector<uint64_t> &data,
    const std::vector<uint16_t> &tags) {
  if (!built)
    return "simulation backend has no mapped graph";
  if (portIdx >= inputBindings.size()) {
    std::ostringstream oss;
    oss << "input port " << portIdx << " out of range";
    return oss.str();
  }
  if (hwOrdinalToSwInputNode[portIdx] == INVALID_ID) {
    std::ostringstream oss;
    oss << "input port " << portIdx << " is not bound to a software input";
    return oss.str();
  }
  if (!tags.empty() && tags.size() != data.size())
    return "input tags must match input data length";

  auto &binding = inputBindings[portIdx];
  binding.clear();
  binding.reserve(data.size());
  for (size_t i = 0; i < data.size(); ++i) {
    Token token;
    token.data = data[i];
    token.hasTag = !tags.empty();
    token.tag = tags.empty() ? 0 : tags[i];
    binding.push_back(token);
  }
  return {};
}

std::string FunctionalSimulationBackend::Impl::setExtMemoryBacking(
    unsigned regionId, uint8_t *data, size_t sizeBytes) {
  if (!built)
    return "simulation backend has no mapped graph";
  if (regionId >= memoryRegions.size()) {
    std::ostringstream oss;
    oss << "memory region " << regionId << " out of range";
    return oss.str();
  }
  memoryRegions[regionId].data = data;
  memoryRegions[regionId].sizeBytes = sizeBytes;
  return {};
}

unsigned FunctionalSimulationBackend::Impl::getNumInputPorts() const {
  return numInputPorts;
}

unsigned FunctionalSimulationBackend::Impl::getNumOutputPorts() const {
  return numOutputPorts;
}

std::vector<uint64_t>
FunctionalSimulationBackend::Impl::getOutput(unsigned portIdx) const {
  if (portIdx >= outputCollectors.size())
    return {};
  std::vector<uint64_t> data;
  data.reserve(outputCollectors[portIdx].size());
  for (const Token &token : outputCollectors[portIdx])
    data.push_back(token.data);
  return data;
}

std::vector<uint16_t>
FunctionalSimulationBackend::Impl::getOutputTags(unsigned portIdx) const {
  if (portIdx >= outputCollectors.size())
    return {};
  std::vector<uint16_t> tags;
  tags.reserve(outputCollectors[portIdx].size());
  for (const Token &token : outputCollectors[portIdx])
    tags.push_back(token.hasTag ? token.tag : 0);
  return tags;
}

void FunctionalSimulationBackend::Impl::resetExecution() {
  edgeQueues.assign(dfg.edges.size(), {});
  outputCollectors.assign(numOutputPorts, {});
  nodeRuntime.assign(dfg.nodes.size(), {});

  for (IdIndex nodeId = 0; nodeId < static_cast<IdIndex>(dfg.nodes.size());
       ++nodeId) {
    const Node *node = dfg.getNode(nodeId);
    if (!node)
      continue;
    nodeRuntime[nodeId].perf.nodeIndex = static_cast<uint32_t>(nodeId);
    if (opMatches(getNodeAttrStr(node, "op_name"), "stream")) {
      nodeRuntime[nodeId].stream.stepOp =
          getNodeAttrString(node, "step_op", "+=");
      nodeRuntime[nodeId].stream.contCond =
          getNodeAttrString(node, "cont_cond", "<");
    }
  }
}

void FunctionalSimulationBackend::Impl::resetAll() {
  built = false;
  configWords = 0;
  numInputPorts = 0;
  numOutputPorts = 0;
  dfg = Graph();
  adg = Graph();
  mapping = MappingState();
  hwInputOrdinals.clear();
  hwOutputOrdinals.clear();
  swInputToHwOrdinal.clear();
  swOutputToHwOrdinal.clear();
  hwOrdinalToSwInputNode.clear();
  hwOrdinalToSwOutputNode.clear();
  inputPortEdge.clear();
  outputPortEdges.clear();
  edgeQueues.clear();
  inputBindings.clear();
  outputCollectors.clear();
  nodeRuntime.clear();
  memoryRegions.clear();
  swMemoryNodeToRegion.clear();
}

SimResult FunctionalSimulationBackend::Impl::invoke(uint32_t epochId,
                                                    uint64_t invocationId) {
  SimResult result;
  result.configCycles = configWords * config.configWordsPerCycle;
  result.totalConfigWrites = configWords;
  result.traceDocument.version = 1;
  result.traceDocument.traceKind = "fcc_cycle_trace";
  result.traceDocument.producer = "fcc";
  result.traceDocument.epochId = epochId;
  result.traceDocument.invocationId = invocationId;
  result.traceDocument.coreId = config.coreId;

  if (!built) {
    result.errorMessage = "simulation backend has no mapped graph";
    result.termination = RunTermination::ContractError;
    return result;
  }

  resetExecution();
  seedInputs();

  if (config.traceMode != TraceMode::Off) {
    TraceEvent start;
    start.cycle = 0;
    start.epochId = epochId;
    start.invocationId = invocationId;
    start.coreId = config.coreId;
    start.eventKind = EventKind::InvocationStart;
    result.traceEvents.push_back(start);
  }

  uint64_t actions = 0;
  std::string errorMessage;
  while (actions < config.maxCycles) {
    bool madeProgress = false;
    for (IdIndex nodeId = 0; nodeId < static_cast<IdIndex>(dfg.nodes.size());
         ++nodeId) {
      Action action = tryExecuteNode(nodeId);
      if (!action.error.empty()) {
        errorMessage = action.error;
        break;
      }
      if (!action.progress)
        continue;
      madeProgress = true;
      ++actions;

      NodeRuntime &runtime = nodeRuntime[nodeId];
      runtime.perf.activeCycles++;
      runtime.perf.tokensIn += action.tokensIn;
      runtime.perf.tokensOut += action.tokensOut;

      if (config.traceMode != TraceMode::Off) {
        TraceEvent fire;
        fire.cycle = actions;
        fire.epochId = epochId;
        fire.invocationId = invocationId;
        fire.coreId = config.coreId;
        fire.hwNodeId =
            nodeId < static_cast<IdIndex>(mapping.swNodeToHwNode.size())
                ? static_cast<uint32_t>(mapping.swNodeToHwNode[nodeId])
                : static_cast<uint32_t>(nodeId);
        fire.eventKind = EventKind::NodeFire;
        fire.arg0 = action.tokensIn;
        fire.arg1 = action.tokensOut;
        result.traceEvents.push_back(fire);
      }
    }

    if (!errorMessage.empty())
      break;
    if (!madeProgress)
      break;
  }

  if (!errorMessage.empty()) {
    result.errorMessage = errorMessage;
    result.termination = RunTermination::DeviceError;
    if (config.traceMode != TraceMode::Off) {
      TraceEvent error;
      error.cycle = actions;
      error.epochId = epochId;
      error.invocationId = invocationId;
      error.coreId = config.coreId;
      error.eventKind = EventKind::DeviceError;
      result.traceEvents.push_back(error);
    }
  } else if (actions >= config.maxCycles) {
    result.errorMessage = "simulation exceeded maxCycles";
    result.termination = RunTermination::Timeout;
  } else {
    result.success = true;
    result.termination = RunTermination::Completed;
  }

  result.totalCycles =
      config.resetOverheadCycles + result.configCycles + actions;
  result.nodePerf.reserve(nodeRuntime.size());
  for (const NodeRuntime &runtime : nodeRuntime)
    result.nodePerf.push_back(runtime.perf);

  if (config.traceMode != TraceMode::Off) {
    TraceEvent done;
    done.cycle = actions;
    done.epochId = epochId;
    done.invocationId = invocationId;
    done.coreId = config.coreId;
    done.eventKind = EventKind::InvocationDone;
    done.arg0 = static_cast<uint32_t>(result.success ? 1 : 0);
    done.arg1 = static_cast<uint32_t>(result.termination);
    result.traceEvents.push_back(done);
  }

  result.traceDocument.events = result.traceEvents;

  return result;
}

// ===----------------------------------------------------------------------===
// Impl -- token management
// ===----------------------------------------------------------------------===

void FunctionalSimulationBackend::Impl::seedInputs() {
  for (unsigned ordinal = 0; ordinal < inputBindings.size(); ++ordinal) {
    IdIndex swNodeId = hwOrdinalToSwInputNode[ordinal];
    if (swNodeId == INVALID_ID)
      continue;
    const Node *node = dfg.getNode(swNodeId);
    if (!node || node->outputPorts.empty())
      continue;
    IdIndex portId = node->outputPorts.front();
    const Port *outPort = dfg.getPort(portId);
    if (!outPort)
      continue;
    for (const Token &token : inputBindings[ordinal]) {
      Token coerced = token;
      coerced.data = coerceValueToType(token.data, outPort->type);
      pushToken(portId, coerced);
    }
  }
}

bool FunctionalSimulationBackend::Impl::hasInputToken(IdIndex portId) const {
  if (portId == INVALID_ID ||
      portId >= static_cast<IdIndex>(inputPortEdge.size()))
    return false;
  IdIndex edgeId = inputPortEdge[portId];
  return edgeId != INVALID_ID &&
         edgeId < static_cast<IdIndex>(edgeQueues.size()) &&
         !edgeQueues[edgeId].empty();
}

Token FunctionalSimulationBackend::Impl::popInputToken(IdIndex portId) {
  IdIndex edgeId = inputPortEdge[portId];
  Token token = edgeQueues[edgeId].front();
  edgeQueues[edgeId].pop_front();
  return token;
}

void FunctionalSimulationBackend::Impl::pushToken(IdIndex portId,
                                                   Token token) {
  if (portId == INVALID_ID ||
      portId >= static_cast<IdIndex>(outputPortEdges.size()))
    return;
  const Port *port = dfg.getPort(portId);
  if (port)
    token.data = coerceValueToType(token.data, port->type);
  for (IdIndex edgeId : outputPortEdges[portId]) {
    if (edgeId == INVALID_ID ||
        edgeId >= static_cast<IdIndex>(edgeQueues.size()))
      continue;
    edgeQueues[edgeId].push_back(token);
  }
}

// ===----------------------------------------------------------------------===
// Impl -- value helpers
// ===----------------------------------------------------------------------===

unsigned FunctionalSimulationBackend::Impl::getSelectIndex(
    uint64_t rawValue, mlir::Type type) const {
  return static_cast<unsigned>(coerceValueToType(rawValue, type));
}

uint64_t FunctionalSimulationBackend::Impl::applyStreamStep(
    uint64_t current, uint64_t step, llvm::StringRef stepOp) const {
  if (stepOp == "+=")
    return current + step;
  if (stepOp == "-=")
    return current - step;
  if (stepOp == "*=")
    return current * step;
  if (stepOp == "/=")
    return step == 0 ? current : current / step;
  if (stepOp == "<<=")
    return current << step;
  if (stepOp == ">>=")
    return current >> step;
  return current + step;
}

bool FunctionalSimulationBackend::Impl::evaluateStreamCond(
    uint64_t lhs, uint64_t rhs, llvm::StringRef cond) const {
  if (cond == "<")
    return signExtendToI64(lhs, 64) < signExtendToI64(rhs, 64);
  if (cond == "<=")
    return signExtendToI64(lhs, 64) <= signExtendToI64(rhs, 64);
  if (cond == ">")
    return signExtendToI64(lhs, 64) > signExtendToI64(rhs, 64);
  if (cond == ">=")
    return signExtendToI64(lhs, 64) >= signExtendToI64(rhs, 64);
  if (cond == "!=")
    return lhs != rhs;
  if (cond == "==")
    return lhs == rhs;
  return lhs < rhs;
}

std::optional<unsigned>
FunctionalSimulationBackend::Impl::getMemoryRegionId(IdIndex swNodeId) const {
  auto it = swMemoryNodeToRegion.find(swNodeId);
  if (it == swMemoryNodeToRegion.end())
    return std::nullopt;
  return it->second;
}

// ===----------------------------------------------------------------------===
// Impl -- memory operations
// ===----------------------------------------------------------------------===

bool FunctionalSimulationBackend::Impl::readMemory(unsigned regionId,
                                                    uint64_t index,
                                                    uint64_t &value,
                                                    std::string &error) const {
  if (regionId >= memoryRegions.size()) {
    error = "memory region out of range";
    return false;
  }
  const MemoryRegionBinding &region = memoryRegions[regionId];
  if (!region.data) {
    std::ostringstream oss;
    oss << "memory region " << regionId << " is not bound";
    error = oss.str();
    return false;
  }
  size_t elemBytes = size_t{1} << region.elemSizeLog2;
  size_t byteOffset = static_cast<size_t>(index) * elemBytes;
  if (byteOffset + elemBytes > region.sizeBytes) {
    std::ostringstream oss;
    oss << "memory region " << regionId << " load OOB at element " << index;
    error = oss.str();
    return false;
  }
  value = 0;
  for (size_t byte = 0; byte < elemBytes; ++byte)
    value |= uint64_t(region.data[byteOffset + byte]) << (byte * 8);
  return true;
}

bool FunctionalSimulationBackend::Impl::writeMemory(unsigned regionId,
                                                     uint64_t index,
                                                     uint64_t value,
                                                     std::string &error) {
  if (regionId >= memoryRegions.size()) {
    error = "memory region out of range";
    return false;
  }
  MemoryRegionBinding &region = memoryRegions[regionId];
  if (!region.data) {
    std::ostringstream oss;
    oss << "memory region " << regionId << " is not bound";
    error = oss.str();
    return false;
  }
  size_t elemBytes = size_t{1} << region.elemSizeLog2;
  size_t byteOffset = static_cast<size_t>(index) * elemBytes;
  if (byteOffset + elemBytes > region.sizeBytes) {
    std::ostringstream oss;
    oss << "memory region " << regionId << " store OOB at element " << index;
    error = oss.str();
    return false;
  }
  for (size_t byte = 0; byte < elemBytes; ++byte)
    region.data[byteOffset + byte] =
        static_cast<uint8_t>((value >> (byte * 8)) & 0xffu);
  return true;
}

// ===----------------------------------------------------------------------===
// Impl -- node dispatch
// ===----------------------------------------------------------------------===

FunctionalSimulationBackend::Impl::Action
FunctionalSimulationBackend::Impl::tryExecuteNode(IdIndex nodeId) {
  Action action;
  const Node *node = dfg.getNode(nodeId);
  if (!node)
    return action;

  llvm::StringRef opName = getNodeAttrStr(node, "op_name");
  if (node->kind == Node::ModuleInputNode)
    return action;
  if (node->kind == Node::ModuleOutputNode)
    return executeModuleOutput(nodeId, node);

  if (opMatches(opName, "join"))
    return executeJoin(nodeId, node);
  if (opMatches(opName, "fork") || opMatches(opName, "lazy_fork") ||
      opMatches(opName, "branch"))
    return executeForkLike(nodeId, node);
  if (opMatches(opName, "sink"))
    return executeSink(nodeId, node);
  if (opMatches(opName, "source"))
    return executeSource(nodeId, node);
  if (opMatches(opName, "merge"))
    return executeMerge(nodeId, node, false);
  if (opMatches(opName, "control_merge"))
    return executeMerge(nodeId, node, true);
  if (opMatches(opName, "constant"))
    return node->inputPorts.empty() ? executeArithConstant(nodeId, node)
                                    : executeHandshakeConstant(nodeId, node);
  if (opMatches(opName, "stream"))
    return executeStream(nodeId, node);
  if (opMatches(opName, "gate"))
    return executeGate(nodeId, node);
  if (opMatches(opName, "carry"))
    return executeCarry(nodeId, node);
  if (opMatches(opName, "invariant"))
    return executeInvariant(nodeId, node);
  if (opMatches(opName, "cond_br"))
    return executeCondBr(nodeId, node);
  if (opMatches(opName, "select"))
    return executeSelect(nodeId, node);
  if (opName == "handshake.mux")
    return executeHandshakeMux(nodeId, node);
  if (opName == "fabric.mux")
    return executeFabricMux(nodeId, node);
  if (opMatches(opName, "load"))
    return executeLoad(nodeId, node);
  if (opMatches(opName, "store"))
    return executeStore(nodeId, node);
  if (opMatches(opName, "memory") || opMatches(opName, "extmemory"))
    return executeMemory(nodeId, node);
  if (opMatches(opName, "index_cast") || opMatches(opName, "index_castui") ||
      opMatches(opName, "trunci") || opMatches(opName, "extui") ||
      opMatches(opName, "extsi"))
    return executeIndexCast(nodeId, node);
  if (opMatches(opName, "cmpi"))
    return executeCmpi(nodeId, node);
  if (opMatches(opName, "addi") || opMatches(opName, "subi") ||
      opMatches(opName, "muli") || opMatches(opName, "divsi") ||
      opMatches(opName, "divui") || opMatches(opName, "remsi") ||
      opMatches(opName, "remui") || opMatches(opName, "andi") ||
      opMatches(opName, "ori") || opMatches(opName, "xori") ||
      opMatches(opName, "shli") || opMatches(opName, "shrsi") ||
      opMatches(opName, "shrui"))
    return executeBinaryArith(nodeId, node, opName);

  action.error = "unsupported DFG op in functional simulator: " + opName.str();
  return action;
}

// ===----------------------------------------------------------------------===
// Public pimpl forwarding
// ===----------------------------------------------------------------------===

FunctionalSimulationBackend::FunctionalSimulationBackend(const SimConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

FunctionalSimulationBackend::~FunctionalSimulationBackend() = default;

std::string FunctionalSimulationBackend::connect() { return impl_->connect(); }

std::string FunctionalSimulationBackend::buildFromMappedState(
    const Graph &dfg, const Graph &adg, const MappingState &mapping) {
  return impl_->buildFromMappedState(dfg, adg, mapping);
}

std::string FunctionalSimulationBackend::buildFromMappedState(
    const Graph &dfg, const Graph &adg, const MappingState &mapping,
    llvm::ArrayRef<PEContainment> peContainment) {
  (void)peContainment;
  return impl_->buildFromMappedState(dfg, adg, mapping);
}

std::string
FunctionalSimulationBackend::loadConfig(const std::vector<uint8_t> &configBlob) {
  return impl_->loadConfig(configBlob);
}

std::string FunctionalSimulationBackend::setInput(
    unsigned portIdx, const std::vector<uint64_t> &data,
    const std::vector<uint16_t> &tags) {
  return impl_->setInput(portIdx, data, tags);
}

std::string FunctionalSimulationBackend::setExtMemoryBacking(unsigned regionId,
                                                             uint8_t *data,
                                                             size_t sizeBytes) {
  return impl_->setExtMemoryBacking(regionId, data, sizeBytes);
}

SimResult FunctionalSimulationBackend::invoke(uint32_t epochId,
                                              uint64_t invocationId) {
  return impl_->invoke(epochId, invocationId);
}

std::vector<uint64_t>
FunctionalSimulationBackend::getOutput(unsigned portIdx) const {
  return impl_->getOutput(portIdx);
}

std::vector<uint16_t>
FunctionalSimulationBackend::getOutputTags(unsigned portIdx) const {
  return impl_->getOutputTags(portIdx);
}

void FunctionalSimulationBackend::resetExecution() { impl_->resetExecution(); }

void FunctionalSimulationBackend::resetAll() { impl_->resetAll(); }

unsigned FunctionalSimulationBackend::getNumInputPorts() const {
  return impl_->getNumInputPorts();
}

unsigned FunctionalSimulationBackend::getNumOutputPorts() const {
  return impl_->getNumOutputPorts();
}

} // namespace sim
} // namespace fcc
