#include "fcc/Simulator/FunctionalBackend.h"

#include "fcc/Mapper/TypeCompat.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace fcc {
namespace sim {

namespace {

constexpr unsigned kInvalidOrdinal = std::numeric_limits<unsigned>::max();

struct Token {
  uint64_t data = 0;
  uint16_t tag = 0;
  bool hasTag = false;
};

mlir::Attribute getNodeAttr(const Node *node, llvm::StringRef key) {
  if (!node)
    return {};
  for (const auto &attr : node->attributes) {
    if (attr.getName() == key)
      return attr.getValue();
  }
  return {};
}

bool getNodeAttrBool(const Node *node, llvm::StringRef key,
                     bool defaultValue = false) {
  if (auto boolAttr = mlir::dyn_cast_or_null<mlir::BoolAttr>(
          getNodeAttr(node, key)))
    return boolAttr.getValue();
  return defaultValue;
}

std::string getNodeAttrString(const Node *node, llvm::StringRef key,
                              llvm::StringRef defaultValue = "") {
  if (auto strAttr = mlir::dyn_cast_or_null<mlir::StringAttr>(
          getNodeAttr(node, key)))
    return strAttr.getValue().str();
  return defaultValue.str();
}

bool opMatches(llvm::StringRef opName, llvm::StringRef shortName) {
  if (opName == shortName)
    return true;
  size_t dot = opName.rfind('.');
  return dot != llvm::StringRef::npos && opName.drop_front(dot + 1) == shortName;
}

unsigned getTypeBitWidth(mlir::Type type) {
  if (auto width = detail::getScalarWidth(type))
    return *width;
  return 64;
}

uint64_t maskToWidth(uint64_t value, unsigned width) {
  if (width == 0)
    return 0;
  if (width >= 64)
    return value;
  return value & ((uint64_t{1} << width) - 1);
}

int64_t signExtendToI64(uint64_t value, unsigned width) {
  if (width == 0)
    return 0;
  if (width >= 64)
    return static_cast<int64_t>(value);
  uint64_t masked = maskToWidth(value, width);
  uint64_t signBit = uint64_t{1} << (width - 1);
  if ((masked & signBit) == 0)
    return static_cast<int64_t>(masked);
  uint64_t extendMask = ~((uint64_t{1} << width) - 1);
  return static_cast<int64_t>(masked | extendMask);
}

uint64_t coerceValueToType(uint64_t value, mlir::Type type) {
  return maskToWidth(value, getTypeBitWidth(type));
}

unsigned getMemoryElemSizeLog2(const Node *swNode, const Node *hwNode,
                               const Graph &dfg, const Graph &adg) {
  if (swNode) {
    for (IdIndex portId : swNode->inputPorts) {
      const Port *port = dfg.getPort(portId);
      if (!port || !mlir::isa<mlir::MemRefType>(port->type))
        continue;
      if (auto log2 = detail::getMemRefElementByteWidthLog2(port->type))
        return *log2;
    }
  }

  if (hwNode) {
    for (IdIndex portId : hwNode->inputPorts) {
      const Port *port = adg.getPort(portId);
      if (!port || !mlir::isa<mlir::MemRefType>(port->type))
        continue;
      if (auto log2 = detail::getMemRefElementByteWidthLog2(port->type))
        return *log2;
    }
  }

  return 2;
}

std::vector<unsigned> buildBoundaryOrdinals(const Graph &graph, Node::Kind kind,
                                            unsigned &count) {
  std::vector<unsigned> ordinals(graph.nodes.size(), kInvalidOrdinal);
  count = 0;
  for (IdIndex nodeId = 0; nodeId < static_cast<IdIndex>(graph.nodes.size());
       ++nodeId) {
    const Node *node = graph.getNode(nodeId);
    if (!node || node->kind != kind)
      continue;
    ordinals[nodeId] = count++;
  }
  return ordinals;
}

std::optional<mlir::arith::CmpIPredicate> getCmpPredicate(const Node *node) {
  mlir::Attribute attr = getNodeAttr(node, "predicate");
  if (!attr)
    return std::nullopt;
  if (auto predAttr = mlir::dyn_cast<mlir::arith::CmpIPredicateAttr>(attr))
    return predAttr.getValue();
  if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(attr))
    return static_cast<mlir::arith::CmpIPredicate>(intAttr.getInt());
  return std::nullopt;
}

bool evaluateCmpPredicate(mlir::arith::CmpIPredicate predicate, uint64_t lhs,
                          uint64_t rhs, unsigned width) {
  switch (predicate) {
  case mlir::arith::CmpIPredicate::eq:
    return maskToWidth(lhs, width) == maskToWidth(rhs, width);
  case mlir::arith::CmpIPredicate::ne:
    return maskToWidth(lhs, width) != maskToWidth(rhs, width);
  case mlir::arith::CmpIPredicate::slt:
    return signExtendToI64(lhs, width) < signExtendToI64(rhs, width);
  case mlir::arith::CmpIPredicate::sle:
    return signExtendToI64(lhs, width) <= signExtendToI64(rhs, width);
  case mlir::arith::CmpIPredicate::sgt:
    return signExtendToI64(lhs, width) > signExtendToI64(rhs, width);
  case mlir::arith::CmpIPredicate::sge:
    return signExtendToI64(lhs, width) >= signExtendToI64(rhs, width);
  case mlir::arith::CmpIPredicate::ult:
    return maskToWidth(lhs, width) < maskToWidth(rhs, width);
  case mlir::arith::CmpIPredicate::ule:
    return maskToWidth(lhs, width) <= maskToWidth(rhs, width);
  case mlir::arith::CmpIPredicate::ugt:
    return maskToWidth(lhs, width) > maskToWidth(rhs, width);
  case mlir::arith::CmpIPredicate::uge:
    return maskToWidth(lhs, width) >= maskToWidth(rhs, width);
  }
  return false;
}

} // namespace

struct FunctionalSimulationBackend::Impl {
  struct MemoryRegionBinding {
    uint8_t *data = nullptr;
    size_t sizeBytes = 0;
    unsigned elemSizeLog2 = 2;
    IdIndex hwNode = INVALID_ID;
    IdIndex swNode = INVALID_ID;
  };

  struct StreamState {
    bool active = false;
    uint64_t nextIdx = 0;
    uint64_t step = 0;
    uint64_t bound = 0;
    std::string stepOp = "+=";
    std::string contCond = "<";
  };

  struct CarryState {
    enum Phase : uint8_t { NeedInit = 0, NeedCond = 1, NeedLoop = 2 };
    Phase phase = NeedInit;
    uint64_t initValue = 0;
  };

  struct InvariantState {
    enum Phase : uint8_t { NeedInit = 0, NeedCond = 1 };
    Phase phase = NeedInit;
    uint64_t storedValue = 0;
  };

  struct GateState {
    enum Phase : uint8_t { NeedHead = 0, NeedNext = 1 };
    Phase phase = NeedHead;
  };

  struct NodeRuntime {
    bool emittedOnce = false;
    PerfSnapshot perf;
    StreamState stream;
    CarryState carry;
    InvariantState invariant;
    GateState gate;
  };

  struct Action {
    bool progress = false;
    unsigned tokensIn = 0;
    unsigned tokensOut = 0;
    std::string error;
  };

  explicit Impl(const SimConfig &cfg) : config(cfg) {}

  SimConfig config;
  Graph dfg;
  Graph adg;
  MappingState mapping;
  bool built = false;
  uint64_t configWords = 0;
  unsigned numInputPorts = 0;
  unsigned numOutputPorts = 0;

  std::vector<unsigned> hwInputOrdinals;
  std::vector<unsigned> hwOutputOrdinals;
  std::vector<unsigned> swInputToHwOrdinal;
  std::vector<unsigned> swOutputToHwOrdinal;
  std::vector<IdIndex> hwOrdinalToSwInputNode;
  std::vector<IdIndex> hwOrdinalToSwOutputNode;

  std::vector<IdIndex> inputPortEdge;
  std::vector<std::vector<IdIndex>> outputPortEdges;
  std::vector<std::deque<Token>> edgeQueues;
  std::vector<std::vector<Token>> inputBindings;
  std::vector<std::vector<Token>> outputCollectors;
  std::vector<NodeRuntime> nodeRuntime;
  std::vector<MemoryRegionBinding> memoryRegions;
  llvm::DenseMap<IdIndex, unsigned> swMemoryNodeToRegion;

  std::string connect() { return {}; }

  std::string buildFromMappedState(const Graph &dfgGraph, const Graph &adgGraph,
                                   const MappingState &state) {
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
      if (edge->dstPort >= 0 && edge->dstPort < static_cast<IdIndex>(inputPortEdge.size()))
        inputPortEdge[edge->dstPort] = edgeId;
      if (edge->srcPort >= 0 &&
          edge->srcPort < static_cast<IdIndex>(outputPortEdges.size()))
        outputPortEdges[edge->srcPort].push_back(edgeId);
    }

    hwInputOrdinals = buildBoundaryOrdinals(adg, Node::ModuleInputNode,
                                            numInputPorts);
    hwOutputOrdinals = buildBoundaryOrdinals(adg, Node::ModuleOutputNode,
                                             numOutputPorts);

    swInputToHwOrdinal.assign(dfg.nodes.size(), kInvalidOrdinal);
    swOutputToHwOrdinal.assign(dfg.nodes.size(), kInvalidOrdinal);
    hwOrdinalToSwInputNode.assign(numInputPorts, INVALID_ID);
    hwOrdinalToSwOutputNode.assign(numOutputPorts, INVALID_ID);

    for (IdIndex swNodeId = 0; swNodeId < static_cast<IdIndex>(dfg.nodes.size());
         ++swNodeId) {
      const Node *swNode = dfg.getNode(swNodeId);
      if (!swNode || swNodeId >= static_cast<IdIndex>(mapping.swNodeToHwNode.size()))
        continue;
      IdIndex hwNodeId = mapping.swNodeToHwNode[swNodeId];
      if (hwNodeId == INVALID_ID || hwNodeId >= static_cast<IdIndex>(adg.nodes.size()))
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
    for (IdIndex hwNodeId = 0; hwNodeId < static_cast<IdIndex>(adg.nodes.size());
         ++hwNodeId) {
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

  std::string loadConfig(const std::vector<uint8_t> &configBlob) {
    if (!built)
      return "simulation backend has no mapped graph";
    configWords = (configBlob.size() + 3) / 4;
    return {};
  }

  std::string setInput(unsigned portIdx, const std::vector<uint64_t> &data,
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

  std::string setExtMemoryBacking(unsigned regionId, uint8_t *data,
                                  size_t sizeBytes) {
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

  unsigned getNumInputPorts() const { return numInputPorts; }
  unsigned getNumOutputPorts() const { return numOutputPorts; }

  std::vector<uint64_t> getOutput(unsigned portIdx) const {
    if (portIdx >= outputCollectors.size())
      return {};
    std::vector<uint64_t> data;
    data.reserve(outputCollectors[portIdx].size());
    for (const Token &token : outputCollectors[portIdx])
      data.push_back(token.data);
    return data;
  }

  std::vector<uint16_t> getOutputTags(unsigned portIdx) const {
    if (portIdx >= outputCollectors.size())
      return {};
    std::vector<uint16_t> tags;
    tags.reserve(outputCollectors[portIdx].size());
    for (const Token &token : outputCollectors[portIdx])
      tags.push_back(token.hasTag ? token.tag : 0);
    return tags;
  }

  void resetExecution() {
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

  void resetAll() {
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

  SimResult invoke(uint32_t epochId, uint64_t invocationId) {
    SimResult result;
    result.configCycles = configWords * config.configWordsPerCycle;
    result.totalConfigWrites = configWords;

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
          fire.hwNodeId = nodeId < static_cast<IdIndex>(mapping.swNodeToHwNode.size())
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

    return result;
  }

private:
  void seedInputs() {
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

  bool hasInputToken(IdIndex portId) const {
    if (portId == INVALID_ID || portId >= static_cast<IdIndex>(inputPortEdge.size()))
      return false;
    IdIndex edgeId = inputPortEdge[portId];
    return edgeId != INVALID_ID && edgeId < static_cast<IdIndex>(edgeQueues.size()) &&
           !edgeQueues[edgeId].empty();
  }

  Token popInputToken(IdIndex portId) {
    IdIndex edgeId = inputPortEdge[portId];
    Token token = edgeQueues[edgeId].front();
    edgeQueues[edgeId].pop_front();
    return token;
  }

  void pushToken(IdIndex portId, Token token) {
    if (portId == INVALID_ID || portId >= static_cast<IdIndex>(outputPortEdges.size()))
      return;
    const Port *port = dfg.getPort(portId);
    if (port)
      token.data = coerceValueToType(token.data, port->type);
    for (IdIndex edgeId : outputPortEdges[portId]) {
      if (edgeId == INVALID_ID || edgeId >= static_cast<IdIndex>(edgeQueues.size()))
        continue;
      edgeQueues[edgeId].push_back(token);
    }
  }

  unsigned getSelectIndex(uint64_t rawValue, mlir::Type type) const {
    return static_cast<unsigned>(coerceValueToType(rawValue, type));
  }

  uint64_t applyStreamStep(uint64_t current, uint64_t step,
                           llvm::StringRef stepOp) const {
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

  bool evaluateStreamCond(uint64_t lhs, uint64_t rhs,
                          llvm::StringRef cond) const {
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

  std::optional<unsigned> getMemoryRegionId(IdIndex swNodeId) const {
    auto it = swMemoryNodeToRegion.find(swNodeId);
    if (it == swMemoryNodeToRegion.end())
      return std::nullopt;
    return it->second;
  }

  bool readMemory(unsigned regionId, uint64_t index, uint64_t &value,
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

  bool writeMemory(unsigned regionId, uint64_t index, uint64_t value,
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

  Action tryExecuteNode(IdIndex nodeId) {
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

  Action executeModuleOutput(IdIndex nodeId, const Node *node) {
    Action action;
    if (node->inputPorts.empty() || !hasInputToken(node->inputPorts.front()))
      return action;
    unsigned ordinal = swOutputToHwOrdinal[nodeId];
    if (ordinal == kInvalidOrdinal || ordinal >= outputCollectors.size()) {
      action.error = "module output is not mapped to a hardware boundary";
      return action;
    }
    Token token = popInputToken(node->inputPorts.front());
    outputCollectors[ordinal].push_back(token);
    action.progress = true;
    action.tokensIn = 1;
    action.tokensOut = 1;
    return action;
  }

  Action executeJoin(IdIndex, const Node *node) {
    Action action;
    for (IdIndex portId : node->inputPorts) {
      if (!hasInputToken(portId))
        return action;
    }
    for (IdIndex portId : node->inputPorts)
      (void)popInputToken(portId);
    if (!node->outputPorts.empty())
      pushToken(node->outputPorts.front(), Token{});
    action.progress = true;
    action.tokensIn = static_cast<unsigned>(node->inputPorts.size());
    action.tokensOut = node->outputPorts.empty() ? 0u : 1u;
    return action;
  }

  Action executeForkLike(IdIndex, const Node *node) {
    Action action;
    if (node->inputPorts.empty() || !hasInputToken(node->inputPorts.front()))
      return action;
    Token token = popInputToken(node->inputPorts.front());
    for (IdIndex outPort : node->outputPorts)
      pushToken(outPort, token);
    action.progress = true;
    action.tokensIn = 1;
    action.tokensOut = static_cast<unsigned>(node->outputPorts.size());
    return action;
  }

  Action executeSink(IdIndex, const Node *node) {
    Action action;
    if (node->inputPorts.empty() || !hasInputToken(node->inputPorts.front()))
      return action;
    (void)popInputToken(node->inputPorts.front());
    action.progress = true;
    action.tokensIn = 1;
    return action;
  }

  Action executeSource(IdIndex nodeId, const Node *node) {
    Action action;
    NodeRuntime &runtime = nodeRuntime[nodeId];
    if (runtime.emittedOnce || node->outputPorts.empty())
      return action;
    pushToken(node->outputPorts.front(), Token{});
    runtime.emittedOnce = true;
    action.progress = true;
    action.tokensOut = 1;
    return action;
  }

  Action executeMerge(IdIndex, const Node *node, bool withIndex) {
    Action action;
    for (unsigned i = 0; i < node->inputPorts.size(); ++i) {
      IdIndex inPort = node->inputPorts[i];
      if (!hasInputToken(inPort))
        continue;
      Token token = popInputToken(inPort);
      if (!node->outputPorts.empty())
        pushToken(node->outputPorts.front(), token);
      if (withIndex && node->outputPorts.size() > 1) {
        Token indexToken;
        indexToken.data = i;
        pushToken(node->outputPorts[1], indexToken);
      }
      action.progress = true;
      action.tokensIn = 1;
      action.tokensOut =
          1 + static_cast<unsigned>(withIndex && node->outputPorts.size() > 1);
      return action;
    }
    return action;
  }

  Action executeHandshakeConstant(IdIndex, const Node *node) {
    Action action;
    if (node->inputPorts.empty() || node->outputPorts.empty() ||
        !hasInputToken(node->inputPorts.front()))
      return action;
    (void)popInputToken(node->inputPorts.front());
    Token token;
    if (auto intAttr =
            mlir::dyn_cast_or_null<mlir::IntegerAttr>(getNodeAttr(node, "value")))
      token.data = intAttr.getValue().getZExtValue();
    pushToken(node->outputPorts.front(), token);
    action.progress = true;
    action.tokensIn = 1;
    action.tokensOut = 1;
    return action;
  }

  Action executeArithConstant(IdIndex nodeId, const Node *node) {
    Action action;
    NodeRuntime &runtime = nodeRuntime[nodeId];
    if (runtime.emittedOnce || node->outputPorts.empty())
      return action;
    Token token;
    if (auto intAttr =
            mlir::dyn_cast_or_null<mlir::IntegerAttr>(getNodeAttr(node, "value")))
      token.data = intAttr.getValue().getZExtValue();
    pushToken(node->outputPorts.front(), token);
    runtime.emittedOnce = true;
    action.progress = true;
    action.tokensOut = 1;
    return action;
  }

  Action executeStream(IdIndex nodeId, const Node *node) {
    Action action;
    NodeRuntime &runtime = nodeRuntime[nodeId];
    if (node->inputPorts.size() < 3 || node->outputPorts.size() < 2)
      return action;

    if (!runtime.stream.active) {
      for (IdIndex portId : node->inputPorts) {
        if (!hasInputToken(portId))
          return action;
      }
      runtime.stream.nextIdx = popInputToken(node->inputPorts[0]).data;
      runtime.stream.step = popInputToken(node->inputPorts[1]).data;
      runtime.stream.bound = popInputToken(node->inputPorts[2]).data;
      runtime.stream.active = true;
      action.progress = true;
      action.tokensIn = 3;
      return action;
    }

    bool willContinue = evaluateStreamCond(
        runtime.stream.nextIdx, runtime.stream.bound, runtime.stream.contCond);
    Token idxToken;
    idxToken.data = runtime.stream.nextIdx;
    Token condToken;
    condToken.data = willContinue ? 1 : 0;
    pushToken(node->outputPorts[0], idxToken);
    pushToken(node->outputPorts[1], condToken);
    if (willContinue)
      runtime.stream.nextIdx = applyStreamStep(runtime.stream.nextIdx,
                                              runtime.stream.step,
                                              runtime.stream.stepOp);
    else
      runtime.stream.active = false;
    action.progress = true;
    action.tokensOut = 2;
    return action;
  }

  Action executeGate(IdIndex nodeId, const Node *node) {
    Action action;
    NodeRuntime &runtime = nodeRuntime[nodeId];
    if (node->inputPorts.size() < 2 || node->outputPorts.size() < 2)
      return action;
    if (!hasInputToken(node->inputPorts[0]) || !hasInputToken(node->inputPorts[1]))
      return action;

    Token value = popInputToken(node->inputPorts[0]);
    Token cond = popInputToken(node->inputPorts[1]);
    bool condBit = (cond.data & 1) != 0;
    action.progress = true;
    action.tokensIn = 2;

    if (runtime.gate.phase == GateState::NeedHead) {
      if (condBit) {
        pushToken(node->outputPorts[0], value);
        action.tokensOut = 1;
        runtime.gate.phase = GateState::NeedNext;
      }
      return action;
    }

    if (condBit) {
      pushToken(node->outputPorts[0], value);
      Token afterCond;
      afterCond.data = 1;
      pushToken(node->outputPorts[1], afterCond);
      action.tokensOut = 2;
    } else {
      Token afterCond;
      afterCond.data = 0;
      pushToken(node->outputPorts[1], afterCond);
      action.tokensOut = 1;
      runtime.gate.phase = GateState::NeedHead;
    }
    return action;
  }

  Action executeCarry(IdIndex nodeId, const Node *node) {
    Action action;
    NodeRuntime &runtime = nodeRuntime[nodeId];
    if (node->inputPorts.size() < 3 || node->outputPorts.empty())
      return action;

    switch (runtime.carry.phase) {
    case CarryState::NeedInit:
      if (!hasInputToken(node->inputPorts[1]))
        return action;
      runtime.carry.initValue = popInputToken(node->inputPorts[1]).data;
      pushToken(node->outputPorts.front(), Token{runtime.carry.initValue});
      runtime.carry.phase = CarryState::NeedCond;
      action.progress = true;
      action.tokensIn = 1;
      action.tokensOut = 1;
      return action;
    case CarryState::NeedCond: {
      if (!hasInputToken(node->inputPorts[0]))
        return action;
      bool cond = (popInputToken(node->inputPorts[0]).data & 1) != 0;
      runtime.carry.phase = cond ? CarryState::NeedLoop : CarryState::NeedInit;
      action.progress = true;
      action.tokensIn = 1;
      return action;
    }
    case CarryState::NeedLoop:
      if (!hasInputToken(node->inputPorts[2]))
        return action;
      pushToken(node->outputPorts.front(), popInputToken(node->inputPorts[2]));
      runtime.carry.phase = CarryState::NeedCond;
      action.progress = true;
      action.tokensIn = 1;
      action.tokensOut = 1;
      return action;
    }
    return action;
  }

  Action executeInvariant(IdIndex nodeId, const Node *node) {
    Action action;
    NodeRuntime &runtime = nodeRuntime[nodeId];
    if (node->inputPorts.size() < 2 || node->outputPorts.empty())
      return action;

    switch (runtime.invariant.phase) {
    case InvariantState::NeedInit:
      if (!hasInputToken(node->inputPorts[1]))
        return action;
      runtime.invariant.storedValue = popInputToken(node->inputPorts[1]).data;
      pushToken(node->outputPorts.front(),
                Token{runtime.invariant.storedValue});
      runtime.invariant.phase = InvariantState::NeedCond;
      action.progress = true;
      action.tokensIn = 1;
      action.tokensOut = 1;
      return action;
    case InvariantState::NeedCond:
      if (!hasInputToken(node->inputPorts[0]))
        return action;
      if ((popInputToken(node->inputPorts[0]).data & 1) != 0) {
        pushToken(node->outputPorts.front(),
                  Token{runtime.invariant.storedValue});
        action.tokensOut = 1;
      } else {
        runtime.invariant.phase = InvariantState::NeedInit;
      }
      action.progress = true;
      action.tokensIn = 1;
      return action;
    }
    return action;
  }

  Action executeCondBr(IdIndex, const Node *node) {
    Action action;
    if (node->inputPorts.size() < 2 || node->outputPorts.size() < 2)
      return action;
    if (!hasInputToken(node->inputPorts[0]) || !hasInputToken(node->inputPorts[1]))
      return action;
    bool cond = (popInputToken(node->inputPorts[0]).data & 1) != 0;
    Token data = popInputToken(node->inputPorts[1]);
    pushToken(node->outputPorts[cond ? 0 : 1], data);
    action.progress = true;
    action.tokensIn = 2;
    action.tokensOut = 1;
    return action;
  }

  Action executeSelect(IdIndex, const Node *node) {
    Action action;
    if (node->inputPorts.size() < 3 || node->outputPorts.empty())
      return action;
    for (IdIndex portId : node->inputPorts) {
      if (!hasInputToken(portId))
        return action;
    }
    bool cond = (popInputToken(node->inputPorts[0]).data & 1) != 0;
    Token trueToken = popInputToken(node->inputPorts[1]);
    Token falseToken = popInputToken(node->inputPorts[2]);
    pushToken(node->outputPorts.front(), cond ? trueToken : falseToken);
    action.progress = true;
    action.tokensIn = 3;
    action.tokensOut = 1;
    return action;
  }

  Action executeHandshakeMux(IdIndex, const Node *node) {
    Action action;
    if (node->inputPorts.size() < 2 || node->outputPorts.empty())
      return action;
    if (!hasInputToken(node->inputPorts[0]))
      return action;
    unsigned select =
        getSelectIndex(edgeQueues[inputPortEdge[node->inputPorts[0]]].front().data,
                       dfg.getPort(node->inputPorts[0])->type);
    if (select + 1 >= node->inputPorts.size()) {
      action.error = "mux select out of range";
      return action;
    }
    IdIndex selectedPort = node->inputPorts[select + 1];
    if (!hasInputToken(selectedPort))
      return action;
    (void)popInputToken(node->inputPorts[0]);
    Token selected = popInputToken(selectedPort);
    pushToken(node->outputPorts.front(), selected);
    action.progress = true;
    action.tokensIn = 2;
    action.tokensOut = 1;
    return action;
  }

  Action executeFabricMux(IdIndex, const Node *node) {
    Action action;
    unsigned numInputs = static_cast<unsigned>(node->inputPorts.size());
    unsigned numOutputs = static_cast<unsigned>(node->outputPorts.size());
    if (numInputs == 0 || numOutputs == 0)
      return action;

    bool disconnect = getNodeAttrBool(node, "disconnect");
    bool discard = getNodeAttrBool(node, "discard");
    unsigned sel = 0;
    if (auto intAttr = mlir::dyn_cast_or_null<mlir::IntegerAttr>(
            getNodeAttr(node, "sel")))
      sel = static_cast<unsigned>(intAttr.getInt());

    if (numInputs == 1 && numOutputs == 1) {
      if (disconnect || discard) {
        action.error = "1:1 mux cannot set discard or disconnect";
        return action;
      }
      if (!hasInputToken(node->inputPorts.front()))
        return action;
      Token token = popInputToken(node->inputPorts.front());
      pushToken(node->outputPorts.front(), token);
      action.progress = true;
      action.tokensIn = 1;
      action.tokensOut = 1;
      return action;
    }

    if (disconnect)
      return action;

    if (numOutputs == 1) {
      if (sel >= numInputs) {
        action.error = "mux select out of range";
        return action;
      }

      unsigned discardedInputs = 0;
      if (discard) {
        for (unsigned i = 0; i < numInputs; ++i) {
          if (i == sel || !hasInputToken(node->inputPorts[i]))
            continue;
          (void)popInputToken(node->inputPorts[i]);
          ++discardedInputs;
        }
      }

      if (!hasInputToken(node->inputPorts[sel])) {
        if (discardedInputs > 0) {
          action.progress = true;
          action.tokensIn = discardedInputs;
        }
        return action;
      }

      Token token = popInputToken(node->inputPorts[sel]);
      pushToken(node->outputPorts.front(), token);
      action.progress = true;
      action.tokensIn = discardedInputs + 1;
      action.tokensOut = 1;
      return action;
    }

    if (numInputs == 1) {
      if (sel >= numOutputs) {
        action.error = "mux select out of range";
        return action;
      }
      if (!hasInputToken(node->inputPorts.front()))
        return action;
      Token token = popInputToken(node->inputPorts.front());
      action.progress = true;
      action.tokensIn = 1;
      if (!discard) {
        pushToken(node->outputPorts[sel], token);
        action.tokensOut = 1;
      }
      return action;
    }

    action.error = "mux must be either M:1 or 1:M";
    return action;
  }

  Action executeLoad(IdIndex, const Node *node) {
    Action action;
    if (node->inputPorts.size() < 3 || node->outputPorts.size() < 2)
      return action;
    unsigned numAddr = static_cast<unsigned>(node->outputPorts.size() - 1);
    unsigned dataInputIdx = numAddr;
    unsigned ctrlInputIdx = numAddr + 1;
    if (ctrlInputIdx >= node->inputPorts.size())
      return action;

    bool canIssueAddr = hasInputToken(node->inputPorts[ctrlInputIdx]);
    for (unsigned i = 0; i < numAddr && canIssueAddr; ++i)
      canIssueAddr = hasInputToken(node->inputPorts[i]);

    if (canIssueAddr) {
      std::vector<Token> addrs;
      addrs.reserve(numAddr);
      for (unsigned i = 0; i < numAddr; ++i)
        addrs.push_back(popInputToken(node->inputPorts[i]));
      (void)popInputToken(node->inputPorts[ctrlInputIdx]);
      for (unsigned i = 0; i < numAddr; ++i)
        pushToken(node->outputPorts[i + 1], addrs[i]);
      action.progress = true;
      action.tokensIn = numAddr + 1;
      action.tokensOut = numAddr;
      return action;
    }

    if (dataInputIdx < node->inputPorts.size() &&
        hasInputToken(node->inputPorts[dataInputIdx])) {
      pushToken(node->outputPorts.front(),
                popInputToken(node->inputPorts[dataInputIdx]));
      action.progress = true;
      action.tokensIn = 1;
      action.tokensOut = 1;
    }
    return action;
  }

  Action executeStore(IdIndex, const Node *node) {
    Action action;
    if (node->inputPorts.size() < 3 || node->outputPorts.size() < 2)
      return action;
    unsigned numAddr = static_cast<unsigned>(node->outputPorts.size() - 1);
    unsigned dataInputIdx = numAddr;
    unsigned ctrlInputIdx = numAddr + 1;
    if (ctrlInputIdx >= node->inputPorts.size())
      return action;

    bool ready = hasInputToken(node->inputPorts[dataInputIdx]) &&
                 hasInputToken(node->inputPorts[ctrlInputIdx]);
    for (unsigned i = 0; i < numAddr && ready; ++i)
      ready = hasInputToken(node->inputPorts[i]);
    if (!ready)
      return action;

    std::vector<Token> addrs;
    addrs.reserve(numAddr);
    for (unsigned i = 0; i < numAddr; ++i)
      addrs.push_back(popInputToken(node->inputPorts[i]));
    Token data = popInputToken(node->inputPorts[dataInputIdx]);
    (void)popInputToken(node->inputPorts[ctrlInputIdx]);
    pushToken(node->outputPorts.front(), data);
    for (unsigned i = 0; i < numAddr; ++i)
      pushToken(node->outputPorts[i + 1], addrs[i]);
    action.progress = true;
    action.tokensIn = numAddr + 2;
    action.tokensOut = numAddr + 1;
    return action;
  }

  Action executeMemory(IdIndex nodeId, const Node *node) {
    Action action;
    auto regionId = getMemoryRegionId(nodeId);
    if (!regionId) {
      action.error = "memory node is missing a bound simulation region";
      return action;
    }

    unsigned ldCount = static_cast<unsigned>(std::max<int64_t>(
        0, getNodeAttrInt(node, "ldCount", 0)));
    unsigned stCount = static_cast<unsigned>(std::max<int64_t>(
        0, getNodeAttrInt(node, "stCount", 0)));
    bool hasMemrefInput =
        !node->inputPorts.empty() &&
        mlir::isa<mlir::MemRefType>(dfg.getPort(node->inputPorts.front())->type);
    unsigned inputBase = hasMemrefInput ? 1u : 0u;

    for (unsigned ldIdx = 0; ldIdx < ldCount; ++ldIdx) {
      unsigned portIdx = inputBase + stCount * 2 + ldIdx;
      if (portIdx >= node->inputPorts.size() || !hasInputToken(node->inputPorts[portIdx]))
        continue;
      Token addrToken = popInputToken(node->inputPorts[portIdx]);
      uint64_t loadedValue = 0;
      if (!readMemory(*regionId, addrToken.data, loadedValue, action.error))
        return action;
      if (ldIdx >= node->outputPorts.size()) {
        action.error = "memory load result index out of range";
        return action;
      }
      Token dataToken;
      dataToken.data = loadedValue;
      pushToken(node->outputPorts[ldIdx], dataToken);
      unsigned donePort = ldCount + stCount + ldIdx;
      if (donePort >= node->outputPorts.size()) {
        action.error = "memory load done index out of range";
        return action;
      }
      pushToken(node->outputPorts[donePort], Token{});
      action.progress = true;
      action.tokensIn = 1;
      action.tokensOut = 2;
      return action;
    }

    for (unsigned stIdx = 0; stIdx < stCount; ++stIdx) {
      unsigned dataPort = inputBase + stIdx * 2;
      unsigned addrPort = dataPort + 1;
      if (addrPort >= node->inputPorts.size() || !hasInputToken(node->inputPorts[dataPort]) ||
          !hasInputToken(node->inputPorts[addrPort]))
        continue;
      Token dataToken = popInputToken(node->inputPorts[dataPort]);
      Token addrToken = popInputToken(node->inputPorts[addrPort]);
      if (!writeMemory(*regionId, addrToken.data, dataToken.data, action.error))
        return action;
      unsigned donePort = ldCount + stIdx;
      if (donePort >= node->outputPorts.size()) {
        action.error = "memory store done index out of range";
        return action;
      }
      pushToken(node->outputPorts[donePort], Token{});
      action.progress = true;
      action.tokensIn = 2;
      action.tokensOut = 1;
      return action;
    }

    return action;
  }

  Action executeIndexCast(IdIndex, const Node *node) {
    Action action;
    if (node->inputPorts.empty() || node->outputPorts.empty() ||
        !hasInputToken(node->inputPorts.front()))
      return action;
    Token input = popInputToken(node->inputPorts.front());
    const Port *srcPort = dfg.getPort(node->inputPorts.front());
    const Port *dstPort = dfg.getPort(node->outputPorts.front());
    unsigned srcWidth = srcPort ? getTypeBitWidth(srcPort->type) : 64;
    uint64_t value = input.data;
    if (dstPort && dstPort->type.isIndex() && srcWidth < 64)
      value = static_cast<uint64_t>(signExtendToI64(input.data, srcWidth));
    Token output = input;
    output.data = dstPort ? coerceValueToType(value, dstPort->type) : value;
    pushToken(node->outputPorts.front(), output);
    action.progress = true;
    action.tokensIn = 1;
    action.tokensOut = 1;
    return action;
  }

  Action executeCmpi(IdIndex, const Node *node) {
    Action action;
    if (node->inputPorts.size() < 2 || node->outputPorts.empty())
      return action;
    if (!hasInputToken(node->inputPorts[0]) || !hasInputToken(node->inputPorts[1]))
      return action;
    auto predicate = getCmpPredicate(node);
    if (!predicate) {
      action.error = "cmpi is missing predicate attribute";
      return action;
    }
    Token lhs = popInputToken(node->inputPorts[0]);
    Token rhs = popInputToken(node->inputPorts[1]);
    const Port *lhsPort = dfg.getPort(node->inputPorts[0]);
    unsigned width = lhsPort ? getTypeBitWidth(lhsPort->type) : 64;
    Token resultToken;
    resultToken.data =
        evaluateCmpPredicate(*predicate, lhs.data, rhs.data, width) ? 1 : 0;
    pushToken(node->outputPorts.front(), resultToken);
    action.progress = true;
    action.tokensIn = 2;
    action.tokensOut = 1;
    return action;
  }

  Action executeBinaryArith(IdIndex, const Node *node, llvm::StringRef opName) {
    Action action;
    if (node->inputPorts.size() < 2 || node->outputPorts.empty())
      return action;
    if (!hasInputToken(node->inputPorts[0]) || !hasInputToken(node->inputPorts[1]))
      return action;
    Token lhs = popInputToken(node->inputPorts[0]);
    Token rhs = popInputToken(node->inputPorts[1]);
    const Port *dstPort = dfg.getPort(node->outputPorts.front());
    const Port *lhsPort = dfg.getPort(node->inputPorts[0]);
    unsigned width = lhsPort ? getTypeBitWidth(lhsPort->type) : 64;
    uint64_t resultValue = 0;

    if (opMatches(opName, "addi"))
      resultValue = lhs.data + rhs.data;
    else if (opMatches(opName, "subi"))
      resultValue = lhs.data - rhs.data;
    else if (opMatches(opName, "muli"))
      resultValue = lhs.data * rhs.data;
    else if (opMatches(opName, "divsi"))
      resultValue =
          rhs.data == 0
              ? 0
              : static_cast<uint64_t>(signExtendToI64(lhs.data, width) /
                                      signExtendToI64(rhs.data, width));
    else if (opMatches(opName, "divui"))
      resultValue = rhs.data == 0 ? 0 : lhs.data / rhs.data;
    else if (opMatches(opName, "remsi"))
      resultValue =
          rhs.data == 0
              ? 0
              : static_cast<uint64_t>(signExtendToI64(lhs.data, width) %
                                      signExtendToI64(rhs.data, width));
    else if (opMatches(opName, "remui"))
      resultValue = rhs.data == 0 ? 0 : lhs.data % rhs.data;
    else if (opMatches(opName, "andi"))
      resultValue = lhs.data & rhs.data;
    else if (opMatches(opName, "ori"))
      resultValue = lhs.data | rhs.data;
    else if (opMatches(opName, "xori"))
      resultValue = lhs.data ^ rhs.data;
    else if (opMatches(opName, "shli"))
      resultValue = lhs.data << rhs.data;
    else if (opMatches(opName, "shrsi"))
      resultValue =
          static_cast<uint64_t>(signExtendToI64(lhs.data, width) >> rhs.data);
    else if (opMatches(opName, "shrui"))
      resultValue = lhs.data >> rhs.data;
    else
      action.error = "unsupported binary arithmetic op";

    if (!action.error.empty())
      return action;

    Token output;
    output.data = dstPort ? coerceValueToType(resultValue, dstPort->type)
                          : resultValue;
    pushToken(node->outputPorts.front(), output);
    action.progress = true;
    action.tokensIn = 2;
    action.tokensOut = 1;
    return action;
  }
};

FunctionalSimulationBackend::FunctionalSimulationBackend(const SimConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

FunctionalSimulationBackend::~FunctionalSimulationBackend() = default;

std::string FunctionalSimulationBackend::connect() { return impl_->connect(); }

std::string FunctionalSimulationBackend::buildFromMappedState(
    const Graph &dfg, const Graph &adg, const MappingState &mapping) {
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
