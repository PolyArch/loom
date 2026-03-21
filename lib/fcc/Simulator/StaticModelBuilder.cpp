#include "fcc/Simulator/StaticModelBuilder.h"

#include "fcc/Mapper/BridgeBinding.h"
#include "fcc/Mapper/TypeCompat.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/StringRef.h"

#include <algorithm>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace fcc {
namespace sim {

namespace {

constexpr unsigned kInvalidOrdinal = std::numeric_limits<unsigned>::max();

mlir::Attribute getNodeAttr(const Node *node, llvm::StringRef key) {
  if (!node)
    return {};
  for (const auto &attr : node->attributes) {
    if (attr.getName() == key)
      return attr.getValue();
  }
  return {};
}

mlir::Attribute getEdgeAttr(const Edge *edge, llvm::StringRef key) {
  if (!edge)
    return {};
  for (const auto &attr : edge->attributes) {
    if (attr.getName() == key)
      return attr.getValue();
  }
  return {};
}

int getEdgeAttrInt(const Edge *edge, llvm::StringRef key, int defaultValue = -1) {
  if (auto intAttr =
          mlir::dyn_cast_or_null<mlir::IntegerAttr>(getEdgeAttr(edge, key)))
    return static_cast<int>(intAttr.getInt());
  return defaultValue;
}

std::string getNodeAttrString(const Node *node, llvm::StringRef key,
                              llvm::StringRef defaultValue = "") {
  if (auto strAttr =
          mlir::dyn_cast_or_null<mlir::StringAttr>(getNodeAttr(node, key)))
    return strAttr.getValue().str();
  return defaultValue.str();
}

bool opMatches(llvm::StringRef opName, llvm::StringRef shortName) {
  if (opName == shortName)
    return true;
  size_t dot = opName.rfind('.');
  return dot != llvm::StringRef::npos &&
         opName.drop_front(dot + 1) == shortName;
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

BridgeLaneRange getMemoryLaneRange(const Node *swNode, const Node *hwNode,
                                   const Graph &dfg, const Graph &adg,
                                   const MappingState &mapping) {
  if (!swNode || !hwNode)
    return {};

  bool isExtMem = (getNodeAttrString(hwNode, "op_kind") == "extmemory");
  DfgMemoryInfo memInfo = DfgMemoryInfo::extract(swNode, dfg, isExtMem);
  BridgeInfo bridge = BridgeInfo::extract(hwNode);
  if (bridge.hasBridge) {
    if (auto range = inferBridgeLaneRange(bridge, memInfo, swNode, mapping))
      return *range;
  }
  return {0u, memInfo.laneSpan()};
}

StaticModuleKind classifyModuleKind(const Node *node) {
  if (!node)
    return StaticModuleKind::Unknown;
  if (node->kind == Node::ModuleInputNode)
    return StaticModuleKind::BoundaryInput;
  if (node->kind == Node::ModuleOutputNode)
    return StaticModuleKind::BoundaryOutput;

  std::string opKind = getNodeAttrString(node, "op_kind");
  if (opKind == "function_unit")
    return StaticModuleKind::FunctionUnit;
  if (opKind == "spatial_sw")
    return StaticModuleKind::SpatialSwitch;
  if (opKind == "temporal_sw")
    return StaticModuleKind::TemporalSwitch;
  if (opKind == "add_tag")
    return StaticModuleKind::AddTag;
  if (opKind == "map_tag")
    return StaticModuleKind::MapTag;
  if (opKind == "del_tag")
    return StaticModuleKind::DelTag;
  if (opKind == "fifo")
    return StaticModuleKind::Fifo;
  if (opKind == "memory")
    return StaticModuleKind::Memory;
  if (opKind == "extmemory")
    return StaticModuleKind::ExtMemory;
  return StaticModuleKind::Unknown;
}

std::vector<unsigned> buildBoundaryOrdinals(const Graph &graph, Node::Kind kind) {
  std::vector<unsigned> ordinals(graph.nodes.size(), kInvalidOrdinal);
  unsigned count = 0;
  for (IdIndex nodeId = 0; nodeId < static_cast<IdIndex>(graph.nodes.size());
       ++nodeId) {
    const Node *node = graph.getNode(nodeId);
    if (!node || node->kind != kind)
      continue;
    ordinals[nodeId] = count++;
  }
  return ordinals;
}

StaticPortDesc buildPortDesc(const Port *port, IdIndex portId) {
  StaticPortDesc desc;
  desc.portId = static_cast<uint32_t>(portId);
  desc.parentNodeId = static_cast<uint32_t>(port->parentNode);
  desc.direction = (port->direction == Port::Input)
                       ? StaticPortDirection::Input
                       : StaticPortDirection::Output;
  desc.isMemRef = mlir::isa<mlir::MemRefType>(port->type);
  desc.isNone = mlir::isa<mlir::NoneType>(port->type);
  if (auto info = detail::getPortTypeInfo(port->type)) {
    desc.isTagged = info->isTagged;
    desc.valueWidth = info->valueWidth;
    desc.tagWidth = info->tagWidth;
  }
  return desc;
}

} // namespace

bool buildStaticMappedModel(const Graph &dfg, const Graph &adg,
                            const MappingState &mapping,
                            llvm::ArrayRef<PEContainment> peContainment,
                            StaticMappedModel &model) {
  model.mutableModules().clear();
  model.mutableChannels().clear();
  model.mutablePorts().clear();
  model.mutablePEs().clear();
  model.mutableInputBindings().clear();
  model.mutableOutputBindings().clear();
  model.mutableMemoryBindings().clear();
  model.mutableCompletionObligations().clear();

  auto &modules = model.mutableModules();
  auto &channels = model.mutableChannels();
  auto &ports = model.mutablePorts();
  auto &pes = model.mutablePEs();
  auto &inputBindings = model.mutableInputBindings();
  auto &outputBindings = model.mutableOutputBindings();
  auto &memoryBindings = model.mutableMemoryBindings();
  auto &obligations = model.mutableCompletionObligations();
  auto &boundaryInputOrdinals = model.mutableBoundaryInputOrdinals();
  auto &boundaryOutputOrdinals = model.mutableBoundaryOutputOrdinals();

  boundaryInputOrdinals = buildBoundaryOrdinals(adg, Node::ModuleInputNode);
  boundaryOutputOrdinals = buildBoundaryOrdinals(adg, Node::ModuleOutputNode);

  ports.reserve(adg.ports.size());
  for (IdIndex portId = 0; portId < static_cast<IdIndex>(adg.ports.size());
       ++portId) {
    const Port *port = adg.getPort(portId);
    if (!port)
      continue;
    ports.push_back(buildPortDesc(port, portId));
  }

  pes.reserve(peContainment.size());
  for (const PEContainment &pe : peContainment) {
    StaticPEDesc desc;
    desc.peName = pe.peName;
    desc.peKind = pe.peKind;
    desc.fuNodeIds.assign(pe.fuNodeIds.begin(), pe.fuNodeIds.end());
    desc.row = pe.row;
    desc.col = pe.col;
    desc.numInputPorts = pe.numInputPorts;
    desc.numOutputPorts = pe.numOutputPorts;
    desc.numInstruction = pe.numInstruction;
    desc.numRegister = pe.numRegister;
    desc.regFifoDepth = pe.regFifoDepth;
    desc.tagWidth = pe.tagWidth;
    desc.enableShareOperandBuffer = pe.enableShareOperandBuffer;
    desc.operandBufferSize = pe.operandBufferSize;
    pes.push_back(std::move(desc));
  }

  modules.reserve(adg.nodes.size());
  for (IdIndex nodeId = 0; nodeId < static_cast<IdIndex>(adg.nodes.size());
       ++nodeId) {
    const Node *node = adg.getNode(nodeId);
    if (!node)
      continue;

    StaticModuleDesc module;
    module.hwNodeId = static_cast<uint32_t>(nodeId);
    module.nodeKind = (node->kind == Node::ModuleInputNode)
                          ? StaticGraphNodeKind::ModuleInput
                          : (node->kind == Node::ModuleOutputNode)
                                ? StaticGraphNodeKind::ModuleOutput
                                : StaticGraphNodeKind::Operation;
    module.kind = classifyModuleKind(node);
    module.name = getNodeAttrString(node, "op_name", "node");
    module.opKind = getNodeAttrString(node, "op_kind");
    module.resourceClass = getNodeAttrString(node, "resource_class");
    module.inputPorts.assign(node->inputPorts.begin(), node->inputPorts.end());
    module.outputPorts.assign(node->outputPorts.begin(), node->outputPorts.end());

    for (const auto &attr : node->attributes) {
      if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(attr.getValue())) {
        module.intAttrs.push_back({attr.getName().str(), intAttr.getInt()});
        continue;
      }
      if (auto boolAttr = mlir::dyn_cast<mlir::BoolAttr>(attr.getValue())) {
        module.intAttrs.push_back(
            {attr.getName().str(), boolAttr.getValue() ? 1 : 0});
        continue;
      }
      if (auto strAttr = mlir::dyn_cast<mlir::StringAttr>(attr.getValue())) {
        module.strAttrs.push_back(
            {attr.getName().str(), strAttr.getValue().str()});
        continue;
      }
      if (auto arrAttr = mlir::dyn_cast<mlir::DenseI8ArrayAttr>(attr.getValue())) {
        module.byteArrayAttrs.push_back(
            {attr.getName().str(),
             std::vector<int8_t>(arrAttr.asArrayRef().begin(),
                                 arrAttr.asArrayRef().end())});
        continue;
      }
      if (auto arrAttr = mlir::dyn_cast<mlir::DenseI64ArrayAttr>(attr.getValue())) {
        module.intArrayAttrs.push_back(
            {attr.getName().str(),
             std::vector<int64_t>(arrAttr.asArrayRef().begin(),
                                  arrAttr.asArrayRef().end())});
        continue;
      }
      if (auto arrayAttr = mlir::dyn_cast<mlir::ArrayAttr>(attr.getValue())) {
        bool allStrings = true;
        std::vector<std::string> values;
        values.reserve(arrayAttr.size());
        for (mlir::Attribute elem : arrayAttr) {
          auto strAttr = mlir::dyn_cast<mlir::StringAttr>(elem);
          if (!strAttr) {
            allStrings = false;
            break;
          }
          values.push_back(strAttr.getValue().str());
        }
        if (allStrings)
          module.stringArrayAttrs.push_back(
              {attr.getName().str(), std::move(values)});
      }
    }
    modules.push_back(std::move(module));
  }

  std::unordered_set<IdIndex> activeHwEdges;
  activeHwEdges.reserve(adg.edges.size());

  auto addIncidentEdges = [&](IdIndex nodeId) {
    const Node *node = adg.getNode(nodeId);
    if (!node)
      return;
    for (IdIndex portId : node->inputPorts) {
      const Port *port = adg.getPort(portId);
      if (!port)
        continue;
      for (IdIndex edgeId : port->connectedEdges) {
        if (edgeId != INVALID_ID && adg.getEdge(edgeId))
          activeHwEdges.insert(edgeId);
      }
    }
    for (IdIndex portId : node->outputPorts) {
      const Port *port = adg.getPort(portId);
      if (!port)
        continue;
      for (IdIndex edgeId : port->connectedEdges) {
        if (edgeId != INVALID_ID && adg.getEdge(edgeId))
          activeHwEdges.insert(edgeId);
      }
    }
  };

  for (IdIndex edgeId = 0;
       edgeId < static_cast<IdIndex>(mapping.hwEdgeToSwEdges.size()); ++edgeId) {
    if (!mapping.hwEdgeToSwEdges[edgeId].empty())
      activeHwEdges.insert(edgeId);
  }

  std::unordered_set<IdIndex> memoryAdjacentHelperNodes;
  for (IdIndex nodeId = 0; nodeId < static_cast<IdIndex>(adg.nodes.size());
       ++nodeId) {
    const Node *node = adg.getNode(nodeId);
    if (!node)
      continue;
    StaticModuleKind kind = classifyModuleKind(node);
    if (kind != StaticModuleKind::Memory &&
        kind != StaticModuleKind::ExtMemory)
      continue;
    addIncidentEdges(nodeId);
    auto recordAdjacentHelpers = [&](llvm::ArrayRef<IdIndex> ports) {
      for (IdIndex portId : ports) {
        const Port *port = adg.getPort(portId);
        if (!port)
          continue;
        for (IdIndex edgeId : port->connectedEdges) {
          const Edge *edge = adg.getEdge(edgeId);
          if (!edge)
            continue;
          IdIndex neighborPortId =
              (edge->srcPort == portId) ? edge->dstPort : edge->srcPort;
          const Port *neighborPort = adg.getPort(neighborPortId);
          if (!neighborPort)
            continue;
          const Node *neighborNode = adg.getNode(neighborPort->parentNode);
          if (!neighborNode)
            continue;
          StaticModuleKind neighborKind = classifyModuleKind(neighborNode);
          if (neighborKind == StaticModuleKind::SpatialSwitch ||
              neighborKind == StaticModuleKind::TemporalSwitch ||
              neighborKind == StaticModuleKind::AddTag ||
              neighborKind == StaticModuleKind::MapTag ||
              neighborKind == StaticModuleKind::DelTag ||
              neighborKind == StaticModuleKind::Fifo) {
            memoryAdjacentHelperNodes.insert(neighborPort->parentNode);
          }
        }
      }
    };
    recordAdjacentHelpers(node->inputPorts);
    recordAdjacentHelpers(node->outputPorts);
  }

  for (IdIndex helperNodeId : memoryAdjacentHelperNodes)
    addIncidentEdges(helperNodeId);

  for (IdIndex nodeId = 0; nodeId < static_cast<IdIndex>(adg.nodes.size());
       ++nodeId) {
    const Node *node = adg.getNode(nodeId);
    if (!node)
      continue;
    if (node->kind == Node::ModuleInputNode || node->kind == Node::ModuleOutputNode)
      addIncidentEdges(nodeId);
    StaticModuleKind kind = classifyModuleKind(node);
    if (kind == StaticModuleKind::AddTag || kind == StaticModuleKind::MapTag ||
        kind == StaticModuleKind::DelTag)
      addIncidentEdges(nodeId);
  }

  std::vector<IdIndex> sortedActiveHwEdges;
  sortedActiveHwEdges.reserve(activeHwEdges.size());
  for (IdIndex edgeId : activeHwEdges)
    sortedActiveHwEdges.push_back(edgeId);
  std::sort(sortedActiveHwEdges.begin(), sortedActiveHwEdges.end());

  channels.reserve(sortedActiveHwEdges.size());
  for (IdIndex edgeId : sortedActiveHwEdges) {
    const Edge *edge = adg.getEdge(edgeId);
    if (!edge)
      continue;
    const Port *srcPort = adg.getPort(edge->srcPort);
    const Port *dstPort = adg.getPort(edge->dstPort);
    if (!srcPort || !dstPort)
      continue;

    StaticChannelDesc channel;
    channel.hwEdgeId = static_cast<uint32_t>(edgeId);
    channel.srcPort = edge->srcPort;
    channel.dstPort = edge->dstPort;
    channel.srcNode = srcPort->parentNode;
    channel.dstNode = dstPort->parentNode;
    channel.peInputIndex = getEdgeAttrInt(edge, "pe_input_index", -1);
    channel.peOutputIndex = getEdgeAttrInt(edge, "pe_output_index", -1);
    channel.touchesBoundaryInput =
        srcPort->parentNode >= 0 &&
        srcPort->parentNode < static_cast<IdIndex>(boundaryInputOrdinals.size()) &&
        boundaryInputOrdinals[srcPort->parentNode] != kInvalidOrdinal;
    channel.touchesBoundaryOutput =
        dstPort->parentNode >= 0 &&
        dstPort->parentNode <
            static_cast<IdIndex>(boundaryOutputOrdinals.size()) &&
        boundaryOutputOrdinals[dstPort->parentNode] != kInvalidOrdinal;
    channels.push_back(std::move(channel));
  }

  for (IdIndex swNodeId = 0; swNodeId < static_cast<IdIndex>(dfg.nodes.size());
       ++swNodeId) {
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
      if (hwNodeId < static_cast<IdIndex>(boundaryInputOrdinals.size()) &&
          boundaryInputOrdinals[hwNodeId] != kInvalidOrdinal) {
        inputBindings.push_back({boundaryInputOrdinals[hwNodeId], swNodeId});
      }
    }

    if (swNode->kind == Node::ModuleOutputNode) {
      bool softwareVisible = true;
      if (!swNode->inputPorts.empty()) {
        const Port *inPort = dfg.getPort(swNode->inputPorts.front());
        if (inPort && mlir::isa<mlir::NoneType>(inPort->type))
          softwareVisible = false;
      }
      if (hwNodeId < static_cast<IdIndex>(boundaryOutputOrdinals.size()) &&
          boundaryOutputOrdinals[hwNodeId] != kInvalidOrdinal) {
        unsigned ordinal = boundaryOutputOrdinals[hwNodeId];
        outputBindings.push_back({ordinal, swNodeId});
        if (softwareVisible) {
          CompletionObligation obligation;
          obligation.kind = CompletionObligationKind::OutputPort;
          obligation.ordinal = ordinal;
          obligation.swNodeId = swNodeId;
          obligation.description =
              "software output bound to hardware boundary output " +
              std::to_string(ordinal);
          obligations.push_back(std::move(obligation));
        }
      }
      continue;
    }

    std::string opName = getNodeAttrString(swNode, "op_name");
    if (!opMatches(opName, "memory") && !opMatches(opName, "extmemory"))
      continue;

    const Node *hwNode = adg.getNode(hwNodeId);
    DfgMemoryInfo memInfo = DfgMemoryInfo::extract(
        swNode, dfg, opMatches(opName, "extmemory"));
    BridgeLaneRange laneRange =
        getMemoryLaneRange(swNode, hwNode, dfg, adg, mapping);
    StaticMemoryBinding binding;
    binding.regionId = static_cast<unsigned>(memoryBindings.size());
    binding.regionIndex = 0;
    binding.swNodeId = swNodeId;
    binding.hwNodeId = hwNodeId;
    binding.startLane = laneRange.start;
    binding.endLane = std::max(laneRange.end, laneRange.start + 1u);
    binding.elemSizeLog2 = getMemoryElemSizeLog2(swNode, hwNode, dfg, adg);
    binding.supportsLoad = memInfo.ldCount > 0;
    binding.supportsStore = memInfo.stCount > 0;
    memoryBindings.push_back(binding);
    if (memInfo.stCount > 0) {
      CompletionObligation obligation;
      obligation.kind = CompletionObligationKind::MemoryRegion;
      obligation.ordinal = binding.regionId;
      obligation.swNodeId = swNodeId;
      obligation.description =
          "software memory side effects for region " +
          std::to_string(binding.regionId);
      obligations.push_back(std::move(obligation));
    }
  }

  std::unordered_map<IdIndex, std::vector<size_t>> bindingIndicesByHwNode;
  for (size_t idx = 0; idx < memoryBindings.size(); ++idx)
    bindingIndicesByHwNode[memoryBindings[idx].hwNodeId].push_back(idx);
  for (auto &entry : bindingIndicesByHwNode) {
    auto &indices = entry.second;
    std::sort(indices.begin(), indices.end(),
              [&](size_t lhs, size_t rhs) {
                const auto &a = memoryBindings[lhs];
                const auto &b = memoryBindings[rhs];
                if (a.startLane != b.startLane)
                  return a.startLane < b.startLane;
                if (a.swNodeId != b.swNodeId)
                  return a.swNodeId < b.swNodeId;
                return a.regionId < b.regionId;
              });
    for (size_t localIndex = 0; localIndex < indices.size(); ++localIndex)
      memoryBindings[indices[localIndex]].regionIndex =
          static_cast<unsigned>(localIndex);
  }

  return true;
}

} // namespace sim
} // namespace fcc
