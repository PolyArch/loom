#include "fcc/Simulator/SimInputSynthesis.h"

#include "fcc/Mapper/TypeCompat.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>

namespace fcc {
namespace sim {

namespace {

bool isMemrefType(mlir::Type type) { return mlir::isa<mlir::MemRefType>(type); }

std::vector<unsigned> buildBoundaryInputOrdinals(const Graph &adg) {
  std::vector<unsigned> ordinals(adg.nodes.size(), 0);
  unsigned nextOrdinal = 0;
  for (IdIndex nodeId = 0; nodeId < static_cast<IdIndex>(adg.nodes.size());
       ++nodeId) {
    const Node *node = adg.getNode(nodeId);
    if (!node || node->kind != Node::ModuleInputNode)
      continue;
    ordinals[nodeId] = nextOrdinal++;
  }
  return ordinals;
}

int64_t findDfgMemrefArgIndex(const Node *swNode, const Graph &dfg) {
  if (!swNode || swNode->inputPorts.empty())
    return -1;
  IdIndex memrefPort = swNode->inputPorts[0];
  const Port *inPort = dfg.getPort(memrefPort);
  if (!inPort)
    return -1;
  for (IdIndex edgeId : inPort->connectedEdges) {
    const Edge *edge = dfg.getEdge(edgeId);
    if (!edge || edge->dstPort != memrefPort)
      continue;
    const Port *srcPort = dfg.getPort(edge->srcPort);
    if (!srcPort || srcPort->parentNode == INVALID_ID)
      continue;
    const Node *srcNode = dfg.getNode(srcPort->parentNode);
    if (!srcNode || srcNode->kind != Node::ModuleInputNode)
      continue;
    return getNodeAttrInt(srcNode, "arg_index", -1);
  }
  return -1;
}

uint32_t getElemSizeLog2(const Node *swNode, const Node *hwNode,
                         const Graph &dfg, const Graph &adg) {
  if (swNode && !swNode->inputPorts.empty()) {
    if (const Port *swMemrefPort = dfg.getPort(swNode->inputPorts[0])) {
      if (auto log2 = detail::getMemRefElementByteWidthLog2(swMemrefPort->type))
        return static_cast<uint32_t>(*log2);
    }
  }

  if (hwNode && !hwNode->inputPorts.empty()) {
    if (const Port *hwMemrefPort = adg.getPort(hwNode->inputPorts[0])) {
      if (auto log2 = detail::getMemRefElementByteWidthLog2(hwMemrefPort->type))
        return static_cast<uint32_t>(*log2);
    }
  }

  return 2;
}

uint64_t synthesizeScalarValue(mlir::Type type, unsigned ordinal) {
  if (type.isIndex())
    return 4;
  if (mlir::isa<mlir::NoneType>(type))
    return 0;
  if (auto width = detail::getScalarWidth(type)) {
    if (*width == 0)
      return 0;
    if (*width >= 64)
      return static_cast<uint64_t>(ordinal + 1);
    uint64_t mask = (uint64_t{1} << *width) - 1;
    return (static_cast<uint64_t>(ordinal + 1) & mask);
  }
  return static_cast<uint64_t>(ordinal + 1);
}

void fillRegionBytes(std::vector<uint8_t> &bytes, uint32_t elemSizeLog2,
                     int64_t memrefArgIndex, bool zeroFill) {
  std::fill(bytes.begin(), bytes.end(), 0);
  if (zeroFill)
    return;

  size_t elemBytes = size_t{1} << elemSizeLog2;
  size_t elemCount = bytes.size() / elemBytes;
  uint64_t base = memrefArgIndex >= 0 ? static_cast<uint64_t>(1 + memrefArgIndex * 4)
                                      : 1u;
  for (size_t elem = 0; elem < elemCount; ++elem) {
    uint64_t value = base + elem;
    for (size_t byte = 0; byte < elemBytes; ++byte)
      bytes[elem * elemBytes + byte] =
          static_cast<uint8_t>((value >> (byte * 8)) & 0xffu);
  }
}

} // namespace

SynthesizedSetup synthesizeSimulationSetup(const Graph &dfg, const Graph &adg,
                                          const MappingState &mapping,
                                          unsigned vectorLength) {
  SynthesizedSetup setup;
  std::vector<unsigned> boundaryInputOrdinals = buildBoundaryInputOrdinals(adg);

  for (IdIndex swNodeId = 0; swNodeId < static_cast<IdIndex>(dfg.nodes.size());
       ++swNodeId) {
    const Node *swNode = dfg.getNode(swNodeId);
    if (!swNode || swNode->kind != Node::ModuleInputNode ||
        swNode->outputPorts.empty())
      continue;
    if (swNodeId >= mapping.swNodeToHwNode.size() ||
        mapping.swNodeToHwNode[swNodeId] == INVALID_ID)
      continue;

    const Port *swPort = dfg.getPort(swNode->outputPorts.front());
    if (!swPort || isMemrefType(swPort->type))
      continue;

    const Node *hwNode = adg.getNode(mapping.swNodeToHwNode[swNodeId]);
    if (!hwNode)
      continue;

    IdIndex hwNodeId = mapping.swNodeToHwNode[swNodeId];
    if (hwNodeId >= boundaryInputOrdinals.size())
      continue;

    SynthesizedInputPort input;
    input.portIdx = boundaryInputOrdinals[hwNodeId];
    llvm::raw_string_ostream typeOS(input.type);
    swPort->type.print(typeOS);
    input.data.push_back(
        synthesizeScalarValue(swPort->type, input.portIdx));
    setup.inputs.push_back(std::move(input));
  }

  std::sort(setup.inputs.begin(), setup.inputs.end(),
            [](const SynthesizedInputPort &a, const SynthesizedInputPort &b) {
              return a.portIdx < b.portIdx;
            });

  unsigned regionId = 0;
  unsigned elemCount = std::max(4u, vectorLength);
  for (IdIndex hwNodeId = 0; hwNodeId < static_cast<IdIndex>(adg.nodes.size());
       ++hwNodeId) {
    const Node *hwNode = adg.getNode(hwNodeId);
    if (!hwNode || getNodeAttrStr(hwNode, "resource_class") != "memory")
      continue;
    if (hwNodeId >= mapping.hwNodeToSwNodes.size())
      continue;

    for (IdIndex swNodeId : mapping.hwNodeToSwNodes[hwNodeId]) {
      const Node *swNode = dfg.getNode(swNodeId);
      if (!swNode)
        continue;

      SynthesizedMemoryRegion region;
      region.regionId = regionId++;
      region.hwNode = hwNodeId;
      region.swNode = swNodeId;
      region.memrefArgIndex = findDfgMemrefArgIndex(swNode, dfg);
      region.elemSizeLog2 = getElemSizeLog2(swNode, hwNode, dfg, adg);
      size_t byteSize = static_cast<size_t>(elemCount) << region.elemSizeLog2;
      region.data.resize(byteSize, 0);

      bool zeroFill = getNodeAttrInt(swNode, "stCount", 0) > 0 &&
                      getNodeAttrInt(swNode, "ldCount", 0) == 0;
      fillRegionBytes(region.data, region.elemSizeLog2, region.memrefArgIndex,
                      zeroFill);
      setup.memoryRegions.push_back(std::move(region));
    }
  }

  return setup;
}

bool writeSetupManifest(const SynthesizedSetup &setup,
                        const std::string &path) {
  std::error_code ec;
  llvm::raw_fd_ostream out(path, ec, llvm::sys::fs::OF_Text);
  if (ec) {
    llvm::errs() << "SimInputSynthesis: cannot open " << path << ": "
                 << ec.message() << "\n";
    return false;
  }

  out << "{\n";
  out << "  \"inputs\": [\n";
  for (size_t i = 0; i < setup.inputs.size(); ++i) {
    const auto &input = setup.inputs[i];
    if (i > 0)
      out << ",\n";
    out << "    {\"port\": " << input.portIdx << ", \"type\": \""
        << input.type << "\", \"data\": [";
    for (size_t j = 0; j < input.data.size(); ++j) {
      if (j > 0)
        out << ", ";
      out << input.data[j];
    }
    out << "]}";
  }
  out << "\n  ],\n";

  out << "  \"memory_regions\": [\n";
  for (size_t i = 0; i < setup.memoryRegions.size(); ++i) {
    const auto &region = setup.memoryRegions[i];
    if (i > 0)
      out << ",\n";
    out << "    {\"region\": " << region.regionId
        << ", \"hw_node\": " << region.hwNode
        << ", \"sw_node\": " << region.swNode
        << ", \"memref_arg_index\": " << region.memrefArgIndex
        << ", \"elem_size_log2\": " << region.elemSizeLog2
        << ", \"byte_size\": " << region.data.size()
        << ", \"preview\": [";
    size_t previewCount = std::min<size_t>(region.data.size(), 16);
    for (size_t j = 0; j < previewCount; ++j) {
      if (j > 0)
        out << ", ";
      out << static_cast<unsigned>(region.data[j]);
    }
    out << "]}";
  }
  out << "\n  ]\n";
  out << "}\n";
  return true;
}

} // namespace sim
} // namespace fcc
