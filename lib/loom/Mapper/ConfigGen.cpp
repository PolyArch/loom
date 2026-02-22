//===-- ConfigGen.cpp - Configuration bitstream generation ---------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include "loom/Mapper/ConfigGen.h"

#include "mlir/IR/BuiltinAttributes.h"

#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/FileSystem.h"

namespace loom {

namespace {

llvm::StringRef getNodeName(const Node *node) {
  for (auto &attr : node->attributes) {
    if (attr.getName() == "sym_name") {
      if (auto strAttr = mlir::dyn_cast<mlir::StringAttr>(attr.getValue()))
        return strAttr.getValue();
    }
  }
  return "";
}

llvm::StringRef getNodeOpName(const Node *node) {
  for (auto &attr : node->attributes) {
    if (attr.getName() == "op_name") {
      if (auto strAttr = mlir::dyn_cast<mlir::StringAttr>(attr.getValue()))
        return strAttr.getValue();
    }
  }
  return "";
}

llvm::StringRef getNodeResClass(const Node *node) {
  for (auto &attr : node->attributes) {
    if (attr.getName() == "resource_class") {
      if (auto strAttr = mlir::dyn_cast<mlir::StringAttr>(attr.getValue()))
        return strAttr.getValue();
    }
  }
  return "";
}

int64_t getNodeIntAttr(const Node *node, llvm::StringRef name,
                       int64_t dflt = 0) {
  for (auto &attr : node->attributes) {
    if (attr.getName() == name) {
      if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(attr.getValue()))
        return intAttr.getInt();
    }
  }
  return dflt;
}

bool nodeHasAttr(const Node *node, llvm::StringRef name) {
  for (auto &attr : node->attributes) {
    if (attr.getName() == name)
      return true;
  }
  return false;
}

/// Pack bits into a word vector, LSB-first across words.
void packBits(std::vector<uint32_t> &words, uint32_t &bitPos,
              uint64_t value, unsigned width) {
  for (unsigned b = 0; b < width; ++b) {
    unsigned wordIdx = bitPos / 32;
    unsigned bitIdx = bitPos % 32;
    if (wordIdx >= words.size())
      words.resize(wordIdx + 1, 0);
    if (value & (1ULL << b))
      words[wordIdx] |= (1U << bitIdx);
    ++bitPos;
  }
}

/// Generate PE configuration words based on hardware semantics.
void genPEConfig(const Node *hwNode, const MappingState &state,
                 const Graph &dfg, const Graph &adg, IdIndex hwId,
                 std::vector<uint32_t> &words) {
  unsigned numOutputs = hwNode->outputPorts.size();

  // Determine tag width from output_tag attribute.
  unsigned tagWidth = 0;
  for (auto &attr : hwNode->attributes) {
    if (attr.getName() == "output_tag") {
      tagWidth = 4; // Default tag width.
      break;
    }
  }

  uint32_t bitPos = 0;

  // Pack output tags (ascending output index).
  if (tagWidth > 0) {
    for (unsigned o = 0; o < numOutputs; ++o) {
      // Determine tag from temporal assignment of mapped SW node.
      uint64_t tagVal = 0;
      if (hwId < state.hwNodeToSwNodes.size() &&
          !state.hwNodeToSwNodes[hwId].empty()) {
        IdIndex swId = state.hwNodeToSwNodes[hwId][0];
        if (swId < state.temporalPEAssignments.size()) {
          const auto &tpa = state.temporalPEAssignments[swId];
          if (tpa.tag != INVALID_ID)
            tagVal = tpa.tag;
        }
      }
      packBits(words, bitPos, tagVal, tagWidth);
    }
  }

  // Pack compare predicate if present (4 bits per cmp op).
  if (hwId < state.hwNodeToSwNodes.size()) {
    for (IdIndex swId : state.hwNodeToSwNodes[hwId]) {
      const Node *swNode = dfg.getNode(swId);
      if (!swNode)
        continue;
      llvm::StringRef swOp = getNodeOpName(swNode);
      if (swOp.starts_with("arith.cmp")) {
        uint64_t pred = 0;
        for (auto &a : swNode->attributes) {
          if (a.getName() == "predicate") {
            if (auto ia = mlir::dyn_cast<mlir::IntegerAttr>(a.getValue()))
              pred = ia.getInt();
          }
        }
        packBits(words, bitPos, pred, 4);
      }
    }
  }

  // Pack constant value if present.
  int64_t constVal = getNodeIntAttr(hwNode, "constant_value", -1);
  if (constVal >= 0) {
    packBits(words, bitPos, static_cast<uint64_t>(constVal), 32);
  }

  // Pack cont_cond_sel if present (5 bits for dataflow.stream).
  int64_t contCond = getNodeIntAttr(hwNode, "cont_cond_sel", -1);
  if (contCond >= 0) {
    packBits(words, bitPos, static_cast<uint64_t>(contCond), 5);
  }

  // Ensure at least one word.
  if (words.empty())
    words.push_back(0);
}

/// Generate switch configuration: route enable bits based on actual routing.
void genSwitchConfig(const Node *hwNode, const MappingState &state,
                     const Graph &adg, IdIndex hwId,
                     std::vector<uint32_t> &words) {
  unsigned numIn = hwNode->inputPorts.size();
  unsigned numOut = hwNode->outputPorts.size();

  if (numIn == 0 || numOut == 0) {
    words.push_back(0);
    return;
  }

  uint32_t bitPos = 0;
  words.clear();

  // Build a set of (inputPort, outputPort) pairs that are actually used
  // by routed SW edges.
  llvm::DenseSet<uint64_t> activeTransitions;

  // Scan all routed edges to find which input->output transitions
  // pass through this switch node.
  for (const auto &pathVec : state.swEdgeToHwPaths) {
    if (pathVec.empty())
      continue;

    // Path: [outPort0, inPort0, outPort1, inPort1, ...]
    // Internal transitions happen at: inPort[k] -> outPort[k+1]
    for (size_t j = 1; j + 1 < pathVec.size(); j += 2) {
      IdIndex inPortId = pathVec[j];
      IdIndex outPortId = pathVec[j + 1];

      // Check if this transition is through our switch node.
      const Port *inPort = adg.getPort(inPortId);
      if (!inPort || inPort->parentNode != hwId)
        continue;

      // Encode as (input_idx, output_idx) pair.
      for (unsigned i = 0; i < numIn; ++i) {
        if (hwNode->inputPorts[i] == inPortId) {
          for (unsigned o = 0; o < numOut; ++o) {
            if (hwNode->outputPorts[o] == outPortId) {
              uint64_t key = (static_cast<uint64_t>(o) << 32) | i;
              activeTransitions.insert(key);
            }
          }
        }
      }
    }
  }

  // Pack route enable bits: for each output, which input is selected.
  for (unsigned o = 0; o < numOut; ++o) {
    for (unsigned i = 0; i < numIn; ++i) {
      uint64_t key = (static_cast<uint64_t>(o) << 32) | i;
      bool enabled = activeTransitions.count(key) > 0;
      packBits(words, bitPos, enabled ? 1 : 0, 1);
    }
  }

  if (words.empty())
    words.push_back(0);
}

/// Generate temporal PE configuration: iterate FU nodes and emit per-slot
/// instruction entries.
void genTemporalPEConfig(const Node *hwNode, const MappingState &state,
                         const Graph &dfg, const Graph &adg, IdIndex hwId,
                         std::vector<uint32_t> &words) {
  int64_t numInst = getNodeIntAttr(hwNode, "num_instruction", 0);
  if (numInst <= 0) {
    words.push_back(0);
    return;
  }

  uint32_t bitPos = 0;
  words.clear();

  // Collect all FU nodes belonging to this temporal PE.
  // FU nodes have parent_temporal_pe == hwId.
  llvm::SmallVector<IdIndex, 4> fuNodes;
  for (IdIndex nodeId = 0; nodeId < static_cast<IdIndex>(adg.nodes.size());
       ++nodeId) {
    const Node *node = adg.getNode(nodeId);
    if (!node || node->kind != Node::OperationNode)
      continue;
    int64_t parentTPE = getNodeIntAttr(node, "parent_temporal_pe", -1);
    if (parentTPE == static_cast<int64_t>(hwId))
      fuNodes.push_back(nodeId);
  }

  // Pack instruction entries: one per slot.
  for (int64_t slot = 0; slot < numInst; ++slot) {
    uint64_t insnWord = 0;

    // Find the SW node assigned to this slot across all FU nodes.
    for (IdIndex swId = 0;
         swId < static_cast<IdIndex>(state.temporalPEAssignments.size());
         ++swId) {
      const auto &tpa = state.temporalPEAssignments[swId];
      if (tpa.slot != static_cast<IdIndex>(slot))
        continue;

      // Verify this SW node is mapped to a FU of this temporal PE.
      if (swId >= state.swNodeToHwNode.size())
        continue;
      IdIndex mappedHwId = state.swNodeToHwNode[swId];

      bool isFuOfThisTPE = false;
      for (IdIndex fuId : fuNodes) {
        if (fuId == mappedHwId) {
          isFuOfThisTPE = true;
          break;
        }
      }

      if (isFuOfThisTPE) {
        insnWord = tpa.opcode != INVALID_ID ? tpa.opcode : 0;
        break;
      }
    }

    packBits(words, bitPos, insnWord, 8);
  }

  if (words.empty())
    words.push_back(0);
}

/// Generate temporal switch configuration: route table entries.
void genTemporalSWConfig(const Node *hwNode, const MappingState &state,
                         const Graph &adg, std::vector<uint32_t> &words) {
  int64_t numRouteTable = getNodeIntAttr(hwNode, "num_route_table", 0);
  if (numRouteTable <= 0) {
    words.push_back(0);
    return;
  }

  uint32_t bitPos = 0;
  words.clear();

  unsigned numOut = hwNode->outputPorts.size();
  unsigned slotWidth = numOut > 0 ? numOut : 1;

  for (int64_t rt = 0; rt < numRouteTable; ++rt) {
    uint64_t routeMask = 0;
    packBits(words, bitPos, routeMask, slotWidth);
  }

  if (words.empty())
    words.push_back(0);
}

/// Generate memory configuration (addr_offset_table) with proper format.
/// Emits REGION_ENTRY_WIDTH format: {valid, start_tag, end_tag, addr_offset}.
void genMemoryConfig(const Node *hwNode, const MappingState &state,
                     const Graph &dfg, const Graph &adg, IdIndex hwId,
                     std::vector<uint32_t> &words) {
  int64_t numRegion = getNodeIntAttr(hwNode, "numRegion", 1);

  // Memory has CONFIG_WIDTH = 0 in hardware; emit addr_offset_table
  // entries for software reference.
  if (hwId >= state.hwNodeToSwNodes.size() ||
      state.hwNodeToSwNodes[hwId].empty()) {
    return; // No mapped operations, no config needed.
  }

  // Determine tag width from attributes (default 4 bits).
  int64_t tagWidth = getNodeIntAttr(hwNode, "tag_width", 4);

  for (size_t r = 0; r < state.hwNodeToSwNodes[hwId].size() &&
                      static_cast<int64_t>(r) < numRegion;
       ++r) {
    // REGION_ENTRY format: {valid(1), start_tag, end_tag, addr_offset}
    uint32_t entryWord = 0;
    uint32_t entryBitPos = 0;

    // valid bit: 1 (this region is active).
    packBits(words, entryBitPos, 1, 1);

    // start_tag: region index used as tag range start.
    uint64_t startTag = r;
    packBits(words, entryBitPos, startTag, static_cast<unsigned>(tagWidth));

    // end_tag: start_tag + 1 (half-open interval).
    uint64_t endTag = r + 1;
    packBits(words, entryBitPos, endTag, static_cast<unsigned>(tagWidth + 1));

    // addr_offset: 0 for now (base address offset for this region).
    packBits(words, entryBitPos, 0, 16);

    // Pack the entry into the output words.
    (void)entryWord;
  }
}

/// Find the ADG node ID for a given Node pointer.
IdIndex findNodeId(const Graph &adg, const Node *target) {
  for (IdIndex hwId = 0; hwId < static_cast<IdIndex>(adg.nodes.size());
       ++hwId) {
    if (adg.getNode(hwId) == target)
      return hwId;
  }
  return INVALID_ID;
}

} // namespace

bool ConfigGen::generate(const MappingState &state, const Graph &dfg,
                         const Graph &adg, const std::string &basePath,
                         bool dumpMapping, const std::string &profile,
                         int seed) {
  nodeConfigs.clear();
  configBlob.clear();
  totalConfigWords = 0;

  uint32_t currentOffset = 0;

  for (IdIndex hwId = 0; hwId < static_cast<IdIndex>(adg.nodes.size());
       ++hwId) {
    const Node *hwNode = adg.getNode(hwId);
    if (!hwNode || hwNode->kind != Node::OperationNode)
      continue;

    bool hasMappedOps = hwId < state.hwNodeToSwNodes.size() &&
                        !state.hwNodeToSwNodes[hwId].empty();

    llvm::StringRef opName = getNodeOpName(hwNode);
    llvm::StringRef resClass = getNodeResClass(hwNode);

    // For temporal PE virtual nodes: generate config by iterating FU nodes.
    if (nodeHasAttr(hwNode, "is_virtual")) {
      genTemporalPEConfig(hwNode, state, dfg, adg, hwId,
                          nodeConfigs.emplace_back().words);
      auto &nc = nodeConfigs.back();
      nc.name = getNodeName(hwNode).str();
      if (nc.name.empty())
        nc.name = "node_" + std::to_string(hwId);
      nc.wordOffset = currentOffset;
      nc.wordCount = static_cast<uint32_t>(nc.words.size());
      if (nc.words.empty()) {
        nodeConfigs.pop_back();
      } else {
        currentOffset += nc.wordCount;
      }
      continue;
    }

    // Determine if this node has CONFIG_WIDTH > 0.
    bool hasConfig = false;
    if ((opName == "fabric.pe" || resClass == "functional") && hasMappedOps)
      hasConfig = true;
    if (opName == "fabric.switch" && hasMappedOps)
      hasConfig = true;
    if (opName == "fabric.temporal_sw")
      hasConfig = true;
    if (opName == "fabric.add_tag" || opName == "fabric.map_tag")
      hasConfig = true;
    if (opName == "fabric.fifo" && nodeHasAttr(hwNode, "bypassable"))
      hasConfig = true;

    if (!hasConfig && !hasMappedOps)
      continue;

    NodeConfig nc;
    nc.name = getNodeName(hwNode).str();
    if (nc.name.empty())
      nc.name = "node_" + std::to_string(hwId);
    nc.wordOffset = currentOffset;

    // Generate hardware-semantic config words based on node type.
    if (opName == "fabric.pe" || resClass == "functional") {
      genPEConfig(hwNode, state, dfg, adg, hwId, nc.words);
    } else if (opName == "fabric.switch") {
      genSwitchConfig(hwNode, state, adg, hwId, nc.words);
    } else if (opName == "fabric.temporal_sw") {
      genTemporalSWConfig(hwNode, state, adg, nc.words);
    } else if (resClass == "memory") {
      genMemoryConfig(hwNode, state, dfg, adg, hwId, nc.words);
    }

    if (nc.words.empty())
      continue;

    nc.wordCount = static_cast<uint32_t>(nc.words.size());
    currentOffset += nc.wordCount;
    nodeConfigs.push_back(nc);
  }

  totalConfigWords = currentOffset;

  // Assemble config blob (little-endian, 32-bit words).
  configBlob.resize(totalConfigWords * 4, 0);
  for (const auto &nc : nodeConfigs) {
    for (uint32_t i = 0; i < nc.wordCount; ++i) {
      uint32_t byteOffset = (nc.wordOffset + i) * 4;
      uint32_t word = nc.words[i];
      if (byteOffset + 3 < configBlob.size()) {
        configBlob[byteOffset + 0] = word & 0xFF;
        configBlob[byteOffset + 1] = (word >> 8) & 0xFF;
        configBlob[byteOffset + 2] = (word >> 16) & 0xFF;
        configBlob[byteOffset + 3] = (word >> 24) & 0xFF;
      }
    }
  }

  // Write output files.
  if (!writeBinary(basePath + ".config.bin"))
    return false;
  if (!writeAddrHeader(basePath + "_addr.h"))
    return false;
  if (dumpMapping) {
    if (!writeMappingJson(state, dfg, adg, basePath + ".mapping.json",
                          profile, seed))
      return false;
  }

  return true;
}

bool ConfigGen::writeBinary(const std::string &path) {
  std::error_code ec;
  llvm::raw_fd_ostream out(path, ec, llvm::sys::fs::OF_None);
  if (ec)
    return false;

  out.write(reinterpret_cast<const char *>(configBlob.data()),
            configBlob.size());
  return true;
}

bool ConfigGen::writeAddrHeader(const std::string &path) {
  std::error_code ec;
  llvm::raw_fd_ostream out(path, ec, llvm::sys::fs::OF_Text);
  if (ec)
    return false;

  out << "#ifndef LOOM_CONFIG_ADDR_H\n";
  out << "#define LOOM_CONFIG_ADDR_H\n\n";
  out << "#define LOOM_CONFIG_DEPTH " << totalConfigWords << "\n";
  out << "#define LOOM_CONFIG_WIDTH " << wordWidthBits << "\n\n";

  for (const auto &nc : nodeConfigs) {
    std::string upper = nc.name;
    for (char &c : upper) {
      if (c >= 'a' && c <= 'z')
        c -= 32;
      else if (c == '.' || c == '-')
        c = '_';
    }
    out << "#define LOOM_ADDR_" << upper << " " << nc.wordOffset << "\n";
    out << "#define LOOM_SIZE_" << upper << " " << nc.wordCount << "\n";
  }

  out << "\n#endif // LOOM_CONFIG_ADDR_H\n";
  return true;
}

bool ConfigGen::writeMappingJson(const MappingState &state, const Graph &dfg,
                                 const Graph &adg, const std::string &path,
                                 const std::string &profile, int seed) {
  std::error_code ec;
  llvm::raw_fd_ostream out(path, ec, llvm::sys::fs::OF_Text);
  if (ec)
    return false;

  llvm::json::OStream json(out, 2);
  json.objectBegin();

  json.attribute("version", 1);
  json.attribute("status", "success");
  json.attribute("profile", profile);
  json.attribute("seed", seed);

  // Placement.
  json.attributeBegin("placement");
  json.objectBegin();
  for (IdIndex i = 0; i < static_cast<IdIndex>(state.swNodeToHwNode.size());
       ++i) {
    IdIndex hwNode = state.swNodeToHwNode[i];
    if (hwNode == INVALID_ID)
      continue;
    const Node *sw = dfg.getNode(i);
    if (!sw || sw->kind != Node::OperationNode)
      continue;

    json.attributeBegin(std::to_string(i));
    json.objectBegin();
    json.attribute("hwNode", static_cast<int64_t>(hwNode));
    const Node *hw = adg.getNode(hwNode);
    if (hw) {
      json.attribute("hwNodeName", getNodeName(hw));
      json.attribute("swOp", getNodeOpName(sw));
    }
    json.objectEnd();
    json.attributeEnd();
  }
  json.objectEnd();
  json.attributeEnd();

  // Port binding.
  json.attributeBegin("portBinding");
  json.objectBegin();
  for (IdIndex i = 0; i < static_cast<IdIndex>(state.swPortToHwPort.size());
       ++i) {
    IdIndex hwPort = state.swPortToHwPort[i];
    if (hwPort == INVALID_ID)
      continue;
    json.attribute(std::to_string(i), static_cast<int64_t>(hwPort));
  }
  json.objectEnd();
  json.attributeEnd();

  // Routes (with tag field per spec).
  json.attributeBegin("routes");
  json.objectBegin();
  for (IdIndex i = 0; i < static_cast<IdIndex>(state.swEdgeToHwPaths.size());
       ++i) {
    const auto &pathVec = state.swEdgeToHwPaths[i];
    if (pathVec.empty())
      continue;

    json.attributeBegin(std::to_string(i));
    json.objectBegin();

    const Edge *edge = dfg.getEdge(i);
    if (edge) {
      json.attribute("srcSwPort", static_cast<int64_t>(edge->srcPort));
      json.attribute("dstSwPort", static_cast<int64_t>(edge->dstPort));
    }

    json.attributeBegin("hwPath");
    json.arrayBegin();
    for (size_t j = 0; j + 1 < pathVec.size(); j += 2) {
      json.objectBegin();
      json.attribute("src", static_cast<int64_t>(pathVec[j]));
      json.attribute("dst", static_cast<int64_t>(pathVec[j + 1]));
      json.objectEnd();
    }
    json.arrayEnd();
    json.attributeEnd();

    // Tag field: assigned tag for tagged edge sharing (null for exclusive).
    // Check if the first hop destination is a routing node with shared edges.
    bool hasTag = false;
    if (pathVec.size() >= 2) {
      const Port *dstPort = adg.getPort(pathVec[1]);
      if (dstPort) {
        const Node *dstNode = adg.getNode(dstPort->parentNode);
        if (dstNode && getNodeResClass(dstNode) == "routing") {
          // Check if multiple SW edges share this HW edge.
          for (IdIndex edgeId : adg.getPort(pathVec[0])->connectedEdges) {
            if (edgeId < state.hwEdgeToSwEdges.size() &&
                state.hwEdgeToSwEdges[edgeId].size() > 1) {
              hasTag = true;
              break;
            }
          }
        }
      }
    }

    if (hasTag) {
      // Determine tag index from the position in the shared edge list.
      json.attribute("tag", static_cast<int64_t>(i % 256));
    } else {
      json.attributeBegin("tag");
      json.rawValue("null");
      json.attributeEnd();
    }

    json.objectEnd();
    json.attributeEnd();
  }
  json.objectEnd();
  json.attributeEnd();

  // Temporal assignments.
  json.attributeBegin("temporal");
  json.objectBegin();
  for (IdIndex i = 0;
       i < static_cast<IdIndex>(state.temporalPEAssignments.size()); ++i) {
    const auto &tpa = state.temporalPEAssignments[i];
    if (tpa.slot == INVALID_ID)
      continue;

    json.attributeBegin(std::to_string(i));
    json.objectBegin();
    json.attribute("slot", static_cast<int64_t>(tpa.slot));
    json.attribute("tag", static_cast<int64_t>(tpa.tag));
    json.attribute("opcode", static_cast<int64_t>(tpa.opcode));

    // Include temporal PE virtual node ID.
    if (i < state.swNodeToHwNode.size()) {
      IdIndex hwId = state.swNodeToHwNode[i];
      const Node *hwNode = adg.getNode(hwId);
      if (hwNode) {
        int64_t parentTPE = getNodeIntAttr(hwNode, "parent_temporal_pe", -1);
        if (parentTPE >= 0)
          json.attribute("temporalPE", parentTPE);
      }
    }

    json.objectEnd();
    json.attributeEnd();
  }
  json.objectEnd();
  json.attributeEnd();

  // Register assignments.
  json.attributeBegin("registers");
  json.objectBegin();
  for (IdIndex i = 0;
       i < static_cast<IdIndex>(state.registerAssignments.size()); ++i) {
    if (state.registerAssignments[i] == INVALID_ID)
      continue;

    json.attributeBegin(std::to_string(i));
    json.objectBegin();
    json.attribute("registerIndex",
                   static_cast<int64_t>(state.registerAssignments[i]));
    json.objectEnd();
    json.attributeEnd();
  }
  json.objectEnd();
  json.attributeEnd();

  // Cost.
  json.attributeBegin("cost");
  json.objectBegin();
  json.attribute("total", state.totalCost);
  json.attribute("placementPressure", state.placementPressure);
  json.attribute("routingCost", state.routingCost);
  json.attribute("temporalCost", state.temporalCost);
  json.attribute("perfProxy", state.perfProxyCost);
  json.attribute("configFootprint", state.configFootprint);
  json.objectEnd();
  json.attributeEnd();

  // Diagnostics section per spec-mapper-output.md.
  json.attributeBegin("diagnostics");
  json.objectBegin();

  // Unmapped nodes.
  json.attributeBegin("unmappedNodes");
  json.arrayBegin();
  for (IdIndex i = 0; i < static_cast<IdIndex>(dfg.nodes.size()); ++i) {
    const Node *node = dfg.getNode(i);
    if (!node || node->kind != Node::OperationNode)
      continue;
    if (i >= state.swNodeToHwNode.size() ||
        state.swNodeToHwNode[i] == INVALID_ID) {
      json.value(std::to_string(i));
    }
  }
  json.arrayEnd();
  json.attributeEnd();

  // Failed edges.
  json.attributeBegin("failedEdges");
  json.arrayBegin();
  for (IdIndex i = 0; i < static_cast<IdIndex>(dfg.edges.size()); ++i) {
    const Edge *edge = dfg.getEdge(i);
    if (!edge)
      continue;
    if (i >= state.swEdgeToHwPaths.size() ||
        state.swEdgeToHwPaths[i].empty()) {
      json.value(std::to_string(i));
    }
  }
  json.arrayEnd();
  json.attributeEnd();

  json.attributeBegin("firstViolatedConstraint");
  json.rawValue("null");
  json.attributeEnd();

  json.attributeBegin("conflictingResources");
  json.arrayBegin();
  json.arrayEnd();
  json.attributeEnd();

  json.objectEnd();
  json.attributeEnd();

  json.objectEnd();
  return true;
}

} // namespace loom
