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
/// Returns total bits packed.
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
                 const Graph &dfg, const Graph &adg,
                 std::vector<uint32_t> &words) {
  // Get number of output ports for tag width.
  unsigned numOutputs = hwNode->outputPorts.size();

  // Determine tag width from output port types.
  unsigned tagWidth = 0;
  for (IdIndex outPortId : hwNode->outputPorts) {
    const Port *port = adg.getPort(outPortId);
    if (!port || !port->type)
      continue;
    // Check if tagged type - extract tag width.
    // For now use a default tag width based on attributes.
    break;
  }

  // Check for output_tag attribute.
  for (auto &attr : hwNode->attributes) {
    if (attr.getName() == "output_tag") {
      // Has configurable output tags.
      tagWidth = 4; // Default tag width.
      break;
    }
  }

  uint32_t bitPos = 0;

  // Pack output tags (ascending output index).
  if (tagWidth > 0) {
    for (unsigned o = 0; o < numOutputs; ++o) {
      // Determine tag from temporal assignment or default.
      uint64_t tagVal = 0;
      packBits(words, bitPos, tagVal, tagWidth);
    }
  }

  // Pack compare predicate if present (4 bits per cmp op).
  // Scan mapped SW nodes for compare operations.
  for (IdIndex hwId = 0;
       hwId < static_cast<IdIndex>(state.hwNodeToSwNodes.size()); ++hwId) {
    const Node *hw = adg.getNode(hwId);
    if (hw != hwNode)
      continue;
    for (IdIndex swId : state.hwNodeToSwNodes[hwId]) {
      const Node *swNode = dfg.getNode(swId);
      if (!swNode)
        continue;
      llvm::StringRef swOp;
      for (auto &a : swNode->attributes) {
        if (a.getName() == "op_name") {
          if (auto s = mlir::dyn_cast<mlir::StringAttr>(a.getValue()))
            swOp = s.getValue();
        }
      }
      if (swOp.starts_with("arith.cmp")) {
        uint64_t pred = 0; // Default predicate.
        for (auto &a : swNode->attributes) {
          if (a.getName() == "predicate") {
            if (auto ia = mlir::dyn_cast<mlir::IntegerAttr>(a.getValue()))
              pred = ia.getInt();
          }
        }
        packBits(words, bitPos, pred, 4);
      }
    }
    break;
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

/// Generate switch configuration: route enable bits.
void genSwitchConfig(const Node *hwNode, const MappingState &state,
                     const Graph &adg, std::vector<uint32_t> &words) {
  // For switches, config is the route enable bits.
  // One bit per connected position in connectivity_table.
  unsigned numIn = hwNode->inputPorts.size();
  unsigned numOut = hwNode->outputPorts.size();
  unsigned numBits = numIn * numOut;

  if (numBits == 0) {
    words.push_back(0);
    return;
  }

  uint32_t bitPos = 0;
  words.clear();

  // For each output, determine which input is selected based on routing.
  for (unsigned o = 0; o < numOut; ++o) {
    for (unsigned i = 0; i < numIn; ++i) {
      // Check if any SW edge is routed through this input->output path.
      IdIndex inPortId = hwNode->inputPorts[i];
      IdIndex outPortId = hwNode->outputPorts[o];
      bool enabled = false;

      if (inPortId < state.hwPortToSwPorts.size() &&
          !state.hwPortToSwPorts[inPortId].empty() &&
          outPortId < state.hwPortToSwPorts.size()) {
        // Check if there's a SW edge routed from this input to this output.
        // This is a simplified check; full check would trace paths.
        enabled = !state.hwPortToSwPorts[outPortId].empty();
      }

      packBits(words, bitPos, enabled ? 1 : 0, 1);
    }
  }

  if (words.empty())
    words.push_back(0);
}

/// Generate temporal PE configuration: instruction memory.
void genTemporalPEConfig(const Node *hwNode, const MappingState &state,
                         const Graph &dfg, const Graph &adg,
                         std::vector<uint32_t> &words) {
  int64_t numInst = getNodeIntAttr(hwNode, "num_instruction", 0);
  if (numInst <= 0) {
    words.push_back(0);
    return;
  }

  uint32_t bitPos = 0;
  words.clear();

  // Get the virtual node ID to find mapped operations.
  IdIndex virtualHwId = INVALID_ID;
  for (IdIndex hwId = 0;
       hwId < static_cast<IdIndex>(adg.nodes.size()); ++hwId) {
    if (adg.getNode(hwId) == hwNode) {
      virtualHwId = hwId;
      break;
    }
  }
  if (virtualHwId == INVALID_ID) {
    words.push_back(0);
    return;
  }

  // Pack instruction entries: one per slot.
  for (int64_t slot = 0; slot < numInst; ++slot) {
    uint64_t insnWord = 0;

    // Find the SW node assigned to this slot.
    for (IdIndex swId = 0;
         swId < static_cast<IdIndex>(state.temporalPEAssignments.size());
         ++swId) {
      const auto &tpa = state.temporalPEAssignments[swId];
      if (tpa.slot == static_cast<IdIndex>(slot)) {
        // Check if this SW node is mapped to a FU of this temporal PE.
        IdIndex hwId = state.swNodeToHwNode[swId];
        const Node *hwN = adg.getNode(hwId);
        if (hwN && getNodeIntAttr(hwN, "parent_temporal_pe", -1) ==
                       static_cast<int64_t>(virtualHwId)) {
          insnWord = tpa.opcode != INVALID_ID ? tpa.opcode : 0;
          break;
        }
      }
    }

    // Pack instruction word (width depends on architecture).
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

/// Generate memory configuration (addr_offset_table).
void genMemoryConfig(const Node *hwNode, const MappingState &state,
                     const Graph &dfg, const Graph &adg, IdIndex hwId,
                     std::vector<uint32_t> &words) {
  // Memory modules have CONFIG_WIDTH = 0 per spec.
  // However, we generate addr_offset_table entries for software reference.
  int64_t numRegion = getNodeIntAttr(hwNode, "numRegion", 1);

  if (hwId < state.hwNodeToSwNodes.size()) {
    for (size_t r = 0; r < state.hwNodeToSwNodes[hwId].size() &&
                        static_cast<int64_t>(r) < numRegion;
         ++r) {
      // Each region gets a tag range entry.
      words.push_back(static_cast<uint32_t>(r)); // region index as placeholder
    }
  }

  // Memory has CONFIG_WIDTH = 0 in hardware; no config words needed.
  // But we still track it for the mapping report.
  if (words.empty())
    return; // No config words for memory.
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

    // Skip nodes with no mapped operations and no config.
    bool hasMappedOps = hwId < state.hwNodeToSwNodes.size() &&
                        !state.hwNodeToSwNodes[hwId].empty();

    llvm::StringRef opName = getNodeOpName(hwNode);
    llvm::StringRef resClass = getNodeResClass(hwNode);

    // Skip virtual temporal PE nodes (FU nodes carry config).
    if (nodeHasAttr(hwNode, "is_virtual"))
      continue;

    // Determine if this node has CONFIG_WIDTH > 0.
    bool hasConfig = false;
    if (opName == "fabric.pe" && hasMappedOps)
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
    if (opName == "fabric.pe" || opName == "fabric.pe" ||
        resClass == "functional") {
      genPEConfig(hwNode, state, dfg, adg, nc.words);
    } else if (opName == "fabric.switch") {
      genSwitchConfig(hwNode, state, adg, nc.words);
    } else if (nodeHasAttr(hwNode, "is_virtual") ||
               opName == "fabric.temporal_pe") {
      genTemporalPEConfig(hwNode, state, dfg, adg, nc.words);
    } else if (opName == "fabric.temporal_sw") {
      genTemporalSWConfig(hwNode, state, adg, nc.words);
    } else if (resClass == "memory") {
      genMemoryConfig(hwNode, state, dfg, adg, hwId, nc.words);
    } else if (hasMappedOps) {
      // Fallback: one config word per mapped operation.
      for (IdIndex swId : state.hwNodeToSwNodes[hwId]) {
        uint32_t configWord = swId;
        if (swId < state.temporalPEAssignments.size()) {
          const auto &tpa = state.temporalPEAssignments[swId];
          if (tpa.slot != INVALID_ID)
            configWord |= (tpa.slot << 16);
          if (tpa.opcode != INVALID_ID)
            configWord |= (tpa.opcode << 24);
        }
        nc.words.push_back(configWord);
      }
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

  // Routes.
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
    // Output structured port pairs.
    for (size_t j = 0; j + 1 < pathVec.size(); j += 2) {
      json.objectBegin();
      json.attribute("src", static_cast<int64_t>(pathVec[j]));
      json.attribute("dst", static_cast<int64_t>(pathVec[j + 1]));
      json.objectEnd();
    }
    json.arrayEnd();
    json.attributeEnd();

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

  json.objectEnd();
  return true;
}

} // namespace loom
