//===-- ConfigGen.cpp - Configuration bitstream generation ---------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include "loom/Mapper/ConfigGen.h"
#include "loom/Mapper/ConfigGenUtil.h"
#include "loom/Dialect/Dataflow/DataflowTypes.h"
#include "loom/Hardware/Common/FabricConstants.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

namespace loom {

using namespace configgen;

namespace {

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
  // First check hwNode for constant_value (ADG-level attribute).
  // Fallback: extract from the mapped DFG node's "value" attribute
  // (handshake.constant stores its payload as {value = N : iW}).
  bool foundConst = false;
  uint64_t constBits = 0;
  unsigned constWidth = static_cast<unsigned>(
      getNodeIntAttr(hwNode, "data_width", 32));
  if (nodeHasAttr(hwNode, "constant_value")) {
    constBits = static_cast<uint64_t>(
        getNodeIntAttr(hwNode, "constant_value", 0));
    foundConst = true;
  } else if (hwId < state.hwNodeToSwNodes.size()) {
    for (IdIndex swId : state.hwNodeToSwNodes[hwId]) {
      const Node *swNode = dfg.getNode(swId);
      if (!swNode)
        continue;
      llvm::StringRef swOp = getNodeOpName(swNode);
      if (swOp != "handshake.constant")
        continue;
      for (auto &a : swNode->attributes) {
        if (a.getName() == "value") {
          if (auto ia = mlir::dyn_cast<mlir::IntegerAttr>(a.getValue())) {
            constBits = static_cast<uint64_t>(ia.getInt());
            foundConst = true;
          } else if (auto fa =
                         mlir::dyn_cast<mlir::FloatAttr>(a.getValue())) {
            // Float constant: store raw bit pattern.
            auto apf = fa.getValue();
            constBits = apf.bitcastToAPInt().getZExtValue();
            foundConst = true;
          }
        }
      }
      if (foundConst)
        break;
    }
  }
  if (foundConst)
    packBits(words, bitPos, constBits, constWidth);
  // Pack cont_cond_sel if present (5 bits for dataflow.stream).
  int64_t contCond = getNodeIntAttr(hwNode, "cont_cond_sel", -1);
  // Fallback: infer cont_cond_sel from the mapped DFG node's stop_cond.
  if (contCond < 0 && hwId < state.hwNodeToSwNodes.size()) {
    for (IdIndex swId : state.hwNodeToSwNodes[hwId]) {
      const Node *swNode = dfg.getNode(swId);
      if (!swNode)
        continue;
      for (auto &a : swNode->attributes) {
        if (a.getName() == "stop_cond") {
          if (auto sa = mlir::dyn_cast<mlir::StringAttr>(a.getValue())) {
            llvm::StringRef sc = sa.getValue();
            if (sc == "slt" || sc == "<") contCond = 0x01;
            else if (sc == "sle" || sc == "<=") contCond = 0x02;
            else if (sc == "sgt" || sc == ">") contCond = 0x04;
            else if (sc == "sge" || sc == ">=") contCond = 0x08;
            else if (sc == "ne" || sc == "!=") contCond = 0x10;
          }
        }
      }
      if (contCond >= 0)
        break;
    }
  }
  if (contCond >= 0)
    packBits(words, bitPos, static_cast<uint64_t>(contCond), 5);
  // If no config bits were packed, leave words empty so caller skips node.
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
  // by routed SW edges. Scan all routed edges for transitions through this node.
  llvm::DenseSet<uint64_t> activeTransitions;
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

  // Pack route enable bits: per spec-fabric-switch.md, one bit per connected
  // position in connectivity_table, scanned in row-major order (output, input).
  // Only positions where connectivity_table[o*numIn+i] == 1 get a bit.
  mlir::DenseI8ArrayAttr connTable;
  for (auto &attr : hwNode->attributes) {
    if (attr.getName() == "connectivity_table") {
      connTable = mlir::dyn_cast<mlir::DenseI8ArrayAttr>(attr.getValue());
      break;
    }
  }

  for (unsigned o = 0; o < numOut; ++o) {
    for (unsigned i = 0; i < numIn; ++i) {
      // Skip unconnected positions.
      if (connTable &&
          static_cast<unsigned>(connTable.size()) == numOut * numIn &&
          connTable[o * numIn + i] == 0)
        continue;
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

  // Collect FU nodes belonging to this temporal PE.
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
/// Per spec: each slot is valid(1) | tag(M) | routes(K) bits,
/// where M = tag width, K = number of connected positions in
/// connectivity_table.
void genTemporalSWConfig(const Node *hwNode, const MappingState &state,
                         const Graph &adg, IdIndex hwId,
                         std::vector<uint32_t> &words) {
  int64_t numRouteTable = getNodeIntAttr(hwNode, "num_route_table", 0);
  if (numRouteTable <= 0) {
    words.push_back(0);
    return;
  }

  uint32_t bitPos = 0;
  words.clear();

  // Determine tag width from port types.
  unsigned tagWidth = 4;
  for (IdIndex portId : hwNode->inputPorts) {
    const Port *port = adg.getPort(portId);
    if (!port)
      continue;
    if (auto taggedType =
            mlir::dyn_cast<loom::dataflow::TaggedType>(port->type)) {
      tagWidth = taggedType.getTagType().getWidth();
      break;
    }
  }

  // Count connected positions (K) from connectivity_table.
  unsigned numIn = hwNode->inputPorts.size(), numOut = hwNode->outputPorts.size();
  unsigned K = numOut * numIn;
  mlir::DenseI8ArrayAttr connTable;
  for (auto &attr : hwNode->attributes) {
    if (attr.getName() == "connectivity_table") {
      connTable = mlir::dyn_cast<mlir::DenseI8ArrayAttr>(attr.getValue());
      break;
    }
  }
  if (connTable && static_cast<unsigned>(connTable.size()) == numOut * numIn) {
    K = 0;
    for (int64_t i = 0; i < connTable.size(); ++i) {
      if (connTable[i] != 0)
        ++K;
    }
  }

  // Per spec: slot_width = 1 (valid) + M (tag) + K (routes)
  unsigned slotWidth = 1 + tagWidth + K;

  // Look up per-slot route masks from temporal SW assignments.
  const llvm::SmallVector<TemporalSWAssignment, 4> *assignments = nullptr;
  if (hwId < state.temporalSWAssignments.size() &&
      !state.temporalSWAssignments[hwId].empty()) {
    assignments = &state.temporalSWAssignments[hwId];
  }

  for (int64_t rt = 0; rt < numRouteTable; ++rt) {
    uint64_t routeMask = 0;
    uint64_t tagVal = 0;
    uint64_t valid = 0;
    if (assignments) {
      for (const auto &tswa : *assignments) {
        if (tswa.slot == static_cast<IdIndex>(rt)) {
          routeMask = tswa.routeMask;
          tagVal = tswa.tag != INVALID_ID ? tswa.tag : 0;
          valid = 1;
          break;
        }
      }
    }
    // Pack: valid(1) | tag(M) | routes(K), LSB first
    uint64_t slotWord = (routeMask << (1 + tagWidth)) |
                        (tagVal << 1) | valid;
    packBits(words, bitPos, slotWord, slotWidth);
  }

  if (words.empty())
    words.push_back(0);
}

/// Generate memory config matching RTL layout (low-to-high per region:
/// addr_offset[ADDR_BIT_WIDTH], end_tag[tw+1], start_tag[tw], valid[1]).
void genMemoryConfig(const Node *hwNode, const MappingState &state,
                     const Graph &dfg, const Graph &adg, IdIndex hwId,
                     std::vector<uint32_t> &words) {
  int64_t numRegion = getNodeIntAttr(hwNode, "numRegion", 1);
  int64_t ldCount = getNodeIntAttr(hwNode, "ldCount", 1);
  int64_t stCount = getNodeIntAttr(hwNode, "stCount", 1);
  bool isBridge = (ldCount > 1 || stCount > 1);
  int64_t tagCount = std::max(ldCount, stCount);

  unsigned tw = 0;
  if (isBridge) { // clog2(max(ldCount, stCount))
    unsigned mc = static_cast<unsigned>(tagCount);
    tw = 1; while ((1u << tw) < mc) ++tw;
  }

  if (numRegion == 0)
    return;
  bool hasMapped = (hwId < state.hwNodeToSwNodes.size() &&
                    !state.hwNodeToSwNodes[hwId].empty());
  size_t mappedCount = hasMapped ? state.hwNodeToSwNodes[hwId].size() : 0;
  // Bridge: each active region uses [0, tagCount) (tags=lanes).
  size_t activeCount = isBridge
      ? 1 : std::min(mappedCount, static_cast<size_t>(numRegion));

  unsigned bitsPerRegion = ADDR_BIT_WIDTH + 1 + (tw > 0 ? tw + (tw + 1) : 0);
  uint32_t bitPos = 0;
  for (size_t r = 0; r < static_cast<size_t>(numRegion); ++r) {
    if (r < activeCount) {
      packBits(words, bitPos, 0, ADDR_BIT_WIDTH); // addr_offset
      if (tw > 0) {
        uint64_t endTag, startTag;
        if (isBridge) { // Bridge: [0, tagCount) for every active region.
          startTag = 0; endTag = static_cast<uint64_t>(tagCount);
        } else {
          startTag = static_cast<uint64_t>(r);
          endTag = static_cast<uint64_t>(r + 1);
        }
        packBits(words, bitPos, endTag, tw + 1);  // end_tag
        packBits(words, bitPos, startTag, tw);     // start_tag
      }
      packBits(words, bitPos, 1, 1); // valid
    } else {
      // Inactive region: pack fields individually (avoids UB for >64-bit).
      packBits(words, bitPos, 0, ADDR_BIT_WIDTH);
      if (tw > 0) { packBits(words, bitPos, 0, tw + 1); packBits(words, bitPos, 0, tw); }
      packBits(words, bitPos, 0, 1); // valid=0
    }
  }
}

} // namespace

bool ConfigGen::generate(const MappingState &state, const Graph &dfg,
                         const Graph &adg, const std::string &basePath,
                         const std::string &profile, int seed) {
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
    if (opName == "fabric.switch" || opName == "fabric.temporal_sw" ||
        opName == "fabric.add_tag" || opName == "fabric.map_tag")
      hasConfig = true;
    if (opName == "fabric.fifo" && nodeHasAttr(hwNode, "bypassable"))
      hasConfig = true;
    // Unmapped bridge memory needs a default addr_offset_table config
    // so that bridge tags remain valid even when unused.
    if (resClass == "memory" && !hasMappedOps) {
      int64_t lc = getNodeIntAttr(hwNode, "ldCount", 1);
      int64_t sc = getNodeIntAttr(hwNode, "stCount", 1);
      if (lc > 1 || sc > 1)
        hasConfig = true;
    }

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
      genTemporalSWConfig(hwNode, state, adg, hwId, nc.words);
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

  // Populate public config slice metadata for simulator consumption.
  configSlices_.clear();
  configSlices_.reserve(nodeConfigs.size());
  for (const auto &nc : nodeConfigs) {
    ConfigSlice slice;
    slice.name = nc.name;
    slice.wordOffset = nc.wordOffset;
    slice.wordCount = nc.wordCount;
    configSlices_.push_back(slice);
  }

  // Write output files.
  if (!writeBinary(basePath + ".config.bin"))
    return false;
  if (!writeAddrHeader(basePath + "_addr.h"))
    return false;
  if (!writeMapJson(state, dfg, adg, basePath + ".map.json", profile, seed))
    return false;
  if (!writeMapText(state, dfg, adg, basePath + ".map.txt"))
    return false;

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

} // namespace loom
