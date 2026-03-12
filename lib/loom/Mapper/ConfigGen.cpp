//===-- ConfigGen.cpp - Configuration bitstream generation ---------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include "loom/Mapper/ConfigGen.h"
#include "loom/Dialect/Dataflow/DataflowTypes.h"
#include "loom/Dialect/Fabric/FabricOps.h"
#include "loom/Hardware/Common/FabricConstants.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

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
  if (constVal >= 0)
    packBits(words, bitPos, static_cast<uint64_t>(constVal), 32);
  // Pack cont_cond_sel if present (5 bits for dataflow.stream).
  int64_t contCond = getNodeIntAttr(hwNode, "cont_cond_sel", -1);
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

/// Find the ADG node ID for a given Node pointer.
IdIndex findNodeId(const Graph &adg, const Node *target) {
  for (IdIndex hwId = 0; hwId < static_cast<IdIndex>(adg.nodes.size());
       ++hwId) {
    if (adg.getNode(hwId) == target)
      return hwId;
  }
  return INVALID_ID;
}

/// Look up an operation by sym_name attribute, searching from the outermost
/// parent module. Used to resolve fabric.instance targets.
mlir::Operation *lookupSymbolInModule(mlir::Operation *from,
                                      llvm::StringRef name) {
  mlir::Operation *scope = from;
  while (scope->getParentOp())
    scope = scope->getParentOp();

  mlir::Operation *result = nullptr;
  scope->walk([&](mlir::Operation *op) -> mlir::WalkResult {
    if (auto attr = op->getAttrOfType<mlir::StringAttr>("sym_name")) {
      if (attr.getValue() == name) {
        result = op;
        return mlir::WalkResult::interrupt();
      }
    }
    return mlir::WalkResult::advance();
  });
  return result;
}

/// Build instruction_mem ArrayAttr for a temporal PE instance.
/// Returns empty ArrayAttr if no entries are generated.
mlir::ArrayAttr buildInstructionMem(
    const Node *hwNode, IdIndex hwId, const MappingState &state,
    const Graph &dfg, const Graph &adg,
    fabric::InstanceOp instanceOp, mlir::Builder &builder) {
  int64_t numInst = getNodeIntAttr(hwNode, "num_instruction", 0);
  int64_t numReg = getNodeIntAttr(hwNode, "num_register", 0);
  if (numInst <= 0)
    return {};

  (void)numReg; // Register encoding handled inline below.

  // Collect FU sub-node IDs in body order (ascending node ID == body order,
  // since ADGFlattener creates FU nodes sequentially in body order).
  llvm::SmallVector<IdIndex, 4> fuNodeIds;
  for (IdIndex nodeId = 0;
       nodeId < static_cast<IdIndex>(adg.nodes.size()); ++nodeId) {
    const Node *node = adg.getNode(nodeId);
    if (!node || node->kind != Node::OperationNode)
      continue;
    int64_t parentTPE = getNodeIntAttr(node, "parent_temporal_pe", -1);
    if (parentTPE == static_cast<int64_t>(hwId))
      fuNodeIds.push_back(nodeId);
  }

  // Build fuNodeId -> fuIndex map.
  llvm::DenseMap<IdIndex, unsigned> fuNodeToIndex;
  for (unsigned i = 0; i < fuNodeIds.size(); ++i)
    fuNodeToIndex[fuNodeIds[i]] = i;

  // Get FU names from the TemporalPEOp definition body.
  llvm::SmallVector<std::string, 4> fuNames;
  mlir::Operation *target =
      lookupSymbolInModule(instanceOp.getOperation(), instanceOp.getModule());
  if (auto tpeOp = mlir::dyn_cast_or_null<fabric::TemporalPEOp>(target)) {
    for (auto &innerOp : tpeOp.getBody().front()) {
      if (innerOp.hasTrait<mlir::OpTrait::IsTerminator>())
        continue;
      if (auto symName =
              innerOp.getAttrOfType<mlir::StringAttr>("sym_name"))
        fuNames.push_back(symName.getValue().str());
      else
        fuNames.push_back("fu_" + std::to_string(fuNames.size()));
    }
  }

  // Get port counts from the virtual node.
  unsigned numOutputs = hwNode->outputPorts.size();
  unsigned numInputs = hwNode->inputPorts.size();

  // Build instruction entries (sparse format: only populated slots).
  llvm::SmallVector<mlir::Attribute, 4> entries;
  for (int64_t slot = 0; slot < numInst; ++slot) {
    // Find SW node assigned to this slot on a FU of this TPE.
    for (IdIndex swId = 0;
         swId < static_cast<IdIndex>(state.temporalPEAssignments.size());
         ++swId) {
      const auto &tpa = state.temporalPEAssignments[swId];
      if (tpa.slot != static_cast<IdIndex>(slot))
        continue;

      // Verify this SW node is mapped to a FU of this temporal PE.
      if (swId >= state.swNodeToHwNode.size())
        continue;
      IdIndex mappedFuId = state.swNodeToHwNode[swId];
      auto fuIdxIt = fuNodeToIndex.find(mappedFuId);
      if (fuIdxIt == fuNodeToIndex.end())
        continue;

      // Derive correct opcode from FU body position (not tpa.opcode).
      unsigned opcode = fuIdxIt->second;
      int64_t tag = tpa.tag != INVALID_ID
          ? static_cast<int64_t>(tpa.tag) : 0;

      std::string fuName = opcode < fuNames.size()
          ? fuNames[opcode]
          : "fu_" + std::to_string(opcode);

      // Format: inst[S]: when(tag=T) dest, ... = fuName(O) src, ...
      // dest: out(idx) or reg(idx), src: in(idx) or reg(idx)
      std::string entry = "inst[" + std::to_string(slot) + "]: when(tag=" +
                          std::to_string(tag) + ") ";

      // Get DFG node for register-aware encoding.
      const Node *swNode = dfg.getNode(swId);

      // Destinations: check if result edges use registers.
      bool firstDest = true;
      if (swNode) {
        for (unsigned o = 0; o < swNode->outputPorts.size(); ++o) {
          IdIndex swPortId = swNode->outputPorts[o];
          const Port *swPort = dfg.getPort(swPortId);
          if (!swPort)
            continue;

          bool hasRegDest = false;
          bool hasOutDest = false;
          IdIndex regIdx = INVALID_ID;

          for (IdIndex edgeId : swPort->connectedEdges) {
            const Edge *edge = dfg.getEdge(edgeId);
            if (!edge || edge->srcPort != swPortId)
              continue;
            if (edgeId < state.registerAssignments.size() &&
                state.registerAssignments[edgeId] != INVALID_ID) {
              hasRegDest = true;
              regIdx = state.registerAssignments[edgeId];
            } else {
              hasOutDest = true;
            }
          }

          if (hasRegDest) {
            if (!firstDest)
              entry += ", ";
            entry += "reg(" + std::to_string(regIdx) + ")";
            firstDest = false;
          }
          if (hasOutDest) {
            if (!firstDest)
              entry += ", ";
            entry += "out(" + std::to_string(o) + ")";
            firstDest = false;
          }
          if (!hasRegDest && !hasOutDest) {
            if (!firstDest)
              entry += ", ";
            entry += "out(" + std::to_string(o) + ")";
            firstDest = false;
          }
        }
      } else {
        for (unsigned o = 0; o < numOutputs; ++o) {
          if (!firstDest)
            entry += ", ";
          entry += "out(" + std::to_string(o) + ")";
          firstDest = false;
        }
      }

      entry += " = " + fuName + "(" + std::to_string(opcode) + ") ";

      // Sources: check if operand edges use registers.
      bool firstSrc = true;
      if (swNode) {
        for (unsigned i = 0; i < swNode->inputPorts.size(); ++i) {
          IdIndex swPortId = swNode->inputPorts[i];
          const Port *swPort = dfg.getPort(swPortId);
          if (!swPort)
            continue;

          bool isReg = false;
          IdIndex regIdx = INVALID_ID;

          for (IdIndex edgeId : swPort->connectedEdges) {
            const Edge *edge = dfg.getEdge(edgeId);
            if (!edge || edge->dstPort != swPortId)
              continue;
            if (edgeId < state.registerAssignments.size() &&
                state.registerAssignments[edgeId] != INVALID_ID) {
              isReg = true;
              regIdx = state.registerAssignments[edgeId];
              break;
            }
          }

          if (!firstSrc)
            entry += ", ";
          if (isReg) {
            entry += "reg(" + std::to_string(regIdx) + ")";
          } else {
            // Find HW input port index via port mapping.
            unsigned hwInIdx = i;
            if (swPortId < state.swPortToHwPort.size()) {
              IdIndex hwPortId = state.swPortToHwPort[swPortId];
              for (unsigned hi = 0; hi < hwNode->inputPorts.size(); ++hi) {
                if (hwNode->inputPorts[hi] == hwPortId) {
                  hwInIdx = hi;
                  break;
                }
              }
            }
            entry += "in(" + std::to_string(hwInIdx) + ")";
          }
          firstSrc = false;
        }
      } else {
        for (unsigned i = 0; i < numInputs; ++i) {
          if (!firstSrc)
            entry += ", ";
          entry += "in(" + std::to_string(i) + ")";
          firstSrc = false;
        }
      }

      entries.push_back(builder.getStringAttr(entry));
      break;
    }
  }

  if (entries.empty())
    return {};
  return builder.getArrayAttr(entries);
}

/// Set output_tag attribute on a tagged PE instance.
/// For non-temporal tagged PEs, reads from taggedPEOutputTags (populated
/// during placement). For temporal PEs, derives from temporalPEAssignments.
void buildPEOutputTag(const Node *hwNode, IdIndex hwId,
                      const MappingState &state,
                      fabric::InstanceOp instanceOp,
                      mlir::Builder &builder) {
  if (!nodeHasAttr(hwNode, "output_tag"))
    return;

  unsigned numOutputs = hwNode->outputPorts.size();
  if (numOutputs == 0)
    return;

  // Non-temporal tagged PE: read from explicit tag assignment.
  uint64_t tagVal = 0;
  bool hasAssignment = false;
  auto tagIt = state.taggedPEOutputTags.find(hwId);
  if (tagIt != state.taggedPEOutputTags.end()) {
    tagVal = tagIt->second;
    hasAssignment = true;
  } else if (hwId < state.hwNodeToSwNodes.size() &&
             !state.hwNodeToSwNodes[hwId].empty()) {
    // Temporal PE: derive from temporal assignment.
    IdIndex swId = state.hwNodeToSwNodes[hwId][0];
    if (swId < state.temporalPEAssignments.size()) {
      const auto &tpa = state.temporalPEAssignments[swId];
      if (tpa.tag != INVALID_ID) {
        tagVal = tpa.tag;
        hasAssignment = true;
      }
    }
  }

  // Only emit output_tag on PEs that have a recorded assignment.
  if (!hasAssignment)
    return;

  llvm::SmallVector<mlir::Attribute, 4> tagAttrs;
  auto i64Type = builder.getIntegerType(64);
  for (unsigned o = 0; o < numOutputs; ++o)
    tagAttrs.push_back(builder.getIntegerAttr(i64Type, tagVal));

  instanceOp->setAttr("output_tag", builder.getArrayAttr(tagAttrs));
}

/// Set compare_predicate attribute on a PE instance when the mapped SW
/// node is a comparison operation.
void buildComparePredicate(const Node *hwNode, IdIndex hwId,
                           const MappingState &state, const Graph &dfg,
                           fabric::InstanceOp instanceOp,
                           mlir::Builder &builder) {
  if (hwId >= state.hwNodeToSwNodes.size())
    return;

  for (IdIndex swId : state.hwNodeToSwNodes[hwId]) {
    const Node *swNode = dfg.getNode(swId);
    if (!swNode)
      continue;
    llvm::StringRef swOp = getNodeOpName(swNode);
    if (swOp.starts_with("arith.cmp")) {
      for (auto &a : swNode->attributes) {
        if (a.getName() == "predicate") {
          if (auto ia = mlir::dyn_cast<mlir::IntegerAttr>(a.getValue())) {
            instanceOp->setAttr("compare_predicate", ia);
            return;
          }
        }
      }
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

bool ConfigGen::writeMapJson(const MappingState &state, const Graph &dfg,
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
      // Use actual assigned per-edge tag from MappingState temporal
      // routing state. Fall back to source node's temporal PE tag.
      int64_t tagVal = 0;
      if (i < state.temporalSWAssignments.size() &&
          !state.temporalSWAssignments[i].empty()) {
        const auto &tswa = state.temporalSWAssignments[i];
        if (tswa[0].tag != INVALID_ID)
          tagVal = static_cast<int64_t>(tswa[0].tag);
      } else if (edge) {
        const Port *srcPort = dfg.getPort(edge->srcPort);
        if (srcPort && srcPort->parentNode != INVALID_ID &&
            srcPort->parentNode < state.temporalPEAssignments.size()) {
          const auto &tpa =
              state.temporalPEAssignments[srcPort->parentNode];
          if (tpa.tag != INVALID_ID)
            tagVal = static_cast<int64_t>(tpa.tag);
        }
      }
      json.attribute("tag", tagVal);
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

    // temporalPE: the HW virtual node ID of the temporal PE where the
    // register is located (per spec-mapper-output.md).
    const Edge *regEdge = dfg.getEdge(i);
    if (regEdge) {
      const Port *srcPort = dfg.getPort(regEdge->srcPort);
      if (srcPort && srcPort->parentNode != INVALID_ID &&
          srcPort->parentNode < state.swNodeToHwNode.size()) {
        IdIndex hwId = state.swNodeToHwNode[srcPort->parentNode];
        const Node *hwNode = adg.getNode(hwId);
        if (hwNode) {
          int64_t parentTPE =
              getNodeIntAttr(hwNode, "parent_temporal_pe", -1);
          if (parentTPE >= 0)
            json.attribute("temporalPE", parentTPE);
        }
      }
    }

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

// ---------------------------------------------------------------------------
// writeMapText -- human-readable mapping report (.map.txt)
// ---------------------------------------------------------------------------

namespace {

/// Get a display name for a DFG (software) node.
std::string swNodeDisplayName(const Node *node, IdIndex id) {
  if (node->kind == Node::ModuleInputNode) {
    int64_t idx = getNodeIntAttr(node, "arg_index", -1);
    return "ModuleInput(arg" + std::to_string(idx >= 0 ? idx : id) + ")";
  }
  if (node->kind == Node::ModuleOutputNode) {
    int64_t idx = getNodeIntAttr(node, "ret_index", -1);
    return "ModuleOutput(ret" + std::to_string(idx >= 0 ? idx : id) + ")";
  }
  llvm::StringRef opName = getNodeOpName(node);
  return opName.empty() ? ("node_" + std::to_string(id)) : opName.str();
}

/// Get a display name for an ADG (hardware) node.
std::string hwNodeDisplayName(const Node *node, IdIndex id) {
  if (node->kind == Node::ModuleInputNode) {
    int64_t idx = getNodeIntAttr(node, "arg_index", -1);
    return "in" + std::to_string(idx >= 0 ? idx : id);
  }
  if (node->kind == Node::ModuleOutputNode) {
    int64_t idx = getNodeIntAttr(node, "ret_index", -1);
    return "out" + std::to_string(idx >= 0 ? idx : id);
  }
  llvm::StringRef symName = getNodeName(node);
  if (!symName.empty())
    return symName.str();
  return "node_" + std::to_string(id);
}

/// Get a description string for an ADG node (op_name, resource_class).
std::string hwNodeDesc(const Node *node) {
  if (node->kind == Node::ModuleInputNode)
    return "ModuleInput";
  if (node->kind == Node::ModuleOutputNode)
    return "ModuleOutput";
  llvm::StringRef opName = getNodeOpName(node);
  llvm::StringRef resClass = getNodeResClass(node);
  std::string desc;
  if (!opName.empty())
    desc = opName.str();
  if (!resClass.empty()) {
    if (!desc.empty())
      desc += ", ";
    desc += resClass.str();
  }
  return desc;
}

/// Get the "loc" string attribute from a node, or empty string.
llvm::StringRef getNodeLocStr(const Node *node) {
  for (auto &attr : node->attributes) {
    if (attr.getName() == "loc") {
      if (auto strAttr = mlir::dyn_cast<mlir::StringAttr>(attr.getValue()))
        return strAttr.getValue();
    }
  }
  return "";
}

/// Format a port reference as "nodeName:in0" or "nodeName:out0".
std::string formatPortRef(const Graph &graph, IdIndex portId,
                          bool isHwGraph) {
  const Port *port = graph.getPort(portId);
  if (!port)
    return "port_" + std::to_string(portId);

  const Node *node = graph.getNode(port->parentNode);
  if (!node)
    return "port_" + std::to_string(portId);

  std::string nodeName = isHwGraph
      ? hwNodeDisplayName(node, port->parentNode)
      : swNodeDisplayName(node, port->parentNode);

  // Find port index within the node's input or output port list.
  const auto &portList = (port->direction == Port::Input)
      ? node->inputPorts : node->outputPorts;
  unsigned portIdx = 0;
  for (unsigned i = 0; i < portList.size(); ++i) {
    if (portList[i] == portId) {
      portIdx = i;
      break;
    }
  }

  std::string dir = (port->direction == Port::Input) ? "in" : "out";
  return nodeName + ":" + dir + std::to_string(portIdx);
}

/// Check if a DFG edge is internal to an operation group (both endpoints
/// mapped to the same HW node AND both are members of a tech-mapped group).
/// Such edges are handled inside the PE and do not need switch-fabric
/// routing. Inter-slot edges on temporal FU sub-nodes are NOT group-internal.
bool isInternalGroupEdge(const Edge *edge, const Graph &dfg,
                         const MappingState &state) {
  const Port *srcPort = dfg.getPort(edge->srcPort);
  const Port *dstPort = dfg.getPort(edge->dstPort);
  if (!srcPort || !dstPort)
    return false;
  IdIndex srcSwNode = srcPort->parentNode;
  IdIndex dstSwNode = dstPort->parentNode;
  if (srcSwNode == INVALID_ID || dstSwNode == INVALID_ID)
    return false;
  if (srcSwNode >= state.swNodeToHwNode.size() ||
      dstSwNode >= state.swNodeToHwNode.size())
    return false;
  IdIndex srcHwNode = state.swNodeToHwNode[srcSwNode];
  IdIndex dstHwNode = state.swNodeToHwNode[dstSwNode];
  if (srcHwNode == INVALID_ID || srcHwNode != dstHwNode)
    return false;
  // Both mapped to same HW node: only truly internal if both are in
  // the same group binding (tech-mapped group).
  auto groupIt = state.groupBindings.find(srcHwNode);
  if (groupIt == state.groupBindings.end())
    return false;
  bool srcInGroup = false, dstInGroup = false;
  for (IdIndex gid : groupIt->second) {
    if (gid == srcSwNode) srcInGroup = true;
    if (gid == dstSwNode) dstInGroup = true;
  }
  return srcInGroup && dstInGroup;
}

} // namespace

bool ConfigGen::writeMapText(const MappingState &state, const Graph &dfg,
                              const Graph &adg, const std::string &path) {
  std::error_code ec;
  llvm::raw_fd_ostream out(path, ec, llvm::sys::fs::OF_Text);
  if (ec)
    return false;

  // --- Header ---
  out << "=== Mapping Report ===\n\n";

  // --- SW Node List ---
  unsigned swNodeCount = 0;
  for (IdIndex i = 0; i < static_cast<IdIndex>(dfg.nodes.size()); ++i) {
    if (dfg.getNode(i))
      ++swNodeCount;
  }
  out << "--- SW Node List (" << swNodeCount << " nodes) ---\n";
  for (IdIndex i = 0; i < static_cast<IdIndex>(dfg.nodes.size()); ++i) {
    const Node *node = dfg.getNode(i);
    if (!node)
      continue;
    out << "  [N" << i << "] " << swNodeDisplayName(node, i);
    llvm::StringRef loc = getNodeLocStr(node);
    if (!loc.empty())
      out << "  " << loc;
    out << "\n";
  }
  out << "\n";

  // --- SW Edge List ---
  unsigned swEdgeCount = 0;
  for (IdIndex i = 0; i < static_cast<IdIndex>(dfg.edges.size()); ++i) {
    if (dfg.getEdge(i))
      ++swEdgeCount;
  }
  out << "--- SW Edge List (" << swEdgeCount << " edges) ---\n";
  for (IdIndex i = 0; i < static_cast<IdIndex>(dfg.edges.size()); ++i) {
    const Edge *edge = dfg.getEdge(i);
    if (!edge)
      continue;
    out << "  [E" << i << "] "
        << formatPortRef(dfg, edge->srcPort, false)
        << " -> "
        << formatPortRef(dfg, edge->dstPort, false);
    // Use destination node's location for the edge.
    const Port *dstPort = dfg.getPort(edge->dstPort);
    if (dstPort) {
      const Node *dstNode = dfg.getNode(dstPort->parentNode);
      if (dstNode) {
        llvm::StringRef loc = getNodeLocStr(dstNode);
        if (!loc.empty())
          out << "  " << loc;
      }
    }
    out << "\n";
  }
  out << "\n";

  // --- HW Node List ---
  unsigned hwNodeCount = 0;
  for (IdIndex i = 0; i < static_cast<IdIndex>(adg.nodes.size()); ++i) {
    if (adg.getNode(i))
      ++hwNodeCount;
  }
  out << "--- HW Node List (" << hwNodeCount << " nodes) ---\n";
  for (IdIndex i = 0; i < static_cast<IdIndex>(adg.nodes.size()); ++i) {
    const Node *node = adg.getNode(i);
    if (!node)
      continue;
    out << "  [H" << i << "] " << hwNodeDisplayName(node, i);
    std::string desc = hwNodeDesc(node);
    if (!desc.empty())
      out << " (" << desc << ")";
    llvm::StringRef loc = getNodeLocStr(node);
    if (!loc.empty())
      out << "  " << loc;
    out << "\n";
  }
  out << "\n";

  // --- Node Mapping ---
  unsigned mappedNodeCount = 0;
  for (IdIndex i = 0; i < static_cast<IdIndex>(state.swNodeToHwNode.size());
       ++i) {
    if (state.swNodeToHwNode[i] != INVALID_ID && dfg.getNode(i))
      ++mappedNodeCount;
  }
  out << "--- Node Mapping (" << mappedNodeCount << " mapped) ---\n";
  for (IdIndex i = 0; i < static_cast<IdIndex>(state.swNodeToHwNode.size());
       ++i) {
    IdIndex hwId = state.swNodeToHwNode[i];
    const Node *swNode = dfg.getNode(i);
    if (!swNode)
      continue;
    if (hwId == INVALID_ID) {
      out << "  [N" << i << "] " << swNodeDisplayName(swNode, i)
          << " -> UNMAPPED\n";
      continue;
    }
    const Node *hwNode = adg.getNode(hwId);
    out << "  [N" << i << "] " << swNodeDisplayName(swNode, i)
        << " -> [H" << hwId << "] "
        << (hwNode ? hwNodeDisplayName(hwNode, hwId) : "???") << "\n";
  }
  out << "\n";

  // --- Edge Routing ---
  unsigned routedEdgeCount = 0;
  unsigned internalEdgeCount = 0;
  for (IdIndex i = 0; i < static_cast<IdIndex>(dfg.edges.size()); ++i) {
    const Edge *edge = dfg.getEdge(i);
    if (!edge)
      continue;
    if (isInternalGroupEdge(edge, dfg, state))
      ++internalEdgeCount;
    else if (i < state.swEdgeToHwPaths.size() &&
             !state.swEdgeToHwPaths[i].empty())
      ++routedEdgeCount;
  }
  out << "--- Edge Routing (" << routedEdgeCount << " routed";
  if (internalEdgeCount > 0)
    out << ", " << internalEdgeCount << " internal";
  out << ") ---\n";
  for (IdIndex i = 0; i < static_cast<IdIndex>(dfg.edges.size()); ++i) {
    const Edge *edge = dfg.getEdge(i);
    if (!edge)
      continue;

    out << "  [E" << i << "] "
        << formatPortRef(dfg, edge->srcPort, false)
        << " -> "
        << formatPortRef(dfg, edge->dstPort, false) << "\n";

    // Internal group edges are handled inside the PE, no routing needed.
    if (isInternalGroupEdge(edge, dfg, state)) {
      out << "    Route: INTERNAL (within PE group)\n\n";
      continue;
    }

    if (i >= state.swEdgeToHwPaths.size() ||
        state.swEdgeToHwPaths[i].empty()) {
      out << "    Route: UNROUTED\n\n";
      continue;
    }

    const auto &hwPath = state.swEdgeToHwPaths[i];
    out << "    Route: ";
    for (size_t j = 0; j < hwPath.size(); ++j) {
      if (j > 0)
        out << " -> ";
      out << formatPortRef(adg, hwPath[j], true);
    }
    out << "\n";
    out << "    Hops: " << (hwPath.size() / 2) << "\n";
    out << "\n";
  }

  // --- Unmapped ---
  out << "--- Unmapped ---\n";
  bool hasUnmapped = false;

  for (IdIndex i = 0; i < static_cast<IdIndex>(dfg.nodes.size()); ++i) {
    const Node *node = dfg.getNode(i);
    if (!node)
      continue;
    if (i >= state.swNodeToHwNode.size() ||
        state.swNodeToHwNode[i] == INVALID_ID) {
      if (!hasUnmapped) {
        out << "  Nodes:\n";
        hasUnmapped = true;
      }
      out << "    [N" << i << "] " << swNodeDisplayName(node, i) << "\n";
    }
  }

  bool hasUnroutedEdges = false;
  for (IdIndex i = 0; i < static_cast<IdIndex>(dfg.edges.size()); ++i) {
    const Edge *edge = dfg.getEdge(i);
    if (!edge)
      continue;
    // Skip internal group edges (handled inside PE, not a routing failure).
    if (isInternalGroupEdge(edge, dfg, state))
      continue;
    if (i >= state.swEdgeToHwPaths.size() ||
        state.swEdgeToHwPaths[i].empty()) {
      if (!hasUnroutedEdges) {
        out << "  Edges:\n";
        hasUnroutedEdges = true;
        hasUnmapped = true;
      }
      out << "    [E" << i << "] "
          << formatPortRef(dfg, edge->srcPort, false) << " -> "
          << formatPortRef(dfg, edge->dstPort, false) << "\n";
    }
  }

  if (!hasUnmapped)
    out << "  (none)\n";
  out << "\n";

  // --- Cost ---
  out << "--- Cost ---\n";
  out << llvm::format("  total           = %.4f\n", state.totalCost);
  out << llvm::format("  placement       = %.4f\n", state.placementPressure);
  out << llvm::format("  routing         = %.4f\n", state.routingCost);
  out << llvm::format("  temporal        = %.4f\n", state.temporalCost);
  out << llvm::format("  perfProxy       = %.4f\n", state.perfProxyCost);
  out << llvm::format("  configFootprint = %.4f\n", state.configFootprint);

  return true;
}

bool ConfigGen::writeConfiguredFabric(
    const MappingState &state, const Graph &dfg, const Graph &adg,
    const llvm::DenseMap<mlir::Operation *, IdIndex> &opMap,
    mlir::Operation *adgModule, const std::string &path) {
  mlir::Builder builder(adgModule->getContext());

  // Walk switch ops and set route_table.
  adgModule->walk([&](fabric::SwitchOp switchOp) {
    mlir::Operation *op = switchOp.getOperation();
    auto it = opMap.find(op);
    if (it == opMap.end())
      return;
    IdIndex hwId = it->second;

    const Node *hwNode = adg.getNode(hwId);
    if (!hwNode)
      return;

    unsigned numIn = hwNode->inputPorts.size();
    unsigned numOut = hwNode->outputPorts.size();
    if (numIn == 0 || numOut == 0)
      return;

    // Build active (output, input) transitions from routed paths.
    llvm::DenseSet<uint64_t> activeTransitions;
    for (const auto &pathVec : state.swEdgeToHwPaths) {
      if (pathVec.empty())
        continue;
      for (size_t j = 1; j + 1 < pathVec.size(); j += 2) {
        IdIndex inPortId = pathVec[j];
        IdIndex outPortId = pathVec[j + 1];
        const Port *inPort = adg.getPort(inPortId);
        if (!inPort || inPort->parentNode != hwId)
          continue;
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

    // Get connectivity_table from the MLIR op attribute.
    auto connTable = switchOp.getConnectivityTableAttr();

    // Build compressed route_table: only entries where connectivity == 1.
    llvm::SmallVector<int8_t> routeTable;
    for (unsigned o = 0; o < numOut; ++o) {
      for (unsigned i = 0; i < numIn; ++i) {
        unsigned idx = o * numIn + i;
        bool connected = true;
        if (connTable && !connTable.empty()) {
          connected =
              idx < static_cast<unsigned>(connTable.size()) && connTable[idx];
        }
        if (!connected)
          continue;
        uint64_t key = (static_cast<uint64_t>(o) << 32) | i;
        routeTable.push_back(activeTransitions.count(key) ? 1 : 0);
      }
    }

    switchOp.setRouteTableAttr(
        mlir::DenseI8ArrayAttr::get(builder.getContext(), routeTable));
  });

  // Walk temporal switch ops and set route_table from temporal SW assignments.
  adgModule->walk([&](fabric::TemporalSwOp tswOp) {
    auto it = opMap.find(tswOp.getOperation());
    if (it == opMap.end())
      return;
    IdIndex hwId = it->second;

    const Node *hwNode = adg.getNode(hwId);
    if (!hwNode)
      return;

    int64_t numRouteTable = getNodeIntAttr(hwNode, "num_route_table", 0);
    if (numRouteTable <= 0)
      return;

    unsigned numIn = hwNode->inputPorts.size();
    unsigned numOut = hwNode->outputPorts.size();
    if (numOut == 0)
      return;

    // Look up temporal SW assignments for this node.
    const llvm::SmallVector<TemporalSWAssignment, 4> *assignments = nullptr;
    if (hwId < state.temporalSWAssignments.size() &&
        !state.temporalSWAssignments[hwId].empty()) {
      assignments = &state.temporalSWAssignments[hwId];
    }

    // Build human-readable route_table entries.
    llvm::SmallVector<mlir::Attribute, 4> routeEntries;
    for (int64_t rt = 0; rt < numRouteTable; ++rt) {
      uint64_t routeMask = 0;
      int64_t tagVal = -1;
      if (assignments) {
        for (const auto &tswa : *assignments) {
          if (tswa.slot == static_cast<IdIndex>(rt)) {
            routeMask = tswa.routeMask;
            if (tswa.tag != INVALID_ID)
              tagVal = static_cast<int64_t>(tswa.tag);
            break;
          }
        }
      }

      // Format: route_table[N]: when(tag=T) O[o]<-I[i], ...
      std::string entry = "route_table[" + std::to_string(rt) + "]: ";
      if (tagVal < 0 && routeMask == 0) {
        entry += "invalid";
      } else {
        if (tagVal >= 0)
          entry += "when(tag=" + std::to_string(tagVal) + ") ";

        // Decode routeMask (connected-position bits) back to (output, input).
        // Get connectivity_table for proper decoding.
        auto connTable = tswOp.getConnectivityTableAttr();
        bool firstRoute = true;
        unsigned posIdx = 0;
        for (unsigned o = 0; o < numOut; ++o) {
          for (unsigned i = 0; i < numIn; ++i) {
            bool connected = true;
            unsigned flatIdx = o * numIn + i;
            if (connTable && !connTable.empty()) {
              connected = flatIdx < static_cast<unsigned>(connTable.size()) &&
                          connTable[flatIdx];
            }
            if (!connected)
              continue;
            if (routeMask & (1ULL << posIdx)) {
              if (!firstRoute)
                entry += ", ";
              entry += "O[" + std::to_string(o) + "]<-I[" +
                       std::to_string(i) + "]";
              firstRoute = false;
            }
            ++posIdx;
          }
        }
        if (firstRoute)
          entry += "(idle)";
      }

      routeEntries.push_back(builder.getStringAttr(entry));
    }

    if (!routeEntries.empty())
      tswOp->setAttr("route_table", builder.getArrayAttr(routeEntries));
  });

  // Walk instance ops and set runtime config attributes.
  adgModule->walk([&](fabric::InstanceOp instanceOp) {
    auto it = opMap.find(instanceOp.getOperation());
    if (it == opMap.end())
      return;
    IdIndex hwId = it->second;
    const Node *hwNode = adg.getNode(hwId);
    if (!hwNode)
      return;

    // Temporal PE instance: set instruction_mem.
    if (nodeHasAttr(hwNode, "is_virtual")) {
      auto instrMem = buildInstructionMem(hwNode, hwId, state, dfg, adg,
                                          instanceOp, builder);
      if (instrMem && !instrMem.empty())
        instanceOp->setAttr("instruction_mem", instrMem);
      return;
    }

    buildPEOutputTag(hwNode, hwId, state, instanceOp, builder);
    buildComparePredicate(hwNode, hwId, state, dfg, instanceOp, builder);
  });

  // Walk memory ops and set addr_offset_table MLIR attribute.
  auto setMemoryAddrOffset = [&](mlir::Operation *op) {
    auto it = opMap.find(op);
    if (it == opMap.end())
      return;
    IdIndex hwId = it->second;
    const Node *hwNode = adg.getNode(hwId);
    if (!hwNode)
      return;

    bool hasMapped = (hwId < state.hwNodeToSwNodes.size() &&
                      !state.hwNodeToSwNodes[hwId].empty());
    int64_t numRegion = getNodeIntAttr(hwNode, "numRegion", 1);
    size_t mappedCount = hasMapped ? state.hwNodeToSwNodes[hwId].size() : 0;
    size_t regionCount =
        std::min(mappedCount, static_cast<size_t>(numRegion));
    int64_t ldCount = getNodeIntAttr(hwNode, "ldCount", 1);
    int64_t stCount = getNodeIntAttr(hwNode, "stCount", 1);
    bool isBridgeMemory = (ldCount > 1 || stCount > 1);
    int64_t tagCount = std::max(ldCount, stCount);

    // Bridge: each active region uses [0, tagCount); non-bridge: per-mapping.
    size_t effectiveRegionCount =
        isBridgeMemory ? 1 : regionCount;

    llvm::SmallVector<int64_t> table; // [valid, start, end, base] * numRegion
    for (size_t r = 0; r < static_cast<size_t>(numRegion); ++r) {
      if (r < effectiveRegionCount) {
        table.push_back(1);
        if (isBridgeMemory) {
          // Bridge lane tags span [0, tagCount) for every active region.
          table.push_back(0); table.push_back(tagCount);
        } else {
          table.push_back(static_cast<int64_t>(r));
          table.push_back(static_cast<int64_t>(r + 1));
        }
        table.push_back(0);
      } else {
        table.append({0, 0, 0, 0});
      }
    }

    op->setAttr("addrOffsetTable", builder.getDenseI64ArrayAttr(table));
    // Emit config node ID so external tools can correlate with _addr.h.
    op->setAttr("loom.configNodeId",
                builder.getI64IntegerAttr(static_cast<int64_t>(hwId)));
  };
  adgModule->walk(
      [&](fabric::MemoryOp memOp) { setMemoryAddrOffset(memOp); });
  adgModule->walk(
      [&](fabric::ExtMemoryOp memOp) { setMemoryAddrOffset(memOp); });

  // Walk add_tag ops and set mapper-assigned tag values.
  adgModule->walk([&](fabric::AddTagOp addTagOp) {
    auto it = opMap.find(addTagOp.getOperation());
    if (it == opMap.end())
      return;
    IdIndex hwId = it->second;
    const Node *hwNode = adg.getNode(hwId);
    if (!hwNode)
      return;

    // Check if this is a bridge add_tag with a direct lane index.
    for (auto &attr : hwNode->attributes) {
      if (attr.getName() == "bridge_lane_index") {
        if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(attr.getValue())) {
          auto resultType = mlir::dyn_cast<loom::dataflow::TaggedType>(
              addTagOp.getResult().getType());
          if (resultType) {
            auto tagType = resultType.getTagType();
            addTagOp.setTagAttr(
                mlir::IntegerAttr::get(tagType, intAttr.getInt()));
          }
          return; // Bridge add_tag handled.
        }
      }
    }

    // Collect all port IDs owned by this add_tag node.
    llvm::DenseSet<IdIndex> nodePortIds;
    for (IdIndex pid : hwNode->inputPorts)
      nodePortIds.insert(pid);
    for (IdIndex pid : hwNode->outputPorts)
      nodePortIds.insert(pid);

    // Scan routed SW edges to find one whose HW path goes through this node.
    for (IdIndex edgeId = 0;
         edgeId < static_cast<IdIndex>(state.swEdgeToHwPaths.size());
         ++edgeId) {
      const auto &pathVec = state.swEdgeToHwPaths[edgeId];
      bool usesNode = false;
      for (IdIndex portId : pathVec) {
        if (nodePortIds.count(portId)) {
          usesNode = true;
          break;
        }
      }
      if (!usesNode)
        continue;

      // Found an edge routed through this add_tag.
      // Get the destination SW node's temporal PE assignment tag.
      const Edge *swEdge = dfg.getEdge(edgeId);
      if (!swEdge)
        continue;
      const Port *dstPort = dfg.getPort(swEdge->dstPort);
      if (!dstPort || dstPort->parentNode == INVALID_ID)
        continue;
      IdIndex dstSwNode = dstPort->parentNode;

      if (dstSwNode < state.temporalPEAssignments.size()) {
        const auto &tpa = state.temporalPEAssignments[dstSwNode];
        if (tpa.tag != INVALID_ID) {
          auto resultType = mlir::dyn_cast<loom::dataflow::TaggedType>(
              addTagOp.getResult().getType());
          if (resultType) {
            auto tagType = resultType.getTagType();
            addTagOp.setTagAttr(
                mlir::IntegerAttr::get(tagType, tpa.tag));
          }
          break;
        }
      }
    }
  });

  // Write the modified module to disk.
  std::error_code ec;
  llvm::raw_fd_ostream out(path, ec, llvm::sys::fs::OF_Text);
  if (ec)
    return false;
  adgModule->print(out);
  out << "\n";
  return true;
}
} // namespace loom

