// ConfigGenConfig.cpp -- Config builder functions and ConfigGen methods
// for binary/JSON/header generation.
// This file is compiled as part of the ConfigGen translation unit.

#include "ConfigGenInternal.h"
#include "fcc/Mapper/OpCompat.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

namespace fcc {
namespace configgen_detail {

namespace {

bool isCommutativeTechMappedOp(llvm::StringRef opName) {
  return opName == "arith.addi" || opName == "arith.addf" ||
         opName == "arith.muli" || opName == "arith.mulf" ||
         opName == "arith.andi" || opName == "arith.ori" ||
         opName == "arith.xori";
}

} // namespace

// ---------------------------------------------------------------------------
// Switch config builders
// ---------------------------------------------------------------------------

GeneratedNodeConfig buildSpatialSwitchConfig(const Node *hwNode, IdIndex hwId,
                                             const MappingState &state,
                                             const Graph &dfg,
                                             const Graph &adg) {
  GeneratedNodeConfig cfg;
  if (!hwNode)
    return cfg;
  unsigned numIn = hwNode->inputPorts.size();
  unsigned numOut = hwNode->outputPorts.size();
  if (numIn == 0 || numOut == 0)
    return cfg;

  auto rows = getBinaryRowsNodeAttr(hwNode, "connectivity_table");
  uint32_t bitPos = 0;
  std::set<std::pair<int, int>> activeTransitions;

  for (IdIndex edgeId = 0;
       edgeId < static_cast<IdIndex>(state.swEdgeToHwPaths.size()); ++edgeId) {
    auto hwPath = buildExportPathForEdge(edgeId, state, dfg, adg);
    if (hwPath.size() < 3)
      continue;
    for (size_t i = 1; i + 1 < hwPath.size(); i += 2) {
      IdIndex inPortId = hwPath[i];
      IdIndex outPortId = hwPath[i + 1];
      const Port *inPort = adg.getPort(inPortId);
      const Port *outPort = adg.getPort(outPortId);
      if (!inPort || !outPort || inPort->parentNode != hwId ||
          outPort->parentNode != hwId)
        continue;
      int inputIdx = findNodeInputIndex(hwNode, inPortId);
      int outputIdx = findNodeOutputIndex(hwNode, outPortId);
      if (inputIdx >= 0 && outputIdx >= 0)
        activeTransitions.insert({inputIdx, outputIdx});
    }
  }

  for (unsigned outIdx = 0; outIdx < numOut; ++outIdx) {
    for (unsigned inIdx = 0; inIdx < numIn; ++inIdx) {
      if (!isConnectedPosition(rows, outIdx, inIdx, numIn))
        continue;
      bool enabled =
          activeTransitions.count({static_cast<int>(inIdx),
                                   static_cast<int>(outIdx)}) > 0;
      packBits(cfg.words, bitPos, enabled ? 1u : 0u, 1);
    }
  }

  return cfg;
}

GeneratedNodeConfig buildTemporalSwitchConfig(const Node *hwNode, IdIndex hwId,
                                              const MappingState &state,
                                              const Graph &dfg,
                                              const Graph &adg) {
  GeneratedNodeConfig cfg;
  if (!hwNode)
    return cfg;

  unsigned numIn = hwNode->inputPorts.size();
  unsigned numOut = hwNode->outputPorts.size();
  unsigned slotCount =
      static_cast<unsigned>(std::max<int64_t>(getNodeAttrInt(hwNode, "num_route_table", 0), 0));
  if (numIn == 0 || numOut == 0 || slotCount == 0)
    return cfg;

  auto rows = getBinaryRowsNodeAttr(hwNode, "connectivity_table");
  unsigned routeBits = countConnectedPositions(rows, numIn, numOut);
  // Hardware tag width is a native parameter of the switch interface and comes
  // directly from the tagged port type, not from any runtime tag-value scan.
  unsigned tagWidth = 0;
  for (IdIndex portId : hwNode->inputPorts) {
    const Port *port = adg.getPort(portId);
    if (!port)
      continue;
    if (auto info = detail::getPortTypeInfo(port->type)) {
      if (info->isTagged) {
        tagWidth = info->tagWidth;
        break;
      }
    }
  }
  if (tagWidth == 0)
    tagWidth = 1;

  std::map<uint64_t, std::vector<unsigned>> routesByTag;
  for (IdIndex edgeId = 0;
       edgeId < static_cast<IdIndex>(state.swEdgeToHwPaths.size()); ++edgeId) {
    auto hwPath = buildExportPathForEdge(edgeId, state, dfg, adg);
    if (hwPath.size() < 3)
      continue;
    for (size_t i = 1; i + 1 < hwPath.size(); i += 2) {
      IdIndex inPortId = hwPath[i];
      IdIndex outPortId = hwPath[i + 1];
      const Port *inPort = adg.getPort(inPortId);
      const Port *outPort = adg.getPort(outPortId);
      if (!inPort || !outPort || inPort->parentNode != hwId ||
          outPort->parentNode != hwId)
        continue;

      int inputIdx = findNodeInputIndex(hwNode, inPortId);
      int outputIdx = findNodeOutputIndex(hwNode, outPortId);
      if (inputIdx < 0 || outputIdx < 0)
        continue;

      auto tag =
          computeTemporalRouteTagValue(edgeId, hwPath, i, state, dfg, adg);
      if (!tag) {
        cfg.complete = false;
        continue;
      }

      unsigned ordinal =
          connectedPositionOrdinal(rows, numIn, numOut, inputIdx, outputIdx);
      auto &routeBitsForTag = routesByTag[*tag];
      if (std::find(routeBitsForTag.begin(), routeBitsForTag.end(), ordinal) ==
          routeBitsForTag.end()) {
        routeBitsForTag.push_back(ordinal);
      }
    }
  }

  uint32_t bitPos = 0;
  unsigned emittedSlots = 0;
  for (const auto &entry : routesByTag) {
    if (emittedSlots >= slotCount) {
      cfg.complete = false;
      break;
    }
    packBits(cfg.words, bitPos, 1u, 1);
    packBits(cfg.words, bitPos, entry.first, tagWidth);
    for (unsigned ordinal = 0; ordinal < routeBits; ++ordinal) {
      bool enabled = std::find(entry.second.begin(), entry.second.end(),
                               ordinal) != entry.second.end();
      packBits(cfg.words, bitPos, enabled ? 1u : 0u, 1);
    }
    ++emittedSlots;
  }
  for (; emittedSlots < slotCount; ++emittedSlots) {
    packBits(cfg.words, bitPos, 0u, 1);
    packBits(cfg.words, bitPos, 0u, tagWidth);
    for (unsigned ordinal = 0; ordinal < routeBits; ++ordinal)
      packBits(cfg.words, bitPos, 0u, 1);
  }

  return cfg;
}

// ---------------------------------------------------------------------------
// Routing config builders
// ---------------------------------------------------------------------------

GeneratedNodeConfig buildAddTagConfig(const Node *hwNode) {
  GeneratedNodeConfig cfg;
  if (auto tag = getUIntNodeAttr(hwNode, "tag"))
    cfg.words.push_back(static_cast<uint32_t>(*tag));
  return cfg;
}

GeneratedNodeConfig buildMapTagConfig(const Node *hwNode, const Graph &adg) {
  GeneratedNodeConfig cfg;
  if (!hwNode)
    return cfg;
  unsigned tableSize =
      static_cast<unsigned>(getNodeAttrInt(hwNode, "table_size", 0));
  if (tableSize == 0)
    return cfg;

  auto [inputTagWidth, outputTagWidth] = getMapTagTagWidths(hwNode, adg);
  std::vector<uint32_t> words;
  uint32_t bitPos = 0;
  auto entries = getMapTagTableEntries(hwNode);
  if (entries.size() != tableSize)
    cfg.complete = false;
  for (unsigned idx = 0; idx < tableSize; ++idx) {
    MapTagTableEntry entry;
    if (idx < entries.size())
      entry = entries[idx];
    packBits(words, bitPos, entry.valid ? 1u : 0u, 1);
    packBits(words, bitPos, entry.srcTag, inputTagWidth);
    packBits(words, bitPos, entry.dstTag, outputTagWidth);
  }
  cfg.words = std::move(words);
  return cfg;
}

GeneratedNodeConfig buildMemoryConfig(const Node *hwNode, IdIndex hwId,
                                      const MappingState &state,
                                      const Graph &dfg, const Graph &adg) {
  GeneratedNodeConfig cfg;
  if (!hwNode)
    return cfg;
  auto addrTable = buildAddrOffsetTable(hwNode, hwId, state, dfg, adg);
  cfg.words.reserve(addrTable.size());
  for (int64_t value : addrTable)
    cfg.words.push_back(static_cast<uint32_t>(value));
  return cfg;
}

GeneratedNodeConfig buildFifoConfig(const Node *hwNode, IdIndex hwId,
                                    const MappingState &state) {
  GeneratedNodeConfig cfg;
  if (!hwNode)
    return cfg;
  bool bypassable = false;
  for (const auto &attr : hwNode->attributes) {
    if (attr.getName() == "bypassable") {
      if (mlir::isa<mlir::BoolAttr>(attr.getValue()))
        bypassable = mlir::cast<mlir::BoolAttr>(attr.getValue()).getValue();
    }
  }
  if (!bypassable)
    return cfg;
  bool bypassed = getEffectiveFifoBypassed(hwNode, hwId, state);
  cfg.words.push_back(bypassed ? 1u : 0u);
  return cfg;
}

// ---------------------------------------------------------------------------
// FU config helpers
// ---------------------------------------------------------------------------

const FUConfigSelection *
findFUConfigSelection(llvm::ArrayRef<FUConfigSelection> fuConfigs,
                      IdIndex hwNodeId) {
  for (const auto &selection : fuConfigs) {
    if (selection.hwNodeId == hwNodeId)
      return &selection;
  }
  return nullptr;
}

const Edge *findEdgeByPorts(const Graph &graph, IdIndex srcPortId,
                            IdIndex dstPortId) {
  const Port *srcPort = graph.getPort(srcPortId);
  if (!srcPort)
    return nullptr;
  for (IdIndex edgeId : srcPort->connectedEdges) {
    const Edge *edge = graph.getEdge(edgeId);
    if (edge && edge->srcPort == srcPortId && edge->dstPort == dstPortId)
      return edge;
  }
  return nullptr;
}

llvm::SmallVector<unsigned, 4> getFUConfigFieldWidths(const Node *hwNode) {
  llvm::SmallVector<unsigned, 4> widths;
  auto attr = getDenseI64NodeAttr(hwNode, "fu_config_field_widths");
  if (!attr)
    return widths;
  for (int64_t width : attr.asArrayRef()) {
    if (width > 0)
      widths.push_back(static_cast<unsigned>(width));
  }
  return widths;
}

unsigned getFUConfigBitWidth(const Node *hwNode) {
  return static_cast<unsigned>(
      std::max<int64_t>(0, getNodeAttrInt(hwNode, "fu_config_bits", 0)));
}

void packFUConfigBits(std::vector<uint32_t> &words, uint32_t &bitPos,
                      const Node *hwNode,
                      const FUConfigSelection *selection, bool &complete) {
  auto widths = getFUConfigFieldWidths(hwNode);
  if (widths.empty())
    return;

  size_t fieldCount = widths.size();
  size_t selectedCount = selection ? selection->fields.size() : 0;
  if (selection && selectedCount != fieldCount)
    complete = false;

  for (size_t i = 0; i < fieldCount; ++i) {
    unsigned width = widths[i];
    if (selection && i < selectedCount) {
      const auto &field = selection->fields[i];
      if (field.kind == FUConfigFieldKind::Mux) {
        unsigned selBits = width >= 2 ? width - 2 : 0;
        packMuxField(words, bitPos, selBits, field.sel, field.discard,
                     field.disconnect);
      } else {
        packBits(words, bitPos, field.value, width);
      }
      continue;
    }
    packBits(words, bitPos, 0u, width);
  }
}

// ---------------------------------------------------------------------------
// PE config planning
// ---------------------------------------------------------------------------

TemporalConfigPlan
buildTemporalConfigPlan(const PEContainment &pe, const MappingState &state,
                        const Graph &dfg, const Graph &adg,
                        llvm::ArrayRef<TechMappedEdgeKind> edgeKinds) {
  TemporalConfigPlan plan;

  for (unsigned ordinal = 0; ordinal < pe.fuNodeIds.size(); ++ordinal) {
    IdIndex fuId = pe.fuNodeIds[ordinal];
    const Node *fuNode = adg.getNode(fuId);
    if (!fuNode)
      continue;
    plan.operandRegsByFU[fuId].assign(fuNode->inputPorts.size(), std::nullopt);
    plan.resultRegsByFU[fuId].assign(fuNode->outputPorts.size(), std::nullopt);
    if (fuId < state.hwNodeToSwNodes.size() && !state.hwNodeToSwNodes[fuId].empty()) {
      plan.slotByFU[fuId] = static_cast<unsigned>(plan.usedFUs.size());
      plan.usedFUs.push_back({ordinal, fuId});
    }
  }

  llvm::DenseMap<IdIndex, unsigned> regBySrcPort;
  for (IdIndex edgeId = 0; edgeId < static_cast<IdIndex>(dfg.edges.size());
       ++edgeId) {
    if (edgeId >= edgeKinds.size() ||
        edgeKinds[edgeId] != TechMappedEdgeKind::TemporalReg)
      continue;
    const Edge *edge = dfg.getEdge(edgeId);
    if (!edge)
      continue;
    const Port *srcPort = dfg.getPort(edge->srcPort);
    const Port *dstPort = dfg.getPort(edge->dstPort);
    if (!srcPort || !dstPort || srcPort->parentNode == INVALID_ID ||
        dstPort->parentNode == INVALID_ID)
      continue;

    IdIndex writerSwNode = srcPort->parentNode;
    IdIndex readerSwNode = dstPort->parentNode;
    if (writerSwNode >= state.swNodeToHwNode.size() ||
        readerSwNode >= state.swNodeToHwNode.size())
      continue;
    IdIndex writerHwNode = state.swNodeToHwNode[writerSwNode];
    IdIndex readerHwNode = state.swNodeToHwNode[readerSwNode];
    if (writerHwNode == INVALID_ID || readerHwNode == INVALID_ID)
      continue;
    const Node *writerHw = adg.getNode(writerHwNode);
    const Node *readerHw = adg.getNode(readerHwNode);
    if (!writerHw || !readerHw)
      continue;
    llvm::StringRef writerPE = getNodeAttrStr(writerHw, "pe_name");
    if (writerPE.empty() || writerPE != pe.peName ||
        writerPE != getNodeAttrStr(readerHw, "pe_name"))
      continue;

    int writerOutputIdx = findNodeOutputIndex(dfg.getNode(writerSwNode), edge->srcPort);
    int readerInputIdx = findNodeInputIndex(dfg.getNode(readerSwNode), edge->dstPort);
    if (writerOutputIdx < 0 || readerInputIdx < 0) {
      plan.complete = false;
      continue;
    }

    unsigned regIdx = 0;
    auto foundReg = regBySrcPort.find(edge->srcPort);
    if (foundReg == regBySrcPort.end()) {
      regIdx = static_cast<unsigned>(regBySrcPort.size());
      regBySrcPort[edge->srcPort] = regIdx;
    } else {
      regIdx = foundReg->second;
    }

    if (regIdx >= pe.numRegister) {
      plan.complete = false;
      continue;
    }

    auto &resultRegs = plan.resultRegsByFU[writerHwNode];
    auto &operandRegs = plan.operandRegsByFU[readerHwNode];
    if (static_cast<size_t>(writerOutputIdx) >= resultRegs.size() ||
        static_cast<size_t>(readerInputIdx) >= operandRegs.size()) {
      plan.complete = false;
      continue;
    }
    if (resultRegs[writerOutputIdx].has_value() &&
        *resultRegs[writerOutputIdx] != regIdx)
      plan.complete = false;
    if (operandRegs[readerInputIdx].has_value() &&
        *operandRegs[readerInputIdx] != regIdx)
      plan.complete = false;

    resultRegs[writerOutputIdx] = regIdx;
    operandRegs[readerInputIdx] = regIdx;

    TemporalRegisterBinding binding;
    binding.swEdgeId = edgeId;
    binding.writerSwNode = writerSwNode;
    binding.readerSwNode = readerSwNode;
    binding.writerHwNode = writerHwNode;
    binding.readerHwNode = readerHwNode;
    binding.srcSwPort = edge->srcPort;
    binding.writerOutputIndex = static_cast<unsigned>(writerOutputIdx);
    binding.readerInputIndex = static_cast<unsigned>(readerInputIdx);
    binding.registerIndex = regIdx;
    binding.peName = pe.peName;
    plan.registerBindings.push_back(std::move(binding));
  }

  return plan;
}

PERouteSummary collectPERouteSummary(const PEContainment &pe,
                                     const MappingState &state,
                                     const Graph &dfg, const Graph &adg) {
  PERouteSummary summary;
  llvm::DenseSet<IdIndex> fuSet(pe.fuNodeIds.begin(), pe.fuNodeIds.end());
  llvm::DenseMap<IdIndex, llvm::DenseMap<int, int>> usedPeOutputsByFU;
  auto findEdgesByPorts = [&](IdIndex srcPortId,
                              IdIndex dstPortId) -> llvm::SmallVector<const Edge *, 4> {
    llvm::SmallVector<const Edge *, 4> matches;
    const Port *srcPort = adg.getPort(srcPortId);
    if (!srcPort)
      return matches;
    for (IdIndex edgeId : srcPort->connectedEdges) {
      const Edge *edge = adg.getEdge(edgeId);
      if (edge && edge->srcPort == srcPortId && edge->dstPort == dstPortId)
        matches.push_back(edge);
    }
    return matches;
  };
  auto chooseBestFlatEdge = [&](IdIndex srcPortId, IdIndex dstPortId,
                                IdIndex srcNodeId, int outputIdx,
                                IdIndex dstNodeId, int inputIdx)
      -> const Edge * {
    auto candidates = findEdgesByPorts(srcPortId, dstPortId);
    if (candidates.empty())
      return nullptr;
    if (candidates.size() == 1)
      return candidates.front();

    auto scoreCandidate = [&](const Edge *edge) {
      int score = 0;
      auto peInputIndex = getUIntEdgeAttr(edge, "pe_input_index");
      auto peOutputIndex = getUIntEdgeAttr(edge, "pe_output_index");

      if (dstNodeId != INVALID_ID && inputIdx >= 0 && peInputIndex) {
        auto it = summary.inputPortSelects.find(dstNodeId);
        if (it != summary.inputPortSelects.end() &&
            static_cast<size_t>(inputIdx) < it->second.size() &&
            it->second[inputIdx] >= 0 &&
            it->second[inputIdx] != static_cast<int>(*peInputIndex)) {
          score += 100;
        }
      }
      if (srcNodeId != INVALID_ID && outputIdx >= 0 && peOutputIndex) {
        auto it = summary.outputPortSelects.find(srcNodeId);
        if (it != summary.outputPortSelects.end() &&
            static_cast<size_t>(outputIdx) < it->second.size() &&
            it->second[outputIdx] >= 0 &&
            it->second[outputIdx] != static_cast<int>(*peOutputIndex)) {
          score += 100;
        }
        auto usedIt = usedPeOutputsByFU.find(srcNodeId);
        if (usedIt != usedPeOutputsByFU.end()) {
          auto existing = usedIt->second.find(static_cast<int>(*peOutputIndex));
          if (existing != usedIt->second.end() && existing->second != outputIdx)
            score += 100;
        }
      }
      if (!peInputIndex)
        score += 1;
      if (!peOutputIndex)
        score += 1;
      return score;
    };

    const Edge *best = candidates.front();
    int bestScore = scoreCandidate(best);
    for (size_t candidateIdx = 1; candidateIdx < candidates.size();
         ++candidateIdx) {
      const Edge *candidate = candidates[candidateIdx];
      int score = scoreCandidate(candidate);
      if (score < bestScore) {
        best = candidate;
        bestScore = score;
      }
    }
    return best;
  };

  for (IdIndex fuId : pe.fuNodeIds) {
    const Node *fuNode = adg.getNode(fuId);
    if (!fuNode)
      continue;
    summary.inputPortSelects[fuId].assign(fuNode->inputPorts.size(), -1);
    summary.outputPortSelects[fuId].assign(fuNode->outputPorts.size(), -1);
  }

  for (IdIndex swEdgeId = 0;
       swEdgeId < static_cast<IdIndex>(state.swEdgeToHwPaths.size());
       ++swEdgeId) {
    auto hwPath = buildExportPathForEdge(swEdgeId, state, dfg, adg);
    if (hwPath.size() < 2)
      continue;

    for (size_t i = 0; i + 1 < hwPath.size(); ++i) {
      IdIndex srcPortId = hwPath[i];
      IdIndex dstPortId = hwPath[i + 1];
      const Port *srcPort = adg.getPort(srcPortId);
      const Port *dstPort = adg.getPort(dstPortId);
      if (!srcPort || !dstPort)
        continue;

      IdIndex srcNodeId = srcPort->parentNode;
      IdIndex dstNodeId = dstPort->parentNode;
      int outputIdx =
          srcNodeId != INVALID_ID ? findNodeOutputIndex(adg.getNode(srcNodeId), srcPortId)
                                  : -1;
      int inputIdx =
          dstNodeId != INVALID_ID ? findNodeInputIndex(adg.getNode(dstNodeId), dstPortId)
                                  : -1;
      const Edge *flatEdge =
          chooseBestFlatEdge(srcPortId, dstPortId, srcNodeId, outputIdx,
                             dstNodeId, inputIdx);
      if (!flatEdge)
        continue;

      if (dstNodeId != INVALID_ID && fuSet.contains(dstNodeId)) {
        auto peInputIndex = getUIntEdgeAttr(flatEdge, "pe_input_index");
        if (peInputIndex) {
          const Node *fuNode = adg.getNode(dstNodeId);
          if (fuNode && inputIdx >= 0) {
            auto &slots = summary.inputPortSelects[dstNodeId];
            if (static_cast<size_t>(inputIdx) < slots.size()) {
              if (slots[inputIdx] >= 0 &&
                  slots[inputIdx] != static_cast<int>(*peInputIndex))
                summary.complete = false;
              slots[inputIdx] = static_cast<int>(*peInputIndex);
            }
            if (pe.peKind == "temporal_pe") {
              if (auto tag =
                      computeTemporalRouteTagValue(swEdgeId, hwPath, i, state,
                                                   dfg, adg)) {
                auto &tags = summary.tagsByFU[dstNodeId];
                if (std::find(tags.begin(), tags.end(), *tag) == tags.end())
                  tags.push_back(*tag);
              }
            }
          }
        }
      }

      if (srcNodeId != INVALID_ID && fuSet.contains(srcNodeId)) {
        auto peOutputIndex = getUIntEdgeAttr(flatEdge, "pe_output_index");
        if (peOutputIndex) {
          const Node *fuNode = adg.getNode(srcNodeId);
          if (fuNode && outputIdx >= 0) {
            auto &slots = summary.outputPortSelects[srcNodeId];
            if (static_cast<size_t>(outputIdx) < slots.size()) {
              if (slots[outputIdx] >= 0 &&
                  slots[outputIdx] != static_cast<int>(*peOutputIndex))
                summary.complete = false;
              slots[outputIdx] = static_cast<int>(*peOutputIndex);
            }
            auto &usedOutputs = usedPeOutputsByFU[srcNodeId];
            auto existing = usedOutputs.find(static_cast<int>(*peOutputIndex));
            if (existing != usedOutputs.end() && existing->second != outputIdx)
              summary.complete = false;
            usedOutputs[static_cast<int>(*peOutputIndex)] = outputIdx;
            if (pe.peKind == "temporal_pe") {
              if (auto tag = computeTemporalRouteTagValue(swEdgeId, hwPath,
                                                          i + 1, state, dfg,
                                                          adg)) {
                auto &tags = summary.tagsByFU[srcNodeId];
                if (std::find(tags.begin(), tags.end(), *tag) == tags.end())
                  tags.push_back(*tag);
              }
            }
          }
        }
      }
    }
  }

  return summary;
}

// ---------------------------------------------------------------------------
// PE config builders
// ---------------------------------------------------------------------------

GeneratedNodeConfig
buildFunctionUnitConfig(llvm::ArrayRef<FUConfigSelection> fuConfigs,
                        IdIndex hwId) {
  GeneratedNodeConfig cfg;
  const FUConfigSelection *selection = findFUConfigSelection(fuConfigs, hwId);
  if (!selection)
    return cfg;
  for (const auto &field : selection->fields) {
    uint32_t word = 0;
    if (field.kind == FUConfigFieldKind::Mux) {
      word = static_cast<uint32_t>(field.sel & 0xffffu);
      if (field.discard)
        word |= (1u << 16);
      if (field.disconnect)
        word |= (1u << 17);
    } else {
      word = static_cast<uint32_t>(field.value & 0xffffffffu);
    }
    cfg.words.push_back(word);
  }
  return cfg;
}

GeneratedNodeConfig
buildSpatialPEConfig(const PEContainment &pe, const MappingState &state,
                     const Graph &dfg, const Graph &adg,
                     llvm::ArrayRef<FUConfigSelection> fuConfigs,
                     bool &globalComplete) {
  GeneratedNodeConfig cfg;
  bool complete = true;
  unsigned fuCount = pe.fuNodeIds.size();
  if (fuCount == 0)
    return cfg;

  unsigned opcodeBits = bitWidthForChoices(fuCount);
  unsigned peInputSelBits = bitWidthForChoices(pe.numInputPorts);
  unsigned peOutputSelBits = bitWidthForChoices(pe.numOutputPorts);

  unsigned maxFuInputs = 0;
  unsigned maxFuOutputs = 0;
  unsigned maxFuConfigBits = 0;
  IdIndex activeFuId = INVALID_ID;
  unsigned activeOpcode = 0;
  unsigned usedFuCount = 0;
  for (unsigned ordinal = 0; ordinal < fuCount; ++ordinal) {
    IdIndex fuId = pe.fuNodeIds[ordinal];
    const Node *fuNode = adg.getNode(fuId);
    if (!fuNode)
      continue;
    maxFuInputs = std::max<unsigned>(maxFuInputs, fuNode->inputPorts.size());
    maxFuOutputs = std::max<unsigned>(maxFuOutputs, fuNode->outputPorts.size());
    maxFuConfigBits = std::max<unsigned>(maxFuConfigBits, getFUConfigBitWidth(fuNode));
    if (fuId < state.hwNodeToSwNodes.size() && !state.hwNodeToSwNodes[fuId].empty()) {
      ++usedFuCount;
      if (activeFuId == INVALID_ID) {
        activeFuId = fuId;
        activeOpcode = ordinal;
      }
    }
  }

  if (usedFuCount > 1)
    complete = false;

  PERouteSummary routes;
  if (activeFuId != INVALID_ID)
    routes = collectPERouteSummary(pe, state, dfg, adg);
  complete = complete && routes.complete;

  uint32_t bitPos = 0;
  bool enabled = activeFuId != INVALID_ID;
  packBits(cfg.words, bitPos, enabled ? 1u : 0u, 1);
  packBits(cfg.words, bitPos, activeOpcode, opcodeBits);

  const Node *activeFu = enabled ? adg.getNode(activeFuId) : nullptr;
  auto inputSelects =
      (enabled && routes.inputPortSelects.count(activeFuId))
          ? routes.inputPortSelects.lookup(activeFuId)
          : llvm::SmallVector<int, 4>();
  auto outputSelects =
      (enabled && routes.outputPortSelects.count(activeFuId))
          ? routes.outputPortSelects.lookup(activeFuId)
          : llvm::SmallVector<int, 4>();

  for (unsigned inputIdx = 0; inputIdx < maxFuInputs; ++inputIdx) {
    bool disconnect = true;
    bool discard = false;
    uint64_t sel = 0;
    if (enabled && activeFu && inputIdx < activeFu->inputPorts.size()) {
      if (inputIdx < inputSelects.size() && inputSelects[inputIdx] >= 0) {
        sel = static_cast<uint64_t>(inputSelects[inputIdx]);
        disconnect = false;
      }
    }
    packMuxField(cfg.words, bitPos, peInputSelBits, sel, discard, disconnect);
  }

  for (unsigned outputIdx = 0; outputIdx < maxFuOutputs; ++outputIdx) {
    bool disconnect = true;
    bool discard = false;
    uint64_t sel = 0;
    if (enabled && activeFu && outputIdx < activeFu->outputPorts.size()) {
      if (outputIdx < outputSelects.size() && outputSelects[outputIdx] >= 0) {
        sel = static_cast<uint64_t>(outputSelects[outputIdx]);
        disconnect = false;
      } else {
        disconnect = false;
        discard = true;
      }
    }
    packMuxField(cfg.words, bitPos, peOutputSelBits, sel, discard, disconnect);
  }

  if (maxFuConfigBits > 0) {
    if (enabled && activeFu) {
      const FUConfigSelection *selection =
          findFUConfigSelection(fuConfigs, activeFuId);
      uint32_t configStart = bitPos;
      packFUConfigBits(cfg.words, bitPos, activeFu, selection, complete);
      unsigned actualBits = bitPos - configStart;
      if (actualBits < maxFuConfigBits)
        packBits(cfg.words, bitPos, 0u, maxFuConfigBits - actualBits);
      else if (actualBits > maxFuConfigBits)
        complete = false;
    } else {
      packBits(cfg.words, bitPos, 0u, maxFuConfigBits);
    }
  }

  cfg.complete = complete;
  globalComplete = globalComplete && complete;
  return cfg;
}

GeneratedNodeConfig
buildTemporalPEConfig(const PEContainment &pe, const MappingState &state,
                      const Graph &dfg, const Graph &adg,
                      llvm::ArrayRef<TechMappedEdgeKind> edgeKinds,
                      llvm::ArrayRef<FUConfigSelection> fuConfigs,
                      bool &globalComplete) {
  GeneratedNodeConfig cfg;
  bool complete = true;
  unsigned fuCount = pe.fuNodeIds.size();
  if (fuCount == 0 || pe.numInstruction == 0)
    return cfg;

  unsigned opcodeBits = bitWidthForChoices(fuCount);
  unsigned regIdxBits = bitWidthForChoices(pe.numRegister);
  unsigned operandCfgWidth = pe.numRegister > 0 ? (1 + regIdxBits) : 0;
  unsigned inputMuxWidth = bitWidthForChoices(pe.numInputPorts) + 2;
  unsigned outputMuxWidth = bitWidthForChoices(pe.numOutputPorts) + 2;
  unsigned resultCfgWidth =
      pe.tagWidth + (pe.numRegister > 0 ? (1 + regIdxBits) : 0);

  unsigned maxFuInputs = 0;
  unsigned maxFuOutputs = 0;
  for (IdIndex fuId : pe.fuNodeIds) {
    const Node *fuNode = adg.getNode(fuId);
    if (!fuNode)
      continue;
    maxFuInputs = std::max<unsigned>(maxFuInputs, fuNode->inputPorts.size());
    maxFuOutputs = std::max<unsigned>(maxFuOutputs, fuNode->outputPorts.size());
  }

  PERouteSummary routes = collectPERouteSummary(pe, state, dfg, adg);
  complete = complete && routes.complete;
  TemporalConfigPlan temporalPlan =
      buildTemporalConfigPlan(pe, state, dfg, adg, edgeKinds);
  complete = complete && temporalPlan.complete;
  auto &usedFUs = temporalPlan.usedFUs;
  if (usedFUs.size() > pe.numInstruction)
    complete = false;

  auto normalizeCommutativeInputSelects =
      [&](IdIndex fuId, llvm::SmallVector<int, 4> &inputSelects) {
        if (inputSelects.size() != 2 || fuId >= state.hwNodeToSwNodes.size() ||
            state.hwNodeToSwNodes[fuId].size() != 1) {
          return;
        }
        IdIndex swNodeId = state.hwNodeToSwNodes[fuId].front();
        const Node *swNode = dfg.getNode(swNodeId);
        if (!swNode ||
            !isCommutativeTechMappedOp(getNodeAttrStr(swNode, "op_name")))
          return;

        auto regIt = temporalPlan.operandRegsByFU.find(fuId);
        llvm::SmallVector<unsigned, 2> externalOperandIndices;
        llvm::SmallVector<int, 2> externalSelects;
        for (unsigned operandIdx = 0; operandIdx < inputSelects.size();
             ++operandIdx) {
          bool isReg = regIt != temporalPlan.operandRegsByFU.end() &&
                       operandIdx < regIt->second.size() &&
                       regIt->second[operandIdx].has_value();
          if (isReg || inputSelects[operandIdx] < 0)
            continue;
          externalOperandIndices.push_back(operandIdx);
          externalSelects.push_back(inputSelects[operandIdx]);
        }
        if (externalSelects.size() < 2)
          return;
        std::sort(externalSelects.begin(), externalSelects.end());
        for (unsigned idx = 0; idx < externalOperandIndices.size(); ++idx)
          inputSelects[externalOperandIndices[idx]] = externalSelects[idx];
      };

  uint32_t bitPos = 0;
  for (unsigned slot = 0; slot < pe.numInstruction; ++slot) {
    bool valid = slot < usedFUs.size();
    uint64_t tag = 0;
    unsigned opcode = 0;
    const Node *fuNode = nullptr;
    llvm::SmallVector<int, 4> inputSelects;
    llvm::SmallVector<int, 4> outputSelects;

    if (valid) {
      opcode = usedFUs[slot].first;
      IdIndex fuId = usedFUs[slot].second;
      fuNode = adg.getNode(fuId);
      if (routes.inputPortSelects.count(fuId))
        inputSelects = routes.inputPortSelects.lookup(fuId);
      if (routes.outputPortSelects.count(fuId))
        outputSelects = routes.outputPortSelects.lookup(fuId);
      normalizeCommutativeInputSelects(fuId, inputSelects);
      auto tags = routes.tagsByFU.lookup(fuId);
      if (tags.empty()) {
        complete = false;
      } else {
        tag = tags.front();
        if (tags.size() > 1)
          complete = false;
      }
    }

    packBits(cfg.words, bitPos, valid ? 1u : 0u, 1);
    packBits(cfg.words, bitPos, tag, pe.tagWidth);
    packBits(cfg.words, bitPos, opcode, opcodeBits);

    for (unsigned operandIdx = 0; operandIdx < maxFuInputs; ++operandIdx) {
      if (operandCfgWidth == 0)
        continue;
      uint64_t regIdx = 0;
      bool isReg = false;
      if (valid && fuNode) {
        auto found = temporalPlan.operandRegsByFU.find(usedFUs[slot].second);
        if (found != temporalPlan.operandRegsByFU.end() &&
            operandIdx < found->second.size() &&
            found->second[operandIdx].has_value()) {
          regIdx = *found->second[operandIdx];
          isReg = true;
        }
      }
      packBits(cfg.words, bitPos, regIdx, regIdxBits);
      packBits(cfg.words, bitPos, isReg ? 1u : 0u, 1);
    }

    for (unsigned inputIdx = 0; inputIdx < maxFuInputs; ++inputIdx) {
      bool disconnect = true;
      bool discard = false;
      uint64_t sel = 0;
      bool isReg = false;
      if (valid && fuNode) {
        auto found = temporalPlan.operandRegsByFU.find(usedFUs[slot].second);
        if (found != temporalPlan.operandRegsByFU.end() &&
            inputIdx < found->second.size() &&
            found->second[inputIdx].has_value()) {
          isReg = true;
        }
      }
      if (!isReg && valid && fuNode && inputIdx < fuNode->inputPorts.size()) {
        if (inputIdx < inputSelects.size() && inputSelects[inputIdx] >= 0) {
          sel = static_cast<uint64_t>(inputSelects[inputIdx]);
          disconnect = false;
        }
      }
      packMuxField(cfg.words, bitPos, inputMuxWidth - 2, sel, discard,
                   disconnect);
    }

    for (unsigned outputIdx = 0; outputIdx < maxFuOutputs; ++outputIdx) {
      bool disconnect = true;
      bool discard = false;
      uint64_t sel = 0;
      if (valid && fuNode && outputIdx < fuNode->outputPorts.size()) {
        if (outputIdx < outputSelects.size() && outputSelects[outputIdx] >= 0) {
          sel = static_cast<uint64_t>(outputSelects[outputIdx]);
          disconnect = false;
        } else {
          disconnect = false;
          discard = true;
        }
      }
      packMuxField(cfg.words, bitPos, outputMuxWidth - 2, sel, discard,
                   disconnect);
    }

    for (unsigned resultIdx = 0; resultIdx < maxFuOutputs; ++resultIdx) {
      if (resultCfgWidth == 0)
        continue;
      uint64_t resultTag = 0;
      uint64_t regIdx = 0;
      bool isReg = false;
      if (valid && fuNode) {
        auto found = temporalPlan.resultRegsByFU.find(usedFUs[slot].second);
        if (found != temporalPlan.resultRegsByFU.end() &&
            resultIdx < found->second.size() &&
            found->second[resultIdx].has_value()) {
          regIdx = *found->second[resultIdx];
          isReg = true;
        }
      }
      if (isReg) {
        resultTag = 0;
        if (resultIdx < outputSelects.size() && outputSelects[resultIdx] >= 0)
          complete = false;
      } else if (valid && fuNode && resultIdx < fuNode->outputPorts.size() &&
                 resultIdx < outputSelects.size() &&
                 outputSelects[resultIdx] >= 0) {
        resultTag = tag;
      }
      packBits(cfg.words, bitPos, resultTag, pe.tagWidth);
      if (pe.numRegister > 0) {
        packBits(cfg.words, bitPos, regIdx, regIdxBits);
        packBits(cfg.words, bitPos, isReg ? 1u : 0u, 1);
      }
    }
  }

  for (IdIndex fuId : pe.fuNodeIds) {
    const Node *fuNode = adg.getNode(fuId);
    const FUConfigSelection *selection = findFUConfigSelection(fuConfigs, fuId);
    packFUConfigBits(cfg.words, bitPos, fuNode, selection, complete);
  }

  cfg.complete = complete;
  globalComplete = globalComplete && complete;
  return cfg;
}

} // namespace configgen_detail

// ===========================================================================
// ConfigGen class method implementations (config artifact building & output)
// ===========================================================================

bool ConfigGen::buildConfigArtifacts(const MappingState &state,
                                     const Graph &dfg, const Graph &adg,
                                     const ADGFlattener &flattener,
                                     llvm::ArrayRef<TechMappedEdgeKind> edgeKinds,
                                     llvm::ArrayRef<FUConfigSelection> fuConfigs) {
  using namespace configgen_detail;

  nodeConfigs_.clear();
  configSlices_.clear();
  configWords_.clear();
  configBlob_.clear();
  configComplete_ = true;

  for (IdIndex hwId = 0; hwId < static_cast<IdIndex>(adg.nodes.size()); ++hwId) {
    const Node *hwNode = adg.getNode(hwId);
    if (!hwNode || hwNode->kind != Node::OperationNode)
      continue;

    GeneratedNodeConfig generated;
    llvm::StringRef resourceClass = getNodeAttrStr(hwNode, "resource_class");
    llvm::StringRef opKind = getNodeAttrStr(hwNode, "op_kind");

    if (resourceClass == "routing") {
      if (opKind == "spatial_sw")
        generated = buildSpatialSwitchConfig(hwNode, hwId, state, dfg, adg);
      else if (opKind == "temporal_sw")
        generated = buildTemporalSwitchConfig(hwNode, hwId, state, dfg, adg);
      else if (opKind == "add_tag")
        generated = buildAddTagConfig(hwNode);
      else if (opKind == "map_tag")
        generated = buildMapTagConfig(hwNode, adg);
      else if (opKind == "fifo")
        generated = buildFifoConfig(hwNode, hwId, state);
      else
        continue;
    } else if (resourceClass == "memory") {
      generated = buildMemoryConfig(hwNode, hwId, state, dfg, adg);
    } else {
      continue;
    }

    if (generated.words.empty())
      continue;

    NodeConfig nodeCfg;
    nodeCfg.name = getNodeAttrStr(hwNode, "op_name").str();
    nodeCfg.kind = opKind.str();
    nodeCfg.hwNode = hwId;
    nodeCfg.complete = generated.complete;
    nodeCfg.words = std::move(generated.words);
    nodeConfigs_.push_back(nodeCfg);

    ConfigSlice slice;
    slice.name = nodeCfg.name;
    slice.kind = nodeCfg.kind;
    slice.hwNode = hwId;
    slice.wordOffset = static_cast<uint32_t>(configWords_.size());
    slice.wordCount = static_cast<uint32_t>(nodeCfg.words.size());
    slice.complete = nodeCfg.complete;
    configSlices_.push_back(slice);

    configWords_.insert(configWords_.end(), nodeCfg.words.begin(),
                        nodeCfg.words.end());
    configComplete_ = configComplete_ && nodeCfg.complete;
  }

  for (const auto &pe : flattener.getPEContainment()) {
    GeneratedNodeConfig generated;
    if (pe.peKind == "spatial_pe") {
      generated = buildSpatialPEConfig(pe, state, dfg, adg, fuConfigs,
                                       configComplete_);
    } else if (pe.peKind == "temporal_pe") {
      generated = buildTemporalPEConfig(pe, state, dfg, adg, edgeKinds,
                                        fuConfigs,
                                        configComplete_);
    } else {
      continue;
    }

    if (generated.words.empty())
      continue;

    NodeConfig nodeCfg;
    nodeCfg.name = pe.peName;
    nodeCfg.kind = pe.peKind;
    nodeCfg.hwNode = INVALID_ID;
    nodeCfg.complete = generated.complete;
    nodeCfg.words = std::move(generated.words);
    nodeConfigs_.push_back(nodeCfg);

    ConfigSlice slice;
    slice.name = nodeCfg.name;
    slice.kind = nodeCfg.kind;
    slice.hwNode = INVALID_ID;
    slice.wordOffset = static_cast<uint32_t>(configWords_.size());
    slice.wordCount = static_cast<uint32_t>(nodeCfg.words.size());
    slice.complete = nodeCfg.complete;
    configSlices_.push_back(slice);

    configWords_.insert(configWords_.end(), nodeCfg.words.begin(),
                        nodeCfg.words.end());
  }

  configBlob_.reserve(configWords_.size() * sizeof(uint32_t));
  for (uint32_t word : configWords_) {
    configBlob_.push_back(static_cast<uint8_t>(word & 0xffu));
    configBlob_.push_back(static_cast<uint8_t>((word >> 8) & 0xffu));
    configBlob_.push_back(static_cast<uint8_t>((word >> 16) & 0xffu));
    configBlob_.push_back(static_cast<uint8_t>((word >> 24) & 0xffu));
  }

  return true;
}

bool ConfigGen::writeConfigBinary(const std::string &path) const {
  std::error_code ec;
  llvm::raw_fd_ostream out(path, ec, llvm::sys::fs::OF_None);
  if (ec) {
    llvm::errs() << "ConfigGen: cannot open " << path << ": " << ec.message()
                 << "\n";
    return false;
  }
  if (!configBlob_.empty())
    out.write(reinterpret_cast<const char *>(configBlob_.data()),
              configBlob_.size());
  return true;
}

bool ConfigGen::writeConfigJson(const std::string &path) const {
  using namespace configgen_detail;

  std::error_code ec;
  llvm::raw_fd_ostream out(path, ec, llvm::sys::fs::OF_Text);
  if (ec) {
    llvm::errs() << "ConfigGen: cannot open " << path << ": " << ec.message()
                 << "\n";
    return false;
  }

  out << "{\n";
  out << "  \"word_width_bits\": 32,\n";
  out << "  \"word_count\": " << configWords_.size() << ",\n";
  out << "  \"byte_size\": " << configBlob_.size() << ",\n";
  out << "  \"complete\": " << (configComplete_ ? "true" : "false") << ",\n";
  out << "  \"coverage_note\": "
      << (configComplete_
              ? "\"all currently modeled configurable slices serialized\""
              : "\"routing, tag, memory, spatial_pe, and temporal_pe slices serialized; some slice contents remain partially modeled\"")
      << ",\n";
  out << "  \"words\": [";
  for (size_t i = 0; i < configWords_.size(); ++i) {
    if (i > 0)
      out << ", ";
    out << configWords_[i];
  }
  out << "],\n";
  out << "  \"slices\": [\n";
  for (size_t i = 0; i < configSlices_.size(); ++i) {
    const auto &slice = configSlices_[i];
    if (i > 0)
      out << ",\n";
    out << "    {\"name\": \"" << slice.name << "\", \"kind\": \""
        << slice.kind << "\", \"hw_node\": ";
    if (slice.hwNode == INVALID_ID)
      out << "null";
    else
      out << slice.hwNode;
    out
        << ", \"word_offset\": " << slice.wordOffset
        << ", \"word_count\": " << slice.wordCount
        << ", \"complete\": " << (slice.complete ? "true" : "false") << "}";
  }
  out << "\n  ]\n";
  out << "}\n";
  return true;
}

bool ConfigGen::writeConfigHeader(const std::string &path) const {
  using namespace configgen_detail;

  std::error_code ec;
  llvm::raw_fd_ostream out(path, ec, llvm::sys::fs::OF_Text);
  if (ec) {
    llvm::errs() << "ConfigGen: cannot open " << path << ": " << ec.message()
                 << "\n";
    return false;
  }

  std::string guard = buildHeaderGuard(path);

  out << "#ifndef " << guard << "\n";
  out << "#define " << guard << "\n\n";
  out << "#include <stdint.h>\n\n";
  out << "static const uint32_t fcc_runtime_config_words[] = {";
  if (configWords_.empty()) {
    out << "0";
  } else {
    for (size_t i = 0; i < configWords_.size(); ++i) {
      if (i == 0)
        out << "\n    ";
      else if ((i % 6) == 0)
        out << ",\n    ";
      else
        out << ", ";
      out << "0x";
      out.write_hex(configWords_[i]);
    }
    out << "\n";
  }
  out << "};\n";
  out << "static const unsigned fcc_runtime_config_word_count = "
      << configWords_.size() << ";\n";
  out << "static const int fcc_runtime_config_complete = "
      << (configComplete_ ? 1 : 0) << ";\n\n";
  out << "#endif\n";
  return true;
}

bool ConfigGen::generate(const MappingState &state, const Graph &dfg,
                         const Graph &adg, const ADGFlattener &flattener,
                         llvm::ArrayRef<TechMappedEdgeKind> edgeKinds,
                         llvm::ArrayRef<FUConfigSelection> fuConfigs,
                         const std::string &basePath, int seed,
                         const TechMapper::Plan *techMapPlan,
                         const TechMapper::PlanMetrics *techMapMetrics,
                         const MapperTimingSummary *timingSummary,
                         const MapperSearchSummary *searchSummary,
                         llvm::StringRef techMapDiagnostics) {
  using namespace configgen_detail;

  if (!buildConfigArtifacts(state, dfg, adg, flattener, edgeKinds, fuConfigs))
    return false;
  if (!writeConfigBinary(basePath + ".config.bin"))
    return false;
  if (!writeConfigJson(basePath + ".config.json"))
    return false;
  if (!writeConfigHeader(getConfigHeaderFilename(basePath)))
    return false;
  if (!writeMapJson(state, dfg, adg, flattener, edgeKinds, fuConfigs,
                    basePath + ".map.json", seed, techMapPlan, techMapMetrics,
                    timingSummary, searchSummary,
                    techMapDiagnostics))
    return false;
  if (!writeMapText(state, dfg, adg, flattener, edgeKinds,
                    basePath + ".map.txt", techMapPlan, techMapMetrics,
                    timingSummary, searchSummary,
                    techMapDiagnostics))
    return false;
  return true;
}

} // namespace fcc
