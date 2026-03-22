#include "ConfigGenInternal.h"

#include "loom/Mapper/ConfigGen.h"
#include "loom/Mapper/Mapper.h"

#include "mlir/IR/BuiltinAttributes.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/raw_ostream.h"

#include <map>
#include <tuple>
#include <vector>

namespace loom {

void ConfigGen::writeMapJsonTailSections(
    llvm::raw_ostream &out, const MappingState &state, const Graph &dfg,
    const Graph &adg, const ADGFlattener &flattener,
    llvm::ArrayRef<TechMappedEdgeKind> edgeKinds,
    llvm::ArrayRef<FUConfigSelection> fuConfigs,
    const MapperTimingSummary *timingSummary,
    const MapperSearchSummary *searchSummary) {
  using namespace configgen_detail;

  out << "  \"search\": {\n";
  if (!searchSummary) {
    out << "    \"available\": false\n";
  } else {
    out << "    \"available\": true";
    out << ",\n    \"techmap_feedback_attempts\": "
        << searchSummary->techMapFeedbackAttempts;
    out << ",\n    \"techmap_feedback_accepted_reconfigurations\": "
        << searchSummary->techMapFeedbackAcceptedReconfigurations;
    out << ",\n    \"placement_seed_lane_count\": "
        << searchSummary->placementSeedLaneCount;
    out << ",\n    \"successful_placement_seed_count\": "
        << searchSummary->successfulPlacementSeedCount;
    out << ",\n    \"routed_lane_count\": "
        << searchSummary->routedLaneCount;
    out << ",\n    \"local_repair_attempts\": "
        << searchSummary->localRepairAttempts;
    out << ",\n    \"local_repair_successes\": "
        << searchSummary->localRepairSuccesses;
    out << ",\n    \"route_aware_refinement_passes\": "
        << searchSummary->routeAwareRefinementPasses;
    out << ",\n    \"route_aware_checkpoint_rescore_passes\": "
        << searchSummary->routeAwareCheckpointRescorePasses;
    out << ",\n    \"route_aware_checkpoint_restore_count\": "
        << searchSummary->routeAwareCheckpointRestoreCount;
    out << ",\n    \"route_aware_neighborhood_attempts\": "
        << searchSummary->routeAwareNeighborhoodAttempts;
    out << ",\n    \"route_aware_neighborhood_accepted_moves\": "
        << searchSummary->routeAwareNeighborhoodAcceptedMoves;
    out << ",\n    \"route_aware_coarse_fallback_moves\": "
        << searchSummary->routeAwareCoarseFallbackMoves;
    out << ",\n    \"fifo_bufferization_accepted_toggles\": "
        << searchSummary->fifoBufferizationAcceptedToggles;
    out << ",\n    \"outer_joint_accepted_rounds\": "
        << searchSummary->outerJointAcceptedRounds << "\n";
  }
  out << "  },\n";
  out << "  \"timing\": {\n";
  if (!timingSummary) {
    out << "    \"available\": false\n";
  } else {
    out << "    \"available\": true";
    out << ",\n    \"estimated_critical_path_delay\": "
        << timingSummary->estimatedCriticalPathDelay;
    out << ",\n    \"estimated_clock_period\": "
        << timingSummary->estimatedClockPeriod;
    out << ",\n    \"estimated_initiation_interval\": "
        << timingSummary->estimatedInitiationInterval;
    out << ",\n    \"estimated_throughput_cost\": "
        << timingSummary->estimatedThroughputCost;
    out << ",\n    \"recurrence_pressure\": "
        << timingSummary->recurrencePressure;
    out << ",\n    \"critical_path_edges\": [";
    for (size_t idx = 0; idx < timingSummary->criticalPathEdges.size(); ++idx) {
      if (idx != 0)
        out << ", ";
      out << timingSummary->criticalPathEdges[idx];
    }
    out << "]";
    out << ",\n    \"fifo_buffer_count\": "
        << timingSummary->fifoBufferCount;
    out << ",\n    \"forced_buffered_fifo_count\": "
        << timingSummary->forcedBufferedFifoCount;
    out << ",\n    \"forced_buffered_fifo_nodes\": [";
    for (size_t idx = 0; idx < timingSummary->forcedBufferedFifoNodes.size();
         ++idx) {
      if (idx != 0)
        out << ", ";
      out << timingSummary->forcedBufferedFifoNodes[idx];
    }
    out << "],\n    \"forced_buffered_fifo_depths\": [";
    for (size_t idx = 0; idx < timingSummary->forcedBufferedFifoDepths.size();
         ++idx) {
      if (idx != 0)
        out << ", ";
      out << timingSummary->forcedBufferedFifoDepths[idx];
    }
    out << "],\n    \"mapper_selected_buffered_fifo_count\": "
        << timingSummary->mapperSelectedBufferedFifoCount;
    out << ",\n    \"mapper_selected_buffered_fifo_nodes\": [";
    for (size_t idx = 0;
         idx < timingSummary->mapperSelectedBufferedFifoNodes.size(); ++idx) {
      if (idx != 0)
        out << ", ";
      out << timingSummary->mapperSelectedBufferedFifoNodes[idx];
    }
    out << "],\n    \"mapper_selected_buffered_fifo_depths\": [";
    for (size_t idx = 0;
         idx < timingSummary->mapperSelectedBufferedFifoDepths.size(); ++idx) {
      if (idx != 0)
        out << ", ";
      out << timingSummary->mapperSelectedBufferedFifoDepths[idx];
    }
    out << "],\n    \"bufferized_edges\": [";
    for (size_t idx = 0; idx < timingSummary->bufferizedEdges.size(); ++idx) {
      if (idx != 0)
        out << ", ";
      out << timingSummary->bufferizedEdges[idx];
    }
    out << "],\n    \"recurrence_cycles\": [";
    for (size_t idx = 0; idx < timingSummary->recurrenceCycles.size(); ++idx) {
      const auto &cycle = timingSummary->recurrenceCycles[idx];
      if (idx != 0)
        out << ", ";
      out << "{\"cycle_id\": " << cycle.cycleId
          << ", \"sw_nodes\": [";
      for (size_t nodeIdx = 0; nodeIdx < cycle.swNodes.size(); ++nodeIdx) {
        if (nodeIdx != 0)
          out << ", ";
        out << cycle.swNodes[nodeIdx];
      }
      out << "], \"sw_edges\": [";
      for (size_t edgeIdx = 0; edgeIdx < cycle.swEdges.size(); ++edgeIdx) {
        if (edgeIdx != 0)
          out << ", ";
        out << cycle.swEdges[edgeIdx];
      }
      out << "], \"recurrence_distance\": " << cycle.recurrenceDistance
          << ", \"sequential_latency_cycles\": "
          << cycle.sequentialLatencyCycles
          << ", \"fifo_stage_cut_contribution\": "
          << cycle.fifoStageCutContribution
          << ", \"max_interval_on_cycle\": "
          << cycle.maxIntervalOnCycle
          << ", \"estimated_cycle_ii\": "
          << cycle.estimatedCycleII
          << ", \"combinational_delay\": "
          << cycle.combinationalDelay << "}";
    }
    out << "]\n";
  }
  out << "  },\n";
  out << "  \"node_mappings\": [\n";

  llvm::DenseMap<IdIndex, unsigned> configClassBySwNode;
  llvm::DenseMap<IdIndex, unsigned> supportClassBySwNode;
  for (const auto &selection : fuConfigs) {
    for (IdIndex swNodeId : selection.swNodeIds) {
      configClassBySwNode[swNodeId] = selection.configClassId;
      supportClassBySwNode[swNodeId] = selection.supportClassId;
    }
  }

  bool first = true;
  for (IdIndex swId = 0; swId < static_cast<IdIndex>(dfg.nodes.size());
       ++swId) {
    const Node *swNode = dfg.getNode(swId);
    if (!swNode || swNode->kind != Node::OperationNode)
      continue;
    if (swId >= state.swNodeToHwNode.size() ||
        state.swNodeToHwNode[swId] == INVALID_ID)
      continue;

    if (!first)
      out << ",\n";
    first = false;

    IdIndex hwId = state.swNodeToHwNode[swId];
    const Node *hwNode = adg.getNode(hwId);

    out << "    {\"sw_node\": " << swId << ", \"hw_node\": " << hwId;
    out << ", \"sw_op\": \"" << getNodeAttrStr(swNode, "op_name") << "\"";
    if (auto it = supportClassBySwNode.find(swId);
        it != supportClassBySwNode.end()) {
      out << ", \"support_class_id\": " << it->second;
    }
    if (auto it = configClassBySwNode.find(swId);
        it != configClassBySwNode.end()) {
      out << ", \"config_class_id\": " << it->second;
    }
    if (hwNode) {
      out << ", \"hw_name\": \"" << getNodeAttrStr(hwNode, "op_name") << "\"";
      out << ", \"pe_name\": \"" << getNodeAttrStr(hwNode, "pe_name") << "\"";
    }
    out << "}";
  }

  out << "\n  ],\n";

  out << "  \"edge_routings\": [\n";
  first = true;
  for (IdIndex eid = 0; eid < static_cast<IdIndex>(dfg.edges.size()); ++eid) {
    const Edge *edge = dfg.getEdge(eid);
    if (!edge)
      continue;

    if (!first)
      out << ",\n";
    first = false;

    out << "    {\"sw_edge\": " << eid;
    if (eid < edgeKinds.size() && edgeKinds[eid] == TechMappedEdgeKind::IntraFU) {
      IdIndex hwNodeId = INVALID_ID;
      const Port *srcPort = dfg.getPort(edge->srcPort);
      if (srcPort && srcPort->parentNode != INVALID_ID &&
          srcPort->parentNode < state.swNodeToHwNode.size()) {
        hwNodeId = state.swNodeToHwNode[srcPort->parentNode];
      }
      out << ", \"kind\": \"intra_fu\"";
      if (hwNodeId != INVALID_ID)
        out << ", \"hw_node\": " << hwNodeId;
      out << ", \"path\": []}";
      continue;
    }
    if (eid < edgeKinds.size() &&
        edgeKinds[eid] == TechMappedEdgeKind::TemporalReg) {
      IdIndex hwNodeId = INVALID_ID;
      const Port *srcPort = dfg.getPort(edge->srcPort);
      if (srcPort && srcPort->parentNode != INVALID_ID &&
          srcPort->parentNode < state.swNodeToHwNode.size()) {
        hwNodeId = state.swNodeToHwNode[srcPort->parentNode];
      }
      out << ", \"kind\": \"temporal_reg\"";
      if (hwNodeId != INVALID_ID)
        out << ", \"hw_node\": " << hwNodeId;
      out << ", \"path\": []}";
      continue;
    }

    if (eid >= state.swEdgeToHwPaths.size() || state.swEdgeToHwPaths[eid].empty()) {
      out << ", \"kind\": \"unrouted\", \"path\": []}";
      continue;
    }

    auto exportPath = buildExportPathForEdge(eid, state, dfg, adg);
    out << ", \"kind\": \"routed\", \"path\": [";
    for (size_t i = 0; i < exportPath.size(); ++i) {
      if (i > 0)
        out << ", ";
      out << exportPath[i];
    }
    out << "]}";
  }

  out << "\n  ],\n";

  struct PortVizInfo {
    bool valid = false;
    std::string kind;
    std::string component;
    std::string pe;
    std::string fu;
    int index = -1;
    std::string dir;
  };

  std::vector<PortVizInfo> portInfo(adg.ports.size());
  out << "  \"port_table\": [\n";
  first = true;
  for (IdIndex pid = 0; pid < static_cast<IdIndex>(adg.ports.size()); ++pid) {
    const Port *p = adg.getPort(pid);
    const Node *pn = p ? adg.getNode(p->parentNode) : nullptr;
    if (!p || !pn)
      continue;

    if (!first)
      out << ",\n";
    first = false;

    out << "    {\"id\": " << pid;
    portInfo[pid].valid = true;
    portInfo[pid].dir =
        (p->direction == Port::Input ? std::string("in") : std::string("out"));

    if (pn->kind == Node::ModuleInputNode) {
      int argIdx = getNodeAttrInt(pn, "arg_index");
      portInfo[pid].kind = "module_in";
      portInfo[pid].component = "module_in";
      portInfo[pid].index = argIdx;
      out << ", \"kind\": \"module_in\", \"index\": " << argIdx;
    } else if (pn->kind == Node::ModuleOutputNode) {
      int resIdx = getNodeAttrInt(pn, "result_index");
      portInfo[pid].kind = "module_out";
      portInfo[pid].component = "module_out";
      portInfo[pid].index = resIdx;
      out << ", \"kind\": \"module_out\", \"index\": " << resIdx;
    } else {
      llvm::StringRef resClass = getNodeAttrStr(pn, "resource_class");
      if (resClass == "routing") {
        llvm::StringRef opName = getNodeAttrStr(pn, "op_name");
        llvm::StringRef opKind = getNodeAttrStr(pn, "op_kind");
        bool isSwitchLike =
            pn->inputPorts.size() > 1 || pn->outputPorts.size() > 1;
        if (opKind == "temporal_sw")
          portInfo[pid].kind = "temporal_sw";
        else if (opKind == "spatial_sw")
          portInfo[pid].kind = "sw";
        else if (opKind == "add_tag" || opKind == "del_tag" ||
                 opKind == "map_tag")
          portInfo[pid].kind = opKind.str();
        else
          portInfo[pid].kind = isSwitchLike ? "sw" : "fifo";
        portInfo[pid].component = opName.str();
        out << ", \"kind\": \"" << portInfo[pid].kind << "\", \"name\": \""
            << opName << "\"";
      } else if (resClass == "memory") {
        llvm::StringRef opName = getNodeAttrStr(pn, "op_name");
        portInfo[pid].kind = "memory";
        portInfo[pid].component = opName.str();
        out << ", \"kind\": \"memory\", \"name\": \"" << opName << "\"";
      } else {
        llvm::StringRef peName = getNodeAttrStr(pn, "pe_name");
        llvm::StringRef opName = getNodeAttrStr(pn, "op_name");
        portInfo[pid].kind = "fu";
        portInfo[pid].component = peName.str();
        portInfo[pid].pe = peName.str();
        portInfo[pid].fu = opName.str();
        out << ", \"kind\": \"fu\", \"pe\": \"" << peName << "\", \"fu\": \""
            << opName << "\"";
      }
      int portIdx = -1;
      if (p->direction == Port::Input) {
        for (unsigned i = 0; i < pn->inputPorts.size(); ++i) {
          if (pn->inputPorts[i] == pid) {
            portIdx = static_cast<int>(i);
            break;
          }
        }
      } else {
        for (unsigned i = 0; i < pn->outputPorts.size(); ++i) {
          if (pn->outputPorts[i] == pid) {
            portIdx = static_cast<int>(i);
            break;
          }
        }
      }
      portInfo[pid].index = portIdx;
      out << ", \"index\": " << portIdx;
    }

    out << ", \"dir\": \""
        << (p->direction == Port::Input ? "in" : "out") << "\"}";
  }
  out << "\n  ],\n";

  out << "  \"switch_routes\": [\n";
  using SwitchRouteKey = std::tuple<std::string, int, int>;
  struct SwitchRouteEntry {
    std::string component;
    IdIndex inputPortId = INVALID_ID;
    IdIndex outputPortId = INVALID_ID;
    int inputPort = -1;
    int outputPort = -1;
    std::vector<IdIndex> swEdges;
  };
  std::map<SwitchRouteKey, SwitchRouteEntry> switchRoutes;
  for (IdIndex eid = 0; eid < static_cast<IdIndex>(state.swEdgeToHwPaths.size());
       ++eid) {
    auto hwPath = buildExportPathForEdge(eid, state, dfg, adg);
    if (hwPath.size() < 3)
      continue;
    for (size_t i = 1; i + 1 < hwPath.size(); i += 2) {
      IdIndex inPortId = hwPath[i];
      IdIndex outPortId = hwPath[i + 1];
      if (inPortId >= portInfo.size() || outPortId >= portInfo.size())
        continue;
      const auto &inInfo = portInfo[inPortId];
      const auto &outInfo = portInfo[outPortId];
      if (!inInfo.valid || !outInfo.valid)
        continue;
      bool inSwitch =
          (inInfo.kind == "sw" || inInfo.kind == "temporal_sw");
      bool outSwitch =
          (outInfo.kind == "sw" || outInfo.kind == "temporal_sw");
      if (!inSwitch || !outSwitch || inInfo.component != outInfo.component)
        continue;

      auto key = std::make_tuple(inInfo.component, inInfo.index, outInfo.index);
      auto &entry = switchRoutes[key];
      if (entry.component.empty()) {
        entry.component = inInfo.component;
        entry.inputPortId = inPortId;
        entry.outputPortId = outPortId;
        entry.inputPort = inInfo.index;
        entry.outputPort = outInfo.index;
      }
      entry.swEdges.push_back(eid);
    }
  }

  first = true;
  for (const auto &it : switchRoutes) {
    const auto &entry = it.second;
    if (!first)
      out << ",\n";
    first = false;

    out << "    {\"component\": \"" << entry.component << "\"";
    out << ", \"input_port_id\": " << entry.inputPortId;
    out << ", \"output_port_id\": " << entry.outputPortId;
    out << ", \"input_port\": " << entry.inputPort;
    out << ", \"output_port\": " << entry.outputPort;
    out << ", \"sw_edges\": [";
    for (size_t i = 0; i < entry.swEdges.size(); ++i) {
      if (i > 0)
        out << ", ";
      out << entry.swEdges[i];
    }
    out << "]}";
  }

  out << "\n  ],\n";

  out << "  \"pe_routes\": [\n";
  using PERouteKey =
      std::tuple<std::string, std::string, std::string, std::string>;
  struct PERouteEntry {
    std::string peName;
    std::string direction;
    std::string pePortKey;
    std::string fuPortKey;
    std::vector<IdIndex> swEdges;
  };
  std::map<PERouteKey, PERouteEntry> peRoutes;
  for (IdIndex eid = 0; eid < static_cast<IdIndex>(state.swEdgeToHwPaths.size());
       ++eid) {
    auto hwPath = buildExportPathForEdge(eid, state, dfg, adg);
    if (hwPath.size() < 2)
      continue;

    for (size_t i = 0; i + 1 < hwPath.size(); ++i) {
      IdIndex srcPortId = hwPath[i];
      IdIndex dstPortId = hwPath[i + 1];
      const Port *srcPort = adg.getPort(srcPortId);
      const Port *dstPort = adg.getPort(dstPortId);
      if (!srcPort || !dstPort)
        continue;

      const Edge *flatEdge = findEdgeByPorts(adg, srcPortId, dstPortId);
      if (!flatEdge)
        continue;

      if (dstPortId < portInfo.size()) {
        const auto &dstInfo = portInfo[dstPortId];
        const Node *dstNode = dstPort->parentNode != INVALID_ID
                                  ? adg.getNode(dstPort->parentNode)
                                  : nullptr;
        auto peInputIndex = getUIntEdgeAttr(flatEdge, "pe_input_index");
        int fuInputIndex = dstNode ? findNodeInputIndex(dstNode, dstPortId) : -1;
        if (dstInfo.valid && dstInfo.kind == "fu" && peInputIndex &&
            fuInputIndex >= 0) {
          std::string peName = dstInfo.pe;
          std::string hwName = dstInfo.fu;
          std::string pePortKey =
              peName + "_in_" + std::to_string(*peInputIndex);
          std::string fuPortKey =
              peName + "/" + hwName + "/in_" + std::to_string(fuInputIndex);
          auto key =
              std::make_tuple(peName, std::string("in"), pePortKey, fuPortKey);
          auto &entry = peRoutes[key];
          if (entry.peName.empty()) {
            entry.peName = peName;
            entry.direction = "in";
            entry.pePortKey = pePortKey;
            entry.fuPortKey = fuPortKey;
          }
          entry.swEdges.push_back(eid);
        }
      }

      if (srcPortId < portInfo.size()) {
        const auto &srcInfo = portInfo[srcPortId];
        const Node *srcNode = srcPort->parentNode != INVALID_ID
                                  ? adg.getNode(srcPort->parentNode)
                                  : nullptr;
        auto peOutputIndex = getUIntEdgeAttr(flatEdge, "pe_output_index");
        int fuOutputIndex =
            srcNode ? findNodeOutputIndex(srcNode, srcPortId) : -1;
        if (srcInfo.valid && srcInfo.kind == "fu" && peOutputIndex &&
            fuOutputIndex >= 0) {
          std::string peName = srcInfo.pe;
          std::string hwName = srcInfo.fu;
          std::string pePortKey =
              peName + "_out_" + std::to_string(*peOutputIndex);
          std::string fuPortKey =
              peName + "/" + hwName + "/out_" + std::to_string(fuOutputIndex);
          auto key = std::make_tuple(peName, std::string("out"), pePortKey,
                                     fuPortKey);
          auto &entry = peRoutes[key];
          if (entry.peName.empty()) {
            entry.peName = peName;
            entry.direction = "out";
            entry.pePortKey = pePortKey;
            entry.fuPortKey = fuPortKey;
          }
          entry.swEdges.push_back(eid);
        }
      }
    }
  }

  first = true;
  for (const auto &it : peRoutes) {
    const auto &entry = it.second;
    if (!first)
      out << ",\n";
    first = false;

    out << "    {\"pe_name\": \"" << entry.peName << "\"";
    out << ", \"direction\": \"" << entry.direction << "\"";
    out << ", \"pe_port_key\": \"" << entry.pePortKey << "\"";
    out << ", \"fu_port_key\": \"" << entry.fuPortKey << "\"";
    out << ", \"sw_edges\": [";
    for (size_t i = 0; i < entry.swEdges.size(); ++i) {
      if (i > 0)
        out << ", ";
      out << entry.swEdges[i];
    }
    out << "]}";
  }
  out << "\n  ],\n";

  out << "  \"fu_configs\": [\n";
  first = true;
  for (const auto &selection : fuConfigs) {
    if (!first)
      out << ",\n";
    first = false;
    out << "    {\"hw_node\": " << selection.hwNodeId;
    out << ", \"hw_name\": \"" << selection.hwName << "\"";
    out << ", \"pe_name\": \"" << selection.peName << "\"";
    out << ", \"support_class_id\": " << selection.supportClassId;
    out << ", \"config_class_id\": " << selection.configClassId;
    out << ", \"sw_nodes\": [";
    for (size_t i = 0; i < selection.swNodeIds.size(); ++i) {
      if (i > 0)
        out << ", ";
      out << selection.swNodeIds[i];
    }
    out << "], \"fields\": [";
    for (size_t i = 0; i < selection.fields.size(); ++i) {
      if (i > 0)
        out << ", ";
      const auto &field = selection.fields[i];
      out << "{\"op_index\": " << field.opIndex;
      out << ", \"op_name\": \"" << field.opName << "\"";
      out << ", \"kind\": \"" << configFieldKindName(field.kind) << "\"";
      out << ", \"bit_width\": " << field.bitWidth;
      out << ", \"value\": " << field.value;
      out << ", \"display\": \"" << formatConfigFieldValue(field) << "\"";
      if (field.kind == FUConfigFieldKind::Mux) {
        out << ", \"sel\": " << field.sel;
        out << ", \"discard\": " << (field.discard ? "true" : "false");
        out << ", \"disconnect\": "
            << (field.disconnect ? "true" : "false");
      }
      out << "}";
    }
    out << "]}";
  }
  out << "\n  ],\n";

  out << "  \"tag_configs\": [\n";
  first = true;
  for (const auto &slice : configSlices_) {
    if (slice.kind != "add_tag" && slice.kind != "map_tag")
      continue;
    const Node *hwNode =
        slice.hwNode == INVALID_ID ? nullptr : adg.getNode(slice.hwNode);
    if (!first)
      out << ",\n";
    first = false;
    out << "    {\"name\": \"" << slice.name << "\", \"kind\": \""
        << slice.kind << "\", \"complete\": "
        << (slice.complete ? "true" : "false");
    if (slice.kind == "add_tag") {
      out << ", \"tag\": " << getNodeAttrInt(hwNode, "tag", 0);
    } else {
      out << ", \"table_size\": " << getNodeAttrInt(hwNode, "table_size", 0);
      auto [inTagWidth, outTagWidth] = getMapTagTagWidths(hwNode, adg);
      out << ", \"input_tag_width\": " << inTagWidth;
      out << ", \"output_tag_width\": " << outTagWidth;
      out << ", \"table\": [";
      bool firstElem = true;
      for (const auto &entry : getMapTagTableEntries(hwNode)) {
        if (!firstElem)
          out << ", ";
        firstElem = false;
        out << "{\"valid\": " << (entry.valid ? "true" : "false")
            << ", \"src_tag\": " << entry.srcTag
            << ", \"dst_tag\": " << entry.dstTag << "}";
      }
      out << "]";
    }
    out << "}";
  }
  out << "\n  ],\n";

  out << "  \"fifo_configs\": [\n";
  first = true;
  for (const auto &slice : configSlices_) {
    if (slice.kind != "fifo")
      continue;
    const Node *hwNode =
        slice.hwNode == INVALID_ID ? nullptr : adg.getNode(slice.hwNode);
    bool bypassable = false;
    bool bypassed = getEffectiveFifoBypassed(hwNode, slice.hwNode, state);
    if (hwNode) {
      for (const auto &attr : hwNode->attributes) {
        if (attr.getName() == "bypassable") {
          if (auto boolAttr = mlir::dyn_cast<mlir::BoolAttr>(attr.getValue()))
            bypassable = boolAttr.getValue();
        }
      }
    }
    if (!first)
      out << ",\n";
    first = false;
    out << "    {\"name\": \"" << slice.name << "\", \"complete\": "
        << (slice.complete ? "true" : "false");
    out << ", \"bypassable\": " << (bypassable ? "true" : "false");
    out << ", \"bypassed\": " << (bypassed ? "true" : "false");
    out << ", \"depth\": "
        << std::max<int64_t>(0, getNodeAttrInt(hwNode, "depth", 0));
    out << "}";
  }
  out << "\n  ],\n";

  out << "  \"temporal_registers\": [\n";
  first = true;
  for (const auto &pe : flattener.getPEContainment()) {
    if (pe.peKind != "temporal_pe")
      continue;
    auto temporalPlan = buildTemporalConfigPlan(pe, state, dfg, adg, edgeKinds);
    for (const auto &binding : temporalPlan.registerBindings) {
      if (!first)
        out << ",\n";
      first = false;
      out << "    {\"pe_name\": \"" << binding.peName << "\"";
      out << ", \"sw_edge\": " << binding.swEdgeId;
      out << ", \"register_index\": " << binding.registerIndex;
      out << ", \"writer_sw_node\": " << binding.writerSwNode;
      out << ", \"reader_sw_node\": " << binding.readerSwNode;
      out << ", \"writer_hw_node\": " << binding.writerHwNode;
      out << ", \"reader_hw_node\": " << binding.readerHwNode;
      out << ", \"writer_output_index\": " << binding.writerOutputIndex;
      out << ", \"reader_input_index\": " << binding.readerInputIndex << "}";
    }
  }
  out << "\n  ],\n";

  out << "  \"memory_regions\": [\n";
  first = true;
  for (IdIndex hwId = 0; hwId < static_cast<IdIndex>(adg.nodes.size());
       ++hwId) {
    const Node *hwNode = adg.getNode(hwId);
    if (!hwNode || getNodeAttrStr(hwNode, "resource_class") != "memory")
      continue;

    if (!first)
      out << ",\n";
    first = false;

    auto regions = collectMemoryRegionsForNode(hwId, state, dfg, adg);
    auto addrTable = buildAddrOffsetTable(hwNode, hwId, state, dfg, adg);

    out << "    {\"hw_node\": " << hwId;
    out << ", \"hw_name\": \"" << getNodeAttrStr(hwNode, "op_name") << "\"";
    out << ", \"memory_kind\": \"" << getNodeAttrStr(hwNode, "op_kind")
        << "\"";
    out << ", \"num_region\": " << getNodeAttrInt(hwNode, "numRegion", 1);
    out << ", \"regions\": [";
    for (size_t i = 0; i < regions.size(); ++i) {
      if (i > 0)
        out << ", ";
      out << "{\"region_index\": " << i;
      out << ", \"sw_node\": " << regions[i].swNode;
      out << ", \"memref_arg_index\": " << regions[i].memrefArgIndex;
      out << ", \"start_lane\": " << regions[i].startLane;
      out << ", \"end_lane\": " << regions[i].endLane;
      out << ", \"ld_count\": " << regions[i].ldCount;
      out << ", \"st_count\": " << regions[i].stCount;
      out << ", \"elem_size_log2\": " << regions[i].elemSizeLog2 << "}";
    }
    out << "], \"addr_offset_table\": [";
    for (size_t i = 0; i < addrTable.size(); ++i) {
      if (i > 0)
        out << ", ";
      out << addrTable[i];
    }
    out << "]}";
  }

  out << "\n  ]\n";
  out << "}\n";
}

} // namespace loom
