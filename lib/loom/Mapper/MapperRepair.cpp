//===-- MapperRepair.cpp - Repair and refinement for mapper --------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include "loom/Mapper/Mapper.h"

#include "mlir/IR/BuiltinAttributes.h"

#include <random>

namespace loom {

namespace {
/// Get op_name string attribute from a node.
llvm::StringRef getNodeOpNameRepair(const Node *node) {
  for (auto &attr : node->attributes)
    if (attr.getName() == "op_name")
      if (auto s = mlir::dyn_cast<mlir::StringAttr>(attr.getValue()))
        return s.getValue();
  return "";
}

/// Check if a node is a memory operation.
bool isMemoryOpRepair(const Node *node) {
  llvm::StringRef opName = getNodeOpNameRepair(node);
  return opName.contains("load") || opName.contains("store") ||
         opName.contains("memory");
}
} // namespace

bool Mapper::runRefinement(MappingState &state, const Graph &dfg,
                           const Graph &adg, const CandidateSet &candidates,
                           const Options &opts) {
  // Repair loop with bounded iterations.
  for (int globalRestart = 0; globalRestart < opts.maxGlobalRestarts;
       ++globalRestart) {

    // Save checkpoint before repair attempt.
    auto checkpoint = state.save();

    for (int attempt = 0; attempt < opts.maxLocalRepairs; ++attempt) {
      // Find unrouted edges.
      llvm::SmallVector<IdIndex, 8> failedEdges;
      for (IdIndex edgeId = 0;
           edgeId < static_cast<IdIndex>(dfg.edges.size()); ++edgeId) {
        const Edge *edge = dfg.getEdge(edgeId);
        if (!edge)
          continue;

        if (edgeId >= state.swEdgeToHwPaths.size() ||
            state.swEdgeToHwPaths[edgeId].empty()) {
          failedEdges.push_back(edgeId);
        }
      }

      if (failedEdges.empty())
        return true; // All edges routed.

      // Strategy selection based on attempt number.
      if (attempt < 6) {
        // Rip-up and reroute: for each failed edge, remove SW edges that
        // block the needed HW path, route the failed edge, then re-route
        // the ripped-up edges.
        for (IdIndex edgeId : failedEdges) {
          const Edge *edge = dfg.getEdge(edgeId);
          if (!edge)
            continue;

          IdIndex srcSwPort = edge->srcPort;
          IdIndex dstSwPort = edge->dstPort;

          if (srcSwPort >= state.swPortToHwPort.size() ||
              dstSwPort >= state.swPortToHwPort.size())
            continue;

          IdIndex srcHwPort = state.swPortToHwPort[srcSwPort];
          IdIndex dstHwPort = state.swPortToHwPort[dstSwPort];

          if (srcHwPort == INVALID_ID || dstHwPort == INVALID_ID)
            continue;

          // First try without rip-up.
          auto path = findPath(srcHwPort, dstHwPort, state, adg);
          if (!path.empty()) {
            state.mapEdge(edgeId, path, dfg, adg);
            continue;
          }

          // Rip-up: find SW edges that use HW edges on the potential path
          // to the destination. Collect SW edges connected to the dest
          // HW port's incoming HW edges.
          llvm::SmallVector<IdIndex, 4> rippedEdges;
          const Port *dstP = adg.getPort(dstHwPort);
          if (dstP) {
            for (IdIndex hwEdgeId : dstP->connectedEdges) {
              if (hwEdgeId >= state.hwEdgeToSwEdges.size())
                continue;
              for (IdIndex swEdge : state.hwEdgeToSwEdges[hwEdgeId]) {
                if (swEdge != edgeId)
                  rippedEdges.push_back(swEdge);
              }
            }
          }

          // Unmap the blocking edges.
          for (IdIndex ripId : rippedEdges)
            state.unmapEdge(ripId, dfg, adg);

          // Try routing the failed edge again.
          path = findPath(srcHwPort, dstHwPort, state, adg);
          if (!path.empty())
            state.mapEdge(edgeId, path, dfg, adg);

          // Re-route the ripped-up edges.
          for (IdIndex ripId : rippedEdges) {
            if (ripId >= state.swEdgeToHwPaths.size() ||
                !state.swEdgeToHwPaths[ripId].empty())
              continue;
            const Edge *ripEdge = dfg.getEdge(ripId);
            if (!ripEdge)
              continue;
            IdIndex rSrc = state.swPortToHwPort[ripEdge->srcPort];
            IdIndex rDst = state.swPortToHwPort[ripEdge->dstPort];
            if (rSrc == INVALID_ID || rDst == INVALID_ID)
              continue;
            auto ripPath = findPath(rSrc, rDst, state, adg);
            if (!ripPath.empty())
              state.mapEdge(ripId, ripPath, dfg, adg);
          }
        }
      } else if (attempt < 9) {
        // Node migration: try moving a node to an alternative candidate.
        for (IdIndex edgeId : failedEdges) {
          const Edge *edge = dfg.getEdge(edgeId);
          if (!edge)
            continue;

          const Port *dstPort = dfg.getPort(edge->dstPort);
          if (!dstPort)
            continue;

          IdIndex swNode = dstPort->parentNode;
          if (swNode == INVALID_ID)
            continue;

          auto candIt = candidates.find(swNode);
          if (candIt == candidates.end())
            continue;

          // Try alternative candidates.
          IdIndex currentHw = state.swNodeToHwNode[swNode];
          for (const auto &cand : candIt->second) {
            if (cand.hwNodeId == currentHw)
              continue;

            // Unmap and try new placement.
            state.unmapNode(swNode, dfg, adg);
            auto result = state.mapNode(swNode, cand.hwNodeId, dfg, adg);
            if (result == ActionResult::Success) {
              // Remap ports using type-aware matching, skipping
              // HW ports already used by other SW nodes.
              const Node *sw = dfg.getNode(swNode);
              const Node *hw = adg.getNode(cand.hwNodeId);
              if (sw && hw) {
                if (sw->inputPorts.size() <= hw->inputPorts.size()) {
                  for (size_t si = 0; si < sw->inputPorts.size(); ++si) {
                    const Port *sp = dfg.getPort(sw->inputPorts[si]);
                    if (!sp) continue;
                    for (size_t hi = 0; hi < hw->inputPorts.size(); ++hi) {
                      IdIndex hwPid = hw->inputPorts[hi];
                      if (!state.hwPortToSwPorts[hwPid].empty()) continue;
                      const Port *hp = adg.getPort(hwPid);
                      if (hp && sp->type == hp->type) {
                        state.mapPort(sw->inputPorts[si], hwPid,
                                      dfg, adg);
                        break;
                      }
                    }
                  }
                } else {
                  for (size_t i = 0; i < sw->inputPorts.size() &&
                                      i < hw->inputPorts.size(); ++i) {
                    state.mapPort(sw->inputPorts[i], hw->inputPorts[i],
                                  dfg, adg);
                  }
                }
                if (sw->outputPorts.size() <= hw->outputPorts.size()) {
                  for (size_t si = 0; si < sw->outputPorts.size(); ++si) {
                    const Port *sp = dfg.getPort(sw->outputPorts[si]);
                    if (!sp) continue;
                    for (size_t hi = 0; hi < hw->outputPorts.size(); ++hi) {
                      IdIndex hwPid = hw->outputPorts[hi];
                      if (!state.hwPortToSwPorts[hwPid].empty()) continue;
                      const Port *hp = adg.getPort(hwPid);
                      if (hp && sp->type == hp->type) {
                        state.mapPort(sw->outputPorts[si], hwPid,
                                      dfg, adg);
                        break;
                      }
                    }
                  }
                } else {
                  for (size_t i = 0; i < sw->outputPorts.size() &&
                                      i < hw->outputPorts.size(); ++i) {
                    state.mapPort(sw->outputPorts[i], hw->outputPorts[i],
                                  dfg, adg);
                  }
                }
              }
              break;
            }
            // Restore if failed.
            state.mapNode(swNode, currentHw, dfg, adg);
          }
        }
      }
    }

    // If we still have failures after max local repairs, try global restart.
    if (globalRestart + 1 < opts.maxGlobalRestarts) {
      state.restore(checkpoint);
      // Clear all edge routing state (both forward and reverse mappings)
      // before re-routing from scratch.
      for (IdIndex edgeId = 0;
           edgeId < static_cast<IdIndex>(dfg.edges.size()); ++edgeId) {
        if (edgeId < state.swEdgeToHwPaths.size())
          state.swEdgeToHwPaths[edgeId].clear();
      }
      for (auto &swEdges : state.hwEdgeToSwEdges)
        swEdges.clear();
      // Use a different seed for each restart to vary routing order.
      runRouting(state, dfg, adg,
                 opts.seed + globalRestart + 1);
    }
  }

  // Check if all edges are routed.
  for (IdIndex edgeId = 0; edgeId < static_cast<IdIndex>(dfg.edges.size());
       ++edgeId) {
    const Edge *edge = dfg.getEdge(edgeId);
    if (!edge)
      continue;
    if (edgeId >= state.swEdgeToHwPaths.size() ||
        state.swEdgeToHwPaths[edgeId].empty())
      return false;
  }

  return true;
}

} // namespace loom
