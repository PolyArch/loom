//===-- MapperRepair.cpp - Repair and refinement for mapper --------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include "loom/Mapper/Mapper.h"
#include "loom/Mapper/TypeCompat.h"

#include "loom/Dialect/Dataflow/DataflowTypes.h"
#include "loom/Hardware/Common/FabricConstants.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

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

/// Get the "resource_class" attribute from a node, or empty string.
llvm::StringRef getNodeResourceClassRepair(const Node *node) {
  for (auto &attr : node->attributes)
    if (attr.getName() == "resource_class")
      if (auto s = mlir::dyn_cast<mlir::StringAttr>(attr.getValue()))
        return s.getValue();
  return "";
}

/// Get bridge boundary port arrays from a node's attributes.
bool getBridgePortsRepair(const Node *node,
                          mlir::DenseI32ArrayAttr &bridgeInPorts,
                          mlir::DenseI32ArrayAttr &bridgeOutPorts) {
  bridgeInPorts = {};
  bridgeOutPorts = {};
  for (auto &attr : node->attributes) {
    if (attr.getName() == "bridge_input_ports")
      bridgeInPorts =
          mlir::dyn_cast<mlir::DenseI32ArrayAttr>(attr.getValue());
    else if (attr.getName() == "bridge_output_ports")
      bridgeOutPorts =
          mlir::dyn_cast<mlir::DenseI32ArrayAttr>(attr.getValue());
  }
  return bridgeInPorts || bridgeOutPorts;
}

/// Get an integer attribute from a node.
int64_t getIntAttrRepair(const Node *node, llvm::StringRef name,
                         int64_t dflt = -1) {
  for (auto &attr : node->attributes)
    if (attr.getName() == name)
      if (auto ia = mlir::dyn_cast<mlir::IntegerAttr>(attr.getValue()))
        return ia.getInt();
  return dflt;
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

      if (failedEdges.empty()) {
        if (log)
          log->logRefinement(attempt, 0, true);
        return true; // All edges routed.
      }

      if (log)
        log->logRefinement(attempt,
                           static_cast<unsigned>(failedEdges.size()),
                           false);

      // Strategy selection based on attempt number.
      if (attempt < 6) {
        // Path-aware rip-up: use relaxed BFS to find the shortest path
        // ignoring exclusivity, then rip up all blocking SW edges along
        // that path. This targets the actual bottleneck, not just the
        // destination port.
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
          auto path = findPath(srcHwPort, dstHwPort, state, adg,
                               &dfg, edgeId);
          if (!path.empty()) {
            state.mapEdge(edgeId, path, dfg, adg);
            continue;
          }

          // Use relaxed BFS to find blocking edges along the shortest
          // path (ignoring C3 exclusivity).
          llvm::SmallVector<IdIndex, 8> blockingEdges;
          auto relaxedPath = findPathRelaxed(srcHwPort, dstHwPort,
                                             state, adg, blockingEdges);

          llvm::SmallVector<IdIndex, 4> rippedEdges;
          if (!relaxedPath.empty() && !blockingEdges.empty()) {
            // Rip up all blocking edges along the relaxed path.
            for (IdIndex swEdge : blockingEdges) {
              if (swEdge != edgeId)
                rippedEdges.push_back(swEdge);
            }
          } else {
            // Fallback: rip edges at the destination port.
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
          }

          // Unmap the blocking edges.
          for (IdIndex ripId : rippedEdges)
            state.unmapEdge(ripId, dfg, adg);

          // Try routing the failed edge again.
          path = findPath(srcHwPort, dstHwPort, state, adg,
                          &dfg, edgeId);
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
            auto ripPath = findPath(rSrc, rDst, state, adg,
                                    &dfg, ripId);
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
              // Remap ports: positional for PE/temporal PE,
              // type-based inputs + positional outputs for memory.
              const Node *sw = dfg.getNode(swNode);
              const Node *hw = adg.getNode(cand.hwNodeId);
              if (sw && hw) {
                bool isMemory =
                    (getNodeResourceClassRepair(hw) == "memory");

                mlir::DenseI32ArrayAttr bridgeInPR, bridgeOutPR;
                bool hasBridgeR = isMemory &&
                    getBridgePortsRepair(hw, bridgeInPR, bridgeOutPR);

                if (hasBridgeR) {
                  // Bridge memory: category-aware binding.
                  unsigned swInSkipR = 0;
                  if (getNodeOpNameRepair(hw) == "fabric.extmemory" &&
                      !sw->inputPorts.empty()) {
                    state.mapPort(sw->inputPorts[0],
                                  hw->inputPorts[0], dfg, adg);
                    swInSkipR = 1;
                  }
                  int32_t storeInCntR = getIntAttrRepair(
                      hw, "bridge_store_input_count", -1);
                  int32_t ldDataOutCntR = getIntAttrRepair(
                      hw, "bridge_ld_data_output_count", -1);

                  if (bridgeInPR) {
                    unsigned stBoundR =
                        (storeInCntR > 0)
                            ? static_cast<unsigned>(storeInCntR) : 0;
                    unsigned inSzR =
                        static_cast<unsigned>(bridgeInPR.size());
                    // Detect DFG store inputs by pair pattern.
                    unsigned dfgStIns = 0;
                    if (stBoundR >= 2) {
                      const Port *sdRef = adg.getPort(
                          static_cast<IdIndex>(bridgeInPR[0]));
                      for (size_t si = swInSkipR;
                           si + 1 < sw->inputPorts.size(); si += 2) {
                        const Port *sp =
                            dfg.getPort(sw->inputPorts[si]);
                        if (sp && sdRef &&
                            isTypeWidthCompatible(sp->type,
                                                  sdRef->type))
                          dfgStIns += 2;
                        else
                          break;
                      }
                    }
                    for (size_t p = swInSkipR;
                         p < sw->inputPorts.size(); ++p) {
                      const Port *sp = dfg.getPort(sw->inputPorts[p]);
                      if (!sp) continue;
                      unsigned rel = p - swInSkipR;
                      bool isSt = (rel < dfgStIns);
                      unsigned lo = isSt ? 0 : stBoundR;
                      unsigned hi = isSt ? stBoundR : inSzR;
                      bool found = false;
                      for (unsigned bi = lo; bi < hi; ++bi) {
                        auto hwPid =
                            static_cast<IdIndex>(bridgeInPR[bi]);
                        if (!state.hwPortToSwPorts[hwPid].empty())
                          continue;
                        const Port *hp = adg.getPort(hwPid);
                        if (hp && isTypeWidthCompatible(
                                      sp->type, hp->type)) {
                          state.mapPort(sw->inputPorts[p], hwPid,
                                        dfg, adg);
                          found = true;
                          break;
                        }
                      }
                      if (!found) {
                        for (unsigned bi = 0; bi < inSzR; ++bi) {
                          auto hwPid =
                              static_cast<IdIndex>(bridgeInPR[bi]);
                          if (!state.hwPortToSwPorts[hwPid].empty())
                            continue;
                          const Port *hp = adg.getPort(hwPid);
                          if (hp && isTypeWidthCompatible(
                                        sp->type, hp->type)) {
                            state.mapPort(sw->inputPorts[p], hwPid,
                                          dfg, adg);
                            break;
                          }
                        }
                      }
                    }
                  }
                  if (bridgeOutPR) {
                    int ldOutR =
                        (ldDataOutCntR > 0) ? ldDataOutCntR : 0;
                    int ldDoneStR = ldOutR;
                    int stDoneStR = ldOutR * 2;
                    int totR =
                        static_cast<int>(bridgeOutPR.size());
                    // Count DFG ld_data outputs.
                    unsigned dfgLdD = 0;
                    const Port *ldDRef =
                        (ldOutR > 0) ? adg.getPort(
                            static_cast<IdIndex>(bridgeOutPR[0]))
                                     : nullptr;
                    for (size_t i = 0; i < sw->outputPorts.size();
                         ++i) {
                      const Port *sp =
                          dfg.getPort(sw->outputPorts[i]);
                      if (sp && ldDRef &&
                          isTypeWidthCompatible(sp->type,
                                                ldDRef->type))
                        ++dfgLdD;
                      else
                        break;
                    }
                    for (size_t p = 0; p < sw->outputPorts.size();
                         ++p) {
                      const Port *sp =
                          dfg.getPort(sw->outputPorts[p]);
                      if (!sp) continue;
                      int lo, hi;
                      if (p < dfgLdD) {
                        lo = 0; hi = ldDoneStR;
                      } else if (p < dfgLdD * 2) {
                        lo = ldDoneStR; hi = stDoneStR;
                      } else {
                        lo = stDoneStR; hi = totR;
                      }
                      bool found = false;
                      for (int bi = lo; bi < hi; ++bi) {
                        auto hwPid =
                            static_cast<IdIndex>(bridgeOutPR[bi]);
                        if (!state.hwPortToSwPorts[hwPid].empty())
                          continue;
                        const Port *hp = adg.getPort(hwPid);
                        if (hp && isTypeWidthCompatible(
                                      sp->type, hp->type)) {
                          state.mapPort(sw->outputPorts[p], hwPid,
                                        dfg, adg);
                          found = true;
                          break;
                        }
                      }
                      if (!found) {
                        for (int bi = 0; bi < totR; ++bi) {
                          auto hwPid =
                              static_cast<IdIndex>(bridgeOutPR[bi]);
                          if (!state.hwPortToSwPorts[hwPid].empty())
                            continue;
                          const Port *hp = adg.getPort(hwPid);
                          if (hp && isTypeWidthCompatible(
                                        sp->type, hp->type)) {
                            state.mapPort(sw->outputPorts[p], hwPid,
                                          dfg, adg);
                            break;
                          }
                        }
                      }
                    }
                  }
                } else if (isMemory) {
                  // Single-port memory: type-based greedy search.
                  llvm::DenseSet<IdIndex> usedInNode;
                  for (size_t si = 0; si < sw->inputPorts.size(); ++si) {
                    const Port *sp = dfg.getPort(sw->inputPorts[si]);
                    if (!sp) continue;
                    for (size_t hi = 0; hi < hw->inputPorts.size();
                         ++hi) {
                      IdIndex hwPid = hw->inputPorts[hi];
                      if (!state.hwPortToSwPorts[hwPid].empty())
                        continue;
                      if (usedInNode.count(hwPid))
                        continue;
                      const Port *hp = adg.getPort(hwPid);
                      if (hp &&
                          isTypeWidthCompatible(sp->type, hp->type)) {
                        state.mapPort(sw->inputPorts[si], hwPid,
                                      dfg, adg);
                        usedInNode.insert(hwPid);
                        break;
                      }
                    }
                  }
                  // Memory outputs: positional.
                  for (size_t i = 0;
                       i < sw->outputPorts.size() &&
                       i < hw->outputPorts.size();
                       ++i) {
                    state.mapPort(sw->outputPorts[i],
                                  hw->outputPorts[i], dfg, adg);
                  }
                } else {
                  // PE / temporal PE FU: positional mapping.
                  for (size_t i = 0;
                       i < sw->inputPorts.size() &&
                       i < hw->inputPorts.size();
                       ++i) {
                    state.mapPort(sw->inputPorts[i],
                                  hw->inputPorts[i], dfg, adg);
                  }
                  for (size_t i = 0;
                       i < sw->outputPorts.size() &&
                       i < hw->outputPorts.size();
                       ++i) {
                    state.mapPort(sw->outputPorts[i],
                                  hw->outputPorts[i], dfg, adg);
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
      if (log)
        log->info("Global restart " + std::to_string(globalRestart + 1) +
                  "/" + std::to_string(opts.maxGlobalRestarts));
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
