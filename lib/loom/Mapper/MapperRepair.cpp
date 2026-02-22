//===-- MapperRepair.cpp - Repair and refinement for mapper --------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include "loom/Mapper/Mapper.h"

#include <random>

namespace loom {

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
      if (attempt < 5) {
        // Rip-up and reroute: remove conflicting edge routes and retry.
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

          auto path = findPath(srcHwPort, dstHwPort, state, adg);
          if (!path.empty()) {
            state.mapEdge(edgeId, path, dfg, adg);
          }
        }
      } else if (attempt < 8) {
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
              // Remap ports.
              const Node *sw = dfg.getNode(swNode);
              const Node *hw = adg.getNode(cand.hwNodeId);
              if (sw && hw) {
                for (size_t i = 0; i < sw->inputPorts.size() &&
                                    i < hw->inputPorts.size();
                     ++i) {
                  state.mapPort(sw->inputPorts[i], hw->inputPorts[i],
                                dfg, adg);
                }
                for (size_t i = 0; i < sw->outputPorts.size() &&
                                    i < hw->outputPorts.size();
                     ++i) {
                  state.mapPort(sw->outputPorts[i], hw->outputPorts[i],
                                dfg, adg);
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
      // Re-route all edges.
      for (IdIndex edgeId = 0;
           edgeId < static_cast<IdIndex>(dfg.edges.size()); ++edgeId) {
        if (edgeId < state.swEdgeToHwPaths.size())
          state.swEdgeToHwPaths[edgeId].clear();
      }
      runRouting(state, dfg, adg);
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
