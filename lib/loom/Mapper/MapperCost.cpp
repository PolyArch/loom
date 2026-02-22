//===-- MapperCost.cpp - Cost computation for mapper --------------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include "loom/Mapper/Mapper.h"

#include "mlir/IR/BuiltinAttributes.h"

#include <cmath>

namespace loom {

void Mapper::getWeights(const std::string &profile, double weights[5]) {
  // Default: balanced.
  weights[0] = 1.0; // placementPressure
  weights[1] = 1.0; // routingCost
  weights[2] = 0.5; // temporalCost
  weights[3] = 0.5; // perfProxyCost
  weights[4] = 0.1; // configFootprint

  if (profile == "throughput_first") {
    weights[0] = 0.3;
    weights[1] = 0.5;
    weights[2] = 0.3;
    weights[3] = 2.0;
    weights[4] = 0.1;
  } else if (profile == "area_power_first") {
    weights[0] = 2.0;
    weights[1] = 0.5;
    weights[2] = 0.5;
    weights[3] = 0.2;
    weights[4] = 1.0;
  } else if (profile == "deterministic_debug") {
    weights[0] = 1.0;
    weights[1] = 1.0;
    weights[2] = 0.5;
    weights[3] = 0.0;
    weights[4] = 0.0;
  }
  // "balanced", "cpsat_full", "heuristic_only" all use default weights.
}

void Mapper::computeCost(MappingState &state, const Graph &dfg,
                         const Graph &adg, const Options &opts) {
  double weights[5];
  getWeights(opts.profile, weights);

  // --- Placement Pressure ---
  // sum_tile (occ(tile) / cap(tile))^2
  double placementPressure = 0.0;
  for (IdIndex i = 0; i < static_cast<IdIndex>(state.hwNodeToSwNodes.size());
       ++i) {
    double occ = static_cast<double>(state.hwNodeToSwNodes[i].size());
    if (occ > 0) {
      double cap = 1.0; // Default capacity.
      const Node *hwNode = adg.getNode(i);
      if (hwNode) {
        for (auto &attr : hwNode->attributes) {
          if (attr.getName() == "num_instruction") {
            if (auto ia = mlir::dyn_cast<mlir::IntegerAttr>(attr.getValue()))
              cap = std::max(1.0, static_cast<double>(ia.getInt()));
          }
        }
      }
      double ratio = occ / cap;
      placementPressure += ratio * ratio;
    }
  }

  // Normalize by number of HW nodes.
  size_t hwNodeCount = adg.countNodes();
  if (hwNodeCount > 0)
    placementPressure /= static_cast<double>(hwNodeCount);

  state.placementPressure = placementPressure;

  // --- Routing Cost ---
  // sum of path lengths (hop count).
  double routingCost = 0.0;
  for (const auto &path : state.swEdgeToHwPaths) {
    if (!path.empty()) {
      routingCost += static_cast<double>(path.size()) / 2.0; // port pairs.
    }
  }
  size_t edgeCount = dfg.countEdges();
  if (edgeCount > 0)
    routingCost /= static_cast<double>(edgeCount);

  state.routingCost = routingCost;

  // --- Temporal Cost ---
  double temporalCost = 0.0;
  double slotUtil = 0.0;
  double regPressure = 0.0;
  uint32_t usedSlots = 0;
  uint32_t usedRegs = 0;

  for (const auto &tpa : state.temporalPEAssignments) {
    if (tpa.slot != INVALID_ID)
      ++usedSlots;
  }
  for (IdIndex r : state.registerAssignments) {
    if (r != INVALID_ID)
      ++usedRegs;
  }

  size_t swNodeCount = dfg.countNodes();
  if (swNodeCount > 0) {
    slotUtil = static_cast<double>(usedSlots) / static_cast<double>(swNodeCount);
    regPressure = static_cast<double>(usedRegs) / static_cast<double>(swNodeCount);
  }
  temporalCost = 0.5 * slotUtil + 0.3 * regPressure;
  state.temporalCost = temporalCost;

  // --- Performance Proxy ---
  // Simple estimate: max path length as critical path.
  double maxPath = 0.0;
  for (const auto &path : state.swEdgeToHwPaths) {
    double pathLen = static_cast<double>(path.size()) / 2.0;
    if (pathLen > maxPath)
      maxPath = pathLen;
  }
  state.criticalPathEst = maxPath;
  state.iiPressure = 0.0; // TODO: compute from temporal utilization
  state.queuePressure = 0.0;
  state.perfProxyCost = maxPath;
  if (edgeCount > 0)
    state.perfProxyCost /= static_cast<double>(edgeCount);

  // --- Config Footprint ---
  state.nonDefaultWords = 0;
  state.totalConfigWords = 0;
  state.configFootprint = 0.0;

  // --- Total Cost ---
  state.totalCost = weights[0] * state.placementPressure +
                    weights[1] * state.routingCost +
                    weights[2] * state.temporalCost +
                    weights[3] * state.perfProxyCost +
                    weights[4] * state.configFootprint;
}

} // namespace loom
