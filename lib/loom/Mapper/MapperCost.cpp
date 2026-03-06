//===-- MapperCost.cpp - Cost computation for mapper --------------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include "loom/Mapper/Mapper.h"

#include "mlir/IR/BuiltinAttributes.h"

#include <cmath>

namespace loom {

namespace {

llvm::StringRef getResClass(const Node *node) {
  for (auto &attr : node->attributes) {
    if (attr.getName() == "resource_class") {
      if (auto strAttr = mlir::dyn_cast<mlir::StringAttr>(attr.getValue()))
        return strAttr.getValue();
    }
  }
  return "";
}

int64_t getIntAttr(const Node *node, llvm::StringRef name, int64_t dflt = 0) {
  for (auto &attr : node->attributes) {
    if (attr.getName() == name) {
      if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(attr.getValue()))
        return intAttr.getInt();
    }
  }
  return dflt;
}

bool hasAttr(const Node *node, llvm::StringRef name) {
  for (auto &attr : node->attributes) {
    if (attr.getName() == name)
      return true;
  }
  return false;
}

} // namespace

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
}

void Mapper::computeCost(MappingState &state, const Graph &dfg,
                         const Graph &adg, const Options &opts) {
  double weights[5];
  getWeights(opts.profile, weights);

  // --- Placement Pressure ---
  // sum_tile (occ(tile) / cap(tile))^2, normalized by HW node count.
  double placementPressure = 0.0;
  for (IdIndex i = 0; i < static_cast<IdIndex>(state.hwNodeToSwNodes.size());
       ++i) {
    double occ = static_cast<double>(state.hwNodeToSwNodes[i].size());
    if (occ > 0) {
      double cap = 1.0;
      const Node *hwNode = adg.getNode(i);
      if (hwNode) {
        int64_t numInst = getIntAttr(hwNode, "num_instruction", 0);
        if (numInst > 0)
          cap = static_cast<double>(numInst);
      }
      double ratio = occ / cap;
      placementPressure += ratio * ratio;
    }
  }
  size_t hwNodeCount = adg.countNodes();
  if (hwNodeCount > 0)
    placementPressure /= static_cast<double>(hwNodeCount);
  state.placementPressure = placementPressure;

  // --- Routing Cost ---
  // Average path length in port pairs (hop count).
  double routingCost = 0.0;
  for (const auto &path : state.swEdgeToHwPaths) {
    if (!path.empty())
      routingCost += static_cast<double>(path.size()) / 2.0;
  }
  size_t edgeCount = dfg.countEdges();
  if (edgeCount > 0)
    routingCost /= static_cast<double>(edgeCount);
  state.routingCost = routingCost;

  // --- Temporal Cost ---
  // 0.5 * slotUtil + 0.3 * regPressure
  double slotUtil = 0.0;
  double regPressure = 0.0;
  uint32_t usedSlots = 0;
  uint32_t totalSlots = 0;
  uint32_t usedRegs = 0;
  uint32_t totalRegs = 0;

  // Collect temporal PE capacities and usage.
  llvm::DenseMap<IdIndex, unsigned> tpeOpCount;
  for (IdIndex swId = 0;
       swId < static_cast<IdIndex>(state.swNodeToHwNode.size()); ++swId) {
    if (state.temporalPEAssignments.size() > swId &&
        state.temporalPEAssignments[swId].slot != INVALID_ID) {
      ++usedSlots;
      IdIndex hwId = state.swNodeToHwNode[swId];
      const Node *hwNode = adg.getNode(hwId);
      if (hwNode) {
        int64_t parent = getIntAttr(hwNode, "parent_temporal_pe", -1);
        if (parent >= 0)
          tpeOpCount[static_cast<IdIndex>(parent)]++;
      }
    }
  }

  for (auto &[tpeId, count] : tpeOpCount) {
    const Node *tpeNode = adg.getNode(tpeId);
    if (tpeNode)
      totalSlots += getIntAttr(tpeNode, "num_instruction", 0);
  }

  for (IdIndex r : state.registerAssignments) {
    if (r != INVALID_ID)
      ++usedRegs;
  }

  for (auto &[tpeId, count] : tpeOpCount) {
    const Node *tpeNode = adg.getNode(tpeId);
    if (tpeNode)
      totalRegs += getIntAttr(tpeNode, "num_register", 0);
  }

  if (totalSlots > 0)
    slotUtil = static_cast<double>(usedSlots) / static_cast<double>(totalSlots);
  if (totalRegs > 0)
    regPressure =
        static_cast<double>(usedRegs) / static_cast<double>(totalRegs);

  state.temporalCost = 0.5 * slotUtil + 0.3 * regPressure;

  // --- Performance Proxy ---
  // critical_path_est + ii_pressure + queue_pressure
  double maxPath = 0.0;
  for (const auto &path : state.swEdgeToHwPaths) {
    double pathLen = static_cast<double>(path.size()) / 2.0;
    if (pathLen > maxPath)
      maxPath = pathLen;
  }
  state.criticalPathEst = maxPath;

  // ii_pressure: ratio of max temporal PE utilization to capacity.
  double iiPressure = 0.0;
  for (auto &[tpeId, count] : tpeOpCount) {
    const Node *tpeNode = adg.getNode(tpeId);
    if (!tpeNode)
      continue;
    int64_t numInst = getIntAttr(tpeNode, "num_instruction", 1);
    double util = static_cast<double>(count) / std::max(1.0, static_cast<double>(numInst));
    if (util > iiPressure)
      iiPressure = util;
  }
  state.iiPressure = iiPressure;

  // queue_pressure: ratio of memory queue usage to capacity.
  double queuePressure = 0.0;
  for (IdIndex i = 0; i < static_cast<IdIndex>(state.hwNodeToSwNodes.size());
       ++i) {
    const Node *hwNode = adg.getNode(i);
    if (!hwNode)
      continue;
    llvm::StringRef rc = getResClass(hwNode);
    if (rc != "memory")
      continue;
    int64_t lsqDepth = getIntAttr(hwNode, "lsqDepth", 0);
    if (lsqDepth <= 0)
      continue;
    double usageRatio = static_cast<double>(state.hwNodeToSwNodes[i].size()) /
                        static_cast<double>(lsqDepth);
    if (usageRatio > queuePressure)
      queuePressure = usageRatio;
  }
  state.queuePressure = queuePressure;

  state.perfProxyCost = state.criticalPathEst;
  if (edgeCount > 0)
    state.perfProxyCost /= static_cast<double>(edgeCount);
  state.perfProxyCost += 0.3 * iiPressure + 0.2 * queuePressure;

  // --- Config Footprint ---
  // non_default_words / total_config_words.
  uint32_t nonDefault = 0;
  uint32_t totalConfig = 0;
  for (IdIndex i = 0; i < static_cast<IdIndex>(state.hwNodeToSwNodes.size());
       ++i) {
    const Node *hwNode = adg.getNode(i);
    if (!hwNode || hwNode->kind != Node::OperationNode)
      continue;
    // Estimate config words per node type.
    llvm::StringRef rc = getResClass(hwNode);
    if (rc == "functional" || rc == "routing") {
      ++totalConfig;
      if (!state.hwNodeToSwNodes[i].empty())
        ++nonDefault;
    }
  }
  state.nonDefaultWords = nonDefault;
  state.totalConfigWords = totalConfig;
  state.configFootprint =
      totalConfig > 0
          ? static_cast<double>(nonDefault) / static_cast<double>(totalConfig)
          : 0.0;

  // --- Total Cost ---
  state.totalCost = weights[0] * state.placementPressure +
                    weights[1] * state.routingCost +
                    weights[2] * state.temporalCost +
                    weights[3] * state.perfProxyCost +
                    weights[4] * state.configFootprint;
}

} // namespace loom
