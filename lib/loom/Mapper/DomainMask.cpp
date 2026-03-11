//===-- DomainMask.cpp - Domain-resource masking implementation ----*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include "loom/Mapper/DomainMask.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/BuiltinAttributes.h"

#include <algorithm>

#define DEBUG_TYPE "domain-mask"

namespace loom {

namespace {

/// Get the string value of a named attribute on a node.
llvm::StringRef getAttrStr(const Node *node, llvm::StringRef name) {
  for (auto &attr : node->attributes) {
    if (attr.getName() == name) {
      if (auto strAttr = mlir::dyn_cast<mlir::StringAttr>(attr.getValue()))
        return strAttr.getValue();
    }
  }
  return "";
}

/// Return true if a node is a removable functional PE instance.
bool isRemovableFunctional(const Node *node) {
  if (!node)
    return false;
  if (node->kind == Node::ModuleInputNode ||
      node->kind == Node::ModuleOutputNode)
    return false;
  llvm::StringRef resClass = getAttrStr(node, "resource_class");
  if (resClass != "functional")
    return false;
  // Virtual temporal PE nodes own ports shared by FU sub-nodes.
  if (getAttrStr(node, "is_virtual") == "true")
    return false;
  return true;
}

} // anonymous namespace

void pruneDomainADG(Graph &adg, const Graph &dfg, unsigned minCandidates) {
  // Run tech-mapping on the full ADG to discover which HW nodes are
  // compatible with each DFG operation.
  TechMapper techMapper;
  CandidateSet candidates = techMapper.map(dfg, adg);

  // Count DFG operations that have at least one functional candidate.
  llvm::DenseSet<IdIndex> dfgNeedsFunctional;
  for (auto &[swId, candList] : candidates) {
    for (auto &cand : candList) {
      const Node *hw = adg.getNode(cand.hwNodeId);
      if (hw && getAttrStr(hw, "resource_class") == "functional") {
        dfgNeedsFunctional.insert(swId);
        break;
      }
    }
  }
  unsigned demand = dfgNeedsFunctional.size();

  // Build reverse map: hwNodeId -> set of DFG nodes that use it as candidate.
  llvm::DenseMap<IdIndex, llvm::DenseSet<IdIndex>> reverseMap;
  for (auto &[swId, candList] : candidates) {
    for (auto &cand : candList)
      reverseMap[cand.hwNodeId].insert(swId);
  }

  // Current candidate count per DFG node.
  llvm::DenseMap<IdIndex, unsigned> candCount;
  for (auto &[swId, candList] : candidates)
    candCount[swId] = static_cast<unsigned>(candList.size());

  // Collect all removable functional nodes with their "usefulness" score.
  struct HwEntry {
    unsigned useCount;
    IdIndex hwId;
  };
  llvm::SmallVector<HwEntry, 128> hwEntries;
  for (IdIndex i = 0; i < static_cast<IdIndex>(adg.nodes.size()); ++i) {
    Node *node = adg.getNode(i);
    if (!isRemovableFunctional(node))
      continue;

    unsigned useCount = 0;
    auto it = reverseMap.find(i);
    if (it != reverseMap.end())
      useCount = it->second.size();

    hwEntries.push_back({useCount, i});
  }

  unsigned totalFunctional = hwEntries.size();

  // Compute removal budget: at most half the surplus beyond demand.
  // This prevents over-pruning that breaks bipartite matching feasibility.
  unsigned surplus = totalFunctional > demand ? totalFunctional - demand : 0;
  unsigned maxRemovals = surplus / 2;

  // If the surplus is tiny relative to demand, masking won't help routing
  // meaningfully. Skip to avoid breaking placement for marginal gain.
  if (surplus < demand / 4) {
    LLVM_DEBUG(llvm::dbgs() << "DomainMask: surplus too small (" << surplus
                            << " surplus for " << demand
                            << " demand), skipping\n");
    return;
  }

  // Sort by ascending usefulness: least-useful nodes removed first.
  std::sort(hwEntries.begin(), hwEntries.end(),
            [](const HwEntry &a, const HwEntry &b) {
              return a.useCount < b.useCount;
            });

  // Greedy coverage-safe removal with removal budget cap.
  // Guards:
  //   1. Per-node guard: each DFG node keeps > minCandidates candidates.
  //   2. Budget cap: total removals <= maxRemovals.
  llvm::SmallVector<IdIndex, 64> toRemove;

  for (auto &entry : hwEntries) {
    if (toRemove.size() >= maxRemovals)
      break;

    IdIndex hwId = entry.hwId;

    // Nodes not serving any DFG operation: always remove (within budget).
    if (entry.useCount == 0) {
      toRemove.push_back(hwId);
      continue;
    }

    // Check per-node safety: all served DFG nodes keep > minCandidates.
    auto revIt = reverseMap.find(hwId);
    if (revIt == reverseMap.end()) {
      toRemove.push_back(hwId);
      continue;
    }

    bool safe = true;
    for (IdIndex swId : revIt->second) {
      auto ccIt = candCount.find(swId);
      if (ccIt == candCount.end() || ccIt->second <= minCandidates) {
        safe = false;
        break;
      }
    }

    if (!safe)
      continue;

    // Safe to remove: decrement candidate counts.
    for (IdIndex swId : revIt->second) {
      auto ccIt = candCount.find(swId);
      if (ccIt != candCount.end())
        ccIt->second--;
    }
    toRemove.push_back(hwId);
  }

  LLVM_DEBUG({
    unsigned remaining =
        totalFunctional - static_cast<unsigned>(toRemove.size());
    llvm::dbgs() << "DomainMask: " << toRemove.size() << " of "
                 << totalFunctional << " functional nodes removed"
                 << " (remaining=" << remaining << ", demand=" << demand
                 << ", surplus=" << surplus
                 << ", maxRemovals=" << maxRemovals
                 << ", minCandidates=" << minCandidates << ")\n";
  });

  // Remove unneeded nodes (cascade-deletes their ports and edges).
  for (IdIndex id : toRemove)
    adg.removeNode(id);
}

} // namespace loom
