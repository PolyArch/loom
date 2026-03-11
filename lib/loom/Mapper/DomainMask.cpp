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
#include <cmath>
#include <map>
#include <queue>
#include <vector>

#define DEBUG_TYPE "domain-mask"

namespace loom {

namespace {

//===----------------------------------------------------------------------===//
// Attribute helpers
//===----------------------------------------------------------------------===//

llvm::StringRef getAttrStr(const Node *node, llvm::StringRef name) {
  for (auto &attr : node->attributes) {
    if (attr.getName() == name) {
      if (auto strAttr = mlir::dyn_cast<mlir::StringAttr>(attr.getValue()))
        return strAttr.getValue();
    }
  }
  return "";
}

bool hasAttrFlag(const Node *node, llvm::StringRef name) {
  for (auto &attr : node->attributes) {
    if (attr.getName() == name) {
      if (mlir::isa<mlir::UnitAttr>(attr.getValue()))
        return true;
      if (auto strAttr = mlir::dyn_cast<mlir::StringAttr>(attr.getValue()))
        return strAttr.getValue() == "true";
    }
  }
  return false;
}

void collectAttrInts(const Node *node, llvm::StringRef name,
                     llvm::SmallVectorImpl<IdIndex> &result) {
  for (auto &attr : node->attributes) {
    if (attr.getName() == name) {
      if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(attr.getValue()))
        result.push_back(static_cast<IdIndex>(intAttr.getInt()));
    }
  }
}

//===----------------------------------------------------------------------===//
// Graph traversal helpers
//===----------------------------------------------------------------------===//

/// Visit all unique neighbor nodes of a given node, calling visitor(neighborId,
/// neighborNode) for each. Traverses through ports and edges.
template <typename VisitorFn>
void visitNeighbors(const Graph &adg, IdIndex nodeId, VisitorFn visitor) {
  const Node *node = adg.getNode(nodeId);
  if (!node)
    return;
  llvm::DenseSet<IdIndex> seen;

  auto scan = [&](IdIndex portId) {
    const Port *port = adg.getPort(portId);
    if (!port)
      return;
    for (IdIndex edgeId : port->connectedEdges) {
      const Edge *edge = adg.getEdge(edgeId);
      if (!edge)
        continue;
      IdIndex otherPortId =
          (edge->srcPort == portId) ? edge->dstPort : edge->srcPort;
      const Port *otherPort = adg.getPort(otherPortId);
      if (!otherPort)
        continue;
      IdIndex otherNodeId = otherPort->parentNode;
      if (otherNodeId == nodeId)
        continue;
      const Node *otherNode = adg.getNode(otherNodeId);
      if (!otherNode)
        continue;
      if (seen.insert(otherNodeId).second)
        visitor(otherNodeId, otherNode);
    }
  };

  for (IdIndex pid : node->inputPorts)
    scan(pid);
  for (IdIndex pid : node->outputPorts)
    scan(pid);
}

/// Get unique routing-node neighbors of a given node.
llvm::SmallVector<IdIndex, 8> getRoutingNeighbors(const Graph &adg,
                                                   IdIndex nodeId) {
  llvm::SmallVector<IdIndex, 8> result;
  visitNeighbors(adg, nodeId,
                 [&](IdIndex nbrId, const Node *nbrNode) {
                   if (getAttrStr(nbrNode, "resource_class") == "routing")
                     result.push_back(nbrId);
                 });
  return result;
}

/// Check if a node has any live port connections (owns at least one port
/// with at least one connected edge).
bool hasLivePortConnections(const Graph &adg, IdIndex nodeId) {
  const Node *node = adg.getNode(nodeId);
  if (!node)
    return false;
  for (IdIndex pid : node->inputPorts) {
    const Port *port = adg.getPort(pid);
    if (port && port->parentNode == nodeId && !port->connectedEdges.empty())
      return true;
  }
  for (IdIndex pid : node->outputPorts) {
    const Port *port = adg.getPort(pid);
    if (port && port->parentNode == nodeId && !port->connectedEdges.empty())
      return true;
  }
  return false;
}

bool isRemovableFunctional(const Node *node) {
  if (!node)
    return false;
  if (node->kind == Node::ModuleInputNode ||
      node->kind == Node::ModuleOutputNode)
    return false;
  llvm::StringRef resClass = getAttrStr(node, "resource_class");
  if (resClass != "functional")
    return false;
  return true;
}

//===----------------------------------------------------------------------===//
// Steiner tree routing pruning
//===----------------------------------------------------------------------===//

/// Prune routing nodes to a Steiner tree approximation connecting all
/// retained endpoints (functional, memory, module I/O).
///
/// Uses a Voronoi-boundary approach:
///   1. Multi-source BFS from all endpoints through routing nodes.
///   2. Each routing node belongs to the Voronoi cell of its nearest endpoint.
///   3. Routing nodes on boundaries between cells (neighbors with different
///      nearest endpoints) lie on inter-endpoint shortest paths.
///   4. Trace from each boundary node back to its nearest endpoint, marking
///      all routing nodes on the path.
///   5. Optionally expand the marked set by bufferHops layers.
///   6. Remove routing nodes not in the marked set.
unsigned pruneSteinerRouting(Graph &adg, unsigned bufferHops) {
  // Step 1: Find retained endpoints (non-routing nodes with port connections).
  llvm::SmallVector<IdIndex, 64> endpoints;
  for (IdIndex i = 0; i < static_cast<IdIndex>(adg.nodes.size()); ++i) {
    const Node *node = adg.getNode(i);
    if (!node)
      continue;
    if (node->kind == Node::ModuleInputNode ||
        node->kind == Node::ModuleOutputNode) {
      endpoints.push_back(i);
      continue;
    }
    llvm::StringRef cls = getAttrStr(node, "resource_class");
    if (cls == "routing" || cls.empty())
      continue;
    // Functional, memory, or other non-routing node.
    if (hasLivePortConnections(adg, i))
      endpoints.push_back(i);
  }

  if (endpoints.size() <= 1)
    return 0;

  // Step 2: Multi-source BFS from all endpoints through routing nodes.
  // Cache neighbor lists to avoid redundant port/edge traversals.
  // For each routing node, record:
  //   nearestEP[r] = the endpoint whose Voronoi cell r belongs to
  //   parent[r]    = predecessor on shortest path back to nearestEP[r]
  llvm::DenseMap<IdIndex, IdIndex> nearestEP;
  llvm::DenseMap<IdIndex, IdIndex> parent;
  llvm::DenseMap<IdIndex, llvm::SmallVector<IdIndex, 8>> nbrCache;

  auto cachedNeighbors = [&](IdIndex id) -> const llvm::SmallVector<IdIndex, 8> & {
    auto it = nbrCache.find(id);
    if (it != nbrCache.end())
      return it->second;
    return nbrCache.try_emplace(id, getRoutingNeighbors(adg, id)).first->second;
  };

  std::queue<IdIndex> bfs;
  for (IdIndex epId : endpoints) {
    for (IdIndex nbr : cachedNeighbors(epId)) {
      if (!nearestEP.count(nbr)) {
        nearestEP[nbr] = epId;
        parent[nbr] = epId;
        bfs.push(nbr);
      }
    }
  }

  while (!bfs.empty()) {
    IdIndex rId = bfs.front();
    bfs.pop();

    for (IdIndex nbr : cachedNeighbors(rId)) {
      if (!nearestEP.count(nbr)) {
        nearestEP[nbr] = nearestEP[rId];
        parent[nbr] = rId;
        bfs.push(nbr);
      }
    }
  }

  // Step 3: Identify boundary edges and trace paths.
  // A routing node is on the Steiner tree if it lies on the boundary between
  // two Voronoi cells or on the path from a boundary node to its endpoint.
  llvm::DenseSet<IdIndex> treeNodes;

  auto tracePath = [&](IdIndex start) {
    IdIndex cur = start;
    while (nearestEP.count(cur)) {
      if (!treeNodes.insert(cur).second)
        break; // Already in tree; rest of path is also in tree.
      auto it = parent.find(cur);
      if (it == parent.end())
        break;
      IdIndex par = it->second;
      // Stop if parent is an endpoint (not a routing node).
      if (!nearestEP.count(par))
        break;
      cur = par;
    }
  };

  for (auto &[rId, ep] : nearestEP) {
    for (IdIndex nbr : cachedNeighbors(rId)) {
      auto it = nearestEP.find(nbr);
      if (it != nearestEP.end() && it->second != ep) {
        // Boundary edge: trace both sides back to their endpoints.
        tracePath(rId);
        tracePath(nbr);
      }
    }
  }

  // Step 4: Expand by bufferHops layers for routing redundancy.
  // Use a frontier approach to avoid copying the full set each hop.
  llvm::SmallVector<IdIndex, 64> frontier(treeNodes.begin(), treeNodes.end());
  for (unsigned hop = 0; hop < bufferHops; ++hop) {
    llvm::SmallVector<IdIndex, 64> nextFrontier;
    for (IdIndex rId : frontier) {
      for (IdIndex nbr : cachedNeighbors(rId)) {
        if (nearestEP.count(nbr) && treeNodes.insert(nbr).second)
          nextFrontier.push_back(nbr);
      }
    }
    frontier = std::move(nextFrontier);
  }

  // Step 5: Remove routing nodes not in the Steiner tree.
  unsigned removed = 0;
  for (IdIndex i = 0; i < static_cast<IdIndex>(adg.nodes.size()); ++i) {
    const Node *node = adg.getNode(i);
    if (!node)
      continue;
    if (node->kind == Node::ModuleInputNode ||
        node->kind == Node::ModuleOutputNode)
      continue;
    if (getAttrStr(node, "resource_class") != "routing")
      continue;
    if (!treeNodes.count(i)) {
      adg.removeNode(i);
      removed++;
    }
  }

  return removed;
}

//===----------------------------------------------------------------------===//
// Dead-end routing pruning
//===----------------------------------------------------------------------===//

unsigned pruneDeadEndRouting(Graph &adg) {
  std::queue<IdIndex> worklist;
  for (IdIndex i = 0; i < static_cast<IdIndex>(adg.nodes.size()); ++i) {
    const Node *node = adg.getNode(i);
    if (!node)
      continue;
    if (node->kind == Node::ModuleInputNode ||
        node->kind == Node::ModuleOutputNode)
      continue;
    if (getAttrStr(node, "resource_class") == "routing")
      worklist.push(i);
  }

  unsigned removed = 0;
  while (!worklist.empty()) {
    IdIndex nodeId = worklist.front();
    worklist.pop();
    if (!adg.getNode(nodeId))
      continue;

    // Count routing neighbors and endpoint neighbors to decide removability.
    unsigned routingCount = 0;
    unsigned endpointCount = 0;
    llvm::SmallVector<IdIndex, 4> routingNbrs;
    visitNeighbors(adg, nodeId,
                   [&](IdIndex nbrId, const Node *nbrNode) {
                     if (getAttrStr(nbrNode, "resource_class") == "routing") {
                       routingCount++;
                       routingNbrs.push_back(nbrId);
                     } else {
                       endpointCount++;
                     }
                   });

    if (endpointCount == 0 && routingCount <= 1) {
      for (IdIndex neighborId : routingNbrs)
        worklist.push(neighborId);
      adg.removeNode(nodeId);
      removed++;
    }
  }

  return removed;
}

} // anonymous namespace

//===----------------------------------------------------------------------===//
// Public API
//===----------------------------------------------------------------------===//

void pruneDomainADG(Graph &adg, const Graph &dfg, unsigned minCandidates) {
  // --- Phase 1: Compute the full candidate universe. -----------------------
  TechMapper techMapper;
  CandidateSet candidates = techMapper.map(dfg, adg);

  // --- Phase 2: Build per-PE reverse map and compatibility classes. --------
  llvm::DenseMap<IdIndex, llvm::SmallVector<IdIndex, 8>> peToOps;
  llvm::DenseSet<IdIndex> retainedMemory;
  for (auto &[dfgId, candList] : candidates) {
    for (auto &cand : candList) {
      const Node *hw = adg.getNode(cand.hwNodeId);
      if (!hw)
        continue;
      llvm::StringRef cls = getAttrStr(hw, "resource_class");
      if (cls == "functional")
        peToOps[cand.hwNodeId].push_back(dfgId);
      else if (cls == "memory")
        retainedMemory.insert(cand.hwNodeId);
    }
  }
  for (auto &[peId, ops] : peToOps) {
    llvm::sort(ops);
    ops.erase(std::unique(ops.begin(), ops.end()), ops.end());
  }

  struct CompatClass {
    llvm::SmallVector<IdIndex, 16> peIds;
    llvm::SmallVector<IdIndex, 8> dfgOps;
  };
  std::map<std::vector<IdIndex>, CompatClass> classes;
  for (auto &[peId, ops] : peToOps) {
    std::vector<IdIndex> key(ops.begin(), ops.end());
    auto &cls = classes[key];
    cls.peIds.push_back(peId);
    if (cls.dfgOps.empty())
      cls.dfgOps.assign(ops.begin(), ops.end());
  }

  // --- Phase 3: Compute demand per class. ----------------------------------
  llvm::DenseMap<IdIndex, unsigned> classesPerOp;
  for (auto &[key, cls] : classes) {
    for (IdIndex dfgOp : cls.dfgOps)
      classesPerOp[dfgOp]++;
  }

  llvm::DenseSet<IdIndex> retainedPEs;
  for (auto &[key, cls] : classes) {
    double demand = 0.0;
    for (IdIndex dfgOp : cls.dfgOps) {
      unsigned numClasses = classesPerOp.lookup(dfgOp);
      demand += 1.0 / std::max(numClasses, 1u);
    }
    unsigned keepCount = static_cast<unsigned>(std::ceil(demand));
    if (minCandidates > 1)
      keepCount += (minCandidates - 1);
    keepCount = std::max(keepCount, 1u);
    keepCount = std::min(keepCount, static_cast<unsigned>(cls.peIds.size()));

    llvm::sort(cls.peIds);
    for (unsigned i = 0; i < keepCount; ++i)
      retainedPEs.insert(cls.peIds[i]);
  }

  // --- Phase 4: Protect virtual temporal PEs with retained FU sub-nodes. ---
  for (IdIndex i = 0; i < static_cast<IdIndex>(adg.nodes.size()); ++i) {
    const Node *node = adg.getNode(i);
    if (!node)
      continue;
    if (getAttrStr(node, "resource_class") != "functional")
      continue;
    if (!hasAttrFlag(node, "is_virtual"))
      continue;

    llvm::SmallVector<IdIndex, 8> fuIds;
    collectAttrInts(node, "fu_node", fuIds);

    bool anyRetained = false;
    for (IdIndex fuId : fuIds) {
      if (retainedPEs.count(fuId)) {
        anyRetained = true;
        break;
      }
    }
    if (anyRetained)
      retainedPEs.insert(i);
  }

  // --- Phase 5: Remove surplus functional PEs. -----------------------------
  llvm::SmallVector<IdIndex, 64> functionalToRemove;
  unsigned totalFunctional = 0;
  for (IdIndex i = 0; i < static_cast<IdIndex>(adg.nodes.size()); ++i) {
    Node *node = adg.getNode(i);
    if (!isRemovableFunctional(node))
      continue;
    totalFunctional++;
    if (!retainedPEs.count(i))
      functionalToRemove.push_back(i);
  }
  for (IdIndex id : functionalToRemove)
    adg.removeNode(id);

  // --- Phase 5b: Remove unused memory nodes. --------------------------------
  // Memory nodes not referenced by any DFG candidate are surplus; removing
  // them frees switch ports that would otherwise be consumed by unused
  // load/store connections.
  unsigned totalMemory = 0;
  unsigned memoryRemoved = 0;
  for (IdIndex i = 0; i < static_cast<IdIndex>(adg.nodes.size()); ++i) {
    const Node *node = adg.getNode(i);
    if (!node)
      continue;
    if (getAttrStr(node, "resource_class") != "memory")
      continue;
    totalMemory++;
    if (!retainedMemory.count(i)) {
      adg.removeNode(i);
      memoryRemoved++;
    }
  }

  // --- Phase 6: Steiner-tree routing pruning. ------------------------------
  unsigned steinerRemoved = pruneSteinerRouting(adg, /*bufferHops=*/2);

  // --- Phase 7: Dead-end routing pruning. ----------------------------------
  unsigned deadEndRemoved = pruneDeadEndRouting(adg);

  LLVM_DEBUG(llvm::dbgs() << "DomainMask: removed " << functionalToRemove.size()
               << " of " << totalFunctional << " functional, "
               << memoryRemoved << " of " << totalMemory << " memory, "
               << steinerRemoved << " Steiner + "
               << deadEndRemoved << " dead-end routing"
               << " (retainedPE=" << retainedPEs.size()
               << ", retainedMem=" << retainedMemory.size()
               << ", classes=" << classes.size() << ")\n");
}

} // namespace loom
