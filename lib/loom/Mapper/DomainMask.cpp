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

#include <queue>

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

/// BFS from retained endpoint nodes to find all reachable routing nodes.
/// Returns the set of routing node IDs that are reachable from at least one
/// retained endpoint (functional, memory, or module I/O).
llvm::DenseSet<IdIndex> findReachableRoutingNodes(const Graph &adg) {
  llvm::DenseSet<IdIndex> reachable;
  std::queue<IdIndex> worklist;

  // Seed the BFS with all retained non-routing endpoint nodes.
  for (IdIndex i = 0; i < static_cast<IdIndex>(adg.nodes.size()); ++i) {
    const Node *node = adg.getNode(i);
    if (!node)
      continue;

    llvm::StringRef resClass = getAttrStr(node, "resource_class");
    bool isRouting = (resClass == "routing");

    // Module I/O sentinels, memory, functional, and unknown-class nodes
    // are endpoints that anchor the routing subgraph.
    if (!isRouting) {
      for (IdIndex portId : node->inputPorts)
        worklist.push(portId);
      for (IdIndex portId : node->outputPorts)
        worklist.push(portId);
    }
  }

  // BFS through edges to discover reachable routing nodes.
  llvm::DenseSet<IdIndex> visitedPorts;
  while (!worklist.empty()) {
    IdIndex portId = worklist.front();
    worklist.pop();

    if (!visitedPorts.insert(portId).second)
      continue;

    const Port *port = adg.getPort(portId);
    if (!port)
      continue;

    for (IdIndex edgeId : port->connectedEdges) {
      const Edge *edge = adg.getEdge(edgeId);
      if (!edge)
        continue;

      IdIndex otherPortId = (edge->srcPort == portId) ? edge->dstPort
                                                      : edge->srcPort;
      if (visitedPorts.count(otherPortId))
        continue;

      const Port *otherPort = adg.getPort(otherPortId);
      if (!otherPort)
        continue;

      IdIndex otherNodeId = otherPort->parentNode;
      const Node *otherNode = adg.getNode(otherNodeId);
      if (!otherNode)
        continue;

      llvm::StringRef otherClass = getAttrStr(otherNode, "resource_class");
      if (otherClass == "routing") {
        reachable.insert(otherNodeId);

        for (IdIndex pid : otherNode->inputPorts)
          worklist.push(pid);
        for (IdIndex pid : otherNode->outputPorts)
          worklist.push(pid);
      }
    }
  }

  return reachable;
}

} // anonymous namespace

void pruneDomainADG(Graph &adg, const Graph &dfg, unsigned minCandidates) {
  // Run tech-mapping to discover which HW nodes are compatible with each
  // DFG operation.
  TechMapper techMapper;
  CandidateSet candidates = techMapper.map(dfg, adg);

  // Build set of HW functional nodes that serve at least one DFG operation.
  llvm::DenseSet<IdIndex> usedFunctional;
  for (auto &[swId, candList] : candidates) {
    for (auto &cand : candList) {
      const Node *hw = adg.getNode(cand.hwNodeId);
      if (hw && getAttrStr(hw, "resource_class") == "functional")
        usedFunctional.insert(cand.hwNodeId);
    }
  }

  // Remove all functional PEs that are NOT candidates for any DFG operation.
  // These nodes consume switch ports without any possibility of placement use.
  // Candidate PEs (useCount > 0) are always retained to avoid breaking
  // placement feasibility.
  llvm::SmallVector<IdIndex, 64> functionalToRemove;
  unsigned totalFunctional = 0;
  for (IdIndex i = 0; i < static_cast<IdIndex>(adg.nodes.size()); ++i) {
    Node *node = adg.getNode(i);
    if (!isRemovableFunctional(node))
      continue;
    totalFunctional++;

    if (!usedFunctional.count(i))
      functionalToRemove.push_back(i);
  }

  // Cascade-delete removed PEs (ports and edges removed automatically).
  for (IdIndex id : functionalToRemove)
    adg.removeNode(id);

  // After functional pruning, find routing nodes still reachable from
  // retained endpoints. Remove unreachable routing nodes to further
  // free routing capacity.
  llvm::DenseSet<IdIndex> reachableRouting = findReachableRoutingNodes(adg);

  llvm::SmallVector<IdIndex, 64> routingToRemove;
  for (IdIndex i = 0; i < static_cast<IdIndex>(adg.nodes.size()); ++i) {
    const Node *node = adg.getNode(i);
    if (!node)
      continue;
    if (node->kind == Node::ModuleInputNode ||
        node->kind == Node::ModuleOutputNode)
      continue;
    if (getAttrStr(node, "resource_class") != "routing")
      continue;
    if (!reachableRouting.count(i))
      routingToRemove.push_back(i);
  }

  for (IdIndex id : routingToRemove)
    adg.removeNode(id);

  LLVM_DEBUG({
    unsigned funcRemaining =
        totalFunctional - static_cast<unsigned>(functionalToRemove.size());
    llvm::dbgs() << "DomainMask: removed " << functionalToRemove.size()
                 << " functional + " << routingToRemove.size()
                 << " routing nodes (funcRemaining=" << funcRemaining
                 << ", candidates=" << usedFunctional.size() << ")\n";
  });
}

} // namespace loom
