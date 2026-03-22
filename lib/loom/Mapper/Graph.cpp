#include "loom/Mapper/Graph.h"

#include "mlir/IR/BuiltinAttributes.h"

namespace loom {

IdIndex Graph::addNode(std::unique_ptr<Node> node) {
  IdIndex id = static_cast<IdIndex>(nodes.size());
  nodes.push_back(std::move(node));
  return id;
}

IdIndex Graph::addPort(std::unique_ptr<Port> port) {
  IdIndex id = static_cast<IdIndex>(ports.size());
  ports.push_back(std::move(port));
  return id;
}

IdIndex Graph::addEdge(std::unique_ptr<Edge> edge) {
  IdIndex id = static_cast<IdIndex>(edges.size());
  edges.push_back(std::move(edge));
  return id;
}

void Graph::removeNode(IdIndex id) {
  if (id >= nodes.size() || !nodes[id])
    return;
  // Cascade-delete owned ports.
  auto &node = nodes[id];
  for (IdIndex pid : node->inputPorts)
    removePort(pid);
  for (IdIndex pid : node->outputPorts)
    removePort(pid);
  nodes[id].reset();
}

void Graph::removePort(IdIndex id) {
  if (id >= ports.size() || !ports[id])
    return;
  auto &port = ports[id];
  // Remove connected edges.
  auto edgeCopy = port->connectedEdges;
  for (IdIndex eid : edgeCopy)
    removeEdge(eid);
  // Remove from parent node port lists.
  if (port->parentNode != INVALID_ID && port->parentNode < nodes.size()) {
    auto &node = nodes[port->parentNode];
    if (node) {
      auto &portList = (port->direction == Port::Input) ? node->inputPorts
                                                        : node->outputPorts;
      portList.erase(std::remove(portList.begin(), portList.end(), id),
                     portList.end());
    }
  }
  ports[id].reset();
}

void Graph::removeEdge(IdIndex id) {
  if (id >= edges.size() || !edges[id])
    return;
  auto &edge = edges[id];
  // Remove from both endpoint ports.
  for (IdIndex pid : {edge->srcPort, edge->dstPort}) {
    if (pid < ports.size() && ports[pid]) {
      auto &ce = ports[pid]->connectedEdges;
      ce.erase(std::remove(ce.begin(), ce.end(), id), ce.end());
    }
  }
  edges[id].reset();
}

Node *Graph::getNode(IdIndex id) const {
  if (id >= nodes.size())
    return nullptr;
  return nodes[id].get();
}

Port *Graph::getPort(IdIndex id) const {
  if (id >= ports.size())
    return nullptr;
  return ports[id].get();
}

Edge *Graph::getEdge(IdIndex id) const {
  if (id >= edges.size())
    return nullptr;
  return edges[id].get();
}

bool Graph::isValid(IdIndex id, EntityKind kind) const {
  switch (kind) {
  case EntityKind::Node:
    return id < nodes.size() && nodes[id] != nullptr;
  case EntityKind::Port:
    return id < ports.size() && ports[id] != nullptr;
  case EntityKind::Edge:
    return id < edges.size() && edges[id] != nullptr;
  }
  return false;
}

size_t Graph::countNodes() const {
  size_t count = 0;
  for (auto &n : nodes)
    if (n)
      ++count;
  return count;
}

size_t Graph::countPorts() const {
  size_t count = 0;
  for (auto &p : ports)
    if (p)
      ++count;
  return count;
}

size_t Graph::countEdges() const {
  size_t count = 0;
  for (auto &e : edges)
    if (e)
      ++count;
  return count;
}

Graph Graph::clone() const {
  Graph g(context);
  g.nodes.resize(nodes.size());
  g.ports.resize(ports.size());
  g.edges.resize(edges.size());
  for (size_t i = 0; i < nodes.size(); ++i) {
    if (nodes[i]) {
      g.nodes[i] = std::make_unique<Node>(*nodes[i]);
    }
  }
  for (size_t i = 0; i < ports.size(); ++i) {
    if (ports[i]) {
      g.ports[i] = std::make_unique<Port>(*ports[i]);
    }
  }
  for (size_t i = 0; i < edges.size(); ++i) {
    if (edges[i]) {
      g.edges[i] = std::make_unique<Edge>(*edges[i]);
    }
  }
  return g;
}

// Attribute helpers.

llvm::StringRef getNodeAttrStr(const Node *node, llvm::StringRef key) {
  if (!node)
    return "";
  // Fast path: return from cache if available.
  if (node->attributeCacheValid) {
    if (key == "resource_class")
      return node->cachedResourceClass;
    if (key == "pe_name")
      return node->cachedPeName;
    if (key == "op_name")
      return node->cachedOpName;
    if (key == "pe_kind")
      return node->cachedPeKind;
    if (key == "op_kind")
      return node->cachedOpKind;
  }
  for (auto &attr : node->attributes) {
    if (attr.getName() == key) {
      if (auto strAttr = mlir::dyn_cast<mlir::StringAttr>(attr.getValue()))
        return strAttr.getValue();
    }
  }
  return "";
}

int64_t getNodeAttrInt(const Node *node, llvm::StringRef key,
                       int64_t defaultVal) {
  if (!node)
    return defaultVal;
  for (auto &attr : node->attributes) {
    if (attr.getName() == key) {
      if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(attr.getValue()))
        return intAttr.getInt();
    }
  }
  return defaultVal;
}

bool nodeHasAttr(const Node *node, llvm::StringRef key) {
  if (!node)
    return false;
  for (auto &attr : node->attributes)
    if (attr.getName() == key)
      return true;
  return false;
}

void Graph::buildAttributeCache() {
  for (auto &nodePtr : nodes) {
    if (!nodePtr)
      continue;
    Node *node = nodePtr.get();
    node->cachedResourceClass = "";
    node->cachedPeName = "";
    node->cachedOpName = "";
    node->cachedPeKind = "";
    node->cachedOpKind = "";
    for (auto &attr : node->attributes) {
      auto strAttr = mlir::dyn_cast<mlir::StringAttr>(attr.getValue());
      if (!strAttr)
        continue;
      llvm::StringRef name = attr.getName();
      if (name == "resource_class")
        node->cachedResourceClass = strAttr.getValue();
      else if (name == "pe_name")
        node->cachedPeName = strAttr.getValue();
      else if (name == "op_name")
        node->cachedOpName = strAttr.getValue();
      else if (name == "pe_kind")
        node->cachedPeKind = strAttr.getValue();
      else if (name == "op_kind")
        node->cachedOpKind = strAttr.getValue();
    }
    node->attributeCacheValid = true;
  }
}

} // namespace loom
