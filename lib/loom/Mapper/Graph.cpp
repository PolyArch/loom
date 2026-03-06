//===-- Graph.cpp - Mapper graph data model implementation --------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Implementation of Graph mutation and query methods, including ownership-based
// deletion cascades.
//
//===----------------------------------------------------------------------===//

#include "loom/Mapper/Graph.h"

#include <algorithm>

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

  // Copy port lists before modification (removePort modifies parent node lists).
  auto inPorts = nodes[id]->inputPorts;
  auto outPorts = nodes[id]->outputPorts;

  for (IdIndex portId : inPorts) {
    if (portId < ports.size() && ports[portId]) {
      if (ports[portId]->parentNode == id) {
        // Node owns this port: cascade-delete.
        removePort(portId);
      }
      // Otherwise: node only references the port (e.g., temporal PE FU node
      // sharing virtual node's ports). Leave the port intact.
    }
  }

  for (IdIndex portId : outPorts) {
    if (portId < ports.size() && ports[portId]) {
      if (ports[portId]->parentNode == id) {
        removePort(portId);
      }
    }
  }

  nodes[id] = nullptr;
}

void Graph::removePort(IdIndex id) {
  if (id >= ports.size() || !ports[id])
    return;

  // Copy connected edges before modification (removeEdge modifies port lists).
  auto edgeIds = ports[id]->connectedEdges;

  for (IdIndex edgeId : edgeIds) {
    removeEdge(edgeId);
  }

  // Remove this port ID from its parent node's port lists.
  IdIndex parentId = ports[id]->parentNode;
  if (parentId < nodes.size() && nodes[parentId]) {
    auto &inPorts = nodes[parentId]->inputPorts;
    auto &outPorts = nodes[parentId]->outputPorts;
    inPorts.erase(std::remove(inPorts.begin(), inPorts.end(), id),
                  inPorts.end());
    outPorts.erase(std::remove(outPorts.begin(), outPorts.end(), id),
                   outPorts.end());
  }

  ports[id] = nullptr;
}

void Graph::removeEdge(IdIndex id) {
  if (id >= edges.size() || !edges[id])
    return;

  IdIndex srcPortId = edges[id]->srcPort;
  IdIndex dstPortId = edges[id]->dstPort;

  // Remove from source port's connected edges.
  if (srcPortId < ports.size() && ports[srcPortId]) {
    auto &ce = ports[srcPortId]->connectedEdges;
    ce.erase(std::remove(ce.begin(), ce.end(), id), ce.end());
  }

  // Remove from destination port's connected edges.
  if (dstPortId < ports.size() && ports[dstPortId]) {
    auto &ce = ports[dstPortId]->connectedEdges;
    ce.erase(std::remove(ce.begin(), ce.end(), id), ce.end());
  }

  edges[id] = nullptr;
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
  for (const auto &n : nodes)
    if (n)
      ++count;
  return count;
}

size_t Graph::countPorts() const {
  size_t count = 0;
  for (const auto &p : ports)
    if (p)
      ++count;
  return count;
}

size_t Graph::countEdges() const {
  size_t count = 0;
  for (const auto &e : edges)
    if (e)
      ++count;
  return count;
}

} // namespace loom
