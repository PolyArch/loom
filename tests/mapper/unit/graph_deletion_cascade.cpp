//===-- graph_deletion_cascade.cpp - Deletion cascade test ---------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Remove node -> owned ports and connected edges removed.
//
//===----------------------------------------------------------------------===//

#include "TestUtil.h"
#include "loom/Mapper/Graph.h"

using namespace loom;

int main() {
  Graph graph;

  // Create 2 nodes connected by an edge.
  auto node0 = std::make_unique<Node>();
  IdIndex n0 = graph.addNode(std::move(node0));

  auto node1 = std::make_unique<Node>();
  IdIndex n1 = graph.addNode(std::move(node1));

  // Node 0: output port (id=0).
  auto outPort = std::make_unique<Port>();
  outPort->parentNode = n0;
  outPort->direction = Port::Output;
  IdIndex op = graph.addPort(std::move(outPort));
  graph.getNode(n0)->outputPorts.push_back(op);

  // Node 1: input port (id=1).
  auto inPort = std::make_unique<Port>();
  inPort->parentNode = n1;
  inPort->direction = Port::Input;
  IdIndex ip = graph.addPort(std::move(inPort));
  graph.getNode(n1)->inputPorts.push_back(ip);

  // Edge connecting them.
  auto edge = std::make_unique<Edge>();
  edge->srcPort = op;
  edge->dstPort = ip;
  IdIndex eid = graph.addEdge(std::move(edge));
  graph.getPort(op)->connectedEdges.push_back(eid);
  graph.getPort(ip)->connectedEdges.push_back(eid);

  TEST_ASSERT(graph.countNodes() == 2);
  TEST_ASSERT(graph.countPorts() == 2);
  TEST_ASSERT(graph.countEdges() == 1);

  // Remove node 0: should cascade-delete its output port and the edge.
  graph.removeNode(n0);

  TEST_ASSERT(graph.getNode(n0) == nullptr);
  TEST_ASSERT(graph.getPort(op) == nullptr);
  TEST_ASSERT(graph.getEdge(eid) == nullptr);

  // Node 1 and its port survive, but connected edges list is now empty.
  TEST_ASSERT(graph.getNode(n1) != nullptr);
  TEST_ASSERT(graph.getPort(ip) != nullptr);
  TEST_ASSERT(graph.getPort(ip)->connectedEdges.empty());

  TEST_ASSERT(graph.countNodes() == 1);
  TEST_ASSERT(graph.countPorts() == 1);
  TEST_ASSERT(graph.countEdges() == 0);

  return 0;
}
