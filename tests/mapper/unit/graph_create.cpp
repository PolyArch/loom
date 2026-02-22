//===-- graph_create.cpp - Graph creation test ---------------------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Create graph with N nodes, M ports, K edges; verify counts and accessors.
//
//===----------------------------------------------------------------------===//

#include "TestUtil.h"
#include "loom/Mapper/Graph.h"

using namespace loom;

int main() {
  Graph graph;

  // Create 3 nodes, each with 1 input and 1 output port.
  for (IdIndex i = 0; i < 3; ++i) {
    auto node = std::make_unique<Node>();
    IdIndex nid = graph.addNode(std::move(node));
    TEST_ASSERT(nid == i);

    auto inPort = std::make_unique<Port>();
    inPort->parentNode = nid;
    inPort->direction = Port::Input;
    IdIndex inId = graph.addPort(std::move(inPort));
    graph.getNode(nid)->inputPorts.push_back(inId);

    auto outPort = std::make_unique<Port>();
    outPort->parentNode = nid;
    outPort->direction = Port::Output;
    IdIndex outId = graph.addPort(std::move(outPort));
    graph.getNode(nid)->outputPorts.push_back(outId);
  }

  TEST_ASSERT(graph.countNodes() == 3);
  TEST_ASSERT(graph.countPorts() == 6);

  // Create 2 edges: node0.out -> node1.in, node1.out -> node2.in.
  // Port layout: node0(in=0, out=1), node1(in=2, out=3), node2(in=4, out=5).
  {
    auto edge = std::make_unique<Edge>();
    edge->srcPort = 1;
    edge->dstPort = 2;
    IdIndex eid = graph.addEdge(std::move(edge));
    graph.getPort(1)->connectedEdges.push_back(eid);
    graph.getPort(2)->connectedEdges.push_back(eid);
  }
  {
    auto edge = std::make_unique<Edge>();
    edge->srcPort = 3;
    edge->dstPort = 4;
    IdIndex eid = graph.addEdge(std::move(edge));
    graph.getPort(3)->connectedEdges.push_back(eid);
    graph.getPort(4)->connectedEdges.push_back(eid);
  }

  TEST_ASSERT(graph.countEdges() == 2);

  // Verify getters for valid and out-of-range IDs.
  TEST_ASSERT(graph.getNode(0) != nullptr);
  TEST_ASSERT(graph.getNode(1) != nullptr);
  TEST_ASSERT(graph.getNode(2) != nullptr);
  TEST_ASSERT(graph.getNode(3) == nullptr);

  TEST_ASSERT(graph.getPort(0) != nullptr);
  TEST_ASSERT(graph.getPort(5) != nullptr);
  TEST_ASSERT(graph.getPort(6) == nullptr);

  TEST_ASSERT(graph.getEdge(0) != nullptr);
  TEST_ASSERT(graph.getEdge(1) != nullptr);
  TEST_ASSERT(graph.getEdge(2) == nullptr);

  // Verify isValid.
  TEST_ASSERT(graph.isValid(0, EntityKind::Node));
  TEST_ASSERT(graph.isValid(2, EntityKind::Node));
  TEST_ASSERT(!graph.isValid(3, EntityKind::Node));
  TEST_ASSERT(!graph.isValid(99, EntityKind::Node));
  TEST_ASSERT(graph.isValid(0, EntityKind::Port));
  TEST_ASSERT(!graph.isValid(6, EntityKind::Port));
  TEST_ASSERT(graph.isValid(0, EntityKind::Edge));
  TEST_ASSERT(!graph.isValid(2, EntityKind::Edge));

  return 0;
}
