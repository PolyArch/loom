//===-- graph_iteration.cpp - Non-null iteration test -------------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Iterate non-null entries skipping deleted slots.
//
//===----------------------------------------------------------------------===//

#include "TestUtil.h"
#include "loom/Mapper/Graph.h"

using namespace loom;

int main() {
  Graph graph;

  // Create 5 nodes, each with input/output ports.
  for (IdIndex i = 0; i < 5; ++i) {
    auto node = std::make_unique<Node>();
    IdIndex nid = graph.addNode(std::move(node));

    auto inPort = std::make_unique<Port>();
    inPort->parentNode = nid;
    inPort->direction = Port::Input;
    IdIndex pid = graph.addPort(std::move(inPort));
    graph.getNode(nid)->inputPorts.push_back(pid);

    auto outPort = std::make_unique<Port>();
    outPort->parentNode = nid;
    outPort->direction = Port::Output;
    IdIndex opid = graph.addPort(std::move(outPort));
    graph.getNode(nid)->outputPorts.push_back(opid);
  }

  // Delete nodes 1 and 3 (and their owned ports).
  graph.removeNode(1);
  graph.removeNode(3);

  // Iterate nodes: should visit exactly 3 non-null entries.
  size_t nodeCount = 0;
  for (Node *n : graph.nodeRange()) {
    TEST_ASSERT(n != nullptr);
    ++nodeCount;
  }
  TEST_ASSERT(nodeCount == 3);
  TEST_ASSERT(nodeCount == graph.countNodes());

  // Iterate ports: nodes 0,2,4 each have 2 ports -> 6 remaining.
  size_t portCount = 0;
  for (Port *p : graph.portRange()) {
    TEST_ASSERT(p != nullptr);
    ++portCount;
  }
  TEST_ASSERT(portCount == 6);
  TEST_ASSERT(portCount == graph.countPorts());

  // Iterate edges: none were created.
  size_t edgeCount = 0;
  for (Edge *e : graph.edgeRange()) {
    TEST_ASSERT(e != nullptr);
    ++edgeCount;
  }
  TEST_ASSERT(edgeCount == 0);

  return 0;
}
