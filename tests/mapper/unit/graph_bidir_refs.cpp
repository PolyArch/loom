//===-- graph_bidir_refs.cpp - Bidirectional reference consistency -*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Port.parentNode and Node.ports are consistent after mutations.
//
//===----------------------------------------------------------------------===//

#include "TestUtil.h"
#include "loom/Mapper/Graph.h"

using namespace loom;

int main() {
  Graph graph;

  // Create a node with 2 input and 1 output port.
  auto node = std::make_unique<Node>();
  IdIndex nid = graph.addNode(std::move(node));

  IdIndex portIds[3];
  for (int i = 0; i < 2; ++i) {
    auto port = std::make_unique<Port>();
    port->parentNode = nid;
    port->direction = Port::Input;
    portIds[i] = graph.addPort(std::move(port));
    graph.getNode(nid)->inputPorts.push_back(portIds[i]);
  }
  {
    auto port = std::make_unique<Port>();
    port->parentNode = nid;
    port->direction = Port::Output;
    portIds[2] = graph.addPort(std::move(port));
    graph.getNode(nid)->outputPorts.push_back(portIds[2]);
  }

  // Verify bidirectional references.
  Node *n = graph.getNode(nid);
  TEST_ASSERT(n->inputPorts.size() == 2);
  TEST_ASSERT(n->outputPorts.size() == 1);

  for (IdIndex pid : n->inputPorts) {
    TEST_ASSERT(graph.getPort(pid)->parentNode == nid);
  }
  for (IdIndex pid : n->outputPorts) {
    TEST_ASSERT(graph.getPort(pid)->parentNode == nid);
  }

  // Remove one input port; verify consistency.
  graph.removePort(portIds[0]);
  TEST_ASSERT(graph.getPort(portIds[0]) == nullptr);

  n = graph.getNode(nid);
  TEST_ASSERT(n->inputPorts.size() == 1);
  TEST_ASSERT(n->inputPorts[0] == portIds[1]);
  TEST_ASSERT(n->outputPorts.size() == 1);
  TEST_ASSERT(n->outputPorts[0] == portIds[2]);

  // Remaining ports still reference the node.
  TEST_ASSERT(graph.getPort(portIds[1])->parentNode == nid);
  TEST_ASSERT(graph.getPort(portIds[2])->parentNode == nid);

  return 0;
}
