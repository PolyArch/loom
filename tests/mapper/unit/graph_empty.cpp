//===-- graph_empty.cpp - Empty graph and single-node edge cases --*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Empty graph and single-node edge cases.
//
//===----------------------------------------------------------------------===//

#include "TestUtil.h"
#include "loom/Mapper/Graph.h"

using namespace loom;

int main() {
  // Empty graph.
  Graph graph;
  TEST_ASSERT(graph.countNodes() == 0);
  TEST_ASSERT(graph.countPorts() == 0);
  TEST_ASSERT(graph.countEdges() == 0);

  // Iteration over empty graph yields nothing.
  for (Node *n : graph.nodeRange()) {
    (void)n;
    TEST_ASSERT(false);
  }
  for (Port *p : graph.portRange()) {
    (void)p;
    TEST_ASSERT(false);
  }
  for (Edge *e : graph.edgeRange()) {
    (void)e;
    TEST_ASSERT(false);
  }

  // Getters on empty graph return nullptr.
  TEST_ASSERT(graph.getNode(0) == nullptr);
  TEST_ASSERT(graph.getPort(0) == nullptr);
  TEST_ASSERT(graph.getEdge(0) == nullptr);

  // isValid on empty graph.
  TEST_ASSERT(!graph.isValid(0, EntityKind::Node));
  TEST_ASSERT(!graph.isValid(INVALID_ID, EntityKind::Node));

  // Single node with no ports.
  auto node = std::make_unique<Node>();
  IdIndex nid = graph.addNode(std::move(node));
  TEST_ASSERT(nid == 0);
  TEST_ASSERT(graph.countNodes() == 1);
  TEST_ASSERT(graph.getNode(nid)->inputPorts.empty());
  TEST_ASSERT(graph.getNode(nid)->outputPorts.empty());

  // Remove the single node.
  graph.removeNode(nid);
  TEST_ASSERT(graph.countNodes() == 0);
  TEST_ASSERT(graph.nodes.size() == 1); // null slot preserved

  // Remove on already-deleted slot is a no-op.
  graph.removeNode(nid);
  TEST_ASSERT(graph.countNodes() == 0);

  // Remove on out-of-range ID is a no-op.
  graph.removeNode(999);
  graph.removePort(999);
  graph.removeEdge(999);

  return 0;
}
