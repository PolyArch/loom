//===-- graph_id_stability.cpp - ID stability after deletion ------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Delete middle node; verify subsequent IDs are unchanged.
//
//===----------------------------------------------------------------------===//

#include "TestUtil.h"
#include "loom/Mapper/Graph.h"

using namespace loom;

int main() {
  Graph graph;

  // Create 5 nodes, each with one input and one output port.
  for (IdIndex i = 0; i < 5; ++i) {
    auto node = std::make_unique<Node>();
    IdIndex nid = graph.addNode(std::move(node));

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

  TEST_ASSERT(graph.countNodes() == 5);

  // Delete middle node (id=2). Its ports (ids 4,5) should also be deleted.
  graph.removeNode(2);

  TEST_ASSERT(graph.getNode(2) == nullptr);
  TEST_ASSERT(!graph.isValid(2, EntityKind::Node));

  // Other nodes retain their original IDs.
  TEST_ASSERT(graph.getNode(0) != nullptr);
  TEST_ASSERT(graph.getNode(1) != nullptr);
  TEST_ASSERT(graph.getNode(3) != nullptr);
  TEST_ASSERT(graph.getNode(4) != nullptr);

  // Count reflects deletion; vector size unchanged (null slot preserved).
  TEST_ASSERT(graph.countNodes() == 4);
  TEST_ASSERT(graph.nodes.size() == 5);

  // Node 3's ports still have correct parentNode.
  // Port layout: node3 has ports at indices 6 (in) and 7 (out).
  TEST_ASSERT(graph.getPort(6) != nullptr);
  TEST_ASSERT(graph.getPort(6)->parentNode == 3);
  TEST_ASSERT(graph.getPort(7) != nullptr);
  TEST_ASSERT(graph.getPort(7)->parentNode == 3);

  // Node 4's ports also intact.
  TEST_ASSERT(graph.getPort(8) != nullptr);
  TEST_ASSERT(graph.getPort(8)->parentNode == 4);

  return 0;
}
