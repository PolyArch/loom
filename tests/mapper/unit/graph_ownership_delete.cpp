//===-- graph_ownership_delete.cpp - Shared port ownership test ---*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Delete FU node sharing ports with virtual node -> ports intact.
// Models the temporal PE scenario where FU nodes reference ports owned by
// the virtual temporal PE node.
//
//===----------------------------------------------------------------------===//

#include "TestUtil.h"
#include "loom/Mapper/Graph.h"

using namespace loom;

int main() {
  Graph graph;

  // Virtual temporal PE node (port owner).
  auto virtualNode = std::make_unique<Node>();
  IdIndex virtualId = graph.addNode(std::move(virtualNode));

  // Create 2 input and 2 output ports owned by the virtual node.
  IdIndex inPorts[2], outPorts[2];
  for (int i = 0; i < 2; ++i) {
    auto port = std::make_unique<Port>();
    port->parentNode = virtualId;
    port->direction = Port::Input;
    inPorts[i] = graph.addPort(std::move(port));
    graph.getNode(virtualId)->inputPorts.push_back(inPorts[i]);
  }
  for (int i = 0; i < 2; ++i) {
    auto port = std::make_unique<Port>();
    port->parentNode = virtualId;
    port->direction = Port::Output;
    outPorts[i] = graph.addPort(std::move(port));
    graph.getNode(virtualId)->outputPorts.push_back(outPorts[i]);
  }

  // Create 2 FU nodes that reference (but don't own) the same ports.
  IdIndex fuIds[2];
  for (int i = 0; i < 2; ++i) {
    auto fu = std::make_unique<Node>();
    fuIds[i] = graph.addNode(std::move(fu));
    graph.getNode(fuIds[i])->inputPorts.push_back(inPorts[0]);
    graph.getNode(fuIds[i])->inputPorts.push_back(inPorts[1]);
    graph.getNode(fuIds[i])->outputPorts.push_back(outPorts[0]);
    graph.getNode(fuIds[i])->outputPorts.push_back(outPorts[1]);
  }

  TEST_ASSERT(graph.countNodes() == 3);
  TEST_ASSERT(graph.countPorts() == 4);

  // Delete FU node 0. Ports are owned by virtual node (parentNode != fuIds[0]),
  // so they must NOT be deleted.
  graph.removeNode(fuIds[0]);

  TEST_ASSERT(graph.getNode(fuIds[0]) == nullptr);
  TEST_ASSERT(graph.countNodes() == 2);

  // All ports still exist.
  for (int i = 0; i < 2; ++i) {
    TEST_ASSERT(graph.getPort(inPorts[i]) != nullptr);
    TEST_ASSERT(graph.getPort(outPorts[i]) != nullptr);
  }
  TEST_ASSERT(graph.countPorts() == 4);

  // FU node 1 still references the ports.
  Node *fu1 = graph.getNode(fuIds[1]);
  TEST_ASSERT(fu1->inputPorts.size() == 2);
  TEST_ASSERT(fu1->outputPorts.size() == 2);

  // Virtual node still owns all ports.
  Node *vn = graph.getNode(virtualId);
  TEST_ASSERT(vn->inputPorts.size() == 2);
  TEST_ASSERT(vn->outputPorts.size() == 2);

  // Now delete the virtual node: ports should cascade-delete since it owns them.
  graph.removeNode(virtualId);
  TEST_ASSERT(graph.getNode(virtualId) == nullptr);
  TEST_ASSERT(graph.countPorts() == 0);

  return 0;
}
