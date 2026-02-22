//===-- graph_fan_out.cpp - Fan-out edge test ----------------------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Output port with 3 connectedEdges; verify all accessible.
//
//===----------------------------------------------------------------------===//

#include "TestUtil.h"
#include "loom/Mapper/Graph.h"

using namespace loom;

int main() {
  Graph graph;

  // Source node with one output port.
  auto src = std::make_unique<Node>();
  IdIndex srcId = graph.addNode(std::move(src));

  auto srcOut = std::make_unique<Port>();
  srcOut->parentNode = srcId;
  srcOut->direction = Port::Output;
  IdIndex srcOutId = graph.addPort(std::move(srcOut));
  graph.getNode(srcId)->outputPorts.push_back(srcOutId);

  // 3 destination nodes, each with one input port.
  IdIndex dstInIds[3];
  for (int i = 0; i < 3; ++i) {
    auto dst = std::make_unique<Node>();
    IdIndex dstId = graph.addNode(std::move(dst));

    auto dstIn = std::make_unique<Port>();
    dstIn->parentNode = dstId;
    dstIn->direction = Port::Input;
    dstInIds[i] = graph.addPort(std::move(dstIn));
    graph.getNode(dstId)->inputPorts.push_back(dstInIds[i]);
  }

  // Create 3 fan-out edges from the source output port.
  IdIndex edgeIds[3];
  for (int i = 0; i < 3; ++i) {
    auto edge = std::make_unique<Edge>();
    edge->srcPort = srcOutId;
    edge->dstPort = dstInIds[i];
    edgeIds[i] = graph.addEdge(std::move(edge));
    graph.getPort(srcOutId)->connectedEdges.push_back(edgeIds[i]);
    graph.getPort(dstInIds[i])->connectedEdges.push_back(edgeIds[i]);
  }

  // Source port has 3 connected edges (fan-out).
  TEST_ASSERT(graph.getPort(srcOutId)->connectedEdges.size() == 3);
  TEST_ASSERT(graph.countEdges() == 3);

  // Each edge has correct endpoints.
  for (int i = 0; i < 3; ++i) {
    Edge *e = graph.getEdge(edgeIds[i]);
    TEST_ASSERT(e != nullptr);
    TEST_ASSERT(e->srcPort == srcOutId);
    TEST_ASSERT(e->dstPort == dstInIds[i]);
  }

  // Remove middle edge; verify fan-out shrinks.
  graph.removeEdge(edgeIds[1]);
  TEST_ASSERT(graph.getPort(srcOutId)->connectedEdges.size() == 2);
  TEST_ASSERT(graph.countEdges() == 2);
  TEST_ASSERT(graph.getEdge(edgeIds[0]) != nullptr);
  TEST_ASSERT(graph.getEdge(edgeIds[1]) == nullptr);
  TEST_ASSERT(graph.getEdge(edgeIds[2]) != nullptr);

  return 0;
}
