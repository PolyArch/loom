//===-- graph_sentinel_adg.cpp - ADG sentinel test ----------------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Create ADG sentinels (ModuleInputNode with output port, ModuleOutputNode
// with input port); verify directions and connectivity.
//
//===----------------------------------------------------------------------===//

#include "TestUtil.h"
#include "loom/Mapper/Graph.h"

using namespace loom;

int main() {
  Graph graph;

  // Create an operation node with 2 inputs and 1 output.
  auto opNode = std::make_unique<Node>();
  opNode->kind = Node::OperationNode;
  IdIndex opId = graph.addNode(std::move(opNode));

  IdIndex opInPorts[2];
  for (int i = 0; i < 2; ++i) {
    auto port = std::make_unique<Port>();
    port->parentNode = opId;
    port->direction = Port::Input;
    opInPorts[i] = graph.addPort(std::move(port));
    graph.getNode(opId)->inputPorts.push_back(opInPorts[i]);
  }
  IdIndex opOutPort;
  {
    auto port = std::make_unique<Port>();
    port->parentNode = opId;
    port->direction = Port::Output;
    opOutPort = graph.addPort(std::move(port));
    graph.getNode(opId)->outputPorts.push_back(opOutPort);
  }

  // 2 ModuleInputNode sentinels (ADG input arguments).
  IdIndex sentinelOutPorts[2];
  for (int i = 0; i < 2; ++i) {
    auto sentinel = std::make_unique<Node>();
    sentinel->kind = Node::ModuleInputNode;
    IdIndex sId = graph.addNode(std::move(sentinel));

    auto port = std::make_unique<Port>();
    port->parentNode = sId;
    port->direction = Port::Output;
    sentinelOutPorts[i] = graph.addPort(std::move(port));
    graph.getNode(sId)->outputPorts.push_back(sentinelOutPorts[i]);
  }

  // 1 ModuleOutputNode sentinel (ADG output).
  IdIndex sentinelInPort;
  {
    auto sentinel = std::make_unique<Node>();
    sentinel->kind = Node::ModuleOutputNode;
    IdIndex sId = graph.addNode(std::move(sentinel));

    auto port = std::make_unique<Port>();
    port->parentNode = sId;
    port->direction = Port::Input;
    sentinelInPort = graph.addPort(std::move(port));
    graph.getNode(sId)->inputPorts.push_back(sentinelInPort);
  }

  // Connect sentinel outputs to operation inputs.
  for (int i = 0; i < 2; ++i) {
    auto edge = std::make_unique<Edge>();
    edge->srcPort = sentinelOutPorts[i];
    edge->dstPort = opInPorts[i];
    IdIndex eId = graph.addEdge(std::move(edge));
    graph.getPort(sentinelOutPorts[i])->connectedEdges.push_back(eId);
    graph.getPort(opInPorts[i])->connectedEdges.push_back(eId);
  }

  // Connect operation output to sentinel input.
  {
    auto edge = std::make_unique<Edge>();
    edge->srcPort = opOutPort;
    edge->dstPort = sentinelInPort;
    IdIndex eId = graph.addEdge(std::move(edge));
    graph.getPort(opOutPort)->connectedEdges.push_back(eId);
    graph.getPort(sentinelInPort)->connectedEdges.push_back(eId);
  }

  // Verify counts: 1 op + 2 input sentinels + 1 output sentinel = 4 nodes.
  TEST_ASSERT(graph.countNodes() == 4);
  // 2 op_in + 1 op_out + 2 sentinel_out + 1 sentinel_in = 6 ports.
  TEST_ASSERT(graph.countPorts() == 6);
  TEST_ASSERT(graph.countEdges() == 3);

  // Verify sentinel port directions.
  TEST_ASSERT(graph.getPort(sentinelOutPorts[0])->direction == Port::Output);
  TEST_ASSERT(graph.getPort(sentinelOutPorts[1])->direction == Port::Output);
  TEST_ASSERT(graph.getPort(sentinelInPort)->direction == Port::Input);

  // Each sentinel port has exactly 1 connected edge.
  TEST_ASSERT(graph.getPort(sentinelOutPorts[0])->connectedEdges.size() == 1);
  TEST_ASSERT(graph.getPort(sentinelOutPorts[1])->connectedEdges.size() == 1);
  TEST_ASSERT(graph.getPort(sentinelInPort)->connectedEdges.size() == 1);

  // Verify edge endpoints.
  for (int i = 0; i < 2; ++i) {
    IdIndex eId = graph.getPort(sentinelOutPorts[i])->connectedEdges[0];
    Edge *e = graph.getEdge(eId);
    TEST_ASSERT(e->srcPort == sentinelOutPorts[i]);
    TEST_ASSERT(e->dstPort == opInPorts[i]);
  }
  {
    IdIndex eId = graph.getPort(sentinelInPort)->connectedEdges[0];
    Edge *e = graph.getEdge(eId);
    TEST_ASSERT(e->srcPort == opOutPort);
    TEST_ASSERT(e->dstPort == sentinelInPort);
  }

  return 0;
}
