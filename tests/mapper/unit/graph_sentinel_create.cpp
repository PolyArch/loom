//===-- graph_sentinel_create.cpp - Sentinel node creation test ---*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Create ModuleInputNode/ModuleOutputNode, verify kind and ports.
//
//===----------------------------------------------------------------------===//

#include "TestUtil.h"
#include "loom/Mapper/Graph.h"

using namespace loom;

int main() {
  Graph graph;

  // ModuleInputNode sentinel: one output port.
  auto inputNode = std::make_unique<Node>();
  inputNode->kind = Node::ModuleInputNode;
  IdIndex inNodeId = graph.addNode(std::move(inputNode));

  auto inOutPort = std::make_unique<Port>();
  inOutPort->parentNode = inNodeId;
  inOutPort->direction = Port::Output;
  IdIndex inOutPortId = graph.addPort(std::move(inOutPort));
  graph.getNode(inNodeId)->outputPorts.push_back(inOutPortId);

  // ModuleOutputNode sentinel: one input port.
  auto outputNode = std::make_unique<Node>();
  outputNode->kind = Node::ModuleOutputNode;
  IdIndex outNodeId = graph.addNode(std::move(outputNode));

  auto outInPort = std::make_unique<Port>();
  outInPort->parentNode = outNodeId;
  outInPort->direction = Port::Input;
  IdIndex outInPortId = graph.addPort(std::move(outInPort));
  graph.getNode(outNodeId)->inputPorts.push_back(outInPortId);

  // Verify kinds.
  TEST_ASSERT(graph.getNode(inNodeId)->kind == Node::ModuleInputNode);
  TEST_ASSERT(graph.getNode(outNodeId)->kind == Node::ModuleOutputNode);

  // ModuleInputNode: output ports only.
  TEST_ASSERT(graph.getNode(inNodeId)->inputPorts.empty());
  TEST_ASSERT(graph.getNode(inNodeId)->outputPorts.size() == 1);

  // ModuleOutputNode: input ports only.
  TEST_ASSERT(graph.getNode(outNodeId)->inputPorts.size() == 1);
  TEST_ASSERT(graph.getNode(outNodeId)->outputPorts.empty());

  // Verify port directions.
  TEST_ASSERT(graph.getPort(inOutPortId)->direction == Port::Output);
  TEST_ASSERT(graph.getPort(outInPortId)->direction == Port::Input);

  return 0;
}
