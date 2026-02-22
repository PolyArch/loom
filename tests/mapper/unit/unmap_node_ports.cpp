//===-- unmap_node_ports.cpp - unmapNode port cleanup test ---------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Verify that unmapNode() clears both node and port bindings, so that
// subsequent re-mapping does not encounter stale port state.
//
//===----------------------------------------------------------------------===//

#include "TestUtil.h"
#include "loom/Mapper/Graph.h"
#include "loom/Mapper/MappingState.h"

#include <algorithm>

using namespace loom;

namespace {

IdIndex addSimpleNode(Graph &g, Node::Kind kind = Node::OperationNode,
                      unsigned numIn = 1, unsigned numOut = 1) {
  auto node = std::make_unique<Node>();
  node->kind = kind;
  IdIndex nodeId = g.addNode(std::move(node));

  for (unsigned i = 0; i < numIn; ++i) {
    auto port = std::make_unique<Port>();
    port->parentNode = nodeId;
    port->direction = Port::Input;
    IdIndex pid = g.addPort(std::move(port));
    g.getNode(nodeId)->inputPorts.push_back(pid);
  }

  for (unsigned i = 0; i < numOut; ++i) {
    auto port = std::make_unique<Port>();
    port->parentNode = nodeId;
    port->direction = Port::Output;
    IdIndex pid = g.addPort(std::move(port));
    g.getNode(nodeId)->outputPorts.push_back(pid);
  }

  return nodeId;
}

} // namespace

int main() {
  // Test 1: unmapNode clears both node mapping and port bindings.
  {
    Graph dfg;
    Graph adg;

    IdIndex sw0 = addSimpleNode(dfg, Node::OperationNode, 2, 1);
    IdIndex hw0 = addSimpleNode(adg, Node::OperationNode, 2, 1);

    MappingState state;
    state.init(dfg, adg);

    // Map node and ports.
    auto r = state.mapNode(sw0, hw0, dfg, adg);
    TEST_ASSERT(r == ActionResult::Success);

    const Node *swNode = dfg.getNode(sw0);
    const Node *hwNode = adg.getNode(hw0);
    TEST_ASSERT(swNode && hwNode);

    // Map input ports.
    for (size_t i = 0; i < swNode->inputPorts.size() &&
                        i < hwNode->inputPorts.size();
         ++i) {
      r = state.mapPort(swNode->inputPorts[i], hwNode->inputPorts[i], dfg, adg);
      TEST_ASSERT(r == ActionResult::Success);
    }

    // Map output ports.
    for (size_t i = 0; i < swNode->outputPorts.size() &&
                        i < hwNode->outputPorts.size();
         ++i) {
      r = state.mapPort(swNode->outputPorts[i], hwNode->outputPorts[i], dfg, adg);
      TEST_ASSERT(r == ActionResult::Success);
    }

    // Verify port bindings are set.
    for (IdIndex swPort : swNode->inputPorts)
      TEST_ASSERT(state.swPortToHwPort[swPort] != INVALID_ID);
    for (IdIndex swPort : swNode->outputPorts)
      TEST_ASSERT(state.swPortToHwPort[swPort] != INVALID_ID);

    // Verify reverse port mappings.
    for (size_t i = 0; i < hwNode->inputPorts.size(); ++i) {
      IdIndex hwPort = hwNode->inputPorts[i];
      TEST_ASSERT(!state.hwPortToSwPorts[hwPort].empty());
    }
    for (size_t i = 0; i < hwNode->outputPorts.size(); ++i) {
      IdIndex hwPort = hwNode->outputPorts[i];
      TEST_ASSERT(!state.hwPortToSwPorts[hwPort].empty());
    }

    // Now unmapNode -- should clear both node and port bindings.
    r = state.unmapNode(sw0, dfg, adg);
    TEST_ASSERT(r == ActionResult::Success);

    // Node mapping should be cleared.
    TEST_ASSERT(state.swNodeToHwNode[sw0] == INVALID_ID);
    TEST_ASSERT(state.hwNodeToSwNodes[hw0].empty());

    // Port forward mappings should be cleared.
    for (IdIndex swPort : swNode->inputPorts)
      TEST_ASSERT(state.swPortToHwPort[swPort] == INVALID_ID);
    for (IdIndex swPort : swNode->outputPorts)
      TEST_ASSERT(state.swPortToHwPort[swPort] == INVALID_ID);

    // Port reverse mappings should be cleared.
    for (IdIndex hwPort : hwNode->inputPorts)
      TEST_ASSERT(state.hwPortToSwPorts[hwPort].empty());
    for (IdIndex hwPort : hwNode->outputPorts)
      TEST_ASSERT(state.hwPortToSwPorts[hwPort].empty());
  }

  // Test 2: After unmapNode, re-mapping the same node to a different HW node
  // succeeds (verifies no stale port state prevents re-mapping).
  {
    Graph dfg;
    Graph adg;

    IdIndex sw0 = addSimpleNode(dfg, Node::OperationNode, 1, 1);
    IdIndex hw0 = addSimpleNode(adg, Node::OperationNode, 1, 1);
    IdIndex hw1 = addSimpleNode(adg, Node::OperationNode, 1, 1);

    MappingState state;
    state.init(dfg, adg);

    const Node *swNode = dfg.getNode(sw0);
    const Node *hwNode0 = adg.getNode(hw0);
    const Node *hwNode1 = adg.getNode(hw1);

    // Map node and ports to hw0.
    state.mapNode(sw0, hw0, dfg, adg);
    state.mapPort(swNode->inputPorts[0], hwNode0->inputPorts[0], dfg, adg);
    state.mapPort(swNode->outputPorts[0], hwNode0->outputPorts[0], dfg, adg);

    // Unmap.
    state.unmapNode(sw0, dfg, adg);

    // Re-map to hw1.
    auto r = state.mapNode(sw0, hw1, dfg, adg);
    TEST_ASSERT(r == ActionResult::Success);
    r = state.mapPort(swNode->inputPorts[0], hwNode1->inputPorts[0], dfg, adg);
    TEST_ASSERT(r == ActionResult::Success);
    r = state.mapPort(swNode->outputPorts[0], hwNode1->outputPorts[0], dfg, adg);
    TEST_ASSERT(r == ActionResult::Success);

    // Verify new mapping.
    TEST_ASSERT(state.swNodeToHwNode[sw0] == hw1);
    TEST_ASSERT(state.swPortToHwPort[swNode->inputPorts[0]] ==
                hwNode1->inputPorts[0]);
    TEST_ASSERT(state.swPortToHwPort[swNode->outputPorts[0]] ==
                hwNode1->outputPorts[0]);

    // Old HW ports should be clean.
    TEST_ASSERT(state.hwPortToSwPorts[hwNode0->inputPorts[0]].empty());
    TEST_ASSERT(state.hwPortToSwPorts[hwNode0->outputPorts[0]].empty());
  }

  // Test 3: unmapNode with no port bindings still succeeds cleanly.
  {
    Graph dfg;
    Graph adg;

    IdIndex sw0 = addSimpleNode(dfg, Node::OperationNode, 1, 1);
    IdIndex hw0 = addSimpleNode(adg, Node::OperationNode, 1, 1);

    MappingState state;
    state.init(dfg, adg);

    // Map only the node, no ports.
    state.mapNode(sw0, hw0, dfg, adg);
    auto r = state.unmapNode(sw0, dfg, adg);
    TEST_ASSERT(r == ActionResult::Success);
    TEST_ASSERT(state.swNodeToHwNode[sw0] == INVALID_ID);
  }

  // Test 4: State validity holds after unmapNode with port cleanup.
  {
    Graph dfg;
    Graph adg;

    IdIndex sw0 = addSimpleNode(dfg, Node::OperationNode, 1, 1);
    IdIndex sw1 = addSimpleNode(dfg, Node::OperationNode, 1, 1);
    IdIndex hw0 = addSimpleNode(adg, Node::OperationNode, 1, 1);
    IdIndex hw1 = addSimpleNode(adg, Node::OperationNode, 1, 1);

    MappingState state;
    state.init(dfg, adg);

    const Node *swNode0 = dfg.getNode(sw0);
    const Node *swNode1 = dfg.getNode(sw1);
    const Node *hwNode0 = adg.getNode(hw0);
    const Node *hwNode1 = adg.getNode(hw1);

    // Map both nodes with ports.
    state.mapNode(sw0, hw0, dfg, adg);
    state.mapPort(swNode0->inputPorts[0], hwNode0->inputPorts[0], dfg, adg);
    state.mapPort(swNode0->outputPorts[0], hwNode0->outputPorts[0], dfg, adg);

    state.mapNode(sw1, hw1, dfg, adg);
    state.mapPort(swNode1->inputPorts[0], hwNode1->inputPorts[0], dfg, adg);
    state.mapPort(swNode1->outputPorts[0], hwNode1->outputPorts[0], dfg, adg);

    // Unmap sw0 only.
    state.unmapNode(sw0, dfg, adg);

    // sw1 should still be fully mapped.
    TEST_ASSERT(state.swNodeToHwNode[sw1] == hw1);
    TEST_ASSERT(state.swPortToHwPort[swNode1->inputPorts[0]] ==
                hwNode1->inputPorts[0]);

    // Overall state validity.
    TEST_ASSERT(state.isValid());
  }

  return 0;
}
