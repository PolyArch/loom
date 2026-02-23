//===-- dfg_extraction_invariants.cpp - DFG structural invariants --*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Verify structural invariants that a correctly extracted DFG must satisfy:
// sentinel node port directions, sentinel edge connectivity, fan-out,
// node count decomposition, and edge source/destination direction invariant.
//
//===----------------------------------------------------------------------===//

#include "TestUtil.h"
#include "loom/Mapper/Graph.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"

using namespace loom;

namespace {

/// Helper: add a named string attribute to a node.
void setStringAttr(Node *node, mlir::MLIRContext &ctx,
                   llvm::StringRef name, llvm::StringRef value) {
  node->attributes.push_back(mlir::NamedAttribute(
      mlir::StringAttr::get(&ctx, name), mlir::StringAttr::get(&ctx, value)));
}

/// Helper: create an operation node with op_name and resource_class.
IdIndex addOpNode(Graph &g, mlir::MLIRContext &ctx,
                  llvm::StringRef opName, llvm::StringRef resClass,
                  unsigned numIn = 1, unsigned numOut = 1) {
  auto node = std::make_unique<Node>();
  node->kind = Node::OperationNode;
  setStringAttr(node.get(), ctx, "op_name", opName);
  setStringAttr(node.get(), ctx, "resource_class", resClass);
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

/// Helper: create a ModuleInputNode sentinel with one output port.
IdIndex addInputSentinel(Graph &g) {
  auto node = std::make_unique<Node>();
  node->kind = Node::ModuleInputNode;
  IdIndex nodeId = g.addNode(std::move(node));

  auto port = std::make_unique<Port>();
  port->parentNode = nodeId;
  port->direction = Port::Output;
  IdIndex portId = g.addPort(std::move(port));
  g.getNode(nodeId)->outputPorts.push_back(portId);

  return nodeId;
}

/// Helper: create a ModuleOutputNode sentinel with one input port.
IdIndex addOutputSentinel(Graph &g) {
  auto node = std::make_unique<Node>();
  node->kind = Node::ModuleOutputNode;
  IdIndex nodeId = g.addNode(std::move(node));

  auto port = std::make_unique<Port>();
  port->parentNode = nodeId;
  port->direction = Port::Input;
  IdIndex portId = g.addPort(std::move(port));
  g.getNode(nodeId)->inputPorts.push_back(portId);

  return nodeId;
}

/// Helper: add an edge between a source output port and a destination input port.
IdIndex addEdgeBetween(Graph &g, IdIndex srcNode, unsigned srcPortIdx,
                       IdIndex dstNode, unsigned dstPortIdx) {
  IdIndex srcPort = g.getNode(srcNode)->outputPorts[srcPortIdx];
  IdIndex dstPort = g.getNode(dstNode)->inputPorts[dstPortIdx];

  auto edge = std::make_unique<Edge>();
  edge->srcPort = srcPort;
  edge->dstPort = dstPort;
  IdIndex eid = g.addEdge(std::move(edge));
  g.getPort(srcPort)->connectedEdges.push_back(eid);
  g.getPort(dstPort)->connectedEdges.push_back(eid);
  return eid;
}

} // namespace

int main() {
  mlir::MLIRContext ctx;
  ctx.loadAllAvailableDialects();

  // Test 1: Sentinel node types - ModuleInputNode has output ports only,
  // ModuleOutputNode has input ports only.
  {
    Graph dfg(&ctx);

    IdIndex inSentinel = addInputSentinel(dfg);
    IdIndex outSentinel = addOutputSentinel(dfg);

    // ModuleInputNode: must have output ports only.
    Node *inNode = dfg.getNode(inSentinel);
    TEST_ASSERT(inNode->kind == Node::ModuleInputNode);
    TEST_ASSERT(inNode->inputPorts.empty());
    TEST_ASSERT(inNode->outputPorts.size() == 1);
    TEST_ASSERT(dfg.getPort(inNode->outputPorts[0])->direction == Port::Output);

    // ModuleOutputNode: must have input ports only.
    Node *outNode = dfg.getNode(outSentinel);
    TEST_ASSERT(outNode->kind == Node::ModuleOutputNode);
    TEST_ASSERT(outNode->outputPorts.empty());
    TEST_ASSERT(outNode->inputPorts.size() == 1);
    TEST_ASSERT(dfg.getPort(outNode->inputPorts[0])->direction == Port::Input);
  }

  // Test 2: Sentinel edge connectivity - all input sentinels have at least
  // one outgoing edge to operation nodes.
  {
    Graph dfg(&ctx);

    // Build: 2 input sentinels -> op -> output sentinel.
    IdIndex in0 = addInputSentinel(dfg);
    IdIndex in1 = addInputSentinel(dfg);
    IdIndex op = addOpNode(dfg, ctx, "arith.addi", "functional", 2, 1);
    IdIndex out = addOutputSentinel(dfg);

    // Connect input sentinels to operation inputs.
    addEdgeBetween(dfg, in0, 0, op, 0);
    addEdgeBetween(dfg, in1, 0, op, 1);
    // Connect operation output to output sentinel.
    addEdgeBetween(dfg, op, 0, out, 0);

    // Verify each input sentinel has at least one outgoing edge.
    for (IdIndex sentinelId : {in0, in1}) {
      Node *sentinel = dfg.getNode(sentinelId);
      TEST_ASSERT(sentinel->kind == Node::ModuleInputNode);
      TEST_ASSERT(!sentinel->outputPorts.empty());

      bool hasOutgoingEdge = false;
      for (IdIndex portId : sentinel->outputPorts) {
        Port *port = dfg.getPort(portId);
        if (!port->connectedEdges.empty())
          hasOutgoingEdge = true;
      }
      TEST_ASSERT(hasOutgoingEdge);

      // Verify the edge destination is an operation node.
      IdIndex outPort = sentinel->outputPorts[0];
      IdIndex edgeId = dfg.getPort(outPort)->connectedEdges[0];
      Edge *edge = dfg.getEdge(edgeId);
      Port *dstPort = dfg.getPort(edge->dstPort);
      Node *dstNode = dfg.getNode(dstPort->parentNode);
      TEST_ASSERT(dstNode->kind == Node::OperationNode);
    }
  }

  // Test 3: Fan-out verification - an operation node with multiple output
  // edges creates fan-out correctly (connectedEdges.size() on output port).
  {
    Graph dfg(&ctx);

    // Build: in -> op (fan-out=3) -> {out0, out1, out2}.
    IdIndex in = addInputSentinel(dfg);
    IdIndex op = addOpNode(dfg, ctx, "arith.addi", "functional", 1, 1);
    IdIndex out0 = addOutputSentinel(dfg);
    IdIndex out1 = addOutputSentinel(dfg);
    IdIndex out2 = addOutputSentinel(dfg);

    addEdgeBetween(dfg, in, 0, op, 0);

    // Fan-out: connect op's output port to 3 output sentinels.
    IdIndex opOutPort = dfg.getNode(op)->outputPorts[0];
    for (IdIndex outSentinel : {out0, out1, out2}) {
      IdIndex dstPort = dfg.getNode(outSentinel)->inputPorts[0];
      auto edge = std::make_unique<Edge>();
      edge->srcPort = opOutPort;
      edge->dstPort = dstPort;
      IdIndex eid = dfg.addEdge(std::move(edge));
      dfg.getPort(opOutPort)->connectedEdges.push_back(eid);
      dfg.getPort(dstPort)->connectedEdges.push_back(eid);
    }

    // Verify fan-out of 3 on the operation's output port.
    TEST_ASSERT(dfg.getPort(opOutPort)->connectedEdges.size() == 3);

    // Each output sentinel's input port has exactly 1 edge.
    for (IdIndex outSentinel : {out0, out1, out2}) {
      IdIndex inPort = dfg.getNode(outSentinel)->inputPorts[0];
      TEST_ASSERT(dfg.getPort(inPort)->connectedEdges.size() == 1);
    }
  }

  // Test 4: DFG node counts - total nodes = operation nodes + input sentinels
  // + output sentinels.
  {
    Graph dfg(&ctx);

    // 2 input sentinels + 3 operation nodes + 1 output sentinel = 6 total.
    IdIndex in0 = addInputSentinel(dfg);
    IdIndex in1 = addInputSentinel(dfg);
    IdIndex op0 = addOpNode(dfg, ctx, "arith.addi", "functional", 2, 1);
    IdIndex op1 = addOpNode(dfg, ctx, "arith.muli", "functional", 1, 1);
    IdIndex op2 = addOpNode(dfg, ctx, "arith.addi", "functional", 1, 1);
    IdIndex out = addOutputSentinel(dfg);

    // Connect: in0,in1 -> op0 -> op1 -> op2 -> out.
    addEdgeBetween(dfg, in0, 0, op0, 0);
    addEdgeBetween(dfg, in1, 0, op0, 1);
    addEdgeBetween(dfg, op0, 0, op1, 0);
    addEdgeBetween(dfg, op1, 0, op2, 0);
    addEdgeBetween(dfg, op2, 0, out, 0);

    // Count by kind.
    size_t inputSentinels = 0, outputSentinels = 0, opNodes = 0;
    for (auto *node : dfg.nodeRange()) {
      if (node->kind == Node::ModuleInputNode)
        ++inputSentinels;
      else if (node->kind == Node::ModuleOutputNode)
        ++outputSentinels;
      else if (node->kind == Node::OperationNode)
        ++opNodes;
    }

    TEST_ASSERT(inputSentinels == 2);
    TEST_ASSERT(outputSentinels == 1);
    TEST_ASSERT(opNodes == 3);
    TEST_ASSERT(dfg.countNodes() == inputSentinels + outputSentinels + opNodes);
    TEST_ASSERT(dfg.countNodes() == 6);

    // Suppress unused-variable warnings for node IDs used only for construction.
    (void)in0; (void)in1; (void)op0; (void)op1; (void)op2; (void)out;
  }

  // Test 5: Edge source/destination invariant - every edge's srcPort is an
  // Output port, dstPort is an Input port.
  {
    Graph dfg(&ctx);

    IdIndex in0 = addInputSentinel(dfg);
    IdIndex in1 = addInputSentinel(dfg);
    IdIndex op0 = addOpNode(dfg, ctx, "arith.addi", "functional", 2, 1);
    IdIndex op1 = addOpNode(dfg, ctx, "arith.muli", "functional", 1, 1);
    IdIndex out = addOutputSentinel(dfg);

    addEdgeBetween(dfg, in0, 0, op0, 0);
    addEdgeBetween(dfg, in1, 0, op0, 1);
    addEdgeBetween(dfg, op0, 0, op1, 0);
    addEdgeBetween(dfg, op1, 0, out, 0);

    // Every edge must have srcPort=Output, dstPort=Input.
    for (auto *edge : dfg.edgeRange()) {
      Port *srcPort = dfg.getPort(edge->srcPort);
      Port *dstPort = dfg.getPort(edge->dstPort);
      TEST_ASSERT(srcPort != nullptr);
      TEST_ASSERT(dstPort != nullptr);
      TEST_ASSERT(srcPort->direction == Port::Output);
      TEST_ASSERT(dstPort->direction == Port::Input);
    }

    // Verify we actually checked some edges.
    TEST_ASSERT(dfg.countEdges() == 4);
  }

  return 0;
}
