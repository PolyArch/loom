//===-- DFGBuilder.cpp - DFG extraction from Handshake MLIR -------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include "loom/Mapper/DFGBuilder.h"

#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"

#include "llvm/ADT/DenseMap.h"

namespace loom {

Graph DFGBuilder::build(circt::handshake::FuncOp funcOp) {
  Graph graph(funcOp.getContext());

  // Map from MLIR Value -> output port ID that produces it.
  llvm::DenseMap<mlir::Value, IdIndex> valueToPort;

  mlir::Builder builder(funcOp.getContext());

  // --- Phase 1: Create OperationNode for each non-terminator operation ---
  auto &block = funcOp.getBody().front();
  for (auto &op : block) {
    if (op.hasTrait<mlir::OpTrait::IsTerminator>())
      continue;

    auto node = std::make_unique<Node>();
    node->kind = Node::OperationNode;
    node->attributes.push_back(
        builder.getNamedAttr("op_name",
                             builder.getStringAttr(op.getName().getStringRef())));

    IdIndex nodeId = graph.addNode(std::move(node));

    // Create input ports for each operand.
    for (unsigned i = 0; i < op.getNumOperands(); ++i) {
      auto port = std::make_unique<Port>();
      port->parentNode = nodeId;
      port->direction = Port::Input;
      port->type = op.getOperand(i).getType();
      IdIndex portId = graph.addPort(std::move(port));
      graph.getNode(nodeId)->inputPorts.push_back(portId);
    }

    // Create output ports for each result.
    for (unsigned i = 0; i < op.getNumResults(); ++i) {
      auto port = std::make_unique<Port>();
      port->parentNode = nodeId;
      port->direction = Port::Output;
      port->type = op.getResult(i).getType();
      IdIndex portId = graph.addPort(std::move(port));
      graph.getNode(nodeId)->outputPorts.push_back(portId);

      // Register this result value -> output port mapping.
      valueToPort[op.getResult(i)] = portId;
    }
  }

  // --- Phase 2: Create ModuleInputNode sentinels for block arguments ---
  for (auto arg : block.getArguments()) {
    auto node = std::make_unique<Node>();
    node->kind = Node::ModuleInputNode;
    node->attributes.push_back(
        builder.getNamedAttr("arg_index",
                             builder.getI32IntegerAttr(arg.getArgNumber())));

    IdIndex nodeId = graph.addNode(std::move(node));

    // Sentinel input nodes have output ports (they produce values).
    auto port = std::make_unique<Port>();
    port->parentNode = nodeId;
    port->direction = Port::Output;
    port->type = arg.getType();
    IdIndex portId = graph.addPort(std::move(port));
    graph.getNode(nodeId)->outputPorts.push_back(portId);

    valueToPort[arg] = portId;
  }

  // --- Phase 3: Create ModuleOutputNode sentinels for return operands ---
  // Find the return operation.
  mlir::Operation *returnOp = block.getTerminator();
  if (returnOp) {
    for (unsigned i = 0; i < returnOp->getNumOperands(); ++i) {
      auto node = std::make_unique<Node>();
      node->kind = Node::ModuleOutputNode;
      node->attributes.push_back(
          builder.getNamedAttr("ret_index",
                               builder.getI32IntegerAttr(i)));

      IdIndex nodeId = graph.addNode(std::move(node));

      // Sentinel output nodes have input ports (they consume values).
      auto port = std::make_unique<Port>();
      port->parentNode = nodeId;
      port->direction = Port::Input;
      port->type = returnOp->getOperand(i).getType();
      IdIndex portId = graph.addPort(std::move(port));
      graph.getNode(nodeId)->inputPorts.push_back(portId);
    }
  }

  // --- Phase 4: Create edges for SSA value uses ---
  // For each non-terminator operation's operands, create edge from
  // producing port to consuming port.
  IdIndex nodeIdx = 0;
  for (auto &op : block) {
    if (op.hasTrait<mlir::OpTrait::IsTerminator>())
      continue;

    Node *node = graph.getNode(nodeIdx);
    for (unsigned i = 0; i < op.getNumOperands(); ++i) {
      mlir::Value operand = op.getOperand(i);
      auto srcIt = valueToPort.find(operand);
      if (srcIt == valueToPort.end())
        continue;

      IdIndex srcPortId = srcIt->second;
      IdIndex dstPortId = node->inputPorts[i];

      auto edge = std::make_unique<Edge>();
      edge->srcPort = srcPortId;
      edge->dstPort = dstPortId;
      IdIndex edgeId = graph.addEdge(std::move(edge));

      graph.getPort(srcPortId)->connectedEdges.push_back(edgeId);
      graph.getPort(dstPortId)->connectedEdges.push_back(edgeId);
    }
    ++nodeIdx;
  }

  // Also create edges for return operands -> sentinel output nodes.
  if (returnOp) {
    // Sentinel output nodes start after all operation nodes and input sentinels.
    IdIndex outputSentinelStart =
        nodeIdx + static_cast<IdIndex>(block.getNumArguments());
    for (unsigned i = 0; i < returnOp->getNumOperands(); ++i) {
      mlir::Value operand = returnOp->getOperand(i);
      auto srcIt = valueToPort.find(operand);
      if (srcIt == valueToPort.end())
        continue;

      Node *outputNode = graph.getNode(outputSentinelStart + i);
      if (!outputNode || outputNode->inputPorts.empty())
        continue;

      IdIndex srcPortId = srcIt->second;
      IdIndex dstPortId = outputNode->inputPorts[0];

      auto edge = std::make_unique<Edge>();
      edge->srcPort = srcPortId;
      edge->dstPort = dstPortId;
      IdIndex edgeId = graph.addEdge(std::move(edge));

      graph.getPort(srcPortId)->connectedEdges.push_back(edgeId);
      graph.getPort(dstPortId)->connectedEdges.push_back(edgeId);
    }
  }

  return graph;
}

} // namespace loom
