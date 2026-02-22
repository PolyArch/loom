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
#include "llvm/Support/raw_ostream.h"

namespace loom {

Graph DFGBuilder::build(circt::handshake::FuncOp funcOp) {
  Graph graph(funcOp.getContext());

  // Map from MLIR Value -> output port ID that produces it.
  // Populated completely in Pass 1 before any edges are created in Pass 2.
  llvm::DenseMap<mlir::Value, IdIndex> valueToPort;

  // Map from Operation* -> graph node ID, for robust lookup in Pass 2.
  llvm::DenseMap<mlir::Operation *, IdIndex> opToNode;

  mlir::Builder builder(funcOp.getContext());
  auto &block = funcOp.getBody().front();

  //===--------------------------------------------------------------------===//
  // PASS 1: Create ALL nodes, ALL ports, and populate the complete valueToPort
  // map. No edges are created in this pass. This ensures that forward
  // references in graph regions (e.g., dataflow.carry using a value defined
  // later) are fully resolved before edge creation.
  //===--------------------------------------------------------------------===//

  // 1a. Create OperationNode for each non-terminator operation.
  for (auto &op : block) {
    if (op.hasTrait<mlir::OpTrait::IsTerminator>())
      continue;

    auto node = std::make_unique<Node>();
    node->kind = Node::OperationNode;
    node->attributes.push_back(
        builder.getNamedAttr("op_name",
                             builder.getStringAttr(op.getName().getStringRef())));

    IdIndex nodeId = graph.addNode(std::move(node));
    opToNode[&op] = nodeId;

    // Create input ports for each operand.
    for (unsigned i = 0; i < op.getNumOperands(); ++i) {
      auto port = std::make_unique<Port>();
      port->parentNode = nodeId;
      port->direction = Port::Input;
      port->type = op.getOperand(i).getType();
      IdIndex portId = graph.addPort(std::move(port));
      graph.getNode(nodeId)->inputPorts.push_back(portId);
    }

    // Create output ports for each result and register in valueToPort.
    for (unsigned i = 0; i < op.getNumResults(); ++i) {
      auto port = std::make_unique<Port>();
      port->parentNode = nodeId;
      port->direction = Port::Output;
      port->type = op.getResult(i).getType();
      IdIndex portId = graph.addPort(std::move(port));
      graph.getNode(nodeId)->outputPorts.push_back(portId);

      valueToPort[op.getResult(i)] = portId;
    }
  }

  // 1b. Create ModuleInputNode sentinels for block arguments.
  for (auto arg : block.getArguments()) {
    auto node = std::make_unique<Node>();
    node->kind = Node::ModuleInputNode;
    node->attributes.push_back(
        builder.getNamedAttr("arg_index",
                             builder.getI32IntegerAttr(arg.getArgNumber())));

    IdIndex nodeId = graph.addNode(std::move(node));

    auto port = std::make_unique<Port>();
    port->parentNode = nodeId;
    port->direction = Port::Output;
    port->type = arg.getType();
    IdIndex portId = graph.addPort(std::move(port));
    graph.getNode(nodeId)->outputPorts.push_back(portId);

    valueToPort[arg] = portId;
  }

  // 1c. Create ModuleOutputNode sentinels for return operands.
  mlir::Operation *returnOp = block.getTerminator();
  llvm::SmallVector<IdIndex> outputSentinelNodes;
  if (returnOp) {
    for (unsigned i = 0; i < returnOp->getNumOperands(); ++i) {
      auto node = std::make_unique<Node>();
      node->kind = Node::ModuleOutputNode;
      node->attributes.push_back(
          builder.getNamedAttr("ret_index",
                               builder.getI32IntegerAttr(i)));

      IdIndex nodeId = graph.addNode(std::move(node));
      outputSentinelNodes.push_back(nodeId);

      auto port = std::make_unique<Port>();
      port->parentNode = nodeId;
      port->direction = Port::Input;
      port->type = returnOp->getOperand(i).getType();
      IdIndex portId = graph.addPort(std::move(port));
      graph.getNode(nodeId)->inputPorts.push_back(portId);
    }
  }

  //===--------------------------------------------------------------------===//
  // PASS 2: Create ALL edges using the now-complete valueToPort map.
  // Forward references are resolved because every Value (including those
  // defined later in textual order) was registered in Pass 1.
  //===--------------------------------------------------------------------===//

  // 2a. Create edges for each non-terminator operation's operands.
  for (auto &op : block) {
    if (op.hasTrait<mlir::OpTrait::IsTerminator>())
      continue;

    auto nodeIt = opToNode.find(&op);
    if (nodeIt == opToNode.end())
      continue;
    Node *node = graph.getNode(nodeIt->second);

    for (unsigned i = 0; i < op.getNumOperands(); ++i) {
      mlir::Value operand = op.getOperand(i);
      auto srcIt = valueToPort.find(operand);
      if (srcIt == valueToPort.end()) {
        llvm::errs() << "warning: DFGBuilder: unresolved operand " << i
                     << " of " << op.getName() << "\n";
        continue;
      }

      IdIndex srcPortId = srcIt->second;
      IdIndex dstPortId = node->inputPorts[i];

      auto edge = std::make_unique<Edge>();
      edge->srcPort = srcPortId;
      edge->dstPort = dstPortId;
      IdIndex edgeId = graph.addEdge(std::move(edge));

      graph.getPort(srcPortId)->connectedEdges.push_back(edgeId);
      graph.getPort(dstPortId)->connectedEdges.push_back(edgeId);
    }
  }

  // 2b. Create edges for return operands -> sentinel output nodes.
  if (returnOp) {
    for (unsigned i = 0; i < returnOp->getNumOperands(); ++i) {
      mlir::Value operand = returnOp->getOperand(i);
      auto srcIt = valueToPort.find(operand);
      if (srcIt == valueToPort.end()) {
        llvm::errs() << "warning: DFGBuilder: unresolved return operand "
                     << i << "\n";
        continue;
      }

      Node *outputNode = graph.getNode(outputSentinelNodes[i]);
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
