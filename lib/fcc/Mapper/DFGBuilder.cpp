#include "fcc/Mapper/DFGBuilder.h"

#include "circt/Dialect/Handshake/HandshakeDialect.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/Support/raw_ostream.h"

namespace fcc {

namespace {

void setNodeAttr(Node *node, llvm::StringRef key, mlir::Attribute val,
                 mlir::MLIRContext *ctx) {
  node->attributes.push_back(
      mlir::NamedAttribute(mlir::StringAttr::get(ctx, key), val));
}

/// Get a descriptive op name for a DFG operation.
std::string getOpName(mlir::Operation *op) {
  return op->getName().getStringRef().str();
}

} // namespace

bool DFGBuilder::build(mlir::ModuleOp module, mlir::MLIRContext *ctx) {
  dfg = Graph(ctx);

  // Find the first handshake.func.
  circt::handshake::FuncOp funcOp;
  module->walk([&](circt::handshake::FuncOp func) {
    if (!funcOp)
      funcOp = func;
  });

  if (!funcOp) {
    llvm::errs() << "DFGBuilder: no handshake.func found\n";
    return false;
  }

  // Map from MLIR Value to output port ID in the DFG.
  llvm::DenseMap<mlir::Value, IdIndex> valueToPort;

  auto &body = funcOp.getBody().front();

  // Create ModuleInputNode sentinels for block arguments.
  for (auto arg : body.getArguments()) {
    auto inputNode = std::make_unique<Node>();
    inputNode->kind = Node::ModuleInputNode;

    setNodeAttr(inputNode.get(), "op_name",
                mlir::StringAttr::get(ctx, "module_input"), ctx);
    setNodeAttr(inputNode.get(), "arg_index",
                mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 32),
                                       arg.getArgNumber()),
                ctx);

    // Mark whether this is a memref or scalar input so the mapper
    // can distinguish them during sentinel binding.
    bool isMemref = mlir::isa<mlir::MemRefType>(arg.getType());
    setNodeAttr(inputNode.get(), "is_memref",
                mlir::BoolAttr::get(ctx, isMemref), ctx);

    // Create an output port for this input argument.
    auto port = std::make_unique<Port>();
    port->direction = Port::Output;
    port->type = arg.getType();
    IdIndex portId = dfg.addPort(std::move(port));
    dfg.ports[portId]->parentNode = static_cast<IdIndex>(dfg.nodes.size());
    inputNode->outputPorts.push_back(portId);

    IdIndex nodeId = dfg.addNode(std::move(inputNode));
    valueToPort[arg] = dfg.nodes[nodeId]->outputPorts[0];
  }

  // Create OperationNode for each non-terminator op.
  for (auto &op : body.getOperations()) {
    // Skip the return terminator.
    if (mlir::isa<circt::handshake::ReturnOp>(op))
      continue;

    auto opNode = std::make_unique<Node>();
    opNode->kind = Node::OperationNode;

    std::string opName = getOpName(&op);
    setNodeAttr(opNode.get(), "op_name",
                mlir::StringAttr::get(ctx, opName), ctx);

    // Copy over relevant attributes from the MLIR op.
    for (auto &attr : op.getAttrs()) {
      // Skip function_type and sym_name which are structural.
      if (attr.getName() == "function_type" || attr.getName() == "sym_name")
        continue;
      setNodeAttr(opNode.get(), attr.getName(), attr.getValue(), ctx);
    }

    // Create input ports.
    for (unsigned i = 0; i < op.getNumOperands(); ++i) {
      auto port = std::make_unique<Port>();
      port->direction = Port::Input;
      port->type = op.getOperand(i).getType();
      IdIndex portId = dfg.addPort(std::move(port));
      dfg.ports[portId]->parentNode = static_cast<IdIndex>(dfg.nodes.size());
      opNode->inputPorts.push_back(portId);
    }

    // Create output ports.
    for (unsigned i = 0; i < op.getNumResults(); ++i) {
      auto port = std::make_unique<Port>();
      port->direction = Port::Output;
      port->type = op.getResult(i).getType();
      IdIndex portId = dfg.addPort(std::move(port));
      dfg.ports[portId]->parentNode = static_cast<IdIndex>(dfg.nodes.size());
      opNode->outputPorts.push_back(portId);
    }

    IdIndex nodeId = dfg.addNode(std::move(opNode));

    // Map results to output ports.
    for (unsigned i = 0; i < op.getNumResults(); ++i) {
      valueToPort[op.getResult(i)] = dfg.nodes[nodeId]->outputPorts[i];
    }
  }

  // Create one ModuleOutputNode per return operand.
  // Each return operand is a separate DFG output (maps to its own ADG output).
  for (auto &op : body.getOperations()) {
    auto returnOp = mlir::dyn_cast<circt::handshake::ReturnOp>(op);
    if (!returnOp)
      continue;

    for (unsigned i = 0; i < returnOp.getNumOperands(); ++i) {
      auto outputNode = std::make_unique<Node>();
      outputNode->kind = Node::ModuleOutputNode;

      setNodeAttr(outputNode.get(), "op_name",
                  mlir::StringAttr::get(ctx, "module_output"), ctx);
      setNodeAttr(outputNode.get(), "result_index",
                  mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 32), i),
                  ctx);

      auto port = std::make_unique<Port>();
      port->direction = Port::Input;
      port->type = returnOp.getOperand(i).getType();
      IdIndex portId = dfg.addPort(std::move(port));
      dfg.ports[portId]->parentNode = static_cast<IdIndex>(dfg.nodes.size());
      outputNode->inputPorts.push_back(portId);

      dfg.addNode(std::move(outputNode));
    }
  }

  // Create edges from SSA value uses.
  for (auto &op : body.getOperations()) {
    if (mlir::isa<circt::handshake::ReturnOp>(op))
      continue;

    // Find the node for this op. We need to look up by matching.
    // Actually, we can iterate nodes and match by scanning operands.
  }

  // Simpler edge creation: for each node's input ports, find the source.
  for (IdIndex nodeId = 0; nodeId < static_cast<IdIndex>(dfg.nodes.size());
       ++nodeId) {
    auto *node = dfg.getNode(nodeId);
    if (!node)
      continue;

    // For input sentinels, no input edges.
    if (node->kind == Node::ModuleInputNode)
      continue;
  }

  // Use a different approach: iterate all operations and create edges
  // from operands to inputs.
  {
    // Build node index: map node ID -> the operation's operands.
    // We know node ordering: first come input sentinels (one per block arg),
    // then operation nodes (one per non-return op), then output sentinel.

    IdIndex numArgs = static_cast<IdIndex>(body.getNumArguments());
    IdIndex opNodeIdx = numArgs; // Operation nodes start here.

    for (auto &op : body.getOperations()) {
      if (mlir::isa<circt::handshake::ReturnOp>(op)) {
        // Handle return op: operand i -> output sentinel node i.
        // Output nodes are at the end of the node list, one per operand.
        IdIndex numReturnOperands = op.getNumOperands();
        IdIndex firstOutNodeId =
            static_cast<IdIndex>(dfg.nodes.size()) - numReturnOperands;

        for (unsigned i = 0; i < op.getNumOperands(); ++i) {
          mlir::Value operand = op.getOperand(i);
          auto srcIt = valueToPort.find(operand);
          if (srcIt == valueToPort.end())
            continue;

          IdIndex outNodeId = firstOutNodeId + i;
          auto *outNode = dfg.getNode(outNodeId);
          if (!outNode || outNode->inputPorts.empty())
            continue;

          IdIndex srcPortId = srcIt->second;
          IdIndex dstPortId = outNode->inputPorts[0]; // Each output node has 1 input

          auto edge = std::make_unique<Edge>();
          edge->srcPort = srcPortId;
          edge->dstPort = dstPortId;
          IdIndex edgeId = dfg.addEdge(std::move(edge));
          dfg.ports[srcPortId]->connectedEdges.push_back(edgeId);
          dfg.ports[dstPortId]->connectedEdges.push_back(edgeId);
        }
        continue;
      }

      auto *node = dfg.getNode(opNodeIdx);
      if (!node) {
        ++opNodeIdx;
        continue;
      }

      for (unsigned i = 0; i < op.getNumOperands(); ++i) {
        mlir::Value operand = op.getOperand(i);
        auto srcIt = valueToPort.find(operand);
        if (srcIt == valueToPort.end())
          continue;
        if (i >= node->inputPorts.size())
          break;

        IdIndex srcPortId = srcIt->second;
        IdIndex dstPortId = node->inputPorts[i];

        auto edge = std::make_unique<Edge>();
        edge->srcPort = srcPortId;
        edge->dstPort = dstPortId;
        IdIndex edgeId = dfg.addEdge(std::move(edge));
        dfg.ports[srcPortId]->connectedEdges.push_back(edgeId);
        dfg.ports[dstPortId]->connectedEdges.push_back(edgeId);
      }

      ++opNodeIdx;
    }
  }

  llvm::outs() << "DFGBuilder: " << dfg.countNodes() << " nodes, "
               << dfg.countPorts() << " ports, " << dfg.countEdges()
               << " edges\n";

  dfg.buildAttributeCache();

  return true;
}

} // namespace fcc
