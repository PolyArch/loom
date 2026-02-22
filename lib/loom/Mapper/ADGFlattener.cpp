//===-- ADGFlattener.cpp - ADG extraction from Fabric MLIR ---------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include "loom/Mapper/ADGFlattener.h"

#include "loom/Dialect/Fabric/FabricOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"

namespace loom {

namespace {

/// Classify an operation name into a resource category for node attributes.
llvm::StringRef classifyOp(llvm::StringRef opName) {
  if (opName == "fabric.pe" || opName == "fabric.temporal_pe")
    return "functional";
  if (opName == "fabric.switch" || opName == "fabric.temporal_sw" ||
      opName == "fabric.add_tag" || opName == "fabric.map_tag" ||
      opName == "fabric.del_tag" || opName == "fabric.fifo")
    return "routing";
  if (opName == "fabric.memory" || opName == "fabric.extmemory")
    return "memory";
  return "unknown";
}

/// Helper to create a node from an operation inside the fabric module.
IdIndex createNodeFromOp(Graph &graph, mlir::Operation &op,
                         mlir::Builder &builder,
                         llvm::DenseMap<mlir::Value, IdIndex> &valueToPort) {
  auto node = std::make_unique<Node>();
  node->kind = Node::OperationNode;

  llvm::StringRef opName = op.getName().getStringRef();
  node->attributes.push_back(
      builder.getNamedAttr("op_name", builder.getStringAttr(opName)));
  node->attributes.push_back(
      builder.getNamedAttr("resource_class",
                           builder.getStringAttr(classifyOp(opName))));

  // Copy sym_name if present.
  if (auto symName = op.getAttrOfType<mlir::StringAttr>("sym_name")) {
    node->attributes.push_back(
        builder.getNamedAttr("sym_name", symName));
  }

  IdIndex nodeId = graph.addNode(std::move(node));

  // Create input ports for operands.
  for (unsigned i = 0; i < op.getNumOperands(); ++i) {
    auto port = std::make_unique<Port>();
    port->parentNode = nodeId;
    port->direction = Port::Input;
    port->type = op.getOperand(i).getType();
    IdIndex portId = graph.addPort(std::move(port));
    graph.getNode(nodeId)->inputPorts.push_back(portId);
  }

  // Create output ports for results.
  for (unsigned i = 0; i < op.getNumResults(); ++i) {
    auto port = std::make_unique<Port>();
    port->parentNode = nodeId;
    port->direction = Port::Output;
    port->type = op.getResult(i).getType();
    IdIndex portId = graph.addPort(std::move(port));
    graph.getNode(nodeId)->outputPorts.push_back(portId);

    valueToPort[op.getResult(i)] = portId;
  }

  return nodeId;
}

} // namespace

Graph ADGFlattener::flatten(fabric::ModuleOp moduleOp) {
  Graph graph(moduleOp.getContext());
  matrix = ConnectivityMatrix();

  mlir::Builder builder(moduleOp.getContext());
  llvm::DenseMap<mlir::Value, IdIndex> valueToPort;

  auto &block = moduleOp.getBody().front();

  // --- Phase A: Create sentinel nodes for module I/O ---
  // Module arguments -> ModuleInputNode sentinels (output ports).
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

  // --- Phase B: Resolve operations ---
  // Walk all operations in the module body and create nodes.
  // fabric.instance is resolved by looking up the referenced module definition.
  llvm::DenseMap<mlir::Operation *, IdIndex> opToNode;

  for (auto &op : block) {
    // Skip the terminator (fabric.yield).
    if (op.hasTrait<mlir::OpTrait::IsTerminator>())
      continue;

    IdIndex nodeId = createNodeFromOp(graph, op, builder, valueToPort);
    opToNode[&op] = nodeId;
  }

  // --- Phase C: Handle temporal_pe ---
  // temporal_pe creates a virtual node plus FU sub-nodes that share ports.
  // This is handled implicitly through the node creation above.
  // The virtual node and FU nodes are distinguished by attributes.
  // For temporal_pe ops, we add metadata attributes.
  for (auto &op : block) {
    if (op.getName().getStringRef() != "fabric.temporal_pe")
      continue;

    auto it = opToNode.find(&op);
    if (it == opToNode.end())
      continue;

    IdIndex virtualNodeId = it->second;
    Node *virtualNode = graph.getNode(virtualNodeId);
    if (!virtualNode)
      continue;

    // Mark as virtual node.
    virtualNode->attributes.push_back(
        builder.getNamedAttr("is_virtual", builder.getUnitAttr()));

    // Extract temporal PE parameters.
    if (auto numInst = op.getAttrOfType<mlir::IntegerAttr>("num_instruction")) {
      virtualNode->attributes.push_back(
          builder.getNamedAttr("num_instruction", numInst));
    }
    if (auto numReg = op.getAttrOfType<mlir::IntegerAttr>("num_register")) {
      virtualNode->attributes.push_back(
          builder.getNamedAttr("num_register", numReg));
    }

    // Create FU sub-nodes from the temporal_pe body.
    if (op.getNumRegions() > 0) {
      auto &tpeBody = op.getRegion(0);
      if (!tpeBody.empty()) {
        llvm::SmallVector<IdIndex, 4> fuNodeIds;
        for (auto &innerOp : tpeBody.front()) {
          if (innerOp.hasTrait<mlir::OpTrait::IsTerminator>())
            continue;

          auto fuNode = std::make_unique<Node>();
          fuNode->kind = Node::OperationNode;
          fuNode->attributes.push_back(
              builder.getNamedAttr("op_name",
                  builder.getStringAttr(innerOp.getName().getStringRef())));
          fuNode->attributes.push_back(
              builder.getNamedAttr("resource_class",
                  builder.getStringAttr("functional")));
          fuNode->attributes.push_back(
              builder.getNamedAttr("parent_temporal_pe",
                  builder.getI32IntegerAttr(virtualNodeId)));

          IdIndex fuId = graph.addNode(std::move(fuNode));
          fuNodeIds.push_back(fuId);

          // FU nodes share the virtual node's ports (reference, not own).
          Node *fuNodePtr = graph.getNode(fuId);
          fuNodePtr->inputPorts = virtualNode->inputPorts;
          fuNodePtr->outputPorts = virtualNode->outputPorts;
        }

        // Store FU node IDs on the virtual node.
        for (IdIndex fuId : fuNodeIds) {
          virtualNode->attributes.push_back(
              builder.getNamedAttr(
                  "fu_node",
                  builder.getI32IntegerAttr(fuId)));
        }
      }
    }
  }

  // --- Phase D: Create edges for hardware connections ---
  // Wire edges based on SSA value uses (operand -> defining result).
  for (auto &op : block) {
    if (op.hasTrait<mlir::OpTrait::IsTerminator>())
      continue;

    auto it = opToNode.find(&op);
    if (it == opToNode.end())
      continue;

    Node *dstNode = graph.getNode(it->second);
    if (!dstNode)
      continue;

    for (unsigned i = 0; i < op.getNumOperands(); ++i) {
      mlir::Value operand = op.getOperand(i);
      auto srcIt = valueToPort.find(operand);
      if (srcIt == valueToPort.end())
        continue;

      IdIndex srcPortId = srcIt->second;
      if (i >= dstNode->inputPorts.size())
        continue;
      IdIndex dstPortId = dstNode->inputPorts[i];

      auto edge = std::make_unique<Edge>();
      edge->srcPort = srcPortId;
      edge->dstPort = dstPortId;
      IdIndex edgeId = graph.addEdge(std::move(edge));

      graph.getPort(srcPortId)->connectedEdges.push_back(edgeId);
      graph.getPort(dstPortId)->connectedEdges.push_back(edgeId);

      // Build connectivity matrix entries.
      matrix.outToIn[srcPortId] = dstPortId;

      // For routing nodes, track internal connectivity.
      Port *srcPort = graph.getPort(srcPortId);
      Port *dstPort = graph.getPort(dstPortId);
      if (srcPort && dstPort && srcPort->parentNode == dstPort->parentNode) {
        // Internal routing within same node.
        matrix.inToOut[dstPortId].push_back(srcPortId);
      }
    }
  }

  // Also handle fabric.yield -> module output sentinels.
  mlir::Operation *yieldOp = block.getTerminator();
  if (yieldOp && yieldOp->getNumOperands() > 0) {
    for (unsigned i = 0; i < yieldOp->getNumOperands(); ++i) {
      mlir::Value operand = yieldOp->getOperand(i);
      auto srcIt = valueToPort.find(operand);
      if (srcIt == valueToPort.end())
        continue;

      // Create ModuleOutputNode sentinel for each yield operand.
      auto node = std::make_unique<Node>();
      node->kind = Node::ModuleOutputNode;
      node->attributes.push_back(
          builder.getNamedAttr("ret_index", builder.getI32IntegerAttr(i)));

      IdIndex nodeId = graph.addNode(std::move(node));

      auto port = std::make_unique<Port>();
      port->parentNode = nodeId;
      port->direction = Port::Input;
      port->type = operand.getType();
      IdIndex portId = graph.addPort(std::move(port));
      graph.getNode(nodeId)->inputPorts.push_back(portId);

      // Create edge.
      IdIndex srcPortId = srcIt->second;
      auto edge = std::make_unique<Edge>();
      edge->srcPort = srcPortId;
      edge->dstPort = portId;
      IdIndex edgeId = graph.addEdge(std::move(edge));

      graph.getPort(srcPortId)->connectedEdges.push_back(edgeId);
      graph.getPort(portId)->connectedEdges.push_back(edgeId);

      matrix.outToIn[srcPortId] = portId;
    }
  }

  // Build inToOut for routing nodes (input port -> reachable output ports).
  // Walk through all routing-class nodes.
  for (auto *node : graph.nodeRange()) {
    if (node->kind != Node::OperationNode)
      continue;

    bool isRouting = false;
    for (auto &attr : node->attributes) {
      if (attr.getName() == "resource_class") {
        if (auto strAttr = mlir::dyn_cast<mlir::StringAttr>(attr.getValue())) {
          if (strAttr.getValue() == "routing")
            isRouting = true;
        }
      }
    }
    if (!isRouting)
      continue;

    // For routing nodes, each input port can reach all output ports.
    for (IdIndex inPortId : node->inputPorts) {
      for (IdIndex outPortId : node->outputPorts) {
        auto &outPorts = matrix.inToOut[inPortId];
        if (std::find(outPorts.begin(), outPorts.end(), outPortId) ==
            outPorts.end()) {
          outPorts.push_back(outPortId);
        }
      }
    }
  }

  return graph;
}

} // namespace loom
