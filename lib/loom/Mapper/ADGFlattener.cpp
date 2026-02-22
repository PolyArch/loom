//===-- ADGFlattener.cpp - ADG extraction from Fabric MLIR ---------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include "loom/Mapper/ADGFlattener.h"

#include "loom/Dialect/Fabric/FabricOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
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

/// Resolve a fabric.instance to its target definition.
/// Returns the target operation, or nullptr if not an instance or not found.
mlir::Operation *resolveInstance(mlir::Operation &op) {
  if (op.getName().getStringRef() != "fabric.instance")
    return nullptr;

  auto moduleRef = op.getAttrOfType<mlir::FlatSymbolRefAttr>("module");
  if (!moduleRef)
    return nullptr;

  // Walk up to the nearest symbol table (the builtin module).
  mlir::Operation *symbolTableOp = op.getParentOp();
  while (symbolTableOp && !symbolTableOp->hasTrait<mlir::OpTrait::SymbolTable>())
    symbolTableOp = symbolTableOp->getParentOp();
  if (!symbolTableOp)
    return nullptr;

  return mlir::SymbolTable::lookupSymbolIn(symbolTableOp, moduleRef);
}

/// Get the effective op name for a node: resolves instances to their target.
llvm::StringRef getEffectiveOpName(mlir::Operation &op,
                                   mlir::Operation *resolved) {
  if (resolved)
    return resolved->getName().getStringRef();
  return op.getName().getStringRef();
}

/// Copy hardware-relevant attributes from a resolved definition to a node.
void copyHwAttributes(Node *node, mlir::Operation *resolved,
                      mlir::Builder &builder) {
  if (!resolved)
    return;

  // Copy key hardware parameters by name.
  static const char *attrsToCopy[] = {
      "num_instruction", "num_register", "reg_fifo_depth",
      "num_route_table", "connectivity_table", "ldCount",
      "stCount", "lsqDepth", "numRegion", "addrOffsetTable",
      "memref_type", "is_private", "depth", "bypassable",
      "tag", "table_size", "table", "constant_value",
      "cont_cond_sel", "lqDepth", "sqDepth",
      "enable_share_operand_buffer", "operand_buffer_size",
      "latency", "interval", "output_tag"};

  for (const char *name : attrsToCopy) {
    if (auto attr = resolved->getAttr(name)) {
      node->attributes.push_back(builder.getNamedAttr(name, attr));
    }
  }
}

/// Create an ADG node from an operation in the fabric module body.
/// Handles instance resolution: if the op is fabric.instance, resolves to
/// the actual definition and uses its type/attributes.
IdIndex createNodeFromOp(Graph &graph, mlir::Operation &op,
                         mlir::Builder &builder,
                         llvm::DenseMap<mlir::Value, IdIndex> &valueToPort) {
  mlir::Operation *resolved = resolveInstance(op);
  llvm::StringRef effectiveOpName = getEffectiveOpName(op, resolved);

  auto node = std::make_unique<Node>();
  node->kind = Node::OperationNode;

  node->attributes.push_back(
      builder.getNamedAttr("op_name", builder.getStringAttr(effectiveOpName)));
  node->attributes.push_back(
      builder.getNamedAttr("resource_class",
                           builder.getStringAttr(classifyOp(effectiveOpName))));

  // Copy sym_name from the instance (not the definition).
  if (auto symName = op.getAttrOfType<mlir::StringAttr>("sym_name")) {
    node->attributes.push_back(builder.getNamedAttr("sym_name", symName));
  }

  // Copy hardware attributes from the resolved definition.
  copyHwAttributes(node.get(), resolved, builder);

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

/// Check if a node has a specific string attribute value.
llvm::StringRef getNodeAttrStr(const Node *node, llvm::StringRef name) {
  for (auto &attr : node->attributes) {
    if (attr.getName() == name) {
      if (auto strAttr = mlir::dyn_cast<mlir::StringAttr>(attr.getValue()))
        return strAttr.getValue();
    }
  }
  return "";
}

/// Build connectivity matrix inToOut entries for a switch/temporal_sw node
/// using its connectivity_table attribute (if present).
/// If no connectivity_table, assumes full crossbar.
void buildSwitchConnectivity(
    const Node *node, ConnectivityMatrix &matrix) {
  // Find connectivity_table attribute.
  mlir::DenseI8ArrayAttr connTable;
  for (auto &attr : node->attributes) {
    if (attr.getName() == "connectivity_table") {
      connTable = mlir::dyn_cast<mlir::DenseI8ArrayAttr>(attr.getValue());
      break;
    }
  }

  unsigned numIn = node->inputPorts.size();
  unsigned numOut = node->outputPorts.size();

  if (!connTable || connTable.empty()) {
    // Full crossbar: every input connects to every output.
    for (IdIndex inPortId : node->inputPorts) {
      for (IdIndex outPortId : node->outputPorts) {
        matrix.inToOut[inPortId].push_back(outPortId);
      }
    }
    return;
  }

  // Parse connectivity_table: output-major, table[o * numIn + i] = 1
  // means output o receives from input i.
  for (unsigned o = 0; o < numOut; ++o) {
    for (unsigned i = 0; i < numIn; ++i) {
      unsigned idx = o * numIn + i;
      if (idx < static_cast<unsigned>(connTable.size()) && connTable[idx]) {
        IdIndex inPortId = node->inputPorts[i];
        IdIndex outPortId = node->outputPorts[o];
        auto &outPorts = matrix.inToOut[inPortId];
        if (std::find(outPorts.begin(), outPorts.end(), outPortId) ==
            outPorts.end()) {
          outPorts.push_back(outPortId);
        }
      }
    }
  }
}

} // namespace

Graph ADGFlattener::flatten(fabric::ModuleOp moduleOp) {
  Graph graph(moduleOp.getContext());
  matrix = ConnectivityMatrix();

  mlir::Builder builder(moduleOp.getContext());
  llvm::DenseMap<mlir::Value, IdIndex> valueToPort;

  auto &block = moduleOp.getBody().front();

  // --- Phase A: Create sentinel nodes for module I/O ---
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

  // --- Phase B: Resolve operations and create ADG nodes ---
  // fabric.instance is resolved by looking up the referenced definition.
  llvm::DenseMap<mlir::Operation *, IdIndex> opToNode;

  // Track which ops are temporal PEs (either direct or via instance).
  llvm::SmallVector<std::pair<mlir::Operation *, mlir::Operation *>, 4>
      temporalPEOps; // (body op, resolved definition)

  for (auto &op : block) {
    if (op.hasTrait<mlir::OpTrait::IsTerminator>())
      continue;

    mlir::Operation *resolved = resolveInstance(op);
    llvm::StringRef effectiveName = getEffectiveOpName(op, resolved);

    IdIndex nodeId = createNodeFromOp(graph, op, builder, valueToPort);
    opToNode[&op] = nodeId;

    // Track temporal_pe ops for Phase C.
    if (effectiveName == "fabric.temporal_pe") {
      temporalPEOps.push_back({&op, resolved});
    }
  }

  // --- Phase C: Handle temporal_pe ---
  // For each temporal_pe, mark virtual node and create FU sub-nodes.
  for (auto &[bodyOp, resolved] : temporalPEOps) {
    auto it = opToNode.find(bodyOp);
    if (it == opToNode.end())
      continue;

    IdIndex virtualNodeId = it->second;
    Node *virtualNode = graph.getNode(virtualNodeId);
    if (!virtualNode)
      continue;

    // Mark as virtual node.
    virtualNode->attributes.push_back(
        builder.getNamedAttr("is_virtual", builder.getUnitAttr()));

    // The temporal_pe body contains FU definitions.
    // For instances, the body is on the resolved definition.
    mlir::Operation *tpeOp = resolved ? resolved : bodyOp;
    if (tpeOp->getNumRegions() == 0)
      continue;

    auto &tpeBody = tpeOp->getRegion(0);
    if (tpeBody.empty())
      continue;

    llvm::SmallVector<IdIndex, 4> fuNodeIds;
    for (auto &innerOp : tpeBody.front()) {
      if (innerOp.hasTrait<mlir::OpTrait::IsTerminator>())
        continue;

      auto fuNode = std::make_unique<Node>();
      fuNode->kind = Node::OperationNode;

      // For FU nodes inside temporal_pe, they may also be instances.
      mlir::Operation *fuResolved = resolveInstance(innerOp);
      llvm::StringRef fuEffName = getEffectiveOpName(innerOp, fuResolved);

      fuNode->attributes.push_back(
          builder.getNamedAttr("op_name", builder.getStringAttr(fuEffName)));
      fuNode->attributes.push_back(
          builder.getNamedAttr("resource_class",
                               builder.getStringAttr("functional")));
      fuNode->attributes.push_back(
          builder.getNamedAttr("parent_temporal_pe",
                               builder.getI32IntegerAttr(virtualNodeId)));

      // Copy hardware attributes from the resolved FU definition.
      copyHwAttributes(fuNode.get(), fuResolved, builder);

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
          builder.getNamedAttr("fu_node", builder.getI32IntegerAttr(fuId)));
    }
  }

  // --- Phase D: Create edges for hardware connections ---
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

      // Build physical edge connectivity.
      matrix.outToIn[srcPortId] = dstPortId;
    }
  }

  // Handle fabric.yield -> module output sentinels.
  mlir::Operation *yieldOp = block.getTerminator();
  if (yieldOp && yieldOp->getNumOperands() > 0) {
    for (unsigned i = 0; i < yieldOp->getNumOperands(); ++i) {
      mlir::Value operand = yieldOp->getOperand(i);
      auto srcIt = valueToPort.find(operand);
      if (srcIt == valueToPort.end())
        continue;

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

  // --- Phase E: Build inToOut for routing nodes ---
  // Uses connectivity_table for switches, full crossbar for others.
  for (auto *node : graph.nodeRange()) {
    if (node->kind != Node::OperationNode)
      continue;

    llvm::StringRef resClass = getNodeAttrStr(node, "resource_class");
    if (resClass != "routing")
      continue;

    llvm::StringRef opName = getNodeAttrStr(node, "op_name");

    if (opName == "fabric.switch" || opName == "fabric.temporal_sw") {
      // Use connectivity_table if available, else full crossbar.
      buildSwitchConnectivity(node, matrix);
    } else {
      // Single-in-single-out routing (add_tag, map_tag, del_tag, fifo).
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
  }

  return graph;
}

} // namespace loom
