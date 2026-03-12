//===-- ADGFlattener.cpp - ADG extraction from Fabric MLIR ---------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include "loom/Mapper/ADGFlattener.h"

#include "loom/Dialect/Fabric/FabricOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Path.h"

namespace loom {

namespace {

/// Extract a human-readable location string from an MLIR Location.
std::string extractLocStr(mlir::Location loc) {
  if (auto fileLoc = mlir::dyn_cast<mlir::FileLineColLoc>(loc)) {
    return (llvm::sys::path::filename(fileLoc.getFilename()) + ":" +
            llvm::Twine(fileLoc.getLine()))
        .str();
  }
  return "";
}

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

/// Extract body_ops and body_edges from a fabric.pe operation's body region
/// and attach them as attributes on the given node. This enables the
/// TechMapper to distinguish between different PE types.
void extractPEBodyOps(mlir::Operation *peOp, Node *node,
                      mlir::Builder &builder) {
  if (!peOp || !node || peOp->getNumRegions() == 0)
    return;

  auto &peBody = peOp->getRegion(0);
  if (peBody.empty())
    return;

  llvm::SmallVector<mlir::Attribute, 4> bodyOps;
  for (auto &innerOp : peBody.front()) {
    if (innerOp.hasTrait<mlir::OpTrait::IsTerminator>())
      continue;
    bodyOps.push_back(
        builder.getStringAttr(innerOp.getName().getStringRef()));
  }
  if (bodyOps.empty())
    return;

  node->attributes.push_back(builder.getNamedAttr(
      "body_ops", builder.getArrayAttr(bodyOps)));

  // Extract body_edges: internal use-def connections between body ops.
  // Each body_edge is a pair (src_op_index, dst_op_index) indicating
  // that operation src produces a value consumed by operation dst.
  llvm::SmallVector<mlir::Attribute, 4> bodyEdges;
  llvm::DenseMap<mlir::Operation *, unsigned> opToIndex;
  unsigned opIdx = 0;
  for (auto &innerOp : peBody.front()) {
    if (innerOp.hasTrait<mlir::OpTrait::IsTerminator>())
      continue;
    opToIndex[&innerOp] = opIdx++;
  }
  for (auto &innerOp : peBody.front()) {
    if (innerOp.hasTrait<mlir::OpTrait::IsTerminator>())
      continue;
    auto dstIt = opToIndex.find(&innerOp);
    if (dstIt == opToIndex.end())
      continue;
    unsigned dstIdx = dstIt->second;
    for (mlir::Value operand : innerOp.getOperands()) {
      if (auto *defOp = operand.getDefiningOp()) {
        auto srcIt = opToIndex.find(defOp);
        if (srcIt != opToIndex.end()) {
          bodyEdges.push_back(
              builder.getI32IntegerAttr(srcIt->second));
          bodyEdges.push_back(
              builder.getI32IntegerAttr(dstIdx));
        }
      }
    }
  }
  if (!bodyEdges.empty()) {
    node->attributes.push_back(builder.getNamedAttr(
        "body_edges", builder.getArrayAttr(bodyEdges)));
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

  // Copy visualization grid coordinates from the instance.
  if (auto vizRow = op.getAttrOfType<mlir::IntegerAttr>("viz_row"))
    node->attributes.push_back(builder.getNamedAttr("viz_row", vizRow));
  if (auto vizCol = op.getAttrOfType<mlir::IntegerAttr>("viz_col"))
    node->attributes.push_back(builder.getNamedAttr("viz_col", vizCol));

  // Store source location for human-readable mapping output.
  std::string locStr = extractLocStr(op.getLoc());
  if (!locStr.empty()) {
    node->attributes.push_back(
        builder.getNamedAttr("loc", builder.getStringAttr(locStr)));
  }

  // Copy hardware attributes from the resolved definition, or from the op
  // itself when it is a direct definition (not wrapped in fabric.instance).
  copyHwAttributes(node.get(), resolved ? resolved : &op, builder);

  // Extract body operations for PE definitions.
  if (effectiveOpName == "fabric.pe") {
    mlir::Operation *peOp = resolved ? resolved : &op;
    extractPEBodyOps(peOp, node.get(), builder);
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

    std::string argLocStr = extractLocStr(arg.getLoc());
    if (!argLocStr.empty()) {
      node->attributes.push_back(
          builder.getNamedAttr("loc", builder.getStringAttr(argLocStr)));
    }

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
    opMap[&op] = nodeId;

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

      // Extract body_ops for FU nodes that are fabric.pe definitions.
      if (fuEffName == "fabric.pe") {
        mlir::Operation *fuPeOp = fuResolved ? fuResolved : &innerOp;
        extractPEBodyOps(fuPeOp, fuNode.get(), builder);
      }

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

      std::string yieldLocStr = extractLocStr(yieldOp->getLoc());
      if (!yieldLocStr.empty()) {
        node->attributes.push_back(
            builder.getNamedAttr("loc", builder.getStringAttr(yieldLocStr)));
      }

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

  // --- Phase F: Detect memory bridge clusters ---
  // For multi-port memory nodes, identify the surrounding add_tag/temporal_sw/
  // del_tag bridge nodes and store boundary port metadata on the memory node.
  // This enables the TechMapper to match DFG per-lane ports against bridge
  // boundary ports rather than the memory's aggregated tagged ports.
  for (IdIndex nodeId = 0; nodeId < static_cast<IdIndex>(graph.nodes.size());
       ++nodeId) {
    Node *node = graph.getNode(nodeId);
    if (!node || node->kind != Node::OperationNode)
      continue;
    if (getNodeAttrStr(node, "resource_class") != "memory")
      continue;

    // Get ldCount and stCount to determine if bridge detection is needed.
    unsigned ldCount = 0, stCount = 0;
    bool isPrivate = false;
    bool isExtMem = (getNodeAttrStr(node, "op_name") == "fabric.extmemory");
    for (auto &attr : node->attributes) {
      if (attr.getName() == "ldCount") {
        if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(attr.getValue()))
          ldCount = intAttr.getInt();
      } else if (attr.getName() == "stCount") {
        if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(attr.getValue()))
          stCount = intAttr.getInt();
      } else if (attr.getName() == "is_private") {
        if (auto boolAttr =
                mlir::dyn_cast<mlir::BoolAttr>(attr.getValue()))
          isPrivate = boolAttr.getValue();
        else
          isPrivate = true; // UnitAttr presence means private.
      }
    }

    // Single-port memory: no bridge cluster needed.
    if (ldCount <= 1 && stCount <= 1)
      continue;

    // Helper: given a port, find the source port of the edge feeding it (for
    // input ports) or the dest port of the edge leaving it (for output ports).
    auto findFeedingPort = [&](IdIndex portId) -> IdIndex {
      const Port *p = graph.getPort(portId);
      if (!p)
        return INVALID_ID;
      for (IdIndex edgeId : p->connectedEdges) {
        const Edge *e = graph.getEdge(edgeId);
        if (e && e->dstPort == portId)
          return e->srcPort;
      }
      return INVALID_ID;
    };

    auto findConsumingPort = [&](IdIndex portId) -> IdIndex {
      const Port *p = graph.getPort(portId);
      if (!p)
        return INVALID_ID;
      for (IdIndex edgeId : p->connectedEdges) {
        const Edge *e = graph.getEdge(edgeId);
        if (e && e->srcPort == portId)
          return e->dstPort;
      }
      return INVALID_ID;
    };

    // Helper: given a port, find the parent node's op_name.
    auto getPortOwnerOp = [&](IdIndex portId) -> llvm::StringRef {
      const Port *p = graph.getPort(portId);
      if (!p)
        return "";
      const Node *n = graph.getNode(p->parentNode);
      if (!n)
        return "";
      return getNodeAttrStr(n, "op_name");
    };

    // Trace input bridge for one memory input port category.
    // Returns the boundary input port IDs (add_tag input ports) in lane order.
    auto traceInputBridge =
        [&](unsigned memInputPortIdx, unsigned laneCount,
            llvm::SmallVectorImpl<IdIndex> &addTagNodes,
            IdIndex &muxNodeId) -> llvm::SmallVector<IdIndex, 4> {
      muxNodeId = INVALID_ID;
      llvm::SmallVector<IdIndex, 4> boundary;
      if (memInputPortIdx >= node->inputPorts.size())
        return boundary;

      IdIndex memInPortId = node->inputPorts[memInputPortIdx];
      IdIndex srcPortId = findFeedingPort(memInPortId);
      if (srcPortId == INVALID_ID)
        return boundary;

      llvm::StringRef srcOp = getPortOwnerOp(srcPortId);
      if (srcOp == "fabric.add_tag") {
        // Single-lane: add_tag directly feeds memory.
        const Port *srcPort = graph.getPort(srcPortId);
        if (srcPort) {
          const Node *atNode = graph.getNode(srcPort->parentNode);
          if (atNode && !atNode->inputPorts.empty()) {
            boundary.push_back(atNode->inputPorts[0]);
            addTagNodes.push_back(srcPort->parentNode);
          }
        }
      } else if (srcOp == "fabric.temporal_sw") {
        // Multi-lane: temporal_sw feeds memory. Trace each temporal_sw input
        // back to its feeding add_tag node.
        const Port *srcPort = graph.getPort(srcPortId);
        if (srcPort) {
          const Node *tswNode = graph.getNode(srcPort->parentNode);
          if (tswNode) {
            muxNodeId = srcPort->parentNode;
            for (unsigned lane = 0; lane < tswNode->inputPorts.size() &&
                                    lane < laneCount;
                 ++lane) {
              IdIndex tswInPort = tswNode->inputPorts[lane];
              IdIndex atOutPort = findFeedingPort(tswInPort);
              if (atOutPort != INVALID_ID &&
                  getPortOwnerOp(atOutPort) == "fabric.add_tag") {
                const Port *aop = graph.getPort(atOutPort);
                if (aop) {
                  const Node *atNode = graph.getNode(aop->parentNode);
                  if (atNode && !atNode->inputPorts.empty()) {
                    boundary.push_back(atNode->inputPorts[0]);
                    addTagNodes.push_back(aop->parentNode);
                  }
                }
              }
            }
          }
        }
      }
      return boundary;
    };

    // Trace output bridge for one memory output port category.
    // Returns the boundary output port IDs (del_tag output ports) in lane
    // order.
    auto traceOutputBridge =
        [&](unsigned memOutputPortIdx, unsigned laneCount,
            IdIndex &demuxNodeId) -> llvm::SmallVector<IdIndex, 4> {
      demuxNodeId = INVALID_ID;
      llvm::SmallVector<IdIndex, 4> boundary;
      if (memOutputPortIdx >= node->outputPorts.size())
        return boundary;

      IdIndex memOutPortId = node->outputPorts[memOutputPortIdx];
      IdIndex dstPortId = findConsumingPort(memOutPortId);
      if (dstPortId == INVALID_ID)
        return boundary;

      llvm::StringRef dstOp = getPortOwnerOp(dstPortId);
      if (dstOp == "fabric.del_tag") {
        // Single-lane: memory directly feeds del_tag.
        const Port *dstPort = graph.getPort(dstPortId);
        if (dstPort) {
          const Node *dtNode = graph.getNode(dstPort->parentNode);
          if (dtNode && !dtNode->outputPorts.empty())
            boundary.push_back(dtNode->outputPorts[0]);
        }
      } else if (dstOp == "fabric.temporal_sw") {
        // Multi-lane: memory feeds temporal_sw demux. Trace each temporal_sw
        // output to its consuming del_tag node.
        const Port *dstPort = graph.getPort(dstPortId);
        if (dstPort) {
          const Node *tswNode = graph.getNode(dstPort->parentNode);
          if (tswNode) {
            demuxNodeId = dstPort->parentNode;
            for (unsigned lane = 0; lane < tswNode->outputPorts.size() &&
                                    lane < laneCount;
                 ++lane) {
              IdIndex tswOutPort = tswNode->outputPorts[lane];
              IdIndex dtInPort = findConsumingPort(tswOutPort);
              if (dtInPort != INVALID_ID &&
                  getPortOwnerOp(dtInPort) == "fabric.del_tag") {
                const Port *dip = graph.getPort(dtInPort);
                if (dip) {
                  const Node *dtNode = graph.getNode(dip->parentNode);
                  if (dtNode && !dtNode->outputPorts.empty())
                    boundary.push_back(dtNode->outputPorts[0]);
                }
              }
            }
          }
        }
      }
      return boundary;
    };

    // Trace input bridges per category.
    // ADG memory input layout: [memref?] [st_data, st_addr] (if stCount>0),
    //                          [ld_addr] (if ldCount>0)
    // DFG memory input layout: [memref?] [st0_data, st0_addr, st1_data,
    //                           st1_addr, ...] (interleaved per store lane),
    //                          [ld0_addr, ld1_addr, ...]
    // We collect in ADG order, then interleave to match DFG ordering.
    llvm::SmallVector<IdIndex, 8> bridgeInputPorts;
    llvm::SmallVector<IdIndex, 8> allAddTagNodes;
    llvm::SmallVector<IdIndex, 4> muxNodes;
    unsigned memInIdx = isExtMem ? 1 : 0; // Skip memref for extmemory.

    llvm::SmallVector<IdIndex, 4> stDataBoundary, stAddrBoundary;
    llvm::SmallVector<IdIndex, 4> stDataATNodes, stAddrATNodes;
    if (stCount > 0) {
      IdIndex muxId;
      stDataBoundary = traceInputBridge(memInIdx++, stCount,
                                         stDataATNodes, muxId);
      if (muxId != INVALID_ID)
        muxNodes.push_back(muxId);
      stAddrBoundary = traceInputBridge(memInIdx++, stCount,
                                         stAddrATNodes, muxId);
      if (muxId != INVALID_ID)
        muxNodes.push_back(muxId);

      // Interleave st_data and st_addr per lane to match DFG ordering:
      // [st0_data, st0_addr, st1_data, st1_addr, ...]
      for (unsigned lane = 0; lane < stCount; ++lane) {
        if (lane < stDataBoundary.size())
          bridgeInputPorts.push_back(stDataBoundary[lane]);
        if (lane < stAddrBoundary.size())
          bridgeInputPorts.push_back(stAddrBoundary[lane]);
      }
      for (unsigned lane = 0; lane < stCount; ++lane) {
        if (lane < stDataATNodes.size())
          allAddTagNodes.push_back(stDataATNodes[lane]);
        if (lane < stAddrATNodes.size())
          allAddTagNodes.push_back(stAddrATNodes[lane]);
      }
    }
    if (ldCount > 0) {
      llvm::SmallVector<IdIndex, 4> atNodes;
      IdIndex muxId;
      auto ldAddrBoundary =
          traceInputBridge(memInIdx++, ldCount, atNodes, muxId);
      bridgeInputPorts.append(ldAddrBoundary.begin(), ldAddrBoundary.end());
      allAddTagNodes.append(atNodes.begin(), atNodes.end());
      if (muxId != INVALID_ID)
        muxNodes.push_back(muxId);
    }

    // Trace output bridges per category.
    // ADG memory output layout: [memref?] [ld_data, ld_done] (if ldCount>0),
    //                            [st_done] (if stCount>0)
    // DFG output layout:         [ld_data] (if ldCount>0),
    //                            [st_done] (if stCount>0),
    //                            [ld_done] (if ldCount>0)
    // We store boundary ports in DFG order: ld_data, st_done, ld_done.
    llvm::SmallVector<IdIndex, 8> ldDataBoundary, ldDoneBoundary,
        stDoneBoundary;
    llvm::SmallVector<IdIndex, 4> demuxNodes;
    unsigned memOutIdx = (!isPrivate && !isExtMem) ? 1 : 0; // Skip memref.

    if (ldCount > 0) {
      IdIndex demuxId;
      ldDataBoundary = traceOutputBridge(memOutIdx++, ldCount, demuxId);
      if (demuxId != INVALID_ID)
        demuxNodes.push_back(demuxId);
      ldDoneBoundary = traceOutputBridge(memOutIdx++, ldCount, demuxId);
      if (demuxId != INVALID_ID)
        demuxNodes.push_back(demuxId);
    }
    if (stCount > 0) {
      IdIndex demuxId;
      stDoneBoundary = traceOutputBridge(memOutIdx++, stCount, demuxId);
      if (demuxId != INVALID_ID)
        demuxNodes.push_back(demuxId);
    }

    // Reorder to DFG output order: ld_data, st_done, ld_done.
    llvm::SmallVector<IdIndex, 8> bridgeOutputPorts;
    bridgeOutputPorts.append(ldDataBoundary.begin(), ldDataBoundary.end());
    bridgeOutputPorts.append(stDoneBoundary.begin(), stDoneBoundary.end());
    bridgeOutputPorts.append(ldDoneBoundary.begin(), ldDoneBoundary.end());

    // Skip if detection failed (incomplete bridge).
    if (bridgeInputPorts.empty() && bridgeOutputPorts.empty())
      continue;

    // Set bridge_lane_index on each add_tag node for ConfigGen.
    // The allAddTagNodes vector is interleaved per store lane:
    // [st0_data_at, st0_addr_at, st1_data_at, st1_addr_at, ...,
    //  ld0_addr_at, ld1_addr_at, ...].
    // Each data/addr pair in a lane shares the same lane index.
    {
      unsigned idx = 0;
      // Store lanes: each lane has 2 add_tag nodes (data + addr).
      for (unsigned lane = 0; lane < stCount && idx < allAddTagNodes.size();
           ++lane) {
        // st_data add_tag for this lane
        if (idx < allAddTagNodes.size()) {
          Node *atNode = graph.getNode(allAddTagNodes[idx++]);
          if (atNode)
            atNode->attributes.push_back(builder.getNamedAttr(
                "bridge_lane_index", builder.getI32IntegerAttr(lane)));
        }
        // st_addr add_tag for this lane
        if (idx < allAddTagNodes.size()) {
          Node *atNode = graph.getNode(allAddTagNodes[idx++]);
          if (atNode)
            atNode->attributes.push_back(builder.getNamedAttr(
                "bridge_lane_index", builder.getI32IntegerAttr(lane)));
        }
      }
      // Load lanes: each has 1 add_tag node (addr).
      for (unsigned lane = 0; lane < ldCount && idx < allAddTagNodes.size();
           ++lane) {
        Node *atNode = graph.getNode(allAddTagNodes[idx++]);
        if (atNode)
          atNode->attributes.push_back(builder.getNamedAttr(
              "bridge_lane_index", builder.getI32IntegerAttr(lane)));
      }
    }

    // Store bridge-boundary metadata as node attributes.
    llvm::SmallVector<int32_t> inPorts(bridgeInputPorts.begin(),
                                       bridgeInputPorts.end());
    llvm::SmallVector<int32_t> outPorts(bridgeOutputPorts.begin(),
                                        bridgeOutputPorts.end());
    node->attributes.push_back(builder.getNamedAttr(
        "bridge_input_ports",
        mlir::DenseI32ArrayAttr::get(builder.getContext(), inPorts)));
    node->attributes.push_back(builder.getNamedAttr(
        "bridge_output_ports",
        mlir::DenseI32ArrayAttr::get(builder.getContext(), outPorts)));

    // Store bridge temporal_sw node IDs for temporal assignment.
    if (!muxNodes.empty()) {
      llvm::SmallVector<int32_t> muxIds(muxNodes.begin(), muxNodes.end());
      node->attributes.push_back(builder.getNamedAttr(
          "bridge_mux_nodes",
          mlir::DenseI32ArrayAttr::get(builder.getContext(), muxIds)));
    }
    if (!demuxNodes.empty()) {
      llvm::SmallVector<int32_t> demuxIds(demuxNodes.begin(),
                                           demuxNodes.end());
      node->attributes.push_back(builder.getNamedAttr(
          "bridge_demux_nodes",
          mlir::DenseI32ArrayAttr::get(builder.getContext(), demuxIds)));
    }
  }

  return graph;
}

} // namespace loom
