#include "fcc/Mapper/ADGFlattener.h"

#include "fcc/Dialect/Fabric/FabricDialect.h"
#include "fcc/Dialect/Fabric/FabricOps.h"
#include "fcc/Dialect/Fabric/FabricTypes.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/StringMap.h"
#include "llvm/Support/raw_ostream.h"

#include <regex>

namespace fcc {

namespace {

/// Parse grid coordinates from a name like "pe_2_3" or "sw_0_1".
std::pair<int, int> parseGridPos(llvm::StringRef name) {
  // Find last two _N patterns.
  auto str = name.str();
  std::regex re("_(\\d+)_(\\d+)$");
  std::smatch m;
  if (std::regex_search(str, m, re)) {
    return {std::stoi(m[1].str()), std::stoi(m[2].str())};
  }
  return {-1, -1};
}

/// Add a named attribute to a node.
void setNodeAttr(Node *node, llvm::StringRef key, mlir::Attribute val,
                 mlir::MLIRContext *ctx) {
  node->attributes.push_back(
      mlir::NamedAttribute(mlir::StringAttr::get(ctx, key), val));
}

/// Extract operation names from an FU body (skipping fabric.yield).
/// Returns an ArrayAttr containing the op names as StringAttrs.
mlir::ArrayAttr extractOpsFromFUBody(fcc::fabric::FunctionUnitOp fuOp,
                                      mlir::MLIRContext *ctx) {
  llvm::SmallVector<mlir::Attribute, 4> opNames;
  auto &fuBody = fuOp.getBody().front();
  for (auto &bodyOp : fuBody.getOperations()) {
    // Skip the yield terminator.
    if (mlir::isa<fcc::fabric::YieldOp>(bodyOp))
      continue;
    std::string opName = bodyOp.getName().getStringRef().str();
    opNames.push_back(mlir::StringAttr::get(ctx, opName));
  }
  return mlir::ArrayAttr::get(ctx, opNames);
}

/// Extract op names and internal DAG edges from an FU body.
/// Returns ops as ArrayAttr and edges as ArrayAttr of [srcIdx, dstIdx] pairs.
std::pair<mlir::ArrayAttr, mlir::ArrayAttr>
extractFUBodyDAG(fcc::fabric::FunctionUnitOp fuOp, mlir::MLIRContext *ctx) {
  llvm::SmallVector<mlir::Attribute, 4> opNames;
  llvm::SmallVector<mlir::Attribute, 4> dagEdges;

  auto &fuBody = fuOp.getBody().front();

  // Map from Value to op index for edge tracking
  llvm::DenseMap<mlir::Value, int> valueToOpIdx;
  // Map block args to index -1 (they are FU inputs, not ops)
  for (auto arg : fuBody.getArguments())
    valueToOpIdx[arg] = -1;

  int opIdx = 0;
  for (auto &bodyOp : fuBody.getOperations()) {
    if (mlir::isa<fcc::fabric::YieldOp>(bodyOp))
      continue;

    std::string opName = bodyOp.getName().getStringRef().str();
    opNames.push_back(mlir::StringAttr::get(ctx, opName));

    // Track which ops this op depends on
    for (auto operand : bodyOp.getOperands()) {
      auto it = valueToOpIdx.find(operand);
      if (it != valueToOpIdx.end() && it->second >= 0) {
        // This op uses the result of op at index it->second
        auto edge = mlir::ArrayAttr::get(
            ctx,
            {mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 32),
                                     it->second),
             mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 32), opIdx)});
        dagEdges.push_back(edge);
      }
    }

    // Map this op's results to its index
    for (auto result : bodyOp.getResults())
      valueToOpIdx[result] = opIdx;

    opIdx++;
  }

  return {mlir::ArrayAttr::get(ctx, opNames),
          mlir::ArrayAttr::get(ctx, dagEdges)};
}

} // namespace

bool ADGFlattener::flatten(mlir::ModuleOp topModule, mlir::MLIRContext *ctx) {
  adg = Graph(ctx);

  // Find the fabric.module inside the top-level MLIR module.
  fcc::fabric::ModuleOp fabricMod;
  topModule->walk([&](fcc::fabric::ModuleOp mod) { fabricMod = mod; });
  if (!fabricMod) {
    llvm::errs() << "ADGFlattener: no fabric.module found\n";
    return false;
  }

  // Map from SSA Value (result of each op) to ADG port IDs, to wire edges.
  llvm::DenseMap<mlir::Value, IdIndex> valueToOutputPort;

  // Process all operations inside the fabric module body.
  auto &body = fabricMod.getBody().front();

  // Pass 0: Create sentinel nodes for module-level I/O boundary ports.
  // Block arguments that are NOT memrefs become ModuleInputNode sentinels.
  // Module results (yield operands) become ModuleOutputNode sentinels.

  auto moduleFnType = fabricMod.getFunctionType();

  for (auto arg : body.getArguments()) {
    mlir::Type argType = arg.getType();

    // Skip memref arguments (those are bound to extmemory instances directly).
    if (mlir::isa<mlir::MemRefType>(argType))
      continue;

    auto inputNode = std::make_unique<Node>();
    inputNode->kind = Node::ModuleInputNode;

    setNodeAttr(inputNode.get(), "op_name",
                mlir::StringAttr::get(ctx, "module_input"), ctx);
    setNodeAttr(inputNode.get(), "arg_index",
                mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 32),
                                       arg.getArgNumber()),
                ctx);
    setNodeAttr(inputNode.get(), "resource_class",
                mlir::StringAttr::get(ctx, "boundary"), ctx);

    // Create an output port for this boundary input.
    auto port = std::make_unique<Port>();
    port->direction = Port::Output;
    port->type = argType;
    IdIndex portId = adg.addPort(std::move(port));
    adg.ports[portId]->parentNode = static_cast<IdIndex>(adg.nodes.size());
    inputNode->outputPorts.push_back(portId);

    IdIndex nodeId = adg.addNode(std::move(inputNode));
    valueToOutputPort[arg] = adg.nodes[nodeId]->outputPorts[0];
  }

  // Module output sentinels will be created after all ops are processed
  // (in the yield-handling section at the end).

  // Pre-pass: Collect PE and SW definitions by symbol name for instance
  // resolution. Scan BOTH the top-level module AND fabric.module body,
  // since definitions may be at either level.
  llvm::StringMap<fcc::fabric::SpatialPEOp> peDefMap;
  llvm::StringMap<fcc::fabric::SpatialSwOp> swDefMap;
  // Scan top-level module (definitions outside fabric.module)
  topModule->walk([&](fcc::fabric::SpatialPEOp peOp) {
    if (auto symName = peOp.getSymName())
      peDefMap[*symName] = peOp;
  });
  topModule->walk([&](fcc::fabric::SpatialSwOp swOp) {
    if (auto symName = swOp.getSymName())
      swDefMap[*symName] = swOp;
  });

  // Helper lambda: create FU nodes from a PE definition for a given instance.
  auto createFUNodesFromPE = [&](fcc::fabric::SpatialPEOp peOp,
                                 llvm::StringRef instanceName) {
    auto gridPos = parseGridPos(instanceName);

    PEContainment pe;
    pe.peName = instanceName.str();
    pe.row = gridPos.first;
    pe.col = gridPos.second;

    // Set PE-level port counts from the PE function type.
    auto peFnType = peOp.getFunctionType();
    pe.numInputPorts = peFnType.getNumInputs();
    pe.numOutputPorts = peFnType.getNumResults();

    auto &peBody = peOp.getBody().front();
    for (auto &innerOp : peBody.getOperations()) {
      auto fuOp = mlir::dyn_cast<fcc::fabric::FunctionUnitOp>(innerOp);
      if (!fuOp)
        continue;

      auto fuNode = std::make_unique<Node>();
      fuNode->kind = Node::OperationNode;

      std::string fuName = fuOp.getSymName().str();
      setNodeAttr(fuNode.get(), "op_name",
                  mlir::StringAttr::get(ctx, fuName), ctx);
      setNodeAttr(fuNode.get(), "resource_class",
                  mlir::StringAttr::get(ctx, "functional"), ctx);
      setNodeAttr(fuNode.get(), "pe_name",
                  mlir::StringAttr::get(ctx, instanceName), ctx);

      if (auto opsAttr = fuOp->getAttr("ops")) {
        setNodeAttr(fuNode.get(), "ops", opsAttr, ctx);
      } else {
        // Extract op names and internal DAG edges from the FU body.
        auto [bodyOps, dagEdges] = extractFUBodyDAG(fuOp, ctx);
        if (!bodyOps.empty())
          setNodeAttr(fuNode.get(), "ops", bodyOps, ctx);
        if (!dagEdges.empty())
          setNodeAttr(fuNode.get(), "internal_edges", dagEdges, ctx);
      }

      if (fuOp.getLatency()) {
        setNodeAttr(fuNode.get(), "latency",
                    mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 64),
                                           *fuOp.getLatency()),
                    ctx);
      }

      auto fnType = fuOp.getFunctionType();
      for (unsigned i = 0; i < fnType.getNumInputs(); ++i) {
        auto port = std::make_unique<Port>();
        port->direction = Port::Input;
        port->type = fnType.getInput(i);
        IdIndex portId = adg.addPort(std::move(port));
        adg.ports[portId]->parentNode =
            static_cast<IdIndex>(adg.nodes.size());
        fuNode->inputPorts.push_back(portId);
      }

      for (unsigned i = 0; i < fnType.getNumResults(); ++i) {
        auto port = std::make_unique<Port>();
        port->direction = Port::Output;
        port->type = fnType.getResult(i);
        IdIndex portId = adg.addPort(std::move(port));
        adg.ports[portId]->parentNode =
            static_cast<IdIndex>(adg.nodes.size());
        fuNode->outputPorts.push_back(portId);
      }

      IdIndex fuNodeId = adg.addNode(std::move(fuNode));
      pe.fuNodeIds.push_back(fuNodeId);
      nodeGridPos[fuNodeId] = gridPos;

      for (IdIndex ip : adg.nodes[fuNodeId]->inputPorts) {
        for (IdIndex op : adg.nodes[fuNodeId]->outputPorts) {
          connectivity.inToOut[ip].push_back(op);
        }
      }
    }

    peContainment.push_back(pe);
  };

  // Pass 1: Create nodes for each hardware resource.
  // For spatial_pe: flatten by creating one FU node per function_unit inside.
  // For switches/fifos/extmemory: create one node each.
  // For fabric.instance: resolve to PE or SW definition and create nodes.

  for (auto &op : body.getOperations()) {
    // Handle fabric.instance ops (resolve to PE or SW definitions).
    if (auto instOp = mlir::dyn_cast<fcc::fabric::InstanceOp>(op)) {
      llvm::StringRef moduleName = instOp.getModule();
      std::string instanceName;
      if (auto symName = instOp.getSymName())
        instanceName = symName->str();

      // Check if this instance references a PE definition.
      auto peIt = peDefMap.find(moduleName);
      if (peIt != peDefMap.end()) {
        createFUNodesFromPE(peIt->second, instanceName);
        continue;
      }

      // Check if this instance references a SW definition.
      auto swIt = swDefMap.find(moduleName);
      if (swIt != swDefMap.end()) {
        auto swDef = swIt->second;
        auto swNode = std::make_unique<Node>();
        swNode->kind = Node::OperationNode;

        setNodeAttr(swNode.get(), "op_name",
                    mlir::StringAttr::get(ctx, instanceName), ctx);
        setNodeAttr(swNode.get(), "resource_class",
                    mlir::StringAttr::get(ctx, "routing"), ctx);

        auto gridPos = parseGridPos(instanceName);
        auto swFnType = swDef.getFunctionType();

        for (unsigned i = 0; i < swFnType.getNumInputs(); ++i) {
          auto port = std::make_unique<Port>();
          port->direction = Port::Input;
          port->type = swFnType.getInput(i);
          IdIndex portId = adg.addPort(std::move(port));
          adg.ports[portId]->parentNode =
              static_cast<IdIndex>(adg.nodes.size());
          swNode->inputPorts.push_back(portId);
        }

        for (unsigned i = 0; i < swFnType.getNumResults(); ++i) {
          auto port = std::make_unique<Port>();
          port->direction = Port::Output;
          port->type = swFnType.getResult(i);
          IdIndex portId = adg.addPort(std::move(port));
          adg.ports[portId]->parentNode =
              static_cast<IdIndex>(adg.nodes.size());
          swNode->outputPorts.push_back(portId);
        }

        IdIndex swNodeId = adg.addNode(std::move(swNode));
        nodeGridPos[swNodeId] = gridPos;

        auto *node = adg.getNode(swNodeId);
        for (IdIndex ip : node->inputPorts) {
          for (IdIndex op : node->outputPorts) {
            connectivity.inToOut[ip].push_back(op);
          }
        }

        for (unsigned i = 0; i < instOp.getNumResults(); ++i) {
          valueToOutputPort[instOp.getResult(i)] = node->outputPorts[i];
        }
        continue;
      }

      // Unknown instance type -- skip.
      continue;
    }

    if (auto peOp = mlir::dyn_cast<fcc::fabric::SpatialPEOp>(op)) {
      // If this PE has a sym_name and is referenced by InstanceOps,
      // skip it here (instances are handled above).
      if (auto symName = peOp.getSymName()) {
        if (peDefMap.count(*symName))
          continue;
      }

      std::string peName;
      if (auto symName = peOp.getSymName())
        peName = symName->str();

      auto gridPos = parseGridPos(peName);

      PEContainment pe;
      pe.peName = peName;
      pe.row = gridPos.first;
      pe.col = gridPos.second;

      // Set PE-level port counts from the PE function type.
      auto peFnType = peOp.getFunctionType();
      pe.numInputPorts = peFnType.getNumInputs();
      pe.numOutputPorts = peFnType.getNumResults();

      // Create one FU node per function_unit inside the PE.
      auto &peBody = peOp.getBody().front();
      for (auto &innerOp : peBody.getOperations()) {
        auto fuOp = mlir::dyn_cast<fcc::fabric::FunctionUnitOp>(innerOp);
        if (!fuOp)
          continue;

        auto fuNode = std::make_unique<Node>();
        fuNode->kind = Node::OperationNode;

        // Store metadata as attributes.
        std::string fuName = fuOp.getSymName().str();

        setNodeAttr(fuNode.get(), "op_name",
                    mlir::StringAttr::get(ctx, fuName), ctx);
        setNodeAttr(fuNode.get(), "resource_class",
                    mlir::StringAttr::get(ctx, "functional"), ctx);
        setNodeAttr(fuNode.get(), "pe_name",
                    mlir::StringAttr::get(ctx, peName), ctx);

        // Store the ops list from the FU.
        if (auto opsAttr = fuOp->getAttr("ops")) {
          setNodeAttr(fuNode.get(), "ops", opsAttr, ctx);
        } else {
          // Extract op names and internal DAG edges from the FU body.
          auto [bodyOps, dagEdges] = extractFUBodyDAG(fuOp, ctx);
          if (!bodyOps.empty())
            setNodeAttr(fuNode.get(), "ops", bodyOps, ctx);
          if (!dagEdges.empty())
            setNodeAttr(fuNode.get(), "internal_edges", dagEdges, ctx);
        }

        // Store latency and interval.
        if (fuOp.getLatency()) {
          setNodeAttr(fuNode.get(), "latency",
                      mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 64),
                                             *fuOp.getLatency()),
                      ctx);
        }

        // Create input ports from FU function type.
        auto fnType = fuOp.getFunctionType();
        for (unsigned i = 0; i < fnType.getNumInputs(); ++i) {
          auto port = std::make_unique<Port>();
          port->direction = Port::Input;
          port->type = fnType.getInput(i);
          IdIndex portId = adg.addPort(std::move(port));
          adg.ports[portId]->parentNode =
              static_cast<IdIndex>(adg.nodes.size());
          fuNode->inputPorts.push_back(portId);
        }

        // Create output ports from FU function type.
        for (unsigned i = 0; i < fnType.getNumResults(); ++i) {
          auto port = std::make_unique<Port>();
          port->direction = Port::Output;
          port->type = fnType.getResult(i);
          IdIndex portId = adg.addPort(std::move(port));
          adg.ports[portId]->parentNode =
              static_cast<IdIndex>(adg.nodes.size());
          fuNode->outputPorts.push_back(portId);
        }

        IdIndex fuNodeId = adg.addNode(std::move(fuNode));
        pe.fuNodeIds.push_back(fuNodeId);
        nodeGridPos[fuNodeId] = gridPos;

        // Register internal connectivity: each input can reach each output.
        for (IdIndex ip : adg.nodes[fuNodeId]->inputPorts) {
          for (IdIndex op : adg.nodes[fuNodeId]->outputPorts) {
            connectivity.inToOut[ip].push_back(op);
          }
        }
      }

      // Map PE SSA results to the first FU's output ports for connectivity.
      // In practice, PE outputs are connected via the switch network, so we
      // track the PE-level SSA values for wiring inter-PE edges.
      for (unsigned i = 0; i < peOp.getNumResults(); ++i) {
        // Store the PE output value -> we will handle wiring in pass 2.
        // For now, just record the value.
      }

      peContainment.push_back(pe);
      continue;
    }

    if (auto swOp = mlir::dyn_cast<fcc::fabric::SpatialSwOp>(op)) {
      // Skip SW definitions that are referenced by InstanceOps.
      if (auto symName = swOp.getSymName()) {
        if (swDefMap.count(*symName))
          continue;
      }

      auto swNode = std::make_unique<Node>();
      swNode->kind = Node::OperationNode;

      std::string swName;
      if (auto symName = swOp.getSymName())
        swName = symName->str();

      setNodeAttr(swNode.get(), "op_name",
                  mlir::StringAttr::get(ctx, swName), ctx);
      setNodeAttr(swNode.get(), "resource_class",
                  mlir::StringAttr::get(ctx, "routing"), ctx);

      auto gridPos = parseGridPos(swName);

      // Create input ports from function type (switch has no SSA operands
      // in graph regions).
      auto swFnType = swOp.getFunctionType();
      for (unsigned i = 0; i < swFnType.getNumInputs(); ++i) {
        auto port = std::make_unique<Port>();
        port->direction = Port::Input;
        port->type = swFnType.getInput(i);
        IdIndex portId = adg.addPort(std::move(port));
        adg.ports[portId]->parentNode =
            static_cast<IdIndex>(adg.nodes.size());
        swNode->inputPorts.push_back(portId);
      }

      // Create output ports.
      for (unsigned i = 0; i < swFnType.getNumResults(); ++i) {
        auto port = std::make_unique<Port>();
        port->direction = Port::Output;
        port->type = swFnType.getResult(i);
        IdIndex portId = adg.addPort(std::move(port));
        adg.ports[portId]->parentNode =
            static_cast<IdIndex>(adg.nodes.size());
        swNode->outputPorts.push_back(portId);
      }

      IdIndex swNodeId = adg.addNode(std::move(swNode));
      nodeGridPos[swNodeId] = gridPos;

      // Register internal connectivity based on connectivity_table.
      auto *node = adg.getNode(swNodeId);
      for (IdIndex ip : node->inputPorts) {
        for (IdIndex op : node->outputPorts) {
          connectivity.inToOut[ip].push_back(op);
        }
      }

      // Map SSA results to output ports.
      for (unsigned i = 0; i < swOp.getNumResults(); ++i) {
        valueToOutputPort[swOp.getResult(i)] = node->outputPorts[i];
      }
      continue;
    }

    if (auto extOp = mlir::dyn_cast<fcc::fabric::ExtMemoryOp>(op)) {
      auto memNode = std::make_unique<Node>();
      memNode->kind = Node::OperationNode;

      std::string memName;
      if (auto symName = extOp.getSymName())
        memName = symName->str();

      setNodeAttr(memNode.get(), "op_name",
                  mlir::StringAttr::get(ctx, memName), ctx);
      setNodeAttr(memNode.get(), "resource_class",
                  mlir::StringAttr::get(ctx, "memory"), ctx);
      setNodeAttr(memNode.get(), "ldCount",
                  mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 64),
                                         extOp.getLdCount()),
                  ctx);
      setNodeAttr(memNode.get(), "stCount",
                  mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 64),
                                         extOp.getStCount()),
                  ctx);

      // Record which module memref argument this extmemory consumes.
      // The memref_arg_index attribute is emitted by ADGBuilder to
      // identify which module block argument (memref) this extmemory binds to.
      if (auto argIdxAttr = extOp->getAttrOfType<mlir::IntegerAttr>(
              "memref_arg_index")) {
        setNodeAttr(memNode.get(), "memref_arg_index", argIdxAttr, ctx);
      }

      // Copy connected_sw attribute so we can wire SW->ExtMem reverse edges.
      if (auto connSwAttr = extOp->getAttrOfType<mlir::ArrayAttr>(
              "connected_sw")) {
        setNodeAttr(memNode.get(), "connected_sw", connSwAttr, ctx);
      }

      // Create input ports from function type.
      auto memFnType = extOp.getFunctionType();
      for (unsigned i = 0; i < memFnType.getNumInputs(); ++i) {
        auto port = std::make_unique<Port>();
        port->direction = Port::Input;
        port->type = memFnType.getInput(i);
        IdIndex portId = adg.addPort(std::move(port));
        adg.ports[portId]->parentNode =
            static_cast<IdIndex>(adg.nodes.size());
        memNode->inputPorts.push_back(portId);
      }

      // Create output ports.
      for (unsigned i = 0; i < memFnType.getNumResults(); ++i) {
        auto port = std::make_unique<Port>();
        port->direction = Port::Output;
        port->type = memFnType.getResult(i);
        IdIndex portId = adg.addPort(std::move(port));
        adg.ports[portId]->parentNode =
            static_cast<IdIndex>(adg.nodes.size());
        memNode->outputPorts.push_back(portId);
      }

      IdIndex memNodeId = adg.addNode(std::move(memNode));

      // Internal connectivity.
      auto *node = adg.getNode(memNodeId);
      for (IdIndex ip : node->inputPorts) {
        for (IdIndex op : node->outputPorts) {
          connectivity.inToOut[ip].push_back(op);
        }
      }

      for (unsigned i = 0; i < extOp.getNumResults(); ++i) {
        valueToOutputPort[extOp.getResult(i)] = node->outputPorts[i];
      }
      continue;
    }

    if (auto fifoOp = mlir::dyn_cast<fcc::fabric::FifoOp>(op)) {
      auto fifoNode = std::make_unique<Node>();
      fifoNode->kind = Node::OperationNode;

      std::string fifoName;
      if (auto symName = fifoOp.getSymName())
        fifoName = symName->str();

      setNodeAttr(fifoNode.get(), "op_name",
                  mlir::StringAttr::get(ctx, fifoName), ctx);
      setNodeAttr(fifoNode.get(), "resource_class",
                  mlir::StringAttr::get(ctx, "routing"), ctx);

      // Create input ports from function type.
      auto fifoFnType = fifoOp.getFunctionType();
      for (unsigned i = 0; i < fifoFnType.getNumInputs(); ++i) {
        auto port = std::make_unique<Port>();
        port->direction = Port::Input;
        port->type = fifoFnType.getInput(i);
        IdIndex portId = adg.addPort(std::move(port));
        adg.ports[portId]->parentNode =
            static_cast<IdIndex>(adg.nodes.size());
        fifoNode->inputPorts.push_back(portId);
      }

      // Create output ports.
      for (unsigned i = 0; i < fifoFnType.getNumResults(); ++i) {
        auto port = std::make_unique<Port>();
        port->direction = Port::Output;
        port->type = fifoFnType.getResult(i);
        IdIndex portId = adg.addPort(std::move(port));
        adg.ports[portId]->parentNode =
            static_cast<IdIndex>(adg.nodes.size());
        fifoNode->outputPorts.push_back(portId);
      }

      IdIndex fifoNodeId = adg.addNode(std::move(fifoNode));

      // Internal connectivity.
      auto *node = adg.getNode(fifoNodeId);
      for (IdIndex ip : node->inputPorts) {
        for (IdIndex op : node->outputPorts) {
          connectivity.inToOut[ip].push_back(op);
        }
      }

      for (unsigned i = 0; i < fifoOp.getNumResults(); ++i) {
        valueToOutputPort[fifoOp.getResult(i)] = node->outputPorts[i];
      }
      continue;
    }
  }

  // Pass 2: Wire edges based on SSA value uses.
  // For each op that consumes an SSA value produced by another op, create
  // an edge from the producer's output port to the consumer's input port.
  // We need to track which node owns which SSA operands.

  // Build a map from mlir::Value to the producing node's output port.
  // The PE outputs are connected to switch inputs. We need to handle the
  // PE -> SW wiring specially since PEs are flattened into FU nodes.

  // For the MVP fabric, PE outputs are directly connected to switch inputs
  // via SSA value flow. Since we flattened PEs into FU nodes, the PE-level
  // output ports don't exist in the flat graph. Instead, we create connectivity
  // between FU output ports and the switch input ports that the PE output
  // feeds into.

  // Re-scan the fabric to build physical connectivity edges.
  // For each SSA value usage: producer output port -> consumer input port.

  // Build a map: SSA Value -> (node_id, output_port_index) in the ADG.
  // For PEs: map PE result values to each FU's output ports (all FUs in the PE
  // can potentially drive PE outputs).

  // We handle the connectivity differently: in this spatial PE model,
  // each PE has 4 input ports and 4 output ports that are the "exterior"
  // ports. These are connected via SSA values to switches. The FUs inside
  // share these PE ports.

  // For the flattened ADG:
  // - Each FU has native-type ports (i32, i1, index, none).
  // - The switch has bits<32> ports.
  // - The mapper will handle type-width matching during routing.
  // - Physical connectivity: PE output -> SW input means every FU output port
  //   in that PE can potentially route to that SW input port.
  //   Similarly, SW output -> PE input means every FU input port is reachable.

  // So we need to create connectivity edges:
  //   For each (PE result i) -> (consumer input j):
  //     For each FU in that PE: FU output port 0..N -> consumer input port j
  //   For each (producer output k) -> (PE operand m):
  //     For each FU in that PE: producer output port k -> FU input port 0..N

  // Collect PE info: which PE has which results, and which operands.
  // This handles both direct SpatialPEOp and InstanceOp referencing PEs.
  struct PEInfo {
    mlir::Operation *op;
    std::vector<IdIndex> fuNodeIds;
  };

  std::vector<PEInfo> peInfos;
  size_t peIdx = 0;
  for (auto &op : body.getOperations()) {
    bool isPE = false;
    if (auto peOp = mlir::dyn_cast<fcc::fabric::SpatialPEOp>(op)) {
      // Skip definitions that are referenced by instances.
      if (auto symName = peOp.getSymName()) {
        if (!peDefMap.count(*symName))
          isPE = true;
      } else {
        isPE = true;
      }
    } else if (auto instOp = mlir::dyn_cast<fcc::fabric::InstanceOp>(op)) {
      isPE = peDefMap.count(instOp.getModule()) > 0;
    }
    if (isPE) {
      PEInfo info;
      info.op = &op;
      if (peIdx < peContainment.size())
        info.fuNodeIds.assign(peContainment[peIdx].fuNodeIds.begin(),
                              peContainment[peIdx].fuNodeIds.end());
      peInfos.push_back(info);
      peIdx++;
    }
  }

  // Now create physical edges.
  // Track which node owns each SSA result for non-PE ops (switches, fifos,
  // extmemory).
  // For PEs, we track the PE -> FU mapping.

  // For each SSA value use:
  // 1. Find the producer: could be a PE result, switch result, fifo result,
  //    extmemory result, or a block argument.
  // 2. Find the consumer: which op uses this value and at which operand idx.
  // 3. Create edges from producer output port(s) to consumer input port(s).

  // Build map: Value -> vector of output port IDs (for PEs, all FU outputs).
  llvm::DenseMap<mlir::Value, llvm::SmallVector<IdIndex, 4>> valueSrcPorts;

  // Non-PE ops: already mapped in valueToOutputPort.
  for (auto &kv : valueToOutputPort) {
    valueSrcPorts[kv.first].push_back(kv.second);
  }

  // PE ops: map each PE result to all FU output ports in that PE.
  for (auto &pi : peInfos) {
    for (unsigned r = 0; r < pi.op->getNumResults(); ++r) {
      mlir::Value val = pi.op->getResult(r);
      for (IdIndex fuId : pi.fuNodeIds) {
        auto *fuNode = adg.getNode(fuId);
        if (!fuNode)
          continue;
        for (IdIndex op : fuNode->outputPorts) {
          valueSrcPorts[val].push_back(op);
        }
      }
    }
  }

  // Build map: (op, operandIdx) -> vector of input port IDs.
  // For non-PE consumer ops: direct input port.
  // For PE consumer ops: all FU input ports in that PE.
  // We need to find which node/port each operand maps to.

  // Rebuild a map from Operation* to its input port list in the ADG.
  llvm::DenseMap<mlir::Operation *, llvm::SmallVector<IdIndex, 8>>
      opToInputPorts;

  // For non-PE ops.
  {
    size_t nonPeNodeIdx = 0;
    for (auto &op : body.getOperations()) {
      if (mlir::isa<fcc::fabric::SpatialPEOp>(op))
        continue;
      if (mlir::isa<fcc::fabric::YieldOp>(op))
        continue;

      // Find the matching node in the ADG.
      // Non-PE nodes were added after all PE FU nodes. We need to match them.
      // Build a different map by name or by op pointer.
    }
  }

  // Simpler approach: iterate ops in order and track which ADG nodes they
  // correspond to. We know the order: all PE FUs first, then switches,
  // extmemory, fifos in order.

  // Collect non-PE, non-SW-instance ops (extmemory, fifo, etc.) that were
  // created in pass 1 as standalone ADG nodes with their own SSA results.
  std::vector<mlir::Operation *> nonPeOps;
  for (auto &op : body.getOperations()) {
    if (mlir::isa<fcc::fabric::SpatialPEOp>(op))
      continue;
    if (mlir::isa<fcc::fabric::YieldOp>(op))
      continue;
    if (mlir::isa<fcc::fabric::SpatialSwOp>(op))
      continue;
    if (auto instOp = mlir::dyn_cast<fcc::fabric::InstanceOp>(op)) {
      if (peDefMap.count(instOp.getModule()) ||
          swDefMap.count(instOp.getModule()))
        continue;
    }
    nonPeOps.push_back(&op);
  }

  // Count total FU nodes.
  IdIndex totalFuNodes = 0;
  for (auto &pe : peContainment)
    totalFuNodes += pe.fuNodeIds.size();

  // The nonPeOps correspond to the last N nodes created in pass 1
  // (after all sentinel, SW instance, and FU nodes).
  IdIndex nonPeStartIdx = static_cast<IdIndex>(adg.nodes.size()) -
                           static_cast<IdIndex>(nonPeOps.size());

  // Wire edges for non-PE/non-SW ops (extmemory, fifo, etc.) based on SSA.
  for (size_t i = 0; i < nonPeOps.size(); ++i) {
    IdIndex nodeId = nonPeStartIdx + static_cast<IdIndex>(i);
    auto *node = adg.getNode(nodeId);
    if (!node)
      continue;

    mlir::Operation *op = nonPeOps[i];

    for (unsigned j = 0; j < op->getNumOperands(); ++j) {
      mlir::Value operand = op->getOperand(j);
      if (j >= node->inputPorts.size())
        break;

      IdIndex dstPortId = node->inputPorts[j];

      auto srcIt = valueSrcPorts.find(operand);
      if (srcIt == valueSrcPorts.end())
        continue;

      for (IdIndex srcPortId : srcIt->second) {
        auto edge = std::make_unique<Edge>();
        edge->srcPort = srcPortId;
        edge->dstPort = dstPortId;
        IdIndex edgeId = adg.addEdge(std::move(edge));
        adg.ports[srcPortId]->connectedEdges.push_back(edgeId);
        adg.ports[dstPortId]->connectedEdges.push_back(edgeId);
        connectivity.outToIn[srcPortId].push_back(dstPortId);
      }
    }
  }

  // Helper: create edges from a set of source ports to a set of dest ports.
  auto createEdgesBetweenPorts =
      [&](llvm::ArrayRef<IdIndex> srcPorts, llvm::ArrayRef<IdIndex> dstPorts) {
        for (IdIndex srcPortId : srcPorts) {
          for (IdIndex dstPortId : dstPorts) {
            auto edge = std::make_unique<Edge>();
            edge->srcPort = srcPortId;
            edge->dstPort = dstPortId;
            IdIndex edgeId = adg.addEdge(std::move(edge));
            adg.ports[srcPortId]->connectedEdges.push_back(edgeId);
            adg.ports[dstPortId]->connectedEdges.push_back(edgeId);
            connectivity.outToIn[srcPortId].push_back(dstPortId);
          }
        }
      };

  // Wire SW instance operands based on SSA def-use chains.
  // Each SW instance op has operands that are SSA values from other ops
  // (neighboring switches, PEs, etc.). We create edges from the producer's
  // output port(s) to the corresponding SW input port.
  for (auto &op : body.getOperations()) {
    auto instOp = mlir::dyn_cast<fcc::fabric::InstanceOp>(op);
    if (!instOp)
      continue;
    if (!swDefMap.count(instOp.getModule()))
      continue;

    // Find the ADG node for this SW instance by matching the output port.
    // The first result's output port was registered in valueToOutputPort.
    if (instOp.getNumResults() == 0)
      continue;
    auto outIt = valueToOutputPort.find(instOp.getResult(0));
    if (outIt == valueToOutputPort.end())
      continue;
    IdIndex firstOutPortId = outIt->second;
    const Port *firstOutPort = adg.getPort(firstOutPortId);
    if (!firstOutPort)
      continue;
    IdIndex swNodeId = firstOutPort->parentNode;
    auto *swNode = adg.getNode(swNodeId);
    if (!swNode)
      continue;

    for (unsigned j = 0; j < instOp.getNumOperands(); ++j) {
      mlir::Value operand = instOp.getOperand(j);
      if (j >= swNode->inputPorts.size())
        break;

      IdIndex dstPortId = swNode->inputPorts[j];

      auto srcIt = valueSrcPorts.find(operand);
      if (srcIt == valueSrcPorts.end())
        continue;

      for (IdIndex srcPortId : srcIt->second) {
        auto edge = std::make_unique<Edge>();
        edge->srcPort = srcPortId;
        edge->dstPort = dstPortId;
        IdIndex edgeId = adg.addEdge(std::move(edge));
        adg.ports[srcPortId]->connectedEdges.push_back(edgeId);
        adg.ports[dstPortId]->connectedEdges.push_back(edgeId);
        connectivity.outToIn[srcPortId].push_back(dstPortId);
      }
    }
  }

  // Wire PE instance operands based on SSA def-use chains.
  // Each PE instance has operands from its local switch. We create edges
  // from the producer's output port(s) to ALL FU input ports in that PE
  // (since any FU inside the PE can consume data arriving at a PE port).
  for (auto &pi : peInfos) {
    mlir::Operation *peOp = pi.op;

    for (unsigned j = 0; j < peOp->getNumOperands(); ++j) {
      mlir::Value operand = peOp->getOperand(j);

      auto srcIt = valueSrcPorts.find(operand);
      if (srcIt == valueSrcPorts.end())
        continue;

      // The destination is all FU input ports in this PE.
      for (IdIndex fuId : pi.fuNodeIds) {
        auto *fuNode = adg.getNode(fuId);
        if (!fuNode)
          continue;
        createEdgesBetweenPorts(srcIt->second, fuNode->inputPorts);
      }
    }
  }

  // Wire reverse connectivity: SW output ports -> ExtMem input ports.
  // The ADGBuilder creates connections from ExtMem outputs to SW inputs via
  // SSA operands, but the reverse direction (SW outputs feeding ExtMem data
  // inputs) is only recorded as metadata (connected_sw attribute).
  // We need to explicitly add these edges so the routing BFS can find paths
  // from compute nodes back to external memory.
  {
    // Build a map from instance name -> SW ADG node ID.
    llvm::DenseMap<llvm::StringRef, IdIndex> swNameToNodeId;
    for (IdIndex nid = 0; nid < static_cast<IdIndex>(adg.nodes.size()); ++nid) {
      const Node *n = adg.getNode(nid);
      if (!n)
        continue;
      if (getNodeAttrStr(n, "resource_class") == "routing") {
        swNameToNodeId[getNodeAttrStr(n, "op_name")] = nid;
      }
    }

    // For each ExtMem node, find connected SW nodes and create edges.
    for (IdIndex nid = 0; nid < static_cast<IdIndex>(adg.nodes.size()); ++nid) {
      const Node *n = adg.getNode(nid);
      if (!n)
        continue;
      if (getNodeAttrStr(n, "resource_class") != "memory")
        continue;

      // Find the connected_sw attribute.
      for (auto &attr : n->attributes) {
        if (attr.getName() != "connected_sw")
          continue;
        auto arrayAttr = mlir::dyn_cast<mlir::ArrayAttr>(attr.getValue());
        if (!arrayAttr)
          continue;
        for (auto elem : arrayAttr) {
          auto strAttr = mlir::dyn_cast<mlir::StringAttr>(elem);
          if (!strAttr)
            continue;
          auto swIt = swNameToNodeId.find(strAttr.getValue());
          if (swIt == swNameToNodeId.end())
            continue;
          IdIndex swNodeId = swIt->second;
          auto *swNode = adg.getNode(swNodeId);
          if (!swNode)
            continue;

          // The ExtMem data input ports start after the memref port (port 0).
          // SW output ports for ExtMem start at port 8 (same base as input).
          // ADGBuilder uses swOutputPortBase = 8, same as swInputPortBase.
          // ExtMem has: input port 0 = memref, ports 1..N = data inputs.
          // SW has: output ports 8..8+N-1 for ExtMem data.

          // Get the ExtMem function type to determine the number of data input
          // ports (excluding the memref port at index 0).
          unsigned numExtMemDataInputs = n->inputPorts.size() > 1
                                             ? n->inputPorts.size() - 1
                                             : 0;
          unsigned swOutputBase = 8; // Matches ADGBuilder convention.

          for (unsigned p = 0; p < numExtMemDataInputs; ++p) {
            unsigned swOutIdx = swOutputBase + p;
            unsigned extMemInIdx = 1 + p; // Skip memref port at 0.
            if (swOutIdx >= swNode->outputPorts.size() ||
                extMemInIdx >= n->inputPorts.size())
              break;

            IdIndex srcPortId = swNode->outputPorts[swOutIdx];
            IdIndex dstPortId = n->inputPorts[extMemInIdx];

            // Add connectivity entry.
            connectivity.outToIn[srcPortId].push_back(dstPortId);

            // Also create an ADG edge for this connection.
            auto edge = std::make_unique<Edge>();
            edge->srcPort = srcPortId;
            edge->dstPort = dstPortId;
            IdIndex edgeId = adg.addEdge(std::move(edge));
            adg.ports[srcPortId]->connectedEdges.push_back(edgeId);
            adg.ports[dstPortId]->connectedEdges.push_back(edgeId);
          }
        }
      }
    }
  }

  // Create module output sentinel nodes from the fabric.yield terminator.
  for (auto &op : body.getOperations()) {
    auto yieldOp = mlir::dyn_cast<fcc::fabric::YieldOp>(op);
    if (!yieldOp)
      continue;

    for (unsigned i = 0; i < yieldOp.getNumOperands(); ++i) {
      auto outputNode = std::make_unique<Node>();
      outputNode->kind = Node::ModuleOutputNode;

      setNodeAttr(outputNode.get(), "op_name",
                  mlir::StringAttr::get(ctx, "module_output"), ctx);
      setNodeAttr(outputNode.get(), "result_index",
                  mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 32), i),
                  ctx);
      setNodeAttr(outputNode.get(), "resource_class",
                  mlir::StringAttr::get(ctx, "boundary"), ctx);

      // Create an input port for this boundary output.
      auto port = std::make_unique<Port>();
      port->direction = Port::Input;
      port->type = yieldOp.getOperand(i).getType();
      IdIndex portId = adg.addPort(std::move(port));
      adg.ports[portId]->parentNode = static_cast<IdIndex>(adg.nodes.size());
      outputNode->inputPorts.push_back(portId);

      IdIndex nodeId = adg.addNode(std::move(outputNode));

      // Wire from the yield operand source.
      mlir::Value yieldOperand = yieldOp.getOperand(i);
      auto srcIt = valueSrcPorts.find(yieldOperand);
      if (srcIt != valueSrcPorts.end()) {
        for (IdIndex srcPortId : srcIt->second) {
          auto edge = std::make_unique<Edge>();
          edge->srcPort = srcPortId;
          edge->dstPort = portId;
          IdIndex edgeId = adg.addEdge(std::move(edge));
          adg.ports[srcPortId]->connectedEdges.push_back(edgeId);
          adg.ports[portId]->connectedEdges.push_back(edgeId);
          connectivity.outToIn[srcPortId].push_back(portId);
        }
      }
    }
  }

  // Boundary sentinel edges are already wired via SSA:
  // - Input sentinels: their output ports were registered in valueSrcPorts
  //   for block arguments, so SW/PE instances that consume those values
  //   already have edges from the sentinel to their input ports.
  // - Output sentinels: wired above from yield operands via valueSrcPorts.

  // Count connectivity entries.
  size_t totalOutToIn = 0;
  for (auto &kv : connectivity.outToIn)
    totalOutToIn += kv.second.size();
  size_t totalInToOut = 0;
  for (auto &kv : connectivity.inToOut)
    totalInToOut += kv.second.size();

  // Count sentinel nodes.
  unsigned inputSentinels = 0, outputSentinels = 0;
  for (IdIndex i = 0; i < static_cast<IdIndex>(adg.nodes.size()); ++i) {
    auto *n = adg.getNode(i);
    if (!n)
      continue;
    if (n->kind == Node::ModuleInputNode)
      inputSentinels++;
    else if (n->kind == Node::ModuleOutputNode)
      outputSentinels++;
  }

  llvm::outs() << "ADGFlattener: " << adg.countNodes() << " nodes, "
               << adg.countPorts() << " ports, " << adg.countEdges()
               << " edges\n";
  llvm::outs() << "  PEs: " << peContainment.size() << " (with "
               << totalFuNodes << " FU nodes)\n";
  llvm::outs() << "  Boundary sentinels: " << inputSentinels << " input, "
               << outputSentinels << " output\n";
  llvm::outs() << "  Connectivity: " << connectivity.outToIn.size()
               << " out ports (" << totalOutToIn << " out->in entries), "
               << connectivity.inToOut.size() << " in ports ("
               << totalInToOut << " in->out entries)\n";


  return true;
}

std::pair<int, int> ADGFlattener::getNodeGridPos(IdIndex nodeId) const {
  auto it = nodeGridPos.find(nodeId);
  if (it != nodeGridPos.end())
    return it->second;
  return {-1, -1};
}

} // namespace fcc
