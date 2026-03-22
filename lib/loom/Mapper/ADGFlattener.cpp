#include "loom/Mapper/ADGFlattener.h"
#include "ADGFlattenerContext.h"

#include "loom/Dialect/Fabric/FabricDialect.h"
#include "loom/Dialect/Fabric/FabricOps.h"
#include "loom/Dialect/Fabric/FabricTypes.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/Support/raw_ostream.h"

namespace loom {

namespace {

bool connectivityPositionEnabled(mlir::ArrayAttr table, unsigned outputIdx,
                                 unsigned inputIdx) {
  if (!table || table.empty())
    return true;
  if (outputIdx >= table.size())
    return false;
  auto rowAttr = mlir::dyn_cast<mlir::StringAttr>(table[outputIdx]);
  if (!rowAttr)
    return false;
  llvm::StringRef row = rowAttr.getValue();
  if (inputIdx >= row.size())
    return false;
  return row[inputIdx] == '1';
}

void addSwitchInternalConnectivity(
    ConnectivityMatrix &connectivity, const Node *node,
    mlir::ArrayAttr connectivityTable = mlir::ArrayAttr()) {
  if (!node)
    return;
  for (unsigned inputIdx = 0; inputIdx < node->inputPorts.size(); ++inputIdx) {
    IdIndex inputPort = node->inputPorts[inputIdx];
    for (unsigned outputIdx = 0; outputIdx < node->outputPorts.size();
         ++outputIdx) {
      if (!connectivityPositionEnabled(connectivityTable, outputIdx, inputIdx))
        continue;
      connectivity.inToOut[inputPort].push_back(node->outputPorts[outputIdx]);
    }
  }
}

} // namespace

// --- FlattenContext method implementations ---

std::string FlattenContext::getOrCreateOpName(mlir::Operation &op) {
  if (auto instOp = mlir::dyn_cast<loom::fabric::InstanceOp>(op)) {
    if (auto symName = instOp.getSymName())
      return symName->str();
    return ("inst_" + std::to_string(opToNodeId.size()));
  }
  if (auto extOp = mlir::dyn_cast<loom::fabric::ExtMemoryOp>(op)) {
    if (auto symNameAttr = extOp.getSymNameAttr())
      return symNameAttr.getValue().str();
    return ("extmemory_" + std::to_string(autoExtMemCount++));
  }
  if (auto memOp = mlir::dyn_cast<loom::fabric::MemoryOp>(op)) {
    if (auto symNameAttr = memOp.getSymNameAttr())
      return symNameAttr.getValue().str();
    return ("memory_" + std::to_string(autoMemCount++));
  }
  if (auto fifoOp = mlir::dyn_cast<loom::fabric::FifoOp>(op)) {
    if (auto symNameAttr = fifoOp.getSymNameAttr())
      return symNameAttr.getValue().str();
    return ("fifo_" + std::to_string(autoFifoCount++));
  }
  if (auto tswOp = mlir::dyn_cast<loom::fabric::TemporalSwOp>(op)) {
    if (auto symNameAttr = tswOp.getSymNameAttr())
      return symNameAttr.getValue().str();
    return ("temporal_sw_" + std::to_string(autoTemporalSwCount++));
  }
  if (mlir::isa<loom::fabric::AddTagOp>(op))
    return ("add_tag_" + std::to_string(autoAddTagCount++));
  if (mlir::isa<loom::fabric::DelTagOp>(op))
    return ("del_tag_" + std::to_string(autoDelTagCount++));
  if (mlir::isa<loom::fabric::MapTagOp>(op))
    return ("map_tag_" + std::to_string(autoMapTagCount++));
  return op.getName().getStringRef().str();
}

bool FlattenContext::isDefinitionOp(mlir::Operation *op,
                                    llvm::StringRef name) const {
  if (mlir::isa<loom::fabric::FunctionUnitOp>(op))
    return true;
  if (!mlir::isa<loom::fabric::SpatialPEOp, loom::fabric::TemporalPEOp,
                 loom::fabric::SpatialSwOp, loom::fabric::TemporalSwOp,
                 loom::fabric::ExtMemoryOp, loom::fabric::MemoryOp,
                 loom::fabric::FifoOp>(op)) {
    return false;
  }
  return !op->hasAttr("inline_instantiation");
}

// --- ADGFlattener::flatten() - top-level coordinator ---

bool ADGFlattener::flatten(mlir::ModuleOp topModule, mlir::MLIRContext *ctx) {
  adg = Graph(ctx);

  // Find the fabric.module inside the top-level MLIR module.
  loom::fabric::ModuleOp fabricMod;
  topModule->walk([&](loom::fabric::ModuleOp mod) { fabricMod = mod; });
  if (!fabricMod) {
    llvm::errs() << "ADGFlattener: no fabric.module found\n";
    return false;
  }

  FlattenContext fctx;
  fctx.ctx = ctx;
  fctx.fabricMod = fabricMod;

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
    fctx.valueToOutputPort[arg] = adg.nodes[nodeId]->outputPorts[0];
  }

  // Module output sentinels will be created after all ops are processed
  // (in the yield-handling section in flattenWireEdges).

  topModule.walk([&](loom::fabric::InstanceOp instOp) {
    fctx.referencedTargetsByBlock[instOp->getBlock()].insert(
        instOp.getModule());
  });

  // Pre-pass: Collect definition symbols for instance resolution.
  topModule->walk([&](loom::fabric::SpatialPEOp peOp) {
    if (auto symNameAttr = peOp.getSymNameAttr();
        symNameAttr && fctx.isDefinitionOp(peOp.getOperation(),
                                           symNameAttr.getValue()))
      fctx.peDefMap[symNameAttr.getValue()] = peOp;
  });
  topModule->walk([&](loom::fabric::TemporalPEOp peOp) {
    if (auto symNameAttr = peOp.getSymNameAttr();
        symNameAttr && fctx.isDefinitionOp(peOp.getOperation(),
                                           symNameAttr.getValue()))
      fctx.temporalPeDefMap[symNameAttr.getValue()] = peOp;
  });
  topModule->walk([&](loom::fabric::SpatialSwOp swOp) {
    if (auto symNameAttr = swOp.getSymNameAttr();
        symNameAttr && fctx.isDefinitionOp(swOp.getOperation(),
                                           symNameAttr.getValue()))
      fctx.swDefMap[symNameAttr.getValue()] = swOp;
  });
  topModule->walk([&](loom::fabric::TemporalSwOp swOp) {
    if (auto symNameAttr = swOp.getSymNameAttr();
        symNameAttr && fctx.isDefinitionOp(swOp.getOperation(),
                                           symNameAttr.getValue()))
      fctx.temporalSwDefMap[symNameAttr.getValue()] = swOp;
  });
  topModule->walk([&](loom::fabric::ExtMemoryOp extOp) {
    if (auto symNameAttr = extOp.getSymNameAttr();
        symNameAttr && fctx.isDefinitionOp(extOp.getOperation(),
                                           symNameAttr.getValue()))
      fctx.extMemoryDefMap[symNameAttr.getValue()] = extOp;
  });
  topModule->walk([&](loom::fabric::MemoryOp memOp) {
    if (auto symNameAttr = memOp.getSymNameAttr();
        symNameAttr && fctx.isDefinitionOp(memOp.getOperation(),
                                           symNameAttr.getValue()))
      fctx.memoryDefMap[symNameAttr.getValue()] = memOp;
  });
  topModule->walk([&](loom::fabric::FifoOp fifoOp) {
    if (auto symNameAttr = fifoOp.getSymNameAttr();
        symNameAttr && fctx.isDefinitionOp(fifoOp.getOperation(),
                                           symNameAttr.getValue()))
      fctx.fifoDefMap[symNameAttr.getValue()] = fifoOp;
  });
  topModule->walk([&](loom::fabric::FunctionUnitOp fuOp) {
    auto symName = fuOp.getSymNameAttr().getValue();
    if (fctx.isDefinitionOp(fuOp.getOperation(), symName))
      fctx.functionUnitDefMap[symName] = fuOp;
  });

  // Pass 1: Create nodes for each hardware resource.
  flattenCreateNodes(fctx, body);

  // Pass 2: Wire edges + legacy fallback + module output sentinels +
  // bridge analysis + finalization.
  flattenWireEdges(fctx, body);

  // Bridge port analysis for multi-lane memory nodes.
  flattenAnalyzeBridges(fctx);

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

  inferMissingNodeGridPositions(adg, nodeGridPos);

  llvm::outs() << "ADGFlattener: " << adg.countNodes() << " nodes, "
               << adg.countPorts() << " ports, " << adg.countEdges()
               << " edges\n";
  llvm::outs() << "  PEs: " << peContainment.size() << " (with "
               << fctx.totalFuNodes << " FU nodes)\n";
  llvm::outs() << "  Boundary sentinels: " << inputSentinels << " input, "
               << outputSentinels << " output\n";
  llvm::outs() << "  Connectivity: " << connectivity.outToIn.size()
               << " out ports (" << totalOutToIn << " out->in entries), "
               << connectivity.inToOut.size() << " in ports ("
               << totalInToOut << " in->out entries)\n";

  adg.buildAttributeCache();

  return true;
}

// --- flattenCreateNodes: Pass 1 - create ADG nodes for hardware resources ---

void ADGFlattener::flattenCreateNodes(FlattenContext &fctx,
                                      mlir::Block &body) {
  mlir::MLIRContext *ctx = fctx.ctx;

  // Helper lambda: create FU nodes from a PE definition for a given instance.
  auto createFUNodesFromPE = [&](auto peOp, llvm::StringRef instanceName,
                                 llvm::StringRef peKind,
                                 unsigned numInstruction,
                                 unsigned numRegister,
                                 unsigned regFifoDepth, unsigned tagWidth,
                                 bool enableShareOperandBuffer,
                                 unsigned operandBufferSize) {
    auto gridPos = parseGridPos(instanceName);

    PEContainment pe;
    pe.peName = instanceName.str();
    pe.peKind = peKind.str();
    pe.row = gridPos.first;
    pe.col = gridPos.second;

    // Set PE-level port counts from the PE function type.
    auto peFnType = peOp.getFunctionType();
    pe.numInputPorts = peFnType.getNumInputs();
    pe.numOutputPorts = peFnType.getNumResults();
    pe.numInstruction = numInstruction;
    pe.numRegister = numRegister;
    pe.regFifoDepth = regFifoDepth;
    pe.tagWidth = tagWidth;
    pe.enableShareOperandBuffer = enableShareOperandBuffer;
    pe.operandBufferSize = operandBufferSize;

    auto instantiateFU = [&](loom::fabric::FunctionUnitOp fuOp,
                             llvm::StringRef fuInstanceName) {
      auto fuNode = std::make_unique<Node>();
      fuNode->kind = Node::OperationNode;

      std::string fuName = fuInstanceName.str();
      if (fuName.empty())
        fuName = fuOp.getSymName().str();
      setNodeAttr(fuNode.get(), "op_name",
                  mlir::StringAttr::get(ctx, fuName), ctx);
      setNodeAttr(fuNode.get(), "op_kind",
                  mlir::StringAttr::get(ctx, "function_unit"), ctx);
      setNodeAttr(fuNode.get(), "resource_class",
                  mlir::StringAttr::get(ctx, "functional"), ctx);
      setNodeAttr(fuNode.get(), "pe_name",
                  mlir::StringAttr::get(ctx, instanceName), ctx);
      setNodeAttr(fuNode.get(), "pe_kind",
                  mlir::StringAttr::get(ctx, peKind), ctx);

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

      if (auto fieldWidths = extractFUConfigFieldWidths(fuOp, ctx)) {
        setNodeAttr(fuNode.get(), "fu_config_field_widths", fieldWidths, ctx);
        int64_t totalBits = 0;
        for (int64_t width : fieldWidths.asArrayRef())
          totalBits += width;
        setNodeAttr(fuNode.get(), "fu_config_bits",
                    mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 64),
                                           totalBits),
                    ctx);
      } else {
        setNodeAttr(fuNode.get(), "fu_config_bits",
                    mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 64), 0),
                    ctx);
      }

      if (fuOp.getLatency()) {
        setNodeAttr(fuNode.get(), "latency",
                    mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 64),
                                           *fuOp.getLatency()),
                    ctx);
      }
      if (fuOp.getInterval()) {
        setNodeAttr(fuNode.get(), "interval",
                    mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 64),
                                           *fuOp.getInterval()),
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
    };

    auto &peBody = peOp.getBody().front();
    auto referencedIt = fctx.referencedTargetsByBlock.find(&peBody);
    const llvm::DenseSet<llvm::StringRef> *referencedTargets =
        referencedIt != fctx.referencedTargetsByBlock.end()
            ? &referencedIt->second
            : nullptr;
    for (auto &innerOp : peBody.getOperations()) {
      if (auto fuOp = mlir::dyn_cast<loom::fabric::FunctionUnitOp>(innerOp)) {
        llvm::StringRef symName = fuOp.getSymNameAttr().getValue();
        if (!symName.empty() && referencedTargets &&
            referencedTargets->contains(symName))
          continue;
        instantiateFU(fuOp, symName);
        continue;
      }

      auto instOp = mlir::dyn_cast<loom::fabric::InstanceOp>(innerOp);
      if (!instOp)
        continue;
      auto fuIt = fctx.functionUnitDefMap.find(instOp.getModule());
      if (fuIt == fctx.functionUnitDefMap.end())
        continue;
      instantiateFU(fuIt->second,
                    instOp.getSymName().value_or(instOp.getModule()));
    }

    peContainment.push_back(pe);
  };

  // Pass 1: Create nodes for each hardware resource.
  // For spatial_pe: flatten by creating one FU node per function_unit inside.
  // For switches/fifos/extmemory: create one node each.
  // For fabric.instance: resolve to PE or SW definition and create nodes.

  for (auto &op : body.getOperations()) {
    // Handle fabric.instance ops (resolve to PE or SW definitions).
    if (auto instOp = mlir::dyn_cast<loom::fabric::InstanceOp>(op)) {
      llvm::StringRef moduleName = instOp.getModule();
      std::string instanceName;
      if (auto symName = instOp.getSymName())
        instanceName = symName->str();

      // Check if this instance references a PE definition.
      auto peIt = fctx.peDefMap.find(moduleName);
      if (peIt != fctx.peDefMap.end()) {
        createFUNodesFromPE(peIt->second, instanceName, "spatial_pe", 0, 0, 0,
                            0, false, 0);
        continue;
      }

      auto temporalPeIt = fctx.temporalPeDefMap.find(moduleName);
      if (temporalPeIt != fctx.temporalPeDefMap.end()) {
        auto temporalFnType = temporalPeIt->second.getFunctionType();
        unsigned tagWidth = 0;
        if (temporalFnType.getNumInputs() > 0) {
          if (auto tagged =
                  mlir::dyn_cast<loom::fabric::TaggedType>(temporalFnType.getInput(0))) {
            if (auto intTy = mlir::dyn_cast<mlir::IntegerType>(tagged.getTagType()))
              tagWidth = intTy.getWidth();
          }
        }
        createFUNodesFromPE(
            temporalPeIt->second, instanceName, "temporal_pe",
            static_cast<unsigned>(std::max<int64_t>(0, temporalPeIt->second.getNumInstruction())),
            static_cast<unsigned>(std::max<int64_t>(0, temporalPeIt->second.getNumRegister())),
            static_cast<unsigned>(std::max<int64_t>(0, temporalPeIt->second.getRegFifoDepth())),
            tagWidth, temporalPeIt->second.getEnableShareOperandBuffer(),
            static_cast<unsigned>(temporalPeIt->second.getOperandBufferSize().value_or(0)));
        continue;
      }

      // Check if this instance references a SW definition.
      auto swIt = fctx.swDefMap.find(moduleName);
      if (swIt != fctx.swDefMap.end()) {
        auto swDef = swIt->second;
        auto swNode = std::make_unique<Node>();
        swNode->kind = Node::OperationNode;

        setNodeAttr(swNode.get(), "op_name",
                    mlir::StringAttr::get(ctx, instanceName), ctx);
        setNodeAttr(swNode.get(), "op_kind",
                    mlir::StringAttr::get(ctx, "spatial_sw"), ctx);
        setNodeAttr(swNode.get(), "resource_class",
                    mlir::StringAttr::get(ctx, "routing"), ctx);
        if (auto ct = swDef.getConnectivityTable())
          setNodeAttr(swNode.get(), "connectivity_table", *ct, ctx);

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
        fctx.opToNodeId[instOp.getOperation()] = swNodeId;
        nodeGridPos[swNodeId] = gridPos;

        auto *node = adg.getNode(swNodeId);
        addSwitchInternalConnectivity(connectivity, node,
                                      swDef.getConnectivityTable().value_or(
                                          mlir::ArrayAttr()));

        for (unsigned i = 0; i < instOp.getNumResults(); ++i) {
          fctx.valueToOutputPort[instOp.getResult(i)] = node->outputPorts[i];
        }
        continue;
      }

      auto tswIt = fctx.temporalSwDefMap.find(moduleName);
      if (tswIt != fctx.temporalSwDefMap.end()) {
        auto tswDef = tswIt->second;
        auto tswNode = std::make_unique<Node>();
        tswNode->kind = Node::OperationNode;

        if (instanceName.empty())
          instanceName = fctx.getOrCreateOpName(op);

        setNodeAttr(tswNode.get(), "op_name",
                    mlir::StringAttr::get(ctx, instanceName), ctx);
        setNodeAttr(tswNode.get(), "op_kind",
                    mlir::StringAttr::get(ctx, "temporal_sw"), ctx);
        setNodeAttr(tswNode.get(), "resource_class",
                    mlir::StringAttr::get(ctx, "routing"), ctx);
        setNodeAttr(tswNode.get(), "num_route_table",
                    mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 64),
                                           tswDef.getNumRouteTable()),
                    ctx);
        if (auto ct = tswDef.getConnectivityTable())
          setNodeAttr(tswNode.get(), "connectivity_table", *ct, ctx);

        auto gridPos = parseGridPos(instanceName);
        auto tswFnType = tswDef.getFunctionType();

        for (unsigned i = 0; i < tswFnType.getNumInputs(); ++i) {
          auto port = std::make_unique<Port>();
          port->direction = Port::Input;
          port->type = tswFnType.getInput(i);
          IdIndex portId = adg.addPort(std::move(port));
          adg.ports[portId]->parentNode =
              static_cast<IdIndex>(adg.nodes.size());
          tswNode->inputPorts.push_back(portId);
        }

        for (unsigned i = 0; i < tswFnType.getNumResults(); ++i) {
          auto port = std::make_unique<Port>();
          port->direction = Port::Output;
          port->type = tswFnType.getResult(i);
          IdIndex portId = adg.addPort(std::move(port));
          adg.ports[portId]->parentNode =
              static_cast<IdIndex>(adg.nodes.size());
          tswNode->outputPorts.push_back(portId);
        }

        IdIndex tswNodeId = adg.addNode(std::move(tswNode));
        fctx.opToNodeId[instOp.getOperation()] = tswNodeId;
        nodeGridPos[tswNodeId] = gridPos;

        auto *node = adg.getNode(tswNodeId);
        addSwitchInternalConnectivity(connectivity, node,
                                      tswDef.getConnectivityTable().value_or(
                                          mlir::ArrayAttr()));

        for (unsigned i = 0; i < instOp.getNumResults(); ++i)
          fctx.valueToOutputPort[instOp.getResult(i)] = node->outputPorts[i];
        continue;
      }

      auto extIt = fctx.extMemoryDefMap.find(moduleName);
      if (extIt != fctx.extMemoryDefMap.end()) {
        auto extDef = extIt->second;
        auto memNode = std::make_unique<Node>();
        memNode->kind = Node::OperationNode;

        if (instanceName.empty())
          instanceName = moduleName.str();
        setNodeAttr(memNode.get(), "op_name",
                    mlir::StringAttr::get(ctx, instanceName), ctx);
        setNodeAttr(memNode.get(), "op_kind",
                    mlir::StringAttr::get(ctx, "extmemory"), ctx);
        setNodeAttr(memNode.get(), "resource_class",
                    mlir::StringAttr::get(ctx, "memory"), ctx);
        setNodeAttr(memNode.get(), "ldCount",
                    mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 64),
                                           extDef.getLdCount()),
                    ctx);
        setNodeAttr(memNode.get(), "stCount",
                    mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 64),
                                           extDef.getStCount()),
                    ctx);
        if (auto numRegionAttr =
                extDef->getAttrOfType<mlir::IntegerAttr>("numRegion")) {
          setNodeAttr(memNode.get(), "numRegion", numRegionAttr, ctx);
        }
        if (auto memrefTypeAttr =
                extDef->getAttrOfType<mlir::TypeAttr>("memref_type")) {
          setNodeAttr(memNode.get(), "memref_type", memrefTypeAttr, ctx);
        }
        if (auto addrOffsetAttr =
                extDef->getAttrOfType<mlir::DenseI64ArrayAttr>("addrOffsetTable")) {
          setNodeAttr(memNode.get(), "addrOffsetTable", addrOffsetAttr, ctx);
        }

        auto memFnType = extDef.getFunctionType();
        for (unsigned i = 0; i < memFnType.getNumInputs(); ++i) {
          auto port = std::make_unique<Port>();
          port->direction = Port::Input;
          port->type = memFnType.getInput(i);
          IdIndex portId = adg.addPort(std::move(port));
          adg.ports[portId]->parentNode =
              static_cast<IdIndex>(adg.nodes.size());
          memNode->inputPorts.push_back(portId);
        }
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
        fctx.opToNodeId[instOp.getOperation()] = memNodeId;
        nodeGridPos[memNodeId] = parseGridPos(instanceName);
        auto *node = adg.getNode(memNodeId);
        for (IdIndex ip : node->inputPorts)
          for (IdIndex opPort : node->outputPorts)
            connectivity.inToOut[ip].push_back(opPort);
        for (unsigned i = 0; i < instOp.getNumResults(); ++i)
          fctx.valueToOutputPort[instOp.getResult(i)] = node->outputPorts[i];
        continue;
      }

      auto memIt = fctx.memoryDefMap.find(moduleName);
      if (memIt != fctx.memoryDefMap.end()) {
        auto memDef = memIt->second;
        auto memNode = std::make_unique<Node>();
        memNode->kind = Node::OperationNode;

        if (instanceName.empty())
          instanceName = moduleName.str();
        setNodeAttr(memNode.get(), "op_name",
                    mlir::StringAttr::get(ctx, instanceName), ctx);
        setNodeAttr(memNode.get(), "op_kind",
                    mlir::StringAttr::get(ctx, "memory"), ctx);
        setNodeAttr(memNode.get(), "resource_class",
                    mlir::StringAttr::get(ctx, "memory"), ctx);
        setNodeAttr(memNode.get(), "ldCount",
                    mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 64),
                                           memDef.getLdCount()),
                    ctx);
        setNodeAttr(memNode.get(), "stCount",
                    mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 64),
                                           memDef.getStCount()),
                    ctx);
        if (auto numRegionAttr =
                memDef->getAttrOfType<mlir::IntegerAttr>("numRegion")) {
          setNodeAttr(memNode.get(), "numRegion", numRegionAttr, ctx);
        }
        if (auto memrefTypeAttr =
                memDef->getAttrOfType<mlir::TypeAttr>("memref_type")) {
          setNodeAttr(memNode.get(), "memref_type", memrefTypeAttr, ctx);
        }
        if (auto addrOffsetAttr =
                memDef->getAttrOfType<mlir::DenseI64ArrayAttr>("addrOffsetTable")) {
          setNodeAttr(memNode.get(), "addrOffsetTable", addrOffsetAttr, ctx);
        }

        auto memFnType = memDef.getFunctionType();
        for (unsigned i = 0; i < memFnType.getNumInputs(); ++i) {
          auto port = std::make_unique<Port>();
          port->direction = Port::Input;
          port->type = memFnType.getInput(i);
          IdIndex portId = adg.addPort(std::move(port));
          adg.ports[portId]->parentNode =
              static_cast<IdIndex>(adg.nodes.size());
          memNode->inputPorts.push_back(portId);
        }
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
        fctx.opToNodeId[instOp.getOperation()] = memNodeId;
        nodeGridPos[memNodeId] = parseGridPos(instanceName);
        auto *node = adg.getNode(memNodeId);
        for (IdIndex ip : node->inputPorts)
          for (IdIndex opPort : node->outputPorts)
            connectivity.inToOut[ip].push_back(opPort);
        for (unsigned i = 0; i < instOp.getNumResults(); ++i)
          fctx.valueToOutputPort[instOp.getResult(i)] = node->outputPorts[i];
        continue;
      }

      auto fifoIt = fctx.fifoDefMap.find(moduleName);
      if (fifoIt != fctx.fifoDefMap.end()) {
        auto fifoDef = fifoIt->second;
        auto fifoNode = std::make_unique<Node>();
        fifoNode->kind = Node::OperationNode;

        if (instanceName.empty())
          instanceName = moduleName.str();
        setNodeAttr(fifoNode.get(), "op_name",
                    mlir::StringAttr::get(ctx, instanceName), ctx);
        setNodeAttr(fifoNode.get(), "op_kind",
                    mlir::StringAttr::get(ctx, "fifo"), ctx);
        setNodeAttr(fifoNode.get(), "resource_class",
                    mlir::StringAttr::get(ctx, "routing"), ctx);
        setNodeAttr(fifoNode.get(), "depth",
                    mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 64),
                                           fifoDef.getDepth()),
                    ctx);
        setNodeAttr(fifoNode.get(), "bypassable",
                    mlir::BoolAttr::get(ctx, static_cast<bool>(fifoDef.getBypassable())),
                    ctx);
        bool bypassed = false;
        if (auto attr = mlir::dyn_cast_or_null<mlir::BoolAttr>(
                fifoDef->getAttr("bypassed"))) {
          bypassed = attr.getValue();
        }
        setNodeAttr(fifoNode.get(), "bypassed",
                    mlir::BoolAttr::get(ctx, bypassed), ctx);

        auto fifoFnType = fifoDef.getFunctionType();
        for (unsigned i = 0; i < fifoFnType.getNumInputs(); ++i) {
          auto port = std::make_unique<Port>();
          port->direction = Port::Input;
          port->type = fifoFnType.getInput(i);
          IdIndex portId = adg.addPort(std::move(port));
          adg.ports[portId]->parentNode =
              static_cast<IdIndex>(adg.nodes.size());
          fifoNode->inputPorts.push_back(portId);
        }
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
        fctx.opToNodeId[instOp.getOperation()] = fifoNodeId;
        nodeGridPos[fifoNodeId] = parseGridPos(instanceName);
        auto *node = adg.getNode(fifoNodeId);
        for (IdIndex ip : node->inputPorts)
          for (IdIndex opPort : node->outputPorts)
            connectivity.inToOut[ip].push_back(opPort);
        for (unsigned i = 0; i < instOp.getNumResults(); ++i)
          fctx.valueToOutputPort[instOp.getResult(i)] = node->outputPorts[i];
        continue;
      }

      // Unknown instance type -- skip.
      continue;
    }

    if (auto peOp = mlir::dyn_cast<loom::fabric::SpatialPEOp>(op)) {
      if (!peOp->hasAttr("inline_instantiation"))
        continue;

      llvm::StringRef symName;
      if (auto symNameAttr = peOp.getSymNameAttr())
        symName = symNameAttr.getValue();

      std::string peName;
      if (!symName.empty())
        peName = symName.str();
      createFUNodesFromPE(peOp, peName, "spatial_pe", 0, 0, 0, 0, false, 0);
      continue;
    }

    if (auto peOp = mlir::dyn_cast<loom::fabric::TemporalPEOp>(op)) {
      if (!peOp->hasAttr("inline_instantiation"))
        continue;

      llvm::StringRef symName;
      if (auto symNameAttr = peOp.getSymNameAttr())
        symName = symNameAttr.getValue();

      std::string peName;
      if (!symName.empty())
        peName = symName.str();
      auto temporalFnType = peOp.getFunctionType();
      unsigned tagWidth = 0;
      if (temporalFnType.getNumInputs() > 0) {
        if (auto tagged =
                mlir::dyn_cast<loom::fabric::TaggedType>(temporalFnType.getInput(0))) {
          if (auto intTy = mlir::dyn_cast<mlir::IntegerType>(tagged.getTagType()))
            tagWidth = intTy.getWidth();
        }
      }
      createFUNodesFromPE(
          peOp, peName, "temporal_pe",
          static_cast<unsigned>(std::max<int64_t>(0, peOp.getNumInstruction())),
          static_cast<unsigned>(std::max<int64_t>(0, peOp.getNumRegister())),
          static_cast<unsigned>(std::max<int64_t>(0, peOp.getRegFifoDepth())),
          tagWidth, peOp.getEnableShareOperandBuffer(),
          static_cast<unsigned>(peOp.getOperandBufferSize().value_or(0)));
      continue;
    }

    if (auto swOp = mlir::dyn_cast<loom::fabric::SpatialSwOp>(op)) {
      if (!swOp->hasAttr("inline_instantiation"))
        continue;
      llvm::StringRef symName;
      if (auto symNameAttr = swOp.getSymNameAttr())
        symName = symNameAttr.getValue();

      auto swNode = std::make_unique<Node>();
      swNode->kind = Node::OperationNode;

      std::string swName;
      if (!symName.empty())
        swName = symName.str();

      setNodeAttr(swNode.get(), "op_name",
                  mlir::StringAttr::get(ctx, swName), ctx);
      setNodeAttr(swNode.get(), "op_kind",
                  mlir::StringAttr::get(ctx, "spatial_sw"), ctx);
      setNodeAttr(swNode.get(), "resource_class",
                  mlir::StringAttr::get(ctx, "routing"), ctx);
      if (auto ct = swOp.getConnectivityTable())
        setNodeAttr(swNode.get(), "connectivity_table", *ct, ctx);

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
      fctx.opToNodeId[swOp.getOperation()] = swNodeId;
      nodeGridPos[swNodeId] = gridPos;

      // Register internal connectivity based on connectivity_table.
      auto *node = adg.getNode(swNodeId);
      addSwitchInternalConnectivity(connectivity, node,
                                    swOp.getConnectivityTable().value_or(
                                        mlir::ArrayAttr()));

      // Map SSA results to output ports.
      for (unsigned i = 0; i < swOp.getNumResults(); ++i) {
        fctx.valueToOutputPort[swOp.getResult(i)] = node->outputPorts[i];
      }
      continue;
    }

    if (auto tswOp = mlir::dyn_cast<loom::fabric::TemporalSwOp>(op)) {
      if (!tswOp->hasAttr("inline_instantiation"))
        continue;
      llvm::StringRef symName;
      if (auto symNameAttr = tswOp.getSymNameAttr())
        symName = symNameAttr.getValue();
      auto tswNode = std::make_unique<Node>();
      tswNode->kind = Node::OperationNode;

      std::string tswName = fctx.getOrCreateOpName(op);
      setNodeAttr(tswNode.get(), "op_name",
                  mlir::StringAttr::get(ctx, tswName), ctx);
      setNodeAttr(tswNode.get(), "op_kind",
                  mlir::StringAttr::get(ctx, "temporal_sw"), ctx);
      setNodeAttr(tswNode.get(), "resource_class",
                  mlir::StringAttr::get(ctx, "routing"), ctx);
      setNodeAttr(tswNode.get(), "num_route_table",
                  mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 64),
                                         tswOp.getNumRouteTable()),
                  ctx);
      if (auto ct = tswOp.getConnectivityTable())
        setNodeAttr(tswNode.get(), "connectivity_table", *ct, ctx);

      auto gridPos = parseGridPos(tswName);
      auto tswFnType = tswOp.getFunctionType();
      for (unsigned i = 0; i < tswFnType.getNumInputs(); ++i) {
        auto port = std::make_unique<Port>();
        port->direction = Port::Input;
        port->type = tswFnType.getInput(i);
        IdIndex portId = adg.addPort(std::move(port));
        adg.ports[portId]->parentNode =
            static_cast<IdIndex>(adg.nodes.size());
        tswNode->inputPorts.push_back(portId);
      }
      for (unsigned i = 0; i < tswFnType.getNumResults(); ++i) {
        auto port = std::make_unique<Port>();
        port->direction = Port::Output;
        port->type = tswFnType.getResult(i);
        IdIndex portId = adg.addPort(std::move(port));
        adg.ports[portId]->parentNode =
            static_cast<IdIndex>(adg.nodes.size());
        tswNode->outputPorts.push_back(portId);
      }

      IdIndex tswNodeId = adg.addNode(std::move(tswNode));
      fctx.opToNodeId[tswOp.getOperation()] = tswNodeId;
      nodeGridPos[tswNodeId] = gridPos;

      auto *node = adg.getNode(tswNodeId);
      addSwitchInternalConnectivity(connectivity, node,
                                    tswOp.getConnectivityTable().value_or(
                                        mlir::ArrayAttr()));
      for (unsigned i = 0; i < tswOp.getNumResults(); ++i)
        fctx.valueToOutputPort[tswOp.getResult(i)] = node->outputPorts[i];
      continue;
    }

    if (auto extOp = mlir::dyn_cast<loom::fabric::ExtMemoryOp>(op)) {
      if (!extOp->hasAttr("inline_instantiation"))
        continue;
      llvm::StringRef symName;
      if (auto symNameAttr = extOp.getSymNameAttr())
        symName = symNameAttr.getValue();
      auto memNode = std::make_unique<Node>();
      memNode->kind = Node::OperationNode;

      std::string memName = fctx.getOrCreateOpName(op);

      setNodeAttr(memNode.get(), "op_name",
                  mlir::StringAttr::get(ctx, memName), ctx);
      setNodeAttr(memNode.get(), "op_kind",
                  mlir::StringAttr::get(ctx, "extmemory"), ctx);
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
      if (auto connSwDetailAttr = extOp->getAttrOfType<mlir::ArrayAttr>(
              "connected_sw_detail")) {
        setNodeAttr(memNode.get(), "connected_sw_detail", connSwDetailAttr, ctx);
      }
      if (auto numRegionAttr =
              extOp->getAttrOfType<mlir::IntegerAttr>("numRegion")) {
        setNodeAttr(memNode.get(), "numRegion", numRegionAttr, ctx);
      }
      if (auto memrefTypeAttr =
              extOp->getAttrOfType<mlir::TypeAttr>("memref_type")) {
        setNodeAttr(memNode.get(), "memref_type", memrefTypeAttr, ctx);
      }
      if (auto addrOffsetAttr =
              extOp->getAttrOfType<mlir::DenseI64ArrayAttr>("addrOffsetTable")) {
        setNodeAttr(memNode.get(), "addrOffsetTable", addrOffsetAttr, ctx);
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
      fctx.opToNodeId[extOp.getOperation()] = memNodeId;

      // Internal connectivity.
      auto *node = adg.getNode(memNodeId);
      for (IdIndex ip : node->inputPorts) {
        for (IdIndex op : node->outputPorts) {
          connectivity.inToOut[ip].push_back(op);
        }
      }

      for (unsigned i = 0; i < extOp.getNumResults(); ++i) {
        fctx.valueToOutputPort[extOp.getResult(i)] = node->outputPorts[i];
      }
      continue;
    }

    if (auto memOp = mlir::dyn_cast<loom::fabric::MemoryOp>(op)) {
      if (!memOp->hasAttr("inline_instantiation"))
        continue;
      llvm::StringRef symName;
      if (auto symNameAttr = memOp.getSymNameAttr())
        symName = symNameAttr.getValue();
      auto memNode = std::make_unique<Node>();
      memNode->kind = Node::OperationNode;

      std::string memName = fctx.getOrCreateOpName(op);
      setNodeAttr(memNode.get(), "op_name",
                  mlir::StringAttr::get(ctx, memName), ctx);
      setNodeAttr(memNode.get(), "op_kind",
                  mlir::StringAttr::get(ctx, "memory"), ctx);
      setNodeAttr(memNode.get(), "resource_class",
                  mlir::StringAttr::get(ctx, "memory"), ctx);
      setNodeAttr(memNode.get(), "ldCount",
                  mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 64),
                                         memOp.getLdCount()),
                  ctx);
      setNodeAttr(memNode.get(), "stCount",
                  mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 64),
                                         memOp.getStCount()),
                  ctx);
      if (auto numRegionAttr =
              memOp->getAttrOfType<mlir::IntegerAttr>("numRegion")) {
        setNodeAttr(memNode.get(), "numRegion", numRegionAttr, ctx);
      }
      if (auto memrefTypeAttr =
              memOp->getAttrOfType<mlir::TypeAttr>("memref_type")) {
        setNodeAttr(memNode.get(), "memref_type", memrefTypeAttr, ctx);
      }
      if (auto addrOffsetAttr =
              memOp->getAttrOfType<mlir::DenseI64ArrayAttr>("addrOffsetTable")) {
        setNodeAttr(memNode.get(), "addrOffsetTable", addrOffsetAttr, ctx);
      }

      auto memFnType = memOp.getFunctionType();
      for (unsigned i = 0; i < memFnType.getNumInputs(); ++i) {
        auto port = std::make_unique<Port>();
        port->direction = Port::Input;
        port->type = memFnType.getInput(i);
        IdIndex portId = adg.addPort(std::move(port));
        adg.ports[portId]->parentNode =
            static_cast<IdIndex>(adg.nodes.size());
        memNode->inputPorts.push_back(portId);
      }
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
      fctx.opToNodeId[memOp.getOperation()] = memNodeId;
      auto *node = adg.getNode(memNodeId);
      for (IdIndex ip : node->inputPorts) {
        for (IdIndex op : node->outputPorts)
          connectivity.inToOut[ip].push_back(op);
      }
      for (unsigned i = 0; i < memOp.getNumResults(); ++i)
        fctx.valueToOutputPort[memOp.getResult(i)] = node->outputPorts[i];
      continue;
    }

    if (auto fifoOp = mlir::dyn_cast<loom::fabric::FifoOp>(op)) {
      if (!fifoOp->hasAttr("inline_instantiation"))
        continue;
      llvm::StringRef symName;
      if (auto symNameAttr = fifoOp.getSymNameAttr())
        symName = symNameAttr.getValue();
      auto fifoNode = std::make_unique<Node>();
      fifoNode->kind = Node::OperationNode;

      std::string fifoName = fctx.getOrCreateOpName(op);

      setNodeAttr(fifoNode.get(), "op_name",
                  mlir::StringAttr::get(ctx, fifoName), ctx);
      setNodeAttr(fifoNode.get(), "op_kind",
                  mlir::StringAttr::get(ctx, "fifo"), ctx);
      setNodeAttr(fifoNode.get(), "resource_class",
                  mlir::StringAttr::get(ctx, "routing"), ctx);
      setNodeAttr(fifoNode.get(), "depth",
                  mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 64),
                                         fifoOp.getDepth()),
                  ctx);
      setNodeAttr(fifoNode.get(), "bypassable",
                  mlir::BoolAttr::get(ctx, static_cast<bool>(fifoOp.getBypassable())),
                  ctx);
      bool bypassed = false;
      if (auto attr =
              mlir::dyn_cast_or_null<mlir::BoolAttr>(fifoOp->getAttr("bypassed")))
        bypassed = attr.getValue();
      setNodeAttr(fifoNode.get(), "bypassed",
                  mlir::BoolAttr::get(ctx, bypassed), ctx);

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
      fctx.opToNodeId[fifoOp.getOperation()] = fifoNodeId;

      // Internal connectivity.
      auto *node = adg.getNode(fifoNodeId);
      for (IdIndex ip : node->inputPorts) {
        for (IdIndex op : node->outputPorts) {
          connectivity.inToOut[ip].push_back(op);
        }
      }

      for (unsigned i = 0; i < fifoOp.getNumResults(); ++i) {
        fctx.valueToOutputPort[fifoOp.getResult(i)] = node->outputPorts[i];
      }
      continue;
    }

    if (auto addTagOp = mlir::dyn_cast<loom::fabric::AddTagOp>(op)) {
      auto routeNode = std::make_unique<Node>();
      routeNode->kind = Node::OperationNode;
      std::string nodeName = fctx.getOrCreateOpName(op);

      setNodeAttr(routeNode.get(), "op_name",
                  mlir::StringAttr::get(ctx, nodeName), ctx);
      setNodeAttr(routeNode.get(), "op_kind",
                  mlir::StringAttr::get(ctx, "add_tag"), ctx);
      setNodeAttr(routeNode.get(), "resource_class",
                  mlir::StringAttr::get(ctx, "routing"), ctx);
      if (auto tagAttr = addTagOp->getAttr("tag"))
        setNodeAttr(routeNode.get(), "tag", tagAttr, ctx);

      auto inPort = std::make_unique<Port>();
      inPort->direction = Port::Input;
      inPort->type = addTagOp.getValue().getType();
      IdIndex inPortId = adg.addPort(std::move(inPort));
      adg.ports[inPortId]->parentNode = static_cast<IdIndex>(adg.nodes.size());
      routeNode->inputPorts.push_back(inPortId);

      auto outPort = std::make_unique<Port>();
      outPort->direction = Port::Output;
      outPort->type = addTagOp.getResult().getType();
      IdIndex outPortId = adg.addPort(std::move(outPort));
      adg.ports[outPortId]->parentNode = static_cast<IdIndex>(adg.nodes.size());
      routeNode->outputPorts.push_back(outPortId);

      IdIndex nodeId = adg.addNode(std::move(routeNode));
      fctx.opToNodeId[addTagOp.getOperation()] = nodeId;
      connectivity.inToOut[inPortId].push_back(outPortId);
      fctx.valueToOutputPort[addTagOp.getResult()] = outPortId;
      continue;
    }

    if (auto delTagOp = mlir::dyn_cast<loom::fabric::DelTagOp>(op)) {
      auto routeNode = std::make_unique<Node>();
      routeNode->kind = Node::OperationNode;
      std::string nodeName = fctx.getOrCreateOpName(op);

      setNodeAttr(routeNode.get(), "op_name",
                  mlir::StringAttr::get(ctx, nodeName), ctx);
      setNodeAttr(routeNode.get(), "op_kind",
                  mlir::StringAttr::get(ctx, "del_tag"), ctx);
      setNodeAttr(routeNode.get(), "resource_class",
                  mlir::StringAttr::get(ctx, "routing"), ctx);

      auto inPort = std::make_unique<Port>();
      inPort->direction = Port::Input;
      inPort->type = delTagOp.getTagged().getType();
      IdIndex inPortId = adg.addPort(std::move(inPort));
      adg.ports[inPortId]->parentNode = static_cast<IdIndex>(adg.nodes.size());
      routeNode->inputPorts.push_back(inPortId);

      auto outPort = std::make_unique<Port>();
      outPort->direction = Port::Output;
      outPort->type = delTagOp.getResult().getType();
      IdIndex outPortId = adg.addPort(std::move(outPort));
      adg.ports[outPortId]->parentNode = static_cast<IdIndex>(adg.nodes.size());
      routeNode->outputPorts.push_back(outPortId);

      IdIndex nodeId = adg.addNode(std::move(routeNode));
      fctx.opToNodeId[delTagOp.getOperation()] = nodeId;
      connectivity.inToOut[inPortId].push_back(outPortId);
      fctx.valueToOutputPort[delTagOp.getResult()] = outPortId;
      continue;
    }

    if (auto mapTagOp = mlir::dyn_cast<loom::fabric::MapTagOp>(op)) {
      auto routeNode = std::make_unique<Node>();
      routeNode->kind = Node::OperationNode;
      std::string nodeName = fctx.getOrCreateOpName(op);

      setNodeAttr(routeNode.get(), "op_name",
                  mlir::StringAttr::get(ctx, nodeName), ctx);
      setNodeAttr(routeNode.get(), "op_kind",
                  mlir::StringAttr::get(ctx, "map_tag"), ctx);
      setNodeAttr(routeNode.get(), "resource_class",
                  mlir::StringAttr::get(ctx, "routing"), ctx);
      if (auto tableSizeAttr = mapTagOp->getAttr("table_size"))
        setNodeAttr(routeNode.get(), "table_size", tableSizeAttr, ctx);
      if (auto tableAttr = mapTagOp->getAttr("table"))
        setNodeAttr(routeNode.get(), "table", tableAttr, ctx);

      auto inPort = std::make_unique<Port>();
      inPort->direction = Port::Input;
      inPort->type = mapTagOp.getTagged().getType();
      IdIndex inPortId = adg.addPort(std::move(inPort));
      adg.ports[inPortId]->parentNode = static_cast<IdIndex>(adg.nodes.size());
      routeNode->inputPorts.push_back(inPortId);

      auto outPort = std::make_unique<Port>();
      outPort->direction = Port::Output;
      outPort->type = mapTagOp.getResult().getType();
      IdIndex outPortId = adg.addPort(std::move(outPort));
      adg.ports[outPortId]->parentNode = static_cast<IdIndex>(adg.nodes.size());
      routeNode->outputPorts.push_back(outPortId);

      IdIndex nodeId = adg.addNode(std::move(routeNode));
      fctx.opToNodeId[mapTagOp.getOperation()] = nodeId;
      connectivity.inToOut[inPortId].push_back(outPortId);
      fctx.valueToOutputPort[mapTagOp.getResult()] = outPortId;
      continue;
    }
  }
}

// --- getNodeGridPos ---

std::pair<int, int> ADGFlattener::getNodeGridPos(IdIndex nodeId) const {
  auto it = nodeGridPos.find(nodeId);
  if (it != nodeGridPos.end())
    return it->second;
  return {-1, -1};
}

} // namespace loom
