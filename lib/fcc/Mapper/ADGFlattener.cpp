#include "fcc/Mapper/ADGFlattener.h"
#include "fcc/Mapper/BridgeBinding.h"

#include "fcc/Dialect/Fabric/FabricDialect.h"
#include "fcc/Dialect/Fabric/FabricOps.h"
#include "fcc/Dialect/Fabric/FabricTypes.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Builders.h"

#include "llvm/ADT/StringMap.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <regex>
#include <type_traits>

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

void setEdgeAttr(Edge *edge, llvm::StringRef key, mlir::Attribute val,
                 mlir::MLIRContext *ctx) {
  edge->attributes.push_back(
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

mlir::DenseI64ArrayAttr
extractFUConfigFieldWidths(fcc::fabric::FunctionUnitOp fuOp,
                           mlir::MLIRContext *ctx) {
  llvm::SmallVector<int64_t, 4> widths;
  for (mlir::Operation &bodyOp : fuOp.getBody().front().getOperations()) {
    auto muxOp = mlir::dyn_cast<fcc::fabric::MuxOp>(bodyOp);
    if (!muxOp)
      continue;
    unsigned numInputs = muxOp.getInputs().size();
    unsigned numResults = muxOp.getResults().size();
    if (numInputs == 1 && numResults == 1)
      continue;
    unsigned branchCount = std::max(numInputs, numResults);
    unsigned selBits = branchCount > 1 ? llvm::Log2_32_Ceil(branchCount) : 0;
    widths.push_back(static_cast<int64_t>(selBits + 2));
  }
  if (widths.empty())
    return {};
  return mlir::DenseI64ArrayAttr::get(ctx, widths);
}

std::optional<llvm::StringRef> getNamedDefinition(mlir::Operation &op) {
  if (auto fuOp = mlir::dyn_cast<fcc::fabric::FunctionUnitOp>(op))
    return fuOp.getSymName();
  if (auto peOp = mlir::dyn_cast<fcc::fabric::SpatialPEOp>(op))
    return peOp.getSymName();
  if (auto peOp = mlir::dyn_cast<fcc::fabric::TemporalPEOp>(op))
    return peOp.getSymName();
  if (auto swOp = mlir::dyn_cast<fcc::fabric::SpatialSwOp>(op))
    return swOp.getSymName();
  if (auto swOp = mlir::dyn_cast<fcc::fabric::TemporalSwOp>(op))
    return swOp.getSymName();
  if (auto extOp = mlir::dyn_cast<fcc::fabric::ExtMemoryOp>(op))
    return extOp.getSymName();
  if (auto memOp = mlir::dyn_cast<fcc::fabric::MemoryOp>(op))
    return memOp.getSymName();
  if (auto fifoOp = mlir::dyn_cast<fcc::fabric::FifoOp>(op))
    return fifoOp.getSymName();
  return std::nullopt;
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

  llvm::DenseMap<mlir::Block *, llvm::DenseSet<llvm::StringRef>>
      referencedTargetsByBlock;
  topModule.walk([&](fcc::fabric::InstanceOp instOp) {
    referencedTargetsByBlock[instOp->getBlock()].insert(instOp.getModule());
  });

  auto isDefinitionOp = [&](mlir::Operation *op,
                            llvm::StringRef name) -> bool {
    if (mlir::isa<fcc::fabric::FunctionUnitOp>(op))
      return true;
    if (!mlir::isa<fcc::fabric::SpatialPEOp, fcc::fabric::TemporalPEOp,
                   fcc::fabric::SpatialSwOp, fcc::fabric::TemporalSwOp,
                   fcc::fabric::ExtMemoryOp, fcc::fabric::MemoryOp,
                   fcc::fabric::FifoOp>(op)) {
      return false;
    }
    return !op->hasAttr("inline_instantiation");
  };

  // Pre-pass: Collect definition symbols for instance resolution.
  llvm::StringMap<fcc::fabric::SpatialPEOp> peDefMap;
  llvm::StringMap<fcc::fabric::TemporalPEOp> temporalPeDefMap;
  llvm::StringMap<fcc::fabric::SpatialSwOp> swDefMap;
  llvm::StringMap<fcc::fabric::TemporalSwOp> temporalSwDefMap;
  llvm::StringMap<fcc::fabric::ExtMemoryOp> extMemoryDefMap;
  llvm::StringMap<fcc::fabric::MemoryOp> memoryDefMap;
  llvm::StringMap<fcc::fabric::FifoOp> fifoDefMap;
  llvm::StringMap<fcc::fabric::FunctionUnitOp> functionUnitDefMap;
  topModule->walk([&](fcc::fabric::SpatialPEOp peOp) {
    if (auto symNameAttr = peOp.getSymNameAttr();
        symNameAttr && isDefinitionOp(peOp.getOperation(),
                                      symNameAttr.getValue()))
      peDefMap[symNameAttr.getValue()] = peOp;
  });
  topModule->walk([&](fcc::fabric::TemporalPEOp peOp) {
    if (auto symNameAttr = peOp.getSymNameAttr();
        symNameAttr && isDefinitionOp(peOp.getOperation(),
                                      symNameAttr.getValue()))
      temporalPeDefMap[symNameAttr.getValue()] = peOp;
  });
  topModule->walk([&](fcc::fabric::SpatialSwOp swOp) {
    if (auto symNameAttr = swOp.getSymNameAttr();
        symNameAttr && isDefinitionOp(swOp.getOperation(),
                                      symNameAttr.getValue()))
      swDefMap[symNameAttr.getValue()] = swOp;
  });
  topModule->walk([&](fcc::fabric::TemporalSwOp swOp) {
    if (auto symNameAttr = swOp.getSymNameAttr();
        symNameAttr && isDefinitionOp(swOp.getOperation(),
                                      symNameAttr.getValue()))
      temporalSwDefMap[symNameAttr.getValue()] = swOp;
  });
  topModule->walk([&](fcc::fabric::ExtMemoryOp extOp) {
    if (auto symNameAttr = extOp.getSymNameAttr();
        symNameAttr && isDefinitionOp(extOp.getOperation(),
                                      symNameAttr.getValue()))
      extMemoryDefMap[symNameAttr.getValue()] = extOp;
  });
  topModule->walk([&](fcc::fabric::MemoryOp memOp) {
    if (auto symNameAttr = memOp.getSymNameAttr();
        symNameAttr && isDefinitionOp(memOp.getOperation(),
                                      symNameAttr.getValue()))
      memoryDefMap[symNameAttr.getValue()] = memOp;
  });
  topModule->walk([&](fcc::fabric::FifoOp fifoOp) {
    if (auto symNameAttr = fifoOp.getSymNameAttr();
        symNameAttr && isDefinitionOp(fifoOp.getOperation(),
                                      symNameAttr.getValue()))
      fifoDefMap[symNameAttr.getValue()] = fifoOp;
  });
  topModule->walk([&](fcc::fabric::FunctionUnitOp fuOp) {
    auto symName = fuOp.getSymNameAttr().getValue();
    if (isDefinitionOp(fuOp.getOperation(), symName))
      functionUnitDefMap[symName] = fuOp;
  });

  llvm::DenseMap<mlir::Operation *, IdIndex> opToNodeId;
  unsigned autoTemporalSwCount = 0;
  unsigned autoExtMemCount = 0;
  unsigned autoMemCount = 0;
  unsigned autoFifoCount = 0;
  unsigned autoAddTagCount = 0;
  unsigned autoDelTagCount = 0;
  unsigned autoMapTagCount = 0;

  auto getOrCreateOpName = [&](mlir::Operation &op) -> std::string {
    if (auto instOp = mlir::dyn_cast<fcc::fabric::InstanceOp>(op)) {
      if (auto symName = instOp.getSymName())
        return symName->str();
      return ("inst_" + std::to_string(opToNodeId.size()));
    }
    if (auto extOp = mlir::dyn_cast<fcc::fabric::ExtMemoryOp>(op)) {
      if (auto symNameAttr = extOp.getSymNameAttr())
        return symNameAttr.getValue().str();
      return ("extmemory_" + std::to_string(autoExtMemCount++));
    }
    if (auto memOp = mlir::dyn_cast<fcc::fabric::MemoryOp>(op)) {
      if (auto symNameAttr = memOp.getSymNameAttr())
        return symNameAttr.getValue().str();
      return ("memory_" + std::to_string(autoMemCount++));
    }
    if (auto fifoOp = mlir::dyn_cast<fcc::fabric::FifoOp>(op)) {
      if (auto symNameAttr = fifoOp.getSymNameAttr())
        return symNameAttr.getValue().str();
      return ("fifo_" + std::to_string(autoFifoCount++));
    }
    if (auto tswOp = mlir::dyn_cast<fcc::fabric::TemporalSwOp>(op)) {
      if (auto symNameAttr = tswOp.getSymNameAttr())
        return symNameAttr.getValue().str();
      return ("temporal_sw_" + std::to_string(autoTemporalSwCount++));
    }
    if (mlir::isa<fcc::fabric::AddTagOp>(op))
      return ("add_tag_" + std::to_string(autoAddTagCount++));
    if (mlir::isa<fcc::fabric::DelTagOp>(op))
      return ("del_tag_" + std::to_string(autoDelTagCount++));
    if (mlir::isa<fcc::fabric::MapTagOp>(op))
      return ("map_tag_" + std::to_string(autoMapTagCount++));
    return op.getName().getStringRef().str();
  };

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

    auto instantiateFU = [&](fcc::fabric::FunctionUnitOp fuOp,
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
    };

    auto &peBody = peOp.getBody().front();
    auto referencedIt = referencedTargetsByBlock.find(&peBody);
    const llvm::DenseSet<llvm::StringRef> *referencedTargets =
        referencedIt != referencedTargetsByBlock.end() ? &referencedIt->second
                                                       : nullptr;
    for (auto &innerOp : peBody.getOperations()) {
      if (auto fuOp = mlir::dyn_cast<fcc::fabric::FunctionUnitOp>(innerOp)) {
        llvm::StringRef symName = fuOp.getSymNameAttr().getValue();
        if (!symName.empty() && referencedTargets &&
            referencedTargets->contains(symName))
          continue;
        instantiateFU(fuOp, symName);
        continue;
      }

      auto instOp = mlir::dyn_cast<fcc::fabric::InstanceOp>(innerOp);
      if (!instOp)
        continue;
      auto fuIt = functionUnitDefMap.find(instOp.getModule());
      if (fuIt == functionUnitDefMap.end())
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
    if (auto instOp = mlir::dyn_cast<fcc::fabric::InstanceOp>(op)) {
      llvm::StringRef moduleName = instOp.getModule();
      std::string instanceName;
      if (auto symName = instOp.getSymName())
        instanceName = symName->str();

      // Check if this instance references a PE definition.
      auto peIt = peDefMap.find(moduleName);
      if (peIt != peDefMap.end()) {
        createFUNodesFromPE(peIt->second, instanceName, "spatial_pe", 0, 0, 0,
                            0, false, 0);
        continue;
      }

      auto temporalPeIt = temporalPeDefMap.find(moduleName);
      if (temporalPeIt != temporalPeDefMap.end()) {
        auto temporalFnType = temporalPeIt->second.getFunctionType();
        unsigned tagWidth = 0;
        if (temporalFnType.getNumInputs() > 0) {
          if (auto tagged =
                  mlir::dyn_cast<fcc::fabric::TaggedType>(temporalFnType.getInput(0))) {
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
      auto swIt = swDefMap.find(moduleName);
      if (swIt != swDefMap.end()) {
        auto swDef = swIt->second;
        auto swNode = std::make_unique<Node>();
        swNode->kind = Node::OperationNode;

        setNodeAttr(swNode.get(), "op_name",
                    mlir::StringAttr::get(ctx, instanceName), ctx);
        setNodeAttr(swNode.get(), "op_kind",
                    mlir::StringAttr::get(ctx, "spatial_sw"), ctx);
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
        opToNodeId[instOp.getOperation()] = swNodeId;
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

      auto tswIt = temporalSwDefMap.find(moduleName);
      if (tswIt != temporalSwDefMap.end()) {
        auto tswDef = tswIt->second;
        auto tswNode = std::make_unique<Node>();
        tswNode->kind = Node::OperationNode;

        if (instanceName.empty())
          instanceName = getOrCreateOpName(op);

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
        opToNodeId[instOp.getOperation()] = tswNodeId;
        nodeGridPos[tswNodeId] = gridPos;

        auto *node = adg.getNode(tswNodeId);
        for (IdIndex ip : node->inputPorts) {
          for (IdIndex op : node->outputPorts)
            connectivity.inToOut[ip].push_back(op);
        }

        for (unsigned i = 0; i < instOp.getNumResults(); ++i)
          valueToOutputPort[instOp.getResult(i)] = node->outputPorts[i];
        continue;
      }

      auto extIt = extMemoryDefMap.find(moduleName);
      if (extIt != extMemoryDefMap.end()) {
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
        opToNodeId[instOp.getOperation()] = memNodeId;
        nodeGridPos[memNodeId] = parseGridPos(instanceName);
        auto *node = adg.getNode(memNodeId);
        for (IdIndex ip : node->inputPorts)
          for (IdIndex opPort : node->outputPorts)
            connectivity.inToOut[ip].push_back(opPort);
        for (unsigned i = 0; i < instOp.getNumResults(); ++i)
          valueToOutputPort[instOp.getResult(i)] = node->outputPorts[i];
        continue;
      }

      auto memIt = memoryDefMap.find(moduleName);
      if (memIt != memoryDefMap.end()) {
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
        opToNodeId[instOp.getOperation()] = memNodeId;
        nodeGridPos[memNodeId] = parseGridPos(instanceName);
        auto *node = adg.getNode(memNodeId);
        for (IdIndex ip : node->inputPorts)
          for (IdIndex opPort : node->outputPorts)
            connectivity.inToOut[ip].push_back(opPort);
        for (unsigned i = 0; i < instOp.getNumResults(); ++i)
          valueToOutputPort[instOp.getResult(i)] = node->outputPorts[i];
        continue;
      }

      auto fifoIt = fifoDefMap.find(moduleName);
      if (fifoIt != fifoDefMap.end()) {
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
                    mlir::StringAttr::get(ctx, "buffer"), ctx);

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
        opToNodeId[instOp.getOperation()] = fifoNodeId;
        nodeGridPos[fifoNodeId] = parseGridPos(instanceName);
        auto *node = adg.getNode(fifoNodeId);
        for (IdIndex ip : node->inputPorts)
          for (IdIndex opPort : node->outputPorts)
            connectivity.inToOut[ip].push_back(opPort);
        for (unsigned i = 0; i < instOp.getNumResults(); ++i)
          valueToOutputPort[instOp.getResult(i)] = node->outputPorts[i];
        continue;
      }

      // Unknown instance type -- skip.
      continue;
    }

    if (auto peOp = mlir::dyn_cast<fcc::fabric::SpatialPEOp>(op)) {
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

    if (auto peOp = mlir::dyn_cast<fcc::fabric::TemporalPEOp>(op)) {
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
                mlir::dyn_cast<fcc::fabric::TaggedType>(temporalFnType.getInput(0))) {
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

    if (auto swOp = mlir::dyn_cast<fcc::fabric::SpatialSwOp>(op)) {
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
      opToNodeId[swOp.getOperation()] = swNodeId;
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

    if (auto tswOp = mlir::dyn_cast<fcc::fabric::TemporalSwOp>(op)) {
      if (!tswOp->hasAttr("inline_instantiation"))
        continue;
      llvm::StringRef symName;
      if (auto symNameAttr = tswOp.getSymNameAttr())
        symName = symNameAttr.getValue();
      auto tswNode = std::make_unique<Node>();
      tswNode->kind = Node::OperationNode;

      std::string tswName = getOrCreateOpName(op);
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
      opToNodeId[tswOp.getOperation()] = tswNodeId;
      nodeGridPos[tswNodeId] = gridPos;

      auto *node = adg.getNode(tswNodeId);
      for (IdIndex ip : node->inputPorts) {
        for (IdIndex op : node->outputPorts)
          connectivity.inToOut[ip].push_back(op);
      }
      for (unsigned i = 0; i < tswOp.getNumResults(); ++i)
        valueToOutputPort[tswOp.getResult(i)] = node->outputPorts[i];
      continue;
    }

    if (auto extOp = mlir::dyn_cast<fcc::fabric::ExtMemoryOp>(op)) {
      if (!extOp->hasAttr("inline_instantiation"))
        continue;
      llvm::StringRef symName;
      if (auto symNameAttr = extOp.getSymNameAttr())
        symName = symNameAttr.getValue();
      auto memNode = std::make_unique<Node>();
      memNode->kind = Node::OperationNode;

      std::string memName = getOrCreateOpName(op);

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
      opToNodeId[extOp.getOperation()] = memNodeId;

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

    if (auto memOp = mlir::dyn_cast<fcc::fabric::MemoryOp>(op)) {
      if (!memOp->hasAttr("inline_instantiation"))
        continue;
      llvm::StringRef symName;
      if (auto symNameAttr = memOp.getSymNameAttr())
        symName = symNameAttr.getValue();
      auto memNode = std::make_unique<Node>();
      memNode->kind = Node::OperationNode;

      std::string memName = getOrCreateOpName(op);
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
      opToNodeId[memOp.getOperation()] = memNodeId;
      auto *node = adg.getNode(memNodeId);
      for (IdIndex ip : node->inputPorts) {
        for (IdIndex op : node->outputPorts)
          connectivity.inToOut[ip].push_back(op);
      }
      for (unsigned i = 0; i < memOp.getNumResults(); ++i)
        valueToOutputPort[memOp.getResult(i)] = node->outputPorts[i];
      continue;
    }

    if (auto fifoOp = mlir::dyn_cast<fcc::fabric::FifoOp>(op)) {
      if (!fifoOp->hasAttr("inline_instantiation"))
        continue;
      llvm::StringRef symName;
      if (auto symNameAttr = fifoOp.getSymNameAttr())
        symName = symNameAttr.getValue();
      auto fifoNode = std::make_unique<Node>();
      fifoNode->kind = Node::OperationNode;

      std::string fifoName = getOrCreateOpName(op);

      setNodeAttr(fifoNode.get(), "op_name",
                  mlir::StringAttr::get(ctx, fifoName), ctx);
      setNodeAttr(fifoNode.get(), "op_kind",
                  mlir::StringAttr::get(ctx, "fifo"), ctx);
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
      opToNodeId[fifoOp.getOperation()] = fifoNodeId;

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

    if (auto addTagOp = mlir::dyn_cast<fcc::fabric::AddTagOp>(op)) {
      auto routeNode = std::make_unique<Node>();
      routeNode->kind = Node::OperationNode;
      std::string nodeName = getOrCreateOpName(op);

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
      opToNodeId[addTagOp.getOperation()] = nodeId;
      connectivity.inToOut[inPortId].push_back(outPortId);
      valueToOutputPort[addTagOp.getResult()] = outPortId;
      continue;
    }

    if (auto delTagOp = mlir::dyn_cast<fcc::fabric::DelTagOp>(op)) {
      auto routeNode = std::make_unique<Node>();
      routeNode->kind = Node::OperationNode;
      std::string nodeName = getOrCreateOpName(op);

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
      opToNodeId[delTagOp.getOperation()] = nodeId;
      connectivity.inToOut[inPortId].push_back(outPortId);
      valueToOutputPort[delTagOp.getResult()] = outPortId;
      continue;
    }

    if (auto mapTagOp = mlir::dyn_cast<fcc::fabric::MapTagOp>(op)) {
      auto routeNode = std::make_unique<Node>();
      routeNode->kind = Node::OperationNode;
      std::string nodeName = getOrCreateOpName(op);

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
      opToNodeId[mapTagOp.getOperation()] = nodeId;
      connectivity.inToOut[inPortId].push_back(outPortId);
      valueToOutputPort[mapTagOp.getResult()] = outPortId;
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
  // This handles both direct PE ops and InstanceOp referencing PE defs.
  struct PEInfo {
    mlir::Operation *op;
    std::vector<IdIndex> fuNodeIds;
  };

  std::vector<PEInfo> peInfos;
  size_t peIdx = 0;
  for (auto &op : body.getOperations()) {
    bool isPE = false;
    if (auto peOp = mlir::dyn_cast<fcc::fabric::SpatialPEOp>(op)) {
      isPE = peOp->hasAttr("inline_instantiation");
    } else if (auto peOp = mlir::dyn_cast<fcc::fabric::TemporalPEOp>(op)) {
      isPE = peOp->hasAttr("inline_instantiation");
    } else if (auto instOp = mlir::dyn_cast<fcc::fabric::InstanceOp>(op)) {
      isPE = peDefMap.count(instOp.getModule()) > 0 ||
             temporalPeDefMap.count(instOp.getModule()) > 0;
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
  struct SourceBinding {
    IdIndex portId = INVALID_ID;
    int peOutputIndex = -1;
  };
  llvm::DenseMap<mlir::Value, llvm::SmallVector<SourceBinding, 4>> valueSrcPorts;

  // Non-PE ops: already mapped in valueToOutputPort.
  for (auto &kv : valueToOutputPort) {
    valueSrcPorts[kv.first].push_back({kv.second, -1});
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
          valueSrcPorts[val].push_back({op, static_cast<int>(r)});
        }
      }
    }
  }

  // Count total FU nodes.
  IdIndex totalFuNodes = 0;
  for (auto &pe : peContainment)
    totalFuNodes += pe.fuNodeIds.size();

  // Wire all single-node operations and instances using the explicit
  // opToNodeId map populated during pass 1.
  for (auto &op : body.getOperations()) {
    auto it = opToNodeId.find(&op);
    if (it == opToNodeId.end())
      continue;
    auto *node = adg.getNode(it->second);
    if (!node)
      continue;

    for (unsigned j = 0; j < op.getNumOperands(); ++j) {
      if (j >= node->inputPorts.size())
        break;
      auto srcIt = valueSrcPorts.find(op.getOperand(j));
      if (srcIt == valueSrcPorts.end())
        continue;

      IdIndex dstPortId = node->inputPorts[j];
      for (const SourceBinding &binding : srcIt->second) {
        IdIndex srcPortId = binding.portId;
        auto edge = std::make_unique<Edge>();
        edge->srcPort = srcPortId;
        edge->dstPort = dstPortId;
        if (binding.peOutputIndex >= 0) {
          setEdgeAttr(edge.get(), "pe_output_index",
                      mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 32),
                                             binding.peOutputIndex),
                      ctx);
        }
        IdIndex edgeId = adg.addEdge(std::move(edge));
        adg.ports[srcPortId]->connectedEdges.push_back(edgeId);
        adg.ports[dstPortId]->connectedEdges.push_back(edgeId);
        connectivity.outToIn[srcPortId].push_back(dstPortId);
      }
    }
  }

  // Helper: create edges from a set of source ports to a set of dest ports.
  auto createEdgesBetweenPorts =
      [&](llvm::ArrayRef<SourceBinding> srcPorts, llvm::ArrayRef<IdIndex> dstPorts,
          int peInputIndex) {
        for (const SourceBinding &binding : srcPorts) {
          IdIndex srcPortId = binding.portId;
          for (IdIndex dstPortId : dstPorts) {
            auto edge = std::make_unique<Edge>();
            edge->srcPort = srcPortId;
            edge->dstPort = dstPortId;
            if (binding.peOutputIndex >= 0) {
              setEdgeAttr(edge.get(), "pe_output_index",
                          mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 32),
                                                 binding.peOutputIndex),
                          ctx);
            }
            if (peInputIndex >= 0) {
              setEdgeAttr(edge.get(), "pe_input_index",
                          mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 32),
                                                 peInputIndex),
                          ctx);
            }
            IdIndex edgeId = adg.addEdge(std::move(edge));
            adg.ports[srcPortId]->connectedEdges.push_back(edgeId);
            adg.ports[dstPortId]->connectedEdges.push_back(edgeId);
            connectivity.outToIn[srcPortId].push_back(dstPortId);
          }
        }
      };

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
        createEdgesBetweenPorts(srcIt->second, fuNode->inputPorts,
                                static_cast<int>(j));
      }
    }
  }

  // Legacy fallback: wire SW output ports -> ExtMem input ports from metadata.
  // Newer ADGs spell these edges with real SSA operands on fabric.extmemory,
  // so this block only runs when the data input ports still have no producer.
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

      bool hasStructuredInputs = false;
      for (unsigned inIdx = 1; inIdx < n->inputPorts.size(); ++inIdx) {
        const Port *inPort = adg.getPort(n->inputPorts[inIdx]);
        if (inPort && !inPort->connectedEdges.empty()) {
          hasStructuredInputs = true;
          break;
        }
      }
      if (hasStructuredInputs)
        continue;

      auto connectSwitchToMemory = [&](llvm::StringRef swName,
                                       unsigned swOutputBase) {
        auto swIt = swNameToNodeId.find(swName);
        if (swIt == swNameToNodeId.end())
          return;
        IdIndex swNodeId = swIt->second;
        auto *swNode = adg.getNode(swNodeId);
        if (!swNode)
          return;

        unsigned numExtMemDataInputs = n->inputPorts.size() > 1
                                           ? n->inputPorts.size() - 1
                                           : 0;
        for (unsigned p = 0; p < numExtMemDataInputs; ++p) {
          unsigned swOutIdx = swOutputBase + p;
          unsigned extMemInIdx = 1 + p; // Skip memref port at 0.
          if (swOutIdx >= swNode->outputPorts.size() ||
              extMemInIdx >= n->inputPorts.size())
            break;

          IdIndex srcPortId = swNode->outputPorts[swOutIdx];
          IdIndex dstPortId = n->inputPorts[extMemInIdx];
          connectivity.outToIn[srcPortId].push_back(dstPortId);

          auto edge = std::make_unique<Edge>();
          edge->srcPort = srcPortId;
          edge->dstPort = dstPortId;
          IdIndex edgeId = adg.addEdge(std::move(edge));
          adg.ports[srcPortId]->connectedEdges.push_back(edgeId);
          adg.ports[dstPortId]->connectedEdges.push_back(edgeId);
        }
      };

      bool usedDetailedMetadata = false;
      for (auto &attr : n->attributes) {
        if (attr.getName() != "connected_sw_detail")
          continue;
        auto arrayAttr = mlir::dyn_cast<mlir::ArrayAttr>(attr.getValue());
        if (!arrayAttr)
          continue;
        usedDetailedMetadata = true;
        for (auto elem : arrayAttr) {
          auto dictAttr = mlir::dyn_cast<mlir::DictionaryAttr>(elem);
          if (!dictAttr)
            continue;
          auto nameAttr = dictAttr.getAs<mlir::StringAttr>("name");
          if (!nameAttr)
            continue;
          unsigned swOutputBase = 0;
          if (auto outBaseAttr =
                  dictAttr.getAs<mlir::IntegerAttr>("output_port_base")) {
            swOutputBase = static_cast<unsigned>(outBaseAttr.getInt());
          }
          connectSwitchToMemory(nameAttr.getValue(), swOutputBase);
        }
      }

      if (usedDetailedMetadata)
        continue;

      // Legacy fallback for older ADGs that only recorded switch names.
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
          connectSwitchToMemory(strAttr.getValue(), /*swOutputBase=*/4);
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
        for (const SourceBinding &binding : srcIt->second) {
          IdIndex srcPortId = binding.portId;
          auto edge = std::make_unique<Edge>();
          edge->srcPort = srcPortId;
          edge->dstPort = portId;
          if (binding.peOutputIndex >= 0) {
            setEdgeAttr(edge.get(), "pe_output_index",
                        mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 32),
                                               binding.peOutputIndex),
                        ctx);
          }
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

  {
    mlir::Builder builder(ctx);

    auto findFeedingPort = [&](IdIndex portId) -> IdIndex {
      const Port *p = adg.getPort(portId);
      if (!p)
        return INVALID_ID;
      for (IdIndex edgeId : p->connectedEdges) {
        const Edge *e = adg.getEdge(edgeId);
        if (e && e->dstPort == portId)
          return e->srcPort;
      }
      return INVALID_ID;
    };

    auto findConsumingPort = [&](IdIndex portId) -> IdIndex {
      const Port *p = adg.getPort(portId);
      if (!p)
        return INVALID_ID;
      for (IdIndex edgeId : p->connectedEdges) {
        const Edge *e = adg.getEdge(edgeId);
        if (e && e->srcPort == portId)
          return e->dstPort;
      }
      return INVALID_ID;
    };

    auto findConsumingPorts = [&](IdIndex portId)
        -> llvm::SmallVector<IdIndex, 4> {
      llvm::SmallVector<IdIndex, 4> consumers;
      const Port *p = adg.getPort(portId);
      if (!p)
        return consumers;
      for (IdIndex edgeId : p->connectedEdges) {
        const Edge *e = adg.getEdge(edgeId);
        if (e && e->srcPort == portId)
          consumers.push_back(e->dstPort);
      }
      return consumers;
    };

    auto getPortOwnerKind = [&](IdIndex portId) -> llvm::StringRef {
      const Port *p = adg.getPort(portId);
      if (!p)
        return "";
      const Node *n = adg.getNode(p->parentNode);
      if (!n)
        return "";
      return getNodeAttrStr(n, "op_kind");
    };

    for (IdIndex nodeId = 0; nodeId < static_cast<IdIndex>(adg.nodes.size());
         ++nodeId) {
      Node *node = adg.getNode(nodeId);
      if (!node || node->kind != Node::OperationNode)
        continue;
      if (getNodeAttrStr(node, "resource_class") != "memory")
        continue;

      unsigned ldCount =
          static_cast<unsigned>(getNodeAttrInt(node, "ldCount", 0));
      unsigned stCount =
          static_cast<unsigned>(getNodeAttrInt(node, "stCount", 0));
      bool isExtMem = getNodeAttrStr(node, "op_kind") == "extmemory";

      if (ldCount <= 1 && stCount <= 1)
        continue;

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

        llvm::DenseSet<IdIndex> seenOutputs;
        llvm::DenseSet<IdIndex> seenBoundaryPorts;
        llvm::DenseSet<IdIndex> seenRouteNodes;
        llvm::SmallVector<std::pair<IdIndex, IdIndex>, 4> boundaryPairs;

        std::function<bool(IdIndex)> visitOutputPort = [&](IdIndex outPortId) {
          if (outPortId == INVALID_ID || !seenOutputs.insert(outPortId).second)
            return false;
          const Port *outPort = adg.getPort(outPortId);
          if (!outPort || outPort->direction != Port::Output)
            return false;
          const Node *owner = adg.getNode(outPort->parentNode);
          if (!owner)
            return false;

          llvm::StringRef kind = getNodeAttrStr(owner, "op_kind");
          if (kind == "add_tag") {
            if (owner->inputPorts.empty())
              return false;
            IdIndex boundaryPort = owner->inputPorts[0];
            bool inserted = seenBoundaryPorts.insert(boundaryPort).second;
            if (inserted)
              boundaryPairs.push_back({boundaryPort, outPort->parentNode});
            return inserted;
          }

          if (kind == "map_tag" || kind == "del_tag" || kind == "fifo") {
            if (owner->inputPorts.empty())
              return false;
            return visitOutputPort(findFeedingPort(owner->inputPorts[0]));
          }

          if (kind == "spatial_sw" || kind == "temporal_sw") {
            if (seenRouteNodes.insert(outPort->parentNode).second &&
                muxNodeId == INVALID_ID) {
              muxNodeId = outPort->parentNode;
            }
            bool foundBoundary = false;
            for (IdIndex inPortId : owner->inputPorts) {
              IdIndex upstreamOutPortId = findFeedingPort(inPortId);
              if (visitOutputPort(upstreamOutPortId)) {
                foundBoundary = true;
                continue;
              }
              const Port *inPort = adg.getPort(inPortId);
              if (inPort && mlir::isa<fcc::fabric::TaggedType>(inPort->type) &&
                  seenBoundaryPorts.insert(inPortId).second) {
                boundaryPairs.push_back({inPortId, INVALID_ID});
                foundBoundary = true;
              }
            }
            return foundBoundary;
          }

          if (mlir::isa<fcc::fabric::TaggedType>(outPort->type)) {
            bool inserted = seenBoundaryPorts.insert(outPortId).second;
            if (inserted)
              boundaryPairs.push_back({outPortId, INVALID_ID});
            return inserted;
          }
          return false;
        };

        visitOutputPort(srcPortId);
        llvm::sort(boundaryPairs,
                   [](const auto &lhs, const auto &rhs) {
                     return lhs.first < rhs.first;
                   });
        if (boundaryPairs.size() > laneCount)
          boundaryPairs.resize(laneCount);
        for (const auto &entry : boundaryPairs) {
          boundary.push_back(entry.first);
          addTagNodes.push_back(entry.second);
        }
        return boundary;
      };

      auto traceOutputBridge =
          [&](unsigned memOutputPortIdx, unsigned laneCount,
              IdIndex &demuxNodeId) -> llvm::SmallVector<IdIndex, 4> {
        demuxNodeId = INVALID_ID;
        llvm::SmallVector<IdIndex, 4> boundary;
        if (memOutputPortIdx >= node->outputPorts.size())
          return boundary;

        IdIndex memOutPortId = node->outputPorts[memOutputPortIdx];
        llvm::DenseSet<IdIndex> seenInputs;
        llvm::DenseSet<IdIndex> seenBoundaryPorts;
        llvm::DenseSet<IdIndex> seenRouteNodes;

        std::function<bool(IdIndex)> visitInputPort = [&](IdIndex inPortId) {
          if (inPortId == INVALID_ID || !seenInputs.insert(inPortId).second)
            return false;
          const Port *inPort = adg.getPort(inPortId);
          if (!inPort || inPort->direction != Port::Input)
            return false;
          const Node *owner = adg.getNode(inPort->parentNode);
          if (!owner)
            return false;

          llvm::StringRef kind = getNodeAttrStr(owner, "op_kind");
          if (kind == "del_tag") {
            if (!owner->outputPorts.empty()) {
              IdIndex boundaryPort = owner->outputPorts[0];
              bool inserted = seenBoundaryPorts.insert(boundaryPort).second;
              if (inserted)
                boundary.push_back(boundaryPort);
              return inserted;
            }
            return false;
          }

          if (kind == "add_tag" || kind == "map_tag" || kind == "fifo") {
            bool foundBoundary = false;
            for (IdIndex outPortId : owner->outputPorts) {
              for (IdIndex nextInPortId : findConsumingPorts(outPortId))
                foundBoundary |= visitInputPort(nextInPortId);
            }
            return foundBoundary;
          }

          if (kind == "spatial_sw" || kind == "temporal_sw") {
            if (seenRouteNodes.insert(inPort->parentNode).second &&
                demuxNodeId == INVALID_ID) {
              demuxNodeId = inPort->parentNode;
            }
            bool foundBoundary = false;
            for (IdIndex outPortId : owner->outputPorts) {
              bool branchFound = false;
              for (IdIndex nextInPortId : findConsumingPorts(outPortId))
                branchFound |= visitInputPort(nextInPortId);
              if (branchFound) {
                foundBoundary = true;
                continue;
              }
              const Port *outPort = adg.getPort(outPortId);
              if (outPort &&
                  mlir::isa<fcc::fabric::TaggedType>(outPort->type) &&
                  seenBoundaryPorts.insert(outPortId).second) {
                boundary.push_back(outPortId);
                foundBoundary = true;
              }
            }
            return foundBoundary;
          }

          if (mlir::isa<fcc::fabric::TaggedType>(inPort->type)) {
            bool inserted = seenBoundaryPorts.insert(inPortId).second;
            if (inserted)
              boundary.push_back(inPortId);
            return inserted;
          }
          return false;
        };

        for (IdIndex dstPortId : findConsumingPorts(memOutPortId))
          visitInputPort(dstPortId);

        llvm::sort(boundary);
        if (boundary.size() > laneCount)
          boundary.resize(laneCount);
        return boundary;
      };

      llvm::SmallVector<IdIndex, 8> bridgeInputPorts;
      llvm::SmallVector<BridgePortCategory, 8> bridgeInputCats;
      llvm::SmallVector<unsigned, 8> bridgeInputLanes;
      llvm::SmallVector<IdIndex, 8> allAddTagNodes;
      llvm::SmallVector<IdIndex, 4> muxNodes;
      unsigned memInIdx = isExtMem ? 1 : 0;

      llvm::SmallVector<IdIndex, 4> ldAddrBoundary, stAddrBoundary,
          stDataBoundary;
      llvm::SmallVector<IdIndex, 4> ldAddrNodes, stAddrNodes, stDataNodes;
      if (ldCount > 0) {
        IdIndex muxId;
        ldAddrBoundary =
            traceInputBridge(memInIdx++, ldCount, ldAddrNodes, muxId);
        if (muxId != INVALID_ID)
          muxNodes.push_back(muxId);
      }
      if (stCount > 0) {
        IdIndex muxId;
        stAddrBoundary =
            traceInputBridge(memInIdx++, stCount, stAddrNodes, muxId);
        if (muxId != INVALID_ID)
          muxNodes.push_back(muxId);
        stDataBoundary =
            traceInputBridge(memInIdx++, stCount, stDataNodes, muxId);
        if (muxId != INVALID_ID)
          muxNodes.push_back(muxId);

        for (unsigned lane = 0; lane < stCount; ++lane) {
          if (lane < stDataBoundary.size()) {
            bridgeInputPorts.push_back(stDataBoundary[lane]);
            bridgeInputCats.push_back(BridgePortCategory::StData);
            bridgeInputLanes.push_back(lane);
          }
          if (lane < stAddrBoundary.size()) {
            bridgeInputPorts.push_back(stAddrBoundary[lane]);
            bridgeInputCats.push_back(BridgePortCategory::StAddr);
            bridgeInputLanes.push_back(lane);
          }
          if (lane < stDataNodes.size())
            allAddTagNodes.push_back(stDataNodes[lane]);
          if (lane < stAddrNodes.size())
            allAddTagNodes.push_back(stAddrNodes[lane]);
        }
      }
      for (unsigned lane = 0; lane < ldCount && lane < ldAddrBoundary.size();
           ++lane) {
        bridgeInputPorts.push_back(ldAddrBoundary[lane]);
        bridgeInputCats.push_back(BridgePortCategory::LdAddr);
        bridgeInputLanes.push_back(lane);
      }
      for (unsigned lane = 0; lane < ldCount && lane < ldAddrNodes.size();
           ++lane)
        allAddTagNodes.push_back(ldAddrNodes[lane]);

      llvm::SmallVector<IdIndex, 8> bridgeOutputPorts;
      llvm::SmallVector<BridgePortCategory, 8> bridgeOutputCats;
      llvm::SmallVector<unsigned, 8> bridgeOutputLanes;
      llvm::SmallVector<IdIndex, 4> demuxNodes;
      unsigned memOutIdx = 0;

      if (ldCount > 0) {
        IdIndex demuxId;
        auto ldDataBoundary =
            traceOutputBridge(memOutIdx++, ldCount, demuxId);
        if (demuxId != INVALID_ID)
          demuxNodes.push_back(demuxId);
        for (unsigned lane = 0; lane < ldCount && lane < ldDataBoundary.size();
             ++lane) {
          bridgeOutputPorts.push_back(ldDataBoundary[lane]);
          bridgeOutputCats.push_back(BridgePortCategory::LdData);
          bridgeOutputLanes.push_back(lane);
        }

        auto ldDoneBoundary =
            traceOutputBridge(memOutIdx++, ldCount, demuxId);
        if (demuxId != INVALID_ID)
          demuxNodes.push_back(demuxId);
        for (unsigned lane = 0; lane < ldCount && lane < ldDoneBoundary.size();
             ++lane) {
          bridgeOutputPorts.push_back(ldDoneBoundary[lane]);
          bridgeOutputCats.push_back(BridgePortCategory::LdDone);
          bridgeOutputLanes.push_back(lane);
        }
      }
      if (stCount > 0) {
        IdIndex demuxId;
        auto stDoneBoundary =
            traceOutputBridge(memOutIdx++, stCount, demuxId);
        if (demuxId != INVALID_ID)
          demuxNodes.push_back(demuxId);
        for (unsigned lane = 0; lane < stCount && lane < stDoneBoundary.size();
             ++lane) {
          bridgeOutputPorts.push_back(stDoneBoundary[lane]);
          bridgeOutputCats.push_back(BridgePortCategory::StDone);
          bridgeOutputLanes.push_back(lane);
        }
      }

      if (bridgeInputPorts.empty() && bridgeOutputPorts.empty()) {
        continue;
      }

      {
        unsigned idx = 0;
        for (unsigned lane = 0; lane < stCount && idx < allAddTagNodes.size();
             ++lane) {
          if (idx < allAddTagNodes.size()) {
            Node *atNode = adg.getNode(allAddTagNodes[idx++]);
            if (atNode) {
              atNode->attributes.push_back(builder.getNamedAttr(
                  "bridge_lane_index", builder.getI32IntegerAttr(lane)));
            }
          }
          if (idx < allAddTagNodes.size()) {
            Node *atNode = adg.getNode(allAddTagNodes[idx++]);
            if (atNode) {
              atNode->attributes.push_back(builder.getNamedAttr(
                  "bridge_lane_index", builder.getI32IntegerAttr(lane)));
            }
          }
        }
        for (unsigned lane = 0; lane < ldCount && idx < allAddTagNodes.size();
             ++lane) {
          Node *atNode = adg.getNode(allAddTagNodes[idx++]);
          if (atNode) {
            atNode->attributes.push_back(builder.getNamedAttr(
                "bridge_lane_index", builder.getI32IntegerAttr(lane)));
          }
        }
      }

      llvm::SmallVector<int32_t> inPorts32(bridgeInputPorts.begin(),
                                           bridgeInputPorts.end());
      llvm::SmallVector<int32_t> inCats32;
      llvm::SmallVector<int32_t> inLanes32;
      for (auto cat : bridgeInputCats)
        inCats32.push_back(static_cast<int32_t>(cat));
      for (unsigned lane : bridgeInputLanes)
        inLanes32.push_back(static_cast<int32_t>(lane));
      node->attributes.push_back(builder.getNamedAttr(
          "bridge_input_ports",
          mlir::DenseI32ArrayAttr::get(builder.getContext(), inPorts32)));
      node->attributes.push_back(builder.getNamedAttr(
          "bridge_input_categories",
          mlir::DenseI32ArrayAttr::get(builder.getContext(), inCats32)));
      node->attributes.push_back(builder.getNamedAttr(
          "bridge_input_lanes",
          mlir::DenseI32ArrayAttr::get(builder.getContext(), inLanes32)));

      llvm::SmallVector<int32_t> outPorts32(bridgeOutputPorts.begin(),
                                            bridgeOutputPorts.end());
      llvm::SmallVector<int32_t> outCats32;
      llvm::SmallVector<int32_t> outLanes32;
      for (auto cat : bridgeOutputCats)
        outCats32.push_back(static_cast<int32_t>(cat));
      for (unsigned lane : bridgeOutputLanes)
        outLanes32.push_back(static_cast<int32_t>(lane));
      node->attributes.push_back(builder.getNamedAttr(
          "bridge_output_ports",
          mlir::DenseI32ArrayAttr::get(builder.getContext(), outPorts32)));
      node->attributes.push_back(builder.getNamedAttr(
          "bridge_output_categories",
          mlir::DenseI32ArrayAttr::get(builder.getContext(), outCats32)));
      node->attributes.push_back(builder.getNamedAttr(
          "bridge_output_lanes",
          mlir::DenseI32ArrayAttr::get(builder.getContext(), outLanes32)));

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
  }

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
