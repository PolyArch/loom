#include "SVGenInternal.h"

#include "fcc/SVGen/SVEmitter.h"
#include "fcc/SVGen/SVModuleRegistry.h"

#include "fcc/Dialect/Fabric/FabricOps.h"
#include "fcc/Dialect/Fabric/FabricTypes.h"
#include "fcc/Mapper/TypeCompat.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Value.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/raw_ostream.h"

#include <string>
#include <vector>

namespace fcc {
namespace svgen {

namespace {

/// Context for top-level module emission.
struct TopEmitContext {
  SVEmitter &emitter;
  SVModuleRegistry &registry;

  /// Map SSA value -> wire name in the top module.
  llvm::DenseMap<mlir::Value, std::string> wireNames;
  unsigned nextWireIdx = 0;

  /// Map from operations to their generated SV module names.
  llvm::DenseMap<mlir::Operation *, std::string> opModuleNames;

  /// Counter for generating unique instance names per op type.
  llvm::StringMap<unsigned> instanceCounters;

  /// Ordered list of instance names that have config ports, matching the
  /// order of computeConfigLayout() slices.
  std::vector<std::string> configInstanceNames;

  std::string getOrCreateWire(mlir::Value val) {
    auto it = wireNames.find(val);
    if (it != wireNames.end())
      return it->second;
    std::string name = "net_" + std::to_string(nextWireIdx++);
    wireNames[val] = name;
    return name;
  }

  std::string makeInstanceName(llvm::StringRef baseName) {
    unsigned idx = instanceCounters[baseName]++;
    return (baseName + "_" + llvm::Twine(idx)).str();
  }
};

/// Get an op's SV module name and generate an instance for it.
static std::string getModuleName(mlir::Operation &op) {
  if (auto spatialSw = mlir::dyn_cast<fcc::fabric::SpatialSwOp>(op)) {
    return "fabric_spatial_sw";
  }
  if (auto temporalSw = mlir::dyn_cast<fcc::fabric::TemporalSwOp>(op)) {
    return "fabric_temporal_sw";
  }
  if (mlir::isa<fcc::fabric::AddTagOp>(op))
    return "fabric_add_tag";
  if (mlir::isa<fcc::fabric::DelTagOp>(op))
    return "fabric_del_tag";
  if (mlir::isa<fcc::fabric::MapTagOp>(op))
    return "fabric_map_tag";
  if (auto fifo = mlir::dyn_cast<fcc::fabric::FifoOp>(op))
    return "fabric_fifo";
  if (auto memOp = mlir::dyn_cast<fcc::fabric::MemoryOp>(op))
    return "fabric_memory";
  if (auto extMem = mlir::dyn_cast<fcc::fabric::ExtMemoryOp>(op))
    return "fabric_extmemory";
  return "";
}

/// Get the instance name hint from an op (sym_name if available).
static std::string getInstanceHint(mlir::Operation &op) {
  if (auto spatialSw = mlir::dyn_cast<fcc::fabric::SpatialSwOp>(op)) {
    if (auto sym = spatialSw.getSymName())
      return SVEmitter::sanitizeName(*sym);
  }
  if (auto temporalSw = mlir::dyn_cast<fcc::fabric::TemporalSwOp>(op)) {
    if (auto sym = temporalSw.getSymName())
      return SVEmitter::sanitizeName(*sym);
  }
  if (auto fifo = mlir::dyn_cast<fcc::fabric::FifoOp>(op)) {
    if (auto sym = fifo.getSymName())
      return SVEmitter::sanitizeName(*sym);
  }
  if (auto memOp = mlir::dyn_cast<fcc::fabric::MemoryOp>(op)) {
    if (auto sym = memOp.getSymName())
      return SVEmitter::sanitizeName(*sym);
  }
  if (auto extMem = mlir::dyn_cast<fcc::fabric::ExtMemoryOp>(op)) {
    if (auto sym = extMem.getSymName())
      return SVEmitter::sanitizeName(*sym);
  }
  return "";
}

/// Emit wire declarations for all SSA results of an op in top scope.
static void emitTopWires(TopEmitContext &ctx, mlir::Operation &op) {
  for (mlir::Value result : op.getResults()) {
    std::string wireName = ctx.getOrCreateWire(result);
    unsigned width = SVEmitter::getTypeWidth(result.getType());
    std::string typeStr = "logic" + SVEmitter::bitRange(width);
    ctx.emitter.emitWire(typeStr, wireName);

    // Also declare valid/ready wires for handshake ports.
    ctx.emitter.emitWire("logic", wireName + "_valid");
    ctx.emitter.emitWire("logic", wireName + "_ready");
  }
}

/// Emit connections for single I/O modules: add_tag, del_tag, map_tag, fifo.
/// These modules use in_data/in_valid/in_ready and out_data/out_valid/out_ready
/// (plus in_tag/out_tag where applicable).
static void emitSingleIOConns(TopEmitContext &ctx, mlir::Operation &op,
                               std::vector<SVConnection> &conns) {
  // Single input operand (skip memrefs).
  for (unsigned i = 0; i < op.getNumOperands(); ++i) {
    if (mlir::isa<mlir::MemRefType>(op.getOperand(i).getType()))
      continue;
    std::string wire = ctx.getOrCreateWire(op.getOperand(i));
    unsigned dw = SVEmitter::getDataWidth(op.getOperand(i).getType());
    unsigned tw = SVEmitter::getTagWidth(op.getOperand(i).getType());
    conns.push_back({"in_data", wire + "[" + std::to_string(dw - 1) + ":0]"});
    conns.push_back({"in_valid", wire + "_valid"});
    conns.push_back({"in_ready", wire + "_ready"});
    if (tw > 0)
      conns.push_back({"in_tag", wire + "[" + std::to_string(dw + tw - 1) +
                                     ":" + std::to_string(dw) + "]"});
    break; // single input
  }

  // Single output result.
  for (unsigned i = 0; i < op.getNumResults(); ++i) {
    std::string wire = ctx.getOrCreateWire(op.getResult(i));
    unsigned outDw = SVEmitter::getDataWidth(op.getResult(i).getType());
    unsigned outTw = SVEmitter::getTagWidth(op.getResult(i).getType());
    conns.push_back({"out_data",
                     wire + "[" + std::to_string(outDw - 1) + ":0]"});
    conns.push_back({"out_valid", wire + "_valid"});
    conns.push_back({"out_ready", wire + "_ready"});
    if (outTw > 0)
      conns.push_back({"out_tag",
                       wire + "[" + std::to_string(outDw + outTw - 1) + ":" +
                           std::to_string(outDw) + "]"});
    break; // single output
  }
}

/// Emit connections for switch modules (spatial_sw, temporal_sw).
/// These use array ports: in_valid[N], in_data[N], in_tag[N], etc.
static void emitSwitchConns(TopEmitContext &ctx, mlir::Operation &op,
                             std::vector<SVConnection> &conns) {
  // Collect non-memref operands for input array ports.
  std::vector<mlir::Value> inputs;
  for (unsigned i = 0; i < op.getNumOperands(); ++i) {
    if (!mlir::isa<mlir::MemRefType>(op.getOperand(i).getType()))
      inputs.push_back(op.getOperand(i));
  }

  // For switch modules, in_valid/in_ready are packed bit vectors and
  // in_data/in_tag are unpacked arrays.  We connect them element-wise.
  for (unsigned i = 0; i < inputs.size(); ++i) {
    std::string wire = ctx.getOrCreateWire(inputs[i]);
    unsigned dw = SVEmitter::getDataWidth(inputs[i].getType());
    unsigned tw = SVEmitter::getTagWidth(inputs[i].getType());
    conns.push_back({"in_valid[" + std::to_string(i) + "]",
                     wire + "_valid"});
    conns.push_back({"in_ready[" + std::to_string(i) + "]",
                     wire + "_ready"});
    conns.push_back({"in_data[" + std::to_string(i) + "]",
                     wire + "[" + std::to_string(dw - 1) + ":0]"});
    if (tw > 0)
      conns.push_back({"in_tag[" + std::to_string(i) + "]",
                       wire + "[" + std::to_string(dw + tw - 1) + ":" +
                           std::to_string(dw) + "]"});
    else
      conns.push_back({"in_tag[" + std::to_string(i) + "]", "'0"});
  }

  for (unsigned i = 0; i < op.getNumResults(); ++i) {
    std::string wire = ctx.getOrCreateWire(op.getResult(i));
    unsigned outDw = SVEmitter::getDataWidth(op.getResult(i).getType());
    unsigned outTw = SVEmitter::getTagWidth(op.getResult(i).getType());
    conns.push_back({"out_valid[" + std::to_string(i) + "]",
                     wire + "_valid"});
    conns.push_back({"out_ready[" + std::to_string(i) + "]",
                     wire + "_ready"});
    conns.push_back({"out_data[" + std::to_string(i) + "]",
                     wire + "[" + std::to_string(outDw - 1) + ":0]"});
    if (outTw > 0)
      conns.push_back({"out_tag[" + std::to_string(i) + "]",
                       wire + "[" + std::to_string(outDw + outTw - 1) + ":" +
                           std::to_string(outDw) + "]"});
    else
      conns.push_back({"out_tag[" + std::to_string(i) + "]", ""});
  }
}

/// Emit connections for memory modules (fabric_memory, fabric_extmemory).
/// These use family-based ports: load_addr_*, store_addr_*, store_data_*,
/// load_data_*, load_done_*, store_done_*.
/// MLIR operand order (non-memref): load_addr, store_addr, store_data.
/// MLIR result order: load_data, load_done, store_done.
static void emitMemoryConns(TopEmitContext &ctx, mlir::Operation &op,
                             std::vector<SVConnection> &conns) {
  // Collect non-memref operands.
  std::vector<mlir::Value> dataOperands;
  for (unsigned i = 0; i < op.getNumOperands(); ++i) {
    if (!mlir::isa<mlir::MemRefType>(op.getOperand(i).getType()))
      dataOperands.push_back(op.getOperand(i));
  }

  // Port family names for inputs in order.
  const char *inPortFamilies[] = {"load_addr", "store_addr", "store_data"};
  for (unsigned i = 0; i < dataOperands.size() && i < 3; ++i) {
    std::string wire = ctx.getOrCreateWire(dataOperands[i]);
    unsigned dw = SVEmitter::getDataWidth(dataOperands[i].getType());
    unsigned tw = SVEmitter::getTagWidth(dataOperands[i].getType());
    std::string family = inPortFamilies[i];
    conns.push_back({family + "_valid", wire + "_valid"});
    conns.push_back({family + "_ready", wire + "_ready"});
    conns.push_back({family + "_data",
                     wire + "[" + std::to_string(dw - 1) + ":0]"});
    if (tw > 0)
      conns.push_back({family + "_tag",
                       wire + "[" + std::to_string(dw + tw - 1) + ":" +
                           std::to_string(dw) + "]"});
    else
      conns.push_back({family + "_tag", "'0"});
  }

  // Port family names for outputs in order.
  // load_data has data+tag; load_done and store_done have tag only.
  const char *outPortFamilies[] = {"load_data", "load_done", "store_done"};
  // load_data has data field; load_done and store_done do not.
  const bool outHasData[] = {true, false, false};
  for (unsigned i = 0; i < op.getNumResults() && i < 3; ++i) {
    std::string wire = ctx.getOrCreateWire(op.getResult(i));
    unsigned outDw = SVEmitter::getDataWidth(op.getResult(i).getType());
    unsigned outTw = SVEmitter::getTagWidth(op.getResult(i).getType());
    std::string family = outPortFamilies[i];
    conns.push_back({family + "_valid", wire + "_valid"});
    conns.push_back({family + "_ready", wire + "_ready"});
    if (outHasData[i]) {
      conns.push_back({family + "_data",
                       wire + "[" + std::to_string(outDw - 1) + ":0]"});
    }
    if (outTw > 0)
      conns.push_back({family + "_tag",
                       wire + "[" + std::to_string(outDw + outTw - 1) + ":" +
                           std::to_string(outDw) + "]"});
    else
      conns.push_back({family + "_tag", ""});
  }
}

/// Emit a pre-written module instance with parameter overrides.
static void emitPrewrittenInstance(TopEmitContext &ctx, mlir::Operation &op,
                                   llvm::StringRef modName,
                                   llvm::StringRef instName) {
  std::vector<std::string> params;
  std::vector<SVConnection> conns;

  // Always connect clk and rst_n first.
  conns.push_back({"clk", "clk"});
  conns.push_back({"rst_n", "rst_n"});

  // Determine data width and tag width from port types.
  unsigned dataWidth = 32;
  unsigned tagWidth = 0;
  if (op.getNumOperands() > 0) {
    for (unsigned i = 0; i < op.getNumOperands(); ++i) {
      if (mlir::isa<mlir::MemRefType>(op.getOperand(i).getType()))
        continue;
      dataWidth = SVEmitter::getDataWidth(op.getOperand(i).getType());
      tagWidth = SVEmitter::getTagWidth(op.getOperand(i).getType());
      break;
    }
  } else if (op.getNumResults() > 0) {
    dataWidth = SVEmitter::getDataWidth(op.getResult(0).getType());
    tagWidth = SVEmitter::getTagWidth(op.getResult(0).getType());
  }

  params.push_back(".DATA_WIDTH(" + std::to_string(dataWidth) + ")");
  if (tagWidth > 0)
    params.push_back(".TAG_WIDTH(" + std::to_string(tagWidth) + ")");

  // Op-specific parameters and connections.
  bool isSwitch = false;
  bool isMemory = false;

  if (auto spatialSw = mlir::dyn_cast<fcc::fabric::SpatialSwOp>(op)) {
    auto swFnType = spatialSw.getFunctionType();
    params.push_back(".NUM_IN(" +
                      std::to_string(swFnType.getNumInputs()) + ")");
    params.push_back(".NUM_OUT(" +
                      std::to_string(swFnType.getNumResults()) + ")");
    int64_t decompBits = spatialSw.getDecomposableBits();
    if (decompBits > 0)
      params.push_back(".DECOMPOSABLE_BITS(" + std::to_string(decompBits) +
                        ")");
    isSwitch = true;
  } else if (auto temporalSw =
                 mlir::dyn_cast<fcc::fabric::TemporalSwOp>(op)) {
    auto tswFnType = temporalSw.getFunctionType();
    params.push_back(
        ".NUM_IN(" + std::to_string(tswFnType.getNumInputs()) + ")");
    params.push_back(
        ".NUM_OUT(" + std::to_string(tswFnType.getNumResults()) + ")");
    params.push_back(".NUM_SLOTS(" +
                      std::to_string(temporalSw.getNumRouteTable()) + ")");
    isSwitch = true;
  } else if (auto fifo = mlir::dyn_cast<fcc::fabric::FifoOp>(op)) {
    params.push_back(".DEPTH(" + std::to_string(fifo.getDepth()) + ")");
    if (fifo.getBypassable())
      params.push_back(".BYPASSABLE(1)");
  } else if (auto addTag = mlir::dyn_cast<fcc::fabric::AddTagOp>(op)) {
    // add_tag: TAG_WIDTH from output type.
    unsigned outTagW = SVEmitter::getTagWidth(addTag.getResult().getType());
    if (outTagW > 0 && tagWidth == 0)
      params.push_back(".TAG_WIDTH(" + std::to_string(outTagW) + ")");
  } else if (auto delTag = mlir::dyn_cast<fcc::fabric::DelTagOp>(op)) {
    // del_tag uses IN_TAG_WIDTH instead of TAG_WIDTH.
    unsigned inTagW = SVEmitter::getTagWidth(delTag.getTagged().getType());
    // Remove TAG_WIDTH if already added, replace with IN_TAG_WIDTH.
    for (auto it = params.begin(); it != params.end(); ++it) {
      if (it->find(".TAG_WIDTH(") != std::string::npos) {
        params.erase(it);
        break;
      }
    }
    if (inTagW > 0)
      params.push_back(".IN_TAG_WIDTH(" + std::to_string(inTagW) + ")");
  } else if (auto mapTag = mlir::dyn_cast<fcc::fabric::MapTagOp>(op)) {
    unsigned inTagW = SVEmitter::getTagWidth(mapTag.getTagged().getType());
    unsigned outTagW = SVEmitter::getTagWidth(mapTag.getResult().getType());
    unsigned tableSize = static_cast<unsigned>(mapTag.getTableSize());
    // map_tag uses IN_TAG_WIDTH, OUT_TAG_WIDTH, TABLE_SIZE.
    for (auto it = params.begin(); it != params.end(); ++it) {
      if (it->find(".TAG_WIDTH(") != std::string::npos) {
        params.erase(it);
        break;
      }
    }
    if (inTagW > 0)
      params.push_back(".IN_TAG_WIDTH(" + std::to_string(inTagW) + ")");
    if (outTagW > 0)
      params.push_back(".OUT_TAG_WIDTH(" + std::to_string(outTagW) + ")");
    params.push_back(".TABLE_SIZE(" + std::to_string(tableSize) + ")");
  } else if (auto memOp = mlir::dyn_cast<fcc::fabric::MemoryOp>(op)) {
    params.push_back(".LD_COUNT(" + std::to_string(memOp.getLdCount()) + ")");
    params.push_back(".ST_COUNT(" + std::to_string(memOp.getStCount()) + ")");
    params.push_back(
        ".IS_PRIVATE(" + std::to_string(memOp.getIsPrivate() ? 1 : 0) + ")");
    params.push_back(".NUM_REGION(" + std::to_string(memOp.getNumRegion()) +
                      ")");
    isMemory = true;
  } else if (auto extMem = mlir::dyn_cast<fcc::fabric::ExtMemoryOp>(op)) {
    params.push_back(".LD_COUNT(" + std::to_string(extMem.getLdCount()) + ")");
    params.push_back(".ST_COUNT(" + std::to_string(extMem.getStCount()) + ")");
    params.push_back(
        ".NUM_REGION(" + std::to_string(extMem.getNumRegion()) + ")");
    isMemory = true;
  }

  // Emit port connections based on module category.
  if (isSwitch) {
    emitSwitchConns(ctx, op, conns);
  } else if (isMemory) {
    emitMemoryConns(ctx, op, conns);
  } else {
    // Single I/O modules: add_tag, del_tag, map_tag, fifo.
    emitSingleIOConns(ctx, op, conns);
  }

  // Config interface (del_tag has no config ports).
  if (!mlir::isa<fcc::fabric::DelTagOp>(op)) {
    conns.push_back({"cfg_valid", instName.str() + "_cfg_valid"});
    conns.push_back({"cfg_wdata", instName.str() + "_cfg_wdata"});
    conns.push_back({"cfg_ready", instName.str() + "_cfg_ready"});

    // Declare config wires.
    ctx.emitter.emitWire("logic", instName.str() + "_cfg_valid");
    ctx.emitter.emitWire("logic [31:0]", instName.str() + "_cfg_wdata");
    ctx.emitter.emitWire("logic", instName.str() + "_cfg_ready");
  }

  ctx.emitter.emitInstance(modName, instName, params, conns);
  ctx.emitter.emitBlankLine();
}

} // namespace

/// Generate the fabric_top.sv top module.
void generateTopModule(fcc::fabric::ModuleOp fabricMod,
                       llvm::raw_ostream &os,
                       SVModuleRegistry &registry,
                       const llvm::DenseMap<mlir::Operation *, std::string>
                           &peModuleNames) {
  std::string topName = SVEmitter::sanitizeName(
      fabricMod.getSymName().str());
  std::string moduleName = "fabric_top_" + topName;

  auto fnType = fabricMod.getFunctionType();
  unsigned numModInputs = fnType.getNumInputs();
  unsigned numModOutputs = fnType.getNumResults();

  SVEmitter emitter(os);
  emitter.emitFileHeader(moduleName);

  // Parameters.
  std::vector<std::string> params;

  // Ports: clk, rst_n, module boundary I/O, config interface.
  std::vector<SVPort> ports;
  ports.push_back({SVPortDir::Input, "logic", "clk"});
  ports.push_back({SVPortDir::Input, "logic", "rst_n"});

  auto &body = fabricMod.getBody().front();

  // Module-level inputs from block arguments.
  for (auto arg : body.getArguments()) {
    mlir::Type argType = arg.getType();
    if (mlir::isa<mlir::MemRefType>(argType))
      continue; // memref ports handled via AXI
    unsigned width = SVEmitter::getTypeWidth(argType);
    std::string argName = "mod_in" + std::to_string(arg.getArgNumber());
    ports.push_back(
        {SVPortDir::Input, "logic" + SVEmitter::bitRange(width), argName});
    ports.push_back({SVPortDir::Input, "logic", argName + "_valid"});
    ports.push_back({SVPortDir::Output, "logic", argName + "_ready"});
  }

  // Module-level outputs from yield operands.
  auto *terminator = body.getTerminator();
  if (auto yieldOp = mlir::dyn_cast<fcc::fabric::YieldOp>(terminator)) {
    for (unsigned i = 0; i < yieldOp.getOperands().size(); ++i) {
      unsigned width =
          SVEmitter::getTypeWidth(yieldOp.getOperand(i).getType());
      std::string outName = "mod_out" + std::to_string(i);
      ports.push_back(
          {SVPortDir::Output, "logic" + SVEmitter::bitRange(width), outName});
      ports.push_back({SVPortDir::Output, "logic", outName + "_valid"});
      ports.push_back({SVPortDir::Input, "logic", outName + "_ready"});
    }
  }

  // Config interface.
  ports.push_back({SVPortDir::Input, "logic", "cfg_valid"});
  ports.push_back({SVPortDir::Input, "logic [31:0]", "cfg_wdata"});
  ports.push_back({SVPortDir::Input, "logic", "cfg_last"});
  ports.push_back({SVPortDir::Output, "logic", "cfg_ready"});

  emitter.emitModuleHeader(moduleName, params, ports);

  // Create TopEmitContext.
  TopEmitContext ctx{emitter, registry, {}, 0, {}, {}, {}};

  // Map block arguments to port names.
  for (auto arg : body.getArguments()) {
    if (mlir::isa<mlir::MemRefType>(arg.getType()))
      continue;
    std::string portName = "mod_in" + std::to_string(arg.getArgNumber());
    ctx.wireNames[arg] = portName;
  }

  // Walk body ops and emit wire declarations + instances.
  emitter.emitComment("Internal wires and module instances");
  emitter.emitBlankLine();

  for (auto &op : body.getOperations()) {
    // Skip definitions (function_unit, PE defs that are not inline).
    if (mlir::isa<fcc::fabric::FunctionUnitOp>(op))
      continue;
    if (mlir::isa<fcc::fabric::YieldOp>(op))
      continue;

    // Instance ops: instantiate the referenced module.
    if (auto instOp = mlir::dyn_cast<fcc::fabric::InstanceOp>(op)) {
      emitTopWires(ctx, op);
      std::string refModName =
          SVEmitter::sanitizeName(instOp.getModule().getValue());
      std::string instName = "inst_" + refModName;
      if (auto sym = instOp.getSymName())
        instName = SVEmitter::sanitizeName(*sym);

      // The referenced module might be a PE -- look up its generated name.
      std::string actualModName = "pe_" + refModName;

      std::vector<std::string> instParams;
      std::vector<SVConnection> conns;

      // Connect operands.
      for (unsigned i = 0; i < instOp.getOperands().size(); ++i) {
        if (mlir::isa<mlir::MemRefType>(instOp.getOperand(i).getType()))
          continue;
        std::string wire = ctx.getOrCreateWire(instOp.getOperand(i));
        conns.push_back({"in" + std::to_string(i), wire});
      }
      // Connect results.
      for (unsigned i = 0; i < instOp.getResults().size(); ++i) {
        std::string wire = ctx.getOrCreateWire(instOp.getResult(i));
        conns.push_back({"out" + std::to_string(i), wire});
      }

      conns.push_back({"clk", "clk"});
      conns.push_back({"rst_n", "rst_n"});

      emitter.emitInstance(actualModName, instName, instParams, conns);
      emitter.emitBlankLine();
      continue;
    }

    // Inline PE ops (with inline_instantiation attribute).
    if (auto spatialPE = mlir::dyn_cast<fcc::fabric::SpatialPEOp>(op)) {
      emitTopWires(ctx, op);
      auto peIt = peModuleNames.find(&op);
      std::string peModName =
          peIt != peModuleNames.end()
              ? peIt->second
              : "pe_" + SVEmitter::sanitizeName(
                            spatialPE.getSymName().value_or("spatial_pe"));
      std::string instName =
          ctx.makeInstanceName(SVEmitter::sanitizeName(
              spatialPE.getSymName().value_or("spe")));

      std::vector<std::string> peParams;
      std::vector<SVConnection> peConns;
      peConns.push_back({"clk", "clk"});
      peConns.push_back({"rst_n", "rst_n"});

      for (unsigned i = 0; i < spatialPE.getInputs().size(); ++i) {
        std::string wire = ctx.getOrCreateWire(spatialPE.getInputs()[i]);
        peConns.push_back({"pe_in" + std::to_string(i), wire});
        peConns.push_back(
            {"pe_in_valid" + std::to_string(i), wire + "_valid"});
        peConns.push_back(
            {"pe_in_ready" + std::to_string(i), wire + "_ready"});
      }
      for (unsigned i = 0; i < spatialPE->getNumResults(); ++i) {
        std::string wire = ctx.getOrCreateWire(spatialPE.getResult(i));
        peConns.push_back({"pe_out" + std::to_string(i), wire});
        peConns.push_back(
            {"pe_out_valid" + std::to_string(i), wire + "_valid"});
        peConns.push_back(
            {"pe_out_ready" + std::to_string(i), wire + "_ready"});
      }

      peConns.push_back(
          {"cfg_valid", instName + "_cfg_valid"});
      peConns.push_back(
          {"cfg_wdata", instName + "_cfg_wdata"});
      peConns.push_back(
          {"cfg_ready", instName + "_cfg_ready"});

      emitter.emitWire("logic", instName + "_cfg_valid");
      emitter.emitWire("logic [31:0]", instName + "_cfg_wdata");
      emitter.emitWire("logic", instName + "_cfg_ready");

      ctx.configInstanceNames.push_back(instName);

      emitter.emitInstance(peModName, instName, peParams, peConns);
      emitter.emitBlankLine();
      continue;
    }

    if (auto temporalPE = mlir::dyn_cast<fcc::fabric::TemporalPEOp>(op)) {
      emitTopWires(ctx, op);
      auto peIt = peModuleNames.find(&op);
      std::string peModName =
          peIt != peModuleNames.end()
              ? peIt->second
              : "pe_" + SVEmitter::sanitizeName(
                            temporalPE.getSymName().value_or("temporal_pe"));
      std::string instName =
          ctx.makeInstanceName(SVEmitter::sanitizeName(
              temporalPE.getSymName().value_or("tpe")));

      std::vector<std::string> peParams;
      std::vector<SVConnection> peConns;
      peConns.push_back({"clk", "clk"});
      peConns.push_back({"rst_n", "rst_n"});

      for (unsigned i = 0; i < temporalPE.getInputs().size(); ++i) {
        std::string wire = ctx.getOrCreateWire(temporalPE.getInputs()[i]);
        peConns.push_back({"pe_in" + std::to_string(i), wire});
        peConns.push_back(
            {"pe_in_valid" + std::to_string(i), wire + "_valid"});
        peConns.push_back(
            {"pe_in_ready" + std::to_string(i), wire + "_ready"});
      }
      for (unsigned i = 0; i < temporalPE->getNumResults(); ++i) {
        std::string wire = ctx.getOrCreateWire(temporalPE.getResult(i));
        peConns.push_back({"pe_out" + std::to_string(i), wire});
        peConns.push_back(
            {"pe_out_valid" + std::to_string(i), wire + "_valid"});
        peConns.push_back(
            {"pe_out_ready" + std::to_string(i), wire + "_ready"});
      }

      peConns.push_back(
          {"cfg_valid", instName + "_cfg_valid"});
      peConns.push_back(
          {"cfg_wdata", instName + "_cfg_wdata"});
      peConns.push_back(
          {"cfg_ready", instName + "_cfg_ready"});

      emitter.emitWire("logic", instName + "_cfg_valid");
      emitter.emitWire("logic [31:0]", instName + "_cfg_wdata");
      emitter.emitWire("logic", instName + "_cfg_ready");

      ctx.configInstanceNames.push_back(instName);

      emitter.emitInstance(peModName, instName, peParams, peConns);
      emitter.emitBlankLine();
      continue;
    }

    // Pre-written module ops (switches, FIFOs, tag ops, memory).
    std::string modName = getModuleName(op);
    if (!modName.empty()) {
      emitTopWires(ctx, op);
      std::string hint = getInstanceHint(op);
      std::string instName =
          hint.empty() ? ctx.makeInstanceName(modName) : hint;
      emitPrewrittenInstance(ctx, op, modName, instName);

      // Track config instance name for modules that have config bits.
      // Must match the set of ops recognized by computeConfigLayout().
      bool hasConfig = false;
      if (mlir::isa<fcc::fabric::SpatialSwOp>(op) ||
          mlir::isa<fcc::fabric::TemporalSwOp>(op) ||
          mlir::isa<fcc::fabric::AddTagOp>(op) ||
          mlir::isa<fcc::fabric::MapTagOp>(op) ||
          mlir::isa<fcc::fabric::MemoryOp>(op) ||
          mlir::isa<fcc::fabric::ExtMemoryOp>(op)) {
        hasConfig = true;
      } else if (auto fifo = mlir::dyn_cast<fcc::fabric::FifoOp>(op)) {
        if (fifo.getBypassable())
          hasConfig = true;
      }
      if (hasConfig)
        ctx.configInstanceNames.push_back(instName);

      continue;
    }
  }

  // Connect yield operands to module outputs.
  emitter.emitBlankLine();
  emitter.emitComment("Module output connections");
  if (auto yieldOp = mlir::dyn_cast<fcc::fabric::YieldOp>(terminator)) {
    for (unsigned i = 0; i < yieldOp.getOperands().size(); ++i) {
      std::string srcWire = ctx.getOrCreateWire(yieldOp.getOperand(i));
      std::string outName = "mod_out" + std::to_string(i);
      emitter.emitAssign(outName, srcWire);
      emitter.emitAssign(outName + "_valid", srcWire + "_valid");
      emitter.emitAssign(srcWire + "_ready", outName + "_ready");
    }
  }

  // Config controller instantiation.
  emitter.emitBlankLine();
  emitter.emitComment("Config controller");
  registry.requireModule("fabric", "fabric_config_ctrl.sv");

  auto configSlices = computeConfigLayout(fabricMod);
  if (!configSlices.empty()) {
    unsigned numSlices = configSlices.size();

    // Compute total words across all slices.
    unsigned totalWords = 0;
    for (const auto &slice : configSlices)
      totalWords += slice.wordCount;

    emitter.emitComment("Config slice table:");
    for (const auto &slice : configSlices) {
      emitter.emitComment("  " + slice.moduleName + ": offset=" +
                          std::to_string(slice.wordOffset) + " words=" +
                          std::to_string(slice.wordCount));
    }
    emitter.emitBlankLine();

    // Build SLICE_OFFSET and SLICE_COUNT parameter array literals.
    // SystemVerilog array parameter: '{val0, val1, ...}
    std::string offsetArr = "'{";
    std::string countArr = "'{";
    for (unsigned i = 0; i < numSlices; ++i) {
      if (i > 0) {
        offsetArr += ", ";
        countArr += ", ";
      }
      offsetArr += std::to_string(configSlices[i].wordOffset);
      countArr += std::to_string(configSlices[i].wordCount);
    }
    offsetArr += "}";
    countArr += "}";

    std::vector<std::string> cfgParams;
    cfgParams.push_back(".NUM_SLICES(" + std::to_string(numSlices) + ")");
    cfgParams.push_back(".TOTAL_WORDS(" + std::to_string(totalWords) + ")");
    cfgParams.push_back(".SLICE_OFFSET(" + offsetArr + ")");
    cfgParams.push_back(".SLICE_COUNT(" + countArr + ")");

    // Declare wires for config controller outputs.
    emitter.emitWire("logic [" + std::to_string(numSlices - 1) + ":0]",
                     "slice_cfg_valid");
    emitter.emitWire("logic [31:0]", "slice_cfg_wdata");
    emitter.emitWire("logic [15:0]", "slice_cfg_word_idx");
    emitter.emitWire("logic", "cfg_done");
    emitter.emitBlankLine();

    std::vector<SVConnection> cfgConns;
    cfgConns.push_back({"clk", "clk"});
    cfgConns.push_back({"rst_n", "rst_n"});
    cfgConns.push_back({"cfg_valid", "cfg_valid"});
    cfgConns.push_back({"cfg_wdata", "cfg_wdata"});
    cfgConns.push_back({"cfg_last", "cfg_last"});
    cfgConns.push_back({"cfg_ready", "cfg_ready"});
    cfgConns.push_back({"slice_cfg_valid", "slice_cfg_valid"});
    cfgConns.push_back({"slice_cfg_wdata", "slice_cfg_wdata"});
    cfgConns.push_back({"slice_cfg_word_idx", "slice_cfg_word_idx"});
    cfgConns.push_back({"cfg_done", "cfg_done"});

    emitter.emitInstance("fabric_config_ctrl", "u_config_ctrl", cfgParams,
                         cfgConns);
    emitter.emitBlankLine();

    // Wire each slice_cfg_valid[i] and slice_cfg_wdata to the
    // corresponding module's config ports.
    emitter.emitComment("Config distribution to module instances");
    for (unsigned i = 0; i < numSlices; ++i) {
      if (i < ctx.configInstanceNames.size()) {
        const std::string &inst = ctx.configInstanceNames[i];
        emitter.emitAssign(inst + "_cfg_valid",
                           "slice_cfg_valid[" + std::to_string(i) + "]");
        emitter.emitAssign(inst + "_cfg_wdata", "slice_cfg_wdata");
      }
    }
  }

  emitter.emitBlankLine();
  emitter.emitModuleFooter(moduleName);
}

} // namespace svgen
} // namespace fcc
