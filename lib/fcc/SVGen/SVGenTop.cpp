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

/// Emit a pre-written module instance with parameter overrides.
static void emitPrewrittenInstance(TopEmitContext &ctx, mlir::Operation &op,
                                   llvm::StringRef modName,
                                   llvm::StringRef instName) {
  std::vector<std::string> params;
  std::vector<SVConnection> conns;

  // Determine data width and tag width from port types.
  unsigned dataWidth = 32;
  unsigned tagWidth = 0;
  if (op.getNumOperands() > 0) {
    dataWidth = SVEmitter::getDataWidth(op.getOperand(0).getType());
    tagWidth = SVEmitter::getTagWidth(op.getOperand(0).getType());
  } else if (op.getNumResults() > 0) {
    dataWidth = SVEmitter::getDataWidth(op.getResult(0).getType());
    tagWidth = SVEmitter::getTagWidth(op.getResult(0).getType());
  }

  params.push_back(".DATA_WIDTH(" + std::to_string(dataWidth) + ")");
  if (tagWidth > 0)
    params.push_back(".TAG_WIDTH(" + std::to_string(tagWidth) + ")");

  // Op-specific parameters.
  if (auto spatialSw = mlir::dyn_cast<fcc::fabric::SpatialSwOp>(op)) {
    params.push_back(".NUM_IN(" +
                      std::to_string(spatialSw.getInputs().size()) + ")");
    params.push_back(".NUM_OUT(" +
                      std::to_string(spatialSw.getOutputs().size()) + ")");
    int64_t decompBits = spatialSw.getDecomposableBits();
    if (decompBits > 0)
      params.push_back(".DECOMPOSABLE_BITS(" + std::to_string(decompBits) +
                        ")");
  } else if (auto temporalSw =
                 mlir::dyn_cast<fcc::fabric::TemporalSwOp>(op)) {
    params.push_back(
        ".NUM_IN(" + std::to_string(temporalSw.getInputs().size()) + ")");
    params.push_back(
        ".NUM_OUT(" + std::to_string(temporalSw.getOutputs().size()) + ")");
    params.push_back(".NUM_SLOTS(" +
                      std::to_string(temporalSw.getNumRouteTable()) + ")");
  } else if (auto fifo = mlir::dyn_cast<fcc::fabric::FifoOp>(op)) {
    params.push_back(".DEPTH(" + std::to_string(fifo.getDepth()) + ")");
    if (fifo.getBypassable())
      params.push_back(".BYPASSABLE(1)");
  } else if (auto memOp = mlir::dyn_cast<fcc::fabric::MemoryOp>(op)) {
    params.push_back(".LD_COUNT(" + std::to_string(memOp.getLdCount()) + ")");
    params.push_back(".ST_COUNT(" + std::to_string(memOp.getStCount()) + ")");
    params.push_back(
        ".IS_PRIVATE(" + std::to_string(memOp.getIsPrivate() ? 1 : 0) + ")");
    params.push_back(".NUM_REGION(" + std::to_string(memOp.getNumRegion()) +
                      ")");
  } else if (auto extMem = mlir::dyn_cast<fcc::fabric::ExtMemoryOp>(op)) {
    params.push_back(".LD_COUNT(" + std::to_string(extMem.getLdCount()) + ")");
    params.push_back(".ST_COUNT(" + std::to_string(extMem.getStCount()) + ")");
    params.push_back(
        ".NUM_REGION(" + std::to_string(extMem.getNumRegion()) + ")");
  }

  // Connect operands (inputs).
  for (unsigned i = 0; i < op.getNumOperands(); ++i) {
    // Skip memref operands (bound separately for memory modules).
    if (mlir::isa<mlir::MemRefType>(op.getOperand(i).getType()))
      continue;
    std::string wire = ctx.getOrCreateWire(op.getOperand(i));
    conns.push_back({"in" + std::to_string(i), wire});
    conns.push_back({"in" + std::to_string(i) + "_valid", wire + "_valid"});
    conns.push_back({"in" + std::to_string(i) + "_ready", wire + "_ready"});
  }

  // Connect results (outputs).
  for (unsigned i = 0; i < op.getNumResults(); ++i) {
    std::string wire = ctx.getOrCreateWire(op.getResult(i));
    conns.push_back({"out" + std::to_string(i), wire});
    conns.push_back(
        {"out" + std::to_string(i) + "_valid", wire + "_valid"});
    conns.push_back(
        {"out" + std::to_string(i) + "_ready", wire + "_ready"});
  }

  // Config interface.
  conns.push_back({"cfg_valid", instName.str() + "_cfg_valid"});
  conns.push_back({"cfg_wdata", instName.str() + "_cfg_wdata"});
  conns.push_back({"cfg_ready", instName.str() + "_cfg_ready"});

  // Declare config wires.
  ctx.emitter.emitWire("logic", instName.str() + "_cfg_valid");
  ctx.emitter.emitWire("logic [31:0]", instName.str() + "_cfg_wdata");
  ctx.emitter.emitWire("logic", instName.str() + "_cfg_ready");

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
  TopEmitContext ctx{emitter, registry, {}, 0, {}, {}};

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
      for (unsigned i = 0; i < spatialPE.getOutputs().size(); ++i) {
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
      for (unsigned i = 0; i < temporalPE.getOutputs().size(); ++i) {
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
    emitter.emitComment("Config slice table:");
    for (const auto &slice : configSlices) {
      emitter.emitComment("  " + slice.moduleName + ": offset=" +
                          std::to_string(slice.wordOffset) + " words=" +
                          std::to_string(slice.wordCount));
    }
    emitter.emitBlankLine();

    std::vector<std::string> cfgParams;
    cfgParams.push_back(".NUM_MODULES(" +
                         std::to_string(configSlices.size()) + ")");

    std::vector<SVConnection> cfgConns;
    cfgConns.push_back({"clk", "clk"});
    cfgConns.push_back({"rst_n", "rst_n"});
    cfgConns.push_back({"cfg_valid", "cfg_valid"});
    cfgConns.push_back({"cfg_wdata", "cfg_wdata"});
    cfgConns.push_back({"cfg_last", "cfg_last"});
    cfgConns.push_back({"cfg_ready", "cfg_ready"});

    emitter.emitInstance("fabric_config_ctrl", "u_config_ctrl", cfgParams,
                         cfgConns);
  }

  emitter.emitBlankLine();
  emitter.emitModuleFooter(moduleName);
}

} // namespace svgen
} // namespace fcc
