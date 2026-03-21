#include "SVGenInternal.h"

#include "fcc/SVGen/SVEmitter.h"
#include "fcc/SVGen/SVModuleRegistry.h"

#include "fcc/Dialect/Fabric/FabricOps.h"
#include "fcc/Dialect/Fabric/FabricTypes.h"
#include "fcc/Mapper/TypeCompat.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Value.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <string>
#include <vector>

namespace fcc {
namespace svgen {

namespace {

/// Context for emitting a single FU body module.
struct FUEmitContext {
  SVEmitter &emitter;
  SVModuleRegistry &registry;
  llvm::StringRef fpIpProfile;

  /// Map from MLIR SSA value to the SV wire name representing it.
  llvm::DenseMap<mlir::Value, std::string> valueNames;
  unsigned nextWireIdx = 0;

  /// Track whether any error was emitted during code generation.
  bool hadError = false;

  std::string getOrCreateWireName(mlir::Value val) {
    auto it = valueNames.find(val);
    if (it != valueNames.end())
      return it->second;
    std::string name = "w_" + std::to_string(nextWireIdx++);
    valueNames[val] = name;
    return name;
  }
};

/// Get the data width for an SSA value.
static unsigned getValueWidth(mlir::Value val) {
  return SVEmitter::getDataWidth(val.getType());
}

/// Emit wire declarations for all SSA results of an op.
static void emitOpResultWires(FUEmitContext &ctx, mlir::Operation &op) {
  for (mlir::Value result : op.getResults()) {
    std::string wireName = ctx.getOrCreateWireName(result);
    unsigned width = getValueWidth(result);
    std::string typeStr = "logic" + SVEmitter::bitRange(width);
    ctx.emitter.emitWire(typeStr, wireName);
  }
}

/// Emit valid/ready wire declarations for an op instance.
/// For each input port: in_valid_N and in_ready_N wires.
/// For output ports: single-output uses out_valid/out_ready,
/// multi-output uses out_valid_N/out_ready_N.
static void emitHandshakeWires(FUEmitContext &ctx,
                               llvm::StringRef instName,
                               unsigned numOperands,
                               unsigned numResults) {
  for (unsigned i = 0; i < numOperands; ++i) {
    ctx.emitter.emitWire(
        "logic", instName.str() + "_in_valid_" + std::to_string(i));
    ctx.emitter.emitWire(
        "logic", instName.str() + "_in_ready_" + std::to_string(i));
  }
  if (numResults == 1) {
    ctx.emitter.emitWire("logic", instName.str() + "_out_valid");
    ctx.emitter.emitWire("logic", instName.str() + "_out_ready");
  } else {
    for (unsigned i = 0; i < numResults; ++i) {
      ctx.emitter.emitWire(
          "logic", instName.str() + "_out_valid_" + std::to_string(i));
      ctx.emitter.emitWire(
          "logic", instName.str() + "_out_ready_" + std::to_string(i));
    }
  }
}

/// Emit a module instantiation for a dialect compute op.
///
/// The pre-written fu_op_* modules use this port convention:
///   - Parameter: WIDTH (not DATA_WIDTH)
///   - Inputs:  in_data_N, in_valid_N, in_ready_N  (per operand, indexed)
///   - Single output:  out_data, out_valid, out_ready
///   - Multi output:   out_data_N, out_valid_N, out_ready_N
///   - Clock/reset:    clk, rst_n
///   - Optional config: cfg_bits (cmpi), cfg_cont_cond (stream), etc.
static void emitDialectOpInstance(FUEmitContext &ctx, mlir::Operation &op) {
  llvm::StringRef opName = op.getName().getStringRef();
  std::string svModName = SVModuleRegistry::getSVModuleName(opName);
  if (svModName.empty())
    return;

  // Determine instance name from operation position.
  std::string instName =
      svModName + "_inst_" + std::to_string(ctx.nextWireIdx);

  // Gather WIDTH parameter from first result or first operand.
  unsigned dataWidth = 32;
  if (op.getNumResults() > 0)
    dataWidth = getValueWidth(op.getResult(0));
  else if (op.getNumOperands() > 0)
    dataWidth = getValueWidth(op.getOperand(0));

  std::vector<std::string> params;
  params.push_back(".WIDTH(" + std::to_string(dataWidth) + ")");

  unsigned numOperands = op.getNumOperands();
  unsigned numResults = op.getNumResults();

  // Emit handshake wires for this instance.
  emitHandshakeWires(ctx, instName, numOperands, numResults);

  // Build connection list.
  std::vector<SVConnection> connections;

  // Clock and reset.
  connections.push_back({"clk", "clk"});
  connections.push_back({"rst_n", "rst_n"});

  // Input data ports: in_data_N.
  for (unsigned i = 0; i < numOperands; ++i) {
    std::string idx = std::to_string(i);
    std::string expr = ctx.getOrCreateWireName(op.getOperand(i));
    connections.push_back({"in_data_" + idx, expr});
    connections.push_back(
        {"in_valid_" + idx, instName + "_in_valid_" + idx});
    connections.push_back(
        {"in_ready_" + idx, instName + "_in_ready_" + idx});
  }

  // Output data ports: out_data / out_data_N depending on result count.
  if (numResults == 1) {
    std::string expr = ctx.getOrCreateWireName(op.getResult(0));
    connections.push_back({"out_data", expr});
    connections.push_back({"out_valid", instName + "_out_valid"});
    connections.push_back({"out_ready", instName + "_out_ready"});
  } else {
    for (unsigned i = 0; i < numResults; ++i) {
      std::string idx = std::to_string(i);
      std::string expr = ctx.getOrCreateWireName(op.getResult(i));
      connections.push_back({"out_data_" + idx, expr});
      connections.push_back(
          {"out_valid_" + idx, instName + "_out_valid_" + idx});
      connections.push_back(
          {"out_ready_" + idx, instName + "_out_ready_" + idx});
    }
  }

  ctx.emitter.emitInstance(svModName, instName, params, connections);
  ctx.emitter.emitBlankLine();
}

/// Emit a fabric_mux instantiation for a fabric.mux op.
///
/// fabric_mux.sv ports:
///   - Parameters: NUM_IN, DATA_WIDTH
///   - clk, rst_n
///   - Config: cfg_valid, cfg_wdata[31:0], cfg_ready
///   - Inputs (array): in_valid[NUM_IN-1:0], in_ready[NUM_IN-1:0],
///                     in_data[0:NUM_IN-1]
///   - Output: out_valid, out_ready, out_data
static void emitMuxInstance(FUEmitContext &ctx, fcc::fabric::MuxOp muxOp) {
  unsigned numIn = muxOp.getInputs().size();

  std::string instName = "mux_inst_" + std::to_string(ctx.nextWireIdx);

  unsigned dataWidth = 32;
  if (muxOp.getResults().size() > 0)
    dataWidth = getValueWidth(muxOp.getResult(0));
  else if (numIn > 0)
    dataWidth = getValueWidth(muxOp.getInputs()[0]);

  std::vector<std::string> params;
  params.push_back(".NUM_IN(" + std::to_string(numIn) + ")");
  params.push_back(".DATA_WIDTH(" + std::to_string(dataWidth) + ")");

  std::vector<SVConnection> connections;

  // Clock and reset.
  connections.push_back({"clk", "clk"});
  connections.push_back({"rst_n", "rst_n"});

  // Config port: driven from FU-level config wiring.
  connections.push_back({"cfg_valid", "cfg_mux_valid_" + instName});
  connections.push_back({"cfg_wdata", "cfg_mux_wdata_" + instName});
  connections.push_back({"cfg_ready", "cfg_mux_ready_" + instName});

  // Input ports: fabric_mux uses packed arrays in_valid[N], in_ready[N],
  // and unpacked array in_data[0:N-1]. We concatenate individual wires
  // into the packed vectors and use per-element assignment for in_data.
  //
  // For the packed in_valid/in_ready vectors, build a concatenation
  // expression: {wire_N-1, ..., wire_1, wire_0}.
  std::string inValidConcat = "{";
  for (int i = static_cast<int>(numIn) - 1; i >= 0; --i) {
    if (i < static_cast<int>(numIn) - 1)
      inValidConcat += ", ";
    inValidConcat += instName + "_in_valid_" + std::to_string(i);
  }
  inValidConcat += "}";
  connections.push_back({"in_valid", inValidConcat});

  // in_ready is an output packed vector; declare per-element wires and
  // connect the packed vector.
  for (unsigned i = 0; i < numIn; ++i) {
    ctx.emitter.emitWire("logic",
                         instName + "_in_valid_" + std::to_string(i));
    ctx.emitter.emitWire("logic",
                         instName + "_in_ready_" + std::to_string(i));
  }

  std::string inReadyConcat = "{";
  for (int i = static_cast<int>(numIn) - 1; i >= 0; --i) {
    if (i < static_cast<int>(numIn) - 1)
      inReadyConcat += ", ";
    inReadyConcat += instName + "_in_ready_" + std::to_string(i);
  }
  inReadyConcat += "}";
  connections.push_back({"in_ready", inReadyConcat});

  // in_data is an unpacked array -- connected via a generate-friendly
  // temporary wire array. Emit an array wire and assign each element.
  std::string dataArrayName = instName + "_in_data";
  ctx.emitter.emitRaw(
      "logic [" + std::to_string(dataWidth - 1) + ":0] " +
      dataArrayName + " [0:" + std::to_string(numIn - 1) + "];\n");
  for (unsigned i = 0; i < numIn; ++i) {
    std::string expr = ctx.getOrCreateWireName(muxOp.getInputs()[i]);
    ctx.emitter.emitAssign(
        dataArrayName + "[" + std::to_string(i) + "]", expr);
  }
  connections.push_back({"in_data", dataArrayName});

  // Output port (single output).
  std::string outExpr = ctx.getOrCreateWireName(muxOp.getResult(0));
  connections.push_back({"out_data", outExpr});
  connections.push_back({"out_valid", instName + "_out_valid"});
  connections.push_back({"out_ready", instName + "_out_ready"});

  ctx.emitter.emitWire("logic", instName + "_out_valid");
  ctx.emitter.emitWire("logic", instName + "_out_ready");

  ctx.emitter.emitInstance("fabric_mux", instName, params, connections);
  ctx.emitter.emitBlankLine();
}

} // namespace

/// Generate a FU body SV module for a FunctionUnitOp.
/// Returns the generated SV module name, or an empty string on error.
std::string generateFUBody(fcc::fabric::FunctionUnitOp fuOp,
                           llvm::raw_ostream &os,
                           SVModuleRegistry &registry,
                           llvm::StringRef fpIpProfile) {
  std::string fuName =
      SVEmitter::sanitizeName(fuOp.getSymName());
  std::string moduleName = "fu_" + fuName;

  SVEmitter emitter(os);
  emitter.emitFileHeader(moduleName);

  auto fnType = fuOp.getFunctionType();
  unsigned numIn = fnType.getNumInputs();
  unsigned numOut = fnType.getNumResults();

  // Determine data width from FU port types.
  unsigned dataWidth = 32;
  if (numIn > 0)
    dataWidth = SVEmitter::getDataWidth(fnType.getInput(0));
  else if (numOut > 0)
    dataWidth = SVEmitter::getDataWidth(fnType.getResult(0));

  // Count mux ops for config bit computation.
  unsigned totalMuxConfigBits = 0;
  fuOp.getBody().front().walk([&](fcc::fabric::MuxOp muxOp) {
    unsigned numMuxIn = muxOp.getInputs().size();
    unsigned selBits = numMuxIn > 1 ? llvm::Log2_32_Ceil(numMuxIn) : 0;
    totalMuxConfigBits += selBits + 2; // sel + discard + disconnect
  });

  // Build parameter list.
  std::vector<std::string> params;
  params.push_back("parameter DATA_WIDTH = " + std::to_string(dataWidth));
  if (totalMuxConfigBits > 0)
    params.push_back("parameter FU_CFG_BITS = " +
                      std::to_string(totalMuxConfigBits));

  // Build port list using in_data_N / in_valid_N / in_ready_N convention.
  std::vector<SVPort> ports;
  ports.push_back({SVPortDir::Input, "logic", "clk"});
  ports.push_back({SVPortDir::Input, "logic", "rst_n"});

  for (unsigned i = 0; i < numIn; ++i) {
    unsigned w = SVEmitter::getDataWidth(fnType.getInput(i));
    std::string idx = std::to_string(i);
    ports.push_back(
        {SVPortDir::Input, "logic" + SVEmitter::bitRange(w),
         "in_data_" + idx});
    ports.push_back({SVPortDir::Input, "logic", "in_valid_" + idx});
    ports.push_back({SVPortDir::Output, "logic", "in_ready_" + idx});
  }
  for (unsigned i = 0; i < numOut; ++i) {
    unsigned w = SVEmitter::getDataWidth(fnType.getResult(i));
    std::string idx = std::to_string(i);
    ports.push_back(
        {SVPortDir::Output, "logic" + SVEmitter::bitRange(w),
         "out_data_" + idx});
    ports.push_back({SVPortDir::Output, "logic", "out_valid_" + idx});
    ports.push_back({SVPortDir::Input, "logic", "out_ready_" + idx});
  }
  if (totalMuxConfigBits > 0)
    ports.push_back(
        {SVPortDir::Input,
         "logic" + SVEmitter::bitRange(totalMuxConfigBits),
         "fu_cfg"});

  emitter.emitModuleHeader(moduleName, params, ports);

  // Set up emission context.
  FUEmitContext ctx{emitter, registry, fpIpProfile, {}, 0, false};

  // Map block arguments to input port data wire names.
  auto &bodyBlock = fuOp.getBody().front();
  for (unsigned i = 0; i < numIn && i < bodyBlock.getNumArguments(); ++i) {
    ctx.valueNames[bodyBlock.getArgument(i)] =
        "in_data_" + std::to_string(i);
  }

  // Declare internal wires and emit logic.
  emitter.emitComment("Internal wiring");

  for (auto &op : bodyBlock.getOperations()) {
    llvm::StringRef opName = op.getName().getStringRef();

    // Skip the yield terminator -- handled separately.
    if (mlir::isa<fcc::fabric::YieldOp>(op))
      continue;

    // Handle fabric.mux specially.
    if (auto muxOp = mlir::dyn_cast<fcc::fabric::MuxOp>(op)) {
      emitOpResultWires(ctx, op);
      emitMuxInstance(ctx, muxOp);
      continue;
    }

    // For known dialect ops, emit wire + instance.
    if (SVModuleRegistry::isKnownOp(opName)) {
      emitOpResultWires(ctx, op);
      emitDialectOpInstance(ctx, op);
      continue;
    }

    // Unknown ops: emit error and mark failure.
    llvm::errs() << "gen-sv error: unsupported-op: " << opName
                 << " in FU " << fuOp.getSymName() << "\n";
    ctx.hadError = true;
  }

  // If there was an unsupported op, return empty to signal error.
  if (ctx.hadError) {
    emitter.emitComment("ERROR: generation aborted due to unsupported ops");
    emitter.emitModuleFooter(moduleName);
    return "";
  }

  // Connect yield operands to output ports.
  emitter.emitBlankLine();
  emitter.emitComment("Output connections");
  if (auto yieldOp = mlir::dyn_cast<fcc::fabric::YieldOp>(
          bodyBlock.getTerminator())) {
    for (unsigned i = 0; i < yieldOp.getOperands().size() && i < numOut;
         ++i) {
      std::string srcWire = ctx.getOrCreateWireName(yieldOp.getOperand(i));
      emitter.emitAssign("out_data_" + std::to_string(i), srcWire);
    }
  }

  // Handshake passthrough for combinational FU:
  // Each input valid is driven from the corresponding FU port,
  // each output ready is driven from the corresponding FU port.
  // For now, simple passthrough: all input valids AND-ed -> all output
  // valids; all output readies AND-ed -> all input readies.
  emitter.emitBlankLine();
  emitter.emitComment("Handshake passthrough (combinational FU)");

  // Build AND of all input valids.
  std::string allInValid;
  if (numIn == 0) {
    allInValid = "1'b1";
  } else if (numIn == 1) {
    allInValid = "in_valid_0";
  } else {
    allInValid = "in_valid_0";
    for (unsigned i = 1; i < numIn; ++i)
      allInValid += " & in_valid_" + std::to_string(i);
  }

  // Build AND of all output readies.
  std::string allOutReady;
  if (numOut == 0) {
    allOutReady = "1'b1";
  } else if (numOut == 1) {
    allOutReady = "out_ready_0";
  } else {
    allOutReady = "out_ready_0";
    for (unsigned i = 1; i < numOut; ++i)
      allOutReady += " & out_ready_" + std::to_string(i);
  }

  for (unsigned i = 0; i < numOut; ++i) {
    emitter.emitAssign("out_valid_" + std::to_string(i),
                        allInValid);
  }
  for (unsigned i = 0; i < numIn; ++i) {
    emitter.emitAssign("in_ready_" + std::to_string(i),
                        allOutReady);
  }

  emitter.emitBlankLine();
  emitter.emitModuleFooter(moduleName);

  return moduleName;
}

} // namespace svgen
} // namespace fcc
