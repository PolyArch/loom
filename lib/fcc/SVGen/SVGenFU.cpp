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

/// Emit a module instantiation for a dialect compute op.
static void emitDialectOpInstance(FUEmitContext &ctx, mlir::Operation &op) {
  llvm::StringRef opName = op.getName().getStringRef();
  std::string svModName = SVModuleRegistry::getSVModuleName(opName);
  if (svModName.empty())
    return;

  // Determine instance name from operation position.
  std::string instName =
      svModName + "_inst_" + std::to_string(ctx.nextWireIdx);

  // Gather DATA_WIDTH parameter from first result.
  unsigned dataWidth = 32;
  if (op.getNumResults() > 0)
    dataWidth = getValueWidth(op.getResult(0));
  else if (op.getNumOperands() > 0)
    dataWidth = getValueWidth(op.getOperand(0));

  std::vector<std::string> params;
  params.push_back(".DATA_WIDTH(" + std::to_string(dataWidth) + ")");

  // Connections: operands -> in0, in1, ...; results -> out0, out1, ...
  std::vector<SVConnection> connections;

  for (unsigned i = 0; i < op.getNumOperands(); ++i) {
    std::string portName = "in" + std::to_string(i);
    std::string expr = ctx.getOrCreateWireName(op.getOperand(i));
    connections.push_back({portName, expr});
  }

  for (unsigned i = 0; i < op.getNumResults(); ++i) {
    std::string portName = "out" + std::to_string(i);
    std::string expr = ctx.getOrCreateWireName(op.getResult(i));
    connections.push_back({portName, expr});
  }

  // Valid/ready handshake ports.
  connections.push_back({"in_valid", "fu_in_valid"});
  connections.push_back({"in_ready", "fu_in_ready_" + instName});
  connections.push_back({"out_valid", "fu_out_valid_" + instName});
  connections.push_back({"out_ready", "fu_out_ready"});

  ctx.emitter.emitInstance(svModName, instName, params, connections);
  ctx.emitter.emitBlankLine();
}

/// Emit a fabric_mux instantiation for a fabric.mux op.
static void emitMuxInstance(FUEmitContext &ctx, fcc::fabric::MuxOp muxOp) {
  unsigned numIn = muxOp.getInputs().size();
  unsigned numOut = muxOp.getResults().size();

  // For a standard N:1 mux.
  std::string instName = "mux_inst_" + std::to_string(ctx.nextWireIdx);

  unsigned dataWidth = 32;
  if (numOut > 0)
    dataWidth = getValueWidth(muxOp.getResult(0));
  else if (numIn > 0)
    dataWidth = getValueWidth(muxOp.getInputs()[0]);

  std::vector<std::string> params;
  params.push_back(".NUM_IN(" + std::to_string(numIn) + ")");
  params.push_back(".DATA_WIDTH(" + std::to_string(dataWidth) + ")");

  std::vector<SVConnection> connections;

  // Input ports: in[0], in[1], ... connected via concatenation or individual.
  // For simplicity, connect each input individually.
  for (unsigned i = 0; i < numIn; ++i) {
    std::string portName = "in" + std::to_string(i);
    std::string expr = ctx.getOrCreateWireName(muxOp.getInputs()[i]);
    connections.push_back({portName, expr});
  }

  // Output port.
  for (unsigned i = 0; i < numOut; ++i) {
    std::string portName = "out" + std::to_string(i);
    std::string expr = ctx.getOrCreateWireName(muxOp.getResult(i));
    connections.push_back({portName, expr});
  }

  // Config connections: sel, discard, disconnect come from config register
  // bits routed through the PE container.
  connections.push_back({"cfg_sel", "cfg_mux_sel_" + instName});
  connections.push_back({"cfg_discard", "cfg_mux_discard_" + instName});
  connections.push_back({"cfg_disconnect", "cfg_mux_disconnect_" + instName});

  ctx.emitter.emitInstance("fabric_mux", instName, params, connections);
  ctx.emitter.emitBlankLine();
}

} // namespace

/// Generate a FU body SV module for a FunctionUnitOp.
/// Returns the generated SV module name.
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

  // Build port list.
  std::vector<SVPort> ports;
  for (unsigned i = 0; i < numIn; ++i) {
    unsigned w = SVEmitter::getDataWidth(fnType.getInput(i));
    ports.push_back(
        {SVPortDir::Input, "logic" + SVEmitter::bitRange(w),
         "in" + std::to_string(i)});
  }
  for (unsigned i = 0; i < numOut; ++i) {
    unsigned w = SVEmitter::getDataWidth(fnType.getResult(i));
    ports.push_back(
        {SVPortDir::Output, "logic" + SVEmitter::bitRange(w),
         "out" + std::to_string(i)});
  }
  ports.push_back({SVPortDir::Input, "logic", "in_valid"});
  ports.push_back({SVPortDir::Output, "logic", "in_ready"});
  ports.push_back({SVPortDir::Output, "logic", "out_valid"});
  ports.push_back({SVPortDir::Input, "logic", "out_ready"});
  if (totalMuxConfigBits > 0)
    ports.push_back(
        {SVPortDir::Input,
         "logic" + SVEmitter::bitRange(totalMuxConfigBits),
         "fu_cfg"});

  emitter.emitModuleHeader(moduleName, params, ports);

  // Set up emission context.
  FUEmitContext ctx{emitter, registry, fpIpProfile, {}, 0};

  // Map block arguments to input port names.
  auto &bodyBlock = fuOp.getBody().front();
  for (unsigned i = 0; i < numIn && i < bodyBlock.getNumArguments(); ++i) {
    ctx.valueNames[bodyBlock.getArgument(i)] = "in" + std::to_string(i);
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

    // Unknown ops: emit wires and a comment placeholder.
    emitOpResultWires(ctx, op);
    emitter.emitComment("TODO: unsupported op " + opName.str());
  }

  // Connect yield operands to output ports.
  emitter.emitBlankLine();
  emitter.emitComment("Output connections");
  if (auto yieldOp = mlir::dyn_cast<fcc::fabric::YieldOp>(
          bodyBlock.getTerminator())) {
    for (unsigned i = 0; i < yieldOp.getOperands().size() && i < numOut;
         ++i) {
      std::string srcWire = ctx.getOrCreateWireName(yieldOp.getOperand(i));
      emitter.emitAssign("out" + std::to_string(i), srcWire);
    }
  }

  // Simple valid/ready passthrough for combinational FU.
  emitter.emitBlankLine();
  emitter.emitComment("Handshake passthrough (combinational FU)");
  emitter.emitAssign("out_valid", "in_valid");
  emitter.emitAssign("in_ready", "out_ready");

  emitter.emitBlankLine();
  emitter.emitModuleFooter(moduleName);

  return moduleName;
}

} // namespace svgen
} // namespace fcc
