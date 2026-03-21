#include "SVGenInternal.h"

#include "fcc/SVGen/SVEmitter.h"
#include "fcc/SVGen/SVModuleRegistry.h"

#include "fcc/Dialect/Fabric/FabricOps.h"
#include "fcc/Dialect/Fabric/FabricTypes.h"
#include "fcc/Mapper/TypeCompat.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Value.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
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

  /// Map from MLIR SSA value to the SV wire base name representing it.
  /// The actual wires are: <base>_data (or just <base> for data),
  /// <base>_valid, <base>_ready.
  llvm::DenseMap<mlir::Value, std::string> valueNames;
  unsigned nextWireIdx = 0;

  /// Track whether any error was emitted during code generation.
  bool hadError = false;

  /// Running config bit offset within fu_cfg for configurable ops.
  unsigned cfgBitOffset = 0;

  /// For each SSA value, track all consumer ready wire names.
  /// The value's ready signal = AND of all consumer readies.
  llvm::DenseMap<mlir::Value, llvm::SmallVector<std::string, 2>>
      valueConsumerReadies;

  std::string getOrCreateWireName(mlir::Value val) {
    auto it = valueNames.find(val);
    if (it != valueNames.end())
      return it->second;
    std::string name = "w_" + std::to_string(nextWireIdx++);
    valueNames[val] = name;
    return name;
  }

  /// Get the valid wire name for an SSA value.
  std::string getValidWire(mlir::Value val) {
    return getOrCreateWireName(val) + "_valid";
  }

  /// Get the ready wire name for an SSA value.
  std::string getReadyWire(mlir::Value val) {
    return getOrCreateWireName(val) + "_ready";
  }

  /// Get the data wire name for an SSA value.
  std::string getDataWire(mlir::Value val) {
    return getOrCreateWireName(val);
  }

  /// Register a consumer ready wire for an SSA value.
  /// Returns the name of the per-consumer ready wire that will be driven
  /// by the consumer instance.
  std::string addConsumerReady(mlir::Value val, llvm::StringRef consumerInst,
                               unsigned operandIdx) {
    std::string readyWire = consumerInst.str() + "_in_ready_" +
                            std::to_string(operandIdx);
    valueConsumerReadies[val].push_back(readyWire);
    return readyWire;
  }
};

/// Get the data width for an SSA value.
static unsigned getValueWidth(mlir::Value val) {
  return SVEmitter::getDataWidth(val.getType());
}

/// Emit data, valid, and ready wire declarations for all SSA results of an op.
static void emitOpResultWires(FUEmitContext &ctx, mlir::Operation &op) {
  for (mlir::Value result : op.getResults()) {
    std::string wireName = ctx.getOrCreateWireName(result);
    unsigned width = getValueWidth(result);
    std::string typeStr = "logic" + SVEmitter::bitRange(width);
    ctx.emitter.emitWire(typeStr, wireName);
    ctx.emitter.emitWire("logic", wireName + "_valid");
    ctx.emitter.emitWire("logic", wireName + "_ready");
  }
}

/// Compute the config bit count for a single body op.
/// Returns 0 for non-configurable ops.
static unsigned getOpConfigBits(mlir::Operation &op, unsigned dataWidth) {
  llvm::StringRef opName = op.getName().getStringRef();
  if (mlir::isa<fcc::fabric::MuxOp>(op)) {
    auto muxOp = mlir::cast<fcc::fabric::MuxOp>(op);
    unsigned numMuxIn = muxOp.getInputs().size();
    unsigned selBits = numMuxIn > 1 ? llvm::Log2_32_Ceil(numMuxIn) : 0;
    return selBits + 2; // sel + discard + disconnect
  }
  if (opName == "arith.cmpi" || opName == "arith.cmpf")
    return 4;
  if (opName == "handshake.constant")
    return dataWidth;
  if (opName == "handshake.join")
    return op.getNumOperands(); // NUM_IN bits for join mask
  if (opName == "dataflow.stream")
    return 5;
  return 0;
}

/// Emit a module instantiation for a standard dialect compute op.
///
/// Standard ops use this port convention:
///   - Parameter: WIDTH
///   - Inputs:  in_data_N, in_valid_N, in_ready_N  (per operand, indexed)
///   - Single output:  out_data, out_valid, out_ready
///   - Multi output:   out_data_N, out_valid_N, out_ready_N
///   - Clock/reset:    clk, rst_n
///   - Optional config: cfg_bits (cmpi/cmpf), cfg_value (constant),
///     cfg_cont_cond (stream)
///
/// SSA-driven handshake: each operand's in_valid comes from the SSA value's
/// valid wire; each result's out_valid drives the SSA result's valid wire.
/// The instance's in_ready outputs are registered as consumer readies for
/// the SSA operand values. The instance's out_ready input comes from the
/// SSA result's ready wire.
static void emitStandardOpInstance(FUEmitContext &ctx, mlir::Operation &op,
                                   unsigned dataWidth) {
  llvm::StringRef opName = op.getName().getStringRef();
  std::string svModName = SVModuleRegistry::getSVModuleName(opName);
  if (svModName.empty())
    return;

  std::string instName =
      svModName + "_inst_" + std::to_string(ctx.nextWireIdx);

  // WIDTH parameter from first result or first operand.
  unsigned opDataWidth = 32;
  if (op.getNumResults() > 0)
    opDataWidth = getValueWidth(op.getResult(0));
  else if (op.getNumOperands() > 0)
    opDataWidth = getValueWidth(op.getOperand(0));

  std::vector<std::string> params;
  params.push_back(".WIDTH(" + std::to_string(opDataWidth) + ")");

  unsigned numOperands = op.getNumOperands();
  unsigned numResults = op.getNumResults();

  // Declare per-operand ready wires (outputs from the instance).
  for (unsigned i = 0; i < numOperands; ++i) {
    std::string readyWire = instName + "_in_ready_" + std::to_string(i);
    ctx.emitter.emitWire("logic", readyWire);
    // Register this ready wire as a consumer of the SSA operand.
    ctx.addConsumerReady(op.getOperand(i), instName, i);
  }

  // Build connection list.
  std::vector<SVConnection> connections;

  connections.push_back({"clk", "clk"});
  connections.push_back({"rst_n", "rst_n"});

  // Input data/valid/ready ports per operand.
  for (unsigned i = 0; i < numOperands; ++i) {
    std::string idx = std::to_string(i);
    std::string dataExpr = ctx.getDataWire(op.getOperand(i));
    std::string validExpr = ctx.getValidWire(op.getOperand(i));
    std::string readyWire = instName + "_in_ready_" + idx;
    connections.push_back({"in_data_" + idx, dataExpr});
    connections.push_back({"in_valid_" + idx, validExpr});
    connections.push_back({"in_ready_" + idx, readyWire});
  }

  // Output data/valid/ready ports.
  if (numResults == 1) {
    std::string dataWire = ctx.getDataWire(op.getResult(0));
    std::string validWire = ctx.getValidWire(op.getResult(0));
    std::string readyWire = ctx.getReadyWire(op.getResult(0));
    connections.push_back({"out_data", dataWire});
    connections.push_back({"out_valid", validWire});
    connections.push_back({"out_ready", readyWire});
  } else {
    for (unsigned i = 0; i < numResults; ++i) {
      std::string idx = std::to_string(i);
      std::string dataWire = ctx.getDataWire(op.getResult(i));
      std::string validWire = ctx.getValidWire(op.getResult(i));
      std::string readyWire = ctx.getReadyWire(op.getResult(i));
      connections.push_back({"out_data_" + idx, dataWire});
      connections.push_back({"out_valid_" + idx, validWire});
      connections.push_back({"out_ready_" + idx, readyWire});
    }
  }

  // Connect op-specific config ports from fu_cfg bit slices.
  unsigned cfgBits = getOpConfigBits(op, dataWidth);
  if (cfgBits > 0) {
    std::string cfgSlice;
    if (cfgBits == 1) {
      cfgSlice = "fu_cfg[" + std::to_string(ctx.cfgBitOffset) + "]";
    } else {
      cfgSlice = "fu_cfg[" +
                 std::to_string(ctx.cfgBitOffset + cfgBits - 1) + ":" +
                 std::to_string(ctx.cfgBitOffset) + "]";
    }

    if (opName == "arith.cmpi" || opName == "arith.cmpf") {
      connections.push_back({"cfg_bits", cfgSlice});
    } else if (opName == "handshake.constant") {
      connections.push_back({"cfg_value", cfgSlice});
    } else if (opName == "dataflow.stream") {
      connections.push_back({"cfg_cont_cond", cfgSlice});
    }

    ctx.cfgBitOffset += cfgBits;
  }

  ctx.emitter.emitInstance(svModName, instName, params, connections);
  ctx.emitter.emitBlankLine();
}

/// Emit a handshake.join instantiation with packed-array ports.
///
/// fu_op_join.sv ports:
///   - Parameters: NUM_IN, WIDTH
///   - in_data  [NUM_IN-1:0][WIDTH-1:0] (packed 2D)
///   - in_valid [NUM_IN-1:0]
///   - in_ready [NUM_IN-1:0]
///   - out_data, out_valid, out_ready
///   - cfg_join_mask [NUM_IN-1:0]
static void emitJoinInstance(FUEmitContext &ctx, mlir::Operation &op,
                             unsigned fuDataWidth) {
  unsigned numIn = op.getNumOperands();

  std::string instName = "fu_op_join_inst_" + std::to_string(ctx.nextWireIdx);

  unsigned opDataWidth = 32;
  if (op.getNumResults() > 0)
    opDataWidth = getValueWidth(op.getResult(0));
  else if (numIn > 0)
    opDataWidth = getValueWidth(op.getOperand(0));

  std::vector<std::string> params;
  params.push_back(".NUM_IN(" + std::to_string(numIn) + ")");
  params.push_back(".WIDTH(" + std::to_string(opDataWidth) + ")");

  // Declare per-input ready wires.
  for (unsigned i = 0; i < numIn; ++i) {
    std::string readyWire = instName + "_in_ready_" + std::to_string(i);
    ctx.emitter.emitWire("logic", readyWire);
    ctx.addConsumerReady(op.getOperand(i), instName, i);
  }

  // Build in_data concatenation: packed [NUM_IN-1:0][WIDTH-1:0].
  // Concatenation order: {data_N-1, data_N-2, ..., data_0}.
  std::string inDataConcat = "{";
  for (int i = static_cast<int>(numIn) - 1; i >= 0; --i) {
    if (i < static_cast<int>(numIn) - 1)
      inDataConcat += ", ";
    inDataConcat += ctx.getDataWire(op.getOperand(i));
  }
  inDataConcat += "}";

  // Build in_valid concatenation.
  std::string inValidConcat = "{";
  for (int i = static_cast<int>(numIn) - 1; i >= 0; --i) {
    if (i < static_cast<int>(numIn) - 1)
      inValidConcat += ", ";
    inValidConcat += ctx.getValidWire(op.getOperand(i));
  }
  inValidConcat += "}";

  // Build in_ready concatenation.
  std::string inReadyConcat = "{";
  for (int i = static_cast<int>(numIn) - 1; i >= 0; --i) {
    if (i < static_cast<int>(numIn) - 1)
      inReadyConcat += ", ";
    inReadyConcat += instName + "_in_ready_" + std::to_string(i);
  }
  inReadyConcat += "}";

  std::vector<SVConnection> connections;

  connections.push_back({"clk", "clk"});
  connections.push_back({"rst_n", "rst_n"});

  connections.push_back({"in_data", inDataConcat});
  connections.push_back({"in_valid", inValidConcat});
  connections.push_back({"in_ready", inReadyConcat});

  // Single output.
  std::string outDataWire = ctx.getDataWire(op.getResult(0));
  std::string outValidWire = ctx.getValidWire(op.getResult(0));
  std::string outReadyWire = ctx.getReadyWire(op.getResult(0));
  connections.push_back({"out_data", outDataWire});
  connections.push_back({"out_valid", outValidWire});
  connections.push_back({"out_ready", outReadyWire});

  // Config: cfg_join_mask from fu_cfg slice.
  unsigned cfgBits = numIn;
  std::string cfgSlice;
  if (cfgBits == 1) {
    cfgSlice = "fu_cfg[" + std::to_string(ctx.cfgBitOffset) + "]";
  } else {
    cfgSlice = "fu_cfg[" +
               std::to_string(ctx.cfgBitOffset + cfgBits - 1) + ":" +
               std::to_string(ctx.cfgBitOffset) + "]";
  }
  connections.push_back({"cfg_join_mask", cfgSlice});
  ctx.cfgBitOffset += cfgBits;

  ctx.emitter.emitInstance("fu_op_join", instName, params, connections);
  ctx.emitter.emitBlankLine();
}

/// Emit a handshake.mux instantiation with special port layout.
///
/// fu_op_mux.sv ports:
///   - Parameters: NUM_DATA, WIDTH
///   - in_data_sel [WIDTH-1:0], in_valid_sel, in_ready_sel
///   - in_data [NUM_DATA-1:0][WIDTH-1:0] (packed)
///   - in_valid [NUM_DATA-1:0], in_ready [NUM_DATA-1:0]
///   - out_data, out_valid, out_ready
///
/// MLIR handshake.mux operands: [sel, data_0, data_1, ...]
static void emitHandshakeMuxInstance(FUEmitContext &ctx, mlir::Operation &op,
                                     unsigned fuDataWidth) {
  unsigned numOperands = op.getNumOperands();
  // Operand 0 is sel, rest are data inputs.
  unsigned numData = numOperands - 1;

  std::string instName =
      "fu_op_mux_inst_" + std::to_string(ctx.nextWireIdx);

  unsigned opDataWidth = 32;
  if (op.getNumResults() > 0)
    opDataWidth = getValueWidth(op.getResult(0));
  else if (numData > 0)
    opDataWidth = getValueWidth(op.getOperand(1));

  std::vector<std::string> params;
  params.push_back(".NUM_DATA(" + std::to_string(numData) + ")");
  params.push_back(".WIDTH(" + std::to_string(opDataWidth) + ")");

  // Declare per-input ready wires.
  // Operand 0 (sel) ready wire.
  std::string selReadyWire = instName + "_in_ready_sel";
  ctx.emitter.emitWire("logic", selReadyWire);
  ctx.valueConsumerReadies[op.getOperand(0)].push_back(selReadyWire);

  // Data input ready wires.
  for (unsigned i = 0; i < numData; ++i) {
    std::string readyWire = instName + "_in_ready_" + std::to_string(i);
    ctx.emitter.emitWire("logic", readyWire);
    // Data operand is at MLIR operand index i+1.
    ctx.valueConsumerReadies[op.getOperand(i + 1)].push_back(readyWire);
  }

  std::vector<SVConnection> connections;

  connections.push_back({"clk", "clk"});
  connections.push_back({"rst_n", "rst_n"});

  // Sel port (operand 0).
  connections.push_back({"in_data_sel", ctx.getDataWire(op.getOperand(0))});
  connections.push_back({"in_valid_sel", ctx.getValidWire(op.getOperand(0))});
  connections.push_back({"in_ready_sel", selReadyWire});

  // Build in_data concatenation for data inputs (packed array).
  std::string inDataConcat = "{";
  for (int i = static_cast<int>(numData) - 1; i >= 0; --i) {
    if (i < static_cast<int>(numData) - 1)
      inDataConcat += ", ";
    inDataConcat += ctx.getDataWire(op.getOperand(i + 1));
  }
  inDataConcat += "}";

  // Build in_valid concatenation for data inputs.
  std::string inValidConcat = "{";
  for (int i = static_cast<int>(numData) - 1; i >= 0; --i) {
    if (i < static_cast<int>(numData) - 1)
      inValidConcat += ", ";
    inValidConcat += ctx.getValidWire(op.getOperand(i + 1));
  }
  inValidConcat += "}";

  // Build in_ready concatenation for data inputs.
  std::string inReadyConcat = "{";
  for (int i = static_cast<int>(numData) - 1; i >= 0; --i) {
    if (i < static_cast<int>(numData) - 1)
      inReadyConcat += ", ";
    inReadyConcat += instName + "_in_ready_" + std::to_string(i);
  }
  inReadyConcat += "}";

  connections.push_back({"in_data", inDataConcat});
  connections.push_back({"in_valid", inValidConcat});
  connections.push_back({"in_ready", inReadyConcat});

  // Single output.
  std::string outDataWire = ctx.getDataWire(op.getResult(0));
  std::string outValidWire = ctx.getValidWire(op.getResult(0));
  std::string outReadyWire = ctx.getReadyWire(op.getResult(0));
  connections.push_back({"out_data", outDataWire});
  connections.push_back({"out_valid", outValidWire});
  connections.push_back({"out_ready", outReadyWire});

  ctx.emitter.emitInstance("fu_op_mux", instName, params, connections);
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
///
/// Config bits from fu_cfg are packed as [sel | discard | disconnect]
/// and driven through a 32-bit cfg_wdata word with cfg_valid tied high.
static void emitFabricMuxInstance(FUEmitContext &ctx,
                                  fcc::fabric::MuxOp muxOp,
                                  unsigned fuDataWidth) {
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

  connections.push_back({"clk", "clk"});
  connections.push_back({"rst_n", "rst_n"});

  // Config port: extract bits from fu_cfg and pack into cfg_wdata.
  unsigned selBits = numIn > 1 ? llvm::Log2_32_Ceil(numIn) : 0;
  unsigned muxCfgBits = selBits + 2;

  std::string cfgWdataWire = instName + "_cfg_wdata";
  ctx.emitter.emitWire("logic [31:0]", cfgWdataWire);

  std::string cfgSlice;
  if (muxCfgBits == 1) {
    cfgSlice = "fu_cfg[" + std::to_string(ctx.cfgBitOffset) + "]";
  } else {
    cfgSlice = "fu_cfg[" +
               std::to_string(ctx.cfgBitOffset + muxCfgBits - 1) + ":" +
               std::to_string(ctx.cfgBitOffset) + "]";
  }
  ctx.emitter.emitAssign(cfgWdataWire,
                         "{" + std::to_string(32 - muxCfgBits) +
                             "'b0, " + cfgSlice + "}");

  connections.push_back({"cfg_valid", "1'b1"});
  connections.push_back({"cfg_wdata", cfgWdataWire});

  std::string cfgReadyWire = instName + "_cfg_ready";
  ctx.emitter.emitWire("logic", cfgReadyWire);
  connections.push_back({"cfg_ready", cfgReadyWire});

  ctx.cfgBitOffset += muxCfgBits;

  // Declare per-input ready wires and register consumer readies.
  for (unsigned i = 0; i < numIn; ++i) {
    std::string readyWire = instName + "_in_ready_" + std::to_string(i);
    ctx.emitter.emitWire("logic", readyWire);
    ctx.addConsumerReady(muxOp.getInputs()[i], instName, i);
  }

  // Build in_valid concatenation: {wire_N-1, ..., wire_0}.
  std::string inValidConcat = "{";
  for (int i = static_cast<int>(numIn) - 1; i >= 0; --i) {
    if (i < static_cast<int>(numIn) - 1)
      inValidConcat += ", ";
    inValidConcat += ctx.getValidWire(muxOp.getInputs()[i]);
  }
  inValidConcat += "}";
  connections.push_back({"in_valid", inValidConcat});

  // Build in_ready concatenation.
  std::string inReadyConcat = "{";
  for (int i = static_cast<int>(numIn) - 1; i >= 0; --i) {
    if (i < static_cast<int>(numIn) - 1)
      inReadyConcat += ", ";
    inReadyConcat += instName + "_in_ready_" + std::to_string(i);
  }
  inReadyConcat += "}";
  connections.push_back({"in_ready", inReadyConcat});

  // in_data is an unpacked array -- connected via a temporary wire array.
  std::string dataArrayName = instName + "_in_data";
  ctx.emitter.emitRaw(
      "logic [" + std::to_string(dataWidth - 1) + ":0] " +
      dataArrayName + " [0:" + std::to_string(numIn - 1) + "];\n");
  for (unsigned i = 0; i < numIn; ++i) {
    std::string expr = ctx.getDataWire(muxOp.getInputs()[i]);
    ctx.emitter.emitAssign(
        dataArrayName + "[" + std::to_string(i) + "]", expr);
  }
  connections.push_back({"in_data", dataArrayName});

  // Output port (single output): connect to SSA result valid/ready.
  connections.push_back({"out_data", ctx.getDataWire(muxOp.getResult(0))});
  connections.push_back({"out_valid", ctx.getValidWire(muxOp.getResult(0))});
  connections.push_back({"out_ready", ctx.getReadyWire(muxOp.getResult(0))});

  ctx.emitter.emitInstance("fabric_mux", instName, params, connections);
  ctx.emitter.emitBlankLine();
}

/// Emit the ready-merge logic for all SSA values that have consumers.
/// For each SSA value, its ready signal = AND of all consumer ready signals.
/// If a value has no consumers (dead code), ready is tied high.
static void emitReadyMergeLogic(FUEmitContext &ctx) {
  for (auto &entry : ctx.valueConsumerReadies) {
    mlir::Value val = entry.first;
    const auto &readies = entry.second;
    std::string readyWire = ctx.getReadyWire(val);

    if (readies.empty()) {
      ctx.emitter.emitAssign(readyWire, "1'b1");
      continue;
    }
    if (readies.size() == 1) {
      ctx.emitter.emitAssign(readyWire, readies[0]);
      continue;
    }
    // AND of all consumer readies.
    std::string expr = readies[0];
    for (unsigned i = 1; i < readies.size(); ++i)
      expr += " & " + readies[i];
    ctx.emitter.emitAssign(readyWire, expr);
  }
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

  // Count total config bits across all configurable body ops.
  unsigned totalConfigBits = 0;
  auto &bodyBlock = fuOp.getBody().front();
  for (auto &op : bodyBlock.getOperations()) {
    if (mlir::isa<fcc::fabric::YieldOp>(op))
      continue;
    totalConfigBits += getOpConfigBits(op, dataWidth);
  }

  // Build parameter list.
  std::vector<std::string> params;
  params.push_back("parameter DATA_WIDTH = " + std::to_string(dataWidth));
  if (totalConfigBits > 0)
    params.push_back("parameter FU_CFG_BITS = " +
                      std::to_string(totalConfigBits));

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
  if (totalConfigBits > 0)
    ports.push_back(
        {SVPortDir::Input,
         "logic" + SVEmitter::bitRange(totalConfigBits),
         "fu_cfg"});

  emitter.emitModuleHeader(moduleName, params, ports);

  // Set up emission context.
  FUEmitContext ctx{emitter, registry, fpIpProfile, {}, 0, false, 0, {}};

  // Map block arguments (FU inputs) to port wire names and declare
  // corresponding valid/ready wires aliased to FU port signals.
  emitter.emitComment("Block argument (FU input) valid/ready aliases");
  for (unsigned i = 0; i < numIn && i < bodyBlock.getNumArguments(); ++i) {
    mlir::Value arg = bodyBlock.getArgument(i);
    std::string idx = std::to_string(i);
    // Data wire name maps directly to the FU port.
    ctx.valueNames[arg] = "in_data_" + idx;
    // Declare valid/ready wires for the block argument so they can be
    // referenced uniformly.  Valid is driven from the FU port; ready
    // will be driven by the consumer merge later.
    ctx.emitter.emitWire("logic", "in_data_" + idx + "_valid");
    ctx.emitter.emitWire("logic", "in_data_" + idx + "_ready");
    ctx.emitter.emitAssign("in_data_" + idx + "_valid",
                           "in_valid_" + idx);
  }

  emitter.emitBlankLine();
  emitter.emitComment("Internal wiring");

  // Emit all ops in body order.
  for (auto &op : bodyBlock.getOperations()) {
    llvm::StringRef opName = op.getName().getStringRef();

    // Skip the yield terminator -- handled separately.
    if (mlir::isa<fcc::fabric::YieldOp>(op))
      continue;

    // Handle fabric.mux specially (config-time structural mux).
    if (auto muxOp = mlir::dyn_cast<fcc::fabric::MuxOp>(op)) {
      emitOpResultWires(ctx, op);
      emitFabricMuxInstance(ctx, muxOp, dataWidth);
      continue;
    }

    // Handle handshake.join specially (packed array ports).
    if (opName == "handshake.join") {
      emitOpResultWires(ctx, op);
      emitJoinInstance(ctx, op, dataWidth);
      continue;
    }

    // Handle handshake.mux specially (sel + packed data array ports).
    if (opName == "handshake.mux") {
      emitOpResultWires(ctx, op);
      emitHandshakeMuxInstance(ctx, op, dataWidth);
      continue;
    }

    // For other known dialect ops, emit with standard port convention.
    if (SVModuleRegistry::isKnownOp(opName)) {
      emitOpResultWires(ctx, op);
      emitStandardOpInstance(ctx, op, dataWidth);
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

  // -------------------------------------------------------------------
  // Yield: connect yield operands to FU output ports.
  // FU out_data = yield operand data wire
  // FU out_valid = yield operand valid wire
  // yield operand ready = FU out_ready (registered as consumer)
  // -------------------------------------------------------------------
  emitter.emitBlankLine();
  emitter.emitComment("Output connections (yield)");
  if (auto yieldOp = mlir::dyn_cast<fcc::fabric::YieldOp>(
          bodyBlock.getTerminator())) {
    for (unsigned i = 0; i < yieldOp.getOperands().size() && i < numOut;
         ++i) {
      mlir::Value yieldOperand = yieldOp.getOperand(i);
      std::string idx = std::to_string(i);
      emitter.emitAssign("out_data_" + idx,
                         ctx.getDataWire(yieldOperand));
      emitter.emitAssign("out_valid_" + idx,
                         ctx.getValidWire(yieldOperand));
      // The FU output ready drives back into the yield operand's ready.
      // Register it as a consumer of the yield operand value.
      ctx.valueConsumerReadies[yieldOperand].push_back("out_ready_" + idx);
    }
  }

  // -------------------------------------------------------------------
  // SSA-driven ready backpropagation
  //
  // For each SSA value, its ready = AND(all consumer readies).
  // For block arguments, the merged ready drives the FU input ready port.
  // Values with no consumers (dead results) get ready tied high.
  // -------------------------------------------------------------------
  emitter.emitBlankLine();
  emitter.emitComment("SSA ready backpropagation");

  // Ensure all SSA values (block args + op results) have a ready driver.
  // If a value has no registered consumers, tie its ready high.
  for (unsigned i = 0; i < bodyBlock.getNumArguments(); ++i) {
    mlir::Value arg = bodyBlock.getArgument(i);
    if (ctx.valueConsumerReadies.find(arg) == ctx.valueConsumerReadies.end())
      ctx.valueConsumerReadies[arg]; // insert empty vector
  }
  for (auto &op : bodyBlock.getOperations()) {
    if (mlir::isa<fcc::fabric::YieldOp>(op))
      continue;
    for (mlir::Value result : op.getResults()) {
      if (ctx.valueConsumerReadies.find(result) ==
          ctx.valueConsumerReadies.end())
        ctx.valueConsumerReadies[result]; // insert empty vector
    }
  }

  emitReadyMergeLogic(ctx);

  // Drive FU input ready ports from block argument ready wires.
  emitter.emitBlankLine();
  emitter.emitComment("FU input ready from block argument ready");
  for (unsigned i = 0; i < numIn && i < bodyBlock.getNumArguments(); ++i) {
    std::string idx = std::to_string(i);
    emitter.emitAssign("in_ready_" + idx,
                       "in_data_" + idx + "_ready");
  }

  emitter.emitBlankLine();
  emitter.emitModuleFooter(moduleName);

  return moduleName;
}

} // namespace svgen
} // namespace fcc
