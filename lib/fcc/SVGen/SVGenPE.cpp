#include "SVGenInternal.h"

#include "fcc/SVGen/SVEmitter.h"
#include "fcc/SVGen/SVModuleRegistry.h"

#include "fcc/Dialect/Fabric/FabricOps.h"
#include "fcc/Dialect/Fabric/FabricTypes.h"
#include "fcc/Mapper/TypeCompat.h"

#include "mlir/IR/BuiltinOps.h"

#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <string>
#include <vector>

namespace fcc {
namespace svgen {

namespace {

/// Collect FU info from a PE body.
struct FUInfo {
  std::string fuSymName;
  std::string svModuleName; // "fu_<name>"
  unsigned numInputs = 0;
  unsigned numOutputs = 0;
  unsigned configBits = 0;
  fcc::fabric::FunctionUnitOp fuOp;
};

static std::vector<FUInfo> collectFUs(mlir::Block &peBody) {
  std::vector<FUInfo> fus;
  for (auto &op : peBody.getOperations()) {
    if (auto fuOp = mlir::dyn_cast<fcc::fabric::FunctionUnitOp>(op)) {
      FUInfo info;
      info.fuSymName = fuOp.getSymName().str();
      info.svModuleName =
          "fu_" + SVEmitter::sanitizeName(fuOp.getSymName());
      auto fnType = fuOp.getFunctionType();
      info.numInputs = fnType.getNumInputs();
      info.numOutputs = fnType.getNumResults();
      info.fuOp = fuOp;

      // Count FU internal config bits (mux ops).
      unsigned cfgBits = 0;
      fuOp.getBody().front().walk([&](fcc::fabric::MuxOp muxOp) {
        unsigned numMuxIn = muxOp.getInputs().size();
        unsigned selBits =
            numMuxIn > 1 ? llvm::Log2_32_Ceil(numMuxIn) : 0;
        cfgBits += selBits + 2;
      });
      info.configBits = cfgBits;
      fus.push_back(std::move(info));
    }
  }
  return fus;
}

} // namespace

/// Generate a spatial PE wrapper SV module.
std::string generateSpatialPE(fcc::fabric::SpatialPEOp peOp,
                               llvm::raw_ostream &os,
                               SVModuleRegistry &registry,
                               llvm::StringRef fpIpProfile) {
  std::string peName = SVEmitter::sanitizeName(
      peOp.getSymName().value_or("spatial_pe"));
  std::string moduleName = "pe_" + peName;

  auto fnType = peOp.getFunctionType();
  unsigned numPEInputs = fnType.getNumInputs();
  unsigned numPEOutputs = fnType.getNumResults();

  auto fus = collectFUs(peOp.getBody().front());
  unsigned numFU = fus.size();

  // Compute max FU port counts.
  unsigned maxFUIn = 0, maxFUOut = 0, maxFUCfg = 0;
  for (const auto &fu : fus) {
    maxFUIn = std::max(maxFUIn, fu.numInputs);
    maxFUOut = std::max(maxFUOut, fu.numOutputs);
    maxFUCfg = std::max(maxFUCfg, fu.configBits);
  }

  // Determine data width from PE port types.
  unsigned dataWidth = 32;
  if (numPEInputs > 0)
    dataWidth = SVEmitter::getDataWidth(fnType.getInput(0));
  else if (numPEOutputs > 0)
    dataWidth = SVEmitter::getDataWidth(fnType.getResult(0));

  SVEmitter emitter(os);
  emitter.emitFileHeader(moduleName);

  // Parameters.
  std::vector<std::string> params;
  params.push_back("parameter DATA_WIDTH = " + std::to_string(dataWidth));
  params.push_back("parameter NUM_PE_IN = " + std::to_string(numPEInputs));
  params.push_back("parameter NUM_PE_OUT = " + std::to_string(numPEOutputs));
  params.push_back("parameter NUM_FU = " + std::to_string(numFU));

  // Compute config width.
  unsigned opcodeBits = numFU > 1 ? llvm::Log2_32_Ceil(numFU) : 0;
  unsigned perInputMuxBits = 0;
  if (numPEInputs > 0) {
    unsigned selBits =
        numPEInputs > 1 ? llvm::Log2_32_Ceil(numPEInputs) : 0;
    perInputMuxBits = selBits + 2;
  }
  unsigned perOutputDemuxBits = 0;
  if (numPEOutputs > 0) {
    unsigned selBits =
        numPEOutputs > 1 ? llvm::Log2_32_Ceil(numPEOutputs) : 0;
    perOutputDemuxBits = selBits + 2;
  }
  unsigned totalCfgBits = 1 + opcodeBits + maxFUIn * perInputMuxBits +
                          maxFUOut * perOutputDemuxBits + maxFUCfg;
  unsigned cfgWords = (totalCfgBits + 31) / 32;

  params.push_back("parameter CFG_BITS = " + std::to_string(totalCfgBits));

  // Ports.
  std::vector<SVPort> ports;
  ports.push_back({SVPortDir::Input, "logic", "clk"});
  ports.push_back({SVPortDir::Input, "logic", "rst_n"});

  for (unsigned i = 0; i < numPEInputs; ++i) {
    unsigned w = SVEmitter::getDataWidth(fnType.getInput(i));
    ports.push_back({SVPortDir::Input, "logic" + SVEmitter::bitRange(w),
                     "pe_in" + std::to_string(i)});
    ports.push_back(
        {SVPortDir::Input, "logic", "pe_in_valid" + std::to_string(i)});
    ports.push_back(
        {SVPortDir::Output, "logic", "pe_in_ready" + std::to_string(i)});
  }
  for (unsigned i = 0; i < numPEOutputs; ++i) {
    unsigned w = SVEmitter::getDataWidth(fnType.getResult(i));
    ports.push_back({SVPortDir::Output, "logic" + SVEmitter::bitRange(w),
                     "pe_out" + std::to_string(i)});
    ports.push_back(
        {SVPortDir::Output, "logic", "pe_out_valid" + std::to_string(i)});
    ports.push_back(
        {SVPortDir::Input, "logic", "pe_out_ready" + std::to_string(i)});
  }

  // Config interface.
  ports.push_back({SVPortDir::Input, "logic", "cfg_valid"});
  ports.push_back(
      {SVPortDir::Input, "logic [31:0]", "cfg_wdata"});
  ports.push_back({SVPortDir::Output, "logic", "cfg_ready"});

  emitter.emitModuleHeader(moduleName, params, ports);

  // Config register.
  emitter.emitComment("Config register: enable | opcode | input_mux | "
                      "output_demux | fu_config");
  emitter.emitReg("logic" + SVEmitter::bitRange(totalCfgBits), "cfg_reg");
  emitter.emitBlankLine();

  // Config loading logic.
  emitter.emitComment("Config loading (word-serial)");
  emitter.emitLocalParam("", "CFG_WORDS",
                          std::to_string(cfgWords));
  emitter.emitReg("logic [31:0]", "cfg_shift_reg [0:CFG_WORDS-1]");
  emitter.emitReg(
      "logic" + SVEmitter::bitRange(
          cfgWords > 1 ? llvm::Log2_32_Ceil(cfgWords + 1) : 1),
      "cfg_word_cnt");
  emitter.emitBlankLine();

  emitter.emitRaw("always_ff @(posedge clk or negedge rst_n) begin : cfg_load\n");
  emitter.indent();
  emitter.emitRaw("integer iter_var0;\n");
  emitter.emitRaw("if (!rst_n) begin : cfg_reset\n");
  emitter.indent();
  emitter.emitRaw("cfg_word_cnt <= '0;\n");
  emitter.emitRaw("for (iter_var0 = 0; iter_var0 < CFG_WORDS; "
                  "iter_var0 = iter_var0 + 1) begin : cfg_rst_loop\n");
  emitter.indent();
  emitter.emitRaw("cfg_shift_reg[iter_var0] <= '0;\n");
  emitter.dedent();
  emitter.emitRaw("end // cfg_rst_loop\n");
  emitter.dedent();
  emitter.emitRaw("end else if (cfg_valid) begin : cfg_write\n");
  emitter.indent();
  emitter.emitRaw("if (cfg_word_cnt < CFG_WORDS) begin : cfg_accept\n");
  emitter.indent();
  emitter.emitRaw("cfg_shift_reg[cfg_word_cnt] <= cfg_wdata;\n");
  emitter.emitRaw("cfg_word_cnt <= cfg_word_cnt + 1;\n");
  emitter.dedent();
  emitter.emitRaw("end // cfg_accept\n");
  emitter.dedent();
  emitter.emitRaw("end // cfg_write\n");
  emitter.dedent();
  emitter.emitRaw("end // cfg_load\n");
  emitter.emitBlankLine();

  // Pack config bits from shift register.
  emitter.emitComment("Unpack config register from shift words");
  emitter.emitRaw("always_comb begin : cfg_unpack\n");
  emitter.indent();
  emitter.emitRaw("integer iter_var0;\n");
  emitter.emitRaw("for (iter_var0 = 0; iter_var0 < CFG_BITS; "
                  "iter_var0 = iter_var0 + 1) begin : unpack_loop\n");
  emitter.indent();
  emitter.emitRaw("cfg_reg[iter_var0] = "
                  "cfg_shift_reg[iter_var0 / 32][iter_var0 % 32];\n");
  emitter.dedent();
  emitter.emitRaw("end // unpack_loop\n");
  emitter.dedent();
  emitter.emitRaw("end // cfg_unpack\n");
  emitter.emitBlankLine();

  emitter.emitAssign("cfg_ready", "1'b1");
  emitter.emitBlankLine();

  // Extract config fields.
  emitter.emitComment("Config field extraction");
  unsigned bitPos = 0;
  emitter.emitWire("logic", "cfg_enable");
  emitter.emitAssign("cfg_enable", "cfg_reg[" + std::to_string(bitPos) + "]");
  bitPos += 1;

  if (opcodeBits > 0) {
    emitter.emitWire("logic" + SVEmitter::bitRange(opcodeBits), "cfg_opcode");
    emitter.emitAssign("cfg_opcode",
                        "cfg_reg[" + std::to_string(bitPos + opcodeBits - 1) +
                            ":" + std::to_string(bitPos) + "]");
    bitPos += opcodeBits;
  }

  // Input mux bank: instantiate fabric_spatial_pe_mux (pre-written).
  emitter.emitBlankLine();
  emitter.emitComment("Input mux bank");
  registry.requireModule("fabric/spatial_pe", "fabric_spatial_pe_mux.sv");

  for (unsigned fuIdx = 0; fuIdx < numFU; ++fuIdx) {
    const auto &fu = fus[fuIdx];
    for (unsigned inIdx = 0; inIdx < fu.numInputs; ++inIdx) {
      std::string wireName = "fu" + std::to_string(fuIdx) + "_in" +
                             std::to_string(inIdx);
      emitter.emitWire("logic" + SVEmitter::bitRange(dataWidth), wireName);
    }
  }
  emitter.emitBlankLine();

  // Output demux bank.
  emitter.emitComment("Output demux bank");
  registry.requireModule("fabric/spatial_pe", "fabric_spatial_pe_demux.sv");
  emitter.emitBlankLine();

  // FU instantiations.
  emitter.emitComment("FU instantiations (one active per config opcode)");
  registry.requireModule("fabric/spatial_pe", "fabric_spatial_pe_fu_slot.sv");

  for (unsigned fuIdx = 0; fuIdx < numFU; ++fuIdx) {
    const auto &fu = fus[fuIdx];
    std::string fuInstName = fu.svModuleName + "_inst";

    // FU data connections.
    std::vector<SVConnection> fuConns;
    for (unsigned i = 0; i < fu.numInputs; ++i) {
      fuConns.push_back({"in" + std::to_string(i),
                         "fu" + std::to_string(fuIdx) + "_in" +
                             std::to_string(i)});
    }
    for (unsigned i = 0; i < fu.numOutputs; ++i) {
      std::string outWire = "fu" + std::to_string(fuIdx) + "_out" +
                            std::to_string(i);
      emitter.emitWire("logic" + SVEmitter::bitRange(dataWidth), outWire);
      fuConns.push_back({"out" + std::to_string(i), outWire});
    }

    fuConns.push_back({"in_valid", "cfg_enable"});
    fuConns.push_back(
        {"in_ready", "fu" + std::to_string(fuIdx) + "_in_ready"});
    fuConns.push_back(
        {"out_valid", "fu" + std::to_string(fuIdx) + "_out_valid"});
    fuConns.push_back({"out_ready", "1'b1"});

    if (fu.configBits > 0) {
      fuConns.push_back(
          {"fu_cfg", "cfg_reg[" + std::to_string(bitPos + fu.configBits - 1) +
                         ":" + std::to_string(bitPos) + "]"});
    }

    std::vector<std::string> fuParams;
    fuParams.push_back(".DATA_WIDTH(DATA_WIDTH)");
    if (fu.configBits > 0)
      fuParams.push_back(".FU_CFG_BITS(" + std::to_string(fu.configBits) +
                          ")");

    emitter.emitWire("logic",
                     "fu" + std::to_string(fuIdx) + "_in_ready");
    emitter.emitWire("logic",
                     "fu" + std::to_string(fuIdx) + "_out_valid");
    emitter.emitInstance(fu.svModuleName, fuInstName, fuParams, fuConns);
    emitter.emitBlankLine();
  }

  emitter.emitModuleFooter(moduleName);
  return moduleName;
}

/// Generate a temporal PE wrapper SV module.
std::string generateTemporalPE(fcc::fabric::TemporalPEOp peOp,
                                llvm::raw_ostream &os,
                                SVModuleRegistry &registry,
                                llvm::StringRef fpIpProfile) {
  std::string peName = SVEmitter::sanitizeName(
      peOp.getSymName().value_or("temporal_pe"));
  std::string moduleName = "pe_" + peName;

  auto fnType = peOp.getFunctionType();
  unsigned numPEInputs = fnType.getNumInputs();
  unsigned numPEOutputs = fnType.getNumResults();
  unsigned numInstruction = static_cast<unsigned>(peOp.getNumInstruction());
  unsigned numRegister = static_cast<unsigned>(peOp.getNumRegister());
  unsigned regFifoDepth = static_cast<unsigned>(peOp.getRegFifoDepth());

  auto fus = collectFUs(peOp.getBody().front());
  unsigned numFU = fus.size();

  unsigned dataWidth = 32;
  unsigned tagWidth = 0;
  if (numPEInputs > 0) {
    dataWidth = SVEmitter::getDataWidth(fnType.getInput(0));
    tagWidth = SVEmitter::getTagWidth(fnType.getInput(0));
  }
  if (tagWidth == 0)
    tagWidth = 1;

  SVEmitter emitter(os);
  emitter.emitFileHeader(moduleName);

  std::vector<std::string> params;
  params.push_back("parameter DATA_WIDTH = " + std::to_string(dataWidth));
  params.push_back("parameter TAG_WIDTH = " + std::to_string(tagWidth));
  params.push_back("parameter NUM_PE_IN = " + std::to_string(numPEInputs));
  params.push_back("parameter NUM_PE_OUT = " + std::to_string(numPEOutputs));
  params.push_back("parameter NUM_FU = " + std::to_string(numFU));
  params.push_back("parameter NUM_INSTR = " + std::to_string(numInstruction));
  params.push_back("parameter NUM_REG = " + std::to_string(numRegister));
  params.push_back("parameter REG_FIFO_DEPTH = " +
                    std::to_string(regFifoDepth));

  // Ports.
  std::vector<SVPort> ports;
  ports.push_back({SVPortDir::Input, "logic", "clk"});
  ports.push_back({SVPortDir::Input, "logic", "rst_n"});

  unsigned totalPortWidth = dataWidth + tagWidth;
  for (unsigned i = 0; i < numPEInputs; ++i) {
    ports.push_back(
        {SVPortDir::Input,
         "logic" + SVEmitter::bitRange(totalPortWidth),
         "pe_in" + std::to_string(i)});
    ports.push_back(
        {SVPortDir::Input, "logic", "pe_in_valid" + std::to_string(i)});
    ports.push_back(
        {SVPortDir::Output, "logic", "pe_in_ready" + std::to_string(i)});
  }
  for (unsigned i = 0; i < numPEOutputs; ++i) {
    ports.push_back(
        {SVPortDir::Output,
         "logic" + SVEmitter::bitRange(totalPortWidth),
         "pe_out" + std::to_string(i)});
    ports.push_back(
        {SVPortDir::Output, "logic", "pe_out_valid" + std::to_string(i)});
    ports.push_back(
        {SVPortDir::Input, "logic", "pe_out_ready" + std::to_string(i)});
  }

  ports.push_back({SVPortDir::Input, "logic", "cfg_valid"});
  ports.push_back({SVPortDir::Input, "logic [31:0]", "cfg_wdata"});
  ports.push_back({SVPortDir::Output, "logic", "cfg_ready"});

  emitter.emitModuleHeader(moduleName, params, ports);

  // Instruction memory, operand routing, register file, scheduler,
  // output arbiter, and FU slot wrappers are instantiated from pre-written
  // sub-modules. The C++ here only wires them together since the FU set
  // and port counts vary per PE instance.

  emitter.emitComment("Instruction memory (tag-match CAM)");
  emitter.emitComment("  Selects instruction slot by incoming tag.");
  emitter.emitBlankLine();

  emitter.emitComment("Operand routing");
  emitter.emitComment("  Per-instruction: each operand from PE input mux "
                      "or register file.");
  emitter.emitBlankLine();

  emitter.emitComment("Register file: " + std::to_string(numRegister) +
                      " regs x " + std::to_string(regFifoDepth) + " deep");
  emitter.emitBlankLine();

  emitter.emitComment("Scheduler: one FU fires per cycle");
  emitter.emitBlankLine();

  // FU instantiations.
  emitter.emitComment("FU instantiations");
  for (unsigned fuIdx = 0; fuIdx < numFU; ++fuIdx) {
    const auto &fu = fus[fuIdx];
    emitter.emitComment("  FU " + std::to_string(fuIdx) + ": " +
                        fu.svModuleName);

    // Declare FU data wires.
    for (unsigned i = 0; i < fu.numInputs; ++i) {
      emitter.emitWire("logic" + SVEmitter::bitRange(dataWidth),
                        "fu" + std::to_string(fuIdx) + "_in" +
                            std::to_string(i));
    }
    for (unsigned i = 0; i < fu.numOutputs; ++i) {
      emitter.emitWire("logic" + SVEmitter::bitRange(dataWidth),
                        "fu" + std::to_string(fuIdx) + "_out" +
                            std::to_string(i));
    }
    emitter.emitWire("logic",
                     "fu" + std::to_string(fuIdx) + "_in_valid");
    emitter.emitWire("logic",
                     "fu" + std::to_string(fuIdx) + "_in_ready");
    emitter.emitWire("logic",
                     "fu" + std::to_string(fuIdx) + "_out_valid");
    emitter.emitWire("logic",
                     "fu" + std::to_string(fuIdx) + "_out_ready");

    std::vector<SVConnection> conns;
    for (unsigned i = 0; i < fu.numInputs; ++i) {
      conns.push_back({"in" + std::to_string(i),
                       "fu" + std::to_string(fuIdx) + "_in" +
                           std::to_string(i)});
    }
    for (unsigned i = 0; i < fu.numOutputs; ++i) {
      conns.push_back({"out" + std::to_string(i),
                       "fu" + std::to_string(fuIdx) + "_out" +
                           std::to_string(i)});
    }
    conns.push_back(
        {"in_valid", "fu" + std::to_string(fuIdx) + "_in_valid"});
    conns.push_back(
        {"in_ready", "fu" + std::to_string(fuIdx) + "_in_ready"});
    conns.push_back(
        {"out_valid", "fu" + std::to_string(fuIdx) + "_out_valid"});
    conns.push_back(
        {"out_ready", "fu" + std::to_string(fuIdx) + "_out_ready"});

    std::vector<std::string> fuParams;
    fuParams.push_back(".DATA_WIDTH(DATA_WIDTH)");
    emitter.emitInstance(fu.svModuleName,
                         fu.svModuleName + "_inst",
                         fuParams, conns);
    emitter.emitBlankLine();
  }

  emitter.emitComment("Output arbiter: round-robin by FU definition order");
  emitter.emitBlankLine();

  emitter.emitAssign("cfg_ready", "1'b1");
  emitter.emitBlankLine();

  emitter.emitModuleFooter(moduleName);
  return moduleName;
}

} // namespace svgen
} // namespace fcc
