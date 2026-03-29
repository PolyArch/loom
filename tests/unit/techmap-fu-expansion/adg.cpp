//===-- adg.cpp - Tech-mapping FU body expansion test --------*- C++ -*-===//
//
// Part of the loom project.
//
//===----------------------------------------------------------------------===//
//
// Builds an ADG with multi-op FU body archetypes for tech-mapping expansion:
//   - MAC: arith.muli feeding arith.addi (with mux to select mul-only or mac)
//   - Compare-select: arith.cmpi feeding arith.select (for min/max patterns)
//   - Simple addi for commutative swap variant testing
//
//===----------------------------------------------------------------------===//

#include "loom/ADG/ADGBuilder.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

using namespace loom::adg;

static void buildTechmapFUExpansionADG(const std::string &outputPath) {
  ADGBuilder builder("techmap_fu_expansion_test");
  constexpr unsigned dataWidth = 32;

  // MAC FU body: muli -> addi, with a mux to select muli-only or mac output.
  std::string macBody;
  macBody += "%m = arith.muli %arg0, %arg1 : i32\n";
  macBody += "%s = arith.addi %m, %arg2 : i32\n";
  macBody += "%o = fabric.mux %m, %s "
             "{sel = 1 : i64, discard = false, disconnect = false} "
             ": i32, i32 -> i32\n";
  macBody += "fabric.yield %o : i32\n";
  auto fuMac = builder.defineFUWithBody(
      "fu_mac", {"i32", "i32", "i32"}, {"i32"}, macBody);

  // Compare-select FU body: arith.cmpi feeding arith.select.
  // Covers min/max patterns via configurable predicate.
  std::string cmpSelBody;
  cmpSelBody += "%cmp = arith.cmpi slt, %arg0, %arg1 : i32\n";
  cmpSelBody += "%sel = arith.select %cmp, %arg0, %arg1 : i32\n";
  cmpSelBody += "fabric.yield %sel : i32\n";
  auto fuCmpSel = builder.defineFUWithBody(
      "fu_cmp_sel", {"i32", "i32"}, {"i32"}, cmpSelBody);

  // Simple binary addi FU for commutative swap variant testing.
  auto fuAddi = builder.defineBinaryFU(
      "fu_addi", "arith.addi", "i32", "i32");

  // Build PEs, each with a single FU.
  SpatialPESpec macPESpec;
  macPESpec.name = "mac_pe";
  macPESpec.numInputs = 3;
  macPESpec.numOutputs = 1;
  macPESpec.bitsWidth = dataWidth;
  macPESpec.functionUnits = {fuMac};
  auto macPE = builder.defineSpatialPE(macPESpec);

  SpatialPESpec cmpSelPESpec;
  cmpSelPESpec.name = "cmp_sel_pe";
  cmpSelPESpec.numInputs = 2;
  cmpSelPESpec.numOutputs = 1;
  cmpSelPESpec.bitsWidth = dataWidth;
  cmpSelPESpec.functionUnits = {fuCmpSel};
  auto cmpSelPE = builder.defineSpatialPE(cmpSelPESpec);

  SpatialPESpec addiPESpec;
  addiPESpec.name = "addi_pe";
  addiPESpec.numInputs = 2;
  addiPESpec.numOutputs = 1;
  addiPESpec.bitsWidth = dataWidth;
  addiPESpec.functionUnits = {fuAddi};
  auto addiPE = builder.defineSpatialPE(addiPESpec);

  // Switch for routing. 3 inputs from module, 3 outputs from PEs going back
  // in, plus 3 outputs to PEs, plus 1 output to module. Use 6 inputs, 6
  // outputs with full connectivity.
  SpatialSWSpec swSpec;
  swSpec.name = "sw_main";
  swSpec.inputTypes = {};
  swSpec.outputTypes = {};
  swSpec.connectivity = {};
  unsigned swPorts = 6;
  for (unsigned iter_var0 = 0; iter_var0 < swPorts; ++iter_var0) {
    swSpec.inputTypes.push_back("!fabric.bits<32>");
    std::vector<bool> row;
    for (unsigned iter_var1 = 0; iter_var1 < swPorts; ++iter_var1)
      row.push_back(true);
    swSpec.connectivity.push_back(row);
  }
  for (unsigned iter_var0 = 0; iter_var0 < swPorts; ++iter_var0)
    swSpec.outputTypes.push_back("!fabric.bits<32>");
  auto sw = builder.defineSpatialSW(swSpec);

  // Module-level I/O.
  auto inA = builder.addScalarInput("a", dataWidth);
  auto inB = builder.addScalarInput("b", dataWidth);
  auto inC = builder.addScalarInput("c", dataWidth);
  auto outMac = builder.addScalarOutput("mac_result", dataWidth);
  auto outCmpSel = builder.addScalarOutput("cmp_sel_result", dataWidth);
  auto outAddi = builder.addScalarOutput("addi_result", dataWidth);

  // Instantiate PEs and switch.
  auto macInst = builder.instantiatePE(macPE, "pe_mac");
  auto cmpSelInst = builder.instantiatePE(cmpSelPE, "pe_cmp_sel");
  auto addiInst = builder.instantiatePE(addiPE, "pe_addi");
  auto swInst = builder.instantiateSW(sw, "sw_0");

  // Wire module inputs to switch.
  builder.connectInputToInstance(inA, swInst, 0);
  builder.connectInputToInstance(inB, swInst, 1);
  builder.connectInputToInstance(inC, swInst, 2);

  // Switch outputs to PE inputs.
  builder.connect(swInst, 0, macInst, 0);   // a -> mac.in0
  builder.connect(swInst, 1, macInst, 1);   // b -> mac.in1
  builder.connect(swInst, 2, macInst, 2);   // c -> mac.in2
  builder.connect(swInst, 0, cmpSelInst, 0);  // a -> cmpsel.in0
  builder.connect(swInst, 1, cmpSelInst, 1);  // b -> cmpsel.in1
  builder.connect(swInst, 0, addiInst, 0);    // a -> addi.in0
  builder.connect(swInst, 1, addiInst, 1);    // b -> addi.in1

  // PE outputs back to switch inputs.
  builder.connect(macInst, 0, swInst, 3);
  builder.connect(cmpSelInst, 0, swInst, 4);
  builder.connect(addiInst, 0, swInst, 5);

  // Switch outputs to module outputs.
  builder.connectInstanceToOutputVector(swInst, 3, {outMac});
  builder.connectInstanceToOutputVector(swInst, 4, {outCmpSel});
  builder.connectInstanceToOutputVector(swInst, 5, {outAddi});

  builder.exportMLIR(outputPath);
}

static llvm::cl::opt<std::string> outputFile(
    llvm::cl::Positional,
    llvm::cl::desc("<output .fabric.mlir file>"),
    llvm::cl::init("techmap-fu-expansion.fabric.mlir"));

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(
      argc, argv, "Tech-mapping FU expansion ADG generator\n");

  auto parentPath = llvm::sys::path::parent_path(outputFile);
  if (!parentPath.empty()) {
    if (auto ec = llvm::sys::fs::create_directories(parentPath)) {
      llvm::errs() << "error: cannot create output directory: "
                   << parentPath << "\n";
      return 1;
    }
  }

  buildTechmapFUExpansionADG(outputFile);
  return 0;
}
