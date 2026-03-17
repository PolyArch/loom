//===-- adg.cpp - GUI test ADG generation -----------------------*- C++ -*-===//
//
// Part of the fcc project.
//
//===----------------------------------------------------------------------===//
//
// Constructs a small builder-based ADG for GUI visualization testing.
//
// The goal here is not stress-routing. It is a compact, self-consistent
// builder-based smoke test that exercises PE/FU rendering and simple mapping.
//
// The ADG uses four tiny PEs wired directly:
//   - add_pe   : arith.addi
//   - const_pe : handshake.constant
//   - mul_pe   : arith.muli
//   - join_pe  : handshake.join
//
// The module exposes three scalar inputs and two scalar outputs.
//
//===----------------------------------------------------------------------===//

#include "fcc/ADG/ADGBuilder.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

using namespace fcc::adg;

static void buildAllPEsTestADG(const std::string &outputPath) {
  ADGBuilder builder("all_pes_test_adg");

  const unsigned dataWidth = 64;

  //=== Define Function Units and single-purpose PEs ===

  auto fuAdd = builder.defineFU(
      "fu_add", {"i32", "i32"}, {"i32"}, {"arith.addi"});
  auto fuConstant = builder.defineFU(
      "fu_constant", {"none"}, {"i32"}, {"handshake.constant"});
  auto fuMul = builder.defineFU(
      "fu_mul", {"i32", "i32"}, {"i32"}, {"arith.muli"});
  auto fuJoin = builder.defineFU(
      "fu_join", {"none"}, {"none"}, {"handshake.join"});

  auto addPE = builder.defineSpatialPE("add_pe", 2, 1, dataWidth, {fuAdd});
  auto constPE = builder.defineSpatialPE("const_pe", 1, 1, dataWidth,
                                         {fuConstant});
  auto mulPE = builder.defineSpatialPE("mul_pe", 2, 1, dataWidth, {fuMul});
  auto joinPE = builder.defineSpatialPE("join_pe", 1, 1, dataWidth, {fuJoin});

  auto addInst = builder.instantiatePE(addPE, "pe_add");
  auto constInst = builder.instantiatePE(constPE, "pe_const");
  auto mulInst = builder.instantiatePE(mulPE, "pe_mul");
  auto joinInst = builder.instantiatePE(joinPE, "pe_join");

  //=== Add scalar boundary inputs and wire to the PEs ===

  auto sIn0 = builder.addScalarInput("scalar_in0", dataWidth);
  auto sIn1 = builder.addScalarInput("scalar_in1", dataWidth);
  auto sCtrl = builder.addScalarInput("scalar_ctrl", dataWidth);

  builder.connectScalarInputToInstance(sIn0, addInst, 0);
  builder.connectScalarInputToInstance(sIn1, addInst, 1);
  builder.connectScalarInputToInstance(sCtrl, constInst, 0);
  builder.connectScalarInputToInstance(sCtrl, joinInst, 0);

  //=== Wire the PE dataflow graph directly ===

  builder.connect(addInst, 0, mulInst, 0);
  builder.connect(constInst, 0, mulInst, 1);

  //=== Add scalar boundary outputs ===

  auto sOut0 = builder.addScalarOutput("scalar_result", dataWidth);
  auto sOut1 = builder.addScalarOutput("scalar_done", dataWidth);
  builder.connectInstanceToScalarOutput(mulInst, 0, sOut0);
  builder.connectInstanceToScalarOutput(joinInst, 0, sOut1);

  //=== Export ===

  builder.exportMLIR(outputPath);
}

//===----------------------------------------------------------------------===//
// Standalone tool entry point
//===----------------------------------------------------------------------===//

static llvm::cl::opt<std::string> outputFile(
    llvm::cl::Positional,
    llvm::cl::desc("<output .fabric.mlir file>"),
    llvm::cl::init("all-pes.fabric.mlir"));

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "GUI test ADG generator\n");

  // Ensure output directory exists
  auto parentPath = llvm::sys::path::parent_path(outputFile);
  if (!parentPath.empty()) {
    if (auto ec = llvm::sys::fs::create_directories(parentPath)) {
      llvm::errs() << "error: cannot create output directory: "
                   << parentPath << "\n";
      return 1;
    }
  }

  llvm::outs() << "Generating all-pes test ADG -> " << outputFile << "\n";
  buildAllPEsTestADG(outputFile);
  llvm::outs() << "Done.\n";

  return 0;
}
