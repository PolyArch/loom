//===-- adg.cpp - Raw FU body builder test ----------------------*- C++ -*-===//
//
// Part of the fcc project.
//
//===----------------------------------------------------------------------===//

#include "fcc/ADG/ADGBuilder.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

using namespace fcc::adg;

static void buildRawFUADG(const std::string &outputPath) {
  ADGBuilder builder("builder_raw_fu_test");
  constexpr unsigned dataWidth = 64;

  std::string rawBody;
  rawBody += "%m = arith.muli %arg0, %arg1 : i32\n";
  rawBody += "%s = arith.addi %m, %arg2 : i32\n";
  rawBody += "%o = fabric.mux %m, %s {sel = 1 : i64, discard = false, disconnect = false} : i32, i32 -> i32\n";
  rawBody += "fabric.yield %o : i32\n";

  auto fuMac = builder.defineFUWithBody("fu_mac", {"i32", "i32", "i32"},
                                        {"i32"}, rawBody);

  SpatialPESpec peSpec;
  peSpec.name = "raw_fu_pe";
  peSpec.numInputs = 3;
  peSpec.numOutputs = 1;
  peSpec.bitsWidth = dataWidth;
  peSpec.functionUnits = {fuMac};
  auto pe = builder.defineSpatialPE(peSpec);

  auto inst = builder.instantiatePE(pe, "pe_0");
  auto in0 = builder.addScalarInput("a", dataWidth);
  auto in1 = builder.addScalarInput("b", dataWidth);
  auto in2 = builder.addScalarInput("c", dataWidth);
  auto out0 = builder.addScalarOutput("result", dataWidth);

  builder.connectInputVectorToInstance({in0, in1, in2}, inst);
  builder.connectInstanceToOutputVector(inst, 0, {out0});

  builder.exportMLIR(outputPath);
}

static llvm::cl::opt<std::string> outputFile(
    llvm::cl::Positional, llvm::cl::desc("<output .fabric.mlir file>"),
    llvm::cl::init("builder-raw-fu.fabric.mlir"));

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv, "Raw FU body ADG generator\n");

  auto parentPath = llvm::sys::path::parent_path(outputFile);
  if (!parentPath.empty()) {
    if (auto ec = llvm::sys::fs::create_directories(parentPath)) {
      llvm::errs() << "error: cannot create output directory: " << parentPath
                   << "\n";
      return 1;
    }
  }

  buildRawFUADG(outputFile);
  return 0;
}
