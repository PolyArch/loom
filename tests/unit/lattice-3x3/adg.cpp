//===-- adg.cpp - Lattice mesh builder test ---------------------*- C++ -*-===//
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

static void buildLatticeADG(const std::string &outputPath) {
  ADGBuilder builder("lattice_3x3_test");
  constexpr unsigned dataWidth = 64;

  auto fuAdd = builder.defineBinaryFU("fu_add", "arith.addi", "i32", "i32");

  auto pe =
      builder.defineSingleFUSpatialPE("lattice_add_pe", 4, 4, dataWidth, fuAdd);

  LatticeMeshOptions options;
  options.topLeftExtraInputs = 2;
  options.bottomRightExtraOutputs = 1;
  auto mesh = builder.buildLatticeMesh(3, 3, pe, options);

  auto in0 = builder.addScalarInput("a", dataWidth);
  auto in1 = builder.addScalarInput("b", dataWidth);
  auto out0 = builder.addScalarOutput("result", dataWidth);

  builder.connectInputToPort(in0, mesh.ingressPorts[0]);
  builder.connectInputToPort(in1, mesh.ingressPorts[1]);
  builder.connectPortToOutput(mesh.egressPorts[0], out0);

  builder.exportMLIR(outputPath);
}

static llvm::cl::opt<std::string> outputFile(
    llvm::cl::Positional, llvm::cl::desc("<output .fabric.mlir file>"),
    llvm::cl::init("lattice-3x3.fabric.mlir"));

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "Lattice mesh ADG generator\n");

  auto parentPath = llvm::sys::path::parent_path(outputFile);
  if (!parentPath.empty()) {
    if (auto ec = llvm::sys::fs::create_directories(parentPath)) {
      llvm::errs() << "error: cannot create output directory: " << parentPath
                   << "\n";
      return 1;
    }
  }

  buildLatticeADG(outputFile);
  return 0;
}
