//===-- adg.cpp - Cube topology builder test --------------------*- C++ -*-===//
//
// Part of the loom project.
//
//===----------------------------------------------------------------------===//

#include "loom/ADG/ADGBuilder.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

using namespace loom::adg;

static void buildCubeADG(const std::string &outputPath) {
  ADGBuilder builder("cube_2x2x2_test");
  constexpr unsigned dataWidth = 64;

  auto fuAdd = builder.defineBinaryFU("fu_add", "arith.addi", "i32", "i32");

  auto pe =
      builder.defineSingleFUSpatialPE("cube_add_pe", 8, 8, dataWidth, fuAdd);

  CubeOptions options;
  options.originExtraInputs = 2;
  options.farCornerExtraOutputs = 1;
  auto cube = builder.buildCube(2, 2, 2, pe, options);

  auto in0 = builder.addScalarInput("a", dataWidth);
  auto in1 = builder.addScalarInput("b", dataWidth);
  auto out0 = builder.addScalarOutput("result", dataWidth);

  builder.connectInputToPort(in0, cube.ingressPorts[0]);
  builder.connectInputToPort(in1, cube.ingressPorts[1]);
  builder.connectPortToOutput(cube.egressPorts[0], out0);

  builder.exportMLIR(outputPath);
}

static llvm::cl::opt<std::string> outputFile(
    llvm::cl::Positional, llvm::cl::desc("<output .fabric.mlir file>"),
    llvm::cl::init("cube-2x2x2.fabric.mlir"));

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv, "Cube ADG generator\n");

  auto parentPath = llvm::sys::path::parent_path(outputFile);
  if (!parentPath.empty()) {
    if (auto ec = llvm::sys::fs::create_directories(parentPath)) {
      llvm::errs() << "error: cannot create output directory: " << parentPath
                   << "\n";
      return 1;
    }
  }

  buildCubeADG(outputFile);
  return 0;
}
