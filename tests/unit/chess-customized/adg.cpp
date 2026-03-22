//===-- adg.cpp - Customized chess mesh builder test ------------*- C++ -*-===//
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

static void buildCustomizedChessADG(const std::string &outputPath) {
  ADGBuilder builder("chess_customized_test");
  constexpr unsigned dataWidth = 64;

  auto fuAdd = builder.defineBinaryFU("fu_add", "arith.addi", "i32", "i32");
  auto fuMul = builder.defineBinaryFU("fu_mul", "arith.muli", "i32", "i32");

  auto addPE =
      builder.defineSingleFUSpatialPE("chess_add_pe", 4, 4, dataWidth, fuAdd);
  auto mulPE =
      builder.defineSingleFUSpatialPE("chess_mul_pe", 4, 4, dataWidth, fuMul);

  ChessMeshOptions options;
  options.topLeftExtraInputs = 2;
  options.bottomRightExtraOutputs = 1;
  auto mesh = builder.buildChessMesh(
      2, 2,
      [&](unsigned row, unsigned col) {
        if (row == 1 && col == 1)
          return mulPE;
        return addPE;
      },
      options);

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
    llvm::cl::init("chess-customized.fabric.mlir"));

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "Customized chess mesh ADG generator\n");

  auto parentPath = llvm::sys::path::parent_path(outputFile);
  if (!parentPath.empty()) {
    if (auto ec = llvm::sys::fs::create_directories(parentPath)) {
      llvm::errs() << "error: cannot create output directory: " << parentPath
                   << "\n";
      return 1;
    }
  }

  buildCustomizedChessADG(outputFile);
  return 0;
}
