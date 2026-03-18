//===-- adg.cpp - Builder PE grid helper test -------------------*- C++ -*-===//
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

static void buildADG(const std::string &outputPath) {
  ADGBuilder builder("builder_pe_grid_test_adg");
  constexpr unsigned dataWidth = 64;

  auto fuAdd = builder.defineBinaryFU("fu_add", "arith.addi", "i32", "i32");
  auto pe = builder.defineSingleFUSpatialPE("add_pe", 2, 1, dataWidth, fuAdd);
  auto grid = builder.instantiatePEGrid(1, 2, pe, "pe");

  auto lhs = builder.addScalarInput("lhs", dataWidth);
  auto rhs = builder.addScalarInput("rhs", dataWidth);
  auto bias = builder.addScalarInput("bias", dataWidth);
  auto sum = builder.addScalarOutput("sum", dataWidth);

  builder.connectInputToInstance(lhs, grid[0][0], 0);
  builder.connectInputToInstance(rhs, grid[0][0], 1);
  builder.connect(grid[0][0], 0, grid[0][1], 0);
  builder.connectInputToInstance(bias, grid[0][1], 1);
  builder.connectInstanceToOutput(grid[0][1], 0, sum);

  builder.exportMLIR(outputPath);
}

static llvm::cl::opt<std::string> outputFile(
    llvm::cl::Positional, llvm::cl::desc("<output .fabric.mlir file>"),
    llvm::cl::init("builder-pe-grid.fabric.mlir"));

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "Builder PE grid helper ADG generator\n");

  auto parentPath = llvm::sys::path::parent_path(outputFile);
  if (!parentPath.empty()) {
    if (auto ec = llvm::sys::fs::create_directories(parentPath)) {
      llvm::errs() << "error: cannot create output directory: " << parentPath
                   << "\n";
      return 1;
    }
  }

  buildADG(outputFile);
  return 0;
}
