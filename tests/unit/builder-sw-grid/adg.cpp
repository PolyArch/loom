//===-- adg.cpp - Builder switch grid helper test --------------*- C++ -*-===//
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
  ADGBuilder builder("builder_sw_grid_test_adg");
  constexpr unsigned dataWidth = 64;

  auto fuAdd = builder.defineBinaryFU("fu_add", "arith.addi", "i32", "i32");
  auto pe = builder.defineSingleFUSpatialPE("add_pe", 2, 1, dataWidth, fuAdd);

  std::vector<unsigned> swWidths(3, dataWidth);
  std::vector<std::vector<bool>> fullCrossbar(
      3, std::vector<bool>(3, true));
  auto sw = builder.defineSpatialSW("grid_sw", swWidths, swWidths,
                                    fullCrossbar);

  auto swGrid = builder.instantiateSWGrid(1, 2, sw, "sw");
  auto peGrid = builder.instantiatePEGrid(1, 2, pe, "pe");

  auto inA = builder.addScalarInput("a", dataWidth);
  auto inB = builder.addScalarInput("b", dataWidth);
  auto inC = builder.addScalarInput("c", dataWidth);
  auto outY = builder.addScalarOutput("y", dataWidth);

  auto sw0 = swGrid[0][0];
  auto sw1 = swGrid[0][1];
  auto pe0 = peGrid[0][0];
  auto pe1 = peGrid[0][1];

  builder.connectInputToInstance(inA, sw0, 0);
  builder.connectInputToInstance(inB, sw0, 1);
  builder.connectInputToInstance(inC, sw0, 2);

  builder.connect(sw0, 0, pe0, 0);
  builder.connect(sw0, 1, pe0, 1);
  builder.connect(sw0, 2, sw1, 0);

  builder.connect(pe0, 0, sw1, 1);

  builder.connect(sw1, 0, pe1, 1);
  builder.connect(sw1, 1, pe1, 0);
  builder.connect(pe1, 0, sw1, 2);
  builder.connectInstanceToOutput(sw1, 2, outY);

  builder.exportMLIR(outputPath);
}

static llvm::cl::opt<std::string> outputFile(
    llvm::cl::Positional, llvm::cl::desc("<output .fabric.mlir file>"),
    llvm::cl::init("builder-sw-grid.fabric.mlir"));

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(
      argc, argv, "Builder switch grid helper ADG generator\n");

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
