//===-- adg.cpp - Builder index width configuration test --------*- C++ -*-===//
//
// Part of the fcc project.
//
//===----------------------------------------------------------------------===//

#include "fcc/ADG/ADGBuilder.h"
#include "fcc/ADG/ADGBuilderDetail.h"
#include "fcc/Dialect/Fabric/FabricTypes.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

using namespace fcc::adg;

static bool buildADG(const std::string &outputPath) {
  auto parsedIndexWidth = detail::tryParseScalarWidth("index");
  unsigned configuredIndexWidth = fcc::fabric::getConfiguredIndexBitWidth();
  if (!parsedIndexWidth || *parsedIndexWidth != configuredIndexWidth) {
    llvm::errs() << "error: index width configuration mismatch\n";
    return false;
  }

  ADGBuilder builder("builder_index_width_test_adg");
  constexpr unsigned dataWidth = 64;

  auto fuAdd = builder.defineBinaryFU("fu_add", "arith.addi", "i32", "i32");
  auto pe = builder.defineSingleFUSpatialPE("add_pe", 2, 1, dataWidth, fuAdd);
  auto grid = builder.instantiatePEGrid(1, 2, pe, "pe");

  auto lhs = builder.addScalarInput("lhs", dataWidth);
  auto rhs = builder.addScalarInput("rhs", dataWidth);
  auto bias = builder.addScalarInput("bias", dataWidth);
  auto sum = builder.addScalarOutput("sum", dataWidth);

  std::string indexBitsTy =
      "!fabric.bits<" + std::to_string(configuredIndexWidth) + ">";
  std::string taggedIndexTy =
      "!fabric.tagged<" + indexBitsTy + ", i1>";
  auto probeIn = builder.addInput("index_probe_in", indexBitsTy);
  auto probeOut = builder.addOutput("index_probe_out", indexBitsTy);
  auto addTag = builder.createAddTagBank(indexBitsTy, taggedIndexTy, {0})[0];
  auto delTag = builder.createDelTagBank(taggedIndexTy, indexBitsTy, 1)[0];

  builder.connectInputToInstance(lhs, grid[0][0], 0);
  builder.connectInputToInstance(rhs, grid[0][0], 1);
  builder.connect(grid[0][0], 0, grid[0][1], 0);
  builder.connectInputToInstance(bias, grid[0][1], 1);
  builder.connectInstanceToOutput(grid[0][1], 0, sum);

  builder.connectInputToInstance(probeIn, addTag, 0);
  builder.connect(addTag, 0, delTag, 0);
  builder.connectInstanceToOutput(delTag, 0, probeOut);

  builder.exportMLIR(outputPath);
  return true;
}

static llvm::cl::opt<std::string> outputFile(
    llvm::cl::Positional, llvm::cl::desc("<output .fabric.mlir file>"),
    llvm::cl::init("builder-index-width.fabric.mlir"));

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "Builder index width ADG generator\n");

  auto parentPath = llvm::sys::path::parent_path(outputFile);
  if (!parentPath.empty()) {
    if (auto ec = llvm::sys::fs::create_directories(parentPath)) {
      llvm::errs() << "error: cannot create output directory: " << parentPath
                   << "\n";
      return 1;
    }
  }

  return buildADG(outputFile) ? 0 : 1;
}
