//===-- adg.cpp - Builder simple op helper test -----------------*- C++ -*-===//
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
  ADGBuilder builder("builder_simple_op_helpers_test_adg");
  constexpr unsigned dataWidth = 64;

  auto fuTrunc =
      builder.defineUnaryFU("fu_trunci", "arith.trunci", "i64", "i32");
  auto fuAdd = builder.defineBinaryFU("fu_add", "arith.addi", "i32", "i32");

  auto truncPE =
      builder.defineSingleFUSpatialPE("trunc_pe", 1, 1, dataWidth, fuTrunc);
  auto addPE =
      builder.defineSingleFUSpatialPE("add_pe", 2, 1, dataWidth, fuAdd);

  auto trunc0 = builder.instantiatePE(truncPE, "pe_trunc0");
  auto trunc1 = builder.instantiatePE(truncPE, "pe_trunc1");
  auto add = builder.instantiatePE(addPE, "pe_add");

  auto lhs = builder.addScalarInput("lhs", dataWidth);
  auto rhs = builder.addScalarInput("rhs", dataWidth);
  auto sum = builder.addScalarOutput("sum", dataWidth);

  builder.connectInputToInstance(lhs, trunc0, 0);
  builder.connectInputToInstance(rhs, trunc1, 0);
  builder.connect(trunc0, 0, add, 0);
  builder.connect(trunc1, 0, add, 1);
  builder.connectInstanceToOutput(add, 0, sum);

  builder.exportMLIR(outputPath);
}

static llvm::cl::opt<std::string> outputFile(
    llvm::cl::Positional, llvm::cl::desc("<output .fabric.mlir file>"),
    llvm::cl::init("builder-simple-op-helpers.fabric.mlir"));

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "Builder simple op helper ADG generator\n");

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
