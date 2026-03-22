//===-- adg.cpp - Builder single temporal PE helper test -------*- C++ -*-===//
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

static void buildADG(const std::string &outputPath) {
  ADGBuilder builder("builder_temporal_single_fu_test");

  const std::string bitsTy = "!fabric.bits<32>";
  const std::string taggedTy = "!fabric.tagged<!fabric.bits<32>, i1>";

  auto fuAdd = builder.defineBinaryFU("fu_add", "arith.addi", "i32", "i32");
  auto tpe = builder.defineSingleFUTemporalPE(
      "add_tpe", {taggedTy, taggedTy}, {taggedTy}, fuAdd, 0, 1, 0, false,
      std::nullopt);

  auto a = builder.addInput("a", bitsTy);
  auto b = builder.addInput("b", bitsTy);
  auto sum = builder.addOutput("sum", bitsTy);

  auto tags = builder.createAddTagBank(bitsTy, taggedTy, {0, 0});
  auto tpeInst = builder.instantiatePE(tpe, "tpe_0");
  auto del = builder.createDelTag(taggedTy, bitsTy);

  builder.connectInputToInstance(a, tags[0], 0);
  builder.connectInputToInstance(b, tags[1], 0);
  builder.connect(tags[0], 0, tpeInst, 0);
  builder.connect(tags[1], 0, tpeInst, 1);
  builder.connect(tpeInst, 0, del, 0);
  builder.connectInstanceToOutput(del, 0, sum);

  builder.exportMLIR(outputPath);
}

static llvm::cl::opt<std::string> outputFile(
    llvm::cl::Positional, llvm::cl::desc("<output .fabric.mlir file>"),
    llvm::cl::init("builder-temporal-single-fu.fabric.mlir"));

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(
      argc, argv, "Builder single temporal PE helper ADG generator\n");

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
