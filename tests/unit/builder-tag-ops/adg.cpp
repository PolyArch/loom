//===-- adg.cpp - Builder tag ops test --------------------------*- C++ -*-===//
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

static void buildTagOpsADG(const std::string &outputPath) {
  ADGBuilder builder("builder_tag_ops_test");

  const std::string bitsTy = "!fabric.bits<32>";
  const std::string taggedInTy = "!fabric.tagged<!fabric.bits<32>, i1>";
  const std::string taggedOutTy = "!fabric.tagged<!fabric.bits<32>, i2>";

  auto in0 = builder.addInput("a", bitsTy);
  auto out0 = builder.addOutput("result", bitsTy);

  auto add = builder.createAddTagBank(bitsTy, taggedInTy, {0})[0];
  auto map = builder.createMapTag(
      taggedInTy, taggedOutTy,
      {{true, 0, 1}, {true, 1, 2}});
  auto del = builder.createDelTagBank(taggedOutTy, bitsTy, 1)[0];

  builder.connectInputToInstance(in0, add, 0);
  builder.connect(add, 0, map, 0);
  builder.connect(map, 0, del, 0);
  builder.connectInstanceToOutput(del, 0, out0);

  builder.exportMLIR(outputPath);
}

static llvm::cl::opt<std::string> outputFile(
    llvm::cl::Positional, llvm::cl::desc("<output .fabric.mlir file>"),
    llvm::cl::init("builder-tag-ops.fabric.mlir"));

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv, "Builder tag ops ADG generator\n");

  auto parentPath = llvm::sys::path::parent_path(outputFile);
  if (!parentPath.empty()) {
    if (auto ec = llvm::sys::fs::create_directories(parentPath)) {
      llvm::errs() << "error: cannot create output directory: " << parentPath
                   << "\n";
      return 1;
    }
  }

  buildTagOpsADG(outputFile);
  return 0;
}
