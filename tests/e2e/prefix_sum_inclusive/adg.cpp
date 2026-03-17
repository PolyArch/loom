//===-- adg.cpp - prefix_sum_inclusive ADG generation ----------*- C++ -*-===//

#include "TemporalDomainADGs.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

static llvm::cl::opt<std::string> outputFile(
    llvm::cl::Positional, llvm::cl::desc("<output .fabric.mlir file>"),
    llvm::cl::init("prefix_sum_inclusive.fabric.mlir"));

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "prefix_sum_inclusive ADG generator\n");

  auto parentPath = llvm::sys::path::parent_path(outputFile);
  if (!parentPath.empty())
    if (auto ec = llvm::sys::fs::create_directories(parentPath)) {
      llvm::errs() << "error: cannot create output directory: " << parentPath
                   << "\n";
      return 1;
    }

  fcc::e2e::TemporalScanDomainOptions opts;
  opts.moduleName = "prefix_sum_inclusive_domain";
  fcc::e2e::buildTemporalScanDomain(outputFile, opts);
  return 0;
}
