//===-- adg.cpp - compact_predicate ADG generation -------------*- C++ -*-===//

#include "DomainADGs.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

static llvm::cl::opt<std::string> outputFile(
    llvm::cl::Positional, llvm::cl::desc("<output .fabric.mlir file>"),
    llvm::cl::init("compact_predicate.fabric.mlir"));

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "compact_predicate ADG generator\n");

  auto parentPath = llvm::sys::path::parent_path(outputFile);
  if (!parentPath.empty())
    if (auto ec = llvm::sys::fs::create_directories(parentPath)) {
      llvm::errs() << "error: cannot create output directory: " << parentPath
                   << "\n";
      return 1;
    }

  fcc::e2e::SpatialVectorDomainOptions opts;
  opts.moduleName = "compact_predicate_domain";
  opts.numPEs = 128;
  opts.numExtMems = 3;
  opts.numScalarInputs = 1;
  opts.numScalarOutputs = 1;
  opts.maxLdCount = 1;
  opts.maxStCount = 1;
  fcc::e2e::buildSpatialVectorDomain(outputFile, opts);
  return 0;
}
