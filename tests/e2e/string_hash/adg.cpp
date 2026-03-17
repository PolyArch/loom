//===-- adg.cpp - string_hash ADG generation -------------------*- C++ -*-===//

#include "DomainADGs.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

static llvm::cl::opt<std::string> outputFile(
    llvm::cl::Positional, llvm::cl::desc("<output .fabric.mlir file>"),
    llvm::cl::init("string_hash.fabric.mlir"));

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "string_hash ADG generator\n");

  auto parentPath = llvm::sys::path::parent_path(outputFile);
  if (!parentPath.empty())
    if (auto ec = llvm::sys::fs::create_directories(parentPath)) {
      llvm::errs() << "error: cannot create output directory: " << parentPath
                   << "\n";
      return 1;
    }

  fcc::e2e::SpatialVectorDomainOptions opts;
  opts.moduleName = "string_hash_domain";
  opts.numPEs = 160;
  opts.numExtMems = 2;
  opts.numScalarInputs = 2;
  opts.numScalarOutputs = 0;
  opts.maxLdCount = 3;
  opts.maxStCount = 1;
  fcc::e2e::buildSpatialVectorDomain(outputFile, opts);
  return 0;
}
