//===-- adg.cpp - delta_encode ADG generation ------------------*- C++ -*-===//

#include "DomainADGs.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

static llvm::cl::opt<std::string> outputFile(
    llvm::cl::Positional, llvm::cl::desc("<output .fabric.mlir file>"),
    llvm::cl::init("delta_encode.fabric.mlir"));

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "delta_encode ADG generator\n");

  auto parentPath = llvm::sys::path::parent_path(outputFile);
  if (!parentPath.empty())
    if (auto ec = llvm::sys::fs::create_directories(parentPath)) {
      llvm::errs() << "error: cannot create output directory: " << parentPath
                   << "\n";
      return 1;
    }

  fcc::e2e::SpatialVectorDomainOptions opts;
  opts.moduleName = "delta_encode_domain";
  opts.numPEs = 48;
  opts.numExtMems = 2;
  opts.numScalarInputs = 1;
  opts.numScalarOutputs = 0;
  opts.maxLdCount = 2;
  opts.maxStCount = 1;
  fcc::e2e::buildSpatialVectorDomain(outputFile, opts);
  return 0;
}
