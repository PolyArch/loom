#include "E2EADGs.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

static llvm::cl::opt<std::string> outputFile(
    llvm::cl::Positional, llvm::cl::desc("<output .fabric.mlir file>"),
    llvm::cl::init("mesh-10x10-extmem-2.fabric.mlir"));

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "mesh-10x10-extmem-2 ADG generator\n");

  auto parent = llvm::sys::path::parent_path(outputFile);
  if (!parent.empty()) {
    if (auto ec = llvm::sys::fs::create_directories(parent)) {
      llvm::errs() << "error: cannot create output directory: " << parent
                   << "\n";
      return 1;
    }
  }

  loom::e2e::buildMesh10x10Extmem2(outputFile);
  return 0;
}
