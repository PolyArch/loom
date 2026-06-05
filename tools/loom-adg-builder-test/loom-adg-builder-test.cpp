#include "ADG/Builder.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

static llvm::cl::opt<bool>
    sharedReduction("shared-reduction",
                    llvm::cl::desc("emit a shared reduction SpatialCore ADG"),
                    llvm::cl::init(false));

static llvm::cl::opt<std::string>
    outputPath("output", llvm::cl::desc("output Fabric MLIR path"),
               llvm::cl::Required);

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(
      argc, argv,
      "loom-adg-builder-test: emit deterministic Fabric MLIR from the ADG "
      "Builder C++ API\n");

  if (!sharedReduction) {
    llvm::errs() << "error: no ADG recipe selected\n";
    return 1;
  }

  std::error_code ec;
  llvm::raw_fd_ostream out(outputPath, ec, llvm::sys::fs::OF_Text);
  if (ec) {
    llvm::errs() << "error: could not open " << outputPath << ": "
                 << ec.message() << "\n";
    return 1;
  }

  if (llvm::Error err = loom::adg::writeSharedReductionAdg(out)) {
    llvm::errs() << "error: " << llvm::toString(std::move(err)) << "\n";
    return 1;
  }
  return 0;
}
