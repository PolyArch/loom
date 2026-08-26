#include "Application/Manifest.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <string>
#include <utility>

namespace {

llvm::cl::opt<std::string>
    manifestPath("manifest", llvm::cl::desc("Application portfolio manifest"),
                 llvm::cl::value_desc("path"), llvm::cl::Required);

} // namespace

int main(int argc, char **argv) {
  llvm::InitLLVM init(argc, argv);
  llvm::cl::ParseCommandLineOptions(
      argc, argv, "Inspect a canonical Loom application manifest\n");
  auto manifest = loom::application::loadApplicationManifest(manifestPath);
  if (!manifest) {
    llvm::errs() << "loom-application-manifest-inspect: error: "
                 << llvm::toString(manifest.takeError()) << '\n';
    return EXIT_FAILURE;
  }
  loom::application::writeApplicationManifestInventoryJson(llvm::outs(),
                                                           *manifest);
  return EXIT_SUCCESS;
}
