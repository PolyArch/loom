#include "loom/ADG/KHGGenerator.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

static llvm::cl::opt<std::string> outputFile(llvm::cl::Positional,
                                              llvm::cl::desc("<output>"),
                                              llvm::cl::Required);

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv, "CISY8 ADG builder\n");
  auto parent = llvm::sys::path::parent_path(outputFile);
  if (!parent.empty())
    llvm::sys::fs::create_directories(parent);
  loom::adg::exportKHGADG(loom::adg::paramsFromTypeId("CISY8"), outputFile);
  return 0;
}
