#include "ADG/Builtin.h"
#include "ADG/Export.h"
#include "Common/ArtifactStore.h"
#include "Common/ArtifactText.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

namespace {

llvm::cl::opt<std::string>
    builtinName("builtin", llvm::cl::desc("builtin Fabric target preset"),
                llvm::cl::value_desc("small|default|large"),
                llvm::cl::Required);

llvm::cl::opt<std::string>
    artifactStorePath("artifact-store",
                      llvm::cl::desc("existing ArtifactStore directory"),
                      llvm::cl::value_desc("path"), llvm::cl::Required);

llvm::cl::opt<std::string>
    outputBase("output",
               llvm::cl::desc("output base for paired .mlir and .html files"),
               llvm::cl::value_desc("path"), llvm::cl::Required);

int reportError(llvm::Error error) {
  llvm::errs() << "error: " << llvm::toString(std::move(error)) << '\n';
  return 1;
}

} // namespace

int main(int argc, char **argv) {
  llvm::InitLLVM init(argc, argv);
  llvm::cl::ParseCommandLineOptions(
      argc, argv,
      "loom-adg: build and export a canonical builtin Fabric target\n");

  auto preset = loom::adg::parseBuiltinTargetPreset(builtinName);
  if (!preset)
    return reportError(preset.takeError());

  loom::ArtifactStore store(artifactStorePath);
  auto design = loom::adg::buildBuiltinTarget(store, *preset);
  if (!design)
    return reportError(design.takeError());
  if (design->roots().size() != 1)
    return reportError(llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "builtin target did not produce exactly one Fabric root"));

  const loom::fabric::FinalizedFabricRoot &root = design->roots().front();
  if (llvm::Error error =
          loom::adg::exportFabricDesign(root, store, outputBase))
    return reportError(std::move(error));

  llvm::outs() << loom::formatArtifactIdentityHex(root.reference().artifact)
               << '\n';
  return 0;
}
