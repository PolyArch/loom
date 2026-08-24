#include "ADG/Builtin.h"
#include "ADG/Export.h"
#include "Common/ArtifactStore.h"
#include "Common/ArtifactText.h"
#include "Config/ResolvedConfig.h"
#include "Fabric/Artifact/FabricTopologyQuality.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

namespace {

llvm::cl::opt<std::string>
    builtinName("builtin", llvm::cl::desc("builtin Fabric target preset"),
                llvm::cl::value_desc("small|coverage|large"),
                llvm::cl::init(""));

llvm::cl::opt<std::string>
    configPath("config",
               llvm::cl::desc("resolved configuration selecting a builtin "
                              "Fabric target"),
               llvm::cl::value_desc("path"), llvm::cl::init(""));

llvm::cl::opt<std::string>
    artifactStorePath("artifact-store",
                      llvm::cl::desc("existing ArtifactStore directory"),
                      llvm::cl::value_desc("path"), llvm::cl::Required);

llvm::cl::opt<std::string>
    outputBase("output",
               llvm::cl::desc("output base for paired .mlir and .html files"),
               llvm::cl::value_desc("path"), llvm::cl::Required);

llvm::cl::opt<std::string> authoringOutput(
    "authoring-output",
    llvm::cl::desc("optional self-contained System authoring MLIR output"),
    llvm::cl::value_desc("path"), llvm::cl::init(""));

llvm::cl::opt<std::string> rootReferencePath(
    "root-reference",
    llvm::cl::desc("optional canonical Fabric root-reference JSON output"),
    llvm::cl::value_desc("path"), llvm::cl::init(""));

llvm::cl::opt<std::string> moduleRootReferencePath(
    "module-root-reference",
    llvm::cl::desc("optional canonical imported Module root-reference JSON "
                   "output"),
    llvm::cl::value_desc("path"), llvm::cl::init(""));

llvm::cl::opt<std::string> topologyQualityPath(
    "topology-quality",
    llvm::cl::desc("optional derived topology-quality JSON output"),
    llvm::cl::value_desc("path"), llvm::cl::init(""));

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

  if (builtinName.empty() == configPath.empty())
    return reportError(llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "exactly one of --builtin and --config is required"));

  loom::ArtifactStore store(artifactStorePath);
  llvm::Expected<loom::adg::FinalizedFabricDesign> design = [&]() {
    if (!builtinName.empty()) {
      auto preset = loom::adg::parseBuiltinTargetPreset(builtinName);
      if (!preset)
        return llvm::Expected<loom::adg::FinalizedFabricDesign>(
            preset.takeError());
      return loom::adg::buildBuiltinTarget(store, *preset);
    }
    auto config = loom::loadResolvedConfig(configPath);
    if (!config)
      return llvm::Expected<loom::adg::FinalizedFabricDesign>(
          config.takeError());
    return loom::adg::buildBuiltinTarget(
        store, config->hardwareTarget.templateIdentity,
        config->hardwareTarget.schemaVersion.major,
        config->hardwareTarget.schemaVersion.minor,
        config->hardwareTarget.parameters);
  }();
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
  if (!authoringOutput.empty()) {
    std::error_code error;
    llvm::raw_fd_ostream output(authoringOutput, error, llvm::sys::fs::OF_Text);
    if (error)
      return reportError(llvm::createStringError(
          error, "cannot open authoring output '%s'", authoringOutput.c_str()));
    if (llvm::Error writeError =
            loom::adg::writeFabricAuthoringMlir(root, store, output))
      return reportError(std::move(writeError));
    output.close();
    if (output.has_error())
      return reportError(llvm::createStringError(
          llvm::inconvertibleErrorCode(), "cannot write authoring output"));
  }
  if (!rootReferencePath.empty())
    if (llvm::Error error = loom::writeArtifactRootReferenceJsonFile(
            rootReferencePath, root.reference()))
      return reportError(std::move(error));
  if (!moduleRootReferencePath.empty()) {
    if (root.directDependencies().size() != 1 ||
        root.directDependencies().front().role !=
            loom::fabric::FabricDependencyRole::ImportedModule)
      return reportError(llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "builtin System does not have exactly one imported Module"));
    if (llvm::Error error = loom::writeArtifactRootReferenceJsonFile(
            moduleRootReferencePath, root.directDependencies().front().root))
      return reportError(std::move(error));
  }
  if (!topologyQualityPath.empty()) {
    auto quality =
        loom::fabric::analyzeFabricTopologyQualityClosure(root.view());
    if (!quality)
      return reportError(quality.takeError());
    std::error_code error;
    llvm::raw_fd_ostream output(topologyQualityPath, error,
                                llvm::sys::fs::OF_Text);
    if (error)
      return reportError(llvm::createStringError(
          error, "cannot open topology-quality output '%s'",
          topologyQualityPath.c_str()));
    if (llvm::Error writeError =
            loom::fabric::writeFabricTopologyQualityJson(*quality, output))
      return reportError(std::move(writeError));
  }

  llvm::outs() << loom::formatArtifactIdentityHex(root.reference().artifact)
               << '\n';
  return 0;
}
