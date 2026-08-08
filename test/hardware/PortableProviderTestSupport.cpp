#include "PortableProviderTestSupport.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <filesystem>
#include <fstream>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace loom::hardware::test {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "portable_provider_conformance_invalid: " +
                                     message);
}

std::string moduleText(mlir::ModuleOp module) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  module.print(stream);
  return text;
}

bool pathContains(const std::filesystem::path &root,
                  const std::filesystem::path &path) {
  auto rootPart = root.begin();
  auto pathPart = path.begin();
  while (rootPart != root.end() && pathPart != path.end() &&
         *rootPart == *pathPart) {
    ++rootPart;
    ++pathPart;
  }
  return rootPart == root.end();
}

} // namespace

llvm::Expected<std::string> specializeAndExportPortableProvider(
    rtl::ModuleRootCirctSkeleton skeleton,
    const FinalizedConfigurationABI &configurationAbi,
    const rtl::FabricOperationProviderRegistry &providers,
    const ExternalImplementationContractCatalog &externalContracts) {
  if (!skeleton.module)
    return invalid("module is absent");
  if (skeleton.operationLeaves.empty())
    return invalid("operation occurrence set is empty");
  if (llvm::Error error = rtl::verifyCommonCirctSkeleton(
          *skeleton.module, configurationAbi.abi(), skeleton.operationLeaves))
    return std::move(error);

  std::vector<rtl::FabricOperationRecipeBinding> recipes;
  recipes.reserve(skeleton.operationLeaves.size());
  for (const rtl::FabricOperationLeafAssociation &association :
       skeleton.operationLeaves)
    recipes.push_back({association.occurrence,
                       rtl::BackendRecipeKey::PortableSystemVerilog,
                       {}});

  const std::string before = moduleText(*skeleton.module);
  auto providerOutput = rtl::specializeFabricOperationLeaves(
      *skeleton.module, configurationAbi, skeleton.operationLeaves, recipes,
      providers, externalContracts);
  if (!providerOutput) {
    llvm::Error error = providerOutput.takeError();
    if (moduleText(*skeleton.module) != before) {
      llvm::consumeError(std::move(error));
      return invalid("failed specialization mutated the common skeleton");
    }
    return std::move(error);
  }
  if (!providerOutput->payloads.empty() ||
      !providerOutput->activityPoints.empty() ||
      !providerOutput->externalImplementationBindings.empty())
    return invalid("portable recipe emitted external implementation state");

  auto systemVerilog =
      rtl::lowerAndExportSpecializedSystemVerilog(*skeleton.module);
  if (!systemVerilog)
    return systemVerilog.takeError();
  return std::move(*systemVerilog);
}

llvm::Error writePortableProviderArtifacts(
    const std::filesystem::path &root,
    llvm::ArrayRef<PortableProviderArtifact> artifacts) {
  if (root.empty())
    return invalid("artifact root is empty");

  std::vector<std::filesystem::path> relativePaths;
  relativePaths.reserve(artifacts.size());
  std::set<std::filesystem::path> uniquePaths;
  for (const PortableProviderArtifact &artifact : artifacts) {
    const std::filesystem::path relative =
        artifact.relativePath.lexically_normal();
    if (relative.empty() || relative.is_absolute() ||
        relative != artifact.relativePath || relative.filename().empty() ||
        llvm::is_contained(relative, std::filesystem::path("..")))
      return invalid("artifact path must be a normalized relative file");
    if (!uniquePaths.insert(relative).second)
      return invalid("artifact path is duplicated");
    relativePaths.push_back(relative);
  }

  std::error_code error;
  std::filesystem::create_directories(root, error);
  if (error)
    return invalid("could not create artifact root: " + error.message());
  const std::filesystem::path canonicalRoot =
      std::filesystem::weakly_canonical(root, error);
  if (error)
    return invalid("could not resolve artifact root: " + error.message());

  std::vector<std::filesystem::path> destinations;
  destinations.reserve(relativePaths.size());
  for (const std::filesystem::path &relative : relativePaths) {
    const std::filesystem::path destination = root / relative;
    std::filesystem::create_directories(destination.parent_path(), error);
    if (error)
      return invalid("could not create artifact directory: " + error.message());
    const std::filesystem::path canonicalParent =
        std::filesystem::weakly_canonical(destination.parent_path(), error);
    if (error || !pathContains(canonicalRoot, canonicalParent))
      return invalid("artifact path escapes its root");
    if (std::filesystem::is_symlink(
            std::filesystem::symlink_status(destination, error)))
      return invalid("artifact target may not be a symbolic link");
    if (error && error != std::errc::no_such_file_or_directory)
      return invalid("could not inspect artifact target: " + error.message());
    error.clear();
    destinations.push_back(destination);
  }

  for (const auto &[artifact, destination] :
       llvm::zip_equal(artifacts, destinations)) {
    std::ofstream output(destination, std::ios::binary | std::ios::trunc);
    output.write(artifact.contents.data(),
                 static_cast<std::streamsize>(artifact.contents.size()));
    output.close();
    if (!output)
      return invalid("could not write artifact file");
  }
  return llvm::Error::success();
}

} // namespace loom::hardware::test
