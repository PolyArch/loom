#include "PortableProviderTestSupport.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
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

llvm::Expected<std::filesystem::path>
createStagingDirectory(const std::filesystem::path &root) {
  for (unsigned attempt = 0; attempt != 32; ++attempt) {
    llvm::SmallString<256> model((root.string() + ".partial-%%%%%%").c_str());
    llvm::SmallString<256> candidate;
    llvm::sys::fs::createUniquePath(model, candidate, false);
    std::error_code error;
    if (std::filesystem::create_directory(candidate.str().str(), error))
      return std::filesystem::path(candidate.str().str());
    if (error != std::errc::file_exists)
      return invalid("could not create artifact staging directory: " +
                     error.message());
  }
  return invalid("could not allocate an artifact staging directory");
}

struct StagingCleanup final {
  std::filesystem::path path;
  bool published = false;

  ~StagingCleanup() {
    if (published)
      return;
    std::error_code ignored;
    std::filesystem::remove_all(path, ignored);
  }
};

} // namespace

llvm::Expected<PortableProviderConformance> specializeAndExportPortableProvider(
    rtl::ModuleRootCirctSkeleton skeleton,
    const FinalizedConfigurationABI &configurationAbi,
    const rtl::FabricOperationProviderRegistry &providers,
    const ExternalImplementationContractCatalog &externalContracts) {
  if (!skeleton.module)
    return invalid("module is absent");
  if (skeleton.operationLeaves.empty())
    return invalid("operation occurrence set is empty");

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
  auto systemVerilog =
      rtl::lowerAndExportSpecializedSystemVerilog(*skeleton.module);
  if (!systemVerilog)
    return systemVerilog.takeError();
  return PortableProviderConformance{std::move(*providerOutput),
                                     std::move(*systemVerilog)};
}

llvm::Error writePortableProviderArtifacts(
    const std::filesystem::path &root,
    llvm::ArrayRef<PortableProviderArtifact> artifacts) {
  if (root.empty())
    return invalid("artifact root is empty");

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
  }

  std::error_code error;
  if (std::filesystem::exists(root, error) || error)
    return invalid("artifact root already exists or is inaccessible");
  const std::filesystem::path parent = root.parent_path().empty()
                                           ? std::filesystem::path(".")
                                           : root.parent_path();
  if (!std::filesystem::is_directory(parent, error) || error)
    return invalid("artifact parent must be an existing directory");

  auto staging = createStagingDirectory(root);
  if (!staging)
    return staging.takeError();
  StagingCleanup cleanup{*staging};

  for (const PortableProviderArtifact &artifact : artifacts) {
    const std::filesystem::path destination = *staging / artifact.relativePath;
    std::filesystem::create_directories(destination.parent_path(), error);
    if (error)
      return invalid("could not create artifact directory: " + error.message());
    std::ofstream output(destination, std::ios::binary | std::ios::trunc);
    output.write(artifact.contents.data(),
                 static_cast<std::streamsize>(artifact.contents.size()));
    output.close();
    if (!output)
      return invalid("could not write artifact file");
  }

  std::filesystem::rename(*staging, root, error);
  if (error)
    return invalid("could not publish artifact root: " + error.message());
  cleanup.published = true;
  return llvm::Error::success();
}

} // namespace loom::hardware::test
