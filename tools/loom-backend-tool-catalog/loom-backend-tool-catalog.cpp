#include "ExternalTool/Provider.h"
#include "ExternalTool/ShellProbe.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"

#include <filesystem>
#include <string>
#include <system_error>

namespace {

llvm::cl::OptionCategory catalogCategory("Loom backend tool catalog options");

llvm::cl::opt<std::string> probeDirectoryOption(
    "probe-dir",
    llvm::cl::desc("Directory for temporary backend tool probe scripts"),
    llvm::cl::value_desc("path"), llvm::cl::init("."),
    llvm::cl::cat(catalogCategory));

llvm::Error catalogError(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "backend_tool_catalog_failed: " + message);
}

llvm::Expected<llvm::SmallString<256>> absoluteProbeDirectory() {
  llvm::SmallString<256> path(probeDirectoryOption);
  if (std::error_code error = llvm::sys::fs::make_absolute(path))
    return catalogError("could not make probe directory absolute: " +
                        error.message());
  llvm::sys::path::remove_dots(path, true);
  std::error_code directoryError;
  if (!std::filesystem::is_directory(std::filesystem::path(path.str().str()),
                                     directoryError) ||
      directoryError)
    return catalogError("probe directory does not exist: " + path.str());
  return path;
}

llvm::Expected<bool> releaseIsAvailable(
    const loom::external_tool::BackendToolCatalogEntry &entry,
    const loom::external_tool::BackendToolReleaseProfile &release,
    llvm::StringRef probeDirectory) {
  for (const std::string &name : entry.provider.binding.executableNames) {
    llvm::ErrorOr<std::string> executable = llvm::sys::findProgramByName(name);
    if (!executable)
      continue;
    loom::external_tool::ShellToolBindingProbe probe(probeDirectory.str(),
                                                     release.exactVersionProbe);
    auto result = probe.probeExecutable(*executable);
    if (!result)
      return result.takeError();
    if (*result)
      return true;
  }
  return false;
}

llvm::Error run() {
  if (llvm::Error error = loom::external_tool::validateBackendToolCatalog())
    return error;
  auto probeDirectory = absoluteProbeDirectory();
  if (!probeDirectory)
    return probeDirectory.takeError();

  llvm::json::Array availableFeatures;
  llvm::json::Array tools;
  for (const auto &entry : loom::external_tool::backendToolCatalog()) {
    llvm::json::Array releases;
    for (const auto &release : entry.validatedReleases) {
      auto available =
          releaseIsAvailable(entry, release, probeDirectory->str());
      if (!available)
        return available.takeError();
      if (*available)
        availableFeatures.push_back(release.conformanceFeature);
      releases.push_back(llvm::json::Object{
          {"conformance_feature", release.conformanceFeature},
          {"module_alias", release.moduleAlias
                               ? llvm::json::Value(*release.moduleAlias)
                               : llvm::json::Value(nullptr)},
          {"available", *available},
      });
    }
    tools.push_back(llvm::json::Object{
        {"logical_tool_key", entry.provider.binding.key},
        {"official_product_name", entry.officialProductName},
        {"validated_releases", std::move(releases)},
    });
  }

  llvm::json::Object projection{
      {"schema", "loom.external_tool.backend_catalog"},
      {"version", "1.0"},
      {"available_features", std::move(availableFeatures)},
      {"tools", std::move(tools)},
  };
  llvm::outs() << llvm::formatv("{0:2}",
                                llvm::json::Value(std::move(projection)))
               << '\n';
  return llvm::Error::success();
}

} // namespace

int main(int argc, char **argv) {
  llvm::InitLLVM initialization(argc, argv);
  llvm::cl::HideUnrelatedOptions(catalogCategory);
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "Loom backend tool catalog probe\n");
  if (llvm::Error error = run()) {
    llvm::errs() << llvm::toString(std::move(error)) << '\n';
    return 1;
  }
  return 0;
}
