#include "ExternalTool/InvocationBundle.h"
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
#include "llvm/Support/raw_ostream.h"

#include <filesystem>
#include <optional>
#include <string>
#include <system_error>
#include <vector>

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

/// Probes one validated release in the current environment and returns the
/// resolved binding when the exact release is available.
llvm::Expected<std::optional<loom::external_tool::ResolvedToolBinding>>
probeRelease(const loom::external_tool::BackendToolCatalogEntry &entry,
             const loom::external_tool::BackendToolReleaseProfile &release,
             llvm::StringRef probeDirectory) {
  loom::external_tool::ShellToolBindingProbe probe(probeDirectory.str(),
                                                   release.exactVersionProbe);
  auto result = loom::external_tool::resolveEnvironmentToolBinding(
      entry.provider.binding,
      loom::external_tool::captureToolEnvironment(entry.provider.binding),
      probe);
  if (!result)
    return result.takeError();
  return std::move(*result);
}

llvm::Error run() {
  if (llvm::Error error = loom::external_tool::validateBackendToolCatalog())
    return error;
  auto probeDirectory = absoluteProbeDirectory();
  if (!probeDirectory)
    return probeDirectory.takeError();

  struct ProbedRelease final {
    const loom::external_tool::BackendToolReleaseProfile *release = nullptr;
    std::optional<loom::external_tool::ResolvedToolBinding> binding;
  };
  struct ProbedEntry final {
    const loom::external_tool::BackendToolCatalogEntry *entry = nullptr;
    std::vector<ProbedRelease> releases;
  };
  std::vector<std::string> availableFeatures;
  std::vector<ProbedEntry> entries;
  for (const auto &entry : loom::external_tool::backendToolCatalog()) {
    ProbedEntry probed{&entry, {}};
    for (const auto &release : entry.validatedReleases) {
      auto binding = probeRelease(entry, release, probeDirectory->str());
      if (!binding)
        return binding.takeError();
      if (*binding)
        availableFeatures.push_back(release.conformanceFeature);
      probed.releases.push_back({&release, std::move(*binding)});
    }
    entries.push_back(std::move(probed));
  }

  // Keys are written in sorted order so the projection stays byte-stable.
  llvm::json::OStream json(llvm::outs(), 2);
  json.object([&] {
    json.attributeArray("available_features", [&] {
      for (const std::string &feature : availableFeatures)
        json.value(feature);
    });
    json.attribute("schema", "loom.external_tool.backend_catalog");
    json.attributeArray("tools", [&] {
      for (const ProbedEntry &probed : entries)
        json.object([&] {
          json.attribute("logical_tool_key",
                         probed.entry->provider.binding.key);
          json.attribute("official_product_name",
                         probed.entry->officialProductName);
          json.attributeArray("validated_releases", [&] {
            for (const ProbedRelease &release : probed.releases)
              json.object([&] {
                if (release.binding) {
                  json.attributeBegin("binding");
                  loom::external_tool::writeResolvedToolBinding(
                      json, *release.binding);
                  json.attributeEnd();
                } else {
                  json.attribute("binding", nullptr);
                }
                json.attribute("conformance_feature",
                               release.release->conformanceFeature);
                if (release.release->moduleAlias)
                  json.attribute("module_alias", *release.release->moduleAlias);
                else
                  json.attribute("module_alias", nullptr);
              });
          });
        });
    });
    json.attribute("version", "1.1");
  });
  llvm::outs() << '\n';
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
