#include "Application/ProductBuild.h"

#include "Common/ArtifactStore.h"
#include "Config/ResolvedConfig.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <filesystem>
#include <string>
#include <system_error>
#include <utility>

namespace {

[[noreturn]] void fail(const llvm::Twine &message) {
  llvm::errs() << "product build policy test: " << message << '\n';
  std::exit(EXIT_FAILURE);
}

void require(bool condition, const llvm::Twine &message) {
  if (!condition)
    fail(message);
}

template <typename T> T take(llvm::Expected<T> value) {
  if (!value)
    fail(llvm::toString(value.takeError()));
  return std::move(*value);
}

class TemporaryDirectory final {
public:
  TemporaryDirectory() {
    if (std::error_code error = llvm::sys::fs::createUniqueDirectory(
            "loom-product-build-policy", path_))
      fail("cannot create temporary directory: " + error.message());
  }

  ~TemporaryDirectory() { llvm::sys::fs::remove_directories(path_); }

  std::string child(llvm::StringRef name) const {
    llvm::SmallString<256> result(path_);
    llvm::sys::path::append(result, name);
    return result.str().str();
  }

  llvm::StringRef path() const { return path_; }

private:
  llvm::SmallString<256> path_;
};

void writeConfig(llvm::StringRef path, const loom::ResolvedConfig &config) {
  std::error_code error;
  llvm::raw_fd_ostream output(path, error, llvm::sys::fs::OF_Text);
  if (error)
    fail("cannot write resolved configuration: " + error.message());
  output << loom::canonicalResolvedConfigJson(config);
  output.close();
  if (output.has_error())
    fail("cannot close resolved configuration");
}

std::string findProductWorkspace(llvm::StringRef parent,
                                 llvm::StringRef deploymentName) {
  const std::string prefix = ("." + deploymentName + ".loom-work-").str();
  std::string workspace;
  std::error_code error;
  for (std::filesystem::directory_iterator iterator(parent.str(), error), end;
       !error && iterator != end; iterator.increment(error)) {
    const std::filesystem::directory_entry &entry = *iterator;
    std::error_code statusError;
    if (!entry.is_directory(statusError) || statusError)
      continue;
    const std::string filename = entry.path().filename().string();
    if (!llvm::StringRef(filename).starts_with(prefix))
      continue;
    if (!workspace.empty())
      fail("product invocation created more than one bounded workspace");
    workspace = entry.path().string();
  }
  if (error)
    fail("cannot enumerate product workspace: " + error.message());
  if (workspace.empty())
    fail("product invocation did not retain its bounded workspace");
  return workspace;
}

void explicitResolvedConfigRemainsTheProductPolicyOwner() {
  TemporaryDirectory directory;
  loom::ResolvedConfig config =
      take(loom::resolveConfigProfile("quick_explore"));
  config.dse.spatialPnr.search.completionGoal =
      loom::ResolvedPnrCompletionGoal::ExhaustConfiguredWork;
  config.dse.systemPnr.search.completionGoal =
      loom::ResolvedPnrCompletionGoal::ExhaustConfiguredWork;

  loom::ResolvedConfig rewritten = config;
  rewritten.dse.spatialPnr.search.completionGoal =
      loom::ResolvedPnrCompletionGoal::FirstVerifiedCandidate;
  rewritten.dse.systemPnr.search.completionGoal =
      loom::ResolvedPnrCompletionGoal::FirstVerifiedCandidate;
  require(loom::resolvedConfigIdentity(config) !=
              loom::resolvedConfigIdentity(rewritten),
          "completion policy did not contribute to ResolvedConfig identity");

  const std::string configPath = directory.child("resolved-config.json");
  writeConfig(configPath, config);
  loom::application::ProductBuildOptions options;
  options.deploymentOutput = directory.child("deployment");
  options.accelerationProfile = configPath;
  auto invocation = take(
      loom::application::ProductBuildInvocation::create(std::move(options)));

  const std::string workspace =
      findProductWorkspace(directory.path(), "deployment");
  loom::ArtifactStore artifacts(
      (std::filesystem::path(workspace) / "artifacts").string());
  const loom::CanonicalSemanticBytes published =
      take(artifacts.get(loom::ResolvedConfig::artifactSchema,
                         loom::resolvedConfigIdentity(config)));
  require(published.bytes() ==
              loom::canonicalResolvedConfigBytes(config).bytes(),
          "product target changed the explicit ResolvedConfig bytes");

  const llvm::StringRef publishedText(
      reinterpret_cast<const char *>(published.bytes().data()),
      published.bytes().size());
  const loom::ResolvedConfig adopted =
      take(loom::parseResolvedConfig(publishedText, "product target"));
  require(adopted.dse.spatialPnr.search.completionGoal ==
                  loom::ResolvedPnrCompletionGoal::ExhaustConfiguredWork &&
              adopted.dse.systemPnr.search.completionGoal ==
                  loom::ResolvedPnrCompletionGoal::ExhaustConfiguredWork,
          "product target replaced exhaustive PnR with a prefix search");

  auto hiddenRewrite = artifacts.get(loom::ResolvedConfig::artifactSchema,
                                     loom::resolvedConfigIdentity(rewritten));
  if (hiddenRewrite)
    fail("product target published a second rewritten ResolvedConfig");
  llvm::consumeError(hiddenRewrite.takeError());
  require(!invocation->compilerArguments().empty(),
          "product invocation did not retain its compiler projection");
}

} // namespace

int main() {
  explicitResolvedConfigRemainsTheProductPolicyOwner();
  return 0;
}
