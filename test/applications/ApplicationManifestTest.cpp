#include "Application/Manifest.h"
#include "Application/SourceAdmission.h"
#include "Common/BlobDigest.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"

#include <array>
#include <cstdlib>
#include <filesystem>
#include <optional>
#include <string>
#include <system_error>
#include <utility>
#include <variant>
#include <vector>

#ifndef LOOM_TEST_REPOSITORY_ROOT
#define LOOM_TEST_REPOSITORY_ROOT ""
#endif

namespace {

using namespace loom;
using namespace loom::application;

[[noreturn]] void fail(const llvm::Twine &message) {
  llvm::errs() << "application manifest test failure: " << message << '\n';
  std::exit(EXIT_FAILURE);
}

template <typename T> T take(llvm::Expected<T> value) {
  if (!value)
    fail(llvm::toString(value.takeError()));
  return std::move(*value);
}

template <typename T>
void requireErrorContains(llvm::Expected<T> value, llvm::StringRef needle) {
  if (value)
    fail("expected an error containing '" + needle + "'");
  const std::string message = llvm::toString(value.takeError());
  if (!llvm::StringRef(message).contains(needle))
    fail("expected error containing '" + needle + "', got: " + message);
}

class TemporaryTree final {
public:
  explicit TemporaryTree(llvm::StringRef path) : root_(path.str()) {
    std::error_code error;
    std::filesystem::remove_all(root_, error);
    error.clear();
    std::filesystem::create_directories(root_, error);
    if (error)
      fail("cannot create temporary tree: " + error.message());
  }

  ~TemporaryTree() {
    std::error_code ignored;
    std::filesystem::remove_all(root_, ignored);
  }

  std::filesystem::path path(llvm::StringRef relative = {}) const {
    return relative.empty() ? root_ : root_ / relative.str();
  }

private:
  std::filesystem::path root_;
};

void writeFile(const std::filesystem::path &path, llvm::StringRef contents) {
  std::error_code error;
  std::filesystem::create_directories(path.parent_path(), error);
  if (error)
    fail("cannot create fixture directory: " + error.message());
  llvm::raw_fd_ostream stream(path.string(), error, llvm::sys::fs::OF_None);
  if (error)
    fail("cannot open fixture file: " + error.message());
  stream << contents;
  stream.close();
  if (stream.has_error())
    fail("cannot write fixture file");
}

std::string readFile(const std::filesystem::path &path) {
  auto buffer = llvm::MemoryBuffer::getFile(path.string(), false, false);
  if (!buffer)
    fail("cannot read command output: " + buffer.getError().message());
  return (*buffer)->getBuffer().str();
}

class GitFixture final {
public:
  explicit GitFixture(const TemporaryTree &tree) : tree_(tree) {
    llvm::ErrorOr<std::string> executable = llvm::sys::findProgramByName("git");
    if (!executable)
      fail("git is required by the application source admission anchor");
    git_ = *executable;
  }

  void run(std::vector<std::string> arguments) {
    CommandResult result = execute(std::move(arguments), false);
    if (result.status != 0 || result.executionFailed)
      fail("git command failed: " + result.error);
  }

  std::string capture(std::vector<std::string> arguments) {
    CommandResult result = execute(std::move(arguments), true);
    if (result.status != 0 || result.executionFailed)
      fail("git command failed: " + result.error);
    return llvm::StringRef(result.output).trim().str();
  }

private:
  struct CommandResult final {
    int status;
    bool executionFailed;
    std::string output;
    std::string error;
  };

  CommandResult execute(std::vector<std::string> arguments, bool capture) {
    llvm::SmallVector<llvm::StringRef, 16> command;
    command.push_back(git_);
    for (const std::string &argument : arguments)
      command.push_back(argument);
    const std::filesystem::path outputPath =
        tree_.path("git-" + std::to_string(invocation_) + ".stdout");
    const std::filesystem::path errorPath =
        tree_.path("git-" + std::to_string(invocation_++) + ".stderr");
    const std::string outputText = outputPath.string();
    const std::string errorText = errorPath.string();
    const std::array<std::optional<llvm::StringRef>, 3> redirects = {
        llvm::StringRef(),
        capture ? std::optional<llvm::StringRef>(outputText)
                : std::optional<llvm::StringRef>(""),
        errorText};
    std::string message;
    bool executionFailed = false;
    const int status =
        llvm::sys::ExecuteAndWait(git_, command, std::nullopt, redirects, 30,
                                  256, &message, &executionFailed);
    std::string output = capture ? readFile(outputPath) : std::string{};
    std::string error = readFile(errorPath);
    if (!message.empty()) {
      if (!error.empty())
        error += '\n';
      error += message;
    }
    return {status, executionFailed, std::move(output), std::move(error)};
  }

  const TemporaryTree &tree_;
  std::string git_;
  std::size_t invocation_ = 0;
};

std::string replaceOnce(std::string text, llvm::StringRef from,
                        llvm::StringRef to) {
  const std::size_t offset = text.find(from.str());
  if (offset == std::string::npos)
    fail("fixture replacement source was not found");
  text.replace(offset, from.size(), to.str());
  return text;
}

std::string manifestText(llvm::StringRef digest) {
  return R"json({
  "schema": "loom.application_portfolio",
  "version": "2.0",
  "applications": [
    {
      "identity": "repository-app",
      "source": {"kind": "repository", "root": "test/apps/local"},
      "build": {
        "entry": "main.cpp",
        "language": "c++",
        "sources": ["main.cpp"],
        "compiler_options": ["-O2"],
        "link_options": []
      },
      "cached_inputs": [],
      "inputs": [
        {
          "name": "local-input",
          "workload": "local-workload",
          "runtime_input": "local-runtime",
          "cached_inputs": [],
          "oracle": {"kind": "typed_invariant", "entry": "test/oracles/local.oracle"},
          "profile": {
            "warmup_samples": 2,
            "measured_samples": 3,
            "oracle_coverage": "all_measured_samples",
            "deadline_milliseconds": 5000
          }
        }
      ],
      "selections": ["smoke"]
    },
    {
      "identity": "upstream-app",
      "source": {"kind": "gitlink", "root": "externals/upstream"},
      "build": {
        "entry": "main.c",
        "language": "c",
        "sources": ["main.c"],
        "compiler_options": ["-O3"],
        "link_options": []
      },
      "cached_inputs": [
        {"logical_name": "weights", "path": "weights.bin", "sha256": ")json" +
         digest.str() + R"json("}
      ],
      "inputs": [
        {
          "name": "cache-free-input",
          "workload": "cache-free-workload",
          "runtime_input": "cache-free-runtime",
          "cached_inputs": [],
          "oracle": {"kind": "exact", "entry": "test/oracles/upstream-free.oracle"},
          "profile": {
            "warmup_samples": 0,
            "measured_samples": 1,
            "oracle_coverage": "all_measured_samples",
            "deadline_milliseconds": 10000
          }
        },
        {
          "name": "wakeword-input",
          "workload": "wakeword-workload",
          "runtime_input": "wakeword-runtime",
          "cached_inputs": ["weights"],
          "oracle": {"kind": "exact", "entry": "test/oracles/upstream.oracle"},
          "profile": {
            "warmup_samples": 0,
            "measured_samples": 1,
            "oracle_coverage": "all_measured_samples",
            "deadline_milliseconds": 30000
          }
        }
      ],
      "selections": ["smoke", "validation"]
    }
  ]
})json";
}

const UnavailableApplicationSource &
requireUnavailable(const ApplicationSourceAdmissionOutcome &outcome,
                   SourceUnavailableReason reason) {
  const auto *unavailable = std::get_if<UnavailableApplicationSource>(&outcome);
  if (!unavailable || unavailable->reason != reason)
    fail("source admission did not preserve its typed unavailable reason");
  return *unavailable;
}

void exerciseManifestAndAdmission(llvm::StringRef temporaryPath) {
  TemporaryTree tree(temporaryPath);
  GitFixture git(tree);
  const std::string repository = tree.path("repository").string();
  const std::string upstream = tree.path("upstream-origin").string();
  const std::string cache = tree.path("cache").string();

  git.run({"init", "--quiet", upstream});
  writeFile(tree.path("upstream-origin/main.c"),
            "int main(void) { return 0; }\n");
  git.run({"-C", upstream, "add", "main.c"});
  git.run({"-C", upstream, "-c", "user.name=Loom Test", "-c",
           "user.email=loom@example.invalid", "commit", "--quiet", "-m",
           "fixture"});
  const std::string upstreamCommit =
      git.capture({"-C", upstream, "rev-parse", "HEAD"});

  git.run({"init", "--quiet", repository});
  writeFile(tree.path("repository/test/apps/local/main.cpp"),
            "int main() { return 0; }\n");
  writeFile(tree.path("repository/test/oracles/local.oracle"), "local\n");
  writeFile(tree.path("repository/test/oracles/upstream-free.oracle"),
            "upstream-free\n");
  writeFile(tree.path("repository/test/oracles/upstream.oracle"), "upstream\n");
  git.run({"-C", repository, "add", "test/apps/local/main.cpp"});
  git.run({"-C", repository, "update-index", "--add", "--cacheinfo",
           "160000," + upstreamCommit + ",externals/upstream"});
  std::filesystem::create_directories(cache);

  constexpr llvm::StringLiteral cacheBytes = "exact cached model bytes\n";
  const BlobDigest digest = computeBlobDigest(llvm::ArrayRef<std::uint8_t>(
      reinterpret_cast<const std::uint8_t *>(cacheBytes.data()),
      cacheBytes.size()));
  const std::string text = manifestText(formatBlobDigestHex(digest));
  ApplicationManifest manifest = take(parseApplicationManifest(text));
  if (manifest.applications().size() != 2 ||
      toString(manifest.applications()[1].build.language) != "c" ||
      manifest.applications()[0].inputs[0].profile.totalSamples() != 5 ||
      toString(manifest.applications()[0].inputs[0].profile.oracleCoverage) !=
          "all_measured_samples")
    fail("manifest parser changed the typed inventory");

  SelectedApplicationInput selected =
      take(selectApplicationInput(manifest, "upstream-app", "wakeword-input"));
  if (selected.applicationIdentity != "upstream-app" ||
      selected.source.kind != SourceKind::Gitlink ||
      selected.source.root != "externals/upstream" ||
      selected.build.entry != "main.c" || selected.cachedInputs.size() != 1 ||
      selected.cachedInputs[0].logicalName != "weights" ||
      selected.input.name != "wakeword-input" ||
      selected.input.profile.warmupSamples != 0 ||
      selected.input.profile.measuredSamples != 1 ||
      selected.input.profile.totalSamples() != 1 ||
      selected.input.profile.deadlineMilliseconds != 30000)
    fail("application input resolver changed the selected production copy");
  selected.input.name = "mutated-copy";
  if (take(selectApplicationInput(manifest, "upstream-app", "wakeword-input"))
          .input.name != "wakeword-input")
    fail("application input resolver did not return an independent copy");
  requireErrorContains(
      selectApplicationInput(manifest, "missing-app", "wakeword-input"),
      "unknown application identity");
  requireErrorContains(
      selectApplicationInput(manifest, "upstream-app", "missing-input"),
      "has no input named");

  const std::vector<std::string> smoke =
      selectApplicationIdentities(manifest, ExecutionSelection::Smoke);
  const std::vector<std::string> validation =
      selectApplicationIdentities(manifest, ExecutionSelection::Validation);
  const std::vector<std::string> scale =
      selectApplicationIdentities(manifest, ExecutionSelection::ScaleEda);
  if (smoke != std::vector<std::string>{"repository-app", "upstream-app"} ||
      validation != std::vector<std::string>{"upstream-app"} || !scale.empty())
    fail("execution selection did not derive a canonical manifest subset");

  const std::filesystem::path manifestPath = tree.path("manifest.json");
  writeFile(manifestPath, text);
  if (take(loadApplicationManifest(manifestPath.string()))
          .applications()
          .size() != 2)
    fail("manifest file loader changed the inventory");

  auto missingCheckout =
      take(admitApplicationSources(manifest, smoke, repository, cache));
  if (!std::holds_alternative<AdmittedApplicationSource>(missingCheckout[0]))
    fail("repository-owned source was not admitted independently");
  requireUnavailable(missingCheckout[1],
                     SourceUnavailableReason::GitlinkCheckout);

  std::filesystem::create_directories(tree.path("repository/externals"));
  git.run({"clone", "--quiet", upstream,
           tree.path("repository/externals/upstream").string()});
  std::filesystem::remove(tree.path("repository/test/oracles/upstream.oracle"));
  auto selectedWithoutCache = take(admitApplicationSource(
      manifest, "upstream-app", "cache-free-input", repository));
  if (!std::holds_alternative<AdmittedApplicationSource>(selectedWithoutCache))
    fail("selected input admission required an unrelated cache or oracle");
  writeFile(tree.path("repository/test/oracles/upstream.oracle"), "upstream\n");
  auto missingCacheRoot =
      take(admitApplicationSources(manifest, smoke, repository));
  requireUnavailable(missingCacheRoot[1], SourceUnavailableReason::CacheRoot);
  auto missingCache =
      take(admitApplicationSources(manifest, smoke, repository, cache));
  requireUnavailable(missingCache[1], SourceUnavailableReason::CachedInput);

  writeFile(tree.path("cache/weights.bin"), cacheBytes);
  auto admitted =
      take(admitApplicationSources(manifest, smoke, repository, cache));
  for (const ApplicationSourceAdmissionOutcome &outcome : admitted) {
    const auto *source = std::get_if<AdmittedApplicationSource>(&outcome);
    if (!source || !std::filesystem::path(source->sourceRoot).is_absolute())
      fail("fully available source did not produce an admitted projection");
  }

  writeFile(tree.path("cache/weights.bin"), "tampered\n");
  requireErrorContains(
      admitApplicationSources(manifest, smoke, repository, cache), "digest");
  writeFile(tree.path("cache/weights.bin"), cacheBytes);

  writeFile(tree.path("repository/externals/upstream/main.c"),
            "int main(void) { return 1; }\n");
  requireErrorContains(
      admitApplicationSources(manifest, smoke, repository, cache),
      "differs from its pinned commit");
  writeFile(tree.path("repository/externals/upstream/main.c"),
            "int main(void) { return 0; }\n");

  const std::string copiedRevision = replaceOnce(
      text, "\"root\": \"externals/upstream\"",
      "\"root\": \"externals/upstream\", \"revision\": \"deadbeef\"");
  requireErrorContains(parseApplicationManifest(copiedRevision),
                       "unknown field 'revision'");
  const std::string oldSchema =
      replaceOnce(text, "\"version\": \"2.0\"", "\"version\": \"1.0\"");
  requireErrorContains(parseApplicationManifest(oldSchema),
                       "unsupported schema or version");
  const std::string invalidCoverage =
      replaceOnce(text, "\"oracle_coverage\": \"all_measured_samples\"",
                  "\"oracle_coverage\": \"sampled\"");
  requireErrorContains(parseApplicationManifest(invalidCoverage),
                       "oracle coverage must be 'all_measured_samples'");
  const std::string zeroMeasured =
      replaceOnce(text, "\"measured_samples\": 3", "\"measured_samples\": 0");
  requireErrorContains(parseApplicationManifest(zeroMeasured),
                       "measured_samples must be greater than zero");
  const std::string zeroDeadline = replaceOnce(
      text, "\"deadline_milliseconds\": 5000", "\"deadline_milliseconds\": 0");
  requireErrorContains(parseApplicationManifest(zeroDeadline),
                       "deadline_milliseconds must be greater than zero");
  const std::string overflowingSamples =
      replaceOnce(text, "\"warmup_samples\": 2",
                  "\"warmup_samples\": 18446744073709551615");
  requireErrorContains(parseApplicationManifest(overflowingSamples),
                       "total sample count overflows uint64");
  const std::string escapingCache = replaceOnce(
      text, "\"path\": \"weights.bin\"", "\"path\": \"../weights.bin\"");
  requireErrorContains(parseApplicationManifest(escapingCache),
                       "non-canonical path component");
  const std::string unsortedInventory =
      replaceOnce(text, "\"identity\": \"repository-app\"",
                  "\"identity\": \"z-repository-app\"");
  requireErrorContains(parseApplicationManifest(unsortedInventory),
                       "strictly ordered");

  const std::string wrongModeText =
      replaceOnce(text, "\"root\": \"externals/upstream\"",
                  "\"root\": \"test/apps/local/main.cpp\"");
  ApplicationManifest wrongMode = take(parseApplicationManifest(wrongModeText));
  requireErrorContains(admitApplicationSources(
                           wrongMode, std::vector<std::string>{"upstream-app"},
                           repository, cache),
                       "mode 160000");

  requireErrorContains(
      admitApplicationSources(
          manifest, std::vector<std::string>{"upstream-app", "repository-app"},
          repository, cache),
      "not canonical and unique");
}

void exerciseRepositoryManifest(llvm::StringRef manifestPath,
                                llvm::StringRef repositoryRoot,
                                llvm::StringRef sharedRepositoryRoot) {
  ApplicationManifest manifest = take(loadApplicationManifest(manifestPath));
  const std::vector<std::string> smoke =
      selectApplicationIdentities(manifest, ExecutionSelection::Smoke);
  const std::vector<std::string> validation =
      selectApplicationIdentities(manifest, ExecutionSelection::Validation);
  const std::vector<std::string> scale =
      selectApplicationIdentities(manifest, ExecutionSelection::ScaleEda);
  if (smoke != std::vector<std::string>{"gapbs-pagerank", "llama2c-kernels",
                                        "loom-multisensor-attention",
                                        "mlperf-tiny-anomaly-detection",
                                        "vecadd-memory"} ||
      !validation.empty() || !scale.empty())
    fail("repository manifest execution selections do not match real rows");
  for (const ApplicationDefinition &application : manifest.applications()) {
    if (application.inputs.size() != 1)
      fail("repository manifest application has an unexpected input count");
    const WorkloadExecutionProfile &profile = application.inputs[0].profile;
    const bool isTinyMl =
        application.identity == "mlperf-tiny-anomaly-detection";
    if (profile.warmupSamples != (isTinyMl ? 1u : 0u) ||
        profile.measuredSamples != (isTinyMl ? 4u : 1u) ||
        profile.oracleCoverage != OracleCoverage::AllMeasuredSamples ||
        profile.deadlineMilliseconds != (isTinyMl ? 10000u : 120000u) ||
        profile.totalSamples() != (isTinyMl ? 5u : 1u))
      fail("repository manifest changed its bounded smoke input profiles");
  }

  SelectedApplicationInput tinyMl = take(selectApplicationInput(
      manifest, "mlperf-tiny-anomaly-detection", "smoke"));
  if (tinyMl.source.kind != SourceKind::Repository ||
      tinyMl.source.root != "test/applications/mlperf-tiny-anomaly" ||
      tinyMl.build.entry != "runner.cpp" ||
      tinyMl.build.language != LanguageMode::Cxx ||
      tinyMl.build.sources != std::vector<std::string>{"runner.cpp"} ||
      tinyMl.build.compilerOptions !=
          std::vector<std::string>{"-std=c++17", "-O1", "-ffp-contract=off",
                                   "-fno-exceptions", "-fno-rtti"} ||
      !tinyMl.build.linkOptions.empty() || tinyMl.cachedInputs.size() != 2 ||
      tinyMl.cachedInputs[0].logicalName != "model" ||
      tinyMl.cachedInputs[0].path !=
          "externals/mlperf-tiny/benchmark/training/anomaly_detection/"
          "trained_models/ad01_int8.tflite" ||
      tinyMl.cachedInputs[1].logicalName != "smoke-dataset" ||
      tinyMl.cachedInputs[1].path !=
          "externals/mlperf-tiny/benchmark/reference_submissions/"
          "anomaly_detection/datasets/dcase01/"
          "normal_id_01_00000000_hist_librosa.bin" ||
      tinyMl.input.workload != "mlperf-tiny-anomaly-smoke" ||
      tinyMl.input.runtimeInput != "mlperf-tiny-anomaly-smoke-input" ||
      tinyMl.input.oracle.kind != OracleKind::Exact ||
      tinyMl.input.oracle.entry !=
          "test/applications/mlperf-tiny-anomaly/expected-smoke.txt" ||
      formatBlobDigestHex(tinyMl.cachedInputs[0].digest) !=
          "87cf24194ef93d1d9b11a591d805526b98008e351655d29883c825c9c106ba24" ||
      formatBlobDigestHex(tinyMl.cachedInputs[1].digest) !=
          "31bc130d27e3732e1c09db946ccc7bfa130f98739bcd90cfa39d590f61f4d6fa")
    fail("repository manifest changed the TinyML inference selection");

  const std::vector<std::string> gitlinkApplications = {"gapbs-pagerank",
                                                        "llama2c-kernels"};
  auto gitlinkOutcomes = take(admitApplicationSources(
      manifest, gitlinkApplications, sharedRepositoryRoot));
  if (gitlinkOutcomes.size() != gitlinkApplications.size())
    fail("repository manifest Gitlink admission changed cardinality");
  for (auto [index, identity] : llvm::enumerate(gitlinkApplications)) {
    const auto *admitted =
        std::get_if<AdmittedApplicationSource>(&gitlinkOutcomes[index]);
    if (!admitted || admitted->applicationIdentity != identity ||
        !std::filesystem::path(admitted->sourceRoot).is_absolute())
      fail("repository manifest Gitlink source was not admitted");
  }

  const std::vector<std::string> repositoryApplications = {
      "loom-multisensor-attention", "vecadd-memory"};
  auto repositoryOutcomes = take(admitApplicationSources(
      manifest, repositoryApplications, repositoryRoot));
  if (repositoryOutcomes.size() != repositoryApplications.size())
    fail("repository-owned source admission changed cardinality");
  for (auto [index, identity] : llvm::enumerate(repositoryApplications)) {
    const auto *admitted =
        std::get_if<AdmittedApplicationSource>(&repositoryOutcomes[index]);
    if (!admitted || admitted->applicationIdentity != identity ||
        !std::filesystem::path(admitted->sourceRoot).is_absolute())
      fail("repository-owned application source was not admitted");
  }

  ApplicationSourceAdmissionOutcome tinyMlAdmission =
      take(admitApplicationSource(manifest,
                                  "mlperf-tiny-anomaly-detection", "smoke",
                                  repositoryRoot, sharedRepositoryRoot));
  const auto *admittedTinyMl =
      std::get_if<AdmittedApplicationSource>(&tinyMlAdmission);
  if (!admittedTinyMl || admittedTinyMl->applicationIdentity !=
                             "mlperf-tiny-anomaly-detection" ||
      !std::filesystem::path(admittedTinyMl->sourceRoot).is_absolute())
    fail("TinyML source, oracle, model, or dataset was not admitted");
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 4)
    fail("expected <temporary-directory> <manifest> <repository-root>");
  exerciseManifestAndAdmission(argv[1]);
  const llvm::StringRef configuredSharedRoot = LOOM_TEST_REPOSITORY_ROOT;
  exerciseRepositoryManifest(argv[2], argv[3], configuredSharedRoot.empty()
                                                    ? argv[3]
                                                    : configuredSharedRoot);
  return 0;
}
