#include "ExternalTool/InvocationBundle.h"

#include "Common/BlobDigest.h"
#include "Common/ExecutionControl.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

using namespace loom::external_tool;

namespace {

[[noreturn]] void fail(const char *test, const std::string &message) {
  std::cerr << test << ": " << message << '\n';
  std::exit(1);
}

void require(const char *test, bool condition, const std::string &message) {
  if (!condition)
    fail(test, message);
}

template <typename T> T take(const char *test, llvm::Expected<T> value) {
  if (!value)
    fail(test, llvm::toString(value.takeError()));
  return std::move(*value);
}

template <typename T>
void requireFailureContains(const char *test, llvm::Expected<T> value,
                            llvm::StringRef reason) {
  if (value)
    fail(test, "expected failure containing " + reason.str());
  const std::string message = llvm::toString(value.takeError());
  require(test, message.find(reason.str()) != std::string::npos,
          "unexpected failure: " + message);
}

void requireFailureContains(const char *test, llvm::Error error,
                            llvm::StringRef reason) {
  if (!error)
    fail(test, "expected failure containing " + reason.str());
  const std::string message = llvm::toString(std::move(error));
  require(test, message.find(reason.str()) != std::string::npos,
          "unexpected failure: " + message);
}

void writeText(const std::filesystem::path &path, llvm::StringRef contents) {
  std::filesystem::create_directories(path.parent_path());
  std::ofstream stream(path, std::ios::binary);
  if (!stream)
    fail(__func__, "cannot write " + path.string());
  stream << contents.str();
}

void writeExecutable(const std::filesystem::path &path,
                     llvm::StringRef contents) {
  writeText(path, contents);
  std::filesystem::permissions(path,
                               std::filesystem::perms::owner_read |
                                   std::filesystem::perms::owner_write |
                                   std::filesystem::perms::owner_exec |
                                   std::filesystem::perms::group_read |
                                   std::filesystem::perms::group_exec,
                               std::filesystem::perm_options::replace);
}

std::string readText(const std::filesystem::path &path) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream)
    fail(__func__, "cannot read " + path.string());
  std::ostringstream result;
  result << stream.rdbuf();
  return result.str();
}

loom::BlobDigest digest(llvm::StringRef contents) {
  const auto *bytes = reinterpret_cast<const std::uint8_t *>(contents.data());
  return loom::computeBlobDigest(
      llvm::ArrayRef<std::uint8_t>(bytes, contents.size()));
}

ExternalToolInvocationBundleSpec baseSpec(const std::filesystem::path &tool) {
  ExternalToolInvocationBundleSpec specification;
  specification.semanticContract.providerIdentity = "fake_schedule_tool@1";
  specification.semanticContract.semanticClosure =
      SemanticInvocationClosure(CandidateGeneratorInvocationClosure{
          {0x01, 0x02}, {0x03, 0x04}, digest("fake-binding").bytes()});
  specification.semanticContract.resultImporterIdentity = std::string(64, 'a');
  specification.tool = ResolvedToolBinding{"fake_schedule_tool",
                                           ToolBindingSource::Explicit,
                                           tool.string(),
                                           "Fake Schedule Tool 1.0",
                                           {},
                                           {},
                                           std::nullopt,
                                           std::nullopt};
  specification.toolVersionProbe =
      ToolVersionProbe{{"--version"}, "Fake Schedule Tool 1.0"};
  specification.runtime.kind = InvocationRuntimeKind::Host;
  return specification;
}

std::string buildDriver(std::size_t ordinal, std::size_t count,
                        llvm::StringRef executable) {
  return executable.str() + "\n" + std::to_string(ordinal) + "\n" +
         std::to_string(count) + "\n";
}

void parallelBuildsJoinBeforeController(const std::filesystem::path &root,
                                        const std::filesystem::path &tool) {
  constexpr std::size_t buildCount = 4;
  ExternalToolInvocationBundleSpec specification = baseSpec(tool);
  std::vector<std::string> executables;
  for (std::size_t ordinal = 0; ordinal != buildCount; ++ordinal) {
    const std::string driver =
        "drivers/build-" + std::to_string(ordinal) + ".txt";
    const std::string executable =
        "work/build-" + std::to_string(ordinal) + "/simulator";
    specification.files.push_back({driver,
                                   buildDriver(ordinal, buildCount, executable),
                                   std::nullopt, false});
    specification.commands.push_back({tool.string(), "compile", driver});
    executables.push_back(executable);
  }
  specification.commands.push_back({executables.front(), "outputs/result.txt",
                                    executables[1], executables[2],
                                    executables[3]});
  specification.declaredOutputs = {"outputs/result.txt"};
  specification.toolProducedExecutables = executables;
  specification.parallelCommandGroups = {{0, buildCount, buildCount}};

  const std::filesystem::path bundle = root / "parallel-builds";
  const PreparedExternalToolInvocation prepared =
      take(__func__, finalizeExternalToolInvocationBundle(bundle.string(),
                                                          specification));
  const ExternalToolInvocationExecutionObservation observation = take(
      __func__, executeExternalToolInvocationBundleObserved(
                    prepared, {}, ExternalToolResultReusePolicy::RequireFresh));
  require(__func__,
          observation.exitCode == 0 &&
              observation.commandExecutions.size() == buildCount + 1,
          "parallel build invocation did not report every command");
  for (std::size_t ordinal = 0; ordinal != observation.commandExecutions.size();
       ++ordinal)
    require(__func__,
            observation.commandExecutions[ordinal].commandOrdinal == ordinal &&
                observation.commandExecutions[ordinal].exitCode == 0,
            "command observations are not canonical");
  for (std::size_t ordinal = 0; ordinal != buildCount; ++ordinal)
    require(__func__,
            observation.commandExecutions[ordinal].wallNanoseconds >=
                10'000'000,
            "build wall-time observation was not collected");
  require(__func__, readText(bundle / "outputs/result.txt") == "joined",
          "controller did not observe the complete build barrier");
  require(__func__,
          readText(bundle / "outputs/stdout.log") ==
                  "stdout-0\nstdout-1\nstdout-2\nstdout-3\n" &&
              readText(bundle / "outputs/stderr.log") ==
                  "stderr-0\nstderr-1\nstderr-2\nstderr-3\n",
          "parallel command logs are not collected in ordinal order");
  require(__func__,
          !std::filesystem::exists(bundle / ".loom-command-execution"),
          "successful parallel execution left launcher scratch");
  for (const std::filesystem::directory_entry &entry :
       std::filesystem::directory_iterator(bundle))
    require(__func__,
            !llvm::StringRef(entry.path().filename().string())
                 .starts_with(".loom-command-observations.partial."),
            "successful parallel execution left a partial observation");
  ExternalToolInvocationExecutionObservation inconsistent = observation;
  inconsistent.commandExecutions.front().exitCode = 1;
  requireFailureContains(__func__,
                         validateExternalToolInvocationExecutionObservation(
                             prepared, inconsistent),
                         "command results are inconsistent");
  const std::string manifest = readText(bundle / "tool-invocation.json");
  require(__func__,
          manifest.find("\"parallel_command_groups\"") != std::string::npos &&
              manifest.find("\"worker_limit\": 4") != std::string::npos,
          "parallel build schedule is absent from the manifest");
}

void lowestOrdinalFailureWins(const std::filesystem::path &root,
                              const std::filesystem::path &tool) {
  ExternalToolInvocationBundleSpec specification = baseSpec(tool);
  specification.commands = {{tool.string(), "fail", "0", "71"},
                            {tool.string(), "fail", "1", "72"},
                            {tool.string(), "fail", "2", "73"}};
  specification.parallelCommandGroups = {{0, 3, 3}};
  const std::filesystem::path bundle = root / "parallel-failure";
  const PreparedExternalToolInvocation prepared =
      take(__func__, finalizeExternalToolInvocationBundle(bundle.string(),
                                                          specification));
  const ExternalToolInvocationExecutionObservation observation = take(
      __func__, executeExternalToolInvocationBundleObserved(
                    prepared, {}, ExternalToolResultReusePolicy::RequireFresh));
  require(__func__,
          observation.exitCode == 71 &&
              observation.commandExecutions.size() == 3,
          "parallel failure did not select the lowest command ordinal");
  for (std::size_t ordinal = 0; ordinal != 3; ++ordinal)
    require(__func__,
            observation.commandExecutions[ordinal].commandOrdinal == ordinal &&
                observation.commandExecutions[ordinal].exitCode ==
                    static_cast<int>(71 + ordinal),
            "parallel failure observations are incomplete");
  require(__func__,
          readText(bundle / "outputs/stdout.log") == "fail-0\nfail-1\nfail-2\n",
          "failed command logs are not canonical");
}

void signaledCommandRetainsTypedExit(const std::filesystem::path &root,
                                     const std::filesystem::path &tool) {
  ExternalToolInvocationBundleSpec specification = baseSpec(tool);
  specification.commands = {{tool.string(), "crash"}};
  const std::filesystem::path bundle = root / "signal-failure";
  const PreparedExternalToolInvocation prepared =
      take(__func__, finalizeExternalToolInvocationBundle(bundle.string(),
                                                          specification));
  const ExternalToolInvocationExecutionObservation observation = take(
      __func__, executeExternalToolInvocationBundleObserved(
                    prepared, {}, ExternalToolResultReusePolicy::RequireFresh));
  const int expectedExitCode = 128 + SIGSEGV;
  require(__func__,
          observation.exitCode == expectedExitCode &&
              observation.commandExecutions.size() == 1 &&
              observation.commandExecutions.front().exitCode ==
                  expectedExitCode,
          "signaled command was not reported as its typed shell exit");
  require(__func__,
          readText(bundle / "outputs/stderr.log").find("Segmentation fault") !=
              std::string::npos,
          "signaled command diagnostic was not retained in stderr");
}

void observationStateObstructionFailsBeforeLaunch(
    const std::filesystem::path &root, const std::filesystem::path &tool) {
  ExternalToolInvocationBundleSpec specification = baseSpec(tool);
  specification.commands = {{tool.string(), "noop"}};
  const std::filesystem::path bundle = root / "observation-initialization";
  const PreparedExternalToolInvocation prepared =
      take(__func__, finalizeExternalToolInvocationBundle(bundle.string(),
                                                          specification));
  std::error_code directoryError;
  std::filesystem::create_directory(bundle / ".loom-command-observations",
                                    directoryError);
  require(__func__, !directoryError,
          "cannot obstruct the observation path: " + directoryError.message());
  requireFailureContains(
      __func__,
      executeExternalToolInvocationBundleObserved(
          prepared, {}, ExternalToolResultReusePolicy::RequireFresh),
      "could not clear prior invocation state");
}

void completedScriptKillsSurvivingChildren(const std::filesystem::path &root,
                                           const std::filesystem::path &tool) {
  ExternalToolInvocationBundleSpec specification = baseSpec(tool);
  specification.commands = {{tool.string(), "orphan", "outputs/late"}};
  const std::filesystem::path bundle = root / "surviving-child";
  const PreparedExternalToolInvocation prepared =
      take(__func__, finalizeExternalToolInvocationBundle(bundle.string(),
                                                          specification));
  const ExternalToolInvocationExecutionObservation observation = take(
      __func__, executeExternalToolInvocationBundleObserved(
                    prepared, {}, ExternalToolResultReusePolicy::RequireFresh));
  require(__func__, observation.exitCode == 0,
          "parent command with a surviving child did not complete");
  std::this_thread::sleep_for(std::chrono::milliseconds(650));
  require(__func__, !std::filesystem::exists(bundle / "outputs/late"),
          "completed run script left a surviving tool child");
}

void auxiliaryCommandOwnershipIsTyped(const std::filesystem::path &root,
                                      const std::filesystem::path &tool,
                                      const std::filesystem::path &auxiliary) {
  const ExternalFileFingerprint fingerprint =
      take(__func__, fingerprintExternalFile(auxiliary.string()));
  ExternalToolInvocationBundleSpec dataOnly = baseSpec(tool);
  dataOnly.commands = {{auxiliary.string(), "noop"}};
  dataOnly.externalFiles.push_back(
      {"schedule_data", "schedule_data", auxiliary.string(), fingerprint});
  requireFailureContains(
      __func__, finalizeExternalToolInvocationBundle(
                    (root / "data-as-command").string(), dataOnly),
      "typed auxiliary tool");

  ExternalToolInvocationBundleSpec typed = baseSpec(tool);
  typed.commands = {{auxiliary.string(), "noop"}};
  typed.auxiliaryToolExecutables.push_back(
      {"schedule_auxiliary", "schedule_auxiliary", auxiliary.string(),
       fingerprint});
  const std::filesystem::path bundle = root / "typed-auxiliary";
  const PreparedExternalToolInvocation prepared = take(
      __func__, finalizeExternalToolInvocationBundle(bundle.string(), typed));
  require(__func__,
          take(__func__, executeExternalToolInvocationBundle(prepared)) == 0,
          "typed auxiliary command did not execute");
  require(__func__,
          readText(bundle / "tool-invocation.json")
                  .find("\"auxiliary_tool_executables\"") !=
              std::string::npos,
          "typed auxiliary command owner is absent from the manifest");
}

void scheduleValidationAndCacheIdentity(const std::filesystem::path &root,
                                        const std::filesystem::path &tool) {
  ExternalToolInvocationBundleSpec invalid = baseSpec(tool);
  invalid.commands = {{tool.string(), "noop"},
                      {tool.string(), "noop"},
                      {tool.string(), "noop"}};
  invalid.parallelCommandGroups = {{0, 2, 2}, {1, 3, 2}};
  requireFailureContains(__func__,
                         finalizeExternalToolInvocationBundle(
                             (root / "overlap").string(), invalid),
                         "nonoverlapping");

  ExternalToolInvocationBundleSpec twoWorkers = baseSpec(tool);
  twoWorkers.commands = invalid.commands;
  twoWorkers.parallelCommandGroups = {{0, 3, 2}};
  ExternalToolInvocationBundleSpec threeWorkers = twoWorkers;
  threeWorkers.parallelCommandGroups = {{0, 3, 3}};
  const PreparedExternalToolInvocation twoPrepared =
      take(__func__, finalizeExternalToolInvocationBundle(
                         (root / "two-workers").string(), twoWorkers));
  const PreparedExternalToolInvocation threePrepared =
      take(__func__, finalizeExternalToolInvocationBundle(
                         (root / "three-workers").string(), threeWorkers));
  const ExternalToolResultCacheKey twoKey =
      take(__func__, deriveExternalToolResultCacheKey(twoPrepared));
  const ExternalToolResultCacheKey threeKey =
      take(__func__, deriveExternalToolResultCacheKey(threePrepared));
  require(__func__,
          twoKey.inputMaterialDigest == threeKey.inputMaterialDigest &&
              twoKey.toolVersionDigest == threeKey.toolVersionDigest &&
              twoKey.executionConfigurationDigest !=
                  threeKey.executionConfigurationDigest,
          "worker limit did not change only execution configuration identity");

  ExternalToolInvocationBundleSpec ordered = baseSpec(tool);
  ordered.commands = invalid.commands;
  const std::filesystem::path currentRoot = root / "ordered-current";
  const PreparedExternalToolInvocation current =
      take(__func__,
           finalizeExternalToolInvocationBundle(currentRoot.string(), ordered));
  std::string legacyManifest = readText(currentRoot / "tool-invocation.json");
  const std::string currentVersion = "\"version\": \"2.4\"";
  const std::size_t versionPosition = legacyManifest.find(currentVersion);
  require(__func__, versionPosition != std::string::npos,
          "current manifest version is absent");
  legacyManifest.replace(versionPosition, currentVersion.size(),
                         "\"version\": \"2.3\"");
  const std::string emptyAuxiliaryTools =
      "  \"auxiliary_tool_executables\": [],\n";
  const std::size_t auxiliaryPosition =
      legacyManifest.find(emptyAuxiliaryTools);
  require(__func__, auxiliaryPosition != std::string::npos,
          "current manifest omits the auxiliary tool domain");
  legacyManifest.erase(auxiliaryPosition, emptyAuxiliaryTools.size());
  const std::filesystem::path legacyRoot = root / "ordered-legacy";
  std::error_code copyError;
  std::filesystem::copy(currentRoot, legacyRoot,
                        std::filesystem::copy_options::recursive, copyError);
  require(__func__, !copyError,
          "cannot construct the legacy manifest fixture: " +
              copyError.message());
  writeText(legacyRoot / "tool-invocation.json", legacyManifest);
  const PreparedExternalToolInvocation legacy{legacyRoot.string(),
                                              digest(legacyManifest)};
  const ExternalToolResultCacheKey currentKey =
      take(__func__, deriveExternalToolResultCacheKey(current));
  const ExternalToolResultCacheKey legacyKey =
      take(__func__, deriveExternalToolResultCacheKey(legacy));
  require(__func__, currentKey == legacyKey,
          "empty 2.4 auxiliary tools changed the ordered 2.3 cache identity");
}

struct ExecutionDeadline final {
  std::chrono::steady_clock::time_point notAfter;
};

bool deadlineReached(const void *opaque) {
  return std::chrono::steady_clock::now() >=
         static_cast<const ExecutionDeadline *>(opaque)->notAfter;
}

std::optional<std::chrono::steady_clock::duration>
deadlineRemaining(const void *opaque) {
  const auto remaining =
      static_cast<const ExecutionDeadline *>(opaque)->notAfter -
      std::chrono::steady_clock::now();
  return remaining > std::chrono::steady_clock::duration::zero()
             ? remaining
             : std::chrono::steady_clock::duration::zero();
}

void controlledStopKillsParallelDescendants(const std::filesystem::path &root,
                                            const std::filesystem::path &tool) {
  ExternalToolInvocationBundleSpec specification = baseSpec(tool);
  specification.commands = {
      {tool.string(), "block", "outputs/child-0.pid", "outputs/late-0"},
      {tool.string(), "block", "outputs/child-1.pid", "outputs/late-1"}};
  specification.parallelCommandGroups = {{0, 2, 2}};
  const std::filesystem::path bundle = root / "parallel-stop";
  const PreparedExternalToolInvocation prepared =
      take(__func__, finalizeExternalToolInvocationBundle(bundle.string(),
                                                          specification));
  const ExecutionDeadline deadline{std::chrono::steady_clock::now() +
                                   std::chrono::milliseconds(150)};
  const loom::ExecutionControlView control{&deadline, deadlineReached,
                                           deadlineRemaining};
  const ExternalToolInvocationExecutionObservation observation =
      take(__func__,
           executeExternalToolInvocationBundleObserved(
               prepared, control, ExternalToolResultReusePolicy::RequireFresh));
  require(__func__,
          observation.exitCode == externalToolExecutionStoppedExitCode &&
              observation.invokedExternalTool &&
              observation.commandExecutions.empty(),
          "controlled stop did not retain its typed disposition");
  std::this_thread::sleep_for(std::chrono::milliseconds(650));
  require(__func__,
          !std::filesystem::exists(bundle / "outputs/late-0") &&
              !std::filesystem::exists(bundle / "outputs/late-1") &&
              !std::filesystem::exists(bundle / ".loom-command-execution"),
          "controlled stop left a parallel descendant or launcher scratch");
  for (const std::filesystem::directory_entry &entry :
       std::filesystem::directory_iterator(bundle))
    require(__func__,
            !llvm::StringRef(entry.path().filename().string())
                 .starts_with(".loom-command-observations.partial."),
            "controlled stop left a partial command observation");
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 2)
    fail("main", "expected one test directory");
  const std::filesystem::path root =
      std::filesystem::absolute(argv[1]).lexically_normal();
  std::filesystem::create_directories(root);
  const std::filesystem::path tool = root / "bin" / "fake-schedule-tool";
  const std::filesystem::path auxiliary =
      root / "bin" / "fake-schedule-auxiliary";
  writeExecutable(tool,
                  R"bash(#!/usr/bin/env bash
set -u
case "${1-}" in
  --version)
    printf '%s\n' 'Fake Schedule Tool 1.0'
    ;;
  compile)
    mapfile -t loom_driver <"$2"
    loom_output=${loom_driver[0]}
    loom_ordinal=${loom_driver[1]}
    loom_count=${loom_driver[2]}
    mkdir -p -- work/parallel-entered "$(dirname -- "$loom_output")"
    : >"work/parallel-entered/$loom_ordinal"
    loom_observed=0
    for ((loom_attempt = 0; loom_attempt != 500; ++loom_attempt)); do
      loom_observed=$(find work/parallel-entered -type f | wc -l)
      if (( loom_observed == loom_count )); then break; fi
      sleep 0.01
    done
    if (( loom_observed != loom_count )); then exit 79; fi
    sleep 0.02
    cat >"$loom_output" <<'LOOM_SIMULATOR'
#!/usr/bin/env bash
set -eu
loom_result=$1
shift
for loom_peer in "$@"; do [[ -x "$loom_peer" ]]; done
printf '%s' joined >"$loom_result"
LOOM_SIMULATOR
    chmod u+x "$loom_output"
    printf 'stdout-%s\n' "$loom_ordinal"
    printf 'stderr-%s\n' "$loom_ordinal" >&2
    ;;
  fail)
    printf 'fail-%s\n' "$2"
    exit "$3"
    ;;
  crash)
    kill -SEGV "$$"
    ;;
  orphan)
    (sleep 0.5; printf '%s' late >"$2") &
    ;;
  noop)
    :
    ;;
  block)
    (sleep 0.5; printf '%s' late >"$3") &
    printf '%s' "$!" >"$2"
    while :; do sleep 0.01; done
    ;;
  *)
    exit 64
    ;;
esac
)bash");
  writeExecutable(auxiliary, readText(tool));
  require("main", ::unsetenv("LOOM_EXTERNAL_TOOL_CACHE_ROOT") == 0,
          "cannot isolate persistent result reuse");
  parallelBuildsJoinBeforeController(root, tool);
  lowestOrdinalFailureWins(root, tool);
  signaledCommandRetainsTypedExit(root, tool);
  observationStateObstructionFailsBeforeLaunch(root, tool);
  completedScriptKillsSurvivingChildren(root, tool);
  auxiliaryCommandOwnershipIsTyped(root, tool, auxiliary);
  scheduleValidationAndCacheIdentity(root, tool);
  controlledStopKillsParallelDescendants(root, tool);
  return 0;
}
