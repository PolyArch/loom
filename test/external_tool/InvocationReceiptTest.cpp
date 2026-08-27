#include "ExternalTool/InvocationBundle.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Program.h"

#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <sys/file.h>
#include <thread>
#include <unistd.h>
#include <utility>
#include <variant>

using namespace loom::external_tool;

namespace {

constexpr int kFixtureToolExitCode = 93;

[[noreturn]] void fail(const char *test, const std::string &message) {
  std::cerr << test << ": " << message << '\n';
  std::exit(1);
}

void require(const char *test, bool condition, const std::string &message) {
  if (!condition)
    fail(test, message);
}

void requireSuccess(const char *test, llvm::Error error) {
  if (error)
    fail(test, llvm::toString(std::move(error)));
}

void requireFailure(const char *test, llvm::Error error,
                    const std::string &message) {
  if (!error)
    fail(test, message);
  llvm::consumeError(std::move(error));
}

template <typename T> T take(const char *test, llvm::Expected<T> value) {
  if (!value)
    fail(test, llvm::toString(value.takeError()));
  return std::move(*value);
}

template <typename T>
void requireFailure(const char *test, llvm::Expected<T> value,
                    const std::string &message) {
  if (value)
    fail(test, message);
  llvm::consumeError(value.takeError());
}

std::string readText(const std::filesystem::path &path) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream)
    fail(__func__, "could not open " + path.string());
  std::ostringstream contents;
  contents << stream.rdbuf();
  return contents.str();
}

void writeText(const std::filesystem::path &path, llvm::StringRef contents) {
  std::filesystem::create_directories(path.parent_path());
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  if (!stream)
    fail(__func__, "could not open " + path.string());
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

struct FenceAdmissionStop final {
  std::atomic<bool> requested{false};
  mutable std::atomic<unsigned> queries{0};
};

bool stopFenceAdmission(const void *context) {
  const auto &stop = *static_cast<const FenceAdmissionStop *>(context);
  stop.queries.fetch_add(1, std::memory_order_release);
  return stop.requested.load(std::memory_order_acquire);
}

bool neverStop(const void *) { return false; }

std::optional<std::chrono::steady_clock::duration>
expiredDeadline(const void *) {
  return std::chrono::steady_clock::duration::zero();
}

struct FenceAdmissionDeadline final {
  mutable std::atomic<unsigned> queries{0};
};

std::optional<std::chrono::steady_clock::duration>
expireWhileWaitingForFence(const void *context) {
  auto &deadline = *static_cast<const FenceAdmissionDeadline *>(context);
  const unsigned query =
      deadline.queries.fetch_add(1, std::memory_order_acq_rel);
  if (query < 2)
    return std::chrono::milliseconds(10);
  return std::chrono::steady_clock::duration::zero();
}

void requireFenceHeld(const char *test, const std::filesystem::path &path) {
  const int descriptor = ::open(path.c_str(), O_RDWR | O_CLOEXEC | O_NOFOLLOW);
  require(test, descriptor >= 0, "could not open the bundle fence probe");
  errno = 0;
  const int locked = ::flock(descriptor, LOCK_EX | LOCK_NB);
  const int lockError = errno;
  if (locked == 0)
    (void)::flock(descriptor, LOCK_UN);
  (void)::close(descriptor);
  require(test,
          locked != 0 && (lockError == EWOULDBLOCK || lockError == EAGAIN),
          "the bundle fence was not held by an independent description");
}

bool hasChildCommand(const llvm::sys::ProcessInfo &process,
                     llvm::StringRef command) {
  const std::string processId = std::to_string(process.Pid);
  std::ifstream children("/proc/" + processId + "/task/" + processId +
                         "/children");
  std::uint64_t child = 0;
  while (children >> child) {
    std::ifstream name("/proc/" + std::to_string(child) + "/comm");
    std::string spelling;
    if (std::getline(name, spelling) && spelling == command)
      return true;
  }
  return false;
}

loom::BlobDigest blobDigest(llvm::StringRef contents) {
  const auto *bytes = reinterpret_cast<const std::uint8_t *>(contents.data());
  return loom::computeBlobDigest(
      llvm::ArrayRef<std::uint8_t>(bytes, contents.size()));
}

ExternalToolInvocationBundleSpec
baseSpec(const std::filesystem::path &tool, llvm::StringRef output,
         llvm::StringRef value = "receipt-output") {
  ExternalToolInvocationBundleSpec spec;
  spec.semanticContract.providerIdentity = "receipt_fixture@1";
  spec.semanticContract.semanticClosure =
      SemanticInvocationClosure(CandidateGeneratorInvocationClosure{
          {0x01}, {0x02}, blobDigest("receipt-binding").bytes()});
  spec.semanticContract.resultImporterIdentity = std::string(64, 'a');
  spec.tool = ResolvedToolBinding{"receipt_fixture",
                                  ToolBindingSource::Explicit,
                                  tool.string(),
                                  "Receipt Fixture 1.0",
                                  {},
                                  {},
                                  std::nullopt,
                                  std::nullopt};
  spec.toolVersionProbe = ToolVersionProbe{{"--version"}, "Receipt Fixture"};
  spec.runtime.kind = InvocationRuntimeKind::Host;
  spec.commands = {{tool.string(), "run", value.str(), output.str()}};
  spec.declaredOutputs = {output.str()};
  return spec;
}

ExternalToolInvocationImportExpectation
importExpectation(const ExternalToolInvocationBundleSpec &spec) {
  return ExternalToolInvocationImportExpectation{
      spec.semanticContract, {}, {}, {}, spec.declaredOutputs};
}

PreparedExternalToolInvocation
prepare(const char *test, const std::filesystem::path &root,
        llvm::StringRef name, const ExternalToolInvocationBundleSpec &spec) {
  return take(test, finalizeExternalToolInvocationBundle(
                        (root / name.str()).string(), spec));
}

void interleavedGenerationRejectsOldCompletionAndReceipt(
    const std::filesystem::path &root, const std::filesystem::path &tool) {
  const std::string output = "outputs/interleaved.txt";
  const ExternalToolInvocationBundleSpec spec = baseSpec(tool, output);
  const PreparedExternalToolInvocation prepared =
      prepare(__func__, root, "interleaved-generation", spec);
  const ExternalToolInvocationExecutionObservation oldExecution = take(
      __func__, executeExternalToolInvocationBundleObserved(
                    prepared, {}, ExternalToolResultReusePolicy::RequireFresh));
  const std::filesystem::path completion =
      root / "interleaved-generation" / "outputs" / "completion.json";
  const std::string oldCompletion = readText(completion);

  const ExternalToolInvocationExecutionObservation newExecution = take(
      __func__, executeExternalToolInvocationBundleObserved(
                    prepared, {}, ExternalToolResultReusePolicy::RequireFresh));
  const loom::BlobDigest newToken = newExecution.attemptToken;
  require(__func__, newToken != oldExecution.attemptToken,
          "a new execution generation reused the prior attempt token");
  writeText(completion, oldCompletion);

  requireFailure(
      __func__,
      importExternalToolInvocationAttempt(prepared, importExpectation(spec)),
      "raw import accepted a completion from an old generation");
  ExternalToolInvocationExecutionObservation reboundExecution = oldExecution;
  reboundExecution.attemptToken = newToken;
  requireSuccess(__func__, validateExternalToolInvocationExecutionObservation(
                               prepared, reboundExecution));
  requireFailure(
      __func__,
      importExternalToolInvocationAttempt(prepared, importExpectation(spec),
                                          reboundExecution),
      "receipt-aware import accepted a rebound receipt from an old generation");
}

void expiredDeadlineDoesNotSupersedeGeneration(
    const std::filesystem::path &root, const std::filesystem::path &tool) {
  const std::string output = "outputs/deadline.txt";
  const ExternalToolInvocationBundleSpec spec = baseSpec(tool, output);
  const PreparedExternalToolInvocation prepared =
      prepare(__func__, root, "expired-deadline", spec);
  const ExternalToolInvocationExecutionObservation retained = take(
      __func__, executeExternalToolInvocationBundleObserved(
                    prepared, {}, ExternalToolResultReusePolicy::RequireFresh));

  const loom::ExecutionControlView control{nullptr, neverStop, expiredDeadline};
  auto stopped = executeExternalToolInvocationBundleObserved(
      prepared, control, ExternalToolResultReusePolicy::RequireFresh);
  require(__func__, !stopped,
          "an expired deadline admitted a new execution generation");
  bool typedAdmissionStop = false;
  llvm::Error remaining = llvm::handleErrors(
      stopped.takeError(),
      [&](const ExternalToolExecutionAdmissionStoppedError &) {
        typedAdmissionStop = true;
      });
  requireSuccess(__func__, std::move(remaining));
  require(__func__, typedAdmissionStop,
          "an expired deadline lost its typed admission failure");

  const InvocationCompletion completion =
      take(__func__, loadExternalToolInvocationCompletion(prepared));
  require(__func__, completion.attemptToken == retained.attemptToken,
          "an expired deadline superseded the durable generation");
  const ImportedExternalToolInvocationBundle imported = take(
      __func__,
      importExternalToolInvocationBundle(prepared, importExpectation(spec)));
  require(__func__,
          take(__func__, readExternalToolInvocationDeclaredOutput(
                             imported, output)) == "receipt-output",
          "an expired deadline changed the retained declared output");
}

void forgedInheritedDescriptorsDoNotBypassFence(
    const std::filesystem::path &root, const std::filesystem::path &tool) {
  const std::string output = "outputs/inherited-fence.txt";
  const ExternalToolInvocationBundleSpec spec = baseSpec(tool, output);
  const PreparedExternalToolInvocation prepared =
      prepare(__func__, root, "forged-inherited-fence", spec);
  const ExternalToolInvocationExecutionObservation retained = take(
      __func__, executeExternalToolInvocationBundleObserved(
                    prepared, {}, ExternalToolResultReusePolicy::RequireFresh));
  const std::string runScript =
      (std::filesystem::path(prepared.bundleRoot) / "run.sh").string();
  llvm::ErrorOr<std::string> bash = llvm::sys::findProgramByName("bash");
  require(__func__, static_cast<bool>(bash), "could not find bash");
  const std::array<llvm::StringRef, 6> arguments{
      *bash,
      "-c",
      "exec 198<>/dev/null; exec 199</; exec \"$1\" \"$2\" "
      "--loom-inherited-execution-fence",
      "loom-forged-fence",
      *bash,
      runScript};
  std::string message;
  bool executionFailed = false;
  const llvm::sys::ProcessInfo process = llvm::sys::ExecuteNoWait(
      *bash, arguments, std::nullopt, {}, 0, &message, &executionFailed);
  require(__func__, !executionFailed && process.Pid != 0,
          "could not start inherited-fence probe: " + message);
  const llvm::sys::ProcessInfo waited =
      llvm::sys::Wait(process, std::nullopt, &message);
  require(__func__,
          waited.Pid == process.Pid &&
              waited.ReturnCode ==
                  static_cast<int>(InvocationLauncherExitCode::LauncherFailure),
          "forged inherited descriptors did not fail in the launcher");

  const std::string fence = prepared.bundleRoot + ".loom-execution.lock";
  const std::array<llvm::StringRef, 8> unlockedArguments{
      *bash,
      "-c",
      "exec 198>>\"$1\"; exec 199<\"$2\"; exec \"$3\" \"$4\" "
      "--loom-inherited-execution-fence",
      "loom-unlocked-fence",
      fence,
      prepared.bundleRoot,
      *bash,
      runScript};
  const llvm::sys::ProcessInfo unlocked =
      llvm::sys::ExecuteNoWait(*bash, unlockedArguments, std::nullopt, {}, 0,
                               &message, &executionFailed);
  require(__func__, !executionFailed && unlocked.Pid != 0,
          "could not start unlocked-fence probe: " + message);
  const llvm::sys::ProcessInfo unlockedWaited =
      llvm::sys::Wait(unlocked, std::nullopt, &message);
  require(__func__,
          unlockedWaited.Pid == unlocked.Pid &&
              unlockedWaited.ReturnCode ==
                  static_cast<int>(InvocationLauncherExitCode::LauncherFailure),
          "an unlocked canonical fence entered inherited execution");

  const InvocationCompletion completion =
      take(__func__, loadExternalToolInvocationCompletion(prepared));
  require(__func__, completion.attemptToken == retained.attemptToken,
          "forged inherited descriptors superseded the durable generation");
  const ImportedExternalToolInvocationBundle imported = take(
      __func__,
      importExternalToolInvocationBundle(prepared, importExpectation(spec)));
  require(__func__,
          take(__func__, readExternalToolInvocationDeclaredOutput(
                             imported, output)) == "receipt-output",
          "forged inherited descriptors changed the retained output");
}

void overlappingExecutionsAreFencedByBundleRoot(
    const std::filesystem::path &root, const std::filesystem::path &tool) {
  const std::filesystem::path counter = root / "overlap-counter";
  const std::filesystem::path entered = root / "overlap-entered";
  const std::filesystem::path release = root / "overlap-release";
  const std::string output = "outputs/overlap.txt";
  ExternalToolInvocationBundleSpec spec = baseSpec(tool, output);
  spec.commands = {{tool.string(), "overlap-run", counter.string(),
                    entered.string(), release.string(), output}};
  const PreparedExternalToolInvocation prepared =
      prepare(__func__, root, "overlapping-executions", spec);

  std::optional<llvm::Expected<ExternalToolInvocationExecutionObservation>>
      firstResult;
  std::atomic<bool> firstReturned{false};
  std::thread first([&] {
    firstResult.emplace(executeExternalToolInvocationBundleObserved(
        prepared, {}, ExternalToolResultReusePolicy::RequireFresh));
    firstReturned.store(true, std::memory_order_release);
  });
  const std::filesystem::path firstEntered = entered.string() + ".detached";
  for (unsigned attempt = 0;
       attempt != 500 && !std::filesystem::exists(firstEntered); ++attempt)
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  require(__func__, std::filesystem::exists(firstEntered),
          "the first execution did not launch its detached descendant");
  const std::filesystem::path fence =
      prepared.bundleRoot + ".loom-execution.lock";
  requireFenceHeld(__func__, fence);
  for (unsigned attempt = 0;
       attempt != 500 && !firstReturned.load(std::memory_order_acquire);
       ++attempt)
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  require(__func__, firstReturned.load(std::memory_order_acquire),
          "the observed executor retained its detached descendant");
  first.join();
  require(__func__, firstResult.has_value(),
          "the first overlapping execution did not return");
  const ExternalToolInvocationExecutionObservation firstExecution =
      take(__func__, std::move(*firstResult));

  FenceAdmissionStop stop;
  std::optional<llvm::Expected<ExternalToolInvocationExecutionObservation>>
      cancelledResult;
  const loom::ExecutionControlView control{&stop, stopFenceAdmission};
  std::thread cancelled([&] {
    cancelledResult.emplace(executeExternalToolInvocationBundleObserved(
        prepared, control, ExternalToolResultReusePolicy::RequireFresh));
  });
  for (unsigned attempt = 0;
       attempt != 500 && stop.queries.load(std::memory_order_acquire) < 2;
       ++attempt)
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  require(__func__, stop.queries.load(std::memory_order_acquire) >= 2,
          "the cancelled execution did not contend for the bundle fence");
  stop.requested.store(true, std::memory_order_release);
  cancelled.join();
  require(__func__, cancelledResult.has_value() && !*cancelledResult,
          "cancelled fence admission returned an execution observation");
  bool typedAdmissionStop = false;
  llvm::Error remaining = llvm::handleErrors(
      cancelledResult->takeError(),
      [&](const ExternalToolExecutionAdmissionStoppedError &) {
        typedAdmissionStop = true;
      });
  requireSuccess(__func__, std::move(remaining));
  require(__func__, typedAdmissionStop,
          "cancelled fence admission lost its typed failure");

  FenceAdmissionDeadline deadline;
  const loom::ExecutionControlView deadlineControl{&deadline, neverStop,
                                                   expireWhileWaitingForFence};
  auto expired = executeExternalToolInvocationBundleObserved(
      prepared, deadlineControl, ExternalToolResultReusePolicy::RequireFresh);
  require(__func__, !expired,
          "a deadline that expired during fence contention admitted an "
          "execution");
  bool typedDeadlineStop = false;
  llvm::Error deadlineError = llvm::handleErrors(
      expired.takeError(),
      [&](const ExternalToolExecutionAdmissionStoppedError &) {
        typedDeadlineStop = true;
      });
  requireSuccess(__func__, std::move(deadlineError));
  require(__func__, typedDeadlineStop && deadline.queries.load() >= 3,
          "contended deadline admission lost its typed failure");
  const InvocationCompletion retainedCompletion =
      take(__func__, loadExternalToolInvocationCompletion(prepared));
  require(__func__,
          retainedCompletion.attemptToken == firstExecution.attemptToken,
          "cancelled fence admission superseded the durable generation");

  llvm::ErrorOr<std::string> bash = llvm::sys::findProgramByName("bash");
  require(__func__, static_cast<bool>(bash), "could not find bash");
  const std::string runScript =
      (std::filesystem::path(prepared.bundleRoot) / "run.sh").string();
  const std::array<llvm::StringRef, 2> directArguments{*bash, runScript};
  std::string message;
  bool executionFailed = false;
  const llvm::sys::ProcessInfo second = llvm::sys::ExecuteNoWait(
      *bash, directArguments, std::nullopt, {}, 0, &message, &executionFailed);
  require(__func__, !executionFailed && second.Pid != 0,
          "could not start direct execution: " + message);
  for (unsigned attempt = 0;
       attempt != 500 && !hasChildCommand(second, "flock"); ++attempt)
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  require(__func__,
          hasChildCommand(second, "flock") &&
              !std::filesystem::exists(entered.string() + ".2"),
          "direct execution did not block on the occupied bundle fence");
  requireFenceHeld(__func__, fence);

  writeText(release, "release");
  const llvm::sys::ProcessInfo waited =
      llvm::sys::Wait(second, std::nullopt, &message);
  require(__func__, waited.Pid == second.Pid && waited.ReturnCode == 0,
          "direct execution of the later generation failed");
  const InvocationCompletion secondCompletion =
      take(__func__, loadExternalToolInvocationCompletion(prepared));
  require(__func__,
          firstExecution.attemptToken != secondCompletion.attemptToken,
          "serialized executions reused an attempt token");

  const ImportedExternalToolInvocationBundle imported = take(
      __func__,
      importExternalToolInvocationBundle(prepared, importExpectation(spec)));
  require(__func__,
          take(__func__, readExternalToolInvocationDeclaredOutput(
                             imported, output)) == "2",
          "the first execution overwrote the later generation");
  requireFailure(__func__,
                 importExternalToolInvocationBundle(
                     prepared, importExpectation(spec), firstExecution),
                 "the superseded execution receipt remained importable");
}

void liveExecutionRemainsBoundToAdmittedBundleRoot(
    const std::filesystem::path &root, const std::filesystem::path &tool) {
  const std::filesystem::path marker = root / "root-rebind-entered";
  const std::filesystem::path release = root / "root-rebind-release";
  const std::string output = "outputs/root-rebind.txt";
  ExternalToolInvocationBundleSpec spec = baseSpec(tool, output);
  spec.commands = {{tool.string(), "root-rebind-run", marker.string(),
                    release.string(), output}};
  const PreparedExternalToolInvocation prepared =
      prepare(__func__, root, "root-rebind", spec);
  const std::filesystem::path logicalRoot(prepared.bundleRoot);
  const std::filesystem::path movedRoot = root / "root-rebind-moved";

  std::optional<llvm::Expected<ExternalToolInvocationExecutionObservation>>
      result;
  std::thread execution([&] {
    result.emplace(executeExternalToolInvocationBundleObserved(
        prepared, {}, ExternalToolResultReusePolicy::AllowExactReuse));
  });
  for (unsigned attempt = 0; attempt != 500 && !std::filesystem::exists(marker);
       ++attempt)
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  require(__func__, std::filesystem::exists(marker),
          "the root-rebind execution did not reach its command");
  std::error_code renameError;
  std::filesystem::rename(logicalRoot, movedRoot, renameError);
  require(__func__, !renameError,
          "could not rebind the logical bundle path: " + renameError.message());
  writeText(release, "release");
  execution.join();
  std::filesystem::rename(movedRoot, logicalRoot, renameError);
  require(__func__, !renameError,
          "could not restore the logical bundle path: " +
              renameError.message());
  require(__func__, result.has_value(),
          "the root-rebind execution did not return");
  const ExternalToolInvocationExecutionObservation observation =
      take(__func__, std::move(*result));
  const ImportedExternalToolInvocationBundle imported =
      take(__func__, importExternalToolInvocationBundle(
                         prepared, importExpectation(spec), observation));
  require(__func__,
          take(__func__, readExternalToolInvocationDeclaredOutput(
                             imported, output)) == "root-bound",
          "the live executor did not seal the admitted bundle-root inode");
}

void launcherExecFailureIsNotAnExecutionObservation(
    const std::filesystem::path &root, const std::filesystem::path &tool) {
  const ExternalToolInvocationBundleSpec spec =
      baseSpec(tool, "outputs/launcher-error.txt");
  const PreparedExternalToolInvocation prepared =
      prepare(__func__, root, "launcher-error", spec);
  const std::filesystem::path invalidPath = root / "invalid-launch-path";
  writeExecutable(invalidPath / "bash", "not an executable image\n");
  const char *oldPathValue = std::getenv("PATH");
  const std::optional<std::string> oldPath =
      oldPathValue ? std::optional<std::string>(oldPathValue) : std::nullopt;
  require(__func__, ::setenv("PATH", invalidPath.c_str(), 1) == 0,
          "could not install the launcher-error PATH");
  auto execution = executeExternalToolInvocationBundleObserved(
      prepared, {}, ExternalToolResultReusePolicy::AllowExactReuse);
  const int restoreResult =
      oldPath ? ::setenv("PATH", oldPath->c_str(), 1) : ::unsetenv("PATH");
  require(__func__, restoreResult == 0, "could not restore PATH");
  require(__func__, !execution,
          "an exec failure became an execution observation");
  const std::string error = llvm::toString(execution.takeError());
  require(__func__,
          error.find("could not execute generated run script") !=
              std::string::npos,
          "an exec failure bypassed the launcher error pipe: " + error);
}

void outputMutationCannotEscapeReceiptAwareImport(
    const std::filesystem::path &root, const std::filesystem::path &tool) {
  const std::string output = "outputs/output-mutation.txt";
  const ExternalToolInvocationBundleSpec spec =
      baseSpec(tool, output, "original-output");
  const PreparedExternalToolInvocation prepared =
      prepare(__func__, root, "output-mutation", spec);
  const ExternalToolInvocationExecutionObservation execution = take(
      __func__, executeExternalToolInvocationBundleObserved(
                    prepared, {}, ExternalToolResultReusePolicy::RequireFresh));
  writeText(root / "output-mutation" / output, "mutated-output");

  requireFailure(
      __func__,
      importExternalToolInvocationBundle(prepared, importExpectation(spec),
                                         execution),
      "receipt-aware import accepted output bytes changed after execution");
}

void completionReplacementInvalidatesReceipt(
    const std::filesystem::path &root, const std::filesystem::path &tool) {
  const std::string output = "outputs/completion-replacement.txt";
  const ExternalToolInvocationBundleSpec spec = baseSpec(tool, output);
  const PreparedExternalToolInvocation prepared =
      prepare(__func__, root, "completion-replacement", spec);
  const ExternalToolInvocationExecutionObservation execution = take(
      __func__, executeExternalToolInvocationBundleObserved(
                    prepared, {}, ExternalToolResultReusePolicy::RequireFresh));
  const std::filesystem::path completion =
      root / "completion-replacement" / "outputs" / "completion.json";
  std::string replacement = readText(completion);
  const std::string success = "\"status\":\"success\",\"exit_code\":0";
  const std::string failure = "\"status\":\"tool_exit\",\"exit_code\":" +
                              std::to_string(kFixtureToolExitCode);
  const std::size_t statusOffset = replacement.find(success);
  require(__func__, statusOffset != std::string::npos,
          "could not locate the canonical completion status");
  replacement.replace(statusOffset, success.size(), failure);
  constexpr llvm::StringLiteral outputMember = "\"output_sha256\":[";
  const std::size_t outputBegin = replacement.find(outputMember.str());
  require(__func__, outputBegin != std::string::npos,
          "could not locate the canonical completion outputs");
  const std::size_t digestBegin = outputBegin + outputMember.size();
  const std::size_t digestEnd = replacement.find(']', digestBegin);
  require(__func__, digestEnd != std::string::npos,
          "could not locate the canonical completion output terminator");
  replacement.erase(digestBegin, digestEnd - digestBegin);
  writeText(completion, replacement);

  const InvocationCompletion parsed =
      take(__func__, loadExternalToolInvocationCompletion(prepared));
  require(__func__,
          parsed.status == InvocationCompletionStatus::ToolExit &&
              parsed.exitCode == kFixtureToolExitCode,
          "the replacement completion was not a valid failed record");
  requireFailure(
      __func__,
      validateExternalToolInvocationExecutionReceipt(prepared, execution),
      "receipt validation accepted a completion with another exit code");
  requireFailure(__func__,
                 importExternalToolInvocationAttempt(
                     prepared, importExpectation(spec), execution),
                 "receipt-aware import accepted a replaced completion record");
}

void cacheHitCarriesAnImportableReceipt(const std::filesystem::path &root,
                                        const std::filesystem::path &tool) {
  const std::filesystem::path cache = root / "result-cache";
  require(__func__,
          ::setenv("LOOM_EXTERNAL_TOOL_CACHE_ROOT", cache.c_str(), 1) == 0,
          "could not enable the result cache");
  const std::filesystem::path counter = root / "cache-tool-entry-count";
  const std::string output = "outputs/cache-hit.txt";
  ExternalToolInvocationBundleSpec spec = baseSpec(tool, output, "cached");
  spec.commands = {
      {tool.string(), "counted-run", counter.string(), "cached", output}};

  const PreparedExternalToolInvocation population =
      prepare(__func__, root, "cache-population", spec);
  const ExternalToolInvocationExecutionObservation populationExecution =
      take(__func__, executeExternalToolInvocationBundleObserved(population));
  require(__func__,
          populationExecution.cacheLookup ==
              ExternalToolResultCacheLookup::Miss,
          "the cache population was not a miss");

  const PreparedExternalToolInvocation hit =
      prepare(__func__, root, "cache-hit", spec);
  const ExternalToolInvocationExecutionObservation hitExecution =
      take(__func__, executeExternalToolInvocationBundleObserved(hit));
  require(__func__,
          hitExecution.cacheLookup == ExternalToolResultCacheLookup::Hit &&
              !hitExecution.invokedExternalTool && readText(counter) == "1",
          "the exact cache hit re-entered the external tool");
  const ImportedExternalToolInvocationBundle imported =
      take(__func__, importExternalToolInvocationBundle(
                         hit, importExpectation(spec), hitExecution));
  require(__func__,
          take(__func__, readExternalToolInvocationDeclaredOutput(
                             imported, output)) == "cached",
          "the cache-hit receipt did not import its exact output");
  require(__func__, ::unsetenv("LOOM_EXTERNAL_TOOL_CACHE_ROOT") == 0,
          "could not disable the result cache");
}

void failedExecutionCarriesAnImportableReceipt(
    const std::filesystem::path &root, const std::filesystem::path &tool) {
  ExternalToolInvocationBundleSpec spec =
      baseSpec(tool, "outputs/unused-failure.txt");
  spec.commands = {{tool.string(), "fail"}};
  spec.declaredOutputs.clear();
  const PreparedExternalToolInvocation prepared =
      prepare(__func__, root, "failed-execution", spec);
  const ExternalToolInvocationExecutionObservation execution = take(
      __func__, executeExternalToolInvocationBundleObserved(
                    prepared, {}, ExternalToolResultReusePolicy::RequireFresh));
  const ExternalToolInvocationAttemptOutcome imported =
      take(__func__, importExternalToolInvocationAttempt(
                         prepared, importExpectation(spec), execution));
  require(__func__,
          std::holds_alternative<FailedExternalToolInvocationAttempt>(imported),
          "a failed receipt did not import as a failed attempt");
  const auto &failure = std::get<FailedExternalToolInvocationAttempt>(imported);
  require(__func__,
          failure.status == InvocationCompletionStatus::ToolExit &&
              failure.exitCode == kFixtureToolExitCode,
          "a failed receipt lost its exact completion disposition");
}

struct StopWhenEntered final {
  std::filesystem::path marker;
};

bool stopWhenEntered(const void *opaque) {
  return std::filesystem::exists(
      static_cast<const StopWhenEntered *>(opaque)->marker);
}

void stoppedExecutionCarriesAnImportableReceipt(
    const std::filesystem::path &root, const std::filesystem::path &tool) {
  const std::string output = "outputs/stopped.txt";
  ExternalToolInvocationBundleSpec spec = baseSpec(tool, output);
  spec.commands = {{tool.string(), "controlled-block", "outputs/entered",
                    "outputs/late", output}};
  const PreparedExternalToolInvocation prepared =
      prepare(__func__, root, "stopped-execution", spec);
  const StopWhenEntered stop{root / "stopped-execution" / "outputs" /
                             "entered"};
  const loom::ExecutionControlView control{&stop, stopWhenEntered};
  const ExternalToolInvocationExecutionObservation execution =
      take(__func__,
           executeExternalToolInvocationBundleObserved(
               prepared, control, ExternalToolResultReusePolicy::RequireFresh));
  require(__func__,
          execution.exitCode == externalToolExecutionStoppedExitCode &&
              execution.invokedExternalTool &&
              !std::filesystem::exists(root / "stopped-execution" / "outputs" /
                                       "completion.json"),
          "controlled execution did not preserve its incomplete disposition");
  const ExternalToolInvocationAttemptOutcome imported =
      take(__func__, importExternalToolInvocationAttempt(
                         prepared, importExpectation(spec), execution));
  require(
      __func__,
      std::holds_alternative<IncompleteExternalToolInvocationAttempt>(imported),
      "a stopped receipt did not import as an incomplete attempt");
  std::this_thread::sleep_for(std::chrono::milliseconds(700));
  require(
      __func__,
      !std::filesystem::exists(root / "stopped-execution" / "outputs" / "late"),
      "a descendant survived the stopped external-tool process group");
}

struct StopAfterCompletion final {
  std::filesystem::path completion;
};

bool stopAfterCompletion(const void *opaque) {
  return std::filesystem::exists(
      static_cast<const StopAfterCompletion *>(opaque)->completion);
}

void stoppedPostflightRemainsIncomplete(const std::filesystem::path &root,
                                        const std::filesystem::path &tool) {
  const std::string output = "outputs/postflight-stop.txt";
  const ExternalToolInvocationBundleSpec spec = baseSpec(tool, output);
  const PreparedExternalToolInvocation prepared =
      prepare(__func__, root, "postflight-stop", spec);
  const StopAfterCompletion stop{root / "postflight-stop" / "outputs" /
                                 "completion.json"};
  const loom::ExecutionControlView control{&stop, stopAfterCompletion};
  const ExternalToolInvocationExecutionObservation execution =
      take(__func__,
           executeExternalToolInvocationBundleObserved(
               prepared, control, ExternalToolResultReusePolicy::RequireFresh));
  require(__func__,
          execution.exitCode == externalToolExecutionStoppedExitCode &&
              execution.invokedExternalTool &&
              !std::filesystem::exists(stop.completion),
          "stopped postflight retained a successful completion");
  const ExternalToolInvocationAttemptOutcome imported =
      take(__func__, importExternalToolInvocationAttempt(
                         prepared, importExpectation(spec), execution));
  require(
      __func__,
      std::holds_alternative<IncompleteExternalToolInvocationAttempt>(imported),
      "stopped postflight did not retain its incomplete disposition");
}

void publicObservationCannotSubstituteForAReceipt(
    const std::filesystem::path &root, const std::filesystem::path &tool) {
  const std::string output = "outputs/unsealed.txt";
  const ExternalToolInvocationBundleSpec spec = baseSpec(tool, output);
  const PreparedExternalToolInvocation prepared =
      prepare(__func__, root, "unsealed-observation", spec);
  ExternalToolInvocationExecutionObservation execution = take(
      __func__, executeExternalToolInvocationBundleObserved(
                    prepared, {}, ExternalToolResultReusePolicy::RequireFresh));
  execution.receipt = {};
  requireFailure(
      __func__,
      importExternalToolInvocationBundle(prepared, importExpectation(spec),
                                         execution),
      "public execution fields substituted for a sealed executor receipt");
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 2)
    fail("main", "expected one test-directory argument");
  const std::filesystem::path root =
      std::filesystem::absolute(argv[1]).lexically_normal();
  std::filesystem::create_directories(root);
  const std::filesystem::path tool = root / "tool bin" / "receipt fixture";
  writeExecutable(tool, "#!/usr/bin/env bash\n"
                        "set -u\n"
                        "case \"${1-}\" in\n"
                        "  --version) printf '%s\\n' 'Receipt Fixture 1.0' ;;\n"
                        "  run) printf '%s' \"$2\" >\"$3\" ;;\n"
                        "  counted-run)\n"
                        "    value=0\n"
                        "    if [[ -f \"$2\" ]]; then IFS= read -r value "
                        "<\"$2\"; fi\n"
                        "    printf '%s' \"$((value + 1))\" >\"$2\"\n"
                        "    printf '%s' \"$3\" >\"$4\"\n"
                        "    ;;\n"
                        "  overlap-run)\n"
                        "    value=0\n"
                        "    if [[ -f \"$2\" ]]; then IFS= read -r value "
                        "<\"$2\"; fi\n"
                        "    value=$((value + 1))\n"
                        "    printf '%s' \"$value\" >\"$2\"\n"
                        "    printf '%s' entered >\"$3.$value\"\n"
                        "    printf '%s' \"$value\" >\"$5\"\n"
                        "    if [[ \"$value\" == 1 ]]; then\n"
                        "      setsid --fork bash -c '\n"
                        "        printf %s entered >\"$1.detached\"\n"
                        "        while [[ ! -f \"$2\" ]]; do sleep 0.01; "
                        "done\n"
                        "        printf %s \"$3\" >\"$4\"\n"
                        "      ' loom-detached \"$3\" \"$4\" \"$value\" "
                        "\"$5\" </dev/null >/dev/null 2>&1 &\n"
                        "    fi\n"
                        "    ;;\n"
                        "  root-rebind-run)\n"
                        "    printf '%s' entered >\"$2\"\n"
                        "    while [[ ! -f \"$3\" ]]; do sleep 0.01; done\n"
                        "    printf '%s' root-bound >\"$4\"\n"
                        "    ;;\n"
                        "  fail) exit 93 ;;\n"
                        "  controlled-block)\n"
                        "    printf '%s' entered >\"$2\"\n"
                        "    (sleep 0.6; printf '%s' late >\"$3\") &\n"
                        "    while :; do sleep 0.01; done\n"
                        "    ;;\n"
                        "  *) exit 64 ;;\n"
                        "esac\n");
  require("main", ::unsetenv("LOOM_EXTERNAL_TOOL_CACHE_ROOT") == 0,
          "could not isolate the result-cache fixture");

  interleavedGenerationRejectsOldCompletionAndReceipt(root, tool);
  expiredDeadlineDoesNotSupersedeGeneration(root, tool);
  forgedInheritedDescriptorsDoNotBypassFence(root, tool);
  overlappingExecutionsAreFencedByBundleRoot(root, tool);
  liveExecutionRemainsBoundToAdmittedBundleRoot(root, tool);
  launcherExecFailureIsNotAnExecutionObservation(root, tool);
  outputMutationCannotEscapeReceiptAwareImport(root, tool);
  completionReplacementInvalidatesReceipt(root, tool);
  cacheHitCarriesAnImportableReceipt(root, tool);
  failedExecutionCarriesAnImportableReceipt(root, tool);
  stoppedExecutionCarriesAnImportableReceipt(root, tool);
  stoppedPostflightRemainsIncomplete(root, tool);
  publicObservationCannotSubstituteForAReceipt(root, tool);
  return 0;
}
