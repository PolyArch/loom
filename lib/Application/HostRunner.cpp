#include "Application/HostRunner.h"

#include "Common/BlobDigest.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <optional>
#include <string>
#include <system_error>
#include <thread>
#include <utility>
#include <variant>
#include <vector>

extern char **environ;

namespace loom::application {
namespace {

using MonotonicClock = std::chrono::steady_clock;

llvm::Error hostRunError(const llvm::Twine &message) {
  return llvm::createStringError(std::make_error_code(std::errc::io_error),
                                 "application_host_run_failed: " + message);
}

class TemporaryDirectory final {
public:
  TemporaryDirectory() = default;
  TemporaryDirectory(const TemporaryDirectory &) = delete;
  TemporaryDirectory &operator=(const TemporaryDirectory &) = delete;
  TemporaryDirectory(TemporaryDirectory &&other) noexcept
      : path_(std::move(other.path_)) {
    other.path_.clear();
  }

  ~TemporaryDirectory() {
    if (path_.empty())
      return;
    std::error_code ignored;
    std::filesystem::remove_all(path_, ignored);
  }

  static llvm::Expected<TemporaryDirectory>
  create(const std::filesystem::path &repositoryRoot) {
    const std::filesystem::path scratchRoot = repositoryRoot / "temp";
    std::error_code filesystemError;
    std::filesystem::create_directories(scratchRoot, filesystemError);
    if (filesystemError)
      return hostRunError("cannot create repository scratch root: " +
                          filesystemError.message());
    const std::filesystem::file_status scratchStatus =
        std::filesystem::symlink_status(scratchRoot, filesystemError);
    if (filesystemError || !std::filesystem::is_directory(scratchStatus) ||
        std::filesystem::is_symlink(scratchStatus))
      return hostRunError("repository scratch root is not a regular directory");

    llvm::SmallString<256> path;
    const std::filesystem::path prefix =
        scratchRoot / "loom-application-host-run";
    if (std::error_code error =
            llvm::sys::fs::createUniqueDirectory(prefix.string(), path))
      return hostRunError("cannot create temporary directory: " +
                          error.message());
    TemporaryDirectory result;
    result.path_ = path.str().str();
    return std::move(result);
  }

  std::string child(llvm::StringRef name) const {
    return (std::filesystem::path(path_) / name.str()).string();
  }

private:
  std::string path_;
};

llvm::Expected<std::string> readFile(llvm::StringRef path,
                                     llvm::StringRef description) {
  auto buffer = llvm::MemoryBuffer::getFile(path, false, false);
  if (!buffer)
    return hostRunError("cannot read " + description + " '" + path +
                        "': " + buffer.getError().message());
  return (*buffer)->getBuffer().str();
}

llvm::Error prepareCaptureFile(llvm::StringRef path,
                               llvm::StringRef description) {
  std::error_code error;
  llvm::raw_fd_ostream output(path, error, llvm::sys::fs::OF_None);
  if (error)
    return hostRunError("cannot create " + description + " '" + path +
                        "': " + error.message());
  output.close();
  if (output.has_error())
    return hostRunError("cannot initialize " + description + " '" + path + "'");
  return llvm::Error::success();
}

std::string appendExecutionMessage(std::string diagnostic,
                                   llvm::StringRef message) {
  if (message.empty())
    return diagnostic;
  if (!diagnostic.empty() && diagnostic.back() != '\n')
    diagnostic.push_back('\n');
  diagnostic.append(message.data(), message.size());
  return diagnostic;
}

llvm::SmallVector<llvm::StringRef, 32>
argumentRefs(const std::vector<std::string> &arguments) {
  llvm::SmallVector<llvm::StringRef, 32> result;
  result.reserve(arguments.size());
  for (const std::string &argument : arguments)
    result.push_back(argument);
  return result;
}

std::vector<std::string> deterministicHostEnvironment() {
  std::vector<std::string> result;
  for (char **entry = environ; entry && *entry; ++entry)
    if (!llvm::StringRef(*entry).starts_with("LC_ALL="))
      result.emplace_back(*entry);
  result.emplace_back("LC_ALL=C");
  return result;
}

struct BlockingCommandResult final {
  std::optional<int> exitStatus;
  bool executionFailed = false;
  std::string diagnostic;
};

llvm::Expected<BlockingCommandResult>
runBlockingCommand(llvm::StringRef executable,
                   const std::vector<std::string> &arguments,
                   llvm::StringRef outputPath, llvm::StringRef errorPath) {
  if (llvm::Error error =
          prepareCaptureFile(outputPath, "compiler stdout capture"))
    return std::move(error);
  if (llvm::Error error =
          prepareCaptureFile(errorPath, "compiler stderr capture"))
    return std::move(error);
  const auto refs = argumentRefs(arguments);
  const std::array<std::optional<llvm::StringRef>, 3> redirects = {
      llvm::StringRef(), outputPath, errorPath};
  std::string message;
  bool executionFailed = false;
  const int status =
      llvm::sys::ExecuteAndWait(executable, refs, std::nullopt, redirects, 0, 0,
                                &message, &executionFailed);
  auto diagnostic = readFile(errorPath, "compiler diagnostic");
  if (!diagnostic)
    return diagnostic.takeError();
  auto compilerOutput = readFile(outputPath, "compiler output");
  if (!compilerOutput)
    return compilerOutput.takeError();
  if (!compilerOutput->empty())
    *diagnostic =
        appendExecutionMessage(std::move(*diagnostic), *compilerOutput);
  return BlockingCommandResult{
      status >= 0 ? std::optional<int>(status) : std::nullopt,
      executionFailed || status < 0,
      appendExecutionMessage(std::move(*diagnostic), message)};
}

std::uint64_t elapsedNanoseconds(MonotonicClock::time_point begin) {
  const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
                           MonotonicClock::now() - begin)
                           .count();
  if (elapsed <= 0)
    return 0;
  using Elapsed = decltype(elapsed);
  if constexpr (sizeof(Elapsed) > sizeof(std::uint64_t))
    if (elapsed >
        static_cast<Elapsed>(std::numeric_limits<std::uint64_t>::max()))
      return std::numeric_limits<std::uint64_t>::max();
  return static_cast<std::uint64_t>(elapsed);
}

MonotonicClock::time_point
executionDeadline(MonotonicClock::time_point begin,
                  std::uint64_t deadlineMilliseconds) {
  using Milliseconds = std::chrono::milliseconds;
  const auto maximumMilliseconds =
      std::chrono::duration_cast<Milliseconds>(MonotonicClock::duration::max())
          .count();
  if (maximumMilliseconds <= 0 ||
      deadlineMilliseconds > static_cast<std::uint64_t>(maximumMilliseconds))
    return MonotonicClock::time_point::max();
  const auto duration = std::chrono::duration_cast<MonotonicClock::duration>(
      Milliseconds(static_cast<Milliseconds::rep>(deadlineMilliseconds)));
  if (duration > MonotonicClock::time_point::max() - begin)
    return MonotonicClock::time_point::max();
  return begin + duration;
}

struct BoundedCommandResult final {
  std::optional<int> exitStatus;
  std::uint64_t wallTimeNanoseconds = 0;
  bool timedOut = false;
  bool executionFailed = false;
  std::string output;
  std::string diagnostic;
};

constexpr std::chrono::milliseconds hostProcessPollInterval{1};
constexpr std::chrono::milliseconds hostProcessCleanupLimit{1000};

enum class HostProcessGroupState : std::uint8_t {
  Empty,
  Present,
  InspectionFailed,
};

HostProcessGroupState inspectHostProcessGroup(llvm::sys::procid_t processId,
                                              std::string &message) {
  if (::kill(-processId, 0) == 0 || errno == EPERM)
    return HostProcessGroupState::Present;
  if (errno == ESRCH)
    return HostProcessGroupState::Empty;
  const int error = errno;
  message = appendExecutionMessage(
      std::move(message),
      "cannot inspect host process group: " +
          std::error_code(error, std::generic_category()).message());
  return HostProcessGroupState::InspectionFailed;
}

void waitForHostProcessGroupExit(llvm::sys::procid_t processId,
                                 std::string &message) {
  const MonotonicClock::time_point cleanupDeadline =
      MonotonicClock::now() + hostProcessCleanupLimit;
  while (true) {
    const HostProcessGroupState state =
        inspectHostProcessGroup(processId, message);
    if (state != HostProcessGroupState::Present)
      return;
    const MonotonicClock::time_point now = MonotonicClock::now();
    if (now >= cleanupDeadline) {
      message = appendExecutionMessage(std::move(message),
                                       "host process group did not terminate");
      return;
    }
    std::this_thread::sleep_for(
        std::min(cleanupDeadline - now,
                 std::chrono::duration_cast<MonotonicClock::duration>(
                     hostProcessPollInterval)));
  }
}

void terminateHostProcessGroup(llvm::sys::procid_t processId,
                               bool leaderMayNeedFallback,
                               std::string &message) {
  if (::kill(-processId, SIGKILL) != 0) {
    const int groupError = errno;
    if (leaderMayNeedFallback && ::kill(processId, SIGKILL) == 0)
      return;
    const int leaderError = errno;
    if (groupError != ESRCH && (!leaderMayNeedFallback || leaderError != ESRCH))
      message = appendExecutionMessage(
          std::move(message),
          "cannot terminate host process group: " +
              std::error_code(groupError, std::generic_category()).message());
  }
}

llvm::sys::ProcessInfo
terminateHostProcess(const llvm::sys::ProcessInfo &process,
                     std::string &message) {
  terminateHostProcessGroup(process.Pid, true, message);
  std::string waitMessage;
  llvm::sys::ProcessInfo waited =
      llvm::sys::Wait(process, std::nullopt, &waitMessage);
  message = appendExecutionMessage(std::move(message), waitMessage);
  waitForHostProcessGroupExit(process.Pid, message);
  return waited;
}

llvm::Expected<BoundedCommandResult>
runBoundedCommand(llvm::StringRef executable,
                  const std::vector<std::string> &arguments,
                  llvm::StringRef outputPath, llvm::StringRef errorPath,
                  std::uint64_t deadlineMilliseconds) {
  if (llvm::Error error = prepareCaptureFile(outputPath, "host stdout capture"))
    return std::move(error);
  if (llvm::Error error = prepareCaptureFile(errorPath, "host stderr capture"))
    return std::move(error);
  const auto refs = argumentRefs(arguments);
  const std::vector<std::string> environment = deterministicHostEnvironment();
  const auto environmentRefs = argumentRefs(environment);
  const std::array<std::optional<llvm::StringRef>, 3> redirects = {
      llvm::StringRef(), outputPath, errorPath};
  std::string message;
  bool executionFailed = false;
  const MonotonicClock::time_point begin = MonotonicClock::now();
  const llvm::sys::ProcessInfo process =
      llvm::sys::ExecuteNoWait(executable, refs, environmentRefs, redirects, 0,
                               &message, &executionFailed, nullptr, true);
  if (executionFailed || process.Pid == llvm::sys::ProcessInfo::InvalidPid) {
    const std::uint64_t wallTimeNanoseconds = elapsedNanoseconds(begin);
    auto output = readFile(outputPath, "host stdout");
    if (!output)
      return output.takeError();
    auto diagnostic = readFile(errorPath, "host stderr");
    if (!diagnostic)
      return diagnostic.takeError();
    return BoundedCommandResult{
        std::nullopt,
        wallTimeNanoseconds,
        false,
        true,
        std::move(*output),
        appendExecutionMessage(std::move(*diagnostic), message)};
  }

  const MonotonicClock::time_point deadline =
      executionDeadline(begin, deadlineMilliseconds);
  bool timedOut = false;
  std::optional<int> exitStatus;
  while (true) {
    const MonotonicClock::time_point now = MonotonicClock::now();
    if (now >= deadline) {
      timedOut = true;
      terminateHostProcess(process, message);
      break;
    }

    message.clear();
    llvm::sys::ProcessInfo waited =
        llvm::sys::Wait(process, 0, &message, nullptr, true);
    if (waited.Pid == process.Pid) {
      if (MonotonicClock::now() >= deadline) {
        timedOut = true;
        terminateHostProcessGroup(process.Pid, false, message);
        waitForHostProcessGroupExit(process.Pid, message);
      } else {
        if (waited.ReturnCode < 0)
          executionFailed = true;
        else
          exitStatus = waited.ReturnCode;
        const HostProcessGroupState groupState =
            inspectHostProcessGroup(process.Pid, message);
        if (groupState != HostProcessGroupState::Empty) {
          executionFailed = true;
          if (groupState == HostProcessGroupState::Present)
            message = appendExecutionMessage(
                std::move(message),
                "host process left descendants after leader exit");
          terminateHostProcessGroup(process.Pid, false, message);
          waitForHostProcessGroupExit(process.Pid, message);
        }
      }
      break;
    }
    if (waited.Pid != llvm::sys::ProcessInfo::InvalidPid || !message.empty()) {
      executionFailed = true;
      terminateHostProcess(process, message);
      break;
    }

    const auto remaining = deadline - MonotonicClock::now();
    std::this_thread::sleep_for(std::min(
        remaining, std::chrono::duration_cast<MonotonicClock::duration>(
                       hostProcessPollInterval)));
  }

  const std::uint64_t wallTimeNanoseconds = elapsedNanoseconds(begin);
  auto output = readFile(outputPath, "host stdout");
  if (!output)
    return output.takeError();
  auto diagnostic = readFile(errorPath, "host stderr");
  if (!diagnostic)
    return diagnostic.takeError();
  return BoundedCommandResult{
      exitStatus,
      wallTimeNanoseconds,
      timedOut,
      executionFailed,
      std::move(*output),
      appendExecutionMessage(std::move(*diagnostic), message)};
}

std::string compilerName(LanguageMode language) {
  return language == LanguageMode::C ? "clang" : "clang++";
}

std::string mismatchDiagnostic(llvm::StringRef expected,
                               llvm::StringRef actual) {
  const std::size_t common = std::min(expected.size(), actual.size());
  const auto mismatch = std::mismatch(
      expected.begin(), expected.begin() + common, actual.begin());
  const std::size_t offset =
      static_cast<std::size_t>(mismatch.first - expected.begin());
  if (offset == common)
    return "stdout byte count " + std::to_string(actual.size()) +
           " differs from exact oracle byte count " +
           std::to_string(expected.size());
  const unsigned expectedByte =
      static_cast<unsigned>(static_cast<unsigned char>(expected[offset]));
  const unsigned actualByte =
      static_cast<unsigned>(static_cast<unsigned char>(actual[offset]));
  return "stdout differs from exact oracle at byte " + std::to_string(offset) +
         ": expected 0x" + llvm::utohexstr(expectedByte, true) +
         ", observed 0x" + llvm::utohexstr(actualByte, true);
}

llvm::StringRef compileStatus(const ApplicationHostRunReport &report) {
  switch (report.outcome) {
  case ApplicationHostRunOutcome::SourceUnavailable:
  case ApplicationHostRunOutcome::UnsupportedOracle:
  case ApplicationHostRunOutcome::UnsupportedProfile:
    return "not_run";
  case ApplicationHostRunOutcome::CompileFailure:
    return "failed";
  case ApplicationHostRunOutcome::Succeeded:
  case ApplicationHostRunOutcome::ExecutionFailure:
  case ApplicationHostRunOutcome::Timeout:
  case ApplicationHostRunOutcome::OracleMismatch:
    return "succeeded";
  }
  llvm_unreachable("unknown ApplicationHostRunOutcome");
}

llvm::StringRef executionStatus(const ApplicationHostRunReport &report) {
  switch (report.outcome) {
  case ApplicationHostRunOutcome::SourceUnavailable:
  case ApplicationHostRunOutcome::UnsupportedOracle:
  case ApplicationHostRunOutcome::UnsupportedProfile:
  case ApplicationHostRunOutcome::CompileFailure:
    return "not_run";
  case ApplicationHostRunOutcome::ExecutionFailure:
    return "failed";
  case ApplicationHostRunOutcome::Timeout:
    return "timed_out";
  case ApplicationHostRunOutcome::Succeeded:
  case ApplicationHostRunOutcome::OracleMismatch:
    return "succeeded";
  }
  llvm_unreachable("unknown ApplicationHostRunOutcome");
}

template <typename Value>
llvm::json::Value optionalJson(const std::optional<Value> &value) {
  return value ? llvm::json::Value(*value) : llvm::json::Value(nullptr);
}

llvm::json::Object
projectApplicationHostRunReportJson(const ApplicationHostRunReport &report) {
  llvm::json::Object sourceAdmission{
      {"status", report.unavailableSource ? "unavailable" : "admitted"},
      {"reason",
       report.unavailableSource
           ? llvm::json::Value(toString(report.unavailableSource->reason))
           : llvm::json::Value(nullptr)},
      {"path", report.unavailableSource
                   ? llvm::json::Value(report.unavailableSource->path)
                   : llvm::json::Value(nullptr)}};
  llvm::json::Object profile{
      {"warmup_samples", report.selection.input.profile.warmupSamples},
      {"measured_samples", report.selection.input.profile.measuredSamples},
      {"oracle_coverage",
       toString(report.selection.input.profile.oracleCoverage)},
      {"deadline_milliseconds",
       report.selection.input.profile.deadlineMilliseconds}};
  if (report.selection.input.profile.maximumSimulatedTicks)
    profile["maximum_simulated_ticks"] =
        *report.selection.input.profile.maximumSimulatedTicks;
  return llvm::json::Object{
      {"schema", ApplicationHostRunReport::schemaIdentity},
      {"version", ApplicationHostRunReport::schemaVersion},
      {"selection", projectSelectedApplicationInputJson(report.selection)},
      {"profile", std::move(profile)},
      {"source_admission", std::move(sourceAdmission)},
      {"compile",
       llvm::json::Object{
           {"status", compileStatus(report)},
           {"compiler", optionalJson(report.compilerExecutable)},
           {"exit_status", optionalJson(report.compileExitStatus)}}},
      {"execution",
       llvm::json::Object{
           {"status", executionStatus(report)},
           {"exit_status", optionalJson(report.executionExitStatus)},
           {"host_wall_time_nanoseconds",
            optionalJson(report.hostWallTimeNanoseconds)}}},
      {"oracle_result",
       llvm::json::Object{{"status", toString(report.oracleStatus)}}},
      {"outcome", toString(report.outcome)}};
}

} // namespace

llvm::StringRef toString(ApplicationHostRunOutcome outcome) {
  switch (outcome) {
  case ApplicationHostRunOutcome::Succeeded:
    return "succeeded";
  case ApplicationHostRunOutcome::SourceUnavailable:
    return "source_unavailable";
  case ApplicationHostRunOutcome::CompileFailure:
    return "compile_failure";
  case ApplicationHostRunOutcome::ExecutionFailure:
    return "execution_failure";
  case ApplicationHostRunOutcome::Timeout:
    return "timeout";
  case ApplicationHostRunOutcome::OracleMismatch:
    return "oracle_mismatch";
  case ApplicationHostRunOutcome::UnsupportedOracle:
    return "unsupported_oracle";
  case ApplicationHostRunOutcome::UnsupportedProfile:
    return "unsupported_profile";
  }
  llvm_unreachable("unknown ApplicationHostRunOutcome");
}

llvm::StringRef toString(ApplicationHostOracleStatus status) {
  switch (status) {
  case ApplicationHostOracleStatus::NotChecked:
    return "not_checked";
  case ApplicationHostOracleStatus::Matched:
    return "matched";
  case ApplicationHostOracleStatus::Mismatched:
    return "mismatched";
  case ApplicationHostOracleStatus::Unsupported:
    return "unsupported";
  }
  llvm_unreachable("unknown ApplicationHostOracleStatus");
}

llvm::Expected<ApplicationHostRunReport>
runApplicationInputOnHost(const ApplicationManifest &manifest,
                          const ApplicationHostRunRequest &request) {
  auto selection = selectApplicationInput(manifest, request.applicationIdentity,
                                          request.inputName);
  if (!selection)
    return selection.takeError();

  ApplicationHostRunReport report{std::move(*selection),
                                  ApplicationHostRunOutcome::Succeeded,
                                  ApplicationHostOracleStatus::NotChecked,
                                  std::nullopt,
                                  std::nullopt,
                                  std::nullopt,
                                  std::nullopt,
                                  std::nullopt,
                                  {}};

  std::optional<llvm::StringRef> cacheRoot;
  if (request.cacheRoot)
    cacheRoot = *request.cacheRoot;
  auto admission = admitApplicationSource(manifest, request.applicationIdentity,
                                          request.inputName,
                                          request.repositoryRoot, cacheRoot);
  if (!admission)
    return admission.takeError();
  if (auto *unavailable =
          std::get_if<UnavailableApplicationSource>(&*admission)) {
    report.outcome = ApplicationHostRunOutcome::SourceUnavailable;
    report.unavailableSource = std::move(*unavailable);
    return report;
  }
  const auto &admitted = std::get<AdmittedApplicationSource>(*admission);
  if (!std::filesystem::path(admitted.repositoryRoot).is_absolute() ||
      admitted.sourcePaths.size() != report.selection.build.sources.size() ||
      admitted.inputs.size() != 1 ||
      admitted.inputs.front().inputName != report.selection.input.name)
    return hostRunError(
        "selected source admission changed path or input cardinality");
  const AdmittedApplicationInput &admittedInput = admitted.inputs.front();
  if (admittedInput.cachedInputs.size() != report.selection.cachedInputs.size())
    return hostRunError("selected source admission changed cache cardinality");
  for (auto [declared, admittedCache] : llvm::zip_equal(
           report.selection.cachedInputs, admittedInput.cachedInputs))
    if (declared.logicalName != admittedCache.logicalName)
      return hostRunError("selected source admission changed cache order");

  if (report.selection.input.oracle.kind != OracleKind::Exact) {
    report.outcome = ApplicationHostRunOutcome::UnsupportedOracle;
    report.oracleStatus = ApplicationHostOracleStatus::Unsupported;
    return report;
  }
  if (report.selection.cachedInputs.empty() &&
      (report.selection.input.profile.warmupSamples != 0 ||
       report.selection.input.profile.measuredSamples != 1)) {
    report.outcome = ApplicationHostRunOutcome::UnsupportedProfile;
    report.diagnostic =
        "a host selection without cached inputs supports exactly zero "
        "warm-up samples and one measured sample";
    return report;
  }

  const std::filesystem::path repositoryRoot(admitted.repositoryRoot);

  const std::string selectedCompiler = request.compilerExecutable.value_or(
      compilerName(report.selection.build.language));
  report.compilerExecutable = selectedCompiler;
  llvm::ErrorOr<std::string> compiler =
      llvm::sys::findProgramByName(selectedCompiler);
  if (!compiler) {
    report.outcome = ApplicationHostRunOutcome::CompileFailure;
    report.diagnostic = "cannot resolve compiler '" + selectedCompiler +
                        "': " + compiler.getError().message();
    return report;
  }
  report.compilerExecutable = *compiler;

  auto temporary = TemporaryDirectory::create(repositoryRoot);
  if (!temporary)
    return temporary.takeError();
  const std::string executablePath = temporary->child("application");
  const std::string compilerOutput = temporary->child("compiler.stdout");
  const std::string compilerError = temporary->child("compiler.stderr");

  std::vector<std::string> compileArguments;
  compileArguments.reserve(7 + report.selection.build.compilerOptions.size() +
                           report.selection.build.sources.size() +
                           report.selection.build.linkOptions.size());
  compileArguments.push_back(*compiler);
  compileArguments.push_back("-working-directory=" + repositoryRoot.string());
  compileArguments.insert(compileArguments.end(),
                          report.selection.build.compilerOptions.begin(),
                          report.selection.build.compilerOptions.end());
  compileArguments.push_back("-DLOOM_APPLICATION_HOST_EXECUTION=1");
  compileArguments.push_back("-x");
  compileArguments.push_back(toString(report.selection.build.language).str());
  compileArguments.insert(compileArguments.end(), admitted.sourcePaths.begin(),
                          admitted.sourcePaths.end());
  compileArguments.insert(compileArguments.end(),
                          report.selection.build.linkOptions.begin(),
                          report.selection.build.linkOptions.end());
  compileArguments.push_back("-o");
  compileArguments.push_back(executablePath);

  auto compilation = runBlockingCommand(*compiler, compileArguments,
                                        compilerOutput, compilerError);
  if (!compilation)
    return compilation.takeError();
  report.compileExitStatus = compilation->exitStatus;
  report.diagnostic = std::move(compilation->diagnostic);
  if (compilation->executionFailed || !compilation->exitStatus ||
      *compilation->exitStatus != 0) {
    report.outcome = ApplicationHostRunOutcome::CompileFailure;
    return report;
  }

  std::vector<std::string> executionArguments;
  executionArguments.reserve(3 + report.selection.cachedInputs.size());
  executionArguments.push_back(executablePath);
  if (!report.selection.cachedInputs.empty()) {
    for (const AdmittedCachedInput &input : admittedInput.cachedInputs)
      executionArguments.push_back(input.path);
    executionArguments.push_back(
        std::to_string(report.selection.input.profile.warmupSamples));
    executionArguments.push_back(
        std::to_string(report.selection.input.profile.measuredSamples));
  }

  const std::string hostOutput = temporary->child("host.stdout");
  const std::string hostError = temporary->child("host.stderr");
  auto execution = runBoundedCommand(
      executablePath, executionArguments, hostOutput, hostError,
      report.selection.input.profile.deadlineMilliseconds);
  if (!execution)
    return execution.takeError();
  report.executionExitStatus = execution->exitStatus;
  report.hostWallTimeNanoseconds = execution->wallTimeNanoseconds;
  report.diagnostic = appendExecutionMessage(std::move(report.diagnostic),
                                             execution->diagnostic);
  if (execution->timedOut) {
    report.outcome = ApplicationHostRunOutcome::Timeout;
    report.diagnostic =
        appendExecutionMessage(std::move(report.diagnostic),
                               "execution exceeded the selected deadline");
    return report;
  }
  if (execution->executionFailed || !execution->exitStatus ||
      *execution->exitStatus != 0) {
    report.outcome = ApplicationHostRunOutcome::ExecutionFailure;
    return report;
  }

  auto expected = readFile(admittedInput.oraclePath, "exact host oracle");
  if (!expected)
    return expected.takeError();
  if (execution->output != *expected) {
    report.outcome = ApplicationHostRunOutcome::OracleMismatch;
    report.oracleStatus = ApplicationHostOracleStatus::Mismatched;
    report.diagnostic = appendExecutionMessage(
        std::move(report.diagnostic),
        mismatchDiagnostic(*expected, execution->output));
    return report;
  }

  report.outcome = ApplicationHostRunOutcome::Succeeded;
  report.oracleStatus = ApplicationHostOracleStatus::Matched;
  return report;
}

llvm::Expected<ApplicationHostSelectionRunReport> runApplicationSelectionOnHost(
    const ApplicationManifest &manifest,
    const ApplicationHostSelectionRunRequest &request) {
  ApplicationHostSelectionRunReport result{request.selection, {}};
  const std::vector<SelectedApplicationInput> selections =
      selectApplicationInputs(manifest, request.selection);
  result.reports.reserve(selections.size());
  for (const SelectedApplicationInput &selection : selections) {
    auto report = runApplicationInputOnHost(
        manifest, ApplicationHostRunRequest{
                      selection.applicationIdentity, selection.input.name,
                      request.repositoryRoot, request.cacheRoot,
                      request.compilerExecutable});
    if (!report)
      return report.takeError();
    result.reports.push_back(std::move(*report));
  }
  return result;
}

void writeApplicationHostRunReportJson(llvm::raw_ostream &output,
                                       const ApplicationHostRunReport &report) {
  llvm::json::OStream json(output, 2);
  json.value(projectApplicationHostRunReportJson(report));
  output << '\n';
}

void writeApplicationHostSelectionRunReportJson(
    llvm::raw_ostream &output,
    const ApplicationHostSelectionRunReport &report) {
  llvm::json::Array reports;
  for (const ApplicationHostRunReport &member : report.reports)
    reports.push_back(projectApplicationHostRunReportJson(member));
  llvm::json::OStream json(output, 2);
  json.value(llvm::json::Object{
      {"schema", ApplicationHostSelectionRunReport::schemaIdentity},
      {"version", ApplicationHostSelectionRunReport::schemaVersion},
      {"execution_selection", toString(report.selection)},
      {"reports", std::move(reports)}});
  output << '\n';
}

bool applicationHostRunSucceeded(const ApplicationHostRunReport &report) {
  return report.outcome == ApplicationHostRunOutcome::Succeeded;
}

bool applicationHostSelectionRunSucceeded(
    const ApplicationHostSelectionRunReport &report) {
  return !report.reports.empty() &&
         llvm::all_of(report.reports, applicationHostRunSucceeded);
}

} // namespace loom::application
