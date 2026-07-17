#include "Evaluation/ToolRunner.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <pthread.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

using namespace loom;
using namespace loom::evaluation;

namespace {

constexpr llvm::StringLiteral kSecretValue = "loom-tool-runner-secret-6f7340b1";
constexpr std::size_t kCapturedStreamBytes = 256 * 1024;

std::filesystem::path selfExecutable;

[[noreturn]] void fail(const char *test, const std::string &message) {
  std::cerr << test << ": " << message << '\n';
  std::exit(1);
}

void require(const char *test, bool condition, const std::string &message) {
  if (!condition)
    fail(test, message);
}

template <typename T>
T takeExpected(const char *test, llvm::Expected<T> value) {
  if (!value)
    fail(test, llvm::toString(value.takeError()));
  return std::move(*value);
}

template <typename T>
void expectErrorContains(const char *test, llvm::Expected<T> value,
                         llvm::StringRef expected) {
  if (value)
    fail(test, "expected an error");
  std::string message = llvm::toString(value.takeError());
  require(test, llvm::StringRef(message).contains(expected),
          "unexpected error: " + message);
}

class TemporaryDirectory {
public:
  TemporaryDirectory() {
    std::string pattern = "/tmp/loom-tool-runner-test-XXXXXX";
    std::vector<char> buffer(pattern.begin(), pattern.end());
    buffer.push_back('\0');
    char *created = ::mkdtemp(buffer.data());
    if (!created)
      fail(__func__, std::string("mkdtemp failed: ") + std::strerror(errno));
    path_ = created;
  }

  ~TemporaryDirectory() {
    std::error_code error;
    std::filesystem::remove_all(path_, error);
  }

  const std::filesystem::path &path() const { return path_; }

private:
  std::filesystem::path path_;
};

void writeFile(const std::filesystem::path &path, llvm::StringRef contents) {
  std::filesystem::create_directories(path.parent_path());
  std::ofstream stream(path, std::ios::binary);
  if (!stream)
    fail(__func__, "could not open " + path.string());
  stream.write(contents.data(), static_cast<std::streamsize>(contents.size()));
  if (!stream)
    fail(__func__, "could not write " + path.string());
}

std::string readFile(const std::filesystem::path &path) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream)
    fail(__func__, "could not open " + path.string());
  std::ostringstream contents;
  contents << stream.rdbuf();
  return contents.str();
}

ToolInvocation baseInvocation(const std::filesystem::path &scratch,
                              llvm::StringRef childMode) {
  ToolInvocation invocation;
  invocation.toolBindingIdentity = "test-tool@1";
  invocation.executablePath = selfExecutable.string();
  invocation.argv = {selfExecutable.string(), childMode.str()};
  invocation.scratchDirectory = scratch.string();
  return invocation;
}

std::vector<pid_t> readPids(const std::filesystem::path &path) {
  std::istringstream stream(readFile(path));
  std::vector<pid_t> pids;
  pid_t pid = 0;
  while (stream >> pid)
    pids.push_back(pid);
  return pids;
}

void requireProcessGone(const char *test, pid_t pid) {
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(2);
  while (::kill(pid, 0) == 0 || errno == EPERM) {
    if (std::chrono::steady_clock::now() >= deadline)
      fail(test,
           "process was not terminated and reaped: " + std::to_string(pid));
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  require(test, errno == ESRCH,
          "process liveness check failed for " + std::to_string(pid));
}

void requireProcessesGone(const char *test, const std::vector<pid_t> &pids) {
  require(test, pids.size() == 2, "helper did not record both process IDs");
  for (pid_t pid : pids)
    requireProcessGone(test, pid);
}

void terminateAndRequireGone(const char *test, pid_t pid) {
  if (::kill(pid, 0) == 0 || errno == EPERM)
    ::kill(pid, SIGKILL);
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(4);
  while (::kill(pid, 0) == 0 || errno == EPERM) {
    if (std::chrono::steady_clock::now() >= deadline)
      fail(test, "detached process remained alive: " + std::to_string(pid));
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  require(test, errno == ESRCH,
          "detached process liveness check failed: " + std::to_string(pid));
}

void literalInvocationUsesOverlayAndScratch() {
  TemporaryDirectory scratch;
  TemporaryDirectory inputs;
  const std::filesystem::path inputPath = inputs.path() / "input.txt";
  const std::filesystem::path shellMarker = inputs.path() / "shell-ran";
  writeFile(inputPath, "materialized-input\n");

  ToolInvocation invocation = baseInvocation(scratch.path(), "--record");
  const std::string metacharacters =
      "$(touch " + shellMarker.string() + "); * | && ;";
  invocation.argv.push_back(inputPath.string());
  invocation.argv.push_back(metacharacters);
  invocation.environmentOverlay = {{"LOOM_TOOL_RUNNER_ENV", "overlay"}};
  invocation.inputs = {
      MaterializedInputArtifact{ArtifactIdentity({0x01}), inputPath.string()}};
  invocation.declaredOutputs = {"report/invocation.txt"};
  invocation.resourceLeaseBindingIdentities = {"cpu-slot/7"};
  invocation.licenseLeaseBindingIdentities = {"license/token-3"};

  ToolRunOutcome outcome = takeExpected(__func__, runTool(invocation));
  require(__func__, outcome.status == ToolRunStatus::Exited,
          "tool did not exit normally");
  require(__func__, outcome.exitCode == 0, "tool exit code changed");
  require(__func__, outcome.toolBindingIdentity == "test-tool@1",
          "tool binding identity was not retained");
  require(__func__,
          outcome.resourceLeaseBindingIdentities ==
              std::vector<std::string>{"cpu-slot/7"},
          "resource lease identity was not retained");
  require(__func__,
          outcome.licenseLeaseBindingIdentities ==
              std::vector<std::string>{"license/token-3"},
          "license lease identity was not retained");
  require(__func__,
          outcome.producedFiles ==
              std::vector<std::string>{"report/invocation.txt"},
          "declared output was not inventoried");

  const std::string record =
      readFile(scratch.path() / "report" / "invocation.txt");
  require(__func__,
          llvm::StringRef(record).contains(
              std::filesystem::canonical(scratch.path()).string()),
          "tool did not run in the allocated scratch directory");
  require(__func__, llvm::StringRef(record).contains("overlay"),
          "environment overlay was not visible to the tool");
  require(__func__, llvm::StringRef(record).contains(metacharacters),
          "literal argv was changed");
  require(__func__, llvm::StringRef(record).contains("materialized-input"),
          "materialized input was not readable");
  require(__func__, !std::filesystem::exists(shellMarker),
          "shell metacharacters were interpreted");
}

void streamsExitAndLaunchFailureAreDistinct() {
  TemporaryDirectory scratch;
  ToolInvocation nonzero = baseInvocation(scratch.path(), "--emit-nonzero");

  ToolRunOutcome exited = takeExpected(__func__, runTool(nonzero));
  require(__func__, exited.status == ToolRunStatus::Exited,
          "nonzero exit was not classified as a normal exit");
  require(__func__, exited.exitCode == 23, "nonzero exit code was lost");
  require(__func__,
          exited.standardOutput == std::string(kCapturedStreamBytes, 'o'),
          "stdout capture changed or deadlocked above pipe capacity");
  require(__func__,
          exited.standardError == std::string(kCapturedStreamBytes, 'e'),
          "stderr capture changed or deadlocked above pipe capacity");

  ToolInvocation signaled = baseInvocation(scratch.path(), "--signal-self");
  ToolRunOutcome signalOutcome = takeExpected(__func__, runTool(signaled));
  require(__func__, signalOutcome.status == ToolRunStatus::Signaled,
          "signal termination was not distinguished");
  require(__func__, signalOutcome.terminationSignal == SIGUSR1,
          "termination signal was lost");

  const std::filesystem::path badExecutable = scratch.path() / "bad-tool";
  writeFile(badExecutable, "not an executable image\n");
  require(__func__, ::chmod(badExecutable.c_str(), 0700) == 0,
          "could not mark launch-failure fixture executable");

  ToolInvocation invalidImage;
  invalidImage.toolBindingIdentity = "bad-tool@1";
  invalidImage.executablePath = badExecutable.string();
  invalidImage.argv = {badExecutable.string()};
  invalidImage.scratchDirectory = scratch.path().string();

  ToolRunOutcome launch = takeExpected(__func__, runTool(invalidImage));
  require(__func__, launch.status == ToolRunStatus::LaunchFailure,
          "exec failure was not classified as launch failure");
  require(__func__, launch.launchErrorNumber == ENOEXEC,
          "launch errno was not retained");
  require(__func__, !launch.launchErrorMessage.empty(),
          "launch failure lacks a diagnostic");
}

void continuousOutputCannotStarveControlOrOtherStreams() {
  TemporaryDirectory scratch;
  ToolInvocation invocation =
      baseInvocation(scratch.path(), "--continuous-output");
  invocation.timeout = std::chrono::milliseconds(150);

  const auto started = std::chrono::steady_clock::now();
  ToolRunOutcome outcome = takeExpected(__func__, runTool(invocation));
  const auto elapsed = std::chrono::steady_clock::now() - started;

  require(__func__, outcome.status == ToolRunStatus::TimedOut,
          "continuous stdout starved timeout delivery");
  require(__func__, elapsed < std::chrono::seconds(2),
          "continuous stdout prevented bounded return");
  require(__func__,
          llvm::StringRef(outcome.standardError).contains("stderr-marker"),
          "continuous stdout starved stderr capture");
}

void descriptorsAreIsolatedAndClosedStandardFdsAreReusable() {
  TemporaryDirectory scratch;
  const int inherited = ::open("/dev/null", O_RDONLY);
  require(__func__, inherited >= 0, "could not open descriptor probe");

  ToolInvocation probe = baseInvocation(scratch.path(), "--probe-fd");
  probe.argv.push_back(std::to_string(inherited));
  llvm::Expected<ToolRunOutcome> probeResult = runTool(probe);
  ::close(inherited);
  ToolRunOutcome probeOutcome = takeExpected(__func__, std::move(probeResult));
  require(__func__, probeOutcome.standardOutput == "closed\n",
          "tool inherited an unintended non-CLOEXEC descriptor");

  const int savedOutput = ::dup(STDOUT_FILENO);
  const int savedError = ::dup(STDERR_FILENO);
  require(__func__, savedOutput >= 0 && savedError >= 0,
          "could not preserve standard descriptors");
  require(__func__, ::close(STDOUT_FILENO) == 0,
          "could not close stdout for descriptor reuse probe");
  require(__func__, ::close(STDERR_FILENO) == 0,
          "could not close stderr for descriptor reuse probe");

  ToolInvocation closedStandardFds =
      baseInvocation(scratch.path(), "--emit-small");
  llvm::Expected<ToolRunOutcome> closedResult = runTool(closedStandardFds);
  const int restoreOutput = ::dup2(savedOutput, STDOUT_FILENO);
  const int restoreError = ::dup2(savedError, STDERR_FILENO);
  ::close(savedOutput);
  ::close(savedError);
  require(__func__,
          restoreOutput == STDOUT_FILENO && restoreError == STDERR_FILENO,
          "could not restore standard descriptors");

  ToolRunOutcome closedOutcome =
      takeExpected(__func__, std::move(closedResult));
  require(__func__, closedOutcome.standardOutput == "stdout-small\n",
          "closed stdout was not remapped to its capture pipe");
  require(__func__, closedOutcome.standardError == "stderr-small\n",
          "closed stderr was not remapped to its capture pipe");
}

void inheritedSignalStateIsNormalized() {
  TemporaryDirectory scratch;

  struct sigaction ignoreChild{};
  struct sigaction previousChild{};
  ignoreChild.sa_handler = SIG_IGN;
  ::sigemptyset(&ignoreChild.sa_mask);
  require(__func__, ::sigaction(SIGCHLD, &ignoreChild, &previousChild) == 0,
          "could not ignore SIGCHLD for probe");
  ToolInvocation childProbe = baseInvocation(scratch.path(), "--exit-zero");
  llvm::Expected<ToolRunOutcome> childResult = runTool(childProbe);
  require(__func__, ::sigaction(SIGCHLD, &previousChild, nullptr) == 0,
          "could not restore SIGCHLD disposition");
  ToolRunOutcome childOutcome = takeExpected(__func__, std::move(childResult));
  require(__func__,
          childOutcome.status == ToolRunStatus::Exited &&
              childOutcome.exitCode == 0,
          "inherited SIGCHLD ignore state broke supervisor waiting");

  struct sigaction ignoreUser{};
  struct sigaction previousUser{};
  ignoreUser.sa_handler = SIG_IGN;
  ::sigemptyset(&ignoreUser.sa_mask);
  require(__func__, ::sigaction(SIGUSR1, &ignoreUser, &previousUser) == 0,
          "could not ignore SIGUSR1 for probe");
  sigset_t blocked;
  sigset_t previousMask;
  ::sigemptyset(&blocked);
  ::sigaddset(&blocked, SIGUSR1);
  require(__func__, ::pthread_sigmask(SIG_BLOCK, &blocked, &previousMask) == 0,
          "could not block SIGUSR1 for probe");

  ToolInvocation userProbe = baseInvocation(scratch.path(), "--signal-self");
  llvm::Expected<ToolRunOutcome> userResult = runTool(userProbe);
  const int restoreMask =
      ::pthread_sigmask(SIG_SETMASK, &previousMask, nullptr);
  const int restoreAction = ::sigaction(SIGUSR1, &previousUser, nullptr);
  require(__func__, restoreMask == 0 && restoreAction == 0,
          "could not restore SIGUSR1 state");

  ToolRunOutcome userOutcome = takeExpected(__func__, std::move(userResult));
  require(__func__,
          userOutcome.status == ToolRunStatus::Signaled &&
              userOutcome.terminationSignal == SIGUSR1,
          "inherited SIGUSR1 state changed tool termination");
}

void timeoutAndCancellationReapProcessGroups() {
  TemporaryDirectory timeoutScratch;
  ToolInvocation timeout =
      baseInvocation(timeoutScratch.path(), "--spawn-descendant");
  timeout.argv.push_back("pids.txt");
  timeout.declaredOutputs = {"pids.txt"};
  timeout.timeout = std::chrono::milliseconds(200);

  ToolRunOutcome timedOut = takeExpected(__func__, runTool(timeout));
  require(__func__, timedOut.status == ToolRunStatus::TimedOut,
          "timeout was not distinguished");
  require(__func__, timedOut.terminationSignal == SIGKILL,
          "timeout did not reach bounded forceful cleanup");
  requireProcessesGone(__func__, readPids(timeoutScratch.path() / "pids.txt"));

  TemporaryDirectory cancelScratch;
  ToolInvocation cancelled =
      baseInvocation(cancelScratch.path(), "--spawn-descendant");
  cancelled.argv.push_back("pids.txt");
  cancelled.declaredOutputs = {"pids.txt"};
  const std::filesystem::path pidFile = cancelScratch.path() / "pids.txt";
  cancelled.cancellationRequested = [pidFile] {
    return std::filesystem::exists(pidFile);
  };

  ToolRunOutcome cancelledOutcome = takeExpected(__func__, runTool(cancelled));
  require(__func__, cancelledOutcome.status == ToolRunStatus::Cancelled,
          "explicit cancellation was not distinguished");
  require(__func__, cancelledOutcome.terminationSignal == SIGKILL,
          "cancellation did not reach bounded forceful cleanup");
  requireProcessesGone(__func__, readPids(pidFile));
}

void completedLeaderWinsAgainstLateCancellation() {
  TemporaryDirectory scratch;
  ToolInvocation invocation =
      baseInvocation(scratch.path(), "--complete-with-descendant");
  invocation.argv.push_back("pids.txt");
  invocation.declaredOutputs = {"pids.txt"};
  const std::filesystem::path pidFile = scratch.path() / "pids.txt";
  invocation.cancellationRequested = [pidFile] {
    if (!std::filesystem::exists(pidFile))
      return false;
    std::this_thread::sleep_for(std::chrono::milliseconds(250));
    return true;
  };

  ToolRunOutcome outcome = takeExpected(__func__, runTool(invocation));
  require(__func__,
          outcome.status == ToolRunStatus::Exited && outcome.exitCode == 0,
          "late cancellation overrode an already-completed leader");
  requireProcessesGone(__func__, readPids(pidFile));
}

void detachedCaptureHolderDoesNotExtendTimeout() {
  TemporaryDirectory scratch;
  ToolInvocation invocation =
      baseInvocation(scratch.path(), "--spawn-detached");
  invocation.argv.push_back("pids.txt");
  invocation.declaredOutputs = {"pids.txt"};
  invocation.timeout = std::chrono::milliseconds(150);

  const auto started = std::chrono::steady_clock::now();
  ToolRunOutcome outcome = takeExpected(__func__, runTool(invocation));
  const auto elapsed = std::chrono::steady_clock::now() - started;
  const std::vector<pid_t> pids = readPids(scratch.path() / "pids.txt");
  require(__func__, pids.size() == 2,
          "detached helper did not record both process IDs");
  require(__func__, outcome.status == ToolRunStatus::TimedOut,
          "detached capture holder changed timeout classification");
  require(__func__, elapsed < std::chrono::seconds(2),
          "detached capture holder kept runTool waiting for pipe EOF");
  requireProcessGone(__func__, pids.front());
  terminateAndRequireGone(__func__, pids.back());
}

void invalidOutputPathsAreRejectedBeforeSpawn() {
  TemporaryDirectory root;
  const std::filesystem::path scratch = root.path() / "scratch";
  const std::filesystem::path outside = root.path() / "outside";
  std::filesystem::create_directories(scratch);
  std::filesystem::create_directories(outside);

  ToolInvocation traversal = baseInvocation(scratch, "--mark-spawned");
  traversal.declaredOutputs = {"../escaped.txt"};
  expectErrorContains(__func__, runTool(traversal), "declared output");
  require(__func__, !std::filesystem::exists(scratch / "spawned.txt"),
          "tool spawned before traversal validation");

  ToolInvocation absolute = baseInvocation(scratch, "--mark-spawned");
  absolute.declaredOutputs = {(outside / "absolute.txt").string()};
  expectErrorContains(__func__, runTool(absolute), "declared output");
  require(__func__, !std::filesystem::exists(scratch / "spawned.txt"),
          "tool spawned before absolute-output validation");

  std::filesystem::create_directory_symlink(outside, scratch / "link");
  ToolInvocation symlink = baseInvocation(scratch, "--mark-spawned");
  symlink.declaredOutputs = {"link/escaped.txt"};
  expectErrorContains(__func__, runTool(symlink), "escapes scratch");
  require(__func__, !std::filesystem::exists(scratch / "spawned.txt"),
          "tool spawned before symlink traversal validation");
}

void producedInventoryIsSortedAndScratchRelative() {
  TemporaryDirectory scratch;
  writeFile(scratch.path() / "out" / "unchanged.txt", "unchanged\n");
  writeFile(scratch.path() / "out" / "changed.txt", "before\n");
  ToolInvocation invocation =
      baseInvocation(scratch.path(), "--create-outputs");
  invocation.declaredOutputs = {"out"};

  ToolRunOutcome outcome = takeExpected(__func__, runTool(invocation));
  const std::vector<std::string> expected = {"out/a.txt", "out/changed.txt",
                                             "out/nested/m.txt", "out/z.txt"};
  require(__func__, outcome.producedFiles == expected,
          "inventory did not isolate files created or changed by this run");
  require(__func__, !outcome.inventoryDiagnostic,
          "successful inventory returned a diagnostic");
  require(__func__,
          std::find(outcome.producedFiles.begin(), outcome.producedFiles.end(),
                    "ignored.txt") == outcome.producedFiles.end(),
          "undeclared scratch file entered the produced inventory");
  for (const std::string &path : outcome.producedFiles) {
    require(__func__, std::filesystem::path(path).is_relative(),
            "produced inventory contains an absolute path");
    require(__func__, path.find("..") == std::string::npos,
            "produced inventory escapes scratch");
  }
}

void inventoryFailureRetainsRawOutcome() {
  TemporaryDirectory root;
  const std::filesystem::path scratch = root.path() / "scratch";
  const std::filesystem::path outside = root.path() / "outside";
  std::filesystem::create_directories(scratch);
  std::filesystem::create_directories(outside);

  ToolInvocation escaping =
      baseInvocation(scratch, "--create-escaping-symlink");
  escaping.argv.push_back(outside.string());
  escaping.declaredOutputs = {"out"};
  ToolRunOutcome escapingOutcome = takeExpected(__func__, runTool(escaping));
  require(__func__,
          escapingOutcome.status == ToolRunStatus::Exited &&
              escapingOutcome.exitCode == 0,
          "inventory symlink failure discarded raw exit facts");
  require(__func__, escapingOutcome.standardOutput == "symlink-created\n",
          "inventory symlink failure discarded captured stdout");
  require(__func__,
          escapingOutcome.inventoryDiagnostic &&
              llvm::StringRef(*escapingOutcome.inventoryDiagnostic)
                  .contains("escapes scratch"),
          "escaping output symlink lacks an inventory diagnostic");

  ToolInvocation unreadable =
      baseInvocation(scratch, "--create-unreadable-output");
  unreadable.declaredOutputs = {"private"};
  ToolRunOutcome unreadableOutcome =
      takeExpected(__func__, runTool(unreadable));
  ::chmod((scratch / "private" / "locked").c_str(), 0700);
  require(__func__,
          unreadableOutcome.status == ToolRunStatus::Exited &&
              unreadableOutcome.exitCode == 0,
          "inventory read failure discarded raw exit facts");
  require(__func__, unreadableOutcome.inventoryDiagnostic.has_value(),
          "inventory read failure lacks a raw diagnostic");
}

void secretEnvironmentValuesAreNotRetained() {
  TemporaryDirectory scratch;
  ToolInvocation invocation = baseInvocation(scratch.path(), "--check-secret");
  invocation.environmentOverlay = {
      {"LOOM_TOOL_RUNNER_SECRET", kSecretValue.str()}};
  invocation.resourceLeaseBindingIdentities = {"resource/non-secret"};
  invocation.licenseLeaseBindingIdentities = {"license/non-secret"};

  ToolRunOutcome outcome = takeExpected(__func__, runTool(invocation));
  require(__func__, outcome.status == ToolRunStatus::Exited,
          "secret-check helper failed");
  require(__func__, outcome.standardOutput == "secret-present\n",
          "secret overlay was unavailable to the tool");

  auto requireNoSecret = [&](llvm::StringRef value) {
    require(__func__, !value.contains(kSecretValue),
            "secret environment value was retained in the outcome");
  };
  requireNoSecret(outcome.toolBindingIdentity);
  requireNoSecret(outcome.standardOutput);
  requireNoSecret(outcome.standardError);
  requireNoSecret(outcome.launchErrorMessage);
  if (outcome.inventoryDiagnostic)
    requireNoSecret(*outcome.inventoryDiagnostic);
  for (const std::string &value : outcome.producedFiles)
    requireNoSecret(value);
  for (const std::string &value : outcome.resourceLeaseBindingIdentities)
    requireNoSecret(value);
  for (const std::string &value : outcome.licenseLeaseBindingIdentities)
    requireNoSecret(value);
}

[[noreturn]] void writeForever(int descriptor) {
  char block[4096];
  std::memset(block, 'o', sizeof(block));
  for (;;) {
    const ssize_t written = ::write(descriptor, block, sizeof(block));
    if (written < 0 && errno == EINTR)
      continue;
    if (written <= 0)
      ::_exit(87);
  }
}

int runChild(int argc, char **argv) {
  const llvm::StringRef mode(argv[1]);
  if (mode == "--record") {
    if (argc != 4)
      return 80;
    std::filesystem::create_directories("report");
    std::ofstream record("report/invocation.txt", std::ios::binary);
    record << std::filesystem::current_path().string() << '\n';
    const char *overlay = std::getenv("LOOM_TOOL_RUNNER_ENV");
    record << (overlay ? overlay : "<missing>") << '\n';
    record << argv[3] << '\n';
    record << readFile(argv[2]);
    return record ? 0 : 81;
  }
  if (mode == "--emit-nonzero") {
    std::cout << std::string(kCapturedStreamBytes, 'o');
    std::cerr << std::string(kCapturedStreamBytes, 'e');
    return 23;
  }
  if (mode == "--emit-small") {
    const char output[] = "stdout-small\n";
    const char error[] = "stderr-small\n";
    if (::write(STDOUT_FILENO, output, sizeof(output) - 1) < 0 ||
        ::write(STDERR_FILENO, error, sizeof(error) - 1) < 0)
      return 90;
    return 0;
  }
  if (mode == "--probe-fd") {
    if (argc != 3)
      return 91;
    const int descriptor = std::stoi(argv[2]);
    errno = 0;
    const int flags = ::fcntl(descriptor, F_GETFD);
    std::cout << (flags < 0 && errno == EBADF ? "closed\n" : "open\n");
    return 0;
  }
  if (mode == "--exit-zero")
    return 0;
  if (mode == "--continuous-output") {
    for (int index = 0; index < 32; ++index) {
      const pid_t writer = ::fork();
      if (writer < 0)
        return 88;
      if (writer == 0)
        writeForever(STDOUT_FILENO);
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    const char marker[] = "stderr-marker\n";
    if (::write(STDERR_FILENO, marker, sizeof(marker) - 1) < 0)
      return 89;
    for (;;)
      ::pause();
  }
  if (mode == "--signal-self") {
    std::raise(SIGUSR1);
    return 86;
  }
  if (mode == "--mark-spawned") {
    writeFile("spawned.txt", "spawned\n");
    return 0;
  }
  if (mode == "--create-outputs") {
    writeFile("out/z.txt", "z\n");
    writeFile("out/nested/m.txt", "m\n");
    writeFile("out/a.txt", "a\n");
    writeFile("out/changed.txt", "changed-after\n");
    writeFile("ignored.txt", "ignored\n");
    return 0;
  }
  if (mode == "--create-escaping-symlink") {
    if (argc != 3)
      return 100;
    std::filesystem::create_directories("out");
    std::filesystem::create_directory_symlink(argv[2], "out/escape");
    std::cout << "symlink-created\n";
    return 0;
  }
  if (mode == "--create-unreadable-output") {
    writeFile("private/locked/file.txt", "private\n");
    if (::chmod("private/locked", 0000) < 0)
      return 101;
    return 0;
  }
  if (mode == "--check-secret") {
    const char *secret = std::getenv("LOOM_TOOL_RUNNER_SECRET");
    if (!secret || secret != kSecretValue)
      return 82;
    std::cout << "secret-present\n";
    return 0;
  }
  if (mode == "--spawn-descendant") {
    if (argc != 3)
      return 83;
    std::signal(SIGTERM, SIG_IGN);
    const pid_t child = ::fork();
    if (child < 0)
      return 84;
    if (child == 0) {
      for (;;)
        ::pause();
    }
    std::ofstream pids(argv[2]);
    pids << ::getpid() << '\n' << child << '\n';
    pids.close();
    for (;;)
      ::pause();
  }
  if (mode == "--complete-with-descendant") {
    if (argc != 3)
      return 92;
    const pid_t child = ::fork();
    if (child < 0)
      return 93;
    if (child == 0) {
      std::signal(SIGTERM, SIG_IGN);
      for (;;)
        ::pause();
    }
    std::ofstream pids(argv[2]);
    pids << ::getpid() << '\n' << child << '\n';
    pids.close();
    return 0;
  }
  if (mode == "--spawn-detached") {
    if (argc != 3)
      return 94;
    int ready[2];
    if (::pipe(ready) < 0)
      return 95;
    const pid_t child = ::fork();
    if (child < 0)
      return 96;
    if (child == 0) {
      ::close(ready[0]);
      if (::setsid() < 0)
        ::_exit(97);
      std::signal(SIGTERM, SIG_IGN);
      ::alarm(3);
      const char marker = 'r';
      if (::write(ready[1], &marker, 1) != 1)
        ::_exit(98);
      ::close(ready[1]);
      for (;;)
        ::pause();
    }
    ::close(ready[1]);
    char marker = 0;
    const ssize_t readyCount = ::read(ready[0], &marker, 1);
    ::close(ready[0]);
    if (readyCount != 1)
      return 99;
    std::ofstream pids(argv[2]);
    pids << ::getpid() << '\n' << child << '\n';
    pids.close();
    for (;;)
      ::pause();
  }
  return 85;
}

} // namespace

int main(int argc, char **argv) {
  if (argc > 1)
    return runChild(argc, argv);

  selfExecutable = std::filesystem::canonical(argv[0]);
  literalInvocationUsesOverlayAndScratch();
  streamsExitAndLaunchFailureAreDistinct();
  continuousOutputCannotStarveControlOrOtherStreams();
  descriptorsAreIsolatedAndClosedStandardFdsAreReusable();
  inheritedSignalStateIsNormalized();
  timeoutAndCancellationReapProcessGroups();
  completedLeaderWinsAgainstLateCancellation();
  detachedCaptureHolderDoesNotExtendTimeout();
  invalidOutputPathsAreRejectedBeforeSpawn();
  producedInventoryIsSortedAndScratchRelative();
  inventoryFailureRetainsRawOutcome();
  secretEnvironmentValuesAreNotRetained();
  return 0;
}
