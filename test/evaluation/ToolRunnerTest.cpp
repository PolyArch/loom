#include "Evaluation/ToolRunner.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstddef>
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
#include <linux/filter.h>
#include <linux/seccomp.h>
#include <poll.h>
#include <pthread.h>
#include <sys/mman.h>
#include <sys/prctl.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

struct NegativeSignalRecord {
  pid_t sender;
  pid_t target;
  int signalNumber;
  int reservationPresent;
};

struct NegativeSignalLog {
  unsigned int count;
  pid_t reservationProcess;
  NegativeSignalRecord records[64];
};

static NegativeSignalLog *activeNegativeSignalLog = nullptr;

extern "C" int kill(pid_t process, int signalNumber) noexcept {
  NegativeSignalLog *log = activeNegativeSignalLog;
  const pid_t sender = ::getpid();
  if (log && process < -1) {
    const unsigned int index =
        __atomic_fetch_add(&log->count, 1U, __ATOMIC_RELAXED);
    if (index < std::size(log->records)) {
      const pid_t reservation =
          __atomic_load_n(&log->reservationProcess, __ATOMIC_RELAXED);
      const int reservationPresent =
          reservation > 0 && ::syscall(SYS_kill, reservation, 0) == 0;
      log->records[index] = NegativeSignalRecord{sender, process, signalNumber,
                                                 reservationPresent};
    }
  }
  return static_cast<int>(::syscall(SYS_kill, process, signalNumber));
}

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

class NegativeSignalRecorder {
public:
  NegativeSignalRecorder() {
    require(__func__, !activeNegativeSignalLog,
            "negative signal recorder is already active");
    void *mapping =
        ::mmap(nullptr, sizeof(NegativeSignalLog), PROT_READ | PROT_WRITE,
               MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    require(__func__, mapping != MAP_FAILED,
            std::string("mmap failed: ") + std::strerror(errno));
    log_ = static_cast<NegativeSignalLog *>(mapping);
    std::memset(log_, 0, sizeof(*log_));
    activeNegativeSignalLog = log_;
  }

  NegativeSignalRecorder(const NegativeSignalRecorder &) = delete;
  NegativeSignalRecorder &operator=(const NegativeSignalRecorder &) = delete;

  ~NegativeSignalRecorder() {
    activeNegativeSignalLog = nullptr;
    ::munmap(log_, sizeof(*log_));
  }

  std::vector<NegativeSignalRecord> records() const {
    const unsigned int recorded =
        __atomic_load_n(&log_->count, __ATOMIC_RELAXED);
    const unsigned int count =
        std::min(recorded, static_cast<unsigned int>(std::size(log_->records)));
    return std::vector<NegativeSignalRecord>(log_->records,
                                             log_->records + count);
  }

  void setReservation(pid_t process) {
    __atomic_store_n(&log_->reservationProcess, process, __ATOMIC_RELAXED);
  }

private:
  NegativeSignalLog *log_ = nullptr;
};

template <typename T>
T takeExpected(const char *test, llvm::Expected<T> value) {
  if (!value)
    fail(test, llvm::toString(value.takeError()));
  return std::move(*value);
}

ArtifactIdentity artifact(std::uint8_t value) {
  ArtifactIdentity::Storage bytes{};
  bytes.front() = value;
  return takeExpected(__func__, ArtifactIdentity::fromBytes(bytes));
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

std::vector<pid_t> readProcessChildren(pid_t process) {
  const std::filesystem::path path =
      "/proc" / std::filesystem::path(std::to_string(process)) / "task" /
      std::filesystem::path(std::to_string(process)) / "children";
  std::ifstream stream(path);
  std::vector<pid_t> children;
  pid_t child = 0;
  while (stream >> child)
    children.push_back(child);
  return children;
}

bool waitForPath(const std::filesystem::path &path) {
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(2);
  while (!std::filesystem::exists(path)) {
    if (std::chrono::steady_clock::now() >= deadline)
      return false;
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  return true;
}

bool waitForChild(pid_t child, int &status) {
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(3);
  for (;;) {
    const pid_t waited = ::waitpid(child, &status, WNOHANG);
    if (waited == child)
      return true;
    if (waited < 0 && errno != EINTR)
      return false;
    if (std::chrono::steady_clock::now() >= deadline)
      return false;
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
}

void observeChildExitWithoutReaping(const char *test, pid_t child) {
  siginfo_t information{};
  int waitResult = -1;
  do {
    waitResult = ::waitid(P_PID, static_cast<id_t>(child), &information,
                          WEXITED | WNOWAIT);
  } while (waitResult < 0 && errno == EINTR);
  require(test, waitResult == 0 && information.si_pid == child,
          "child exit was not observable before the late control send");
}

void terminateProcesses(const std::vector<pid_t> &pids) {
  for (pid_t pid : pids)
    ::kill(pid, SIGKILL);
}

std::vector<std::pair<int, std::string>> processSocketDescriptors() {
  std::vector<std::pair<int, std::string>> sockets;
  for (const std::filesystem::directory_entry &entry :
       std::filesystem::directory_iterator("/proc/self/fd")) {
    const std::string name = entry.path().filename().string();
    char *end = nullptr;
    const long descriptor = std::strtol(name.c_str(), &end, 10);
    if (!end || *end != '\0' || descriptor < 0)
      continue;
    std::error_code error;
    const std::filesystem::path target =
        std::filesystem::read_symlink(entry.path(), error);
    if (!error && llvm::StringRef(target.string()).starts_with("socket:["))
      sockets.emplace_back(static_cast<int>(descriptor), target.string());
  }
  return sockets;
}

pid_t waitForSupervisor(pid_t caller) {
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(2);
  while (std::chrono::steady_clock::now() < deadline) {
    for (pid_t child : readProcessChildren(caller)) {
      if (!readProcessChildren(child).empty())
        return child;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  return -1;
}

pid_t waitForReservation(pid_t toolProcess, pid_t workloadChild) {
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(2);
  while (std::chrono::steady_clock::now() < deadline) {
    for (pid_t child : readProcessChildren(toolProcess)) {
      if (child != workloadChild)
        return child;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  return -1;
}

bool recordReservationWhenReady(const std::filesystem::path &pidFile,
                                NegativeSignalRecorder &recorder) {
  if (!std::filesystem::exists(pidFile))
    return false;
  const std::vector<pid_t> pids = readPids(pidFile);
  if (pids.size() != 2)
    return false;
  for (pid_t child : readProcessChildren(pids.front())) {
    if (child != pids.back()) {
      recorder.setReservation(child);
      return true;
    }
  }
  return false;
}

void makeCloseRangeUnavailable() {
#ifdef SYS_close_range
  sock_filter filter[] = {
      BPF_STMT(BPF_LD | BPF_W | BPF_ABS, offsetof(struct seccomp_data, nr)),
      BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, SYS_close_range, 0, 1),
      BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ERRNO | ENOSYS),
      BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ALLOW),
  };
  sock_fprog program{static_cast<unsigned short>(std::size(filter)), filter};
  if (::prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) < 0 ||
      ::prctl(PR_SET_SECCOMP, SECCOMP_MODE_FILTER, &program) < 0)
    ::_exit(120);
#endif
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
      MaterializedInputArtifact{artifact(0x01), inputPath.string()}};
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
  TemporaryDirectory timeoutScratch;
  ToolInvocation timeout =
      baseInvocation(timeoutScratch.path(), "--continuous-output");
  timeout.timeout = std::chrono::milliseconds(150);

  const auto started = std::chrono::steady_clock::now();
  ToolRunOutcome outcome = takeExpected(__func__, runTool(timeout));
  const auto elapsed = std::chrono::steady_clock::now() - started;

  if (outcome.status != ToolRunStatus::TimedOut) {
    std::ostringstream message;
    message << "continuous stdout starved timeout delivery: status="
            << static_cast<int>(outcome.status);
    if (outcome.terminationSignal)
      message << " signal=" << *outcome.terminationSignal;
    if (outcome.infrastructureDiagnostic)
      message << " infrastructure=" << *outcome.infrastructureDiagnostic;
    fail(__func__, message.str());
  }
  require(__func__, elapsed < std::chrono::seconds(2),
          "continuous stdout prevented bounded return");

  TemporaryDirectory captureScratch;
  ToolInvocation capture =
      baseInvocation(captureScratch.path(), "--continuous-output");
  const std::filesystem::path ready = captureScratch.path() / "ready.txt";
  capture.cancellationRequested = [ready] {
    return std::filesystem::exists(ready);
  };
  capture.timeout = std::chrono::seconds(2);
  outcome = takeExpected(__func__, runTool(capture));

  require(__func__, outcome.status == ToolRunStatus::Cancelled,
          "continuous stdout starved ready-driven cancellation");
  require(__func__,
          llvm::StringRef(outcome.standardOutput).contains("stdout-marker"),
          "continuous output lost the stdout marker");
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

void concurrentSupervisorsDoNotRetainPeerControlSockets() {
  TemporaryDirectory firstScratch;
  TemporaryDirectory secondScratch;
  const std::filesystem::path firstPids = firstScratch.path() / "pids.txt";
  const std::filesystem::path secondPids = secondScratch.path() / "pids.txt";
  const std::vector<std::pair<int, std::string>> baseline =
      processSocketDescriptors();
  std::atomic<bool> firstReturned{false};
  std::atomic<bool> cancelSecond{false};
  std::optional<ToolRunOutcome> firstOutcome;

  std::thread first([&] {
    ToolInvocation invocation =
        baseInvocation(firstScratch.path(), "--spawn-descendant");
    invocation.argv.push_back("pids.txt");
    invocation.declaredOutputs = {"pids.txt"};
    llvm::Expected<ToolRunOutcome> result = runTool(invocation);
    if (result)
      firstOutcome = std::move(*result);
    else
      llvm::consumeError(result.takeError());
    firstReturned.store(true, std::memory_order_release);
  });
  require(__func__, waitForPath(firstPids),
          "first concurrent invocation did not start");

  int firstControl = -1;
  for (const auto &socket : processSocketDescriptors()) {
    const int descriptor = socket.first;
    const std::string &target = socket.second;
    const auto existing =
        std::find_if(baseline.begin(), baseline.end(), [&](const auto &entry) {
          return entry.first == descriptor && entry.second == target;
        });
    if (existing == baseline.end()) {
      require(__func__, firstControl < 0,
              "first invocation exposed multiple parent control sockets");
      firstControl = descriptor;
    }
  }
  require(__func__, firstControl >= 0,
          "could not identify first invocation control socket");

  std::thread second([&] {
    ToolInvocation invocation =
        baseInvocation(secondScratch.path(), "--spawn-descendant");
    invocation.argv.push_back("pids.txt");
    invocation.declaredOutputs = {"pids.txt"};
    invocation.cancellationRequested = [&] {
      return cancelSecond.load(std::memory_order_relaxed);
    };
    llvm::Expected<ToolRunOutcome> result = runTool(invocation);
    if (!result)
      llvm::consumeError(result.takeError());
  });
  require(__func__, waitForPath(secondPids),
          "second concurrent invocation did not start");
  require(__func__, ::close(firstControl) == 0,
          "could not close first invocation control socket");

  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::milliseconds(1500);
  while (!firstReturned.load(std::memory_order_acquire) &&
         std::chrono::steady_clock::now() < deadline)
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  const bool returnedBeforePeerExit =
      firstReturned.load(std::memory_order_acquire);

  cancelSecond.store(true, std::memory_order_relaxed);
  second.join();
  first.join();
  require(__func__, returnedBeforePeerExit,
          "a concurrent supervisor retained its peer control socket");
  require(__func__,
          firstOutcome &&
              firstOutcome->status == ToolRunStatus::InfrastructureFailure,
          "parent control EOF was not returned as infrastructure failure");
  requireProcessesGone(__func__, readPids(firstPids));
  requireProcessesGone(__func__, readPids(secondPids));
}

void descriptorFallbackClosesHighFdsAboveLoweredLimits() {
  TemporaryDirectory scratch;
  const pid_t caller = ::fork();
  require(__func__, caller >= 0, "could not fork descriptor fallback probe");
  if (caller == 0) {
    const int source = ::open("/dev/null", O_RDONLY);
    if (source < 0)
      ::_exit(121);
    const int inherited = ::fcntl(source, F_DUPFD, 256);
    ::close(source);
    if (inherited < 256)
      ::_exit(122);

    rlimit limits{64, 64};
    if (::setrlimit(RLIMIT_NOFILE, &limits) < 0)
      ::_exit(123);
    makeCloseRangeUnavailable();

    ToolInvocation probe = baseInvocation(scratch.path(), "--probe-fd");
    probe.argv.push_back(std::to_string(inherited));
    llvm::Expected<ToolRunOutcome> result = runTool(probe);
    if (!result) {
      llvm::consumeError(result.takeError());
      ::_exit(124);
    }
    ::_exit(result->standardOutput == "closed\n" ? 0 : 125);
  }

  int status = 0;
  require(__func__, waitForChild(caller, status),
          "descriptor fallback probe did not return");
  require(__func__, WIFEXITED(status) && WEXITSTATUS(status) == 0,
          "fallback left a high descriptor open after limits were lowered");
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

void invokedToolCannotEscapeItsProcessGroup() {
  TemporaryDirectory scratch;
  ToolInvocation invocation =
      baseInvocation(scratch.path(), "--attempt-self-escape");
  invocation.declaredOutputs = {"escape.txt"};
  const std::filesystem::path reportPath = scratch.path() / "escape.txt";
  invocation.cancellationRequested = [reportPath] {
    return std::filesystem::exists(reportPath);
  };

  ToolRunOutcome outcome = takeExpected(__func__, runTool(invocation));
  std::istringstream report(readFile(reportPath));
  std::string processLabel;
  std::string groupLabel;
  std::string deathLabel;
  std::string sessionLabel;
  std::string setGroupLabel;
  pid_t process = -1;
  pid_t processGroup = -1;
  int deathSignal = -1;
  int sessionResult = 0;
  int sessionError = 0;
  int groupResult = 0;
  int groupError = 0;
  report >> processLabel >> process >> groupLabel >> processGroup >>
      deathLabel >> deathSignal >> sessionLabel >> sessionResult >>
      sessionError >> setGroupLabel >> groupResult >> groupError;

  require(__func__,
          !report.fail() && processLabel == "pid" && groupLabel == "pgrp" &&
              deathLabel == "pdeath" && sessionLabel == "setsid" &&
              setGroupLabel == "setpgid" && process > 0,
          "self-escape helper returned an invalid report");

  errno = 0;
  const bool processGone =
      ::syscall(SYS_kill, process, 0) < 0 && errno == ESRCH;
  if (!processGone)
    terminateAndRequireGone(__func__, process);

  require(__func__, process > 0 && processGroup == process,
          "invoked executable did not lead its process group");
  require(__func__, deathSignal == 0,
          "invoked executable did not clear PDEATHSIG");
  require(__func__, sessionResult < 0 && sessionError == EPERM,
          "invoked executable escaped with setsid");
  require(__func__, groupResult < 0 && groupError == EPERM,
          "invoked executable escaped with setpgid");
  require(__func__, processGone,
          "invoked executable survived process-group interruption cleanup");
  require(__func__, outcome.status == ToolRunStatus::Cancelled,
          "self-escape attempt changed cancellation classification");
}

void cancellationReapsProcessGroups() {
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

void groupSignalsEndBeforePgidReservationRelease() {
  TemporaryDirectory scratch;
  ToolInvocation invocation =
      baseInvocation(scratch.path(), "--spawn-descendant");
  invocation.argv.push_back("pids.txt");
  invocation.declaredOutputs = {"pids.txt"};
  invocation.timeout = std::chrono::seconds(1);
  const std::filesystem::path pidFile = scratch.path() / "pids.txt";

  NegativeSignalRecorder recorder;
  invocation.cancellationRequested = [&] {
    return recordReservationWhenReady(pidFile, recorder);
  };
  struct sigaction ignoreChild{};
  struct sigaction previousChild{};
  ignoreChild.sa_handler = SIG_IGN;
  ::sigemptyset(&ignoreChild.sa_mask);
  require(__func__, ::sigaction(SIGCHLD, &ignoreChild, &previousChild) == 0,
          "could not ignore SIGCHLD for PGID reservation probe");
  llvm::Expected<ToolRunOutcome> result = runTool(invocation);
  const int restoreChild = ::sigaction(SIGCHLD, &previousChild, nullptr);
  require(__func__, restoreChild == 0,
          "could not restore SIGCHLD for PGID reservation probe");

  ToolRunOutcome outcome = takeExpected(__func__, std::move(result));
  const std::vector<pid_t> pids = readPids(pidFile);
  requireProcessesGone(__func__, pids);
  require(__func__, outcome.status == ToolRunStatus::Cancelled,
          "PGID reservation probe did not use explicit cancellation");

  const std::vector<NegativeSignalRecord> signals = recorder.records();
  const int expectedSignals[] = {SIGSTOP, SIGTERM, SIGCONT, SIGKILL};
  require(__func__, signals.size() == std::size(expectedSignals),
          "PGID reservation probe observed an unexpected signal count");
  const pid_t expectedTarget = -pids.front();
  const pid_t expectedOwner = signals.front().sender;
  require(__func__, expectedOwner != ::getpid(),
          "caller claimed normal process-group signal ownership");
  for (std::size_t index = 0; index < signals.size(); ++index) {
    require(__func__, signals[index].sender == expectedOwner,
            "final process-group ownership changed during cleanup");
    require(__func__, signals[index].target == expectedTarget,
            "runner signaled a process group not led by the invoked tool");
    require(__func__, signals[index].reservationPresent != 0,
            "runner signaled after the PGID reservation was released");
    require(__func__, signals[index].signalNumber == expectedSignals[index],
            "runner changed the required process-group signal ordering");
  }
}

void completedLeaderWinsAgainstLateCancellation() {
  TemporaryDirectory scratch;
  ToolInvocation invocation =
      baseInvocation(scratch.path(), "--exit-on-signal");
  invocation.declaredOutputs = {"ready.txt"};
  const std::filesystem::path readyFile = scratch.path() / "ready.txt";
  bool callbackRan = false;
  pid_t leader = -1;
  pid_t reservation = -1;
  pid_t supervisor = -1;
  invocation.cancellationRequested = [&] {
    if (!std::filesystem::exists(readyFile))
      return false;
    const std::vector<pid_t> pids = readPids(readyFile);
    require(__func__, pids.size() == 1,
            "late cancellation helper returned an invalid PID record");
    leader = pids.front();
    const std::vector<pid_t> children = readProcessChildren(leader);
    require(__func__, children.size() == 1,
            "late cancellation could not identify the PGID reservation");
    reservation = children.front();
    supervisor = waitForSupervisor(::getpid());
    require(__func__, supervisor > 0,
            "late cancellation could not identify the supervisor");
    require(__func__, ::kill(leader, SIGUSR1) == 0,
            "late cancellation could not complete the leader");
    observeChildExitWithoutReaping(__func__, supervisor);
    callbackRan = true;
    return true;
  };

  ToolRunOutcome outcome = takeExpected(__func__, runTool(invocation));
  require(__func__, callbackRan,
          "late cancellation callback did not publish the race");
  require(__func__,
          outcome.status == ToolRunStatus::Exited && outcome.exitCode == 0 &&
              !outcome.infrastructureDiagnostic,
          "late cancellation overrode an already-completed leader");
  requireProcessGone(__func__, leader);
  requireProcessGone(__func__, reservation);
  requireProcessGone(__func__, supervisor);
}

void detachedCaptureHolderDoesNotExtendInterruption() {
  TemporaryDirectory scratch;
  ToolInvocation invocation =
      baseInvocation(scratch.path(), "--spawn-detached");
  invocation.argv.push_back("pids.txt");
  invocation.declaredOutputs = {"pids.txt"};
  const std::filesystem::path pidFile = scratch.path() / "pids.txt";
  std::optional<std::chrono::steady_clock::time_point> interruptedAt;
  invocation.cancellationRequested = [&] {
    if (!std::filesystem::exists(pidFile))
      return false;
    interruptedAt = std::chrono::steady_clock::now();
    return true;
  };

  ToolRunOutcome outcome = takeExpected(__func__, runTool(invocation));
  const auto returnedAt = std::chrono::steady_clock::now();
  const std::vector<pid_t> pids = readPids(pidFile);
  require(__func__, pids.size() == 2,
          "detached helper did not record both process IDs");
  require(__func__, outcome.status == ToolRunStatus::Cancelled,
          "detached capture holder changed cancellation classification");
  require(__func__, interruptedAt.has_value(),
          "detached helper was not interrupted after becoming ready");
  require(__func__, returnedAt - *interruptedAt < std::chrono::seconds(2),
          "detached capture holder kept runTool waiting for pipe EOF");
  requireProcessGone(__func__, pids.front());
  terminateAndRequireGone(__func__, pids.back());
}

void callerDeathAbortsOwnedProcessGroup() {
  TemporaryDirectory scratch;
  const std::filesystem::path pidFile = scratch.path() / "pids.txt";
  const pid_t caller = ::fork();
  require(__func__, caller >= 0, "could not fork caller-death probe");
  if (caller == 0) {
    ToolInvocation invocation =
        baseInvocation(scratch.path(), "--spawn-descendant");
    invocation.argv.push_back("pids.txt");
    invocation.declaredOutputs = {"pids.txt"};
    llvm::Expected<ToolRunOutcome> result = runTool(invocation);
    if (!result)
      llvm::consumeError(result.takeError());
    ::_exit(126);
  }

  require(__func__, waitForPath(pidFile),
          "caller-death helper did not start its process group");
  const std::vector<pid_t> ownedProcesses = readPids(pidFile);
  require(__func__, ownedProcesses.size() == 2,
          "caller-death helper did not record both process IDs");
  const pid_t reservation =
      waitForReservation(ownedProcesses.front(), ownedProcesses.back());
  require(__func__, reservation > 0,
          "caller-death probe could not identify the PGID reservation");
  const std::vector<pid_t> callerChildren = readProcessChildren(caller);
  require(__func__, !callerChildren.empty(),
          "caller-death probe could not observe runner children");
  require(__func__, ::kill(caller, SIGKILL) == 0,
          "could not terminate ToolRunner caller");
  int status = 0;
  require(__func__, waitForChild(caller, status),
          "terminated ToolRunner caller was not reaped");
  require(__func__, WIFSIGNALED(status) && WTERMSIG(status) == SIGKILL,
          "caller-death probe changed caller termination");
  requireProcessesGone(__func__, ownedProcesses);
  requireProcessGone(__func__, reservation);
  for (pid_t child : callerChildren)
    requireProcessGone(__func__, child);
}

void supervisorDeathUsesEmergencyGroupOwnership() {
  TemporaryDirectory scratch;
  const std::filesystem::path pidFile = scratch.path() / "pids.txt";
  const pid_t caller = ::fork();
  require(__func__, caller >= 0, "could not fork supervisor-death probe");
  if (caller == 0) {
    ToolInvocation invocation =
        baseInvocation(scratch.path(), "--spawn-descendant");
    invocation.argv.push_back("pids.txt");
    invocation.declaredOutputs = {"pids.txt"};
    llvm::Expected<ToolRunOutcome> result = runTool(invocation);
    if (!result) {
      llvm::consumeError(result.takeError());
      ::_exit(127);
    }
    ::_exit(result->status == ToolRunStatus::InfrastructureFailure &&
                    result->infrastructureDiagnostic
                ? 0
                : 128);
  }

  require(__func__, waitForPath(pidFile),
          "supervisor-death helper did not start its process group");
  const std::vector<pid_t> ownedProcesses = readPids(pidFile);
  require(__func__, ownedProcesses.size() == 2,
          "supervisor-death helper did not record both process IDs");
  const pid_t reservation =
      waitForReservation(ownedProcesses.front(), ownedProcesses.back());
  require(__func__, reservation > 0,
          "supervisor-death probe could not identify the PGID reservation");
  const pid_t supervisor = waitForSupervisor(caller);
  require(__func__, supervisor > 0, "could not identify ToolRunner supervisor");
  const std::vector<pid_t> callerChildren = readProcessChildren(caller);
  require(__func__, ::kill(supervisor, SIGKILL) == 0,
          "could not terminate ToolRunner supervisor");

  int status = 0;
  const bool callerReturned = waitForChild(caller, status);
  if (!callerReturned)
    ::kill(caller, SIGKILL);
  if (!callerReturned || !WIFEXITED(status) || WEXITSTATUS(status) != 0) {
    terminateProcesses(ownedProcesses);
    terminateProcesses(callerChildren);
  }
  require(__func__, callerReturned,
          "runTool did not return after supervisor death");
  require(__func__, WIFEXITED(status) && WEXITSTATUS(status) == 0,
          "supervisor death was not returned as raw infrastructure failure");
  requireProcessesGone(__func__, ownedProcesses);
  requireProcessGone(__func__, reservation);
  for (pid_t child : callerChildren)
    requireProcessGone(__func__, child);
}

void interruptTransportFailureReturnsRawOutcome() {
  TemporaryDirectory scratch;
  const std::filesystem::path readyFile = scratch.path() / "ready.txt";
  ToolInvocation invocation = baseInvocation(scratch.path(), "--emit-and-wait");
  invocation.declaredOutputs = {"ready.txt"};

  bool callbackRan = false;
  pid_t tool = -1;
  pid_t reservation = -1;
  pid_t supervisor = -1;
  NegativeSignalRecorder recorder;
  invocation.cancellationRequested = [&] {
    if (callbackRan)
      return true;
    if (!std::filesystem::exists(readyFile))
      return false;

    const std::vector<pid_t> pids = readPids(readyFile);
    require(__func__, pids.size() == 1,
            "interrupt race helper returned an invalid PID record");
    tool = pids.front();
    const std::vector<pid_t> toolChildren = readProcessChildren(tool);
    require(__func__, toolChildren.size() == 1,
            "interrupt race could not identify the PGID reservation");
    reservation = toolChildren.front();
    recorder.setReservation(reservation);
    supervisor = waitForSupervisor(::getpid());
    require(__func__, supervisor > 0,
            "interrupt race could not identify the supervisor");
    require(__func__, ::kill(supervisor, SIGKILL) == 0,
            "interrupt race could not terminate the supervisor");

    observeChildExitWithoutReaping(__func__, supervisor);
    callbackRan = true;
    return true;
  };

  ToolRunOutcome outcome = takeExpected(__func__, runTool(invocation));
  require(__func__, callbackRan,
          "interrupt race did not execute the cancellation callback");
  require(__func__,
          outcome.status == ToolRunStatus::InfrastructureFailure &&
              outcome.infrastructureDiagnostic,
          "interrupt transport failure was not returned as a raw outcome");
  require(__func__, outcome.standardOutput == "stdout-before-interrupt\n",
          "interrupt transport failure lost captured stdout");
  require(__func__, outcome.standardError == "stderr-before-interrupt\n",
          "interrupt transport failure lost captured stderr");
  require(__func__,
          outcome.producedFiles == std::vector<std::string>{"ready.txt"},
          "interrupt transport failure skipped output inventory");
  require(__func__,
          outcome.startedAt.time_since_epoch().count() != 0 &&
              outcome.startedAt <= outcome.endedAt,
          "interrupt transport failure lost run timestamps");
  requireProcessGone(__func__, tool);
  requireProcessGone(__func__, reservation);
  requireProcessGone(__func__, supervisor);
  const std::vector<NegativeSignalRecord> signals = recorder.records();
  require(__func__,
          signals.size() == 1 && signals.front().sender == ::getpid() &&
              signals.front().target == -tool &&
              signals.front().signalNumber == SIGKILL &&
              signals.front().reservationPresent != 0,
          "interrupt race did not use one reserved emergency group signal");
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

void stablePreflightBoundariesRejectBeforeSpawn() {
  TemporaryDirectory root;
  const std::filesystem::path scratch = root.path() / "scratch";
  std::filesystem::create_directory(scratch);

  struct Rejection {
    const char *name;
    std::function<void(ToolInvocation &)> invalidate;
    llvm::StringLiteral diagnostic;
  };
  const std::vector<Rejection> rejections = {
      {"executable",
       [](ToolInvocation &invocation) { invocation.executablePath = "tool"; },
       "executable path"},
      {"argv", [](ToolInvocation &invocation) { invocation.argv.clear(); },
       "argv"},
      {"scratch",
       [](ToolInvocation &invocation) {
         invocation.scratchDirectory = "relative-scratch";
       },
       "scratch directory"},
      {"environment",
       [](ToolInvocation &invocation) {
         invocation.environmentOverlay = {{"BAD=NAME", "value"}};
       },
       "environment overlay"},
      {"timeout",
       [](ToolInvocation &invocation) {
         invocation.timeout = std::chrono::milliseconds(-1);
       },
       "timeout"},
      {"resource lease",
       [](ToolInvocation &invocation) {
         invocation.resourceLeaseBindingIdentities = {""};
       },
       "resource lease"},
      {"license lease",
       [](ToolInvocation &invocation) {
         invocation.licenseLeaseBindingIdentities = {""};
       },
       "license lease"},
  };

  for (const Rejection &rejection : rejections) {
    ToolInvocation invocation = baseInvocation(scratch, "--mark-spawned");
    rejection.invalidate(invocation);
    llvm::Expected<ToolRunOutcome> result = runTool(invocation);
    if (result)
      fail(__func__, std::string(rejection.name) + " was accepted");
    const std::string message = llvm::toString(result.takeError());
    require(__func__, llvm::StringRef(message).contains(rejection.diagnostic),
            std::string(rejection.name) +
                " returned unexpected diagnostic: " + message);
    require(__func__, !std::filesystem::exists(scratch / "spawned.txt"),
            std::string(rejection.name) + " spawned before rejection");
  }
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

void secretEnvironmentValuesAreNotStructurallyRetained() {
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
  requireNoSecret(outcome.launchErrorMessage);
  if (outcome.infrastructureDiagnostic)
    requireNoSecret(*outcome.infrastructureDiagnostic);
  if (outcome.inventoryDiagnostic)
    requireNoSecret(*outcome.inventoryDiagnostic);
  for (const std::string &value : outcome.producedFiles)
    requireNoSecret(value);
  for (const std::string &value : outcome.resourceLeaseBindingIdentities)
    requireNoSecret(value);
  for (const std::string &value : outcome.licenseLeaseBindingIdentities)
    requireNoSecret(value);
}

void capturedStreamsRemainVerbatim() {
  TemporaryDirectory scratch;
  ToolInvocation invocation =
      baseInvocation(scratch.path(), "--echo-secret-streams");
  invocation.environmentOverlay = {
      {"LOOM_TOOL_RUNNER_SECRET", kSecretValue.str()}};

  ToolRunOutcome outcome = takeExpected(__func__, runTool(invocation));
  require(__func__, outcome.standardOutput == kSecretValue.str(),
          "stdout was not captured verbatim");
  require(__func__, outcome.standardError == kSecretValue.str(),
          "stderr was not captured verbatim");
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
    const char outputMarker[] = "stdout-marker\n";
    const char errorMarker[] = "stderr-marker\n";
    if (::write(STDOUT_FILENO, outputMarker, sizeof(outputMarker) - 1) < 0 ||
        ::write(STDERR_FILENO, errorMarker, sizeof(errorMarker) - 1) < 0)
      return 89;
    const pid_t writer = ::fork();
    if (writer < 0)
      return 88;
    if (writer == 0)
      writeForever(STDOUT_FILENO);
    writeFile("./ready.txt", "ready\n");
    for (;;)
      ::pause();
  }
  if (mode == "--signal-self") {
    std::raise(SIGUSR1);
    return 86;
  }
  if (mode == "--attempt-self-escape") {
    int deathSignal = -1;
    if (::prctl(PR_SET_PDEATHSIG, 0) < 0 ||
        ::prctl(PR_GET_PDEATHSIG, &deathSignal) < 0)
      return 104;

    errno = 0;
    const int sessionResult = ::setsid();
    const int sessionError = errno;
    int ready[2];
    if (::pipe(ready) < 0)
      return 105;
    const pid_t alternateGroup = ::fork();
    if (alternateGroup < 0)
      return 106;
    if (alternateGroup == 0) {
      ::close(ready[0]);
      if (::setpgid(0, 0) < 0)
        ::_exit(107);
      const char marker = 'r';
      if (::write(ready[1], &marker, 1) != 1)
        ::_exit(108);
      ::close(ready[1]);
      for (;;)
        ::pause();
    }
    ::close(ready[1]);
    char marker = 0;
    const ssize_t readyCount = ::read(ready[0], &marker, 1);
    ::close(ready[0]);
    if (readyCount != 1)
      return 109;

    errno = 0;
    const int groupResult = ::setpgid(0, alternateGroup);
    const int groupError = errno;
    ::kill(alternateGroup, SIGKILL);
    while (::waitpid(alternateGroup, nullptr, 0) < 0 && errno == EINTR) {
    }

    std::ofstream report("escape.txt");
    report << "pid " << ::getpid() << '\n';
    report << "pgrp " << ::getpgrp() << '\n';
    report << "pdeath " << deathSignal << '\n';
    report << "setsid " << sessionResult << ' ' << sessionError << '\n';
    report << "setpgid " << groupResult << ' ' << groupError << '\n';
    report.close();
    if (!report)
      return 110;
    std::signal(SIGTERM, SIG_IGN);
    for (;;)
      ::pause();
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
  if (mode == "--echo-secret-streams") {
    const char *secret = std::getenv("LOOM_TOOL_RUNNER_SECRET");
    if (!secret)
      return 102;
    const std::size_t size = std::strlen(secret);
    if (::write(STDOUT_FILENO, secret, size) != static_cast<ssize_t>(size) ||
        ::write(STDERR_FILENO, secret, size) != static_cast<ssize_t>(size))
      return 103;
    return 0;
  }
  if (mode == "--emit-and-wait") {
    const char output[] = "stdout-before-interrupt\n";
    const char error[] = "stderr-before-interrupt\n";
    if (::write(STDOUT_FILENO, output, sizeof(output) - 1) < 0 ||
        ::write(STDERR_FILENO, error, sizeof(error) - 1) < 0)
      return 111;
    std::ofstream ready("ready.txt");
    ready << ::getpid() << '\n';
    ready.close();
    if (!ready)
      return 112;
    std::signal(SIGTERM, SIG_IGN);
    for (;;)
      ::pause();
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
  if (mode == "--exit-on-signal") {
    sigset_t signalSet;
    ::sigemptyset(&signalSet);
    ::sigaddset(&signalSet, SIGUSR1);
    if (::pthread_sigmask(SIG_BLOCK, &signalSet, nullptr) != 0)
      return 92;
    std::ofstream ready("ready.txt");
    ready << ::getpid() << '\n';
    ready.close();
    if (!ready)
      return 93;
    int received = 0;
    if (::sigwait(&signalSet, &received) != 0 || received != SIGUSR1)
      return 94;
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
  concurrentSupervisorsDoNotRetainPeerControlSockets();
  descriptorFallbackClosesHighFdsAboveLoweredLimits();
  inheritedSignalStateIsNormalized();
  cancellationReapsProcessGroups();
  groupSignalsEndBeforePgidReservationRelease();
  invokedToolCannotEscapeItsProcessGroup();
  completedLeaderWinsAgainstLateCancellation();
  detachedCaptureHolderDoesNotExtendInterruption();
  callerDeathAbortsOwnedProcessGroup();
  supervisorDeathUsesEmergencyGroupOwnership();
  interruptTransportFailureReturnsRawOutcome();
  invalidOutputPathsAreRejectedBeforeSpawn();
  stablePreflightBoundariesRejectBeforeSpawn();
  producedInventoryIsSortedAndScratchRelative();
  inventoryFailureRetainsRawOutcome();
  secretEnvironmentValuesAreNotStructurallyRetained();
  capturedStreamsRemainVerbatim();
  return 0;
}
