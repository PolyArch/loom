#include "Evaluation/ToolRunner.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <climits>
#include <csignal>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <map>
#include <optional>
#include <poll.h>
#include <set>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <pthread.h>
#include <sys/prctl.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

extern char **environ;

namespace loom::evaluation {
namespace {

namespace fs = std::filesystem;

constexpr std::chrono::milliseconds kPollInterval(10);
constexpr std::chrono::milliseconds kGracefulTermination(100);
constexpr std::chrono::milliseconds kForcefulTermination(500);
constexpr std::size_t kDrainQuota = 64 * 1024;

class OwnedFileDescriptor {
public:
  OwnedFileDescriptor() = default;
  explicit OwnedFileDescriptor(int descriptor) : descriptor_(descriptor) {}
  OwnedFileDescriptor(const OwnedFileDescriptor &) = delete;
  OwnedFileDescriptor &operator=(const OwnedFileDescriptor &) = delete;

  OwnedFileDescriptor(OwnedFileDescriptor &&other) noexcept
      : descriptor_(other.release()) {}
  OwnedFileDescriptor &operator=(OwnedFileDescriptor &&other) noexcept {
    if (this != &other)
      reset(other.release());
    return *this;
  }

  ~OwnedFileDescriptor() { reset(); }

  int get() const { return descriptor_; }
  explicit operator bool() const { return descriptor_ >= 0; }

  int release() {
    const int descriptor = descriptor_;
    descriptor_ = -1;
    return descriptor;
  }

  void reset(int descriptor = -1) {
    if (descriptor_ >= 0)
      ::close(descriptor_);
    descriptor_ = descriptor;
  }

private:
  int descriptor_ = -1;
};

struct Pipe {
  OwnedFileDescriptor read;
  OwnedFileDescriptor write;
};

struct ControlChannel {
  OwnedFileDescriptor parent;
  OwnedFileDescriptor supervisor;
};

enum class LaunchStage : int {
  SupervisorSetup,
  ProcessFork,
  ProcessGroup,
  WorkingDirectory,
  StandardOutput,
  StandardError,
  DescriptorIsolation,
  Execute,
};

struct LaunchRecord {
  int stage;
  int errorNumber;
};

enum class InterruptKind : int { None, TimedOut, Cancelled, Abort };

struct WaitRecord {
  int valid;
  int waitStatus;
  int errorNumber;
  int cleanupComplete;
  int interruptKind;
};

struct PreparedInvocation {
  fs::path scratchDirectory;
  std::vector<fs::path> declaredOutputs;
  std::vector<std::string> argvStorage;
  std::vector<char *> argv;
  std::vector<std::string> environmentStorage;
  std::vector<char *> environment;
  int maximumDescriptor = 0;
};

struct FileSnapshot {
  std::uint64_t device;
  std::uint64_t inode;
  std::uint64_t size;
  std::int64_t modifiedSeconds;
  std::int64_t modifiedNanoseconds;
  std::int64_t changedSeconds;
  std::int64_t changedNanoseconds;

  friend bool operator==(const FileSnapshot &lhs, const FileSnapshot &rhs) {
    return lhs.device == rhs.device && lhs.inode == rhs.inode &&
           lhs.size == rhs.size && lhs.modifiedSeconds == rhs.modifiedSeconds &&
           lhs.modifiedNanoseconds == rhs.modifiedNanoseconds &&
           lhs.changedSeconds == rhs.changedSeconds &&
           lhs.changedNanoseconds == rhs.changedNanoseconds;
  }
};

using InventorySnapshot = std::map<std::string, FileSnapshot>;

llvm::Error invocationError(const std::string &message) {
  return llvm::createStringError(std::errc::invalid_argument, "%s",
                                 message.c_str());
}

llvm::Error systemError(const std::string &operation, int errorNumber) {
  return llvm::createStringError(
      std::error_code(errorNumber, std::generic_category()), "%s",
      operation.c_str());
}

bool containsNull(llvm::StringRef value) { return value.contains('\0'); }

bool pathIsWithin(const fs::path &root, const fs::path &candidate) {
  auto rootComponent = root.begin();
  auto candidateComponent = candidate.begin();
  for (; rootComponent != root.end(); ++rootComponent, ++candidateComponent) {
    if (candidateComponent == candidate.end() ||
        *candidateComponent != *rootComponent)
      return false;
  }
  return true;
}

llvm::Error moveAboveStandard(OwnedFileDescriptor &descriptor) {
  if (descriptor.get() > STDERR_FILENO)
    return llvm::Error::success();
  const int moved =
      ::fcntl(descriptor.get(), F_DUPFD_CLOEXEC, STDERR_FILENO + 1);
  if (moved < 0)
    return systemError("could not isolate ToolRunner descriptor", errno);
  descriptor.reset(moved);
  return llvm::Error::success();
}

llvm::Expected<Pipe> createPipe() {
  int descriptors[2];
  if (::pipe2(descriptors, O_CLOEXEC) < 0)
    return systemError("could not create ToolRunner pipe", errno);
  Pipe pipe{OwnedFileDescriptor(descriptors[0]),
            OwnedFileDescriptor(descriptors[1])};
  if (llvm::Error error = moveAboveStandard(pipe.read))
    return std::move(error);
  if (llvm::Error error = moveAboveStandard(pipe.write))
    return std::move(error);
  return pipe;
}

llvm::Expected<ControlChannel> createControlChannel() {
  int descriptors[2];
  if (::socketpair(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC, 0, descriptors) < 0)
    return systemError("could not create ToolRunner control channel", errno);
  ControlChannel channel{OwnedFileDescriptor(descriptors[0]),
                         OwnedFileDescriptor(descriptors[1])};
  if (llvm::Error error = moveAboveStandard(channel.parent))
    return std::move(error);
  if (llvm::Error error = moveAboveStandard(channel.supervisor))
    return std::move(error);
  return channel;
}

llvm::Error setNonBlocking(int descriptor) {
  const int flags = ::fcntl(descriptor, F_GETFL);
  if (flags < 0)
    return systemError("could not read ToolRunner pipe flags", errno);
  if (::fcntl(descriptor, F_SETFL, flags | O_NONBLOCK) < 0)
    return systemError("could not make ToolRunner pipe nonblocking", errno);
  return llvm::Error::success();
}

llvm::Expected<fs::path> canonicalExistingPath(const fs::path &path,
                                               llvm::StringRef description) {
  std::error_code error;
  fs::path canonical = fs::canonical(path, error);
  if (error)
    return invocationError(description.str() + " path '" + path.string() +
                           "' is invalid: " + error.message());
  return canonical;
}

llvm::Error validateIdentity(llvm::StringRef identity,
                             llvm::StringRef description) {
  if (identity.empty())
    return invocationError(description.str() + " identity is empty");
  if (containsNull(identity))
    return invocationError(description.str() + " identity contains NUL");
  return llvm::Error::success();
}

llvm::Expected<PreparedInvocation>
prepareInvocation(const ToolInvocation &invocation) {
  if (llvm::Error error =
          validateIdentity(invocation.toolBindingIdentity, "tool binding"))
    return std::move(error);

  if (invocation.executablePath.empty() ||
      containsNull(invocation.executablePath))
    return invocationError("executable path is empty or contains NUL");
  const fs::path executable(invocation.executablePath);
  if (!executable.is_absolute())
    return invocationError("executable path must be absolute");
  std::error_code fileError;
  const fs::file_status executableStatus = fs::status(executable, fileError);
  if (fileError || !fs::is_regular_file(executableStatus))
    return invocationError("executable path '" + executable.string() +
                           "' is not a regular file");
  if (::access(executable.c_str(), X_OK) < 0)
    return invocationError("executable path '" + executable.string() +
                           "' is not executable: " + std::strerror(errno));

  if (invocation.argv.empty())
    return invocationError("argv must contain argv[0]");
  for (const std::string &argument : invocation.argv) {
    if (containsNull(argument))
      return invocationError("argv contains NUL");
  }

  if (invocation.scratchDirectory.empty() ||
      containsNull(invocation.scratchDirectory))
    return invocationError("scratch directory is empty or contains NUL");
  const fs::path requestedScratch(invocation.scratchDirectory);
  if (!requestedScratch.is_absolute())
    return invocationError("scratch directory must be absolute");
  llvm::Expected<fs::path> scratch =
      canonicalExistingPath(requestedScratch, "scratch directory");
  if (!scratch)
    return scratch.takeError();
  if (!fs::is_directory(*scratch))
    return invocationError("scratch path is not a directory");
  if (::access(scratch->c_str(), W_OK | X_OK) < 0)
    return invocationError("scratch directory is not writable: " +
                           std::string(std::strerror(errno)));

  for (const MaterializedInputArtifact &input : invocation.inputs) {
    if (input.artifact.empty())
      return invocationError("materialized input artifact identity is empty");
    if (input.path.empty() || containsNull(input.path))
      return invocationError(
          "materialized input path is empty or contains NUL");
    const fs::path inputPath(input.path);
    if (!inputPath.is_absolute())
      return invocationError("materialized input path must be absolute");
    std::error_code inputError;
    const fs::file_status status = fs::status(inputPath, inputError);
    if (inputError || !fs::exists(status))
      return invocationError("materialized input path '" + input.path +
                             "' does not exist");
  }

  std::vector<fs::path> outputs;
  std::set<std::string> outputSpellings;
  outputs.reserve(invocation.declaredOutputs.size());
  for (const std::string &spelling : invocation.declaredOutputs) {
    const fs::path output(spelling);
    if (spelling.empty() || containsNull(spelling) || output.is_absolute())
      return invocationError(
          "declared output must be a nonempty relative path");
    for (const fs::path &component : output) {
      if (component == "..")
        return invocationError("declared output path may not contain '..'");
    }
    const fs::path normalized = output.lexically_normal();
    if (normalized.empty() || normalized == ".")
      return invocationError("declared output must name a path under scratch");

    std::error_code outputError;
    const fs::path resolved =
        fs::weakly_canonical(*scratch / normalized, outputError);
    if (outputError)
      return invocationError("declared output path '" + spelling +
                             "' is invalid: " + outputError.message());
    if (!pathIsWithin(*scratch, resolved))
      return invocationError("declared output path '" + spelling +
                             "' escapes scratch");

    const std::string normalizedSpelling = normalized.generic_string();
    if (!outputSpellings.insert(normalizedSpelling).second)
      return invocationError("declared output path is duplicated: " +
                             normalizedSpelling);
    outputs.push_back(normalized);
  }

  if (invocation.timeout && invocation.timeout->count() < 0)
    return invocationError("timeout may not be negative");

  for (const std::string &identity :
       invocation.resourceLeaseBindingIdentities) {
    if (llvm::Error error =
            validateIdentity(identity, "resource lease binding"))
      return std::move(error);
  }
  for (const std::string &identity : invocation.licenseLeaseBindingIdentities) {
    if (llvm::Error error = validateIdentity(identity, "license lease binding"))
      return std::move(error);
  }

  std::map<std::string, std::string> environmentValues;
  if (environ) {
    for (char **entry = environ; *entry; ++entry) {
      llvm::StringRef assignment(*entry);
      const std::size_t separator = assignment.find('=');
      if (separator == llvm::StringRef::npos)
        continue;
      environmentValues[assignment.take_front(separator).str()] =
          assignment.drop_front(separator + 1).str();
    }
  }
  std::set<std::string> overlayNames;
  for (const EnvironmentVariable &variable : invocation.environmentOverlay) {
    if (variable.name.empty() || variable.name.find('=') != std::string::npos ||
        containsNull(variable.name))
      return invocationError("environment overlay has an invalid name");
    if (containsNull(variable.value))
      return invocationError("environment overlay value contains NUL");
    if (!overlayNames.insert(variable.name).second)
      return invocationError("environment overlay name is duplicated: " +
                             variable.name);
    environmentValues[variable.name] = variable.value;
  }

  PreparedInvocation prepared;
  prepared.scratchDirectory = std::move(*scratch);
  prepared.declaredOutputs = std::move(outputs);
  prepared.argvStorage = invocation.argv;
  prepared.argv.reserve(prepared.argvStorage.size() + 1);
  for (std::string &argument : prepared.argvStorage)
    prepared.argv.push_back(argument.data());
  prepared.argv.push_back(nullptr);

  prepared.environmentStorage.reserve(environmentValues.size());
  for (const auto &entry : environmentValues)
    prepared.environmentStorage.push_back(entry.first + "=" + entry.second);
  prepared.environment.reserve(prepared.environmentStorage.size() + 1);
  for (std::string &entry : prepared.environmentStorage)
    prepared.environment.push_back(entry.data());
  prepared.environment.push_back(nullptr);
  const long maximumDescriptor = ::sysconf(_SC_OPEN_MAX);
  prepared.maximumDescriptor =
      maximumDescriptor > 0 && maximumDescriptor <= INT_MAX
          ? static_cast<int>(maximumDescriptor)
          : 65536;
  return prepared;
}

template <typename Record>
void writeRecord(int descriptor, const Record &record) {
  const char *bytes = reinterpret_cast<const char *>(&record);
  std::size_t remaining = sizeof(record);
  while (remaining > 0) {
    const ssize_t written = ::write(descriptor, bytes, remaining);
    if (written > 0) {
      bytes += written;
      remaining -= static_cast<std::size_t>(written);
      continue;
    }
    if (written < 0 && errno == EINTR)
      continue;
    return;
  }
}

[[noreturn]] void reportLaunchFailure(int descriptor, LaunchStage stage,
                                      int errorNumber) {
  writeRecord(descriptor, LaunchRecord{static_cast<int>(stage), errorNumber});
  ::_exit(127);
}

timespec monotonicNow() {
  timespec now{};
  ::clock_gettime(CLOCK_MONOTONIC, &now);
  return now;
}

timespec addMilliseconds(timespec value, long milliseconds) {
  value.tv_sec += milliseconds / 1000;
  value.tv_nsec += (milliseconds % 1000) * 1000000L;
  if (value.tv_nsec >= 1000000000L) {
    ++value.tv_sec;
    value.tv_nsec -= 1000000000L;
  }
  return value;
}

bool before(timespec lhs, timespec rhs) {
  return lhs.tv_sec < rhs.tv_sec ||
         (lhs.tv_sec == rhs.tv_sec && lhs.tv_nsec < rhs.tv_nsec);
}

void sleepUntil(timespec deadline) {
  const timespec pause{0, 10000000L};
  while (before(monotonicNow(), deadline))
    ::nanosleep(&pause, nullptr);
}

int observeLeaderCompletion(pid_t process) {
  siginfo_t information{};
  if (::waitid(P_PID, static_cast<id_t>(process), &information,
               WEXITED | WNOHANG | WNOWAIT) < 0)
    return -errno;
  return information.si_pid == process ? 1 : 0;
}

int freezeLeaderForInterrupt(pid_t process) {
  if (::kill(-process, SIGSTOP) < 0 && errno != ESRCH)
    return -errno;
  const timespec deadline = addMilliseconds(
      monotonicNow(), static_cast<long>(kForcefulTermination.count()));
  const timespec pause{0, 10000000L};
  for (;;) {
    siginfo_t information{};
    if (::waitid(P_PID, static_cast<id_t>(process), &information,
                 WEXITED | WSTOPPED | WNOHANG | WNOWAIT) < 0)
      return -errno;
    if (information.si_pid == process) {
      if (information.si_code == CLD_EXITED ||
          information.si_code == CLD_KILLED ||
          information.si_code == CLD_DUMPED)
        return 1;
      if (information.si_code == CLD_STOPPED)
        return 0;
    }
    if (!before(monotonicNow(), deadline))
      return -ETIMEDOUT;
    ::nanosleep(&pause, nullptr);
  }
}

bool reapProcessGroupChildren(pid_t processGroup) {
  const timespec deadline = addMilliseconds(
      monotonicNow(), static_cast<long>(kForcefulTermination.count()));
  const timespec pause{0, 10000000L};
  for (;;) {
    int status = 0;
    pid_t waited = -1;
    do {
      waited = ::waitpid(-processGroup, &status, WNOHANG);
    } while (waited < 0 && errno == EINTR);
    if (waited < 0 && errno == ECHILD)
      return true;
    if (waited < 0)
      return false;
    if (waited == 0) {
      if (!before(monotonicNow(), deadline))
        return false;
      ::nanosleep(&pause, nullptr);
    }
  }
}

void reapExitedChildren() {
  int status = 0;
  while (::waitpid(-1, &status, WNOHANG) > 0) {
  }
}

int sendControlMessage(int descriptor, InterruptKind interrupt) {
  const std::uint8_t value = static_cast<std::uint8_t>(interrupt);
  for (;;) {
    const ssize_t sent =
        ::send(descriptor, &value, sizeof(value), MSG_NOSIGNAL | MSG_DONTWAIT);
    if (sent == static_cast<ssize_t>(sizeof(value)))
      return 0;
    if (sent < 0 && errno == EINTR)
      continue;
    if (sent < 0 && (errno == EPIPE || errno == ECONNRESET))
      return 0;
    return sent < 0 ? errno : EIO;
  }
}

int receiveControlMessage(int descriptor, InterruptKind &interrupt) {
  std::uint8_t value = 0;
  const ssize_t received =
      ::recv(descriptor, &value, sizeof(value), MSG_DONTWAIT);
  if (received == 0)
    return 0;
  if (received < 0 && (errno == EAGAIN || errno == EWOULDBLOCK))
    return 0;
  if (received < 0 && errno == EINTR)
    return 0;
  if (received < 0)
    return -errno;
  if (received != static_cast<ssize_t>(sizeof(value)) ||
      value < static_cast<std::uint8_t>(InterruptKind::TimedOut) ||
      value > static_cast<std::uint8_t>(InterruptKind::Abort))
    return -EPROTO;
  interrupt = static_cast<InterruptKind>(value);
  return 1;
}

int normalizeSignalState() {
  struct sigaction action{};
  action.sa_handler = SIG_DFL;
  ::sigemptyset(&action.sa_mask);
  for (int signalNumber = 1; signalNumber < NSIG; ++signalNumber) {
    if (signalNumber == SIGKILL || signalNumber == SIGSTOP)
      continue;
    if (::sigaction(signalNumber, &action, nullptr) < 0 && errno != EINVAL)
      return errno;
  }
  sigset_t emptyMask;
  ::sigemptyset(&emptyMask);
  if (::sigprocmask(SIG_SETMASK, &emptyMask, nullptr) < 0)
    return errno;
  return 0;
}

int closeUnintendedDescriptors(int preserved, int maximumDescriptor) {
#ifdef SYS_close_range
  auto closeRange = [](unsigned int first, unsigned int last) -> int {
    if (first > last)
      return 0;
    while (::syscall(SYS_close_range, first, last, 0) < 0) {
      if (errno == EINTR)
        continue;
      return errno;
    }
    return 0;
  };
  int closeError = 0;
  if (preserved > STDERR_FILENO + 1)
    closeError =
        closeRange(STDERR_FILENO + 1, static_cast<unsigned int>(preserved - 1));
  if (closeError == 0 && preserved < INT_MAX)
    closeError = closeRange(static_cast<unsigned int>(preserved + 1), UINT_MAX);
  if (closeError == 0)
    return 0;
  if (closeError != ENOSYS && closeError != EINVAL)
    return closeError;
#endif
  for (int descriptor = STDERR_FILENO + 1; descriptor < maximumDescriptor;
       ++descriptor) {
    if (descriptor != preserved)
      ::close(descriptor);
  }
  return 0;
}

class ActiveRunGuard {
public:
  ActiveRunGuard(pid_t supervisor, int controlDescriptor)
      : supervisor_(supervisor), controlDescriptor_(controlDescriptor) {}
  ActiveRunGuard(const ActiveRunGuard &) = delete;
  ActiveRunGuard &operator=(const ActiveRunGuard &) = delete;

  ~ActiveRunGuard() {
    if (supervisor_ <= 0)
      return;
    sendControlMessage(controlDescriptor_, InterruptKind::Abort);
    const pid_t alreadyExited = ::waitpid(supervisor_, nullptr, WNOHANG);
    if (alreadyExited == supervisor_ || (alreadyExited < 0 && errno == ECHILD))
      return;

    const timespec deadline = addMilliseconds(
        monotonicNow(),
        static_cast<long>(
            (kGracefulTermination + kForcefulTermination + kForcefulTermination)
                .count()));
    const timespec pause{0, 10000000L};
    while (before(monotonicNow(), deadline)) {
      const pid_t waited = ::waitpid(supervisor_, nullptr, WNOHANG);
      if (waited == supervisor_ || (waited < 0 && errno == ECHILD))
        return;
      ::nanosleep(&pause, nullptr);
    }
    ::kill(supervisor_, SIGKILL);
    while (::waitpid(supervisor_, nullptr, 0) < 0 && errno == EINTR) {
    }
  }

  void release() { supervisor_ = -1; }

private:
  pid_t supervisor_;
  int controlDescriptor_;
};

[[noreturn]] void superviseTool(int standardOutputRead, int standardOutputWrite,
                                int standardErrorRead, int standardErrorWrite,
                                int launchRead, int launchWrite,
                                int controlParent, int controlSupervisor,
                                int resultRead, int resultWrite,
                                const PreparedInvocation &prepared,
                                llvm::StringRef executablePath) {
  ::close(standardOutputRead);
  ::close(standardErrorRead);
  ::close(launchRead);
  ::close(controlParent);
  ::close(resultRead);

  const int signalError = normalizeSignalState();
  if (signalError != 0) {
    writeRecord(launchWrite,
                LaunchRecord{static_cast<int>(LaunchStage::SupervisorSetup),
                             signalError});
    writeRecord(resultWrite, WaitRecord{0, 0, signalError, 1,
                                        static_cast<int>(InterruptKind::None)});
    ::_exit(127);
  }

  if (::prctl(PR_SET_CHILD_SUBREAPER, 1) < 0) {
    const int errorNumber = errno;
    writeRecord(launchWrite,
                LaunchRecord{static_cast<int>(LaunchStage::SupervisorSetup),
                             errorNumber});
    writeRecord(resultWrite, WaitRecord{0, 0, errorNumber, 1,
                                        static_cast<int>(InterruptKind::None)});
    ::_exit(127);
  }

  const pid_t toolProcess = ::fork();
  if (toolProcess < 0) {
    const int errorNumber = errno;
    writeRecord(
        launchWrite,
        LaunchRecord{static_cast<int>(LaunchStage::ProcessFork), errorNumber});
    writeRecord(resultWrite, WaitRecord{0, 0, errorNumber, 1,
                                        static_cast<int>(InterruptKind::None)});
    ::_exit(127);
  }

  if (toolProcess == 0) {
    ::close(controlSupervisor);
    ::close(resultWrite);
    if (::setpgid(0, 0) < 0)
      reportLaunchFailure(launchWrite, LaunchStage::ProcessGroup, errno);
    if (::chdir(prepared.scratchDirectory.c_str()) < 0)
      reportLaunchFailure(launchWrite, LaunchStage::WorkingDirectory, errno);
    if (::dup2(standardOutputWrite, STDOUT_FILENO) < 0)
      reportLaunchFailure(launchWrite, LaunchStage::StandardOutput, errno);
    if (::dup2(standardErrorWrite, STDERR_FILENO) < 0)
      reportLaunchFailure(launchWrite, LaunchStage::StandardError, errno);
    if (standardOutputWrite != STDOUT_FILENO)
      ::close(standardOutputWrite);
    if (standardErrorWrite != STDERR_FILENO)
      ::close(standardErrorWrite);
    const int descriptorError =
        closeUnintendedDescriptors(launchWrite, prepared.maximumDescriptor);
    if (descriptorError != 0)
      reportLaunchFailure(launchWrite, LaunchStage::DescriptorIsolation,
                          descriptorError);
    ::execve(executablePath.data(), prepared.argv.data(),
             prepared.environment.data());
    reportLaunchFailure(launchWrite, LaunchStage::Execute, errno);
  }

  ::close(standardOutputWrite);
  ::close(standardErrorWrite);

  if (::setpgid(toolProcess, toolProcess) < 0 && errno != EACCES &&
      errno != ESRCH) {
    const int errorNumber = errno;
    ::kill(toolProcess, SIGKILL);
    int status = 0;
    while (::waitpid(toolProcess, &status, 0) < 0 && errno == EINTR) {
    }
    writeRecord(
        launchWrite,
        LaunchRecord{static_cast<int>(LaunchStage::ProcessGroup), errorNumber});
    writeRecord(resultWrite, WaitRecord{1, status, 0, 1,
                                        static_cast<int>(InterruptKind::None)});
    ::_exit(127);
  }
  ::close(launchWrite);

  InterruptKind interrupt = InterruptKind::None;
  bool terminationStarted = false;
  bool forceSent = false;
  bool leaderCompleted = false;
  timespec gracefulDeadline{};
  timespec forcefulDeadline{};
  int monitorError = 0;

  for (;;) {
    const int completion = observeLeaderCompletion(toolProcess);
    if (completion > 0) {
      leaderCompleted = true;
      break;
    }
    if (completion < 0 && monitorError == 0)
      monitorError = -completion;

    const timespec now = monotonicNow();
    if ((monitorError != 0 || interrupt != InterruptKind::None) &&
        !terminationStarted) {
      if (interrupt == InterruptKind::None)
        interrupt = InterruptKind::Abort;
      ::kill(-toolProcess, SIGTERM);
      terminationStarted = true;
      gracefulDeadline =
          addMilliseconds(now, static_cast<long>(kGracefulTermination.count()));
    }
    if (terminationStarted && !forceSent && !before(now, gracefulDeadline)) {
      ::kill(-toolProcess, SIGKILL);
      forceSent = true;
      forcefulDeadline =
          addMilliseconds(now, static_cast<long>(kForcefulTermination.count()));
    }
    if (forceSent && !before(now, forcefulDeadline)) {
      if (monitorError == 0)
        monitorError = ETIMEDOUT;
      break;
    }

    pollfd controlPoll{controlSupervisor, POLLIN | POLLHUP, 0};
    const int pollResult =
        ::poll(&controlPoll, 1, static_cast<int>(kPollInterval.count()));
    if (pollResult < 0 && errno != EINTR) {
      if (monitorError == 0)
        monitorError = errno;
      continue;
    }
    if (pollResult <= 0 || interrupt != InterruptKind::None ||
        monitorError != 0)
      continue;

    InterruptKind requested = InterruptKind::None;
    int receiveResult = 0;
    if (controlPoll.revents & POLLIN)
      receiveResult = receiveControlMessage(controlSupervisor, requested);
    else if (controlPoll.revents & POLLHUP) {
      requested = InterruptKind::Abort;
      receiveResult = 1;
    }
    if (receiveResult < 0) {
      monitorError = -receiveResult;
      continue;
    }
    if (receiveResult == 0)
      continue;

    const int freezeResult = freezeLeaderForInterrupt(toolProcess);
    if (freezeResult > 0) {
      leaderCompleted = true;
      break;
    }
    if (freezeResult < 0) {
      monitorError = -freezeResult;
      continue;
    }
    interrupt = requested;
    ::kill(-toolProcess, SIGTERM);
    ::kill(-toolProcess, SIGCONT);
    terminationStarted = true;
    gracefulDeadline = addMilliseconds(
        monotonicNow(), static_cast<long>(kGracefulTermination.count()));
  }

  if (leaderCompleted) {
    const timespec now = monotonicNow();
    if (!terminationStarted) {
      ::kill(-toolProcess, SIGTERM);
      terminationStarted = true;
      gracefulDeadline =
          addMilliseconds(now, static_cast<long>(kGracefulTermination.count()));
    }
    if (!forceSent) {
      sleepUntil(gracefulDeadline);
      ::kill(-toolProcess, SIGKILL);
      forceSent = true;
    }
  }

  int waitStatus = 0;
  pid_t waited = -1;
  if (leaderCompleted) {
    do {
      waited = ::waitpid(toolProcess, &waitStatus, 0);
    } while (waited < 0 && errno == EINTR);
  }
  const int waitError =
      waited < 0 ? (monitorError != 0 ? monitorError : errno) : monitorError;
  const bool cleanupComplete =
      leaderCompleted && reapProcessGroupChildren(toolProcess);
  reapExitedChildren();
  writeRecord(resultWrite,
              WaitRecord{waited == toolProcess ? 1 : 0, waitStatus, waitError,
                         cleanupComplete ? 1 : 0, static_cast<int>(interrupt)});
  ::_exit(0);
}

std::size_t pipeCapacity(const OwnedFileDescriptor &descriptor) {
  if (!descriptor)
    return 0;
#ifdef F_GETPIPE_SZ
  const int capacity = ::fcntl(descriptor.get(), F_GETPIPE_SZ);
  if (capacity > 0)
    return static_cast<std::size_t>(capacity);
#endif
  return kDrainQuota;
}

template <typename Buffer>
llvm::Error drainDescriptor(OwnedFileDescriptor &descriptor, Buffer &buffer,
                            std::size_t quota = kDrainQuota) {
  char bytes[4096];
  std::size_t drained = 0;
  while (descriptor && drained < quota) {
    const std::size_t remaining = quota - drained;
    const ssize_t count =
        ::read(descriptor.get(), bytes, std::min(sizeof(bytes), remaining));
    if (count > 0) {
      buffer.insert(buffer.end(), bytes, bytes + count);
      drained += static_cast<std::size_t>(count);
      continue;
    }
    if (count == 0) {
      descriptor.reset();
      return llvm::Error::success();
    }
    if (errno == EINTR)
      continue;
    if (errno == EAGAIN || errno == EWOULDBLOCK)
      return llvm::Error::success();
    return systemError("could not read ToolRunner pipe", errno);
  }
  return llvm::Error::success();
}

template <typename Record>
llvm::Expected<Record> decodeRecord(const std::vector<char> &bytes,
                                    llvm::StringRef description) {
  if (bytes.size() != sizeof(Record))
    return llvm::createStringError(
        std::errc::io_error, "%s record has %zu bytes, expected %zu",
        description.str().c_str(), bytes.size(), sizeof(Record));
  Record record;
  std::memcpy(&record, bytes.data(), sizeof(record));
  return record;
}

llvm::StringRef launchStageName(LaunchStage stage) {
  switch (stage) {
  case LaunchStage::SupervisorSetup:
    return "launcher setup";
  case LaunchStage::ProcessFork:
    return "fork";
  case LaunchStage::ProcessGroup:
    return "process group setup";
  case LaunchStage::WorkingDirectory:
    return "working directory setup";
  case LaunchStage::StandardOutput:
    return "stdout setup";
  case LaunchStage::StandardError:
    return "stderr setup";
  case LaunchStage::DescriptorIsolation:
    return "descriptor isolation";
  case LaunchStage::Execute:
    return "execve";
  }
  return "unknown launch stage";
}

std::string launchErrorMessage(const LaunchRecord &record) {
  const LaunchStage stage = static_cast<LaunchStage>(record.stage);
  return launchStageName(stage).str() + ": " +
         std::error_code(record.errorNumber, std::generic_category()).message();
}

llvm::Expected<InventorySnapshot>
snapshotDeclaredOutputs(const PreparedInvocation &prepared) {
  InventorySnapshot snapshot;

  auto relativeSpelling =
      [&](const fs::path &path) -> llvm::Expected<std::string> {
    const fs::path relative =
        path.lexically_relative(prepared.scratchDirectory);
    if (relative.empty() || relative.is_absolute())
      return llvm::createStringError(std::errc::invalid_argument,
                                     "could not relativize output path");
    return relative.generic_string();
  };

  auto validateSymlink = [&](const fs::path &path) -> llvm::Error {
    llvm::Expected<std::string> relative = relativeSpelling(path);
    if (!relative)
      return relative.takeError();
    std::error_code canonicalError;
    const fs::path canonical = fs::canonical(path, canonicalError);
    if (canonicalError)
      return llvm::createStringError(canonicalError,
                                     "output symlink '%s' cannot be resolved",
                                     relative->c_str());
    if (!pathIsWithin(prepared.scratchDirectory, canonical))
      return llvm::createStringError(std::errc::permission_denied,
                                     "output symlink '%s' escapes scratch",
                                     relative->c_str());
    return llvm::Error::success();
  };

  auto addFile = [&](const fs::path &path) -> llvm::Error {
    llvm::Expected<std::string> relative = relativeSpelling(path);
    if (!relative)
      return relative.takeError();
    std::error_code canonicalError;
    const fs::path canonical = fs::canonical(path, canonicalError);
    if (canonicalError)
      return llvm::createStringError(
          canonicalError, "could not inventory output '%s'", relative->c_str());
    if (!pathIsWithin(prepared.scratchDirectory, canonical))
      return llvm::createStringError(std::errc::permission_denied,
                                     "output file '%s' escapes scratch",
                                     relative->c_str());

    struct stat status{};
    if (::stat(path.c_str(), &status) < 0)
      return llvm::createStringError(
          std::error_code(errno, std::generic_category()),
          "could not stat output '%s'", relative->c_str());
    snapshot[*relative] =
        FileSnapshot{static_cast<std::uint64_t>(status.st_dev),
                     static_cast<std::uint64_t>(status.st_ino),
                     static_cast<std::uint64_t>(status.st_size),
                     static_cast<std::int64_t>(status.st_mtim.tv_sec),
                     static_cast<std::int64_t>(status.st_mtim.tv_nsec),
                     static_cast<std::int64_t>(status.st_ctim.tv_sec),
                     static_cast<std::int64_t>(status.st_ctim.tv_nsec)};
    return llvm::Error::success();
  };

  for (const fs::path &declared : prepared.declaredOutputs) {
    const fs::path root = prepared.scratchDirectory / declared;
    std::error_code statusError;
    const fs::file_status status = fs::symlink_status(root, statusError);
    if (statusError == std::errc::no_such_file_or_directory)
      continue;
    if (statusError)
      return llvm::createStringError(
          statusError, "could not inspect output '%s'", declared.c_str());
    if (fs::is_symlink(status)) {
      if (llvm::Error error = validateSymlink(root))
        return std::move(error);
      continue;
    }
    if (fs::is_regular_file(status)) {
      if (llvm::Error error = addFile(root))
        return std::move(error);
      continue;
    }
    if (!fs::is_directory(status))
      continue;

    std::error_code iteratorError;
    fs::recursive_directory_iterator iterator(root, iteratorError);
    const fs::recursive_directory_iterator end;
    if (iteratorError)
      return llvm::createStringError(
          iteratorError, "could not enumerate output '%s'", declared.c_str());
    while (iterator != end) {
      const fs::directory_entry entry = *iterator;
      std::error_code entryError;
      const fs::file_status entryStatus = entry.symlink_status(entryError);
      llvm::Expected<std::string> relative = relativeSpelling(entry.path());
      if (!relative)
        return relative.takeError();
      if (entryError)
        return llvm::createStringError(
            entryError, "could not inspect output '%s'", relative->c_str());
      if (fs::is_symlink(entryStatus)) {
        iterator.disable_recursion_pending();
        if (llvm::Error error = validateSymlink(entry.path()))
          return std::move(error);
      } else if (fs::is_regular_file(entryStatus)) {
        if (llvm::Error error = addFile(entry.path()))
          return std::move(error);
      }
      iterator.increment(iteratorError);
      if (iteratorError)
        return llvm::createStringError(
            iteratorError, "could not enumerate output '%s'", declared.c_str());
    }
  }

  return snapshot;
}

std::vector<std::string> changedProducedFiles(const InventorySnapshot &before,
                                              const InventorySnapshot &after) {
  std::vector<std::string> produced;
  for (const auto &entry : after) {
    const auto previous = before.find(entry.first);
    if (previous == before.end() || !(previous->second == entry.second))
      produced.push_back(entry.first);
  }
  return produced;
}

ToolRunOutcome baseOutcome(const ToolInvocation &invocation,
                           std::chrono::system_clock::time_point startedAt) {
  ToolRunOutcome outcome;
  outcome.startedAt = startedAt;
  outcome.toolBindingIdentity = invocation.toolBindingIdentity;
  outcome.resourceLeaseBindingIdentities =
      invocation.resourceLeaseBindingIdentities;
  outcome.licenseLeaseBindingIdentities =
      invocation.licenseLeaseBindingIdentities;
  return outcome;
}

} // namespace

llvm::Expected<ToolRunOutcome> runTool(const ToolInvocation &invocation) {
  llvm::Expected<PreparedInvocation> preparedValue =
      prepareInvocation(invocation);
  if (!preparedValue)
    return preparedValue.takeError();
  PreparedInvocation prepared = std::move(*preparedValue);
  llvm::Expected<InventorySnapshot> inventoryBeforeValue =
      snapshotDeclaredOutputs(prepared);
  if (!inventoryBeforeValue)
    return inventoryBeforeValue.takeError();
  InventorySnapshot inventoryBefore = std::move(*inventoryBeforeValue);

  const auto startedAt = std::chrono::system_clock::now();
  const auto steadyStart = std::chrono::steady_clock::now();
  ToolRunOutcome outcome = baseOutcome(invocation, startedAt);

  llvm::Expected<Pipe> standardOutputValue = createPipe();
  if (!standardOutputValue)
    return standardOutputValue.takeError();
  Pipe standardOutput = std::move(*standardOutputValue);
  llvm::Expected<Pipe> standardErrorValue = createPipe();
  if (!standardErrorValue)
    return standardErrorValue.takeError();
  Pipe standardError = std::move(*standardErrorValue);
  llvm::Expected<Pipe> launchValue = createPipe();
  if (!launchValue)
    return launchValue.takeError();
  Pipe launch = std::move(*launchValue);
  llvm::Expected<ControlChannel> controlValue = createControlChannel();
  if (!controlValue)
    return controlValue.takeError();
  ControlChannel control = std::move(*controlValue);
  llvm::Expected<Pipe> resultValue = createPipe();
  if (!resultValue)
    return resultValue.takeError();
  Pipe result = std::move(*resultValue);

  sigset_t allSignals;
  sigset_t previousSignalMask;
  ::sigfillset(&allSignals);
  const int blockError =
      ::pthread_sigmask(SIG_SETMASK, &allSignals, &previousSignalMask);
  if (blockError != 0)
    return systemError("could not block signals before ToolRunner fork",
                       blockError);

  const pid_t supervisor = ::fork();
  const int forkError = supervisor < 0 ? errno : 0;
  if (supervisor == 0) {
    superviseTool(standardOutput.read.get(), standardOutput.write.get(),
                  standardError.read.get(), standardError.write.get(),
                  launch.read.get(), launch.write.get(), control.parent.get(),
                  control.supervisor.get(), result.read.get(),
                  result.write.get(), prepared, invocation.executablePath);
  }

  const int restoreMaskError =
      ::pthread_sigmask(SIG_SETMASK, &previousSignalMask, nullptr);
  if (supervisor < 0) {
    const int errorNumber = forkError;
    outcome.status = ToolRunStatus::LaunchFailure;
    outcome.launchErrorNumber = errorNumber;
    outcome.launchErrorMessage =
        "fork: " +
        std::error_code(errorNumber, std::generic_category()).message();
    outcome.endedAt = std::chrono::system_clock::now();
    return outcome;
  }

  if (restoreMaskError != 0) {
    sendControlMessage(control.parent.get(), InterruptKind::Abort);
    while (::waitpid(supervisor, nullptr, 0) < 0 && errno == EINTR) {
    }
    return systemError("could not restore signals after ToolRunner fork",
                       restoreMaskError);
  }

  ActiveRunGuard activeRun(supervisor, control.parent.get());

  standardOutput.write.reset();
  standardError.write.reset();
  launch.write.reset();
  control.supervisor.reset();
  result.write.reset();

  if (llvm::Error error = setNonBlocking(standardOutput.read.get()))
    return std::move(error);
  if (llvm::Error error = setNonBlocking(standardError.read.get()))
    return std::move(error);
  if (llvm::Error error = setNonBlocking(launch.read.get()))
    return std::move(error);
  if (llvm::Error error = setNonBlocking(result.read.get()))
    return std::move(error);

  std::vector<char> launchBytes;
  std::vector<char> resultBytes;
  bool interruptRequested = false;

  while (result.read) {
    if (llvm::Error error =
            drainDescriptor(standardOutput.read, outcome.standardOutput))
      return std::move(error);
    if (llvm::Error error =
            drainDescriptor(standardError.read, outcome.standardError))
      return std::move(error);
    if (llvm::Error error = drainDescriptor(launch.read, launchBytes))
      return std::move(error);
    if (llvm::Error error = drainDescriptor(result.read, resultBytes))
      return std::move(error);

    if (!result.read)
      break;

    if (!interruptRequested) {
      bool cancelled = false;
      if (invocation.cancellationRequested)
        cancelled = invocation.cancellationRequested();
      const auto now = std::chrono::steady_clock::now();
      const bool timedOut =
          invocation.timeout && now - steadyStart >= *invocation.timeout;
      if (cancelled || timedOut) {
        const InterruptKind requested =
            cancelled ? InterruptKind::Cancelled : InterruptKind::TimedOut;
        const int controlError =
            sendControlMessage(control.parent.get(), requested);
        if (controlError != 0)
          return systemError("could not interrupt ToolRunner supervisor",
                             controlError);
        interruptRequested = true;
      }
    }

    pollfd descriptors[4];
    nfds_t count = 0;
    auto append = [&](const OwnedFileDescriptor &descriptor) {
      if (descriptor)
        descriptors[count++] = pollfd{descriptor.get(), POLLIN | POLLHUP, 0};
    };
    append(standardOutput.read);
    append(standardError.read);
    append(launch.read);
    append(result.read);
    const int pollResult =
        ::poll(descriptors, count, static_cast<int>(kPollInterval.count()));
    if (pollResult < 0 && errno != EINTR)
      return systemError("ToolRunner poll failed", errno);
  }

  const std::size_t finalOutputQuota = pipeCapacity(standardOutput.read);
  const std::size_t finalErrorQuota = pipeCapacity(standardError.read);
  if (llvm::Error error = drainDescriptor(
          standardOutput.read, outcome.standardOutput, finalOutputQuota))
    return std::move(error);
  if (llvm::Error error = drainDescriptor(
          standardError.read, outcome.standardError, finalErrorQuota))
    return std::move(error);
  if (llvm::Error error = drainDescriptor(launch.read, launchBytes))
    return std::move(error);
  standardOutput.read.reset();
  standardError.read.reset();
  launch.read.reset();

  pid_t waitedSupervisor = -1;
  do {
    waitedSupervisor = ::waitpid(supervisor, nullptr, 0);
  } while (waitedSupervisor < 0 && errno == EINTR);
  if (waitedSupervisor < 0 && errno != ECHILD)
    return systemError("could not wait for ToolRunner supervisor", errno);
  activeRun.release();

  std::optional<LaunchRecord> launchRecord;
  if (!launchBytes.empty()) {
    llvm::Expected<LaunchRecord> decoded =
        decodeRecord<LaunchRecord>(launchBytes, "launch");
    if (!decoded)
      return decoded.takeError();
    launchRecord = *decoded;
  }

  std::optional<WaitRecord> waitRecord;
  if (!resultBytes.empty()) {
    llvm::Expected<WaitRecord> decoded =
        decodeRecord<WaitRecord>(resultBytes, "wait");
    if (!decoded)
      return decoded.takeError();
    waitRecord = *decoded;
  }

  if (launchRecord) {
    outcome.status = ToolRunStatus::LaunchFailure;
    outcome.launchErrorNumber = launchRecord->errorNumber;
    outcome.launchErrorMessage = launchErrorMessage(*launchRecord);
  } else {
    if (!waitRecord || !waitRecord->valid)
      return systemError("ToolRunner could not wait for tool process",
                         waitRecord ? waitRecord->errorNumber : ECHILD);
    if (!waitRecord->cleanupComplete)
      return llvm::createStringError(
          std::errc::timed_out,
          "ToolRunner could not terminate and reap the process group");

    if (WIFEXITED(waitRecord->waitStatus))
      outcome.exitCode = WEXITSTATUS(waitRecord->waitStatus);
    if (WIFSIGNALED(waitRecord->waitStatus))
      outcome.terminationSignal = WTERMSIG(waitRecord->waitStatus);

    const InterruptKind interrupt =
        static_cast<InterruptKind>(waitRecord->interruptKind);
    if (interrupt == InterruptKind::TimedOut)
      outcome.status = ToolRunStatus::TimedOut;
    else if (interrupt == InterruptKind::Cancelled)
      outcome.status = ToolRunStatus::Cancelled;
    else if (interrupt == InterruptKind::Abort)
      return llvm::createStringError(std::errc::operation_canceled,
                                     "ToolRunner supervisor aborted");
    else if (interrupt != InterruptKind::None)
      return llvm::createStringError(
          std::errc::protocol_error,
          "ToolRunner returned an invalid interrupt");
    else if (outcome.exitCode)
      outcome.status = ToolRunStatus::Exited;
    else if (outcome.terminationSignal)
      outcome.status = ToolRunStatus::Signaled;
    else
      return llvm::createStringError(std::errc::io_error,
                                     "tool returned an unknown wait status");
  }

  llvm::Expected<InventorySnapshot> inventoryAfter =
      snapshotDeclaredOutputs(prepared);
  if (!inventoryAfter)
    outcome.inventoryDiagnostic = llvm::toString(inventoryAfter.takeError());
  else
    outcome.producedFiles =
        changedProducedFiles(inventoryBefore, *inventoryAfter);
  outcome.endedAt = std::chrono::system_clock::now();
  return outcome;
}

} // namespace loom::evaluation
