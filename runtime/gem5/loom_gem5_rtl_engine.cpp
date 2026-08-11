#include "Gem5BridgeWire.h"
#include "Vloom_mapped_rtl_testbench.h"
#include "verilated.h"

#include <cerrno>
#include <csignal>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <string>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <unistd.h>
#include <utility>
#include <vector>

namespace {

struct Options final {
  std::string socketPath;
  std::string expectedLaunchPath;
  std::string mappedResultPath;
  std::string gem5Executable;
  std::string gem5OutputDirectory;
  std::string gem5Configuration;
  std::string projection;
  std::string systemResultPath;
  std::uint64_t ticksPerCycle = 0;
};

bool parseUnsigned(const std::string &text, std::uint64_t &value) {
  if (text.empty())
    return false;
  value = 0;
  for (const char character : text) {
    if (character < '0' || character > '9')
      return false;
    const std::uint64_t digit = static_cast<std::uint64_t>(character - '0');
    if (value > (std::numeric_limits<std::uint64_t>::max() - digit) / 10)
      return false;
    value = value * 10 + digit;
  }
  return true;
}

bool parseArguments(int argc, char **argv, Options &options,
                    std::vector<std::string> &verilatorArguments) {
  verilatorArguments.push_back(argv[0]);
  for (int index = 1; index < argc; ++index) {
    const std::string argument(argv[index]);
    if (!argument.empty() && argument.front() == '+') {
      verilatorArguments.push_back(argument);
      continue;
    }
    if (index + 1 >= argc)
      return false;
    const std::string value(argv[++index]);
    if (argument == "--socket")
      options.socketPath = value;
    else if (argument == "--expected-launch")
      options.expectedLaunchPath = value;
    else if (argument == "--mapped-result")
      options.mappedResultPath = value;
    else if (argument == "--ticks-per-cycle") {
      if (!parseUnsigned(value, options.ticksPerCycle))
        return false;
    } else if (argument == "--gem5")
      options.gem5Executable = value;
    else if (argument == "--gem5-output")
      options.gem5OutputDirectory = value;
    else if (argument == "--gem5-config")
      options.gem5Configuration = value;
    else if (argument == "--projection")
      options.projection = value;
    else if (argument == "--system-result")
      options.systemResultPath = value;
    else
      return false;
  }
  return !options.socketPath.empty() && !options.expectedLaunchPath.empty() &&
         !options.mappedResultPath.empty() && !options.gem5Executable.empty() &&
         !options.gem5OutputDirectory.empty() &&
         !options.gem5Configuration.empty() && !options.projection.empty() &&
         !options.systemResultPath.empty() && options.ticksPerCycle != 0;
}

bool readAll(int descriptor, std::uint8_t *bytes, std::size_t size) {
  while (size != 0) {
    const ssize_t count = ::read(descriptor, bytes, size);
    if (count == 0)
      return false;
    if (count < 0) {
      if (errno == EINTR)
        continue;
      return false;
    }
    bytes += count;
    size -= static_cast<std::size_t>(count);
  }
  return true;
}

bool writeAll(int descriptor, const std::uint8_t *bytes, std::size_t size) {
  while (size != 0) {
    const ssize_t count = ::write(descriptor, bytes, size);
    if (count < 0) {
      if (errno == EINTR)
        continue;
      return false;
    }
    bytes += count;
    size -= static_cast<std::size_t>(count);
  }
  return true;
}

bool readFile(const std::string &path, std::vector<std::uint8_t> &bytes) {
  std::ifstream input(path, std::ios::binary);
  if (!input)
    return false;
  bytes.assign(std::istreambuf_iterator<char>(input),
               std::istreambuf_iterator<char>());
  return input.good() || input.eof();
}

int openServer(const std::string &path) {
  if (path.size() >= sizeof(sockaddr_un::sun_path))
    return -1;
  const int server = ::socket(AF_UNIX, SOCK_STREAM, 0);
  if (server < 0)
    return -1;
  ::unlink(path.c_str());
  sockaddr_un address{};
  address.sun_family = AF_UNIX;
  std::memcpy(address.sun_path, path.c_str(), path.size() + 1);
  if (::bind(server, reinterpret_cast<sockaddr *>(&address),
             sizeof(address)) != 0 ||
      ::listen(server, 1) != 0) {
    ::close(server);
    ::unlink(path.c_str());
    return -1;
  }
  return server;
}

pid_t launchGem5(const Options &options) {
  const pid_t child = ::fork();
  if (child != 0)
    return child;
  ::execl(options.gem5Executable.c_str(), options.gem5Executable.c_str(), "-d",
          options.gem5OutputDirectory.c_str(),
          options.gem5Configuration.c_str(), "--projection",
          options.projection.c_str(), "--result",
          options.systemResultPath.c_str(), static_cast<char *>(nullptr));
  _exit(127);
}

bool receiveLaunch(int connection,
                   loom::runtime::Gem5BridgeMessage &message) {
  std::vector<std::uint8_t> bytes(loom::runtime::gem5BridgeWireHeaderBytes);
  if (!readAll(connection, bytes.data(), bytes.size()))
    return false;
  const std::uint64_t payloadSize =
      loom::runtime::detail::readGem5BridgeU64(bytes.data() + 16);
  if (payloadSize > 64ULL * 1024ULL * 1024ULL ||
      payloadSize > std::numeric_limits<std::size_t>::max())
    return false;
  const std::size_t headerSize = bytes.size();
  bytes.resize(headerSize + static_cast<std::size_t>(payloadSize));
  if (payloadSize != 0 &&
      !readAll(connection, bytes.data() + headerSize,
               static_cast<std::size_t>(payloadSize)))
    return false;
  std::string diagnostic;
  return loom::runtime::decodeGem5BridgeWireMessage(bytes, message,
                                                    diagnostic);
}

bool sendCompletion(int connection, std::uint64_t sequence,
                    std::uint64_t delay, std::uint32_t status,
                    std::vector<std::uint8_t> result) {
  const loom::runtime::Gem5BridgeCompletion completion{
      delay, status, std::move(result)};
  const loom::runtime::Gem5BridgeMessage message{
      loom::runtime::Gem5BridgeMessageKind::Completion, sequence,
      loom::runtime::encodeGem5BridgeCompletion(completion)};
  const std::vector<std::uint8_t> bytes =
      loom::runtime::encodeGem5BridgeWireMessage(message);
  return writeAll(connection, bytes.data(), bytes.size());
}

bool waitForChild(pid_t child) {
  int status = 0;
  while (::waitpid(child, &status, 0) < 0) {
    if (errno != EINTR)
      return false;
  }
  return WIFEXITED(status) && WEXITSTATUS(status) == 0;
}

void stopChild(pid_t child) {
  if (child <= 0)
    return;
  ::kill(child, SIGTERM);
  int status = 0;
  while (::waitpid(child, &status, 0) < 0 && errno == EINTR) {
  }
}

} // namespace

int main(int argc, char **argv) {
  Options options;
  std::vector<std::string> verilatorArguments;
  if (!parseArguments(argc, argv, options, verilatorArguments)) {
    std::cerr << "invalid gem5 RTL engine arguments\n";
    return 2;
  }
  std::vector<char *> verilatorPointers;
  for (std::string &argument : verilatorArguments)
    verilatorPointers.push_back(argument.data());
  VerilatedContext context;
  context.commandArgs(static_cast<int>(verilatorPointers.size()),
                      verilatorPointers.data());

  std::vector<std::uint8_t> expectedLaunch;
  if (!readFile(options.expectedLaunchPath, expectedLaunch)) {
    std::cerr << "cannot read the expected Spatial launch\n";
    return 3;
  }
  const int server = openServer(options.socketPath);
  if (server < 0) {
    std::cerr << "cannot publish the bridge socket\n";
    return 4;
  }
  const pid_t gem5 = launchGem5(options);
  if (gem5 < 0) {
    ::close(server);
    ::unlink(options.socketPath.c_str());
    std::cerr << "cannot launch gem5\n";
    return 5;
  }
  const int connection = ::accept(server, nullptr, nullptr);
  ::close(server);
  if (connection < 0) {
    stopChild(gem5);
    ::unlink(options.socketPath.c_str());
    std::cerr << "cannot accept the bridge connection\n";
    return 6;
  }
  loom::runtime::Gem5BridgeMessage launch;
  if (!receiveLaunch(connection, launch) ||
      launch.kind != loom::runtime::Gem5BridgeMessageKind::SpatialLaunch ||
      launch.sequence != 0 || launch.payload != expectedLaunch) {
    ::close(connection);
    stopChild(gem5);
    ::unlink(options.socketPath.c_str());
    std::cerr << "bridge launch differs from the exact Deployment\n";
    return 7;
  }

  Vloom_mapped_rtl_testbench top{&context};
  top.eval();
  while (!context.gotFinish()) {
    if (!top.eventsPending()) {
      ::close(connection);
      stopChild(gem5);
      ::unlink(options.socketPath.c_str());
      std::cerr << "RTL engine has no pending event before terminal\n";
      return 8;
    }
    context.time(top.nextTimeSlot());
    top.eval();
  }
  top.final();
  std::vector<std::uint8_t> result;
  if (!readFile(options.mappedResultPath, result)) {
    ::close(connection);
    stopChild(gem5);
    ::unlink(options.socketPath.c_str());
    std::cerr << "RTL harness did not publish its result\n";
    return 9;
  }
  const bool retired = top.loom_engine_retired != 0;
  const std::uint64_t launchCycle = top.loom_engine_launch_cycle;
  const std::uint64_t retirementCycle = top.loom_engine_retirement_cycle;
  if (retired && retirementCycle < launchCycle) {
    ::close(connection);
    stopChild(gem5);
    ::unlink(options.socketPath.c_str());
    std::cerr << "RTL harness published reversed progress\n";
    return 10;
  }
  const std::uint64_t cycles = retired ? retirementCycle - launchCycle : 0;
  if (cycles > std::numeric_limits<std::uint64_t>::max() /
                   options.ticksPerCycle) {
    ::close(connection);
    stopChild(gem5);
    ::unlink(options.socketPath.c_str());
    std::cerr << "RTL completion delay overflows gem5 ticks\n";
    return 11;
  }
  if (!sendCompletion(connection, launch.sequence,
                      cycles * options.ticksPerCycle, retired ? 0U : 1U,
                      std::move(result))) {
    ::close(connection);
    stopChild(gem5);
    ::unlink(options.socketPath.c_str());
    std::cerr << "cannot send the RTL completion\n";
    return 12;
  }
  ::close(connection);
  ::unlink(options.socketPath.c_str());
  if (!waitForChild(gem5)) {
    std::cerr << "gem5 did not complete successfully\n";
    return 13;
  }
  return 0;
}
