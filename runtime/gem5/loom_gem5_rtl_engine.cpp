#include "Gem5BridgeWire.h"
#include "Gem5SpatialChannelPlan.h"
#include "Vloom_mapped_rtl_testbench.h"
#include "verilated.h"

#include <cerrno>
#include <csignal>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <poll.h>
#include <set>
#include <sstream>
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
  std::string channelPlanPath;
  std::string peerManifestPath;
  std::vector<std::string> peerExecutables;
  std::string gem5Executable;
  std::string gem5OutputDirectory;
  std::string gem5Configuration;
  std::string projection;
  std::string systemResultPath;
  std::uint64_t ticksPerCycle = 0;
  bool peer = false;
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
    if (argument == "--peer") {
      options.peer = true;
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
    else if (argument == "--channel-plan")
      options.channelPlanPath = value;
    else if (argument == "--peer-manifest")
      options.peerManifestPath = value;
    else if (argument == "--peer-executable")
      options.peerExecutables.push_back(value);
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
  if (options.socketPath.empty() || options.expectedLaunchPath.empty() ||
      options.mappedResultPath.empty() || options.channelPlanPath.empty() ||
      options.ticksPerCycle == 0)
    return false;
  if (options.peer)
    return options.peerManifestPath.empty() &&
           options.peerExecutables.empty() && options.gem5Executable.empty() &&
           options.gem5OutputDirectory.empty() &&
           options.gem5Configuration.empty() && options.projection.empty() &&
           options.systemResultPath.empty();
  return !options.peerManifestPath.empty() && !options.gem5Executable.empty() &&
         !options.gem5OutputDirectory.empty() &&
         !options.gem5Configuration.empty() && !options.projection.empty() &&
         !options.systemResultPath.empty();
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
  if (::bind(server, reinterpret_cast<sockaddr *>(&address), sizeof(address)) !=
          0 ||
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

bool receiveLaunch(int connection, loom::runtime::Gem5BridgeMessage &message) {
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
  if (payloadSize != 0 && !readAll(connection, bytes.data() + headerSize,
                                   static_cast<std::size_t>(payloadSize)))
    return false;
  std::string diagnostic;
  return loom::runtime::decodeGem5BridgeWireMessage(bytes, message, diagnostic);
}

bool sendMessage(int connection,
                 const loom::runtime::Gem5BridgeMessage &message) {
  const std::vector<std::uint8_t> bytes =
      loom::runtime::encodeGem5BridgeWireMessage(message);
  return writeAll(connection, bytes.data(), bytes.size());
}

bool transactMemory(int connection, std::uint64_t sequence,
                    loom::runtime::Gem5BridgeMessageKind kind,
                    loom::runtime::Gem5BridgeMemoryRequest request,
                    loom::runtime::Gem5BridgeMemoryResponse &response) {
  if (!sendMessage(connection,
                   {kind, sequence,
                    loom::runtime::encodeGem5BridgeMemoryRequest(request)}))
    return false;
  loom::runtime::Gem5BridgeMessage message;
  if (!receiveLaunch(connection, message) ||
      message.kind != loom::runtime::Gem5BridgeMessageKind::MemoryResponse ||
      message.sequence != sequence)
    return false;
  std::string diagnostic;
  if (!loom::runtime::decodeGem5BridgeMemoryResponse(message.payload, response,
                                                     diagnostic) ||
      response.requestId != request.requestId || !response.success)
    return false;
  if (request.operation == loom::runtime::Gem5BridgeMemoryOperation::Read)
    return response.data.size() == request.size;
  return response.data.empty();
}

bool readChannelPayload(
    int connection, std::uint64_t sequence,
    const loom::runtime::Gem5SpatialChannelEngineInput &input,
    std::uint64_t ticksPerCycle, std::uint64_t &requestId,
    std::vector<std::uint8_t> &payload) {
  constexpr std::uint64_t maximumPolls = 1'000'000;
  for (std::uint64_t attempt = 0; attempt != maximumPolls; ++attempt) {
    loom::runtime::Gem5BridgeMemoryResponse header;
    if (!transactMemory(connection, sequence,
                        loom::runtime::Gem5BridgeMessageKind::MemoryRequest,
                        {loom::runtime::Gem5BridgeMemoryOperation::Read,
                         attempt == 0 ? 0 : ticksPerCycle,
                         requestId++,
                         input.address,
                         loom::runtime::gem5SpatialChannelBufferHeaderBytes,
                         {}},
                        header))
      return false;
    if (std::all_of(header.data.begin(), header.data.end(),
                    [](std::uint8_t byte) { return byte == 0; }))
      continue;
    std::uint64_t payloadBytes = 0;
    std::string diagnostic;
    if (!loom::runtime::decodeGem5SpatialChannelBufferHeaderPortable(
            header.data.data(), header.data.size(), payloadBytes, diagnostic) ||
        payloadBytes > input.capacityBytes -
                           loom::runtime::gem5SpatialChannelBufferHeaderBytes)
      return false;
    loom::runtime::Gem5BridgeMemoryResponse response;
    if (!transactMemory(
            connection, sequence,
            loom::runtime::Gem5BridgeMessageKind::MemoryRequest,
            {loom::runtime::Gem5BridgeMemoryOperation::Read,
             0,
             requestId++,
             input.address + loom::runtime::gem5SpatialChannelBufferHeaderBytes,
             payloadBytes,
             {}},
            response))
      return false;
    payload = std::move(response.data);
    return true;
  }
  return false;
}

bool extractMappedStream(const std::vector<std::uint8_t> &payload,
                         std::uint64_t observationOrdinal,
                         std::uint32_t tokenBitWidth,
                         std::vector<std::string> &tokens) {
  const std::string text(payload.begin(), payload.end());
  std::istringstream input(text);
  std::string line;
  if (!std::getline(input, line) || line != "loom.mapped_rtl_result 1.0")
    return false;
  bool retired = false;
  bool selected = false;
  while (std::getline(input, line)) {
    if (line == "terminal retired")
      retired = true;
    if (line.rfind("stream ", 0) != 0)
      continue;
    std::istringstream fields(line);
    std::string label;
    std::string termination;
    std::uint64_t ordinal = 0;
    std::uint64_t width = 0;
    std::uint64_t count = 0;
    if (!(fields >> label >> ordinal >> termination >> width >> count) ||
        label != "stream" || termination != "closed")
      return false;
    if (ordinal != observationOrdinal)
      continue;
    if (selected || width != tokenBitWidth || count > payload.size())
      return false;
    tokens.reserve(static_cast<std::size_t>(count));
    for (std::uint64_t token = 0; token != count; ++token) {
      std::string bits;
      if (!(fields >> bits) || bits.size() != width + 1 ||
          bits.front() != 'b' ||
          !std::all_of(bits.begin() + 1, bits.end(),
                       [](char bit) { return bit == '0' || bit == '1'; }))
        return false;
      tokens.push_back(bits.substr(1));
    }
    std::string trailing;
    if (fields >> trailing)
      return false;
    selected = true;
  }
  return retired && selected;
}

bool materializeRuntimeStream(const std::string &mappedResultPath,
                              std::uint64_t ordinal,
                              const std::vector<std::string> &tokens,
                              std::string &path) {
  const std::filesystem::path launchOutput =
      std::filesystem::path(mappedResultPath).parent_path();
  path = (launchOutput / ("runtime-stream-" + std::to_string(ordinal) + ".txt"))
             .generic_string();
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  if (!output)
    return false;
  for (const std::string &token : tokens)
    output << token << '\n';
  output.close();
  return output.good();
}

bool publishChannelOutputs(
    int connection, std::uint64_t sequence,
    const loom::runtime::Gem5SpatialChannelEnginePlan &plan,
    const std::vector<std::uint8_t> &result, std::uint64_t &requestId) {
  for (const loom::runtime::Gem5SpatialChannelEngineOutput &output :
       plan.outputs) {
    if (result.empty() ||
        result.size() > output.capacityBytes -
                            loom::runtime::gem5SpatialChannelBufferHeaderBytes)
      return false;
    loom::runtime::Gem5BridgeMemoryResponse response;
    if (!transactMemory(connection, sequence,
                        loom::runtime::Gem5BridgeMessageKind::ChannelTransfer,
                        {loom::runtime::Gem5BridgeMemoryOperation::Write, 0,
                         requestId++,
                         output.address +
                             loom::runtime::gem5SpatialChannelBufferHeaderBytes,
                         result.size(), result},
                        response))
      return false;
    const auto header =
        loom::runtime::encodeGem5SpatialChannelBufferHeaderPortable(
            result.size());
    if (!transactMemory(
            connection, sequence,
            loom::runtime::Gem5BridgeMessageKind::ChannelTransfer,
            {loom::runtime::Gem5BridgeMemoryOperation::Write, 0, requestId++,
             output.address, header.size(),
             std::vector<std::uint8_t>(header.begin(), header.end())},
            response))
      return false;
  }
  return true;
}

bool sendCompletion(int connection, std::uint64_t sequence, std::uint64_t delay,
                    std::uint32_t status, std::vector<std::uint8_t> result) {
  const loom::runtime::Gem5BridgeCompletion completion{delay, status,
                                                       std::move(result)};
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

struct PeerCommand final {
  std::vector<std::string> arguments;
  std::string socketPath;
};

struct ChildGroup final {
  std::vector<pid_t> children;

  ~ChildGroup() {
    for (pid_t child : children)
      stopChild(child);
  }

  bool waitAll() {
    constexpr unsigned maximumPolls = 500;
    bool success = true;
    for (unsigned poll = 0; poll != maximumPolls && !children.empty(); ++poll) {
      auto child = children.begin();
      while (child != children.end()) {
        int status = 0;
        const pid_t result = ::waitpid(*child, &status, WNOHANG);
        if (result == 0) {
          ++child;
          continue;
        }
        if (result < 0) {
          if (errno == EINTR)
            continue;
          success = false;
        } else {
          success = success && WIFEXITED(status) && WEXITSTATUS(status) == 0;
        }
        child = children.erase(child);
      }
      if (!children.empty())
        ::usleep(10'000);
    }
    if (!children.empty()) {
      success = false;
      for (pid_t child : children)
        stopChild(child);
      children.clear();
    }
    return success;
  }
};

int acceptWhileChildLives(int server, pid_t child, bool &childReaped) {
  childReaped = false;
  for (;;) {
    pollfd descriptor{server, POLLIN, 0};
    const int pollResult = ::poll(&descriptor, 1, 10);
    if (pollResult > 0 && (descriptor.revents & POLLIN) != 0)
      return ::accept(server, nullptr, nullptr);
    if (pollResult < 0 && errno != EINTR)
      return -1;
    if (pollResult > 0 &&
        (descriptor.revents & (POLLERR | POLLHUP | POLLNVAL)) != 0)
      return -1;

    int status = 0;
    const pid_t result = ::waitpid(child, &status, WNOHANG);
    if (result == child) {
      childReaped = true;
      return -1;
    }
    if (result < 0 && errno != EINTR)
      return -1;
  }
}

bool parsePeerManifest(const std::string &path,
                       std::vector<PeerCommand> &commands) {
  std::ifstream input(path);
  if (!input)
    return false;
  std::string line;
  if (!std::getline(input, line) || line != "loom.gem5_rtl_peers 1.0")
    return false;
  bool ended = false;
  while (std::getline(input, line)) {
    if (line == "end") {
      ended = true;
      break;
    }
    PeerCommand command;
    std::size_t begin = 0;
    while (begin <= line.size()) {
      const std::size_t separator = line.find('\t', begin);
      const std::string field = line.substr(begin, separator - begin);
      if (field.empty())
        return false;
      command.arguments.push_back(field);
      if (separator == std::string::npos)
        break;
      begin = separator + 1;
    }
    for (std::size_t index = 0; index + 1 < command.arguments.size(); ++index)
      if (command.arguments[index] == "--socket")
        command.socketPath = command.arguments[index + 1];
    if (command.arguments.empty() || command.socketPath.empty())
      return false;
    commands.push_back(std::move(command));
  }
  return ended && input.peek() == std::char_traits<char>::eof();
}

bool validatePeerExecutables(const std::vector<PeerCommand> &commands,
                             const std::vector<std::string> &executables) {
  if (commands.size() != executables.size())
    return false;
  std::set<std::string> unique;
  for (std::size_t index = 0; index != commands.size(); ++index)
    if (executables[index].empty() || commands[index].arguments.empty() ||
        commands[index].arguments.front() != executables[index] ||
        !unique.insert(executables[index]).second)
      return false;
  return true;
}

bool launchPeers(const std::vector<PeerCommand> &commands,
                 const std::vector<std::string> &verilatorArguments,
                 ChildGroup &children) {
  for (const PeerCommand &command : commands) {
    const pid_t child = ::fork();
    if (child < 0)
      return false;
    if (child == 0) {
      std::vector<char *> arguments;
      arguments.reserve(command.arguments.size() + verilatorArguments.size());
      for (const std::string &argument : command.arguments)
        arguments.push_back(const_cast<char *>(argument.c_str()));
      for (std::size_t index = 1; index != verilatorArguments.size(); ++index)
        arguments.push_back(
            const_cast<char *>(verilatorArguments[index].c_str()));
      arguments.push_back(nullptr);
      ::execv(arguments.front(), arguments.data());
      _exit(127);
    }
    children.children.push_back(child);
  }
  constexpr unsigned maximumSocketPolls = 500;
  for (const PeerCommand &command : commands) {
    unsigned poll = 0;
    while (::access(command.socketPath.c_str(), F_OK) != 0 &&
           poll != maximumSocketPolls) {
      ::usleep(10'000);
      ++poll;
    }
    if (poll == maximumSocketPolls)
      return false;
  }
  return true;
}

} // namespace

int main(int argc, char **argv) {
  Options options;
  std::vector<std::string> verilatorArguments;
  if (!parseArguments(argc, argv, options, verilatorArguments)) {
    std::cerr << "invalid gem5 RTL engine arguments\n";
    return 2;
  }

  std::vector<std::uint8_t> expectedLaunch;
  if (!readFile(options.expectedLaunchPath, expectedLaunch)) {
    std::cerr << "cannot read the expected Spatial launch\n";
    return 3;
  }
  std::vector<std::uint8_t> channelPlanBytes;
  if (!readFile(options.channelPlanPath, channelPlanBytes)) {
    std::cerr << "cannot read the Spatial channel plan\n";
    return 3;
  }
  loom::runtime::Gem5SpatialChannelEnginePlan channelPlan;
  std::string channelDiagnostic;
  if (!loom::runtime::decodeGem5SpatialChannelEnginePlan(
          std::string_view(
              reinterpret_cast<const char *>(channelPlanBytes.data()),
              channelPlanBytes.size()),
          channelPlan, channelDiagnostic)) {
    std::cerr << channelDiagnostic << '\n';
    return 3;
  }
  const int server = openServer(options.socketPath);
  if (server < 0) {
    std::cerr << "cannot publish the bridge socket\n";
    return 4;
  }
  std::vector<PeerCommand> peerCommands;
  ChildGroup peerChildren;
  if (!options.peer &&
      (!parsePeerManifest(options.peerManifestPath, peerCommands) ||
       !validatePeerExecutables(peerCommands, options.peerExecutables) ||
       !launchPeers(peerCommands, verilatorArguments, peerChildren))) {
    ::close(server);
    ::unlink(options.socketPath.c_str());
    std::cerr << "cannot launch the peer RTL engines\n";
    return 5;
  }
  const pid_t gem5 = options.peer ? 0 : launchGem5(options);
  if (!options.peer && gem5 < 0) {
    ::close(server);
    ::unlink(options.socketPath.c_str());
    std::cerr << "cannot launch gem5\n";
    return 5;
  }
  bool gem5Reaped = false;
  const int connection = options.peer
                             ? ::accept(server, nullptr, nullptr)
                             : acceptWhileChildLives(server, gem5, gem5Reaped);
  ::close(server);
  if (connection < 0) {
    if (!options.peer && !gem5Reaped)
      stopChild(gem5);
    ::unlink(options.socketPath.c_str());
    if (gem5Reaped)
      std::cerr << "gem5 exited before opening the bridge connection\n";
    else
      std::cerr << "cannot accept the bridge connection\n";
    return 6;
  }
  loom::runtime::Gem5BridgeMessage launch;
  if (!receiveLaunch(connection, launch) ||
      launch.kind != loom::runtime::Gem5BridgeMessageKind::SpatialLaunch ||
      launch.sequence != 0 || launch.payload != expectedLaunch) {
    ::close(connection);
    if (!options.peer)
      stopChild(gem5);
    ::unlink(options.socketPath.c_str());
    std::cerr << "bridge launch differs from the exact Deployment\n";
    return 7;
  }

  std::uint64_t requestId = 0;
  for (const loom::runtime::Gem5SpatialChannelEngineInput &input :
       channelPlan.inputs) {
    std::vector<std::uint8_t> payload;
    if (!readChannelPayload(connection, launch.sequence, input,
                            options.ticksPerCycle, requestId, payload)) {
      ::close(connection);
      if (!options.peer)
        stopChild(gem5);
      ::unlink(options.socketPath.c_str());
      std::cerr << "could not receive a Spatial channel input\n";
      return 8;
    }
    std::vector<std::string> tokens;
    if (!extractMappedStream(payload, input.producerObservationOrdinal,
                             input.tokenBitWidth, tokens)) {
      ::close(connection);
      if (!options.peer)
        stopChild(gem5);
      ::unlink(options.socketPath.c_str());
      std::cerr << "Spatial channel payload is not a retired mapped stream\n";
      return 8;
    }
    std::string path;
    if (!materializeRuntimeStream(options.mappedResultPath,
                                  input.consumerStreamInputOrdinal, tokens,
                                  path)) {
      ::close(connection);
      if (!options.peer)
        stopChild(gem5);
      ::unlink(options.socketPath.c_str());
      std::cerr << "could not materialize a Spatial channel input\n";
      return 8;
    }
    verilatorArguments.push_back(
        "+LOOM_STREAM_INPUT_" +
        std::to_string(input.consumerStreamInputOrdinal) + "=" + path);
  }

  std::vector<char *> verilatorPointers;
  for (std::string &argument : verilatorArguments)
    verilatorPointers.push_back(argument.data());
  VerilatedContext context;
  context.commandArgs(static_cast<int>(verilatorPointers.size()),
                      verilatorPointers.data());

  Vloom_mapped_rtl_testbench top{&context};
  top.eval();
  while (!context.gotFinish()) {
    if (!top.eventsPending()) {
      ::close(connection);
      if (!options.peer)
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
    if (!options.peer)
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
    if (!options.peer)
      stopChild(gem5);
    ::unlink(options.socketPath.c_str());
    std::cerr << "RTL harness published reversed progress\n";
    return 10;
  }
  const std::uint64_t cycles = retired ? retirementCycle - launchCycle : 0;
  if (cycles >
      std::numeric_limits<std::uint64_t>::max() / options.ticksPerCycle) {
    ::close(connection);
    if (!options.peer)
      stopChild(gem5);
    ::unlink(options.socketPath.c_str());
    std::cerr << "RTL completion delay overflows gem5 ticks\n";
    return 11;
  }
  if (!publishChannelOutputs(connection, launch.sequence, channelPlan, result,
                             requestId)) {
    ::close(connection);
    if (!options.peer)
      stopChild(gem5);
    ::unlink(options.socketPath.c_str());
    std::cerr << "could not publish a Spatial channel output\n";
    return 12;
  }
  if (!sendCompletion(connection, launch.sequence,
                      cycles * options.ticksPerCycle, retired ? 0U : 1U,
                      std::move(result))) {
    ::close(connection);
    if (!options.peer)
      stopChild(gem5);
    ::unlink(options.socketPath.c_str());
    std::cerr << "cannot send the RTL completion\n";
    return 12;
  }
  ::close(connection);
  ::unlink(options.socketPath.c_str());
  if (!options.peer && !waitForChild(gem5)) {
    std::cerr << "gem5 did not complete successfully\n";
    return 13;
  }
  if (!options.peer && !peerChildren.waitAll()) {
    std::cerr << "a peer RTL engine did not complete successfully\n";
    return 14;
  }
  return 0;
}
