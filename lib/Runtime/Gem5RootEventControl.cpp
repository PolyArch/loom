#include "Runtime/Gem5RootEventControl.h"

#include "Common/ArtifactStore.h"
#include "Dataflow/IR/DataflowCanonicalEntity.h"
#include "Dataflow/IR/DataflowEventDerivation.h"
#include "Mapping/Artifact/SystemMappingArtifact.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <fcntl.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include <array>
#include <atomic>
#include <cerrno>
#include <cstring>
#include <mutex>
#include <optional>
#include <thread>
#include <utility>

namespace loom::runtime {
namespace {

/// Bounded wait between stop-flag checks while the controller blocks on the
/// device. The device itself bounds every socket operation by
/// `gem5RootEventControlTimeoutMilliseconds`.
constexpr int kPollSliceMilliseconds = 100;

llvm::Error reject(Gem5RootEventControlErrorReason reason,
                   const llvm::Twine &message) {
  return llvm::make_error<Gem5RootEventControlError>(reason, message.str());
}

void writeU32(std::uint8_t *output, std::uint32_t value) {
  for (int shift = 24; shift >= 0; shift -= 8)
    *output++ = static_cast<std::uint8_t>(value >> shift);
}

void writeU64(std::uint8_t *output, std::uint64_t value) {
  for (int shift = 56; shift >= 0; shift -= 8)
    *output++ = static_cast<std::uint8_t>(value >> shift);
}

std::uint32_t readU32(const std::uint8_t *input) {
  std::uint32_t value = 0;
  for (std::size_t index = 0; index != 4; ++index)
    value = (value << 8) | input[index];
  return value;
}

std::uint64_t readU64(const std::uint8_t *input) {
  std::uint64_t value = 0;
  for (std::size_t index = 0; index != 8; ++index)
    value = (value << 8) | input[index];
  return value;
}

enum class ReadOutcome : std::uint8_t { Complete, Closed, Stopped, Failed };

ReadOutcome readExact(int descriptor, std::uint8_t *buffer, std::size_t size,
                      const std::atomic<bool> &stop) {
  std::size_t received = 0;
  while (received != size) {
    if (stop.load())
      return ReadOutcome::Stopped;
    pollfd waiting{descriptor, POLLIN, 0};
    const int ready = ::poll(&waiting, 1, kPollSliceMilliseconds);
    if (ready < 0) {
      if (errno == EINTR)
        continue;
      return ReadOutcome::Failed;
    }
    if (ready == 0)
      continue;
    const ssize_t count =
        ::read(descriptor, buffer + received, size - received);
    if (count == 0)
      return received == 0 ? ReadOutcome::Closed : ReadOutcome::Failed;
    if (count < 0) {
      if (errno == EINTR || errno == EAGAIN)
        continue;
      return ReadOutcome::Failed;
    }
    received += static_cast<std::size_t>(count);
  }
  return ReadOutcome::Complete;
}

bool writeExact(int descriptor, const std::uint8_t *buffer, std::size_t size) {
  std::size_t sent = 0;
  while (sent != size) {
    const ssize_t count = ::write(descriptor, buffer + sent, size - sent);
    if (count < 0) {
      if (errno == EINTR || errno == EAGAIN)
        continue;
      return false;
    }
    sent += static_cast<std::size_t>(count);
  }
  return true;
}

} // namespace

char Gem5RootEventControlError::ID = 0;

void Gem5RootEventControlError::log(llvm::raw_ostream &stream) const {
  stream << message_;
}

std::error_code Gem5RootEventControlError::convertToErrorCode() const {
  return llvm::inconvertibleErrorCode();
}

llvm::StringRef gem5RootEventControlErrorReasonSpelling(
    Gem5RootEventControlErrorReason reason) {
  switch (reason) {
  case Gem5RootEventControlErrorReason::EndpointWithoutDeployment:
    return "endpoint_without_deployment";
  case Gem5RootEventControlErrorReason::NonTerminalEdge:
    return "non_terminal_edge";
  case Gem5RootEventControlErrorReason::EndpointBoundExceeded:
    return "endpoint_bound_exceeded";
  case Gem5RootEventControlErrorReason::SocketUnavailable:
    return "socket_unavailable";
  case Gem5RootEventControlErrorReason::ProtocolFailure:
    return "protocol_failure";
  case Gem5RootEventControlErrorReason::ControllerRejected:
    return "controller_rejected";
  }
  llvm_unreachable("unknown gem5 root event control error reason");
}

llvm::Expected<Gem5RootEventEndpointTable>
deriveGem5RootEventEndpointTable(const pnr::ResourceTimeTransitionGraph &graph,
                                 const ArtifactStore &artifacts) {
  if (llvm::Error error = pnr::validateResourceTimeTransitionGraph(graph))
    return std::move(error);
  if (!graph.entry.deployment)
    return reject(Gem5RootEventControlErrorReason::EndpointWithoutDeployment,
                  "transition graph entry has no Deployment");
  auto mapping = mapping::importSystemMapping(graph.entry.mapping, artifacts);
  if (!mapping)
    return mapping.takeError();
  const std::size_t rootCount =
      mapping->view().executionBindings().rootThreadLaunches().size();
  Gem5RootEventEndpointTable table{mapping->view().dataflowIdentity(),
                                   {*graph.entry.deployment}};
  for (const pnr::ResourceTimeTransitionEndpointReference &endpoint :
       graph.endpoints) {
    if (endpoint == graph.entry)
      continue;
    if (!endpoint.deployment)
      return reject(Gem5RootEventControlErrorReason::EndpointWithoutDeployment,
                    "transition graph endpoint has no Deployment");
    table.deployments.push_back(*endpoint.deployment);
  }
  if (table.deployments.size() > gem5MaximumDynamicSpatialInvocations)
    return reject(Gem5RootEventControlErrorReason::EndpointBoundExceeded,
                  "transition graph exceeds the gem5 endpoint bound");
  for (const pnr::ResourceTimeTransition &transition : graph.transitions) {
    if (!transition.afterActive.empty() ||
        transition.completedBefore.size() + 1 != rootCount)
      return reject(Gem5RootEventControlErrorReason::NonTerminalEdge,
                    "gem5 root event control admits only terminal edges: a "
                    "child endpoint has no dispatch targets, so a region "
                    "that continues under the child cannot be launched");
  }
  return table;
}

class Gem5RootEventController::Impl final {
public:
  Impl(int directory, std::string socketName, int listener,
       ArtifactIdentity dataflow, Gem5RootEventDecisionCallback callback)
      : directory_(directory), socketName_(std::move(socketName)),
        listener_(listener), dataflow_(dataflow),
        callback_(std::move(callback)) {}

  ~Impl() { close(); }

  void start() {
    thread_ = std::thread([this] { serve(); });
  }

  llvm::Expected<std::vector<Gem5RootEventAcknowledgement>> finish() {
    stop_.store(true);
    if (thread_.joinable())
      thread_.join();
    close();
    std::lock_guard<std::mutex> guard(mutex_);
    if (failure_)
      return reject(failure_->first, failure_->second);
    return acknowledgements_;
  }

private:
  void fail(Gem5RootEventControlErrorReason reason, std::string message) {
    std::lock_guard<std::mutex> guard(mutex_);
    if (!failure_)
      failure_.emplace(reason, std::move(message));
  }

  std::optional<int> acceptConnection() {
    while (!stop_.load()) {
      pollfd waiting{listener_, POLLIN, 0};
      const int ready = ::poll(&waiting, 1, kPollSliceMilliseconds);
      if (ready < 0) {
        if (errno == EINTR)
          continue;
        fail(Gem5RootEventControlErrorReason::SocketUnavailable,
             std::string("root event control accept failed: ") +
                 std::strerror(errno));
        return std::nullopt;
      }
      if (ready == 0)
        continue;
      const int connection =
          ::accept4(listener_, nullptr, nullptr, SOCK_CLOEXEC);
      if (connection < 0) {
        if (errno == EINTR || errno == EAGAIN)
          continue;
        fail(Gem5RootEventControlErrorReason::SocketUnavailable,
             std::string("root event control accept failed: ") +
                 std::strerror(errno));
        return std::nullopt;
      }
      return connection;
    }
    return std::nullopt;
  }

  void serve() {
    const std::optional<int> connection = acceptConnection();
    if (!connection)
      return;
    std::uint64_t expectedGeneration = 1;
    while (!stop_.load()) {
      std::array<std::uint8_t, gem5RootEventControlRequestBytes> request{};
      const ReadOutcome outcome =
          readExact(*connection, request.data(), request.size(), stop_);
      if (outcome == ReadOutcome::Closed || outcome == ReadOutcome::Stopped)
        break;
      if (outcome == ReadOutcome::Failed) {
        fail(Gem5RootEventControlErrorReason::ProtocolFailure,
             "root event control request was truncated");
        break;
      }
      const std::uint64_t generation = readU64(request.data() + 4);
      const std::uint64_t entity = readU64(request.data() + 12);
      const std::uint64_t occurrence = readU64(request.data() + 20);
      const std::uint32_t action = readU32(request.data() + 28);
      const std::uint64_t tick = readU64(request.data() + 32);
      const std::uint64_t delta = readU64(request.data() + 40);
      if (readU32(request.data()) != gem5RootEventControlRequestMagic ||
          generation != expectedGeneration || occurrence == 0 ||
          action > static_cast<std::uint32_t>(
                       Gem5RootLifecycleAction::Completion)) {
        fail(Gem5RootEventControlErrorReason::ProtocolFailure,
             "root event control request is not canonical");
        break;
      }
      const dataflow::RootThreadLaunchRef root{
          dataflow_, dataflow::RootThreadLaunchId(entity)};
      const bool start =
          action == static_cast<std::uint32_t>(Gem5RootLifecycleAction::Start);
      const sim::SystemRootLifecycleObservation observation{
          start ? dataflow::rootThreadStartEventFamily(root)
                : dataflow::rootThreadCompletionEventFamily(root),
          occurrence,
          {tick, delta}};
      llvm::Expected<Gem5RootEventDecision> decided = callback_(observation);
      Gem5RootEventDecision decision{Gem5RootEventControlDecision::Reject, 0};
      if (decided)
        decision = *decided;
      else
        fail(Gem5RootEventControlErrorReason::ControllerRejected,
             llvm::toString(decided.takeError()));
      std::array<std::uint8_t, gem5RootEventControlAckBytes> acknowledgement{};
      writeU32(acknowledgement.data(), gem5RootEventControlAckMagic);
      writeU64(acknowledgement.data() + 4, generation);
      writeU32(acknowledgement.data() + 12,
               static_cast<std::uint32_t>(decision.decision));
      writeU64(acknowledgement.data() + 16, decision.endpoint);
      if (!writeExact(*connection, acknowledgement.data(),
                      acknowledgement.size())) {
        fail(Gem5RootEventControlErrorReason::ProtocolFailure,
             "root event control acknowledgement was not delivered");
        break;
      }
      if (!decided)
        break;
      {
        std::lock_guard<std::mutex> guard(mutex_);
        acknowledgements_.push_back({generation, observation, decision});
      }
      ++expectedGeneration;
    }
    ::close(*connection);
  }

  void close() {
    if (listener_ >= 0) {
      ::close(listener_);
      listener_ = -1;
    }
    if (directory_ >= 0) {
      ::unlinkat(directory_, socketName_.c_str(), 0);
      ::close(directory_);
      directory_ = -1;
    }
  }

  int directory_;
  std::string socketName_;
  int listener_;
  ArtifactIdentity dataflow_;
  Gem5RootEventDecisionCallback callback_;
  std::thread thread_;
  std::atomic<bool> stop_{false};
  std::mutex mutex_;
  std::vector<Gem5RootEventAcknowledgement> acknowledgements_;
  std::optional<std::pair<Gem5RootEventControlErrorReason, std::string>>
      failure_;
};

Gem5RootEventController::Gem5RootEventController(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

Gem5RootEventController::Gem5RootEventController(
    Gem5RootEventController &&) noexcept = default;
Gem5RootEventController &Gem5RootEventController::operator=(
    Gem5RootEventController &&) noexcept = default;

Gem5RootEventController::~Gem5RootEventController() {
  if (impl_)
    llvm::consumeError(impl_->finish().takeError());
}

llvm::Expected<Gem5RootEventController>
Gem5RootEventController::listen(llvm::StringRef bundleRoot,
                                ArtifactIdentity dataflow,
                                Gem5RootEventDecisionCallback callback) {
  const llvm::StringRef socketPath(gem5RootEventControlSocketPath);
  llvm::SmallString<256> directoryPath(bundleRoot);
  llvm::sys::path::append(directoryPath,
                          llvm::sys::path::parent_path(socketPath));
  const std::string socketName = llvm::sys::path::filename(socketPath).str();
  // The bundle root may exceed the AF_UNIX path bound, so the socket is bound
  // through its directory descriptor while the device connects with the
  // bundle-relative path from the launcher's working directory.
  const int directory =
      ::open(directoryPath.c_str(), O_PATH | O_DIRECTORY | O_CLOEXEC);
  if (directory < 0)
    return reject(Gem5RootEventControlErrorReason::SocketUnavailable,
                  "cannot open the bundle output directory for the root "
                  "event control socket: " +
                      llvm::Twine(std::strerror(errno)));
  ::unlinkat(directory, socketName.c_str(), 0);
  const int listener = ::socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
  if (listener < 0) {
    ::close(directory);
    return reject(Gem5RootEventControlErrorReason::SocketUnavailable,
                  "cannot create the root event control socket: " +
                      llvm::Twine(std::strerror(errno)));
  }
  auto impl = std::make_unique<Impl>(directory, socketName, listener,
                                     dataflow, std::move(callback));
  const std::string bindPath =
      "/proc/self/fd/" + std::to_string(directory) + "/" + socketName;
  sockaddr_un address{};
  address.sun_family = AF_UNIX;
  if (bindPath.size() >= sizeof(address.sun_path))
    return reject(Gem5RootEventControlErrorReason::SocketUnavailable,
                  "root event control socket path exceeds the AF_UNIX bound");
  std::memcpy(address.sun_path, bindPath.c_str(), bindPath.size() + 1);
  if (::bind(listener, reinterpret_cast<const sockaddr *>(&address),
             sizeof(address)) != 0 ||
      ::listen(listener, 1) != 0)
    return reject(Gem5RootEventControlErrorReason::SocketUnavailable,
                  "cannot listen on the root event control socket: " +
                      llvm::Twine(std::strerror(errno)));
  impl->start();
  return Gem5RootEventController(std::move(impl));
}

llvm::Expected<std::vector<Gem5RootEventAcknowledgement>>
Gem5RootEventController::finish() {
  if (!impl_)
    return reject(Gem5RootEventControlErrorReason::SocketUnavailable,
                  "root event controller was already finished");
  auto records = impl_->finish();
  impl_.reset();
  return records;
}

} // namespace loom::runtime
