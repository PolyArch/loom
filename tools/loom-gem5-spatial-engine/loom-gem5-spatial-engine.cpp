#include "Common/ArtifactStore.h"
#include "Common/ArtifactText.h"
#include "Common/ComponentViewDigest.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/Identity/FabricRefBytes.h"
#include "Mapping/Artifact/MappingArtifact.h"
#include "Runtime/Gem5BridgeWire.h"
#include "Runtime/Gem5SpatialChannel.h"
#include "Runtime/OrderedChannelSequence.h"
#include "Simulator/CGRAAdmission.h"
#include "Simulator/CGRASimulator.h"
#include "Simulator/DFGSimulator.h"
#include "Simulator/SimulationArtifacts.h"
#include "Simulator/SimulationExecution.h"
#include "Simulator/SpatialChannelWire.h"
#include "Simulator/SpatialInvocation.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <optional>
#include <poll.h>
#include <set>
#include <string>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <utility>
#include <vector>

namespace {

llvm::cl::opt<std::string>
    artifactStorePath("artifact-store",
                      llvm::cl::desc("Invocation package ArtifactStore"),
                      llvm::cl::Required);
llvm::cl::opt<std::string>
    socketPath("socket", llvm::cl::desc("Invocation-local bridge socket"),
               llvm::cl::Required);
llvm::cl::list<std::string>
    expectedLaunchPaths("expected-launch",
                        llvm::cl::desc("Exact launch payload"),
                        llvm::cl::OneOrMore);
llvm::cl::list<std::string>
    workloadIdentities("workload",
                       llvm::cl::desc("Spatial workload ArtifactIdentity"),
                       llvm::cl::OneOrMore);
llvm::cl::list<std::string> runtimeInputIdentities(
    "runtime-input",
    llvm::cl::desc("Spatial runtime input ArtifactIdentity or 'none'"),
    llvm::cl::OneOrMore);
llvm::cl::list<std::string> channelProjectionPaths(
    "channel-projection",
    llvm::cl::desc("Invocation-local Spatial channel projection"),
    llvm::cl::OneOrMore);
llvm::cl::list<std::uint64_t> bridgeOrdinals(
    "bridge-ordinal",
    llvm::cl::desc("System bridge session ordinal for the preceding entry"),
    llvm::cl::OneOrMore);
llvm::cl::opt<std::string>
    dataflowIdentity("dataflow",
                     llvm::cl::desc("Canonical Dataflow ArtifactIdentity"),
                     llvm::cl::Required);
llvm::cl::list<std::string>
    fabricIdentities("fabric", llvm::cl::desc("Fabric ArtifactIdentity"));
llvm::cl::list<std::string>
    spatialMappingIdentities("spatial-mapping",
                             llvm::cl::desc("SpatialMapping ArtifactIdentity"));
llvm::cl::opt<std::uint64_t>
    maximumWork("maximum-work", llvm::cl::desc("Engine semantic work limit"),
                llvm::cl::init(100000));
llvm::cl::opt<std::uint64_t>
    ticksPerCycle("ticks-per-cycle",
                  llvm::cl::desc("gem5 ticks per Spatial cycle"),
                  llvm::cl::init(1000));
llvm::cl::opt<std::uint64_t> maximumInvocations(
    "maximum-invocations",
    llvm::cl::desc("Maximum dynamic invocations in this engine session"),
    llvm::cl::init(4096));
llvm::cl::opt<std::uint64_t> bridgeCount(
    "bridge-count",
    llvm::cl::desc("Number of bridge connections sharing this System session"),
    llvm::cl::init(1));

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      "gem5_spatial_engine_invalid: " + message);
}

int report(llvm::Error error) {
  llvm::errs() << llvm::toString(std::move(error)) << '\n';
  return 1;
}

#if defined(LOOM_GEM5_SPATIAL_ENGINE_DFG)
using PreparedSpatialExecution = loom::sim::PreparedDfgExecution;
#else
using PreparedSpatialExecution = loom::sim::PreparedCgraExecution;
#endif

struct SpatialSessionEntry final {
  loom::sim::ImportedSpatialSimulationWorkload workload;
  std::optional<loom::sim::CanonicalSimulationRuntimeInput> staticRuntime;
  loom::runtime::Gem5SpatialChannelProjection channels;
  std::vector<std::uint8_t> expectedLaunch;
  std::uint64_t bridgeOrdinal = 0;
  std::uint64_t sessionEntryOrdinal = 0;
  std::size_t preparedOrdinal = 0;
};

struct ChannelReservation final {
  std::uint64_t address = 0;
  std::uint32_t consumerOrdinal = 0;
  std::uint64_t sequence = 0;
};

struct ChannelSequenceState final {
  loom::runtime::OrderedChannelSequence sequence;
  std::map<std::string, std::uint32_t> consumerOrdinals;
};

llvm::Expected<loom::ArtifactRootReference>
root(llvm::StringRef text, const loom::ArtifactSchemaDescriptor &schema) {
  auto identity = loom::parseArtifactIdentityHex(text);
  if (!identity)
    return identity.takeError();
  return loom::ArtifactRootReference{schema.identity.str(), schema.version,
                                     std::move(*identity)};
}

llvm::Expected<std::vector<std::uint8_t>> readFile(llvm::StringRef path) {
  auto buffer = llvm::MemoryBuffer::getFile(path, false, false);
  if (!buffer)
    return invalid("cannot read '" + path +
                   "': " + buffer.getError().message());
  const llvm::StringRef contents = (*buffer)->getBuffer();
  return std::vector<std::uint8_t>(contents.bytes_begin(),
                                   contents.bytes_end());
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

llvm::Expected<int> openServer() {
  if (socketPath.size() >= sizeof(sockaddr_un::sun_path))
    return invalid("socket path is too long");
  const int server = ::socket(AF_UNIX, SOCK_STREAM, 0);
  if (server < 0)
    return invalid("cannot create the bridge socket");
  ::unlink(socketPath.c_str());
  sockaddr_un address{};
  address.sun_family = AF_UNIX;
  std::memcpy(address.sun_path, socketPath.c_str(), socketPath.size() + 1);
  if (::bind(server, reinterpret_cast<sockaddr *>(&address), sizeof(address)) !=
          0 ||
      ::listen(server, static_cast<int>(std::min<std::uint64_t>(
                           bridgeCount, std::numeric_limits<int>::max()))) !=
          0) {
    ::close(server);
    return invalid("cannot publish the bridge socket");
  }
  return server;
}

llvm::Expected<loom::runtime::Gem5BridgeMessage> readMessage(int connection) {
  std::vector<std::uint8_t> bytes(loom::runtime::gem5BridgeWireHeaderBytes);
  if (!readAll(connection, bytes.data(), bytes.size()))
    return invalid("bridge disconnected before the message header");
  const std::uint64_t payloadSize =
      loom::runtime::detail::readGem5BridgeU64(bytes.data() + 16);
  if (payloadSize > std::numeric_limits<std::size_t>::max() ||
      payloadSize > 64ULL * 1024ULL * 1024ULL)
    return invalid("bridge message exceeds the engine limit");
  const std::size_t headerSize = bytes.size();
  bytes.resize(headerSize + static_cast<std::size_t>(payloadSize));
  if (payloadSize != 0 && !readAll(connection, bytes.data() + headerSize,
                                   static_cast<std::size_t>(payloadSize)))
    return invalid("bridge disconnected before the message payload");
  loom::runtime::Gem5BridgeMessage message;
  std::string diagnostic;
  if (!loom::runtime::decodeGem5BridgeWireMessage(bytes, message, diagnostic))
    return invalid(diagnostic);
  return message;
}

llvm::Error writeMessage(int connection,
                         const loom::runtime::Gem5BridgeMessage &message) {
  const std::vector<std::uint8_t> bytes =
      loom::runtime::encodeGem5BridgeWireMessage(message);
  if (!writeAll(connection, bytes.data(), bytes.size()))
    return invalid("cannot write the bridge completion");
  return llvm::Error::success();
}

llvm::Expected<loom::runtime::Gem5BridgeMemoryResponse>
transactMemory(int connection, std::uint64_t sequence,
               loom::runtime::Gem5BridgeMessageKind kind,
               loom::runtime::Gem5BridgeMemoryRequest request) {
  if (llvm::Error error = writeMessage(
          connection, {kind, sequence,
                       loom::runtime::encodeGem5BridgeMemoryRequest(request)}))
    return std::move(error);
  auto message = readMessage(connection);
  if (!message)
    return message.takeError();
  if (message->kind != loom::runtime::Gem5BridgeMessageKind::MemoryResponse ||
      message->sequence != sequence)
    return invalid("bridge returned the wrong memory response envelope");
  loom::runtime::Gem5BridgeMemoryResponse response;
  std::string diagnostic;
  if (!loom::runtime::decodeGem5BridgeMemoryResponse(message->payload, response,
                                                     diagnostic))
    return invalid(diagnostic);
  if (response.requestId != request.requestId || !response.success)
    return invalid("bridge memory response does not match its request");
  if (request.operation == loom::runtime::Gem5BridgeMemoryOperation::Read &&
      response.data.size() != request.size)
    return invalid("bridge memory read returned the wrong byte count");
  if (request.operation == loom::runtime::Gem5BridgeMemoryOperation::Write &&
      !response.data.empty())
    return invalid("bridge memory write returned unexpected data");
  return response;
}

llvm::Expected<std::uint64_t>
spatialCoordinateTicks(const loom::sim::SpatialEventCoordinate &coordinate) {
  using u128 = unsigned __int128;
  const u128 scaled = static_cast<u128>(coordinate.referenceCycle.numerator()) *
                      static_cast<std::uint64_t>(ticksPerCycle);
  const std::uint64_t denominator = coordinate.referenceCycle.denominator();
  if (scaled % denominator != 0 ||
      scaled / denominator > std::numeric_limits<std::uint64_t>::max())
    return invalid("Spatial service coordinate is not an integral gem5 tick");
  return static_cast<std::uint64_t>(scaled / denominator);
}

class Gem5CgraExternalMemoryProvider final
    : public loom::sim::CgraExternalMemoryProvider {
public:
  Gem5CgraExternalMemoryProvider(
      int connection, std::uint64_t sequence,
      const loom::runtime::SpatialInvocationWire &invocation,
      std::uint64_t &requestId)
      : connection_(connection), sequence_(sequence), invocation_(&invocation),
        requestId_(&requestId) {}

  llvm::Expected<loom::sim::CgraExternalMemoryResponse>
  transact(const loom::sim::CgraExternalMemoryRequest &request) override {
    if (request.elements.empty())
      return invalid("CGRA external memory request has no active element");
    if (request.objectOrdinal >= invocation_->memoryObjects.size())
      return invalid("CGRA external memory request names no guest object");
    if (lastCoordinate_ && loom::sim::compareSpatialEventCoordinates(
                               *lastCoordinate_, request.readyCoordinate) > 0)
      return invalid("CGRA external memory requests are not time ordered");
    auto readyTick = spatialCoordinateTicks(request.readyCoordinate);
    if (!readyTick)
      return readyTick.takeError();
    std::uint64_t priorTick = 0;
    if (lastCoordinate_) {
      auto projected = spatialCoordinateTicks(*lastCoordinate_);
      if (!projected)
        return projected.takeError();
      priorTick = *projected;
    }
    if (*readyTick < priorTick)
      return invalid("CGRA external memory tick projection moved backward");
    const std::uint64_t initialDelay = *readyTick - priorTick;

    const loom::runtime::SpatialInvocationMemoryObject &object =
        invocation_->memoryObjects[request.objectOrdinal];
    loom::sim::CgraExternalMemoryResponse result;
    if (request.operation == loom::sim::CgraExternalMemoryOperation::Read)
      result.readData.reserve(request.elements.size());
    for (std::size_t ordinal = 0; ordinal != request.elements.size();
         ++ordinal) {
      const loom::sim::CgraExternalMemoryElement &element =
          request.elements[ordinal];
      if (element.byteCount == 0 ||
          element.byteOffset > object.initialBytes.size() ||
          element.byteCount > object.initialBytes.size() - element.byteOffset ||
          object.address >
              std::numeric_limits<std::uint64_t>::max() - element.byteOffset)
        return invalid("CGRA external memory element exceeds its guest object");
      const bool write =
          request.operation == loom::sim::CgraExternalMemoryOperation::Write;
      if ((write && element.writeData.size() != element.byteCount) ||
          (!write && !element.writeData.empty()))
        return invalid("CGRA external memory element has the wrong payload");
      loom::runtime::Gem5BridgeMemoryRequest bridgeRequest{
          write ? loom::runtime::Gem5BridgeMemoryOperation::Write
                : loom::runtime::Gem5BridgeMemoryOperation::Read,
          ordinal == 0 ? initialDelay : 0,
          (*requestId_)++,
          object.address + element.byteOffset,
          element.byteCount,
          element.writeData};
      auto response =
          transactMemory(connection_, sequence_,
                         loom::runtime::Gem5BridgeMessageKind::MemoryRequest,
                         std::move(bridgeRequest));
      if (!response)
        return response.takeError();
      if (write) {
        for (std::size_t byte = 0; byte != element.writeData.size(); ++byte)
          externallyCommittedBytes_[object.address + element.byteOffset +
                                    byte] = element.writeData[byte];
      } else {
        result.readData.push_back(std::move(response->data));
      }
    }
    lastCoordinate_ = request.readyCoordinate;
    return result;
  }

  std::optional<loom::sim::SpatialEventCoordinate> lastCoordinate() const {
    return lastCoordinate_;
  }

  std::vector<loom::sim::SpatialInvocationMemoryWrite> retainUncommittedWrites(
      llvm::ArrayRef<loom::sim::SpatialInvocationMemoryWrite> writes) const {
    struct Interval final {
      std::uint64_t begin = 0;
      std::uint64_t end = 0;
    };
    std::vector<Interval> resultDestinations;
    resultDestinations.reserve(invocation_->results.size());
    for (const auto &destination : invocation_->results) {
      const std::uint64_t byteCount =
          (static_cast<std::uint64_t>(destination.bitCount) + 7) / 8;
      resultDestinations.push_back(
          {destination.address, destination.address + byteCount});
    }
    const auto isResultDestination = [&](std::uint64_t address) {
      return llvm::any_of(resultDestinations, [&](const Interval &interval) {
        return interval.begin <= address && address < interval.end;
      });
    };

    std::vector<loom::sim::SpatialInvocationMemoryWrite> retained;
    for (const loom::sim::SpatialInvocationMemoryWrite &write : writes) {
      std::optional<loom::sim::SpatialInvocationMemoryWrite> run;
      for (std::size_t ordinal = 0; ordinal != write.bytes.size(); ++ordinal) {
        const std::uint64_t address = write.address + ordinal;
        auto committed = externallyCommittedBytes_.find(address);
        const bool keep = isResultDestination(address) ||
                          committed == externallyCommittedBytes_.end() ||
                          committed->second != write.bytes[ordinal];
        if (!keep) {
          if (run) {
            retained.push_back(std::move(*run));
            run.reset();
          }
          continue;
        }
        if (!run)
          run = loom::sim::SpatialInvocationMemoryWrite{address, {}};
        run->bytes.push_back(write.bytes[ordinal]);
      }
      if (run)
        retained.push_back(std::move(*run));
    }
    return retained;
  }

private:
  int connection_ = -1;
  std::uint64_t sequence_ = 0;
  const loom::runtime::SpatialInvocationWire *invocation_ = nullptr;
  std::uint64_t *requestId_ = nullptr;
  std::optional<loom::sim::SpatialEventCoordinate> lastCoordinate_;
  std::map<std::uint64_t, std::uint8_t> externallyCommittedBytes_;
};

void appendChannelKeyU64(std::string &key, std::uint64_t value) {
  for (unsigned byte = 0; byte != 8; ++byte)
    key.push_back(static_cast<char>(value >> (byte * 8)));
}

llvm::Expected<std::string>
channelConsumerKey(const loom::sim::ImportedSpatialSimulationWorkload &workload,
                   const loom::runtime::Gem5SpatialChannelInput &input) {
  const auto *spatial = workload.workload.spatial();
  if (!spatial)
    return invalid("channel consumer key lost its Spatial workload");
  std::string key;
  appendChannelKeyU64(key, input.address);
  appendChannelKeyU64(key, spatial->launchRef.rootThreadLaunch.entity.value());
  appendChannelKeyU64(key, spatial->launchRef.staticGraphLaunch.entity.value());
  appendChannelKeyU64(key, input.consumerStreamInputOrdinal);
  key.push_back(static_cast<char>(spatial->denseCoordinates.size()));
  for (std::uint64_t coordinate : spatial->denseCoordinates)
    appendChannelKeyU64(key, coordinate);
  return key;
}

llvm::Expected<loom::sim::CanonicalSimulationRuntimeInput> bindChannelInputs(
    int connection, std::uint64_t sequence,
    const loom::runtime::Gem5SpatialChannelProjection &projection,
    const loom::sim::ImportedSpatialSimulationWorkload &workload,
    const loom::sim::CanonicalSimulationRuntimeInput &runtimeInput,
    std::uint64_t &requestId,
    std::map<std::uint64_t, ChannelSequenceState> &channelSequences,
    std::vector<ChannelReservation> &reservations) {
  const auto *base = runtimeInput.spatial();
  if (!base)
    return invalid("Spatial runtime input lost its typed payload");
  const auto *spatialWorkload = workload.workload.spatial();
  if (!spatialWorkload)
    return invalid("Spatial workload lost its typed payload");
  auto view = workload.dataflow.view();
  if (!view)
    return view.takeError();
  loom::sim::SpatialSimulationRuntimeInputDraft draft{
      workload.workload.identity()};
  draft.runtimeValues = base->runtimeValues;
  draft.runtimeStreams = base->runtimeStreams;
  draft.memoryObjects = base->memoryObjects;
  draft.memoryRootBindings.reserve(base->memoryRootBindings.size());
  for (const loom::sim::MemoryRootBindingEntry &binding :
       base->memoryRootBindings)
    draft.memoryRootBindings.push_back({binding.root,
                                        binding.binding.objectOrdinal,
                                        binding.binding.byteOffset});

  for (const loom::runtime::Gem5SpatialChannelInput &channel :
       projection.inputs) {
    if (channel.consumerStreamInputOrdinal >= draft.runtimeStreams.size())
      return invalid("channel projection names an absent consumer input");
    auto state = channelSequences.find(channel.address);
    if (state == channelSequences.end())
      return invalid("channel input has no ordered sequence state");
    auto key = channelConsumerKey(workload, channel);
    if (!key)
      return key.takeError();
    auto branch = state->second.consumerOrdinals.find(*key);
    if (branch == state->second.consumerOrdinals.end())
      return invalid("channel input has no deterministic consumer branch");
    auto received = state->second.sequence.reserve(branch->second);
    if (!received)
      return received.takeError();
    if (received->kind == loom::runtime::OrderedChannelReceiveKind::WouldBlock)
      return llvm::createStringError(std::errc::timed_out,
                                     "ordered channel input is waiting for "
                                     "the next SendSeq");
    if (received->kind == loom::runtime::OrderedChannelReceiveKind::Closed)
      return invalid("ordered channel input reached channel close");
    reservations.push_back(
        {channel.address, branch->second, received->sequence});
    auto stream = loom::sim::decodeSpatialChannelStream(
        received->payload, *view, spatialWorkload->launchRef,
        channel.consumerStreamInputOrdinal, draft.memoryObjects.size());
    if (!stream)
      return stream.takeError();
    draft.runtimeStreams[channel.consumerStreamInputOrdinal] =
        std::move(*stream);
  }
  return loom::sim::finalizeSimulationRuntimeInput(draft, workload.workload,
                                                   *view);
}

llvm::Error publishChannelOutputs(
    int connection, std::uint64_t sequence,
    const loom::runtime::Gem5SpatialChannelProjection &projection,
    const loom::sim::SpatialEngineBoundaryResult &result,
    const loom::sim::ImportedSpatialSimulationWorkload &workload,
    const loom::sim::CanonicalSimulationRuntimeInput &runtimeInput,
    std::uint64_t &requestId,
    std::map<std::uint64_t, ChannelSequenceState> &channelSequences) {
  if (projection.outputs.empty())
    return llvm::Error::success();
  if (!std::holds_alternative<loom::sim::RetiredExecution>(result.terminal))
    return invalid("channel producer did not retire");
  const auto *spatialWorkload = workload.workload.spatial();
  const auto *spatialRuntime = runtimeInput.spatial();
  if (!spatialWorkload || !spatialRuntime)
    return invalid("Spatial channel owner lost its typed payload");
  auto view = workload.dataflow.view();
  if (!view)
    return view.takeError();
  for (const loom::runtime::Gem5SpatialChannelOutput &channel :
       projection.outputs) {
    const auto found =
        llvm::find(spatialWorkload->observableContract.streamOutputs,
                   channel.producerStreamOutputOrdinal);
    if (found == spatialWorkload->observableContract.streamOutputs.end())
      return invalid("channel producer output is not observable");
    const std::size_t observation = static_cast<std::size_t>(std::distance(
        spatialWorkload->observableContract.streamOutputs.begin(), found));
    if (observation >= result.functionalObservations.streamOutputs.size())
      return invalid("channel producer omitted its selected stream output");
    auto encodedStream = loom::sim::encodeSpatialChannelStream(
        result.functionalObservations.streamOutputs[observation], *view,
        spatialWorkload->launchRef, channel.producerStreamOutputOrdinal,
        spatialRuntime->memoryObjects.size());
    if (!encodedStream)
      return encodedStream.takeError();
    if (encodedStream->empty() ||
        encodedStream->size() >
            channel.capacityBytes -
                loom::runtime::gem5SpatialChannelBufferHeaderBytes)
      return invalid("channel stream exceeds its selected buffer");
    auto state = channelSequences.find(channel.address);
    if (state == channelSequences.end())
      return invalid("channel output has no ordered sequence state");
    const loom::sim::CanonicalStreamSequence &stream =
        result.functionalObservations.streamOutputs[observation];
    // ClosedAfterLast belongs to this graph observation horizon; the Dataflow
    // channel contract has no implicit EOS. Repeated producer launches append
    // to the same SendSeq stream until an explicit channel protocol closes it.
    if (stream.values.tokenCount != 0) {
      if (stream.values.lanes.size() % stream.values.tokenCount != 0)
        return invalid("channel stream lane count is not token aligned");
      const std::size_t lanesPerToken =
          stream.values.lanes.size() / stream.values.tokenCount;
      for (std::uint64_t token = 0; token != stream.values.tokenCount;
           ++token) {
        loom::sim::CanonicalStreamSequence one;
        one.values.tokenCount = 1;
        const auto begin = stream.values.lanes.begin() +
                           static_cast<std::size_t>(token) * lanesPerToken;
        one.values.lanes.assign(begin, begin + lanesPerToken);
        one.termination = token + 1 == stream.values.tokenCount
                              ? stream.termination
                              : loom::sim::StreamTermination::OpenAfterLast;
        auto encodedToken = loom::sim::encodeSpatialChannelStream(
            one, *view, spatialWorkload->launchRef,
            channel.producerStreamOutputOrdinal,
            spatialRuntime->memoryObjects.size());
        if (!encodedToken)
          return encodedToken.takeError();
        auto published = state->second.sequence.publish(*encodedToken);
        if (!published)
          return published.takeError();
      }
    }
    const std::size_t payloadBytes = encodedStream->size();
    loom::runtime::Gem5BridgeMemoryRequest payload{
        loom::runtime::Gem5BridgeMemoryOperation::Write,
        0,
        requestId++,
        channel.address + loom::runtime::gem5SpatialChannelBufferHeaderBytes,
        payloadBytes,
        std::move(*encodedStream)};
    auto payloadResponse =
        transactMemory(connection, sequence,
                       loom::runtime::Gem5BridgeMessageKind::ChannelTransfer,
                       std::move(payload));
    if (!payloadResponse)
      return payloadResponse.takeError();
    const auto header =
        loom::runtime::encodeGem5SpatialChannelBufferHeader(payloadBytes);
    loom::runtime::Gem5BridgeMemoryRequest commit{
        loom::runtime::Gem5BridgeMemoryOperation::Write,
        0,
        requestId++,
        channel.address,
        header.size(),
        std::vector<std::uint8_t>(header.begin(), header.end())};
    auto commitResponse =
        transactMemory(connection, sequence,
                       loom::runtime::Gem5BridgeMessageKind::ChannelTransfer,
                       std::move(commit));
    if (!commitResponse)
      return commitResponse.takeError();
  }
  return llvm::Error::success();
}

llvm::Error publishInvocationResults(
    int connection, std::uint64_t sequence,
    llvm::ArrayRef<loom::sim::SpatialInvocationMemoryWrite> writes,
    std::uint64_t readyAfterTicks, std::uint64_t &requestId) {
  for (std::size_t ordinal = 0; ordinal != writes.size(); ++ordinal) {
    const loom::sim::SpatialInvocationMemoryWrite &write = writes[ordinal];
    loom::runtime::Gem5BridgeMemoryRequest request{
        loom::runtime::Gem5BridgeMemoryOperation::Write,
        ordinal == 0 ? readyAfterTicks : 0,
        requestId++,
        write.address,
        write.bytes.size(),
        write.bytes};
    auto response =
        transactMemory(connection, sequence,
                       loom::runtime::Gem5BridgeMessageKind::MemoryRequest,
                       std::move(request));
    if (!response)
      return response.takeError();
  }
  return llvm::Error::success();
}

#if defined(LOOM_GEM5_SPATIAL_ENGINE_DFG)
llvm::Expected<loom::sim::SpatialEngineBoundaryResult>
runDfg(const loom::sim::PreparedDfgExecution &prepared,
       const loom::sim::ImportedSpatialSimulationWorkload &workload,
       const loom::sim::CanonicalSimulationRuntimeInput &runtimeInput) {
  auto retired = loom::sim::simulateRetiredDfgWorkload(
      prepared, workload.workload, runtimeInput, maximumWork);
  if (!retired)
    return retired.takeError();
  auto zero = loom::evaluation::ExactRatio::get(0, 1);
  auto retirement =
      loom::evaluation::ExactRatio::get(retired->report.wavefrontSteps, 1);
  if (!zero)
    return zero.takeError();
  if (!retirement)
    return retirement.takeError();
  const loom::sim::SpatialEventCoordinate launch{std::move(*zero), 0};
  const loom::sim::SpatialEventCoordinate terminal{std::move(*retirement), 0};
  return loom::sim::SpatialEngineBoundaryResult{
      loom::sim::RetiredExecution{},
      std::move(retired->observations),
      {launch, terminal, terminal},
      {}};
}

#else
llvm::Expected<loom::sim::SpatialEngineBoundaryResult>
runCgra(const loom::sim::PreparedCgraExecution &prepared,
        const loom::sim::ImportedSpatialSimulationWorkload &workload,
        const loom::sim::CanonicalSimulationRuntimeInput &runtimeInput,
        loom::sim::CgraExternalMemoryProvider *externalMemoryProvider) {
  auto outcome = loom::sim::simulateCgraWorkload(
      prepared, workload.workload, runtimeInput, maximumWork, std::nullopt,
      externalMemoryProvider);
  if (!outcome)
    return outcome.takeError();
  if (outcome->state == loom::sim::SpatialExecutionSessionState::StoppedByLimit)
    return llvm::createStringError(
        std::make_error_code(std::errc::timed_out),
        "CGRA engine reached its work limit: frames=" +
            std::to_string(outcome->counters.eventFrameCount) +
            ", actor_commits=" +
            std::to_string(outcome->counters.actorCommitCount) +
            ", actor_retirements=" +
            std::to_string(outcome->counters.actorRetirementCount) +
            ", token_publications=" +
            std::to_string(outcome->counters.tokenPublicationCount) +
            ", physical_requests=" +
            std::to_string(outcome->counters.physicalRequestCount) +
            ", physical_grants=" +
            std::to_string(outcome->counters.physicalGrantCount) +
            ", physical_retirements=" +
            std::to_string(outcome->counters.physicalRetirementCount) +
            ", empty_frames=" +
            std::to_string(outcome->counters.emptyEventFrameCount) +
            ", source_frames=" +
            std::to_string(outcome->counters.computeSourceFrameCount) + "/" +
            std::to_string(outcome->counters.memorySourceFrameCount) + "/" +
            std::to_string(outcome->counters.transportSourceFrameCount) + "/" +
            std::to_string(outcome->counters.physicalSourceFrameCount));
  if (outcome->state != loom::sim::SpatialExecutionSessionState::Retired ||
      !outcome->retired) {
    const llvm::StringRef state = [&] {
      switch (outcome->state) {
      case loom::sim::SpatialExecutionSessionState::Runnable:
        return llvm::StringRef("runnable");
      case loom::sim::SpatialExecutionSessionState::Retired:
        return llvm::StringRef("retired_without_result");
      case loom::sim::SpatialExecutionSessionState::Halted:
        return llvm::StringRef("halted");
      case loom::sim::SpatialExecutionSessionState::StoppedByLimit:
        return llvm::StringRef("stopped_by_limit");
      case loom::sim::SpatialExecutionSessionState::Failed:
        return llvm::StringRef("failed");
      }
      llvm_unreachable("unknown Spatial execution session state");
    }();
    std::string diagnostic;
    llvm::raw_string_ostream stream(diagnostic);
    stream << "CGRA engine did not retire the graph: state=" << state
           << ", frames=" << outcome->counters.eventFrameCount
           << ", actor_commits=" << outcome->counters.actorCommitCount
           << ", actor_retirements=" << outcome->counters.actorRetirementCount
           << ", token_publications=" << outcome->counters.tokenPublicationCount
           << ", memory_linearizations="
           << outcome->counters.memoryLinearizationCount
           << ", physical_requests=" << outcome->counters.physicalRequestCount
           << ", physical_grants=" << outcome->counters.physicalGrantCount
           << ", physical_retirements="
           << outcome->counters.physicalRetirementCount;
    if (outcome->closedWaitSet)
      stream << ", pending_actor_firings="
             << outcome->closedWaitSet->pendingActorFirings
             << ", pending_transfers="
             << outcome->closedWaitSet->pendingTransfers
             << ", pending_physical_actions="
             << outcome->closedWaitSet->pendingPhysicalActions
             << ", graph_retirement_visible="
             << outcome->closedWaitSet->graphRetirementVisible;
    if (outcome->closedWaitSet)
      for (const auto &firing : outcome->closedWaitSet->actorFirings)
        stream << ", actor_firing={actor=" << firing.semanticActorOrdinal
               << ", occurrence=" << firing.occurrenceOrdinal
               << ", transition=" << firing.transitionCaseOrdinal
               << ", transfers=" << firing.completedTransfers << "/"
               << firing.expectedTransfers
               << ", physical_complete=" << firing.physicalComplete
               << ", causal_release=" << firing.causalReleaseSatisfied << "}";
    if (outcome->closedWaitSet)
      for (const auto &transfer : outcome->closedWaitSet->transfers) {
        stream << ", transfer={binding=" << transfer.bindingOrdinal
               << ", occurrence=" << transfer.occurrenceOrdinal
               << ", producer=" << transfer.producerActorOrdinal << ":"
               << transfer.producerResultOrdinal
               << ", blocked=" << transfer.blocked
               << ", arrival_scheduled=" << transfer.arrivalScheduled
               << ", publication_ready=" << transfer.publicationReady
               << ", published=" << transfer.published
               << ", consumed_requested=" << transfer.consumedRequested
               << ", operand_reserved=" << transfer.operandCapacityReserved
               << ", operand_blocked=" << transfer.operandCapacityBlocked
               << ", produced=" << transfer.producedRetired << "/"
               << transfer.producedPermitted
               << ", traversal=" << transfer.traversalRetired << "/"
               << transfer.traversalPermitted << ", traversal_terminals="
               << transfer.traversalTerminalsPermitted
               << ", consumed=" << transfer.consumedRetired << "/"
               << transfer.consumedPermitted
               << ", ready_sinks=" << transfer.readySinkCount
               << ", published_sinks=" << transfer.publishedSinkCount << "/"
               << transfer.sinkCount
               << ", publications=" << transfer.publishedPublicationCount << "/"
               << transfer.requestedPublicationCount << "/"
               << transfer.publicationCount << ", blocking_traversal="
               << transfer.blockingTraversalNodeOrdinal << ":"
               << transfer.blockingTraversalWaitingForStorage
               << ", blocking_storage=" << transfer.blockingStorageOrdinal
               << ":" << transfer.blockingStorageOccupancy << "+"
               << transfer.blockingStorageReservations << "/"
               << transfer.blockingStorageCapacity << ":";
        if (transfer.blockingStorageHead)
          stream << transfer.blockingStorageHead->bindingOrdinal << ":"
                 << transfer.blockingStorageHead->occurrenceOrdinal << ":"
                 << transfer.blockingStorageHead->traversalNodeOrdinal;
        else
          stream << "none";
        stream << ", downstream=" << transfer.blockingDownstreamStorageCount
               << ":" << transfer.blockingUnbufferedSinkCount << ":"
               << transfer.blockingDownstreamStorageOrdinal << ":"
               << transfer.blockingDownstreamStorageOccupancy << "+"
               << transfer.blockingDownstreamStorageReservations << "/"
               << transfer.blockingDownstreamStorageCapacity << ":"
               << transfer.blockingDownstreamStorageReserved << ":";
        if (transfer.blockingDownstreamStorageHead)
          stream
              << transfer.blockingDownstreamStorageHead->bindingOrdinal << ":"
              << transfer.blockingDownstreamStorageHead->occurrenceOrdinal
              << ":"
              << transfer.blockingDownstreamStorageHead->traversalNodeOrdinal;
        else
          stream << "none";
        stream << ", blocking_route_targets=[";
        for (auto indexed : llvm::enumerate(transfer.blockingTraversals)) {
          if (indexed.index())
            stream << ",";
          stream << llvm::toHex(
              loom::fabric::canonicalFabricBytes(indexed.value()), true);
        }
        stream << "], downstream_route_targets=[";
        for (auto indexed :
             llvm::enumerate(transfer.blockingDownstreamTraversals)) {
          if (indexed.index())
            stream << ",";
          stream << llvm::toHex(
              loom::fabric::canonicalFabricBytes(indexed.value()), true);
        }
        stream << "]";
        stream << ", blocking_actor=" << transfer.blockingActorOrdinal
               << ", blocking_ready=" << transfer.blockingReadyTokenCount
               << ", blocking_queue=" << transfer.blockingQueueOccupancy << "+"
               << transfer.blockingQueueReservations << "/"
               << transfer.blockingQueueCapacity << ", unpublished=[";
        for (std::size_t index = 0;
             index != transfer.unpublishedActorOrdinals.size(); ++index) {
          if (index != 0)
            stream << ",";
          stream << transfer.unpublishedActorOrdinals[index] << ":"
                 << transfer.unpublishedInputOrdinals[index] << ":"
                 << transfer.unpublishedReadyTokenCounts[index];
        }
        stream << "]}";
      }
    if (outcome->closedWaitSet) {
      stream << ", blocked_actor_inputs=[";
      for (std::size_t index = 0;
           index != outcome->closedWaitSet->blockedActorInputs.size();
           ++index) {
        if (index != 0)
          stream << ",";
        const auto &input = outcome->closedWaitSet->blockedActorInputs[index];
        stream << input.semanticActorOrdinal << ":" << input.actorEntityId
               << ":" << input.inputOrdinal << ":" << input.channelOrdinal
               << ":" << static_cast<unsigned>(input.sourceKind) << ":"
               << input.definingActorOrdinal << ":"
               << input.definingActorEntityId << ":"
               << input.definingActorTerminal;
      }
      stream << "]";
      stream << ", transfer_wait_cycle=[";
      for (std::size_t index = 0;
           index != outcome->closedWaitSet->transferWaitCycle.size(); ++index) {
        if (index != 0)
          stream << ",";
        const auto &edge = outcome->closedWaitSet->transferWaitCycle[index];
        stream << edge.waitingBindingOrdinal << ":"
               << edge.waitingOccurrenceOrdinal << "->"
               << edge.blockingActorOrdinal << ":"
               << edge.blockingBindingOrdinal << ":"
               << edge.blockingOccurrenceOrdinal << ":"
               << static_cast<unsigned>(edge.kind);
      }
      stream << "]";
      stream << ", actor_wait_cycle=[";
      for (std::size_t index = 0;
           index != outcome->closedWaitSet->actorWaitCycle.size(); ++index) {
        if (index != 0)
          stream << ",";
        const auto &edge = outcome->closedWaitSet->actorWaitCycle[index];
        stream << edge.waitingActorOrdinal << "->" << edge.blockingActorOrdinal
               << ":" << static_cast<unsigned>(edge.kind);
      }
      stream << "]";
      stream
          << ", operand_queue_summary={groups="
          << outcome->closedWaitSet->operandQueueGroupCount
          << ",blocking_groups="
          << outcome->closedWaitSet->operandQueuePotentiallyBlockingGroupCount
          << ",shared_ingress="
          << outcome->closedWaitSet->operandQueueSharedIngressPressure
          << ",distinct_ingress="
          << outcome->closedWaitSet->operandQueueDistinctIngressCount
          << ",pairing_keys="
          << outcome->closedWaitSet->operandQueuePairingKeyCount << ",status="
          << static_cast<unsigned>(
                 outcome->closedWaitSet->operandQueueProgressStatus)
          << ",support="
          << static_cast<unsigned>(
                 outcome->closedWaitSet->operandQueueProgressSupport)
          << ",digest=";
      if (outcome->closedWaitSet->operandQueueProjectionDigest)
        stream << loom::formatComponentViewDigestHex(
            *outcome->closedWaitSet->operandQueueProjectionDigest);
      else
        stream << "none";
      stream << "}";
      stream << ", operand_queue_heads=[";
      for (std::size_t index = 0;
           index != outcome->closedWaitSet->operandQueueHeads.size() &&
           index != 16;
           ++index) {
        if (index != 0)
          stream << ",";
        const auto &head = outcome->closedWaitSet->operandQueueHeads[index];
        llvm::SmallString<32> tag;
        head.headTag.toStringUnsigned(tag, 16);
        stream << "{context="
               << llvm::toHex(
                      ::loom::fabric::canonicalFabricBytes(head.queue.context),
                      true)
               << ",fu_occurrence=" << head.queue.fuOccurrence
               << ",fu_input=" << head.queue.fuInput
               << ",unit=" << head.allocationUnit
               << ",occupancy=" << head.occupancy
               << ",reservations=" << head.reservations
               << ",capacity=" << head.capacity
               << ",head=" << head.headBindingOrdinal << ":"
               << head.headOccurrenceOrdinal << ":"
               << head.headProducerSequenceOrdinal << ":" << tag.str()
               << ",exact=" << head.exactHead << "}";
      }
      if (outcome->closedWaitSet->operandQueueHeads.size() > 16)
        stream << ",...";
      stream << "]";
    }
    if (outcome->closedWaitSet)
      for (const auto &action : outcome->closedWaitSet->physicalActions)
        stream << ", physical_action={action=" << action.actionOrdinal
               << ", occurrence=" << action.occurrenceOrdinal
               << ", client=" << static_cast<unsigned>(action.clientKind)
               << ", granted=" << action.granted
               << ", has_commit=" << action.hasCommit
               << ", requires_causal_release=" << action.requiresCausalRelease
               << ", intrinsic_release=" << action.intrinsicReleaseReached
               << ", causal_release=" << action.causalReleaseReached << "}";
    return invalid(diagnostic);
  }
  return loom::sim::SpatialEngineBoundaryResult{
      loom::sim::RetiredExecution{},
      std::move(outcome->retired->observations),
      std::move(outcome->retired->progress),
      {}};
}
#endif

llvm::Expected<std::uint64_t> completionDelay(
    const loom::sim::SpatialEngineBoundaryResult &result,
    std::optional<loom::sim::SpatialEventCoordinate> externallyServicedThrough =
        std::nullopt) {
#if defined(LOOM_GEM5_SPATIAL_ENGINE_DFG)
  return 0;
#else
  if (!result.progressObservations.graphRetirementVisible)
    return 0;
  const loom::sim::SpatialEventCoordinate terminal =
      *result.progressObservations.graphRetirementVisible;
  auto terminalTick = spatialCoordinateTicks(terminal);
  if (!terminalTick)
    return terminalTick.takeError();
  if (!externallyServicedThrough)
    return *terminalTick;
  if (loom::sim::compareSpatialEventCoordinates(*externallyServicedThrough,
                                                terminal) > 0)
    return invalid("external memory service follows Spatial retirement");
  auto servicedTick = spatialCoordinateTicks(*externallyServicedThrough);
  if (!servicedTick)
    return servicedTick.takeError();
  if (*servicedTick > *terminalTick)
    return invalid("external memory service tick follows Spatial retirement");
  return *terminalTick - *servicedTick;
#endif
}

#if defined(LOOM_GEM5_SPATIAL_ENGINE_DFG)
void appendKeyU64(std::string &key, std::uint64_t value) {
  for (unsigned byte = 0; byte != 8; ++byte)
    key.push_back(static_cast<char>(value >> (byte * 8)));
}
#endif

void appendKeyIdentity(std::string &key,
                       const loom::ArtifactIdentity &identity) {
  const auto &bytes = identity.bytes();
  key.append(reinterpret_cast<const char *>(bytes.data()), bytes.size());
}

llvm::Expected<std::size_t> selectSessionEntry(
    llvm::ArrayRef<SpatialSessionEntry> entries,
    const loom::runtime::Gem5SpatialLaunchEnvelope &launch,
    const std::optional<loom::runtime::SpatialInvocationWire> &invocation) {
  std::optional<std::size_t> selected;
  for (const auto indexed : llvm::enumerate(entries)) {
    const SpatialSessionEntry &entry = indexed.value();
    if (entry.bridgeOrdinal != launch.bridgeSessionOrdinal)
      continue;
    if (entry.expectedLaunch != launch.staticLaunch)
      continue;
    const auto *workload = entry.workload.workload.spatial();
    if (!workload)
      return invalid("Spatial session entry lost its workload payload");
    bool matches = false;
    if (invocation) {
      matches = invocation->canonicalDataflowIdentity ==
                    entry.workload.dataflow.identity().bytes() &&
                invocation->rootThreadLaunchEntity ==
                    workload->launchRef.rootThreadLaunch.entity.value() &&
                invocation->graphLaunchEntity ==
                    workload->launchRef.staticGraphLaunch.entity.value() &&
                invocation->denseCoordinates == workload->denseCoordinates;
    } else {
      matches = entry.staticRuntime.has_value();
    }
    if (!matches)
      continue;
    if (selected)
      return invalid("Spatial launch matches multiple session entries");
    selected = indexed.index();
  }
  if (!selected)
    return invalid("Spatial launch matches no session entry");
  return *selected;
}

} // namespace

int main(int argc, char **argv) {
  llvm::InitLLVM initialization(argc, argv);
  llvm::cl::ParseCommandLineOptions(argc, argv);
  if (maximumWork == 0 || ticksPerCycle == 0 || maximumInvocations == 0 ||
      bridgeCount == 0 || bridgeCount > std::numeric_limits<int>::max())
    return report(invalid("work, timing, and invocation limits must be "
                          "positive"));

  loom::ArtifactStore store(artifactStorePath);
  auto dataflow = root(dataflowIdentity, dataflow::canonicalDataflowSchema);
  if (!dataflow)
    return report(dataflow.takeError());
  const std::size_t entryCount = workloadIdentities.size();
  if (expectedLaunchPaths.size() != entryCount ||
      runtimeInputIdentities.size() != entryCount ||
      channelProjectionPaths.size() != entryCount ||
      bridgeOrdinals.size() != entryCount)
    return report(invalid("Spatial session argument tables have different "
                          "lengths"));
#if defined(LOOM_GEM5_SPATIAL_ENGINE_DFG)
  if (!fabricIdentities.empty() || !spatialMappingIdentities.empty())
    return report(invalid("DFG session received CGRA owner references"));
#else
  if (fabricIdentities.size() != entryCount ||
      spatialMappingIdentities.size() != entryCount)
    return report(invalid("CGRA session owner tables are not total"));
#endif

  std::vector<SpatialSessionEntry> entries;
  entries.reserve(entryCount);
  std::vector<PreparedSpatialExecution> preparedExecutions;
  std::map<std::string, std::size_t> preparedByKey;
  std::map<std::uint64_t, std::uint64_t> nextSessionEntryOrdinal;
  for (std::size_t ordinal = 0; ordinal != entryCount; ++ordinal) {
    if (bridgeOrdinals[ordinal] >= bridgeCount)
      return report(invalid("Spatial entry names an absent bridge session"));
    auto workload =
        root(workloadIdentities[ordinal], loom::sim::simulationWorkloadSchema);
    if (!workload)
      return report(workload.takeError());
    auto importedWorkload =
        loom::sim::importSpatialSimulationWorkload(*workload, store);
    if (!importedWorkload)
      return report(importedWorkload.takeError());
    if (importedWorkload->dataflow.identity() != dataflow->artifact)
      return report(invalid("Spatial inputs name a foreign Dataflow owner"));
    std::optional<loom::sim::CanonicalSimulationRuntimeInput> staticRuntime;
    if (runtimeInputIdentities[ordinal] != "none") {
      auto runtime = root(runtimeInputIdentities[ordinal],
                          loom::sim::simulationRuntimeInputSchema);
      if (!runtime)
        return report(runtime.takeError());
      auto imported = loom::sim::importSpatialSimulationRuntimeInput(
          *runtime, *importedWorkload, store);
      if (!imported)
        return report(imported.takeError());
      staticRuntime.emplace(std::move(*imported));
    }
    auto expectedLaunch = readFile(expectedLaunchPaths[ordinal]);
    if (!expectedLaunch)
      return report(expectedLaunch.takeError());
    auto channelBytes = readFile(channelProjectionPaths[ordinal]);
    if (!channelBytes)
      return report(channelBytes.takeError());
    auto channels =
        loom::runtime::decodeGem5SpatialChannelProjection(*channelBytes);
    if (!channels)
      return report(channels.takeError());
    std::string preparedKey;
    appendKeyIdentity(preparedKey, dataflow->artifact);
#if defined(LOOM_GEM5_SPATIAL_ENGINE_DFG)
    const auto *spatialWorkload = importedWorkload->workload.spatial();
    if (!spatialWorkload)
      return report(invalid("Spatial workload lost its typed payload"));
    appendKeyU64(preparedKey,
                 spatialWorkload->launchRef.rootThreadLaunch.entity.value());
    appendKeyU64(preparedKey,
                 spatialWorkload->launchRef.staticGraphLaunch.entity.value());
#else
    auto fabric =
        root(fabricIdentities[ordinal], loom::fabric::fabricArtifactSchema);
    auto mapping = root(spatialMappingIdentities[ordinal],
                        loom::mapping::mappingArtifactSchema);
    if (!fabric)
      return report(fabric.takeError());
    if (!mapping)
      return report(mapping.takeError());
    appendKeyIdentity(preparedKey, fabric->artifact);
    appendKeyIdentity(preparedKey, mapping->artifact);
#endif
    auto prepared = preparedByKey.find(preparedKey);
    std::size_t preparedOrdinal = 0;
    if (prepared == preparedByKey.end()) {
#if defined(LOOM_GEM5_SPATIAL_ENGINE_DFG)
      auto built = loom::sim::prepareDfgExecution(importedWorkload->dataflow,
                                                  spatialWorkload->launchRef);
#else
      auto built =
          loom::sim::prepareCgraExecution(*dataflow, *fabric, *mapping, store);
#endif
      if (!built)
        return report(built.takeError());
      preparedOrdinal = preparedExecutions.size();
      preparedExecutions.push_back(std::move(*built));
      preparedByKey.emplace(std::move(preparedKey), preparedOrdinal);
    } else {
      preparedOrdinal = prepared->second;
    }
    const std::uint64_t sessionEntryOrdinal =
        nextSessionEntryOrdinal[bridgeOrdinals[ordinal]]++;
    entries.push_back({std::move(*importedWorkload), std::move(staticRuntime),
                       std::move(*channels), std::move(*expectedLaunch),
                       bridgeOrdinals[ordinal], sessionEntryOrdinal,
                       preparedOrdinal});
  }
  if (nextSessionEntryOrdinal.size() != bridgeCount)
    return report(invalid("Spatial entry table does not cover every bridge"));

  std::map<std::uint64_t, std::uint64_t> channelCapacities;
  std::map<std::uint64_t, std::set<std::string>> channelConsumerKeys;
  for (const SpatialSessionEntry &entry : entries) {
    for (const auto &output : entry.channels.outputs) {
      auto [position, inserted] =
          channelCapacities.emplace(output.address, output.capacityBytes);
      if (!inserted && position->second != output.capacityBytes)
        return report(invalid("ordered channel outputs disagree on capacity"));
    }
    for (const auto &input : entry.channels.inputs) {
      auto [position, inserted] =
          channelCapacities.emplace(input.address, input.capacityBytes);
      if (!inserted && position->second != input.capacityBytes)
        return report(invalid("ordered channel inputs disagree on capacity"));
      auto key = channelConsumerKey(entry.workload, input);
      if (!key)
        return report(key.takeError());
      channelConsumerKeys[input.address].insert(std::move(*key));
    }
  }
  std::map<std::uint64_t, ChannelSequenceState> channelSequences;
  for (const auto &[address, capacityBytes] : channelCapacities) {
    const auto keys = channelConsumerKeys.find(address);
    if (keys == channelConsumerKeys.end() || keys->second.empty())
      return report(invalid("ordered channel has no consumer branch"));
    if (keys->second.size() > std::numeric_limits<std::uint32_t>::max())
      return report(invalid("ordered channel consumer count exceeds u32"));
    const std::uint32_t consumers =
        static_cast<std::uint32_t>(keys->second.size());
    if (consumers == 0)
      return report(invalid("ordered channel has no consumer branch"));
    if (capacityBytes <= loom::runtime::gem5SpatialChannelBufferHeaderBytes)
      return report(invalid("ordered channel capacity has no payload space"));
    auto sequence = loom::runtime::OrderedChannelSequence::create(
        capacityBytes - loom::runtime::gem5SpatialChannelBufferHeaderBytes,
        consumers);
    if (!sequence)
      return report(sequence.takeError());
    channelSequences.emplace(address,
                             ChannelSequenceState{std::move(*sequence), {}});
    auto &ordinals = channelSequences.find(address)->second.consumerOrdinals;
    std::uint32_t ordinal = 0;
    for (const std::string &key : keys->second)
      ordinals.emplace(key, ordinal++);
  }
  for (const SpatialSessionEntry &entry : entries) {
    for (const auto &input : entry.channels.inputs) {
      auto state = channelSequences.find(input.address);
      if (state == channelSequences.end())
        return report(invalid("ordered channel input has no sequence state"));
      auto key = channelConsumerKey(entry.workload, input);
      if (!key)
        return report(key.takeError());
      if (state->second.consumerOrdinals.find(*key) ==
          state->second.consumerOrdinals.end())
        return report(invalid("ordered channel branch identity was not "
                              "materialized canonically"));
    }
  }

  auto server = openServer();
  if (!server)
    return report(server.takeError());
  struct BridgeConnection final {
    int descriptor = -1;
    std::uint64_t nextSequence = 0;
  };
  std::optional<int> serverDescriptor(*server);
  std::vector<BridgeConnection> connections;
  connections.reserve(static_cast<std::size_t>(bridgeCount));
  std::uint64_t acceptedConnections = 0;
  while (serverDescriptor || !connections.empty()) {
    std::vector<pollfd> descriptors;
    descriptors.reserve(connections.size() + (serverDescriptor ? 1 : 0));
    if (serverDescriptor)
      descriptors.push_back({*serverDescriptor, POLLIN, 0});
    for (const BridgeConnection &connection : connections)
      descriptors.push_back({connection.descriptor, POLLIN, 0});
    int readyCount = 0;
    do {
      readyCount = ::poll(descriptors.data(), descriptors.size(), -1);
    } while (readyCount < 0 && errno == EINTR);
    if (readyCount < 0)
      return report(invalid("cannot wait for a bridge message"));

    const std::size_t connectionOffset = serverDescriptor ? 1 : 0;
    if (serverDescriptor && (descriptors.front().revents & POLLIN) != 0) {
      const int connection = ::accept(*serverDescriptor, nullptr, nullptr);
      if (connection < 0)
        return report(invalid("cannot accept a bridge connection"));
      connections.push_back({connection, 0});
      ++acceptedConnections;
      if (acceptedConnections == bridgeCount) {
        ::close(*serverDescriptor);
        serverDescriptor.reset();
      }
      continue;
    }

    std::optional<std::size_t> readyConnection;
    for (std::size_t ordinal = 0; ordinal != connections.size(); ++ordinal) {
      const short events = descriptors[ordinal + connectionOffset].revents;
      if ((events & POLLIN) != 0) {
        readyConnection = ordinal;
        break;
      }
      if ((events & (POLLERR | POLLHUP | POLLNVAL)) != 0) {
        ::close(connections[ordinal].descriptor);
        connections.erase(connections.begin() + ordinal);
        break;
      }
    }
    if (!readyConnection)
      continue;
    BridgeConnection &bridge = connections[*readyConnection];
    if (bridge.nextSequence >= maximumInvocations)
      return report(invalid("bridge exceeded its invocation limit"));
    const int connection = bridge.descriptor;
    const std::uint64_t sequence = bridge.nextSequence;
    auto message = readMessage(connection);
    if (!message) {
      ::close(connection);
      return report(message.takeError());
    }
    loom::runtime::Gem5SpatialLaunchEnvelope launch;
    std::string launchDiagnostic;
    if (message->kind != loom::runtime::Gem5BridgeMessageKind::SpatialLaunch ||
        message->sequence != sequence ||
        !loom::runtime::decodeGem5SpatialLaunchEnvelope(
            message->payload, launch, launchDiagnostic)) {
      ::close(connection);
      return report(
          invalid("bridge launch does not match the exact Deployment: " +
                  launchDiagnostic));
    }

    std::optional<loom::runtime::SpatialInvocationWire> invocation;
    std::optional<loom::sim::CanonicalSimulationRuntimeInput> invocationRuntime;
    if (!launch.invocation.empty()) {
      loom::runtime::SpatialInvocationWire wire;
      std::string diagnostic;
      if (!loom::runtime::decodeSpatialInvocationWire(launch.invocation, wire,
                                                      diagnostic)) {
        ::close(connection);
        return report(invalid(diagnostic));
      }
      invocation = std::move(wire);
    }
    auto selected = selectSessionEntry(entries, launch, invocation);
    if (!selected) {
      ::close(connection);
      return report(selected.takeError());
    }
    SpatialSessionEntry &entry = entries[*selected];
    if (invocation) {
      if (entry.staticRuntime) {
        ::close(connection);
        return report(
            invalid("dynamic invocation has a competing static runtime input"));
      }
      auto runtime = loom::sim::materializeSpatialInvocationRuntimeInput(
          entry.workload, *invocation);
      if (!runtime) {
        ::close(connection);
        return report(runtime.takeError());
      }
      invocationRuntime.emplace(std::move(*runtime));
    } else if (!entry.staticRuntime) {
      ::close(connection);
      return report(invalid("static launch has no Spatial runtime input"));
    }
    const loom::sim::CanonicalSimulationRuntimeInput &baseRuntime =
        invocationRuntime ? *invocationRuntime : *entry.staticRuntime;
    std::uint64_t requestId = 0;
    std::vector<ChannelReservation> channelReservations;
    const auto settleChannelReservations = [&](bool commit) -> llvm::Error {
      for (auto it = channelReservations.rbegin();
           it != channelReservations.rend(); ++it) {
        auto state = channelSequences.find(it->address);
        if (state == channelSequences.end())
          return invalid("channel reservation lost its sequence state");
        llvm::Error error = commit ? state->second.sequence.commit(
                                         it->consumerOrdinal, it->sequence)
                                   : state->second.sequence.cancel(
                                         it->consumerOrdinal, it->sequence);
        if (error)
          return error;
      }
      channelReservations.clear();
      return llvm::Error::success();
    };
    auto runtime = bindChannelInputs(
        connection, message->sequence, entry.channels, entry.workload,
        baseRuntime, requestId, channelSequences, channelReservations);
    if (!runtime) {
      llvm::consumeError(settleChannelReservations(false));
      ::close(connection);
      return report(runtime.takeError());
    }
#if defined(LOOM_GEM5_SPATIAL_ENGINE_DFG)
    auto result = runDfg(preparedExecutions[entry.preparedOrdinal],
                         entry.workload, *runtime);
    std::optional<loom::sim::SpatialEventCoordinate> externallyServicedThrough;
#else
    std::optional<Gem5CgraExternalMemoryProvider> externalMemoryProvider;
    if (invocation)
      externalMemoryProvider.emplace(connection, message->sequence, *invocation,
                                     requestId);
    auto result = runCgra(
        preparedExecutions[entry.preparedOrdinal], entry.workload, *runtime,
        externalMemoryProvider ? &*externalMemoryProvider : nullptr);
    std::optional<loom::sim::SpatialEventCoordinate> externallyServicedThrough;
    if (externalMemoryProvider)
      externallyServicedThrough = externalMemoryProvider->lastCoordinate();
#endif
    if (!result) {
      llvm::consumeError(settleChannelReservations(false));
      ::close(connection);
      std::string diagnostic = llvm::toString(result.takeError());
      const auto *spatial = entry.workload.workload.spatial();
      diagnostic =
          "Spatial invocation sequence " + std::to_string(sequence) +
          " (session_entry=" + std::to_string(*selected) +
          ", prepared_entry=" + std::to_string(entry.preparedOrdinal) +
          ", runtime_objects=" +
          std::to_string(runtime->spatial()->memoryObjects.size()) +
          ", runtime_values=" +
          std::to_string(runtime->spatial()->runtimeValues.size()) +
          ", runtime_streams=" +
          std::to_string(runtime->spatial()->runtimeStreams.size()) +
          ", runtime_identity=" +
          formatArtifactIdentityHex(runtime->identity()) +
          ", dense_coordinates=" +
          (spatial ? std::to_string(spatial->denseCoordinates.size()) : "0") +
          "): " + diagnostic;
      return report(llvm::createStringError(
          std::make_error_code(std::errc::state_not_recoverable), diagnostic));
    }
    auto encoded = loom::sim::encodeSpatialEngineBoundaryResult(
        *result, entry.workload, *runtime);
    if (!encoded) {
      llvm::consumeError(settleChannelReservations(false));
      ::close(connection);
      return report(encoded.takeError());
    }
    if (llvm::Error error = publishChannelOutputs(
            connection, message->sequence, entry.channels, *result,
            entry.workload, *runtime, requestId, channelSequences)) {
      llvm::consumeError(settleChannelReservations(false));
      ::close(connection);
      return report(std::move(error));
    }
    auto delay = completionDelay(*result, externallyServicedThrough);
    if (!delay) {
      llvm::consumeError(settleChannelReservations(false));
      ::close(connection);
      return report(delay.takeError());
    }
    bool invocationDelayApplied = false;
    if (invocation) {
      auto writes = loom::sim::projectSpatialInvocationResultWrites(
          *invocation, entry.workload, result->functionalObservations);
      if (!writes) {
        llvm::consumeError(settleChannelReservations(false));
        ::close(connection);
        return report(writes.takeError());
      }
#if !defined(LOOM_GEM5_SPATIAL_ENGINE_DFG)
      if (externalMemoryProvider)
        *writes = externalMemoryProvider->retainUncommittedWrites(*writes);
#endif
      invocationDelayApplied = !writes->empty();
      if (llvm::Error error = publishInvocationResults(
              connection, message->sequence, *writes, *delay, requestId)) {
        llvm::consumeError(settleChannelReservations(false));
        ::close(connection);
        return report(std::move(error));
      }
    }
    if (llvm::Error error = settleChannelReservations(
            std::holds_alternative<loom::sim::RetiredExecution>(
                result->terminal))) {
      ::close(connection);
      return report(std::move(error));
    }
    const std::uint32_t status =
        std::holds_alternative<loom::sim::RetiredExecution>(result->terminal)
            ? 0
            : 1;
    std::optional<loom::runtime::SpatialInvocationRuntimeInputSnapshot>
        runtimeSnapshot;
    if (invocation)
      runtimeSnapshot.emplace(
          loom::runtime::SpatialInvocationRuntimeInputSnapshot{
              runtime->identity().bytes(),
              std::vector<std::uint8_t>(
                  runtime->canonicalBytes().bytes().begin(),
                  runtime->canonicalBytes().bytes().end())});
    std::vector<std::uint8_t> completionResult =
        loom::runtime::encodeSpatialInvocationResultWire(
            {entry.sessionEntryOrdinal, launch.invocation,
             std::move(runtimeSnapshot), std::move(*encoded)});
    if (completionResult.empty()) {
      ::close(connection);
      return report(invalid("cannot encode Spatial invocation result"));
    }
    const loom::runtime::Gem5BridgeCompletion completion{
        invocationDelayApplied ? 0 : *delay, status,
        std::move(completionResult)};
    if (llvm::Error error = writeMessage(
            connection,
            {loom::runtime::Gem5BridgeMessageKind::Completion,
             message->sequence,
             loom::runtime::encodeGem5BridgeCompletion(completion)})) {
      ::close(connection);
      return report(std::move(error));
    }
    ++bridge.nextSequence;
  }
  for (const BridgeConnection &connection : connections)
    ::close(connection.descriptor);
  ::unlink(socketPath.c_str());
  return 0;
}
