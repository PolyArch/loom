#include "Common/ArtifactStore.h"
#include "Common/ArtifactText.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Mapping/Artifact/MappingArtifact.h"
#include "Runtime/Gem5BridgeWire.h"
#include "Runtime/Gem5SpatialChannel.h"
#include "Simulator/CGRAAdmission.h"
#include "Simulator/CGRASimulator.h"
#include "Simulator/DFGSimulator.h"
#include "Simulator/SimulationArtifacts.h"
#include "Simulator/SimulationExecution.h"
#include "Simulator/SpatialInvocation.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include <cerrno>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
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
llvm::cl::opt<std::string>
    expectedLaunchPath("expected-launch",
                       llvm::cl::desc("Exact launch payload"),
                       llvm::cl::Required);
llvm::cl::opt<std::string>
    workloadIdentity("workload",
                     llvm::cl::desc("Spatial workload ArtifactIdentity"),
                     llvm::cl::Required);
llvm::cl::opt<std::string> runtimeInputIdentity(
    "runtime-input", llvm::cl::desc("Spatial runtime input ArtifactIdentity"),
    llvm::cl::init(""));
llvm::cl::opt<std::string> channelProjectionPath(
    "channel-projection",
    llvm::cl::desc("Invocation-local Spatial channel projection"),
    llvm::cl::Required);
llvm::cl::opt<std::string>
    dataflowIdentity("dataflow",
                     llvm::cl::desc("Canonical Dataflow ArtifactIdentity"),
                     llvm::cl::Required);
llvm::cl::opt<std::string>
    fabricIdentity("fabric", llvm::cl::desc("Fabric ArtifactIdentity"),
                   llvm::cl::init(""));
llvm::cl::opt<std::string>
    spatialMappingIdentity("spatial-mapping",
                           llvm::cl::desc("SpatialMapping ArtifactIdentity"),
                           llvm::cl::init(""));
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

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      "gem5_spatial_engine_invalid: " + message);
}

int report(llvm::Error error) {
  llvm::errs() << llvm::toString(std::move(error)) << '\n';
  return 1;
}

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
      ::listen(server, 1) != 0) {
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

llvm::Expected<std::vector<std::uint8_t>>
readChannelPayload(int connection, std::uint64_t sequence,
                   const loom::runtime::Gem5SpatialChannelInput &input,
                   std::uint64_t &requestId) {
  for (std::uint64_t attempt = 0; attempt != maximumWork; ++attempt) {
    const loom::runtime::Gem5BridgeMemoryRequest headerRequest{
        loom::runtime::Gem5BridgeMemoryOperation::Read,
        attempt == 0 ? 0 : static_cast<std::uint64_t>(ticksPerCycle),
        requestId++,
        input.address,
        loom::runtime::gem5SpatialChannelBufferHeaderBytes,
        {}};
    auto header = transactMemory(
        connection, sequence,
        loom::runtime::Gem5BridgeMessageKind::MemoryRequest, headerRequest);
    if (!header)
      return header.takeError();
    if (llvm::all_of(header->data, [](std::uint8_t byte) { return byte == 0; }))
      continue;
    auto payloadBytes =
        loom::runtime::decodeGem5SpatialChannelBufferHeader(header->data);
    if (!payloadBytes)
      return payloadBytes.takeError();
    if (*payloadBytes > input.capacityBytes -
                            loom::runtime::gem5SpatialChannelBufferHeaderBytes)
      return invalid("channel payload exceeds its selected buffer");
    const loom::runtime::Gem5BridgeMemoryRequest payloadRequest{
        loom::runtime::Gem5BridgeMemoryOperation::Read,
        0,
        requestId++,
        input.address + loom::runtime::gem5SpatialChannelBufferHeaderBytes,
        *payloadBytes,
        {}};
    auto payload = transactMemory(
        connection, sequence,
        loom::runtime::Gem5BridgeMessageKind::MemoryRequest, payloadRequest);
    if (!payload)
      return payload.takeError();
    return std::move(payload->data);
  }
  return llvm::createStringError(std::errc::timed_out,
                                 "channel input did not become ready");
}

struct ChannelInputContext final {
  loom::sim::ImportedSpatialSimulationInputs inputs;
  std::size_t streamObservation = 0;
};

llvm::Expected<std::vector<ChannelInputContext>> importChannelInputContexts(
    const loom::runtime::Gem5SpatialChannelProjection &projection,
    const loom::ArtifactStore &store) {
  std::vector<ChannelInputContext> contexts;
  contexts.reserve(projection.inputs.size());
  for (const loom::runtime::Gem5SpatialChannelInput &channel :
       projection.inputs) {
    auto inputs = loom::sim::importSpatialSimulationInputs(
        channel.producerWorkload, channel.producerRuntimeInput, store);
    if (!inputs)
      return inputs.takeError();
    const auto *workload = inputs->workload.spatial();
    if (!workload)
      return invalid("channel producer workload lost its typed payload");
    const auto found = llvm::find(workload->observableContract.streamOutputs,
                                  channel.producerStreamOutputOrdinal);
    if (found == workload->observableContract.streamOutputs.end())
      return invalid("channel producer output is not observable");
    const std::size_t observation = static_cast<std::size_t>(std::distance(
        workload->observableContract.streamOutputs.begin(), found));
    contexts.push_back({std::move(*inputs), observation});
  }
  return contexts;
}

llvm::Expected<loom::sim::CanonicalSimulationRuntimeInput> bindChannelInputs(
    int connection, std::uint64_t sequence,
    const loom::runtime::Gem5SpatialChannelProjection &projection,
    llvm::ArrayRef<ChannelInputContext> channelContexts,
    const loom::sim::ImportedSpatialSimulationWorkload &workload,
    const loom::sim::CanonicalSimulationRuntimeInput &runtimeInput,
    std::uint64_t &requestId) {
  if (channelContexts.size() != projection.inputs.size())
    return invalid("channel input context is not total over the projection");
  const auto *base = runtimeInput.spatial();
  if (!base)
    return invalid("Spatial runtime input lost its typed payload");
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

  for (const auto indexed : llvm::enumerate(projection.inputs)) {
    const loom::runtime::Gem5SpatialChannelInput &channel = indexed.value();
    const ChannelInputContext &context = channelContexts[indexed.index()];
    if (channel.consumerStreamInputOrdinal >= draft.runtimeStreams.size())
      return invalid("channel projection names an absent consumer input");
    auto payload = readChannelPayload(connection, sequence, channel, requestId);
    if (!payload)
      return payload.takeError();
    auto producerResult =
        loom::sim::decodeSpatialEngineBoundaryResult(*payload, context.inputs);
    if (!producerResult)
      return producerResult.takeError();
    if (!std::holds_alternative<loom::sim::RetiredExecution>(
            producerResult->terminal))
      return invalid("channel producer did not retire");
    if (context.streamObservation >=
        producerResult->functionalObservations.streamOutputs.size())
      return invalid("channel producer omitted its selected stream output");
    draft.runtimeStreams[channel.consumerStreamInputOrdinal] =
        producerResult->functionalObservations
            .streamOutputs[context.streamObservation];
  }
  auto view = workload.dataflow.view();
  if (!view)
    return view.takeError();
  return loom::sim::finalizeSimulationRuntimeInput(draft, workload.workload,
                                                   *view);
}

llvm::Error publishChannelOutputs(
    int connection, std::uint64_t sequence,
    const loom::runtime::Gem5SpatialChannelProjection &projection,
    llvm::ArrayRef<std::uint8_t> encodedResult, std::uint64_t &requestId) {
  for (const loom::runtime::Gem5SpatialChannelOutput &channel :
       projection.outputs) {
    if (encodedResult.empty() ||
        encodedResult.size() >
            channel.capacityBytes -
                loom::runtime::gem5SpatialChannelBufferHeaderBytes)
      return invalid("channel result exceeds its selected buffer");
    loom::runtime::Gem5BridgeMemoryRequest payload{
        loom::runtime::Gem5BridgeMemoryOperation::Write,
        0,
        requestId++,
        channel.address + loom::runtime::gem5SpatialChannelBufferHeaderBytes,
        encodedResult.size(),
        std::vector<std::uint8_t>(encodedResult.begin(), encodedResult.end())};
    auto payloadResponse =
        transactMemory(connection, sequence,
                       loom::runtime::Gem5BridgeMessageKind::ChannelTransfer,
                       std::move(payload));
    if (!payloadResponse)
      return payloadResponse.takeError();
    const auto header = loom::runtime::encodeGem5SpatialChannelBufferHeader(
        encodedResult.size());
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
        const loom::sim::CanonicalSimulationRuntimeInput &runtimeInput) {
  auto outcome = loom::sim::simulateCgraWorkload(prepared, workload.workload,
                                                 runtimeInput, maximumWork);
  if (!outcome)
    return outcome.takeError();
  if (outcome->state == loom::sim::SpatialExecutionSessionState::StoppedByLimit)
    return llvm::createStringError(std::errc::timed_out,
                                   "CGRA engine reached its work limit");
  if (outcome->state != loom::sim::SpatialExecutionSessionState::Retired ||
      !outcome->retired)
    return invalid("CGRA engine did not retire the graph");
  return loom::sim::SpatialEngineBoundaryResult{
      loom::sim::RetiredExecution{},
      std::move(outcome->retired->observations),
      std::move(outcome->retired->progress),
      {}};
}
#endif

llvm::Expected<std::uint64_t>
completionDelay(const loom::sim::SpatialEngineBoundaryResult &result) {
#if defined(LOOM_GEM5_SPATIAL_ENGINE_DFG)
  return 0;
#else
  if (!result.progressObservations.graphRetirementVisible)
    return 0;
  const loom::evaluation::ExactRatio cycles =
      result.progressObservations.graphRetirementVisible->referenceCycle;
  if (cycles.denominator() != 1 ||
      cycles.numerator() >
          std::numeric_limits<std::uint64_t>::max() / ticksPerCycle)
    return invalid("Spatial completion delay is not an integral gem5 tick");
  return cycles.numerator() * ticksPerCycle;
#endif
}

} // namespace

int main(int argc, char **argv) {
  llvm::InitLLVM initialization(argc, argv);
  llvm::cl::ParseCommandLineOptions(argc, argv);
  if (maximumWork == 0 || ticksPerCycle == 0 || maximumInvocations == 0)
    return report(invalid("work, timing, and invocation limits must be "
                          "positive"));

  loom::ArtifactStore store(artifactStorePath);
  auto workload = root(workloadIdentity, loom::sim::simulationWorkloadSchema);
  auto dataflow = root(dataflowIdentity, dataflow::canonicalDataflowSchema);
  if (!workload)
    return report(workload.takeError());
  if (!dataflow)
    return report(dataflow.takeError());
  auto importedWorkload =
      loom::sim::importSpatialSimulationWorkload(*workload, store);
  if (!importedWorkload)
    return report(importedWorkload.takeError());
  if (importedWorkload->dataflow.identity() != dataflow->artifact)
    return report(invalid("Spatial inputs name a foreign Dataflow owner"));
  std::optional<loom::sim::CanonicalSimulationRuntimeInput> staticRuntime;
  if (!runtimeInputIdentity.empty()) {
    auto runtime =
        root(runtimeInputIdentity, loom::sim::simulationRuntimeInputSchema);
    if (!runtime)
      return report(runtime.takeError());
    auto imported = loom::sim::importSpatialSimulationRuntimeInput(
        *runtime, *importedWorkload, store);
    if (!imported)
      return report(imported.takeError());
    staticRuntime.emplace(std::move(*imported));
  }
#if defined(LOOM_GEM5_SPATIAL_ENGINE_DFG)
  const auto *spatialWorkload = importedWorkload->workload.spatial();
  if (!spatialWorkload)
    return report(invalid("Spatial workload lost its typed payload"));
  auto prepared = loom::sim::prepareDfgExecution(importedWorkload->dataflow,
                                                 spatialWorkload->launchRef);
#else
  if (fabricIdentity.empty() || spatialMappingIdentity.empty())
    return report(invalid("CGRA engine owner references are not total"));
  auto fabric = root(fabricIdentity, loom::fabric::fabricArtifactSchema);
  auto mapping =
      root(spatialMappingIdentity, loom::mapping::mappingArtifactSchema);
  if (!fabric)
    return report(fabric.takeError());
  if (!mapping)
    return report(mapping.takeError());
  auto prepared =
      loom::sim::prepareCgraExecution(*dataflow, *fabric, *mapping, store);
#endif
  if (!prepared)
    return report(prepared.takeError());
  auto expectedLaunch = readFile(expectedLaunchPath);
  if (!expectedLaunch)
    return report(expectedLaunch.takeError());
  auto channelBytes = readFile(channelProjectionPath);
  if (!channelBytes)
    return report(channelBytes.takeError());
  auto channels =
      loom::runtime::decodeGem5SpatialChannelProjection(*channelBytes);
  if (!channels)
    return report(channels.takeError());
  auto channelContexts = importChannelInputContexts(*channels, store);
  if (!channelContexts)
    return report(channelContexts.takeError());

  auto server = openServer();
  if (!server)
    return report(server.takeError());
  const int connection = ::accept(*server, nullptr, nullptr);
  ::close(*server);
  if (connection < 0)
    return report(invalid("cannot accept the bridge connection"));
  for (std::uint64_t sequence = 0; sequence != maximumInvocations; ++sequence) {
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
            message->payload, launch, launchDiagnostic) ||
        launch.staticLaunch != *expectedLaunch) {
      ::close(connection);
      return report(
          invalid("bridge launch does not match the exact Deployment: " +
                  launchDiagnostic));
    }

    std::optional<loom::runtime::SpatialInvocationWire> invocation;
    std::optional<loom::sim::CanonicalSimulationRuntimeInput> invocationRuntime;
    if (launch.invocation.empty()) {
      if (!staticRuntime) {
        ::close(connection);
        return report(invalid("static launch has no Spatial runtime input"));
      }
    } else {
      if (staticRuntime) {
        ::close(connection);
        return report(
            invalid("dynamic invocation has a competing static runtime input"));
      }
      loom::runtime::SpatialInvocationWire wire;
      std::string diagnostic;
      if (!loom::runtime::decodeSpatialInvocationWire(launch.invocation, wire,
                                                      diagnostic)) {
        ::close(connection);
        return report(invalid(diagnostic));
      }
      auto runtime = loom::sim::materializeSpatialInvocationRuntimeInput(
          *importedWorkload, wire);
      if (!runtime) {
        ::close(connection);
        return report(runtime.takeError());
      }
      invocation = std::move(wire);
      invocationRuntime.emplace(std::move(*runtime));
    }
    const loom::sim::CanonicalSimulationRuntimeInput &baseRuntime =
        invocationRuntime ? *invocationRuntime : *staticRuntime;
    std::uint64_t requestId = 0;
    auto runtime = bindChannelInputs(connection, message->sequence, *channels,
                                     *channelContexts, *importedWorkload,
                                     baseRuntime, requestId);
    if (!runtime) {
      ::close(connection);
      return report(runtime.takeError());
    }
#if defined(LOOM_GEM5_SPATIAL_ENGINE_DFG)
    auto result = runDfg(*prepared, *importedWorkload, *runtime);
#else
    auto result = runCgra(*prepared, *importedWorkload, *runtime);
#endif
    if (!result) {
      ::close(connection);
      return report(result.takeError());
    }
    auto encoded = loom::sim::encodeSpatialEngineBoundaryResult(
        *result, *importedWorkload, *runtime);
    if (!encoded) {
      ::close(connection);
      return report(encoded.takeError());
    }
    if (llvm::Error error = publishChannelOutputs(
            connection, message->sequence, *channels, *encoded, requestId)) {
      ::close(connection);
      return report(std::move(error));
    }
    auto delay = completionDelay(*result);
    if (!delay) {
      ::close(connection);
      return report(delay.takeError());
    }
    if (invocation) {
      auto writes = loom::sim::projectSpatialInvocationResultWrites(
          *invocation, *importedWorkload, result->functionalObservations);
      if (!writes) {
        ::close(connection);
        return report(writes.takeError());
      }
      if (llvm::Error error = publishInvocationResults(
              connection, message->sequence, *writes, *delay, requestId)) {
        ::close(connection);
        return report(std::move(error));
      }
    }
    const std::uint32_t status =
        std::holds_alternative<loom::sim::RetiredExecution>(result->terminal)
            ? 0
            : 1;
    std::vector<std::uint8_t> completionResult =
        loom::runtime::encodeSpatialInvocationResultWire(
            {launch.invocation, std::move(*encoded)});
    const loom::runtime::Gem5BridgeCompletion completion{
        invocation ? 0 : *delay, status, std::move(completionResult)};
    if (llvm::Error error = writeMessage(
            connection,
            {loom::runtime::Gem5BridgeMessageKind::Completion,
             message->sequence,
             loom::runtime::encodeGem5BridgeCompletion(completion)})) {
      ::close(connection);
      return report(std::move(error));
    }
  }
  ::close(connection);
  ::unlink(socketPath.c_str());
  return 0;
}
