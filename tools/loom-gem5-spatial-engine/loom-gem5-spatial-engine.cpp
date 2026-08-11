#include "Common/ArtifactStore.h"
#include "Common/ArtifactText.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Mapping/Artifact/MappingArtifact.h"
#include "Runtime/Gem5BridgeWire.h"
#include "Simulator/CGRAAdmission.h"
#include "Simulator/CGRASimulator.h"
#include "Simulator/DFGSimulator.h"
#include "Simulator/SimulationArtifacts.h"
#include "Simulator/SimulationExecution.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include <cerrno>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <utility>
#include <vector>

namespace {

llvm::cl::opt<std::string> artifactStorePath(
    "artifact-store", llvm::cl::desc("Invocation package ArtifactStore"),
    llvm::cl::Required);
llvm::cl::opt<std::string> socketPath(
    "socket", llvm::cl::desc("Invocation-local bridge socket"),
    llvm::cl::Required);
llvm::cl::opt<std::string> expectedLaunchPath(
    "expected-launch", llvm::cl::desc("Exact launch payload"),
    llvm::cl::Required);
llvm::cl::opt<std::string> workloadIdentity(
    "workload", llvm::cl::desc("Spatial workload ArtifactIdentity"),
    llvm::cl::Required);
llvm::cl::opt<std::string> runtimeInputIdentity(
    "runtime-input", llvm::cl::desc("Spatial runtime input ArtifactIdentity"),
    llvm::cl::Required);
llvm::cl::opt<std::string> dataflowIdentity(
    "dataflow", llvm::cl::desc("Canonical Dataflow ArtifactIdentity"),
    llvm::cl::Required);
llvm::cl::opt<std::string> fabricIdentity(
    "fabric", llvm::cl::desc("Fabric ArtifactIdentity"),
    llvm::cl::init(""));
llvm::cl::opt<std::string> spatialMappingIdentity(
    "spatial-mapping", llvm::cl::desc("SpatialMapping ArtifactIdentity"),
    llvm::cl::init(""));
llvm::cl::opt<std::uint64_t> maximumWork(
    "maximum-work", llvm::cl::desc("Engine semantic work limit"),
    llvm::cl::init(100000));
llvm::cl::opt<std::uint64_t> ticksPerCycle(
    "ticks-per-cycle", llvm::cl::desc("gem5 ticks per Spatial cycle"),
    llvm::cl::init(1000));

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
    return invalid("cannot read '" + path + "': " +
                   buffer.getError().message());
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
  if (::bind(server, reinterpret_cast<sockaddr *>(&address),
             sizeof(address)) != 0 ||
      ::listen(server, 1) != 0) {
    ::close(server);
    return invalid("cannot publish the bridge socket");
  }
  return server;
}

llvm::Expected<loom::runtime::Gem5BridgeMessage>
readMessage(int connection) {
  std::vector<std::uint8_t> bytes(
      loom::runtime::gem5BridgeWireHeaderBytes);
  if (!readAll(connection, bytes.data(), bytes.size()))
    return invalid("bridge disconnected before the message header");
  const std::uint64_t payloadSize =
      loom::runtime::detail::readGem5BridgeU64(bytes.data() + 16);
  if (payloadSize > std::numeric_limits<std::size_t>::max() ||
      payloadSize > 64ULL * 1024ULL * 1024ULL)
    return invalid("bridge message exceeds the engine limit");
  const std::size_t headerSize = bytes.size();
  bytes.resize(headerSize + static_cast<std::size_t>(payloadSize));
  if (payloadSize != 0 &&
      !readAll(connection, bytes.data() + headerSize,
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

llvm::Expected<loom::sim::SpatialEngineBoundaryResult>
runDfg(const loom::sim::ImportedSpatialSimulationInputs &inputs) {
  auto retired = loom::sim::simulateRetiredDfgWorkload(
      inputs.dataflow, inputs.workload, inputs.runtimeInput, maximumWork);
  if (!retired)
    return retired.takeError();
  auto zero = loom::evaluation::ExactRatio::get(0, 1);
  auto retirement = loom::evaluation::ExactRatio::get(
      retired->report.wavefrontSteps, 1);
  if (!zero)
    return zero.takeError();
  if (!retirement)
    return retirement.takeError();
  const loom::sim::SpatialEventCoordinate launch{std::move(*zero), 0};
  const loom::sim::SpatialEventCoordinate terminal{std::move(*retirement), 0};
  return loom::sim::SpatialEngineBoundaryResult{
      loom::sim::RetiredExecution{}, std::move(retired->observations),
      {launch, terminal, terminal}, {}};
}

llvm::Expected<loom::sim::SpatialEngineBoundaryResult>
runCgra(const loom::sim::ImportedSpatialSimulationInputs &inputs,
        const loom::ArtifactStore &store) {
  if (fabricIdentity.empty() || spatialMappingIdentity.empty())
    return invalid("CGRA engine owner references are not total");
  auto fabric = root(fabricIdentity, loom::fabric::fabricArtifactSchema);
  auto mapping = root(spatialMappingIdentity,
                      loom::mapping::mappingArtifactSchema);
  if (!fabric)
    return fabric.takeError();
  if (!mapping)
    return mapping.takeError();
  auto dataflow = root(dataflowIdentity, dataflow::canonicalDataflowSchema);
  if (!dataflow)
    return dataflow.takeError();
  auto prepared = loom::sim::prepareCgraExecution(
      *dataflow, *fabric, *mapping, store);
  if (!prepared)
    return prepared.takeError();
  auto outcome = loom::sim::simulateCgraWorkload(
      *prepared, inputs.workload, inputs.runtimeInput, maximumWork);
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
      std::move(outcome->retired->progress), {}};
}

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
  if (maximumWork == 0 || ticksPerCycle == 0)
    return report(invalid("work and timing limits must be positive"));

  loom::ArtifactStore store(artifactStorePath);
  auto workload = root(workloadIdentity, loom::sim::simulationWorkloadSchema);
  auto runtime =
      root(runtimeInputIdentity, loom::sim::simulationRuntimeInputSchema);
  auto dataflow = root(dataflowIdentity, dataflow::canonicalDataflowSchema);
  if (!workload)
    return report(workload.takeError());
  if (!runtime)
    return report(runtime.takeError());
  if (!dataflow)
    return report(dataflow.takeError());
  auto inputs =
      loom::sim::importSpatialSimulationInputs(*workload, *runtime, store);
  if (!inputs)
    return report(inputs.takeError());
  if (inputs->dataflow.identity() != dataflow->artifact)
    return report(invalid("Spatial inputs name a foreign Dataflow owner"));
  auto expectedLaunch = readFile(expectedLaunchPath);
  if (!expectedLaunch)
    return report(expectedLaunch.takeError());

  auto server = openServer();
  if (!server)
    return report(server.takeError());
  const int connection = ::accept(*server, nullptr, nullptr);
  ::close(*server);
  if (connection < 0)
    return report(invalid("cannot accept the bridge connection"));
  auto message = readMessage(connection);
  if (!message) {
    ::close(connection);
    return report(message.takeError());
  }
  if (message->kind != loom::runtime::Gem5BridgeMessageKind::SpatialLaunch ||
      message->sequence != 0 || message->payload != *expectedLaunch) {
    ::close(connection);
    return report(invalid("bridge launch does not match the exact Deployment"));
  }

#if defined(LOOM_GEM5_SPATIAL_ENGINE_DFG)
  auto result = runDfg(*inputs);
#else
  auto result = runCgra(*inputs, store);
#endif
  if (!result) {
    ::close(connection);
    return report(result.takeError());
  }
  auto encoded = loom::sim::encodeSpatialEngineBoundaryResult(
      *result, *workload, *runtime, store);
  if (!encoded) {
    ::close(connection);
    return report(encoded.takeError());
  }
  auto delay = completionDelay(*result);
  if (!delay) {
    ::close(connection);
    return report(delay.takeError());
  }
  const std::uint32_t status =
      std::holds_alternative<loom::sim::RetiredExecution>(result->terminal)
          ? 0
          : 1;
  const loom::runtime::Gem5BridgeCompletion completion{
      *delay, status, std::move(*encoded)};
  llvm::Error writeError = writeMessage(
      connection,
      {loom::runtime::Gem5BridgeMessageKind::Completion, message->sequence,
       loom::runtime::encodeGem5BridgeCompletion(completion)});
  ::close(connection);
  ::unlink(socketPath.c_str());
  if (writeError)
    return report(std::move(writeError));
  return 0;
}
