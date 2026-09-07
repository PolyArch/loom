#include "Common/ArtifactStore.h"
#include "Common/ArtifactText.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Mapping/Artifact/MappingArtifact.h"
#include "Simulator/SimulationArtifacts.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include <cerrno>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <optional>
#include <poll.h>
#include <string>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <utility>
#include <vector>

#include "Runtime/Gem5BridgeSocket.h"
#include "SpatialEngineSession.h"

namespace {
using namespace loom::gem5engine;

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
    llvm::cl::desc("Number of physical bridges sharing this System session"),
    llvm::cl::init(1));
llvm::cl::opt<std::string> performanceProfilePath(
    "performance-profile",
    llvm::cl::desc("CGRA engine active performance profile output"),
    llvm::cl::init(""));

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

} // namespace

int main(int argc, char **argv) {
  llvm::InitLLVM initialization(argc, argv);
  llvm::cl::ParseCommandLineOptions(argc, argv);
  if (maximumWork == 0 || ticksPerCycle == 0 || maximumInvocations == 0 ||
      bridgeCount == 0 || bridgeCount > std::numeric_limits<int>::max())
    return report(invalid("work, timing, and invocation limits must be "
                          "positive"));
#if defined(LOOM_GEM5_SPATIAL_ENGINE_DFG)
  if (!performanceProfilePath.empty())
    return report(
        invalid("DFG engine does not provide a CGRA performance profile"));
#endif

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
    if (importedWorkload->dataflow->identity() != dataflow->artifact)
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
      auto built = loom::sim::prepareDfgExecution(*importedWorkload->dataflow,
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
  auto session = SpatialEngineSession::create(
      std::move(entries), std::move(preparedExecutions),
      {maximumWork, ticksPerCycle, maximumInvocations}, performanceProfilePath);
  if (!session)
    return report(session.takeError());
  auto server = openServer();
  if (!server)
    return report(server.takeError());
  const int connection = ::accept(*server, nullptr, nullptr);
  ::close(*server);
  if (connection < 0)
    return report(invalid("cannot accept the engine session connection"));
  while (true) {
    pollfd descriptor{connection, POLLIN, 0};
    int ready;
    do {
      ready = ::poll(&descriptor, 1, -1);
    } while (ready < 0 && errno == EINTR);
    if (ready < 0) {
      ::close(connection);
      return report(invalid("cannot await a causal engine input"));
    }
    if ((descriptor.revents & POLLIN) == 0)
      break;
    // Clean EOF ends the existing externally managed engine lifetime.
    std::uint8_t firstByte;
    const ssize_t available = ::recv(connection, &firstByte, 1, MSG_PEEK);
    if (available == 0)
      break;
    loom::runtime::Gem5BridgeAdvance input;
    std::string diagnostic;
    if (!loom::runtime::readGem5BridgeAdvance(
            connection, 1, loom::runtime::gem5BridgeDefaultMaximumMessageBytes,
            input, diagnostic)) {
      ::close(connection);
      return report(invalid(diagnostic));
    }
    auto response = (*session)->advance(input);
    if (!response) {
      ::close(connection);
      return report(response.takeError());
    }
    if (!loom::runtime::writeGem5BridgeAdvance(connection, *response,
                                               diagnostic)) {
      ::close(connection);
      return report(invalid(diagnostic));
    }
  }
  ::close(connection);
  ::unlink(socketPath.c_str());
  return 0;
}
