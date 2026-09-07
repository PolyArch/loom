#include "runtime/gem5/loom_spatial_engine_session.hh"

#include "Runtime/Gem5BridgeSocket.h"
#include "runtime/gem5/loom_spatial_bridge.hh"

#include "base/logging.hh"
#include "sim/sim_exit.hh"

#include <algorithm>
#include <cstring>
#include <limits>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

namespace gem5 {
namespace {
constexpr char advanceExitCause[] = "Loom Spatial causal advance";
}

LoomSpatialEngineSession::LoomSpatialEngineSession(const Params &params)
    : SimObject(params), socketPath(params.engine_socket),
      bridgeCount(params.bridge_count) {
  panic_if(socketPath.empty() || bridgeCount == 0,
           "Loom Spatial engine session needs a socket and bridges");
}

LoomSpatialEngineSession::~LoomSpatialEngineSession() {
  if (socket >= 0)
    ::close(socket);
}

void LoomSpatialEngineSession::registerBridge(std::uint64_t ordinal,
                                              LoomSpatialBridge &bridge) {
  panic_if(bridges.size() >= bridgeCount ||
               !bridges.emplace(ordinal, &bridge).second,
           "Loom Spatial engine bridge membership is not unique");
}

bool LoomSpatialEngineSession::isAdvanceExit(const std::string &cause) const {
  return cause == advanceExitCause;
}

void LoomSpatialEngineSession::submit(
    loom::runtime::Gem5BridgeMessage message) {
  panic_if(bridges.count(message.bridgeSessionOrdinal) == 0,
           "Loom Spatial causal input names a foreign bridge");
  panic_if(generation == std::numeric_limits<std::uint64_t>::max(),
           "Loom Spatial causal generation exhausted");
  pending.push_back({++generation, curTick(), {std::move(message)}});
  if (pending.size() == 1)
    exitSimLoopNow(advanceExitCause);
}

void LoomSpatialEngineSession::connectEngine() {
  if (socket >= 0)
    return;
  fatal_if(socketPath.size() >= sizeof(sockaddr_un::sun_path),
           "Loom Spatial engine socket path is too long");
  socket = ::socket(AF_UNIX, SOCK_STREAM, 0);
  fatal_if(socket < 0, "Cannot create Loom Spatial engine connection");
  sockaddr_un address{};
  address.sun_family = AF_UNIX;
  std::memcpy(address.sun_path, socketPath.c_str(), socketPath.size() + 1);
  fatal_if(::connect(socket, reinterpret_cast<sockaddr *>(&address),
                     sizeof(address)) != 0,
           "Cannot connect to Loom Spatial engine");
}

void LoomSpatialEngineSession::advance() {
  if (pending.empty())
    return;
  EventQueue::ScopedMigration migrate(eventQueue());
  fatal_if(bridges.size() != bridgeCount,
           "Loom Spatial engine session has incomplete bridge membership");
  connectEngine();
  std::uint64_t messageLimit = 0;
  for (const auto &[ordinal, bridge] : bridges)
    messageLimit = std::max(messageLimit, bridge->maximumMessageBytes);
  while (!pending.empty()) {
    const auto &input = pending.front();
    fatal_if(curTick() != input.causalTick,
             "gem5 advanced beyond an unresolved Spatial causal input");
    auto &initiator = *bridges.at(input.messages.front().bridgeSessionOrdinal);
    const auto accounting = initiator.beginCallbackAccounting();
    std::string diagnostic;
    fatal_if(!loom::runtime::writeGem5BridgeAdvance(socket, input, diagnostic),
             "Loom Spatial causal input failed: %s", diagnostic.c_str());
    initiator.startEngineWait();
    loom::runtime::Gem5BridgeAdvance response;
    const bool received = loom::runtime::readGem5BridgeAdvance(
        socket,
        bridgeCount * loom::runtime::gem5BridgeMaximumBridgeActionsPerAdvance,
        messageLimit, response, diagnostic);
    initiator.finishEngineWait();
    fatal_if(!received, "Loom Spatial causal response failed: %s",
             diagnostic.c_str());
    fatal_if(response.generation != input.generation ||
                 response.causalTick != input.causalTick,
             "Loom Spatial causal response has stale or foreign identity");
    // One Bridge may receive several concurrent memory transactions in one
    // response; every other boundary action stays unique per Bridge because
    // the engine emits it only once per causal advance.
    for (const auto &message : response.messages) {
      const auto found = bridges.find(message.bridgeSessionOrdinal);
      fatal_if(found == bridges.end(),
               "Loom Spatial response names a foreign bridge");
      fatal_if(message.payload.size() +
                       loom::runtime::gem5BridgeWireHeaderBytes >
                   found->second->maximumMessageBytes,
               "Loom Spatial response exceeds its target bridge limit");
    }
    const Tick causalTick = input.causalTick;
    pending.pop_front();
    for (const auto &message : response.messages)
      bridges.at(message.bridgeSessionOrdinal)
          ->acceptBoundary(message, causalTick);
    initiator.finishCallbackAccounting(accounting);
  }
}

} // namespace gem5
