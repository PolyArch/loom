#ifndef LOOM_FABRIC_IDENTITY_FABRICMEMORYSERVICEHANDSHAKE_H
#define LOOM_FABRIC_IDENTITY_FABRICMEMORYSERVICEHANDSHAKE_H

#include "Fabric/Identity/FabricHandshake.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>
#include <variant>
#include <vector>

namespace loom::fabric {

enum class MemoryServiceHandshakeChannel : std::uint8_t {
  Request,
  Response,
};

struct MemoryServiceHandshakeSignalRef final {
  FabricMemoryEndpointRef endpoint;
  MemoryServiceHandshakeChannel channel =
      MemoryServiceHandshakeChannel::Request;
  HandshakeSignalKind signal = HandshakeSignalKind::Valid;

  friend bool operator==(const MemoryServiceHandshakeSignalRef &lhs,
                         const MemoryServiceHandshakeSignalRef &rhs) {
    return lhs.endpoint == rhs.endpoint && lhs.channel == rhs.channel &&
           lhs.signal == rhs.signal;
  }
};

using FabricMemoryHandshakeServiceTarget =
    std::variant<LocalMemoryServiceRef, ManagerEndpointRef>;

/// The exact service target selected for one operation-row handshake plan.
/// This is an invocation-local projection of SpatialMapping and has no wire
/// encoding or identity independent of that Mapping.
struct FabricMemoryOperationServiceHandshakeSelection final {
  FabricMemoryHandshakePlacement placement;
  FabricMemoryCapabilityAlternativeRef capability;
  FabricMemoryHandshakeServiceTarget target;
};

/// Every configured target that one subordinate provider decode may select.
/// Runtime address values choose among these rows, so the complete canonical
/// target set participates in structural handshake closure.
struct FabricMemoryProviderServiceHandshakeSelection final {
  SubordinateEndpointRef subordinate;
  std::vector<FabricMemoryHandshakeServiceTarget> targets;
};

struct FabricMemoryServiceHandshakeSelection final {
  std::vector<FabricMemoryOperationServiceHandshakeSelection> operations;
  std::vector<FabricMemoryProviderServiceHandshakeSelection> providers;
};

/// One Module-boundary signal after transport and memory-service planes have
/// been unified. `memoryChannel` is absent exactly for a token-transport
/// boundary and present exactly for a memory-service boundary.
struct ModuleBoundaryHandshakeSignalRef final {
  FabricModuleBoundaryEndpointRef boundary;
  std::optional<MemoryServiceHandshakeChannel> memoryChannel;
  HandshakeSignalKind signal = HandshakeSignalKind::Valid;

  friend bool operator==(const ModuleBoundaryHandshakeSignalRef &lhs,
                         const ModuleBoundaryHandshakeSignalRef &rhs) {
    return lhs.boundary == rhs.boundary &&
           lhs.memoryChannel == rhs.memoryChannel && lhs.signal == rhs.signal;
  }
};

struct ModuleBoundaryHandshakeDependencyArc final {
  ModuleBoundaryHandshakeSignalRef source;
  ModuleBoundaryHandshakeSignalRef destination;
};

/// Rebuilds the complete selected Module-local graph, including the ordinary
/// transport owner graph, memory operation dispatch, subordinate forwarding,
/// the root-owned memory network, and both boundary planes. A cycle is
/// rejected before boundary reachability is returned.
llvm::Expected<std::vector<ModuleBoundaryHandshakeDependencyArc>>
deriveSelectedModuleBoundaryHandshakeReachability(
    const FabricArtifactView &view,
    const FabricHandshakeSelection &transportSelection,
    const FabricMemoryServiceHandshakeSelection &memorySelection,
    const FabricHandshakeContext &context,
    ExecutionControlView executionControl = {});

llvm::Error verifySelectedMemoryServiceHandshakeAcyclic(
    const FabricArtifactView &view,
    const FabricHandshakeSelection &transportSelection,
    const FabricMemoryServiceHandshakeSelection &memorySelection,
    const FabricHandshakeContext &context,
    ExecutionControlView executionControl = {});

} // namespace loom::fabric

#endif // LOOM_FABRIC_IDENTITY_FABRICMEMORYSERVICEHANDSHAKE_H
