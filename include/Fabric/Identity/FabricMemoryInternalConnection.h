#ifndef LOOM_FABRIC_IDENTITY_FABRICMEMORYINTERNALCONNECTION_H
#define LOOM_FABRIC_IDENTITY_FABRICMEMORYINTERNALCONNECTION_H

#include "Fabric/Identity/FabricRefs.h"

#include "llvm/ADT/ArrayRef.h"

#include <cstdint>

namespace loom::fabric {

enum class FabricMemoryInternalConnectionUseKind : std::uint8_t {
  Producer,
  Consumer,
};

/// One selected role use of an occurrence-local physical internal connection.
/// Role schemas and endpoint eligibility are validated by their respective
/// Fabric owners before constructing this relation.
struct FabricMemoryInternalConnectionUse final {
  FabricMemoryOccurrenceRef occurrence;
  FabricOrdinal connection = 0;
  FabricMemoryInternalConnectionUseKind kind =
      FabricMemoryInternalConnectionUseKind::Consumer;
};

enum class FabricMemoryInternalConnectionClosure : std::uint8_t {
  Closed,
  Open,
  MultipleProducers,
};

/// The sole Fabric-owned closure predicate for selected occurrence-local
/// internal connections. An unused connection is absent, while every selected
/// connection has exactly one producing role and at least one consuming role.
FabricMemoryInternalConnectionClosure
deriveFabricMemoryInternalConnectionClosure(
    llvm::ArrayRef<FabricMemoryInternalConnectionUse> uses);

} // namespace loom::fabric

#endif // LOOM_FABRIC_IDENTITY_FABRICMEMORYINTERNALCONNECTION_H
