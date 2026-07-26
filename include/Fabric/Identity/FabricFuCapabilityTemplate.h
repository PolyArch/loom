#ifndef LOOM_FABRIC_IDENTITY_FABRICFUCAPABILITYTEMPLATE_H
#define LOOM_FABRIC_IDENTITY_FABRICFUCAPABILITYTEMPLATE_H

#include "Fabric/Identity/FabricRefs.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <utility>
#include <variant>
#include <vector>

namespace loom::fabric {

enum class FabricFuCapabilityTemplateEndpointKind : std::uint32_t {
  BoundaryPort = 0,
  NodePort = 1,
};

struct FabricFuCapabilityTemplateEndpointRef {
  using Payload = std::variant<FabricFuTemplatePortRef, FabricFuNodePortRef>;

  Payload payload;

  FabricFuCapabilityTemplateEndpointKind kind() const {
    return static_cast<FabricFuCapabilityTemplateEndpointKind>(payload.index());
  }

  static FabricFuCapabilityTemplateEndpointRef
  boundaryPort(FabricFuTemplatePortRef port) {
    return FabricFuCapabilityTemplateEndpointRef{
        Payload(std::in_place_type<FabricFuTemplatePortRef>, std::move(port))};
  }

  static FabricFuCapabilityTemplateEndpointRef
  nodePort(FabricFuNodePortRef port) {
    return FabricFuCapabilityTemplateEndpointRef{
        Payload(std::in_place_type<FabricFuNodePortRef>, std::move(port))};
  }

  friend bool operator==(const FabricFuCapabilityTemplateEndpointRef &lhs,
                         const FabricFuCapabilityTemplateEndpointRef &rhs) {
    return lhs.payload == rhs.payload;
  }
  friend bool operator!=(const FabricFuCapabilityTemplateEndpointRef &lhs,
                         const FabricFuCapabilityTemplateEndpointRef &rhs) {
    return !(lhs == rhs);
  }
};

struct FabricFuCapabilityTemplateEdge {
  FabricFuCapabilityTemplateEndpointRef source;
  FabricFuCapabilityTemplateEndpointRef destination;

  friend bool operator==(const FabricFuCapabilityTemplateEdge &lhs,
                         const FabricFuCapabilityTemplateEdge &rhs) {
    return lhs.source == rhs.source && lhs.destination == rhs.destination;
  }
  friend bool operator!=(const FabricFuCapabilityTemplateEdge &lhs,
                         const FabricFuCapabilityTemplateEdge &rhs) {
    return !(lhs == rhs);
  }
};

struct FabricFuCapabilityTemplateRecord {
  std::vector<FabricFuTemplateNodeRef> activeNodes;
  std::vector<FabricFuCapabilityTemplateEdge> activeEdges;

  friend bool operator==(const FabricFuCapabilityTemplateRecord &lhs,
                         const FabricFuCapabilityTemplateRecord &rhs) {
    return lhs.activeNodes == rhs.activeNodes &&
           lhs.activeEdges == rhs.activeEdges;
  }
  friend bool operator!=(const FabricFuCapabilityTemplateRecord &lhs,
                         const FabricFuCapabilityTemplateRecord &rhs) {
    return !(lhs == rhs);
  }
};

/// Sorts authoring-order nodes and edges into their canonical semantic order.
/// Empty node sets, duplicate members, mixed FU owners, invalid edge
/// directions, and edges naming inactive nodes are rejected rather than
/// silently repaired.
llvm::Expected<FabricFuCapabilityTemplateRecord>
normalizeFabricFuCapabilityTemplateRecord(
    FabricFuCapabilityTemplateRecord record);

/// Canonical record bytes: u64be node count, node references, u64be edge
/// count, then directed endpoint pairs. Endpoint variants use u32be tags.
llvm::Expected<std::vector<std::uint8_t>>
canonicalFabricFuCapabilityTemplateBytes(
    const FabricFuCapabilityTemplateRecord &record);

/// Strict decoding accepts only normalized bytes and rejects trailing data.
llvm::Expected<FabricFuCapabilityTemplateRecord>
decodeFabricFuCapabilityTemplateRecord(llvm::ArrayRef<std::uint8_t> bytes);

/// Normalizes every record, orders records by canonical bytes, and rejects
/// duplicate physical templates. Dense ordinals are their resulting indices.
llvm::Expected<std::vector<FabricFuCapabilityTemplateRecord>>
normalizeFabricFuCapabilityTemplateInventory(
    llvm::ArrayRef<FabricFuCapabilityTemplateRecord> records);

/// Resolves one exact dense reference in O(1) against an already normalized
/// and owner-validated inventory.
llvm::Error validateFabricFuCapabilityTemplateRef(
    llvm::ArrayRef<FabricFuCapabilityTemplateRecord> inventory,
    const FabricFuCapabilityTemplateRef &ref);

} // namespace loom::fabric

#endif // LOOM_FABRIC_IDENTITY_FABRICFUCAPABILITYTEMPLATE_H
