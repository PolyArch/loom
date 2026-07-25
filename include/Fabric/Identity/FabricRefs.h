#ifndef LOOM_FABRIC_IDENTITY_FABRICREFS_H
#define LOOM_FABRIC_IDENTITY_FABRICREFS_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <string>
#include <system_error>

namespace loom {
namespace fabric {

/// Artifact-global entity identifier of one finalized Fabric Hardware
/// Description. All entity kinds share this single unsigned 64-bit namespace;
/// native PnR index width never narrows the persistent range.
using FabricEntityId = std::uint64_t;

/// Owner-relative structural ordinal. Its meaning is fixed by the owning
/// reference family, and it is only ever interpreted against the canonical
/// inventory that family selects.
using FabricOrdinal = std::uint64_t;

#define LOOM_FABRIC_ENTITY(Name, Keyword) Name,
enum class FabricEntityKind : std::uint32_t {
#include "Fabric/Identity/FabricRefs.def"
};

#define LOOM_FABRIC_ROOT_KIND(Name, Keyword) Name,
enum class FabricRootKind : std::uint32_t {
#include "Fabric/Identity/FabricRefs.def"
};

#define LOOM_FABRIC_FU_NODE_KIND(Name, Keyword) Name,
enum class FabricFuNodeKind : std::uint32_t {
#include "Fabric/Identity/FabricRefs.def"
};

#define LOOM_FABRIC_PORT_DIRECTION(Name, Keyword) Name,
enum class FabricPortDirection : std::uint32_t {
#include "Fabric/Identity/FabricRefs.def"
};

#define LOOM_FABRIC_MEMORY_SERVICE(Name, Keyword, Member, Type) Name,
enum class FabricMemoryServiceKind : std::uint32_t {
#include "Fabric/Identity/FabricRefs.def"
};

#define LOOM_FABRIC_FIFO_MODE(Name, Keyword) Name,
enum class FabricFifoTraversalMode : std::uint32_t {
#include "Fabric/Identity/FabricRefs.def"
};

#define LOOM_FABRIC_REGISTER_FIFO_PATH_ROLE(Name, Keyword) Name,
enum class FabricRegisterFifoPathRole : std::uint32_t {
#include "Fabric/Identity/FabricRefs.def"
};

#define LOOM_FABRIC_TRAVERSAL(Name, Keyword, Member, Type) Name,
enum class FabricPhysicalTraversalKind : std::uint32_t {
#include "Fabric/Identity/FabricRefs.def"
};

#define LOOM_FABRIC_INVENTORY(Name, Keyword) Name,
enum class FabricInventoryKind : std::uint32_t {
#include "Fabric/Identity/FabricRefs.def"
};

#define LOOM_FABRIC_TRANSPORT_OWNER(Name, Member, Type) Name,
enum class FabricTransportEndpointOwnerKind : std::uint32_t {
#include "Fabric/Identity/FabricRefs.def"
};

#define LOOM_FABRIC_MEMORY_OWNER(Name, Member, Type) Name,
enum class FabricMemoryEndpointOwnerKind : std::uint32_t {
#include "Fabric/Identity/FabricRefs.def"
};

#define LOOM_FABRIC_INVENTORY_OWNER(Name, Member, Type) Name,
enum class FabricInventoryOwnerKind : std::uint32_t {
#include "Fabric/Identity/FabricRefs.def"
};

#define LOOM_FABRIC_REF_ERROR(Name, Keyword) Name,
/// Typed classification of an invalid persistent reference. A well-formed
/// reference whose target cannot support the requested software operation is
/// a Mapping feasibility failure and is deliberately not classified here.
enum class FabricRefErrorKind : std::uint32_t {
#include "Fabric/Identity/FabricRefs.def"
};

/// Canonical keyword of each closed sum: one overload set, each projecting
/// the same catalog declaration. These are the only spellings the strict text
/// codec accepts and the only ones it prints.
llvm::StringRef fabricRefKeyword(FabricRootKind kind);
llvm::StringRef fabricRefKeyword(FabricFuNodeKind kind);
llvm::StringRef fabricRefKeyword(FabricPortDirection direction);
llvm::StringRef fabricRefKeyword(FabricMemoryServiceKind kind);
llvm::StringRef fabricRefKeyword(FabricFifoTraversalMode mode);
llvm::StringRef fabricRefKeyword(FabricRegisterFifoPathRole role);
llvm::StringRef fabricRefKeyword(FabricPhysicalTraversalKind kind);
llvm::StringRef fabricRefKeyword(FabricInventoryKind kind);
llvm::StringRef fabricRefKeyword(FabricRefErrorKind kind);

/// Declared cardinality and diagnostic name of each closed sum, taken from
/// the last enumerator of its one catalog declaration. Both codecs and the
/// importer read these instead of repeating a bound.
#define LOOM_FABRIC_CLOSED_SUM(Enum, LastName, Text)                           \
  inline std::uint32_t fabricClosedBound(Enum) {                               \
    return static_cast<std::uint32_t>(Enum::LastName) + 1;                     \
  }                                                                            \
  inline llvm::StringRef fabricClosedName(Enum) { return Text; }

LOOM_FABRIC_CLOSED_SUM(FabricFuNodeKind, Demux, "FU node kind")
LOOM_FABRIC_CLOSED_SUM(FabricPortDirection, Output, "port direction")
LOOM_FABRIC_CLOSED_SUM(FabricMemoryServiceKind, System, "memory service kind")
LOOM_FABRIC_CLOSED_SUM(FabricFifoTraversalMode, Bypass, "FIFO traversal mode")
LOOM_FABRIC_CLOSED_SUM(FabricRegisterFifoPathRole, Read,
                       "register FIFO path role")
LOOM_FABRIC_CLOSED_SUM(FabricPhysicalTraversalKind, SystemTransferPatternLeg,
                       "physical traversal kind")
LOOM_FABRIC_CLOSED_SUM(FabricTransportEndpointOwnerKind, ExternalBoundary,
                       "transport endpoint owner")
LOOM_FABRIC_CLOSED_SUM(FabricMemoryEndpointOwnerKind, ExternalBoundary,
                       "memory endpoint owner")
LOOM_FABRIC_CLOSED_SUM(FabricInventoryOwnerKind, ExternalBoundary,
                       "inventory owner")

#undef LOOM_FABRIC_CLOSED_SUM

/// Failure of a persistent reference carrying its typed classification.
class FabricRefError : public llvm::ErrorInfo<FabricRefError> {
public:
  static char ID;

  FabricRefError(FabricRefErrorKind kind, std::string message)
      : kind_(kind), message_(std::move(message)) {}

  FabricRefErrorKind kind() const { return kind_; }
  void log(llvm::raw_ostream &os) const override {
    os << fabricRefKeyword(kind_) << ": " << message_;
  }
  std::error_code convertToErrorCode() const override {
    return llvm::inconvertibleErrorCode();
  }

private:
  FabricRefErrorKind kind_;
  std::string message_;
};

llvm::Error makeFabricRefError(FabricRefErrorKind kind,
                               const llvm::Twine &message);

/// Consumes `error` and returns its classification. A non-Fabric error is
/// reported as malformed input rather than silently reclassified.
FabricRefErrorKind takeFabricRefErrorKind(llvm::Error error);

//===---------------------------------------------------------------------===//
// Entity references
//===---------------------------------------------------------------------===//

constexpr llvm::StringLiteral fabricRefKeyword(FabricEntityKind kind) {
  switch (kind) {
#define LOOM_FABRIC_ENTITY(Name, Keyword)                                      \
  case FabricEntityKind::Name:                                                 \
    return llvm::StringLiteral(Keyword);
#include "Fabric/Identity/FabricRefs.def"
  }
  return llvm::StringLiteral("");
}

/// One entity of the artifact-global namespace. The entity kind is a static
/// schema fact, so a typed reference stores only its semantic identifier and
/// two references of different kinds are never interchangeable, whatever
/// their identifiers.
template <FabricEntityKind Kind> class FabricTypedEntityRef {
public:
  static constexpr llvm::StringLiteral familyKeyword =
      fabricRefKeyword(Kind);

  FabricTypedEntityRef() = default;
  explicit constexpr FabricTypedEntityRef(FabricEntityId id) : id_(id) {}

  constexpr FabricEntityId id() const { return id_; }

  friend bool operator==(FabricTypedEntityRef lhs, FabricTypedEntityRef rhs) {
    return lhs.id_ == rhs.id_;
  }
  friend bool operator!=(FabricTypedEntityRef lhs, FabricTypedEntityRef rhs) {
    return !(lhs == rhs);
  }

private:
  FabricEntityId id_ = 0;
};

#define LOOM_FABRIC_ENTITY(Name, Keyword)                                      \
  using Name##Ref = FabricTypedEntityRef<FabricEntityKind::Name>;
#include "Fabric/Identity/FabricRefs.def"

//===---------------------------------------------------------------------===//
// Structural references
//
// Every family below declares its fields exactly once, in the declaration
// order of docs/spec-fabric-identity.md, through `visitFields`. The strict
// text codec, the canonical byte codec, and the importer all walk that one
// declaration.
//===---------------------------------------------------------------------===//

#define LOOM_FABRIC_REF_FIELDS(...)                                            \
  template <typename Self, typename Visitor>                                   \
  static void visitFields(Self &self, Visitor &visitor) {                      \
    __VA_ARGS__                                                                \
  }

// Families with a fixed field list compare exactly the fields their one
// `visitFields` declaration names.
#define LOOM_FABRIC_REF_EQUALITY(Type, ...)                                    \
  inline bool operator==(const Type &lhs, const Type &rhs) {                   \
    return __VA_ARGS__;                                                        \
  }                                                                            \
  inline bool operator!=(const Type &lhs, const Type &rhs) {                   \
    return !(lhs == rhs);                                                      \
  }

/// The one SpatialCore attachment of an AccCore. Its fixed ordinal zero is a
/// schema constant, so the semantic content is the AccCore alone.
struct SpatialCoreOccurrenceRef {
  static constexpr llvm::StringLiteral familyKeyword =
      llvm::StringLiteral("fabric.spatial_core_occurrence");
  AccCoreOccurrenceRef core;

  LOOM_FABRIC_REF_FIELDS(visitor.ref(self.core);)
};

/// The AccCore-resident InstructionCore context. It shares the numeric
/// content of a SpatialCore attachment and remains a different typed domain.
struct InstructionCoreContextRef {
  static constexpr llvm::StringLiteral familyKeyword =
      llvm::StringLiteral("fabric.instruction_core_context");
  AccCoreOccurrenceRef core;

  LOOM_FABRIC_REF_FIELDS(visitor.ref(self.core);)
};

/// The PE-owned resident instruction context.
struct InstructionContextRef {
  static constexpr llvm::StringLiteral familyKeyword =
      llvm::StringLiteral("fabric.instruction_context");
  FabricPeOccurrenceRef pe;
  FabricOrdinal ordinal = 0;

  LOOM_FABRIC_REF_FIELDS(visitor.ref(self.pe); visitor.ordinal(self.ordinal);)
};

/// A node of the FU-internal configured graph of one FU template. Inner nodes
/// are not independently placeable and therefore hold no EntityId.
struct FabricFuTemplateNodeRef {
  static constexpr llvm::StringLiteral familyKeyword =
      llvm::StringLiteral("fabric.fu_template_node");
  FabricFuNodeKind node = FabricFuNodeKind::Op;
  FabricFuTemplateRef fu;
  FabricOrdinal ordinal = 0;

  LOOM_FABRIC_REF_FIELDS(visitor.tag(self.node); visitor.ref(self.fu);
                         visitor.ordinal(self.ordinal);)
};

/// The same node domain of one physical FU occurrence.
struct FabricFuOccurrenceNodeRef {
  static constexpr llvm::StringLiteral familyKeyword =
      llvm::StringLiteral("fabric.fu_occurrence_node");
  FabricFuNodeKind node = FabricFuNodeKind::Op;
  FabricFuOccurrenceRef fu;
  FabricOrdinal ordinal = 0;

  LOOM_FABRIC_REF_FIELDS(visitor.tag(self.node); visitor.ref(self.fu);
                         visitor.ordinal(self.ordinal);)
};

struct FabricFuTemplatePortRef {
  static constexpr llvm::StringLiteral familyKeyword =
      llvm::StringLiteral("fabric.fu_template_port");
  FabricFuTemplateRef fu;
  FabricPortDirection direction = FabricPortDirection::Input;
  FabricOrdinal ordinal = 0;

  LOOM_FABRIC_REF_FIELDS(visitor.ref(self.fu); visitor.tag(self.direction);
                         visitor.ordinal(self.ordinal);)
};

struct FabricFuNodePortRef {
  static constexpr llvm::StringLiteral familyKeyword =
      llvm::StringLiteral("fabric.fu_node_port");
  FabricFuTemplateNodeRef node;
  FabricPortDirection direction = FabricPortDirection::Input;
  FabricOrdinal ordinal = 0;

  LOOM_FABRIC_REF_FIELDS(visitor.ref(self.node); visitor.tag(self.direction);
                         visitor.ordinal(self.ordinal);)
};

struct FabricFuOccurrencePortRef {
  static constexpr llvm::StringLiteral familyKeyword =
      llvm::StringLiteral("fabric.fu_occurrence_port");
  FabricFuOccurrenceRef fu;
  FabricPortDirection direction = FabricPortDirection::Input;
  FabricOrdinal ordinal = 0;

  LOOM_FABRIC_REF_FIELDS(visitor.ref(self.fu); visitor.tag(self.direction);
                         visitor.ordinal(self.ordinal);)
};

struct FabricMemoryOperationPortRef {
  static constexpr llvm::StringLiteral familyKeyword =
      llvm::StringLiteral("fabric.memory_operation_port");
  FabricMemoryOccurrenceRef memory;
  FabricOrdinal ordinal = 0;

  LOOM_FABRIC_REF_FIELDS(visitor.ref(self.memory);
                         visitor.ordinal(self.ordinal);)
};

struct FabricMemoryCapabilityAlternativeRef {
  static constexpr llvm::StringLiteral familyKeyword =
      llvm::StringLiteral("fabric.memory_capability_alternative");
  FabricMemoryOperationPortRef port;
  FabricOrdinal ordinal = 0;

  LOOM_FABRIC_REF_FIELDS(visitor.ref(self.port); visitor.ordinal(self.ordinal);)
};

struct FabricMemoryOperationContextRef {
  static constexpr llvm::StringLiteral familyKeyword =
      llvm::StringLiteral("fabric.memory_operation_context");
  FabricMemoryOperationPortRef port;
  FabricOrdinal ordinal = 0;

  LOOM_FABRIC_REF_FIELDS(visitor.ref(self.port); visitor.ordinal(self.ordinal);)
};

/// A memory service is either the optional Local Memory Service of one memory
/// occurrence or one system memory service entity.
struct FabricMemoryServiceRef {
  static constexpr llvm::StringLiteral familyKeyword =
      llvm::StringLiteral("fabric.memory_service");
  FabricMemoryServiceKind kind = FabricMemoryServiceKind::Local;
  union Payload {
    Payload() : local() {}
#define LOOM_FABRIC_MEMORY_SERVICE(Name, Keyword, Member, Type) Type Member;
#include "Fabric/Identity/FabricRefs.def"
  } payload;

  static FabricMemoryServiceRef local(FabricMemoryOccurrenceRef memory) {
    FabricMemoryServiceRef ref;
    ref.kind = FabricMemoryServiceKind::Local;
    ref.payload.local = memory;
    return ref;
  }
  static FabricMemoryServiceRef system(SystemMemoryServiceRef service) {
    FabricMemoryServiceRef ref;
    ref.kind = FabricMemoryServiceKind::System;
    ref.payload.system = service;
    return ref;
  }
};

inline bool operator==(const FabricMemoryServiceRef &lhs,
                       const FabricMemoryServiceRef &rhs) {
  if (lhs.kind != rhs.kind)
    return false;
  switch (lhs.kind) {
#define LOOM_FABRIC_MEMORY_SERVICE(Name, Keyword, Member, Type)                \
  case FabricMemoryServiceKind::Name:                                          \
    return lhs.payload.Member == rhs.payload.Member;
#include "Fabric/Identity/FabricRefs.def"
  }
  return false;
}
inline bool operator!=(const FabricMemoryServiceRef &lhs,
                       const FabricMemoryServiceRef &rhs) {
  return !(lhs == rhs);
}

struct FabricMemoryServiceRegionRef {
  static constexpr llvm::StringLiteral familyKeyword =
      llvm::StringLiteral("fabric.memory_service_region");
  FabricMemoryServiceRef service;
  FabricOrdinal ordinal = 0;

  LOOM_FABRIC_REF_FIELDS(visitor.ref(self.service);
                         visitor.ordinal(self.ordinal);)
};

struct FabricTransferPatternRef {
  static constexpr llvm::StringLiteral familyKeyword =
      llvm::StringLiteral("fabric.transfer_pattern");
  SystemTransportResourceRef resource;
  FabricOrdinal ordinal = 0;

  LOOM_FABRIC_REF_FIELDS(visitor.ref(self.resource);
                         visitor.ordinal(self.ordinal);)
};

LOOM_FABRIC_REF_EQUALITY(SpatialCoreOccurrenceRef, lhs.core == rhs.core)
LOOM_FABRIC_REF_EQUALITY(InstructionCoreContextRef, lhs.core == rhs.core)
LOOM_FABRIC_REF_EQUALITY(InstructionContextRef,
                         lhs.pe == rhs.pe && lhs.ordinal == rhs.ordinal)
LOOM_FABRIC_REF_EQUALITY(FabricFuTemplateNodeRef,
                         lhs.node == rhs.node && lhs.fu == rhs.fu &&
                             lhs.ordinal == rhs.ordinal)
LOOM_FABRIC_REF_EQUALITY(FabricFuOccurrenceNodeRef,
                         lhs.node == rhs.node && lhs.fu == rhs.fu &&
                             lhs.ordinal == rhs.ordinal)
LOOM_FABRIC_REF_EQUALITY(FabricFuTemplatePortRef,
                         lhs.fu == rhs.fu && lhs.direction == rhs.direction &&
                             lhs.ordinal == rhs.ordinal)
LOOM_FABRIC_REF_EQUALITY(FabricFuNodePortRef,
                         lhs.node == rhs.node &&
                             lhs.direction == rhs.direction &&
                             lhs.ordinal == rhs.ordinal)
LOOM_FABRIC_REF_EQUALITY(FabricFuOccurrencePortRef,
                         lhs.fu == rhs.fu && lhs.direction == rhs.direction &&
                             lhs.ordinal == rhs.ordinal)
LOOM_FABRIC_REF_EQUALITY(FabricMemoryOperationPortRef,
                         lhs.memory == rhs.memory &&
                             lhs.ordinal == rhs.ordinal)
LOOM_FABRIC_REF_EQUALITY(FabricMemoryCapabilityAlternativeRef,
                         lhs.port == rhs.port && lhs.ordinal == rhs.ordinal)
LOOM_FABRIC_REF_EQUALITY(FabricMemoryOperationContextRef,
                         lhs.port == rhs.port && lhs.ordinal == rhs.ordinal)
LOOM_FABRIC_REF_EQUALITY(FabricMemoryServiceRegionRef,
                         lhs.service == rhs.service &&
                             lhs.ordinal == rhs.ordinal)
LOOM_FABRIC_REF_EQUALITY(FabricTransferPatternRef,
                         lhs.resource == rhs.resource &&
                             lhs.ordinal == rhs.ordinal)

//===---------------------------------------------------------------------===//
// Closed owner unions
//
// A union stores its selected constructor payload directly. The text codec
// spells a union as that payload's own canonical reference, so the
// constructor is recovered from the payload family and never written twice;
// canonical bytes prepend the constructor discriminant.
//===---------------------------------------------------------------------===//

/// Closed union of Mapping-visible owners exposing token terminals.
struct FabricTransportEndpointOwnerRef {
  FabricTransportEndpointOwnerKind kind =
      FabricTransportEndpointOwnerKind::SpatialCoreOccurrence;
  union Payload {
    Payload() : spatialCore() {}
#define LOOM_FABRIC_TRANSPORT_OWNER(Name, Member, Type) Type Member;
#include "Fabric/Identity/FabricRefs.def"
  } payload;

#define LOOM_FABRIC_TRANSPORT_OWNER(Name, Member, Type)                        \
  static FabricTransportEndpointOwnerRef of(const Type &value) {               \
    FabricTransportEndpointOwnerRef owner;                                     \
    owner.kind = FabricTransportEndpointOwnerKind::Name;                       \
    owner.payload.Member = value;                                              \
    return owner;                                                              \
  }
#include "Fabric/Identity/FabricRefs.def"
};

/// Closed union of owners exposing a manager or subordinate memory-service
/// endpoint inventory.
struct FabricMemoryEndpointOwnerRef {
  FabricMemoryEndpointOwnerKind kind =
      FabricMemoryEndpointOwnerKind::SpatialCoreOccurrence;
  union Payload {
    Payload() : spatialCore() {}
#define LOOM_FABRIC_MEMORY_OWNER(Name, Member, Type) Type Member;
#include "Fabric/Identity/FabricRefs.def"
  } payload;

#define LOOM_FABRIC_MEMORY_OWNER(Name, Member, Type)                           \
  static FabricMemoryEndpointOwnerRef of(const Type &value) {                  \
    FabricMemoryEndpointOwnerRef owner;                                        \
    owner.kind = FabricMemoryEndpointOwnerKind::Name;                          \
    owner.payload.Member = value;                                              \
    return owner;                                                              \
  }
#include "Fabric/Identity/FabricRefs.def"
};

/// The one closed constructor catalog behind the resource-state, use-pattern,
/// semantic-config-field, and physical-refinement-domain owner projections.
/// Sharing the catalog avoids four independently drifting copies; the
/// consuming family selects which canonical inventory the ordinal indexes.
struct FabricInventoryOwnerRef {
  FabricInventoryOwnerKind kind = FabricInventoryOwnerKind::ModuleTemplate;
  union Payload {
    Payload() : moduleTemplate() {}
#define LOOM_FABRIC_INVENTORY_OWNER(Name, Member, Type) Type Member;
#include "Fabric/Identity/FabricRefs.def"
  } payload;

#define LOOM_FABRIC_INVENTORY_OWNER(Name, Member, Type)                        \
  static FabricInventoryOwnerRef of(const Type &value) {                       \
    FabricInventoryOwnerRef owner;                                             \
    owner.kind = FabricInventoryOwnerKind::Name;                               \
    owner.payload.Member = value;                                              \
    return owner;                                                              \
  }
#include "Fabric/Identity/FabricRefs.def"
};

inline bool operator==(const FabricTransportEndpointOwnerRef &lhs,
                       const FabricTransportEndpointOwnerRef &rhs) {
  if (lhs.kind != rhs.kind)
    return false;
  switch (lhs.kind) {
#define LOOM_FABRIC_TRANSPORT_OWNER(Name, Member, Type)                        \
  case FabricTransportEndpointOwnerKind::Name:                                 \
    return lhs.payload.Member == rhs.payload.Member;
#include "Fabric/Identity/FabricRefs.def"
  }
  return false;
}
inline bool operator!=(const FabricTransportEndpointOwnerRef &lhs,
                       const FabricTransportEndpointOwnerRef &rhs) {
  return !(lhs == rhs);
}

inline bool operator==(const FabricMemoryEndpointOwnerRef &lhs,
                       const FabricMemoryEndpointOwnerRef &rhs) {
  if (lhs.kind != rhs.kind)
    return false;
  switch (lhs.kind) {
#define LOOM_FABRIC_MEMORY_OWNER(Name, Member, Type)                           \
  case FabricMemoryEndpointOwnerKind::Name:                                    \
    return lhs.payload.Member == rhs.payload.Member;
#include "Fabric/Identity/FabricRefs.def"
  }
  return false;
}
inline bool operator!=(const FabricMemoryEndpointOwnerRef &lhs,
                       const FabricMemoryEndpointOwnerRef &rhs) {
  return !(lhs == rhs);
}

inline bool operator==(const FabricInventoryOwnerRef &lhs,
                       const FabricInventoryOwnerRef &rhs) {
  if (lhs.kind != rhs.kind)
    return false;
  switch (lhs.kind) {
#define LOOM_FABRIC_INVENTORY_OWNER(Name, Member, Type)                        \
  case FabricInventoryOwnerKind::Name:                                         \
    return lhs.payload.Member == rhs.payload.Member;
#include "Fabric/Identity/FabricRefs.def"
  }
  return false;
}
inline bool operator!=(const FabricInventoryOwnerRef &lhs,
                       const FabricInventoryOwnerRef &rhs) {
  return !(lhs == rhs);
}

//===---------------------------------------------------------------------===//
// Endpoint and owner-relative families
//===---------------------------------------------------------------------===//

/// A token terminal of one owner's canonical transport inventory.
struct FabricTransportEndpointRef {
  static constexpr llvm::StringLiteral familyKeyword =
      llvm::StringLiteral("fabric.transport_endpoint");
  FabricTransportEndpointOwnerRef owner;
  FabricOrdinal ordinal = 0;

  LOOM_FABRIC_REF_FIELDS(visitor.ref(self.owner);
                         visitor.ordinal(self.ordinal);)
};

/// A memory-service capability endpoint of one owner's canonical memory
/// inventory. Equal ordinals never make it a token terminal.
struct FabricMemoryEndpointRef {
  static constexpr llvm::StringLiteral familyKeyword =
      llvm::StringLiteral("fabric.memory_endpoint");
  FabricMemoryEndpointOwnerRef owner;
  FabricOrdinal ordinal = 0;

  LOOM_FABRIC_REF_FIELDS(visitor.ref(self.owner);
                         visitor.ordinal(self.ordinal);)
};

#define LOOM_FABRIC_OWNER_RELATIVE_FAMILY(Name, Keyword)                       \
  struct Name {                                                                \
    static constexpr llvm::StringLiteral familyKeyword =                       \
        llvm::StringLiteral(Keyword);                                          \
    FabricInventoryOwnerRef owner;                                             \
    FabricOrdinal ordinal = 0;                                                 \
                                                                               \
    LOOM_FABRIC_REF_FIELDS(visitor.ref(self.owner);                            \
                           visitor.ordinal(self.ordinal);)                     \
  };

LOOM_FABRIC_OWNER_RELATIVE_FAMILY(FabricResourceStateRef, "fabric.resource_state")
LOOM_FABRIC_OWNER_RELATIVE_FAMILY(FabricUsePatternRef, "fabric.use_pattern")
LOOM_FABRIC_OWNER_RELATIVE_FAMILY(FabricSemanticConfigFieldRef,
                                  "fabric.semantic_config_field")
LOOM_FABRIC_OWNER_RELATIVE_FAMILY(FabricPhysicalRefinementDomainRef,
                                  "fabric.refinement_domain")

#undef LOOM_FABRIC_OWNER_RELATIVE_FAMILY

LOOM_FABRIC_REF_EQUALITY(FabricTransportEndpointRef,
                         lhs.owner == rhs.owner && lhs.ordinal == rhs.ordinal)
LOOM_FABRIC_REF_EQUALITY(FabricMemoryEndpointRef,
                         lhs.owner == rhs.owner && lhs.ordinal == rhs.ordinal)
LOOM_FABRIC_REF_EQUALITY(FabricResourceStateRef,
                         lhs.owner == rhs.owner && lhs.ordinal == rhs.ordinal)
LOOM_FABRIC_REF_EQUALITY(FabricUsePatternRef,
                         lhs.owner == rhs.owner && lhs.ordinal == rhs.ordinal)
LOOM_FABRIC_REF_EQUALITY(FabricSemanticConfigFieldRef,
                         lhs.owner == rhs.owner && lhs.ordinal == rhs.ordinal)
LOOM_FABRIC_REF_EQUALITY(FabricPhysicalRefinementDomainRef,
                         lhs.owner == rhs.owner && lhs.ordinal == rhs.ordinal)

//===---------------------------------------------------------------------===//
// Directed physical traversals
//===---------------------------------------------------------------------===//

struct FabricPointConnectionPayload {
  FabricTransportEndpointRef source;
  FabricTransportEndpointRef destination;

  LOOM_FABRIC_REF_FIELDS(visitor.ref(self.source);
                         visitor.ref(self.destination);)
};

struct FabricPeSelectorPayload {
  FabricPeOccurrenceRef owner;
  FabricTransportEndpointRef source;
  FabricTransportEndpointRef destination;

  LOOM_FABRIC_REF_FIELDS(visitor.ref(self.owner); visitor.ref(self.source);
                         visitor.ref(self.destination);)
};

struct FabricPeRegisterFifoPayload {
  FabricPeOccurrenceRef owner;
  FabricOrdinal registerFifo = 0;
  FabricRegisterFifoPathRole role = FabricRegisterFifoPathRole::Write;

  LOOM_FABRIC_REF_FIELDS(visitor.ref(self.owner);
                         visitor.ordinal(self.registerFifo);
                         visitor.tag(self.role);)
};

struct FabricSwitchTraversalPayload {
  FabricSwitchOccurrenceRef owner;
  FabricOrdinal input = 0;
  FabricOrdinal output = 0;

  LOOM_FABRIC_REF_FIELDS(visitor.ref(self.owner); visitor.ordinal(self.input);
                         visitor.ordinal(self.output);)
};

struct FabricFifoTraversalPayload {
  FabricFifoOccurrenceRef owner;
  FabricFifoTraversalMode mode = FabricFifoTraversalMode::Buffered;

  LOOM_FABRIC_REF_FIELDS(visitor.ref(self.owner); visitor.tag(self.mode);)
};

struct FabricBoundaryTraversalPayload {
  FabricBoundaryOccurrenceRef owner;
  FabricOrdinal output = 0;

  LOOM_FABRIC_REF_FIELDS(visitor.ref(self.owner); visitor.ordinal(self.output);)
};

struct FabricTransferPatternLegPayload {
  FabricTransferPatternRef owner;
  FabricOrdinal egress = 0;

  LOOM_FABRIC_REF_FIELDS(visitor.ref(self.owner); visitor.ordinal(self.egress);)
};

#define LOOM_FABRIC_TRAVERSAL_PAYLOAD_EQUALITY(Type, ...)                      \
  inline bool operator==(const Type &lhs, const Type &rhs) {                   \
    return __VA_ARGS__;                                                        \
  }

LOOM_FABRIC_TRAVERSAL_PAYLOAD_EQUALITY(FabricPointConnectionPayload,
                                       lhs.source == rhs.source &&
                                           lhs.destination == rhs.destination)
LOOM_FABRIC_TRAVERSAL_PAYLOAD_EQUALITY(FabricPeSelectorPayload,
                                       lhs.owner == rhs.owner &&
                                           lhs.source == rhs.source &&
                                           lhs.destination == rhs.destination)
LOOM_FABRIC_TRAVERSAL_PAYLOAD_EQUALITY(FabricPeRegisterFifoPayload,
                                       lhs.owner == rhs.owner &&
                                           lhs.registerFifo ==
                                               rhs.registerFifo &&
                                           lhs.role == rhs.role)
LOOM_FABRIC_TRAVERSAL_PAYLOAD_EQUALITY(FabricSwitchTraversalPayload,
                                       lhs.owner == rhs.owner &&
                                           lhs.input == rhs.input &&
                                           lhs.output == rhs.output)
LOOM_FABRIC_TRAVERSAL_PAYLOAD_EQUALITY(FabricFifoTraversalPayload,
                                       lhs.owner == rhs.owner &&
                                           lhs.mode == rhs.mode)
LOOM_FABRIC_TRAVERSAL_PAYLOAD_EQUALITY(FabricBoundaryTraversalPayload,
                                       lhs.owner == rhs.owner &&
                                           lhs.output == rhs.output)
LOOM_FABRIC_TRAVERSAL_PAYLOAD_EQUALITY(FabricTransferPatternLegPayload,
                                       lhs.owner == rhs.owner &&
                                           lhs.egress == rhs.egress)

#undef LOOM_FABRIC_TRAVERSAL_PAYLOAD_EQUALITY

/// One directed physical traversal. A switch traversal, a point connection,
/// and a system transfer-pattern leg are variants of this one closed sum and
/// never collapse into a route-table entry, capacity resource, or the
/// resource states a traversal induces.
struct FabricPhysicalTraversalRef {
  static constexpr llvm::StringLiteral familyKeyword =
      llvm::StringLiteral("fabric.traversal");
  FabricPhysicalTraversalKind kind =
      FabricPhysicalTraversalKind::PointConnection;
  union Payload {
    Payload() : pointConnection() {}
#define LOOM_FABRIC_TRAVERSAL(Name, Keyword, Member, Type) Type Member;
#include "Fabric/Identity/FabricRefs.def"
  } payload;

  static FabricPhysicalTraversalRef
  pointConnection(FabricTransportEndpointRef source,
                  FabricTransportEndpointRef destination) {
    FabricPhysicalTraversalRef ref;
    ref.kind = FabricPhysicalTraversalKind::PointConnection;
    ref.payload.pointConnection = {source, destination};
    return ref;
  }
  static FabricPhysicalTraversalRef
  peSelector(FabricPeOccurrenceRef owner, FabricTransportEndpointRef source,
             FabricTransportEndpointRef destination) {
    FabricPhysicalTraversalRef ref;
    ref.kind = FabricPhysicalTraversalKind::PeSelectorTraversal;
    ref.payload.peSelector = {owner, source, destination};
    return ref;
  }
  static FabricPhysicalTraversalRef
  peRegisterFifo(FabricPeOccurrenceRef owner, FabricOrdinal registerFifo,
                 FabricRegisterFifoPathRole role) {
    FabricPhysicalTraversalRef ref;
    ref.kind = FabricPhysicalTraversalKind::PeRegisterFifoTraversal;
    ref.payload.peRegisterFifo = {owner, registerFifo, role};
    return ref;
  }
  static FabricPhysicalTraversalRef
  switchTraversal(FabricSwitchOccurrenceRef owner, FabricOrdinal input,
                  FabricOrdinal output) {
    FabricPhysicalTraversalRef ref;
    ref.kind = FabricPhysicalTraversalKind::SwitchTraversal;
    ref.payload.switchTraversal = {owner, input, output};
    return ref;
  }
  static FabricPhysicalTraversalRef
  fifoTraversal(FabricFifoOccurrenceRef owner, FabricFifoTraversalMode mode) {
    FabricPhysicalTraversalRef ref;
    ref.kind = FabricPhysicalTraversalKind::FifoTraversal;
    ref.payload.fifoTraversal = {owner, mode};
    return ref;
  }
  static FabricPhysicalTraversalRef
  boundaryTraversal(FabricBoundaryOccurrenceRef owner, FabricOrdinal output) {
    FabricPhysicalTraversalRef ref;
    ref.kind = FabricPhysicalTraversalKind::BoundaryTraversal;
    ref.payload.boundaryTraversal = {owner, output};
    return ref;
  }
  static FabricPhysicalTraversalRef
  transferPatternLeg(FabricTransferPatternRef owner, FabricOrdinal egress) {
    FabricPhysicalTraversalRef ref;
    ref.kind = FabricPhysicalTraversalKind::SystemTransferPatternLeg;
    ref.payload.transferPatternLeg = {owner, egress};
    return ref;
  }
};

inline bool operator==(const FabricPhysicalTraversalRef &lhs,
                       const FabricPhysicalTraversalRef &rhs) {
  if (lhs.kind != rhs.kind)
    return false;
  switch (lhs.kind) {
#define LOOM_FABRIC_TRAVERSAL(Name, Keyword, Member, Type)                     \
  case FabricPhysicalTraversalKind::Name:                                      \
    return lhs.payload.Member == rhs.payload.Member;
#include "Fabric/Identity/FabricRefs.def"
  }
  return false;
}
inline bool operator!=(const FabricPhysicalTraversalRef &lhs,
                       const FabricPhysicalTraversalRef &rhs) {
  return !(lhs == rhs);
}

#undef LOOM_FABRIC_REF_EQUALITY
#undef LOOM_FABRIC_REF_FIELDS

} // namespace fabric
} // namespace loom

#endif // LOOM_FABRIC_IDENTITY_FABRICREFS_H
