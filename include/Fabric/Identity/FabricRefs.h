#ifndef LOOM_FABRIC_IDENTITY_FABRICREFS_H
#define LOOM_FABRIC_IDENTITY_FABRICREFS_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <system_error>
#include <utility>
#include <variant>

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

#define LOOM_FABRIC_MEMORY_SERVICE(Name, Keyword, Type) Name,
enum class FabricMemoryServiceKind : std::uint32_t {
#include "Fabric/Identity/FabricRefs.def"
};

#define LOOM_FABRIC_HARDWARE_DOMAIN_KIND(Name, Keyword) Name,
enum class FabricHardwareDomainKind : std::uint32_t {
#include "Fabric/Identity/FabricRefs.def"
};

#define LOOM_FABRIC_MEMORY_ENDPOINT_ROLE(Name, Keyword) Name,
enum class FabricMemoryEndpointRole : std::uint32_t {
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

#define LOOM_FABRIC_TRAVERSAL(Name, Keyword, Type) Name,
enum class FabricPhysicalTraversalKind : std::uint32_t {
#include "Fabric/Identity/FabricRefs.def"
};

#define LOOM_FABRIC_INVENTORY(Name, Keyword) Name,
enum class FabricInventoryKind : std::uint32_t {
#include "Fabric/Identity/FabricRefs.def"
};

#define LOOM_FABRIC_TRANSPORT_OWNER(Ordinal, Name, Type) Name = Ordinal,
enum class FabricTransportEndpointOwnerKind : std::uint32_t {
#include "Fabric/Identity/FabricRefs.def"
};

#define LOOM_FABRIC_MEMORY_OWNER(Ordinal, Name, Type) Name = Ordinal,
enum class FabricMemoryEndpointOwnerKind : std::uint32_t {
#include "Fabric/Identity/FabricRefs.def"
};

#define LOOM_FABRIC_INVENTORY_OWNER(Name, Type) Name,
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
llvm::StringRef fabricRefKeyword(FabricHardwareDomainKind kind);
llvm::StringRef fabricRefKeyword(FabricMemoryEndpointRole role);
llvm::StringRef fabricRefKeyword(FabricFifoTraversalMode mode);
llvm::StringRef fabricRefKeyword(FabricRegisterFifoPathRole role);
llvm::StringRef fabricRefKeyword(FabricPhysicalTraversalKind kind);
llvm::StringRef fabricRefKeyword(FabricInventoryKind kind);
llvm::StringRef fabricRefKeyword(FabricRefErrorKind kind);

/// Declared cardinality and diagnostic name of each closed sum. Every bound
/// is counted from the catalog declaration itself, so appending a variant
/// never requires a second edit. Both codecs and the importer read these.
#define LOOM_FABRIC_COUNT_ENTRY ++count;

#define LOOM_FABRIC_ENTITY(Name, Keyword) LOOM_FABRIC_COUNT_ENTRY
inline std::uint32_t fabricClosedBound(FabricEntityKind) {
  std::uint32_t count = 0;
#include "Fabric/Identity/FabricRefs.def"
  return count;
}
inline llvm::StringRef fabricClosedName(FabricEntityKind) {
  return "entity kind";
}

#define LOOM_FABRIC_FU_NODE_KIND(Name, Keyword) LOOM_FABRIC_COUNT_ENTRY
inline std::uint32_t fabricClosedBound(FabricFuNodeKind) {
  std::uint32_t count = 0;
#include "Fabric/Identity/FabricRefs.def"
  return count;
}
inline llvm::StringRef fabricClosedName(FabricFuNodeKind) {
  return "FU node kind";
}

#define LOOM_FABRIC_PORT_DIRECTION(Name, Keyword) LOOM_FABRIC_COUNT_ENTRY
inline std::uint32_t fabricClosedBound(FabricPortDirection) {
  std::uint32_t count = 0;
#include "Fabric/Identity/FabricRefs.def"
  return count;
}
inline llvm::StringRef fabricClosedName(FabricPortDirection) {
  return "port direction";
}

#define LOOM_FABRIC_MEMORY_SERVICE(Name, Keyword, Type) LOOM_FABRIC_COUNT_ENTRY
inline std::uint32_t fabricClosedBound(FabricMemoryServiceKind) {
  std::uint32_t count = 0;
#include "Fabric/Identity/FabricRefs.def"
  return count;
}
inline llvm::StringRef fabricClosedName(FabricMemoryServiceKind) {
  return "memory service kind";
}

#define LOOM_FABRIC_HARDWARE_DOMAIN_KIND(Name, Keyword) LOOM_FABRIC_COUNT_ENTRY
inline std::uint32_t fabricClosedBound(FabricHardwareDomainKind) {
  std::uint32_t count = 0;
#include "Fabric/Identity/FabricRefs.def"
  return count;
}
inline llvm::StringRef fabricClosedName(FabricHardwareDomainKind) {
  return "hardware domain kind";
}

#define LOOM_FABRIC_MEMORY_ENDPOINT_ROLE(Name, Keyword) LOOM_FABRIC_COUNT_ENTRY
inline std::uint32_t fabricClosedBound(FabricMemoryEndpointRole) {
  std::uint32_t count = 0;
#include "Fabric/Identity/FabricRefs.def"
  return count;
}
inline llvm::StringRef fabricClosedName(FabricMemoryEndpointRole) {
  return "memory endpoint role";
}

#define LOOM_FABRIC_FIFO_MODE(Name, Keyword) LOOM_FABRIC_COUNT_ENTRY
inline std::uint32_t fabricClosedBound(FabricFifoTraversalMode) {
  std::uint32_t count = 0;
#include "Fabric/Identity/FabricRefs.def"
  return count;
}
inline llvm::StringRef fabricClosedName(FabricFifoTraversalMode) {
  return "FIFO traversal mode";
}

#define LOOM_FABRIC_REGISTER_FIFO_PATH_ROLE(Name, Keyword)                     \
  LOOM_FABRIC_COUNT_ENTRY
inline std::uint32_t fabricClosedBound(FabricRegisterFifoPathRole) {
  std::uint32_t count = 0;
#include "Fabric/Identity/FabricRefs.def"
  return count;
}
inline llvm::StringRef fabricClosedName(FabricRegisterFifoPathRole) {
  return "register FIFO path role";
}

#define LOOM_FABRIC_TRAVERSAL(Name, Keyword, Type) LOOM_FABRIC_COUNT_ENTRY
inline std::uint32_t fabricClosedBound(FabricPhysicalTraversalKind) {
  std::uint32_t count = 0;
#include "Fabric/Identity/FabricRefs.def"
  return count;
}
inline llvm::StringRef fabricClosedName(FabricPhysicalTraversalKind) {
  return "physical traversal kind";
}

#define LOOM_FABRIC_INVENTORY(Name, Keyword) LOOM_FABRIC_COUNT_ENTRY
inline std::uint32_t fabricClosedBound(FabricInventoryKind) {
  std::uint32_t count = 0;
#include "Fabric/Identity/FabricRefs.def"
  return count;
}
inline llvm::StringRef fabricClosedName(FabricInventoryKind) {
  return "inventory kind";
}

#define LOOM_FABRIC_TRANSPORT_OWNER(Ordinal, Name, Type)                       \
  bound = std::max(bound, static_cast<std::uint32_t>(Ordinal) + 1);
inline std::uint32_t fabricClosedBound(FabricTransportEndpointOwnerKind) {
  std::uint32_t bound = 0;
#include "Fabric/Identity/FabricRefs.def"
  return bound;
}
inline llvm::StringRef fabricClosedName(FabricTransportEndpointOwnerKind) {
  return "transport endpoint owner";
}

#define LOOM_FABRIC_MEMORY_OWNER(Ordinal, Name, Type)                          \
  bound = std::max(bound, static_cast<std::uint32_t>(Ordinal) + 1);
inline std::uint32_t fabricClosedBound(FabricMemoryEndpointOwnerKind) {
  std::uint32_t bound = 0;
#include "Fabric/Identity/FabricRefs.def"
  return bound;
}
inline llvm::StringRef fabricClosedName(FabricMemoryEndpointOwnerKind) {
  return "memory endpoint owner";
}

#define LOOM_FABRIC_INVENTORY_OWNER(Name, Type) LOOM_FABRIC_COUNT_ENTRY
inline std::uint32_t fabricClosedBound(FabricInventoryOwnerKind) {
  std::uint32_t count = 0;
#include "Fabric/Identity/FabricRefs.def"
  return count;
}
inline llvm::StringRef fabricClosedName(FabricInventoryOwnerKind) {
  return "inventory owner";
}

#undef LOOM_FABRIC_COUNT_ENTRY

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

/// Builds a variant whose alternative order is exactly the catalog order. The
/// leading placeholder absorbs the list's separating commas and is dropped, so
/// `index()` is the declared discriminant and no second tag is stored.
template <typename Placeholder, typename... Alternatives>
struct FabricCatalogVariant {
  using type = std::variant<Alternatives...>;
};

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
  static constexpr llvm::StringLiteral familyKeyword = fabricRefKeyword(Kind);

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

/// One reusable module-template boundary endpoint. This is attachment
/// correspondence, not a concrete occurrence endpoint or routing resource.
struct FabricModuleBoundaryEndpointRef {
  static constexpr llvm::StringLiteral familyKeyword =
      llvm::StringLiteral("fabric.module_boundary_endpoint");
  FabricModuleTemplateRef module;
  FabricPortDirection direction = FabricPortDirection::Input;
  FabricOrdinal ordinal = 0;

  LOOM_FABRIC_REF_FIELDS(visitor.ref(self.module); visitor.tag(self.direction);
                         visitor.ordinal(self.ordinal);)
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

/// One condition-relevant physical graph template owned by an FU definition.
/// The referenced record inventory is defined by the Fabric artifact owner.
struct FabricFuCapabilityTemplateRef {
  static constexpr llvm::StringLiteral familyKeyword =
      llvm::StringLiteral("fabric.fu_capability_template");
  FabricFuTemplateRef fu;
  FabricOrdinal ordinal = 0;

  LOOM_FABRIC_REF_FIELDS(visitor.ref(self.fu); visitor.ordinal(self.ordinal);)
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

struct FabricMemoryEngineTemplateOperationPortRef {
  static constexpr llvm::StringLiteral familyKeyword =
      llvm::StringLiteral("fabric.memory_engine_template_operation_port");
  FabricMemoryEngineTemplateRef engine;
  FabricOrdinal ordinal = 0;

  LOOM_FABRIC_REF_FIELDS(visitor.ref(self.engine);
                         visitor.ordinal(self.ordinal);)
};

struct FabricMemoryEngineTemplateCapabilityAlternativeRef {
  static constexpr llvm::StringLiteral familyKeyword = llvm::StringLiteral(
      "fabric.memory_engine_template_capability_alternative");
  FabricMemoryEngineTemplateOperationPortRef port;
  FabricOrdinal ordinal = 0;

  LOOM_FABRIC_REF_FIELDS(visitor.ref(self.port); visitor.ordinal(self.ordinal);)
};

struct FabricMemoryEngineTemplateEndpointRef {
  static constexpr llvm::StringLiteral familyKeyword =
      llvm::StringLiteral("fabric.memory_engine_template_endpoint");
  FabricMemoryEngineTemplateRef engine;
  FabricOrdinal ordinal = 0;

  LOOM_FABRIC_REF_FIELDS(visitor.ref(self.engine);
                         visitor.ordinal(self.ordinal);)
};

struct FabricMemoryEngineTemplateInternalConnectionRef {
  static constexpr llvm::StringLiteral familyKeyword =
      llvm::StringLiteral("fabric.memory_engine_template_internal_connection");
  FabricMemoryEngineTemplateRef engine;
  FabricMemoryEngineTemplateEndpointRef source;
  FabricMemoryEngineTemplateEndpointRef sink;

  LOOM_FABRIC_REF_FIELDS(visitor.ref(self.engine); visitor.ref(self.source);
                         visitor.ref(self.sink);)
};

/// A memory service is either the optional Local Memory Service of one memory
/// occurrence or one system memory service entity. The selected alternative is
/// the discriminant; there is no separate tag field to keep in step.
struct FabricMemoryServiceRef {
  static constexpr llvm::StringLiteral familyKeyword =
      llvm::StringLiteral("fabric.memory_service");
  using Payload = typename FabricCatalogVariant<void
#define LOOM_FABRIC_MEMORY_SERVICE(Name, Keyword, Type) , Type
#include "Fabric/Identity/FabricRefs.def"
                                                >::type;

  Payload payload;

  FabricMemoryServiceKind kind() const {
    return static_cast<FabricMemoryServiceKind>(payload.index());
  }

#define LOOM_FABRIC_MEMORY_SERVICE(Name, Keyword, Type)                        \
  static_assert(                                                               \
      std::is_same_v<                                                          \
          std::variant_alternative_t<static_cast<std::size_t>(                 \
                                         FabricMemoryServiceKind::Name),       \
                                     Payload>,                                 \
          Type>,                                                               \
      "alternative order must match the discriminants");
#include "Fabric/Identity/FabricRefs.def"

  static FabricMemoryServiceRef local(FabricMemoryOccurrenceRef memory) {
    return FabricMemoryServiceRef{
        Payload(std::in_place_type<FabricMemoryOccurrenceRef>, memory)};
  }
  static FabricMemoryServiceRef system(SystemMemoryServiceRef service) {
    return FabricMemoryServiceRef{
        Payload(std::in_place_type<SystemMemoryServiceRef>, service)};
  }
};

inline bool operator==(const FabricMemoryServiceRef &lhs,
                       const FabricMemoryServiceRef &rhs) {
  return lhs.payload == rhs.payload;
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
LOOM_FABRIC_REF_EQUALITY(FabricModuleBoundaryEndpointRef,
                         lhs.module == rhs.module &&
                             lhs.direction == rhs.direction &&
                             lhs.ordinal == rhs.ordinal)
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
LOOM_FABRIC_REF_EQUALITY(FabricFuCapabilityTemplateRef,
                         lhs.fu == rhs.fu && lhs.ordinal == rhs.ordinal)
LOOM_FABRIC_REF_EQUALITY(FabricMemoryOperationPortRef,
                         lhs.memory == rhs.memory && lhs.ordinal == rhs.ordinal)
LOOM_FABRIC_REF_EQUALITY(FabricMemoryCapabilityAlternativeRef,
                         lhs.port == rhs.port && lhs.ordinal == rhs.ordinal)
LOOM_FABRIC_REF_EQUALITY(FabricMemoryOperationContextRef,
                         lhs.port == rhs.port && lhs.ordinal == rhs.ordinal)
LOOM_FABRIC_REF_EQUALITY(FabricMemoryEngineTemplateOperationPortRef,
                         lhs.engine == rhs.engine && lhs.ordinal == rhs.ordinal)
LOOM_FABRIC_REF_EQUALITY(FabricMemoryEngineTemplateCapabilityAlternativeRef,
                         lhs.port == rhs.port && lhs.ordinal == rhs.ordinal)
LOOM_FABRIC_REF_EQUALITY(FabricMemoryEngineTemplateEndpointRef,
                         lhs.engine == rhs.engine && lhs.ordinal == rhs.ordinal)
LOOM_FABRIC_REF_EQUALITY(FabricMemoryEngineTemplateInternalConnectionRef,
                         lhs.engine == rhs.engine && lhs.source == rhs.source &&
                             lhs.sink == rhs.sink)
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
  using Payload = typename FabricCatalogVariant<void
#define LOOM_FABRIC_TRANSPORT_OWNER(Ordinal, Name, Type) , Type
#include "Fabric/Identity/FabricRefs.def"
                                                >::type;

  Payload payload;

  FabricTransportEndpointOwnerKind kind() const {
#define LOOM_FABRIC_TRANSPORT_OWNER(Ordinal, Name, Type)                       \
  if (std::holds_alternative<Type>(payload))                                   \
    return FabricTransportEndpointOwnerKind::Name;
#include "Fabric/Identity/FabricRefs.def"
    llvm_unreachable("unknown transport endpoint owner payload");
  }

#define LOOM_FABRIC_TRANSPORT_OWNER(Ordinal, Name, Type)                       \
  static FabricTransportEndpointOwnerRef of(const Type &value) {               \
    return FabricTransportEndpointOwnerRef{                                    \
        Payload(std::in_place_type<Type>, value)};                             \
  }
#include "Fabric/Identity/FabricRefs.def"
};

inline bool operator==(const FabricTransportEndpointOwnerRef &lhs,
                       const FabricTransportEndpointOwnerRef &rhs) {
  return lhs.payload == rhs.payload;
}
inline bool operator!=(const FabricTransportEndpointOwnerRef &lhs,
                       const FabricTransportEndpointOwnerRef &rhs) {
  return !(lhs == rhs);
}

/// Closed union of owners exposing a manager or subordinate memory-service
/// endpoint inventory.
struct FabricMemoryEndpointOwnerRef {
  using Payload = typename FabricCatalogVariant<void
#define LOOM_FABRIC_MEMORY_OWNER(Ordinal, Name, Type) , Type
#include "Fabric/Identity/FabricRefs.def"
                                                >::type;

  Payload payload;

  FabricMemoryEndpointOwnerKind kind() const {
#define LOOM_FABRIC_MEMORY_OWNER(Ordinal, Name, Type)                          \
  if (std::holds_alternative<Type>(payload))                                   \
    return FabricMemoryEndpointOwnerKind::Name;
#include "Fabric/Identity/FabricRefs.def"
    llvm_unreachable("unknown memory endpoint owner payload");
  }

#define LOOM_FABRIC_MEMORY_OWNER(Ordinal, Name, Type)                          \
  static FabricMemoryEndpointOwnerRef of(const Type &value) {                  \
    return FabricMemoryEndpointOwnerRef{                                       \
        Payload(std::in_place_type<Type>, value)};                             \
  }
#include "Fabric/Identity/FabricRefs.def"
};

inline bool operator==(const FabricMemoryEndpointOwnerRef &lhs,
                       const FabricMemoryEndpointOwnerRef &rhs) {
  return lhs.payload == rhs.payload;
}
inline bool operator!=(const FabricMemoryEndpointOwnerRef &lhs,
                       const FabricMemoryEndpointOwnerRef &rhs) {
  return !(lhs == rhs);
}

/// The one closed constructor catalog behind every owner projection. Sharing
/// the catalog avoids four independently drifting copies; the projection that
/// selects a canonical inventory is static type information alone.
struct FabricInventoryOwnerRef {
  using Payload = typename FabricCatalogVariant<void
#define LOOM_FABRIC_INVENTORY_OWNER(Name, Type) , Type
#include "Fabric/Identity/FabricRefs.def"
                                                >::type;

  Payload payload;

  FabricInventoryOwnerKind kind() const {
    return static_cast<FabricInventoryOwnerKind>(payload.index());
  }

#define LOOM_FABRIC_INVENTORY_OWNER(Name, Type)                                \
  static_assert(                                                               \
      std::is_same_v<                                                          \
          std::variant_alternative_t<static_cast<std::size_t>(                 \
                                         FabricInventoryOwnerKind::Name),      \
                                     Payload>,                                 \
          Type>,                                                               \
      "alternative order must match the discriminants");                       \
  static FabricInventoryOwnerRef of(const Type &value) {                       \
    return FabricInventoryOwnerRef{Payload(std::in_place_type<Type>, value)};  \
  }
#include "Fabric/Identity/FabricRefs.def"
};

inline bool operator==(const FabricInventoryOwnerRef &lhs,
                       const FabricInventoryOwnerRef &rhs) {
  return lhs.payload == rhs.payload;
}
inline bool operator!=(const FabricInventoryOwnerRef &lhs,
                       const FabricInventoryOwnerRef &rhs) {
  return !(lhs == rhs);
}

/// Projects either endpoint-owner plane into the shared inventory-owner
/// catalog while preserving the complete typed owner. A system memory service
/// uses the catalog's canonical MemoryService::System constructor.
FabricInventoryOwnerRef
projectFabricInventoryOwner(const FabricTransportEndpointOwnerRef &owner);
FabricInventoryOwnerRef
projectFabricInventoryOwner(const FabricMemoryEndpointOwnerRef &owner);

/// A role-specific owner projection over that one catalog. Four distinct
/// static types share one constructor declaration and one canonical encoding;
/// the role only selects which owner-declared inventory an ordinal indexes.
template <FabricInventoryKind Inventory> class FabricOwnerProjection {
public:
  static constexpr FabricInventoryKind inventory = Inventory;

  FabricOwnerProjection() = default;
  explicit FabricOwnerProjection(FabricInventoryOwnerRef catalog)
      : catalog_(std::move(catalog)) {}

  const FabricInventoryOwnerRef &catalog() const { return catalog_; }

  friend bool operator==(const FabricOwnerProjection &lhs,
                         const FabricOwnerProjection &rhs) {
    return lhs.catalog_ == rhs.catalog_;
  }
  friend bool operator!=(const FabricOwnerProjection &lhs,
                         const FabricOwnerProjection &rhs) {
    return !(lhs == rhs);
  }

private:
  FabricInventoryOwnerRef catalog_;
};

#define LOOM_FABRIC_OWNER_ROLE(Alias, Inventory, Family, Keyword)              \
  using Alias = FabricOwnerProjection<FabricInventoryKind::Inventory>;
#include "Fabric/Identity/FabricRefs.def"

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

/// The owner-relative families. Each names its own owner projection, so a
/// resource-state owner can never be passed where a use-pattern owner is
/// required even though both carry the same catalog constructor.
#define LOOM_FABRIC_OWNER_ROLE(Alias, Inventory, Family, Keyword)              \
  struct Family {                                                              \
    static constexpr llvm::StringLiteral familyKeyword =                       \
        llvm::StringLiteral(Keyword);                                          \
    Alias owner;                                                               \
    FabricOrdinal ordinal = 0;                                                 \
                                                                               \
    LOOM_FABRIC_REF_FIELDS(visitor.ref(self.owner);                            \
                           visitor.ordinal(self.ordinal);)                     \
  };                                                                           \
  LOOM_FABRIC_REF_EQUALITY(Family, lhs.owner == rhs.owner &&                   \
                                       lhs.ordinal == rhs.ordinal)
#include "Fabric/Identity/FabricRefs.def"

LOOM_FABRIC_REF_EQUALITY(FabricTransportEndpointRef,
                         lhs.owner == rhs.owner && lhs.ordinal == rhs.ordinal)
LOOM_FABRIC_REF_EQUALITY(FabricMemoryEndpointRef,
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
  using Payload = typename FabricCatalogVariant<void
#define LOOM_FABRIC_TRAVERSAL(Name, Keyword, Type) , Type
#include "Fabric/Identity/FabricRefs.def"
                                                >::type;

  Payload payload;

  FabricPhysicalTraversalKind kind() const {
    return static_cast<FabricPhysicalTraversalKind>(payload.index());
  }

#define LOOM_FABRIC_TRAVERSAL(Name, Keyword, Type)                             \
  static_assert(                                                               \
      std::is_same_v<                                                          \
          std::variant_alternative_t<static_cast<std::size_t>(                 \
                                         FabricPhysicalTraversalKind::Name),   \
                                     Payload>,                                 \
          Type>,                                                               \
      "alternative order must match the discriminants");
#include "Fabric/Identity/FabricRefs.def"

  template <typename PayloadType>
  static FabricPhysicalTraversalRef of(PayloadType value) {
    return FabricPhysicalTraversalRef{
        Payload(std::in_place_type<PayloadType>, std::move(value))};
  }

  static FabricPhysicalTraversalRef
  pointConnection(FabricTransportEndpointRef source,
                  FabricTransportEndpointRef destination) {
    return of(FabricPointConnectionPayload{source, destination});
  }
  static FabricPhysicalTraversalRef
  peSelector(FabricPeOccurrenceRef owner, FabricTransportEndpointRef source,
             FabricTransportEndpointRef destination) {
    return of(FabricPeSelectorPayload{owner, source, destination});
  }
  static FabricPhysicalTraversalRef
  peRegisterFifo(FabricPeOccurrenceRef owner, FabricOrdinal registerFifo,
                 FabricRegisterFifoPathRole role) {
    return of(FabricPeRegisterFifoPayload{owner, registerFifo, role});
  }
  static FabricPhysicalTraversalRef
  switchTraversal(FabricSwitchOccurrenceRef owner, FabricOrdinal input,
                  FabricOrdinal output) {
    return of(FabricSwitchTraversalPayload{owner, input, output});
  }
  static FabricPhysicalTraversalRef
  fifoTraversal(FabricFifoOccurrenceRef owner, FabricFifoTraversalMode mode) {
    return of(FabricFifoTraversalPayload{owner, mode});
  }
  static FabricPhysicalTraversalRef
  boundaryTraversal(FabricBoundaryOccurrenceRef owner, FabricOrdinal output) {
    return of(FabricBoundaryTraversalPayload{owner, output});
  }
  static FabricPhysicalTraversalRef
  transferPatternLeg(FabricTransferPatternRef owner, FabricOrdinal egress) {
    return of(FabricTransferPatternLegPayload{owner, egress});
  }
};

inline bool operator==(const FabricPhysicalTraversalRef &lhs,
                       const FabricPhysicalTraversalRef &rhs) {
  return lhs.payload == rhs.payload;
}
inline bool operator!=(const FabricPhysicalTraversalRef &lhs,
                       const FabricPhysicalTraversalRef &rhs) {
  return !(lhs == rhs);
}

#define LOOM_FABRIC_REFINEMENT(Name, Alias, Underlying) Name,
enum class FabricRefinementKind : std::uint32_t {
#include "Fabric/Identity/FabricRefs.def"
};

/// A typed refinement of an underlying reference. The refined name is selected
/// by the consuming field's static type; the canonical text and bytes remain
/// exactly the underlying ones, so no role field, wrapper tag, or second
/// identity is ever serialized. Validation checks the fact the owner already
/// declares.
template <FabricRefinementKind Refinement, typename Underlying>
class FabricRefinedRef {
public:
  FabricRefinedRef() = default;
  explicit FabricRefinedRef(Underlying underlying)
      : underlying_(std::move(underlying)) {}

  const Underlying &underlying() const { return underlying_; }

  friend bool operator==(const FabricRefinedRef &lhs,
                         const FabricRefinedRef &rhs) {
    return lhs.underlying_ == rhs.underlying_;
  }
  friend bool operator!=(const FabricRefinedRef &lhs,
                         const FabricRefinedRef &rhs) {
    return !(lhs == rhs);
  }

private:
  Underlying underlying_;
};

#define LOOM_FABRIC_REFINEMENT(Name, Alias, Underlying)                        \
  using Alias = FabricRefinedRef<FabricRefinementKind::Name, Underlying>;
#include "Fabric/Identity/FabricRefs.def"

#undef LOOM_FABRIC_REF_EQUALITY
#undef LOOM_FABRIC_REF_FIELDS

} // namespace fabric
} // namespace loom

#endif // LOOM_FABRIC_IDENTITY_FABRICREFS_H
