#ifndef LOOM_DATAFLOW_IR_DATAFLOW_SERVICE_SCHEMA_H
#define LOOM_DATAFLOW_IR_DATAFLOW_SERVICE_SCHEMA_H

#include "Dataflow/IR/DataflowActorSemantics.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>
#include <variant>

namespace dataflow::semantics {

//===----------------------------------------------------------------------===//
// Canonical memory access view
//===----------------------------------------------------------------------===//

/// How one addressed access names the memory elements it reaches.
enum class MemoryAccessForm : std::uint8_t {
  /// One complete memory element, even when that element is itself a vector.
  Element,
  /// One access lane shape reached contiguously from one scalar address.
  Contiguous,
  /// One access lane shape with one address per lane.
  Indexed,
};

/// Which canonical addressed Dataflow actor a view projects.
enum class MemoryAccessOperation : std::uint8_t {
  Load,
  Store,
  AtomicRmw,
  CompareExchange,
};

/// Whether the actor restricts its lanes with a dynamic mask.
enum class MemoryMaskForm : std::uint8_t { Absent, Dynamic };

/// The nonpersistent `CanonicalMemoryAccessView` of one addressed canonical
/// Dataflow actor. It holds the two existing owners of that actor's memory
/// semantics -- the shared access geometry and the actor's one aggregate
/// contract -- and derives every projected fact from them, including the
/// flattened widths a downstream compatibility relation compares.
///
/// It is produced only by `getCanonicalMemoryAccessView` from one exact
/// verified actor: it cannot be default constructed, assembled field by
/// field, or overwritten in place. It has no identity of its own, is never
/// serialized, and never becomes a second owner of the actor's type, shape,
/// mask, ordering, volatility, granularity, or scope.
class CanonicalMemoryAccessView {
public:
  /// A derived projection is produced whole and never overwritten. Deleting
  /// the copy assignment also withdraws the implicit move assignment, so the
  /// projection stays a value that can be passed on but not mutated.
  CanonicalMemoryAccessView(const CanonicalMemoryAccessView &) = default;
  CanonicalMemoryAccessView &
  operator=(const CanonicalMemoryAccessView &) = delete;

  MemoryAccessOperation operation() const { return accessOperation; }

  /// The one shared geometry analysis of this access.
  const MemoryAccessType &geometry() const { return accessGeometry; }

  /// The one aggregate contract this actor owns.
  const MemoryActorContract &contract() const { return actorContract; }

  /// The actor's exact mask operand type, null when it carries no mask.
  mlir::Type maskType() const { return actorMaskType; }

  MemoryAccessForm form() const;

  MemoryMaskForm maskForm() const {
    return actorMaskType ? MemoryMaskForm::Dynamic : MemoryMaskForm::Absent;
  }

  mlir::Type memoryElementType() const { return accessGeometry.elementType; }

  /// The exact ranked access shape, empty for an `Element` access.
  llvm::ArrayRef<std::int64_t> laneShape() const;

  std::uint64_t laneCount() const { return derived.laneCount; }

  /// One logical address per lane for an indexed access, otherwise one.
  std::uint64_t addressCount() const { return derived.addressCount; }

  std::optional<dataflow::VectorAtomicGranularity> atomicGranularity() const {
    return actorContract.vectorGranularity;
  }

  /// `AddressOf`: `index` for an element or contiguous access, the exact
  /// same-shape address vector for an indexed access.
  mlir::Type addressType() const;

  /// `DataOf`: the exact memory element type, or the complete access vector.
  mlir::Type dataType() const;

  /// `SuccessOf`, defined only for a compare-exchange view.
  mlir::Type successType() const;

  /// The width of one complete memory element.
  std::uint64_t elementBits() const { return derived.elementBits; }

  std::uint64_t dataBits() const { return derived.dataBits; }

  /// The canonical `index` width in this actor's closest enclosing scope.
  unsigned indexBits() const { return derived.indexBits; }

  std::uint64_t addressBits() const { return derived.addressBits; }

  std::uint64_t maskBits() const { return derived.maskBits; }

private:
  /// The exact counts and widths of one access. Each is a product of facts the
  /// actor owns, resolved once when the projection is derived and only after
  /// that product is proven representable. Every getter above reads one of
  /// these and none recomputes a product, so no count or width this projection
  /// publishes can be a wrapped value.
  struct DerivedGeometry {
    std::uint64_t laneCount;
    std::uint64_t addressCount;
    std::uint64_t elementBits;
    std::uint64_t dataBits;
    unsigned indexBits;
    std::uint64_t addressBits;
    std::uint64_t maskBits;
  };

  CanonicalMemoryAccessView(MemoryAccessOperation operation,
                            const MemoryAccessType &geometry,
                            const MemoryActorContract &contract,
                            mlir::Type maskType, const DerivedGeometry &derived)
      : accessOperation(operation), accessGeometry(geometry),
        actorContract(contract), actorMaskType(maskType), derived(derived) {}

  friend llvm::Expected<CanonicalMemoryAccessView>
  getCanonicalMemoryAccessView(mlir::Operation *op);

  MemoryAccessOperation accessOperation;
  MemoryAccessType accessGeometry;
  MemoryActorContract actorContract;
  mlir::Type actorMaskType;
  DerivedGeometry derived;
};

/// The addressed-access view of `op`. Fails when there is no operation, when
/// `op` is not one of the four addressed canonical memory actors, and
/// therefore for `dataflow.fence`, which addresses no memory. Fails as well
/// when the operation's own verification rejects it, so a view exists only
/// for a well formed actor and no malformed operand, result, or contract can
/// reach a projection.
llvm::Expected<CanonicalMemoryAccessView>
getCanonicalMemoryAccessView(mlir::Operation *op);

//===----------------------------------------------------------------------===//
// Canonical Service Schema 2.0
//===----------------------------------------------------------------------===//

/// The exactly six parameterized kinds of Canonical Service Schema 2.0.
enum class ServiceKind : std::uint8_t {
  MessageTransfer,
  MemoryRead,
  MemoryWrite,
  MemoryAtomicRmw,
  MemoryCompareExchange,
  MemoryFence,
};

/// The one typed effect a service kind has. The addressed effects name the
/// logical memory service; `Order` names the memory consistency domain.
enum class ServiceEffect : std::uint8_t {
  None,
  Read,
  Write,
  ReadModifyWrite,
  CompareExchange,
  Order,
};

/// Every value role the schema names. A role occurs at most once in one
/// signature, and no kind uses one role as both an argument and a result.
enum class ServiceValueRole : std::uint8_t {
  Payload,
  Address,
  Data,
  Update,
  Expected,
  Desired,
  Mask,
  Control,
  Old,
  Success,
  Completion,
};

/// One ordered schema value: its role and its exact type.
struct ServiceValue {
  ServiceValueRole role;
  mlir::Type type;
};

using ServiceValues = llvm::SmallVector<ServiceValue, 5>;

/// Which of the two operation-relative endpoints sends one transfer leg. The
/// initiator is the message source or the memory manager; the server is the
/// message sink, the memory provider, or a fence's consistency provider.
enum class ServiceLegDirection : std::uint8_t {
  InitiatorToServer,
  ServerToInitiator,
};

/// The schema spelling of one kind.
llvm::StringRef stringifyServiceKind(ServiceKind kind);

/// The service kind one addressed access operation obliges.
ServiceKind getServiceKind(MemoryAccessOperation operation);

/// One nonpersistent Canonical Service Schema 2.0 service: an exact kind bound
/// to the one parameter that kind takes -- a message payload type, an
/// addressed access view, or a fence contract.
///
/// The schema is the sole owner of the deterministic argument and result
/// order, the typed effect, the transfer-leg direction and ordinal, and the
/// completion leg. Every value type it names remains owned by the actor or the
/// payload type it projects.
class CanonicalService {
public:
  /// `message_transfer<Payload>` over one exact payload type from the
  /// Dataflow transfer payload domain.
  static llvm::Expected<CanonicalService> messageTransfer(mlir::Type payload);

  /// The service one canonical Dataflow memory actor obliges. Fails when there
  /// is no operation, when `op` is not a canonical memory actor, and when the
  /// actor's own geometry or contract owner rejects it.
  static llvm::Expected<CanonicalService> forActor(mlir::Operation *op);

  /// The same projection, rejecting an actor that obliges another kind.
  static llvm::Expected<CanonicalService> forActor(mlir::Operation *op,
                                                   ServiceKind expected);

  ServiceKind kind() const;
  ServiceEffect effect() const;

  /// This kind's arguments and results in schema order. An addressed kind
  /// names its mask exactly when its actor carries one.
  ServiceValues arguments() const;
  ServiceValues results() const;

  /// One leg for a message transfer; a request and a response for every memory
  /// kind.
  unsigned legCount() const;
  ServiceLegDirection legDirection(unsigned ordinal) const;

  /// The initiating leg carries the arguments; the responding leg carries the
  /// results.
  ServiceValues legPayload(unsigned ordinal) const;

  /// The leg whose acceptance is this service's completion event: always the
  /// final leg.
  unsigned completionLeg() const { return legCount() - 1; }

  /// The exact parameter of this service. Each accessor requires its own kind.
  mlir::Type payload() const;
  const CanonicalMemoryAccessView &access() const;
  FenceContractAttr fenceContract() const;

private:
  using Parameter =
      std::variant<mlir::Type, CanonicalMemoryAccessView, FenceContractAttr>;

  explicit CanonicalService(Parameter parameter)
      : parameter(std::move(parameter)) {}

  mlir::MLIRContext *context() const;
  mlir::Type typeOf(ServiceValueRole role) const;

  Parameter parameter;
};

} // namespace dataflow::semantics

#endif // LOOM_DATAFLOW_IR_DATAFLOW_SERVICE_SCHEMA_H
