//===- DataflowServiceSchema.cpp - Canonical Service Schema 2.0 -----------===//
//
// The nonpersistent Canonical Service Schema 2.0 projection of the Dataflow
// semantic layer: the addressed access view derived from one canonical memory
// actor, and the six parameterized service kinds derived from that view, a
// fence contract, or one exact message payload type.
//
// This library owns the deterministic argument and result order, the typed
// effect, the transfer-leg direction and ordinal, and the completion leg of
// each kind. It owns nothing else: every type, shape, mask, contract, and
// access geometry it names is read back from the actor that already owns it.
//
//===----------------------------------------------------------------------===//

#include "Dataflow/IR/DataflowServiceSchema.h"

#include "Common/IndexWidth.h"
#include "Common/VectorWidth.h"
#include "Dataflow/IR/DataflowOps.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <cassert>
#include <string>

using namespace mlir;
using namespace dataflow;
using namespace dataflow::semantics;

namespace {

llvm::Error schemaError(const llvm::Twine &message) {
  return llvm::createStringError(std::errc::invalid_argument, "%s",
                                 message.str().c_str());
}

//===----------------------------------------------------------------------===//
// Actor projection
//===----------------------------------------------------------------------===//

/// Structural legality of one canonical actor, answered by that operation's
/// own generated invariants and verifier rather than by a second constraint
/// set. Everything below this gate reads a verified operation, so no
/// generated accessor or type cast can meet a shape it does not admit.
///
/// The verifier reports through diagnostics. A projection query is a question,
/// not a pass, so this captures them and returns the first as its error
/// instead of emitting anything.
llvm::Error verifyActorStructure(Operation *op) {
  llvm::SmallVector<std::string, 2> diagnostics;
  mlir::ScopedDiagnosticHandler capture(
      op->getContext(), [&](mlir::Diagnostic &diagnostic) {
        diagnostics.push_back(diagnostic.str());
        return mlir::success();
      });
  if (mlir::succeeded(mlir::verify(op, /*verifyRecursively=*/false)))
    return llvm::Error::success();
  if (diagnostics.empty())
    return schemaError("'" + op->getName().getStringRef() +
                       "' is not a well formed operation");
  return schemaError(diagnostics.front());
}

/// The width of one complete memory element, read from the owners that
/// already define it. This runs only after the actor's own verification and
/// `analyzeMemoryAccessType` have accepted the access, so a vector element is
/// already inside the canonical semantic element domain and this adds no type
/// rule of its own. A scalar `index` has no width of its own, so the shared
/// closest-scope resolver answers for it exactly as it does for the address.
llvm::Expected<std::uint64_t> getElementBitWidth(Type elementType,
                                                 Operation *op) {
  if (auto vector = llvm::dyn_cast<VectorType>(elementType))
    return getFlattenedVectorBitWidth(vector);
  if (llvm::isa<IndexType>(elementType)) {
    llvm::Expected<unsigned> index = loom::getIndexBitWidth(op);
    if (!index)
      return index.takeError();
    return static_cast<std::uint64_t>(*index);
  }
  if (llvm::isa<IntegerType, FloatType>(elementType))
    return elementType.getIntOrFloatBitWidth();

  std::string spelling;
  llvm::raw_string_ostream stream(spelling);
  stream << elementType;
  return schemaError("memory element type '" + spelling +
                     "' has no exact bit width");
}

/// The operands and access data type one addressed canonical actor exposes.
struct AddressedActor {
  MemoryAccessOperation operation;
  Value memory;
  Value address;
  Type dataType;
  Value mask;
};

std::optional<AddressedActor> getAddressedActor(Operation *op) {
  return llvm::TypeSwitch<Operation *, std::optional<AddressedActor>>(op)
      .Case<LoadOp>([](LoadOp load) {
        return AddressedActor{MemoryAccessOperation::Load, load.getMem(),
                              load.getAddr(), load.getData().getType(),
                              load.getMask()};
      })
      .Case<StoreOp>([](StoreOp store) {
        return AddressedActor{MemoryAccessOperation::Store, store.getMem(),
                              store.getAddr(), store.getData().getType(),
                              store.getMask()};
      })
      .Case<AtomicRmwOp>([](AtomicRmwOp rmw) {
        return AddressedActor{MemoryAccessOperation::AtomicRmw, rmw.getMem(),
                              rmw.getAddr(), rmw.getValue().getType(),
                              rmw.getMask()};
      })
      .Case<CmpXchgOp>([](CmpXchgOp cmpxchg) {
        return AddressedActor{MemoryAccessOperation::CompareExchange,
                              cmpxchg.getMem(), cmpxchg.getAddr(),
                              cmpxchg.getExpected().getType(),
                              cmpxchg.getMask()};
      })
      .Default([](Operation *) { return std::nullopt; });
}

//===----------------------------------------------------------------------===//
// The schema table
//
// One entry per kind, written exactly as the Canonical Service Schema states
// it. Every addressed kind lists its optional mask; an instance whose actor
// carries no mask drops that one role and nothing else.
//===----------------------------------------------------------------------===//

using Role = ServiceValueRole;

constexpr Role messageArguments[] = {Role::Payload};
constexpr Role messageResults[] = {Role::Completion};
constexpr Role readArguments[] = {Role::Address, Role::Mask, Role::Control};
constexpr Role readResults[] = {Role::Data, Role::Completion};
constexpr Role writeArguments[] = {Role::Address, Role::Data, Role::Mask,
                                   Role::Control};
constexpr Role writeResults[] = {Role::Completion};
constexpr Role rmwArguments[] = {Role::Address, Role::Update, Role::Mask,
                                 Role::Control};
constexpr Role rmwResults[] = {Role::Old, Role::Completion};
constexpr Role compareExchangeArguments[] = {
    Role::Address, Role::Expected, Role::Desired, Role::Mask, Role::Control};
constexpr Role compareExchangeResults[] = {Role::Old, Role::Success,
                                           Role::Completion};
constexpr Role fenceArguments[] = {Role::Control};
constexpr Role fenceResults[] = {Role::Completion};

/// The static facts one service kind owns.
struct KindSchema {
  ServiceEffect effect;
  llvm::ArrayRef<Role> arguments;
  llvm::ArrayRef<Role> results;
  unsigned legCount;
};

constexpr KindSchema messageSchema{ServiceEffect::None, messageArguments,
                                   messageResults, 1};
constexpr KindSchema readSchema{ServiceEffect::Read, readArguments, readResults,
                                2};
constexpr KindSchema writeSchema{ServiceEffect::Write, writeArguments,
                                 writeResults, 2};
constexpr KindSchema rmwSchema{ServiceEffect::ReadModifyWrite, rmwArguments,
                               rmwResults, 2};
constexpr KindSchema compareExchangeSchema{ServiceEffect::CompareExchange,
                                           compareExchangeArguments,
                                           compareExchangeResults, 2};
constexpr KindSchema fenceSchema{ServiceEffect::Order, fenceArguments,
                                 fenceResults, 2};

const KindSchema &getKindSchema(ServiceKind kind) {
  switch (kind) {
  case ServiceKind::MessageTransfer:
    return messageSchema;
  case ServiceKind::MemoryRead:
    return readSchema;
  case ServiceKind::MemoryWrite:
    return writeSchema;
  case ServiceKind::MemoryAtomicRmw:
    return rmwSchema;
  case ServiceKind::MemoryCompareExchange:
    return compareExchangeSchema;
  case ServiceKind::MemoryFence:
    return fenceSchema;
  }
  llvm_unreachable("unhandled canonical service kind");
}

} // namespace

//===----------------------------------------------------------------------===//
// Canonical memory access view
//===----------------------------------------------------------------------===//

MemoryAccessForm CanonicalMemoryAccessView::form() const {
  if (accessGeometry.isGather())
    return MemoryAccessForm::Indexed;
  return accessGeometry.isVector() ? MemoryAccessForm::Contiguous
                                   : MemoryAccessForm::Element;
}

bool CanonicalMemoryAccessView::operator==(
    const CanonicalMemoryAccessView &other) const {
  return sourceActor == other.sourceActor &&
         accessOperation == other.accessOperation &&
         accessGeometry.elementType == other.accessGeometry.elementType &&
         accessGeometry.dataType == other.accessGeometry.dataType &&
         accessGeometry.addressType == other.accessGeometry.addressType &&
         accessGeometry.vectorType == other.accessGeometry.vectorType &&
         accessGeometry.addressVectorType ==
             other.accessGeometry.addressVectorType &&
         accessGeometry.addressForm == other.accessGeometry.addressForm &&
         accessGeometry.pointerLayout == other.accessGeometry.pointerLayout &&
         accessGeometry.dataPointerLayout ==
             other.accessGeometry.dataPointerLayout &&
         actorContract.aggregate == other.actorContract.aggregate &&
         actorContract.atomic == other.actorContract.atomic &&
         actorContract.isVolatile == other.actorContract.isVolatile &&
         actorContract.sourceAlignmentBytes ==
             other.actorContract.sourceAlignmentBytes &&
         actorContract.vectorGranularity ==
             other.actorContract.vectorGranularity &&
         actorContract.syncScope == other.actorContract.syncScope &&
         actorMaskType == other.actorMaskType &&
         derived.laneCount == other.derived.laneCount &&
         derived.addressCount == other.derived.addressCount &&
         derived.elementBits == other.derived.elementBits &&
         derived.dataBits == other.derived.dataBits &&
         derived.addressLaneBits == other.derived.addressLaneBits &&
         derived.addressBits == other.derived.addressBits &&
         derived.maskBits == other.derived.maskBits;
}

llvm::ArrayRef<std::int64_t> CanonicalMemoryAccessView::laneShape() const {
  if (!accessGeometry.isVector())
    return {};
  return accessGeometry.vectorType.getShape();
}

Type CanonicalMemoryAccessView::addressType() const {
  return accessGeometry.addressType;
}

Type CanonicalMemoryAccessView::dataType() const {
  return accessGeometry.dataType;
}

Type CanonicalMemoryAccessView::successType() const {
  assert(accessOperation == MemoryAccessOperation::CompareExchange &&
         "only a compare-exchange access publishes a success result");
  return getCompareExchangeSuccessType(accessGeometry);
}

llvm::Expected<CanonicalMemoryAccessView>
dataflow::semantics::getCanonicalMemoryAccessView(Operation *op) {
  if (!op)
    return schemaError("there is no operation to project");
  // Classification reads only the registered operation name, so it is the one
  // question that is safe to ask before the structural gate below.
  if (!llvm::isa<LoadOp, StoreOp, AtomicRmwOp, CmpXchgOp>(op))
    return schemaError("operation is not an addressed canonical memory actor");
  if (llvm::Error error = verifyActorStructure(op))
    return std::move(error);
  llvm::Expected<CanonicalActorSchemaProjection> projection =
      projectRegisteredActorSchemaProjection(op);
  if (!projection)
    return projection.takeError();

  std::optional<AddressedActor> actor = getAddressedActor(op);
  assert(actor && "a verified addressed actor exposes its access operands");
  Type maskType = actor->mask ? actor->mask.getType() : Type();
  llvm::Expected<MemoryAccessType> geometry = analyzeMemoryAccessType(
      llvm::cast<MemRefType>(actor->memory.getType()), actor->dataType,
      actor->address.getType(), op, maskType);
  if (!geometry)
    return geometry.takeError();
  std::optional<MemoryActorContract> contract = getMemoryActorContract(op);
  assert(contract && "a verified addressed actor owns one aggregate contract");

  // Both widths come from the owners that already resolve them, at this
  // actor's own scope, so the projection introduces no width policy.
  llvm::Expected<std::uint64_t> elementBits =
      getElementBitWidth(geometry->elementType, op);
  if (!elementBits)
    return elementBits.takeError();
  llvm::Expected<unsigned> addressLaneBits = [&]() -> llvm::Expected<unsigned> {
    if (geometry->addressForm == MemoryAddressForm::PointerAddressed) {
      assert(geometry->pointerLayout &&
             "verified pointer address has no pointer layout");
      return geometry->pointerLayout->representationBits;
    }
    return loom::getIndexBitWidth(op);
  }();
  if (!addressLaneBits)
    return addressLaneBits.takeError();

  // A verified access can still name a shape whose product no exact count can
  // hold. Each count and width below is that product under one element width,
  // formed by the single owner of a fixed vector product rather than by a
  // second one here, so an unrepresentable access is refused before it is
  // published and no published value can be a wrapped one.
  std::uint64_t laneCount = 1;
  std::uint64_t dataBits = geometry->dataPointerLayout
                               ? geometry->dataPointerLayout->representationBits
                               : *elementBits;
  if (geometry->isVector()) {
    llvm::Expected<std::uint64_t> lanes =
        loom::getFixedVectorBitWidth(geometry->vectorType, 1);
    if (!lanes)
      return lanes.takeError();
    llvm::Expected<std::uint64_t> data =
        loom::getFixedVectorBitWidth(geometry->vectorType, *elementBits);
    if (!data)
      return data.takeError();
    laneCount = *lanes;
    dataBits = *data;
  }

  std::uint64_t addressCount = 1;
  std::uint64_t addressBits = *addressLaneBits;
  if (geometry->isGather()) {
    llvm::Expected<std::uint64_t> addresses =
        loom::getFixedVectorBitWidth(geometry->addressVectorType, 1);
    if (!addresses)
      return addresses.takeError();
    llvm::Expected<std::uint64_t> bits = loom::getFixedVectorBitWidth(
        geometry->addressVectorType, *addressLaneBits);
    if (!bits)
      return bits.takeError();
    addressCount = *addresses;
    addressBits = *bits;
  }

  return CanonicalMemoryAccessView(
      std::move(*projection), actor->operation, *geometry, *contract, maskType,
      CanonicalMemoryAccessView::DerivedGeometry{
          laneCount, addressCount, *elementBits, dataBits, *addressLaneBits,
          addressBits, maskType ? laneCount : 0});
}

//===----------------------------------------------------------------------===//
// Canonical Service Schema 2.0
//===----------------------------------------------------------------------===//

llvm::StringRef dataflow::semantics::stringifyServiceKind(ServiceKind kind) {
  switch (kind) {
  case ServiceKind::MessageTransfer:
    return "message_transfer";
  case ServiceKind::MemoryRead:
    return "memory_read";
  case ServiceKind::MemoryWrite:
    return "memory_write";
  case ServiceKind::MemoryAtomicRmw:
    return "memory_atomic_rmw";
  case ServiceKind::MemoryCompareExchange:
    return "memory_compare_exchange";
  case ServiceKind::MemoryFence:
    return "memory_fence";
  }
  llvm_unreachable("unhandled canonical service kind");
}

ServiceKind dataflow::semantics::getServiceKind(MemoryAccessOperation op) {
  switch (op) {
  case MemoryAccessOperation::Load:
    return ServiceKind::MemoryRead;
  case MemoryAccessOperation::Store:
    return ServiceKind::MemoryWrite;
  case MemoryAccessOperation::AtomicRmw:
    return ServiceKind::MemoryAtomicRmw;
  case MemoryAccessOperation::CompareExchange:
    return ServiceKind::MemoryCompareExchange;
  }
  llvm_unreachable("unhandled canonical memory access operation");
}

const ServiceRoleSchema &
dataflow::semantics::getServiceRoleSchema(ServiceKind kind) {
  static const ServiceRoleSchema message{messageSchema.arguments,
                                         messageSchema.results};
  static const ServiceRoleSchema read{readSchema.arguments, readSchema.results};
  static const ServiceRoleSchema write{writeSchema.arguments,
                                       writeSchema.results};
  static const ServiceRoleSchema rmw{rmwSchema.arguments, rmwSchema.results};
  static const ServiceRoleSchema compareExchange{
      compareExchangeSchema.arguments, compareExchangeSchema.results};
  static const ServiceRoleSchema fence{fenceSchema.arguments,
                                       fenceSchema.results};

  switch (kind) {
  case ServiceKind::MessageTransfer:
    return message;
  case ServiceKind::MemoryRead:
    return read;
  case ServiceKind::MemoryWrite:
    return write;
  case ServiceKind::MemoryAtomicRmw:
    return rmw;
  case ServiceKind::MemoryCompareExchange:
    return compareExchange;
  case ServiceKind::MemoryFence:
    return fence;
  }
  llvm_unreachable("unhandled canonical service kind");
}

llvm::Expected<ServiceKind>
dataflow::semantics::getMemoryServiceKind(OperationSchemaId actorSchema) {
  switch (actorSchema) {
  case OperationSchemaId::DataflowLoad:
    return ServiceKind::MemoryRead;
  case OperationSchemaId::DataflowStore:
    return ServiceKind::MemoryWrite;
  case OperationSchemaId::DataflowAtomicRmw:
    return ServiceKind::MemoryAtomicRmw;
  case OperationSchemaId::DataflowCmpXchg:
    return ServiceKind::MemoryCompareExchange;
  case OperationSchemaId::DataflowFence:
    return ServiceKind::MemoryFence;
  default:
    return schemaError("operation schema '" +
                       operationSchemaSpelling(actorSchema) +
                       "' is not a canonical memory actor");
  }
}

llvm::Expected<CanonicalService>
CanonicalService::messageTransfer(Type payload) {
  if (!payload)
    return schemaError("a message transfer requires one exact payload type");
  // The exact supported payload domain is the one Dataflow transfer payload
  // domain a channel element type already obeys. This schema defers to that
  // owner rather than restating which types a transfer may carry.
  if (llvm::Error error = DataflowDialect::validateTransferPayloadType(
          payload, "message transfer payload"))
    return std::move(error);
  return CanonicalService(Parameter(std::in_place_type<Type>, payload));
}

llvm::Expected<CanonicalService> CanonicalService::forActor(Operation *op) {
  // A fence orders its consistency domain and addresses no memory, so its one
  // parameter is the exact contract it already owns. The same structural gate
  // runs first, so that contract is read only after the operation's own
  // verification has accepted it.
  if (llvm::isa_and_present<FenceOp>(op)) {
    if (llvm::Error error = verifyActorStructure(op))
      return std::move(error);
    llvm::Expected<CanonicalActorSchemaProjection> projection =
        projectRegisteredActorSchemaProjection(op);
    if (!projection)
      return projection.takeError();
    FenceServiceParameter fence{std::move(*projection),
                                llvm::cast<FenceOp>(op).getContract()};
    return CanonicalService(
        Parameter(std::in_place_type<FenceServiceParameter>, std::move(fence)));
  }

  llvm::Expected<CanonicalMemoryAccessView> view =
      getCanonicalMemoryAccessView(op);
  if (!view)
    return view.takeError();
  return CanonicalService(
      Parameter(std::in_place_type<CanonicalMemoryAccessView>, *view));
}

llvm::Expected<CanonicalService>
CanonicalService::forActor(Operation *op, ServiceKind expected) {
  llvm::Expected<CanonicalService> service = forActor(op);
  if (!service)
    return service;
  if (service->kind() != expected)
    return schemaError(
        "actor obliges the '" + stringifyServiceKind(service->kind()) +
        "' service, not '" + stringifyServiceKind(expected) + "'");
  return service;
}

ServiceKind CanonicalService::kind() const {
  if (const auto *view = std::get_if<CanonicalMemoryAccessView>(&parameter))
    return getServiceKind(view->operation());
  return std::holds_alternative<Type>(parameter) ? ServiceKind::MessageTransfer
                                                 : ServiceKind::MemoryFence;
}

ServiceEffect CanonicalService::effect() const {
  return getKindSchema(kind()).effect;
}

unsigned CanonicalService::legCount() const {
  return getKindSchema(kind()).legCount;
}

ServiceLegDirection CanonicalService::legDirection(unsigned ordinal) const {
  assert(ordinal < legCount() && "leg ordinal is outside this service kind");
  return ordinal == 0 ? ServiceLegDirection::InitiatorToServer
                      : ServiceLegDirection::ServerToInitiator;
}

ServiceValues CanonicalService::legPayload(unsigned ordinal) const {
  return legDirection(ordinal) == ServiceLegDirection::InitiatorToServer
             ? arguments()
             : results();
}

MLIRContext *CanonicalService::context() const {
  if (const auto *view = std::get_if<CanonicalMemoryAccessView>(&parameter))
    return view->geometry().elementType.getContext();
  if (const auto *payloadType = std::get_if<Type>(&parameter))
    return payloadType->getContext();
  return std::get<FenceServiceParameter>(parameter).contract.getContext();
}

Type CanonicalService::typeOf(ServiceValueRole role) const {
  switch (role) {
  case Role::Payload:
    return payload();
  case Role::Address:
    return access().addressType();
  // One access publishes and consumes one payload type, so the write data, the
  // read-modify-write update, both compare-exchange comparands, and the old
  // value are all `DataOf`.
  case Role::Data:
  case Role::Update:
  case Role::Expected:
  case Role::Desired:
  case Role::Old:
    return access().dataType();
  case Role::Mask:
    return access().maskType();
  case Role::Success:
    return access().successType();
  // Control and completion are pure events.
  case Role::Control:
  case Role::Completion:
    return NoneType::get(context());
  }
  llvm_unreachable("unhandled canonical service value role");
}

ServiceValues CanonicalService::arguments() const {
  ServiceValues values;
  for (Role role : getServiceRoleSchema(kind()).arguments) {
    // Only an addressed kind lists a mask, and it names one exactly when its
    // actor carries a dynamic mask.
    if (role == Role::Mask && access().maskForm() == MemoryMaskForm::Absent)
      continue;
    values.push_back({role, typeOf(role)});
  }
  return values;
}

ServiceValues CanonicalService::results() const {
  ServiceValues values;
  for (Role role : getServiceRoleSchema(kind()).results)
    values.push_back({role, typeOf(role)});
  return values;
}

Type CanonicalService::payload() const {
  assert(kind() == ServiceKind::MessageTransfer &&
         "only a message transfer carries a payload type");
  return std::get<Type>(parameter);
}

const CanonicalMemoryAccessView &CanonicalService::access() const {
  assert(std::holds_alternative<CanonicalMemoryAccessView>(parameter) &&
         "only an addressed memory kind has an access view");
  return std::get<CanonicalMemoryAccessView>(parameter);
}

FenceContractAttr CanonicalService::fenceContract() const {
  assert(kind() == ServiceKind::MemoryFence &&
         "only a fence carries a fence contract");
  return std::get<FenceServiceParameter>(parameter).contract;
}

llvm::Error dataflow::semantics::validateCanonicalMemoryActorCorrespondence(
    const CanonicalActorSchemaProjection &actor,
    const CanonicalService &service, const CanonicalMemoryAccessView *access) {
  const auto *serviceAccess =
      std::get_if<CanonicalMemoryAccessView>(&service.parameter);
  const CanonicalActorSchemaProjection *serviceActor = nullptr;
  if (serviceAccess) {
    serviceActor = &serviceAccess->sourceActor;
  } else if (const auto *fence =
                 std::get_if<CanonicalService::FenceServiceParameter>(
                     &service.parameter)) {
    serviceActor = &fence->sourceActor;
  } else {
    return schemaError(
        "parent service is not derived from a canonical memory actor");
  }

  if (*serviceActor != actor)
    return schemaError(
        "parent service does not match the exact actor projection");

  if (!serviceAccess) {
    if (access)
      return schemaError("fence correspondence must not carry an access view");
    return llvm::Error::success();
  }
  if (!access)
    return schemaError("addressed correspondence requires an access view");
  if (*serviceAccess != *access)
    return schemaError(
        "access view does not match the complete parent service projection");
  return llvm::Error::success();
}
