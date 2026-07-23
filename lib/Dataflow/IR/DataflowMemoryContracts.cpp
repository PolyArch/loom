//===- DataflowMemoryContracts.cpp - Memory access contract semantics -----===//
//
// Owns the typed memory access contracts of the canonical Dataflow memory
// actors: their well-formedness, their static legality against one access
// shape, and the nonpersistent projection consumers derive from them.
//
//===----------------------------------------------------------------------===//

#include "Dataflow/IR/DataflowActorSemantics.h"
#include "Dataflow/IR/DataflowAttrs.h"
#include "Dataflow/IR/DataflowOps.h"

#include "mlir/IR/Builders.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace dataflow;

namespace {

llvm::Error contractError(const llvm::Twine &message) {
  return llvm::createStringError(std::errc::invalid_argument, "%s",
                                 message.str().c_str());
}

llvm::Error contractError(const llvm::Twine &prefix, Type type,
                          const llvm::Twine &suffix = "") {
  std::string message;
  llvm::raw_string_ostream stream(message);
  stream << prefix << '\'' << type << '\'' << suffix;
  return contractError(message);
}

//===----------------------------------------------------------------------===//
// Atomic legality
//
// The rules below are the pinned LLVM atomic legality matrix. The element
// category of an `atomicrmw` action is taken from `llvm::AtomicRMWInst` so no
// local classification list can drift from it.
//===----------------------------------------------------------------------===//

llvm::AtomicRMWInst::BinOp toLLVMBinOp(AtomicRmwKind kind) {
  switch (kind) {
  case AtomicRmwKind::Xchg:
    return llvm::AtomicRMWInst::Xchg;
  case AtomicRmwKind::Add:
    return llvm::AtomicRMWInst::Add;
  case AtomicRmwKind::Sub:
    return llvm::AtomicRMWInst::Sub;
  case AtomicRmwKind::And:
    return llvm::AtomicRMWInst::And;
  case AtomicRmwKind::Nand:
    return llvm::AtomicRMWInst::Nand;
  case AtomicRmwKind::Or:
    return llvm::AtomicRMWInst::Or;
  case AtomicRmwKind::Xor:
    return llvm::AtomicRMWInst::Xor;
  case AtomicRmwKind::Max:
    return llvm::AtomicRMWInst::Max;
  case AtomicRmwKind::Min:
    return llvm::AtomicRMWInst::Min;
  case AtomicRmwKind::UMax:
    return llvm::AtomicRMWInst::UMax;
  case AtomicRmwKind::UMin:
    return llvm::AtomicRMWInst::UMin;
  case AtomicRmwKind::FAdd:
    return llvm::AtomicRMWInst::FAdd;
  case AtomicRmwKind::FSub:
    return llvm::AtomicRMWInst::FSub;
  case AtomicRmwKind::FMax:
    return llvm::AtomicRMWInst::FMax;
  case AtomicRmwKind::FMin:
    return llvm::AtomicRMWInst::FMin;
  case AtomicRmwKind::UIncWrap:
    return llvm::AtomicRMWInst::UIncWrap;
  case AtomicRmwKind::UDecWrap:
    return llvm::AtomicRMWInst::UDecWrap;
  case AtomicRmwKind::USubCond:
    return llvm::AtomicRMWInst::USubCond;
  case AtomicRmwKind::USubSat:
    return llvm::AtomicRMWInst::USubSat;
  case AtomicRmwKind::FMaximum:
    return llvm::AtomicRMWInst::FMaximum;
  case AtomicRmwKind::FMinimum:
    return llvm::AtomicRMWInst::FMinimum;
  case AtomicRmwKind::FMaximumNum:
    return llvm::AtomicRMWInst::FMaximumNum;
  case AtomicRmwKind::FMinimumNum:
    return llvm::AtomicRMWInst::FMinimumNum;
  }
  llvm_unreachable("unhandled atomic read-modify-write kind");
}

/// The scalar element categories one actor admits for its atomic action.
enum class AtomicElementCategory : std::uint8_t {
  Integer,
  FloatingPoint,
  IntegerOrFloatingPoint
};

AtomicElementCategory getAtomicElementCategory(Operation *op) {
  auto rmw = llvm::dyn_cast<AtomicRmwOp>(op);
  if (!rmw)
    return llvm::isa<CmpXchgOp>(op)
               ? AtomicElementCategory::Integer
               : AtomicElementCategory::IntegerOrFloatingPoint;
  llvm::AtomicRMWInst::BinOp binOp = toLLVMBinOp(rmw.getContract().getKind());
  if (binOp == llvm::AtomicRMWInst::Xchg)
    return AtomicElementCategory::IntegerOrFloatingPoint;
  return llvm::AtomicRMWInst::isFPOperation(binOp)
             ? AtomicElementCategory::FloatingPoint
             : AtomicElementCategory::Integer;
}

/// How the actor names itself in an atomic legality diagnostic.
std::string getAtomicActionName(Operation *op) {
  if (auto rmw = llvm::dyn_cast<AtomicRmwOp>(op))
    return ("atomicrmw '" + stringifyAtomicRmwKind(rmw.getContract().getKind()) +
            "'")
        .str();
  if (llvm::isa<CmpXchgOp>(op))
    return "compare-exchange";
  return llvm::isa<LoadOp>(op) ? "atomic load" : "atomic store";
}

llvm::Error validateAtomicElementCategory(Operation *op, Type element,
                                          AtomicElementCategory category) {
  const bool isInteger = llvm::isa<IntegerType>(element);
  const bool isFloat = llvm::isa<FloatType>(element);
  const char *required = "";
  switch (category) {
  case AtomicElementCategory::Integer:
    if (isInteger)
      return llvm::Error::success();
    required = "integer";
    break;
  case AtomicElementCategory::FloatingPoint:
    if (isFloat)
      return llvm::Error::success();
    required = "floating-point";
    break;
  case AtomicElementCategory::IntegerOrFloatingPoint:
    if (isInteger || isFloat)
      return llvm::Error::success();
    required = "integer or floating-point";
    break;
  }
  return contractError(getAtomicActionName(op) + " operand must have " +
                           required + " element type, got ",
                       element);
}

/// One atomic object is indivisible, so its width must be an exact static fact
/// of the actor's own types. A width that needs a data layout, such as
/// `index`, fails closed instead of assuming a host width.
llvm::Error validateAtomicObject(Type object) {
  std::uint64_t width = 0;
  if (auto vector = llvm::dyn_cast<VectorType>(object)) {
    llvm::Expected<std::uint64_t> flattened =
        semantics::getFlattenedVectorBitWidth(vector);
    if (!flattened)
      return flattened.takeError();
    width = *flattened;
  } else if (llvm::isa<IntegerType, FloatType>(object)) {
    width = object.getIntOrFloatBitWidth();
  } else {
    return contractError("atomic object type ", object,
                         " has no exact bit width without a data layout");
  }
  if (width < 8 || !llvm::isPowerOf2_64(width))
    return contractError("atomic object ", object,
                         llvm::Twine(" size ") + llvm::Twine(width) +
                             " must be a power of two of at least 8 bits");
  return llvm::Error::success();
}

/// The exact atomic type rules of one addressed atomic access. One atomic
/// object is one complete memory element: `whole_payload` names the single
/// addressed element and `per_lane` names each accessed element. The action
/// itself applies to that element's scalar type, which a vector-valued element
/// unwraps.
llvm::Error
validateAtomicAccessType(Operation *op,
                         const semantics::MemoryAccessType &access) {
  Type object = access.elementType;
  auto vector = llvm::dyn_cast<VectorType>(object);
  if (llvm::Error error = validateAtomicElementCategory(
          op, vector ? vector.getElementType() : object,
          getAtomicElementCategory(op)))
    return error;
  return validateAtomicObject(object);
}

/// Vector atomic granularity follows the canonical access geometry, not the
/// MLIR type of the complete payload. An `element` access names exactly one
/// atomic object even when that memory element is itself a vector, so it is
/// `whole_payload`; a contiguous or indexed access names one independent
/// atomic object per lane, so it is `per_lane`.
llvm::Error validateGranularity(const semantics::MemoryAccessType &access,
                                bool atomic,
                                std::optional<VectorAtomicGranularity> value) {
  // A plain contract structurally carries no granularity.
  if (!atomic)
    return llvm::Error::success();
  // An element access to a scalar memory element has one atomic object either
  // way, so both vector cases degenerate and it declares neither.
  if (!access.isVector() && !llvm::isa<VectorType>(access.elementType)) {
    if (value)
      return contractError("scalar atomic access must not declare a vector "
                           "atomic granularity");
    return llvm::Error::success();
  }
  if (!value)
    return contractError("vector atomic access must declare a vector atomic "
                         "granularity");
  if (access.isVector()) {
    if (*value == VectorAtomicGranularity::PerLane)
      return llvm::Error::success();
    return contractError("'whole_payload' atomic granularity requires an "
                         "access to one complete memory element");
  }
  if (*value == VectorAtomicGranularity::WholePayload)
    return llvm::Error::success();
  return contractError("'per_lane' atomic granularity requires an access to "
                       "independent memory elements");
}

/// The success result of one compare-exchange firing against the shape its
/// canonical access geometry publishes.
llvm::Error
validateCompareExchangeSuccess(CmpXchgOp op,
                               const semantics::MemoryAccessType &access) {
  Type expected = semantics::getCompareExchangeSuccessType(access);
  if (op.getSuccess().getType() == expected)
    return llvm::Error::success();
  return contractError(access.isVector()
                           ? "'per_lane' compare-exchange success result must "
                             "be "
                           : "scalar or 'whole_payload' compare-exchange "
                             "success result must be ",
                       expected);
}

/// No attribute other than the actor's single `contract` slot may state a
/// memory access contract, or one of the contract's own owned values, on this
/// actor. The name matches the operation definitions in DataflowOps.td, which
/// is where the one slot is declared. This is an actor-local rule: the same
/// typed values remain ordinary attribute values everywhere else, and
/// unrelated discardable metadata stays legal.
llvm::Error validateSingleContractOwner(Operation *op) {
  for (NamedAttribute attribute : op->getAttrs()) {
    if (attribute.getName() == "contract")
      continue;
    llvm::StringRef name = attribute.getName().strref();
    if (llvm::isa<PlainAccessContractAttr, AtomicAccessContractAttr,
                  AtomicRmwContractAttr, CompareExchangeContractAttr,
                  FenceContractAttr>(attribute.getValue()))
      return contractError("'" + name +
                           "' must not carry a second aggregate memory "
                           "contract");
    if (llvm::isa<SyncScopeRefAttr>(attribute.getValue()))
      return contractError("'" + name +
                           "' must not carry a second synchronization scope");
  }
  return llvm::Error::success();
}

/// A target-namespaced scope key is representable, but no authoritative
/// resolver owns its meaning yet, so an actor referencing one fails closed.
llvm::Error validateResolvedSyncScope(SyncScopeRefAttr scope) {
  if (!scope || scope.getKind() != SyncScopeKind::Target)
    return llvm::Error::success();
  return contractError(
      "target synchronization scope '" + scope.getTargetNamespace().getValue() +
      "::" + scope.getTargetKey().getValue() +
      "' is unresolved: no compiler-target contract can prove it yet");
}

/// The orderings an atomic load and a compare-exchange failure path reject.
bool isReleaseOrAcqRel(AtomicOrdering ordering) {
  return ordering == AtomicOrdering::Release ||
         ordering == AtomicOrdering::AcqRel;
}

/// The orderings an atomic store rejects.
bool isAcquireOrAcqRel(AtomicOrdering ordering) {
  return ordering == AtomicOrdering::Acquire ||
         ordering == AtomicOrdering::AcqRel;
}

} // namespace

//===----------------------------------------------------------------------===//
// Attribute well-formedness
//===----------------------------------------------------------------------===//

LogicalResult
SyncScopeRefAttr::verify(llvm::function_ref<InFlightDiagnostic()> emitError,
                         SyncScopeKind kind, StringAttr targetNamespace,
                         StringAttr targetKey) {
  const bool named = targetNamespace && targetKey &&
                     !targetNamespace.getValue().empty() &&
                     !targetKey.getValue().empty();
  if (kind == SyncScopeKind::Target && !named)
    return emitError() << "'target' synchronization scope requires a target "
                          "namespace and key";
  if (kind != SyncScopeKind::Target && (targetNamespace || targetKey))
    return emitError() << "only a 'target' synchronization scope carries a "
                          "target namespace and key";
  return success();
}

LogicalResult AtomicAccessContractAttr::verify(
    llvm::function_ref<InFlightDiagnostic()> emitError, AtomicOrdering ordering,
    SyncScopeRefAttr syncScope,
    std::optional<VectorAtomicGranularity> granularity, bool isVolatile) {
  if (!syncScope)
    return emitError() << "atomic access contract requires a synchronization "
                          "scope";
  return success();
}

LogicalResult CompareExchangeContractAttr::verify(
    llvm::function_ref<InFlightDiagnostic()> emitError,
    AtomicOrdering successOrdering, AtomicOrdering failureOrdering,
    SyncScopeRefAttr syncScope,
    std::optional<VectorAtomicGranularity> granularity, bool weak,
    bool isVolatile) {
  if (!syncScope)
    return emitError() << "compare-exchange contract requires a "
                          "synchronization scope";
  if (successOrdering == AtomicOrdering::Unordered ||
      failureOrdering == AtomicOrdering::Unordered)
    return emitError() << "compare-exchange orderings must not be 'unordered'";
  if (isReleaseOrAcqRel(failureOrdering))
    return emitError() << "compare-exchange failure ordering must not be "
                          "'release' or 'acq_rel'";
  return success();
}

LogicalResult
FenceContractAttr::verify(llvm::function_ref<InFlightDiagnostic()> emitError,
                          AtomicOrdering ordering, SyncScopeRefAttr syncScope) {
  if (!syncScope)
    return emitError() << "fence contract requires a synchronization scope";
  if (ordering == AtomicOrdering::Unordered ||
      ordering == AtomicOrdering::Monotonic)
    return emitError() << "fence ordering must be 'acquire', 'release', "
                          "'acq_rel', or 'seq_cst'";
  return success();
}

//===----------------------------------------------------------------------===//
// Actor projection and legality
//===----------------------------------------------------------------------===//

/// An access with lanes is `per_lane` and publishes one bit per lane; an
/// `element` access has one atomic object and publishes one bit, whatever that
/// element's payload type is.
Type dataflow::semantics::getCompareExchangeSuccessType(
    const MemoryAccessType &access) {
  Type success = Builder(access.elementType.getContext()).getI1Type();
  if (access.isVector())
    return VectorType::get(access.vectorType.getShape(), success);
  return success;
}

std::optional<semantics::MemoryActorContract>
dataflow::semantics::getMemoryActorContract(Operation *op) {
  return llvm::TypeSwitch<Operation *, std::optional<MemoryActorContract>>(op)
      .Case<LoadOp, StoreOp>([](auto typedOp) {
        Attribute aggregate = typedOp.getContractAttr();
        if (!aggregate)
          aggregate = PlainAccessContractAttr::get(typedOp.getContext(),
                                                   /*is_volatile=*/false);
        if (auto plain = llvm::dyn_cast<PlainAccessContractAttr>(aggregate))
          return MemoryActorContract{aggregate, /*atomic=*/false,
                                     plain.getIsVolatile(), std::nullopt,
                                     SyncScopeRefAttr()};
        auto access = llvm::cast<AtomicAccessContractAttr>(aggregate);
        return MemoryActorContract{aggregate, /*atomic=*/true,
                                   access.getIsVolatile(),
                                   access.getVectorGranularity(),
                                   access.getSyncScope()};
      })
      .Case<AtomicRmwOp>([](AtomicRmwOp typedOp) {
        AtomicRmwContractAttr aggregate = typedOp.getContract();
        AtomicAccessContractAttr access = aggregate.getAccess();
        return MemoryActorContract{aggregate, /*atomic=*/true,
                                   access.getIsVolatile(),
                                   access.getVectorGranularity(),
                                   access.getSyncScope()};
      })
      .Case<CmpXchgOp>([](CmpXchgOp typedOp) {
        CompareExchangeContractAttr aggregate = typedOp.getContract();
        return MemoryActorContract{aggregate, /*atomic=*/true,
                                   aggregate.getIsVolatile(),
                                   aggregate.getVectorGranularity(),
                                   aggregate.getSyncScope()};
      })
      // A fence is ordered by construction and addresses no memory.
      .Case<FenceOp>([](FenceOp typedOp) {
        FenceContractAttr aggregate = typedOp.getContract();
        return MemoryActorContract{aggregate, /*atomic=*/true,
                                   /*isVolatile=*/false, std::nullopt,
                                   aggregate.getSyncScope()};
      })
      .Default([](Operation *) { return std::nullopt; });
}

Value dataflow::semantics::getMemoryActorDone(Operation *op) {
  return llvm::TypeSwitch<Operation *, Value>(op)
      .Case<LoadOp, StoreOp, AtomicRmwOp, CmpXchgOp, FenceOp>(
          [](auto typedOp) { return typedOp.getDone(); })
      .Default([](Operation *) { return Value(); });
}

Value dataflow::semantics::getMemoryActorControl(Operation *op) {
  return llvm::TypeSwitch<Operation *, Value>(op)
      .Case<LoadOp, StoreOp, AtomicRmwOp, CmpXchgOp, FenceOp>(
          [](auto typedOp) { return typedOp.getCtrl(); })
      .Default([](Operation *) { return Value(); });
}

void dataflow::semantics::getMemoryActorEffects(
    Operation *op,
    llvm::SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  std::optional<MemoryActorContract> contract = getMemoryActorContract(op);
  if (!contract)
    return;

  // The addressed base projection names the actor's memory operand: a load
  // observes it, a store modifies it, and an atomic read-modify-write or
  // compare-exchange does both.
  OpOperand *memory =
      llvm::TypeSwitch<Operation *, OpOperand *>(op)
          .Case<LoadOp, StoreOp, AtomicRmwOp, CmpXchgOp>(
              [](auto typedOp) { return &typedOp.getMemMutable(); })
          .Default([](Operation *) { return nullptr; });
  if (memory) {
    if (!llvm::isa<StoreOp>(op))
      effects.emplace_back(MemoryEffects::Read::get(), memory);
    if (!llvm::isa<LoadOp>(op))
      effects.emplace_back(MemoryEffects::Write::get(), memory);
  }

  // An atomic actor participates in its consistency domain's dynamic state,
  // and a volatile actor carries an observability contract that outlives having
  // no consumer. Neither is a claim that the actor synchronizes: an unordered
  // or monotonic atomic need not form any synchronizes-with relation. Both
  // facts are read back from the one aggregate contract and projected
  // conservatively as unbound effects, because MLIR may erase an operation
  // whose effects are all reads, so the unbound write is what keeps such an
  // actor alive. A fence addresses no memory and therefore publishes only
  // these. Duplicate effect kinds across the bound and unbound forms are
  // intentional.
  if (contract->atomic || contract->isVolatile) {
    effects.emplace_back(MemoryEffects::Read::get());
    effects.emplace_back(MemoryEffects::Write::get());
  }
}

llvm::Error dataflow::semantics::validateMemoryActorContract(
    Operation *op, const std::optional<MemoryAccessType> &access) {
  std::optional<MemoryActorContract> contract = getMemoryActorContract(op);
  if (!contract)
    return contractError("operation is not a canonical memory actor");
  if (llvm::Error error = validateSingleContractOwner(op))
    return error;
  if (llvm::Error error = validateResolvedSyncScope(contract->syncScope))
    return error;

  // Only dataflow.load and dataflow.store nest a bare atomic access contract,
  // and each rejects the orderings its direction cannot express.
  if (auto atomic =
          llvm::dyn_cast<AtomicAccessContractAttr>(contract->aggregate)) {
    if (llvm::isa<LoadOp>(op) && isReleaseOrAcqRel(atomic.getOrdering()))
      return contractError("atomic load ordering must not be 'release' or "
                           "'acq_rel'");
    if (llvm::isa<StoreOp>(op) && isAcquireOrAcqRel(atomic.getOrdering()))
      return contractError("atomic store ordering must not be 'acquire' or "
                           "'acq_rel'");
  }
  if (auto rmw = llvm::dyn_cast<AtomicRmwOp>(op))
    if (rmw.getContract().getAccess().getOrdering() == AtomicOrdering::Unordered)
      return contractError("atomic read-modify-write ordering must not be "
                           "'unordered'");
  if (!access)
    return llvm::Error::success();

  if (llvm::Error error = validateGranularity(*access, contract->atomic,
                                              contract->vectorGranularity))
    return error;
  if (contract->atomic)
    if (llvm::Error error = validateAtomicAccessType(op, *access))
      return error;
  if (auto cmpxchg = llvm::dyn_cast<CmpXchgOp>(op))
    return validateCompareExchangeSuccess(cmpxchg, *access);
  return llvm::Error::success();
}
