#ifndef DATAFLOW_IR_OPERATIONSCHEMA_H
#define DATAFLOW_IR_OPERATIONSCHEMA_H

#include "Common/SpecialMathAccuracy.h"
#include "Dataflow/IR/DataflowEnums.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>
#include <variant>
#include <vector>

namespace dataflow {

/// The stable identity of one registered canonical actor operation schema.
///
/// `include/Dataflow/IR/OperationSchemas.td` is the one source of this domain.
/// The enumerators are dense internal indices for generated tables. Persistent
/// identity uses the registry-owned wire codec, never these numeric values.
enum class OperationSchemaId : std::uint32_t {
#define LOOM_OPERATION_SCHEMA(Name, Id, WireTag, OpClass, ActorKind,           \
                              SemanticsCase, SelectorKind, SelectorValue,      \
                              ElementwiseDecomposable)                         \
  Name = Id,
#include "Dataflow/IR/OperationSchemas.inc"
};

/// The closed typed semantic vocabulary. A case names which typed record an
/// actor's identity-critical semantic payload uses; no semantic value is ever
/// reached through an attribute name.
enum class OperationSemanticsCase : std::uint32_t {
#define LOOM_OPERATION_SEMANTICS_CASE(Name, Id, WireTag) Name = Id,
#include "Dataflow/IR/OperationSchemas.inc"
};

bool isValidFastMathFlags(::mlir::arith::FastMathFlags flags);

enum class CanonicalDataflowActorKind : std::uint32_t {
  Compute,
  Control,
  Memory
};

/// Count of registered schemas. Every schema id is in `[0, count)`.
std::uint32_t operationSchemaCount();

/// The stable readable spelling of one schema. A whole-class schema derives
/// it from the registered operation name; a generic carrier derives it from
/// the selected source-owner registry. It is never a hand-written alias.
llvm::StringRef operationSchemaSpelling(OperationSchemaId schema);

/// The stable spelling of one semantic case, used only for diagnostics.
llvm::StringRef operationSemanticsCaseSpelling(OperationSemanticsCase kind);

/// The schema registered for `spelling`, or absent when none is. Lookup is one
/// hash probe over a table built once.
std::optional<OperationSchemaId> findOperationSchema(llvm::StringRef spelling);

/// The schema of `op`, or absent when `op` is not a registered canonical
/// actor.
std::optional<OperationSchemaId> operationSchemaOf(::mlir::Operation *op);

/// The schema of an operation already known to be registered. Calling it for
/// an unregistered operation is a programming error, not a runtime condition.
OperationSchemaId requireOperationSchema(::mlir::Operation *op);

CanonicalDataflowActorKind actorKind(OperationSchemaId schema);
OperationSemanticsCase semanticsCase(OperationSchemaId schema);

/// Whether a fixed-vector instance is exactly the pointwise lift of the same
/// scalar operation and may therefore use the canonical elementwise
/// decomposition rewrite.
bool supportsElementwiseVectorDecomposition(OperationSchemaId schema);

/// Recomputes source-owned selector state after a type-preserving-schema
/// rewrite changes one actor's exact function type. Whole-class carriers need
/// no mutation; a registered LLVM intrinsic carrier receives the exact
/// overloaded spelling derived by LLVM's registry. The resulting instance is
/// verified against `schema`.
llvm::Error canonicalizeRegisteredActorInstance(OperationSchemaId schema,
                                                ::mlir::Operation *op);

/// Canonical actor classification. An operation is a canonical actor exactly
/// when the registry declares a schema for it; the classifier is a derived
/// query over that one registry rather than another whitelist. An operation
/// with no registered schema, such as an LLVM-dialect alias that mechanical
/// raising has not yet normalized to its standard schema, fails closed here.
std::optional<CanonicalDataflowActorKind>
classifyCanonicalDataflowActor(::mlir::Operation *op);

bool isCanonicalDataflowActor(::mlir::Operation *op);
bool isCanonicalDataflowActor(::mlir::Operation *op,
                              CanonicalDataflowActorKind kind);

//===---------------------------------------------------------------------===//
// Closed typed semantic payload records
//
// One record per declared semantic case, each holding the exact typed values
// its owning operation class exposes through typed accessors. There is no
// attribute dictionary, raw attribute alternative, or generic property bag.
//===---------------------------------------------------------------------===//

/// The actor states no semantic fact beyond its exact function type.
struct NoPayload {
  friend bool operator==(NoPayload, NoPayload) { return true; }
};

/// Floating-point policy not encoded by the actor's function type.
struct FloatingPointPayload {
  ::mlir::arith::FastMathFlags flags = ::mlir::arith::FastMathFlags::none;
  std::optional<::mlir::arith::RoundingMode> roundingMode;

  friend bool operator==(FloatingPointPayload lhs, FloatingPointPayload rhs) {
    return lhs.flags == rhs.flags && lhs.roundingMode == rhs.roundingMode;
  }
};

/// Native floating permissions and the selected special-math accuracy.
struct SpecialMathPayload {
  ::mlir::arith::FastMathFlags flags = ::mlir::arith::FastMathFlags::none;
  ::loom::SpecialMathAccuracyTier accuracy =
      ::loom::SpecialMathAccuracyTier::CorrectlyRounded;

  friend bool operator==(SpecialMathPayload lhs, SpecialMathPayload rhs) {
    return lhs.flags == rhs.flags && lhs.accuracy == rhs.accuracy;
  }
};

/// The exact flag on integer division and right-shift actors.
struct ExactPayload {
  bool isExact = false;

  friend bool operator==(ExactPayload lhs, ExactPayload rhs) {
    return lhs.isExact == rhs.isExact;
  }
};

/// The non-negative operand assumption on unsigned conversion actors.
struct NonNegativePayload {
  bool isNonNegative = false;

  friend bool operator==(NonNegativePayload lhs, NonNegativePayload rhs) {
    return lhs.isNonNegative == rhs.isNonNegative;
  }
};

/// Integer overflow assumptions, which constrain the legal software inputs of
/// one firing.
struct IntegerOverflowPayload {
  ::mlir::arith::IntegerOverflowFlags flags =
      ::mlir::arith::IntegerOverflowFlags::none;

  friend bool operator==(IntegerOverflowPayload lhs,
                         IntegerOverflowPayload rhs) {
    return lhs.flags == rhs.flags;
  }
};

struct IntegerComparePayload {
  ::mlir::arith::CmpIPredicate predicate = ::mlir::arith::CmpIPredicate::eq;

  friend bool operator==(IntegerComparePayload lhs, IntegerComparePayload rhs) {
    return lhs.predicate == rhs.predicate;
  }
};

struct FloatComparePayload {
  ::mlir::arith::CmpFPredicate predicate =
      ::mlir::arith::CmpFPredicate::AlwaysFalse;
  ::mlir::arith::FastMathFlags flags = ::mlir::arith::FastMathFlags::none;

  friend bool operator==(FloatComparePayload lhs, FloatComparePayload rhs) {
    return lhs.predicate == rhs.predicate && lhs.flags == rhs.flags;
  }
};

/// The zero-input poison contract of an LLVM count-zeros actor. The flag-free
/// form is exactly a math operation and never reaches this projection.
struct ZeroPoisonPayload {
  bool isZeroPoison = false;

  friend bool operator==(ZeroPoisonPayload lhs, ZeroPoisonPayload rhs) {
    return lhs.isZeroPoison == rhs.isZeroPoison;
  }
};

/// The integer-minimum poison contract of an LLVM absolute-value actor. The
/// flag-free form is exactly a math operation and never reaches this
/// projection.
struct IntegerMinPoisonPayload {
  bool isIntMinPoison = false;

  friend bool operator==(IntegerMinPoisonPayload lhs,
                         IntegerMinPoisonPayload rhs) {
    return lhs.isIntMinPoison == rhs.isIntMinPoison;
  }
};

/// The no-common-set-bits poison contract of an LLVM integer OR actor. The
/// flag-free form is mechanically raised to arith.ori and never reaches this
/// projection.
struct DisjointPayload {
  bool isDisjoint = false;

  friend bool operator==(DisjointPayload lhs, DisjointPayload rhs) {
    return lhs.isDisjoint == rhs.isDisjoint;
  }
};

/// The statically selected element of an LLVM aggregate actor.
struct AggregatePositionPayload {
  std::vector<std::int64_t> position;

  friend bool operator==(const AggregatePositionPayload &lhs,
                         const AggregatePositionPayload &rhs) {
    return lhs.position == rhs.position;
  }
};

/// The exact typed indexing path and no-wrap contract of one LLVM GEP. The
/// function type owns the base, dynamic indices, and result pointer types.
struct GetElementPtrPayload {
  ::mlir::Type sourceElementType;
  std::vector<std::int32_t> rawConstantIndices;
  ::mlir::LLVM::GEPNoWrapFlags noWrapFlags = ::mlir::LLVM::GEPNoWrapFlags::none;

  friend bool operator==(const GetElementPtrPayload &lhs,
                         const GetElementPtrPayload &rhs) {
    return lhs.sourceElementType == rhs.sourceElementType &&
           lhs.rawConstantIndices == rhs.rawConstantIndices &&
           lhs.noWrapFlags == rhs.noWrapFlags;
  }
};

/// The static dimension selections of a fixed-vector extract or insert.
/// Dynamic selections remain operands and are already represented by the
/// actor's function type.
struct VectorStaticPositionPayload {
  std::vector<std::int64_t> position;

  friend bool operator==(const VectorStaticPositionPayload &lhs,
                         const VectorStaticPositionPayload &rhs) {
    return lhs.position == rhs.position;
  }
};

/// The exact lane selection of a fixed-vector shuffle. A value of -1 retains
/// its native poison meaning.
struct VectorShuffleMaskPayload {
  std::vector<std::int64_t> mask;

  friend bool operator==(const VectorShuffleMaskPayload &lhs,
                         const VectorShuffleMaskPayload &rhs) {
    return lhs.mask == rhs.mask;
  }
};

/// The exact typed value a constant actor emits. It is the operation's own
/// typed value attribute, whose type equals the actor's result type.
struct ConstantValuePayload {
  ::mlir::TypedAttr value;

  friend bool operator==(ConstantValuePayload lhs, ConstantValuePayload rhs) {
    return lhs.value == rhs.value;
  }
};

/// The exact recurrence a stream actor performs.
struct StreamRecurrencePayload {
  StreamStepKind stepKind = StreamStepKind::Add;
  ::mlir::arith::CmpIPredicate predicate = ::mlir::arith::CmpIPredicate::slt;

  friend bool operator==(StreamRecurrencePayload lhs,
                         StreamRecurrencePayload rhs) {
    return lhs.stepKind == rhs.stepKind && lhs.predicate == rhs.predicate;
  }
};

//===---------------------------------------------------------------------===//
// Memory contract projection
//
// The Dataflow memory contract attributes are already closed typed records.
// The projection restates their exact typed fields so a consumer reads a
// value rather than an attribute, and so a contract shape that does not match
// the actor's declared schema fails closed.
//===---------------------------------------------------------------------===//

struct SyncScopeProjection {
  SyncScopeKind kind = SyncScopeKind::System;
  ::mlir::StringAttr targetNamespace;
  ::mlir::StringAttr targetKey;

  friend bool operator==(const SyncScopeProjection &lhs,
                         const SyncScopeProjection &rhs) {
    return lhs.kind == rhs.kind && lhs.targetNamespace == rhs.targetNamespace &&
           lhs.targetKey == rhs.targetKey;
  }
};

struct PlainAccessProjection {
  bool isVolatile = false;

  friend bool operator==(PlainAccessProjection lhs, PlainAccessProjection rhs) {
    return lhs.isVolatile == rhs.isVolatile;
  }
};

struct AtomicAccessProjection {
  AtomicOrdering ordering = AtomicOrdering::Unordered;
  SyncScopeProjection scope;
  std::uint64_t sourceAlignmentBytes = 1;
  std::optional<VectorAtomicGranularity> vectorGranularity;
  bool isVolatile = false;

  friend bool operator==(const AtomicAccessProjection &lhs,
                         const AtomicAccessProjection &rhs) {
    return lhs.ordering == rhs.ordering && lhs.scope == rhs.scope &&
           lhs.sourceAlignmentBytes == rhs.sourceAlignmentBytes &&
           lhs.vectorGranularity == rhs.vectorGranularity &&
           lhs.isVolatile == rhs.isVolatile;
  }
};

struct AtomicRmwProjection {
  AtomicRmwKind kind = AtomicRmwKind::Xchg;
  AtomicAccessProjection access;

  friend bool operator==(const AtomicRmwProjection &lhs,
                         const AtomicRmwProjection &rhs) {
    return lhs.kind == rhs.kind && lhs.access == rhs.access;
  }
};

struct CompareExchangeProjection {
  AtomicOrdering successOrdering = AtomicOrdering::Unordered;
  AtomicOrdering failureOrdering = AtomicOrdering::Unordered;
  SyncScopeProjection scope;
  std::uint64_t sourceAlignmentBytes = 1;
  std::optional<VectorAtomicGranularity> vectorGranularity;
  bool weak = false;
  bool isVolatile = false;

  friend bool operator==(const CompareExchangeProjection &lhs,
                         const CompareExchangeProjection &rhs) {
    return lhs.successOrdering == rhs.successOrdering &&
           lhs.failureOrdering == rhs.failureOrdering &&
           lhs.scope == rhs.scope &&
           lhs.sourceAlignmentBytes == rhs.sourceAlignmentBytes &&
           lhs.vectorGranularity == rhs.vectorGranularity &&
           lhs.weak == rhs.weak && lhs.isVolatile == rhs.isVolatile;
  }
};

struct FenceProjection {
  AtomicOrdering ordering = AtomicOrdering::Unordered;
  SyncScopeProjection scope;

  friend bool operator==(const FenceProjection &lhs,
                         const FenceProjection &rhs) {
    return lhs.ordering == rhs.ordering && lhs.scope == rhs.scope;
  }
};

/// The closed memory-contract sum. Its alternatives are the exact contract
/// shapes the Dataflow memory actors declare.
using MemoryContractPayload =
    std::variant<PlainAccessProjection, AtomicAccessProjection,
                 AtomicRmwProjection, CompareExchangeProjection,
                 FenceProjection>;

/// The closed payload sum, one alternative per declared semantic case.
using SemanticPayload = std::variant<
    NoPayload, FloatingPointPayload, SpecialMathPayload, ExactPayload,
    NonNegativePayload, IntegerOverflowPayload, IntegerComparePayload,
    FloatComparePayload, ConstantValuePayload, StreamRecurrencePayload,
    MemoryContractPayload, ZeroPoisonPayload, IntegerMinPoisonPayload,
    AggregatePositionPayload, VectorStaticPositionPayload,
    VectorShuffleMaskPayload, DisjointPayload, GetElementPtrPayload>;

/// The complete identity-critical projection of one canonical actor instance.
struct CanonicalActorSchemaProjection {
  OperationSchemaId schema;
  ::mlir::FunctionType type;
  SemanticPayload payload;

  friend bool operator==(const CanonicalActorSchemaProjection &lhs,
                         const CanonicalActorSchemaProjection &rhs) {
    return lhs.schema == rhs.schema && lhs.type == rhs.type &&
           lhs.payload == rhs.payload;
  }
  friend bool operator!=(const CanonicalActorSchemaProjection &lhs,
                         const CanonicalActorSchemaProjection &rhs) {
    return !(lhs == rhs);
  }
};

/// The transition descriptor identity of one actor instance. Two instances
/// share a descriptor exactly when their typed projections are equal, so a
/// consumer may key one transition implementation on it.
struct TransitionDescriptorIdentity {
  CanonicalActorSchemaProjection projection;

  friend bool operator==(const TransitionDescriptorIdentity &lhs,
                         const TransitionDescriptorIdentity &rhs) {
    return lhs.projection == rhs.projection;
  }
  friend bool operator!=(const TransitionDescriptorIdentity &lhs,
                         const TransitionDescriptorIdentity &rhs) {
    return !(lhs == rhs);
  }
};

/// Projects a registered actor instance. Failure means the instance does not
/// carry the exact typed state its declared semantic case requires; there is
/// no empty-payload fallback.
llvm::Expected<CanonicalActorSchemaProjection>
projectRegisteredActorSchemaProjection(::mlir::Operation *op);

/// Returns the one LLVM pointer address space used by an actor's exact
/// function type. Actors without pointer endpoints return no value. Mixed
/// address spaces are rejected because one concrete admission query must use
/// one exact DataLayout-derived pointer format.
llvm::Expected<std::optional<std::uint32_t>> projectActorPointerAddressSpace(
    const CanonicalActorSchemaProjection &projection);

/// Verifies that a registered actor instance carries exactly the typed state
/// its schema's declared semantic case owns.
::mlir::LogicalResult verifyRegisteredActorInstance(::mlir::Operation *op);

llvm::Expected<TransitionDescriptorIdentity>
deriveTransitionDescriptorIdentity(::mlir::Operation *op);

} // namespace dataflow

#endif // DATAFLOW_IR_OPERATIONSCHEMA_H
