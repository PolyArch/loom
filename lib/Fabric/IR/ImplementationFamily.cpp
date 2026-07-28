//===- ImplementationFamily.cpp - Normative HSG family registry ----------===//
//
// Expands the one generated implementation-family registry into dense tables.
// The member relation, the capability-parameter schema binding, and the typed
// admission provider binding all come from the same generated rows, so there
// is no second member list, family-shape switch, or backend-local table.
//
//===----------------------------------------------------------------------===//

#include "Fabric/IR/ImplementationFamily.h"

#include "Common/IndexWidth.h"
#include "Common/VectorWidth.h"
#include "Dataflow/IR/DataflowActorSemantics.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <array>
#include <cstddef>
#include <string>
#include <type_traits>

namespace {

constexpr std::size_t kFamilyCount = 0
#define LOOM_IMPLEMENTATION_FAMILY(Name, Id, CapabilityParams, TypedAdmission) \
  +1
#include "Fabric/IR/ImplementationFamilies.inc"
    ;

/// The admitted schemas of each family, laid out one contiguous array per
/// family so a descriptor holds a span into immutable generated storage.
#define LOOM_IMPLEMENTATION_FAMILY(Name, Id, CapabilityParams, TypedAdmission) \
  constexpr ::dataflow::OperationSchemaId kMembers##Name[] = {
#define LOOM_IMPLEMENTATION_FAMILY_MEMBER(Family, Schema)                      \
  ::dataflow::OperationSchemaId::Schema,
#define LOOM_IMPLEMENTATION_FAMILY_END(Name)                                   \
  }                                                                            \
  ;
#include "Fabric/IR/ImplementationFamilies.inc"

const std::array<fabric::ImplementationFamilyDescriptor, kFamilyCount> &
familyTable() {
  static const std::array<fabric::ImplementationFamilyDescriptor, kFamilyCount>
      table = {{
#define LOOM_IMPLEMENTATION_FAMILY(Name, Id, CapabilityParams, TypedAdmission) \
  fabric::ImplementationFamilyDescriptor{                                      \
      fabric::ImplementationFamilyId::Name,                                    \
      llvm::ArrayRef<::dataflow::OperationSchemaId>(kMembers##Name),           \
      fabric::CapabilityParamsSchemaId::CapabilityParams,                      \
      fabric::TypedAdmissionProviderId::TypedAdmission},
#include "Fabric/IR/ImplementationFamilies.inc"
      }};
  return table;
}

} // namespace

namespace {

using dataflow::CanonicalActorSchemaProjection;
using dataflow::OperationSchemaId;
using fabric::FamilyCapabilityParams;
using fabric::FloatBehaviorProfile;
using fabric::FloatFormat;
using fabric::FloatFormatSet;
using fabric::FloatNaNBehavior;
using fabric::FloatSignedZeroBehavior;
using fabric::FloatSubnormalBehavior;
using fabric::IntegerPredicateSet;
using fabric::IntegerWidth;
using fabric::IntegerWidthSet;

llvm::Error reject(const llvm::Twine &message);
IntegerWidthSet ordinaryIntegerWidths();
IntegerWidthSet logicIntegerWidths();
llvm::Error validateIntegerWidths(IntegerWidthSet widths,
                                  IntegerWidthSet allowed,
                                  llvm::StringRef description);
llvm::Error validateFloatFormats(FloatFormatSet formats,
                                 llvm::StringRef description);
llvm::Error validateFloatBehavior(const FloatBehaviorProfile &behavior);
llvm::Expected<IntegerWidth> integerWidth(::mlir::Type type,
                                          llvm::StringRef relation);
unsigned integerBitWidth(IntegerWidth width);
llvm::Expected<FloatFormat> floatFormat(::mlir::Type type,
                                        llvm::StringRef relation);
unsigned floatBitWidth(FloatFormat format);
llvm::Error requireArity(const CanonicalActorSchemaProjection &actor,
                         unsigned inputs, unsigned results);
llvm::Error requireUniformType(const CanonicalActorSchemaProjection &actor,
                               unsigned inputs);
llvm::Error
admitFloatBehavior(const FloatBehaviorProfile &behavior,
                   ::mlir::arith::FastMathFlags actorFlags,
                   std::optional<::mlir::arith::RoundingMode> rounding,
                   FloatNaNBehavior nanBehavior);
llvm::Expected<::mlir::arith::FastMathFlags>
floatingFlags(const CanonicalActorSchemaProjection &actor);
std::optional<::mlir::arith::RoundingMode>
arithmeticRounding(const CanonicalActorSchemaProjection &actor);
llvm::Expected<::mlir::arith::CmpIPredicate>
integerPredicate(const CanonicalActorSchemaProjection &actor);
llvm::Expected<::mlir::arith::CmpFPredicate>
floatPredicate(const CanonicalActorSchemaProjection &actor);

llvm::Error admitScalarOrdinaryIntegerAdmission(
    const FamilyCapabilityParams &capability,
    const CanonicalActorSchemaProjection &actor);
llvm::Error admitScalarUnaryIntegerAdmission(
    const FamilyCapabilityParams &capability,
    const CanonicalActorSchemaProjection &actor);
llvm::Error
admitScalarLogicIntegerAdmission(const FamilyCapabilityParams &capability,
                                 const CanonicalActorSchemaProjection &actor);
llvm::Error
admitScalarIntegerCompareAdmission(const FamilyCapabilityParams &capability,
                                   const CanonicalActorSchemaProjection &actor);
llvm::Error
admitScalarIntegerCastAdmission(const FamilyCapabilityParams &capability,
                                const CanonicalActorSchemaProjection &actor);
llvm::Error
admitScalarBitReinterpretAdmission(const FamilyCapabilityParams &capability,
                                   const CanonicalActorSchemaProjection &actor);
llvm::Error
admitScalarValueSelectAdmission(const FamilyCapabilityParams &capability,
                                const CanonicalActorSchemaProjection &actor);
llvm::Error
admitScalarUniformFloatAdmission(const FamilyCapabilityParams &capability,
                                 const CanonicalActorSchemaProjection &actor);
llvm::Error
admitScalarFloatCompareAdmission(const FamilyCapabilityParams &capability,
                                 const CanonicalActorSchemaProjection &actor);
llvm::Error
admitScalarFloatCastAdmission(const FamilyCapabilityParams &capability,
                              const CanonicalActorSchemaProjection &actor);
llvm::Error admitScalarIntegerFloatConversionAdmission(
    const FamilyCapabilityParams &capability,
    const CanonicalActorSchemaProjection &actor);
llvm::Error admitStreamAdmission(const FamilyCapabilityParams &capability,
                                 const CanonicalActorSchemaProjection &actor);
llvm::Error
admitTokenPlaneAdmission(const FamilyCapabilityParams &capability,
                         const CanonicalActorSchemaProjection &actor);
llvm::Error admitFixedVectorOrdinaryIntegerAdmission(
    const FamilyCapabilityParams &capability,
    const CanonicalActorSchemaProjection &actor);
llvm::Error admitFixedVectorUnaryIntegerAdmission(
    const FamilyCapabilityParams &capability,
    const CanonicalActorSchemaProjection &actor);
llvm::Error admitFixedVectorLogicIntegerAdmission(
    const FamilyCapabilityParams &capability,
    const CanonicalActorSchemaProjection &actor);
llvm::Error admitFixedVectorIntegerCompareAdmission(
    const FamilyCapabilityParams &capability,
    const CanonicalActorSchemaProjection &actor);
llvm::Error admitFixedVectorValueSelectAdmission(
    const FamilyCapabilityParams &capability,
    const CanonicalActorSchemaProjection &actor);
llvm::Error admitFixedVectorUniformFloatAdmission(
    const FamilyCapabilityParams &capability,
    const CanonicalActorSchemaProjection &actor);
llvm::Error admitFixedVectorFloatCompareAdmission(
    const FamilyCapabilityParams &capability,
    const CanonicalActorSchemaProjection &actor);
llvm::Error
admitFixedVectorAdapterAdmission(const FamilyCapabilityParams &capability,
                                 const CanonicalActorSchemaProjection &actor);
llvm::Error
admitConstantTokenAdmission(const FamilyCapabilityParams &capability,
                            const CanonicalActorSchemaProjection &actor);
llvm::Error
admitSyncTokenAdmission(const FamilyCapabilityParams &capability,
                        const CanonicalActorSchemaProjection &actor);
llvm::Error admitMuxTokenAdmission(const FamilyCapabilityParams &capability,
                                   const CanonicalActorSchemaProjection &actor);
llvm::Error
admitDemuxTokenAdmission(const FamilyCapabilityParams &capability,
                         const CanonicalActorSchemaProjection &actor);

bool isValidStreamStepKind(::dataflow::StreamStepKind kind) {
  switch (kind) {
  case ::dataflow::StreamStepKind::Add:
  case ::dataflow::StreamStepKind::Sub:
  case ::dataflow::StreamStepKind::Mul:
  case ::dataflow::StreamStepKind::SDiv:
  case ::dataflow::StreamStepKind::UDiv:
  case ::dataflow::StreamStepKind::ShL:
  case ::dataflow::StreamStepKind::AShr:
  case ::dataflow::StreamStepKind::LShr:
    return true;
  }
  return false;
}

llvm::Error admitStreamAdmission(const FamilyCapabilityParams &capability,
                                 const CanonicalActorSchemaProjection &actor) {
  const auto &params = std::get<fabric::LoopStreamParams>(capability);
  if (llvm::Error error = validateIntegerWidths(
          params.integerWidths, ordinaryIntegerWidths(), "stream"))
    return error;
  if (!isValidStreamStepKind(params.fixedStepKind))
    return reject("fixed stream step kind is invalid");
  if (!params.continuationPredicates.valid())
    return reject("invalid continuation predicate set");
  if (params.continuationPredicates.empty())
    return reject("non-empty continuation predicate set required");
  if (llvm::Error error = requireArity(actor, 3, 2))
    return error;
  ::mlir::Type recurrenceType = actor.type.getInput(0);
  if (actor.type.getInput(1) != recurrenceType ||
      actor.type.getInput(2) != recurrenceType ||
      actor.type.getResult(0) != recurrenceType)
    return reject("stream recurrence types do not agree");
  auto phase = ::llvm::dyn_cast<::mlir::IntegerType>(actor.type.getResult(1));
  if (!phase || !phase.isSignless() || phase.getWidth() != 1)
    return reject("stream phase result must be scalar i1");
  llvm::Expected<IntegerWidth> width =
      integerWidth(recurrenceType, "stream integer width admission");
  if (!width)
    return width.takeError();
  if (!params.integerWidths.contains(*width))
    return reject("stream integer width is not admitted");

  const auto *payload =
      std::get_if<::dataflow::StreamRecurrencePayload>(&actor.payload);
  if (!payload)
    return reject("stream has no typed recurrence projection");
  if (!isValidStreamStepKind(payload->stepKind) ||
      payload->stepKind != params.fixedStepKind)
    return reject("fixed stream step kind does not match the actor");
  if (!params.continuationPredicates.contains(payload->predicate))
    return reject("continuation predicate is not admitted");
  return llvm::Error::success();
}

llvm::Error admitTokenPlanePayload(::mlir::Type payloadType) {
  if (::llvm::isa<::mlir::NoneType>(payloadType))
    return llvm::Error::success();
  if (auto integer = ::llvm::dyn_cast<::mlir::IntegerType>(payloadType)) {
    if (integer.isSignless())
      return llvm::Error::success();
    return reject("token-plane payload integer must be signless");
  }
  if (::llvm::isa<::mlir::FloatType>(payloadType))
    return llvm::Error::success();
  if (::llvm::isa<::mlir::VectorType>(payloadType)) {
    llvm::Expected<::mlir::VectorType> vector =
        ::dataflow::semantics::analyzeFixedRankDataVector(
            payloadType, ::dataflow::semantics::VectorRank::AnyFixed);
    if (!vector)
      return reject("token-plane payload is not a fixed-ranked integer or "
                    "floating vector: " +
                    llvm::toString(vector.takeError()));
    return llvm::Error::success();
  }
  return reject("token-plane payload must be scalar integer, floating point, "
                "fixed-ranked vector, or none");
}

llvm::Error
admitTokenPlaneAdmission(const FamilyCapabilityParams &capability,
                         const CanonicalActorSchemaProjection &actor) {
  (void)std::get<fabric::TokenPlaneParams>(capability);
  ::mlir::Type payloadType;
  switch (actor.schema) {
  case OperationSchemaId::DataflowCarry:
    if (llvm::Error error = requireArity(actor, 3, 1))
      return error;
    payloadType = actor.type.getResult(0);
    if (actor.type.getInput(1) != payloadType ||
        actor.type.getInput(2) != payloadType)
      return reject("carry payload types do not agree");
    break;
  case OperationSchemaId::DataflowInvariant:
    if (llvm::Error error = requireArity(actor, 2, 1))
      return error;
    payloadType = actor.type.getResult(0);
    if (actor.type.getInput(1) != payloadType)
      return reject("invariant payload types do not agree");
    break;
  case OperationSchemaId::DataflowGate:
    if (llvm::Error error = requireArity(actor, 2, 2))
      return error;
    payloadType = actor.type.getResult(1);
    if (actor.type.getInput(1) != payloadType)
      return reject("gate payload types do not agree");
    break;
  default:
    return reject("token-plane admission provider received an unsupported "
                  "schema");
  }
  auto condition =
      ::llvm::dyn_cast<::mlir::IntegerType>(actor.type.getInput(0));
  if (!condition || !condition.isSignless() || condition.getWidth() != 1)
    return reject("token-plane condition must be scalar i1");
  if (actor.schema == OperationSchemaId::DataflowGate) {
    auto result =
        ::llvm::dyn_cast<::mlir::IntegerType>(actor.type.getResult(0));
    if (!result || !result.isSignless() || result.getWidth() != 1)
      return reject("gate condition result must be scalar i1");
  }
  return admitTokenPlanePayload(payloadType);
}

llvm::Expected<::mlir::VectorType> fixedVector(::mlir::Type type,
                                               std::uint32_t maxPayloadBits,
                                               llvm::StringRef relation) {
  llvm::Expected<::mlir::VectorType> vector =
      ::dataflow::semantics::analyzeFixedRankDataVector(
          type, ::dataflow::semantics::VectorRank::AnyFixed);
  if (!vector)
    return reject(relation + " requires a fixed vector: " +
                  llvm::toString(vector.takeError()));
  llvm::Expected<std::uint64_t> width =
      ::dataflow::semantics::getFlattenedVectorBitWidth(*vector);
  if (!width)
    return reject(relation + " has no finite payload width: " +
                  llvm::toString(width.takeError()));
  if (*width > maxPayloadBits)
    return reject(relation + " exceeds payload capacity");
  return *vector;
}

llvm::Error admitVectorIntegerElement(::mlir::VectorType vector,
                                      IntegerWidthSet widths,
                                      llvm::StringRef relation) {
  llvm::Expected<IntegerWidth> width =
      integerWidth(vector.getElementType(), relation);
  if (!width)
    return width.takeError();
  if (!widths.contains(*width))
    return reject(relation + " element width is not admitted");
  return llvm::Error::success();
}

llvm::Error admitVectorFloatElement(::mlir::VectorType vector,
                                    FloatFormatSet formats,
                                    llvm::StringRef relation) {
  llvm::Expected<FloatFormat> format =
      floatFormat(vector.getElementType(), relation);
  if (!format)
    return format.takeError();
  if (!formats.contains(*format))
    return reject(relation + " element type is not admitted");
  return llvm::Error::success();
}

llvm::Error validateVectorCapacity(std::uint32_t capacity) {
  if (capacity == 0)
    return reject("fixed-vector payload capacity must be positive");
  return llvm::Error::success();
}

llvm::Error admitFixedVectorOrdinaryIntegerAdmission(
    const FamilyCapabilityParams &capability,
    const CanonicalActorSchemaProjection &actor) {
  const auto &params = std::get<fabric::FixedVectorIntegerParams>(capability);
  if (llvm::Error error = validateIntegerWidths(
          params.elementWidths, ordinaryIntegerWidths(), "fixed vector"))
    return error;
  if (llvm::Error error = validateVectorCapacity(params.maxPayloadBits))
    return error;
  if (llvm::Error error = requireUniformType(actor, 2))
    return error;
  auto vector = fixedVector(actor.type.getInput(0), params.maxPayloadBits,
                            "fixed-vector integer admission");
  if (!vector)
    return vector.takeError();
  return admitVectorIntegerElement(*vector, params.elementWidths,
                                   "fixed-vector integer admission");
}

llvm::Error admitFixedVectorUnaryIntegerAdmission(
    const FamilyCapabilityParams &capability,
    const CanonicalActorSchemaProjection &actor) {
  const auto &params = std::get<fabric::FixedVectorIntegerParams>(capability);
  if (llvm::Error error = validateIntegerWidths(
          params.elementWidths, ordinaryIntegerWidths(), "fixed vector"))
    return error;
  if (llvm::Error error = validateVectorCapacity(params.maxPayloadBits))
    return error;
  if (llvm::Error error = requireUniformType(actor, 1))
    return error;
  auto vector = fixedVector(actor.type.getInput(0), params.maxPayloadBits,
                            "fixed-vector unary integer admission");
  if (!vector)
    return vector.takeError();
  return admitVectorIntegerElement(*vector, params.elementWidths,
                                   "fixed-vector unary integer admission");
}

llvm::Error admitFixedVectorLogicIntegerAdmission(
    const FamilyCapabilityParams &capability,
    const CanonicalActorSchemaProjection &actor) {
  const auto &params = std::get<fabric::FixedVectorIntegerParams>(capability);
  if (llvm::Error error = validateIntegerWidths(
          params.elementWidths, logicIntegerWidths(), "fixed vector logic"))
    return error;
  switch (actor.schema) {
  case OperationSchemaId::ArithAndI:
  case OperationSchemaId::ArithOrI:
  case OperationSchemaId::ArithXOrI:
  case OperationSchemaId::LLVMOrDisjoint:
    break;
  default:
    return reject("fixed-vector logic provider received an unsupported schema");
  }
  if (llvm::Error error = requireUniformType(actor, 2))
    return error;
  auto vector = fixedVector(actor.type.getInput(0), params.maxPayloadBits,
                            "fixed-vector logic admission");
  if (!vector)
    return vector.takeError();
  return admitVectorIntegerElement(*vector, params.elementWidths,
                                   "fixed-vector logic admission");
}

llvm::Error admitFixedVectorIntegerCompareAdmission(
    const FamilyCapabilityParams &capability,
    const CanonicalActorSchemaProjection &actor) {
  const auto &params =
      std::get<fabric::FixedVectorIntegerCompareMinMaxParams>(capability);
  if (llvm::Error error =
          validateIntegerWidths(params.elementWidths, ordinaryIntegerWidths(),
                                "fixed vector comparison"))
    return error;
  if (!params.predicates.valid() || params.predicates.empty())
    return reject("fixed-vector integer predicates must be non-empty");

  ::mlir::Type operandType;
  if (actor.schema == OperationSchemaId::ArithCmpI) {
    if (llvm::Error error = requireArity(actor, 2, 1))
      return error;
    if (actor.type.getInput(0) != actor.type.getInput(1))
      return reject("fixed-vector comparison operand types do not agree");
    auto operands = fixedVector(actor.type.getInput(0), params.maxPayloadBits,
                                "fixed-vector integer comparison");
    if (!operands)
      return operands.takeError();
    auto result = fixedVector(actor.type.getResult(0), params.maxPayloadBits,
                              "fixed-vector integer comparison result");
    if (!result)
      return result.takeError();
    if (result->getShape() != operands->getShape() ||
        !result->getElementType().isInteger(1))
      return reject(
          "fixed-vector comparison result must be matching vector<i1>");
    operandType = actor.type.getInput(0);
  } else {
    if (llvm::Error error = requireUniformType(actor, 2))
      return error;
    operandType = actor.type.getInput(0);
  }
  auto vector = fixedVector(operandType, params.maxPayloadBits,
                            "fixed-vector integer comparison");
  if (!vector)
    return vector.takeError();
  if (llvm::Error error = admitVectorIntegerElement(
          *vector, params.elementWidths, "fixed-vector integer comparison"))
    return error;
  llvm::Expected<::mlir::arith::CmpIPredicate> predicate =
      integerPredicate(actor);
  if (!predicate)
    return predicate.takeError();
  if (!params.predicates.contains(*predicate))
    return reject("fixed-vector integer predicate is not admitted");
  return llvm::Error::success();
}

llvm::Error admitFixedVectorValueSelectAdmission(
    const FamilyCapabilityParams &capability,
    const CanonicalActorSchemaProjection &actor) {
  const auto &params =
      std::get<fabric::FixedVectorValueSelectParams>(capability);
  if (llvm::Error error = requireArity(actor, 3, 1))
    return error;
  ::mlir::Type valueType = actor.type.getInput(1);
  if (actor.type.getInput(2) != valueType ||
      actor.type.getResult(0) != valueType)
    return reject("fixed-vector select value types do not agree");
  auto values = fixedVector(valueType, params.maxPayloadBits,
                            "fixed-vector value select");
  if (!values)
    return values.takeError();
  auto condition = fixedVector(actor.type.getInput(0), params.maxPayloadBits,
                               "fixed-vector select condition");
  if (!condition)
    return condition.takeError();
  if (condition->getShape() != values->getShape() ||
      !condition->getElementType().isInteger(1))
    return reject("fixed-vector select condition must be matching vector<i1>");
  if (::llvm::isa<::mlir::IntegerType>(values->getElementType()))
    return admitVectorIntegerElement(*values, params.integerElementWidths,
                                     "fixed-vector value select");
  return admitVectorFloatElement(*values, params.floatElementFormats,
                                 "fixed-vector value select");
}

unsigned fixedVectorFloatInputCount(OperationSchemaId schema) {
  switch (schema) {
  case OperationSchemaId::ArithNegF:
  case OperationSchemaId::MathAbsF:
    return 1;
  case OperationSchemaId::ArithAddF:
  case OperationSchemaId::ArithSubF:
  case OperationSchemaId::ArithMulF:
    return 2;
  case OperationSchemaId::MathFma:
    return 3;
  default:
    return 0;
  }
}

llvm::Error admitFixedVectorUniformFloatAdmission(
    const FamilyCapabilityParams &capability,
    const CanonicalActorSchemaProjection &actor) {
  const auto &params = std::get<fabric::FixedVectorFloatParams>(capability);
  if (llvm::Error error =
          validateFloatFormats(params.elementFormats, "fixed vector"))
    return error;
  unsigned inputCount = fixedVectorFloatInputCount(actor.schema);
  if (inputCount == 0)
    return reject(
        "fixed-vector floating provider received an unsupported schema");
  if (llvm::Error error = requireUniformType(actor, inputCount))
    return error;
  auto vector = fixedVector(actor.type.getInput(0), params.maxPayloadBits,
                            "fixed-vector floating admission");
  if (!vector)
    return vector.takeError();
  if (llvm::Error error = admitVectorFloatElement(
          *vector, params.elementFormats, "fixed-vector floating admission"))
    return error;
  llvm::Expected<::mlir::arith::FastMathFlags> flags = floatingFlags(actor);
  if (!flags)
    return flags.takeError();
  const bool rounded = actor.schema == OperationSchemaId::ArithAddF ||
                       actor.schema == OperationSchemaId::ArithSubF ||
                       actor.schema == OperationSchemaId::ArithMulF ||
                       actor.schema == OperationSchemaId::MathFma;
  return admitFloatBehavior(params.behavior, *flags,
                            rounded ? arithmeticRounding(actor) : std::nullopt,
                            FloatNaNBehavior::IEEE);
}

llvm::Error admitFixedVectorFloatCompareAdmission(
    const FamilyCapabilityParams &capability,
    const CanonicalActorSchemaProjection &actor) {
  const auto &params =
      std::get<fabric::FixedVectorFloatCompareMinMaxParams>(capability);
  if (llvm::Error error = validateFloatFormats(params.elementFormats,
                                               "fixed vector comparison"))
    return error;
  if (!params.predicates.valid() || params.predicates.empty())
    return reject("fixed-vector floating predicates must be non-empty");
  ::mlir::Type operandType;
  if (actor.schema == OperationSchemaId::ArithCmpF) {
    if (llvm::Error error = requireArity(actor, 2, 1))
      return error;
    if (actor.type.getInput(0) != actor.type.getInput(1))
      return reject("fixed-vector floating operands do not agree");
    auto operands = fixedVector(actor.type.getInput(0), params.maxPayloadBits,
                                "fixed-vector floating comparison");
    if (!operands)
      return operands.takeError();
    auto result = fixedVector(actor.type.getResult(0), params.maxPayloadBits,
                              "fixed-vector floating comparison result");
    if (!result)
      return result.takeError();
    if (result->getShape() != operands->getShape() ||
        !result->getElementType().isInteger(1))
      return reject("fixed-vector floating result must be matching vector<i1>");
    operandType = actor.type.getInput(0);
  } else {
    if (llvm::Error error = requireUniformType(actor, 2))
      return error;
    operandType = actor.type.getInput(0);
  }
  auto vector = fixedVector(operandType, params.maxPayloadBits,
                            "fixed-vector floating comparison");
  if (!vector)
    return vector.takeError();
  if (llvm::Error error = admitVectorFloatElement(
          *vector, params.elementFormats, "fixed-vector floating comparison"))
    return error;
  llvm::Expected<::mlir::arith::CmpFPredicate> predicate =
      floatPredicate(actor);
  if (!predicate)
    return predicate.takeError();
  if (!params.predicates.contains(*predicate))
    return reject("fixed-vector floating predicate is not admitted");
  llvm::Expected<::mlir::arith::FastMathFlags> flags = floatingFlags(actor);
  if (!flags)
    return flags.takeError();
  FloatNaNBehavior nanBehavior =
      actor.schema == OperationSchemaId::ArithMinNumF ||
              actor.schema == OperationSchemaId::ArithMaxNumF
          ? FloatNaNBehavior::NumberPreferred
          : FloatNaNBehavior::IEEE;
  return admitFloatBehavior(params.behavior, *flags, std::nullopt, nanBehavior);
}

llvm::Error admitAdapterVector(::mlir::Type type,
                               const fabric::FixedVectorAdapterParams &params,
                               llvm::StringRef relation,
                               bool requireRankOne = false) {
  auto vector = fixedVector(type, params.maxPayloadBits, relation);
  if (!vector)
    return vector.takeError();
  if (requireRankOne && vector->getRank() != 1)
    return reject(relation + " requires a rank-one vector");
  if (::llvm::isa<::mlir::IntegerType>(vector->getElementType()))
    return admitVectorIntegerElement(*vector, params.integerElementWidths,
                                     relation);
  return admitVectorFloatElement(*vector, params.floatElementFormats, relation);
}

llvm::Error
admitFixedVectorAdapterAdmission(const FamilyCapabilityParams &capability,
                                 const CanonicalActorSchemaProjection &actor) {
  const auto &params = std::get<fabric::FixedVectorAdapterParams>(capability);
  switch (actor.schema) {
  case OperationSchemaId::DataflowPack:
  case OperationSchemaId::DataflowUnpack: {
    if (llvm::Error error = requireArity(actor, 1, 1))
      return error;
    ::mlir::Type vectorType = actor.schema == OperationSchemaId::DataflowPack
                                  ? actor.type.getInput(0)
                                  : actor.type.getResult(0);
    ::mlir::Type packedType = actor.schema == OperationSchemaId::DataflowPack
                                  ? actor.type.getResult(0)
                                  : actor.type.getInput(0);
    if (llvm::Error error =
            admitAdapterVector(vectorType, params, "fixed-vector adapter"))
      return error;
    auto packed = ::llvm::dyn_cast<::mlir::IntegerType>(packedType);
    auto vector = ::llvm::cast<::mlir::VectorType>(vectorType);
    auto width = ::dataflow::semantics::getFlattenedVectorBitWidth(vector);
    if (!packed || !packed.isSignless() || !width ||
        packed.getWidth() != *width)
      return reject("fixed-vector adapter packed width does not match");
    return llvm::Error::success();
  }
  case OperationSchemaId::DataflowParallelize: {
    if (llvm::Error error = requireArity(actor, 2, 3))
      return error;
    if (llvm::Error error = admitAdapterVector(actor.type.getResult(0), params,
                                               "parallelize", true))
      return error;
    auto vector = ::llvm::cast<::mlir::VectorType>(actor.type.getResult(0));
    auto mask = ::llvm::dyn_cast<::mlir::VectorType>(actor.type.getResult(1));
    if (actor.type.getInput(0) != vector.getElementType() || !mask ||
        mask.getShape() != vector.getShape() ||
        !mask.getElementType().isInteger(1) ||
        !actor.type.getInput(1).isInteger(1) ||
        !actor.type.getResult(2).isInteger(1))
      return reject("parallelize type relation is malformed");
    return llvm::Error::success();
  }
  case OperationSchemaId::DataflowSerialize: {
    if (llvm::Error error = requireArity(actor, 3, 2))
      return error;
    if (llvm::Error error = admitAdapterVector(actor.type.getInput(0), params,
                                               "serialize", true))
      return error;
    auto vector = ::llvm::cast<::mlir::VectorType>(actor.type.getInput(0));
    auto mask = ::llvm::dyn_cast<::mlir::VectorType>(actor.type.getInput(1));
    if (!mask || mask.getShape() != vector.getShape() ||
        !mask.getElementType().isInteger(1) ||
        !actor.type.getInput(2).isInteger(1) ||
        actor.type.getResult(0) != vector.getElementType() ||
        !actor.type.getResult(1).isInteger(1))
      return reject("serialize type relation is malformed");
    return llvm::Error::success();
  }
  default:
    return reject("fixed-vector adapter received an unsupported schema");
  }
}

llvm::Expected<std::uint64_t> payloadBitWidth(::mlir::Type type) {
  if (::llvm::isa<::mlir::NoneType>(type))
    return 0;
  if (auto integer = ::llvm::dyn_cast<::mlir::IntegerType>(type)) {
    if (!integer.isSignless())
      return reject("payload integer must be signless");
    return integer.getWidth();
  }
  if (auto floating = ::llvm::dyn_cast<::mlir::FloatType>(type))
    return floating.getWidth();
  if (auto vector = ::llvm::dyn_cast<::mlir::VectorType>(type))
    return ::dataflow::semantics::getFlattenedVectorBitWidth(vector);
  return reject("payload must be scalar, fixed vector, or none");
}

llvm::Error admitPayload(::mlir::Type type, std::uint32_t capacity) {
  if (llvm::Error error = admitTokenPlanePayload(type))
    return error;
  auto width = payloadBitWidth(type);
  if (!width)
    return width.takeError();
  if (*width > capacity)
    return reject("token payload exceeds payload capacity");
  return llvm::Error::success();
}

llvm::Error
admitConstantTokenAdmission(const FamilyCapabilityParams &capability,
                            const CanonicalActorSchemaProjection &actor) {
  const auto &params = std::get<fabric::PayloadCapacityParams>(capability);
  if (llvm::Error error = requireArity(actor, 1, 1))
    return error;
  if (!::llvm::isa<::mlir::NoneType>(actor.type.getInput(0)))
    return reject("constant control input must be none");
  return admitPayload(actor.type.getResult(0), params.maxPayloadBits);
}

llvm::Error
admitSyncTokenAdmission(const FamilyCapabilityParams &capability,
                        const CanonicalActorSchemaProjection &actor) {
  const auto &params = std::get<fabric::RoutedTokenParams>(capability);
  const unsigned lanes = actor.type.getNumInputs();
  if (lanes == 0 || lanes > params.maxFan ||
      actor.type.getNumResults() != lanes)
    return reject("sync lane count exceeds routed-token fan capacity");
  for (unsigned lane = 0; lane < lanes; ++lane) {
    if (actor.type.getInput(lane) != actor.type.getResult(lane))
      return reject("sync lane types do not agree");
    if (llvm::Error error =
            admitPayload(actor.type.getInput(lane), params.maxPayloadBits))
      return error;
  }
  return llvm::Error::success();
}

llvm::Error validateSelector(::mlir::Type selector, unsigned fan) {
  if (fan == 2)
    return selector.isInteger(1)
               ? llvm::Error::success()
               : reject("two-way token route requires an i1 selector");
  return ::llvm::isa<::mlir::IndexType>(selector)
             ? llvm::Error::success()
             : reject("multi-way token route requires an index selector");
}

llvm::Error
admitMuxTokenAdmission(const FamilyCapabilityParams &capability,
                       const CanonicalActorSchemaProjection &actor) {
  const auto &params = std::get<fabric::RoutedTokenParams>(capability);
  if (actor.type.getNumInputs() < 3 || actor.type.getNumResults() != 1)
    return reject("token mux arity is malformed");
  const unsigned fan = actor.type.getNumInputs() - 1;
  if (fan > params.maxFan)
    return reject("token mux exceeds routed-token fan capacity");
  if (llvm::Error error = validateSelector(actor.type.getInput(0), fan))
    return error;
  ::mlir::Type payload = actor.type.getResult(0);
  for (unsigned lane = 1; lane < actor.type.getNumInputs(); ++lane)
    if (actor.type.getInput(lane) != payload)
      return reject("token mux payload types do not agree");
  return admitPayload(payload, params.maxPayloadBits);
}

llvm::Error
admitDemuxTokenAdmission(const FamilyCapabilityParams &capability,
                         const CanonicalActorSchemaProjection &actor) {
  const auto &params = std::get<fabric::RoutedTokenParams>(capability);
  const unsigned fan = actor.type.getNumResults();
  if (actor.type.getNumInputs() != 2 || fan < 2 || fan > params.maxFan)
    return reject("token demux exceeds routed-token fan capacity");
  if (llvm::Error error = validateSelector(actor.type.getInput(0), fan))
    return error;
  ::mlir::Type payload = actor.type.getInput(1);
  for (unsigned lane = 0; lane < fan; ++lane)
    if (actor.type.getResult(lane) != payload)
      return reject("token demux payload types do not agree");
  return admitPayload(payload, params.maxPayloadBits);
}

} // namespace

std::optional<fabric::ResolvedIndexWidth>
fabric::symbolizeResolvedIndexWidth(unsigned bitWidth) {
  switch (bitWidth) {
  case 32:
    return ResolvedIndexWidth::I32;
  case 64:
    return ResolvedIndexWidth::I64;
  default:
    return std::nullopt;
  }
}

unsigned fabric::getResolvedIndexBitWidth(ResolvedIndexWidth width) {
  switch (width) {
  case ResolvedIndexWidth::I32:
    return 32;
  case ResolvedIndexWidth::I64:
    return 64;
  }
  llvm_unreachable("unknown resolved index width");
}

fabric::CapabilityParamsSchemaId
fabric::capabilityParamsSchema(const FamilyCapabilityParams &params) {
  return std::visit(
      [](const auto &typedParams) {
        using Params = std::decay_t<decltype(typedParams)>;
        return Params::schemaId;
      },
      params);
}

llvm::Error fabric::verifyImplementationFamilyAdmission(
    ImplementationFamilyId family, const FamilyCapabilityParams *params,
    const ::dataflow::CanonicalActorSchemaProjection &actor) {
  std::uint32_t familyIndex = static_cast<std::uint32_t>(family);
  if (familyIndex >= implementationFamilyCount())
    return reject("implementation family is not registered");

  const ImplementationFamilyDescriptor &descriptor =
      implementationFamily(family);
  if (!llvm::is_contained(descriptor.admittedSchemas, actor.schema))
    return reject("actor schema is not admitted by the implementation family");
  if (!params)
    return reject("capability parameters are absent");
  if (capabilityParamsSchema(*params) != descriptor.capabilityParamsSchema)
    return reject("capability parameter schema does not match the generated "
                  "family descriptor");

  switch (descriptor.typedAdmissionProvider) {
#define LOOM_TYPED_ADMISSION_PROVIDER(Name, Id)                                \
  case TypedAdmissionProviderId::Name:                                         \
    return admit##Name(*params, actor);
#include "Fabric/IR/ImplementationFamilies.inc"
  }
  return reject("typed admission provider is not registered");
}

llvm::Expected<::dataflow::CanonicalActorSchemaProjection>
fabric::projectResolvedIndexTypes(
    const ::dataflow::CanonicalActorSchemaProjection &actor,
    unsigned indexBitWidth) {
  if (!symbolizeResolvedIndexWidth(indexBitWidth))
    return reject("resolved index width must be 32 or 64");
  const auto representType = [&](::mlir::Type type) -> ::mlir::Type {
    if (::llvm::isa<::mlir::IndexType>(type))
      return ::mlir::IntegerType::get(type.getContext(), indexBitWidth);
    if (auto vector = ::llvm::dyn_cast<::mlir::VectorType>(type);
        vector && ::llvm::isa<::mlir::IndexType>(vector.getElementType()))
      return ::mlir::VectorType::get(
          vector.getShape(),
          ::mlir::IntegerType::get(type.getContext(), indexBitWidth),
          vector.getScalableDims());
    return type;
  };
  llvm::SmallVector<::mlir::Type, 4> inputs;
  llvm::SmallVector<::mlir::Type, 4> results;
  llvm::transform(actor.type.getInputs(), std::back_inserter(inputs),
                  representType);
  llvm::transform(actor.type.getResults(), std::back_inserter(results),
                  representType);
  ::dataflow::CanonicalActorSchemaProjection represented = actor;
  represented.type =
      ::mlir::FunctionType::get(actor.type.getContext(), inputs, results);
  return represented;
}

llvm::Error fabric::verifyImplementationFamilyAdmission(
    ImplementationFamilyId family, const FamilyCapabilityParams *params,
    const ::dataflow::CanonicalActorSchemaProjection &actor,
    unsigned indexBitWidth) {
  const std::optional<ResolvedIndexWidth> resolved =
      symbolizeResolvedIndexWidth(indexBitWidth);
  if (!resolved)
    return reject("resolved index width must be 32 or 64");
  const bool isIndexCast =
      actor.schema == ::dataflow::OperationSchemaId::ArithIndexCast ||
      actor.schema == ::dataflow::OperationSchemaId::ArithIndexCastUI;
  if (isIndexCast) {
    if (llvm::Error error =
            verifyImplementationFamilyAdmission(family, params, actor))
      return error;
    const auto *cast =
        params ? std::get_if<ScalarIntegerCastParams>(params) : nullptr;
    if (!cast || !cast->relation.resolvedIndexWidths.contains(*resolved))
      return reject("concrete cast relation does not admit the resolved index "
                    "width");
    const bool sourceIsIndex =
        ::llvm::isa<::mlir::IndexType>(actor.type.getInput(0));
    ::mlir::Type integerType =
        sourceIsIndex ? actor.type.getResult(0) : actor.type.getInput(0);
    llvm::Expected<IntegerWidth> integer =
        integerWidth(integerType, "integer cast relation");
    if (!integer)
      return integer.takeError();
    const IntegerWidth index = *resolved == ResolvedIndexWidth::I32
                                   ? IntegerWidth::I32
                                   : IntegerWidth::I64;
    const IntegerWidth source = sourceIsIndex ? index : *integer;
    const IntegerWidth destination = sourceIsIndex ? *integer : index;
    if (!cast->relation.widthPairs.contains(source, destination))
      return reject("integer cast relation does not admit the exact resolved "
                    "index endpoint pair");
    return llvm::Error::success();
  }
  auto represented = projectResolvedIndexTypes(actor, indexBitWidth);
  if (!represented)
    return represented.takeError();
  return verifyImplementationFamilyAdmission(family, params, *represented);
}

llvm::Expected<bool> fabric::requiresSemanticConfigurationField(
    ImplementationFamilyId family, const FamilyCapabilityParams &params,
    llvm::ArrayRef<::dataflow::OperationSchemaId> enabledSchemas,
    std::uint32_t physicalInputCount, std::uint32_t physicalResultCount) {
  const std::uint32_t familyIndex = static_cast<std::uint32_t>(family);
  if (familyIndex >= implementationFamilyCount())
    return reject("implementation family is not registered");
  const ImplementationFamilyDescriptor &descriptor =
      implementationFamily(family);
  if (capabilityParamsSchema(params) != descriptor.capabilityParamsSchema)
    return reject("capability parameter schema does not match the generated "
                  "family descriptor");
  if (enabledSchemas.empty())
    return reject("concrete operation capability has no enabled schema");
  for (OperationSchemaId schema : enabledSchemas)
    if (!llvm::is_contained(descriptor.admittedSchemas, schema))
      return reject("concrete operation capability escapes its generated "
                    "implementation family");

  // Selecting among enabled members always changes the configured function.
  // All remaining cases below identify a parameterized singleton relation
  // whose exact actor point changes real hardware behavior. Width-only
  // admission for a bitwise or modular datapath deliberately creates no field.
  if (enabledSchemas.size() > 1)
    return true;

  const auto floatBehaviorVaries = [](const FloatBehaviorProfile &behavior) {
    return behavior.roundingModes.size() > 1 ||
           behavior.nanBehaviors.size() > 1 ||
           behavior.subnormalBehaviors.size() > 1 ||
           behavior.signedZeroBehaviors.size() > 1;
  };
  const auto hasSchema = [&](OperationSchemaId schema) {
    return llvm::is_contained(enabledSchemas, schema);
  };
  const auto hasSignedIntegerBehavior = [&] {
    if (hasSchema(OperationSchemaId::ArithShRSI) ||
        hasSchema(OperationSchemaId::ArithDivSI) ||
        hasSchema(OperationSchemaId::ArithRemSI) ||
        hasSchema(OperationSchemaId::ArithMinSI) ||
        hasSchema(OperationSchemaId::ArithMaxSI))
      return true;
    if (!hasSchema(OperationSchemaId::ArithCmpI))
      return false;
    const auto &compare = std::get<ScalarIntegerCompareMinMaxParams>(params);
    using Predicate = ::mlir::arith::CmpIPredicate;
    return compare.predicates.contains(Predicate::slt) ||
           compare.predicates.contains(Predicate::sle) ||
           compare.predicates.contains(Predicate::sgt) ||
           compare.predicates.contains(Predicate::sge);
  };

  switch (descriptor.typedAdmissionProvider) {
  case TypedAdmissionProviderId::ScalarOrdinaryIntegerAdmission:
    return std::get<ScalarIntegerParams>(params).integerWidths.size() > 1 &&
           hasSignedIntegerBehavior();
  case TypedAdmissionProviderId::ScalarUnaryIntegerAdmission:
    return false;
  case TypedAdmissionProviderId::ScalarLogicIntegerAdmission:
  case TypedAdmissionProviderId::ScalarBitReinterpretAdmission:
  case TypedAdmissionProviderId::ScalarValueSelectAdmission:
  case TypedAdmissionProviderId::TokenPlaneAdmission:
  case TypedAdmissionProviderId::FixedVectorLogicIntegerAdmission:
    return false;
  case TypedAdmissionProviderId::ScalarIntegerCompareAdmission: {
    const auto &typed = std::get<ScalarIntegerCompareMinMaxParams>(params);
    return typed.predicates.size() > 1 ||
           (typed.operandWidths.size() > 1 && hasSignedIntegerBehavior());
  }
  case TypedAdmissionProviderId::ScalarIntegerCastAdmission:
    return std::get<ScalarIntegerCastParams>(params)
               .relation.widthPairs.size() > 1;
  case TypedAdmissionProviderId::ScalarUniformFloatAdmission: {
    const auto &typed = std::get<ScalarFloatParams>(params);
    return typed.formats.size() > 1 || floatBehaviorVaries(typed.behavior);
  }
  case TypedAdmissionProviderId::ScalarFloatCompareAdmission: {
    const auto &typed = std::get<ScalarFloatCompareMinMaxParams>(params);
    return typed.formats.size() > 1 || typed.predicates.size() > 1 ||
           floatBehaviorVaries(typed.behavior);
  }
  case TypedAdmissionProviderId::ScalarFloatCastAdmission: {
    const auto &typed = std::get<ScalarFloatWidthCastParams>(params);
    return typed.formatPairs.size() > 1 || floatBehaviorVaries(typed.behavior);
  }
  case TypedAdmissionProviderId::ScalarIntegerFloatConversionAdmission: {
    const auto &typed = std::get<ScalarIntegerFloatConversionParams>(params);
    return typed.formatPairs.size() > 1 || floatBehaviorVaries(typed.behavior);
  }
  case TypedAdmissionProviderId::StreamAdmission: {
    const auto &typed = std::get<LoopStreamParams>(params);
    return typed.integerWidths.size() > 1 ||
           typed.continuationPredicates.size() > 1;
  }
  case TypedAdmissionProviderId::FixedVectorOrdinaryIntegerAdmission:
    return std::get<FixedVectorIntegerParams>(params).elementWidths.size() > 1;
  case TypedAdmissionProviderId::FixedVectorUnaryIntegerAdmission:
    return false;
  case TypedAdmissionProviderId::FixedVectorIntegerCompareAdmission: {
    const auto &typed = std::get<FixedVectorIntegerCompareMinMaxParams>(params);
    return typed.elementWidths.size() > 1 || typed.predicates.size() > 1;
  }
  case TypedAdmissionProviderId::FixedVectorValueSelectAdmission: {
    const auto &typed = std::get<FixedVectorValueSelectParams>(params);
    llvm::SmallVector<unsigned, 8> elementWidths;
    const auto addWidth = [&](unsigned width) {
      if (!llvm::is_contained(elementWidths, width))
        elementWidths.push_back(width);
    };
    constexpr IntegerWidth integerWidths[] = {
        IntegerWidth::I1, IntegerWidth::I8, IntegerWidth::I16,
        IntegerWidth::I32, IntegerWidth::I64};
    for (IntegerWidth width : integerWidths)
      if (typed.integerElementWidths.contains(width))
        addWidth(integerBitWidth(width));
    constexpr FloatFormat floatFormats[] = {FloatFormat::F16, FloatFormat::BF16,
                                            FloatFormat::F32, FloatFormat::F64};
    for (FloatFormat format : floatFormats)
      if (typed.floatElementFormats.contains(format))
        addWidth(floatBitWidth(format));
    return elementWidths.size() > 1;
  }
  case TypedAdmissionProviderId::FixedVectorUniformFloatAdmission: {
    const auto &typed = std::get<FixedVectorFloatParams>(params);
    return typed.elementFormats.size() > 1 ||
           floatBehaviorVaries(typed.behavior);
  }
  case TypedAdmissionProviderId::FixedVectorFloatCompareAdmission: {
    const auto &typed = std::get<FixedVectorFloatCompareMinMaxParams>(params);
    return typed.elementFormats.size() > 1 || typed.predicates.size() > 1 ||
           floatBehaviorVaries(typed.behavior);
  }
  case TypedAdmissionProviderId::FixedVectorAdapterAdmission: {
    if (!hasSchema(OperationSchemaId::DataflowParallelize) &&
        !hasSchema(OperationSchemaId::DataflowSerialize))
      return false;
    const auto &typed = std::get<FixedVectorAdapterParams>(params);
    return typed.integerElementWidths.size() +
               typed.floatElementFormats.size() >
           1;
  }
  case TypedAdmissionProviderId::ConstantTokenAdmission:
    return true;
  case TypedAdmissionProviderId::SyncTokenAdmission:
    return physicalInputCount > 1 || physicalResultCount > 1;
  case TypedAdmissionProviderId::MuxTokenAdmission:
    return physicalInputCount > 3;
  case TypedAdmissionProviderId::DemuxTokenAdmission:
    return physicalResultCount > 2;
  }
  return reject("typed admission provider is not registered");
}

mlir::FailureOr<unsigned> fabric::getSemanticPayloadWidth(mlir::Type type,
                                                          std::string &error) {
  if (auto integer = mlir::dyn_cast<mlir::IntegerType>(type))
    return integer.getWidth();
  if (auto floating = mlir::dyn_cast<mlir::FloatType>(type))
    return floating.getWidth();
  if (mlir::isa<mlir::IndexType, mlir::LLVM::LLVMPointerType>(type))
    return ::loom::getIndexWidth();
  if (mlir::isa<mlir::NoneType>(type))
    return 0u;
  if (auto vector = mlir::dyn_cast<mlir::VectorType>(type)) {
    auto elementWidth = getSemanticPayloadWidth(vector.getElementType(), error);
    if (mlir::failed(elementWidth))
      return mlir::failure();
    auto width = ::loom::getFixedVectorBitWidth(vector, *elementWidth);
    if (!width) {
      error = llvm::toString(width.takeError());
      return mlir::failure();
    }
    if (static_cast<unsigned>(*width) != *width) {
      error = "semantic payload width " + std::to_string(*width) +
              " exceeds the physical payload width";
      return mlir::failure();
    }
    return static_cast<unsigned>(*width);
  }

  std::string spelling;
  llvm::raw_string_ostream stream(spelling);
  type.print(stream);
  error = "unsupported semantic payload type " + stream.str();
  return mlir::failure();
}

namespace {

llvm::Expected<unsigned>
bitReinterpretEndpoint(::mlir::Type type,
                       const fabric::ScalarBitReinterpretParams &params) {
  if (::llvm::isa<::mlir::IntegerType>(type)) {
    llvm::Expected<IntegerWidth> width =
        integerWidth(type, "scalar bit reinterpretation");
    if (!width)
      return width.takeError();
    if (*width == IntegerWidth::I1 || !params.integerWidths.contains(*width))
      return reject("bit reinterpretation integer width is not admitted");
    return integerBitWidth(*width);
  }
  llvm::Expected<FloatFormat> format =
      floatFormat(type, "scalar bit reinterpretation");
  if (!format)
    return format.takeError();
  if (!params.floatFormats.contains(*format))
    return reject("bit reinterpretation floating format is not admitted");
  return floatBitWidth(*format);
}

llvm::Error admitScalarBitReinterpretAdmission(
    const FamilyCapabilityParams &capability,
    const CanonicalActorSchemaProjection &actor) {
  const auto &params = std::get<fabric::ScalarBitReinterpretParams>(capability);
  if (!params.integerWidths.valid())
    return reject("invalid bit reinterpretation integer width set");
  if (!params.integerWidths.isSubsetOf(ordinaryIntegerWidths()))
    return reject("bit reinterpretation integer width set contains an "
                  "unsupported width");
  if (!params.floatFormats.valid())
    return reject("invalid bit reinterpretation floating format set");
  if (params.integerWidths.empty() && params.floatFormats.empty())
    return reject("bit reinterpretation requires a non-empty endpoint domain");
  if (llvm::Error error = requireArity(actor, 1, 1))
    return error;
  llvm::Expected<unsigned> source =
      bitReinterpretEndpoint(actor.type.getInput(0), params);
  if (!source)
    return source.takeError();
  llvm::Expected<unsigned> destination =
      bitReinterpretEndpoint(actor.type.getResult(0), params);
  if (!destination)
    return destination.takeError();
  if (*source != *destination)
    return reject("bit reinterpretation requires equal semantic width");
  return llvm::Error::success();
}

llvm::Error admitUniformFloatType(const CanonicalActorSchemaProjection &actor,
                                  FloatFormatSet formats, unsigned inputCount,
                                  FloatFormat &format) {
  if (llvm::Error error = requireUniformType(actor, inputCount))
    return error;
  llvm::Expected<FloatFormat> actorFormat =
      floatFormat(actor.type.getInput(0), "floating format admission");
  if (!actorFormat)
    return actorFormat.takeError();
  if (!formats.contains(*actorFormat))
    return reject("floating format is not admitted");
  format = *actorFormat;
  return llvm::Error::success();
}

llvm::Error
admitScalarUniformFloatAdmission(const FamilyCapabilityParams &capability,
                                 const CanonicalActorSchemaProjection &actor) {
  const auto &params = std::get<fabric::ScalarFloatParams>(capability);
  if (llvm::Error error = validateFloatFormats(params.formats, "scalar"))
    return error;

  unsigned inputCount = 0;
  bool hasArithmeticRounding = false;
  switch (actor.schema) {
  case OperationSchemaId::ArithNegF:
  case OperationSchemaId::MathAbsF:
  case OperationSchemaId::MathSin:
  case OperationSchemaId::MathCos:
  case OperationSchemaId::MathTan:
  case OperationSchemaId::MathSinh:
  case OperationSchemaId::MathCosh:
  case OperationSchemaId::MathTanh:
  case OperationSchemaId::MathExp:
  case OperationSchemaId::MathExp2:
  case OperationSchemaId::MathExpM1:
  case OperationSchemaId::MathLog:
  case OperationSchemaId::MathLog2:
  case OperationSchemaId::MathLog10:
  case OperationSchemaId::MathLog1p:
  case OperationSchemaId::MathFloor:
  case OperationSchemaId::MathCeil:
  case OperationSchemaId::MathRound:
  case OperationSchemaId::MathTrunc:
  case OperationSchemaId::MathRoundEven:
  case OperationSchemaId::MathSqrt:
  case OperationSchemaId::MathRsqrt:
  case OperationSchemaId::MathErf:
    inputCount = 1;
    break;
  case OperationSchemaId::ArithAddF:
  case OperationSchemaId::ArithSubF:
  case OperationSchemaId::ArithMulF:
  case OperationSchemaId::ArithDivF:
  case OperationSchemaId::ArithRemF:
    inputCount = 2;
    hasArithmeticRounding = true;
    break;
  case OperationSchemaId::MathFma:
    inputCount = 3;
    hasArithmeticRounding = true;
    break;
  default:
    return reject("floating admission provider received an unsupported schema");
  }
  FloatFormat format;
  if (llvm::Error error =
          admitUniformFloatType(actor, params.formats, inputCount, format))
    return error;
  (void)format;
  llvm::Expected<::mlir::arith::FastMathFlags> flags = floatingFlags(actor);
  if (!flags)
    return flags.takeError();
  std::optional<::mlir::arith::RoundingMode> rounding =
      hasArithmeticRounding ? arithmeticRounding(actor) : std::nullopt;
  return admitFloatBehavior(params.behavior, *flags, rounding,
                            FloatNaNBehavior::IEEE);
}

llvm::Expected<::mlir::arith::CmpFPredicate>
floatPredicate(const CanonicalActorSchemaProjection &actor) {
  if (actor.schema == OperationSchemaId::ArithCmpF) {
    const auto *payload =
        std::get_if<::dataflow::FloatComparePayload>(&actor.payload);
    if (!payload)
      return reject("floating comparison has no typed predicate projection");
    return payload->predicate;
  }
  switch (actor.schema) {
  case OperationSchemaId::ArithMinimumF:
  case OperationSchemaId::ArithMinNumF:
    return ::mlir::arith::CmpFPredicate::OLT;
  case OperationSchemaId::ArithMaximumF:
  case OperationSchemaId::ArithMaxNumF:
    return ::mlir::arith::CmpFPredicate::OGT;
  default:
    return reject("floating compare provider received an unsupported schema");
  }
}

llvm::Error
admitScalarFloatCompareAdmission(const FamilyCapabilityParams &capability,
                                 const CanonicalActorSchemaProjection &actor) {
  const auto &params =
      std::get<fabric::ScalarFloatCompareMinMaxParams>(capability);
  if (llvm::Error error = validateFloatFormats(params.formats, "comparison"))
    return error;
  if (!params.predicates.valid())
    return reject("invalid floating predicate set");
  if (params.predicates.empty())
    return reject("non-empty floating predicate set required");

  ::mlir::Type operandType;
  if (actor.schema == OperationSchemaId::ArithCmpF) {
    if (llvm::Error error = requireArity(actor, 2, 1))
      return error;
    if (actor.type.getInput(0) != actor.type.getInput(1))
      return reject("floating comparison operand types do not agree");
    auto result =
        ::llvm::dyn_cast<::mlir::IntegerType>(actor.type.getResult(0));
    if (!result || !result.isSignless() || result.getWidth() != 1)
      return reject("floating comparison result must be scalar i1");
    operandType = actor.type.getInput(0);
  } else {
    if (llvm::Error error = requireUniformType(actor, 2))
      return error;
    operandType = actor.type.getInput(0);
  }
  llvm::Expected<FloatFormat> format =
      floatFormat(operandType, "floating comparison");
  if (!format)
    return format.takeError();
  if (!params.formats.contains(*format))
    return reject("floating format is not admitted for comparison");

  llvm::Expected<::mlir::arith::CmpFPredicate> predicate =
      floatPredicate(actor);
  if (!predicate)
    return predicate.takeError();
  if (!params.predicates.contains(*predicate))
    return reject("floating predicate is not admitted");
  llvm::Expected<::mlir::arith::FastMathFlags> flags = floatingFlags(actor);
  if (!flags)
    return flags.takeError();
  FloatNaNBehavior nanBehavior =
      actor.schema == OperationSchemaId::ArithMinNumF ||
              actor.schema == OperationSchemaId::ArithMaxNumF
          ? FloatNaNBehavior::NumberPreferred
          : FloatNaNBehavior::IEEE;
  return admitFloatBehavior(params.behavior, *flags, std::nullopt, nanBehavior);
}

llvm::Error
admitScalarFloatCastAdmission(const FamilyCapabilityParams &capability,
                              const CanonicalActorSchemaProjection &actor) {
  const auto &params = std::get<fabric::ScalarFloatWidthCastParams>(capability);
  if (!params.formatPairs.valid())
    return reject("invalid floating cast relation");
  if (params.formatPairs.empty())
    return reject("non-empty floating cast relation required");
  if (llvm::Error error = requireArity(actor, 1, 1))
    return error;
  llvm::Expected<FloatFormat> source =
      floatFormat(actor.type.getInput(0), "floating cast relation");
  if (!source)
    return source.takeError();
  llvm::Expected<FloatFormat> destination =
      floatFormat(actor.type.getResult(0), "floating cast relation");
  if (!destination)
    return destination.takeError();
  if (!params.formatPairs.contains(*source, *destination))
    return reject("floating cast relation does not admit the endpoint pair");

  bool truncates = false;
  switch (actor.schema) {
  case OperationSchemaId::ArithExtF:
    if (floatBitWidth(*source) >= floatBitWidth(*destination))
      return reject("floating extension must widen");
    break;
  case OperationSchemaId::ArithTruncF:
    if (floatBitWidth(*source) <= floatBitWidth(*destination))
      return reject("floating truncation must narrow");
    truncates = true;
    break;
  default:
    return reject("floating cast provider received an unsupported schema");
  }
  llvm::Expected<::mlir::arith::FastMathFlags> flags = floatingFlags(actor);
  if (!flags)
    return flags.takeError();
  std::optional<::mlir::arith::RoundingMode> rounding =
      truncates ? arithmeticRounding(actor) : std::nullopt;
  return admitFloatBehavior(params.behavior, *flags, rounding,
                            FloatNaNBehavior::IEEE);
}

llvm::Error admitScalarIntegerFloatConversionAdmission(
    const FamilyCapabilityParams &capability,
    const CanonicalActorSchemaProjection &actor) {
  const auto &params =
      std::get<fabric::ScalarIntegerFloatConversionParams>(capability);
  if (!params.formatPairs.valid())
    return reject("invalid integer and floating relation");
  if (params.formatPairs.empty())
    return reject("non-empty integer and floating relation required");
  if (llvm::Error error = requireArity(actor, 1, 1))
    return error;

  ::mlir::Type integerType;
  ::mlir::Type floatType;
  switch (actor.schema) {
  case OperationSchemaId::ArithSIToFP:
  case OperationSchemaId::ArithUIToFP:
    integerType = actor.type.getInput(0);
    floatType = actor.type.getResult(0);
    break;
  case OperationSchemaId::ArithFPToSI:
  case OperationSchemaId::ArithFPToUI:
    floatType = actor.type.getInput(0);
    integerType = actor.type.getResult(0);
    break;
  default:
    return reject("integer and floating conversion provider received an "
                  "unsupported schema");
  }
  llvm::Expected<IntegerWidth> integer =
      integerWidth(integerType, "integer and floating relation");
  if (!integer)
    return integer.takeError();
  if (!ordinaryIntegerWidths().contains(*integer))
    return reject("integer width is not admitted for floating conversion");
  llvm::Expected<FloatFormat> format =
      floatFormat(floatType, "integer and floating relation");
  if (!format)
    return format.takeError();
  if (!params.formatPairs.contains(*integer, *format))
    return reject("integer and floating relation does not admit the endpoint "
                  "pair");

  // Conversion schemas have no arithmetic rounding attribute in their
  // registered projection. Their exact conversion semantics are therefore
  // checked without inventing an arithmetic rounding configuration field.
  return admitFloatBehavior(params.behavior, ::mlir::arith::FastMathFlags::none,
                            std::nullopt, FloatNaNBehavior::IEEE);
}

} // namespace

namespace {

llvm::Error reject(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(), message);
}

IntegerWidthSet ordinaryIntegerWidths() {
  return IntegerWidthSet::get({IntegerWidth::I8, IntegerWidth::I16,
                               IntegerWidth::I32, IntegerWidth::I64});
}

IntegerWidthSet logicIntegerWidths() {
  return IntegerWidthSet::get({IntegerWidth::I1, IntegerWidth::I8,
                               IntegerWidth::I16, IntegerWidth::I32,
                               IntegerWidth::I64});
}

llvm::Error validateIntegerWidths(IntegerWidthSet widths,
                                  IntegerWidthSet allowed,
                                  llvm::StringRef description) {
  if (!widths.valid())
    return reject("invalid " + description + " integer width set");
  if (widths.empty())
    return reject("non-empty integer width set required for " + description);
  if (!widths.isSubsetOf(allowed))
    return reject(description + " integer width set contains an unsupported "
                                "width");
  return llvm::Error::success();
}

llvm::Error validateFloatFormats(FloatFormatSet formats,
                                 llvm::StringRef description) {
  if (!formats.valid())
    return reject("invalid " + description + " floating format set");
  if (formats.empty())
    return reject("non-empty " + description + " floating format set required");
  return llvm::Error::success();
}

llvm::Error validateFloatBehavior(const FloatBehaviorProfile &behavior) {
  if (!behavior.roundingModes.valid())
    return reject("invalid rounding mode set");
  if (behavior.roundingModes.empty())
    return reject("non-empty rounding behavior domain required");
  if (!behavior.nanBehaviors.valid())
    return reject("invalid NaN behavior set");
  if (behavior.nanBehaviors.empty())
    return reject("non-empty NaN behavior domain required");
  if (!behavior.subnormalBehaviors.valid())
    return reject("invalid subnormal behavior set");
  if (behavior.subnormalBehaviors.empty())
    return reject("non-empty subnormal behavior domain required");
  if (!behavior.signedZeroBehaviors.valid())
    return reject("invalid signed-zero behavior set");
  if (behavior.signedZeroBehaviors.empty())
    return reject("non-empty signed-zero behavior domain required");

  using FastMathBits = std::underlying_type_t<::mlir::arith::FastMathFlags>;
  FastMathBits admitted = static_cast<FastMathBits>(behavior.admittedFastMath);
  FastMathBits known =
      static_cast<FastMathBits>(::mlir::arith::FastMathFlags::fast);
  if ((admitted & ~known) != 0)
    return reject("invalid fast-math behavior mask");
  return llvm::Error::success();
}

llvm::Expected<IntegerWidth> integerWidth(::mlir::Type type,
                                          llvm::StringRef relation) {
  if (::llvm::isa<::mlir::VectorType>(type))
    return reject("scalar actor required by " + relation);
  auto integer = ::llvm::dyn_cast<::mlir::IntegerType>(type);
  if (!integer || !integer.isSignless())
    return reject(relation + " requires a scalar signless integer");
  switch (integer.getWidth()) {
  case 1:
    return IntegerWidth::I1;
  case 8:
    return IntegerWidth::I8;
  case 16:
    return IntegerWidth::I16;
  case 32:
    return IntegerWidth::I32;
  case 64:
    return IntegerWidth::I64;
  default:
    return reject(relation + " rejects the actor integer width");
  }
}

unsigned integerBitWidth(IntegerWidth width) {
  switch (width) {
  case IntegerWidth::I1:
    return 1;
  case IntegerWidth::I8:
    return 8;
  case IntegerWidth::I16:
    return 16;
  case IntegerWidth::I32:
    return 32;
  case IntegerWidth::I64:
    return 64;
  }
  llvm_unreachable("invalid integer width");
}

llvm::Expected<FloatFormat> floatFormat(::mlir::Type type,
                                        llvm::StringRef relation) {
  if (::llvm::isa<::mlir::VectorType>(type))
    return reject("scalar actor required by " + relation);
  auto floating = ::llvm::dyn_cast<::mlir::FloatType>(type);
  if (!floating)
    return reject(relation + " requires a scalar floating type");
  if (floating.isF16())
    return FloatFormat::F16;
  if (floating.isBF16())
    return FloatFormat::BF16;
  if (floating.isF32())
    return FloatFormat::F32;
  if (floating.isF64())
    return FloatFormat::F64;
  return reject(relation + " rejects the actor floating format");
}

unsigned floatBitWidth(FloatFormat format) {
  switch (format) {
  case FloatFormat::F16:
  case FloatFormat::BF16:
    return 16;
  case FloatFormat::F32:
    return 32;
  case FloatFormat::F64:
    return 64;
  }
  llvm_unreachable("invalid floating format");
}

llvm::Error requireArity(const CanonicalActorSchemaProjection &actor,
                         unsigned inputs, unsigned results) {
  if (actor.type.getNumInputs() != inputs ||
      actor.type.getNumResults() != results)
    return reject("actor function type has the wrong arity");
  return llvm::Error::success();
}

llvm::Error requireUniformType(const CanonicalActorSchemaProjection &actor,
                               unsigned inputs) {
  if (llvm::Error error = requireArity(actor, inputs, 1))
    return error;
  ::mlir::Type type = actor.type.getInput(0);
  for (unsigned index = 1; index < inputs; ++index)
    if (actor.type.getInput(index) != type)
      return reject("actor function type is not uniform");
  if (actor.type.getResult(0) != type)
    return reject("actor result type differs from its operands");
  return llvm::Error::success();
}

bool hasFastMathFlag(::mlir::arith::FastMathFlags flags,
                     ::mlir::arith::FastMathFlags flag) {
  using Bits = std::underlying_type_t<::mlir::arith::FastMathFlags>;
  return (static_cast<Bits>(flags) & static_cast<Bits>(flag)) != 0;
}

llvm::Error
admitFloatBehavior(const FloatBehaviorProfile &behavior,
                   ::mlir::arith::FastMathFlags actorFlags,
                   std::optional<::mlir::arith::RoundingMode> rounding,
                   FloatNaNBehavior nanBehavior) {
  if (llvm::Error error = validateFloatBehavior(behavior))
    return error;

  using Bits = std::underlying_type_t<::mlir::arith::FastMathFlags>;
  Bits actor = static_cast<Bits>(actorFlags);
  Bits admitted = static_cast<Bits>(behavior.admittedFastMath);
  if ((actor & ~admitted) != 0)
    return reject("fast-math behavior is not admitted");
  if (rounding && !behavior.roundingModes.contains(*rounding))
    return reject("rounding behavior is not admitted");
  if (!hasFastMathFlag(actorFlags, ::mlir::arith::FastMathFlags::nnan) &&
      !behavior.nanBehaviors.contains(nanBehavior))
    return reject("NaN behavior is not admitted");
  if (!behavior.subnormalBehaviors.contains(FloatSubnormalBehavior::Preserve))
    return reject("subnormal behavior is not admitted");
  if (!hasFastMathFlag(actorFlags, ::mlir::arith::FastMathFlags::nsz) &&
      !behavior.signedZeroBehaviors.contains(FloatSignedZeroBehavior::Preserve))
    return reject("signed-zero behavior is not admitted");
  return llvm::Error::success();
}

llvm::Expected<::mlir::arith::FastMathFlags>
floatingFlags(const CanonicalActorSchemaProjection &actor) {
  if (const auto *payload =
          std::get_if<::dataflow::FloatingPointPayload>(&actor.payload))
    return payload->flags;
  if (const auto *payload =
          std::get_if<::dataflow::FloatComparePayload>(&actor.payload))
    return payload->flags;
  return reject(
      "registered floating actor has no floating behavior projection");
}

std::optional<::mlir::arith::RoundingMode>
arithmeticRounding(const CanonicalActorSchemaProjection &actor) {
  const auto *payload =
      std::get_if<::dataflow::FloatingPointPayload>(&actor.payload);
  if (!payload)
    return std::nullopt;
  return payload->roundingMode.value_or(
      ::mlir::arith::RoundingMode::to_nearest_even);
}

} // namespace

std::uint32_t fabric::implementationFamilyCount() {
  return static_cast<std::uint32_t>(kFamilyCount);
}

const fabric::ImplementationFamilyDescriptor &
fabric::implementationFamily(ImplementationFamilyId family) {
  return familyTable()[static_cast<std::size_t>(family)];
}

llvm::StringRef
fabric::implementationFamilyKeyword(ImplementationFamilyId family) {
  return stringifyImplementationFamilyId(family);
}

std::optional<fabric::ImplementationFamilyId>
fabric::findImplementationFamily(llvm::StringRef keyword) {
  return symbolizeImplementationFamilyId(keyword);
}

bool fabric::admitsOperationSchema(ImplementationFamilyId family,
                                   ::dataflow::OperationSchemaId schema) {
  return llvm::is_contained(implementationFamily(family).admittedSchemas,
                            schema);
}

llvm::SmallVector<fabric::ImplementationFamilyId, 2>
fabric::implementationFamiliesFor(::dataflow::OperationSchemaId schema) {
  llvm::SmallVector<ImplementationFamilyId, 2> families;
  for (std::uint32_t index = 0; index < implementationFamilyCount(); ++index) {
    auto family = static_cast<ImplementationFamilyId>(index);
    if (admitsOperationSchema(family, schema))
      families.push_back(family);
  }
  return families;
}

llvm::StringRef
fabric::capabilityParamsSchemaKeyword(CapabilityParamsSchemaId schema) {
  switch (schema) {
#define LOOM_CAPABILITY_PARAMS_SCHEMA(Name, Id)                                \
  case CapabilityParamsSchemaId::Name:                                         \
    return #Name;
#include "Fabric/IR/ImplementationFamilies.inc"
  }
  llvm_unreachable("unregistered capability parameter schema");
}

llvm::StringRef
fabric::typedAdmissionProviderKeyword(TypedAdmissionProviderId provider) {
  switch (provider) {
#define LOOM_TYPED_ADMISSION_PROVIDER(Name, Id)                                \
  case TypedAdmissionProviderId::Name:                                         \
    return #Name;
#include "Fabric/IR/ImplementationFamilies.inc"
  }
  llvm_unreachable("unregistered typed admission provider");
}

namespace {

llvm::Error admitUniformInteger(const CanonicalActorSchemaProjection &actor,
                                IntegerWidthSet widths, unsigned inputCount) {
  if (llvm::Error error = requireUniformType(actor, inputCount))
    return error;
  llvm::Expected<IntegerWidth> width =
      integerWidth(actor.type.getInput(0), "integer width admission");
  if (!width)
    return width.takeError();
  if (!widths.contains(*width))
    return reject("integer width is not admitted");
  return llvm::Error::success();
}

llvm::Error admitScalarOrdinaryIntegerAdmission(
    const FamilyCapabilityParams &capability,
    const CanonicalActorSchemaProjection &actor) {
  const auto &params = std::get<fabric::ScalarIntegerParams>(capability);
  if (llvm::Error error = validateIntegerWidths(
          params.integerWidths, ordinaryIntegerWidths(), "ordinary scalar"))
    return error;
  return admitUniformInteger(actor, params.integerWidths, 2);
}

llvm::Error admitScalarUnaryIntegerAdmission(
    const FamilyCapabilityParams &capability,
    const CanonicalActorSchemaProjection &actor) {
  const auto &params = std::get<fabric::ScalarIntegerParams>(capability);
  if (llvm::Error error = validateIntegerWidths(
          params.integerWidths, ordinaryIntegerWidths(), "ordinary scalar"))
    return error;
  return admitUniformInteger(actor, params.integerWidths, 1);
}

llvm::Error
admitScalarLogicIntegerAdmission(const FamilyCapabilityParams &capability,
                                 const CanonicalActorSchemaProjection &actor) {
  const auto &params = std::get<fabric::ScalarIntegerParams>(capability);
  if (llvm::Error error = validateIntegerWidths(
          params.integerWidths, logicIntegerWidths(), "logic scalar"))
    return error;
  switch (actor.schema) {
  case OperationSchemaId::ArithAndI:
  case OperationSchemaId::ArithOrI:
  case OperationSchemaId::ArithXOrI:
  case OperationSchemaId::LLVMOrDisjoint:
    return admitUniformInteger(actor, params.integerWidths, 2);
  default:
    return reject("logic admission provider received an unsupported schema");
  }
}

llvm::Expected<::mlir::arith::CmpIPredicate>
integerPredicate(const CanonicalActorSchemaProjection &actor) {
  if (actor.schema == OperationSchemaId::ArithCmpI) {
    const auto *payload =
        std::get_if<::dataflow::IntegerComparePayload>(&actor.payload);
    if (!payload)
      return reject("integer comparison has no typed predicate projection");
    return payload->predicate;
  }
  switch (actor.schema) {
  case OperationSchemaId::ArithMinSI:
    return ::mlir::arith::CmpIPredicate::slt;
  case OperationSchemaId::ArithMaxSI:
    return ::mlir::arith::CmpIPredicate::sgt;
  case OperationSchemaId::ArithMinUI:
    return ::mlir::arith::CmpIPredicate::ult;
  case OperationSchemaId::ArithMaxUI:
    return ::mlir::arith::CmpIPredicate::ugt;
  default:
    return reject("integer compare provider received an unsupported schema");
  }
}

llvm::Error admitScalarIntegerCompareAdmission(
    const FamilyCapabilityParams &capability,
    const CanonicalActorSchemaProjection &actor) {
  const auto &params =
      std::get<fabric::ScalarIntegerCompareMinMaxParams>(capability);
  if (llvm::Error error = validateIntegerWidths(
          params.operandWidths, ordinaryIntegerWidths(), "comparison operand"))
    return error;
  if (!params.predicates.valid())
    return reject("invalid integer predicate set");
  if (params.predicates.empty())
    return reject("non-empty integer predicate set required");

  ::mlir::Type operandType;
  if (actor.schema == OperationSchemaId::ArithCmpI) {
    if (llvm::Error error = requireArity(actor, 2, 1))
      return error;
    if (actor.type.getInput(0) != actor.type.getInput(1) ||
        actor.type.getResult(0) !=
            ::mlir::IntegerType::get(actor.type.getContext(), 1))
      return reject("integer comparison function type is malformed");
    operandType = actor.type.getInput(0);
  } else {
    if (llvm::Error error = requireUniformType(actor, 2))
      return error;
    operandType = actor.type.getInput(0);
  }
  llvm::Expected<IntegerWidth> width =
      integerWidth(operandType, "integer comparison");
  if (!width)
    return width.takeError();
  if (!params.operandWidths.contains(*width))
    return reject("integer width is not admitted for comparison");

  llvm::Expected<::mlir::arith::CmpIPredicate> predicate =
      integerPredicate(actor);
  if (!predicate)
    return predicate.takeError();
  if (!params.predicates.contains(*predicate))
    return reject("integer predicate is not admitted");
  return llvm::Error::success();
}

llvm::Error
admitScalarValueSelectAdmission(const FamilyCapabilityParams &capability,
                                const CanonicalActorSchemaProjection &actor) {
  const auto &params = std::get<fabric::ScalarValueSelectParams>(capability);
  if (!params.integerWidths.valid())
    return reject("invalid select integer width set");
  if (!params.integerWidths.isSubsetOf(logicIntegerWidths()))
    return reject("select integer width set contains an unsupported width");
  if (!params.floatFormats.valid())
    return reject("invalid select floating format set");
  if (params.integerWidths.empty() && params.floatFormats.empty())
    return reject("select requires a non-empty scalar value domain");
  if (llvm::Error error = requireArity(actor, 3, 1))
    return error;
  auto condition =
      ::llvm::dyn_cast<::mlir::IntegerType>(actor.type.getInput(0));
  if (!condition || !condition.isSignless() || condition.getWidth() != 1)
    return reject("select condition must be scalar i1");
  ::mlir::Type valueType = actor.type.getInput(1);
  if (actor.type.getInput(2) != valueType ||
      actor.type.getResult(0) != valueType)
    return reject("select value types do not agree");
  if (::llvm::isa<::mlir::IntegerType>(valueType)) {
    llvm::Expected<IntegerWidth> width =
        integerWidth(valueType, "scalar value select");
    if (!width)
      return width.takeError();
    if (!params.integerWidths.contains(*width))
      return reject("select integer width is not admitted");
    return llvm::Error::success();
  }
  llvm::Expected<FloatFormat> format =
      floatFormat(valueType, "scalar value select");
  if (!format)
    return format.takeError();
  if (!params.floatFormats.contains(*format))
    return reject("select floating format is not admitted");
  return llvm::Error::success();
}

llvm::Error
admitScalarIntegerCastAdmission(const FamilyCapabilityParams &capability,
                                const CanonicalActorSchemaProjection &actor) {
  const auto &params = std::get<fabric::ScalarIntegerCastParams>(capability);
  if (!params.relation.widthPairs.valid() ||
      !params.relation.resolvedIndexWidths.valid())
    return reject("invalid integer cast relation");
  if (params.relation.widthPairs.empty())
    return reject("non-empty integer cast relation required");
  if (llvm::Error error = requireArity(actor, 1, 1))
    return error;

  ::mlir::Type sourceType = actor.type.getInput(0);
  ::mlir::Type destinationType = actor.type.getResult(0);
  if (actor.schema == OperationSchemaId::ArithIndexCast ||
      actor.schema == OperationSchemaId::ArithIndexCastUI) {
    bool sourceIsIndex = ::llvm::isa<::mlir::IndexType>(sourceType);
    bool destinationIsIndex = ::llvm::isa<::mlir::IndexType>(destinationType);
    if (sourceIsIndex == destinationIsIndex)
      return reject("index cast requires exactly one index endpoint");
    llvm::Expected<IntegerWidth> integer = integerWidth(
        sourceIsIndex ? destinationType : sourceType, "integer cast relation");
    if (!integer)
      return integer.takeError();
    for (fabric::ResolvedIndexWidth resolved :
         fabric::resolvedIndexWidthDomain) {
      if (!params.relation.resolvedIndexWidths.contains(resolved))
        continue;
      const IntegerWidth index = resolved == fabric::ResolvedIndexWidth::I32
                                     ? IntegerWidth::I32
                                     : IntegerWidth::I64;
      const IntegerWidth source = sourceIsIndex ? index : *integer;
      const IntegerWidth destination = sourceIsIndex ? *integer : index;
      if (params.relation.widthPairs.contains(source, destination))
        return llvm::Error::success();
    }
    return reject("integer cast relation does not admit the resolved index "
                  "endpoint pair");
  }

  llvm::Expected<IntegerWidth> source =
      integerWidth(sourceType, "integer cast relation");
  if (!source)
    return source.takeError();
  llvm::Expected<IntegerWidth> destination =
      integerWidth(destinationType, "integer cast relation");
  if (!destination)
    return destination.takeError();
  if (!params.relation.widthPairs.contains(*source, *destination))
    return reject("integer cast relation does not admit the endpoint pair");
  unsigned sourceBits = integerBitWidth(*source);
  unsigned destinationBits = integerBitWidth(*destination);
  switch (actor.schema) {
  case OperationSchemaId::ArithExtSI:
  case OperationSchemaId::ArithExtUI:
    if (sourceBits >= destinationBits)
      return reject("integer extension must widen");
    break;
  case OperationSchemaId::ArithTruncI:
    if (sourceBits <= destinationBits)
      return reject("integer truncation must narrow");
    break;
  case OperationSchemaId::ArithIndexCast:
  case OperationSchemaId::ArithIndexCastUI:
    break;
  default:
    return reject("integer cast provider received an unsupported schema");
  }
  return llvm::Error::success();
}

} // namespace
