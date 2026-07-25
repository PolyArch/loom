//===- ImplementationFamily.cpp - Normative HSG family registry ----------===//
//
// Expands the one generated implementation-family registry into dense tables.
// The member relation, the capability-parameter schema binding, and the typed
// admission provider binding all come from the same generated rows, so there
// is no second member list, family-shape switch, or backend-local table.
//
//===----------------------------------------------------------------------===//

#include "Fabric/IR/ImplementationFamily.h"

#include "Dataflow/IR/DataflowActorSemantics.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"

#include <array>
#include <cstddef>
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

using dataflow::CanonicalActorSemantics;
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
llvm::Error requireArity(const CanonicalActorSemantics &actor, unsigned inputs,
                         unsigned results);
llvm::Error requireUniformType(const CanonicalActorSemantics &actor,
                               unsigned inputs);
llvm::Error
admitFloatBehavior(const FloatBehaviorProfile &behavior,
                   ::mlir::arith::FastMathFlags actorFlags,
                   std::optional<::mlir::arith::RoundingMode> rounding,
                   FloatNaNBehavior nanBehavior);
llvm::Expected<::mlir::arith::FastMathFlags>
floatingFlags(const CanonicalActorSemantics &actor);
std::optional<::mlir::arith::RoundingMode>
arithmeticRounding(const CanonicalActorSemantics &actor);

llvm::Error
admitScalarOrdinaryIntegerAdmission(const FamilyCapabilityParams &capability,
                                    const CanonicalActorSemantics &actor);
llvm::Error
admitScalarLogicIntegerAdmission(const FamilyCapabilityParams &capability,
                                 const CanonicalActorSemantics &actor);
llvm::Error
admitScalarIntegerCompareAdmission(const FamilyCapabilityParams &capability,
                                   const CanonicalActorSemantics &actor);
llvm::Error
admitScalarIntegerCastAdmission(const FamilyCapabilityParams &capability,
                                const CanonicalActorSemantics &actor);
llvm::Error
admitScalarBitReinterpretAdmission(const FamilyCapabilityParams &capability,
                                   const CanonicalActorSemantics &actor);
llvm::Error
admitScalarValueSelectAdmission(const FamilyCapabilityParams &capability,
                                const CanonicalActorSemantics &actor);
llvm::Error
admitScalarUniformFloatAdmission(const FamilyCapabilityParams &capability,
                                 const CanonicalActorSemantics &actor);
llvm::Error
admitScalarFloatCompareAdmission(const FamilyCapabilityParams &capability,
                                 const CanonicalActorSemantics &actor);
llvm::Error
admitScalarFloatCastAdmission(const FamilyCapabilityParams &capability,
                              const CanonicalActorSemantics &actor);
llvm::Error admitScalarIntegerFloatConversionAdmission(
    const FamilyCapabilityParams &capability,
    const CanonicalActorSemantics &actor);
llvm::Error admitStreamAdmission(const FamilyCapabilityParams &capability,
                                 const CanonicalActorSemantics &actor);
llvm::Error admitTokenPlaneAdmission(const FamilyCapabilityParams &capability,
                                     const CanonicalActorSemantics &actor);

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
                                 const CanonicalActorSemantics &actor) {
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

llvm::Error admitTokenPlaneAdmission(const FamilyCapabilityParams &capability,
                                     const CanonicalActorSemantics &actor) {
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

} // namespace

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
    const ::dataflow::CanonicalActorSemantics &actor) {
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

llvm::Error
admitScalarBitReinterpretAdmission(const FamilyCapabilityParams &capability,
                                   const CanonicalActorSemantics &actor) {
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

llvm::Error admitUniformFloatType(const CanonicalActorSemantics &actor,
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
                                 const CanonicalActorSemantics &actor) {
  const auto &params = std::get<fabric::ScalarFloatParams>(capability);
  if (llvm::Error error = validateFloatFormats(params.formats, "scalar"))
    return error;

  unsigned inputCount = 0;
  bool hasArithmeticRounding = false;
  switch (actor.schema) {
  case OperationSchemaId::ArithNegF:
  case OperationSchemaId::MathAbsF:
    inputCount = 1;
    break;
  case OperationSchemaId::ArithAddF:
  case OperationSchemaId::ArithSubF:
  case OperationSchemaId::ArithMulF:
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
floatPredicate(const CanonicalActorSemantics &actor) {
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
                                 const CanonicalActorSemantics &actor) {
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
                              const CanonicalActorSemantics &actor) {
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
    const CanonicalActorSemantics &actor) {
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

llvm::Error requireArity(const CanonicalActorSemantics &actor, unsigned inputs,
                         unsigned results) {
  if (actor.type.getNumInputs() != inputs ||
      actor.type.getNumResults() != results)
    return reject("actor function type has the wrong arity");
  return llvm::Error::success();
}

llvm::Error requireUniformType(const CanonicalActorSemantics &actor,
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
floatingFlags(const CanonicalActorSemantics &actor) {
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
arithmeticRounding(const CanonicalActorSemantics &actor) {
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

llvm::Error admitUniformInteger(const CanonicalActorSemantics &actor,
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

llvm::Error
admitScalarOrdinaryIntegerAdmission(const FamilyCapabilityParams &capability,
                                    const CanonicalActorSemantics &actor) {
  const auto &params = std::get<fabric::ScalarIntegerParams>(capability);
  if (llvm::Error error = validateIntegerWidths(
          params.integerWidths, ordinaryIntegerWidths(), "ordinary scalar"))
    return error;
  switch (actor.schema) {
  case OperationSchemaId::ArithAddI:
  case OperationSchemaId::ArithSubI:
  case OperationSchemaId::ArithShLI:
  case OperationSchemaId::ArithShRSI:
  case OperationSchemaId::ArithShRUI:
  case OperationSchemaId::ArithMulI:
    return admitUniformInteger(actor, params.integerWidths, 2);
  default:
    return reject("integer admission provider received an unsupported schema");
  }
}

llvm::Error
admitScalarLogicIntegerAdmission(const FamilyCapabilityParams &capability,
                                 const CanonicalActorSemantics &actor) {
  const auto &params = std::get<fabric::ScalarIntegerParams>(capability);
  if (llvm::Error error = validateIntegerWidths(
          params.integerWidths, logicIntegerWidths(), "logic scalar"))
    return error;
  switch (actor.schema) {
  case OperationSchemaId::ArithAndI:
  case OperationSchemaId::ArithOrI:
  case OperationSchemaId::ArithXOrI:
    return admitUniformInteger(actor, params.integerWidths, 2);
  default:
    return reject("logic admission provider received an unsupported schema");
  }
}

llvm::Expected<::mlir::arith::CmpIPredicate>
integerPredicate(const CanonicalActorSemantics &actor) {
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

llvm::Error
admitScalarIntegerCompareAdmission(const FamilyCapabilityParams &capability,
                                   const CanonicalActorSemantics &actor) {
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
                                const CanonicalActorSemantics &actor) {
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

llvm::Expected<IntegerWidth>
resolvedIndexWidth(const fabric::IntegerCastRelation &relation) {
  if (!relation.resolvedIndexWidth)
    return reject("resolved index width is required");
  switch (*relation.resolvedIndexWidth) {
  case fabric::ResolvedIndexWidth::I32:
    return IntegerWidth::I32;
  case fabric::ResolvedIndexWidth::I64:
    return IntegerWidth::I64;
  }
  return reject("resolved index width is invalid");
}

llvm::Error
admitScalarIntegerCastAdmission(const FamilyCapabilityParams &capability,
                                const CanonicalActorSemantics &actor) {
  const auto &params = std::get<fabric::ScalarIntegerCastParams>(capability);
  if (!params.relation.widthPairs.valid())
    return reject("invalid integer cast relation");
  if (params.relation.widthPairs.empty())
    return reject("non-empty integer cast relation required");
  if (llvm::Error error = requireArity(actor, 1, 1))
    return error;

  ::mlir::Type sourceType = actor.type.getInput(0);
  ::mlir::Type destinationType = actor.type.getResult(0);
  IntegerWidth source;
  IntegerWidth destination;
  if (actor.schema == OperationSchemaId::ArithIndexCast ||
      actor.schema == OperationSchemaId::ArithIndexCastUI) {
    bool sourceIsIndex = ::llvm::isa<::mlir::IndexType>(sourceType);
    bool destinationIsIndex = ::llvm::isa<::mlir::IndexType>(destinationType);
    if (sourceIsIndex == destinationIsIndex)
      return reject("index cast requires exactly one index endpoint");
    llvm::Expected<IntegerWidth> index = resolvedIndexWidth(params.relation);
    if (!index)
      return index.takeError();
    llvm::Expected<IntegerWidth> integer = integerWidth(
        sourceIsIndex ? destinationType : sourceType, "integer cast relation");
    if (!integer)
      return integer.takeError();
    source = sourceIsIndex ? *index : *integer;
    destination = sourceIsIndex ? *integer : *index;
  } else {
    llvm::Expected<IntegerWidth> sourceWidth =
        integerWidth(sourceType, "integer cast relation");
    if (!sourceWidth)
      return sourceWidth.takeError();
    llvm::Expected<IntegerWidth> destinationWidth =
        integerWidth(destinationType, "integer cast relation");
    if (!destinationWidth)
      return destinationWidth.takeError();
    source = *sourceWidth;
    destination = *destinationWidth;
  }

  if (!params.relation.widthPairs.contains(source, destination))
    return reject("integer cast relation does not admit the endpoint pair");
  unsigned sourceBits = integerBitWidth(source);
  unsigned destinationBits = integerBitWidth(destination);
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
