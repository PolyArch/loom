//===- ImplementationFamilyQueries.cpp - HSG capability queries ----------===//
//
// Implements read-only projections over the generated implementation-family
// registry and concrete capability parameters.
//
//===----------------------------------------------------------------------===//

#include "Fabric/IR/ImplementationFamily.h"

#include "Common/IndexWidth.h"
#include "Common/VectorWidth.h"
#include "Dataflow/IR/DataflowActorSemantics.h"
#include "Dataflow/IR/OperationSchemaCodec.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CheckedArithmetic.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace {

constexpr std::size_t kFamilyCount = 0
#define LOOM_IMPLEMENTATION_FAMILY(Name, Id, CapabilityParams, TypedAdmission) \
  +1
#include "Fabric/IR/ImplementationFamilies.inc"
    ;

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

llvm::Error reject(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(), message);
}

constexpr char kScalarIntegerCompareConfigurationDomain[] =
    "loom.fabric.scalar-integer-compare-min-max-configuration\0";
constexpr char kFixedVectorIntegerAddSubConfigurationDomain[] =
    "loom.fabric.fixed-vector-integer-add-sub-configuration\0";
constexpr char kFixedVectorIntegerMultiplyConfigurationDomain[] =
    "loom.fabric.fixed-vector-integer-multiply-configuration\0";
constexpr char kFixedVectorValueSelectConfigurationDomain[] =
    "loom.fabric.fixed-vector-value-select-configuration\0";
constexpr char kIntegerLogicConfigurationDomain[] =
    "loom.fabric.integer-logic-configuration\0";
constexpr char kScalarIntegerCastConfigurationDomain[] =
    "loom.fabric.scalar-integer-cast-configuration\0";
constexpr std::uint32_t kConfigurationCodecMajor = 1;
constexpr std::uint32_t kConfigurationCodecMinor = 0;
constexpr std::array<mlir::arith::CmpIPredicate, 4> kSignedIntegerPredicates = {
    mlir::arith::CmpIPredicate::slt, mlir::arith::CmpIPredicate::sle,
    mlir::arith::CmpIPredicate::sgt, mlir::arith::CmpIPredicate::sge};

void appendU32(std::vector<std::uint8_t> &bytes, std::uint32_t value) {
  bytes.push_back(static_cast<std::uint8_t>(value >> 24));
  bytes.push_back(static_cast<std::uint8_t>(value >> 16));
  bytes.push_back(static_cast<std::uint8_t>(value >> 8));
  bytes.push_back(static_cast<std::uint8_t>(value));
}

void appendU64(std::vector<std::uint8_t> &bytes, std::uint64_t value) {
  for (unsigned shift = 56; shift != 0; shift -= 8)
    bytes.push_back(static_cast<std::uint8_t>(value >> shift));
  bytes.push_back(static_cast<std::uint8_t>(value));
}

void appendFramed(std::vector<std::uint8_t> &bytes,
                  llvm::ArrayRef<std::uint8_t> value) {
  appendU64(bytes, value.size());
  bytes.insert(bytes.end(), value.begin(), value.end());
}

std::uint32_t bitsForCardinality(std::uint32_t cardinality) {
  return cardinality <= 1 ? 0 : llvm::Log2_32_Ceil(cardinality);
}

std::uint32_t bitsForInclusiveMaximum(std::uint32_t maximum) {
  return maximum == std::numeric_limits<std::uint32_t>::max()
             ? 32
             : bitsForCardinality(maximum + 1);
}

llvm::Error appendField(std::uint32_t &cursor, std::uint32_t width) {
  auto next = llvm::checkedAddUnsigned(cursor, width);
  if (!next)
    return reject("semantic configuration layout exceeds uint32");
  cursor = *next;
  return llvm::Error::success();
}

void setPackedField(std::vector<std::uint8_t> &bytes, std::uint32_t offset,
                    std::uint32_t width, std::uint64_t value) {
  for (std::uint32_t bit = 0; bit != width; ++bit) {
    const std::size_t byte = (offset + bit) / 8;
    const std::uint8_t mask = std::uint8_t(1U << ((offset + bit) % 8));
    if (((value >> bit) & 1U) != 0)
      bytes[byte] |= mask;
  }
}

std::vector<std::uint8_t> emptyPackedValue(std::uint32_t bitCount) {
  return std::vector<std::uint8_t>(
      bitCount / 8 + static_cast<std::uint32_t>((bitCount % 8) != 0), 0);
}

llvm::Expected<std::uint64_t> vectorStructuralWidth(mlir::Type type) {
  if (auto vector = llvm::dyn_cast<mlir::VectorType>(type))
    return ::dataflow::semantics::getFlattenedVectorBitWidth(vector);
  if (auto integer = llvm::dyn_cast<mlir::IntegerType>(type))
    return integer.getWidth();
  if (auto floating = llvm::dyn_cast<mlir::FloatType>(type))
    return floating.getWidth();
  return reject("vector structural value has no fixed payload width");
}

struct FixedVectorSliceProjection final {
  std::uint64_t staticOffsetBits = 0;
  std::uint64_t sliceWidthBits = 0;
  std::vector<std::uint64_t> dynamicStrideBits;
};

llvm::Expected<FixedVectorSliceProjection> projectFixedVectorSlice(
    const ::dataflow::CanonicalActorSchemaProjection &actor) {
  const auto *payload =
      std::get_if<::dataflow::VectorStaticPositionPayload>(&actor.payload);
  if (!payload)
    return reject("vector slice actor has the wrong semantic payload");
  mlir::VectorType container;
  mlir::Type slice;
  if (actor.schema == ::dataflow::OperationSchemaId::VectorExtract) {
    if (actor.type.getNumInputs() == 0 || actor.type.getNumResults() != 1)
      return reject("vector extract projector has incomplete actor ports");
    container = llvm::dyn_cast<mlir::VectorType>(actor.type.getInput(0));
    slice = actor.type.getResult(0);
  } else if (actor.schema == ::dataflow::OperationSchemaId::VectorInsert) {
    if (actor.type.getNumInputs() < 2 || actor.type.getNumResults() != 1)
      return reject("vector insert projector has incomplete actor ports");
    slice = actor.type.getInput(0);
    container = llvm::dyn_cast<mlir::VectorType>(actor.type.getInput(1));
  } else {
    return reject("vector slice projector received a different schema");
  }
  if (!container)
    return reject("vector slice projector has no container vector");
  auto sliceWidth = vectorStructuralWidth(slice);
  if (!sliceWidth)
    return sliceWidth.takeError();

  FixedVectorSliceProjection projection;
  projection.sliceWidthBits = *sliceWidth;
  for (auto [dimension, position] : llvm::enumerate(payload->position)) {
    std::uint64_t stride = container.getElementTypeBitWidth();
    for (std::int64_t extent : container.getShape().drop_front(dimension + 1)) {
      auto next =
          llvm::checkedMulUnsigned(stride, static_cast<std::uint64_t>(extent));
      if (!next)
        return reject("vector slice stride overflows uint64");
      stride = *next;
    }
    if (position == mlir::ShapedType::kDynamic) {
      projection.dynamicStrideBits.push_back(stride);
      continue;
    }
    auto contribution =
        llvm::checkedMulUnsigned(stride, static_cast<std::uint64_t>(position));
    if (!contribution)
      return reject("vector slice static offset overflows uint64");
    auto next =
        llvm::checkedAddUnsigned(projection.staticOffsetBits, *contribution);
    if (!next)
      return reject("vector slice static offset overflows uint64");
    projection.staticOffsetBits = *next;
  }
  return projection;
}

bool isSignedIntegerPredicate(mlir::arith::CmpIPredicate predicate) {
  return llvm::is_contained(kSignedIntegerPredicates, predicate);
}

enum class IntegerLogicOperation : std::uint32_t { And, Or, Xor };

using fabric::IntegerWidth;
using fabric::integerWidthDomain;
using fabric::ResolvedIndexWidth;
using fabric::resolvedIndexWidthDomain;
using fabric::ScalarIntegerCastParams;

enum class ScalarIntegerCastOperation : std::uint32_t {
  Identity = 0,
  SignExtend = 1,
  ZeroExtend = 2,
  Truncate = 3,
};

struct ScalarIntegerCastBehavior final {
  ScalarIntegerCastOperation operation;
  IntegerWidth sourceWidth;
  IntegerWidth destinationWidth;

  friend bool operator==(const ScalarIntegerCastBehavior &lhs,
                         const ScalarIntegerCastBehavior &rhs) {
    return lhs.operation == rhs.operation &&
           lhs.sourceWidth == rhs.sourceWidth &&
           lhs.destinationWidth == rhs.destinationWidth;
  }
};

struct ScalarIntegerCastCase final {
  ::dataflow::OperationSchemaId schema;
  ScalarIntegerCastBehavior behavior;
  bool sourceIsIndex = false;
  bool destinationIsIndex = false;
  std::optional<ResolvedIndexWidth> resolvedIndexWidth;
};

ScalarIntegerCastOperation
classifyScalarIntegerCast(::dataflow::OperationSchemaId schema,
                          IntegerWidth source, IntegerWidth destination) {
  const unsigned sourceBits = fabric::getBitWidth(source);
  const unsigned destinationBits = fabric::getBitWidth(destination);
  if (sourceBits == destinationBits)
    return ScalarIntegerCastOperation::Identity;
  if (sourceBits > destinationBits)
    return ScalarIntegerCastOperation::Truncate;
  if (schema == ::dataflow::OperationSchemaId::ArithExtSI ||
      schema == ::dataflow::OperationSchemaId::ArithIndexCast)
    return ScalarIntegerCastOperation::SignExtend;
  return ScalarIntegerCastOperation::ZeroExtend;
}

llvm::Expected<std::vector<ScalarIntegerCastCase>>
enumerateScalarIntegerCastCases(
    const ScalarIntegerCastParams &parameters,
    llvm::ArrayRef<::dataflow::OperationSchemaId> enabledSchemas) {
  if (!parameters.relation.widthPairs.valid() ||
      !parameters.relation.resolvedIndexWidths.valid())
    return reject("invalid integer cast relation");
  if (parameters.relation.widthPairs.empty())
    return reject("non-empty integer cast relation required");

  std::vector<ScalarIntegerCastCase> cases;
  for (::dataflow::OperationSchemaId schema : enabledSchemas) {
    const std::size_t schemaStart = cases.size();
    using Schema = ::dataflow::OperationSchemaId;
    const bool isIndexCast =
        schema == Schema::ArithIndexCast || schema == Schema::ArithIndexCastUI;
    if (isIndexCast) {
      for (ResolvedIndexWidth resolved : resolvedIndexWidthDomain) {
        if (!parameters.relation.resolvedIndexWidths.contains(resolved))
          continue;
        const IntegerWidth indexWidth = resolved == ResolvedIndexWidth::I32
                                            ? IntegerWidth::I32
                                            : IntegerWidth::I64;
        for (IntegerWidth source : integerWidthDomain) {
          for (IntegerWidth destination : integerWidthDomain) {
            if (!parameters.relation.widthPairs.contains(source, destination))
              continue;
            const ScalarIntegerCastBehavior behavior{
                classifyScalarIntegerCast(schema, source, destination), source,
                destination};
            if (source == indexWidth)
              cases.push_back({schema, behavior, true, false, resolved});
            if (destination == indexWidth)
              cases.push_back({schema, behavior, false, true, resolved});
          }
        }
      }
      if (cases.size() == schemaStart)
        return reject("integer cast schema has no admitted finite behavior");
      continue;
    }

    if (schema != Schema::ArithExtSI && schema != Schema::ArithExtUI &&
        schema != Schema::ArithTruncI)
      return reject("integer cast capability contains a non-cast schema");
    for (IntegerWidth source : integerWidthDomain) {
      for (IntegerWidth destination : integerWidthDomain) {
        if (!parameters.relation.widthPairs.contains(source, destination))
          continue;
        const unsigned sourceBits = fabric::getBitWidth(source);
        const unsigned destinationBits = fabric::getBitWidth(destination);
        const bool admittedPair = schema == Schema::ArithTruncI
                                      ? sourceBits > destinationBits
                                      : sourceBits < destinationBits;
        if (!admittedPair)
          continue;
        cases.push_back(
            {schema,
             {classifyScalarIntegerCast(schema, source, destination), source,
              destination},
             false,
             false,
             std::nullopt});
      }
    }
    if (cases.size() == schemaStart)
      return reject("integer cast schema has no admitted finite behavior");
  }
  for (IntegerWidth source : integerWidthDomain) {
    for (IntegerWidth destination : integerWidthDomain) {
      if (!parameters.relation.widthPairs.contains(source, destination))
        continue;
      const bool interpreted =
          llvm::any_of(cases, [&](const ScalarIntegerCastCase &castCase) {
            return castCase.behavior.sourceWidth == source &&
                   castCase.behavior.destinationWidth == destination;
          });
      if (!interpreted)
        return reject("orphan integer cast width pair");
    }
  }
  for (ResolvedIndexWidth resolved : resolvedIndexWidthDomain) {
    if (!parameters.relation.resolvedIndexWidths.contains(resolved))
      continue;
    const bool interpreted =
        llvm::any_of(cases, [&](const ScalarIntegerCastCase &castCase) {
          return castCase.resolvedIndexWidth == resolved;
        });
    if (!interpreted)
      return reject("orphan resolved index width");
  }
  if (cases.empty())
    return reject("integer cast capability has no admitted finite behavior");
  return cases;
}

std::vector<ScalarIntegerCastBehavior>
uniqueScalarIntegerCastBehaviors(llvm::ArrayRef<ScalarIntegerCastCase> cases) {
  std::vector<ScalarIntegerCastBehavior> behaviors;
  for (const ScalarIntegerCastCase &castCase : cases)
    if (!llvm::is_contained(behaviors, castCase.behavior))
      behaviors.push_back(castCase.behavior);
  return behaviors;
}

llvm::Expected<::dataflow::SemanticPayload>
makeScalarIntegerCastPayload(::dataflow::OperationSchemaId schema) {
  using Case = ::dataflow::OperationSemanticsCase;
  switch (::dataflow::semanticsCase(schema)) {
  case Case::NoSemanticPayload:
    return ::dataflow::NoPayload{};
  case Case::ArithNonNegative:
    return ::dataflow::NonNegativePayload{};
  case Case::ArithIntegerOverflow:
    return ::dataflow::IntegerOverflowPayload{};
  default:
    return reject("integer cast schema has an unexpected semantic payload");
  }
}

llvm::Expected<::dataflow::CanonicalActorSchemaProjection>
makeScalarIntegerCastActor(mlir::MLIRContext &context,
                           const ScalarIntegerCastCase &castCase) {
  auto payload = makeScalarIntegerCastPayload(castCase.schema);
  if (!payload)
    return payload.takeError();
  mlir::Type source =
      castCase.sourceIsIndex
          ? mlir::Type(mlir::IndexType::get(&context))
          : mlir::Type(mlir::IntegerType::get(
                &context, fabric::getBitWidth(castCase.behavior.sourceWidth)));
  mlir::Type destination =
      castCase.destinationIsIndex
          ? mlir::Type(mlir::IndexType::get(&context))
          : mlir::Type(mlir::IntegerType::get(
                &context,
                fabric::getBitWidth(castCase.behavior.destinationWidth)));
  return ::dataflow::CanonicalActorSchemaProjection{
      castCase.schema,
      mlir::FunctionType::get(&context, {source}, {destination}),
      std::move(*payload)};
}

llvm::Expected<IntegerWidth> scalarIntegerCastWidth(mlir::Type type) {
  auto integer = llvm::dyn_cast<mlir::IntegerType>(type);
  if (!integer || !integer.isSignless())
    return reject("integer cast endpoint is not a signless integer");
  for (IntegerWidth width : integerWidthDomain)
    if (fabric::getBitWidth(width) == integer.getWidth())
      return width;
  return reject("integer cast endpoint width is outside the closed domain");
}

llvm::Expected<ScalarIntegerCastBehavior> resolveScalarIntegerCastBehavior(
    const ::dataflow::CanonicalActorSchemaProjection &actor,
    std::optional<ResolvedIndexWidth> resolvedIndexWidth) {
  if (actor.type.getNumInputs() != 1 || actor.type.getNumResults() != 1)
    return reject("integer cast behavior has wrong arity");
  using Schema = ::dataflow::OperationSchemaId;
  const bool isIndexCast = actor.schema == Schema::ArithIndexCast ||
                           actor.schema == Schema::ArithIndexCastUI;
  if (!isIndexCast && actor.schema != Schema::ArithExtSI &&
      actor.schema != Schema::ArithExtUI && actor.schema != Schema::ArithTruncI)
    return reject("actor is not an integer cast schema");

  mlir::Type sourceType = actor.type.getInput(0);
  mlir::Type destinationType = actor.type.getResult(0);
  const bool sourceIsIndex = llvm::isa<mlir::IndexType>(sourceType);
  const bool destinationIsIndex = llvm::isa<mlir::IndexType>(destinationType);
  if (isIndexCast) {
    if (sourceIsIndex == destinationIsIndex)
      return reject("index cast requires exactly one index endpoint");
    if (!resolvedIndexWidth)
      return reject("index cast behavior has no resolved index width");
  } else if (sourceIsIndex || destinationIsIndex) {
    return reject("ordinary integer cast has an index endpoint");
  }

  const IntegerWidth indexWidth =
      resolvedIndexWidth && *resolvedIndexWidth == ResolvedIndexWidth::I32
          ? IntegerWidth::I32
          : IntegerWidth::I64;
  auto source = sourceIsIndex ? llvm::Expected<IntegerWidth>(indexWidth)
                              : scalarIntegerCastWidth(sourceType);
  if (!source)
    return source.takeError();
  auto destination = destinationIsIndex
                         ? llvm::Expected<IntegerWidth>(indexWidth)
                         : scalarIntegerCastWidth(destinationType);
  if (!destination)
    return destination.takeError();
  return ScalarIntegerCastBehavior{
      classifyScalarIntegerCast(actor.schema, *source, *destination), *source,
      *destination};
}

llvm::Expected<IntegerLogicOperation>
integerLogicOperation(::dataflow::OperationSchemaId schema) {
  using Schema = ::dataflow::OperationSchemaId;
  switch (schema) {
  case Schema::ArithAndI:
    return IntegerLogicOperation::And;
  case Schema::ArithOrI:
  case Schema::LLVMOrDisjoint:
    return IntegerLogicOperation::Or;
  case Schema::ArithXOrI:
    return IntegerLogicOperation::Xor;
  default:
    return reject("integer logic capability contains a non-logic schema");
  }
}

::dataflow::CanonicalActorSchemaProjection makeScalarIntegerCompareActor(
    mlir::MLIRContext &context, unsigned width,
    ::dataflow::OperationSchemaId schema,
    std::optional<mlir::arith::CmpIPredicate> predicate = std::nullopt) {
  mlir::Type operand = mlir::IntegerType::get(&context, width);
  const bool comparison = schema == ::dataflow::OperationSchemaId::ArithCmpI;
  mlir::FunctionType type = mlir::FunctionType::get(
      &context, {operand, operand},
      {comparison ? mlir::IntegerType::get(&context, 1) : operand});
  if (comparison)
    return {schema, type,
            ::dataflow::IntegerComparePayload{
                predicate.value_or(mlir::arith::CmpIPredicate::eq)}};
  return {schema, type, ::dataflow::NoPayload{}};
}

llvm::Expected<::dataflow::SemanticPayload>
makeIntegerLogicPayload(::dataflow::OperationSchemaId schema) {
  using Schema = ::dataflow::OperationSchemaId;
  switch (schema) {
  case Schema::ArithAndI:
  case Schema::ArithOrI:
  case Schema::ArithXOrI:
    return ::dataflow::SemanticPayload(::dataflow::NoPayload{});
  case Schema::LLVMOrDisjoint:
    return ::dataflow::SemanticPayload(::dataflow::DisjointPayload{true});
  default:
    return reject("integer logic capability contains a non-logic schema");
  }
}

llvm::Expected<::dataflow::CanonicalActorSchemaProjection>
makeScalarIntegerLogicActor(mlir::MLIRContext &context, unsigned width,
                            ::dataflow::OperationSchemaId schema) {
  auto payload = makeIntegerLogicPayload(schema);
  if (!payload)
    return payload.takeError();
  mlir::Type operand = mlir::IntegerType::get(&context, width);
  return ::dataflow::CanonicalActorSchemaProjection{
      schema, mlir::FunctionType::get(&context, {operand, operand}, {operand}),
      std::move(*payload)};
}

llvm::Expected<mlir::VectorType>
makeMaximalVectorType(mlir::Type element, std::uint32_t maxPayloadBits) {
  const unsigned elementWidth = element.getIntOrFloatBitWidth();
  if (elementWidth == 0)
    return reject("fixed-vector element has no finite bit width");
  const std::uint32_t laneCount = maxPayloadBits / elementWidth;
  if (laneCount == 0)
    return reject("fixed-vector element width exceeds payload capacity");
  return mlir::VectorType::get({static_cast<std::int64_t>(laneCount)}, element);
}

llvm::Expected<::dataflow::CanonicalActorSchemaProjection>
makeFixedVectorIntegerActor(mlir::MLIRContext &context, unsigned elementWidth,
                            std::uint32_t maxPayloadBits,
                            ::dataflow::OperationSchemaId schema,
                            ::dataflow::SemanticPayload payload) {
  auto vector = makeMaximalVectorType(
      mlir::IntegerType::get(&context, elementWidth), maxPayloadBits);
  if (!vector)
    return vector.takeError();
  return ::dataflow::CanonicalActorSchemaProjection{
      schema, mlir::FunctionType::get(&context, {*vector, *vector}, {*vector}),
      std::move(payload)};
}

llvm::Expected<::dataflow::CanonicalActorSchemaProjection>
makeFixedVectorIntegerAddSubActor(mlir::MLIRContext &context,
                                  unsigned elementWidth,
                                  std::uint32_t maxPayloadBits,
                                  ::dataflow::OperationSchemaId schema) {
  return makeFixedVectorIntegerActor(context, elementWidth, maxPayloadBits,
                                     schema,
                                     ::dataflow::IntegerOverflowPayload{});
}

llvm::Expected<::dataflow::CanonicalActorSchemaProjection>
makeFixedVectorValueSelectActor(mlir::MLIRContext &context, mlir::Type element,
                                std::uint32_t maxPayloadBits) {
  auto values = makeMaximalVectorType(element, maxPayloadBits);
  if (!values)
    return values.takeError();
  mlir::Type condition = mlir::VectorType::get(
      values->getShape(), mlir::IntegerType::get(&context, 1));
  return ::dataflow::CanonicalActorSchemaProjection{
      ::dataflow::OperationSchemaId::ArithSelect,
      mlir::FunctionType::get(&context, {condition, *values, *values},
                              {*values}),
      ::dataflow::NoPayload{}};
}

loom::CanonicalSemanticBytes
encodeElementWidthConfiguration(llvm::StringRef domain, unsigned elementWidth) {
  constexpr std::uint32_t kElementWidthComponent = 1U << 0;
  std::vector<std::uint8_t> bytes;
  bytes.insert(bytes.end(), domain.bytes_begin(), domain.bytes_end());
  appendU32(bytes, kConfigurationCodecMajor);
  appendU32(bytes, kConfigurationCodecMinor);
  appendU32(bytes, kElementWidthComponent);
  appendU32(bytes, elementWidth);
  return loom::CanonicalSemanticBytes(std::move(bytes));
}

llvm::Expected<loom::CanonicalSemanticBytes>
encodeIntegerLogicConfiguration(::dataflow::OperationSchemaId schema) {
  auto operation = integerLogicOperation(schema);
  if (!operation)
    return operation.takeError();
  std::vector<std::uint8_t> bytes;
  const llvm::StringRef domain(kIntegerLogicConfigurationDomain,
                               sizeof(kIntegerLogicConfigurationDomain) - 1);
  bytes.insert(bytes.end(), domain.bytes_begin(), domain.bytes_end());
  appendU32(bytes, kConfigurationCodecMajor);
  appendU32(bytes, kConfigurationCodecMinor);
  appendU32(bytes, static_cast<std::uint32_t>(*operation));
  return loom::CanonicalSemanticBytes(std::move(bytes));
}

mlir::FloatType floatType(mlir::MLIRContext &context,
                          fabric::FloatFormat format) {
  switch (format) {
  case fabric::FloatFormat::F16:
    return mlir::Float16Type::get(&context);
  case fabric::FloatFormat::BF16:
    return mlir::BFloat16Type::get(&context);
  case fabric::FloatFormat::F32:
    return mlir::Float32Type::get(&context);
  case fabric::FloatFormat::F64:
    return mlir::Float64Type::get(&context);
  }
  llvm_unreachable("unknown floating format");
}

::dataflow::CanonicalActorSchemaProjection
makeScalarFloatFmaActor(mlir::MLIRContext &context,
                        fabric::FloatFormat format) {
  mlir::Type type = floatType(context, format);
  return {::dataflow::OperationSchemaId::MathFma,
          mlir::FunctionType::get(&context, {type, type, type}, {type}),
          ::dataflow::FloatingPointPayload{}};
}

bool isStrictIEEE(const fabric::FloatBehaviorProfile &behavior) {
  using mlir::arith::FastMathFlags;
  using mlir::arith::RoundingMode;
  return behavior.roundingModes.size() == 1 &&
         behavior.roundingModes.contains(RoundingMode::to_nearest_even) &&
         behavior.nanBehaviors.size() == 1 &&
         behavior.nanBehaviors.contains(fabric::FloatNaNBehavior::IEEE) &&
         behavior.subnormalBehaviors.size() == 1 &&
         behavior.subnormalBehaviors.contains(
             fabric::FloatSubnormalBehavior::Preserve) &&
         behavior.signedZeroBehaviors.size() == 1 &&
         behavior.signedZeroBehaviors.contains(
             fabric::FloatSignedZeroBehavior::Preserve) &&
         behavior.requiredFastMath == FastMathFlags::none;
}

mlir::FailureOr<unsigned>
semanticPayloadWidth(mlir::Type type, std::optional<unsigned> indexBitWidth,
                     const ::loom::PointerLayout *pointerLayout,
                     std::string &error) {
  if (auto integer = mlir::dyn_cast<mlir::IntegerType>(type))
    return integer.getWidth();
  if (auto floating = mlir::dyn_cast<mlir::FloatType>(type))
    return floating.getWidth();
  if (mlir::isa<mlir::IndexType>(type)) {
    if (indexBitWidth) {
      if (!fabric::symbolizeResolvedIndexWidth(*indexBitWidth)) {
        error = "resolved index width must be 32 or 64";
        return mlir::failure();
      }
      return *indexBitWidth;
    }
    return ::loom::getIndexWidth();
  }
  if (auto pointer = mlir::dyn_cast<mlir::LLVM::LLVMPointerType>(type)) {
    if (!pointerLayout ||
        pointerLayout->addressSpace != pointer.getAddressSpace()) {
      error = "LLVM pointer payload requires its exact DataLayout projection";
      return mlir::failure();
    }
    return pointerLayout->representationBits;
  }
  if (mlir::isa<mlir::NoneType>(type))
    return 0u;
  if (auto vector = mlir::dyn_cast<mlir::VectorType>(type)) {
    auto elementWidth = semanticPayloadWidth(
        vector.getElementType(), indexBitWidth, pointerLayout, error);
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

unsigned fabric::getBitWidth(IntegerWidth width) {
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

unsigned fabric::getBitWidth(FloatFormat format) {
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

llvm::Expected<fabric::FixedVectorSliceAlignMergeConfigurationLayout>
fabric::resolveFixedVectorSliceAlignMergeConfigurationLayout(
    const FixedVectorSliceAlignMergeParams &params,
    llvm::ArrayRef<::dataflow::OperationSchemaId> enabledSchemas) {
  if (params.maxContainerPayloadBits == 0 || params.maxSlicePayloadBits == 0 ||
      params.maxSlicePayloadBits > params.maxContainerPayloadBits)
    return reject("vector slice payload capacities are invalid");
  if (!params.integerElementWidths.valid() ||
      !params.floatElementFormats.valid() ||
      (params.integerElementWidths.empty() &&
       params.floatElementFormats.empty()))
    return reject("vector slice element domain is invalid");
  if (!params.resolvedIndexWidths.valid() ||
      ((params.maxDynamicPositionRank == 0) !=
       params.resolvedIndexWidths.empty()))
    return reject("vector slice dynamic rank and index domain disagree");

  bool extract = false;
  bool insert = false;
  for (::dataflow::OperationSchemaId schema : enabledSchemas) {
    bool *selected = nullptr;
    if (schema == ::dataflow::OperationSchemaId::VectorExtract)
      selected = &extract;
    else if (schema == ::dataflow::OperationSchemaId::VectorInsert)
      selected = &insert;
    else
      return reject("vector slice capability enables a foreign schema");
    if (*selected)
      return reject("vector slice capability enables a schema twice");
    *selected = true;
  }
  if (!extract && !insert)
    return reject("vector slice capability has no enabled schema");

  FixedVectorSliceAlignMergeConfigurationLayout layout;
  std::uint32_t cursor = 0;
  layout.encodesMode = extract && insert;
  layout.modeBitOffset = cursor;
  if (layout.encodesMode)
    if (llvm::Error error = appendField(cursor, 1))
      return std::move(error);
  layout.staticOffsetBitOffset = cursor;
  layout.offsetBitCount = bitsForCardinality(params.maxContainerPayloadBits);
  if (llvm::Error error = appendField(cursor, layout.offsetBitCount))
    return std::move(error);
  layout.sliceWidthBitOffset = cursor;
  layout.sliceWidthBitCount = bitsForCardinality(params.maxSlicePayloadBits);
  if (llvm::Error error = appendField(cursor, layout.sliceWidthBitCount))
    return std::move(error);
  layout.dynamicStrideBitOffset = cursor;
  layout.dynamicStrideCount = params.maxDynamicPositionRank;
  layout.dynamicStrideBitCount =
      bitsForInclusiveMaximum(params.maxContainerPayloadBits);
  auto strideFieldBits = llvm::checkedMulUnsigned(
      layout.dynamicStrideBitCount, params.maxDynamicPositionRank);
  if (!strideFieldBits)
    return reject("vector slice stride layout exceeds uint32");
  if (llvm::Error error = appendField(cursor, *strideFieldBits))
    return std::move(error);
  layout.encodedBitCount = cursor;
  return layout;
}

llvm::Expected<fabric::FixedVectorShuffleConfigurationLayout>
fabric::resolveFixedVectorShuffleConfigurationLayout(
    const FixedVectorShuffleParams &params) {
  if (!params.integerElementWidths.valid() ||
      !params.floatElementFormats.valid() ||
      (params.integerElementWidths.empty() &&
       params.floatElementFormats.empty()))
    return reject("vector shuffle element domain is invalid");
  if (params.maxOperandPayloadBits == 0 || params.maxResultPayloadBits == 0 ||
      params.maxBlockPayloadBits == 0 ||
      params.maxBlockPayloadBits > params.maxOperandPayloadBits ||
      params.maxBlockPayloadBits > params.maxResultPayloadBits ||
      params.maxSourceBlocks < 2 || params.maxResultBlocks == 0)
    return reject("vector shuffle capacities are invalid");

  FixedVectorShuffleConfigurationLayout layout;
  std::uint32_t cursor = 0;
  layout.blockWidthBitOffset = cursor;
  layout.blockWidthBitCount = bitsForCardinality(params.maxBlockPayloadBits);
  if (llvm::Error error = appendField(cursor, layout.blockWidthBitCount))
    return std::move(error);
  layout.leftBlockCountBitOffset = cursor;
  layout.blockCountBitCount = bitsForCardinality(params.maxSourceBlocks);
  if (llvm::Error error = appendField(cursor, layout.blockCountBitCount))
    return std::move(error);
  layout.selectorBitOffset = cursor;
  layout.selectorBitCount = bitsForInclusiveMaximum(params.maxSourceBlocks);
  layout.selectorCount = params.maxResultBlocks;
  layout.poisonSelector = params.maxSourceBlocks;
  auto selectorFieldBits =
      llvm::checkedMulUnsigned(layout.selectorBitCount, layout.selectorCount);
  if (!selectorFieldBits)
    return reject("vector shuffle selector layout exceeds uint32");
  if (llvm::Error error = appendField(cursor, *selectorFieldBits))
    return std::move(error);
  layout.encodedBitCount = cursor;
  return layout;
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
  for (::dataflow::OperationSchemaId schema : enabledSchemas)
    if (!llvm::is_contained(descriptor.admittedSchemas, schema))
      return reject("concrete operation capability escapes its generated "
                    "implementation family");

  if (descriptor.typedAdmissionProvider ==
      TypedAdmissionProviderId::ScalarIntegerCastAdmission) {
    auto cases = enumerateScalarIntegerCastCases(
        std::get<ScalarIntegerCastParams>(params), enabledSchemas);
    if (!cases)
      return cases.takeError();
    return uniqueScalarIntegerCastBehaviors(*cases).size() > 1;
  }

  if (descriptor.typedAdmissionProvider ==
      TypedAdmissionProviderId::FixedVectorSliceAlignMergeAdmission) {
    const auto &typed = std::get<FixedVectorSliceAlignMergeParams>(params);
    if (physicalInputCount < 2 + typed.maxDynamicPositionRank ||
        physicalResultCount < 1)
      return reject("vector slice physical role inventory is incomplete");
    auto layout = resolveFixedVectorSliceAlignMergeConfigurationLayout(
        typed, enabledSchemas);
    if (!layout)
      return layout.takeError();
    return layout->encodedBitCount != 0;
  }
  if (descriptor.typedAdmissionProvider ==
      TypedAdmissionProviderId::FixedVectorShuffleAdmission) {
    if (physicalInputCount < 2 || physicalResultCount < 1)
      return reject("vector shuffle physical role inventory is incomplete");
    auto layout = resolveFixedVectorShuffleConfigurationLayout(
        std::get<FixedVectorShuffleParams>(params));
    if (!layout)
      return layout.takeError();
    return layout->encodedBitCount != 0;
  }

  const bool logicFamily =
      descriptor.typedAdmissionProvider ==
          TypedAdmissionProviderId::ScalarLogicIntegerAdmission ||
      descriptor.typedAdmissionProvider ==
          TypedAdmissionProviderId::FixedVectorLogicIntegerAdmission;
  if (logicFamily && enabledSchemas.size() > 1) {
    std::array<bool, 3> selectedOperations{};
    for (::dataflow::OperationSchemaId schema : enabledSchemas) {
      auto operation = integerLogicOperation(schema);
      if (!operation)
        return operation.takeError();
      selectedOperations[static_cast<std::size_t>(*operation)] = true;
    }
    return llvm::count(selectedOperations, true) > 1;
  }
  if (enabledSchemas.size() > 1)
    return true;

  const auto floatBehaviorVaries = [](const FloatBehaviorProfile &behavior) {
    return behavior.roundingModes.size() > 1 ||
           behavior.nanBehaviors.size() > 1 ||
           behavior.subnormalBehaviors.size() > 1 ||
           behavior.signedZeroBehaviors.size() > 1;
  };
  const auto hasSchema = [&](::dataflow::OperationSchemaId schema) {
    return llvm::is_contained(enabledSchemas, schema);
  };
  const auto hasSignedIntegerBehavior = [&] {
    using ::dataflow::OperationSchemaId;
    if (hasSchema(OperationSchemaId::ArithShRSI) ||
        hasSchema(OperationSchemaId::ArithDivSI) ||
        hasSchema(OperationSchemaId::ArithRemSI) ||
        hasSchema(OperationSchemaId::ArithMinSI) ||
        hasSchema(OperationSchemaId::ArithMaxSI))
      return true;
    if (!hasSchema(OperationSchemaId::ArithCmpI))
      return false;
    const auto &compare = std::get<ScalarIntegerCompareMinMaxParams>(params);
    return llvm::any_of(kSignedIntegerPredicates, [&](auto predicate) {
      return compare.predicates.contains(predicate);
    });
  };

  using ::dataflow::OperationSchemaId;
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
    return (hasSchema(OperationSchemaId::ArithCmpI) &&
            typed.predicates.size() > 1) ||
           (typed.operandWidths.size() > 1 && hasSignedIntegerBehavior());
  }
  case TypedAdmissionProviderId::ScalarIntegerCastAdmission:
    llvm_unreachable("scalar integer cast handled before generic selection");
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
    for (IntegerWidth width : integerWidthDomain)
      if (typed.integerElementWidths.contains(width))
        addWidth(getBitWidth(width));
    for (FloatFormat format : floatFormatDomain)
      if (typed.floatElementFormats.contains(format))
        addWidth(getBitWidth(format));
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
  case TypedAdmissionProviderId::FixedVectorSliceAlignMergeAdmission:
  case TypedAdmissionProviderId::FixedVectorShuffleAdmission:
    llvm_unreachable("structural vector families handled before selection");
  }
  return reject("typed admission provider is not registered");
}

llvm::Expected<loom::CanonicalSemanticBytes>
fabric::encodeImplementationFamilySemanticConfiguration(
    ImplementationFamilyId family, const FamilyCapabilityParams &params,
    llvm::ArrayRef<::dataflow::OperationSchemaId> enabledSchemas,
    std::uint32_t physicalInputCount, std::uint32_t physicalResultCount,
    const ::dataflow::CanonicalActorSchemaProjection &actor,
    std::optional<ResolvedIndexWidth> resolvedIndexWidth) {
  auto needsConfiguration = requiresSemanticConfigurationField(
      family, params, enabledSchemas, physicalInputCount, physicalResultCount);
  if (!needsConfiguration)
    return needsConfiguration.takeError();
  if (!*needsConfiguration)
    return reject("capability has no semantic configuration field");
  if (!llvm::is_contained(enabledSchemas, actor.schema))
    return reject("actor schema is not enabled by the concrete capability");

  if (family == ImplementationFamilyId::FixedVectorSliceAlignMerge) {
    const auto *parameters =
        std::get_if<FixedVectorSliceAlignMergeParams>(&params);
    if (!parameters)
      return reject("capability has the wrong parameter schema");
    const auto *position =
        std::get_if<::dataflow::VectorStaticPositionPayload>(&actor.payload);
    if (!position)
      return reject("vector slice actor has the wrong semantic payload");
    const bool hasDynamicPosition =
        llvm::is_contained(position->position, mlir::ShapedType::kDynamic);
    if (hasDynamicPosition) {
      if (!resolvedIndexWidth)
        return reject("vector slice has no resolved dynamic index width");
      if (llvm::Error error = verifyImplementationFamilyAdmission(
              family, &params, actor,
              getResolvedIndexBitWidth(*resolvedIndexWidth)))
        return std::move(error);
    } else if (llvm::Error error = verifyImplementationFamilyAdmission(
                   family, &params, actor)) {
      return std::move(error);
    }
    auto projected = projectFixedVectorSlice(actor);
    if (!projected)
      return projected.takeError();
    auto layout = resolveFixedVectorSliceAlignMergeConfigurationLayout(
        *parameters, enabledSchemas);
    if (!layout)
      return layout.takeError();
    if (layout->encodedBitCount == 0)
      return reject("vector slice capability has no semantic field");
    if (projected->staticOffsetBits >= parameters->maxContainerPayloadBits ||
        projected->sliceWidthBits == 0 ||
        projected->sliceWidthBits > parameters->maxSlicePayloadBits ||
        projected->dynamicStrideBits.size() >
            parameters->maxDynamicPositionRank)
      return reject("vector slice projection exceeds configured capacity");

    std::vector<std::uint8_t> bytes = emptyPackedValue(layout->encodedBitCount);
    if (layout->encodesMode)
      setPackedField(
          bytes, layout->modeBitOffset, 1,
          actor.schema == ::dataflow::OperationSchemaId::VectorInsert ? 1 : 0);
    setPackedField(bytes, layout->staticOffsetBitOffset, layout->offsetBitCount,
                   projected->staticOffsetBits);
    setPackedField(bytes, layout->sliceWidthBitOffset,
                   layout->sliceWidthBitCount, projected->sliceWidthBits - 1);
    for (auto [ordinal, stride] :
         llvm::enumerate(projected->dynamicStrideBits)) {
      if (stride > parameters->maxContainerPayloadBits)
        return reject("vector slice stride exceeds configured capacity");
      setPackedField(bytes,
                     layout->dynamicStrideBitOffset +
                         ordinal * layout->dynamicStrideBitCount,
                     layout->dynamicStrideBitCount, stride);
    }
    return loom::CanonicalSemanticBytes(std::move(bytes));
  }

  if (family == ImplementationFamilyId::FixedVectorShuffle) {
    const auto *parameters = std::get_if<FixedVectorShuffleParams>(&params);
    if (!parameters)
      return reject("capability has the wrong parameter schema");
    if (llvm::Error error =
            verifyImplementationFamilyAdmission(family, &params, actor))
      return std::move(error);
    auto layout = resolveFixedVectorShuffleConfigurationLayout(*parameters);
    if (!layout)
      return layout.takeError();
    auto left = llvm::dyn_cast<mlir::VectorType>(actor.type.getInput(0));
    auto result = llvm::dyn_cast<mlir::VectorType>(actor.type.getResult(0));
    const auto *payload =
        std::get_if<::dataflow::VectorShuffleMaskPayload>(&actor.payload);
    if (!left || !result || !payload)
      return reject("vector shuffle projector received a malformed actor");
    auto resultWidth =
        ::dataflow::semantics::getFlattenedVectorBitWidth(result);
    if (!resultWidth || result.getDimSize(0) <= 0)
      return reject("vector shuffle has no finite block width");
    const std::uint64_t blockWidth =
        *resultWidth / static_cast<std::uint64_t>(result.getDimSize(0));
    std::vector<std::uint8_t> bytes = emptyPackedValue(layout->encodedBitCount);
    setPackedField(bytes, layout->blockWidthBitOffset,
                   layout->blockWidthBitCount, blockWidth - 1);
    setPackedField(bytes, layout->leftBlockCountBitOffset,
                   layout->blockCountBitCount,
                   static_cast<std::uint64_t>(left.getDimSize(0) - 1));
    for (std::uint32_t ordinal = 0; ordinal != layout->selectorCount;
         ++ordinal) {
      const std::uint64_t selector =
          ordinal < payload->mask.size() && payload->mask[ordinal] >= 0
              ? static_cast<std::uint64_t>(payload->mask[ordinal])
              : layout->poisonSelector;
      setPackedField(
          bytes, layout->selectorBitOffset + ordinal * layout->selectorBitCount,
          layout->selectorBitCount, selector);
    }
    return loom::CanonicalSemanticBytes(std::move(bytes));
  }

  if (family == ImplementationFamilyId::ScalarIntegerCast) {
    const auto *parameters = std::get_if<ScalarIntegerCastParams>(&params);
    if (!parameters)
      return reject("capability has the wrong parameter schema");
    const bool isIndexCast =
        actor.schema == ::dataflow::OperationSchemaId::ArithIndexCast ||
        actor.schema == ::dataflow::OperationSchemaId::ArithIndexCastUI;
    if (isIndexCast) {
      if (!resolvedIndexWidth)
        return reject("index cast behavior has no resolved index width");
      if (llvm::Error error = verifyImplementationFamilyAdmission(
              family, &params, actor,
              getResolvedIndexBitWidth(*resolvedIndexWidth)))
        return std::move(error);
    } else if (llvm::Error error = verifyImplementationFamilyAdmission(
                   family, &params, actor)) {
      return std::move(error);
    }
    auto behavior = resolveScalarIntegerCastBehavior(actor, resolvedIndexWidth);
    if (!behavior)
      return behavior.takeError();
    std::vector<std::uint8_t> bytes;
    const llvm::StringRef domain(kScalarIntegerCastConfigurationDomain,
                                 sizeof(kScalarIntegerCastConfigurationDomain) -
                                     1);
    bytes.insert(bytes.end(), domain.bytes_begin(), domain.bytes_end());
    appendU32(bytes, kConfigurationCodecMajor);
    appendU32(bytes, kConfigurationCodecMinor);
    appendU32(bytes, static_cast<std::uint32_t>(behavior->operation));
    appendU32(bytes, getBitWidth(behavior->sourceWidth));
    appendU32(bytes, getBitWidth(behavior->destinationWidth));
    return loom::CanonicalSemanticBytes(std::move(bytes));
  }

  bool selectionOnly = enabledSchemas.size() > 1;
  for (::dataflow::OperationSchemaId enabled : enabledSchemas) {
    auto singletonNeedsConfiguration = requiresSemanticConfigurationField(
        family, params,
        llvm::ArrayRef<::dataflow::OperationSchemaId>(&enabled, 1),
        physicalInputCount, physicalResultCount);
    if (!singletonNeedsConfiguration)
      return singletonNeedsConfiguration.takeError();
    if (*singletonNeedsConfiguration) {
      selectionOnly = false;
      break;
    }
  }
  if (family == ImplementationFamilyId::ScalarIntegerLogic ||
      family == ImplementationFamilyId::FixedVectorIntegerLogic) {
    if (llvm::Error error =
            verifyImplementationFamilyAdmission(family, &params, actor))
      return std::move(error);
    return encodeIntegerLogicConfiguration(actor.schema);
  }

  if (selectionOnly)
    return ::dataflow::encodeOperationSchemaId(actor.schema);

  if (family == ImplementationFamilyId::ScalarFloatFma) {
    const auto *parameters = std::get_if<ScalarFloatParams>(&params);
    if (!parameters)
      return reject("capability has the wrong parameter schema");
    if (!isStrictIEEE(parameters->behavior))
      return reject("scalar FMA semantic codec supports only the strict IEEE "
                    "behavior profile");
    if (actor.schema != ::dataflow::OperationSchemaId::MathFma ||
        actor.type.getNumInputs() != 3 || actor.type.getNumResults() != 1)
      return reject("actor is not a scalar floating FMA");
    mlir::Type type = actor.type.getInput(0);
    if (actor.type.getInput(1) != type || actor.type.getInput(2) != type ||
        actor.type.getResult(0) != type)
      return reject("scalar FMA actor is not uniform");
    if (llvm::Error error =
            verifyImplementationFamilyAdmission(family, &params, actor))
      return std::move(error);
    return ::dataflow::encodeCanonicalType(type);
  }

  if (family == ImplementationFamilyId::FixedVectorIntegerMultiply) {
    const auto *parameters = std::get_if<FixedVectorIntegerParams>(&params);
    if (!parameters)
      return reject("capability has the wrong parameter schema");
    if (actor.schema != ::dataflow::OperationSchemaId::ArithMulI ||
        actor.type.getNumInputs() != 2 || actor.type.getNumResults() != 1)
      return reject("actor is not a fixed-vector integer multiply");
    auto vector = llvm::dyn_cast<mlir::VectorType>(actor.type.getInput(0));
    if (!vector || actor.type.getInput(1) != vector ||
        actor.type.getResult(0) != vector)
      return reject("fixed-vector integer multiply actor is not uniform");
    auto element = llvm::dyn_cast<mlir::IntegerType>(vector.getElementType());
    if (!element)
      return reject("fixed-vector multiply element is not an integer");
    if (llvm::Error error =
            verifyImplementationFamilyAdmission(family, &params, actor))
      return std::move(error);
    const llvm::StringRef domain(
        kFixedVectorIntegerMultiplyConfigurationDomain,
        sizeof(kFixedVectorIntegerMultiplyConfigurationDomain) - 1);
    return encodeElementWidthConfiguration(domain, element.getWidth());
  }

  if (family == ImplementationFamilyId::FixedVectorValueSelect) {
    const auto *parameters = std::get_if<FixedVectorValueSelectParams>(&params);
    if (!parameters)
      return reject("capability has the wrong parameter schema");
    if (actor.schema != ::dataflow::OperationSchemaId::ArithSelect ||
        actor.type.getNumInputs() != 3 || actor.type.getNumResults() != 1)
      return reject("actor is not a fixed-vector value select");
    auto values = llvm::dyn_cast<mlir::VectorType>(actor.type.getInput(1));
    if (!values || actor.type.getInput(2) != values ||
        actor.type.getResult(0) != values)
      return reject("fixed-vector value select actor is not uniform");
    if (llvm::Error error =
            verifyImplementationFamilyAdmission(family, &params, actor))
      return std::move(error);
    const llvm::StringRef domain(
        kFixedVectorValueSelectConfigurationDomain,
        sizeof(kFixedVectorValueSelectConfigurationDomain) - 1);
    return encodeElementWidthConfiguration(domain,
                                           values.getElementTypeBitWidth());
  }

  if (family == ImplementationFamilyId::FixedVectorIntegerAddSub) {
    const auto *parameters = std::get_if<FixedVectorIntegerParams>(&params);
    if (!parameters)
      return reject("capability has the wrong parameter schema");
    using Schema = ::dataflow::OperationSchemaId;
    if (actor.schema != Schema::ArithAddI && actor.schema != Schema::ArithSubI)
      return reject("actor is not a fixed-vector integer add/sub schema");
    if (actor.type.getNumInputs() != 2 || actor.type.getNumResults() != 1)
      return reject("fixed-vector integer add/sub actor has wrong arity");
    auto vector = llvm::dyn_cast<mlir::VectorType>(actor.type.getInput(0));
    if (!vector || actor.type.getInput(1) != vector ||
        actor.type.getResult(0) != vector)
      return reject("fixed-vector integer add/sub actor is not uniform");
    auto element = llvm::dyn_cast<mlir::IntegerType>(vector.getElementType());
    if (!element)
      return reject("fixed-vector add/sub element is not an integer");

    const bool encodeSchema = enabledSchemas.size() > 1;
    const bool encodeElementWidth = parameters->elementWidths.size() > 1;
    constexpr std::uint32_t kSchemaComponent = 1U << 0;
    constexpr std::uint32_t kElementWidthComponent = 1U << 1;
    const std::uint32_t componentMask =
        (encodeSchema ? kSchemaComponent : 0U) |
        (encodeElementWidth ? kElementWidthComponent : 0U);

    std::vector<std::uint8_t> bytes;
    const llvm::StringRef domain(
        kFixedVectorIntegerAddSubConfigurationDomain,
        sizeof(kFixedVectorIntegerAddSubConfigurationDomain) - 1);
    bytes.insert(bytes.end(), domain.bytes_begin(), domain.bytes_end());
    appendU32(bytes, kConfigurationCodecMajor);
    appendU32(bytes, kConfigurationCodecMinor);
    appendU32(bytes, componentMask);
    if (encodeSchema) {
      auto schemaBytes = ::dataflow::encodeOperationSchemaId(actor.schema);
      if (!schemaBytes)
        return schemaBytes.takeError();
      appendFramed(bytes, schemaBytes->bytes());
    }
    if (encodeElementWidth)
      appendU32(bytes, element.getWidth());
    return loom::CanonicalSemanticBytes(std::move(bytes));
  }

  if (family != ImplementationFamilyId::ScalarIntegerCompareMinMax)
    return reject("semantic field-domain codec is not implemented for the "
                  "capability family");
  const auto *parameters =
      std::get_if<ScalarIntegerCompareMinMaxParams>(&params);
  if (!parameters)
    return reject("capability has the wrong parameter schema");

  std::optional<mlir::arith::CmpIPredicate> predicate;
  bool signedBehavior = false;
  using Schema = ::dataflow::OperationSchemaId;
  switch (actor.schema) {
  case Schema::ArithCmpI: {
    const auto *payload =
        std::get_if<::dataflow::IntegerComparePayload>(&actor.payload);
    if (!payload)
      return reject("integer comparison has no typed predicate");
    predicate = payload->predicate;
    signedBehavior = isSignedIntegerPredicate(*predicate);
    break;
  }
  case Schema::ArithMinSI:
  case Schema::ArithMaxSI:
    signedBehavior = true;
    break;
  case Schema::ArithMinUI:
  case Schema::ArithMaxUI:
    break;
  default:
    return reject("actor is not an integer compare/min/max schema");
  }

  auto schemaBytes = ::dataflow::encodeOperationSchemaId(actor.schema);
  if (!schemaBytes)
    return schemaBytes.takeError();
  std::optional<loom::CanonicalSemanticBytes> predicateBytes;
  if (predicate) {
    auto encoded = ::dataflow::encodeIntegerComparePredicate(*predicate);
    if (!encoded)
      return encoded.takeError();
    predicateBytes.emplace(std::move(*encoded));
  }

  std::optional<std::uint32_t> signedOperandWidth;
  if (signedBehavior && parameters->operandWidths.size() > 1) {
    if (actor.type.getNumInputs() != 2)
      return reject("signed comparison has no binary operand type");
    auto operand = llvm::dyn_cast<mlir::IntegerType>(actor.type.getInput(0));
    if (!operand)
      return reject("signed comparison operand is not an integer");
    signedOperandWidth = operand.getWidth();
  }

  const bool encodeSchema = enabledSchemas.size() > 1;
  const bool encodePredicate =
      actor.schema == Schema::ArithCmpI && parameters->predicates.size() > 1;
  constexpr std::uint32_t kSchemaComponent = 1U << 0;
  constexpr std::uint32_t kPredicateComponent = 1U << 1;
  constexpr std::uint32_t kSignedWidthComponent = 1U << 2;
  const std::uint32_t componentMask =
      (encodeSchema ? kSchemaComponent : 0U) |
      (encodePredicate ? kPredicateComponent : 0U) |
      (signedOperandWidth ? kSignedWidthComponent : 0U);

  std::vector<std::uint8_t> bytes;
  const llvm::StringRef domain(
      kScalarIntegerCompareConfigurationDomain,
      sizeof(kScalarIntegerCompareConfigurationDomain) - 1);
  bytes.insert(bytes.end(), domain.bytes_begin(), domain.bytes_end());
  appendU32(bytes, kConfigurationCodecMajor);
  appendU32(bytes, kConfigurationCodecMinor);
  appendU32(bytes, componentMask);
  if (encodeSchema)
    appendFramed(bytes, schemaBytes->bytes());
  if (encodePredicate)
    appendFramed(bytes, predicateBytes->bytes());
  if (signedOperandWidth)
    appendU32(bytes, *signedOperandWidth);
  return loom::CanonicalSemanticBytes(std::move(bytes));
}

llvm::Expected<std::vector<fabric::FiniteImplementationFamilyBehaviorPoint>>
fabric::resolveFiniteImplementationFamilyBehaviorDomain(
    ImplementationFamilyId family, const FamilyCapabilityParams &params,
    llvm::ArrayRef<::dataflow::OperationSchemaId> enabledSchemas,
    std::uint32_t physicalInputCount, std::uint32_t physicalResultCount,
    mlir::MLIRContext &context,
    llvm::function_ref<
        llvm::Error(const ::dataflow::CanonicalActorSchemaProjection &,
                    std::optional<ResolvedIndexWidth>)>
        verifyConcreteActor) {
  auto needsConfiguration = requiresSemanticConfigurationField(
      family, params, enabledSchemas, physicalInputCount, physicalResultCount);
  if (!needsConfiguration)
    return needsConfiguration.takeError();

  struct ResolvedBehaviorActor final {
    ::dataflow::CanonicalActorSchemaProjection actor;
    std::optional<ResolvedIndexWidth> resolvedIndexWidth;
  };
  std::vector<ResolvedBehaviorActor> actors;
  if (family == ImplementationFamilyId::ScalarIntegerLogic) {
    const auto *parameters = std::get_if<ScalarIntegerParams>(&params);
    if (!parameters)
      return reject("capability has the wrong parameter schema");
    for (::dataflow::OperationSchemaId schema : enabledSchemas) {
      for (IntegerWidth width : integerWidthDomain) {
        if (!parameters->integerWidths.contains(width))
          continue;
        auto actor =
            makeScalarIntegerLogicActor(context, getBitWidth(width), schema);
        if (!actor)
          return actor.takeError();
        actors.push_back({std::move(*actor), std::nullopt});
      }
    }
  } else if (family == ImplementationFamilyId::FixedVectorIntegerLogic) {
    const auto *parameters = std::get_if<FixedVectorIntegerParams>(&params);
    if (!parameters)
      return reject("capability has the wrong parameter schema");
    for (::dataflow::OperationSchemaId schema : enabledSchemas) {
      for (IntegerWidth width : integerWidthDomain) {
        if (!parameters->elementWidths.contains(width))
          continue;
        auto payload = makeIntegerLogicPayload(schema);
        if (!payload)
          return payload.takeError();
        auto actor = makeFixedVectorIntegerActor(context, getBitWidth(width),
                                                 parameters->maxPayloadBits,
                                                 schema, std::move(*payload));
        if (!actor)
          return actor.takeError();
        actors.push_back({std::move(*actor), std::nullopt});
      }
    }
  } else if (family == ImplementationFamilyId::FixedVectorIntegerMultiply) {
    const auto *parameters = std::get_if<FixedVectorIntegerParams>(&params);
    if (!parameters)
      return reject("capability has the wrong parameter schema");
    if (enabledSchemas.size() != 1 ||
        enabledSchemas.front() != ::dataflow::OperationSchemaId::ArithMulI)
      return reject("fixed-vector multiply capability contains a non-multiply "
                    "schema");
    for (IntegerWidth width : integerWidthDomain) {
      if (!parameters->elementWidths.contains(width))
        continue;
      auto actor = makeFixedVectorIntegerActor(
          context, getBitWidth(width), parameters->maxPayloadBits,
          enabledSchemas.front(), ::dataflow::IntegerOverflowPayload{});
      if (!actor)
        return actor.takeError();
      actors.push_back({std::move(*actor), std::nullopt});
    }
  } else if (family == ImplementationFamilyId::FixedVectorValueSelect) {
    const auto *parameters = std::get_if<FixedVectorValueSelectParams>(&params);
    if (!parameters)
      return reject("capability has the wrong parameter schema");
    if (enabledSchemas.size() != 1 ||
        enabledSchemas.front() != ::dataflow::OperationSchemaId::ArithSelect)
      return reject("fixed-vector select capability contains a non-select "
                    "schema");
    for (IntegerWidth width : integerWidthDomain) {
      if (!parameters->integerElementWidths.contains(width))
        continue;
      auto actor = makeFixedVectorValueSelectActor(
          context, mlir::IntegerType::get(&context, getBitWidth(width)),
          parameters->maxPayloadBits);
      if (!actor)
        return actor.takeError();
      actors.push_back({std::move(*actor), std::nullopt});
    }
    for (FloatFormat format : floatFormatDomain) {
      if (!parameters->floatElementFormats.contains(format))
        continue;
      auto actor = makeFixedVectorValueSelectActor(
          context, floatType(context, format), parameters->maxPayloadBits);
      if (!actor)
        return actor.takeError();
      actors.push_back({std::move(*actor), std::nullopt});
    }
  } else if (family == ImplementationFamilyId::ScalarIntegerCast) {
    const auto *parameters = std::get_if<ScalarIntegerCastParams>(&params);
    if (!parameters)
      return reject("capability has the wrong parameter schema");
    auto cases = enumerateScalarIntegerCastCases(*parameters, enabledSchemas);
    if (!cases)
      return cases.takeError();
    actors.reserve(cases->size());
    for (const ScalarIntegerCastCase &castCase : *cases) {
      auto actor = makeScalarIntegerCastActor(context, castCase);
      if (!actor)
        return actor.takeError();
      actors.push_back({std::move(*actor), castCase.resolvedIndexWidth});
    }
  } else if (family == ImplementationFamilyId::ScalarIntegerCompareMinMax) {
    const auto *parameters =
        std::get_if<ScalarIntegerCompareMinMaxParams>(&params);
    if (!parameters)
      return reject("capability has the wrong parameter schema");
    const auto appendWidthDomain =
        [&](::dataflow::OperationSchemaId schema,
            std::optional<mlir::arith::CmpIPredicate> predicate =
                std::nullopt) {
          for (IntegerWidth width : integerWidthDomain) {
            if (!parameters->operandWidths.contains(width))
              continue;
            actors.push_back(
                {makeScalarIntegerCompareActor(context, getBitWidth(width),
                                               schema, predicate),
                 std::nullopt});
          }
        };
    for (::dataflow::OperationSchemaId schema : enabledSchemas) {
      switch (schema) {
      case ::dataflow::OperationSchemaId::ArithCmpI:
        for (std::uint32_t ordinal = 0;
             ordinal <= mlir::arith::getMaxEnumValForCmpIPredicate();
             ++ordinal) {
          const auto predicate =
              static_cast<mlir::arith::CmpIPredicate>(ordinal);
          if (parameters->predicates.contains(predicate))
            appendWidthDomain(schema, predicate);
        }
        break;
      case ::dataflow::OperationSchemaId::ArithMinSI:
      case ::dataflow::OperationSchemaId::ArithMaxSI:
      case ::dataflow::OperationSchemaId::ArithMinUI:
      case ::dataflow::OperationSchemaId::ArithMaxUI:
        appendWidthDomain(schema);
        break;
      default:
        return reject("capability contains a non-compare/min-max schema");
      }
    }
  } else if (family == ImplementationFamilyId::FixedVectorIntegerAddSub) {
    const auto *parameters = std::get_if<FixedVectorIntegerParams>(&params);
    if (!parameters)
      return reject("capability has the wrong parameter schema");
    for (::dataflow::OperationSchemaId schema : enabledSchemas) {
      if (schema != ::dataflow::OperationSchemaId::ArithAddI &&
          schema != ::dataflow::OperationSchemaId::ArithSubI)
        return reject("capability contains a non-add/sub schema");
      for (IntegerWidth width : integerWidthDomain) {
        if (!parameters->elementWidths.contains(width))
          continue;
        auto actor = makeFixedVectorIntegerAddSubActor(
            context, getBitWidth(width), parameters->maxPayloadBits, schema);
        if (!actor)
          return actor.takeError();
        actors.push_back({std::move(*actor), std::nullopt});
      }
    }
  } else if (family == ImplementationFamilyId::ScalarFloatFma) {
    const auto *parameters = std::get_if<ScalarFloatParams>(&params);
    if (!parameters)
      return reject("capability has the wrong parameter schema");
    if (enabledSchemas.size() != 1 ||
        enabledSchemas.front() != ::dataflow::OperationSchemaId::MathFma)
      return reject("scalar FMA capability contains a non-FMA schema");
    if (!isStrictIEEE(parameters->behavior))
      return reject("scalar FMA behavior-domain projection supports only the "
                    "strict IEEE behavior profile");
    for (FloatFormat format : floatFormatDomain)
      if (parameters->formats.contains(format))
        actors.push_back(
            {makeScalarFloatFmaActor(context, format), std::nullopt});
  } else if (family == ImplementationFamilyId::FixedVectorSliceAlignMerge ||
             family == ImplementationFamilyId::FixedVectorShuffle) {
    return reject("direct structural vector configuration has no finite "
                  "behavior-domain enumeration");
  } else {
    return reject("finite behavior-domain projection is not implemented for "
                  "the capability family");
  }
  if (actors.empty())
    return reject("capability has no admitted finite behavior");

  std::vector<FiniteImplementationFamilyBehaviorPoint> points;
  for (ResolvedBehaviorActor &resolvedActor : actors) {
    auto &actor = resolvedActor.actor;
    if (resolvedActor.resolvedIndexWidth) {
      if (llvm::Error error = verifyImplementationFamilyAdmission(
              family, &params, actor,
              getResolvedIndexBitWidth(*resolvedActor.resolvedIndexWidth)))
        return std::move(error);
    } else if (llvm::Error error = verifyImplementationFamilyAdmission(
                   family, &params, actor)) {
      return std::move(error);
    }
    if (llvm::Error error =
            verifyConcreteActor(actor, resolvedActor.resolvedIndexWidth))
      return std::move(error);
    if (!*needsConfiguration) {
      if (points.empty())
        points.push_back(
            {std::move(actor), std::nullopt, resolvedActor.resolvedIndexWidth});
      continue;
    }
    auto semantic = encodeImplementationFamilySemanticConfiguration(
        family, params, enabledSchemas, physicalInputCount, physicalResultCount,
        actor, resolvedActor.resolvedIndexWidth);
    if (!semantic)
      return semantic.takeError();
    const auto duplicate = llvm::find_if(points, [&](const auto &point) {
      return point.semanticConfiguration &&
             point.semanticConfiguration->bytes().equals(semantic->bytes());
    });
    if (duplicate == points.end())
      points.push_back({std::move(actor), std::move(*semantic),
                        resolvedActor.resolvedIndexWidth});
  }
  llvm::sort(points, [](const auto &lhs, const auto &rhs) {
    if (!lhs.semanticConfiguration)
      return rhs.semanticConfiguration.has_value();
    if (!rhs.semanticConfiguration)
      return false;
    return std::lexicographical_compare(
        lhs.semanticConfiguration->bytes().begin(),
        lhs.semanticConfiguration->bytes().end(),
        rhs.semanticConfiguration->bytes().begin(),
        rhs.semanticConfiguration->bytes().end());
  });
  return points;
}

mlir::FailureOr<unsigned> fabric::getSemanticPayloadWidth(mlir::Type type,
                                                          std::string &error) {
  return semanticPayloadWidth(type, std::nullopt, nullptr, error);
}

mlir::FailureOr<unsigned>
fabric::getSemanticPayloadWidth(mlir::Type type,
                                const ::loom::PointerLayout *pointerLayout,
                                std::string &error) {
  return semanticPayloadWidth(type, std::nullopt, pointerLayout, error);
}

mlir::FailureOr<unsigned>
fabric::getSemanticPayloadWidth(mlir::Type type, unsigned indexBitWidth,
                                const ::loom::PointerLayout *pointerLayout,
                                std::string &error) {
  return semanticPayloadWidth(type, indexBitWidth, pointerLayout, error);
}
