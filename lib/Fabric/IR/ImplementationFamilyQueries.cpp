//===- ImplementationFamilyQueries.cpp - HSG capability queries ----------===//
//
// Implements read-only projections over the generated implementation-family
// registry and concrete capability parameters.
//
//===----------------------------------------------------------------------===//

#include "Fabric/IR/ImplementationFamily.h"

#include "Common/IndexWidth.h"
#include "Common/VectorWidth.h"
#include "Dataflow/IR/OperationSchemaCodec.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
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

bool isSignedIntegerPredicate(mlir::arith::CmpIPredicate predicate) {
  return llvm::is_contained(kSignedIntegerPredicates, predicate);
}

enum class IntegerLogicOperation : std::uint32_t { And, Or, Xor };

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
  }
  return reject("typed admission provider is not registered");
}

llvm::Expected<loom::CanonicalSemanticBytes>
fabric::encodeImplementationFamilySemanticConfiguration(
    ImplementationFamilyId family, const FamilyCapabilityParams &params,
    llvm::ArrayRef<::dataflow::OperationSchemaId> enabledSchemas,
    std::uint32_t physicalInputCount, std::uint32_t physicalResultCount,
    const ::dataflow::CanonicalActorSchemaProjection &actor) {
  auto needsConfiguration = requiresSemanticConfigurationField(
      family, params, enabledSchemas, physicalInputCount, physicalResultCount);
  if (!needsConfiguration)
    return needsConfiguration.takeError();
  if (!*needsConfiguration)
    return reject("capability has no semantic configuration field");
  if (!llvm::is_contained(enabledSchemas, actor.schema))
    return reject("actor schema is not enabled by the concrete capability");

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
        llvm::Error(const ::dataflow::CanonicalActorSchemaProjection &)>
        verifyConcreteActor) {
  auto needsConfiguration = requiresSemanticConfigurationField(
      family, params, enabledSchemas, physicalInputCount, physicalResultCount);
  if (!needsConfiguration)
    return needsConfiguration.takeError();

  std::vector<::dataflow::CanonicalActorSchemaProjection> actors;
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
        actors.push_back(std::move(*actor));
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
        actors.push_back(std::move(*actor));
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
      actors.push_back(std::move(*actor));
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
      actors.push_back(std::move(*actor));
    }
    for (FloatFormat format : floatFormatDomain) {
      if (!parameters->floatElementFormats.contains(format))
        continue;
      auto actor = makeFixedVectorValueSelectActor(
          context, floatType(context, format), parameters->maxPayloadBits);
      if (!actor)
        return actor.takeError();
      actors.push_back(std::move(*actor));
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
            actors.push_back(makeScalarIntegerCompareActor(
                context, getBitWidth(width), schema, predicate));
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
        actors.push_back(std::move(*actor));
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
        actors.push_back(makeScalarFloatFmaActor(context, format));
  } else {
    return reject("finite behavior-domain projection is not implemented for "
                  "the capability family");
  }
  if (actors.empty())
    return reject("capability has no admitted finite behavior");

  std::vector<FiniteImplementationFamilyBehaviorPoint> points;
  for (auto &actor : actors) {
    if (llvm::Error error =
            verifyImplementationFamilyAdmission(family, &params, actor))
      return std::move(error);
    if (llvm::Error error = verifyConcreteActor(actor))
      return std::move(error);
    if (!*needsConfiguration) {
      if (points.empty())
        points.push_back({std::move(actor), std::nullopt});
      continue;
    }
    auto semantic = encodeImplementationFamilySemanticConfiguration(
        family, params, enabledSchemas, physicalInputCount, physicalResultCount,
        actor);
    if (!semantic)
      return semantic.takeError();
    const auto duplicate = llvm::find_if(points, [&](const auto &point) {
      return point.semanticConfiguration &&
             point.semanticConfiguration->bytes().equals(semantic->bytes());
    });
    if (duplicate == points.end())
      points.push_back({std::move(actor), std::move(*semantic)});
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
