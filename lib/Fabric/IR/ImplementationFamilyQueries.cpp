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
    constexpr FloatFormat floatFormats[] = {FloatFormat::F16, FloatFormat::BF16,
                                            FloatFormat::F32, FloatFormat::F64};
    for (FloatFormat format : floatFormats)
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
  if (selectionOnly)
    return ::dataflow::encodeOperationSchemaId(actor.schema);

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
  if (family != ImplementationFamilyId::ScalarIntegerCompareMinMax)
    return reject("finite behavior-domain projection is not implemented for "
                  "the capability family");
  const auto *parameters =
      std::get_if<ScalarIntegerCompareMinMaxParams>(&params);
  if (!parameters)
    return reject("capability has the wrong parameter schema");

  std::vector<::dataflow::CanonicalActorSchemaProjection> actors;
  const auto appendWidthDomain =
      [&](::dataflow::OperationSchemaId schema,
          std::optional<mlir::arith::CmpIPredicate> predicate = std::nullopt) {
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
           ordinal <= mlir::arith::getMaxEnumValForCmpIPredicate(); ++ordinal) {
        const auto predicate = static_cast<mlir::arith::CmpIPredicate>(ordinal);
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
  if (actors.empty())
    return reject("capability has no admitted compare/min-max behavior");

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
