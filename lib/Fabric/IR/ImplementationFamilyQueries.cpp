//===- ImplementationFamilyQueries.cpp - HSG capability queries ----------===//
//
// Implements read-only projections over the generated implementation-family
// registry and concrete capability parameters.
//
//===----------------------------------------------------------------------===//

#include "Fabric/IR/ImplementationFamily.h"
#include "ImplementationFamilyBehaviorInternal.h"
#include "ImplementationFamilySpecialMath.h"

#include "Common/IndexWidth.h"
#include "Common/VectorWidth.h"
#include "Dataflow/IR/DataflowActorSemantics.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
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

constexpr std::array<mlir::arith::CmpIPredicate, 4> kSignedIntegerPredicates = {
    mlir::arith::CmpIPredicate::slt, mlir::arith::CmpIPredicate::sle,
    mlir::arith::CmpIPredicate::sgt, mlir::arith::CmpIPredicate::sge};

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

namespace fabric::detail {

llvm::Expected<bool> semanticConfigurationRequiresField(
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
  case TypedAdmissionProviderId::ScalarSpecialMathAdmission: {
    const auto &typed = std::get<ScalarSpecialMathParams>(params);
    if (llvm::Error error =
            validateScalarSpecialMathBehaviorProfile(typed.behavior))
      return std::move(error);
    return typed.formats.size() > 1;
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
    return typed.formatPairs.size() > 1;
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
    std::optional<unsigned> firstReachableWidth;
    const auto addWidth = [&](unsigned width) {
      if (typed.maxPayloadBits < width)
        return false;
      if (typed.maxPayloadBits / width > 1)
        return true;
      if (firstReachableWidth && *firstReachableWidth != width)
        return true;
      firstReachableWidth = width;
      return false;
    };
    for (IntegerWidth width : integerWidthDomain)
      if (typed.integerElementWidths.contains(width) &&
          addWidth(getBitWidth(width)))
        return true;
    for (FloatFormat format : floatFormatDomain)
      if (typed.floatElementFormats.contains(format) &&
          addWidth(getBitWidth(format)))
        return true;
    return false;
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

} // namespace fabric::detail

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
