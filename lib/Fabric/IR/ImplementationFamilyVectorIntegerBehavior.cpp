//===- ImplementationFamilyVectorIntegerBehavior.cpp ---------------------===//
//
// Owns the finite behavior quotient of fixed-vector integer resources.
//
//===----------------------------------------------------------------------===//

#include "ImplementationFamilyVectorIntegerBehavior.h"

#include "ImplementationFamilyBehaviorInternal.h"

#include "Dataflow/IR/OperationSchemaCodec.h"

#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <numeric>
#include <optional>
#include <utility>
#include <vector>

namespace {

using namespace fabric;
using ::dataflow::OperationSchemaId;

struct BehaviorFact final {
  ::dataflow::CanonicalActorSchemaProjection actor;
  llvm::StringRef role;
  std::optional<::loom::CanonicalSemanticBytes> predicate;
  std::uint32_t elementWidth = 0;
  bool observesElementWidth = false;
  std::vector<std::uint64_t> operandPorts;
  std::vector<std::uint64_t> resultPorts;
};

llvm::Error reject(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(), message);
}

bool equalBytes(const std::optional<::loom::CanonicalSemanticBytes> &lhs,
                const std::optional<::loom::CanonicalSemanticBytes> &rhs) {
  if (lhs.has_value() != rhs.has_value())
    return false;
  return !lhs || lhs->bytes().equals(rhs->bytes());
}

bool sameBehavior(const BehaviorFact &lhs, const BehaviorFact &rhs) {
  return lhs.role == rhs.role && equalBytes(lhs.predicate, rhs.predicate) &&
         lhs.observesElementWidth == rhs.observesElementWidth &&
         (!lhs.observesElementWidth || lhs.elementWidth == rhs.elementWidth);
}

bool sameKey(const FiniteImplementationFamilyBehaviorPoint &lhs,
             const FiniteImplementationFamilyBehaviorPoint &rhs) {
  return lhs.semanticConfiguration && rhs.semanticConfiguration &&
         lhs.semanticConfiguration->bytes().equals(
             rhs.semanticConfiguration->bytes());
}

bool keyLess(const FiniteImplementationFamilyBehaviorPoint &lhs,
             const FiniteImplementationFamilyBehaviorPoint &rhs) {
  return std::lexicographical_compare(
      lhs.semanticConfiguration->bytes().begin(),
      lhs.semanticConfiguration->bytes().end(),
      rhs.semanticConfiguration->bytes().begin(),
      rhs.semanticConfiguration->bytes().end());
}

llvm::Expected<mlir::FloatType> floatType(mlir::MLIRContext &context,
                                          FloatFormat format) {
  switch (format) {
  case FloatFormat::F16:
    return mlir::Float16Type::get(&context);
  case FloatFormat::BF16:
    return mlir::BFloat16Type::get(&context);
  case FloatFormat::F32:
    return mlir::Float32Type::get(&context);
  case FloatFormat::F64:
    return mlir::Float64Type::get(&context);
  }
  return reject("fixed-vector select contains an unknown floating format");
}

llvm::Expected<::dataflow::SemanticPayload>
payloadFor(OperationSchemaId schema,
           std::optional<mlir::arith::CmpIPredicate> predicate) {
  using Case = ::dataflow::OperationSemanticsCase;
  switch (::dataflow::semanticsCase(schema)) {
  case Case::NoSemanticPayload:
    return ::dataflow::NoPayload{};
  case Case::ArithIntegerOverflow:
    return ::dataflow::IntegerOverflowPayload{};
  case Case::ArithExact:
    return ::dataflow::ExactPayload{};
  case Case::ArithIntegerCompare:
    if (!predicate)
      return reject("fixed-vector comparison has no predicate");
    return ::dataflow::IntegerComparePayload{*predicate};
  case Case::LLVMZeroPoison:
    return ::dataflow::ZeroPoisonPayload{true};
  case Case::LLVMDisjoint:
    return ::dataflow::DisjointPayload{true};
  default:
    return reject("fixed-vector integer schema has an unsupported payload");
  }
}

llvm::Expected<llvm::StringRef> roleFor(OperationSchemaId schema) {
  switch (schema) {
  case OperationSchemaId::ArithAddI:
    return "Add";
  case OperationSchemaId::ArithSubI:
    return "Sub";
  case OperationSchemaId::ArithAndI:
    return "And";
  case OperationSchemaId::ArithOrI:
  case OperationSchemaId::LLVMOrDisjoint:
    return "Or";
  case OperationSchemaId::ArithXOrI:
    return "Xor";
  case OperationSchemaId::ArithShLI:
    return "Left";
  case OperationSchemaId::ArithShRUI:
    return "LogicalRight";
  case OperationSchemaId::ArithShRSI:
    return "ArithmeticRight";
  case OperationSchemaId::ArithCmpI:
    return "Compare";
  case OperationSchemaId::ArithMinSI:
    return "SignedMin";
  case OperationSchemaId::ArithMaxSI:
    return "SignedMax";
  case OperationSchemaId::ArithMinUI:
    return "UnsignedMin";
  case OperationSchemaId::ArithMaxUI:
    return "UnsignedMax";
  case OperationSchemaId::ArithSelect:
  case OperationSchemaId::ArithMulI:
    return "";
  case OperationSchemaId::LLVMSAddSat:
    return "SignedAdd";
  case OperationSchemaId::LLVMUAddSat:
    return "UnsignedAdd";
  case OperationSchemaId::LLVMSSubSat:
    return "SignedSub";
  case OperationSchemaId::LLVMUSubSat:
    return "UnsignedSub";
  case OperationSchemaId::MathCountLeadingZeros:
  case OperationSchemaId::LLVMCountLeadingZeros:
    return "Leading";
  case OperationSchemaId::MathCountTrailingZeros:
  case OperationSchemaId::LLVMCountTrailingZeros:
    return "Trailing";
  default:
    return reject("fixed-vector integer schema has no canonical role");
  }
}

bool observesElementWidth(ImplementationFamilyId family) {
  return family != ImplementationFamilyId::FixedVectorIntegerLogic;
}

bool isUnary(OperationSchemaId schema) {
  return schema == OperationSchemaId::MathCountLeadingZeros ||
         schema == OperationSchemaId::LLVMCountLeadingZeros ||
         schema == OperationSchemaId::MathCountTrailingZeros ||
         schema == OperationSchemaId::LLVMCountTrailingZeros;
}

bool isComparison(OperationSchemaId schema) {
  return schema == OperationSchemaId::ArithCmpI;
}

bool isSelect(OperationSchemaId schema) {
  return schema == OperationSchemaId::ArithSelect;
}

std::uint32_t payloadCapacity(const FamilyCapabilityParams &params) {
  if (const auto *integers = std::get_if<FixedVectorIntegerParams>(&params))
    return integers->maxPayloadBits;
  if (const auto *compare =
          std::get_if<FixedVectorIntegerCompareMinMaxParams>(&params))
    return compare->maxPayloadBits;
  if (const auto *select = std::get_if<FixedVectorValueSelectParams>(&params))
    return select->maxPayloadBits;
  return 0;
}

llvm::Expected<std::uint64_t>
reachableLaneCount(OperationSchemaId schema, std::uint32_t elementWidth,
                   std::uint32_t maxPayloadBits,
                   llvm::ArrayRef<std::uint32_t> physicalInputWidths,
                   llvm::ArrayRef<std::uint32_t> physicalResultWidths) {
  const std::size_t inputCount = isUnary(schema) ? 1 : isSelect(schema) ? 3 : 2;
  if (physicalInputWidths.size() < inputCount || physicalResultWidths.empty())
    return reject("fixed-vector integer physical role inventory is incomplete");

  std::uint64_t lanes = std::numeric_limits<std::uint64_t>::max();
  const auto admitEndpoint = [&](std::uint32_t physicalWidth,
                                 std::uint32_t laneWidth) {
    lanes = std::min<std::uint64_t>(lanes, physicalWidth / laneWidth);
    lanes = std::min<std::uint64_t>(lanes, maxPayloadBits / laneWidth);
  };
  if (isSelect(schema)) {
    admitEndpoint(physicalInputWidths[0], 1);
    admitEndpoint(physicalInputWidths[1], elementWidth);
    admitEndpoint(physicalInputWidths[2], elementWidth);
  } else {
    for (std::size_t ordinal = 0; ordinal != inputCount; ++ordinal)
      admitEndpoint(physicalInputWidths[ordinal], elementWidth);
  }
  admitEndpoint(physicalResultWidths[0],
                isComparison(schema) ? 1 : elementWidth);
  if (lanes >
      static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()))
    return reject("fixed-vector lane count exceeds int64");
  return lanes;
}

llvm::Expected<::dataflow::CanonicalActorSchemaProjection>
makeActor(mlir::MLIRContext &context, OperationSchemaId schema,
          mlir::Type element, std::uint64_t laneCount,
          std::optional<mlir::arith::CmpIPredicate> predicate) {
  if (laneCount == 0)
    return reject("fixed-vector behavior has no reachable lane");
  auto values =
      mlir::VectorType::get({static_cast<std::int64_t>(laneCount)}, element);
  llvm::SmallVector<mlir::Type, 3> inputs;
  mlir::Type result = values;
  if (isUnary(schema)) {
    inputs.push_back(values);
  } else if (isSelect(schema)) {
    inputs.push_back(mlir::VectorType::get(
        values.getShape(), mlir::IntegerType::get(&context, 1)));
    inputs.push_back(values);
    inputs.push_back(values);
  } else {
    inputs.push_back(values);
    inputs.push_back(values);
    if (isComparison(schema))
      result = mlir::VectorType::get(values.getShape(),
                                     mlir::IntegerType::get(&context, 1));
  }
  auto payload = payloadFor(schema, predicate);
  if (!payload)
    return payload.takeError();
  return ::dataflow::CanonicalActorSchemaProjection{
      schema, mlir::FunctionType::get(&context, inputs, {result}),
      std::move(*payload)};
}

llvm::Expected<BehaviorFact>
describeActor(ImplementationFamilyId family,
              const ::dataflow::CanonicalActorSchemaProjection &actor) {
  auto role = roleFor(actor.schema);
  if (!role)
    return role.takeError();

  mlir::Type valueType =
      isSelect(actor.schema) ? actor.type.getInput(1) : actor.type.getInput(0);
  auto vector = llvm::dyn_cast<mlir::VectorType>(valueType);
  if (!vector || vector.isScalable() || vector.getRank() == 0 ||
      vector.getNumElements() == 0)
    return reject("fixed-vector integer behavior actor has invalid shape");
  const std::uint64_t width = vector.getElementTypeBitWidth();
  if (width == 0 || width > std::numeric_limits<std::uint32_t>::max())
    return reject("fixed-vector integer element width exceeds uint32");

  std::optional<::loom::CanonicalSemanticBytes> predicateBytes;
  if (actor.schema == OperationSchemaId::ArithCmpI) {
    const auto *payload =
        std::get_if<::dataflow::IntegerComparePayload>(&actor.payload);
    if (!payload)
      return reject("fixed-vector comparison has no typed predicate");
    auto encoded =
        ::dataflow::encodeIntegerComparePredicate(payload->predicate);
    if (!encoded)
      return encoded.takeError();
    predicateBytes = std::move(*encoded);
  }

  BehaviorFact fact{actor,
                    *role,
                    std::move(predicateBytes),
                    static_cast<std::uint32_t>(width),
                    observesElementWidth(family),
                    {},
                    {}};
  fact.operandPorts.resize(actor.type.getNumInputs());
  fact.resultPorts.resize(actor.type.getNumResults());
  std::iota(fact.operandPorts.begin(), fact.operandPorts.end(), 0);
  std::iota(fact.resultPorts.begin(), fact.resultPorts.end(), 0);
  return fact;
}

llvm::Error appendCandidate(
    std::vector<BehaviorFact> &candidates, ImplementationFamilyId family,
    const FamilyCapabilityParams &params, OperationSchemaId schema,
    mlir::Type element, std::uint32_t maxPayloadBits,
    llvm::ArrayRef<std::uint32_t> physicalInputWidths,
    llvm::ArrayRef<std::uint32_t> physicalResultWidths,
    mlir::MLIRContext &context,
    std::optional<mlir::arith::CmpIPredicate> predicate = std::nullopt) {
  const std::uint32_t width = element.getIntOrFloatBitWidth();
  auto lanes = reachableLaneCount(schema, width, maxPayloadBits,
                                  physicalInputWidths, physicalResultWidths);
  if (!lanes)
    return lanes.takeError();
  if (*lanes == 0)
    return llvm::Error::success();
  auto actor = makeActor(context, schema, element, *lanes, predicate);
  if (!actor)
    return actor.takeError();
  if (llvm::Error error =
          verifyImplementationFamilyAdmission(family, &params, *actor))
    return error;
  auto fact = describeActor(family, *actor);
  if (!fact)
    return fact.takeError();
  candidates.push_back(std::move(*fact));
  return llvm::Error::success();
}

bool predicateVaries(llvm::ArrayRef<BehaviorFact> facts, llvm::StringRef role) {
  const BehaviorFact *first = nullptr;
  for (const BehaviorFact &fact : facts) {
    if (fact.role != role || !fact.predicate)
      continue;
    if (!first)
      first = &fact;
    else if (!equalBytes(first->predicate, fact.predicate))
      return true;
  }
  return false;
}

llvm::Expected<std::vector<FiniteImplementationFamilyBehaviorPoint>>
encodeDomain(ImplementationFamilyId family,
             std::vector<BehaviorFact> candidates) {
  std::vector<BehaviorFact> facts;
  for (BehaviorFact &candidate : candidates) {
    if (llvm::none_of(facts, [&](const BehaviorFact &fact) {
          return sameBehavior(candidate, fact);
        }))
      facts.push_back(std::move(candidate));
  }
  if (facts.empty())
    return reject("fixed-vector integer capability has no reachable behavior");

  if (facts.size() == 1) {
    BehaviorFact &fact = facts.front();
    std::vector<FiniteImplementationFamilyBehaviorPoint> singleton;
    singleton.emplace_back(std::move(fact.actor), std::nullopt, std::nullopt,
                           std::move(fact.operandPorts),
                           std::move(fact.resultPorts));
    return singleton;
  }

  const bool encodeRole = llvm::any_of(facts, [&](const BehaviorFact &fact) {
    return fact.role != facts.front().role;
  });
  const BehaviorFact *firstObservedWidth = nullptr;
  bool encodeWidth = false;
  for (const BehaviorFact &fact : facts) {
    if (!fact.observesElementWidth)
      continue;
    if (!firstObservedWidth)
      firstObservedWidth = &fact;
    else if (fact.elementWidth != firstObservedWidth->elementWidth)
      encodeWidth = true;
  }

  std::vector<FiniteImplementationFamilyBehaviorPoint> points;
  points.reserve(facts.size());
  for (BehaviorFact &fact : facts) {
    llvm::SmallVector<detail::ImplementationFamilyBehaviorKeyComponent, 2>
        components;
    if (fact.predicate && predicateVaries(facts, fact.role))
      components.emplace_back(*fact.predicate);
    if (fact.observesElementWidth && encodeWidth)
      components.emplace_back(fact.elementWidth);
    auto key = detail::encodeImplementationFamilyBehaviorKey(
        family, encodeRole ? fact.role : "", components);
    if (!key)
      return key.takeError();
    points.emplace_back(std::move(fact.actor), std::move(*key), std::nullopt,
                        std::move(fact.operandPorts),
                        std::move(fact.resultPorts));
  }
  llvm::sort(points, keyLess);
  if (std::adjacent_find(points.begin(), points.end(), sameKey) != points.end())
    return reject("fixed-vector integer behavior codec is not injective");
  return points;
}

llvm::Expected<std::vector<mlir::Type>>
selectElementDomain(const FixedVectorValueSelectParams &params,
                    mlir::MLIRContext &context) {
  std::vector<mlir::Type> elements;
  for (IntegerWidth width : integerWidthDomain)
    if (params.integerElementWidths.contains(width))
      elements.push_back(mlir::IntegerType::get(&context, getBitWidth(width)));
  for (FloatFormat format : floatFormatDomain) {
    if (!params.floatElementFormats.contains(format))
      continue;
    auto type = floatType(context, format);
    if (!type)
      return type.takeError();
    elements.push_back(*type);
  }
  return elements;
}

} // namespace

bool fabric::detail::ownsFixedVectorIntegerBehaviorRelation(
    ImplementationFamilyId family) {
  switch (family) {
  case ImplementationFamilyId::FixedVectorIntegerAddSub:
  case ImplementationFamilyId::FixedVectorIntegerLogic:
  case ImplementationFamilyId::FixedVectorIntegerShift:
  case ImplementationFamilyId::FixedVectorIntegerCompareMinMax:
  case ImplementationFamilyId::FixedVectorValueSelect:
  case ImplementationFamilyId::FixedVectorIntegerMultiply:
  case ImplementationFamilyId::FixedVectorIntegerSaturatingAddSub:
  case ImplementationFamilyId::FixedVectorIntegerCountZeros:
    return true;
  default:
    return false;
  }
}

llvm::Expected<std::vector<fabric::FiniteImplementationFamilyBehaviorPoint>>
fabric::detail::resolveFixedVectorIntegerBehaviorDomain(
    ImplementationFamilyId family, const FamilyCapabilityParams &params,
    llvm::ArrayRef<OperationSchemaId> enabledSchemas,
    llvm::ArrayRef<std::uint32_t> physicalInputWidths,
    llvm::ArrayRef<std::uint32_t> physicalResultWidths,
    mlir::MLIRContext &context) {
  if (!ownsFixedVectorIntegerBehaviorRelation(family))
    return reject("capability family has no fixed-vector integer relation");
  if (enabledSchemas.empty())
    return reject("fixed-vector integer capability has no enabled schema");

  const ImplementationFamilyDescriptor &descriptor =
      implementationFamily(family);
  if (capabilityParamsSchema(params) != descriptor.capabilityParamsSchema)
    return reject(
        "fixed-vector integer capability has the wrong parameter schema");
  for (auto [ordinal, schema] : llvm::enumerate(enabledSchemas)) {
    if (!llvm::is_contained(descriptor.admittedSchemas, schema))
      return reject("fixed-vector integer capability enables a foreign schema");
    if (llvm::is_contained(enabledSchemas.take_front(ordinal), schema))
      return reject("fixed-vector integer capability enables a schema twice");
  }

  const std::uint32_t maxPayloadBits = payloadCapacity(params);
  if (maxPayloadBits == 0)
    return reject("fixed-vector integer capability has invalid parameters");

  std::vector<BehaviorFact> candidates;
  for (OperationSchemaId schema : descriptor.admittedSchemas) {
    if (!llvm::is_contained(enabledSchemas, schema))
      continue;
    if (family == ImplementationFamilyId::FixedVectorValueSelect) {
      const auto *select = std::get_if<FixedVectorValueSelectParams>(&params);
      if (!select)
        return reject("fixed-vector select has the wrong parameter schema");
      auto elements = selectElementDomain(*select, context);
      if (!elements)
        return elements.takeError();
      for (mlir::Type element : *elements)
        if (llvm::Error error = appendCandidate(
                candidates, family, params, schema, element, maxPayloadBits,
                physicalInputWidths, physicalResultWidths, context))
          return std::move(error);
      continue;
    }

    const IntegerWidthSet *widths = nullptr;
    const IntegerPredicateSet *predicates = nullptr;
    if (const auto *integers = std::get_if<FixedVectorIntegerParams>(&params)) {
      widths = &integers->elementWidths;
    } else if (const auto *compare =
                   std::get_if<FixedVectorIntegerCompareMinMaxParams>(
                       &params)) {
      widths = &compare->elementWidths;
      predicates = &compare->predicates;
    } else {
      return reject(
          "fixed-vector integer family has the wrong parameter schema");
    }
    for (IntegerWidth width : integerWidthDomain) {
      if (!widths->contains(width))
        continue;
      mlir::Type element = mlir::IntegerType::get(&context, getBitWidth(width));
      if (schema == OperationSchemaId::ArithCmpI) {
        for (std::uint32_t ordinal = 0;
             ordinal <= mlir::arith::getMaxEnumValForCmpIPredicate();
             ++ordinal) {
          const auto predicate =
              static_cast<mlir::arith::CmpIPredicate>(ordinal);
          if (predicates && predicates->contains(predicate))
            if (llvm::Error error =
                    appendCandidate(candidates, family, params, schema, element,
                                    maxPayloadBits, physicalInputWidths,
                                    physicalResultWidths, context, predicate))
              return std::move(error);
        }
      } else if (llvm::Error error = appendCandidate(
                     candidates, family, params, schema, element,
                     maxPayloadBits, physicalInputWidths, physicalResultWidths,
                     context)) {
        return std::move(error);
      }
    }
  }
  for (OperationSchemaId schema : enabledSchemas)
    if (llvm::none_of(candidates, [&](const BehaviorFact &candidate) {
          return candidate.actor.schema == schema;
        }))
      return reject("fixed-vector integer enabled schema has no reachable "
                    "behavior witness");
  return encodeDomain(family, std::move(candidates));
}

llvm::Expected<::loom::CanonicalSemanticBytes>
fabric::detail::projectFixedVectorIntegerBehavior(
    ImplementationFamilyId family,
    const ::dataflow::CanonicalActorSchemaProjection &actor,
    llvm::ArrayRef<FiniteImplementationFamilyBehaviorPoint> domain) {
  if (!ownsFixedVectorIntegerBehaviorRelation(family))
    return reject("capability family has no fixed-vector integer projector");
  auto projected = describeActor(family, actor);
  if (!projected)
    return projected.takeError();
  for (const FiniteImplementationFamilyBehaviorPoint &point : domain) {
    auto witness = describeActor(family, point.representativeActor);
    if (!witness)
      return witness.takeError();
    if (!sameBehavior(*projected, *witness))
      continue;
    if (!point.semanticConfiguration)
      return reject("fixed-vector integer relation has no semantic field");
    return *point.semanticConfiguration;
  }
  return reject("actor is outside the fixed-vector integer behavior image");
}
