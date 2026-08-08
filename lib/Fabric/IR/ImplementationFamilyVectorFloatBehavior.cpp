//===- ImplementationFamilyVectorFloatBehavior.cpp ----------------------===//
//
// Owns the finite behavior quotient of fixed-vector floating resources.
//
//===----------------------------------------------------------------------===//

#include "ImplementationFamilyVectorFloatBehavior.h"

#include "ImplementationFamilyBehaviorInternal.h"

#include "Dataflow/IR/OperationSchemaCodec.h"

#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <numeric>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

using namespace fabric;
using ::dataflow::OperationSchemaId;
using ::mlir::arith::CmpFPredicate;
using ::mlir::arith::FastMathFlags;
using ::mlir::arith::RoundingMode;

struct BehaviorFact final {
  ::dataflow::CanonicalActorSchemaProjection actor;
  ::loom::CanonicalSemanticBytes canonicalActor;
  llvm::StringRef role;
  std::optional<CmpFPredicate> predicate;
  std::optional<RoundingMode> rounding;
  ::mlir::FloatType elementType;
  std::uint32_t elementWidth = 0;
  bool nanRelaxed = false;
  FloatNaNBehavior strictNaNBehavior = FloatNaNBehavior::IEEE;
  std::vector<std::uint64_t> operandPorts;
  std::vector<std::uint64_t> resultPorts;
};

struct CoverPoint final {
  BehaviorFact fact;
  bool normalized = false;
};

enum class NumericFormatKind : std::uint8_t { None, Exact, Width };

struct NumericFormat final {
  NumericFormatKind kind = NumericFormatKind::None;
  ::mlir::Type exactType;
  std::uint32_t width = 0;
};

struct ParameterView final {
  const FloatFormatSet *formats = nullptr;
  const FloatBehaviorProfile *behavior = nullptr;
  const FloatPredicateSet *predicates = nullptr;
  std::uint32_t maxPayloadBits = 0;
};

llvm::Error reject(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(), message);
}

bool hasFlag(FastMathFlags flags, FastMathFlags flag) {
  using Bits = std::underlying_type_t<FastMathFlags>;
  return (static_cast<Bits>(flags) & static_cast<Bits>(flag)) != 0;
}

FastMathFlags addFlag(FastMathFlags flags, FastMathFlags flag) {
  using Bits = std::underlying_type_t<FastMathFlags>;
  return static_cast<FastMathFlags>(static_cast<Bits>(flags) |
                                    static_cast<Bits>(flag));
}

llvm::Expected<::mlir::FloatType> floatType(::mlir::MLIRContext &context,
                                            FloatFormat format) {
  switch (format) {
  case FloatFormat::F16:
    return ::mlir::Float16Type::get(&context);
  case FloatFormat::BF16:
    return ::mlir::BFloat16Type::get(&context);
  case FloatFormat::F32:
    return ::mlir::Float32Type::get(&context);
  case FloatFormat::F64:
    return ::mlir::Float64Type::get(&context);
  }
  return reject("fixed-vector floating relation contains an unknown format");
}

bool isCompareFamily(ImplementationFamilyId family) {
  return family == ImplementationFamilyId::FixedVectorFloatCompareMinMax;
}

bool selectsRounding(ImplementationFamilyId family) {
  switch (family) {
  case ImplementationFamilyId::FixedVectorFloatAddSub:
  case ImplementationFamilyId::FixedVectorFloatMultiply:
  case ImplementationFamilyId::FixedVectorFloatFma:
    return true;
  default:
    return false;
  }
}

bool isComparison(OperationSchemaId schema) {
  return schema == OperationSchemaId::ArithCmpF;
}

llvm::Expected<unsigned> inputCount(OperationSchemaId schema) {
  switch (schema) {
  case OperationSchemaId::ArithNegF:
  case OperationSchemaId::MathAbsF:
    return 1;
  case OperationSchemaId::ArithAddF:
  case OperationSchemaId::ArithSubF:
  case OperationSchemaId::ArithMulF:
  case OperationSchemaId::ArithCmpF:
  case OperationSchemaId::ArithMinimumF:
  case OperationSchemaId::ArithMaximumF:
  case OperationSchemaId::ArithMinNumF:
  case OperationSchemaId::ArithMaxNumF:
    return 2;
  case OperationSchemaId::MathFma:
    return 3;
  default:
    return reject("fixed-vector floating schema has no physical role shape");
  }
}

llvm::Expected<llvm::StringRef> roleFor(OperationSchemaId schema) {
  switch (schema) {
  case OperationSchemaId::ArithNegF:
    return "Negate";
  case OperationSchemaId::MathAbsF:
    return "Absolute";
  case OperationSchemaId::ArithAddF:
    return "Add";
  case OperationSchemaId::ArithSubF:
    return "Sub";
  case OperationSchemaId::ArithCmpF:
    return "Compare";
  case OperationSchemaId::ArithMinimumF:
    return "Minimum";
  case OperationSchemaId::ArithMaximumF:
    return "Maximum";
  case OperationSchemaId::ArithMinNumF:
    return "MinNumber";
  case OperationSchemaId::ArithMaxNumF:
    return "MaxNumber";
  case OperationSchemaId::ArithMulF:
  case OperationSchemaId::MathFma:
    return "";
  default:
    return reject("fixed-vector floating schema has no canonical role");
  }
}

FloatNaNBehavior nanBehaviorFor(OperationSchemaId schema) {
  return schema == OperationSchemaId::ArithMinNumF ||
                 schema == OperationSchemaId::ArithMaxNumF
             ? FloatNaNBehavior::NumberPreferred
             : FloatNaNBehavior::IEEE;
}

llvm::StringRef normalizedRole(llvm::StringRef role) {
  if (role == "MinNumber")
    return "Minimum";
  if (role == "MaxNumber")
    return "Maximum";
  return role;
}

CmpFPredicate normalizedPredicate(CmpFPredicate predicate) {
  switch (predicate) {
  case CmpFPredicate::UEQ:
    return CmpFPredicate::OEQ;
  case CmpFPredicate::UGT:
    return CmpFPredicate::OGT;
  case CmpFPredicate::UGE:
    return CmpFPredicate::OGE;
  case CmpFPredicate::ULT:
    return CmpFPredicate::OLT;
  case CmpFPredicate::ULE:
    return CmpFPredicate::OLE;
  case CmpFPredicate::UNE:
    return CmpFPredicate::ONE;
  case CmpFPredicate::ORD:
    return CmpFPredicate::AlwaysTrue;
  case CmpFPredicate::UNO:
    return CmpFPredicate::AlwaysFalse;
  default:
    return predicate;
  }
}

bool isConstantPredicate(CmpFPredicate predicate) {
  return predicate == CmpFPredicate::AlwaysFalse ||
         predicate == CmpFPredicate::AlwaysTrue;
}

llvm::Expected<ParameterView>
parametersFor(ImplementationFamilyId family,
              const FamilyCapabilityParams &params) {
  if (isCompareFamily(family)) {
    const auto *compare =
        std::get_if<FixedVectorFloatCompareMinMaxParams>(&params);
    if (!compare)
      return reject("fixed-vector floating family has the wrong parameter "
                    "schema");
    return ParameterView{&compare->elementFormats, &compare->behavior,
                         &compare->predicates, compare->maxPayloadBits};
  }
  const auto *ordinary = std::get_if<FixedVectorFloatParams>(&params);
  if (!ordinary)
    return reject(
        "fixed-vector floating family has the wrong parameter schema");
  return ParameterView{&ordinary->elementFormats, &ordinary->behavior, nullptr,
                       ordinary->maxPayloadBits};
}

llvm::Expected<std::uint64_t>
reachableLaneCount(OperationSchemaId schema, std::uint32_t elementWidth,
                   std::uint32_t maxPayloadBits,
                   llvm::ArrayRef<std::uint32_t> physicalInputWidths,
                   llvm::ArrayRef<std::uint32_t> physicalResultWidths) {
  auto count = inputCount(schema);
  if (!count)
    return count.takeError();
  if (physicalInputWidths.size() < *count || physicalResultWidths.empty())
    return reject(
        "fixed-vector floating physical role inventory is incomplete");
  if (elementWidth == 0 || maxPayloadBits == 0)
    return std::uint64_t{0};

  std::uint64_t lanes = maxPayloadBits / elementWidth;
  for (unsigned ordinal = 0; ordinal != *count; ++ordinal)
    lanes = std::min<std::uint64_t>(lanes, physicalInputWidths[ordinal] /
                                               elementWidth);
  const std::uint32_t resultElementWidth =
      isComparison(schema) ? 1 : elementWidth;
  lanes = std::min<std::uint64_t>(lanes, physicalResultWidths.front() /
                                             resultElementWidth);
  if (lanes >
      static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()))
    return reject("fixed-vector floating lane count exceeds int64");
  return lanes;
}

llvm::Expected<::dataflow::CanonicalActorSchemaProjection>
makeActor(::mlir::MLIRContext &context, OperationSchemaId schema,
          ::mlir::FloatType elementType, std::uint64_t laneCount,
          FastMathFlags flags, std::optional<RoundingMode> rounding,
          std::optional<CmpFPredicate> predicate) {
  if (laneCount == 0)
    return reject("fixed-vector floating behavior has no reachable lane");
  auto count = inputCount(schema);
  if (!count)
    return count.takeError();
  auto values = ::mlir::VectorType::get({static_cast<std::int64_t>(laneCount)},
                                        elementType);
  llvm::SmallVector<::mlir::Type, 3> inputs(*count, values);
  ::mlir::Type result = values;
  ::dataflow::SemanticPayload payload =
      ::dataflow::FloatingPointPayload{flags, rounding};
  if (isComparison(schema)) {
    if (!predicate)
      return reject("fixed-vector floating comparison has no predicate");
    result = ::mlir::VectorType::get(values.getShape(),
                                     ::mlir::IntegerType::get(&context, 1));
    payload = ::dataflow::FloatComparePayload{*predicate, flags};
  }
  return ::dataflow::CanonicalActorSchemaProjection{
      schema, ::mlir::FunctionType::get(&context, inputs, {result}),
      std::move(payload)};
}

llvm::Expected<BehaviorFact>
describeActor(ImplementationFamilyId family,
              const ::dataflow::CanonicalActorSchemaProjection &actor) {
  if (!fabric::detail::ownsFixedVectorFloatBehaviorRelation(family))
    return reject("capability family has no fixed-vector floating relation");
  if (!llvm::is_contained(implementationFamily(family).admittedSchemas,
                          actor.schema))
    return reject("fixed-vector floating actor schema is not admitted by its "
                  "family");
  auto role = roleFor(actor.schema);
  if (!role)
    return role.takeError();
  auto count = inputCount(actor.schema);
  if (!count)
    return count.takeError();
  if (actor.type.getNumInputs() != *count || actor.type.getNumResults() != 1)
    return reject("fixed-vector floating actor has the wrong arity");

  auto values = llvm::dyn_cast<::mlir::VectorType>(actor.type.getInput(0));
  if (!values || values.isScalable() || values.getRank() == 0 ||
      llvm::any_of(values.getShape(),
                   [](std::int64_t extent) { return extent <= 0; }))
    return reject("fixed-vector floating actor has an invalid fixed shape");
  for (unsigned ordinal = 1; ordinal != *count; ++ordinal)
    if (actor.type.getInput(ordinal) != values)
      return reject("fixed-vector floating operand types do not agree");

  auto elementType = llvm::dyn_cast<::mlir::FloatType>(values.getElementType());
  if (!elementType)
    return reject("fixed-vector floating actor has a non-floating element");
  const unsigned rawWidth = elementType.getWidth();
  if (rawWidth == 0 || rawWidth > std::numeric_limits<std::uint32_t>::max())
    return reject("fixed-vector floating element width exceeds uint32");

  FastMathFlags flags = FastMathFlags::none;
  std::optional<CmpFPredicate> predicate;
  std::optional<RoundingMode> rounding;
  if (isComparison(actor.schema)) {
    const auto *payload =
        std::get_if<::dataflow::FloatComparePayload>(&actor.payload);
    if (!payload)
      return reject("fixed-vector floating comparison has no typed payload");
    flags = payload->flags;
    predicate = payload->predicate;
    auto result = llvm::dyn_cast<::mlir::VectorType>(actor.type.getResult(0));
    if (!result || result.isScalable() ||
        result.getShape() != values.getShape() ||
        !result.getElementType().isInteger(1))
      return reject("fixed-vector floating comparison result has the wrong "
                    "shape");
  } else {
    const auto *payload =
        std::get_if<::dataflow::FloatingPointPayload>(&actor.payload);
    if (!payload)
      return reject(
          "fixed-vector floating actor has no typed floating payload");
    flags = payload->flags;
    if (selectsRounding(family))
      rounding = payload->roundingMode.value_or(RoundingMode::to_nearest_even);
    if (actor.type.getResult(0) != values)
      return reject(
          "fixed-vector floating result type differs from its operands");
  }

  auto canonical = ::dataflow::encodeCanonicalActorSchemaProjection(actor);
  if (!canonical)
    return canonical.takeError();
  BehaviorFact fact{actor,
                    std::move(*canonical),
                    *role,
                    predicate,
                    rounding,
                    elementType,
                    static_cast<std::uint32_t>(rawWidth),
                    hasFlag(flags, FastMathFlags::nnan),
                    nanBehaviorFor(actor.schema),
                    {},
                    {}};
  fact.operandPorts.resize(*count);
  fact.resultPorts.resize(1);
  std::iota(fact.operandPorts.begin(), fact.operandPorts.end(), 0);
  fact.resultPorts.front() = 0;
  return fact;
}

bool sameRawRole(const BehaviorFact &lhs, const BehaviorFact &rhs) {
  return lhs.role == rhs.role && lhs.predicate == rhs.predicate;
}

bool sameRawBehavior(ImplementationFamilyId family, const BehaviorFact &lhs,
                     const BehaviorFact &rhs) {
  if (!sameRawRole(lhs, rhs))
    return false;
  if (family == ImplementationFamilyId::FixedVectorFloatSign)
    return lhs.elementWidth == rhs.elementWidth;
  if (isCompareFamily(family)) {
    const bool constant = lhs.predicate && isConstantPredicate(*lhs.predicate);
    return constant || lhs.elementType == rhs.elementType;
  }
  return lhs.elementType == rhs.elementType && lhs.rounding == rhs.rounding;
}

bool strictRefinesRelaxed(const BehaviorFact &strict,
                          const BehaviorFact &relaxed) {
  if (strict.nanRelaxed || !relaxed.nanRelaxed)
    return false;
  const bool sameUnrefined = sameRawRole(strict, relaxed);
  const bool sameNormalized =
      normalizedRole(strict.role) == normalizedRole(relaxed.role) &&
      ((!strict.predicate && !relaxed.predicate) ||
       (strict.predicate && relaxed.predicate &&
        normalizedPredicate(*strict.predicate) ==
            normalizedPredicate(*relaxed.predicate)));
  if (!sameUnrefined && !sameNormalized)
    return false;
  if (strict.predicate && isConstantPredicate(*strict.predicate))
    return true;
  return strict.elementType == relaxed.elementType;
}

bool sameNormalizedBehavior(const BehaviorFact &lhs, const BehaviorFact &rhs) {
  if (!lhs.nanRelaxed || !rhs.nanRelaxed ||
      normalizedRole(lhs.role) != normalizedRole(rhs.role))
    return false;
  if (lhs.predicate || rhs.predicate) {
    if (!lhs.predicate || !rhs.predicate ||
        normalizedPredicate(*lhs.predicate) !=
            normalizedPredicate(*rhs.predicate))
      return false;
    if (isConstantPredicate(normalizedPredicate(*lhs.predicate)))
      return true;
  }
  return lhs.elementWidth == rhs.elementWidth;
}

NumericFormat numericFormat(ImplementationFamilyId family,
                            const CoverPoint &point) {
  if (family == ImplementationFamilyId::FixedVectorFloatSign)
    return {NumericFormatKind::Width, {}, point.fact.elementWidth};
  if (!isCompareFamily(family))
    return {NumericFormatKind::Exact, point.fact.elementType, 0};
  if (point.fact.predicate) {
    CmpFPredicate predicate = point.normalized
                                  ? normalizedPredicate(*point.fact.predicate)
                                  : *point.fact.predicate;
    if (isConstantPredicate(predicate))
      return {};
  }
  return point.normalized ? NumericFormat{NumericFormatKind::Width,
                                          {},
                                          point.fact.elementWidth}
                          : NumericFormat{NumericFormatKind::Exact,
                                          point.fact.elementType, 0};
}

bool sameNumericFormat(const NumericFormat &lhs, const NumericFormat &rhs) {
  if (lhs.kind != rhs.kind)
    return false;
  switch (lhs.kind) {
  case NumericFormatKind::None:
    return true;
  case NumericFormatKind::Exact:
    return lhs.exactType == rhs.exactType;
  case NumericFormatKind::Width:
    return lhs.width == rhs.width;
  }
  llvm_unreachable("unhandled numeric format kind");
}

bool valueVaries(
    llvm::ArrayRef<CoverPoint> points,
    llvm::function_ref<bool(const CoverPoint &, const CoverPoint &)> equal,
    llvm::function_ref<bool(const CoverPoint &)> observes) {
  const CoverPoint *first = nullptr;
  for (const CoverPoint &point : points) {
    if (!observes(point))
      continue;
    if (!first)
      first = &point;
    else if (!equal(*first, point))
      return true;
  }
  return false;
}

llvm::Expected<std::vector<FiniteImplementationFamilyBehaviorPoint>>
encodeDomain(ImplementationFamilyId family, std::vector<CoverPoint> cover) {
  if (cover.empty())
    return reject("fixed-vector floating capability has no reachable behavior");

  const bool encodeRole = valueVaries(
      cover,
      [](const CoverPoint &lhs, const CoverPoint &rhs) {
        const llvm::StringRef lhsRole =
            lhs.normalized ? normalizedRole(lhs.fact.role) : lhs.fact.role;
        const llvm::StringRef rhsRole =
            rhs.normalized ? normalizedRole(rhs.fact.role) : rhs.fact.role;
        return lhsRole == rhsRole;
      },
      [](const CoverPoint &) { return true; });
  const bool encodeRounding = valueVaries(
      cover,
      [](const CoverPoint &lhs, const CoverPoint &rhs) {
        return lhs.fact.rounding == rhs.fact.rounding;
      },
      [](const CoverPoint &point) { return point.fact.rounding.has_value(); });
  const bool encodeNumeric = valueVaries(
      cover,
      [&](const CoverPoint &lhs, const CoverPoint &rhs) {
        return sameNumericFormat(numericFormat(family, lhs),
                                 numericFormat(family, rhs));
      },
      [&](const CoverPoint &point) {
        return numericFormat(family, point).kind != NumericFormatKind::None;
      });

  const auto predicateVaries = [&](llvm::StringRef role) {
    return valueVaries(
        cover,
        [](const CoverPoint &lhs, const CoverPoint &rhs) {
          const auto lhsPredicate =
              lhs.fact.predicate
                  ? std::optional<CmpFPredicate>(
                        lhs.normalized
                            ? normalizedPredicate(*lhs.fact.predicate)
                            : *lhs.fact.predicate)
                  : std::nullopt;
          const auto rhsPredicate =
              rhs.fact.predicate
                  ? std::optional<CmpFPredicate>(
                        rhs.normalized
                            ? normalizedPredicate(*rhs.fact.predicate)
                            : *rhs.fact.predicate)
                  : std::nullopt;
          return lhsPredicate == rhsPredicate;
        },
        [&](const CoverPoint &point) {
          const llvm::StringRef pointRole =
              point.normalized ? normalizedRole(point.fact.role)
                               : point.fact.role;
          return pointRole == role && point.fact.predicate.has_value();
        });
  };

  std::vector<FiniteImplementationFamilyBehaviorPoint> domain;
  domain.reserve(cover.size());
  for (CoverPoint &point : cover) {
    const llvm::StringRef role =
        point.normalized ? normalizedRole(point.fact.role) : point.fact.role;
    llvm::SmallVector<detail::ImplementationFamilyBehaviorKeyComponent, 3>
        components;
    if (point.fact.predicate && predicateVaries(role)) {
      const CmpFPredicate predicate =
          point.normalized ? normalizedPredicate(*point.fact.predicate)
                           : *point.fact.predicate;
      auto encoded = ::dataflow::encodeFloatComparePredicate(predicate);
      if (!encoded)
        return encoded.takeError();
      components.emplace_back(std::move(*encoded));
    }
    const NumericFormat format = numericFormat(family, point);
    if (encodeNumeric && format.kind != NumericFormatKind::None) {
      if (format.kind == NumericFormatKind::Exact) {
        auto encoded = ::dataflow::encodeCanonicalType(format.exactType);
        if (!encoded)
          return encoded.takeError();
        components.emplace_back(std::move(*encoded));
      } else {
        components.emplace_back(format.width);
      }
    }
    if (encodeRounding && point.fact.rounding) {
      auto encoded = ::dataflow::encodeRoundingMode(*point.fact.rounding);
      if (!encoded)
        return encoded.takeError();
      components.emplace_back(std::move(*encoded));
    }
    auto key = detail::encodeImplementationFamilyBehaviorKey(
        family, encodeRole ? role : "", components);
    if (!key)
      return key.takeError();
    domain.emplace_back(std::move(point.fact.actor), std::move(*key),
                        std::nullopt, std::move(point.fact.operandPorts),
                        std::move(point.fact.resultPorts));
  }
  llvm::sort(domain, [](const auto &lhs, const auto &rhs) {
    return std::lexicographical_compare(
        lhs.semanticConfiguration->bytes().begin(),
        lhs.semanticConfiguration->bytes().end(),
        rhs.semanticConfiguration->bytes().begin(),
        rhs.semanticConfiguration->bytes().end());
  });
  if (std::adjacent_find(domain.begin(), domain.end(),
                         [](const auto &lhs, const auto &rhs) {
                           return lhs.semanticConfiguration->bytes().equals(
                               rhs.semanticConfiguration->bytes());
                         }) != domain.end())
    return reject("fixed-vector floating behavior codec is not injective");
  if (domain.size() == 1)
    domain.front().semanticConfiguration = std::nullopt;
  return domain;
}

llvm::Error validateContextualProfile(const ParameterView &parameters,
                                      llvm::ArrayRef<BehaviorFact> candidates) {
  if (parameters.behavior->roundingModes.size() > 1) {
    for (std::uint32_t ordinal = 0;
         ordinal <= ::mlir::arith::getMaxEnumValForRoundingMode(); ++ordinal) {
      const auto mode = static_cast<RoundingMode>(ordinal);
      if (!parameters.behavior->roundingModes.contains(mode))
        continue;
      if (llvm::none_of(candidates, [&](const BehaviorFact &candidate) {
            return candidate.rounding == mode;
          }))
        return reject("fixed-vector floating rounding profile contains an "
                      "orphan behavior");
    }
  }
  if (parameters.behavior->nanBehaviors.size() > 1) {
    constexpr std::array behaviors = {FloatNaNBehavior::IEEE,
                                      FloatNaNBehavior::NumberPreferred};
    for (FloatNaNBehavior behavior : behaviors) {
      if (!parameters.behavior->nanBehaviors.contains(behavior))
        continue;
      if (llvm::none_of(candidates, [&](const BehaviorFact &candidate) {
            return !candidate.nanRelaxed &&
                   candidate.strictNaNBehavior == behavior;
          }))
        return reject("fixed-vector floating NaN profile contains an orphan "
                      "behavior");
    }
  }
  return llvm::Error::success();
}

llvm::Expected<std::vector<CoverPoint>>
buildCover(ImplementationFamilyId family,
           std::vector<BehaviorFact> candidates) {
  llvm::sort(candidates, [](const BehaviorFact &lhs, const BehaviorFact &rhs) {
    if (lhs.nanRelaxed != rhs.nanRelaxed)
      return !lhs.nanRelaxed;
    return std::lexicographical_compare(
        lhs.canonicalActor.bytes().begin(), lhs.canonicalActor.bytes().end(),
        rhs.canonicalActor.bytes().begin(), rhs.canonicalActor.bytes().end());
  });

  std::vector<CoverPoint> cover;
  for (BehaviorFact &candidate : candidates) {
    if (!isCompareFamily(family)) {
      if (llvm::any_of(cover, [&](const CoverPoint &point) {
            return sameRawBehavior(family, candidate, point.fact);
          }))
        continue;
      cover.push_back({std::move(candidate), false});
      continue;
    }

    if (!candidate.nanRelaxed) {
      if (llvm::any_of(cover, [&](const CoverPoint &point) {
            return !point.normalized && !point.fact.nanRelaxed &&
                   sameRawBehavior(family, candidate, point.fact);
          }))
        continue;
      cover.push_back({std::move(candidate), false});
      continue;
    }

    const CoverPoint *preferred = nullptr;
    bool preferredSameRole = false;
    for (const CoverPoint &point : cover) {
      if (point.normalized || !strictRefinesRelaxed(point.fact, candidate))
        continue;
      const bool sameRole = sameRawRole(point.fact, candidate);
      if (!preferred || (sameRole && !preferredSameRole) ||
          (sameRole == preferredSameRole &&
           std::lexicographical_compare(
               point.fact.canonicalActor.bytes().begin(),
               point.fact.canonicalActor.bytes().end(),
               preferred->fact.canonicalActor.bytes().begin(),
               preferred->fact.canonicalActor.bytes().end()))) {
        preferred = &point;
        preferredSameRole = sameRole;
      }
    }
    if (preferred)
      continue;
    if (llvm::any_of(cover, [&](const CoverPoint &point) {
          return point.normalized &&
                 sameNormalizedBehavior(point.fact, candidate);
        }))
      continue;
    cover.push_back({std::move(candidate), true});
  }
  return cover;
}

} // namespace

bool fabric::detail::ownsFixedVectorFloatBehaviorRelation(
    ImplementationFamilyId family) {
  switch (family) {
  case ImplementationFamilyId::FixedVectorFloatSign:
  case ImplementationFamilyId::FixedVectorFloatAddSub:
  case ImplementationFamilyId::FixedVectorFloatCompareMinMax:
  case ImplementationFamilyId::FixedVectorFloatMultiply:
  case ImplementationFamilyId::FixedVectorFloatFma:
    return true;
  default:
    return false;
  }
}

llvm::Expected<std::vector<fabric::FiniteImplementationFamilyBehaviorPoint>>
fabric::detail::resolveFixedVectorFloatBehaviorDomain(
    ImplementationFamilyId family, const FamilyCapabilityParams &params,
    llvm::ArrayRef<OperationSchemaId> enabledSchemas,
    llvm::ArrayRef<std::uint32_t> physicalInputWidths,
    llvm::ArrayRef<std::uint32_t> physicalResultWidths,
    ::mlir::MLIRContext &context) {
  if (!ownsFixedVectorFloatBehaviorRelation(family))
    return reject("capability family has no fixed-vector floating relation");
  if (enabledSchemas.empty())
    return reject("fixed-vector floating capability has no enabled schema");
  const ImplementationFamilyDescriptor &descriptor =
      implementationFamily(family);
  for (auto [ordinal, schema] : llvm::enumerate(enabledSchemas)) {
    if (!llvm::is_contained(descriptor.admittedSchemas, schema))
      return reject("fixed-vector floating capability enables a foreign "
                    "schema");
    if (llvm::is_contained(enabledSchemas.take_front(ordinal), schema))
      return reject("fixed-vector floating capability enables a schema twice");
  }

  auto parameters = parametersFor(family, params);
  if (!parameters)
    return parameters.takeError();
  if (parameters->maxPayloadBits == 0)
    return reject("fixed-vector floating capability has invalid parameters");

  const FastMathFlags minimalFlags =
      detail::minimalFloatingActorPermissions(*parameters->behavior);
  std::vector<FastMathFlags> flagDomain;
  if (!hasFlag(parameters->behavior->requiredFastMath, FastMathFlags::nnan))
    flagDomain.push_back(
        ::mlir::arith::bitEnumClear(minimalFlags, FastMathFlags::nnan));
  flagDomain.push_back(addFlag(minimalFlags, FastMathFlags::nnan));

  std::vector<BehaviorFact> candidates;
  for (OperationSchemaId schema : descriptor.admittedSchemas) {
    if (!llvm::is_contained(enabledSchemas, schema))
      continue;
    const std::size_t candidateStart = candidates.size();
    std::string firstAdmissionRejection;
    for (FloatFormat format : floatFormatDomain) {
      if (!parameters->formats->contains(format))
        continue;
      auto elementType = floatType(context, format);
      if (!elementType)
        return elementType.takeError();
      auto lanes = reachableLaneCount(
          schema, elementType->getWidth(), parameters->maxPayloadBits,
          physicalInputWidths, physicalResultWidths);
      if (!lanes)
        return lanes.takeError();
      if (*lanes == 0)
        continue;

      std::vector<std::optional<CmpFPredicate>> predicates = {std::nullopt};
      if (isComparison(schema)) {
        predicates.clear();
        for (std::uint32_t ordinal = 0;
             ordinal <= ::mlir::arith::getMaxEnumValForCmpFPredicate();
             ++ordinal) {
          const auto predicate = static_cast<CmpFPredicate>(ordinal);
          if (parameters->predicates &&
              parameters->predicates->contains(predicate))
            predicates.emplace_back(predicate);
        }
      }

      std::vector<std::optional<RoundingMode>> roundingModes = {std::nullopt};
      if (selectsRounding(family)) {
        roundingModes.clear();
        for (std::uint32_t ordinal = 0;
             ordinal <= ::mlir::arith::getMaxEnumValForRoundingMode();
             ++ordinal) {
          const auto mode = static_cast<RoundingMode>(ordinal);
          if (parameters->behavior->roundingModes.contains(mode))
            roundingModes.emplace_back(mode);
        }
      }

      for (std::optional<CmpFPredicate> predicate : predicates) {
        for (std::optional<RoundingMode> rounding : roundingModes) {
          for (FastMathFlags flags : flagDomain) {
            auto actor = makeActor(context, schema, *elementType, *lanes, flags,
                                   rounding, predicate);
            if (!actor)
              return actor.takeError();
            if (llvm::Error error = verifyImplementationFamilyAdmission(
                    family, &params, *actor)) {
              if (firstAdmissionRejection.empty())
                firstAdmissionRejection = llvm::toString(std::move(error));
              else
                llvm::consumeError(std::move(error));
              continue;
            }
            auto fact = describeActor(family, *actor);
            if (!fact)
              return fact.takeError();
            candidates.push_back(std::move(*fact));
          }
        }
      }
    }
    if (candidates.size() != candidateStart)
      continue;
    if (!firstAdmissionRejection.empty())
      return reject(firstAdmissionRejection);
    return reject(
        "fixed-vector floating enabled schema has no reachable behavior");
  }
  if (llvm::Error error = validateContextualProfile(*parameters, candidates))
    return std::move(error);
  auto cover = buildCover(family, std::move(candidates));
  if (!cover)
    return cover.takeError();
  return encodeDomain(family, std::move(*cover));
}

llvm::Expected<::loom::CanonicalSemanticBytes>
fabric::detail::projectFixedVectorFloatBehavior(
    ImplementationFamilyId family,
    const ::dataflow::CanonicalActorSchemaProjection &actor,
    llvm::ArrayRef<FiniteImplementationFamilyBehaviorPoint> domain) {
  if (!ownsFixedVectorFloatBehaviorRelation(family))
    return reject("capability family has no fixed-vector floating projector");
  auto projected = describeActor(family, actor);
  if (!projected)
    return projected.takeError();

  const FiniteImplementationFamilyBehaviorPoint *best = nullptr;
  bool bestSameRole = false;
  for (const FiniteImplementationFamilyBehaviorPoint &point : domain) {
    auto witness = describeActor(family, point.representativeActor);
    if (!witness)
      return witness.takeError();
    bool matches = false;
    bool sameRole = false;
    if (!isCompareFamily(family)) {
      matches = sameRawBehavior(family, *projected, *witness);
    } else if (!projected->nanRelaxed) {
      matches =
          !witness->nanRelaxed && sameRawBehavior(family, *projected, *witness);
    } else if (!witness->nanRelaxed) {
      matches = strictRefinesRelaxed(*witness, *projected);
      sameRole = matches && sameRawRole(*witness, *projected);
    } else {
      matches = sameNormalizedBehavior(*projected, *witness);
    }
    if (!matches)
      continue;
    if (!point.semanticConfiguration)
      return reject("fixed-vector floating relation has no semantic field");
    if (!best || (sameRole && !bestSameRole) ||
        (sameRole == bestSameRole &&
         std::lexicographical_compare(
             point.semanticConfiguration->bytes().begin(),
             point.semanticConfiguration->bytes().end(),
             best->semanticConfiguration->bytes().begin(),
             best->semanticConfiguration->bytes().end()))) {
      best = &point;
      bestSameRole = sameRole;
    }
  }
  if (!best)
    return reject("actor is outside the fixed-vector floating behavior image");
  return *best->semanticConfiguration;
}
