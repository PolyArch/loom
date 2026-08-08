//===- ImplementationFamilyScalarIntegerBehavior.cpp --------------------===//
//
// Owns the closed finite behavior quotients of scalar integer families.
//
//===----------------------------------------------------------------------===//

#include "ImplementationFamilyScalarIntegerBehavior.h"

#include "ImplementationFamilyBehaviorInternal.h"

#include "Dataflow/IR/OperationSchemaCodec.h"

#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace {

using namespace fabric;

llvm::Error reject(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(), message);
}

enum class BehaviorComponentSlot : std::uint8_t {
  Predicate,
  SourceWidth,
  DestinationWidth,
  ActiveWidth,
};

using BehaviorComponentValue =
    std::variant<std::uint32_t, ::loom::CanonicalSemanticBytes>;

struct BehaviorComponent final {
  BehaviorComponentSlot slot;
  BehaviorComponentValue value;
};

struct ScalarIntegerBehaviorCandidate final {
  ::dataflow::CanonicalActorSchemaProjection actor;
  std::optional<ResolvedIndexWidth> resolvedIndexWidth;
  llvm::StringRef role;
  std::vector<BehaviorComponent> components;
  std::vector<std::uint64_t> operandPorts;
  std::vector<std::uint64_t> resultPorts;
};

bool equalComponentValue(const BehaviorComponentValue &lhs,
                         const BehaviorComponentValue &rhs) {
  if (lhs.index() != rhs.index())
    return false;
  if (const auto *left = std::get_if<std::uint32_t>(&lhs))
    return *left == std::get<std::uint32_t>(rhs);
  return std::get<::loom::CanonicalSemanticBytes>(lhs).bytes().equals(
      std::get<::loom::CanonicalSemanticBytes>(rhs).bytes());
}

const BehaviorComponent *
findComponent(const ScalarIntegerBehaviorCandidate &candidate,
              BehaviorComponentSlot slot) {
  auto found = llvm::find_if(candidate.components,
                             [&](const BehaviorComponent &component) {
                               return component.slot == slot;
                             });
  return found == candidate.components.end() ? nullptr : &*found;
}

bool componentVaries(llvm::ArrayRef<ScalarIntegerBehaviorCandidate> candidates,
                     llvm::StringRef role, BehaviorComponentSlot slot) {
  const BehaviorComponentValue *first = nullptr;
  for (const ScalarIntegerBehaviorCandidate &candidate : candidates) {
    if (candidate.role != role)
      continue;
    const BehaviorComponent *component = findComponent(candidate, slot);
    if (!component)
      continue;
    if (!first) {
      first = &component->value;
      continue;
    }
    if (!equalComponentValue(*first, component->value))
      return true;
  }
  return false;
}

std::vector<std::uint64_t> identityPorts(unsigned count) {
  std::vector<std::uint64_t> ports(count);
  std::iota(ports.begin(), ports.end(), 0);
  return ports;
}

void appendCandidate(
    std::vector<ScalarIntegerBehaviorCandidate> &candidates,
    ::dataflow::CanonicalActorSchemaProjection actor, llvm::StringRef role,
    std::vector<BehaviorComponent> components = {},
    std::optional<ResolvedIndexWidth> resolvedIndexWidth = std::nullopt) {
  std::vector<std::uint64_t> operandPorts =
      identityPorts(actor.type.getNumInputs());
  std::vector<std::uint64_t> resultPorts =
      identityPorts(actor.type.getNumResults());
  candidates.push_back({std::move(actor), resolvedIndexWidth, role,
                        std::move(components), std::move(operandPorts),
                        std::move(resultPorts)});
}

::dataflow::CanonicalActorSchemaProjection
makeUniformActor(mlir::MLIRContext &context, unsigned width,
                 ::dataflow::OperationSchemaId schema,
                 ::dataflow::SemanticPayload payload, unsigned inputCount,
                 std::optional<unsigned> resultWidth = std::nullopt) {
  mlir::Type input = mlir::IntegerType::get(&context, width);
  std::vector<mlir::Type> inputs(inputCount, input);
  mlir::Type result =
      mlir::IntegerType::get(&context, resultWidth.value_or(width));
  return {schema, mlir::FunctionType::get(&context, inputs, {result}),
          std::move(payload)};
}

llvm::Expected<::dataflow::SemanticPayload>
payloadForScalarIntegerSchema(::dataflow::OperationSchemaId schema) {
  using Schema = ::dataflow::OperationSchemaId;
  switch (::dataflow::semanticsCase(schema)) {
  case ::dataflow::OperationSemanticsCase::NoSemanticPayload:
    return ::dataflow::NoPayload{};
  case ::dataflow::OperationSemanticsCase::ArithExact:
    return ::dataflow::ExactPayload{};
  case ::dataflow::OperationSemanticsCase::ArithNonNegative:
    return ::dataflow::NonNegativePayload{};
  case ::dataflow::OperationSemanticsCase::ArithIntegerOverflow:
    return ::dataflow::IntegerOverflowPayload{};
  case ::dataflow::OperationSemanticsCase::LLVMZeroPoison:
    if (schema == Schema::LLVMCountLeadingZeros ||
        schema == Schema::LLVMCountTrailingZeros)
      return ::dataflow::ZeroPoisonPayload{true};
    break;
  case ::dataflow::OperationSemanticsCase::LLVMDisjoint:
    if (schema == Schema::LLVMOrDisjoint)
      return ::dataflow::DisjointPayload{true};
    break;
  default:
    break;
  }
  return reject("scalar integer schema has an unsupported semantic payload");
}

bool isSignedPredicate(mlir::arith::CmpIPredicate predicate) {
  using Predicate = mlir::arith::CmpIPredicate;
  switch (predicate) {
  case Predicate::slt:
  case Predicate::sle:
  case Predicate::sgt:
  case Predicate::sge:
    return true;
  case Predicate::eq:
  case Predicate::ne:
  case Predicate::ult:
  case Predicate::ule:
  case Predicate::ugt:
  case Predicate::uge:
    return false;
  }
  llvm_unreachable("unknown integer predicate");
}

llvm::Expected<const ScalarIntegerParams *>
requireScalarIntegerParams(const FamilyCapabilityParams &params) {
  const auto *typed = std::get_if<ScalarIntegerParams>(&params);
  if (!typed)
    return reject("capability has the wrong scalar integer parameter schema");
  if (!typed->integerWidths.valid() || typed->integerWidths.empty())
    return reject("scalar integer width domain is invalid");
  if (!typed->pointerFormats.valid())
    return reject("scalar integer pointer format relation is invalid");
  return typed;
}

llvm::Error appendOrdinaryCandidates(
    ImplementationFamilyId family, const ScalarIntegerParams &params,
    llvm::ArrayRef<::dataflow::OperationSchemaId> orderedSchemas,
    mlir::MLIRContext &context,
    std::vector<ScalarIntegerBehaviorCandidate> &candidates) {
  using Schema = ::dataflow::OperationSchemaId;
  for (Schema schema : orderedSchemas) {
    llvm::StringRef role;
    bool unary = false;
    bool activeWidth = false;
    switch (family) {
    case ImplementationFamilyId::ScalarIntegerAddSub:
      if (schema == Schema::ArithAddI)
        role = "Add";
      else if (schema == Schema::ArithSubI)
        role = "Sub";
      else
        return reject(
            "scalar add/sub capability contains a non-add/sub schema");
      break;
    case ImplementationFamilyId::ScalarIntegerLogic:
      if (schema == Schema::ArithAndI)
        role = "And";
      else if (schema == Schema::ArithOrI || schema == Schema::LLVMOrDisjoint)
        role = "Or";
      else if (schema == Schema::ArithXOrI)
        role = "Xor";
      else
        return reject("scalar logic capability contains a non-logic schema");
      break;
    case ImplementationFamilyId::ScalarIntegerShift:
      if (schema == Schema::ArithShLI)
        role = "Left";
      else if (schema == Schema::ArithShRUI)
        role = "LogicalRight";
      else if (schema == Schema::ArithShRSI) {
        role = "ArithmeticRight";
        activeWidth = true;
      } else {
        return reject("scalar shift capability contains a non-shift schema");
      }
      break;
    case ImplementationFamilyId::ScalarSignedIntegerDivRem:
      if (schema == Schema::ArithDivSI)
        role = "Quotient";
      else if (schema == Schema::ArithRemSI)
        role = "Remainder";
      else
        return reject("signed divider capability contains a foreign schema");
      activeWidth = true;
      break;
    case ImplementationFamilyId::ScalarUnsignedIntegerDivRem:
      if (schema == Schema::ArithDivUI)
        role = "Quotient";
      else if (schema == Schema::ArithRemUI)
        role = "Remainder";
      else
        return reject("unsigned divider capability contains a foreign schema");
      activeWidth = true;
      break;
    case ImplementationFamilyId::ScalarIntegerSaturatingAddSub:
      if (schema == Schema::LLVMSAddSat)
        role = "SignedAdd";
      else if (schema == Schema::LLVMUAddSat)
        role = "UnsignedAdd";
      else if (schema == Schema::LLVMSSubSat)
        role = "SignedSub";
      else if (schema == Schema::LLVMUSubSat)
        role = "UnsignedSub";
      else
        return reject(
            "saturating integer capability contains a foreign schema");
      activeWidth = true;
      break;
    case ImplementationFamilyId::ScalarIntegerCountZeros:
      if (schema == Schema::MathCountLeadingZeros ||
          schema == Schema::LLVMCountLeadingZeros)
        role = "Leading";
      else if (schema == Schema::MathCountTrailingZeros ||
               schema == Schema::LLVMCountTrailingZeros)
        role = "Trailing";
      else
        return reject("count-zero capability contains a foreign schema");
      unary = true;
      activeWidth = true;
      break;
    default:
      return reject("family is not an ordinary scalar integer quotient");
    }

    auto payload = payloadForScalarIntegerSchema(schema);
    if (!payload)
      return payload.takeError();
    for (IntegerWidth width : integerWidthDomain) {
      if (!params.integerWidths.contains(width))
        continue;
      std::vector<BehaviorComponent> components;
      if (activeWidth)
        components.push_back(
            {BehaviorComponentSlot::ActiveWidth, getBitWidth(width)});
      appendCandidate(candidates,
                      makeUniformActor(context, getBitWidth(width), schema,
                                       *payload, unary ? 1 : 2),
                      role, std::move(components));
    }
  }
  return llvm::Error::success();
}

llvm::Error appendCompareCandidates(
    const ScalarIntegerCompareMinMaxParams &params,
    llvm::ArrayRef<::dataflow::OperationSchemaId> orderedSchemas,
    mlir::MLIRContext &context,
    std::vector<ScalarIntegerBehaviorCandidate> &candidates) {
  if (!params.operandWidths.valid() || params.operandWidths.empty() ||
      !params.predicates.valid() || params.predicates.empty())
    return reject("scalar integer compare parameter domain is invalid");

  using Schema = ::dataflow::OperationSchemaId;
  for (Schema schema : orderedSchemas) {
    if (schema == Schema::ArithCmpI) {
      for (std::uint32_t ordinal = 0;
           ordinal <= mlir::arith::getMaxEnumValForCmpIPredicate(); ++ordinal) {
        const auto predicate = static_cast<mlir::arith::CmpIPredicate>(ordinal);
        if (!params.predicates.contains(predicate))
          continue;
        auto encodedPredicate =
            ::dataflow::encodeIntegerComparePredicate(predicate);
        if (!encodedPredicate)
          return encodedPredicate.takeError();
        for (IntegerWidth width : integerWidthDomain) {
          if (!params.operandWidths.contains(width))
            continue;
          std::vector<BehaviorComponent> components;
          components.push_back(
              {BehaviorComponentSlot::Predicate, *encodedPredicate});
          if (isSignedPredicate(predicate))
            components.push_back(
                {BehaviorComponentSlot::ActiveWidth, getBitWidth(width)});
          mlir::Type operand =
              mlir::IntegerType::get(&context, getBitWidth(width));
          appendCandidate(
              candidates,
              {schema,
               mlir::FunctionType::get(&context, {operand, operand},
                                       {mlir::IntegerType::get(&context, 1)}),
               ::dataflow::IntegerComparePayload{predicate}},
              "Compare", std::move(components));
        }
      }
      continue;
    }

    llvm::StringRef role;
    bool signedBehavior = false;
    if (schema == Schema::ArithMinSI) {
      role = "SignedMin";
      signedBehavior = true;
    } else if (schema == Schema::ArithMaxSI) {
      role = "SignedMax";
      signedBehavior = true;
    } else if (schema == Schema::ArithMinUI) {
      role = "UnsignedMin";
    } else if (schema == Schema::ArithMaxUI) {
      role = "UnsignedMax";
    } else {
      return reject("compare/min/max capability contains a foreign schema");
    }
    for (IntegerWidth width : integerWidthDomain) {
      if (!params.operandWidths.contains(width))
        continue;
      std::vector<BehaviorComponent> components;
      if (signedBehavior)
        components.push_back(
            {BehaviorComponentSlot::ActiveWidth, getBitWidth(width)});
      appendCandidate(candidates,
                      makeUniformActor(context, getBitWidth(width), schema,
                                       ::dataflow::NoPayload{}, 2),
                      role, std::move(components));
    }
  }
  return llvm::Error::success();
}

enum class CastRole : std::uint8_t {
  Identity,
  SignExtend,
  ZeroExtend,
  Truncate,
};

struct CastCase final {
  ::dataflow::OperationSchemaId schema;
  CastRole role;
  IntegerWidth source;
  IntegerWidth destination;
  bool sourceIsIndex = false;
  bool destinationIsIndex = false;
  std::optional<ResolvedIndexWidth> resolvedIndexWidth;
};

CastRole classifyCast(::dataflow::OperationSchemaId schema, IntegerWidth source,
                      IntegerWidth destination) {
  const unsigned sourceBits = getBitWidth(source);
  const unsigned destinationBits = getBitWidth(destination);
  if (sourceBits == destinationBits)
    return CastRole::Identity;
  if (sourceBits > destinationBits)
    return CastRole::Truncate;
  if (schema == ::dataflow::OperationSchemaId::ArithExtSI ||
      schema == ::dataflow::OperationSchemaId::ArithIndexCast)
    return CastRole::SignExtend;
  return CastRole::ZeroExtend;
}

llvm::StringRef castRoleSpelling(CastRole role) {
  switch (role) {
  case CastRole::Identity:
    return "Identity";
  case CastRole::SignExtend:
    return "SignExtend";
  case CastRole::ZeroExtend:
    return "ZeroExtend";
  case CastRole::Truncate:
    return "Truncate";
  }
  llvm_unreachable("unknown scalar integer cast role");
}

llvm::Expected<std::vector<CastCase>> enumerateCastCases(
    const ScalarIntegerCastParams &params,
    llvm::ArrayRef<::dataflow::OperationSchemaId> orderedSchemas) {
  if (!params.relation.widthPairs.valid() ||
      params.relation.widthPairs.empty() ||
      !params.relation.resolvedIndexWidths.valid())
    return reject("scalar integer cast relation is invalid");

  using Schema = ::dataflow::OperationSchemaId;
  std::vector<CastCase> cases;
  for (Schema schema : orderedSchemas) {
    const std::size_t begin = cases.size();
    const bool indexCast =
        schema == Schema::ArithIndexCast || schema == Schema::ArithIndexCastUI;
    if (indexCast) {
      for (ResolvedIndexWidth resolved : resolvedIndexWidthDomain) {
        if (!params.relation.resolvedIndexWidths.contains(resolved))
          continue;
        const IntegerWidth indexWidth = resolved == ResolvedIndexWidth::I32
                                            ? IntegerWidth::I32
                                            : IntegerWidth::I64;
        for (IntegerWidth source : integerWidthDomain) {
          for (IntegerWidth destination : integerWidthDomain) {
            if (!params.relation.widthPairs.contains(source, destination))
              continue;
            const CastRole role = classifyCast(schema, source, destination);
            if (source == indexWidth)
              cases.push_back(
                  {schema, role, source, destination, true, false, resolved});
            if (destination == indexWidth)
              cases.push_back(
                  {schema, role, source, destination, false, true, resolved});
          }
        }
      }
    } else {
      if (schema != Schema::ArithExtSI && schema != Schema::ArithExtUI &&
          schema != Schema::ArithTruncI)
        return reject(
            "scalar integer cast capability contains a foreign schema");
      for (IntegerWidth source : integerWidthDomain) {
        for (IntegerWidth destination : integerWidthDomain) {
          if (!params.relation.widthPairs.contains(source, destination))
            continue;
          const bool admitted =
              schema == Schema::ArithTruncI
                  ? getBitWidth(source) > getBitWidth(destination)
                  : getBitWidth(source) < getBitWidth(destination);
          if (admitted)
            cases.push_back({schema, classifyCast(schema, source, destination),
                             source, destination, false, false, std::nullopt});
        }
      }
    }
    if (cases.size() == begin)
      return reject("scalar integer cast schema has no admitted behavior");
  }

  for (IntegerWidth source : integerWidthDomain) {
    for (IntegerWidth destination : integerWidthDomain) {
      if (!params.relation.widthPairs.contains(source, destination))
        continue;
      if (!llvm::any_of(cases, [&](const CastCase &castCase) {
            return castCase.source == source &&
                   castCase.destination == destination;
          }))
        return reject("scalar integer cast relation has an orphan width pair");
    }
  }
  for (ResolvedIndexWidth resolved : resolvedIndexWidthDomain) {
    if (params.relation.resolvedIndexWidths.contains(resolved) &&
        !llvm::any_of(cases, [&](const CastCase &castCase) {
          return castCase.resolvedIndexWidth == resolved;
        }))
      return reject("scalar integer cast relation has an orphan index width");
  }
  return cases;
}

llvm::Error appendCastCandidates(
    const ScalarIntegerCastParams &params,
    llvm::ArrayRef<::dataflow::OperationSchemaId> orderedSchemas,
    mlir::MLIRContext &context,
    std::vector<ScalarIntegerBehaviorCandidate> &candidates) {
  auto cases = enumerateCastCases(params, orderedSchemas);
  if (!cases)
    return cases.takeError();
  for (const CastCase &castCase : *cases) {
    auto payload = payloadForScalarIntegerSchema(castCase.schema);
    if (!payload)
      return payload.takeError();
    mlir::Type source = castCase.sourceIsIndex
                            ? mlir::Type(mlir::IndexType::get(&context))
                            : mlir::Type(mlir::IntegerType::get(
                                  &context, getBitWidth(castCase.source)));
    mlir::Type destination =
        castCase.destinationIsIndex
            ? mlir::Type(mlir::IndexType::get(&context))
            : mlir::Type(mlir::IntegerType::get(
                  &context, getBitWidth(castCase.destination)));
    std::vector<BehaviorComponent> components;
    components.push_back(
        {BehaviorComponentSlot::SourceWidth, getBitWidth(castCase.source)});
    components.push_back({BehaviorComponentSlot::DestinationWidth,
                          getBitWidth(castCase.destination)});
    appendCandidate(candidates,
                    {castCase.schema,
                     mlir::FunctionType::get(&context, {source}, {destination}),
                     std::move(*payload)},
                    castRoleSpelling(castCase.role), std::move(components),
                    castCase.resolvedIndexWidth);
  }
  return llvm::Error::success();
}

llvm::Expected<bool>
isPhysicallyReachable(const ScalarIntegerBehaviorCandidate &candidate,
                      llvm::ArrayRef<std::uint32_t> physicalInputWidths,
                      llvm::ArrayRef<std::uint32_t> physicalResultWidths) {
  ::dataflow::CanonicalActorSchemaProjection represented = candidate.actor;
  if (candidate.resolvedIndexWidth) {
    auto projected = projectResolvedIndexTypes(
        candidate.actor,
        getResolvedIndexBitWidth(*candidate.resolvedIndexWidth));
    if (!projected)
      return projected.takeError();
    represented = std::move(*projected);
  }
  const auto fits =
      [](llvm::ArrayRef<mlir::Type> types,
         llvm::ArrayRef<std::uint32_t> widths) -> llvm::Expected<bool> {
    if (types.size() > widths.size())
      return false;
    for (auto [type, width] : llvm::zip(types, widths)) {
      std::string error;
      auto payloadWidth = getSemanticPayloadWidth(type, error);
      if (mlir::failed(payloadWidth))
        return reject(error);
      if (*payloadWidth > width)
        return false;
    }
    return true;
  };
  auto inputs = fits(represented.type.getInputs(), physicalInputWidths);
  if (!inputs || !*inputs)
    return inputs;
  return fits(represented.type.getResults(), physicalResultWidths);
}

llvm::Expected<::loom::CanonicalSemanticBytes>
encodeCandidate(ImplementationFamilyId family,
                llvm::ArrayRef<ScalarIntegerBehaviorCandidate> candidates,
                const ScalarIntegerBehaviorCandidate &candidate) {
  std::vector<detail::ImplementationFamilyBehaviorKeyComponent> components;
  for (const BehaviorComponent &component : candidate.components) {
    if (!componentVaries(candidates, candidate.role, component.slot))
      continue;
    if (const auto *width = std::get_if<std::uint32_t>(&component.value))
      components.emplace_back(*width);
    else
      components.emplace_back(
          std::get<::loom::CanonicalSemanticBytes>(component.value));
  }
  return detail::encodeImplementationFamilyBehaviorKey(family, candidate.role,
                                                       components);
}

bool lessKey(const FiniteImplementationFamilyBehaviorPoint &lhs,
             const FiniteImplementationFamilyBehaviorPoint &rhs) {
  return std::lexicographical_compare(
      lhs.semanticConfiguration->bytes().begin(),
      lhs.semanticConfiguration->bytes().end(),
      rhs.semanticConfiguration->bytes().begin(),
      rhs.semanticConfiguration->bytes().end());
}

} // namespace

llvm::Expected<std::vector<fabric::FiniteImplementationFamilyBehaviorPoint>>
fabric::detail::resolveScalarIntegerBehaviorDomain(
    ImplementationFamilyId family, const FamilyCapabilityParams &params,
    llvm::ArrayRef<::dataflow::OperationSchemaId> enabledSchemas,
    llvm::ArrayRef<std::uint32_t> physicalInputWidths,
    llvm::ArrayRef<std::uint32_t> physicalResultWidths,
    mlir::MLIRContext &context) {
  if (static_cast<std::uint32_t>(family) >= implementationFamilyCount())
    return reject("implementation family is not registered");
  if (enabledSchemas.empty())
    return reject("scalar integer capability has no enabled schema");
  if (llvm::is_contained(enabledSchemas,
                         ::dataflow::OperationSchemaId::LLVMGetElementPtr))
    return reject("GEP has no bounded scalar integer behavior relation");

  const ImplementationFamilyDescriptor &descriptor =
      implementationFamily(family);
  if (capabilityParamsSchema(params) != descriptor.capabilityParamsSchema)
    return reject("capability parameter schema does not match its family");
  for (auto [ordinal, schema] : llvm::enumerate(enabledSchemas)) {
    if (!llvm::is_contained(descriptor.admittedSchemas, schema))
      return reject("scalar integer capability enables a foreign schema");
    if (llvm::is_contained(enabledSchemas.take_front(ordinal), schema))
      return reject("scalar integer capability enables a schema twice");
  }

  std::vector<::dataflow::OperationSchemaId> orderedSchemas;
  for (::dataflow::OperationSchemaId schema : descriptor.admittedSchemas)
    if (llvm::is_contained(enabledSchemas, schema))
      orderedSchemas.push_back(schema);

  std::vector<ScalarIntegerBehaviorCandidate> candidates;
  if (family == ImplementationFamilyId::ScalarIntegerCompareMinMax) {
    const auto *typed = std::get_if<ScalarIntegerCompareMinMaxParams>(&params);
    if (!typed)
      return reject("capability has the wrong compare parameter schema");
    if (llvm::Error error = appendCompareCandidates(*typed, orderedSchemas,
                                                    context, candidates))
      return std::move(error);
  } else if (family == ImplementationFamilyId::ScalarIntegerCast) {
    const auto *typed = std::get_if<ScalarIntegerCastParams>(&params);
    if (!typed)
      return reject("capability has the wrong cast parameter schema");
    if (llvm::Error error =
            appendCastCandidates(*typed, orderedSchemas, context, candidates))
      return std::move(error);
  } else {
    switch (family) {
    case ImplementationFamilyId::ScalarIntegerAddSub:
    case ImplementationFamilyId::ScalarIntegerLogic:
    case ImplementationFamilyId::ScalarIntegerShift:
    case ImplementationFamilyId::ScalarSignedIntegerDivRem:
    case ImplementationFamilyId::ScalarUnsignedIntegerDivRem:
    case ImplementationFamilyId::ScalarIntegerSaturatingAddSub:
    case ImplementationFamilyId::ScalarIntegerCountZeros:
      break;
    default:
      return reject("family has no scalar integer finite quotient");
    }
    auto typed = requireScalarIntegerParams(params);
    if (!typed)
      return typed.takeError();
    if (llvm::Error error = appendOrdinaryCandidates(
            family, **typed, orderedSchemas, context, candidates))
      return std::move(error);
  }
  if (candidates.empty())
    return reject("scalar integer capability has no behavior candidate");

  std::vector<ScalarIntegerBehaviorCandidate> reachable;
  for (ScalarIntegerBehaviorCandidate &candidate : candidates) {
    llvm::Error admission =
        candidate.resolvedIndexWidth
            ? verifyImplementationFamilyAdmission(
                  family, &params, candidate.actor,
                  getResolvedIndexBitWidth(*candidate.resolvedIndexWidth))
            : verifyImplementationFamilyAdmission(family, &params,
                                                  candidate.actor);
    if (admission)
      return std::move(admission);
    if (llvm::Error error = verifyImplementationFamilyPortCorrespondence(
            family, candidate.actor, candidate.operandPorts,
            candidate.resultPorts))
      return std::move(error);
    auto physical = isPhysicallyReachable(candidate, physicalInputWidths,
                                          physicalResultWidths);
    if (!physical)
      return physical.takeError();
    if (*physical)
      reachable.push_back(std::move(candidate));
  }
  if (reachable.empty())
    return reject(
        "scalar integer capability has no physically reachable behavior");

  std::vector<FiniteImplementationFamilyBehaviorPoint> points;
  for (ScalarIntegerBehaviorCandidate &candidate : reachable) {
    auto key = encodeCandidate(family, reachable, candidate);
    if (!key)
      return key.takeError();
    const bool duplicate = llvm::any_of(points, [&](const auto &point) {
      return point.semanticConfiguration &&
             point.semanticConfiguration->bytes().equals(key->bytes());
    });
    if (duplicate)
      continue;
    points.push_back({std::move(candidate.actor), std::move(*key),
                      candidate.resolvedIndexWidth,
                      std::move(candidate.operandPorts),
                      std::move(candidate.resultPorts)});
  }
  llvm::sort(points, lessKey);
  if (points.size() == 1)
    points.front().semanticConfiguration = std::nullopt;
  return points;
}
