//===- ImplementationFamilyScalarFloatCompareBehavior.cpp ---------------===//
//
// Owns the scalar floating compare/minmax behavior refinement cover.
//
//===----------------------------------------------------------------------===//

#include "ImplementationFamilyScalarFloatCompareBehavior.h"

#include "ImplementationFamilyBehaviorInternal.h"

#include "Dataflow/IR/OperationSchemaCodec.h"

#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace {

using namespace fabric;
using ::dataflow::OperationSchemaId;

constexpr std::uint32_t kExactFormatTag = 1;
constexpr std::uint32_t kRepresentationWidthTag = 2;

llvm::Error reject(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(), message);
}

bool hasFastMathFlag(mlir::arith::FastMathFlags flags,
                     mlir::arith::FastMathFlags flag) {
  using Bits = std::underlying_type_t<mlir::arith::FastMathFlags>;
  return (static_cast<Bits>(flags) & static_cast<Bits>(flag)) != 0;
}

mlir::arith::FastMathFlags addFastMathFlag(mlir::arith::FastMathFlags flags,
                                           mlir::arith::FastMathFlags flag) {
  using Bits = std::underlying_type_t<mlir::arith::FastMathFlags>;
  return static_cast<mlir::arith::FastMathFlags>(static_cast<Bits>(flags) |
                                                 static_cast<Bits>(flag));
}

mlir::arith::FastMathFlags removeFastMathFlag(mlir::arith::FastMathFlags flags,
                                              mlir::arith::FastMathFlags flag) {
  using Bits = std::underlying_type_t<mlir::arith::FastMathFlags>;
  return static_cast<mlir::arith::FastMathFlags>(static_cast<Bits>(flags) &
                                                 ~static_cast<Bits>(flag));
}

mlir::Type floatType(mlir::MLIRContext &context, FloatFormat format) {
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
  llvm_unreachable("unknown floating format");
}

llvm::Expected<FloatFormat> formatOf(mlir::Type type) {
  if (llvm::isa<mlir::Float16Type>(type))
    return FloatFormat::F16;
  if (llvm::isa<mlir::BFloat16Type>(type))
    return FloatFormat::BF16;
  if (llvm::isa<mlir::Float32Type>(type))
    return FloatFormat::F32;
  if (llvm::isa<mlir::Float64Type>(type))
    return FloatFormat::F64;
  return reject("scalar floating compare behavior has a non-floating type");
}

std::vector<std::uint64_t> identityPorts(unsigned count) {
  std::vector<std::uint64_t> ports(count);
  for (std::uint64_t ordinal = 0; ordinal != count; ++ordinal)
    ports[ordinal] = ordinal;
  return ports;
}

bool isIEEEBehaviorSchema(OperationSchemaId schema) {
  return schema == OperationSchemaId::ArithCmpF ||
         schema == OperationSchemaId::ArithMinimumF ||
         schema == OperationSchemaId::ArithMaximumF;
}

bool isNumberPreferredSchema(OperationSchemaId schema) {
  return schema == OperationSchemaId::ArithMinNumF ||
         schema == OperationSchemaId::ArithMaxNumF;
}

bool isMinimumSchema(OperationSchemaId schema) {
  return schema == OperationSchemaId::ArithMinimumF ||
         schema == OperationSchemaId::ArithMinNumF;
}

bool isMaximumSchema(OperationSchemaId schema) {
  return schema == OperationSchemaId::ArithMaximumF ||
         schema == OperationSchemaId::ArithMaxNumF;
}

struct ValidatedCapability final {
  const ScalarFloatCompareMinMaxParams *params;
  std::vector<OperationSchemaId> orderedSchemas;
};

llvm::Expected<ValidatedCapability>
validateCapability(ImplementationFamilyId family,
                   const FamilyCapabilityParams &params,
                   llvm::ArrayRef<OperationSchemaId> enabledSchemas) {
  if (family != ImplementationFamilyId::ScalarFloatCompareMinMax)
    return reject("family is not scalar floating compare/minmax");
  const ImplementationFamilyDescriptor &descriptor =
      implementationFamily(family);
  if (capabilityParamsSchema(params) != descriptor.capabilityParamsSchema)
    return reject("capability parameter schema does not match its family");
  const auto *typed = std::get_if<ScalarFloatCompareMinMaxParams>(&params);
  if (!typed)
    return reject("capability has the wrong floating compare parameter schema");
  if (!typed->formats.valid() || typed->formats.empty())
    return reject("floating compare format domain is invalid");
  if (!typed->predicates.valid() || typed->predicates.empty())
    return reject("floating compare predicate domain is invalid");
  if (!typed->behavior.roundingModes.valid() ||
      typed->behavior.roundingModes.size() != 1 ||
      !typed->behavior.roundingModes.contains(
          mlir::arith::RoundingMode::to_nearest_even))
    return reject("floating compare rounding domain must contain only RNE");
  if (!typed->behavior.nanBehaviors.valid() ||
      typed->behavior.nanBehaviors.empty())
    return reject("floating compare NaN behavior domain is invalid");
  if (typed->behavior.nanBehaviors.size() > 1 &&
      hasFastMathFlag(typed->behavior.requiredFastMath,
                      mlir::arith::FastMathFlags::nnan))
    return reject(
        "multiple NaN behaviors have no observable actor distinction");
  if (enabledSchemas.empty())
    return reject("floating compare capability has no enabled schema");

  for (auto [ordinal, schema] : llvm::enumerate(enabledSchemas)) {
    if (!llvm::is_contained(descriptor.admittedSchemas, schema))
      return reject("floating compare capability enables a foreign schema");
    if (llvm::is_contained(enabledSchemas.take_front(ordinal), schema))
      return reject("floating compare capability enables a schema twice");
  }

  const bool observesIEEE = llvm::any_of(enabledSchemas, isIEEEBehaviorSchema);
  const bool observesNumberPreferred =
      llvm::any_of(enabledSchemas, isNumberPreferredSchema);
  if (typed->behavior.nanBehaviors.contains(FloatNaNBehavior::IEEE) &&
      !observesIEEE)
    return reject("IEEE NaN behavior has no enabled actor role");
  if (typed->behavior.nanBehaviors.contains(
          FloatNaNBehavior::NumberPreferred) &&
      !observesNumberPreferred)
    return reject("number-preferred NaN behavior has no enabled actor role");

  const bool hasCompare =
      llvm::is_contained(enabledSchemas, OperationSchemaId::ArithCmpF);
  const bool hasMinimum = llvm::any_of(enabledSchemas, isMinimumSchema);
  const bool hasMaximum = llvm::any_of(enabledSchemas, isMaximumSchema);
  if (hasMinimum &&
      !typed->predicates.contains(mlir::arith::CmpFPredicate::OLT))
    return reject("minimum role has no admitted OLT predicate");
  if (hasMaximum &&
      !typed->predicates.contains(mlir::arith::CmpFPredicate::OGT))
    return reject("maximum role has no admitted OGT predicate");
  if (!hasCompare) {
    for (std::uint32_t ordinal = 0;
         ordinal <= mlir::arith::getMaxEnumValForCmpFPredicate(); ++ordinal) {
      const auto predicate = static_cast<mlir::arith::CmpFPredicate>(ordinal);
      if (!typed->predicates.contains(predicate))
        continue;
      const bool interpreted =
          (predicate == mlir::arith::CmpFPredicate::OLT && hasMinimum) ||
          (predicate == mlir::arith::CmpFPredicate::OGT && hasMaximum);
      if (!interpreted)
        return reject("floating predicate has no enabled actor role");
    }
  }

  std::vector<OperationSchemaId> orderedSchemas;
  for (OperationSchemaId schema : descriptor.admittedSchemas)
    if (llvm::is_contained(enabledSchemas, schema))
      orderedSchemas.push_back(schema);
  return ValidatedCapability{typed, std::move(orderedSchemas)};
}

std::string rawRole(OperationSchemaId schema) {
  switch (schema) {
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
  default:
    llvm_unreachable("foreign floating compare schema");
  }
}

FloatNaNBehavior rawNaNBehavior(OperationSchemaId schema) {
  return isNumberPreferredSchema(schema) ? FloatNaNBehavior::NumberPreferred
                                         : FloatNaNBehavior::IEEE;
}

mlir::arith::FastMathFlags
minimalActorPermissionsForSchema(const FloatBehaviorProfile &behavior,
                                 OperationSchemaId schema) {
  mlir::arith::FastMathFlags flags =
      detail::minimalFloatingActorPermissions(behavior);
  const bool requiresNnan = hasFastMathFlag(behavior.requiredFastMath,
                                            mlir::arith::FastMathFlags::nnan);
  const bool admitsRawNaNBehavior =
      behavior.nanBehaviors.contains(rawNaNBehavior(schema));
  if (!requiresNnan && admitsRawNaNBehavior)
    return removeFastMathFlag(flags, mlir::arith::FastMathFlags::nnan);
  return addFastMathFlag(flags, mlir::arith::FastMathFlags::nnan);
}

mlir::arith::CmpFPredicate implicitPredicate(OperationSchemaId schema) {
  return isMinimumSchema(schema) ? mlir::arith::CmpFPredicate::OLT
                                 : mlir::arith::CmpFPredicate::OGT;
}

::dataflow::CanonicalActorSchemaProjection
makeActor(mlir::MLIRContext &context, FloatFormat format,
          OperationSchemaId schema,
          std::optional<mlir::arith::CmpFPredicate> predicate,
          mlir::arith::FastMathFlags flags) {
  mlir::Type operand = floatType(context, format);
  if (schema == OperationSchemaId::ArithCmpF)
    return {schema,
            mlir::FunctionType::get(&context, {operand, operand},
                                    {mlir::IntegerType::get(&context, 1)}),
            ::dataflow::FloatComparePayload{*predicate, flags}};
  return {schema,
          mlir::FunctionType::get(&context, {operand, operand}, {operand}),
          ::dataflow::FloatingPointPayload{flags, std::nullopt}};
}

struct BehaviorCandidate final {
  ::dataflow::CanonicalActorSchemaProjection actor;
  std::string role;
  std::optional<mlir::arith::CmpFPredicate> predicate;
  FloatFormat format;
  bool weak;
  std::vector<std::uint8_t> canonicalActor;
  std::vector<std::uint64_t> operandPorts = {0, 1};
  std::vector<std::uint64_t> resultPorts = {0};
};

llvm::Expected<BehaviorCandidate>
describeActor(const ::dataflow::CanonicalActorSchemaProjection &actor) {
  if (actor.type.getNumInputs() != 2 || actor.type.getNumResults() != 1)
    return reject("floating compare behavior actor has the wrong arity");
  auto format = formatOf(actor.type.getInput(0));
  if (!format)
    return format.takeError();

  std::optional<mlir::arith::CmpFPredicate> predicate;
  mlir::arith::FastMathFlags flags;
  if (actor.schema == OperationSchemaId::ArithCmpF) {
    const auto *payload =
        std::get_if<::dataflow::FloatComparePayload>(&actor.payload);
    if (!payload)
      return reject("floating comparison has no typed predicate payload");
    predicate = payload->predicate;
    flags = payload->flags;
  } else if (isIEEEBehaviorSchema(actor.schema) ||
             isNumberPreferredSchema(actor.schema)) {
    const auto *payload =
        std::get_if<::dataflow::FloatingPointPayload>(&actor.payload);
    if (!payload || payload->roundingMode)
      return reject("floating min/max has a noncanonical payload");
    predicate = std::nullopt;
    flags = payload->flags;
  } else {
    return reject("actor schema is outside floating compare/minmax");
  }

  auto canonical = ::dataflow::encodeCanonicalActorSchemaProjection(actor);
  if (!canonical)
    return canonical.takeError();
  return BehaviorCandidate{
      actor,
      rawRole(actor.schema),
      predicate,
      *format,
      hasFastMathFlag(flags, mlir::arith::FastMathFlags::nnan),
      std::vector<std::uint8_t>(canonical->bytes().begin(),
                                canonical->bytes().end())};
}

llvm::Expected<bool>
isPhysicallyReachable(const BehaviorCandidate &candidate,
                      llvm::ArrayRef<std::uint32_t> physicalInputWidths,
                      llvm::ArrayRef<std::uint32_t> physicalResultWidths) {
  const auto fits =
      [](llvm::ArrayRef<mlir::Type> types,
         llvm::ArrayRef<std::uint32_t> widths) -> llvm::Expected<bool> {
    if (types.size() > widths.size())
      return false;
    for (auto [type, width] : llvm::zip(types, widths)) {
      std::string message;
      auto required = getSemanticPayloadWidth(type, message);
      if (mlir::failed(required))
        return reject(message);
      if (*required > width)
        return false;
    }
    return true;
  };
  auto inputs = fits(candidate.actor.type.getInputs(), physicalInputWidths);
  if (!inputs || !*inputs)
    return inputs;
  return fits(candidate.actor.type.getResults(), physicalResultWidths);
}

bool isConstantPredicate(mlir::arith::CmpFPredicate predicate);

llvm::Expected<std::vector<BehaviorCandidate>>
enumerateCandidates(ImplementationFamilyId family,
                    const FamilyCapabilityParams &params,
                    const ValidatedCapability &capability,
                    llvm::ArrayRef<std::uint32_t> physicalInputWidths,
                    llvm::ArrayRef<std::uint32_t> physicalResultWidths,
                    mlir::MLIRContext &context) {
  std::vector<BehaviorCandidate> candidates;
  for (OperationSchemaId schema : capability.orderedSchemas) {
    const mlir::arith::FastMathFlags baseFlags =
        minimalActorPermissionsForSchema(capability.params->behavior, schema);
    for (FloatFormat format : floatFormatDomain) {
      if (!capability.params->formats.contains(format))
        continue;
      const auto appendFlags =
          [&](std::optional<mlir::arith::CmpFPredicate> predicate)
          -> llvm::Error {
        std::array<mlir::arith::FastMathFlags, 2> flags = {
            baseFlags,
            addFastMathFlag(baseFlags, mlir::arith::FastMathFlags::nnan)};
        for (auto [ordinal, actorFlags] : llvm::enumerate(flags)) {
          const bool weak =
              hasFastMathFlag(actorFlags, mlir::arith::FastMathFlags::nnan);
          if ((ordinal == 1 && flags[0] == flags[1]) ||
              (!weak && !capability.params->behavior.nanBehaviors.contains(
                            rawNaNBehavior(schema))))
            continue;
          auto described = describeActor(
              makeActor(context, format, schema, predicate, actorFlags));
          if (!described)
            return described.takeError();
          if (llvm::Error error = verifyImplementationFamilyAdmission(
                  family, &params, described->actor))
            return error;
          if (llvm::Error error = verifyImplementationFamilyPortCorrespondence(
                  family, params, described->actor, described->operandPorts,
                  described->resultPorts, physicalInputWidths,
                  physicalResultWidths))
            return error;
          auto reachable = isPhysicallyReachable(
              *described, physicalInputWidths, physicalResultWidths);
          if (!reachable)
            return reachable.takeError();
          if (*reachable)
            candidates.push_back(std::move(*described));
        }
        return llvm::Error::success();
      };

      if (schema == OperationSchemaId::ArithCmpF) {
        for (std::uint32_t ordinal = 0;
             ordinal <= mlir::arith::getMaxEnumValForCmpFPredicate();
             ++ordinal) {
          const auto predicate =
              static_cast<mlir::arith::CmpFPredicate>(ordinal);
          if (capability.params->predicates.contains(predicate))
            if (llvm::Error error = appendFlags(predicate))
              return std::move(error);
        }
      } else if (llvm::Error error = appendFlags(implicitPredicate(schema))) {
        return std::move(error);
      }
    }
  }
  if (candidates.empty())
    return reject(
        "floating compare capability has no physically reachable behavior");
  llvm::sort(candidates,
             [](const BehaviorCandidate &lhs, const BehaviorCandidate &rhs) {
               if (lhs.weak != rhs.weak)
                 return !lhs.weak;
               return std::lexicographical_compare(
                   lhs.canonicalActor.begin(), lhs.canonicalActor.end(),
                   rhs.canonicalActor.begin(), rhs.canonicalActor.end());
             });
  return candidates;
}

llvm::Error
validateReachableNaNBehaviors(const ScalarFloatCompareMinMaxParams &params,
                              llvm::ArrayRef<BehaviorCandidate> candidates) {
  const auto hasReachableRole = [&](FloatNaNBehavior behavior) {
    return llvm::any_of(candidates, [&](const BehaviorCandidate &candidate) {
      return rawNaNBehavior(candidate.actor.schema) == behavior;
    });
  };
  if (params.behavior.nanBehaviors.contains(FloatNaNBehavior::IEEE) &&
      !hasReachableRole(FloatNaNBehavior::IEEE))
    return reject("IEEE NaN behavior has no physically reachable actor role");
  if (params.behavior.nanBehaviors.contains(
          FloatNaNBehavior::NumberPreferred) &&
      !hasReachableRole(FloatNaNBehavior::NumberPreferred))
    return reject(
        "number-preferred NaN behavior has no physically reachable actor role");

  if (params.behavior.nanBehaviors.size() <= 1)
    return llvm::Error::success();
  const auto hasObservableBehavior = [&](FloatNaNBehavior behavior) {
    return llvm::any_of(candidates, [&](const BehaviorCandidate &candidate) {
      const bool constantCompare =
          candidate.predicate && isConstantPredicate(*candidate.predicate);
      return !candidate.weak && !constantCompare &&
             rawNaNBehavior(candidate.actor.schema) == behavior;
    });
  };
  if (!hasObservableBehavior(FloatNaNBehavior::IEEE) ||
      !hasObservableBehavior(FloatNaNBehavior::NumberPreferred))
    return reject(
        "multiple NaN behaviors have no observable actor distinction");
  return llvm::Error::success();
}

mlir::arith::CmpFPredicate
normalizePredicate(mlir::arith::CmpFPredicate predicate) {
  using Predicate = mlir::arith::CmpFPredicate;
  switch (predicate) {
  case Predicate::UEQ:
    return Predicate::OEQ;
  case Predicate::UGT:
    return Predicate::OGT;
  case Predicate::UGE:
    return Predicate::OGE;
  case Predicate::ULT:
    return Predicate::OLT;
  case Predicate::ULE:
    return Predicate::OLE;
  case Predicate::UNE:
    return Predicate::ONE;
  case Predicate::ORD:
    return Predicate::AlwaysTrue;
  case Predicate::UNO:
    return Predicate::AlwaysFalse;
  case Predicate::AlwaysFalse:
  case Predicate::OEQ:
  case Predicate::OGT:
  case Predicate::OGE:
  case Predicate::OLT:
  case Predicate::OLE:
  case Predicate::ONE:
  case Predicate::AlwaysTrue:
    return predicate;
  }
  llvm_unreachable("unknown floating predicate");
}

bool isConstantPredicate(mlir::arith::CmpFPredicate predicate) {
  return predicate == mlir::arith::CmpFPredicate::AlwaysFalse ||
         predicate == mlir::arith::CmpFPredicate::AlwaysTrue;
}

std::string normalizedRole(const BehaviorCandidate &candidate) {
  if (candidate.role == "MinNumber")
    return "Minimum";
  if (candidate.role == "MaxNumber")
    return "Maximum";
  return candidate.role;
}

std::optional<mlir::arith::CmpFPredicate>
normalizedPredicate(const BehaviorCandidate &candidate) {
  if (!candidate.predicate)
    return std::nullopt;
  return normalizePredicate(*candidate.predicate);
}

struct NumericFormat final {
  enum class Kind : std::uint8_t { None, Exact, Width };
  Kind kind = Kind::None;
  FloatFormat format = FloatFormat::F16;
  std::uint32_t width = 0;

  friend bool operator==(const NumericFormat &lhs, const NumericFormat &rhs) {
    if (lhs.kind != rhs.kind)
      return false;
    if (lhs.kind == Kind::Exact)
      return lhs.format == rhs.format;
    if (lhs.kind == Kind::Width)
      return lhs.width == rhs.width;
    return true;
  }
};

struct LogicalBehavior final {
  std::string role;
  std::optional<mlir::arith::CmpFPredicate> predicate;
  NumericFormat numericFormat;

  friend bool operator==(const LogicalBehavior &lhs,
                         const LogicalBehavior &rhs) {
    return lhs.role == rhs.role && lhs.predicate == rhs.predicate &&
           lhs.numericFormat == rhs.numericFormat;
  }
};

LogicalBehavior exactBehavior(const BehaviorCandidate &candidate) {
  NumericFormat numeric;
  if (!candidate.predicate || !isConstantPredicate(*candidate.predicate))
    numeric = {NumericFormat::Kind::Exact, candidate.format, 0};
  return {candidate.role, candidate.predicate, numeric};
}

LogicalBehavior normalizedBehavior(const BehaviorCandidate &candidate) {
  const auto predicate = normalizedPredicate(candidate);
  NumericFormat numeric;
  if (!predicate || !isConstantPredicate(*predicate))
    numeric = {NumericFormat::Kind::Width, FloatFormat::F16,
               getBitWidth(candidate.format)};
  return {normalizedRole(candidate), predicate, numeric};
}

bool strictFormatRefinesWeakFormat(FloatFormat strictFormat,
                                   FloatFormat weakFormat) {
  if (strictFormat == weakFormat)
    return true;
  return strictFormat == FloatFormat::BF16 && weakFormat == FloatFormat::F16;
}

struct CoverRepresentative final {
  std::size_t candidateIndex;
  LogicalBehavior behavior;
  std::optional<::loom::CanonicalSemanticBytes> key;
};

bool representativeRefinesWeak(const CoverRepresentative &representative,
                               const BehaviorCandidate &witness,
                               const BehaviorCandidate &weak) {
  if (normalizedRole(witness) != normalizedRole(weak) ||
      normalizedPredicate(witness) != normalizedPredicate(weak))
    return false;
  switch (representative.behavior.numericFormat.kind) {
  case NumericFormat::Kind::None:
    return true;
  case NumericFormat::Kind::Exact:
    return strictFormatRefinesWeakFormat(
        representative.behavior.numericFormat.format, weak.format);
  case NumericFormat::Kind::Width:
    return representative.behavior.numericFormat.width ==
           getBitWidth(weak.format);
  }
  llvm_unreachable("unknown numeric format kind");
}

std::vector<CoverRepresentative>
buildCover(llvm::ArrayRef<BehaviorCandidate> candidates) {
  std::vector<CoverRepresentative> cover;
  for (std::size_t index = 0; index != candidates.size(); ++index) {
    const BehaviorCandidate &candidate = candidates[index];
    if (candidate.weak)
      continue;
    LogicalBehavior behavior = exactBehavior(candidate);
    if (!llvm::any_of(cover, [&](const CoverRepresentative &representative) {
          return representative.behavior == behavior;
        }))
      cover.push_back({index, std::move(behavior), std::nullopt});
  }
  for (std::size_t index = 0; index != candidates.size(); ++index) {
    const BehaviorCandidate &candidate = candidates[index];
    if (!candidate.weak)
      continue;
    if (llvm::any_of(cover, [&](const CoverRepresentative &representative) {
          return representativeRefinesWeak(
              representative, candidates[representative.candidateIndex],
              candidate);
        }))
      continue;
    cover.push_back({index, normalizedBehavior(candidate), std::nullopt});
  }
  return cover;
}

bool predicateVaries(llvm::ArrayRef<CoverRepresentative> cover,
                     llvm::StringRef role) {
  std::optional<mlir::arith::CmpFPredicate> first;
  bool found = false;
  for (const CoverRepresentative &representative : cover) {
    if (representative.behavior.role != role ||
        !representative.behavior.predicate)
      continue;
    if (!found) {
      first = representative.behavior.predicate;
      found = true;
    } else if (first != representative.behavior.predicate) {
      return true;
    }
  }
  return false;
}

bool numericFormatVaries(llvm::ArrayRef<CoverRepresentative> cover,
                         llvm::StringRef role) {
  std::optional<NumericFormat> first;
  for (const CoverRepresentative &representative : cover) {
    if (representative.behavior.role != role ||
        representative.behavior.numericFormat.kind == NumericFormat::Kind::None)
      continue;
    if (!first)
      first = representative.behavior.numericFormat;
    else if (!(*first == representative.behavior.numericFormat))
      return true;
  }
  return false;
}

bool roleVaries(llvm::ArrayRef<CoverRepresentative> cover) {
  return llvm::any_of(cover, [&](const CoverRepresentative &representative) {
    return representative.behavior.role != cover.front().behavior.role;
  });
}

llvm::Expected<::loom::CanonicalSemanticBytes>
encodeRepresentative(ImplementationFamilyId family,
                     llvm::ArrayRef<CoverRepresentative> cover,
                     const CoverRepresentative &representative,
                     const BehaviorCandidate &witness) {
  std::vector<detail::ImplementationFamilyBehaviorKeyComponent> components;
  if (representative.behavior.predicate &&
      predicateVaries(cover, representative.behavior.role)) {
    auto predicate = ::dataflow::encodeFloatComparePredicate(
        *representative.behavior.predicate);
    if (!predicate)
      return predicate.takeError();
    components.emplace_back(std::move(*predicate));
  }
  if (representative.behavior.numericFormat.kind != NumericFormat::Kind::None &&
      numericFormatVaries(cover, representative.behavior.role)) {
    if (representative.behavior.numericFormat.kind ==
        NumericFormat::Kind::Exact) {
      components.emplace_back(kExactFormatTag);
      auto type =
          ::dataflow::encodeCanonicalType(witness.actor.type.getInput(0));
      if (!type)
        return type.takeError();
      components.emplace_back(std::move(*type));
    } else {
      components.emplace_back(kRepresentationWidthTag);
      components.emplace_back(representative.behavior.numericFormat.width);
    }
  }
  return detail::encodeImplementationFamilyBehaviorKey(
      family, roleVaries(cover) ? representative.behavior.role : "",
      components);
}

bool lessBytes(llvm::ArrayRef<std::uint8_t> lhs,
               llvm::ArrayRef<std::uint8_t> rhs) {
  return std::lexicographical_compare(lhs.begin(), lhs.end(), rhs.begin(),
                                      rhs.end());
}

llvm::Error encodeKeys(ImplementationFamilyId family,
                       llvm::ArrayRef<BehaviorCandidate> candidates,
                       std::vector<CoverRepresentative> &cover) {
  for (CoverRepresentative &representative : cover) {
    auto key = encodeRepresentative(family, cover, representative,
                                    candidates[representative.candidateIndex]);
    if (!key)
      return key.takeError();
    representative.key = std::move(*key);
  }
  return llvm::Error::success();
}

llvm::Error assignKeys(ImplementationFamilyId family,
                       llvm::ArrayRef<BehaviorCandidate> candidates,
                       std::vector<CoverRepresentative> &cover) {
  if (llvm::Error error = encodeKeys(family, candidates, cover))
    return error;
  llvm::sort(cover, [](const CoverRepresentative &lhs,
                       const CoverRepresentative &rhs) {
    return lessBytes(lhs.key->bytes(), rhs.key->bytes());
  });
  for (auto pair : llvm::zip(cover, llvm::drop_begin(cover))) {
    const auto &lhs = std::get<0>(pair);
    const auto &rhs = std::get<1>(pair);
    if (lhs.key->bytes().equals(rhs.key->bytes()))
      return reject("floating compare cover contains a duplicate key");
  }
  return llvm::Error::success();
}

llvm::Expected<std::vector<CoverRepresentative>>
validateDomain(ImplementationFamilyId family,
               const FamilyCapabilityParams &params,
               const ValidatedCapability &capability,
               llvm::ArrayRef<FiniteImplementationFamilyBehaviorPoint> domain) {
  if (domain.empty())
    return reject("floating compare behavior domain is empty");
  std::vector<BehaviorCandidate> witnesses;
  witnesses.reserve(domain.size());
  for (const FiniteImplementationFamilyBehaviorPoint &point : domain) {
    if (point.resolvedIndexWidth)
      return reject("floating compare behavior has an index-width selector");
    auto witness = describeActor(point.representativeActor);
    if (!witness)
      return witness.takeError();
    if (!llvm::is_contained(capability.orderedSchemas,
                            point.representativeActor.schema))
      return reject("floating compare representative schema is not enabled");
    if (llvm::Error error = verifyImplementationFamilyAdmission(
            family, &params, point.representativeActor))
      return std::move(error);
    if (point.operandPorts != identityPorts(2) ||
        point.resultPorts != identityPorts(1))
      return reject("floating compare representative changes a physical role");
    witnesses.push_back(std::move(*witness));
  }

  std::vector<CoverRepresentative> cover;
  for (auto [index, witness] : llvm::enumerate(witnesses)) {
    cover.push_back(
        {index,
         witness.weak ? normalizedBehavior(witness) : exactBehavior(witness),
         std::nullopt});
  }
  for (std::size_t weakIndex = 0; weakIndex != witnesses.size(); ++weakIndex) {
    if (!witnesses[weakIndex].weak)
      continue;
    for (std::size_t representativeIndex = 0;
         representativeIndex != cover.size(); ++representativeIndex) {
      if (representativeIndex == weakIndex)
        continue;
      if (representativeRefinesWeak(cover[representativeIndex],
                                    witnesses[representativeIndex],
                                    witnesses[weakIndex]))
        return reject(
            "floating compare relation contains a redundant weak mode");
    }
  }
  if (llvm::Error error = encodeKeys(family, witnesses, cover))
    return std::move(error);

  if (domain.size() == 1) {
    if (domain.front().semanticConfiguration)
      return reject("singleton floating compare relation did not collapse");
    cover.front().key = std::nullopt;
    return cover;
  }
  for (auto [ordinal, point] : llvm::enumerate(domain)) {
    if (!point.semanticConfiguration)
      return reject("non-singleton floating compare relation has no key");
    if (!point.semanticConfiguration->bytes().equals(
            cover[ordinal].key->bytes()))
      return reject(
          "floating compare representative/key binding is noncanonical");
    if (ordinal != 0 &&
        !lessBytes(domain[ordinal - 1].semanticConfiguration->bytes(),
                   point.semanticConfiguration->bytes()))
      return reject("floating compare relation key order is noncanonical");
  }
  return cover;
}

bool sameUnrefinedExactBehavior(const BehaviorCandidate &representative,
                                const BehaviorCandidate &weak) {
  return !representative.weak && representative.role == weak.role &&
         representative.predicate == weak.predicate &&
         representative.format == weak.format;
}

} // namespace

bool fabric::detail::ownsScalarFloatCompareBehaviorRelation(
    ImplementationFamilyId family) {
  return family == ImplementationFamilyId::ScalarFloatCompareMinMax;
}

llvm::Expected<std::vector<fabric::FiniteImplementationFamilyBehaviorPoint>>
fabric::detail::resolveScalarFloatCompareBehaviorDomain(
    ImplementationFamilyId family, const FamilyCapabilityParams &params,
    llvm::ArrayRef<::dataflow::OperationSchemaId> enabledSchemas,
    llvm::ArrayRef<std::uint32_t> physicalInputWidths,
    llvm::ArrayRef<std::uint32_t> physicalResultWidths,
    mlir::MLIRContext &context) {
  auto capability = validateCapability(family, params, enabledSchemas);
  if (!capability)
    return capability.takeError();
  auto candidates =
      enumerateCandidates(family, params, *capability, physicalInputWidths,
                          physicalResultWidths, context);
  if (!candidates)
    return candidates.takeError();
  if (llvm::Error error =
          validateReachableNaNBehaviors(*capability->params, *candidates))
    return std::move(error);
  for (::dataflow::OperationSchemaId schema : capability->orderedSchemas)
    if (!llvm::any_of(*candidates, [&](const BehaviorCandidate &candidate) {
          return candidate.actor.schema == schema;
        }))
      return reject(
          "enabled schema has no physically reachable floating compare actor");
  std::vector<CoverRepresentative> cover = buildCover(*candidates);
  if (llvm::Error error = assignKeys(family, *candidates, cover))
    return std::move(error);

  std::vector<FiniteImplementationFamilyBehaviorPoint> points;
  points.reserve(cover.size());
  for (CoverRepresentative &representative : cover) {
    const BehaviorCandidate &witness =
        (*candidates)[representative.candidateIndex];
    points.emplace_back(witness.actor, std::move(representative.key),
                        std::nullopt, witness.operandPorts,
                        witness.resultPorts);
  }
  if (points.size() == 1)
    points.front().semanticConfiguration = std::nullopt;
  return points;
}

llvm::Expected<::loom::CanonicalSemanticBytes>
fabric::detail::projectScalarFloatCompareBehavior(
    ImplementationFamilyId family, const FamilyCapabilityParams &params,
    llvm::ArrayRef<::dataflow::OperationSchemaId> enabledSchemas,
    const ::dataflow::CanonicalActorSchemaProjection &actor,
    llvm::ArrayRef<FiniteImplementationFamilyBehaviorPoint> domain) {
  auto capability = validateCapability(family, params, enabledSchemas);
  if (!capability)
    return capability.takeError();
  if (!llvm::is_contained(capability->orderedSchemas, actor.schema))
    return reject("actor schema is not enabled by the concrete capability");
  if (llvm::Error error =
          verifyImplementationFamilyAdmission(family, &params, actor))
    return std::move(error);
  auto projected = describeActor(actor);
  if (!projected)
    return projected.takeError();
  auto cover = validateDomain(family, params, *capability, domain);
  if (!cover)
    return cover.takeError();

  std::optional<std::size_t> selected;
  if (!projected->weak) {
    const LogicalBehavior behavior = exactBehavior(*projected);
    for (const CoverRepresentative &representative : *cover) {
      if (representative.behavior == behavior) {
        selected = representative.candidateIndex;
        break;
      }
    }
  } else {
    for (const CoverRepresentative &representative : *cover) {
      auto witness = describeActor(
          domain[representative.candidateIndex].representativeActor);
      if (!witness)
        return witness.takeError();
      if (representativeRefinesWeak(representative, *witness, *projected) &&
          sameUnrefinedExactBehavior(*witness, *projected)) {
        selected = representative.candidateIndex;
        break;
      }
    }
    if (!selected) {
      for (const CoverRepresentative &representative : *cover) {
        auto witness = describeActor(
            domain[representative.candidateIndex].representativeActor);
        if (!witness)
          return witness.takeError();
        if (representativeRefinesWeak(representative, *witness, *projected)) {
          selected = representative.candidateIndex;
          break;
        }
      }
    }
  }
  if (!selected)
    return reject("actor is outside the floating compare behavior image");
  if (!domain[*selected].semanticConfiguration)
    return reject("floating compare relation has no semantic field");
  return *domain[*selected].semanticConfiguration;
}
