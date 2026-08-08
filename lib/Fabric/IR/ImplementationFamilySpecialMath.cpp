//===- ImplementationFamilySpecialMath.cpp -------------------------------===//

#include "ImplementationFamilySpecialMath.h"
#include "ImplementationFamilyBehaviorInternal.h"

#include "Dataflow/IR/OperationSchemaCodec.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

llvm::Error reject(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(), message);
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
makeActor(mlir::MLIRContext &context, fabric::FloatFormat format,
          ::dataflow::OperationSchemaId schema,
          mlir::arith::FastMathFlags flags,
          ::loom::SpecialMathAccuracyTier accuracy) {
  mlir::Type type = floatType(context, format);
  const unsigned inputCount =
      schema == ::dataflow::OperationSchemaId::MathPowF ? 2 : 1;
  return {
      schema,
      mlir::FunctionType::get(
          &context, llvm::SmallVector<mlir::Type, 2>(inputCount, type), {type}),
      ::dataflow::SpecialMathPayload{flags, accuracy}};
}

llvm::Error requireSpecialMathFamily(fabric::ImplementationFamilyId family) {
  if (fabric::implementationFamily(family).typedAdmissionProvider !=
      fabric::TypedAdmissionProviderId::ScalarSpecialMathAdmission)
    return reject("implementation family is not scalar special math");
  return llvm::Error::success();
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

mlir::arith::FastMathFlags
minimalActorFlags(const fabric::FloatBehaviorProfile &behavior) {
  mlir::arith::FastMathFlags flags = behavior.requiredFastMath;
  if (!behavior.nanBehaviors.contains(fabric::FloatNaNBehavior::IEEE))
    flags = addFastMathFlag(flags, mlir::arith::FastMathFlags::nnan);
  if (!behavior.signedZeroBehaviors.contains(
          fabric::FloatSignedZeroBehavior::Preserve))
    flags = addFastMathFlag(flags, mlir::arith::FastMathFlags::nsz);
  return flags;
}

llvm::Error validateSpecialMathBehaviorProfile(
    const fabric::FloatBehaviorProfile &behavior) {
  if (!behavior.roundingModes.valid() || behavior.roundingModes.size() != 1 ||
      !behavior.roundingModes.contains(
          mlir::arith::RoundingMode::to_nearest_even))
    return reject("special-math behavior requires exactly "
                  "round-to-nearest-even rounding");
  if (!behavior.nanBehaviors.valid() || behavior.nanBehaviors.size() != 1)
    return reject("special-math behavior requires exactly one NaN behavior");
  return llvm::Error::success();
}

bool lessPoint(const fabric::FiniteImplementationFamilyBehaviorPoint &lhs,
               const fabric::FiniteImplementationFamilyBehaviorPoint &rhs) {
  return std::lexicographical_compare(
      lhs.semanticConfiguration->bytes().begin(),
      lhs.semanticConfiguration->bytes().end(),
      rhs.semanticConfiguration->bytes().begin(),
      rhs.semanticConfiguration->bytes().end());
}

llvm::Expected<loom::CanonicalSemanticBytes>
encodeBehaviorKey(fabric::ImplementationFamilyId family, mlir::Type type) {
  auto encodedType = dataflow::encodeCanonicalType(type);
  if (!encodedType)
    return encodedType.takeError();
  const std::array<fabric::detail::ImplementationFamilyBehaviorKeyComponent, 1>
      components = {std::move(*encodedType)};
  return fabric::detail::encodeImplementationFamilyBehaviorKey(family, "",
                                                               components);
}

} // namespace

bool fabric::detail::ownsScalarSpecialMathBehaviorRelation(
    ImplementationFamilyId family) {
  const std::uint32_t ordinal = static_cast<std::uint32_t>(family);
  return ordinal < implementationFamilyCount() &&
         implementationFamily(family).typedAdmissionProvider ==
             TypedAdmissionProviderId::ScalarSpecialMathAdmission;
}

llvm::Expected<std::vector<fabric::FiniteImplementationFamilyBehaviorPoint>>
fabric::detail::resolveScalarSpecialMathBehaviorDomain(
    ImplementationFamilyId family, const FamilyCapabilityParams &params,
    llvm::ArrayRef<::dataflow::OperationSchemaId> enabledSchemas,
    llvm::ArrayRef<std::uint32_t> physicalInputWidths,
    llvm::ArrayRef<std::uint32_t> physicalResultWidths,
    mlir::MLIRContext &context) {
  if (!ownsScalarSpecialMathBehaviorRelation(family))
    return reject("implementation family is not scalar special math");
  const ImplementationFamilyDescriptor &descriptor =
      implementationFamily(family);
  if (capabilityParamsSchema(params) != descriptor.capabilityParamsSchema)
    return reject("capability has the wrong parameter schema");
  const auto *parameters = std::get_if<ScalarSpecialMathParams>(&params);
  if (!parameters)
    return reject("capability has the wrong parameter schema");
  if (enabledSchemas.empty())
    return reject("special-math capability has no enabled schema");
  for (auto [ordinal, schema] : llvm::enumerate(enabledSchemas)) {
    if (!llvm::is_contained(descriptor.admittedSchemas, schema))
      return reject("special-math capability enables a foreign schema");
    if (llvm::is_contained(enabledSchemas.take_front(ordinal), schema))
      return reject("special-math capability enables a schema twice");
  }
  if (descriptor.admittedSchemas.size() != 1 || enabledSchemas.size() != 1)
    return reject("special-math family requires its exact generated schema");
  if (llvm::Error error =
          validateSpecialMathBehaviorProfile(parameters->behavior))
    return std::move(error);

  const unsigned inputCount =
      enabledSchemas.front() == ::dataflow::OperationSchemaId::MathPowF ? 2 : 1;
  if (physicalInputWidths.size() != inputCount ||
      physicalResultWidths.size() != 1)
    return reject("special-math physical role inventory is incomplete");

  mlir::arith::FastMathFlags actorFlags =
      minimalActorFlags(parameters->behavior);
  if (parameters->accuracyGuarantee !=
      ::loom::SpecialMathAccuracyTier::CorrectlyRounded)
    actorFlags = addFastMathFlag(actorFlags, mlir::arith::FastMathFlags::afn);

  std::vector<FiniteImplementationFamilyBehaviorPoint> points;
  for (FloatFormat format : floatFormatDomain) {
    if (!parameters->formats.contains(format))
      continue;
    const std::uint32_t width = getBitWidth(format);
    if (llvm::any_of(
            physicalInputWidths,
            [&](std::uint32_t physical) { return physical < width; }) ||
        physicalResultWidths.front() < width)
      continue;

    ::dataflow::CanonicalActorSchemaProjection actor =
        makeActor(context, format, enabledSchemas.front(), actorFlags,
                  parameters->accuracyGuarantee);
    std::vector<std::uint64_t> operandPorts(inputCount);
    for (std::uint64_t ordinal = 0; ordinal != inputCount; ++ordinal)
      operandPorts[ordinal] = ordinal;
    std::vector<std::uint64_t> resultPorts = {0};
    if (llvm::Error error = validateImplementationFamilyBehaviorPoint(
            family, params, actor, operandPorts, resultPorts,
            physicalInputWidths, physicalResultWidths))
      return std::move(error);
    auto key = encodeBehaviorKey(family, actor.type.getInput(0));
    if (!key)
      return key.takeError();
    points.push_back({std::move(actor), std::move(*key), std::nullopt,
                      std::move(operandPorts), std::move(resultPorts)});
  }
  if (points.empty())
    return reject(
        "special-math capability has no physically reachable behavior");
  llvm::sort(points, lessPoint);
  if (points.size() == 1)
    points.front().semanticConfiguration = std::nullopt;
  return points;
}

llvm::Expected<loom::CanonicalSemanticBytes>
fabric::detail::projectScalarSpecialMathBehavior(
    ImplementationFamilyId family,
    const ::dataflow::CanonicalActorSchemaProjection &actor,
    llvm::ArrayRef<FiniteImplementationFamilyBehaviorPoint> domain) {
  if (!ownsScalarSpecialMathBehaviorRelation(family))
    return reject("capability family has no scalar special-math projector");
  if (!llvm::is_contained(implementationFamily(family).admittedSchemas,
                          actor.schema))
    return reject("actor schema is outside the special-math family");
  const auto *payload =
      std::get_if<::dataflow::SpecialMathPayload>(&actor.payload);
  if (!payload)
    return reject("special-math actor has no typed accuracy projection");
  if (llvm::Error error = ::loom::validateSpecialMathAccuracyContract(
          payload->accuracy,
          hasFastMathFlag(payload->flags, mlir::arith::FastMathFlags::afn)))
    return std::move(error);
  auto encodedActor = ::dataflow::encodeCanonicalActorSchemaProjection(actor);
  if (!encodedActor)
    return encodedActor.takeError();
  if (actor.type.getNumInputs() == 0 || actor.type.getNumResults() != 1)
    return reject("special-math actor has the wrong function type");
  for (const FiniteImplementationFamilyBehaviorPoint &point : domain) {
    if (actor.type.getInput(0) != point.representativeActor.type.getInput(0))
      continue;
    if (!point.semanticConfiguration)
      return reject("special-math relation has no semantic field");
    return *point.semanticConfiguration;
  }
  return reject("actor is outside the special-math behavior image");
}

llvm::Expected<loom::CanonicalSemanticBytes>
fabric::detail::encodeScalarSpecialMathSemanticConfiguration(
    ImplementationFamilyId family, const FamilyCapabilityParams &params,
    const ::dataflow::CanonicalActorSchemaProjection &actor) {
  if (llvm::Error error = requireSpecialMathFamily(family))
    return std::move(error);
  if (!std::get_if<ScalarSpecialMathParams>(&params))
    return reject("capability has the wrong parameter schema");
  const auto *payload =
      std::get_if<::dataflow::SpecialMathPayload>(&actor.payload);
  if (!payload)
    return reject("special-math actor has no typed accuracy projection");
  if (llvm::Error error = ::loom::validateSpecialMathAccuracyContract(
          payload->accuracy,
          hasFastMathFlag(payload->flags, mlir::arith::FastMathFlags::afn)))
    return std::move(error);
  if (llvm::Error error =
          verifyImplementationFamilyAdmission(family, &params, actor))
    return std::move(error);
  return ::dataflow::encodeCanonicalType(actor.type.getInput(0));
}

llvm::Expected<std::vector<dataflow::CanonicalActorSchemaProjection>>
fabric::detail::enumerateScalarSpecialMathBehaviorActors(
    ImplementationFamilyId family, const FamilyCapabilityParams &params,
    llvm::ArrayRef<::dataflow::OperationSchemaId> enabledSchemas,
    ::mlir::MLIRContext &context) {
  if (llvm::Error error = requireSpecialMathFamily(family))
    return std::move(error);
  const auto *parameters = std::get_if<ScalarSpecialMathParams>(&params);
  if (!parameters)
    return reject("capability has the wrong parameter schema");
  const bool permitsApproximateFunctions = hasFastMathFlag(
      parameters->behavior.requiredFastMath, mlir::arith::FastMathFlags::afn);
  if (llvm::Error error = ::loom::validateSpecialMathAccuracyContract(
          parameters->accuracyGuarantee, permitsApproximateFunctions))
    return std::move(error);
  std::vector<::dataflow::CanonicalActorSchemaProjection> actors;
  const mlir::arith::FastMathFlags baseActorFlags =
      minimalActorFlags(parameters->behavior);
  for (::dataflow::OperationSchemaId schema : enabledSchemas) {
    for (FloatFormat format : floatFormatDomain) {
      if (!parameters->formats.contains(format))
        continue;
      for (::loom::SpecialMathAccuracyTier accuracy :
           ::loom::specialMathAccuracyTiers()) {
        mlir::arith::FastMathFlags actorFlags = baseActorFlags;
        if (llvm::Error error = ::loom::validateSpecialMathAccuracyContract(
                accuracy,
                hasFastMathFlag(actorFlags, mlir::arith::FastMathFlags::afn))) {
          llvm::consumeError(std::move(error));
          actorFlags =
              addFastMathFlag(actorFlags, mlir::arith::FastMathFlags::afn);
          if (llvm::Error retry = ::loom::validateSpecialMathAccuracyContract(
                  accuracy,
                  hasFastMathFlag(actorFlags, mlir::arith::FastMathFlags::afn)))
            return std::move(retry);
        }
        auto refines = ::loom::specialMathAccuracyRefines(
            parameters->accuracyGuarantee, accuracy);
        if (!refines)
          return refines.takeError();
        if (!*refines)
          continue;
        actors.push_back(
            makeActor(context, format, schema, actorFlags, accuracy));
      }
    }
  }
  return actors;
}
