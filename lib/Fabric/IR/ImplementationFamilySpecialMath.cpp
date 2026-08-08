//===- ImplementationFamilySpecialMath.cpp -------------------------------===//

#include "ImplementationFamilySpecialMath.h"

#include "Dataflow/IR/OperationSchemaCodec.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"

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

} // namespace

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
