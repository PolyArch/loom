//===- SpecialMathImplementationFamilyTest.cpp ---------------------------===//

#include "Common/SpecialMathAccuracy.h"
#include "Dataflow/IR/OperationSchema.h"
#include "Fabric/IR/ImplementationFamily.h"
#include "ImplementationFamilySpecialMath.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <array>
#include <optional>
#include <string>
#include <utility>
#include <vector>

using namespace mlir;
using namespace fabric;

namespace {

using dataflow::CanonicalActorSchemaProjection;
using dataflow::OperationSchemaId;
using dataflow::SemanticPayload;
using loom::SpecialMathAccuracyTier;

bool expectError(llvm::Error error, llvm::StringRef expected) {
  if (!error) {
    llvm::errs() << "expected failure containing '" << expected << "'\n";
    return false;
  }
  std::string message = llvm::toString(std::move(error));
  if (!llvm::StringRef(message).contains(expected)) {
    llvm::errs() << "unexpected failure: " << message << '\n';
    return false;
  }
  return true;
}

template <typename T>
bool expectFailure(llvm::Expected<T> result, llvm::StringRef expected) {
  if (result) {
    llvm::errs() << "expected failure containing '" << expected << "'\n";
    return false;
  }
  return expectError(result.takeError(), expected);
}

CanonicalActorSchemaProjection makeSinActor(MLIRContext &context, Type type,
                                            SemanticPayload payload) {
  return {OperationSchemaId::MathSin,
          FunctionType::get(&context, {type}, {type}), std::move(payload)};
}

CanonicalActorSchemaProjection makePowActor(MLIRContext &context, Type type,
                                            SemanticPayload payload) {
  return {OperationSchemaId::MathPowF,
          FunctionType::get(&context, {type, type}, {type}),
          std::move(payload)};
}

bool hasFastMathFlag(arith::FastMathFlags flags, arith::FastMathFlags flag) {
  using Bits = std::underlying_type_t<arith::FastMathFlags>;
  return (static_cast<Bits>(flags) & static_cast<Bits>(flag)) != 0;
}

FloatBehaviorProfile approximateBehavior() {
  FloatBehaviorProfile behavior = FloatBehaviorProfile::strictIEEE();
  behavior.requiredFastMath = arith::FastMathFlags::afn;
  return behavior;
}

FloatBehaviorProfile relaxedOnlyBehavior() {
  FloatBehaviorProfile behavior = FloatBehaviorProfile::strictIEEE();
  behavior.nanBehaviors =
      FloatNaNBehaviorSet::get({FloatNaNBehavior::NumberPreferred});
  behavior.signedZeroBehaviors =
      FloatSignedZeroBehaviorSet::get({FloatSignedZeroBehavior::IgnoreSign});
  return behavior;
}

FloatBehaviorProfile multiValuedBehavior() {
  FloatBehaviorProfile behavior = FloatBehaviorProfile::strictIEEE();
  behavior.roundingModes = RoundingModeSet::get(
      {arith::RoundingMode::to_nearest_even, arith::RoundingMode::downward});
  behavior.nanBehaviors = FloatNaNBehaviorSet::get(
      {FloatNaNBehavior::IEEE, FloatNaNBehavior::NumberPreferred});
  behavior.subnormalBehaviors = FloatSubnormalBehaviorSet::get(
      {FloatSubnormalBehavior::Preserve, FloatSubnormalBehavior::FlushToZero});
  behavior.signedZeroBehaviors = FloatSignedZeroBehaviorSet::get(
      {FloatSignedZeroBehavior::Preserve, FloatSignedZeroBehavior::IgnoreSign});
  return behavior;
}

FloatBehaviorProfile downwardOnlyBehavior() {
  FloatBehaviorProfile behavior = FloatBehaviorProfile::strictIEEE();
  behavior.roundingModes =
      RoundingModeSet::get({arith::RoundingMode::downward});
  return behavior;
}

FamilyCapabilityParams makeCapability(FloatFormatSet formats,
                                      FloatBehaviorProfile behavior,
                                      SpecialMathAccuracyTier guarantee) {
  return ScalarSpecialMathParams{formats, behavior, guarantee};
}

bool checkCentralAccuracyDomain() {
  llvm::ArrayRef<SpecialMathAccuracyTier> tiers =
      loom::specialMathAccuracyTiers();
  if (tiers.size() != 4) {
    llvm::errs() << "special-math accuracy domain is not closed\n";
    return false;
  }

  std::vector<loom::CanonicalSemanticBytes> encodings;
  for (SpecialMathAccuracyTier tier : tiers) {
    auto encoded = loom::encodeSpecialMathAccuracyTier(tier);
    if (!encoded) {
      llvm::errs() << llvm::toString(encoded.takeError()) << '\n';
      return false;
    }
    auto decoded = loom::decodeSpecialMathAccuracyTier(encoded->bytes());
    if (!decoded || *decoded != tier) {
      if (!decoded)
        llvm::errs() << llvm::toString(decoded.takeError()) << '\n';
      llvm::errs() << "special-math accuracy codec did not round-trip\n";
      return false;
    }
    for (const loom::CanonicalSemanticBytes &prior : encodings) {
      if (prior.bytes().equals(encoded->bytes())) {
        llvm::errs() << "special-math accuracy codec is not injective\n";
        return false;
      }
    }
    encodings.push_back(std::move(*encoded));
  }

  auto exactRefinesFour = loom::specialMathAccuracyRefines(
      SpecialMathAccuracyTier::CorrectlyRounded,
      SpecialMathAccuracyTier::Max4Ulp);
  auto twoRefinesFour = loom::specialMathAccuracyRefines(
      SpecialMathAccuracyTier::Max2Ulp, SpecialMathAccuracyTier::Max4Ulp);
  auto twoRefinesOne = loom::specialMathAccuracyRefines(
      SpecialMathAccuracyTier::Max2Ulp, SpecialMathAccuracyTier::Max1Ulp);
  if (!exactRefinesFour || !twoRefinesFour || !twoRefinesOne ||
      !*exactRefinesFour || !*twoRefinesFour || *twoRefinesOne) {
    if (!exactRefinesFour)
      llvm::errs() << llvm::toString(exactRefinesFour.takeError()) << '\n';
    if (!twoRefinesFour)
      llvm::errs() << llvm::toString(twoRefinesFour.takeError()) << '\n';
    if (!twoRefinesOne)
      llvm::errs() << llvm::toString(twoRefinesOne.takeError()) << '\n';
    llvm::errs() << "special-math accuracy refinement order is invalid\n";
    return false;
  }
  if (!expectError(loom::validateSpecialMathAccuracyContract(
                       SpecialMathAccuracyTier::Max1Ulp, false),
                   "afn") ||
      !expectError(loom::validateSpecialMathAccuracyContract(
                       static_cast<SpecialMathAccuracyTier>(0xff), true),
                   "unknown"))
    return false;
  return true;
}

bool checkRegisteredFamilyRelation() {
  std::vector<unsigned> memberships(dataflow::operationSchemaCount(), 0);
  unsigned specialFamilies = 0;
  bool ok = true;
  for (std::uint32_t index = 0; index != implementationFamilyCount(); ++index) {
    const ImplementationFamilyDescriptor &descriptor =
        implementationFamily(static_cast<ImplementationFamilyId>(index));
    if (descriptor.typedAdmissionProvider !=
        TypedAdmissionProviderId::ScalarSpecialMathAdmission)
      continue;
    ++specialFamilies;
    if (descriptor.capabilityParamsSchema !=
        CapabilityParamsSchemaId::ScalarSpecialMathParams) {
      llvm::errs() << "special-math admission has the wrong parameter owner\n";
      ok = false;
    }
    for (OperationSchemaId schema : descriptor.admittedSchemas) {
      ++memberships[static_cast<std::uint32_t>(schema)];
      if (dataflow::semanticsCase(schema) !=
          dataflow::OperationSemanticsCase::SpecialMathAccuracy) {
        llvm::errs() << "special-math family admitted another semantic case\n";
        ok = false;
      }
    }
  }

  unsigned specialSchemas = 0;
  for (std::uint32_t index = 0; index != dataflow::operationSchemaCount();
       ++index) {
    const bool isSpecial =
        dataflow::semanticsCase(static_cast<OperationSchemaId>(index)) ==
        dataflow::OperationSemanticsCase::SpecialMathAccuracy;
    if (isSpecial)
      ++specialSchemas;
    if (memberships[index] != (isSpecial ? 1U : 0U)) {
      llvm::errs() << "special-math schema and family relation is not exact\n";
      ok = false;
    }
  }
  if (specialFamilies != 22 || specialSchemas != 22) {
    llvm::errs()
        << "special-math registry does not contain 22 exact families\n";
    ok = false;
  }
  return ok;
}

bool checkCapabilityCodec(MLIRContext &context) {
  FamilyCapabilityParams capability =
      makeCapability(FloatFormatSet::get({FloatFormat::F32, FloatFormat::F64}),
                     approximateBehavior(), SpecialMathAccuracyTier::Max2Ulp);
  DictionaryAttr encoded = getFamilyCapabilityParamsAttr(&context, capability);
  auto decoded = parseFamilyCapabilityParams(
      ImplementationFamilyId::ScalarMathSin, encoded);
  if (!decoded ||
      getFamilyCapabilityParamsAttr(&context, *decoded) != encoded) {
    if (!decoded)
      llvm::errs() << llvm::toString(decoded.takeError()) << '\n';
    llvm::errs() << "special-math capability did not round-trip\n";
    return false;
  }

  FamilyCapabilityParams unauthorized = makeCapability(
      FloatFormatSet::get({FloatFormat::F32}),
      FloatBehaviorProfile::strictIEEE(), SpecialMathAccuracyTier::Max1Ulp);
  auto rejected = parseFamilyCapabilityParams(
      ImplementationFamilyId::ScalarMathSin,
      getFamilyCapabilityParamsAttr(&context, unauthorized));
  if (!expectFailure(std::move(rejected), "afn"))
    return false;

  SmallVector<NamedAttribute> fields(encoded.begin(), encoded.end());
  for (NamedAttribute &field : fields) {
    if (field.getName() == "accuracy_guarantee")
      field.setValue(StringAttr::get(&context, "Unbounded"));
  }
  return expectFailure(
      parseFamilyCapabilityParams(ImplementationFamilyId::ScalarMathSin,
                                  DictionaryAttr::get(&context, fields)),
      "accuracy_guarantee");
}

bool checkAdmission(MLIRContext &context) {
  const auto invalidTier = static_cast<SpecialMathAccuracyTier>(0xff);
  Type f32 = Float32Type::get(&context);
  Type f64 = Float64Type::get(&context);
  FamilyCapabilityParams correctlyRounded =
      makeCapability(FloatFormatSet::get({FloatFormat::F32}),
                     FloatBehaviorProfile::strictIEEE(),
                     SpecialMathAccuracyTier::CorrectlyRounded);
  FamilyCapabilityParams twoUlp =
      makeCapability(FloatFormatSet::get({FloatFormat::F32}),
                     approximateBehavior(), SpecialMathAccuracyTier::Max2Ulp);
  FamilyCapabilityParams downwardOnly = makeCapability(
      FloatFormatSet::get({FloatFormat::F32}), downwardOnlyBehavior(),
      SpecialMathAccuracyTier::CorrectlyRounded);

  auto strictActor = makeSinActor(
      context, f32,
      dataflow::SpecialMathPayload{arith::FastMathFlags::none,
                                   SpecialMathAccuracyTier::CorrectlyRounded});
  auto oneUlpActor = makeSinActor(
      context, f32,
      dataflow::SpecialMathPayload{arith::FastMathFlags::afn,
                                   SpecialMathAccuracyTier::Max1Ulp});
  auto twoUlpActor = makeSinActor(
      context, f32,
      dataflow::SpecialMathPayload{arith::FastMathFlags::afn,
                                   SpecialMathAccuracyTier::Max2Ulp});
  auto fourUlpActor = makeSinActor(
      context, f32,
      dataflow::SpecialMathPayload{arith::FastMathFlags::afn,
                                   SpecialMathAccuracyTier::Max4Ulp});
  auto wrongFormatActor = makeSinActor(
      context, f64,
      dataflow::SpecialMathPayload{arith::FastMathFlags::afn,
                                   SpecialMathAccuracyTier::Max2Ulp});
  auto wrongPayloadActor = makeSinActor(
      context, f32,
      dataflow::FloatingPointPayload{arith::FastMathFlags::afn, std::nullopt});
  auto relaxedWithoutAfnActor = makeSinActor(
      context, f32,
      dataflow::SpecialMathPayload{arith::FastMathFlags::none,
                                   SpecialMathAccuracyTier::Max2Ulp});
  auto invalidTierActor = makeSinActor(
      context, f32,
      dataflow::SpecialMathPayload{arith::FastMathFlags::afn, invalidTier});
  FamilyCapabilityParams invalidGuarantee =
      makeCapability(FloatFormatSet::get({FloatFormat::F32}),
                     approximateBehavior(), invalidTier);

  if (llvm::Error error = verifyImplementationFamilyAdmission(
          ImplementationFamilyId::ScalarMathSin, &correctlyRounded,
          strictActor)) {
    llvm::errs() << llvm::toString(std::move(error)) << '\n';
    return false;
  }
  if (llvm::Error error = verifyImplementationFamilyAdmission(
          ImplementationFamilyId::ScalarMathSin, &correctlyRounded,
          twoUlpActor)) {
    llvm::errs() << llvm::toString(std::move(error)) << '\n';
    return false;
  }
  if (llvm::Error error = verifyImplementationFamilyAdmission(
          ImplementationFamilyId::ScalarMathSin, &twoUlp, twoUlpActor)) {
    llvm::errs() << llvm::toString(std::move(error)) << '\n';
    return false;
  }
  if (llvm::Error error = verifyImplementationFamilyAdmission(
          ImplementationFamilyId::ScalarMathSin, &twoUlp, fourUlpActor)) {
    llvm::errs() << llvm::toString(std::move(error)) << '\n';
    return false;
  }

  bool ok = true;
  ok &= expectError(
      verifyImplementationFamilyAdmission(ImplementationFamilyId::ScalarMathSin,
                                          &twoUlp, oneUlpActor),
      "accuracy");
  ok &= expectError(
      verifyImplementationFamilyAdmission(ImplementationFamilyId::ScalarMathSin,
                                          &twoUlp, strictActor),
      "fast-math behavior");
  ok &= expectError(
      verifyImplementationFamilyAdmission(ImplementationFamilyId::ScalarMathSin,
                                          &twoUlp, wrongFormatActor),
      "format");
  ok &= expectError(
      verifyImplementationFamilyAdmission(ImplementationFamilyId::ScalarMathSin,
                                          &twoUlp, wrongPayloadActor),
      "typed accuracy projection");
  ok &= expectError(verifyImplementationFamilyAdmission(
                        ImplementationFamilyId::ScalarMathSin,
                        &correctlyRounded, relaxedWithoutAfnActor),
                    "afn");
  ok &= expectError(
      verifyImplementationFamilyAdmission(ImplementationFamilyId::ScalarMathSin,
                                          &twoUlp, invalidTierActor),
      "unknown");
  ok &= expectError(
      verifyImplementationFamilyAdmission(ImplementationFamilyId::ScalarMathSin,
                                          &invalidGuarantee, fourUlpActor),
      "unknown");
  ok &= expectError(
      verifyImplementationFamilyAdmission(ImplementationFamilyId::ScalarMathSin,
                                          &downwardOnly, strictActor),
      "rounding");
  return ok;
}

bool checkFiniteBehaviorDomain(MLIRContext &context) {
  const auto invalidTier = static_cast<SpecialMathAccuracyTier>(0xff);
  FamilyCapabilityParams invalidCapability =
      makeCapability(FloatFormatSet::get({FloatFormat::F32}),
                     approximateBehavior(), invalidTier);
  auto invalidDomain = resolveFiniteImplementationFamilyBehaviorDomain(
      ImplementationFamilyId::ScalarMathSin, invalidCapability,
      {OperationSchemaId::MathSin}, 1, 1, context,
      [](const CanonicalActorSchemaProjection &,
         std::optional<ResolvedIndexWidth>) -> llvm::Error {
        return llvm::Error::success();
      });
  if (!expectFailure(std::move(invalidDomain), "unknown special-math"))
    return false;

  FamilyCapabilityParams strict =
      makeCapability(FloatFormatSet::get({FloatFormat::F32}),
                     FloatBehaviorProfile::strictIEEE(),
                     SpecialMathAccuracyTier::CorrectlyRounded);
  std::vector<std::pair<SpecialMathAccuracyTier, arith::FastMathFlags>>
      strictWitnesses;
  auto strictDomain = resolveFiniteImplementationFamilyBehaviorDomain(
      ImplementationFamilyId::ScalarMathSin, strict,
      {OperationSchemaId::MathSin}, 1, 1, context,
      [&](const CanonicalActorSchemaProjection &actor,
          std::optional<ResolvedIndexWidth>) -> llvm::Error {
        const auto *payload =
            std::get_if<dataflow::SpecialMathPayload>(&actor.payload);
        if (!payload)
          return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                         "missing special-math payload");
        strictWitnesses.emplace_back(payload->accuracy, payload->flags);
        return llvm::Error::success();
      });
  if (!strictDomain || strictDomain->size() != 1 ||
      strictDomain->front().semanticConfiguration ||
      strictWitnesses !=
          std::vector<std::pair<SpecialMathAccuracyTier, arith::FastMathFlags>>{
              {SpecialMathAccuracyTier::CorrectlyRounded,
               arith::FastMathFlags::none},
              {SpecialMathAccuracyTier::Max1Ulp, arith::FastMathFlags::afn},
              {SpecialMathAccuracyTier::Max2Ulp, arith::FastMathFlags::afn},
              {SpecialMathAccuracyTier::Max4Ulp, arith::FastMathFlags::afn}}) {
    if (!strictDomain)
      llvm::errs() << llvm::toString(strictDomain.takeError()) << '\n';
    llvm::errs() << "strict special-math finite domain is incomplete\n";
    return false;
  }

  FamilyCapabilityParams singleton =
      makeCapability(FloatFormatSet::get({FloatFormat::F32}),
                     approximateBehavior(), SpecialMathAccuracyTier::Max2Ulp);
  auto singletonNeedsConfiguration = requiresSemanticConfigurationField(
      ImplementationFamilyId::ScalarMathSin, singleton,
      {OperationSchemaId::MathSin}, 1, 1);
  if (!singletonNeedsConfiguration || *singletonNeedsConfiguration) {
    if (!singletonNeedsConfiguration)
      llvm::errs() << llvm::toString(singletonNeedsConfiguration.takeError())
                   << '\n';
    llvm::errs() << "accuracy guarantee created a physical configuration\n";
    return false;
  }

  std::vector<SpecialMathAccuracyTier> observedTiers;
  auto singletonDomain = resolveFiniteImplementationFamilyBehaviorDomain(
      ImplementationFamilyId::ScalarMathSin, singleton,
      {OperationSchemaId::MathSin}, 1, 1, context,
      [&](const CanonicalActorSchemaProjection &actor,
          std::optional<ResolvedIndexWidth>) -> llvm::Error {
        const auto *payload =
            std::get_if<dataflow::SpecialMathPayload>(&actor.payload);
        if (!payload)
          return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                         "missing special-math payload");
        observedTiers.push_back(payload->accuracy);
        return llvm::Error::success();
      });
  if (!singletonDomain || singletonDomain->size() != 1 ||
      singletonDomain->front().semanticConfiguration ||
      observedTiers != std::vector<SpecialMathAccuracyTier>{
                           SpecialMathAccuracyTier::Max2Ulp,
                           SpecialMathAccuracyTier::Max4Ulp}) {
    if (!singletonDomain)
      llvm::errs() << llvm::toString(singletonDomain.takeError()) << '\n';
    llvm::errs() << "singleton special-math finite domain is incomplete\n";
    return false;
  }

  FamilyCapabilityParams multiFormat =
      makeCapability(FloatFormatSet::get({FloatFormat::F32, FloatFormat::F64}),
                     approximateBehavior(), SpecialMathAccuracyTier::Max2Ulp);
  std::vector<std::pair<unsigned, SpecialMathAccuracyTier>> observed;
  auto domain = resolveFiniteImplementationFamilyBehaviorDomain(
      ImplementationFamilyId::ScalarMathSin, multiFormat,
      {OperationSchemaId::MathSin}, 1, 1, context,
      [&](const CanonicalActorSchemaProjection &actor,
          std::optional<ResolvedIndexWidth>) -> llvm::Error {
        const auto *payload =
            std::get_if<dataflow::SpecialMathPayload>(&actor.payload);
        auto type = dyn_cast<FloatType>(actor.type.getInput(0));
        if (!payload || !type)
          return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                         "malformed special-math witness");
        observed.emplace_back(type.getWidth(), payload->accuracy);
        return llvm::Error::success();
      });
  if (!domain || domain->size() != 2 || observed.size() != 4 ||
      !domain->front().semanticConfiguration ||
      !domain->back().semanticConfiguration ||
      domain->front().semanticConfiguration->bytes().equals(
          domain->back().semanticConfiguration->bytes())) {
    if (!domain)
      llvm::errs() << llvm::toString(domain.takeError()) << '\n';
    llvm::errs() << "multi-format special-math finite domain is incomplete\n";
    return false;
  }

  Type f32 = Float32Type::get(&context);
  auto max2 = makeSinActor(
      context, f32,
      dataflow::SpecialMathPayload{arith::FastMathFlags::afn,
                                   SpecialMathAccuracyTier::Max2Ulp});
  auto max4 = makeSinActor(
      context, f32,
      dataflow::SpecialMathPayload{arith::FastMathFlags::afn,
                                   SpecialMathAccuracyTier::Max4Ulp});
  auto invalid = makeSinActor(
      context, f32,
      dataflow::SpecialMathPayload{arith::FastMathFlags::afn, invalidTier});
  auto max2Configuration = encodeImplementationFamilySemanticConfiguration(
      ImplementationFamilyId::ScalarMathSin, multiFormat,
      {OperationSchemaId::MathSin}, 1, 1, max2, {0}, {0});
  auto max4Configuration = encodeImplementationFamilySemanticConfiguration(
      ImplementationFamilyId::ScalarMathSin, multiFormat,
      {OperationSchemaId::MathSin}, 1, 1, max4, {0}, {0});
  auto invalidConfiguration = encodeImplementationFamilySemanticConfiguration(
      ImplementationFamilyId::ScalarMathSin, multiFormat,
      {OperationSchemaId::MathSin}, 1, 1, invalid, {0}, {0});
  if (!expectFailure(std::move(invalidConfiguration), "unknown special-math"))
    return false;
  if (!max2Configuration || !max4Configuration ||
      !max2Configuration->bytes().equals(max4Configuration->bytes())) {
    if (!max2Configuration)
      llvm::errs() << llvm::toString(max2Configuration.takeError()) << '\n';
    if (!max4Configuration)
      llvm::errs() << llvm::toString(max4Configuration.takeError()) << '\n';
    llvm::errs() << "accepted accuracy leaked into physical configuration\n";
    return false;
  }
  return true;
}

bool checkBehaviorWitnesses(MLIRContext &context) {
  FamilyCapabilityParams multiValued = makeCapability(
      FloatFormatSet::get({FloatFormat::F32}), multiValuedBehavior(),
      SpecialMathAccuracyTier::CorrectlyRounded);
  constexpr std::array inputWidths = {32U};
  constexpr std::array resultWidths = {32U};
  if (!expectFailure(fabric::detail::resolveScalarSpecialMathBehaviorDomain(
                         ImplementationFamilyId::ScalarMathSin, multiValued,
                         {OperationSchemaId::MathSin}, inputWidths,
                         resultWidths, context),
                     "rounding"))
    return false;

  FamilyCapabilityParams relaxedOnly = makeCapability(
      FloatFormatSet::get({FloatFormat::F32}), relaxedOnlyBehavior(),
      SpecialMathAccuracyTier::CorrectlyRounded);
  unsigned witnessCount = 0;
  auto domain = resolveFiniteImplementationFamilyBehaviorDomain(
      ImplementationFamilyId::ScalarMathSin, relaxedOnly,
      {OperationSchemaId::MathSin}, 1, 1, context,
      [&](const CanonicalActorSchemaProjection &actor,
          std::optional<ResolvedIndexWidth>) -> llvm::Error {
        const auto *payload =
            std::get_if<dataflow::SpecialMathPayload>(&actor.payload);
        if (!payload ||
            !hasFastMathFlag(payload->flags, arith::FastMathFlags::nnan) ||
            !hasFastMathFlag(payload->flags, arith::FastMathFlags::nsz))
          return llvm::createStringError(
              llvm::inconvertibleErrorCode(),
              "relaxed-only special-math witness lacks permissions");
        if (payload->accuracy == SpecialMathAccuracyTier::CorrectlyRounded) {
          if (hasFastMathFlag(payload->flags, arith::FastMathFlags::afn))
            return llvm::createStringError(
                llvm::inconvertibleErrorCode(),
                "correctly-rounded witness gained afn");
        } else if (!hasFastMathFlag(payload->flags,
                                    arith::FastMathFlags::afn)) {
          return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                         "relaxed witness lacks afn");
        }
        ++witnessCount;
        return llvm::Error::success();
      });
  if (!domain || domain->size() != 1 || domain->front().semanticConfiguration ||
      witnessCount != 4) {
    if (!domain)
      llvm::errs() << llvm::toString(domain.takeError()) << '\n';
    llvm::errs() << "relaxed-only special-math domain is incomplete\n";
    return false;
  }
  return true;
}

bool checkPowBehavior(MLIRContext &context) {
  FamilyCapabilityParams capability =
      makeCapability(FloatFormatSet::get({FloatFormat::F32}),
                     FloatBehaviorProfile::strictIEEE(),
                     SpecialMathAccuracyTier::CorrectlyRounded);
  Type f32 = Float32Type::get(&context);
  auto actor = makePowActor(
      context, f32,
      dataflow::SpecialMathPayload{arith::FastMathFlags::afn,
                                   SpecialMathAccuracyTier::Max2Ulp});
  if (llvm::Error error = verifyImplementationFamilyAdmission(
          ImplementationFamilyId::ScalarMathPow, &capability, actor)) {
    llvm::errs() << llvm::toString(std::move(error)) << '\n';
    return false;
  }

  unsigned witnesses = 0;
  auto domain = resolveFiniteImplementationFamilyBehaviorDomain(
      ImplementationFamilyId::ScalarMathPow, capability,
      {OperationSchemaId::MathPowF}, 2, 1, context,
      [&](const CanonicalActorSchemaProjection &witness,
          std::optional<ResolvedIndexWidth>) -> llvm::Error {
        if (witness.type.getNumInputs() != 2 ||
            witness.type.getNumResults() != 1)
          return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                         "pow witness has wrong arity");
        ++witnesses;
        return llvm::Error::success();
      });
  if (!domain || domain->size() != 1 || domain->front().semanticConfiguration ||
      witnesses != 4) {
    if (!domain)
      llvm::errs() << llvm::toString(domain.takeError()) << '\n';
    llvm::errs() << "pow special-math domain is incomplete\n";
    return false;
  }
  return true;
}

bool checkSealedBehaviorRelation(MLIRContext &context) {
  constexpr std::array inputWidths = {64U};
  constexpr std::array resultWidths = {64U};
  constexpr std::array schemas = {OperationSchemaId::MathSin};
  FamilyCapabilityParams multiFormat =
      makeCapability(FloatFormatSet::get({FloatFormat::F32, FloatFormat::F64}),
                     approximateBehavior(), SpecialMathAccuracyTier::Max2Ulp);
  auto domain = fabric::detail::resolveScalarSpecialMathBehaviorDomain(
      ImplementationFamilyId::ScalarMathSin, multiFormat, schemas, inputWidths,
      resultWidths, context);
  if (!domain || domain->size() != 2 ||
      !domain->front().semanticConfiguration ||
      !domain->back().semanticConfiguration) {
    if (!domain)
      llvm::errs() << llvm::toString(domain.takeError()) << '\n';
    llvm::errs() << "sealed special-math relation lost its format quotient\n";
    return false;
  }
  for (const FiniteImplementationFamilyBehaviorPoint &point : *domain) {
    if (point.operandPorts != std::vector<std::uint64_t>{0} ||
        point.resultPorts != std::vector<std::uint64_t>{0}) {
      llvm::errs() << "special-math relation lost physical correspondence\n";
      return false;
    }
  }
  constexpr std::array<std::uint8_t, 113> expectedF32Key = {
      'l', 'o', 'o', 'm', '.', 'f', 'a', 'b', 'r', 'i', 'c', '.', 'o', 'p', 'e',
      'r', 'a', 't', 'i', 'o', 'n', '-', 'b', 'e', 'h', 'a', 'v', 'i', 'o', 'r',
      '-', 'k', 'e', 'y', 0,   0,   0,   0,   1,   0,   0,   0,   0,   0,   0,
      0,   13,  'S', 'c', 'a', 'l', 'a', 'r', 'M', 'a', 't', 'h', 'S', 'i', 'n',
      0,   0,   0,   0,   0,   0,   0,   45,  'l', 'o', 'o', 'm', '.', 'd', 'a',
      't', 'a', 'f', 'l', 'o', 'w', '.', 'c', 'a', 'n', 'o', 'n', 'i', 'c', 'a',
      'l', '-', 't', 'y', 'p', 'e', 0,   0,   0,   0,   1,   0,   0,   0,   0,
      0,   0,   0,   4,   0,   0,   0,   15};
  if (!domain->front().semanticConfiguration->bytes().equals(expectedF32Key)) {
    llvm::errs() << "special-math relation changed its canonical f32 key\n";
    return false;
  }

  Type f32 = Float32Type::get(&context);
  auto max2 = makeSinActor(
      context, f32,
      dataflow::SpecialMathPayload{arith::FastMathFlags::afn,
                                   SpecialMathAccuracyTier::Max2Ulp});
  auto max4 = makeSinActor(
      context, f32,
      dataflow::SpecialMathPayload{arith::FastMathFlags::afn,
                                   SpecialMathAccuracyTier::Max4Ulp});
  auto projectedMax2 = fabric::detail::projectScalarSpecialMathBehavior(
      ImplementationFamilyId::ScalarMathSin, max2, *domain);
  auto projectedMax4 = fabric::detail::projectScalarSpecialMathBehavior(
      ImplementationFamilyId::ScalarMathSin, max4, *domain);
  if (!projectedMax2 || !projectedMax4 ||
      !projectedMax2->bytes().equals(projectedMax4->bytes())) {
    if (!projectedMax2)
      llvm::errs() << llvm::toString(projectedMax2.takeError()) << '\n';
    if (!projectedMax4)
      llvm::errs() << llvm::toString(projectedMax4.takeError()) << '\n';
    llvm::errs() << "accuracy permission leaked into the sealed relation\n";
    return false;
  }

  FamilyCapabilityParams singleton =
      makeCapability(FloatFormatSet::get({FloatFormat::F32}),
                     approximateBehavior(), SpecialMathAccuracyTier::Max2Ulp);
  auto singletonDomain = fabric::detail::resolveScalarSpecialMathBehaviorDomain(
      ImplementationFamilyId::ScalarMathSin, singleton, schemas, inputWidths,
      resultWidths, context);
  if (!singletonDomain || singletonDomain->size() != 1 ||
      singletonDomain->front().semanticConfiguration) {
    if (!singletonDomain)
      llvm::errs() << llvm::toString(singletonDomain.takeError()) << '\n';
    llvm::errs() << "singleton special-math relation did not collapse\n";
    return false;
  }

  constexpr std::array duplicateSchemas = {OperationSchemaId::MathSin,
                                           OperationSchemaId::MathSin};
  if (!expectFailure(fabric::detail::resolveScalarSpecialMathBehaviorDomain(
                         ImplementationFamilyId::ScalarMathSin, singleton,
                         duplicateSchemas, inputWidths, resultWidths, context),
                     "twice"))
    return false;
  constexpr std::array foreignSchemas = {OperationSchemaId::MathCos};
  if (!expectFailure(fabric::detail::resolveScalarSpecialMathBehaviorDomain(
                         ImplementationFamilyId::ScalarMathSin, singleton,
                         foreignSchemas, inputWidths, resultWidths, context),
                     "foreign"))
    return false;
  constexpr std::array narrowWidths = {16U};
  if (!expectFailure(fabric::detail::resolveScalarSpecialMathBehaviorDomain(
                         ImplementationFamilyId::ScalarMathSin, singleton,
                         schemas, narrowWidths, resultWidths, context),
                     "physically reachable"))
    return false;

  FloatBehaviorProfile orphanRounding = FloatBehaviorProfile::strictIEEE();
  orphanRounding.roundingModes = RoundingModeSet::get(
      {arith::RoundingMode::to_nearest_even, arith::RoundingMode::downward});
  FamilyCapabilityParams invalidRounding =
      makeCapability(FloatFormatSet::get({FloatFormat::F32}), orphanRounding,
                     SpecialMathAccuracyTier::CorrectlyRounded);
  if (!expectFailure(fabric::detail::resolveScalarSpecialMathBehaviorDomain(
                         ImplementationFamilyId::ScalarMathSin, invalidRounding,
                         schemas, inputWidths, resultWidths, context),
                     "rounding"))
    return false;
  FamilyCapabilityParams downward = makeCapability(
      FloatFormatSet::get({FloatFormat::F32}), downwardOnlyBehavior(),
      SpecialMathAccuracyTier::CorrectlyRounded);
  if (!expectFailure(fabric::detail::resolveScalarSpecialMathBehaviorDomain(
                         ImplementationFamilyId::ScalarMathSin, downward,
                         schemas, inputWidths, resultWidths, context),
                     "rounding"))
    return false;

  FloatBehaviorProfile orphanNaN = FloatBehaviorProfile::strictIEEE();
  orphanNaN.nanBehaviors = FloatNaNBehaviorSet::get(
      {FloatNaNBehavior::IEEE, FloatNaNBehavior::NumberPreferred});
  FamilyCapabilityParams invalidNaN =
      makeCapability(FloatFormatSet::get({FloatFormat::F32}), orphanNaN,
                     SpecialMathAccuracyTier::CorrectlyRounded);
  if (!expectFailure(fabric::detail::resolveScalarSpecialMathBehaviorDomain(
                         ImplementationFamilyId::ScalarMathSin, invalidNaN,
                         schemas, inputWidths, resultWidths, context),
                     "NaN"))
    return false;

  constexpr std::array expectedFamilies = {
      ImplementationFamilyId::ScalarMathSin,
      ImplementationFamilyId::ScalarMathCos,
      ImplementationFamilyId::ScalarMathTan,
      ImplementationFamilyId::ScalarMathSinh,
      ImplementationFamilyId::ScalarMathCosh,
      ImplementationFamilyId::ScalarMathTanh,
      ImplementationFamilyId::ScalarMathExp,
      ImplementationFamilyId::ScalarMathExp2,
      ImplementationFamilyId::ScalarMathExpM1,
      ImplementationFamilyId::ScalarMathLog,
      ImplementationFamilyId::ScalarMathLog2,
      ImplementationFamilyId::ScalarMathLog10,
      ImplementationFamilyId::ScalarMathLog1p,
      ImplementationFamilyId::ScalarMathFloor,
      ImplementationFamilyId::ScalarMathCeil,
      ImplementationFamilyId::ScalarMathRound,
      ImplementationFamilyId::ScalarMathTrunc,
      ImplementationFamilyId::ScalarMathRoundEven,
      ImplementationFamilyId::ScalarMathSqrt,
      ImplementationFamilyId::ScalarMathRsqrt,
      ImplementationFamilyId::ScalarMathErf,
      ImplementationFamilyId::ScalarMathPow};
  for (std::uint32_t ordinal = 0; ordinal != implementationFamilyCount();
       ++ordinal) {
    const auto family = static_cast<ImplementationFamilyId>(ordinal);
    const bool expected = llvm::is_contained(expectedFamilies, family);
    if (fabric::detail::ownsScalarSpecialMathBehaviorRelation(family) !=
        expected) {
      llvm::errs() << "special-math relation owns the wrong family set\n";
      return false;
    }
    if (!expected)
      continue;
    const auto &descriptor = implementationFamily(family);
    if (descriptor.admittedSchemas.size() != 1) {
      llvm::errs() << "special-math relation ownership is incomplete\n";
      return false;
    }
    const unsigned inputCount =
        descriptor.admittedSchemas.front() == OperationSchemaId::MathPowF ? 2
                                                                          : 1;
    std::vector<std::uint32_t> familyInputs(inputCount, 32U);
    constexpr std::array familyResults = {32U};
    auto familyDomain = fabric::detail::resolveScalarSpecialMathBehaviorDomain(
        family, singleton, descriptor.admittedSchemas, familyInputs,
        familyResults, context);
    if (!familyDomain || familyDomain->size() != 1) {
      if (!familyDomain)
        llvm::errs() << llvm::toString(familyDomain.takeError()) << '\n';
      llvm::errs() << "registered special-math family has no relation\n";
      return false;
    }
    if (descriptor.admittedSchemas.front() == OperationSchemaId::MathPowF &&
        familyDomain->front().operandPorts !=
            std::vector<std::uint64_t>({0, 1})) {
      llvm::errs() << "pow relation lost its ordered input correspondence\n";
      return false;
    }
  }
  return true;
}

} // namespace

int main() {
  MLIRContext context;
  bool ok = true;
  ok &= checkCentralAccuracyDomain();
  ok &= checkRegisteredFamilyRelation();
  ok &= checkCapabilityCodec(context);
  ok &= checkAdmission(context);
  ok &= checkFiniteBehaviorDomain(context);
  ok &= checkBehaviorWitnesses(context);
  ok &= checkPowBehavior(context);
  ok &= checkSealedBehaviorRelation(context);
  return ok ? 0 : 1;
}
