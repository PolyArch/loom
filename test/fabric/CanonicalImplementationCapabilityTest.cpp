//===- CanonicalImplementationCapabilityTest.cpp -------------------------===//

#include "Fabric/IR/ImplementationFamily.h"

#include "Dataflow/IR/DataflowDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <array>
#include <cstdlib>
#include <optional>
#include <vector>

using namespace fabric;
using namespace mlir;
using dataflow::OperationSchemaId;

namespace {

[[noreturn]] void fail(const llvm::Twine &message) {
  llvm::errs() << "canonical capability test failed: " << message << '\n';
  std::exit(EXIT_FAILURE);
}

void require(bool condition, const llvm::Twine &message) {
  if (!condition)
    fail(message);
}

void requireFailure(llvm::Error error, const llvm::Twine &message) {
  if (!error)
    fail(message);
  llvm::consumeError(std::move(error));
}

template <typename T>
void requireExpectedFailure(llvm::Expected<T> value,
                            const llvm::Twine &message) {
  if (value)
    fail(message);
  llvm::consumeError(value.takeError());
}

void requireDerivationFailure(
    llvm::Expected<CanonicalImplementationCapability> value,
    CanonicalCapabilityDerivationFailure expected) {
  if (value)
    fail("capability derivation unexpectedly succeeded");
  std::optional<CanonicalCapabilityDerivationFailure> observed;
  llvm::Error remaining = llvm::handleErrors(
      value.takeError(), [&](const CanonicalCapabilityDerivationError &error) {
        observed = error.failure();
      });
  if (remaining)
    fail("capability derivation returned a foreign error: " +
         llvm::toString(std::move(remaining)));
  require(observed == expected,
          "capability derivation returned the wrong typed failure");
}

dataflow::CanonicalActorSchemaProjection syncActor(MLIRContext &context) {
  Type none = NoneType::get(&context);
  return {OperationSchemaId::DataflowSync,
          FunctionType::get(&context, {none}, {none}), dataflow::NoPayload{}};
}

void checkRoutedTokenParameterClosure(MLIRContext &context) {
  const auto actor = syncActor(context);
  const FamilyCapabilityParams minimum =
      RoutedTokenParams{RoutedTokenParams::minimumPayloadCapacityBits,
                        RoutedTokenParams::minimumFanCapacity};
  DictionaryAttr encoded = getFamilyCapabilityParamsAttr(&context, minimum);
  auto decoded =
      parseFamilyCapabilityParams(ImplementationFamilyId::TokenSync, encoded);
  require(static_cast<bool>(decoded),
          "minimum routed-token params did not decode");
  const auto *roundTrip = std::get_if<RoutedTokenParams>(&*decoded);
  require(roundTrip &&
              roundTrip->maxPayloadBits ==
                  RoutedTokenParams::minimumPayloadCapacityBits &&
              roundTrip->maxFan == RoutedTokenParams::minimumFanCapacity &&
              getFamilyCapabilityParamsAttr(&context, *decoded) == encoded,
          "minimum routed-token params did not round-trip exactly");

  constexpr std::array enabled = {OperationSchemaId::DataflowSync};
  constexpr std::array<std::uint32_t, 1> physicalWidths = {1};
  const std::array invalid = {RoutedTokenParams{0, 2}, RoutedTokenParams{1, 1}};
  for (const RoutedTokenParams params : invalid) {
    requireFailure(verifyRoutedTokenParams(params),
                   "routed-token validator accepted an invalid lower bound");
    FamilyCapabilityParams capability = params;
    requireExpectedFailure(
        parseFamilyCapabilityParams(
            ImplementationFamilyId::TokenSync,
            getFamilyCapabilityParamsAttr(&context, capability)),
        "routed-token parser accepted an invalid lower bound");
    requireFailure(verifyImplementationFamilyAdmission(
                       ImplementationFamilyId::TokenSync, &capability, actor),
                   "routed-token admission accepted an invalid lower bound");
    requireFailure(
        forEachImplementationFamilyPortCorrespondence(
            ImplementationFamilyId::TokenSync, capability, actor,
            physicalWidths, physicalWidths,
            [](llvm::ArrayRef<std::uint64_t>, llvm::ArrayRef<std::uint64_t>)
                -> llvm::Expected<bool> { return true; }),
        "routed-token correspondence accepted an invalid lower bound");
    requireExpectedFailure(
        resolveFabricOpSemanticFieldRelation(
            ImplementationFamilyId::TokenSync, capability, enabled,
            physicalWidths, physicalWidths, context),
        "routed-token behavior relation accepted an invalid lower bound");
  }
}

void checkCanonicalInverse(MLIRContext &context) {
  Type i32 = IntegerType::get(&context, 32);
  const dataflow::CanonicalActorSchemaProjection add{
      OperationSchemaId::ArithAddI,
      FunctionType::get(&context, {i32, i32}, {i32}),
      dataflow::IntegerOverflowPayload{}};
  const dataflow::CanonicalActorSchemaProjection sub{
      OperationSchemaId::ArithSubI,
      FunctionType::get(&context, {i32, i32}, {i32}),
      dataflow::IntegerOverflowPayload{}};
  const std::array arithmetic = {add, sub};
  auto capability = deriveCanonicalImplementationCapability(
      ImplementationFamilyId::ScalarIntegerAddSub, arithmetic);
  require(static_cast<bool>(capability),
          "explicit scalar family did not derive its exact envelope");
  const auto *integer =
      std::get_if<ScalarIntegerParams>(&capability->parameters);
  require(capability->family == ImplementationFamilyId::ScalarIntegerAddSub &&
              integer && integer->integerWidths.size() == 1 &&
              integer->integerWidths.contains(IntegerWidth::I32),
          "scalar inverse did not derive the least integer-width envelope");
  require(capability->enabledSchemas ==
              std::vector<OperationSchemaId>{OperationSchemaId::ArithAddI,
                                             OperationSchemaId::ArithSubI},
          "scalar inverse did not retain the exact schema projection");

  const auto controlSync = syncActor(context);
  auto syncCapability = deriveCanonicalImplementationCapability(
      ImplementationFamilyId::TokenSync, {controlSync});
  require(static_cast<bool>(syncCapability),
          "control-only sync did not derive its least envelope");
  const auto *routed =
      std::get_if<RoutedTokenParams>(&syncCapability->parameters);
  require(
      routed &&
          routed->maxPayloadBits ==
              RoutedTokenParams::minimumPayloadCapacityBits &&
          routed->maxFan == RoutedTokenParams::minimumFanCapacity &&
          syncCapability->enabledSchemas ==
              std::vector<OperationSchemaId>{OperationSchemaId::DataflowSync},
      "control-only sync did not retain its minimum routed envelope");

  requireDerivationFailure(deriveCanonicalImplementationCapability(
                               ImplementationFamilyId::ScalarIntegerAddSub, {}),
                           CanonicalCapabilityDerivationFailure::EmptyActorSet);
  requireDerivationFailure(
      deriveCanonicalImplementationCapability(ImplementationFamilyId::TokenSync,
                                              arithmetic),
      CanonicalCapabilityDerivationFailure::FamilyDoesNotOwnSchema);
  requireDerivationFailure(
      deriveCanonicalImplementationCapability(
          static_cast<ImplementationFamilyId>(implementationFamilyCount()),
          arithmetic),
      CanonicalCapabilityDerivationFailure::InvalidFamily);

  Type index = IndexType::get(&context);
  const dataflow::CanonicalActorSchemaProjection indexAdd{
      OperationSchemaId::ArithAddI,
      FunctionType::get(&context, {index, index}, {index}),
      dataflow::IntegerOverflowPayload{}};
  requireDerivationFailure(
      deriveCanonicalImplementationCapability(
          ImplementationFamilyId::ScalarIntegerAddSub, {indexAdd}),
      CanonicalCapabilityDerivationFailure::UnsupportedAdmissionProvider);
  const dataflow::CanonicalActorSchemaProjection malformedIndexAdd{
      OperationSchemaId::ArithAddI,
      FunctionType::get(&context, {index, index}, {index}),
      dataflow::NoPayload{}};
  requireDerivationFailure(
      deriveCanonicalImplementationCapability(
          ImplementationFamilyId::ScalarIntegerAddSub, {malformedIndexAdd}),
      CanonicalCapabilityDerivationFailure::InvalidActorProjection);

  Type i1 = IntegerType::get(&context, 1);
  const dataflow::CanonicalActorSchemaProjection predicateAdd{
      OperationSchemaId::ArithAddI, FunctionType::get(&context, {i1, i1}, {i1}),
      dataflow::IntegerOverflowPayload{}};
  requireDerivationFailure(
      deriveCanonicalImplementationCapability(
          ImplementationFamilyId::ScalarIntegerAddSub, {predicateAdd}),
      CanonicalCapabilityDerivationFailure::NoAdmittingFamily);
  const dataflow::CanonicalActorSchemaProjection malformedPredicateAdd{
      OperationSchemaId::ArithAddI,
      FunctionType::get(&context, {i32, i32}, {i1}),
      dataflow::IntegerOverflowPayload{}};
  requireDerivationFailure(
      deriveCanonicalImplementationCapability(
          ImplementationFamilyId::ScalarIntegerAddSub, {malformedPredicateAdd}),
      CanonicalCapabilityDerivationFailure::InvalidActorProjection);

  Type pointer = LLVM::LLVMPointerType::get(&context);
  const dataflow::CanonicalActorSchemaProjection gep{
      OperationSchemaId::LLVMGetElementPtr,
      FunctionType::get(&context, {pointer, i32}, {pointer}),
      dataflow::GetElementPtrPayload{
          i32, {LLVM::GEPOp::kDynamicIndex}, LLVM::GEPNoWrapFlags::none}};
  requireDerivationFailure(
      deriveCanonicalImplementationCapability(
          ImplementationFamilyId::ScalarIntegerAddSub, {gep}),
      CanonicalCapabilityDerivationFailure::UnsupportedAdmissionProvider);
  const dataflow::CanonicalActorSchemaProjection malformedGep{
      OperationSchemaId::LLVMGetElementPtr,
      FunctionType::get(&context, {pointer, i32}, {pointer}),
      dataflow::GetElementPtrPayload{
          i32,
          {LLVM::GEPOp::kDynamicIndex, LLVM::GEPOp::kDynamicIndex},
          LLVM::GEPNoWrapFlags::none}};
  requireDerivationFailure(
      deriveCanonicalImplementationCapability(
          ImplementationFamilyId::ScalarIntegerAddSub, {malformedGep}),
      CanonicalCapabilityDerivationFailure::InvalidActorProjection);
  const dataflow::CanonicalActorSchemaProjection pointerAdd{
      OperationSchemaId::ArithAddI,
      FunctionType::get(&context, {pointer, pointer}, {pointer}),
      dataflow::IntegerOverflowPayload{}};
  requireDerivationFailure(
      deriveCanonicalImplementationCapability(
          ImplementationFamilyId::ScalarIntegerAddSub, {pointerAdd}),
      CanonicalCapabilityDerivationFailure::InvalidActorProjection);

  Type i128 = IntegerType::get(&context, 128);
  const dataflow::CanonicalActorSchemaProjection wideAdd{
      OperationSchemaId::ArithAddI,
      FunctionType::get(&context, {i128, i128}, {i128}),
      dataflow::IntegerOverflowPayload{}};
  const std::array forward = {gep, wideAdd};
  const std::array reverse = {wideAdd, gep};
  requireDerivationFailure(
      deriveCanonicalImplementationCapability(
          ImplementationFamilyId::ScalarIntegerAddSub, forward),
      CanonicalCapabilityDerivationFailure::UnsupportedAdmissionProvider);
  requireDerivationFailure(
      deriveCanonicalImplementationCapability(
          ImplementationFamilyId::ScalarIntegerAddSub, reverse),
      CanonicalCapabilityDerivationFailure::UnsupportedAdmissionProvider);

  Type vector = VectorType::get({4}, i32);
  const dataflow::CanonicalActorSchemaProjection vectorAdd{
      OperationSchemaId::ArithAddI,
      FunctionType::get(&context, {vector, vector}, {vector}),
      dataflow::IntegerOverflowPayload{}};
  requireDerivationFailure(
      deriveCanonicalImplementationCapability(
          ImplementationFamilyId::FixedVectorIntegerAddSub, {vectorAdd}),
      CanonicalCapabilityDerivationFailure::UnsupportedAdmissionProvider);
}

/// The floating datapaths and the token plane a counted loop lowers to. Each
/// policy derives the least envelope its actor set uses, and each refuses
/// exactly what one capability of its family cannot reproduce.
void checkComposedDatapathInverse(MLIRContext &context) {
  Type f32 = Float32Type::get(&context);
  Type f16 = Float16Type::get(&context);
  const dataflow::CanonicalActorSchemaProjection multiply{
      OperationSchemaId::ArithMulF,
      FunctionType::get(&context, {f32, f32}, {f32}),
      dataflow::FloatingPointPayload{}};
  const dataflow::CanonicalActorSchemaProjection negate{
      OperationSchemaId::ArithNegF, FunctionType::get(&context, {f16}, {f16}),
      dataflow::FloatingPointPayload{}};
  auto scalarFloat = deriveCanonicalImplementationCapability(
      ImplementationFamilyId::ScalarFloatMultiply, {multiply});
  require(static_cast<bool>(scalarFloat),
          "scalar floating family did not derive its exact envelope");
  const auto *floatParams =
      std::get_if<ScalarFloatParams>(&scalarFloat->parameters);
  require(floatParams && floatParams->formats.size() == 1 &&
              floatParams->formats.contains(FloatFormat::F32) &&
              floatParams->behavior.roundingModes.size() == 1 &&
              floatParams->behavior.roundingModes.contains(
                  arith::RoundingMode::to_nearest_even) &&
              floatParams->behavior.requiredFastMath ==
                  arith::FastMathFlags::none,
          "scalar floating inverse did not derive the least format and "
          "behavior envelope");
  // A sign resource never rounds, so its envelope keeps the canonical default
  // rather than inventing a rounding domain from an actor that has none.
  auto signFloat = deriveCanonicalImplementationCapability(
      ImplementationFamilyId::ScalarFloatSign, {negate});
  require(static_cast<bool>(signFloat) &&
              std::get<ScalarFloatParams>(signFloat->parameters)
                  .formats.contains(FloatFormat::F16),
          "scalar floating inverse did not admit a sign resource");

  Type f32x4 = VectorType::get({4}, f32);
  Type f32x2 = VectorType::get({2}, f32);
  const dataflow::CanonicalActorSchemaProjection wideAdd{
      OperationSchemaId::ArithAddF,
      FunctionType::get(&context, {f32x4, f32x4}, {f32x4}),
      dataflow::FloatingPointPayload{}};
  const dataflow::CanonicalActorSchemaProjection narrowAdd{
      OperationSchemaId::ArithAddF,
      FunctionType::get(&context, {f32x2, f32x2}, {f32x2}),
      dataflow::FloatingPointPayload{}};
  const std::array vectorFloatActors = {wideAdd, narrowAdd};
  auto vectorFloat = deriveCanonicalImplementationCapability(
      ImplementationFamilyId::FixedVectorFloatAddSub, vectorFloatActors);
  require(static_cast<bool>(vectorFloat),
          "fixed-vector floating family did not derive its exact envelope");
  const auto *vectorParams =
      std::get_if<FixedVectorFloatParams>(&vectorFloat->parameters);
  require(vectorParams && vectorParams->elementFormats.size() == 1 &&
              vectorParams->elementFormats.contains(FloatFormat::F32) &&
              vectorParams->maxPayloadBits == 128,
          "fixed-vector floating inverse did not cover the widest vector its "
          "actors present");

  Type i1 = IntegerType::get(&context, 1);
  Type i32 = IntegerType::get(&context, 32);
  Type none = NoneType::get(&context);
  const dataflow::CanonicalActorSchemaProjection ascending{
      OperationSchemaId::DataflowStream,
      FunctionType::get(&context, {i32, i32, i32}, {i32, i1}),
      dataflow::StreamRecurrencePayload{dataflow::StreamStepKind::Add,
                                        arith::CmpIPredicate::slt}};
  const dataflow::CanonicalActorSchemaProjection unsignedAscending{
      OperationSchemaId::DataflowStream,
      FunctionType::get(&context, {i32, i32, i32}, {i32, i1}),
      dataflow::StreamRecurrencePayload{dataflow::StreamStepKind::Add,
                                        arith::CmpIPredicate::ult}};
  const dataflow::CanonicalActorSchemaProjection scaling{
      OperationSchemaId::DataflowStream,
      FunctionType::get(&context, {i32, i32, i32}, {i32, i1}),
      dataflow::StreamRecurrencePayload{dataflow::StreamStepKind::Mul,
                                        arith::CmpIPredicate::slt}};
  const std::array predicatePair = {ascending, unsignedAscending};
  auto stream = deriveCanonicalImplementationCapability(
      ImplementationFamilyId::LoopStream, predicatePair);
  require(static_cast<bool>(stream),
          "loop stream family did not derive its exact envelope");
  const auto *streamParams = std::get_if<LoopStreamParams>(&stream->parameters);
  require(streamParams &&
              streamParams->fixedStepKind == dataflow::StreamStepKind::Add &&
              streamParams->integerWidths.size() == 1 &&
              streamParams->integerWidths.contains(IntegerWidth::I32) &&
              streamParams->continuationPredicates.size() == 2,
          "loop stream inverse did not derive the least recurrence and "
          "continuation envelope");
  // The step kind is one fixed implementation parameter, not a domain, so two
  // streams that step differently are two resources rather than one capability
  // that silently takes the first actor's kind.
  const std::array stepPair = {ascending, scaling};
  requireDerivationFailure(
      deriveCanonicalImplementationCapability(
          ImplementationFamilyId::LoopStream, stepPair),
      CanonicalCapabilityDerivationFailure::NoAdmittingFamily);

  const dataflow::CanonicalActorSchemaProjection constant{
      OperationSchemaId::DataflowConstant,
      FunctionType::get(&context, {none}, {i32}),
      dataflow::ConstantValuePayload{IntegerAttr::get(i32, 7)}};
  auto constantToken = deriveCanonicalImplementationCapability(
      ImplementationFamilyId::TokenConstant, {constant});
  require(static_cast<bool>(constantToken) &&
              std::get<PayloadCapacityParams>(constantToken->parameters)
                      .maxPayloadBits == 32,
          "constant token inverse did not cover the value its actor emits");

  const dataflow::CanonicalActorSchemaProjection demux{
      OperationSchemaId::DataflowDemux,
      FunctionType::get(&context, {i1, none}, {none, none}),
      dataflow::NoPayload{}};
  auto routedDemux = deriveCanonicalImplementationCapability(
      ImplementationFamilyId::TokenDemux, {demux});
  require(static_cast<bool>(routedDemux),
          "token demux family did not derive its exact envelope");
  const auto *demuxParams =
      std::get_if<RoutedTokenParams>(&routedDemux->parameters);
  require(demuxParams &&
              demuxParams->maxFan == RoutedTokenParams::minimumFanCapacity &&
              demuxParams->maxPayloadBits ==
                  RoutedTokenParams::minimumPayloadCapacityBits,
          "token demux inverse did not derive its least routed envelope");

  const dataflow::CanonicalActorSchemaProjection mux{
      OperationSchemaId::DataflowMux,
      FunctionType::get(&context, {i1, i32, i32}, {i32}),
      dataflow::NoPayload{}};
  auto routedMux = deriveCanonicalImplementationCapability(
      ImplementationFamilyId::TokenMux, {mux});
  require(static_cast<bool>(routedMux) &&
              std::get<RoutedTokenParams>(routedMux->parameters).maxFan == 2 &&
              std::get<RoutedTokenParams>(routedMux->parameters)
                      .maxPayloadBits == 32,
          "token mux inverse did not derive its least routed envelope");

  // A route wider than two ways selects by index, and an index payload is an
  // index too: both resolve their width in the program, not in the actor.
  Type index = IndexType::get(&context);
  const dataflow::CanonicalActorSchemaProjection wideDemux{
      OperationSchemaId::DataflowDemux,
      FunctionType::get(&context, {index, i32}, {i32, i32, i32}),
      dataflow::NoPayload{}};
  requireDerivationFailure(
      deriveCanonicalImplementationCapability(ImplementationFamilyId::TokenDemux,
                                              {wideDemux}),
      CanonicalCapabilityDerivationFailure::UnsupportedAdmissionProvider);
  const dataflow::CanonicalActorSchemaProjection indexStream{
      OperationSchemaId::DataflowStream,
      FunctionType::get(&context, {index, index, index}, {index, i1}),
      dataflow::StreamRecurrencePayload{dataflow::StreamStepKind::Add,
                                        arith::CmpIPredicate::slt}};
  requireDerivationFailure(
      deriveCanonicalImplementationCapability(ImplementationFamilyId::LoopStream,
                                              {indexStream}),
      CanonicalCapabilityDerivationFailure::UnsupportedAdmissionProvider);
}

} // namespace

int main() {
  DialectRegistry registry;
  registry.insert<arith::ArithDialect, LLVM::LLVMDialect,
                  dataflow::DataflowDialect>();
  MLIRContext context(registry, MLIRContext::Threading::DISABLED);
  context.loadAllAvailableDialects();
  checkRoutedTokenParameterClosure(context);
  checkCanonicalInverse(context);
  checkComposedDatapathInverse(context);
  return EXIT_SUCCESS;
}
