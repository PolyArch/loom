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

} // namespace

int main() {
  DialectRegistry registry;
  registry.insert<arith::ArithDialect, LLVM::LLVMDialect,
                  dataflow::DataflowDialect>();
  MLIRContext context(registry, MLIRContext::Threading::DISABLED);
  context.loadAllAvailableDialects();
  checkRoutedTokenParameterClosure(context);
  checkCanonicalInverse(context);
  return EXIT_SUCCESS;
}
