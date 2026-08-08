#include "Fabric/IR/ImplementationFamily.h"

#include "ImplementationFamilyFixedBehavior.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace fabric;
using ::dataflow::OperationSchemaId;

[[noreturn]] void fail(const char *test, const std::string &message) {
  llvm::errs() << test << ": " << message << '\n';
  std::exit(EXIT_FAILURE);
}

void require(const char *test, bool condition, const std::string &message) {
  if (!condition)
    fail(test, message);
}

template <typename T> T take(const char *test, llvm::Expected<T> value) {
  if (!value)
    fail(test, llvm::toString(value.takeError()));
  return std::move(*value);
}

template <typename T>
void expectError(const char *test, llvm::Expected<T> value,
                 llvm::StringRef fragment) {
  if (value)
    fail(test, "expected relation rejection");
  const std::string message = llvm::toString(value.takeError());
  require(test, llvm::StringRef(message).contains(fragment),
          "unexpected rejection: " + message);
}

void requireCanonicalNone(const char *test, ImplementationFamilyId family,
                          const FamilyCapabilityParams &params,
                          llvm::ArrayRef<OperationSchemaId> schemas,
                          llvm::ArrayRef<std::uint32_t> physicalInputWidths,
                          llvm::ArrayRef<std::uint32_t> physicalResultWidths,
                          mlir::MLIRContext &context) {
  auto domain = take(test, detail::resolveFixedBehaviorDomain(
                               family, params, schemas, physicalInputWidths,
                               physicalResultWidths, context));
  require(test, domain.size() == 1,
          "fixed behavior relation did not collapse to one witness");
  const auto &point = domain.front();
  require(test, !point.semanticConfiguration,
          "fixed behavior relation retained a configuration value");
  require(test, point.representativeActor.schema == schemas.front(),
          "fixed behavior relation selected the wrong actor schema");
  require(test,
          point.operandPorts.size() ==
              point.representativeActor.type.getNumInputs(),
          "fixed behavior witness has the wrong operand arity");
  require(test,
          point.resultPorts.size() ==
              point.representativeActor.type.getNumResults(),
          "fixed behavior witness has the wrong result arity");
  for (auto [ordinal, port] : llvm::enumerate(point.operandPorts))
    require(test, port == ordinal,
            "fixed behavior witness changed an operand role");
  for (auto [ordinal, port] : llvm::enumerate(point.resultPorts))
    require(test, port == ordinal,
            "fixed behavior witness changed a result role");
}

void allDeclaredFamiliesOwnCanonicalNone() {
  const char *test = __func__;
  mlir::MLIRContext context(mlir::MLIRContext::Threading::DISABLED);
  const FamilyCapabilityParams select = ScalarValueSelectParams{
      IntegerWidthSet::get({IntegerWidth::I8, IntegerWidth::I64}),
      FloatFormatSet::get({FloatFormat::F32})};
  const FamilyCapabilityParams reinterpret =
      ScalarBitReinterpretParams{IntegerWidthSet::get({IntegerWidth::I32}),
                                 FloatFormatSet::get({FloatFormat::F32})};
  const FamilyCapabilityParams multiply = ScalarIntegerParams{
      IntegerWidthSet::get({IntegerWidth::I8, IntegerWidth::I64})};
  const FamilyCapabilityParams tokenPlane = TokenPlaneParams{};
  const FamilyCapabilityParams adapter =
      FixedVectorAdapterParams{IntegerWidthSet::get({IntegerWidth::I8}),
                               FloatFormatSet::get({FloatFormat::F32}), 128};

  constexpr std::array selectSchema = {OperationSchemaId::ArithSelect};
  constexpr std::array reinterpretSchema = {OperationSchemaId::ArithBitcast};
  constexpr std::array multiplySchema = {OperationSchemaId::ArithMulI};
  constexpr std::array carrySchema = {OperationSchemaId::DataflowCarry};
  constexpr std::array invariantSchema = {OperationSchemaId::DataflowInvariant};
  constexpr std::array gateSchema = {OperationSchemaId::DataflowGate};
  constexpr std::array packSchema = {OperationSchemaId::DataflowPack};
  constexpr std::array unpackSchema = {OperationSchemaId::DataflowUnpack};

  constexpr std::array selectInputs = {1U, 64U, 64U};
  constexpr std::array binaryInputs = {64U, 64U};
  constexpr std::array unaryInput = {128U};
  constexpr std::array carryInputs = {1U, 64U, 64U};
  constexpr std::array invariantInputs = {1U, 64U};
  constexpr std::array gateInputs = {1U, 64U};
  constexpr std::array scalarResult = {64U};
  constexpr std::array packedResult = {128U};
  constexpr std::array gateResults = {1U, 64U};

  requireCanonicalNone(test, ImplementationFamilyId::ScalarValueSelect, select,
                       selectSchema, selectInputs, scalarResult, context);
  requireCanonicalNone(test, ImplementationFamilyId::ScalarBitReinterpret,
                       reinterpret, reinterpretSchema, unaryInput, scalarResult,
                       context);
  requireCanonicalNone(test, ImplementationFamilyId::ScalarIntegerMultiply,
                       multiply, multiplySchema, binaryInputs, scalarResult,
                       context);
  requireCanonicalNone(test, ImplementationFamilyId::LoopCarry, tokenPlane,
                       carrySchema, carryInputs, scalarResult, context);
  requireCanonicalNone(test, ImplementationFamilyId::LoopInvariant, tokenPlane,
                       invariantSchema, invariantInputs, scalarResult, context);
  requireCanonicalNone(test, ImplementationFamilyId::LoopGate, tokenPlane,
                       gateSchema, gateInputs, gateResults, context);
  requireCanonicalNone(test, ImplementationFamilyId::FixedVectorPack, adapter,
                       packSchema, unaryInput, packedResult, context);
  requireCanonicalNone(test, ImplementationFamilyId::FixedVectorUnpack, adapter,
                       unpackSchema, unaryInput, packedResult, context);
}

void ownershipIsClosed() {
  const char *test = __func__;
  constexpr std::array owned = {
      ImplementationFamilyId::ScalarValueSelect,
      ImplementationFamilyId::ScalarBitReinterpret,
      ImplementationFamilyId::ScalarIntegerMultiply,
      ImplementationFamilyId::LoopCarry,
      ImplementationFamilyId::LoopInvariant,
      ImplementationFamilyId::LoopGate,
      ImplementationFamilyId::FixedVectorPack,
      ImplementationFamilyId::FixedVectorUnpack,
  };
  for (ImplementationFamilyId family : owned)
    require(test, detail::ownsFixedBehaviorRelation(family),
            "declared fixed behavior family has no owner");
  require(test,
          !detail::ownsFixedBehaviorRelation(
              ImplementationFamilyId::ScalarIntegerAddSub),
          "fixed behavior owner captured a configurable family");
}

void typedParametersAndSchemasFailClosed() {
  const char *test = __func__;
  mlir::MLIRContext context(mlir::MLIRContext::Threading::DISABLED);
  constexpr std::array multiplySchema = {OperationSchemaId::ArithMulI};
  constexpr std::array binaryInputs = {32U, 32U};
  constexpr std::array scalarResult = {32U};

  const FamilyCapabilityParams wrongParams = TokenPlaneParams{};
  expectError(test,
              detail::resolveFixedBehaviorDomain(
                  ImplementationFamilyId::ScalarIntegerMultiply, wrongParams,
                  multiplySchema, binaryInputs, scalarResult, context),
              "parameter schema");

  const FamilyCapabilityParams emptyMultiply =
      ScalarIntegerParams{IntegerWidthSet::get({})};
  expectError(test,
              detail::resolveFixedBehaviorDomain(
                  ImplementationFamilyId::ScalarIntegerMultiply, emptyMultiply,
                  multiplySchema, binaryInputs, scalarResult, context),
              "non-empty");

  const FamilyCapabilityParams pointerMultiply = ScalarIntegerParams{
      IntegerWidthSet::get({IntegerWidth::I32}),
      PointerFormatRelation::get(
          {{0, 64, 64, ::loom::PointerLayoutKind::StableIntegral}})};
  expectError(test,
              detail::resolveFixedBehaviorDomain(
                  ImplementationFamilyId::ScalarIntegerMultiply,
                  pointerMultiply, multiplySchema, binaryInputs, scalarResult,
                  context),
              "empty pointer format");

  FloatFormatSet invalidFormats;
  invalidFormats.insert(static_cast<FloatFormat>(99));
  const FamilyCapabilityParams invalidAdapter = FixedVectorAdapterParams{
      IntegerWidthSet::get({IntegerWidth::I8}), invalidFormats, 32};
  constexpr std::array packSchema = {OperationSchemaId::DataflowPack};
  constexpr std::array unaryInput = {32U};
  expectError(test,
              detail::resolveFixedBehaviorDomain(
                  ImplementationFamilyId::FixedVectorPack, invalidAdapter,
                  packSchema, unaryInput, scalarResult, context),
              "floating format");

  constexpr std::array foreignSchema = {OperationSchemaId::ArithAddI};
  const FamilyCapabilityParams multiply =
      ScalarIntegerParams{IntegerWidthSet::get({IntegerWidth::I32})};
  expectError(test,
              detail::resolveFixedBehaviorDomain(
                  ImplementationFamilyId::ScalarIntegerMultiply, multiply,
                  foreignSchema, binaryInputs, scalarResult, context),
              "exactly its registered schema");

  constexpr std::array duplicateSchemas = {OperationSchemaId::ArithMulI,
                                           OperationSchemaId::ArithMulI};
  expectError(test,
              detail::resolveFixedBehaviorDomain(
                  ImplementationFamilyId::ScalarIntegerMultiply, multiply,
                  duplicateSchemas, binaryInputs, scalarResult, context),
              "exactly its registered schema");
}

void physicalReachabilityIsNotCountOnly() {
  const char *test = __func__;
  mlir::MLIRContext context(mlir::MLIRContext::Threading::DISABLED);
  const FamilyCapabilityParams select = ScalarValueSelectParams{
      IntegerWidthSet::get({IntegerWidth::I8}), FloatFormatSet{}};
  constexpr std::array selectSchema = {OperationSchemaId::ArithSelect};
  constexpr std::array reachableInputs = {1U, 8U, 8U};
  constexpr std::array unreachableInputs = {1U, 7U, 7U};
  constexpr std::array reachableResult = {8U};
  constexpr std::array unreachableResult = {7U};

  requireCanonicalNone(test, ImplementationFamilyId::ScalarValueSelect, select,
                       selectSchema, reachableInputs, reachableResult, context);
  expectError(test,
              detail::resolveFixedBehaviorDomain(
                  ImplementationFamilyId::ScalarValueSelect, select,
                  selectSchema, unreachableInputs, unreachableResult, context),
              "narrower");

  const FamilyCapabilityParams tokenPlane = TokenPlaneParams{};
  constexpr std::array carrySchema = {OperationSchemaId::DataflowCarry};
  constexpr std::array narrowCondition = {0U, 64U, 64U};
  constexpr std::array payloadResult = {64U};
  expectError(test,
              detail::resolveFixedBehaviorDomain(
                  ImplementationFamilyId::LoopCarry, tokenPlane, carrySchema,
                  narrowCondition, payloadResult, context),
              "narrower");

  const FamilyCapabilityParams adapter = FixedVectorAdapterParams{
      IntegerWidthSet::get({IntegerWidth::I8}), FloatFormatSet{}, 8};
  constexpr std::array packSchema = {OperationSchemaId::DataflowPack};
  constexpr std::array narrowPacked = {7U};
  expectError(test,
              detail::resolveFixedBehaviorDomain(
                  ImplementationFamilyId::FixedVectorPack, adapter, packSchema,
                  narrowPacked, narrowPacked, context),
              "narrower");
}

} // namespace

int main() {
  ownershipIsClosed();
  allDeclaredFamiliesOwnCanonicalNone();
  typedParametersAndSchemasFailClosed();
  physicalReachabilityIsNotCountOnly();
  return EXIT_SUCCESS;
}
