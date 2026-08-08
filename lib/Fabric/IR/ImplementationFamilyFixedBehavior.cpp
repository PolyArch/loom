//===- ImplementationFamilyFixedBehavior.cpp ----------------------------===//
//
// Owns implementation families whose admitted actors select one fixed
// physical behavior and therefore have no semantic configuration field.
//
//===----------------------------------------------------------------------===//

#include "ImplementationFamilyFixedBehavior.h"

#include "ImplementationFamilyBehaviorInternal.h"

#include "mlir/IR/BuiltinTypes.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <numeric>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace fabric;
using ::dataflow::CanonicalActorSchemaProjection;
using ::dataflow::OperationSchemaId;

llvm::Error reject(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(), message);
}

std::vector<std::uint64_t> identityPorts(unsigned count) {
  std::vector<std::uint64_t> ports(count);
  std::iota(ports.begin(), ports.end(), 0);
  return ports;
}

bool containsOnlyOrdinaryIntegerWidths(IntegerWidthSet widths) {
  return widths.valid() && !widths.contains(IntegerWidth::I1);
}

llvm::Error validateFixedParameters(ImplementationFamilyId family,
                                    const FamilyCapabilityParams &params) {
  const ImplementationFamilyDescriptor &descriptor =
      implementationFamily(family);
  if (capabilityParamsSchema(params) != descriptor.capabilityParamsSchema)
    return reject("fixed behavior parameter schema does not match the family");

  switch (family) {
  case ImplementationFamilyId::ScalarValueSelect: {
    const auto &typed = std::get<ScalarValueSelectParams>(params);
    if (!typed.integerWidths.valid())
      return reject("fixed select integer width domain is invalid");
    if (!typed.floatFormats.valid())
      return reject("fixed select floating format domain is invalid");
    if (typed.integerWidths.empty() && typed.floatFormats.empty())
      return reject("fixed select requires a non-empty type domain");
    return llvm::Error::success();
  }
  case ImplementationFamilyId::ScalarBitReinterpret: {
    const auto &typed = std::get<ScalarBitReinterpretParams>(params);
    if (!containsOnlyOrdinaryIntegerWidths(typed.integerWidths))
      return reject("fixed reinterpret integer width domain is invalid");
    if (!typed.floatFormats.valid())
      return reject("fixed reinterpret floating format domain is invalid");
    if (typed.integerWidths.empty() && typed.floatFormats.empty())
      return reject("fixed reinterpret requires a non-empty type domain");
    return llvm::Error::success();
  }
  case ImplementationFamilyId::ScalarIntegerMultiply: {
    const auto &typed = std::get<ScalarIntegerParams>(params);
    if (!containsOnlyOrdinaryIntegerWidths(typed.integerWidths) ||
        typed.integerWidths.empty())
      return reject("fixed multiply requires a non-empty ordinary integer "
                    "width domain");
    if (!typed.pointerFormats.valid())
      return reject("fixed multiply pointer format relation is invalid");
    return llvm::Error::success();
  }
  case ImplementationFamilyId::LoopCarry:
  case ImplementationFamilyId::LoopInvariant:
  case ImplementationFamilyId::LoopGate:
    (void)std::get<TokenPlaneParams>(params);
    return llvm::Error::success();
  case ImplementationFamilyId::FixedVectorPack:
  case ImplementationFamilyId::FixedVectorUnpack: {
    const auto &typed = std::get<FixedVectorAdapterParams>(params);
    if (!typed.integerElementWidths.valid())
      return reject("fixed adapter integer width domain is invalid");
    if (!typed.floatElementFormats.valid())
      return reject("fixed adapter floating format domain is invalid");
    if (typed.integerElementWidths.empty() && typed.floatElementFormats.empty())
      return reject("fixed adapter requires a non-empty element domain");
    if (typed.maxPayloadBits == 0)
      return reject("fixed adapter payload capacity must be positive");
    return llvm::Error::success();
  }
  default:
    return reject("implementation family has no fixed behavior relation");
  }
}

llvm::Error
requireExactSchema(ImplementationFamilyId family,
                   llvm::ArrayRef<OperationSchemaId> enabledSchemas) {
  const ImplementationFamilyDescriptor &descriptor =
      implementationFamily(family);
  if (descriptor.admittedSchemas.size() != 1 || enabledSchemas.size() != 1 ||
      enabledSchemas.front() != descriptor.admittedSchemas.front())
    return reject(implementationFamilyKeyword(family) +
                  " must enable exactly its registered schema");
  return llvm::Error::success();
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

CanonicalActorSchemaProjection makeScalarSelectActor(mlir::MLIRContext &context,
                                                     mlir::Type valueType) {
  mlir::Type condition = mlir::IntegerType::get(&context, 1);
  return {OperationSchemaId::ArithSelect,
          mlir::FunctionType::get(&context, {condition, valueType, valueType},
                                  {valueType}),
          ::dataflow::NoPayload{}};
}

CanonicalActorSchemaProjection
makeScalarReinterpretActor(mlir::MLIRContext &context, mlir::Type type) {
  return {OperationSchemaId::ArithBitcast,
          mlir::FunctionType::get(&context, {type}, {type}),
          ::dataflow::NoPayload{}};
}

CanonicalActorSchemaProjection
makeScalarMultiplyActor(mlir::MLIRContext &context, IntegerWidth width) {
  mlir::Type type = mlir::IntegerType::get(&context, getBitWidth(width));
  return {OperationSchemaId::ArithMulI,
          mlir::FunctionType::get(&context, {type, type}, {type}),
          ::dataflow::IntegerOverflowPayload{}};
}

CanonicalActorSchemaProjection
makeTokenPlaneActor(ImplementationFamilyId family, mlir::MLIRContext &context) {
  mlir::Type condition = mlir::IntegerType::get(&context, 1);
  mlir::Type payload = mlir::NoneType::get(&context);
  switch (family) {
  case ImplementationFamilyId::LoopCarry:
    return {OperationSchemaId::DataflowCarry,
            mlir::FunctionType::get(&context, {condition, payload, payload},
                                    {payload}),
            ::dataflow::NoPayload{}};
  case ImplementationFamilyId::LoopInvariant:
    return {OperationSchemaId::DataflowInvariant,
            mlir::FunctionType::get(&context, {condition, payload}, {payload}),
            ::dataflow::NoPayload{}};
  case ImplementationFamilyId::LoopGate:
    return {OperationSchemaId::DataflowGate,
            mlir::FunctionType::get(&context, {condition, payload},
                                    {condition, payload}),
            ::dataflow::NoPayload{}};
  default:
    llvm_unreachable("non-token family reached token actor construction");
  }
}

CanonicalActorSchemaProjection makeAdapterActor(ImplementationFamilyId family,
                                                mlir::MLIRContext &context,
                                                mlir::Type element) {
  mlir::Type vector = mlir::VectorType::get({1}, element);
  mlir::Type packed =
      mlir::IntegerType::get(&context, element.getIntOrFloatBitWidth());
  if (family == ImplementationFamilyId::FixedVectorPack)
    return {OperationSchemaId::DataflowPack,
            mlir::FunctionType::get(&context, {vector}, {packed}),
            ::dataflow::NoPayload{}};
  return {OperationSchemaId::DataflowUnpack,
          mlir::FunctionType::get(&context, {packed}, {vector}),
          ::dataflow::NoPayload{}};
}

std::vector<CanonicalActorSchemaProjection>
enumerateActors(ImplementationFamilyId family,
                const FamilyCapabilityParams &params,
                mlir::MLIRContext &context) {
  std::vector<CanonicalActorSchemaProjection> actors;
  switch (family) {
  case ImplementationFamilyId::ScalarValueSelect: {
    const auto &typed = std::get<ScalarValueSelectParams>(params);
    for (IntegerWidth width : integerWidthDomain)
      if (typed.integerWidths.contains(width))
        actors.push_back(makeScalarSelectActor(
            context, mlir::IntegerType::get(&context, getBitWidth(width))));
    for (FloatFormat format : floatFormatDomain)
      if (typed.floatFormats.contains(format))
        actors.push_back(
            makeScalarSelectActor(context, floatType(context, format)));
    break;
  }
  case ImplementationFamilyId::ScalarBitReinterpret: {
    const auto &typed = std::get<ScalarBitReinterpretParams>(params);
    for (IntegerWidth width : integerWidthDomain)
      if (typed.integerWidths.contains(width))
        actors.push_back(makeScalarReinterpretActor(
            context, mlir::IntegerType::get(&context, getBitWidth(width))));
    for (FloatFormat format : floatFormatDomain)
      if (typed.floatFormats.contains(format))
        actors.push_back(
            makeScalarReinterpretActor(context, floatType(context, format)));
    break;
  }
  case ImplementationFamilyId::ScalarIntegerMultiply: {
    const auto &typed = std::get<ScalarIntegerParams>(params);
    for (IntegerWidth width : integerWidthDomain)
      if (typed.integerWidths.contains(width))
        actors.push_back(makeScalarMultiplyActor(context, width));
    break;
  }
  case ImplementationFamilyId::LoopCarry:
  case ImplementationFamilyId::LoopInvariant:
  case ImplementationFamilyId::LoopGate:
    actors.push_back(makeTokenPlaneActor(family, context));
    break;
  case ImplementationFamilyId::FixedVectorPack:
  case ImplementationFamilyId::FixedVectorUnpack: {
    const auto &typed = std::get<FixedVectorAdapterParams>(params);
    for (IntegerWidth width : integerWidthDomain)
      if (typed.integerElementWidths.contains(width))
        actors.push_back(makeAdapterActor(
            family, context,
            mlir::IntegerType::get(&context, getBitWidth(width))));
    for (FloatFormat format : floatFormatDomain)
      if (typed.floatElementFormats.contains(format))
        actors.push_back(
            makeAdapterActor(family, context, floatType(context, format)));
    break;
  }
  default:
    llvm_unreachable("non-fixed family reached actor enumeration");
  }
  return actors;
}

} // namespace

bool fabric::detail::ownsFixedBehaviorRelation(ImplementationFamilyId family) {
  switch (family) {
  case ImplementationFamilyId::ScalarValueSelect:
  case ImplementationFamilyId::ScalarBitReinterpret:
  case ImplementationFamilyId::ScalarIntegerMultiply:
  case ImplementationFamilyId::LoopCarry:
  case ImplementationFamilyId::LoopInvariant:
  case ImplementationFamilyId::LoopGate:
  case ImplementationFamilyId::FixedVectorPack:
  case ImplementationFamilyId::FixedVectorUnpack:
    return true;
  default:
    return false;
  }
}

llvm::Expected<std::vector<fabric::FiniteImplementationFamilyBehaviorPoint>>
fabric::detail::resolveFixedBehaviorDomain(
    ImplementationFamilyId family, const FamilyCapabilityParams &params,
    llvm::ArrayRef<OperationSchemaId> enabledSchemas,
    llvm::ArrayRef<std::uint32_t> physicalInputWidths,
    llvm::ArrayRef<std::uint32_t> physicalResultWidths,
    mlir::MLIRContext &context) {
  if (!ownsFixedBehaviorRelation(family))
    return reject("implementation family has no fixed behavior relation");
  if (llvm::Error error = validateFixedParameters(family, params))
    return std::move(error);
  if (llvm::Error error = requireExactSchema(family, enabledSchemas))
    return std::move(error);

  std::string firstRejection;
  for (CanonicalActorSchemaProjection &actor :
       enumerateActors(family, params, context)) {
    std::vector<std::uint64_t> operandPorts =
        identityPorts(actor.type.getNumInputs());
    std::vector<std::uint64_t> resultPorts =
        identityPorts(actor.type.getNumResults());
    if (llvm::Error error = validateImplementationFamilyBehaviorPoint(
            family, params, actor, operandPorts, resultPorts,
            physicalInputWidths, physicalResultWidths)) {
      if (firstRejection.empty())
        firstRejection = llvm::toString(std::move(error));
      else
        llvm::consumeError(std::move(error));
      continue;
    }
    std::vector<FiniteImplementationFamilyBehaviorPoint> domain;
    domain.emplace_back(std::move(actor), std::nullopt, std::nullopt,
                        std::move(operandPorts), std::move(resultPorts));
    return domain;
  }
  return reject(firstRejection.empty()
                    ? "fixed behavior capability has no legal actor"
                    : firstRejection);
}
