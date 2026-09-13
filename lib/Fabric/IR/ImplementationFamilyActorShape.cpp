//===- ImplementationFamilyActorShape.cpp - provider actor shapes --------===//
//
// The capability-independent shape each typed admission provider requires of
// an actor: its ordered arity, the agreement among its payload lanes, and the
// typed semantic projection it must carry. Forward admission and canonical
// capability derivation share these owners, so neither restates the other's
// shape rule and a derivation can prove a shape before it has a capability.
//
//===----------------------------------------------------------------------===//

#include "ImplementationFamilyActorShape.h"

#include "Dataflow/IR/DataflowActorSemantics.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <limits>
#include <optional>
#include <variant>

namespace {

using namespace fabric;
using ::dataflow::CanonicalActorSchemaProjection;
using ::dataflow::OperationSchemaId;

llvm::Error reject(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(), message);
}

/// The selector one routed-token actor uses for its exact fan: a two-way
/// route is predicated, and a wider one indexes its choice.
llvm::Error validateSelector(::mlir::Type selector, unsigned fan) {
  if (fan == 2)
    return selector.isInteger(1)
               ? llvm::Error::success()
               : reject("two-way token route requires an i1 selector");
  return ::llvm::isa<::mlir::IndexType>(selector)
             ? llvm::Error::success()
             : reject("multi-way token route requires an index selector");
}

} // namespace

namespace fabric::detail {

bool isValidStreamStepKind(::dataflow::StreamStepKind kind) {
  switch (kind) {
  case ::dataflow::StreamStepKind::Add:
  case ::dataflow::StreamStepKind::Sub:
  case ::dataflow::StreamStepKind::Mul:
  case ::dataflow::StreamStepKind::SDiv:
  case ::dataflow::StreamStepKind::UDiv:
  case ::dataflow::StreamStepKind::ShL:
  case ::dataflow::StreamStepKind::AShr:
  case ::dataflow::StreamStepKind::LShr:
    return true;
  }
  return false;
}

UniformFloatShape uniformFloatShape(FloatDatapath datapath,
                                    OperationSchemaId schema) {
  switch (schema) {
  case OperationSchemaId::ArithNegF:
  case OperationSchemaId::MathAbsF:
    return {1, false};
  case OperationSchemaId::ArithAddF:
  case OperationSchemaId::ArithSubF:
  case OperationSchemaId::ArithMulF:
    return {2, true};
  case OperationSchemaId::ArithDivF:
  case OperationSchemaId::ArithRemF:
    return datapath == FloatDatapath::Scalar ? UniformFloatShape{2, true}
                                             : UniformFloatShape{};
  case OperationSchemaId::MathFma:
    return {3, true};
  default:
    return {};
  }
}

std::optional<::mlir::arith::RoundingMode>
arithmeticRounding(const CanonicalActorSchemaProjection &actor) {
  const auto *payload =
      std::get_if<::dataflow::FloatingPointPayload>(&actor.payload);
  if (!payload)
    return std::nullopt;
  return payload->roundingMode.value_or(
      ::mlir::arith::RoundingMode::to_nearest_even);
}

llvm::Error requireArity(const CanonicalActorSchemaProjection &actor,
                         unsigned inputs, unsigned results) {
  if (actor.type.getNumInputs() != inputs ||
      actor.type.getNumResults() != results)
    return reject("actor function type has the wrong arity");
  return llvm::Error::success();
}

llvm::Error requireUniformType(const CanonicalActorSchemaProjection &actor,
                               unsigned inputs) {
  if (llvm::Error error = requireArity(actor, inputs, 1))
    return error;
  ::mlir::Type type = actor.type.getInput(0);
  for (unsigned index = 1; index < inputs; ++index)
    if (actor.type.getInput(index) != type)
      return reject("actor function type is not uniform");
  if (actor.type.getResult(0) != type)
    return reject("actor result type differs from its operands");
  return llvm::Error::success();
}

llvm::Error verifyScalarOrdinaryIntegerActorShape(
    const CanonicalActorSchemaProjection &actor) {
  if (actor.schema == OperationSchemaId::LLVMGetElementPtr) {
    const auto *payload =
        std::get_if<dataflow::GetElementPtrPayload>(&actor.payload);
    if (!payload || !payload->sourceElementType)
      return reject("GEP actor has no exact source element type");
    if (actor.type.getNumInputs() == 0 || actor.type.getNumResults() != 1)
      return reject("GEP actor has invalid arity");
    auto base =
        ::mlir::dyn_cast<::mlir::LLVM::LLVMPointerType>(actor.type.getInput(0));
    auto result = ::mlir::dyn_cast<::mlir::LLVM::LLVMPointerType>(
        actor.type.getResult(0));
    if (!base || !result || base.getAddressSpace() != result.getAddressSpace())
      return reject("GEP pointer address spaces do not agree");
    unsigned dynamicCount = 0;
    for (std::int32_t raw : payload->rawConstantIndices)
      dynamicCount += raw == ::mlir::LLVM::GEPOp::kDynamicIndex;
    if (dynamicCount + 1 != actor.type.getNumInputs())
      return reject("GEP dynamic index pattern does not match its function "
                    "type");
    for (::mlir::Type type : actor.type.getInputs().drop_front()) {
      auto integer = ::llvm::dyn_cast<::mlir::IntegerType>(type);
      if (!::llvm::isa<::mlir::IndexType>(type) &&
          (!integer || !integer.isSignless()))
        return reject("GEP dynamic index is not an integer or index type");
    }
    return llvm::Error::success();
  }

  if (llvm::Error error = requireUniformType(actor, 2))
    return error;
  ::mlir::Type type = actor.type.getInput(0);
  auto integer = ::llvm::dyn_cast<::mlir::IntegerType>(type);
  if (!::llvm::isa<::mlir::IndexType>(type) &&
      (!integer || !integer.isSignless()))
    return reject("integer width admission requires a scalar signless integer "
                  "or index type");
  return llvm::Error::success();
}

llvm::Error
verifyScalarIntegerCastActorShape(const CanonicalActorSchemaProjection &actor) {
  if (llvm::Error error = requireArity(actor, 1, 1))
    return error;
  for (::mlir::Type type : {actor.type.getInput(0), actor.type.getResult(0)}) {
    auto integer = ::llvm::dyn_cast<::mlir::IntegerType>(type);
    if (!::llvm::isa<::mlir::IndexType>(type) &&
        (!integer || !integer.isSignless()))
      return reject("integer cast endpoints must be scalar signless integer "
                    "or index types");
  }
  return llvm::Error::success();
}

llvm::Error
verifyTokenPlaneActorShape(const CanonicalActorSchemaProjection &actor) {
  ::mlir::Type payloadType;
  switch (actor.schema) {
  case OperationSchemaId::DataflowCarry:
    if (llvm::Error error = requireArity(actor, 3, 1))
      return error;
    payloadType = actor.type.getResult(0);
    if (actor.type.getInput(1) != payloadType ||
        actor.type.getInput(2) != payloadType)
      return reject("carry payload types do not agree");
    break;
  case OperationSchemaId::DataflowInvariant:
    if (llvm::Error error = requireArity(actor, 2, 1))
      return error;
    payloadType = actor.type.getResult(0);
    if (actor.type.getInput(1) != payloadType)
      return reject("invariant payload types do not agree");
    break;
  case OperationSchemaId::DataflowGate:
    if (llvm::Error error = requireArity(actor, 2, 2))
      return error;
    payloadType = actor.type.getResult(1);
    if (actor.type.getInput(1) != payloadType)
      return reject("gate payload types do not agree");
    break;
  default:
    return reject("token-plane admission provider received an unsupported "
                  "schema");
  }
  auto condition =
      ::llvm::dyn_cast<::mlir::IntegerType>(actor.type.getInput(0));
  if (!condition || !condition.isSignless() || condition.getWidth() != 1)
    return reject("token-plane condition must be scalar i1");
  if (actor.schema == OperationSchemaId::DataflowGate) {
    auto result =
        ::llvm::dyn_cast<::mlir::IntegerType>(actor.type.getResult(0));
    if (!result || !result.isSignless() || result.getWidth() != 1)
      return reject("gate condition result must be scalar i1");
  }
  return llvm::Error::success();
}

llvm::Error
verifySyncTokenActorShape(const CanonicalActorSchemaProjection &actor) {
  if (!std::holds_alternative<::dataflow::NoPayload>(actor.payload))
    return reject("sync actor has a noncanonical semantic payload");
  std::optional<std::uint32_t> lanes =
      fabric::routedTokenLaneCount(ImplementationFamilyId::TokenSync, actor);
  if (!lanes)
    return reject("sync actor has an invalid lane inventory");
  for (unsigned lane = 0; lane < *lanes; ++lane)
    if (actor.type.getInput(lane) != actor.type.getResult(lane))
      return reject("sync lane types do not agree");
  return llvm::Error::success();
}

llvm::Error verifyScalarUniformFloatActorShape(
    const CanonicalActorSchemaProjection &actor) {
  const UniformFloatShape shape =
      uniformFloatShape(FloatDatapath::Scalar, actor.schema);
  if (shape.inputCount == 0)
    return reject("floating admission provider received an unsupported schema");
  if (llvm::Error error = requireUniformType(actor, shape.inputCount))
    return error;
  if (!::llvm::isa<::mlir::FloatType>(actor.type.getInput(0)))
    return reject("scalar floating admission requires a scalar floating type");
  return llvm::Error::success();
}

llvm::Error verifyFixedVectorUniformFloatActorShape(
    const CanonicalActorSchemaProjection &actor) {
  const UniformFloatShape shape =
      uniformFloatShape(FloatDatapath::FixedVector, actor.schema);
  if (shape.inputCount == 0)
    return reject(
        "fixed-vector floating provider received an unsupported schema");
  if (llvm::Error error = requireUniformType(actor, shape.inputCount))
    return error;
  llvm::Expected<::mlir::VectorType> vector =
      ::dataflow::semantics::analyzeFixedRankDataVector(
          actor.type.getInput(0), ::dataflow::semantics::VectorRank::AnyFixed);
  if (!vector)
    return reject("fixed-vector floating admission requires a fixed vector: " +
                  llvm::toString(vector.takeError()));
  if (!::llvm::isa<::mlir::FloatType>(vector->getElementType()))
    return reject("fixed-vector floating admission requires a floating element "
                  "type");
  return llvm::Error::success();
}

llvm::Error
verifyStreamActorShape(const CanonicalActorSchemaProjection &actor) {
  if (llvm::Error error = requireArity(actor, 3, 2))
    return error;
  ::mlir::Type recurrenceType = actor.type.getInput(0);
  if (actor.type.getInput(1) != recurrenceType ||
      actor.type.getInput(2) != recurrenceType ||
      actor.type.getResult(0) != recurrenceType)
    return reject("stream recurrence types do not agree");
  auto recurrence = ::llvm::dyn_cast<::mlir::IntegerType>(recurrenceType);
  if (!::llvm::isa<::mlir::IndexType>(recurrenceType) &&
      (!recurrence || !recurrence.isSignless()))
    return reject("stream recurrence must be a scalar signless integer or "
                  "index type");
  auto phase = ::llvm::dyn_cast<::mlir::IntegerType>(actor.type.getResult(1));
  if (!phase || !phase.isSignless() || phase.getWidth() != 1)
    return reject("stream phase result must be scalar i1");
  const auto *payload =
      std::get_if<::dataflow::StreamRecurrencePayload>(&actor.payload);
  if (!payload)
    return reject("stream has no typed recurrence projection");
  if (!isValidStreamStepKind(payload->stepKind))
    return reject("stream actor step kind is invalid");
  return llvm::Error::success();
}

llvm::Error
verifyConstantTokenActorShape(const CanonicalActorSchemaProjection &actor) {
  if (llvm::Error error = requireArity(actor, 1, 1))
    return error;
  if (!::llvm::isa<::mlir::NoneType>(actor.type.getInput(0)))
    return reject("constant control input must be none");
  return llvm::Error::success();
}

llvm::Error
verifyMuxTokenActorShape(const CanonicalActorSchemaProjection &actor) {
  std::optional<std::uint32_t> fan =
      fabric::routedTokenLaneCount(ImplementationFamilyId::TokenMux, actor);
  if (!fan)
    return reject("token mux arity is malformed");
  if (llvm::Error error = validateSelector(actor.type.getInput(0), *fan))
    return error;
  ::mlir::Type payload = actor.type.getResult(0);
  for (unsigned lane = 1; lane < actor.type.getNumInputs(); ++lane)
    if (actor.type.getInput(lane) != payload)
      return reject("token mux payload types do not agree");
  return llvm::Error::success();
}

llvm::Error
verifyDemuxTokenActorShape(const CanonicalActorSchemaProjection &actor) {
  std::optional<std::uint32_t> fan =
      fabric::routedTokenLaneCount(ImplementationFamilyId::TokenDemux, actor);
  if (!fan)
    return reject("token demux arity is malformed");
  if (llvm::Error error = validateSelector(actor.type.getInput(0), *fan))
    return error;
  ::mlir::Type payload = actor.type.getInput(1);
  for (unsigned lane = 0; lane < *fan; ++lane)
    if (actor.type.getResult(lane) != payload)
      return reject("token demux payload types do not agree");
  return llvm::Error::success();
}

} // namespace fabric::detail

llvm::Error fabric::verifyImplementationFamilyActorShape(
    ImplementationFamilyId family,
    const ::dataflow::CanonicalActorSchemaProjection &actor) {
  const std::uint32_t familyIndex = static_cast<std::uint32_t>(family);
  if (familyIndex >= implementationFamilyCount())
    return reject("implementation family is not registered");
  const ImplementationFamilyDescriptor &descriptor =
      implementationFamily(family);
  if (!llvm::is_contained(descriptor.admittedSchemas, actor.schema))
    return reject("actor schema is not admitted by the implementation family");
  switch (descriptor.typedAdmissionProvider) {
  case TypedAdmissionProviderId::ScalarOrdinaryIntegerAdmission:
    return detail::verifyScalarOrdinaryIntegerActorShape(actor);
  case TypedAdmissionProviderId::SyncTokenAdmission:
    return detail::verifySyncTokenActorShape(actor);
  case TypedAdmissionProviderId::ScalarIntegerCastAdmission:
    return detail::verifyScalarIntegerCastActorShape(actor);
  case TypedAdmissionProviderId::TokenPlaneAdmission:
    return detail::verifyTokenPlaneActorShape(actor);
  case TypedAdmissionProviderId::ScalarUniformFloatAdmission:
    return detail::verifyScalarUniformFloatActorShape(actor);
  case TypedAdmissionProviderId::FixedVectorUniformFloatAdmission:
    return detail::verifyFixedVectorUniformFloatActorShape(actor);
  case TypedAdmissionProviderId::StreamAdmission:
    return detail::verifyStreamActorShape(actor);
  case TypedAdmissionProviderId::ConstantTokenAdmission:
    return detail::verifyConstantTokenActorShape(actor);
  case TypedAdmissionProviderId::MuxTokenAdmission:
    return detail::verifyMuxTokenActorShape(actor);
  case TypedAdmissionProviderId::DemuxTokenAdmission:
    return detail::verifyDemuxTokenActorShape(actor);
  default:
    return reject("implementation-family admission provider has no shared "
                  "capability-independent shape validator");
  }
}

std::optional<std::uint32_t> fabric::routedTokenLaneCount(
    ImplementationFamilyId family,
    const ::dataflow::CanonicalActorSchemaProjection &actor) {
  const std::size_t inputs = actor.type.getNumInputs();
  const std::size_t results = actor.type.getNumResults();
  const std::uint32_t familyIndex = static_cast<std::uint32_t>(family);
  if (familyIndex >= implementationFamilyCount())
    return std::nullopt;
  std::size_t lanes = 0;
  switch (implementationFamily(family).typedAdmissionProvider) {
  case TypedAdmissionProviderId::SyncTokenAdmission:
    if (inputs != results)
      return std::nullopt;
    lanes = inputs;
    break;
  case TypedAdmissionProviderId::MuxTokenAdmission:
    if (inputs < 3 || results != 1)
      return std::nullopt;
    lanes = inputs - 1;
    break;
  case TypedAdmissionProviderId::DemuxTokenAdmission:
    if (inputs != 2 || results < 2)
      return std::nullopt;
    lanes = results;
    break;
  default:
    return std::nullopt;
  }
  if (lanes == 0 || lanes > std::numeric_limits<std::uint32_t>::max())
    return std::nullopt;
  return static_cast<std::uint32_t>(lanes);
}
