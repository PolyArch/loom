//===- OperationSchema.cpp - Canonical operation schema registry ----------===//
//
// Expands the one generated registry into dense tables and implements the
// typed projection each registered actor presents through
// `CanonicalDataflowActorOpInterface`.
//
// Every table is built once and indexed by the dense internal
// `OperationSchemaId`, so a consumer never scans a list or compares an
// operation name to decide semantics. Persistent identity uses the separate
// generated wire tags. Every payload reader reads exact typed accessors of the
// operation class its schema names and fails closed when the instance does not
// carry the state its declared case owns.
//
//===----------------------------------------------------------------------===//

#include "Dataflow/IR/OperationSchema.h"

#include "Dataflow/IR/DataflowActorSemantics.h"
#include "Dataflow/IR/DataflowAttrs.h"
#include "Dataflow/IR/DataflowCanonicalEntity.h"
#include "Dataflow/IR/DataflowOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/ErrorHandling.h"

#include <array>
#include <cstddef>
#include <system_error>

using namespace mlir;

namespace {

constexpr std::size_t kSchemaCount = 0
#define LOOM_OPERATION_SCHEMA(Name, Id, WireTag, OpClass, ActorKind,           \
                              SemanticsCase)                                   \
  +1
#include "Dataflow/IR/OperationSchemas.inc"
    ;

constexpr std::size_t kSemanticsCaseCount = 0
#define LOOM_OPERATION_SEMANTICS_CASE(Name, Id, WireTag) +1
#include "Dataflow/IR/OperationSchemas.inc"
    ;

struct SchemaRecord {
  llvm::StringRef spelling;
  dataflow::CanonicalDataflowActorKind actorKind;
  dataflow::OperationSemanticsCase semantics;
};

/// The dense schema table. Its order is the generated id order, so the id is
/// the index and no search is ever needed.
const std::array<SchemaRecord, kSchemaCount> &schemaTable() {
  static const std::array<SchemaRecord, kSchemaCount> table = {{
#define LOOM_OPERATION_SCHEMA(Name, Id, WireTag, OpClass, ActorKind,           \
                              SemanticsCase)                                   \
  SchemaRecord{OpClass::getOperationName(),                                    \
               dataflow::CanonicalDataflowActorKind::ActorKind,                \
               dataflow::OperationSemanticsCase::SemanticsCase},
#include "Dataflow/IR/OperationSchemas.inc"
  }};
  return table;
}

const std::array<llvm::StringRef, kSemanticsCaseCount> &semanticsCaseTable() {
  static const std::array<llvm::StringRef, kSemanticsCaseCount> table = {{
#define LOOM_OPERATION_SEMANTICS_CASE(Name, Id, WireTag) llvm::StringRef(#Name),
#include "Dataflow/IR/OperationSchemas.inc"
  }};
  return table;
}

/// Spelling-to-id resolution. One hash probe over a table built once; the
/// spelling is the operation's own registered name, so this is identity
/// resolution rather than a semantic decision.
const llvm::StringMap<dataflow::OperationSchemaId> &spellingIndex() {
  static const llvm::StringMap<dataflow::OperationSchemaId> index = [] {
    llvm::StringMap<dataflow::OperationSchemaId> built;
    for (auto [position, record] : llvm::enumerate(schemaTable()))
      built.insert({record.spelling,
                    static_cast<dataflow::OperationSchemaId>(position)});
    return built;
  }();
  return index;
}

llvm::Error mismatch(Operation *op, dataflow::OperationSemanticsCase kind) {
  return llvm::createStringError(
      std::errc::invalid_argument,
      "'%s' does not carry the typed state its registered semantic case '%s' "
      "declares",
      op->getName().getStringRef().str().c_str(),
      dataflow::operationSemanticsCaseSpelling(kind).str().c_str());
}

dataflow::SyncScopeProjection projectScope(dataflow::SyncScopeRefAttr scope) {
  return dataflow::SyncScopeProjection{
      scope.getKind(), scope.getTargetNamespace(), scope.getTargetKey()};
}

dataflow::AtomicAccessProjection
projectAtomicAccess(dataflow::AtomicAccessContractAttr access) {
  return dataflow::AtomicAccessProjection{
      access.getOrdering(), projectScope(access.getSyncScope()),
      access.getSourceAlignmentBytes(), access.getVectorGranularity(),
      access.getIsVolatile()};
}

/// Reads the contract of one plain-or-atomic access actor. An absent optional
/// contract is the canonical plain non-volatile contract its operation
/// definition owns, not a missing fact.
llvm::Expected<dataflow::MemoryContractPayload>
projectAccessContract(Operation *op, Attribute contract) {
  if (!contract)
    return dataflow::MemoryContractPayload{
        dataflow::PlainAccessProjection{false}};
  if (auto plain = llvm::dyn_cast<dataflow::PlainAccessContractAttr>(contract))
    return dataflow::MemoryContractPayload{
        dataflow::PlainAccessProjection{plain.getIsVolatile()}};
  if (auto atomic =
          llvm::dyn_cast<dataflow::AtomicAccessContractAttr>(contract))
    return dataflow::MemoryContractPayload{projectAtomicAccess(atomic)};
  return mismatch(op, dataflow::OperationSemanticsCase::MemoryContract);
}

llvm::Expected<dataflow::SemanticPayload> readMemoryContract(Operation *op) {
  return llvm::TypeSwitch<Operation *,
                          llvm::Expected<dataflow::SemanticPayload>>(op)
      .Case<dataflow::LoadOp, dataflow::StoreOp>(
          [&](auto access) -> llvm::Expected<dataflow::SemanticPayload> {
            llvm::Expected<dataflow::MemoryContractPayload> contract =
                projectAccessContract(op, access.getContractAttr());
            if (!contract)
              return contract.takeError();
            return dataflow::SemanticPayload{*contract};
          })
      .Case<dataflow::AtomicRmwOp>(
          [&](auto rmw) -> llvm::Expected<dataflow::SemanticPayload> {
            dataflow::AtomicRmwContractAttr contract = rmw.getContractAttr();
            if (!contract)
              return mismatch(op,
                              dataflow::OperationSemanticsCase::MemoryContract);
            return dataflow::SemanticPayload{
                dataflow::MemoryContractPayload{dataflow::AtomicRmwProjection{
                    contract.getKind(),
                    projectAtomicAccess(contract.getAccess())}}};
          })
      .Case<dataflow::CmpXchgOp>(
          [&](auto exchange) -> llvm::Expected<dataflow::SemanticPayload> {
            dataflow::CompareExchangeContractAttr contract =
                exchange.getContractAttr();
            if (!contract)
              return mismatch(op,
                              dataflow::OperationSemanticsCase::MemoryContract);
            return dataflow::SemanticPayload{dataflow::MemoryContractPayload{
                dataflow::CompareExchangeProjection{
                    contract.getSuccessOrdering(),
                    contract.getFailureOrdering(),
                    projectScope(contract.getSyncScope()),
                    contract.getSourceAlignmentBytes(),
                    contract.getVectorGranularity(), contract.getWeak(),
                    contract.getIsVolatile()}}};
          })
      .Case<dataflow::FenceOp>(
          [&](auto fence) -> llvm::Expected<dataflow::SemanticPayload> {
            dataflow::FenceContractAttr contract = fence.getContractAttr();
            if (!contract)
              return mismatch(op,
                              dataflow::OperationSemanticsCase::MemoryContract);
            return dataflow::SemanticPayload{
                dataflow::MemoryContractPayload{dataflow::FenceProjection{
                    contract.getOrdering(),
                    projectScope(contract.getSyncScope())}}};
          })
      .Default([&](Operation *) -> llvm::Expected<dataflow::SemanticPayload> {
        return mismatch(op, dataflow::OperationSemanticsCase::MemoryContract);
      });
}

llvm::Expected<dataflow::SemanticPayload> readConstantValue(Operation *op) {
  if (auto constant = llvm::dyn_cast<arith::ConstantOp>(op))
    return dataflow::SemanticPayload{
        dataflow::ConstantValuePayload{constant.getValue()}};
  if (auto constant = llvm::dyn_cast<dataflow::ConstantOp>(op)) {
    auto value =
        llvm::dyn_cast_or_null<TypedAttr>(constant.getConstValueAttr());
    if (!value)
      return mismatch(op, dataflow::OperationSemanticsCase::TypedConstantValue);
    return dataflow::SemanticPayload{dataflow::ConstantValuePayload{value}};
  }
  return mismatch(op, dataflow::OperationSemanticsCase::TypedConstantValue);
}

/// Reads the one typed payload the schema's declared case owns. Every arm
/// requires the exact operation class or interface that case names, so a row
/// that pairs a schema with the wrong case fails closed here.
llvm::Expected<dataflow::SemanticPayload>
readPayload(Operation *op, dataflow::OperationSemanticsCase kind) {
  using Case = dataflow::OperationSemanticsCase;
  for (NamedAttribute attr : op->getDiscardableAttrDictionary()) {
    if (attr.getName().getValue() != dataflow::kEntityIdAttrName ||
        !llvm::isa<dataflow::EntityIdAttr>(attr.getValue()))
      return mismatch(op, kind);
  }
  switch (kind) {
  case Case::NoSemanticPayload: {
    if (auto poison = llvm::dyn_cast<ub::PoisonOp>(op)) {
      ub::PoisonAttrInterface value = poison.getValue();
      if (value && !llvm::isa<ub::PoisonAttr>(value))
        return mismatch(op, kind);
      Attribute properties = op->getPropertiesAsAttribute();
      if (properties) {
        auto dictionary = llvm::dyn_cast<DictionaryAttr>(properties);
        if (!dictionary || dictionary.size() != 1 ||
            dictionary.get(poison.getValueAttrName()) != value)
          return mismatch(op, kind);
      }
      return dataflow::SemanticPayload{dataflow::NoPayload{}};
    }
    if (op->getPropertiesAsAttribute())
      return mismatch(op, kind);
    return dataflow::SemanticPayload{dataflow::NoPayload{}};
  }
  case Case::ArithFloatingPoint: {
    auto fastMath = llvm::dyn_cast<arith::ArithFastMathInterface>(op);
    auto rounding = llvm::dyn_cast<arith::ArithRoundingModeInterface>(op);
    if (!fastMath && !rounding)
      return mismatch(op, kind);
    arith::FastMathFlags flags = arith::FastMathFlags::none;
    if (fastMath) {
      arith::FastMathFlagsAttr attr = fastMath.getFastMathFlagsAttr();
      if (attr)
        flags = attr.getValue();
    }
    std::optional<arith::RoundingMode> roundingMode;
    if (rounding) {
      arith::RoundingModeAttr attr = rounding.getRoundingModeAttr();
      if (attr)
        roundingMode = attr.getValue();
    }
    return dataflow::SemanticPayload{
        dataflow::FloatingPointPayload{flags, roundingMode}};
  }
  case Case::ArithExact: {
    return llvm::TypeSwitch<Operation *,
                            llvm::Expected<dataflow::SemanticPayload>>(op)
        .Case<arith::DivSIOp, arith::DivUIOp, arith::ShRSIOp, arith::ShRUIOp>(
            [](auto actor) {
              return dataflow::SemanticPayload{
                  dataflow::ExactPayload{actor.getIsExact()}};
            })
        .Default([&](Operation *) { return mismatch(op, kind); });
  }
  case Case::ArithNonNegative: {
    auto actor = llvm::dyn_cast<arith::ArithNonNegFlagInterface>(op);
    if (!actor)
      return mismatch(op, kind);
    return dataflow::SemanticPayload{
        dataflow::NonNegativePayload{actor.getNonNeg()}};
  }
  case Case::ArithIntegerOverflow: {
    auto actor = llvm::dyn_cast<arith::ArithIntegerOverflowFlagsInterface>(op);
    if (!actor)
      return mismatch(op, kind);
    return dataflow::SemanticPayload{
        dataflow::IntegerOverflowPayload{actor.getOverflowAttr().getValue()}};
  }
  case Case::ArithIntegerCompare: {
    auto actor = llvm::dyn_cast<arith::CmpIOp>(op);
    if (!actor)
      return mismatch(op, kind);
    return dataflow::SemanticPayload{
        dataflow::IntegerComparePayload{actor.getPredicate()}};
  }
  case Case::ArithFloatCompare: {
    auto actor = llvm::dyn_cast<arith::CmpFOp>(op);
    if (!actor)
      return mismatch(op, kind);
    return dataflow::SemanticPayload{dataflow::FloatComparePayload{
        actor.getPredicate(), actor.getFastmath()}};
  }
  case Case::LLVMZeroPoison: {
    return llvm::TypeSwitch<Operation *,
                            llvm::Expected<dataflow::SemanticPayload>>(op)
        .Case<LLVM::CountLeadingZerosOp, LLVM::CountTrailingZerosOp>(
            [](auto actor) {
              return dataflow::SemanticPayload{
                  dataflow::ZeroPoisonPayload{actor.getIsZeroPoison()}};
            })
        .Default([&](Operation *) { return mismatch(op, kind); });
  }
  case Case::LLVMIntegerMinPoison: {
    auto actor = llvm::dyn_cast<LLVM::AbsOp>(op);
    if (!actor)
      return mismatch(op, kind);
    return dataflow::SemanticPayload{
        dataflow::IntegerMinPoisonPayload{actor.getIsIntMinPoison()}};
  }
  case Case::LLVMDisjoint: {
    auto actor = llvm::dyn_cast<LLVM::OrOp>(op);
    if (!actor)
      return mismatch(op, kind);
    return dataflow::SemanticPayload{
        dataflow::DisjointPayload{actor.getIsDisjoint()}};
  }
  case Case::LLVMAggregatePosition:
    return llvm::TypeSwitch<Operation *,
                            llvm::Expected<dataflow::SemanticPayload>>(op)
        .Case<LLVM::ExtractValueOp, LLVM::InsertValueOp>([](auto actor) {
          llvm::ArrayRef<std::int64_t> position = actor.getPosition();
          return dataflow::SemanticPayload{dataflow::AggregatePositionPayload{
              std::vector<std::int64_t>(position.begin(), position.end())}};
        })
        .Default([&](Operation *) { return mismatch(op, kind); });
  case Case::VectorStaticPosition:
    return llvm::TypeSwitch<Operation *,
                            llvm::Expected<dataflow::SemanticPayload>>(op)
        .Case<vector::ExtractOp, vector::InsertOp>([](auto actor) {
          llvm::ArrayRef<std::int64_t> position = actor.getStaticPosition();
          return dataflow::SemanticPayload{
              dataflow::VectorStaticPositionPayload{
                  std::vector<std::int64_t>(position.begin(), position.end())}};
        })
        .Default([&](Operation *) { return mismatch(op, kind); });
  case Case::VectorShuffleMask: {
    auto actor = llvm::dyn_cast<vector::ShuffleOp>(op);
    if (!actor)
      return mismatch(op, kind);
    llvm::ArrayRef<std::int64_t> mask = actor.getMask();
    return dataflow::SemanticPayload{dataflow::VectorShuffleMaskPayload{
        std::vector<std::int64_t>(mask.begin(), mask.end())}};
  }
  case Case::TypedConstantValue:
    return readConstantValue(op);
  case Case::StreamRecurrence: {
    auto actor = llvm::dyn_cast<dataflow::StreamOp>(op);
    if (!actor)
      return mismatch(op, kind);
    return dataflow::SemanticPayload{dataflow::StreamRecurrencePayload{
        actor.getStepKind(), actor.getPredicate()}};
  }
  case Case::MemoryContract:
    return readMemoryContract(op);
  }
  llvm_unreachable("unhandled registered semantic case");
}

/// The external model through which a registered operation outside the
/// Dataflow dialect projects. It adds no facts of its own: every method
/// resolves through the one registry.
template <typename OpTy>
struct CanonicalActorModel final
    : dataflow::CanonicalDataflowActorOpInterface::ExternalModel<
          CanonicalActorModel<OpTy>, OpTy> {};

/// Attaches the external model unless the operation already implements the
/// interface directly, as the Dataflow dialect's own actors do. Both paths
/// are driven by the same generated rows, so no second list decides which
/// operations project.
template <typename OpTy> void attachActorModel(MLIRContext &context) {
  OperationName name(OpTy::getOperationName(), &context);
  if (name.hasInterface<dataflow::CanonicalDataflowActorOpInterface>())
    return;
  OpTy::template attachInterface<CanonicalActorModel<OpTy>>(context);
}

} // namespace

std::uint32_t dataflow::operationSchemaCount() {
  return static_cast<std::uint32_t>(kSchemaCount);
}

llvm::StringRef dataflow::operationSchemaSpelling(OperationSchemaId schema) {
  return schemaTable()[static_cast<std::size_t>(schema)].spelling;
}

llvm::StringRef
dataflow::operationSemanticsCaseSpelling(OperationSemanticsCase kind) {
  return semanticsCaseTable()[static_cast<std::size_t>(kind)];
}

std::optional<dataflow::OperationSchemaId>
dataflow::findOperationSchema(llvm::StringRef spelling) {
  const auto &index = spellingIndex();
  auto found = index.find(spelling);
  if (found == index.end())
    return std::nullopt;
  return found->second;
}

std::optional<dataflow::OperationSchemaId>
dataflow::operationSchemaOf(Operation *op) {
  if (!op)
    return std::nullopt;
  std::optional<OperationSchemaId> schema =
      findOperationSchema(op->getName().getStringRef());
  if (!schema)
    return std::nullopt;
  switch (semanticsCase(*schema)) {
  case OperationSemanticsCase::VectorStaticPosition:
  case OperationSemanticsCase::VectorShuffleMask: {
    for (Type type :
         llvm::concat<Type>(op->getOperandTypes(), op->getResultTypes())) {
      if (!llvm::isa<VectorType>(type))
        continue;
      llvm::Expected<VectorType> vector =
          dataflow::semantics::analyzeFixedRankDataVector(
              type, dataflow::semantics::VectorRank::AnyFixed);
      if (vector)
        continue;
      llvm::consumeError(vector.takeError());
      return std::nullopt;
    }
    break;
  }
  default:
    break;
  }
  switch (*schema) {
  case OperationSchemaId::LLVMCountLeadingZeros: {
    auto actor = llvm::dyn_cast<LLVM::CountLeadingZerosOp>(op);
    if (!actor || !actor.getIsZeroPoison())
      return std::nullopt;
    break;
  }
  case OperationSchemaId::LLVMCountTrailingZeros: {
    auto actor = llvm::dyn_cast<LLVM::CountTrailingZerosOp>(op);
    if (!actor || !actor.getIsZeroPoison())
      return std::nullopt;
    break;
  }
  case OperationSchemaId::LLVMAbs: {
    auto actor = llvm::dyn_cast<LLVM::AbsOp>(op);
    if (!actor || !actor.getIsIntMinPoison())
      return std::nullopt;
    break;
  }
  case OperationSchemaId::LLVMOrDisjoint: {
    auto actor = llvm::dyn_cast<LLVM::OrOp>(op);
    if (!actor || !actor.getIsDisjoint())
      return std::nullopt;
    break;
  }
  default:
    break;
  }
  return schema;
}

dataflow::OperationSchemaId dataflow::requireOperationSchema(Operation *op) {
  std::optional<OperationSchemaId> schema = operationSchemaOf(op);
  assert(schema && "operation is not a registered canonical actor");
  return *schema;
}

dataflow::CanonicalDataflowActorKind
dataflow::actorKind(OperationSchemaId schema) {
  return schemaTable()[static_cast<std::size_t>(schema)].actorKind;
}

dataflow::OperationSemanticsCase
dataflow::semanticsCase(OperationSchemaId schema) {
  return schemaTable()[static_cast<std::size_t>(schema)].semantics;
}

std::optional<dataflow::CanonicalDataflowActorKind>
dataflow::classifyCanonicalDataflowActor(Operation *op) {
  std::optional<OperationSchemaId> schema = operationSchemaOf(op);
  if (!schema)
    return std::nullopt;
  return actorKind(*schema);
}

bool dataflow::isCanonicalDataflowActor(Operation *op) {
  return operationSchemaOf(op).has_value();
}

bool dataflow::isCanonicalDataflowActor(Operation *op,
                                        CanonicalDataflowActorKind kind) {
  return classifyCanonicalDataflowActor(op) == kind;
}

llvm::Expected<dataflow::CanonicalActorSchemaProjection>
dataflow::projectRegisteredActorSchemaProjection(Operation *op) {
  std::optional<OperationSchemaId> schema = operationSchemaOf(op);
  if (!schema)
    return llvm::createStringError(
        std::errc::invalid_argument,
        "'%s' is not a registered canonical Dataflow actor",
        op->getName().getStringRef().str().c_str());
  llvm::Expected<SemanticPayload> payload =
      readPayload(op, semanticsCase(*schema));
  if (!payload)
    return payload.takeError();
  FunctionType type = FunctionType::get(op->getContext(), op->getOperandTypes(),
                                        op->getResultTypes());
  return CanonicalActorSchemaProjection{*schema, type, *payload};
}

LogicalResult dataflow::verifyRegisteredActorInstance(Operation *op) {
  llvm::Expected<CanonicalActorSchemaProjection> projection =
      projectRegisteredActorSchemaProjection(op);
  if (projection)
    return success();
  return op->emitOpError(llvm::toString(projection.takeError()));
}

llvm::Expected<dataflow::TransitionDescriptorIdentity>
dataflow::deriveTransitionDescriptorIdentity(Operation *op) {
  llvm::Expected<CanonicalActorSchemaProjection> projection =
      projectRegisteredActorSchemaProjection(op);
  if (!projection)
    return projection.takeError();
  return TransitionDescriptorIdentity{*projection};
}

void dataflow::attachCanonicalDataflowActorInterfaces(MLIRContext &context) {
  context.getOrLoadDialect<arith::ArithDialect>();
  context.getOrLoadDialect<math::MathDialect>();
  context.getOrLoadDialect<LLVM::LLVMDialect>();
  context.getOrLoadDialect<ub::UBDialect>();
  context.getOrLoadDialect<vector::VectorDialect>();

#define LOOM_OPERATION_SCHEMA(Name, Id, WireTag, OpClass, ActorKind,           \
                              SemanticsCase)                                   \
  attachActorModel<OpClass>(context);
#include "Dataflow/IR/OperationSchemas.inc"
}
