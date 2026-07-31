#include "Fabric/IR/MemoryPortTransaction.h"

#include "Dataflow/IR/DataflowAttrs.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"

#include <system_error>

using namespace dataflow;
using namespace dataflow::semantics;
using namespace mlir;

namespace fabric {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(std::errc::invalid_argument, "%s",
                                 message.str().c_str());
}

bool sameScope(const SyncScopeProjection &projection, SyncScopeRefAttr scope) {
  return scope && projection.kind == scope.getKind() &&
         projection.targetNamespace == scope.getTargetNamespace() &&
         projection.targetKey == scope.getTargetKey();
}

bool sameAtomicAccess(const AtomicAccessProjection &projection,
                      AtomicAccessContractAttr access) {
  return access && projection.ordering == access.getOrdering() &&
         sameScope(projection.scope, access.getSyncScope()) &&
         projection.sourceAlignmentBytes == access.getSourceAlignmentBytes() &&
         projection.vectorGranularity == access.getVectorGranularity() &&
         projection.isVolatile == access.getIsVolatile();
}

bool actorContractMatchesAccess(const CanonicalActorSchemaProjection &actor,
                                const CanonicalMemoryAccessView &access) {
  const auto *payload = std::get_if<MemoryContractPayload>(&actor.payload);
  if (!payload)
    return false;

  Attribute aggregate = access.contract().aggregate;
  switch (actor.schema) {
  case OperationSchemaId::DataflowLoad:
  case OperationSchemaId::DataflowStore:
    if (const auto *plain = std::get_if<PlainAccessProjection>(payload)) {
      auto contract = llvm::dyn_cast<PlainAccessContractAttr>(aggregate);
      return contract && plain->isVolatile == contract.getIsVolatile();
    }
    if (const auto *atomic = std::get_if<AtomicAccessProjection>(payload))
      return sameAtomicAccess(
          *atomic, llvm::dyn_cast<AtomicAccessContractAttr>(aggregate));
    return false;
  case OperationSchemaId::DataflowAtomicRmw: {
    const auto *rmw = std::get_if<AtomicRmwProjection>(payload);
    auto contract = llvm::dyn_cast<AtomicRmwContractAttr>(aggregate);
    return rmw && contract && rmw->kind == contract.getKind() &&
           sameAtomicAccess(rmw->access, contract.getAccess());
  }
  case OperationSchemaId::DataflowCmpXchg: {
    const auto *exchange = std::get_if<CompareExchangeProjection>(payload);
    auto contract = llvm::dyn_cast<CompareExchangeContractAttr>(aggregate);
    return exchange && contract &&
           exchange->successOrdering == contract.getSuccessOrdering() &&
           exchange->failureOrdering == contract.getFailureOrdering() &&
           sameScope(exchange->scope, contract.getSyncScope()) &&
           exchange->sourceAlignmentBytes ==
               contract.getSourceAlignmentBytes() &&
           exchange->vectorGranularity == contract.getVectorGranularity() &&
           exchange->weak == contract.getWeak() &&
           exchange->isVolatile == contract.getIsVolatile();
  }
  default:
    return false;
  }
}

bool actorContractMatchesFence(const CanonicalActorSchemaProjection &actor,
                               FenceContractAttr contract) {
  if (actor.schema != OperationSchemaId::DataflowFence)
    return false;
  const auto *payload = std::get_if<MemoryContractPayload>(&actor.payload);
  const auto *fence = payload ? std::get_if<FenceProjection>(payload) : nullptr;
  return fence && contract && fence->ordering == contract.getOrdering() &&
         sameScope(fence->scope, contract.getSyncScope());
}

bool sameAccess(const CanonicalMemoryAccessView &lhs,
                const CanonicalMemoryAccessView &rhs) {
  const MemoryAccessType &left = lhs.geometry();
  const MemoryAccessType &right = rhs.geometry();
  return lhs.operation() == rhs.operation() &&
         left.elementType == right.elementType &&
         left.vectorType == right.vectorType &&
         left.addressVectorType == right.addressVectorType &&
         lhs.maskType() == rhs.maskType() &&
         lhs.laneCount() == rhs.laneCount() &&
         lhs.addressCount() == rhs.addressCount() &&
         lhs.elementBits() == rhs.elementBits() &&
         lhs.dataBits() == rhs.dataBits() &&
         lhs.addressLaneBits() == rhs.addressLaneBits() &&
         lhs.addressBits() == rhs.addressBits() &&
         lhs.maskBits() == rhs.maskBits() &&
         lhs.contract().aggregate == rhs.contract().aggregate;
}

llvm::Expected<ServiceKind> expectedServiceKind(OperationSchemaId actorSchema) {
  switch (actorSchema) {
  case OperationSchemaId::DataflowLoad:
    return ServiceKind::MemoryRead;
  case OperationSchemaId::DataflowStore:
    return ServiceKind::MemoryWrite;
  case OperationSchemaId::DataflowAtomicRmw:
    return ServiceKind::MemoryAtomicRmw;
  case OperationSchemaId::DataflowCmpXchg:
    return ServiceKind::MemoryCompareExchange;
  case OperationSchemaId::DataflowFence:
    return ServiceKind::MemoryFence;
  default:
    return invalid("actor schema is not a canonical memory actor");
  }
}

llvm::Expected<MemoryAccessOperation>
expectedAccessOperation(OperationSchemaId actorSchema) {
  switch (actorSchema) {
  case OperationSchemaId::DataflowLoad:
    return MemoryAccessOperation::Load;
  case OperationSchemaId::DataflowStore:
    return MemoryAccessOperation::Store;
  case OperationSchemaId::DataflowAtomicRmw:
    return MemoryAccessOperation::AtomicRmw;
  case OperationSchemaId::DataflowCmpXchg:
    return MemoryAccessOperation::CompareExchange;
  default:
    return invalid("actor schema does not describe an addressed memory actor");
  }
}

llvm::Error
validateProjectionLegality(MemoryPortTransactionProjection projection,
                           const CanonicalMemoryAccessView &access) {
  const std::optional<VectorAtomicGranularity> granularity =
      access.atomicGranularity();
  if (granularity == VectorAtomicGranularity::WholePayload &&
      projection != MemoryPortTransactionProjection::Direct)
    return invalid("whole-payload atomic access requires Direct projection");
  if (granularity == VectorAtomicGranularity::PerLane &&
      projection != MemoryPortTransactionProjection::ActiveLanesRowMajor)
    return invalid(
        "per-lane atomic access requires ActiveLanesRowMajor projection");
  if (access.contract().atomic && !granularity &&
      projection != MemoryPortTransactionProjection::Direct)
    return invalid("scalar atomic access requires Direct projection");
  if (projection == MemoryPortTransactionProjection::ActiveLanesRowMajor &&
      access.form() == MemoryAccessForm::Element)
    return invalid("ActiveLanesRowMajor requires contiguous or indexed access");
  return llvm::Error::success();
}

} // namespace

std::uint8_t
getCanonicalTag(MemoryPortTransactionProjection transactionProjection) {
  switch (transactionProjection) {
  case MemoryPortTransactionProjection::Direct:
    return 0;
  case MemoryPortTransactionProjection::ActiveLanesRowMajor:
    return 1;
  }
  llvm_unreachable("unknown memory transaction projection");
}

llvm::Expected<MemoryPortTransactionProjection>
decodeMemoryPortTransactionProjection(std::uint8_t tag) {
  switch (tag) {
  case 0:
    return MemoryPortTransactionProjection::Direct;
  case 1:
    return MemoryPortTransactionProjection::ActiveLanesRowMajor;
  default:
    return invalid("unknown memory transaction projection tag");
  }
}

llvm::Expected<MemoryOperationPortResourceView>
MemoryOperationPortResourceView::create(
    loom::fabric::FabricMemoryOperationPortRef owner,
    ResourceContract resourceContract,
    llvm::ArrayRef<MemoryPortTransactionProjection> transactionProjections) {
  if (transactionProjections.size() != resourceContract.usePatternCount())
    return invalid("memory operation pattern semantics must match the complete "
                   "use-pattern inventory");
  for (MemoryPortTransactionProjection projection : transactionProjections)
    if (projection != MemoryPortTransactionProjection::Direct &&
        projection != MemoryPortTransactionProjection::ActiveLanesRowMajor)
      return invalid("memory operation pattern has an unknown projection");

  const loom::fabric::FabricInventoryOwnerRef catalog =
      loom::fabric::FabricInventoryOwnerRef::of(owner);
  const loom::fabric::FabricResourceStateOwnerRef stateOwner(catalog);
  const loom::fabric::FabricUsePatternOwnerRef usePatternOwner(catalog);

  std::vector<loom::fabric::FabricResourceStateRef> states;
  states.reserve(resourceContract.stateCount());
  for (std::uint32_t ordinal = 0; ordinal != resourceContract.stateCount();
       ++ordinal)
    states.push_back({stateOwner, ordinal});

  std::vector<loom::fabric::FabricUsePatternRef> patterns;
  patterns.reserve(resourceContract.usePatternCount());
  for (std::uint32_t ordinal = 0; ordinal != resourceContract.usePatternCount();
       ++ordinal)
    patterns.push_back({usePatternOwner, ordinal});

  return MemoryOperationPortResourceView(
      std::move(owner), std::move(resourceContract), std::move(states),
      std::move(patterns),
      std::vector<MemoryPortTransactionProjection>(
          transactionProjections.begin(), transactionProjections.end()));
}

llvm::Expected<MemoryOperationPatternView>
MemoryOperationPortResourceView::operationPattern(
    const loom::fabric::FabricUsePatternRef &usePattern) const {
  const loom::fabric::FabricUsePatternOwnerRef expectedOwner(
      loom::fabric::FabricInventoryOwnerRef::of(owner_));
  if (usePattern.owner != expectedOwner)
    return invalid("memory operation pattern has the wrong resource owner");
  if (usePattern.ordinal >= resourceContract_.usePatternCount())
    return invalid(
        "memory operation pattern references an unknown use pattern");
  return MemoryOperationPatternView(
      *this, usePattern, transactionProjections_[usePattern.ordinal]);
}

UsePattern MemoryOperationPatternView::usePattern() const {
  return operationPort_->resourceContract().usePattern(
      UsePatternKey(static_cast<std::uint32_t>(usePattern_.ordinal)));
}

MemoryPortAssembly
MemoryPortAssembly::derive(const CanonicalService &parentService,
                           MemoryPortTransactionProjection projection,
                           const CanonicalMemoryAccessView *access) {
  std::vector<MemoryResultAssembly> results;
  const bool masked = access && access->maskForm() == MemoryMaskForm::Dynamic;
  for (const ServiceValue result : parentService.results()) {
    if (result.role == ServiceValueRole::Completion)
      continue;
    if (projection == MemoryPortTransactionProjection::Direct) {
      results.push_back(MemoryResultAssembly(
          result.role,
          masked ? MemoryResultAssemblyStrategy::ParentResponseOrZeroOnEmptyMask
                 : MemoryResultAssemblyStrategy::PassThroughParent,
          0, MemoryInactiveAssemblyValue::NotApplicable));
      continue;
    }
    results.push_back(MemoryResultAssembly(
        result.role, MemoryResultAssemblyStrategy::RowMajorLaneValues,
        access->laneCount(),
        masked ? MemoryInactiveAssemblyValue::ZeroBits
               : MemoryInactiveAssemblyValue::NotApplicable));
  }
  return MemoryPortAssembly(std::move(results));
}

llvm::Expected<MemoryPortTransactionPlan> deriveMemoryPortTransactionPlan(
    const MemoryOperationPatternView &pattern,
    const CanonicalActorSchemaProjection &actor,
    const CanonicalService &parentService,
    const std::optional<CanonicalMemoryAccessView> &access) {
  llvm::Expected<ServiceKind> expectedKind = expectedServiceKind(actor.schema);
  if (!expectedKind)
    return expectedKind.takeError();
  if (parentService.kind() != *expectedKind)
    return invalid("parent service kind does not match the actor schema");

  const bool fence = actor.schema == OperationSchemaId::DataflowFence;
  if (fence) {
    if (access)
      return invalid("fence transaction plan must not carry an access view");
    if (!actorContractMatchesFence(actor, parentService.fenceContract()))
      return invalid("parent fence service does not match the actor contract");
    if (pattern.transactionProjection() !=
        MemoryPortTransactionProjection::Direct)
      return invalid("fence requires Direct projection");
  } else {
    if (!access)
      return invalid("addressed transaction plan requires an access view");
    llvm::Expected<MemoryAccessOperation> expectedOperation =
        expectedAccessOperation(actor.schema);
    if (!expectedOperation)
      return expectedOperation.takeError();
    if (access->operation() != *expectedOperation ||
        !actorContractMatchesAccess(actor, *access))
      return invalid("access view does not match the actor projection");
    if (!sameAccess(parentService.access(), *access))
      return invalid(
          "parent service and access view describe different actors");
    if (llvm::Error error = validateProjectionLegality(
            pattern.transactionProjection(), *access))
      return std::move(error);
  }

  const UsePattern use = pattern.usePattern();
  const std::uint64_t childCount =
      pattern.transactionProjection() == MemoryPortTransactionProjection::Direct
          ? 1
          : access->laneCount();
  if (pattern.transactionProjection() ==
          MemoryPortTransactionProjection::Direct &&
      use.internalTransactionCount != 1)
    return invalid("Direct use pattern must declare one internal transaction");
  if (use.internalTransactionCount < childCount)
    return invalid(
        "use pattern has fewer internal transactions than the selected access");

  std::vector<MemoryPortChildTransaction> transactions;
  transactions.reserve(childCount);
  const bool masked = access && access->maskForm() == MemoryMaskForm::Dynamic;
  for (std::uint64_t ordinal = 0; ordinal != childCount; ++ordinal) {
    if (pattern.transactionProjection() ==
        MemoryPortTransactionProjection::Direct) {
      transactions.push_back(MemoryPortChildTransaction(
          ordinal,
          masked ? MemoryChildActivation::parentMaskAny()
                 : MemoryChildActivation::always(),
          MemoryChildProjection::parentRequest()));
      continue;
    }
    transactions.push_back(MemoryPortChildTransaction(
        ordinal,
        masked ? MemoryChildActivation::parentMaskLane(ordinal)
               : MemoryChildActivation::always(),
        MemoryChildProjection::elementLane(ordinal)));
  }

  return MemoryPortTransactionPlan(
      parentService, std::move(transactions),
      MemoryPortAssembly::derive(parentService, pattern.transactionProjection(),
                                 access ? &*access : nullptr));
}

} // namespace fabric
