//===- DFGSimulatorAtomicMemory.cpp - Scalar atomic memory semantics ------===//
//
// The logical byte store owns values, MemoryAtomicOrder owns each object's
// modification order and reads-from choices, and MemorySynchronization owns
// sequenced-before, synchronizes-with, and happens-before. This provider only
// chooses one deterministic legal action and binds those existing owners.
//
//===----------------------------------------------------------------------===//

#include "DFGSimulatorInternal.h"

#include "Dataflow/IR/DataflowOps.h"

#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>
#include <system_error>
#include <utility>

namespace loom::sim {
namespace LLVM_LIBRARY_VISIBILITY_NAMESPACE detail {
namespace {

constexpr std::uint64_t kSystemSyncDomain = 0;
constexpr std::uint64_t kSingleThreadSyncDomain = 1;

enum class AtomicEffectShape { Read, Write, ReadWrite };

struct AtomicRelation {
  std::optional<AtomicReadId> read;
  std::optional<AtomicVersionId> version;
};

bool fail(SimulatorState &state, RunFailure failure, llvm::Error error) {
  state.diagnostics.push_back(llvm::toString(std::move(error)));
  state.failure = failure;
  return false;
}

bool fail(SimulatorState &state, RunFailure failure, llvm::StringRef message) {
  state.diagnostics.push_back(message.str());
  state.failure = failure;
  return false;
}

llvm::Expected<SyncDomainId>
resolveSyncDomain(dataflow::SyncScopeRefAttr scope) {
  switch (scope.getKind()) {
  case dataflow::SyncScopeKind::System:
    return SyncDomainId(kSystemSyncDomain);
  case dataflow::SyncScopeKind::SingleThread:
    return SyncDomainId(kSingleThreadSyncDomain);
  case dataflow::SyncScopeKind::Target:
    return llvm::createStringError(
        std::errc::not_supported,
        "target-specific synchronization scope has no DFG-sim domain "
        "resolver");
  }
  llvm_unreachable("closed synchronization scope");
}

std::optional<SyncRoleKind> operationRole(dataflow::AtomicOrdering ordering,
                                          AtomicEffectShape shape) {
  switch (ordering) {
  case dataflow::AtomicOrdering::Unordered:
  case dataflow::AtomicOrdering::Monotonic:
    return std::nullopt;
  case dataflow::AtomicOrdering::Acquire:
    return SyncRoleKind::Acquire;
  case dataflow::AtomicOrdering::Release:
    return SyncRoleKind::Release;
  case dataflow::AtomicOrdering::AcqRel:
    return SyncRoleKind::AcqRel;
  case dataflow::AtomicOrdering::SeqCst:
    if (shape == AtomicEffectShape::Read)
      return SyncRoleKind::Acquire;
    if (shape == AtomicEffectShape::Write)
      return SyncRoleKind::Release;
    return SyncRoleKind::AcqRel;
  }
  llvm_unreachable("closed atomic ordering");
}

bool isSequentiallyConsistent(dataflow::AtomicOrdering ordering) {
  return ordering == dataflow::AtomicOrdering::SeqCst;
}

std::optional<ReadyMemoryAction> projectAtomicAction(mlir::Operation *operation,
                                                     SimulatorState &state) {
  MemoryActionProjection projection =
      projectReadyMemoryAction(operation, state);
  for (std::string &diagnostic : projection.diagnostics)
    state.diagnostics.push_back(std::move(diagnostic));
  if (!projection.ready)
    return std::nullopt;

  const MemoryActorExecutionPlan &plan = *state.currentActorPlan->memory;
  ReadyMemoryAction &ready = *projection.ready;
  if (plan.access.isVector() || ready.activeLanes.getBitWidth() != 1 ||
      !ready.activeLanes[0] || ready.slots.size() != 1) {
    fail(state, RunFailure::ProviderInvariant,
         "admitted scalar atomic action did not resolve exactly one object");
    return std::nullopt;
  }
  return std::move(*projection.ready);
}

bool admitAtomicMemoryAction(const ReadyMemoryAction &ready,
                             SimulatorState &state) {
  if (!state.memoryActions.queryCrossKind(ready.action).empty())
    return fail(state, RunFailure::UnsupportedCapability,
                "mixed atomic/plain write hazard has no exact DFG "
                "value/version correspondence");
  if (state.memoryActions.hasInexactAtomicHazard(ready.action))
    return fail(state, RunFailure::UnsupportedCapability,
                "overlapping atomic actions do not share one "
                "AtomicObjectKey");
  return true;
}

AtomicObjectKey atomicObjectKey(const ReadyMemoryAction &ready,
                                const MemoryActorExecutionPlan &plan) {
  return AtomicObjectKey{ready.view.memory->logicalRootId, ready.slots.front(),
                         plan.elementLayout.byteCount};
}

bool validateDynamicAlignment(const ReadyMemoryAction &ready,
                              std::uint64_t sourceAlignment,
                              SimulatorState &state) {
  if (sourceAlignment != 0 && ready.slots.front() % sourceAlignment == 0)
    return true;
  return fail(state, RunFailure::ProviderInvariant,
              "atomic address violates its source alignment guarantee");
}

llvm::Expected<AtomicVersionId>
currentAtomicVersion(SimulatorState &state, const AtomicObjectKey &key) {
  (void)memorySynchronization(state);
  if (auto order = state.memoryOrder->modificationOrder(key))
    return order->back();
  return state.memoryOrder->initializeObject(key);
}

std::optional<MemoryOrderFrontierId> publishAtomicRelation(
    const ReadyMemoryAction &ready, const AtomicRelation &relation,
    dataflow::AtomicOrdering ordering, AtomicEffectShape shape,
    SyncDomainId domain, SimulatorState &state) {
  MemorySynchronization &synchronization = memorySynchronization(state);
  auto declared =
      synchronization.declareEffectSequencedAfter(ready.ctrlFrontier);
  if (!declared) {
    fail(state, RunFailure::ProviderInvariant, declared.takeError());
    return std::nullopt;
  }
  const SyncEffectId effect = *declared;

  if (relation.version) {
    if (llvm::Error error = synchronization.registerWrite(
            effect, domain, *relation.version, relation.read)) {
      fail(state, RunFailure::ProviderInvariant, std::move(error));
      return std::nullopt;
    }
  } else if (relation.read) {
    if (llvm::Error error =
            synchronization.registerRead(effect, domain, *relation.read)) {
      fail(state, RunFailure::ProviderInvariant, std::move(error));
      return std::nullopt;
    }
  }

  if (std::optional<SyncRoleKind> role = operationRole(ordering, shape)) {
    if (llvm::Error error =
            synchronization.declareOperationRole(effect, *role)) {
      fail(state, RunFailure::ProviderInvariant, std::move(error));
      return std::nullopt;
    }
  }

  if (isSequentiallyConsistent(ordering)) {
    if (llvm::Error error =
            synchronization.appendSequentiallyConsistent(effect, domain)) {
      fail(state, RunFailure::ProviderInvariant, std::move(error));
      return std::nullopt;
    }
  }
  state.memoryActions.retain(ready.action, effect, synchronization);
  return state.memoryOrderFrontiers.internCanonical(effect);
}

PrimitiveValue exceptionalRmwResult(const PrimitiveValue &oldValue,
                                    const PrimitiveValue &update) {
  if (oldValue.state == PrimitiveValueState::Poison ||
      update.state == PrimitiveValueState::Poison)
    return PrimitiveValue::poison();
  return PrimitiveValue::undef();
}

llvm::Expected<PrimitiveValue> evaluateIntegerRmw(dataflow::AtomicRmwKind kind,
                                                  const llvm::APInt &oldValue,
                                                  const llvm::APInt &update) {
  switch (kind) {
  case dataflow::AtomicRmwKind::Xchg:
    return PrimitiveValue::integer(update);
  case dataflow::AtomicRmwKind::Add:
    return PrimitiveValue::integer(oldValue + update);
  case dataflow::AtomicRmwKind::Sub:
    return PrimitiveValue::integer(oldValue - update);
  case dataflow::AtomicRmwKind::And:
    return PrimitiveValue::integer(oldValue & update);
  case dataflow::AtomicRmwKind::Nand:
    return PrimitiveValue::integer(~(oldValue & update));
  case dataflow::AtomicRmwKind::Or:
    return PrimitiveValue::integer(oldValue | update);
  case dataflow::AtomicRmwKind::Xor:
    return PrimitiveValue::integer(oldValue ^ update);
  case dataflow::AtomicRmwKind::Max:
    return PrimitiveValue::integer(oldValue.sgt(update) ? oldValue : update);
  case dataflow::AtomicRmwKind::Min:
    return PrimitiveValue::integer(oldValue.sle(update) ? oldValue : update);
  case dataflow::AtomicRmwKind::UMax:
    return PrimitiveValue::integer(oldValue.ugt(update) ? oldValue : update);
  case dataflow::AtomicRmwKind::UMin:
    return PrimitiveValue::integer(oldValue.ule(update) ? oldValue : update);
  case dataflow::AtomicRmwKind::UIncWrap:
    return PrimitiveValue::integer(oldValue.uge(update)
                                       ? llvm::APInt(oldValue.getBitWidth(), 0)
                                       : oldValue + 1);
  case dataflow::AtomicRmwKind::UDecWrap:
    return PrimitiveValue::integer(
        oldValue.isZero() || oldValue.ugt(update) ? update : oldValue - 1);
  case dataflow::AtomicRmwKind::USubCond:
    return PrimitiveValue::integer(oldValue.uge(update) ? oldValue - update
                                                        : oldValue);
  case dataflow::AtomicRmwKind::USubSat:
    return PrimitiveValue::integer(oldValue.ult(update)
                                       ? llvm::APInt(oldValue.getBitWidth(), 0)
                                       : oldValue - update);
  case dataflow::AtomicRmwKind::FAdd:
  case dataflow::AtomicRmwKind::FSub:
  case dataflow::AtomicRmwKind::FMax:
  case dataflow::AtomicRmwKind::FMin:
  case dataflow::AtomicRmwKind::FMaximum:
  case dataflow::AtomicRmwKind::FMinimum:
  case dataflow::AtomicRmwKind::FMaximumNum:
  case dataflow::AtomicRmwKind::FMinimumNum:
    break;
  }
  return llvm::createStringError(
      std::errc::invalid_argument,
      "floating-point atomic RMW kind received an integer object");
}

llvm::Expected<PrimitiveValue>
evaluateFloatingRmw(dataflow::AtomicRmwKind kind, llvm::APFloat oldValue,
                    const llvm::APFloat &update) {
  switch (kind) {
  case dataflow::AtomicRmwKind::Xchg:
    return PrimitiveValue::floating(update);
  case dataflow::AtomicRmwKind::FAdd:
    (void)oldValue.add(update, llvm::APFloat::rmNearestTiesToEven);
    return PrimitiveValue::floating(oldValue);
  case dataflow::AtomicRmwKind::FSub:
    (void)oldValue.subtract(update, llvm::APFloat::rmNearestTiesToEven);
    return PrimitiveValue::floating(oldValue);
  case dataflow::AtomicRmwKind::FMax:
    return PrimitiveValue::floating(llvm::maxnum(oldValue, update));
  case dataflow::AtomicRmwKind::FMin:
    return PrimitiveValue::floating(llvm::minnum(oldValue, update));
  case dataflow::AtomicRmwKind::FMaximum:
    return PrimitiveValue::floating(llvm::maximum(oldValue, update));
  case dataflow::AtomicRmwKind::FMinimum:
    return PrimitiveValue::floating(llvm::minimum(oldValue, update));
  case dataflow::AtomicRmwKind::FMaximumNum:
    return PrimitiveValue::floating(llvm::maximumnum(oldValue, update));
  case dataflow::AtomicRmwKind::FMinimumNum:
    return PrimitiveValue::floating(llvm::minimumnum(oldValue, update));
  case dataflow::AtomicRmwKind::Add:
  case dataflow::AtomicRmwKind::Sub:
  case dataflow::AtomicRmwKind::And:
  case dataflow::AtomicRmwKind::Nand:
  case dataflow::AtomicRmwKind::Or:
  case dataflow::AtomicRmwKind::Xor:
  case dataflow::AtomicRmwKind::Max:
  case dataflow::AtomicRmwKind::Min:
  case dataflow::AtomicRmwKind::UMax:
  case dataflow::AtomicRmwKind::UMin:
  case dataflow::AtomicRmwKind::UIncWrap:
  case dataflow::AtomicRmwKind::UDecWrap:
  case dataflow::AtomicRmwKind::USubCond:
  case dataflow::AtomicRmwKind::USubSat:
    break;
  }
  return llvm::createStringError(
      std::errc::invalid_argument,
      "integer atomic RMW kind received a floating-point object");
}

llvm::Expected<Token> evaluateAtomicRmw(dataflow::AtomicRmwKind kind,
                                        const Token &oldToken,
                                        const Token &updateToken,
                                        mlir::Type type,
                                        unsigned indexBitWidth) {
  auto oldValue = primitiveValueFromToken(oldToken, type, indexBitWidth);
  if (!oldValue)
    return oldValue.takeError();
  auto update = primitiveValueFromToken(updateToken, type, indexBitWidth);
  if (!update)
    return update.takeError();
  if (kind == dataflow::AtomicRmwKind::Xchg)
    return tokenFromPrimitiveValue(*update, type);
  if (!oldValue->isDefined() || !update->isDefined())
    return tokenFromPrimitiveValue(exceptionalRmwResult(*oldValue, *update),
                                   type);

  llvm::Expected<PrimitiveValue> result =
      mlir::isa<mlir::IntegerType>(type)
          ? evaluateIntegerRmw(kind, *oldValue->bits, *update->bits)
          : evaluateFloatingRmw(
                kind,
                llvm::APFloat(
                    mlir::cast<mlir::FloatType>(type).getFloatSemantics(),
                    *oldValue->bits),
                llvm::APFloat(
                    mlir::cast<mlir::FloatType>(type).getFloatSemantics(),
                    *update->bits));
  if (!result)
    return result.takeError();
  return tokenFromPrimitiveValue(*result, type);
}

std::optional<SyncDomainId> admitDomain(dataflow::SyncScopeRefAttr scope,
                                        SimulatorState &state) {
  auto domain = resolveSyncDomain(scope);
  if (!domain) {
    fail(state, RunFailure::UnsupportedCapability, domain.takeError());
    return std::nullopt;
  }
  return *domain;
}

} // namespace

bool fireAtomicLoad(dataflow::LoadOp op, SimulatorState &state) {
  auto ready = projectAtomicAction(op, state);
  if (!ready)
    return false;
  const auto contract =
      mlir::cast<dataflow::AtomicAccessContractAttr>(op.getContractAttr());
  if (!validateDynamicAlignment(*ready, contract.getSourceAlignmentBytes(),
                                state))
    return false;
  auto domain = admitDomain(contract.getSyncScope(), state);
  if (!domain)
    return false;
  ready->action.isWrite = false;
  if (!admitAtomicMemoryAction(*ready, state))
    return false;
  const MemoryActorExecutionPlan &plan = *state.currentActorPlan->memory;
  auto read = prepareMemoryRead(*ready, plan, state);
  if (!read)
    return false;
  const AtomicObjectKey key = atomicObjectKey(*ready, plan);
  auto source = currentAtomicVersion(state, key);
  if (!source)
    return fail(state, RunFailure::ProviderInvariant, source.takeError());
  auto relation = state.memoryOrder->atomicLoad(key, *source);
  if (!relation)
    return fail(state, RunFailure::ProviderInvariant, relation.takeError());
  auto publication = publishAtomicRelation(
      *ready, AtomicRelation{*relation, std::nullopt}, contract.getOrdering(),
      AtomicEffectShape::Read, *domain, state);
  if (!publication)
    return false;

  consumeMemoryIssueInputs(*ready, plan, state);
  emitResultTokenWithMemoryOrder(state, 0, read->data, *publication);
  emitResultTokenWithMemoryOrder(state, 1, noneToken(), *publication);
  return true;
}

bool fireAtomicStore(dataflow::StoreOp op, SimulatorState &state) {
  auto ready = projectAtomicAction(op, state);
  if (!ready)
    return false;
  const auto contract =
      mlir::cast<dataflow::AtomicAccessContractAttr>(op.getContractAttr());
  if (!validateDynamicAlignment(*ready, contract.getSourceAlignmentBytes(),
                                state))
    return false;
  auto domain = admitDomain(contract.getSyncScope(), state);
  if (!domain)
    return false;
  if (!admitAtomicMemoryAction(*ready, state))
    return false;
  const MemoryActorExecutionPlan &plan = *state.currentActorPlan->memory;
  const Token &data = peekInputToken(state, *plan.dataOperandOrdinal);
  auto write = prepareMemoryWrite(data, *ready, plan, state);
  if (!write)
    return false;
  const AtomicObjectKey key = atomicObjectKey(*ready, plan);
  if (auto source = currentAtomicVersion(state, key); !source)
    return fail(state, RunFailure::ProviderInvariant, source.takeError());
  auto version = state.memoryOrder->atomicStore(key);
  if (!version)
    return fail(state, RunFailure::ProviderInvariant, version.takeError());
  auto publication = publishAtomicRelation(
      *ready, AtomicRelation{std::nullopt, *version}, contract.getOrdering(),
      AtomicEffectShape::Write, *domain, state);
  if (!publication)
    return false;

  MemoryView view = ready->view;
  consumeMemoryIssueInputs(*ready, plan, state);
  commitDataflowMemoryWrite(view, *write);
  emitResultTokenWithMemoryOrder(state, 0, noneToken(), *publication);
  return true;
}

bool fireAtomicRmw(dataflow::AtomicRmwOp op, SimulatorState &state) {
  auto ready = projectAtomicAction(op, state);
  if (!ready)
    return false;
  const dataflow::AtomicAccessContractAttr contract =
      op.getContract().getAccess();
  if (!validateDynamicAlignment(*ready, contract.getSourceAlignmentBytes(),
                                state))
    return false;
  auto domain = admitDomain(contract.getSyncScope(), state);
  if (!domain)
    return false;
  if (!admitAtomicMemoryAction(*ready, state))
    return false;
  const MemoryActorExecutionPlan &plan = *state.currentActorPlan->memory;
  auto oldValue = prepareMemoryRead(*ready, plan, state);
  if (!oldValue)
    return false;
  const Token &update = peekInputToken(state, *plan.dataOperandOrdinal);
  auto newValue =
      evaluateAtomicRmw(op.getContract().getKind(), oldValue->data, update,
                        op.getValue().getType(), plan.indexBitWidth);
  if (!newValue)
    return fail(state, RunFailure::ProviderInvariant, newValue.takeError());
  auto write = prepareMemoryWrite(*newValue, *ready, plan, state);
  if (!write)
    return false;

  const AtomicObjectKey key = atomicObjectKey(*ready, plan);
  auto source = currentAtomicVersion(state, key);
  if (!source)
    return fail(state, RunFailure::ProviderInvariant, source.takeError());
  auto relation = state.memoryOrder->atomicRmw(key, *source);
  if (!relation)
    return fail(state, RunFailure::ProviderInvariant, relation.takeError());
  auto publication = publishAtomicRelation(
      *ready, AtomicRelation{relation->read, relation->version},
      contract.getOrdering(), AtomicEffectShape::ReadWrite, *domain, state);
  if (!publication)
    return false;

  MemoryView view = ready->view;
  consumeMemoryIssueInputs(*ready, plan, state);
  commitDataflowMemoryWrite(view, *write);
  emitResultTokenWithMemoryOrder(state, 0, oldValue->data, *publication);
  emitResultTokenWithMemoryOrder(state, 1, noneToken(), *publication);
  return true;
}

bool fireCompareExchange(dataflow::CmpXchgOp op, SimulatorState &state) {
  auto ready = projectAtomicAction(op, state);
  if (!ready)
    return false;
  const dataflow::CompareExchangeContractAttr contract = op.getContract();
  if (!validateDynamicAlignment(*ready, contract.getSourceAlignmentBytes(),
                                state))
    return false;
  auto domain = admitDomain(contract.getSyncScope(), state);
  if (!domain)
    return false;
  const MemoryActorExecutionPlan &plan = *state.currentActorPlan->memory;
  auto oldValue = prepareMemoryRead(*ready, plan, state);
  if (!oldValue)
    return false;
  const Token &expected = peekInputToken(state, *plan.dataOperandOrdinal);
  const Token &desired = peekInputToken(state, *plan.desiredOperandOrdinal);
  if (oldValue->data.valueState != PrimitiveValueState::Defined ||
      expected.valueState != PrimitiveValueState::Defined) {
    return fail(state, RunFailure::UnsupportedCapability,
                "compare-exchange exceptional comparison has no exact "
                "single-path provider");
  }
  auto oldBits = tokenBitPattern(oldValue->data, op.getOld().getType());
  if (!oldBits)
    return fail(state, RunFailure::ProviderInvariant, oldBits.takeError());
  auto expectedBits = tokenBitPattern(expected, op.getExpected().getType());
  if (!expectedBits)
    return fail(state, RunFailure::ProviderInvariant, expectedBits.takeError());
  const bool success = *oldBits == *expectedBits;

  std::optional<DataflowMemoryWrite> write;
  if (success) {
    write = prepareMemoryWrite(desired, *ready, plan, state);
    if (!write)
      return false;
  } else {
    ready->action.isWrite = false;
  }

  if (!admitAtomicMemoryAction(*ready, state))
    return false;

  const AtomicObjectKey key = atomicObjectKey(*ready, plan);
  auto source = currentAtomicVersion(state, key);
  if (!source)
    return fail(state, RunFailure::ProviderInvariant, source.takeError());
  const AtomicCompareExchangeDecision decision =
      success ? AtomicCompareExchangeDecision::Success
              : AtomicCompareExchangeDecision::ComparisonFailure;
  auto relation = state.memoryOrder->compareExchange(key, *source, decision);
  if (!relation)
    return fail(state, RunFailure::ProviderInvariant, relation.takeError());
  const dataflow::AtomicOrdering ordering =
      success ? contract.getSuccessOrdering() : contract.getFailureOrdering();
  auto publication = publishAtomicRelation(
      *ready, AtomicRelation{relation->read, relation->version}, ordering,
      success ? AtomicEffectShape::ReadWrite : AtomicEffectShape::Read, *domain,
      state);
  if (!publication)
    return false;

  MemoryView view = ready->view;
  consumeMemoryIssueInputs(*ready, plan, state);
  if (write)
    commitDataflowMemoryWrite(view, *write);
  emitResultTokenWithMemoryOrder(state, 0, oldValue->data, *publication);
  emitResultTokenWithMemoryOrder(state, 1, boolValueToken(success),
                                 *publication);
  emitResultTokenWithMemoryOrder(state, 2, noneToken(), *publication);
  return true;
}

bool fireFence(dataflow::FenceOp op, SimulatorState &state) {
  const dataflow::FenceContractAttr contract = op.getContract();
  auto domain = admitDomain(contract.getSyncScope(), state);
  if (!domain)
    return false;
  if (!hasInputToken(state, 0))
    return false;
  llvm::SmallVector<SyncEffectId, 2> frontier;
  state.memoryOrderFrontiers.appendCanonicalEffects(
      peekInputToken(state, 0).memoryOrder, frontier);
  MemorySynchronization &synchronization = memorySynchronization(state);
  auto declared = synchronization.declareEffectSequencedAfter(frontier);
  if (!declared)
    return fail(state, RunFailure::ProviderInvariant, declared.takeError());
  const SyncEffectId effect = *declared;
  const SyncRoleKind role =
      *operationRole(contract.getOrdering(), AtomicEffectShape::ReadWrite);
  if (llvm::Error error =
          synchronization.declareFenceRole(effect, role, *domain))
    return fail(state, RunFailure::ProviderInvariant, std::move(error));
  if (isSequentiallyConsistent(contract.getOrdering()))
    if (llvm::Error error =
            synchronization.appendSequentiallyConsistent(effect, *domain))
      return fail(state, RunFailure::ProviderInvariant, std::move(error));
  const MemoryOrderFrontierId publication =
      state.memoryOrderFrontiers.internCanonical(effect);

  (void)popInputToken(state, 0);
  emitResultTokenWithMemoryOrder(state, 0, noneToken(), publication);
  return true;
}

} // namespace LLVM_LIBRARY_VISIBILITY_NAMESPACE detail
} // namespace loom::sim
