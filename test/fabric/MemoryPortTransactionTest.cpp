#include "Fabric/IR/MemoryPortTransaction.h"

#include "Dataflow/IR/DataflowDialect.h"
#include "Dataflow/IR/OperationSchema.h"

#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <cstdlib>
#include <optional>
#include <string>
#include <utility>

using namespace dataflow;
using namespace dataflow::semantics;
using namespace fabric;
using namespace loom::fabric;

namespace {

constexpr llvm::StringLiteral fixture = R"mlir(
module {
  func.func @plain_vector(%mem: memref<8xf32>, %address: index, %ctrl: none)
      -> (vector<4xf32>, none) {
    %data, %done = dataflow.load %mem[%address] %ctrl
        : memref<8xf32>, vector<4xf32>
    return %data, %done : vector<4xf32>, none
  }

  func.func @masked_vector(%mem: memref<8xf32>, %address: index,
                           %mask: vector<4xi1>, %ctrl: none)
      -> (vector<4xf32>, none) {
    %data, %done = dataflow.load %mem[%address] %ctrl mask %mask
        : memref<8xf32>, vector<4xf32>
    return %data, %done : vector<4xf32>, none
  }

  func.func @per_lane_vector(%mem: memref<8xf32>, %address: index,
                             %mask: vector<4xi1>, %ctrl: none)
      -> (vector<4xf32>, none) {
    %data, %done = dataflow.load %mem[%address] %ctrl mask %mask
        {contract = #dataflow.atomic_access<
            ordering = monotonic, sync_scope = <system>,
            source_alignment_bytes = 4, vector_granularity = per_lane>}
        : memref<8xf32>, vector<4xf32>
    return %data, %done : vector<4xf32>, none
  }

  func.func @whole_payload(%mem: memref<8xvector<4xf32>>, %address: index,
                           %ctrl: none) -> (vector<4xf32>, none) {
    %data, %done = dataflow.load %mem[%address] %ctrl
        {contract = #dataflow.atomic_access<
            ordering = acquire, sync_scope = <system>,
            source_alignment_bytes = 16,
            vector_granularity = whole_payload>}
        : memref<8xvector<4xf32>>
    return %data, %done : vector<4xf32>, none
  }

  func.func @fence(%ctrl: none) -> none {
    %done = dataflow.fence %ctrl
        {contract = #dataflow.fence_contract<ordering = seq_cst,
                                             sync_scope = <system>>}
    return %done : none
  }
}
)mlir";

[[noreturn]] void fail(llvm::StringRef test, const llvm::Twine &message) {
  llvm::errs() << test << ": " << message << '\n';
  std::exit(EXIT_FAILURE);
}

void require(llvm::StringRef test, bool condition, const llvm::Twine &message) {
  if (!condition)
    fail(test, message);
}

template <typename T> T take(llvm::StringRef test, llvm::Expected<T> value) {
  if (!value)
    fail(test, llvm::toString(value.takeError()));
  return std::move(*value);
}

template <typename T>
void expectRejected(llvm::StringRef test, llvm::Expected<T> value) {
  if (value)
    fail(test, "accepted an invalid transaction plan");
  llvm::consumeError(value.takeError());
}

mlir::Operation *findActor(mlir::ModuleOp module, llvm::StringRef name) {
  mlir::Operation *actor = nullptr;
  module.walk([&](mlir::func::FuncOp function) {
    if (function.getSymName() != name)
      return;
    function.walk([&](mlir::Operation *operation) {
      if (operation->getName().getDialectNamespace() == "dataflow")
        actor = operation;
    });
  });
  if (!actor)
    fail(name, "actor was not found");
  return actor;
}

ResourceContract contractWithTransactions(std::uint32_t count) {
  ResourceContractDeclaration declaration;
  declaration.states = {ResourceStateDeclaration{
      StateKey(0),
      {CapacityDimensionDeclaration{CapacityDimensionKey(0), CapacityUnits(1),
                                    CapacityUnits(0)}}}};
  declaration.requesters = {RequesterKey(0)};
  declaration.eligibilityCount = 1;
  declaration.eventCount = 1;
  declaration.timingContracts = {
      TimingContractDeclaration{TimingContractKey(0), {0}}};

  UsePatternDeclaration pattern{UsePatternKey(0),
                                RequesterKey(0),
                                EligibilityKey(0),
                                EventKey(0),
                                EventKey(0),
                                std::nullopt,
                                TimingContractKey(0),
                                {},
                                {}};
  pattern.internalTransactions.resize(count);
  declaration.usePatterns.push_back(std::move(pattern));
  return take("resource contract", ResourceContract::create(declaration));
}

FabricMemoryOperationPortRef operationPortRef(std::uint64_t memory,
                                              std::uint64_t port) {
  return FabricMemoryOperationPortRef{FabricMemoryOccurrenceRef(memory), port};
}

MemoryOperationPortResourceView
operationPort(ResourceContract contract,
              MemoryPortTransactionProjection projection) {
  return take("operation port",
              MemoryOperationPortResourceView::create(
                  operationPortRef(7, 2), std::move(contract), {projection}));
}

struct ActorInputs {
  CanonicalActorSchemaProjection actor;
  CanonicalService service;
  std::optional<CanonicalMemoryAccessView> access;
};

ActorInputs inputsFor(mlir::ModuleOp module, llvm::StringRef name) {
  mlir::Operation *operation = findActor(module, name);
  CanonicalActorSchemaProjection actor =
      take(name, projectRegisteredActorSchemaProjection(operation));
  CanonicalService service = take(name, CanonicalService::forActor(operation));
  if (name == "fence")
    return ActorInputs{std::move(actor), std::move(service), std::nullopt};
  return ActorInputs{std::move(actor), std::move(service),
                     take(name, getCanonicalMemoryAccessView(operation))};
}

MemoryOperationPatternView
pattern(const MemoryOperationPortResourceView &port) {
  return take("operation pattern",
              port.operationPattern(port.usePatterns().front()));
}

void portOwnsCompleteResourceContract() {
  MemoryOperationPortResourceView port = operationPort(
      contractWithTransactions(1), MemoryPortTransactionProjection::Direct);
  require(__func__,
          port.resourceContract().stateCount() == 1 &&
              port.resourceContract().usePatternCount() == 1,
          "the complete ResourceContract was not retained");
  require(__func__,
          port.resourceStates().size() == 1 &&
              port.resourceStates().front().ordinal == 0 &&
              port.usePatterns().size() == 1 &&
              port.usePatterns().front().ordinal == 0,
          "owner-relative resource references are incomplete");
  const FabricInventoryOwnerRef expectedOwner =
      FabricInventoryOwnerRef::of(port.owner());
  require(__func__,
          port.resourceStates().front().owner ==
                  FabricResourceStateOwnerRef(expectedOwner) &&
              port.usePatterns().front().owner ==
                  FabricUsePatternOwnerRef(expectedOwner),
          "resource references do not retain the exact port owner");

  MemoryOperationPatternView selected =
      take(__func__, port.operationPattern(port.usePatterns().front()));
  require(__func__,
          selected.operationPort().owner() == port.owner() &&
              selected.usePatternRef() == port.usePatterns().front() &&
              selected.transactionProjection() ==
                  MemoryPortTransactionProjection::Direct,
          "the operation pattern changed its owner-defined identity");

  const FabricMemoryOperationPortRef other = operationPortRef(7, 3);
  const FabricUsePatternRef wrongOwner{
      FabricUsePatternOwnerRef(FabricInventoryOwnerRef::of(other)), 0};
  expectRejected<MemoryOperationPatternView>(__func__,
                                             port.operationPattern(wrongOwner));

  FabricUsePatternRef unknown = port.usePatterns().front();
  unknown.ordinal = 1;
  expectRejected<MemoryOperationPatternView>(__func__,
                                             port.operationPattern(unknown));

  expectRejected<MemoryOperationPortResourceView>(
      __func__, MemoryOperationPortResourceView::create(
                    operationPortRef(7, 2), contractWithTransactions(1), {}));
  expectRejected<MemoryOperationPortResourceView>(
      __func__, MemoryOperationPortResourceView::create(
                    operationPortRef(7, 2), contractWithTransactions(1),
                    {static_cast<MemoryPortTransactionProjection>(255)}));
}

void projectionCodecIsStable() {
  require(__func__,
          getCanonicalTag(MemoryPortTransactionProjection::Direct) == 0 &&
              getCanonicalTag(
                  MemoryPortTransactionProjection::ActiveLanesRowMajor) == 1,
          "transaction projection tags changed");
  require(__func__,
          take(__func__, decodeMemoryPortTransactionProjection(0)) ==
                  MemoryPortTransactionProjection::Direct &&
              take(__func__, decodeMemoryPortTransactionProjection(1)) ==
                  MemoryPortTransactionProjection::ActiveLanesRowMajor,
          "transaction projection tags do not round-trip");
  expectRejected<MemoryPortTransactionProjection>(
      __func__, decodeMemoryPortTransactionProjection(2));
}

void directPreservesOneParent(mlir::ModuleOp module) {
  ActorInputs input = inputsFor(module, "plain_vector");
  MemoryOperationPortResourceView port = operationPort(
      contractWithTransactions(1), MemoryPortTransactionProjection::Direct);
  MemoryPortTransactionPlan plan = take(
      __func__, deriveMemoryPortTransactionPlan(pattern(port), input.actor,
                                                input.service, input.access));

  require(__func__, plan.parentService().kind() == ServiceKind::MemoryRead,
          "the parent service changed");
  require(__func__, plan.transactions().size() == 1,
          "Direct did not produce exactly one child");
  const MemoryPortChildTransaction &child = plan.transactions().front();
  require(__func__,
          child.ordinal() == 0 &&
              child.activation().kind() == MemoryChildActivationKind::Always &&
              child.projection().kind() ==
                  MemoryChildProjectionKind::ParentRequest,
          "Direct child semantics differ");
  require(__func__, plan.assembly().results().size() == 1,
          "the load result assembly is incomplete");
  require(__func__,
          plan.assembly().results().front().role() == ServiceValueRole::Data &&
              plan.assembly().results().front().strategy() ==
                  MemoryResultAssemblyStrategy::PassThroughParent,
          "unmasked Direct assembly differs");
  require(__func__,
          plan.assembly().retirement() ==
              MemoryParentRetirement::SingleParentRetirement,
          "the parent retirement identity changed");
}

void maskedDirectCanCompleteLocally(mlir::ModuleOp module) {
  ActorInputs input = inputsFor(module, "masked_vector");
  MemoryOperationPortResourceView port = operationPort(
      contractWithTransactions(1), MemoryPortTransactionProjection::Direct);
  MemoryPortTransactionPlan plan = take(
      __func__, deriveMemoryPortTransactionPlan(pattern(port), input.actor,
                                                input.service, input.access));

  require(__func__,
          plan.transactions().front().activation().kind() ==
              MemoryChildActivationKind::ParentMaskAny,
          "masked Direct did not suppress an all-zero parent mask");
  require(__func__,
          plan.assembly().results().front().strategy() ==
              MemoryResultAssemblyStrategy::ParentResponseOrZeroOnEmptyMask,
          "masked Direct lacks local all-zero result production");
}

void activeLanesAreRowMajorChildren(mlir::ModuleOp module) {
  ActorInputs input = inputsFor(module, "per_lane_vector");
  MemoryOperationPortResourceView port =
      operationPort(contractWithTransactions(4),
                    MemoryPortTransactionProjection::ActiveLanesRowMajor);
  MemoryPortTransactionPlan plan = take(
      __func__, deriveMemoryPortTransactionPlan(pattern(port), input.actor,
                                                input.service, input.access));

  require(__func__, plan.transactions().size() == 4,
          "row-major projection did not create one child per lane");
  for (std::uint64_t lane = 0; lane != 4; ++lane) {
    const MemoryPortChildTransaction &child = plan.transactions()[lane];
    require(__func__,
            child.ordinal() == lane &&
                child.activation().kind() ==
                    MemoryChildActivationKind::ParentMaskLane &&
                child.activation().lane() == lane &&
                child.projection().kind() ==
                    MemoryChildProjectionKind::ElementLane &&
                child.projection().lane() == lane,
            "row-major child order or activation differs");
  }

  const MemoryResultAssembly &result = plan.assembly().results().front();
  require(__func__,
          result.strategy() ==
                  MemoryResultAssemblyStrategy::RowMajorLaneValues &&
              result.laneCount() == 4 &&
              result.inactiveValue() == MemoryInactiveAssemblyValue::ZeroBits,
          "masked row-major assembly differs");

  MemoryOperationPortResourceView tooSmall =
      operationPort(contractWithTransactions(3),
                    MemoryPortTransactionProjection::ActiveLanesRowMajor);
  expectRejected<MemoryPortTransactionPlan>(
      "internal transaction capacity",
      deriveMemoryPortTransactionPlan(pattern(tooSmall), input.actor,
                                      input.service, input.access));
}

void projectionLegalityIsExact(mlir::ModuleOp module) {
  ActorInputs perLane = inputsFor(module, "per_lane_vector");
  MemoryOperationPortResourceView direct = operationPort(
      contractWithTransactions(1), MemoryPortTransactionProjection::Direct);
  expectRejected<MemoryPortTransactionPlan>(
      "per-lane Direct",
      deriveMemoryPortTransactionPlan(pattern(direct), perLane.actor,
                                      perLane.service, perLane.access));

  ActorInputs whole = inputsFor(module, "whole_payload");
  MemoryOperationPortResourceView lanes =
      operationPort(contractWithTransactions(4),
                    MemoryPortTransactionProjection::ActiveLanesRowMajor);
  expectRejected<MemoryPortTransactionPlan>(
      "whole-payload lanes",
      deriveMemoryPortTransactionPlan(pattern(lanes), whole.actor,
                                      whole.service, whole.access));

  ActorInputs fence = inputsFor(module, "fence");
  MemoryPortTransactionPlan fencePlan =
      take("fence Direct",
           deriveMemoryPortTransactionPlan(pattern(direct), fence.actor,
                                           fence.service, fence.access));
  require("fence Direct",
          fencePlan.transactions().size() == 1 &&
              fencePlan.assembly().results().empty(),
          "fence Direct created result assembly or lost its child");
  expectRejected<MemoryPortTransactionPlan>(
      "fence lanes",
      deriveMemoryPortTransactionPlan(pattern(lanes), fence.actor,
                                      fence.service, fence.access));
}

} // namespace

int main() {
  mlir::DialectRegistry registry;
  registry
      .insert<DataflowDialect, mlir::func::FuncDialect, mlir::DLTIDialect>();
  mlir::MLIRContext context(registry,
                            mlir::MLIRContext::Threading::DISABLED);
  context.loadAllAvailableDialects();
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(fixture, &context);
  if (!module)
    fail("fixture", "failed to parse");

  portOwnsCompleteResourceContract();
  projectionCodecIsStable();
  directPreservesOneParent(*module);
  maskedDirectCanCompleteLocally(*module);
  activeLanesAreRowMajorChildren(*module);
  projectionLegalityIsExact(*module);
  return EXIT_SUCCESS;
}
