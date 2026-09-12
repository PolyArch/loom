#include "CgraGraphActivationRuntime.h"

#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Dataflow/IR/DataflowDialect.h"
#include "Evaluation/NumericValue.h"
#include "Fabric/IR/ResourceContract.h"
#include "Simulator/CGRASimulator.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdlib>
#include <initializer_list>
#include <optional>
#include <system_error>
#include <tuple>
#include <utility>

namespace {

using namespace loom::sim::detail;

[[noreturn]] void fail(llvm::StringRef message) {
  llvm::errs() << "CGRA graph activation test: " << message << '\n';
  std::exit(EXIT_FAILURE);
}

template <typename T> T take(llvm::Expected<T> value) {
  if (!value)
    fail(llvm::toString(value.takeError()));
  return std::move(*value);
}

void require(bool condition, llvm::StringRef message) {
  if (!condition)
    fail(message);
}

class FixedExternalMemoryProvider final
    : public loom::sim::CgraExternalMemoryProvider {
public:
  enum class Completion { Immediate, Deferred };

  explicit FixedExternalMemoryProvider(
      Completion completion = Completion::Immediate)
      : completion_(completion) {}

  static loom::sim::CgraExternalMemoryResponse response(unsigned ordinal) {
    return {{{static_cast<std::uint8_t>(13 + 4 * ordinal), 0, 0, 0}}};
  }

  std::uint64_t outstandingCapacity() const override { return 2; }

  llvm::Expected<loom::sim::CgraExternalMemorySubmission>
  submit(const loom::sim::CgraExternalMemoryRequest &request) override {
    if (request.objectOrdinal != 0 ||
        request.operation != loom::sim::CgraExternalMemoryOperation::Read ||
        request.elements.size() != 1 ||
        request.elements.front().byteOffset != 4 ||
        request.elements.front().byteCount != 4 ||
        !request.elements.front().writeData.empty())
      return llvm::createStringError(
          std::errc::invalid_argument,
          "external memory provider received the wrong logical request");
    requests.push_back(request.id);
    if (completion_ == Completion::Deferred)
      return loom::sim::CgraExternalMemoryPending{};
    return response(requests.size() - 1);
  }

  std::vector<loom::sim::CgraExternalMemoryRequestId> requests;

private:
  Completion completion_;
};

mlir::MLIRContext &context() {
  static mlir::MLIRContext *instance = [] {
    mlir::DialectRegistry registry;
    registry.insert<dataflow::DataflowDialect, mlir::arith::ArithDialect,
                    mlir::DLTIDialect, mlir::func::FuncDialect>();
    auto *result =
        new mlir::MLIRContext(registry, mlir::MLIRContext::Threading::DISABLED);
    result->loadAllAvailableDialects();
    return result;
  }();
  return *instance;
}

dataflow::CanonicalDataflowArtifact program() {
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  dataflow.graph private @local(
      %start: none, %lhs: i32, %rhs: i32) -> (i32)
      attributes {
        input_segments = array<i32: 2, 0, 0>,
        result_segments = array<i32: 1, 0, 0>
      } {
    %sum = arith.addi %lhs, %rhs : i32
    %published:2 = dataflow.sync %start, %sum
        : (none, i32) -> (none, i32)
    dataflow.graph.return values(%published#1 : i32) streams() memories()
        complete(%published#0 : none)
  }
  dataflow.thread private @worker
      domain(#dataflow.thread_domain<dense>)(%lhs: i32, %rhs: i32)
      ctrl (%ctrl: none) {
    %value, %done = dataflow.graph.launch @local deps(%ctrl)
        values(%lhs, %rhs) stream_inputs() memories() stream_outputs()
        : (none, i32, i32) -> (i32, none)
    dataflow.thread.yield %done : none
  }
  func.func private @host(%lhs: i32, %rhs: i32) {
    %thread = dataflow.thread.launch @worker(%lhs, %rhs)
        : (i32, i32) -> !dataflow.thread_token
    return
  }
}
)mlir",
                                                        &context());
  if (!module)
    fail("failed to parse graph activation fixture");
  return take(dataflow::finalizeCanonicalDataflow(module.get()));
}

dataflow::CanonicalDataflowArtifact memoryProgram() {
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module attributes {
  dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 64>>
} {
  dataflow.graph private @load(
      %start: none, %address: index, %memory: memref<4xi32>) -> (i32)
      attributes {
        input_segments = array<i32: 1, 0, 1>,
        result_segments = array<i32: 1, 0, 0>
      } {
    %loaded, %loaded_done = dataflow.load %memory[%address] %start
        : memref<4xi32>
    %sum = arith.addi %loaded, %loaded : i32
    %published:2 = dataflow.sync %loaded_done, %sum
        : (none, i32) -> (none, i32)
    dataflow.graph.return values(%published#1 : i32) streams() memories()
        complete(%published#0 : none)
  }
  dataflow.thread private @worker
      domain(#dataflow.thread_domain<dense>)(
          %address: index, %memory: memref<4xi32>) ctrl (%ctrl: none) {
    %value, %done = dataflow.graph.launch @load deps(%ctrl)
        values(%address) stream_inputs() memories(%memory) stream_outputs()
        : (none, index, memref<4xi32>) -> (i32, none)
    dataflow.thread.yield %done : none
  }
  func.func private @host(%address: index, %memory: memref<4xi32>) {
    %thread = dataflow.thread.launch @worker(%address, %memory)
        : (index, memref<4xi32>) -> !dataflow.thread_token
    return
  }
}
)mlir",
                                                        &context());
  if (!module)
    fail("failed to parse graph memory fixture");
  return take(dataflow::finalizeCanonicalDataflow(module.get()));
}

dataflow::CanonicalDataflowArtifact memoryChainProgram() {
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module attributes {
  dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 64>>
} {
  dataflow.graph private @load_then_store(
      %start: none, %load_index: index, %store_index: index,
      %load_memory: memref<4xi32>, %store_memory: memref<4xi32>) -> ()
      attributes {
        input_segments = array<i32: 2, 0, 2>,
        result_segments = array<i32: 0, 0, 0>
      } {
    %value, %load_done =
        dataflow.load %load_memory[%load_index] %start : memref<4xi32>
    %store_done = dataflow.store
        %store_memory[%store_index] %value %load_done : memref<4xi32>
    dataflow.graph.return values() streams() memories()
        complete(%store_done : none)
  }
  dataflow.thread private @memory_worker
      domain(#dataflow.thread_domain<dense>)(
          %load_index: index, %store_index: index,
          %load_memory: memref<4xi32>, %store_memory: memref<4xi32>)
      ctrl (%ctrl: none) {
    %done = dataflow.graph.launch @load_then_store deps(%ctrl)
        values(%load_index, %store_index) stream_inputs()
        memories(%load_memory, %store_memory) stream_outputs()
        : (none, index, index, memref<4xi32>, memref<4xi32>) -> none
    dataflow.thread.yield %done : none
  }
  func.func private @memory_host(
      %load_index: index, %store_index: index,
      %load_memory: memref<4xi32>, %store_memory: memref<4xi32>) {
    %token = dataflow.thread.launch @memory_worker(
        %load_index, %store_index, %load_memory, %store_memory)
        : (index, index, memref<4xi32>, memref<4xi32>)
          -> !dataflow.thread_token
    return
  }
}
)mlir",
                                                        &context());
  if (!module)
    fail("failed to parse internal-memory-chain fixture");
  return take(dataflow::finalizeCanonicalDataflow(module.get()));
}

dataflow::CanonicalDataflowArtifact unsupportedMemoryProgram() {
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module attributes {
  dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 64>>
} {
  dataflow.graph private @volatile_access(
      %start: none, %mem: memref<1xi32>) -> (i32)
      attributes {input_segments = array<i32: 0, 0, 1>,
                  result_segments = array<i32: 1, 0, 0>} {
    %addr = dataflow.constant %start {const_value = 0 : index} : index
    %value, %done = dataflow.load %mem[%addr] %start
        {contract = #dataflow.plain_access<is_volatile = true>}
        : memref<1xi32>
    dataflow.graph.return values(%value : i32) streams() memories()
        complete(%done : none)
  }
  dataflow.graph private @atomic_access(
      %start: none, %mem: memref<1xi32>) -> (i32)
      attributes {input_segments = array<i32: 0, 0, 1>,
                  result_segments = array<i32: 1, 0, 0>} {
    %addr = dataflow.constant %start {const_value = 0 : index} : index
    %value, %done = dataflow.load %mem[%addr] %start
        {contract = #dataflow.atomic_access<ordering = acquire,
                                            sync_scope = <system>,
                                            source_alignment_bytes = 4>}
        : memref<1xi32>
    dataflow.graph.return values(%value : i32) streams() memories()
        complete(%done : none)
  }
  dataflow.graph private @atomic_rmw(
      %start: none, %mem: memref<1xi32>) -> (i32)
      attributes {input_segments = array<i32: 0, 0, 1>,
                  result_segments = array<i32: 1, 0, 0>} {
    %addr = dataflow.constant %start {const_value = 0 : index} : index
    %update = dataflow.constant %start {const_value = 1 : i32} : i32
    %old, %done = dataflow.atomic_rmw %mem[%addr] %update %start
        {contract = #dataflow.rmw_contract<
            kind = add,
            access = <ordering = monotonic, sync_scope = <system>,
                      source_alignment_bytes = 4>>}
        : memref<1xi32>
    dataflow.graph.return values(%old : i32) streams() memories()
        complete(%done : none)
  }
  dataflow.graph private @compare_exchange(
      %start: none, %mem: memref<1xi32>) -> (i32, i1)
      attributes {input_segments = array<i32: 0, 0, 1>,
                  result_segments = array<i32: 2, 0, 0>} {
    %addr = dataflow.constant %start {const_value = 0 : index} : index
    %expected = dataflow.constant %start {const_value = 1 : i32} : i32
    %desired = dataflow.constant %start {const_value = 2 : i32} : i32
    %old, %ok, %done = dataflow.cmpxchg
        %mem[%addr] %expected %desired %start
        {contract = #dataflow.cmpxchg_contract<
            success_ordering = acq_rel, failure_ordering = acquire,
            sync_scope = <system>, source_alignment_bytes = 4>}
        : memref<1xi32> -> i1
    dataflow.graph.return values(%old, %ok : i32, i1) streams() memories()
        complete(%done : none)
  }
  dataflow.graph private @fence(%start: none) -> ()
      attributes {input_segments = array<i32: 0, 0, 0>,
                  result_segments = array<i32: 0, 0, 0>} {
    %done = dataflow.fence %start
        {contract = #dataflow.fence_contract<ordering = seq_cst,
                                             sync_scope = <system>>}
    dataflow.graph.return values() streams() memories()
        complete(%done : none)
  }
}
)mlir",
                                                        &context());
  if (!module)
    fail("failed to parse unsupported CGRA memory fixture");
  return take(dataflow::finalizeCanonicalDataflow(module.get()));
}

dataflow::RootedGraphLaunchRef
onlyLaunch(const dataflow::CanonicalDataflowProgramView &view) {
  require(view.rootThreadLaunches().size() == 1 &&
              view.staticGraphLaunches().size() == 1,
          "fixture must have one rooted graph launch");
  return {view.rootThreadLaunches().front().ref,
          view.staticGraphLaunches().front().ref};
}

fabric::ResourceContract resourceContract() {
  using namespace fabric;
  ResourceContractDeclaration declaration;
  declaration.states = {
      {StateKey(0),
       {{CapacityDimensionKey(0), CapacityUnits(1), CapacityUnits(0)}}}};
  declaration.resourceTransitions = {ResourceTransitionKey(0)};
  declaration.timingContracts = {{TimingContractKey(0), {0, 1, 2}}};
  declaration.requesters = {RequesterKey(0)};
  declaration.eligibilityCount = 1;
  declaration.eventCount = 3;
  declaration.usePatterns = {
      {UsePatternKey(0),
       RequesterKey(0),
       EligibilityKey(0),
       EventKey(0),
       EventKey(2),
       CommitDeclaration{EventKey(1), ResourceTransitionKey(0)},
       TimingContractKey(0),
       {{ClaimKey(0), StateKey(0), CapacityDimensionKey(0), CapacityUnits(1)}},
       {}}};
  return take(ResourceContract::create(declaration));
}

loom::sim::SpatialEventCoordinate coordinate(std::uint64_t cycle,
                                             std::uint64_t delta = 0) {
  return {take(loom::evaluation::ExactRatio::get(cycle, 1)), delta};
}

ActorExecutionPlan &semanticActor(PreparedGraphExecution &execution,
                                  mlir::Operation *operation) {
  for (ActorExecutionPlan &actor : execution.actorPlans)
    if (actor.operation == operation)
      return actor;
  fail("canonical actor is absent from prepared graph execution");
}

void appendTransfer(CgraFrozenExecutionPlan &plan,
                    dataflow::CanonicalGraphProducerEndpointRef producer,
                    dataflow::CanonicalGraphConsumerEndpointRef consumer,
                    dataflow::GraphRef graph) {
  const std::uint64_t sink = plan.transport.localTransferSinks.size();
  plan.transport.localTransferSinks.push_back({std::move(consumer)});
  plan.transport.localTransfers.push_back(
      {std::move(producer), graph, sink, 1});
}

void selectExternalMemoryRoles(CgraMemoryActorPlan &plan,
                               mlir::Operation *operation) {
  constexpr std::size_t roleCount =
      static_cast<std::size_t>(
          dataflow::semantics::ServiceValueRole::Completion) +
      1;
  auto service =
      take(dataflow::semantics::CanonicalService::forActor(operation));
  plan.roleSources.resize(roleCount);
  plan.roleDestinations.resize(roleCount);
  for (auto [ordinal, argument] : llvm::enumerate(service.arguments()))
    plan.roleSources[static_cast<std::size_t>(argument.role)] =
        loom::fabric::FabricMemoryHandshakeExternalRoleSource{
            static_cast<loom::fabric::FabricOrdinal>(ordinal)};
  for (auto [ordinal, result] : llvm::enumerate(service.results())) {
    loom::fabric::FabricMemoryHandshakeRoleDestination destination;
    destination.externalEndpoint =
        static_cast<loom::fabric::FabricOrdinal>(ordinal);
    plan.roleDestinations[static_cast<std::size_t>(result.role)] =
        std::move(destination);
  }
}

void graphActivationCoordinatesComputeAndTransport() {
  auto artifact = program();
  const auto &view = artifact.view();
  const dataflow::RootedGraphLaunchRef launch = onlyLaunch(view);
  const dataflow::CanonicalActorView *add = nullptr;
  const dataflow::CanonicalActorView *sync = nullptr;
  for (const dataflow::CanonicalActorView &actor : view.actors()) {
    const auto schema = dataflow::operationSchemaOf(actor.op);
    if (schema == dataflow::OperationSchemaId::ArithAddI)
      add = &actor;
    if (schema == dataflow::OperationSchemaId::DataflowSync)
      sync = &actor;
  }
  require(add && sync, "fixture lacks add or sync actor");
  auto graphView = take(view.resolve(add->graph));
  auto graph = mlir::cast<dataflow::GraphOp>(graphView.op);
  GraphPreparationResult preparedResult =
      take(prepareGraphExecution(artifact.module(), graph));
  auto *prepared = std::get_if<PreparedGraphExecution>(&preparedResult);
  require(prepared, "graph activation preparation failed");

  CgraFrozenExecutionPlan plan;
  std::vector<CgraResourcePatternSelection> selections;
  const fabric::ResourceContract contract = resourceContract();
  for (const dataflow::CanonicalActorView *actor : {add, sync}) {
    ActorExecutionPlan &semantic = semanticActor(*prepared, actor->op);
    const std::uint64_t transitionOffset = plan.computeTransitions.size();
    for (const auto &handshake : semantic.handshakeCases) {
      const std::uint64_t action = plan.physicalUseTimings.size();
      plan.computeTransitions.push_back({handshake.ordinal, action, 1});
      plan.actorTransitionPhysicalUses.push_back(action);
      plan.physicalUseClients.push_back(
          CgraPhysicalUseClientKind::ComputeTransition);
      plan.physicalUseTimings.push_back({action, 0, 1, 2, 0, 2, 1});
      selections.push_back({0, fabric::UsePatternKey(0)});
    }
    plan.computeActors.push_back(
        {actor->ref,
         actor->graph,
         {},
         {},
         transitionOffset,
         static_cast<std::uint32_t>(semantic.handshakeCases.size()),
         std::nullopt,
         0});
  }

  const std::uint64_t producedAction = plan.physicalUseTimings.size();
  plan.physicalUseClients.push_back(
      CgraPhysicalUseClientKind::ProducedTransport);
  plan.physicalUseTimings.push_back({producedAction, 0, 1, 2, 0, 2, 1});
  selections.push_back({0, fabric::UsePatternKey(0)});
  const std::uint64_t consumedAction = plan.physicalUseTimings.size();
  plan.physicalUseClients.push_back(
      CgraPhysicalUseClientKind::ConsumedTransport);
  plan.physicalUseTimings.push_back({consumedAction, 0, 1, 2, 0, 2, 1});
  selections.push_back({0, fabric::UsePatternKey(0)});
  plan.transport.endpointPhysicalUses = {producedAction, consumedAction};
  plan.transport.producedUses.push_back(
      {{dataflow::ActorTokenResultRef{add->ref, 0}}, 0, 1});
  plan.transport.consumedUses.push_back(
      {{dataflow::ActorTokenOperandRef{sync->ref, 1}}, 1, 1});

  appendTransfer(
      plan,
      dataflow::GraphIngressTokenRef{dataflow::GraphStartTokenRef{add->graph}},
      dataflow::ActorTokenOperandRef{sync->ref, 0}, add->graph);
  appendTransfer(plan,
                 dataflow::GraphIngressTokenRef{
                     dataflow::GraphValueInputTokenRef{add->graph, 0}},
                 dataflow::ActorTokenOperandRef{add->ref, 0}, add->graph);
  appendTransfer(plan,
                 dataflow::GraphIngressTokenRef{
                     dataflow::GraphValueInputTokenRef{add->graph, 1}},
                 dataflow::ActorTokenOperandRef{add->ref, 1}, add->graph);
  appendTransfer(plan, dataflow::ActorTokenResultRef{add->ref, 0},
                 dataflow::ActorTokenOperandRef{sync->ref, 1}, add->graph);
  appendTransfer(plan, dataflow::ActorTokenResultRef{sync->ref, 0},
                 dataflow::GraphEgressTokenRef{
                     dataflow::GraphCompletionFrontierTokenRef{add->graph, 0}},
                 add->graph);
  appendTransfer(plan, dataflow::ActorTokenResultRef{sync->ref, 1},
                 dataflow::GraphEgressTokenRef{
                     dataflow::GraphValueOutputTokenRef{add->graph, 0}},
                 add->graph);

  const fabric::ResourceContract *contracts[] = {&contract};
  plan.resources = take(freezeCgraResourceRuntimePlan(contracts, selections));

  SimulatorState state;
  state.graphScope = graph.getOperation();
  initializeRunState(state, *prepared);
  llvm::SmallVector<GraphIngressEmission, 3> ingress;
  state.graphIngressCapture = &ingress;
  mlir::Block &entry = graph.getBody().front();
  seedBlockArgument(state, entry.getArgument(0), noneToken());
  seedBlockArgument(state, entry.getArgument(1),
                    take(tokenFromBitPattern(llvm::APInt(32, 7),
                                             entry.getArgument(1).getType())));
  seedBlockArgument(state, entry.getArgument(2),
                    take(tokenFromBitPattern(llvm::APInt(32, 9),
                                             entry.getArgument(2).getType())));
  seedBlockArgument(state, entry.getArgument(0), noneToken());
  seedBlockArgument(state, entry.getArgument(1),
                    take(tokenFromBitPattern(llvm::APInt(32, 2),
                                             entry.getArgument(1).getType())));
  seedBlockArgument(state, entry.getArgument(2),
                    take(tokenFromBitPattern(llvm::APInt(32, 5),
                                             entry.getArgument(2).getType())));
  state.graphIngressCapture = nullptr;

  auto runtimeTransportGraph =
      take(freezeCgraTransportGraph(plan, view, add->graph, *prepared));
  auto runtime = take(CgraGraphActivationRuntime::create(
      plan, view, launch, add->graph, *prepared, runtimeTransportGraph, state,
      /*captureMicroarchitecture=*/false));
  if (llvm::Error error = runtime.start(coordinate(0), ingress))
    fail(llvm::toString(std::move(error)));

  std::uint64_t committed = 0;
  std::uint64_t retired = 0;
  std::uint64_t publications = 0;
  std::uint64_t physicalEvents = 0;
  for (unsigned iteration = 0; iteration != 64 && runtime.hasPendingEvents();
       ++iteration) {
    auto frame = take(runtime.advance());
    require(frame.has_value(), "pending activation lost its next frame");
    for (const auto &event : frame->actorEvents)
      if (event.kind == CgraActorLifecycleKind::Committed)
        ++committed;
      else
        ++retired;
    publications += frame->publications.size();
    physicalEvents += frame->physicalEvents.size();
  }

  require(!runtime.hasPendingEvents(), "graph activation did not quiesce");
  require(committed == 4 && retired == 4,
          "graph activation did not commit and retire both actor occurrences");
  require(publications == 12,
          "graph activation did not publish every ingress and actor result");
  require(physicalEvents >= 24,
          "graph activation bypassed selected physical lifecycles");
  auto output =
      state.observedOutputs.find(graph.getBody().front().back().getOperand(0));
  require(output != state.observedOutputs.end() && output->second.size() == 2 &&
              take(tokenBitPattern(output->second.front(),
                                   mlir::IntegerType::get(&context(), 32))) ==
                  llvm::APInt(32, 16) &&
              take(tokenBitPattern(output->second.back(),
                                   mlir::IntegerType::get(&context(), 32))) ==
                  llvm::APInt(32, 7),
          "graph activation lost or reordered repeated ingress values");
}

void graphActivationExecutesSelectedLocalMemory() {
  auto artifact = memoryProgram();
  const auto &view = artifact.view();
  const dataflow::RootedGraphLaunchRef launch = onlyLaunch(view);
  const dataflow::CanonicalActorView *load = nullptr;
  const dataflow::CanonicalActorView *add = nullptr;
  const dataflow::CanonicalActorView *sync = nullptr;
  for (const dataflow::CanonicalActorView &actor : view.actors()) {
    const auto schema = dataflow::operationSchemaOf(actor.op);
    if (schema == dataflow::OperationSchemaId::DataflowLoad)
      load = &actor;
    if (schema == dataflow::OperationSchemaId::ArithAddI)
      add = &actor;
    if (schema == dataflow::OperationSchemaId::DataflowSync)
      sync = &actor;
  }
  require(load && add && sync, "memory fixture lacks selected actors");
  auto graphView = take(view.resolve(load->graph));
  auto graph = mlir::cast<dataflow::GraphOp>(graphView.op);
  GraphPreparationResult preparedResult =
      take(prepareGraphExecution(artifact.module(), graph));
  auto *prepared = std::get_if<PreparedGraphExecution>(&preparedResult);
  require(prepared, "memory graph preparation failed");

  CgraFrozenExecutionPlan plan;
  std::vector<CgraResourcePatternSelection> selections;
  const fabric::ResourceContract contract = resourceContract();
  for (const dataflow::CanonicalActorView *actor : {add, sync}) {
    ActorExecutionPlan &semantic = semanticActor(*prepared, actor->op);
    const std::uint64_t transitionOffset = plan.computeTransitions.size();
    for (const auto &handshake : semantic.handshakeCases) {
      const std::uint64_t action = plan.physicalUseTimings.size();
      plan.computeTransitions.push_back({handshake.ordinal, action, 1});
      plan.actorTransitionPhysicalUses.push_back(action);
      plan.physicalUseClients.push_back(
          CgraPhysicalUseClientKind::ComputeTransition);
      plan.physicalUseTimings.push_back({action, 0, 1, 2, 0, 2, 1});
      selections.push_back({0, fabric::UsePatternKey(0)});
    }
    plan.computeActors.push_back(
        {actor->ref,
         actor->graph,
         {},
         {},
         transitionOffset,
         static_cast<std::uint32_t>(semantic.handshakeCases.size()),
         std::nullopt,
         0});
  }

  const std::uint64_t operationAction = plan.physicalUseTimings.size();
  plan.physicalUseClients.push_back(
      CgraPhysicalUseClientKind::MemoryTransition);
  plan.physicalUseTimings.push_back(
      {operationAction, 0, std::nullopt, 1, 0, 1, std::nullopt});
  selections.push_back({1, fabric::UsePatternKey(0)});
  const std::uint64_t serviceAction = plan.physicalUseTimings.size();
  plan.physicalUseClients.push_back(
      CgraPhysicalUseClientKind::MemoryTransition);
  plan.physicalUseTimings.push_back(
      {serviceAction, 0, std::nullopt, 1, 0, 1, std::nullopt});
  selections.push_back({2, fabric::UsePatternKey(0)});

  const loom::fabric::FabricMemoryOccurrenceRef occurrence{};
  const loom::fabric::FabricMemoryOperationPortRef port{occurrence, 0};
  const loom::fabric::LocalMemoryServiceRef service(
      loom::fabric::FabricMemoryServiceRef::local(occurrence));
  plan.memory.rootedUses.push_back(
      {launch, 0, CgraMemoryServiceTarget(service), serviceAction});
  plan.memory.childTransactions.push_back(
      {fabric::MemoryChildActivationKind::Always, std::nullopt,
       fabric::MemoryChildProjectionKind::ParentRequest, std::nullopt});
  plan.memory.resultAssemblies.push_back(
      {dataflow::semantics::ServiceValueRole::Data,
       fabric::MemoryResultAssemblyStrategy::PassThroughParent, std::nullopt,
       std::nullopt});
  plan.memory.actors.push_back(
      {load->ref,
       load->graph,
       occurrence,
       loom::mapping::SpatialMemoryOperationPlacementView(port),
       loom::fabric::FabricMemoryCapabilityAlternativeRef{port, 0},
       ::fabric::serializedMemoryOperationIssueDepth,
       operationAction,
       0,
       1,
       0,
       1,
       0,
       1,
       {},
       {}});
  selectExternalMemoryRoles(plan.memory.actors.back(), load->op);

  appendTransfer(
      plan,
      dataflow::GraphIngressTokenRef{dataflow::GraphStartTokenRef{load->graph}},
      dataflow::ActorTokenOperandRef{load->ref, 2}, load->graph);
  appendTransfer(plan,
                 dataflow::GraphIngressTokenRef{
                     dataflow::GraphValueInputTokenRef{load->graph, 0}},
                 dataflow::ActorTokenOperandRef{load->ref, 1}, load->graph);
  appendTransfer(plan, dataflow::ActorTokenResultRef{load->ref, 0},
                 dataflow::ActorTokenOperandRef{add->ref, 0}, load->graph);
  plan.transport.localTransferSinks.push_back(
      {dataflow::ActorTokenOperandRef{add->ref, 1}});
  plan.transport.localTransfers.back().sinkCount = 2;
  appendTransfer(plan, dataflow::ActorTokenResultRef{load->ref, 1},
                 dataflow::ActorTokenOperandRef{sync->ref, 0}, load->graph);
  appendTransfer(plan, dataflow::ActorTokenResultRef{add->ref, 0},
                 dataflow::ActorTokenOperandRef{sync->ref, 1}, load->graph);
  appendTransfer(plan, dataflow::ActorTokenResultRef{sync->ref, 0},
                 dataflow::GraphEgressTokenRef{
                     dataflow::GraphCompletionFrontierTokenRef{load->graph, 0}},
                 load->graph);
  appendTransfer(plan, dataflow::ActorTokenResultRef{sync->ref, 1},
                 dataflow::GraphEgressTokenRef{
                     dataflow::GraphValueOutputTokenRef{load->graph, 0}},
                 load->graph);

  const fabric::ResourceContract *contracts[] = {&contract, &contract,
                                                 &contract};
  plan.resources = take(freezeCgraResourceRuntimePlan(contracts, selections));

  mlir::Block &entry = graph.getBody().front();
  const auto seedState =
      [&](SimulatorState &state,
          llvm::SmallVectorImpl<GraphIngressEmission> &ingress,
          unsigned repetitions = 1) {
        state.graphScope = graph.getOperation();
        initializeRunState(state, *prepared);
        auto memory = std::make_shared<MemoryValue>();
        memory->logicalRootId = 0;
        for (std::uint32_t element : {3u, 11u, 5u, 7u}) {
          auto bytes = take(encodeMemoryElement(
              take(tokenFromBitPattern(llvm::APInt(32, element),
                                       mlir::IntegerType::get(&context(), 32))),
              mlir::IntegerType::get(&context(), 32), graph.getOperation()));
          memory->bytes.append(bytes.begin(), bytes.end());
        }
        memory->initialized = llvm::SmallBitVector(memory->bytes.size(), true);
        state.graphIngressCapture = &ingress;
        for (unsigned repetition = 0; repetition != repetitions; ++repetition) {
          seedBlockArgument(state, entry.getArgument(0), noneToken());
          seedBlockArgument(state, entry.getArgument(1),
                            indexToken(llvm::APInt(64, 1)));
        }
        state.graphIngressCapture = nullptr;
        state.memories[entry.getArgument(2)] = memory;
        state.memoryViews[entry.getArgument(2)] =
            MemoryView{memory, entry.getArgument(2), 0,
                       mlir::IntegerType::get(&context(), 32)};
      };

  plan.memory.rootedUses.front().target =
      loom::fabric::ManagerEndpointRef(loom::fabric::FabricMemoryEndpointRef{
          loom::fabric::FabricMemoryEndpointOwnerRef::of(occurrence), 0});
  plan.memory.rootedUses.front().localServicePhysicalUseOrdinal.reset();
  SimulatorState externalState;
  llvm::SmallVector<GraphIngressEmission, 2> externalIngress;
  seedState(externalState, externalIngress, 2);
  auto externalTransportGraph =
      take(freezeCgraTransportGraph(plan, view, load->graph, *prepared));
  auto unsupportedRuntime = CgraGraphActivationRuntime::create(
      plan, view, launch, load->graph, *prepared,
      externalTransportGraph, externalState,
      /*captureMicroarchitecture=*/false);
  require(!unsupportedRuntime,
          "manager memory target unexpectedly acquired a CGRA provider");
  require(llvm::errorToErrorCode(unsupportedRuntime.takeError()) ==
              std::make_error_code(std::errc::not_supported),
          "manager memory target did not fail as typed unsupported");

  // A deferred provider response changes when the model consumes it, but not
  // which lifecycle events the model produces: every actor still commits,
  // publishes, and linearizes exactly once.
  struct ActivationEvidence final {
    std::size_t physicalEvents = 0;
    std::size_t actorEvents = 0;
    std::size_t publications = 0;
    std::size_t memoryLinearizations = 0;

    void accumulate(const CgraGraphActivationFrame &frame) {
      physicalEvents += frame.physicalEvents.size();
      actorEvents += frame.actorEvents.size();
      publications += frame.publications.size();
      memoryLinearizations += frame.memoryLinearizations.size();
    }

    bool operator==(const ActivationEvidence &other) const {
      return std::tie(physicalEvents, actorEvents, publications,
                      memoryLinearizations) ==
             std::tie(other.physicalEvents, other.actorEvents,
                      other.publications, other.memoryLinearizations);
    }
  };
  ActivationEvidence immediateEvidence;
  FixedExternalMemoryProvider externalMemory;
  auto externalRuntime = take(CgraGraphActivationRuntime::create(
      plan, view, launch, load->graph, *prepared, externalTransportGraph,
      externalState, /*captureMicroarchitecture=*/false, &externalMemory));
  if (llvm::Error error = externalRuntime.start(coordinate(0), externalIngress))
    fail(llvm::toString(std::move(error)));
  for (unsigned iteration = 0;
       iteration != 96 && externalRuntime.hasPendingEvents(); ++iteration) {
    auto frame = take(externalRuntime.advance());
    require(frame.has_value(),
            "external memory activation lost its pending frame");
    immediateEvidence.accumulate(*frame);
  }
  require(!externalRuntime.hasPendingEvents() &&
              externalMemory.requests.size() == 2 &&
              !(externalMemory.requests[0] == externalMemory.requests[1]),
          "manager memory requests did not retain occurrence identity");

  const auto requireExternalOutput = [&](SimulatorState &state) {
    auto output = state.observedOutputs.find(
        graph.getBody().front().back().getOperand(0));
    require(output != state.observedOutputs.end() &&
                output->second.size() == 2 &&
                take(tokenBitPattern(output->second.front(),
                                     mlir::IntegerType::get(&context(), 32))) ==
                    llvm::APInt(32, 26) &&
                take(tokenBitPattern(output->second.back(),
                                     mlir::IntegerType::get(&context(), 32))) ==
                    llvm::APInt(32, 34),
            "manager memory activation lost or reordered a provider response");
  };
  requireExternalOutput(externalState);

  FixedExternalMemoryProvider deferredMemory(
      FixedExternalMemoryProvider::Completion::Deferred);
  SimulatorState deferredState;
  llvm::SmallVector<GraphIngressEmission, 4> deferredIngress;
  seedState(deferredState, deferredIngress, 2);
  auto deferredRuntime = take(CgraGraphActivationRuntime::create(
      plan, view, launch, load->graph, *prepared, externalTransportGraph,
      deferredState,
      /*captureMicroarchitecture=*/false, &deferredMemory));
  if (llvm::Error error = deferredRuntime.start(coordinate(0), deferredIngress))
    fail(llvm::toString(std::move(error)));
  ActivationEvidence deferredEvidence;
  for (unsigned iteration = 0;
       iteration != 96 && deferredRuntime.hasPendingEvents(); ++iteration) {
    auto frame = take(deferredRuntime.advance());
    if (frame) {
      deferredEvidence.accumulate(*frame);
      continue;
    }
    require(deferredRuntime.waitingForExternalMemory() &&
                !deferredMemory.requests.empty(),
            "deferred memory lost its pending request");
    require(!take(deferredRuntime.advance()).has_value(),
            "host waiting advanced an incomplete model frame");
    const unsigned ordinal = deferredMemory.requests.size() - 1;
    // The same actor and occurrence from a different execution is foreign.
    llvm::Error foreign = deferredRuntime.completeExternalMemory(
        externalMemory.requests[ordinal],
        FixedExternalMemoryProvider::response(ordinal));
    require(static_cast<bool>(foreign),
            "deferred memory accepted another execution's request identity");
    llvm::consumeError(std::move(foreign));
    llvm::Error malformed = deferredRuntime.completeExternalMemory(
        deferredMemory.requests.back(), {{{1, 2, 3}}});
    require(static_cast<bool>(malformed),
            "deferred memory accepted an incomplete logical response");
    llvm::consumeError(std::move(malformed));
    if (llvm::Error error = deferredRuntime.completeExternalMemory(
            deferredMemory.requests.back(),
            FixedExternalMemoryProvider::response(ordinal)))
      fail(llvm::toString(std::move(error)));
    llvm::Error duplicate = deferredRuntime.completeExternalMemory(
        deferredMemory.requests.back(),
        FixedExternalMemoryProvider::response(ordinal));
    require(static_cast<bool>(duplicate),
            "deferred memory accepted the same completion twice");
    llvm::consumeError(std::move(duplicate));
  }
  require(!deferredRuntime.hasPendingEvents() &&
              deferredMemory.requests.size() == 2 &&
              deferredEvidence == immediateEvidence,
          "deferred external memory changed CGRA lifecycle evidence");
  requireExternalOutput(deferredState);

  // The Fabric memory Operation Engine owns how many firings one bound memory
  // actor may hold outstanding. Under the serialized depth the two load
  // firings never overlap: the second is admitted only after the first
  // retires, so exactly one request is ever outstanding and the second reaches
  // the service only once the first has been answered. Depth two lets the
  // second firing issue while the first awaits its response, so both requests
  // are outstanding together and the second reaches the service strictly
  // earlier. This harness answers a deferred request only when the activation
  // has no other work, so it cannot witness a shorter activation; the
  // overlap and the earlier second issue are what the depth owns. The two
  // runs below differ in nothing else: the same provider, the same two
  // requests in the same order, and the same outputs.
  const auto deferredRunOutstandingPeak =
      [&](std::uint64_t issueDepth,
          std::optional<loom::sim::SpatialEventCoordinate> &secondIssue)
      -> std::size_t {
    plan.memory.actors.back().operationIssueDepth = issueDepth;
    FixedExternalMemoryProvider memory(
        FixedExternalMemoryProvider::Completion::Deferred);
    SimulatorState runState;
    llvm::SmallVector<GraphIngressEmission, 4> runIngress;
    seedState(runState, runIngress, 2);
    auto runtime = take(CgraGraphActivationRuntime::create(
        plan, view, launch, load->graph, *prepared, externalTransportGraph,
        runState, /*captureMicroarchitecture=*/false, &memory));
    if (llvm::Error error = runtime.start(coordinate(0), runIngress))
      fail(llvm::toString(std::move(error)));
    std::size_t completed = 0;
    std::size_t peak = 0;
    for (unsigned iteration = 0;
         iteration != 96 && runtime.hasPendingEvents(); ++iteration) {
      auto frame = take(runtime.advance());
      if (frame) {
        if (memory.requests.size() == 2 && !secondIssue)
          secondIssue = frame->coordinate;
        peak = std::max(peak, memory.requests.size() - completed);
        continue;
      }
      require(runtime.waitingForExternalMemory() &&
                  completed != memory.requests.size(),
              "depth-bounded memory lost its pending request");
      peak = std::max(peak, memory.requests.size() - completed);
      // The consistency domain linearizes firings in issue order however the
      // provider answers, so the oldest outstanding request is completed with
      // its own response.
      if (llvm::Error error = runtime.completeExternalMemory(
              memory.requests[completed],
              FixedExternalMemoryProvider::response(completed)))
        fail(llvm::toString(std::move(error)));
      ++completed;
    }
    require(!runtime.hasPendingEvents() && memory.requests.size() == 2 &&
                completed == 2,
            "depth-bounded memory did not retire both firings");
    requireExternalOutput(runState);
    return peak;
  };
  std::optional<loom::sim::SpatialEventCoordinate> pipelinedSecondIssue;
  std::optional<loom::sim::SpatialEventCoordinate> serializedSecondIssue;
  require(deferredRunOutstandingPeak(2, pipelinedSecondIssue) == 2,
          "issue depth two did not overlap its two memory firings");
  require(deferredRunOutstandingPeak(
              ::fabric::serializedMemoryOperationIssueDepth,
              serializedSecondIssue) == 1,
          "the serialized engine overlapped two memory firings");
  require(pipelinedSecondIssue && serializedSecondIssue &&
              loom::sim::compareSpatialEventCoordinates(
                  *pipelinedSecondIssue, *serializedSecondIssue) < 0,
          "issue depth two did not reach the service earlier than the "
          "serialized engine");
  plan.memory.actors.back().operationIssueDepth =
      ::fabric::serializedMemoryOperationIssueDepth;

  plan.memory.rootedUses.front().target = service;
  plan.memory.rootedUses.front().localServicePhysicalUseOrdinal = serviceAction;
  SimulatorState state;
  llvm::SmallVector<GraphIngressEmission, 2> ingress;
  seedState(state, ingress);

  auto runtimeTransportGraph =
      take(freezeCgraTransportGraph(plan, view, load->graph, *prepared));
  auto runtime = take(CgraGraphActivationRuntime::create(
      plan, view, launch, load->graph, *prepared, runtimeTransportGraph, state,
      /*captureMicroarchitecture=*/false));
  if (llvm::Error error = runtime.start(coordinate(0), ingress))
    fail(llvm::toString(std::move(error)));
  std::uint64_t memoryPhysicalEvents = 0;
  std::uint64_t memoryLinearizations = 0;
  for (unsigned iteration = 0; iteration != 96 && runtime.hasPendingEvents();
       ++iteration) {
    auto frame = take(runtime.advance());
    require(frame.has_value(), "pending memory activation lost its frame");
    for (const auto &event : frame->physicalEvents)
      if (event.actionOrdinal == operationAction ||
          event.actionOrdinal == serviceAction)
        ++memoryPhysicalEvents;
    memoryLinearizations += frame->memoryLinearizations.size();
  }
  require(!runtime.hasPendingEvents(), "memory graph did not quiesce");
  require(memoryPhysicalEvents == 6,
          "memory graph did not execute both selected physical actions");
  require(memoryLinearizations == 1,
          "memory graph did not expose its primitive linearization");
  auto output =
      state.observedOutputs.find(graph.getBody().front().back().getOperand(0));
  require(output != state.observedOutputs.end() && output->second.size() == 1 &&
              take(tokenBitPattern(output->second.front(),
                                   mlir::IntegerType::get(&context(), 32))) ==
                  llvm::APInt(32, 22),
          "memory graph produced the wrong loaded value");
}

void graphActivationExecutesExactMemoryInternalConnections() {
  auto artifact = memoryChainProgram();
  const auto &view = artifact.view();
  const dataflow::RootedGraphLaunchRef launch = onlyLaunch(view);
  const dataflow::CanonicalActorView *load = nullptr;
  const dataflow::CanonicalActorView *store = nullptr;
  for (const dataflow::CanonicalActorView &actor : view.actors()) {
    const auto schema = dataflow::operationSchemaOf(actor.op);
    if (schema == dataflow::OperationSchemaId::DataflowLoad)
      load = &actor;
    if (schema == dataflow::OperationSchemaId::DataflowStore)
      store = &actor;
  }
  require(load && store, "internal-memory fixture lacks load or store");
  auto graphView = take(view.resolve(load->graph));
  auto graph = mlir::cast<dataflow::GraphOp>(graphView.op);
  GraphPreparationResult preparedResult =
      take(prepareGraphExecution(artifact.module(), graph));
  auto *prepared = std::get_if<PreparedGraphExecution>(&preparedResult);
  require(prepared, "internal-memory graph preparation failed");

  CgraFrozenExecutionPlan plan;
  std::vector<CgraResourcePatternSelection> selections;
  const fabric::ResourceContract contract = resourceContract();
  for (std::uint64_t action = 0; action != 4; ++action) {
    plan.physicalUseClients.push_back(
        CgraPhysicalUseClientKind::MemoryTransition);
    plan.physicalUseTimings.push_back(
        {action, 0, std::nullopt, 1, 0, 1, std::nullopt});
    selections.push_back({action, fabric::UsePatternKey(0)});
  }

  const loom::fabric::FabricMemoryOccurrenceRef occurrence{};
  const loom::fabric::LocalMemoryServiceRef service(
      loom::fabric::FabricMemoryServiceRef::local(occurrence));
  plan.memory.rootedUses.push_back(
      {launch, 0, CgraMemoryServiceTarget(service), 1});
  plan.memory.rootedUses.push_back(
      {launch, 1, CgraMemoryServiceTarget(service), 3});
  plan.memory.childTransactions.push_back(
      {fabric::MemoryChildActivationKind::Always, std::nullopt,
       fabric::MemoryChildProjectionKind::ParentRequest, std::nullopt});
  plan.memory.childTransactions.push_back(
      {fabric::MemoryChildActivationKind::Always, std::nullopt,
       fabric::MemoryChildProjectionKind::ParentRequest, std::nullopt});
  plan.memory.resultAssemblies.push_back(
      {dataflow::semantics::ServiceValueRole::Data,
       fabric::MemoryResultAssemblyStrategy::PassThroughParent, std::nullopt,
       std::nullopt});
  const loom::fabric::FabricMemoryOperationPortRef loadPort{occurrence, 0};
  const loom::fabric::FabricMemoryOperationPortRef storePort{occurrence, 1};
  plan.memory.actors.push_back(
      {load->ref,
       load->graph,
       occurrence,
       loom::mapping::SpatialMemoryOperationPlacementView(loadPort),
       loom::fabric::FabricMemoryCapabilityAlternativeRef{loadPort, 0},
       ::fabric::serializedMemoryOperationIssueDepth,
       0,
       0,
       1,
       0,
       1,
       0,
       1,
       {},
       {}});
  plan.memory.actors.push_back(
      {store->ref,
       store->graph,
       occurrence,
       loom::mapping::SpatialMemoryOperationPlacementView(storePort),
       loom::fabric::FabricMemoryCapabilityAlternativeRef{storePort, 0},
       ::fabric::serializedMemoryOperationIssueDepth,
       2,
       1,
       1,
       1,
       1,
       1,
       0,
       {},
       {}});
  selectExternalMemoryRoles(plan.memory.actors[0], load->op);
  selectExternalMemoryRoles(plan.memory.actors[1], store->op);

  auto loadService =
      take(dataflow::semantics::CanonicalService::forActor(load->op));
  auto storeService =
      take(dataflow::semantics::CanonicalService::forActor(store->op));
  loom::fabric::FabricOrdinal nextConnection = 0;
  for (auto [resultOrdinal, result] : llvm::enumerate(loadService.results())) {
    auto resultValue = loadService.resultValue(load->op, resultOrdinal);
    if (!resultValue)
      fail(llvm::toString(resultValue.takeError()));
    const mlir::OpOperand *storeUse = nullptr;
    for (const mlir::OpOperand &use : resultValue->getUses())
      if (use.getOwner() == store->op)
        storeUse = &use;
    require(storeUse, "load result has no selected store consumer");
    std::optional<dataflow::semantics::ServiceValueRole> storeRole;
    for (auto [argumentOrdinal, argument] :
         llvm::enumerate(storeService.arguments())) {
      auto value = storeService.argumentValue(store->op, argumentOrdinal);
      if (!value)
        fail(llvm::toString(value.takeError()));
      if (*value == storeUse)
        storeRole = argument.role;
    }
    require(storeRole.has_value(),
            "store consumer has no canonical service role");
    const std::size_t resultRole = static_cast<std::size_t>(result.role);
    const std::size_t argumentRole = static_cast<std::size_t>(*storeRole);
    auto &destination = *plan.memory.actors[0].roleDestinations[resultRole];
    destination.externalEndpoint.reset();
    destination.internalConnections = {nextConnection};
    plan.memory.actors[1].roleSources[argumentRole] =
        loom::fabric::FabricMemoryHandshakeInternalRoleSource{nextConnection};
    plan.memory.internalConnections.push_back(
        {occurrence,
         nextConnection,
         {load->ref, static_cast<dataflow::StructuralOrdinal>(
                         resultValue->getResultNumber())},
         {store->ref, static_cast<dataflow::StructuralOrdinal>(
                          storeUse->getOperandNumber())}});
    ++nextConnection;
  }
  require(plan.memory.internalConnections.size() == 2,
          "memory chain did not select both internal connections");

  const auto appendExternalInputs =
      [&](const dataflow::CanonicalActorView &actor,
          const CgraMemoryActorPlan &memoryActor, std::uint32_t addressInput,
          bool hasControl) {
        auto actorService =
            take(dataflow::semantics::CanonicalService::forActor(actor.op));
        for (auto [argumentOrdinal, argument] :
             llvm::enumerate(actorService.arguments())) {
          const std::size_t role = static_cast<std::size_t>(argument.role);
          if (!memoryActor.roleSources[role] ||
              !std::holds_alternative<
                  loom::fabric::FabricMemoryHandshakeExternalRoleSource>(
                  *memoryActor.roleSources[role]))
            continue;
          auto value = actorService.argumentValue(actor.op, argumentOrdinal);
          if (!value)
            fail(llvm::toString(value.takeError()));
          dataflow::CanonicalGraphProducerEndpointRef producer(
              dataflow::GraphIngressTokenRef{dataflow::GraphValueInputTokenRef{
                  actor.graph, addressInput}});
          if (argument.role == dataflow::semantics::ServiceValueRole::Control &&
              hasControl) {
            producer = dataflow::GraphIngressTokenRef{
                dataflow::GraphStartTokenRef{actor.graph}};
          }
          appendTransfer(
              plan, std::move(producer),
              dataflow::ActorTokenOperandRef{
                  actor.ref, static_cast<dataflow::StructuralOrdinal>(
                                 (*value)->getOperandNumber())},
              actor.graph);
        }
      };
  appendExternalInputs(*load, plan.memory.actors[0], 0, true);
  appendExternalInputs(*store, plan.memory.actors[1], 1, false);
  appendTransfer(
      plan, dataflow::ActorTokenResultRef{store->ref, 0},
      dataflow::GraphEgressTokenRef{
          dataflow::GraphCompletionFrontierTokenRef{store->graph, 0}},
      store->graph);

  const std::array<const fabric::ResourceContract *, 4> contracts = {
      &contract, &contract, &contract, &contract};
  plan.resources = take(freezeCgraResourceRuntimePlan(contracts, selections));

  SimulatorState state;
  state.graphScope = graph.getOperation();
  initializeRunState(state, *prepared);
  const auto makeMemory = [&](std::initializer_list<std::uint32_t> values) {
    auto memory = std::make_shared<MemoryValue>();
    for (std::uint32_t value : values) {
      auto bytes = take(encodeMemoryElement(
          take(tokenFromBitPattern(llvm::APInt(32, value),
                                   mlir::IntegerType::get(&context(), 32))),
          mlir::IntegerType::get(&context(), 32), graph.getOperation()));
      memory->bytes.append(bytes.begin(), bytes.end());
    }
    memory->initialized = llvm::SmallBitVector(memory->bytes.size(), true);
    return memory;
  };
  auto loadMemory = makeMemory({3, 11, 5, 7});
  auto storeMemory = makeMemory({0, 0, 0, 0});
  loadMemory->logicalRootId = 0;
  storeMemory->logicalRootId = 1;
  mlir::Block &entry = graph.getBody().front();
  llvm::SmallVector<GraphIngressEmission, 3> ingress;
  state.graphIngressCapture = &ingress;
  seedBlockArgument(state, entry.getArgument(0), noneToken());
  seedBlockArgument(state, entry.getArgument(1),
                    indexToken(llvm::APInt(64, 1)));
  seedBlockArgument(state, entry.getArgument(2),
                    indexToken(llvm::APInt(64, 2)));
  state.graphIngressCapture = nullptr;
  state.memories[entry.getArgument(3)] = loadMemory;
  state.memories[entry.getArgument(4)] = storeMemory;
  const MemoryView loadView{loadMemory, entry.getArgument(3), 0,
                            mlir::IntegerType::get(&context(), 32)};
  const MemoryView storeView{storeMemory, entry.getArgument(4), 0,
                             mlir::IntegerType::get(&context(), 32)};
  state.memoryViews[entry.getArgument(3)] = loadView;
  state.memoryViews[entry.getArgument(4)] = storeView;

  CgraFrozenExecutionPlan tampered = plan;
  bool changedConnection = false;
  for (auto &source : tampered.memory.actors[1].roleSources) {
    auto *internal =
        source ? std::get_if<
                     loom::fabric::FabricMemoryHandshakeInternalRoleSource>(
                     &*source)
               : nullptr;
    if (!internal)
      continue;
    ++internal->connection;
    changedConnection = true;
    break;
  }
  require(changedConnection, "memory role tamper found no internal source");
  auto rejectedTransportGraph =
      take(freezeCgraTransportGraph(tampered, view, load->graph, *prepared));
  auto rejected = CgraGraphActivationRuntime::create(
      tampered, view, launch, load->graph, *prepared, rejectedTransportGraph,
      state, /*captureMicroarchitecture=*/false);
  require(!rejected,
          "tampered memory internal activation was accepted by runtime");
  llvm::consumeError(rejected.takeError());

  auto runtimeTransportGraph =
      take(freezeCgraTransportGraph(plan, view, load->graph, *prepared));
  auto runtime = take(CgraGraphActivationRuntime::create(
      plan, view, launch, load->graph, *prepared, runtimeTransportGraph, state,
      /*captureMicroarchitecture=*/false));
  if (llvm::Error error = runtime.start(coordinate(100), ingress))
    fail(llvm::toString(std::move(error)));
  bool sawAtomicLoadPacket = false;
  for (unsigned iteration = 0; iteration != 128 && runtime.hasPendingEvents();
       ++iteration) {
    auto frame = take(runtime.advance());
    require(frame.has_value(),
            "internal-memory activation lost its pending frame");
    std::uint32_t loadPublications = 0;
    for (const CgraTokenPublication &publication : frame->publications) {
      const auto *result =
          std::get_if<dataflow::ActorTokenResultRef>(&publication.producer);
      if (result && result->actor == load->ref)
        ++loadPublications;
    }
    require(loadPublications == 0 || loadPublications == 2,
            "memory completion packet published only one role");
    sawAtomicLoadPacket |= loadPublications == 2;
  }
  require(!runtime.hasPendingEvents(),
          "internal-memory activation did not quiesce");
  require(sawAtomicLoadPacket,
          "memory completion packet did not publish both internal roles");
  auto stored = readMemoryElement(
      storeView, 8, mlir::IntegerType::get(&context(), 32), state,
      graph.getOperation(), "internal-memory verification");
  require(stored && take(tokenBitPattern(
                        *stored, mlir::IntegerType::get(&context(), 32))) ==
                        llvm::APInt(32, 11),
          "exact memory internal connections did not feed the store");
}

void graphActivationRejectsUnsupportedMemoryContracts() {
  auto artifact = unsupportedMemoryProgram();
  const auto &view = artifact.view();
  using ContractClass = dataflow::MemoryContractClass;
  // One graph per non-plain contract class; index 0 (Plain) stays unused.
  std::array<bool, 6> seen{};
  seen[static_cast<std::size_t>(ContractClass::Plain)] = true;
  require(view.graphs().size() == seen.size() - 1,
          "unsupported CGRA memory fixture lost a graph");

  for (const dataflow::CanonicalGraphView &graphView : view.graphs()) {
    auto graph = mlir::cast<dataflow::GraphOp>(graphView.op);
    GraphPreparationResult preparedResult =
        take(prepareGraphExecution(artifact.module(), graph));
    auto *prepared = std::get_if<PreparedGraphExecution>(&preparedResult);
    require(prepared, "unsupported CGRA memory graph preparation failed");
    std::optional<ContractClass> kind;
    std::optional<dataflow::ActorRef> actorRef;
    llvm::StringRef actorName;
    for (const ActorExecutionPlan &actor : prepared->actorPlans) {
      const auto *memory = std::get_if<dataflow::MemoryContractPayload>(
          &actor.projection.payload);
      if (!memory)
        continue;
      const ContractClass contractClass =
          dataflow::classifyMemoryContract(*memory);
      if (contractClass == ContractClass::Plain)
        continue;
      kind = contractClass;
      actorName = actor.operation->getName().getStringRef();
      auto canonical = llvm::find_if(view.actors(), [&](const auto &candidate) {
        return candidate.op == actor.operation;
      });
      require(canonical != view.actors().end(),
              "unsupported memory actor has no canonical reference");
      actorRef = canonical->ref;
    }
    require(kind.has_value() && actorRef.has_value(),
            "unsupported CGRA memory graph lacks its contract actor");
    const std::size_t kindIndex = static_cast<std::size_t>(*kind);
    require(!seen[kindIndex],
            "unsupported CGRA memory fixture repeated a contract kind");
    seen[kindIndex] = true;

    SimulatorState state;
    state.graphScope = graph.getOperation();
    initializeRunState(state, *prepared);
    CgraFrozenExecutionPlan plan;
    const dataflow::RootedGraphLaunchRef unusedLaunch{
        {view.identity(), dataflow::RootThreadLaunchId(0)},
        {view.identity(), dataflow::StaticGraphLaunchId(0)}};
    auto rejectedTransportGraph =
        take(freezeCgraTransportGraph(plan, view, graphView.ref, *prepared));
    auto rejected = CgraGraphActivationRuntime::create(
        plan, view, unusedLaunch, graphView.ref, *prepared,
        rejectedTransportGraph, state, /*captureMicroarchitecture=*/false);
    require(!rejected,
            "CGRA activation accepted an unsupported memory contract");
    llvm::Error rejection = rejected.takeError();
    require(rejection.isA<loom::sim::CgraExecutionUnsupported>(),
            "CGRA memory contract rejection lost its typed payload");
    std::error_code code;
    std::string diagnostic;
    std::optional<loom::sim::CgraUnsupportedMemoryContract> contract;
    llvm::handleAllErrors(
        std::move(rejection),
        [&](const loom::sim::CgraExecutionUnsupported &failure) {
          code = failure.convertToErrorCode();
          llvm::raw_string_ostream stream(diagnostic);
          failure.log(stream);
          contract = failure.memoryContract();
        });
    require(code == std::make_error_code(std::errc::not_supported),
            "CGRA memory contract rejection was not typed Unsupported");
    require(llvm::StringRef(diagnostic).contains(actorName),
            "CGRA memory contract rejection named the wrong actor");
    require(contract && contract->contractClass == *kind &&
                contract->actor == *actorRef,
            "CGRA memory contract rejection lost its class or actor identity");
  }
  require(llvm::all_of(seen, [](bool value) { return value; }),
          "CGRA activation rejection did not cover every memory contract");
}

} // namespace

int main() {
  graphActivationCoordinatesComputeAndTransport();
  graphActivationExecutesSelectedLocalMemory();
  graphActivationExecutesExactMemoryInternalConnections();
  graphActivationRejectsUnsupportedMemoryContracts();
  return EXIT_SUCCESS;
}
