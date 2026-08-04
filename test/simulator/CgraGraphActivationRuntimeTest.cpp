#include "CgraGraphActivationRuntime.h"

#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Dataflow/IR/DataflowDialect.h"
#include "Evaluation/NumericValue.h"
#include "Fabric/IR/ResourceContract.h"

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

#include <cstdlib>
#include <system_error>
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

void graphActivationCoordinatesComputeAndTransport() {
  auto artifact = program();
  auto view = take(artifact.view());
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
         static_cast<std::uint32_t>(semantic.handshakeCases.size())});
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
  state.graphIngressCapture = nullptr;

  auto runtime = take(CgraGraphActivationRuntime::create(
      plan, view, launch, add->graph, *prepared, state,
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
  require(committed == 2 && retired == 2,
          "graph activation did not commit and retire both actors once");
  require(publications == 6,
          "graph activation did not publish every ingress and actor result");
  require(physicalEvents >= 12,
          "graph activation bypassed selected physical lifecycles");
  auto output =
      state.observedOutputs.find(graph.getBody().front().back().getOperand(0));
  require(output != state.observedOutputs.end() && output->second.size() == 1 &&
              take(tokenBitPattern(output->second.front(),
                                   mlir::IntegerType::get(&context(), 32))) ==
                  llvm::APInt(32, 16),
          "graph activation produced the wrong functional value");
}

void graphActivationExecutesSelectedLocalMemory() {
  auto artifact = memoryProgram();
  auto view = take(artifact.view());
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
         static_cast<std::uint32_t>(semantic.handshakeCases.size())});
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
      {load->ref, load->graph, occurrence,
       loom::mapping::SpatialMemoryOperationPlacementView(port),
       loom::fabric::FabricMemoryCapabilityAlternativeRef{port, 0},
       operationAction, 0, 1, 0, 1, 0, 1});

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

  SimulatorState state;
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

  llvm::SmallVector<GraphIngressEmission, 2> ingress;
  state.graphIngressCapture = &ingress;
  mlir::Block &entry = graph.getBody().front();
  seedBlockArgument(state, entry.getArgument(0), noneToken());
  seedBlockArgument(state, entry.getArgument(1),
                    indexToken(llvm::APInt(64, 1)));
  state.graphIngressCapture = nullptr;
  state.memories[entry.getArgument(2)] = memory;
  state.memoryViews[entry.getArgument(2)] = MemoryView{
      memory, entry.getArgument(2), 0, mlir::IntegerType::get(&context(), 32)};

  plan.memory.rootedUses.front().target =
      loom::fabric::ManagerEndpointRef(loom::fabric::FabricMemoryEndpointRef{
          loom::fabric::FabricMemoryEndpointOwnerRef::of(occurrence), 0});
  auto unsupportedRuntime = CgraGraphActivationRuntime::create(
      plan, view, launch, load->graph, *prepared, state,
      /*captureMicroarchitecture=*/false);
  require(!unsupportedRuntime,
          "manager memory target unexpectedly acquired a CGRA provider");
  require(llvm::errorToErrorCode(unsupportedRuntime.takeError()) ==
              std::make_error_code(std::errc::not_supported),
          "manager memory target did not fail as typed unsupported");
  plan.memory.rootedUses.front().target = service;

  auto runtime = take(CgraGraphActivationRuntime::create(
      plan, view, launch, load->graph, *prepared, state,
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

} // namespace

int main() {
  graphActivationCoordinatesComputeAndTransport();
  graphActivationExecutesSelectedLocalMemory();
  return EXIT_SUCCESS;
}
