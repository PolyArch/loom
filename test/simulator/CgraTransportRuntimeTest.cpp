#include "CgraTransportRuntime.h"

#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Dataflow/IR/DataflowDialect.h"
#include "Evaluation/NumericValue.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <array>
#include <cstdlib>
#include <limits>
#include <utility>

namespace {

using namespace loom::sim::detail;

[[noreturn]] void fail(llvm::StringRef message) {
  llvm::errs() << "CGRA transport runtime test: " << message << '\n';
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
    registry.insert<dataflow::DataflowDialect, mlir::arith::ArithDialect>();
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
}
)mlir",
                                                        &context());
  if (!module)
    fail("failed to parse local-transfer fixture");
  return take(dataflow::finalizeCanonicalDataflow(module.get()));
}

dataflow::CanonicalDataflowArtifact fanoutProgram() {
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  dataflow.graph private @fanout(
      %start: none, %lhs: i32, %rhs: i32) -> (i32)
      attributes {
        input_segments = array<i32: 2, 0, 0>,
        result_segments = array<i32: 1, 0, 0>
      } {
    %left = arith.addi %lhs, %rhs : i32
    %right = arith.addi %lhs, %rhs : i32
    %sum = arith.addi %left, %right : i32
    %published:2 = dataflow.sync %start, %sum
        : (none, i32) -> (none, i32)
    dataflow.graph.return values(%published#1 : i32) streams() memories()
        complete(%published#0 : none)
  }
}
)mlir",
                                                        &context());
  if (!module)
    fail("failed to parse operand-queue fanout fixture");
  return take(dataflow::finalizeCanonicalDataflow(module.get()));
}

loom::sim::SpatialEventCoordinate coordinate(std::uint64_t cycle,
                                             std::uint64_t delta = 0) {
  return {take(loom::evaluation::ExactRatio::get(cycle, 1)), delta};
}

void localRealizationEdgePublishesThroughExactConsumer() {
  auto artifact = program();
  auto view = take(artifact.view());
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
  require(prepared, "local-transfer graph preparation failed");

  CgraFrozenExecutionPlan plan;
  plan.computeActors.push_back(
      {add->ref, add->graph, {}, {}, 0, 0, std::nullopt, 0});
  plan.transport.localTransfers.push_back(
      {{dataflow::GraphIngressTokenRef{
           dataflow::GraphValueInputTokenRef{add->graph, 0}}},
       add->graph,
       0,
       1});
  plan.transport.localTransferSinks.push_back(
      {{dataflow::ActorTokenOperandRef{add->ref, 0}}});
  plan.transport.localTransfers.push_back(
      {{dataflow::ActorTokenResultRef{add->ref, 0}}, add->graph, 1, 1});
  plan.transport.localTransferSinks.push_back(
      {{dataflow::ActorTokenOperandRef{sync->ref, 1}}});

  SimulatorState state;
  state.graphScope = graph.getOperation();
  initializeRunState(state, *prepared);
  auto physical = take(CgraPhysicalActionRuntime::create(
      plan.resources, plan.physicalUseTimings));
  auto runtime = take(CgraTransportRuntime::create(plan, view, add->graph,
                                                   *prepared, state, physical));
  llvm::SmallVector<GraphIngressEmission, 2> ingress;
  state.graphIngressCapture = &ingress;
  seedBlockArgument(
      state, graph.getBody().front().getArgument(1),
      take(tokenFromBitPattern(llvm::APInt(32, 7),
                               mlir::IntegerType::get(&context(), 32))));
  state.graphIngressCapture = nullptr;
  require(channelQueue(state, add->op->getOpOperand(0)).empty(),
          "CGRA input seeding bypassed selected transport");
  if (llvm::Error error =
          runtime.acceptGraphIngressEmissions(coordinate(1), ingress))
    fail(llvm::toString(std::move(error)));
  auto ingressFrame = take(runtime.advance());
  require(ingressFrame &&
              loom::sim::compareSpatialEventCoordinates(
                  ingressFrame->coordinate, coordinate(1, 1)) == 0 &&
              ingressFrame->publications.size() == 1 &&
              channelQueue(state, add->op->getOpOperand(0)).size() == 1,
          "graph ingress did not traverse its selected local transfer");

  llvm::SmallVector<CgraActorEmission, 1> emissions;
  emissions.push_back(
      {0, 0, 0, 0,
       take(tokenFromBitPattern(llvm::APInt(32, 15),
                                mlir::IntegerType::get(&context(), 32)))});
  emissions.push_back(
      {0, 0, 0, 1,
       take(tokenFromBitPattern(llvm::APInt(32, 99),
                                mlir::IntegerType::get(&context(), 32)))});
  llvm::Error rejected = runtime.acceptActorEmissions(coordinate(2), emissions);
  require(static_cast<bool>(rejected),
          "partially bound actor emissions were accepted");
  llvm::consumeError(std::move(rejected));
  require(!runtime.hasPendingEvents(),
          "rejected actor emission batch changed transport state");

  emissions.clear();
  emissions.push_back(
      {0, 0, 0, 0,
       take(tokenFromBitPattern(llvm::APInt(32, 16),
                                mlir::IntegerType::get(&context(), 32)))});
  if (llvm::Error error =
          runtime.acceptActorEmissions(coordinate(3), emissions))
    fail(llvm::toString(std::move(error)));
  auto frame = take(runtime.advance());
  require(
      frame &&
          loom::sim::compareSpatialEventCoordinates(frame->coordinate,
                                                    coordinate(3, 1)) == 0 &&
          frame->publications.size() == 1 &&
          frame->publications.front().producer ==
              dataflow::CanonicalGraphProducerEndpointRef(
                  dataflow::ActorTokenResultRef{add->ref, 0}) &&
          channelQueue(state, sync->op->getOpOperand(1)).size() == 1 &&
          take(tokenBitPattern(
              channelQueue(state, sync->op->getOpOperand(1)).front(),
              mlir::IntegerType::get(&context(), 32))) == llvm::APInt(32, 16),
      "FU-local transfer did not publish one exact consumer token");
  channelQueue(state, sync->op->getOpOperand(1)).pop_front();

  CgraFrozenExecutionPlan physicalPlan = plan;
  physicalPlan.transport.localTransfers.pop_back();
  physicalPlan.transport.localTransferSinks.pop_back();
  physicalPlan.transport.traversals.resize(2);
  physicalPlan.transport.traversals.front().kind =
      loom::fabric::FabricPhysicalTraversalKind::BoundaryTraversal;
  physicalPlan.transport.traversals.front().impliedUseOffset = 0;
  physicalPlan.transport.traversals.front().impliedUseCount = 1;
  physicalPlan.transport.traversals[1].kind =
      loom::fabric::FabricPhysicalTraversalKind::SwitchTraversal;
  physicalPlan.transport.traversals[1].impliedUseOffset = 1;
  physicalPlan.transport.traversals[1].impliedUseCount = 1;
  physicalPlan.transport.traversalUses.push_back({{}, {}, 0, 2});
  physicalPlan.transport.traversalUses.push_back({{}, {}, 1, 3});
  physicalPlan.transport.routeNodes.push_back(
      {std::numeric_limits<std::uint32_t>::max(), invalidCgraTransportOrdinal});
  physicalPlan.transport.routeNodes.push_back({0, 1});
  physicalPlan.transport.routeSinks.push_back(
      {{dataflow::ActorTokenOperandRef{sync->ref, 1}},
       1,
       invalidCgraTransportOrdinal});
  physicalPlan.transport.routes.push_back(
      {{dataflow::ActorTokenResultRef{add->ref, 0}},
       add->graph,
       0,
       0,
       2,
       0,
       1});
  physicalPlan.physicalUseClients.push_back(
      CgraPhysicalUseClientKind::ProducedTransport);
  physicalPlan.physicalUseClients.push_back(
      CgraPhysicalUseClientKind::ConsumedTransport);
  physicalPlan.physicalUseClients.push_back(
      CgraPhysicalUseClientKind::TraversalTransport);
  physicalPlan.physicalUseClients.push_back(
      CgraPhysicalUseClientKind::TraversalTransport);
  physicalPlan.resources.selectedUses.push_back({});
  physicalPlan.resources.selectedUses.push_back({});
  physicalPlan.resources.selectedUses.push_back({});
  physicalPlan.resources.selectedUses.push_back({});
  physicalPlan.physicalUseTimings.push_back(
      {0, 0, std::nullopt, 1, 0, 1, std::nullopt});
  physicalPlan.physicalUseTimings.push_back(
      {1, 0, std::nullopt, 1, 0, 1, std::nullopt});
  physicalPlan.physicalUseTimings.push_back(
      {2, 0, std::nullopt, 1, 0, 1, std::nullopt});
  physicalPlan.physicalUseTimings.push_back(
      {3, 0, std::nullopt, 1, 0, 1, std::nullopt});
  physicalPlan.transport.endpointPhysicalUses.push_back(0);
  physicalPlan.transport.endpointPhysicalUses.push_back(1);
  physicalPlan.transport.producedUses.push_back(
      {{dataflow::ActorTokenResultRef{add->ref, 0}}, 0, 1});
  physicalPlan.transport.consumedUses.push_back(
      {{dataflow::ActorTokenOperandRef{sync->ref, 1}}, 1, 1});
  auto selectedPhysical = take(CgraPhysicalActionRuntime::create(
      physicalPlan.resources, physicalPlan.physicalUseTimings));
  auto selectedTransport = take(CgraTransportRuntime::create(
      physicalPlan, view, add->graph, *prepared, state, selectedPhysical));
  emissions.clear();
  emissions.push_back(
      {0, 1, 0, 0,
       take(tokenFromBitPattern(llvm::APInt(32, 17),
                                mlir::IntegerType::get(&context(), 32)))});
  if (llvm::Error error =
          selectedTransport.acceptActorEmissions(coordinate(4), emissions))
    fail(llvm::toString(std::move(error)));
  auto requested = take(selectedTransport.advance());
  require(requested && requested->physicalEvents.size() == 1 &&
              requested->physicalEvents.front().kind ==
                  CgraPhysicalLifecycleKind::Requested &&
              requested->publications.empty(),
          "selected Produced use did not block publication at request");
  auto granted = take(selectedPhysical.advance());
  require(granted && granted->events.size() == 1 &&
              granted->events.front().kind ==
                  CgraPhysicalLifecycleKind::Granted,
          "selected Produced use did not grant");
  (void)take(selectedTransport.acceptPhysicalEvents(*granted));
  auto traversalRequest = take(selectedTransport.advance());
  require(traversalRequest && traversalRequest->physicalEvents.size() == 1 &&
              traversalRequest->physicalEvents.front().kind ==
                  CgraPhysicalLifecycleKind::Requested &&
              traversalRequest->physicalEvents.front().actionOrdinal == 2 &&
              traversalRequest->publications.empty() &&
              loom::sim::compareSpatialEventCoordinates(
                  traversalRequest->coordinate, coordinate(4, 1)) == 0,
          "selected traversal use did not gate route arrival");
  auto traversalGrant = take(selectedPhysical.advance());
  require(traversalGrant && traversalGrant->events.size() == 1 &&
              traversalGrant->events.front().kind ==
                  CgraPhysicalLifecycleKind::Granted &&
              traversalGrant->events.front().actionOrdinal == 2,
          "selected traversal use did not grant");
  (void)take(selectedTransport.acceptPhysicalEvents(*traversalGrant));
  auto childTraversalRequest = take(selectedTransport.advance());
  require(childTraversalRequest &&
              childTraversalRequest->physicalEvents.size() == 1 &&
              childTraversalRequest->physicalEvents.front().kind ==
                  CgraPhysicalLifecycleKind::Requested &&
              childTraversalRequest->physicalEvents.front().actionOrdinal ==
                  3 &&
              childTraversalRequest->publications.empty() &&
              loom::sim::compareSpatialEventCoordinates(
                  childTraversalRequest->coordinate, coordinate(4, 2)) == 0,
          "child traversal did not wait for its selected parent traversal");
  auto childTraversalGrant = take(selectedPhysical.advance());
  require(childTraversalGrant && childTraversalGrant->events.size() == 1 &&
              childTraversalGrant->events.front().kind ==
                  CgraPhysicalLifecycleKind::Granted &&
              childTraversalGrant->events.front().actionOrdinal == 3,
          "child traversal use did not grant");
  (void)take(selectedTransport.acceptPhysicalEvents(*childTraversalGrant));
  auto consumedRequest = take(selectedTransport.advance());
  require(consumedRequest && consumedRequest->physicalEvents.size() == 1 &&
              consumedRequest->physicalEvents.front().kind ==
                  CgraPhysicalLifecycleKind::Requested &&
              consumedRequest->physicalEvents.front().actionOrdinal == 1 &&
              consumedRequest->publications.empty() &&
              loom::sim::compareSpatialEventCoordinates(
                  consumedRequest->coordinate, coordinate(4, 3)) == 0,
          "selected Consumed use did not gate sink publication");
  auto consumedGrant = take(selectedPhysical.advance());
  require(consumedGrant && consumedGrant->events.size() == 1 &&
              consumedGrant->events.front().kind ==
                  CgraPhysicalLifecycleKind::Granted &&
              consumedGrant->events.front().actionOrdinal == 1,
          "selected Consumed use did not grant");
  (void)take(selectedTransport.acceptPhysicalEvents(*consumedGrant));
  auto selectedPublication = take(selectedTransport.advance());
  require(selectedPublication &&
              selectedPublication->publications.size() == 1 &&
              loom::sim::compareSpatialEventCoordinates(
                  selectedPublication->coordinate, coordinate(4, 4)) == 0,
          "selected endpoint uses did not gate token publication");

  CgraFrozenExecutionPlan bufferedPlan = plan;
  bufferedPlan.transport.localTransfers.pop_back();
  bufferedPlan.transport.localTransferSinks.pop_back();
  bufferedPlan.transport.traversals.resize(1);
  bufferedPlan.transport.traversals.front().kind =
      loom::fabric::FabricPhysicalTraversalKind::FifoTraversal;
  bufferedPlan.transport.traversals.front().storageKind =
      CgraTraversalStorageKind::BufferedFifo;
  bufferedPlan.transport.traversals.front().storageOrdinal = 0;
  bufferedPlan.transport.traversalStorages.push_back({});
  auto &storage = bufferedPlan.transport.traversalStorages.front();
  storage.kind = CgraTraversalStorageKind::BufferedFifo;
  storage.capacity = 1;
  storage.enqueuePhysicalUseOrdinal = 0;
  storage.dequeuePhysicalUseOrdinal = 1;
  storage.simultaneousPhysicalUseOrdinal = 2;
  bufferedPlan.transport.routeNodes.push_back(
      {std::numeric_limits<std::uint32_t>::max(), invalidCgraTransportOrdinal});
  bufferedPlan.transport.routeSinks.push_back(
      {{dataflow::ActorTokenOperandRef{sync->ref, 1}},
       0,
       invalidCgraTransportOrdinal});
  bufferedPlan.transport.routes.push_back(
      {{dataflow::ActorTokenResultRef{add->ref, 0}},
       add->graph,
       0,
       0,
       1,
       0,
       1});
  for (std::uint64_t action = 0; action != 3; ++action) {
    bufferedPlan.physicalUseClients.push_back(
        CgraPhysicalUseClientKind::TraversalTransport);
    bufferedPlan.resources.selectedUses.push_back({});
    bufferedPlan.physicalUseTimings.push_back({action, 0, 1, 2, 0, 2, 1});
  }
  auto bufferedPhysical = take(CgraPhysicalActionRuntime::create(
      bufferedPlan.resources, bufferedPlan.physicalUseTimings));
  auto bufferedTransport = take(CgraTransportRuntime::create(
      bufferedPlan, view, add->graph, *prepared, state, bufferedPhysical));

  channelQueue(state, sync->op->getOpOperand(1)).clear();
  channelQueue(state, sync->op->getOpOperand(1))
      .push_back(take(tokenFromBitPattern(
          llvm::APInt(32, 1), mlir::IntegerType::get(&context(), 32))));
  emissions.clear();
  emissions.push_back(
      {0, 2, 0, 0,
       take(tokenFromBitPattern(llvm::APInt(32, 23),
                                mlir::IntegerType::get(&context(), 32)))});
  if (llvm::Error error =
          bufferedTransport.acceptActorEmissions(coordinate(10), emissions))
    fail(llvm::toString(std::move(error)));

  bool sawEnqueue = false;
  bool sawBlocked = false;
  bool sawDequeue = false;
  bool sawPublication = false;
  for (unsigned iteration = 0; iteration != 32 && !sawPublication;
       ++iteration) {
    const auto transportCoordinate = bufferedTransport.nextCoordinate();
    const auto physicalCoordinate = bufferedPhysical.nextCoordinate();
    const bool advancePhysical =
        physicalCoordinate &&
        (!transportCoordinate ||
         loom::sim::compareSpatialEventCoordinates(*physicalCoordinate,
                                                   *transportCoordinate) <= 0);
    if (advancePhysical) {
      auto frame = take(bufferedPhysical.advance());
      require(frame.has_value(), "buffered physical event disappeared");
      auto completions = take(bufferedTransport.acceptPhysicalEvents(*frame));
      (void)completions;
      continue;
    }
    require(transportCoordinate.has_value(),
            "buffered transfer became quiescent before publication");
    auto frame = take(bufferedTransport.advance());
    require(frame.has_value(), "buffered transport event disappeared");
    for (const CgraPhysicalLifecycleEvent &event : frame->physicalEvents) {
      sawEnqueue |= event.kind == CgraPhysicalLifecycleKind::Requested &&
                    event.actionOrdinal == 0;
      sawDequeue |= event.kind == CgraPhysicalLifecycleKind::Requested &&
                    event.actionOrdinal == 1;
    }
    if (!frame->blockedTransfers.empty() && !sawBlocked) {
      require(sawEnqueue && !sawDequeue && frame->publications.empty(),
              "buffered FIFO bypassed its occupied downstream");
      sawBlocked = true;
      channelQueue(state, sync->op->getOpOperand(1)).pop_front();
      if (llvm::Error error = bufferedTransport.retryBlocked(frame->coordinate))
        fail(llvm::toString(std::move(error)));
    }
    if (!frame->publications.empty()) {
      require(sawDequeue && frame->publications.size() == 1,
              "buffered token published without selected dequeue");
      sawPublication = true;
    }
  }
  require(sawEnqueue && sawBlocked && sawDequeue && sawPublication &&
              channelQueue(state, sync->op->getOpOperand(1)).size() == 1 &&
              take(tokenBitPattern(
                  channelQueue(state, sync->op->getOpOperand(1)).front(),
                  mlir::IntegerType::get(&context(), 32))) ==
                  llvm::APInt(32, 23),
          "buffered traversal did not preserve delayed token delivery");
}

void registerFifoWriteAndReadShareOneDurableQueue() {
  auto artifact = program();
  auto view = take(artifact.view());
  const dataflow::CanonicalActorView *add = nullptr;
  const dataflow::CanonicalActorView *sync = nullptr;
  for (const dataflow::CanonicalActorView &actor : view.actors()) {
    const auto schema = dataflow::operationSchemaOf(actor.op);
    if (schema == dataflow::OperationSchemaId::ArithAddI)
      add = &actor;
    if (schema == dataflow::OperationSchemaId::DataflowSync)
      sync = &actor;
  }
  require(add && sync, "register-FIFO fixture lacks add or sync actor");
  auto graphView = take(view.resolve(add->graph));
  auto graph = mlir::cast<dataflow::GraphOp>(graphView.op);
  GraphPreparationResult preparedResult =
      take(prepareGraphExecution(artifact.module(), graph));
  auto *prepared = std::get_if<PreparedGraphExecution>(&preparedResult);
  require(prepared, "register-FIFO graph preparation failed");

  CgraFrozenExecutionPlan plan;
  plan.computeActors.push_back(
      {add->ref, add->graph, {}, {}, 0, 0, std::nullopt, 0});
  plan.transport.traversals.resize(2);
  plan.transport.traversals[0].kind =
      loom::fabric::FabricPhysicalTraversalKind::PeRegisterFifoTraversal;
  plan.transport.traversals[0].storageKind =
      CgraTraversalStorageKind::RegisterFifoWrite;
  plan.transport.traversals[0].storageOrdinal = 0;
  plan.transport.traversals[1].kind =
      loom::fabric::FabricPhysicalTraversalKind::PeRegisterFifoTraversal;
  plan.transport.traversals[1].storageKind =
      CgraTraversalStorageKind::RegisterFifoRead;
  plan.transport.traversals[1].storageOrdinal = 0;
  plan.transport.physicalTags.push_back({llvm::APInt(2, 1)});
  plan.transport.traversalStorages.push_back({});
  CgraTraversalStoragePlan &storage = plan.transport.traversalStorages.front();
  storage.kind = CgraTraversalStorageKind::RegisterFifoWrite;
  storage.capacity = 1;
  storage.enqueuePhysicalUseOrdinal = 0;
  storage.dequeuePhysicalUseOrdinal = 1;
  storage.independentReadWriteServices = true;
  plan.transport.routeNodes.push_back(
      {std::numeric_limits<std::uint32_t>::max(), invalidCgraTransportOrdinal,
       0});
  plan.transport.routeSinks.push_back(
      {{dataflow::ActorTokenOperandRef{sync->ref, 1}}, 0, 1});
  plan.transport.routes.push_back({{dataflow::ActorTokenResultRef{add->ref, 0}},
                                   add->graph,
                                   0,
                                   0,
                                   1,
                                   0,
                                   1});
  for (std::uint64_t action = 0; action != 2; ++action) {
    plan.physicalUseClients.push_back(
        CgraPhysicalUseClientKind::TraversalTransport);
    plan.resources.selectedUses.push_back({});
    plan.physicalUseTimings.push_back({action, 0, 0, 1, 0, 1, 0});
  }

  SimulatorState state;
  state.graphScope = graph.getOperation();
  initializeRunState(state, *prepared);
  auto physical = take(CgraPhysicalActionRuntime::create(
      plan.resources, plan.physicalUseTimings));
  auto transport = take(CgraTransportRuntime::create(
      plan, view, add->graph, *prepared, state, physical));
  llvm::SmallVector<CgraActorEmission, 1> emissions;
  emissions.push_back(
      {0, 0, 0, 0,
       take(tokenFromBitPattern(llvm::APInt(32, 31),
                                mlir::IntegerType::get(&context(), 32)))});
  if (llvm::Error error =
          transport.acceptActorEmissions(coordinate(20), emissions))
    fail(llvm::toString(std::move(error)));

  bool sawWrite = false;
  bool sawRead = false;
  bool sawPublication = false;
  std::optional<loom::evaluation::ExactRatio> writeCycle;
  std::optional<loom::evaluation::ExactRatio> readCycle;
  for (unsigned iteration = 0; iteration != 32 && !sawPublication;
       ++iteration) {
    const auto transportCoordinate = transport.nextCoordinate();
    const auto physicalCoordinate = physical.nextCoordinate();
    const bool advancePhysical =
        physicalCoordinate &&
        (!transportCoordinate ||
         loom::sim::compareSpatialEventCoordinates(*physicalCoordinate,
                                                   *transportCoordinate) <= 0);
    if (advancePhysical) {
      auto frame = take(physical.advance());
      require(frame.has_value(), "register-FIFO physical event disappeared");
      (void)take(transport.acceptPhysicalEvents(*frame));
      continue;
    }
    require(transportCoordinate.has_value(),
            "register-FIFO transfer became quiescent before publication");
    auto frame = take(transport.advance());
    require(frame.has_value(), "register-FIFO transport event disappeared");
    for (const CgraPhysicalLifecycleEvent &event : frame->physicalEvents) {
      if (event.kind != CgraPhysicalLifecycleKind::Requested)
        continue;
      if (event.actionOrdinal == 0) {
        sawWrite = true;
        writeCycle = event.coordinate.referenceCycle;
      }
      if (event.actionOrdinal == 1) {
        sawRead = true;
        readCycle = event.coordinate.referenceCycle;
      }
    }
    sawPublication |= !frame->publications.empty();
  }

  require(sawWrite && sawRead && sawPublication && writeCycle && readCycle &&
              static_cast<__uint128_t>(writeCycle->numerator()) *
                      readCycle->denominator() <
                  static_cast<__uint128_t>(readCycle->numerator()) *
                      writeCycle->denominator() &&
              channelQueue(state, sync->op->getOpOperand(1)).size() == 1 &&
              take(tokenBitPattern(
                  channelQueue(state, sync->op->getOpOperand(1)).front(),
                  mlir::IntegerType::get(&context(), 32))) ==
                  llvm::APInt(32, 31),
          "register-FIFO write/read did not preserve durable ordered storage");

  CgraFrozenExecutionPlan dualPlan = plan;
  const std::uint64_t ingressNodeOffset = dualPlan.transport.routeNodes.size();
  const std::uint64_t ingressSinkOffset = dualPlan.transport.routeSinks.size();
  dualPlan.transport.routeNodes.push_back(
      {std::numeric_limits<std::uint32_t>::max(), invalidCgraTransportOrdinal});
  dualPlan.transport.routeSinks.push_back(
      {{dataflow::ActorTokenOperandRef{add->ref, 0}}, 0, 1});
  dualPlan.transport.routes.push_back(
      {{dataflow::GraphIngressTokenRef{
           dataflow::GraphValueInputTokenRef{add->graph, 0}}},
       add->graph,
       0,
       ingressNodeOffset,
       1,
       ingressSinkOffset,
       1});

  SimulatorState dualState;
  dualState.graphScope = graph.getOperation();
  initializeRunState(dualState, *prepared);
  auto dualPhysical = take(CgraPhysicalActionRuntime::create(
      dualPlan.resources, dualPlan.physicalUseTimings));
  auto dualTransport = take(CgraTransportRuntime::create(
      dualPlan, view, add->graph, *prepared, dualState, dualPhysical));
  channelQueue(dualState, add->op->getOpOperand(0))
      .push_back(take(tokenFromBitPattern(
          llvm::APInt(32, 99), mlir::IntegerType::get(&context(), 32))));
  llvm::SmallVector<GraphIngressEmission, 1> ingress;
  dualState.graphIngressCapture = &ingress;
  seedBlockArgument(
      dualState, graph.getBody().front().getArgument(1),
      take(tokenFromBitPattern(llvm::APInt(32, 41),
                               mlir::IntegerType::get(&context(), 32))));
  dualState.graphIngressCapture = nullptr;
  if (llvm::Error error =
          dualTransport.acceptGraphIngressEmissions(coordinate(40), ingress))
    fail(llvm::toString(std::move(error)));

  bool queueBlocked = false;
  for (unsigned iteration = 0; iteration != 24 && !queueBlocked; ++iteration) {
    const auto transportCoordinate = dualTransport.nextCoordinate();
    const auto physicalCoordinate = dualPhysical.nextCoordinate();
    const bool advancePhysical =
        physicalCoordinate &&
        (!transportCoordinate ||
         loom::sim::compareSpatialEventCoordinates(*physicalCoordinate,
                                                   *transportCoordinate) <= 0);
    if (advancePhysical) {
      auto frame = take(dualPhysical.advance());
      require(frame.has_value(), "dual-port physical event disappeared");
      (void)take(dualTransport.acceptPhysicalEvents(*frame));
      continue;
    }
    require(transportCoordinate.has_value(),
            "dual-port register FIFO became quiescent before blocking");
    auto frame = take(dualTransport.advance());
    require(frame.has_value(), "dual-port transport event disappeared");
    queueBlocked = !frame->blockedTransfers.empty();
  }
  require(queueBlocked,
          "full register FIFO did not preserve downstream backpressure");

  llvm::SmallVector<CgraActorEmission, 1> replacement;
  replacement.push_back(
      {0, 1, 0, 0,
       take(tokenFromBitPattern(llvm::APInt(32, 43),
                                mlir::IntegerType::get(&context(), 32)))});
  if (llvm::Error error =
          dualTransport.acceptActorEmissions(coordinate(50), replacement))
    fail(llvm::toString(std::move(error)));
  channelQueue(dualState, add->op->getOpOperand(0)).pop_front();
  if (llvm::Error error = dualTransport.retryBlocked(coordinate(50)))
    fail(llvm::toString(std::move(error)));

  bool sawAtomicPair = false;
  bool sawIngressPublication = false;
  bool sawReplacementPublication = false;
  for (unsigned iteration = 0; iteration != 48 && !sawReplacementPublication;
       ++iteration) {
    const auto transportCoordinate = dualTransport.nextCoordinate();
    const auto physicalCoordinate = dualPhysical.nextCoordinate();
    const bool advancePhysical =
        physicalCoordinate &&
        (!transportCoordinate ||
         loom::sim::compareSpatialEventCoordinates(*physicalCoordinate,
                                                   *transportCoordinate) <= 0);
    if (advancePhysical) {
      auto frame = take(dualPhysical.advance());
      require(frame.has_value(), "dual-port physical event disappeared");
      (void)take(dualTransport.acceptPhysicalEvents(*frame));
      continue;
    }
    require(transportCoordinate.has_value(),
            "dual-port replacement became quiescent before publication");
    auto frame = take(dualTransport.advance());
    require(frame.has_value(), "dual-port transport event disappeared");
    unsigned requested = 0;
    bool requestedWrite = false;
    bool requestedRead = false;
    for (const CgraPhysicalLifecycleEvent &event : frame->physicalEvents) {
      if (event.kind != CgraPhysicalLifecycleKind::Requested)
        continue;
      ++requested;
      requestedWrite |= event.actionOrdinal == 0;
      requestedRead |= event.actionOrdinal == 1;
    }
    sawAtomicPair |= requested == 2 && requestedWrite && requestedRead;
    for (const CgraTokenPublication &publication : frame->publications) {
      const llvm::APInt value = take(tokenBitPattern(
          publication.token, mlir::IntegerType::get(&context(), 32)));
      sawIngressPublication |= value == llvm::APInt(32, 41);
      sawReplacementPublication |= value == llvm::APInt(32, 43);
    }
  }
  require(sawAtomicPair && sawIngressPublication && sawReplacementPublication &&
              channelQueue(dualState, add->op->getOpOperand(0)).size() == 1 &&
              channelQueue(dualState, sync->op->getOpOperand(1)).size() == 1,
          "dual-port full-queue replacement was not one atomic cycle update");
}

void temporalOperandQueueCapacityAndFanoutAreAtomic() {
  auto artifact = fanoutProgram();
  auto view = take(artifact.view());
  llvm::SmallVector<const dataflow::CanonicalActorView *, 3> adds;
  for (const dataflow::CanonicalActorView &actor : view.actors())
    if (dataflow::operationSchemaOf(actor.op) ==
        dataflow::OperationSchemaId::ArithAddI)
      adds.push_back(&actor);
  require(adds.size() == 3,
          "operand-queue fixture lacks its three add actors");
  const dataflow::CanonicalActorView &left = *adds[0];
  const dataflow::CanonicalActorView &right = *adds[1];
  auto graphView = take(view.resolve(left.graph));
  auto graph = mlir::cast<dataflow::GraphOp>(graphView.op);
  GraphPreparationResult preparedResult =
      take(prepareGraphExecution(artifact.module(), graph));
  auto *prepared = std::get_if<PreparedGraphExecution>(&preparedResult);
  require(prepared, "operand-queue graph preparation failed");

  std::uint64_t leftSemantic = std::numeric_limits<std::uint64_t>::max();
  for (auto [ordinal, actor] : llvm::enumerate(prepared->actorPlans))
    if (actor.operation == left.op)
      leftSemantic = ordinal;
  require(leftSemantic != std::numeric_limits<std::uint64_t>::max(),
          "operand-queue actor has no semantic binding");

  const dataflow::CanonicalGraphProducerEndpointRef producer(
      dataflow::GraphIngressTokenRef{
          dataflow::GraphValueInputTokenRef{left.graph, 0}});
  const dataflow::CanonicalGraphConsumerEndpointRef leftSink(
      dataflow::ActorTokenOperandRef{left.ref, 0});
  const dataflow::CanonicalGraphConsumerEndpointRef rightSink(
      dataflow::ActorTokenOperandRef{right.ref, 0});
  loom::fabric::FabricPeOccurrenceRef pe;
  const fabric::LogicalOperandQueueKey leftQueue{{pe, 0}, 0, 0};
  const fabric::LogicalOperandQueueKey rightQueue{{pe, 0}, 1, 0};

  CgraFrozenExecutionPlan plan;
  plan.transport.localTransfers.push_back(
      {producer, left.graph, 0, 2});
  plan.transport.localTransferSinks.push_back({leftSink});
  plan.transport.localTransferSinks.push_back({rightSink});
  plan.transport.consumedUses.push_back({leftSink, 0, 0});
  plan.transport.consumedUses.push_back({rightSink, 0, 0});
  plan.transport.operandQueueMatches.push_back(
      {leftSink, leftQueue, 0, 2});
  plan.transport.operandQueueMatches.push_back(
      {rightSink, rightQueue, 1, 2});
  plan.transport.operandQueueActivations.push_back(
      {producer,
       {loom::fabric::FabricTransportEndpointOwnerRef::of(pe), 0},
       llvm::APInt(1, 0),
       0,
       2});

  SimulatorState state;
  state.graphScope = graph.getOperation();
  initializeRunState(state, *prepared);
  TokenQueue &leftChannel = channelQueue(state, left.op->getOpOperand(0));
  TokenQueue &rightChannel = channelQueue(state, right.op->getOpOperand(0));
  leftChannel.push_back(take(tokenFromBitPattern(
      llvm::APInt(32, 7), mlir::IntegerType::get(&context(), 32))));
  auto physical = take(CgraPhysicalActionRuntime::create(
      plan.resources, plan.physicalUseTimings));
  auto transport = take(CgraTransportRuntime::create(
      plan, view, left.graph, *prepared, state, physical));

  llvm::SmallVector<GraphIngressEmission, 1> first;
  first.push_back(
      {1, 0,
       take(tokenFromBitPattern(llvm::APInt(32, 11),
                                mlir::IntegerType::get(&context(), 32)))});
  if (llvm::Error error =
          transport.acceptGraphIngressEmissions(coordinate(70), first))
    fail(llvm::toString(std::move(error)));
  auto firstFrame = take(transport.advance());
  require(firstFrame && firstFrame->publications.size() == 1 &&
              leftChannel.size() == 2 && rightChannel.size() == 1,
          "Temporal operand fanout did not fill both allocation units");

  llvm::SmallVector<GraphIngressEmission, 1> second;
  second.push_back(
      {1, 1,
       take(tokenFromBitPattern(llvm::APInt(32, 13),
                                mlir::IntegerType::get(&context(), 32)))});
  if (llvm::Error error =
          transport.acceptGraphIngressEmissions(coordinate(71), second))
    fail(llvm::toString(std::move(error)));
  auto blocked = take(transport.advance());
  require(blocked && blocked->publications.empty() &&
              blocked->blockedTransfers.size() == 1 &&
              leftChannel.size() == 2 && rightChannel.size() == 1,
          "full Temporal operand unit allowed partial fanout");

  leftChannel.pop_front();
  const std::array<CgraActorLifecycleEvent, 1> committed = {
      CgraActorLifecycleEvent{CgraActorLifecycleKind::Committed, leftSemantic,
                              0, 0, 0, blocked->coordinate}};
  if (llvm::Error error = transport.acceptActorCommits(committed))
    fail(llvm::toString(std::move(error)));
  if (llvm::Error error = transport.retryBlocked(blocked->coordinate))
    fail(llvm::toString(std::move(error)));
  auto replacement = take(transport.advance());
  require(replacement && replacement->publications.size() == 1 &&
              leftChannel.size() == 2 && rightChannel.size() == 2 &&
              take(tokenBitPattern(
                  leftChannel.front(),
                  mlir::IntegerType::get(&context(), 32))) ==
                  llvm::APInt(32, 11),
          "Temporal operand dequeue replacement did not commit atomically");
}

} // namespace

int main() {
  localRealizationEdgePublishesThroughExactConsumer();
  registerFifoWriteAndReadShareOneDurableQueue();
  temporalOperandQueueCapacityAndFanoutAreAtomic();
  return EXIT_SUCCESS;
}
