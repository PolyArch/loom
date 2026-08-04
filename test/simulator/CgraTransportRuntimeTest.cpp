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

#include <cstdlib>
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
  plan.computeActors.push_back({add->ref, add->graph, {}, {}, 0, 0});
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

  llvm::SmallVector<CgraComputeActorEmission, 1> emissions;
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
  physicalPlan.physicalUseClients.push_back(
      CgraPhysicalUseClientKind::ProducedTransport);
  physicalPlan.physicalUseClients.push_back(
      CgraPhysicalUseClientKind::ConsumedTransport);
  physicalPlan.resources.selectedUses.push_back({});
  physicalPlan.resources.selectedUses.push_back({});
  physicalPlan.physicalUseTimings.push_back(
      {0, 0, std::nullopt, 1, 0, 1, std::nullopt});
  physicalPlan.physicalUseTimings.push_back(
      {1, 0, std::nullopt, 1, 0, 1, std::nullopt});
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
  if (llvm::Error error = selectedTransport.acceptPhysicalEvents(*granted))
    fail(llvm::toString(std::move(error)));
  auto consumedRequest = take(selectedTransport.advance());
  require(consumedRequest && consumedRequest->physicalEvents.size() == 1 &&
              consumedRequest->physicalEvents.front().kind ==
                  CgraPhysicalLifecycleKind::Requested &&
              consumedRequest->physicalEvents.front().actionOrdinal == 1 &&
              consumedRequest->publications.empty() &&
              loom::sim::compareSpatialEventCoordinates(
                  consumedRequest->coordinate, coordinate(4, 1)) == 0,
          "selected Consumed use did not gate sink publication");
  auto consumedGrant = take(selectedPhysical.advance());
  require(consumedGrant && consumedGrant->events.size() == 1 &&
              consumedGrant->events.front().kind ==
                  CgraPhysicalLifecycleKind::Granted &&
              consumedGrant->events.front().actionOrdinal == 1,
          "selected Consumed use did not grant");
  if (llvm::Error error =
          selectedTransport.acceptPhysicalEvents(*consumedGrant))
    fail(llvm::toString(std::move(error)));
  auto selectedPublication = take(selectedTransport.advance());
  require(selectedPublication &&
              selectedPublication->publications.size() == 1 &&
              loom::sim::compareSpatialEventCoordinates(
                  selectedPublication->coordinate, coordinate(4, 2)) == 0,
          "selected endpoint uses did not gate token publication");
}

} // namespace

int main() {
  localRealizationEdgePublishesThroughExactConsumer();
  return EXIT_SUCCESS;
}
