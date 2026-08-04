#include "CgraComputeRuntime.h"

#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Dataflow/IR/DataflowDialect.h"
#include "Evaluation/NumericValue.h"
#include "Fabric/IR/ResourceContract.h"

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
  llvm::errs() << "CGRA compute runtime test: " << message << '\n';
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
  dataflow.graph private @add(
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

  dataflow.graph private @stream(
      %start: none, %init: i32, %limit: i32, %step: i32) -> (i32)
      attributes {
        input_segments = array<i32: 3, 0, 0>,
        result_segments = array<i32: 0, 1, 0>
      } {
    %iv, %phase = dataflow.stream %init, %limit, %step
        step add while slt : i32
    %tokens = dataflow.invariant %phase, %start : none
    %complete:2 = dataflow.demux %phase, %tokens
        : (i1, none) -> (none, none)
    dataflow.graph.return values() streams(%iv : i32) memories()
        complete(%complete#0 : none)
  }
}
)mlir",
                                                        &context());
  if (!module)
    fail("failed to parse compute fixture");
  return take(dataflow::finalizeCanonicalDataflow(module.get()));
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

bool hasPhysical(const CgraComputeLifecycleFrame &frame,
                 CgraPhysicalLifecycleKind kind) {
  return llvm::any_of(frame.physicalEvents,
                      [kind](const CgraPhysicalLifecycleEvent &event) {
                        return event.kind == kind;
                      });
}

ActorExecutionPlan &semanticActor(PreparedGraphExecution &execution,
                                  mlir::Operation *operation) {
  for (ActorExecutionPlan &actor : execution.actorPlans)
    if (actor.operation == operation)
      return actor;
  fail("canonical actor is absent from prepared graph execution");
}

CgraFrozenExecutionPlan selectedPlan(const dataflow::CanonicalActorView &actor,
                                     const ActorExecutionPlan &semantic,
                                     const fabric::ResourceContract &contract) {
  CgraFrozenExecutionPlan plan;
  plan.computeActors.push_back(
      {actor.ref,
       actor.graph,
       {},
       {},
       0,
       static_cast<std::uint32_t>(semantic.handshakeCases.size())});
  std::vector<CgraResourcePatternSelection> selections;
  for (const auto &transition : semantic.handshakeCases) {
    const std::uint64_t action = plan.physicalUseTimings.size();
    plan.computeTransitions.push_back({transition.ordinal, action, 1});
    plan.actorTransitionPhysicalUses.push_back(action);
    plan.physicalUseTimings.push_back({action, 0, 1, 2, 0, 2, 1});
    selections.push_back({0, fabric::UsePatternKey(0)});
  }
  const fabric::ResourceContract *contracts[] = {&contract};
  plan.resources = take(freezeCgraResourceRuntimePlan(contracts, selections));
  return plan;
}

void computeCommitWaitsForExactPhysicalLifecycle() {
  auto artifact = program();
  auto view = take(artifact.view());
  const dataflow::CanonicalActorView *add = nullptr;
  for (const dataflow::CanonicalActorView &actor : view.actors()) {
    if (dataflow::operationSchemaOf(actor.op) ==
        dataflow::OperationSchemaId::ArithAddI) {
      add = &actor;
      break;
    }
  }
  require(add != nullptr, "fixture has no canonical add actor");
  auto graphView = take(view.resolve(add->graph));
  auto graph = mlir::cast<dataflow::GraphOp>(graphView.op);
  GraphPreparationResult preparedResult =
      take(prepareGraphExecution(artifact.module(), graph));
  auto *prepared = std::get_if<PreparedGraphExecution>(&preparedResult);
  require(prepared != nullptr, "compute graph preparation failed");

  const fabric::ResourceContract contract = resourceContract();
  CgraFrozenExecutionPlan plan =
      selectedPlan(*add, semanticActor(*prepared, add->op), contract);

  SimulatorState state;
  state.graphScope = graph.getOperation();
  initializeRunState(state, *prepared);
  seedBlockArgument(state, graph.getStart(), noneToken());
  mlir::Block &entry = graph.getBody().front();
  seedBlockArgument(state, entry.getArgument(1),
                    take(tokenFromBitPattern(llvm::APInt(32, 7),
                                             entry.getArgument(1).getType())));
  seedBlockArgument(state, entry.getArgument(2),
                    take(tokenFromBitPattern(llvm::APInt(32, 9),
                                             entry.getArgument(2).getType())));

  auto runtime = take(
      CgraComputeRuntime::create(plan, view, add->graph, *prepared, state));
  auto frame = take(runtime.start(coordinate(0)));
  require(frame && hasPhysical(*frame, CgraPhysicalLifecycleKind::Requested) &&
              hasPhysical(*frame, CgraPhysicalLifecycleKind::Granted) &&
              frame->actorEvents.empty(),
          "initial readiness did not request and grant the selected use");

  frame = take(runtime.advance());
  require(frame &&
              frame->coordinate.referenceCycle ==
                  take(loom::evaluation::ExactRatio::get(1, 1)) &&
              frame->coordinate.delta == 0 &&
              hasPhysical(*frame, CgraPhysicalLifecycleKind::Committed) &&
              frame->actorEvents.empty() &&
              state.pendingChannelOrdinals.empty(),
          "software actor committed before the owner resource transition");

  frame = take(runtime.advance());
  require(frame &&
              frame->coordinate.referenceCycle ==
                  take(loom::evaluation::ExactRatio::get(1, 1)) &&
              frame->coordinate.delta == 1 && frame->actorEvents.size() == 1 &&
              frame->actorEvents.front().kind ==
                  CgraComputeActorLifecycleKind::Committed &&
              state.pendingChannelOrdinals.size() == 1,
          "actor commit did not use the shared provider at the next delta");

  frame = take(runtime.advance());
  require(frame &&
              frame->coordinate.referenceCycle ==
                  take(loom::evaluation::ExactRatio::get(2, 1)) &&
              hasPhysical(*frame, CgraPhysicalLifecycleKind::Retired) &&
              frame->actorEvents.empty() &&
              frame->physicalCompletions.size() == 1 &&
              !runtime.hasPendingEvents(),
          "physical retirement fabricated actor retirement or stayed active");
}

void statefulActorCannotBypassUnmodeledTransport() {
  auto artifact = program();
  auto view = take(artifact.view());
  const dataflow::CanonicalActorView *stream = nullptr;
  for (const dataflow::CanonicalActorView &actor : view.actors())
    if (dataflow::operationSchemaOf(actor.op) ==
        dataflow::OperationSchemaId::DataflowStream)
      stream = &actor;
  require(stream != nullptr, "fixture has no canonical stream actor");
  auto graphView = take(view.resolve(stream->graph));
  auto graph = mlir::cast<dataflow::GraphOp>(graphView.op);
  GraphPreparationResult preparedResult =
      take(prepareGraphExecution(artifact.module(), graph));
  auto *prepared = std::get_if<PreparedGraphExecution>(&preparedResult);
  require(prepared != nullptr, "stream graph preparation failed");

  const fabric::ResourceContract contract = resourceContract();
  CgraFrozenExecutionPlan plan =
      selectedPlan(*stream, semanticActor(*prepared, stream->op), contract);
  SimulatorState state;
  state.graphScope = graph.getOperation();
  initializeRunState(state, *prepared);
  seedBlockArgument(state, graph.getStart(), noneToken());
  mlir::Block &entry = graph.getBody().front();
  for (unsigned ordinal = 1; ordinal != 4; ++ordinal)
    seedBlockArgument(state, entry.getArgument(ordinal),
                      take(tokenFromBitPattern(
                          llvm::APInt(32, ordinal == 2 ? 3 : ordinal - 1),
                          entry.getArgument(ordinal).getType())));

  auto runtime = take(
      CgraComputeRuntime::create(plan, view, stream->graph, *prepared, state));
  (void)take(runtime.start(coordinate(0)));
  (void)take(runtime.advance());
  auto committed = take(runtime.advance());
  require(committed && committed->actorEvents.size() == 1,
          "stream transition did not commit");
  auto next = take(runtime.advance());
  require(next &&
              next->coordinate.referenceCycle ==
                  take(loom::evaluation::ExactRatio::get(2, 1)) &&
              !hasPhysical(*next, CgraPhysicalLifecycleKind::Requested),
          "stateful actor bypassed its pending transport obligation");
}

} // namespace

int main() {
  computeCommitWaitsForExactPhysicalLifecycle();
  statefulActorCannotBypassUnmodeledTransport();
  return EXIT_SUCCESS;
}
