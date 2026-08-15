#include "CgraComputeRuntime.h"

#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Dataflow/IR/DataflowDialect.h"
#include "Evaluation/NumericValue.h"
#include "Fabric/IR/OperationResourceContract.h"
#include "Fabric/IR/ResourceContract.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <initializer_list>
#include <system_error>
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
    registry.insert<dataflow::DataflowDialect, mlir::arith::ArithDialect,
                    mlir::vector::VectorDialect>();
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

  dataflow.graph private @shuffle(
      %start: none, %lhs: vector<2xi8>, %rhs: vector<1xi8>)
      -> (vector<3xi8>)
      attributes {
        input_segments = array<i32: 2, 0, 0>,
        result_segments = array<i32: 1, 0, 0>
      } {
    %value = vector.shuffle %lhs, %rhs [1, 2, -1]
        : vector<2xi8>, vector<1xi8>
    %published:2 = dataflow.sync %start, %value
        : (none, vector<3xi8>) -> (none, vector<3xi8>)
    dataflow.graph.return values(%published#1 : vector<3xi8>) streams()
        memories() complete(%published#0 : none)
  }

  dataflow.graph private @serialize(
      %start: none, %init: i8, %limit: i8, %step: i8)
      -> (i8, i1)
      attributes {
        input_segments = array<i32: 3, 0, 0>,
        result_segments = array<i32: 0, 2, 0>
      } {
    %iv, %phase = dataflow.stream %init, %limit, %step
        step add while ult : i8
    %values, %mask, %group_phase =
        dataflow.parallelize %iv, %phase
          : (i8, i1) -> (vector<2xi8>, vector<2xi1>, i1)
    %scalar, %scalar_phase =
        dataflow.serialize %values, %mask, %group_phase
        : (vector<2xi8>, vector<2xi1>, i1) -> (i8, i1)
    %units = dataflow.invariant %scalar_phase, %start : none
    %complete:2 = dataflow.demux %scalar_phase, %units
        : (i1, none) -> (none, none)
    dataflow.graph.return values()
        streams(%scalar, %scalar_phase : i8, i1) memories()
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

bool isCoordinate(const loom::sim::SpatialEventCoordinate &actual,
                  const loom::sim::SpatialEventCoordinate &expected) {
  return loom::sim::compareSpatialEventCoordinates(actual, expected) == 0;
}

bool hasPhysical(const CgraComputeLifecycleFrame &frame,
                 CgraPhysicalLifecycleKind kind) {
  return llvm::any_of(frame.physicalEvents,
                      [kind](const CgraPhysicalLifecycleEvent &event) {
                        return event.kind == kind;
                      });
}

bool hasPhysical(const CgraComputeLifecycleFrame &frame,
                 CgraPhysicalLifecycleKind kind,
                 std::uint32_t ownerEventOrdinal) {
  return llvm::any_of(
      frame.physicalEvents,
      [kind, ownerEventOrdinal](const CgraPhysicalLifecycleEvent &event) {
        return event.kind == kind &&
               event.ownerEventOrdinal == ownerEventOrdinal;
      });
}

Token vectorToken(mlir::VectorType type,
                  std::initializer_list<std::uint8_t> lanes) {
  require(static_cast<std::size_t>(type.getNumElements()) == lanes.size(),
          "vector token lane count does not match its type");
  llvm::APInt bits(static_cast<unsigned>(lanes.size()) * 8, 0);
  unsigned ordinal = 0;
  for (std::uint8_t lane : lanes)
    bits.insertBits(llvm::APInt(8, lane), 8 * ordinal++);
  return take(tokenFromBitPattern(bits, type));
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
       static_cast<std::uint32_t>(semantic.handshakeCases.size()),
       std::nullopt,
       0});
  std::vector<CgraResourcePatternSelection> selections;
  const fabric::UsePattern use =
      contract.usePattern(fabric::UsePatternKey(0));
  const auto ranks = contract.eventOrder(use.timingAndProgress);
  for (const auto &transition : semantic.handshakeCases) {
    const std::uint64_t action = plan.physicalUseTimings.size();
    plan.computeTransitions.push_back({transition.ordinal, action, 1});
    plan.actorTransitionPhysicalUses.push_back(action);
    plan.physicalUseClients.push_back(
        CgraPhysicalUseClientKind::ComputeTransition);
    plan.physicalUseTimings.push_back(
        {action,
         ranks[use.acquire.ordinal()],
         use.commit ? std::optional<std::uint32_t>(
                          ranks[use.commit->event.ordinal()])
                    : std::nullopt,
         ranks[use.release.ordinal()],
         use.acquire.ordinal(),
         use.release.ordinal(),
         use.commit ? std::optional<std::uint32_t>(use.commit->event.ordinal())
                    : std::nullopt});
    selections.push_back({0, fabric::UsePatternKey(0)});
  }
  const fabric::ResourceContract *contracts[] = {&contract};
  plan.resources = take(freezeCgraResourceRuntimePlan(contracts, selections));
  return plan;
}

void temporalDispatchFollowsFabricRoundRobinSlots() {
  CgraTemporalDispatchDomainPlan domain;
  domain.candidateCount = 2;
  domain.resetPosition = 0;

  require(isCoordinate(take(projectCgraTemporalDispatchCoordinate(
                           domain, 0, coordinate(0))),
                       coordinate(0)),
          "reset dispatch candidate did not receive the reset slot");
  require(isCoordinate(take(projectCgraTemporalDispatchCoordinate(
                           domain, 1, coordinate(0))),
                       coordinate(1)),
          "second dispatch candidate did not follow the reset slot");
  require(isCoordinate(take(projectCgraTemporalDispatchCoordinate(
                           domain, 1, coordinate(1, 3))),
                       coordinate(1, 3)),
          "selected dispatch candidate lost its current delta");
  require(isCoordinate(take(projectCgraTemporalDispatchCoordinate(
                           domain, 0, coordinate(1))),
                       coordinate(2)),
          "dispatch cursor did not advance after the second candidate");

  loom::sim::SpatialEventCoordinate halfCycle{
      take(loom::evaluation::ExactRatio::get(1, 2)), 7};
  require(isCoordinate(take(projectCgraTemporalDispatchCoordinate(
                           domain, 1, halfCycle)),
                       coordinate(1)),
          "fractional request did not wait for the next selected PE slot");

  domain.resetPosition = 1;
  require(isCoordinate(take(projectCgraTemporalDispatchCoordinate(
                           domain, 1, coordinate(0))),
                       coordinate(0)) &&
              isCoordinate(take(projectCgraTemporalDispatchCoordinate(
                               domain, 0, coordinate(0))),
                           coordinate(1)),
          "dispatch reset position did not come from the Fabric policy");
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

  // This fixture selects Fabric-local intrinsic timing only. Mapping-owned
  // causal release extension is deliberately absent from this selected use.
  const fabric::ResourceContract contract =
      fabric::oneCycleElasticOperationResourceContract();
  CgraFrozenExecutionPlan plan =
      selectedPlan(*add, semanticActor(*prepared, add->op), contract);
  const std::uint64_t transportAction = plan.physicalUseTimings.size();
  plan.physicalUseClients.push_back(
      CgraPhysicalUseClientKind::ProducedTransport);
  plan.resources.selectedUses.push_back({});
  plan.physicalUseTimings.push_back(
      {transportAction, 0, std::nullopt, 0, 0, 0, std::nullopt});

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

  auto physical = take(CgraPhysicalActionRuntime::create(
      plan.resources, plan.physicalUseTimings));
  (void)take(physical.request(transportAction, 0, coordinate(0)));
  auto runtime = take(CgraComputeRuntime::create(plan, view, add->graph,
                                                 *prepared, state, physical));
  if (llvm::Error error = runtime.start(coordinate(0)))
    fail(llvm::toString(std::move(error)));
  auto frame = take(runtime.advance());
  require(frame && hasPhysical(*frame, CgraPhysicalLifecycleKind::Requested) &&
              frame->actorEvents.empty(),
          "compute action did not expose its request lifecycle");

  auto physicalFrame = take(physical.advance());
  require(physicalFrame && physicalFrame->events.size() == 2,
          "shared physical calendar did not retain both clients");
  auto computePhysical = take(runtime.acceptPhysicalEvents(*physicalFrame));
  require(computePhysical.coordinate.referenceCycle ==
                  take(loom::evaluation::ExactRatio::get(0, 1)) &&
              computePhysical.coordinate.delta == 0 &&
              hasPhysical(computePhysical, CgraPhysicalLifecycleKind::Granted,
                          0) &&
              computePhysical.actorEvents.empty(),
          "compute client did not receive its exact grant");

  physicalFrame = take(physical.advance());
  require(physicalFrame && loom::sim::compareSpatialEventCoordinates(
                               physicalFrame->coordinate, coordinate(0)) == 0,
          "zero-hold transport action did not retire at its grant coordinate");
  computePhysical = take(runtime.acceptPhysicalEvents(*physicalFrame));
  require(computePhysical.physicalEvents.empty(),
          "compute client consumed another client's retirement");

  physicalFrame = take(physical.advance());
  require(physicalFrame.has_value(),
          "physical publication and release frame is missing");
  computePhysical = take(runtime.acceptPhysicalEvents(*physicalFrame));
  require(
      computePhysical.coordinate.referenceCycle ==
              take(loom::evaluation::ExactRatio::get(1, 1)) &&
          computePhysical.coordinate.delta == 0 &&
          hasPhysical(computePhysical, CgraPhysicalLifecycleKind::Committed,
                      1) &&
          hasPhysical(computePhysical, CgraPhysicalLifecycleKind::Retired, 2) &&
          computePhysical.actorEvents.empty() &&
          computePhysical.physicalCompletions.empty() &&
          state.pendingChannelOrdinals.empty(),
      "intrinsic result-slot lifecycle did not publish and release at t + 1");

  frame = take(runtime.advance());
  require(frame &&
              frame->coordinate.referenceCycle ==
                  take(loom::evaluation::ExactRatio::get(1, 1)) &&
              frame->coordinate.delta == 1 && frame->actorEvents.size() == 1 &&
              frame->actorEvents.front().kind ==
                  CgraActorLifecycleKind::Committed &&
              frame->actorEmissions.size() == 1 &&
              frame->actorEmissions.front().resultOrdinal == 0 &&
              take(tokenBitPattern(frame->actorEmissions.front().token,
                                   mlir::IntegerType::get(&context(), 32))) ==
                  llvm::APInt(32, 16) &&
              frame->physicalCompletions.size() == 1 &&
              state.pendingChannelOrdinals.empty() &&
              !physical.hasPendingActions(),
          "actor commit did not hand its shared-provider token to transport");
  require(!take(physical.advance()),
          "intrinsic result-slot lifecycle retained a later physical event");
  if (llvm::Error error = runtime.retireActor(0, 0, coordinate(1, 1)))
    fail(llvm::toString(std::move(error)));
  require(!runtime.hasActiveActors(),
          "coordinated actor retirement did not release the firing");
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

  auto physical = take(CgraPhysicalActionRuntime::create(
      plan.resources, plan.physicalUseTimings));
  auto runtime = take(CgraComputeRuntime::create(plan, view, stream->graph,
                                                 *prepared, state, physical));
  if (llvm::Error error = runtime.start(coordinate(0)))
    fail(llvm::toString(std::move(error)));
  (void)take(runtime.advance());
  auto physicalFrame = take(physical.advance());
  require(physicalFrame.has_value(), "stream grant frame is missing");
  (void)take(runtime.acceptPhysicalEvents(*physicalFrame));
  physicalFrame = take(physical.advance());
  require(physicalFrame.has_value(), "stream commit frame is missing");
  (void)take(runtime.acceptPhysicalEvents(*physicalFrame));
  auto committed = take(runtime.advance());
  require(committed && committed->actorEvents.size() == 1,
          "stream transition did not commit");
  physicalFrame = take(physical.advance());
  require(physicalFrame.has_value(), "stream retirement frame is missing");
  auto retired = take(runtime.acceptPhysicalEvents(*physicalFrame));
  require(retired.physicalCompletions.size() == 1,
          "stream physical execution did not complete");
  llvm::SmallBitVector candidates(prepared->actorPlans.size(), true);
  if (llvm::Error error =
          runtime.acceptReadyCandidates(coordinate(2), candidates))
    fail(llvm::toString(std::move(error)));
  require(!runtime.hasPendingEvents() && runtime.hasActiveActors(),
          "stateful actor bypassed its pending transport obligation");
}

void structuralVectorUsesSharedPhysicalLifecycle() {
  auto artifact = program();
  auto view = take(artifact.view());
  const dataflow::CanonicalActorView *shuffle = nullptr;
  for (const dataflow::CanonicalActorView &actor : view.actors())
    if (dataflow::operationSchemaOf(actor.op) ==
        dataflow::OperationSchemaId::VectorShuffle)
      shuffle = &actor;
  require(shuffle != nullptr, "fixture has no canonical shuffle actor");

  auto graphView = take(view.resolve(shuffle->graph));
  auto graph = mlir::cast<dataflow::GraphOp>(graphView.op);
  GraphPreparationResult preparedResult =
      take(prepareGraphExecution(artifact.module(), graph));
  auto *prepared = std::get_if<PreparedGraphExecution>(&preparedResult);
  require(prepared != nullptr, "shuffle graph preparation failed");

  const fabric::ResourceContract contract = resourceContract();
  CgraFrozenExecutionPlan plan =
      selectedPlan(*shuffle, semanticActor(*prepared, shuffle->op), contract);
  SimulatorState state;
  state.graphScope = graph.getOperation();
  initializeRunState(state, *prepared);
  seedBlockArgument(state, graph.getStart(), noneToken());
  mlir::Block &entry = graph.getBody().front();
  seedBlockArgument(
      state, entry.getArgument(1),
      vectorToken(mlir::cast<mlir::VectorType>(entry.getArgument(1).getType()),
                  {1, 2}));
  seedBlockArgument(
      state, entry.getArgument(2),
      vectorToken(mlir::cast<mlir::VectorType>(entry.getArgument(2).getType()),
                  {3}));

  auto physical = take(CgraPhysicalActionRuntime::create(
      plan.resources, plan.physicalUseTimings));
  auto runtime = take(CgraComputeRuntime::create(plan, view, shuffle->graph,
                                                 *prepared, state, physical));
  if (llvm::Error error = runtime.start(coordinate(0)))
    fail(llvm::toString(std::move(error)));
  auto requested = take(runtime.advance());
  require(requested &&
              hasPhysical(*requested, CgraPhysicalLifecycleKind::Requested),
          "shuffle compute action did not request its physical resource");

  auto physicalFrame = take(physical.advance());
  require(physicalFrame.has_value(), "shuffle grant frame is missing");
  auto accepted = take(runtime.acceptPhysicalEvents(*physicalFrame));
  require(hasPhysical(accepted, CgraPhysicalLifecycleKind::Granted),
          "shuffle compute action did not receive its grant");

  physicalFrame = take(physical.advance());
  require(physicalFrame.has_value(), "shuffle commit frame is missing");
  accepted = take(runtime.acceptPhysicalEvents(*physicalFrame));
  require(hasPhysical(accepted, CgraPhysicalLifecycleKind::Committed),
          "shuffle resource did not expose its commit transition");

  auto committed = take(runtime.advance());
  require(committed && committed->actorEvents.size() == 1 &&
              committed->actorEvents.front().kind ==
                  CgraActorLifecycleKind::Committed &&
              committed->actorEmissions.size() == 1,
          "shuffle actor did not commit through the CGRA lifecycle");
  const CgraActorLifecycleEvent committedActor = committed->actorEvents.front();
  auto lanes = take(vectorPrimitiveValues(
      committed->actorEmissions.front().token,
      mlir::cast<mlir::VectorType>(shuffle->op->getResult(0).getType()),
      shuffle->op));
  require(lanes.size() == 3 && lanes[0].isDefined() &&
              *lanes[0].bits == llvm::APInt(8, 2) && lanes[1].isDefined() &&
              *lanes[1].bits == llvm::APInt(8, 3) &&
              lanes[2].state == loom::sim::PrimitiveValueState::Poison,
          "CGRA shuffle result diverged from the shared mixed-lane semantics");

  physicalFrame = take(physical.advance());
  require(physicalFrame.has_value(), "shuffle retirement frame is missing");
  accepted = take(runtime.acceptPhysicalEvents(*physicalFrame));
  require(hasPhysical(accepted, CgraPhysicalLifecycleKind::Retired) &&
              accepted.physicalCompletions.size() == 1,
          "shuffle physical execution did not retire");
  if (llvm::Error error =
          runtime.retireActor(committedActor.semanticActorOrdinal,
                              committedActor.occurrenceOrdinal, coordinate(2)))
    fail(llvm::toString(std::move(error)));
  require(!runtime.hasActiveActors(),
          "shuffle actor stayed active after coordinated retirement");
}

void exceptionalSerializeRejectsBeforePhysicalRequest() {
  auto artifact = program();
  auto view = take(artifact.view());
  const dataflow::CanonicalActorView *serialize = nullptr;
  for (const dataflow::CanonicalActorView &actor : view.actors())
    if (dataflow::operationSchemaOf(actor.op) ==
        dataflow::OperationSchemaId::DataflowSerialize)
      serialize = &actor;
  require(serialize != nullptr, "fixture has no canonical serialize actor");

  auto graphView = take(view.resolve(serialize->graph));
  auto graph = mlir::cast<dataflow::GraphOp>(graphView.op);
  GraphPreparationResult preparedResult =
      take(prepareGraphExecution(artifact.module(), graph));
  auto *prepared = std::get_if<PreparedGraphExecution>(&preparedResult);
  require(prepared != nullptr, "serialize graph preparation failed");

  ActorExecutionPlan &semantic = semanticActor(*prepared, serialize->op);
  const fabric::ResourceContract contract = resourceContract();
  CgraFrozenExecutionPlan plan = selectedPlan(*serialize, semantic, contract);
  SimulatorState state;
  state.graphScope = graph.getOperation();
  initializeRunState(state, *prepared);
  state.channelSlots[semantic.firstInputChannel].ready.push_back(vectorToken(
      mlir::cast<mlir::VectorType>(serialize->op->getOperand(0).getType()),
      {1, 2}));
  const llvm::SmallVector<loom::sim::PrimitiveValue, 2> mask = {
      loom::sim::PrimitiveValue::integer(llvm::APInt(1, 1)),
      loom::sim::PrimitiveValue::poison()};
  state.channelSlots[semantic.firstInputChannel + 1].ready.push_back(
      take(tokenFromVectorPrimitiveValues(
          mask,
          mlir::cast<mlir::VectorType>(serialize->op->getOperand(1).getType()),
          serialize->op)));
  state.channelSlots[semantic.firstInputChannel + 2].ready.push_back(
      boolValueToken(true));

  auto physical = take(CgraPhysicalActionRuntime::create(
      plan.resources, plan.physicalUseTimings));
  auto runtime = take(CgraComputeRuntime::create(plan, view, serialize->graph,
                                                 *prepared, state, physical));
  const std::uint64_t mutationEpoch = state.actorMutationEpoch;
  llvm::Error error = runtime.start(coordinate(0));
  if (!error)
    fail("CGRA serialize accepted an exceptional mask");
  const std::error_code code = llvm::errorToErrorCode(std::move(error));
  require(code == std::make_error_code(std::errc::not_supported),
          "CGRA serialize exceptional mask was not typed Unsupported");
  require(state.actorMutationEpoch == mutationEpoch &&
              state.channelSlots[semantic.firstInputChannel].ready.size() ==
                  1 &&
              state.channelSlots[semantic.firstInputChannel + 1].ready.size() ==
                  1 &&
              state.channelSlots[semantic.firstInputChannel + 2].ready.size() ==
                  1 &&
              !physical.nextCoordinate() && !runtime.hasPendingEvents(),
          "CGRA serialize exceptional mask advanced execution state");
}

} // namespace

int main() {
  temporalDispatchFollowsFabricRoundRobinSlots();
  computeCommitWaitsForExactPhysicalLifecycle();
  statefulActorCannotBypassUnmodeledTransport();
  structuralVectorUsesSharedPhysicalLifecycle();
  exceptionalSerializeRejectsBeforePhysicalRequest();
  return EXIT_SUCCESS;
}
