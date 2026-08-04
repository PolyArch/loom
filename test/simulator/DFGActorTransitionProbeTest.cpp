#include "DFGSimulatorInternal.h"

#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Dataflow/IR/DataflowDialect.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/APInt.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <optional>
#include <utility>

namespace {

using namespace loom::sim::detail;

[[noreturn]] void fail(llvm::StringRef message) {
  llvm::errs() << "DFG actor transition probe test: " << message << '\n';
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
    registry.insert<dataflow::DataflowDialect>();
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
  dataflow.graph private @select(
      %start: none, %selector: index,
      %lane0: i32, %lane1: i32, %lane2: i32) -> (i32)
      attributes {
        input_segments = array<i32: 4, 0, 0>,
        result_segments = array<i32: 1, 0, 0>
      } {
    %selected = dataflow.mux %selector, %lane0, %lane1, %lane2
        : (index, i32, i32, i32) -> i32
    %published:2 = dataflow.sync %start, %selected
        : (none, i32) -> (none, i32)
    dataflow.graph.return values(%published#1 : i32) streams() memories()
        complete(%published#0 : none)
  }

  dataflow.graph private @stream_probe(
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
    fail("failed to parse probe fixture");
  return take(dataflow::finalizeCanonicalDataflow(module.get()));
}

dataflow::GraphOp findGraph(mlir::ModuleOp module,
                            dataflow::OperationSchemaId schema) {
  for (dataflow::GraphOp graph : module.getOps<dataflow::GraphOp>())
    for (mlir::Operation &operation : graph.getBody().front())
      if (dataflow::operationSchemaOf(&operation) == schema)
        return graph;
  fail("fixture graph is missing its actor schema");
}

PreparedGraphExecution prepare(mlir::ModuleOp module, dataflow::GraphOp graph) {
  GraphPreparationResult result = take(prepareGraphExecution(module, graph));
  if (auto *failure = std::get_if<GraphPreparationFailure>(&result))
    fail(failure->diagnostics.empty() ? "graph preparation failed"
                                      : failure->diagnostics.front());
  return std::move(std::get<PreparedGraphExecution>(result));
}

ActorExecutionPlan &findActor(PreparedGraphExecution &execution,
                              dataflow::OperationSchemaId schema) {
  for (ActorExecutionPlan &actor : execution.actorPlans)
    if (actor.projection.schema == schema)
      return actor;
  fail("prepared graph is missing its actor plan");
}

std::size_t readyTokenCount(const SimulatorState &state) {
  std::size_t count = 0;
  for (const ChannelSlot &slot : state.channelSlots)
    count += slot.ready.size();
  return count;
}

Token i32(std::uint32_t value) {
  return take(tokenFromBitPattern(llvm::APInt(32, value),
                                  mlir::IntegerType::get(&context(), 32)));
}

void selectiveProbeIsExactAndNonMutating(
    dataflow::CanonicalDataflowArtifact &artifact) {
  dataflow::GraphOp graph =
      findGraph(artifact.module(), dataflow::OperationSchemaId::DataflowMux);
  PreparedGraphExecution execution = prepare(artifact.module(), graph);
  ActorExecutionPlan &mux =
      findActor(execution, dataflow::OperationSchemaId::DataflowMux);
  SimulatorState state;
  initializeRunState(state, execution);
  state.currentActorPlan = &mux;

  mlir::Block &entry = graph.getBody().front();
  seedBlockArgument(state, entry.getArgument(1),
                    indexToken(llvm::APInt(64, 2)));
  seedBlockArgument(state, entry.getArgument(2), i32(11));
  const std::size_t beforeBlocked = readyTokenCount(state);
  auto blocked = take(probeActorTransition(mux, state));
  require(!blocked && readyTokenCount(state) == beforeBlocked &&
              state.diagnostics.empty(),
          "probe consumed an unselected lane or mutated blocked state");

  seedBlockArgument(state, entry.getArgument(4), i32(33));
  const std::size_t beforeReady = readyTokenCount(state);
  auto selected = take(probeActorTransition(mux, state));
  require(selected && *selected == 2 && readyTokenCount(state) == beforeReady &&
              state.diagnostics.empty(),
          "probe did not select the exact mux lane without mutation");
}

void statefulProbeTracksSchemaCasesWithoutMutation(
    dataflow::CanonicalDataflowArtifact &artifact) {
  dataflow::GraphOp graph =
      findGraph(artifact.module(), dataflow::OperationSchemaId::DataflowStream);
  PreparedGraphExecution execution = prepare(artifact.module(), graph);
  ActorExecutionPlan &stream =
      findActor(execution, dataflow::OperationSchemaId::DataflowStream);
  SimulatorState state;
  initializeRunState(state, execution);
  state.currentActorPlan = &stream;

  mlir::Block &entry = graph.getBody().front();
  seedBlockArgument(state, entry.getArgument(1), i32(0));
  seedBlockArgument(state, entry.getArgument(2), i32(2));
  seedBlockArgument(state, entry.getArgument(3), i32(1));
  const std::size_t beforeStart = readyTokenCount(state);
  auto start = take(probeActorTransition(stream, state));
  require(start && *start == 0 && readyTokenCount(state) == beforeStart &&
              state.streamStates.empty(),
          "stream probe did not select StartTrue without creating state");

  require(fireActorOperation(stream, state),
          "stream provider did not commit the probed transition");
  const auto current = state.streamStates.find(stream.operation);
  require(current != state.streamStates.end() &&
              current->second.mode == dataflow::semantics::StreamMode::Running,
          "stream provider did not enter its running state");
  const llvm::APInt currentIv = current->second.current;
  const std::size_t beforeContinue = readyTokenCount(state);
  auto continuation = take(probeActorTransition(stream, state));
  require(continuation && *continuation == 2 &&
              readyTokenCount(state) == beforeContinue &&
              state.streamStates.find(stream.operation)->second.current ==
                  currentIv,
          "stream probe did not select ContinueTrue without mutation");
}

} // namespace

int main() {
  auto artifact = program();
  selectiveProbeIsExactAndNonMutating(artifact);
  statefulProbeTracksSchemaCasesWithoutMutation(artifact);
  return EXIT_SUCCESS;
}
