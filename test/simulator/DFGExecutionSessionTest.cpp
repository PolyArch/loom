#include "Simulator/DFGSimulator.h"

#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Dataflow/IR/DataflowDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/APInt.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <cstdlib>
#include <string>
#include <utility>

namespace {

[[noreturn]] void fail(llvm::StringRef message) {
  llvm::errs() << "DFGExecutionSessionTest: " << message << "\n";
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
                    mlir::func::FuncDialect>();
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
  dataflow.graph private @add(%ctrl: none, %lhs: i32, %rhs: i32) -> i32
      attributes {
        input_segments = array<i32: 2, 0, 0>,
        result_segments = array<i32: 1, 0, 0>
      } {
    %sum = arith.addi %lhs, %rhs : i32
    %published:2 = dataflow.sync %ctrl, %sum
        : (none, i32) -> (none, i32)
    dataflow.graph.return values(%published#1 : i32) streams() memories()
        complete(%published#0 : none)
  }
  dataflow.thread private @worker domain(#dataflow.thread_domain<dense>)()
      ctrl (%ctrl: none) {
    %lhs = arith.constant 7 : i32
    %rhs = arith.constant 9 : i32
    %value, %done = dataflow.graph.launch @add deps(%ctrl)
        values(%lhs, %rhs) stream_inputs() memories() stream_outputs()
        : (none, i32, i32) -> (i32, none)
    dataflow.thread.yield %done : none
  }
  func.func private @host() {
    %thread = dataflow.thread.launch @worker()
        : () -> !dataflow.thread_token
    return
  }
}
)mlir",
                                                        &context());
  if (!module)
    fail("failed to parse the session fixture");
  return take(dataflow::finalizeCanonicalDataflow(module.get()));
}

dataflow::CanonicalDataflowArtifact nonRetiringProgram() {
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  dataflow.graph private @detached_actor(%start: none, %input: i32) -> (i32)
      attributes {
        input_segments = array<i32: 1, 0, 0>,
        result_segments = array<i32: 1, 0, 0>
      } {
    %published:2 = dataflow.sync %start, %input
        : (none, i32) -> (none, i32)
    %first = arith.addi %input, %input : i32
    %detached = arith.addi %first, %input : i32
    dataflow.graph.return values(%published#1 : i32) streams() memories()
        complete(%published#0 : none)
  }
  dataflow.thread private @worker domain(#dataflow.thread_domain<dense>)()
      ctrl (%ctrl: none) {
    %input = arith.constant 7 : i32
    %value, %done = dataflow.graph.launch @detached_actor deps(%ctrl)
        values(%input) stream_inputs() memories() stream_outputs()
        : (none, i32) -> (i32, none)
    dataflow.thread.yield %done : none
  }
  func.func private @host() {
    %thread = dataflow.thread.launch @worker()
        : () -> !dataflow.thread_token
    return
  }
}
)mlir",
                                                        &context());
  if (!module)
    fail("failed to parse the non-retiring fixture");
  return take(dataflow::finalizeCanonicalDataflow(module.get()));
}

dataflow::RootedGraphLaunchRef
onlyLaunch(const dataflow::CanonicalDataflowProgramView &view) {
  require(view.rootThreadLaunches().size() == 1 &&
              view.staticGraphLaunches().size() == 1,
          "fixture must contain one rooted graph launch");
  return {view.rootThreadLaunches().front().ref,
          view.staticGraphLaunches().front().ref};
}

loom::sim::CanonicalValueSequence value(std::uint32_t bits) {
  return {1, {loom::sim::SemanticLane::defined(llvm::APInt(32, bits))}};
}

std::uint64_t observedI32(const loom::sim::RetiredDFGSimulation &execution) {
  require(execution.observations.valueResults.size() == 1,
          "execution did not return one selected value");
  const auto *published = std::get_if<loom::sim::PublishedValueResult>(
      &execution.observations.valueResults.front());
  require(published && published->value.tokenCount == 1 &&
              published->value.lanes.size() == 1 &&
              published->value.lanes.front().state ==
                  loom::sim::SemanticState::Defined,
          "execution did not publish one defined scalar");
  return published->value.lanes.front().bits.getZExtValue();
}

void incrementalExecutionMatchesRunToCompletion() {
  dataflow::CanonicalDataflowArtifact artifact = program();
  auto view = take(artifact.view());
  const dataflow::RootedGraphLaunchRef launch = onlyLaunch(view);

  loom::sim::SpatialSimulationWorkload workloadDraft{launch};
  workloadDraft.valueInputPlan = {loom::sim::RuntimeValueInput{},
                                  loom::sim::RuntimeValueInput{}};
  workloadDraft.observableContract.valueResults = {0};
  auto workload =
      take(loom::sim::finalizeSimulationWorkload(workloadDraft, view));

  loom::sim::SpatialSimulationRuntimeInputDraft runtimeDraft{
      workload.identity()};
  runtimeDraft.runtimeValues = {{0, value(7)}, {1, value(9)}};
  auto runtime = take(
      loom::sim::finalizeSimulationRuntimeInput(runtimeDraft, workload, view));
  auto prepared = take(loom::sim::prepareDfgExecution(artifact, launch));

  auto session =
      take(loom::sim::startDfgExecutionSession(prepared, workload, runtime));
  require(session.state() == loom::sim::DfgExecutionSessionState::Runnable,
          "a fresh session is not runnable");
  require(session.wavefrontSteps() == 0,
          "a fresh session already consumed a wavefront");

  auto state = take(session.advance(1));
  require(session.wavefrontSteps() == 1,
          "one advance did not consume exactly one wavefront");
  while (state == loom::sim::DfgExecutionSessionState::Runnable)
    state = take(session.advance(1));
  if (state != loom::sim::DfgExecutionSessionState::Retired)
    fail(("the incrementally driven graph stopped after " +
          std::to_string(session.wavefrontSteps()) + " wavefronts")
             .c_str());

  const std::uint64_t retiredSteps = session.wavefrontSteps();
  require(take(session.advance(1)) ==
                  loom::sim::DfgExecutionSessionState::Retired &&
              session.wavefrontSteps() == retiredSteps,
          "advancing a retired session changed its execution");

  auto incremental = take(session.takeRetiredSimulation());
  auto complete =
      take(loom::sim::simulateRetiredDfgWorkload(prepared, workload, runtime));
  require(incremental.report.wavefrontSteps == complete.report.wavefrontSteps &&
              incremental.report.eventCount == complete.report.eventCount &&
              incremental.report.operationFireCounts ==
                  complete.report.operationFireCounts &&
              incremental.report.finalOutputs == complete.report.finalOutputs &&
              observedI32(incremental) == 16 && observedI32(complete) == 16,
          "incremental and run-to-completion execution diverged");
}

void typedRetiredApiPreservesRetirementFailure() {
  dataflow::CanonicalDataflowArtifact artifact = nonRetiringProgram();
  auto view = take(artifact.view());
  const dataflow::RootedGraphLaunchRef launch = onlyLaunch(view);

  loom::sim::SpatialSimulationWorkload workloadDraft{launch};
  workloadDraft.valueInputPlan = {loom::sim::RuntimeValueInput{}};
  workloadDraft.observableContract.valueResults = {0};
  auto workload =
      take(loom::sim::finalizeSimulationWorkload(workloadDraft, view));
  loom::sim::SpatialSimulationRuntimeInputDraft runtimeDraft{
      workload.identity()};
  runtimeDraft.runtimeValues = {{0, value(7)}};
  auto runtime = take(
      loom::sim::finalizeSimulationRuntimeInput(runtimeDraft, workload, view));

  auto execution =
      loom::sim::simulateRetiredDfgWorkload(artifact, workload, runtime);
  if (execution)
    fail("detached actor unexpectedly satisfied graph retirement");
  bool sawRetirementFailure = false;
  llvm::handleAllErrors(
      execution.takeError(),
      [&](const loom::sim::NonRetiredDFGExecutionError &failure) {
        sawRetirementFailure = failure.report().status == "invalid";
      },
      [&](const llvm::ErrorInfoBase &failure) { fail(failure.message()); });
  require(sawRetirementFailure,
          "typed retired API erased the retirement failure class");
}

} // namespace

int main() {
  incrementalExecutionMatchesRunToCompletion();
  typedRetiredApiPreservesRetirementFailure();
  return EXIT_SUCCESS;
}
