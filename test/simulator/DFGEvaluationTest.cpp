#include "Evaluation/Models/DfgSimulation.h"

#include "Common/ArtifactStore.h"
#include "Common/ResolvedConfig.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Dataflow/IR/DataflowDialect.h"
#include "Evaluation/Evidence.h"
#include "Simulator/SimulationArtifacts.h"
#include "Simulator/SimulationExecution.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <string>
#include <utility>

namespace {

[[noreturn]] void fail(llvm::StringRef message) {
  llvm::errs() << "DFGEvaluationTest: " << message << '\n';
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

class TemporaryDirectory {
public:
  TemporaryDirectory() {
    if (std::error_code error =
            llvm::sys::fs::createUniqueDirectory("loom-dfg-evaluation", path_))
      fail(error.message());
  }

  ~TemporaryDirectory() {
    if (std::error_code error = llvm::sys::fs::remove_directories(path_))
      llvm::errs() << "temporary directory cleanup failed: " << error.message()
                   << '\n';
  }

  llvm::StringRef path() const { return path_; }

private:
  llvm::SmallString<128> path_;
};

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
    fail("failed to parse the DFG Evaluation fixture");
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

void retiredExecutionBecomesEvidenceOutput() {
  TemporaryDirectory directory;
  const loom::ArtifactStore store(directory.path());
  dataflow::CanonicalDataflowArtifact dataflow = program();
  dataflow::CanonicalDataflowProgramView view = take(dataflow.view());
  const loom::ArtifactRootReference dataflowRef =
      take(dataflow::publishCanonicalDataflow(dataflow, store));

  loom::sim::SpatialSimulationWorkload workloadDraft{onlyLaunch(view)};
  workloadDraft.valueInputPlan = {loom::sim::RuntimeValueInput{},
                                  loom::sim::RuntimeValueInput{}};
  workloadDraft.observableContract.valueResults = {0};
  loom::sim::CanonicalSimulationWorkload workload =
      take(loom::sim::finalizeSimulationWorkload(workloadDraft, view));

  loom::sim::SpatialSimulationRuntimeInputDraft runtimeDraft{
      workload.identity()};
  runtimeDraft.runtimeValues = {{0, value(7)}, {1, value(9)}};
  loom::sim::CanonicalSimulationRuntimeInput runtime = take(
      loom::sim::finalizeSimulationRuntimeInput(runtimeDraft, workload, view));
  const loom::ArtifactRootReference workloadRef =
      take(loom::sim::publishSimulationWorkload(workload, store));
  const loom::ArtifactRootReference runtimeRef =
      take(loom::sim::publishSimulationRuntimeInput(runtime, store));

  auto prepared = take(loom::evaluation::models::prepareDfgSimulationEvaluation(
      dataflowRef, workloadRef, runtimeRef, loom::defaultResolvedConfig(),
      store));
  auto limited = take(loom::evaluation::models::evaluateDfgSimulation(
      prepared, {1, std::nullopt}, store));
  require(limited.outcomeKind() ==
                  loom::evaluation::EvidenceOutcomeKind::CancelledOrTimeout &&
              limited.outputBindings().size() == 1 &&
              limited.outputBindings().front().artifacts.empty(),
          "wavefront-limited attempt retained a fabricated execution");

  auto evidence = take(loom::evaluation::models::evaluateDfgSimulation(
      prepared, {64, std::nullopt}, store));
  require(evidence.outcomeKind() ==
              loom::evaluation::EvidenceOutcomeKind::Completed,
          "retired DFG run did not produce Completed Evidence");
  require(evidence.outputBindings().size() == 1 &&
              evidence.outputBindings().front().artifacts.size() == 1,
          "Completed Evidence did not bind one execution output");

  const loom::ArtifactRootReference executionRef =
      evidence.outputBindings().front().artifacts.front();
  auto execution = take(loom::sim::importSimulationExecution(
      executionRef, prepared.resolution, store));
  require(execution.request() ==
              loom::evaluation::evaluationRequestReference(prepared.request),
          "execution is not coupled to the exact EvaluationRequest");
  require(
      std::holds_alternative<loom::sim::RetiredExecution>(execution.terminal()),
      "successful DFG run did not retain a Retired terminal");

  require(execution.functionalObservations().valueResults.size() == 1,
          "execution value observations are not total");
  const auto *published = std::get_if<loom::sim::PublishedValueResult>(
      &execution.functionalObservations().valueResults.front());
  require(published && published->value.lanes.size() == 1 &&
              published->value.lanes.front().bits.getZExtValue() == 16,
          "execution did not preserve the real DFG result");

  const auto &progress = execution.progressObservations();
  require(progress.launchAccepted.referenceCycle.isZero() &&
              progress.graphRetirementVisible.has_value() &&
              progress.graphRetirementVisible->referenceCycle.numerator() > 0 &&
              progress.graphRetirementVisible->referenceCycle.denominator() ==
                  1 &&
              progress.terminalObserved.referenceCycle ==
                  progress.graphRetirementVisible->referenceCycle,
          "execution progress is not derived from DFG retirement");

  const auto *completed =
      std::get_if<loom::evaluation::CompletedEvidence>(&evidence.outcome());
  require(completed && completed->metricResults.size() == 1,
          "DFG Evidence did not contain its cycle-count result");
  const auto *point = std::get_if<loom::evaluation::PointObservation>(
      &completed->metricResults.front().observation);
  const auto *cycles =
      point ? std::get_if<loom::evaluation::IntegerValue>(&point->value)
            : nullptr;
  require(
      cycles &&
          cycles->value() ==
              static_cast<std::int64_t>(
                  progress.graphRetirementVisible->referenceCycle.numerator()),
      "cycle-count Evidence diverged from execution progress");

  const loom::ArtifactRootReference evidenceRef =
      take(loom::evaluation::publishEvaluationEvidence(evidence, store));
  auto importedEvidence = take(loom::evaluation::importEvaluationEvidence(
      evidenceRef, prepared.resolution, store));
  require(importedEvidence.outputBindings() == evidence.outputBindings(),
          "Evidence strict import changed the execution binding");
}

} // namespace

int main() {
  retiredExecutionBecomesEvidenceOutput();
  return EXIT_SUCCESS;
}
