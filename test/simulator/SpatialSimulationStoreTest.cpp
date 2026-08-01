#include "Simulator/SimulationArtifacts.h"

#include "Common/ArtifactStore.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Dataflow/IR/DataflowDialect.h"

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
  llvm::errs() << "SpatialSimulationStoreTest: " << message << "\n";
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
            llvm::sys::fs::createUniqueDirectory("loom-spatial-store", path_))
      fail(error.message());
  }

  ~TemporaryDirectory() {
    if (std::error_code error = llvm::sys::fs::remove_directories(path_))
      llvm::errs() << "temporary directory cleanup failed: " << error.message()
                   << "\n";
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
    fail("failed to parse the store fixture");
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

void storedSpatialInputsRecoverTheirSoleDataflowOwner() {
  TemporaryDirectory directory;
  const loom::ArtifactStore store(directory.path());
  auto dataflow = program();
  auto view = take(dataflow.view());
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

  const auto dataflowRef =
      take(dataflow::publishCanonicalDataflow(dataflow, store));
  const auto workloadRef =
      take(loom::sim::publishSimulationWorkload(workload, store));
  const auto runtimeRef =
      take(loom::sim::publishSimulationRuntimeInput(runtime, store));
  auto imported = take(
      loom::sim::importSpatialSimulationInputs(workloadRef, runtimeRef, store));

  require(imported.dataflow.identity() == dataflowRef.artifact,
          "import recovered a different Dataflow owner");
  require(imported.workload.identity() == workload.identity(),
          "workload identity changed during strict import");
  require(imported.runtimeInput.identity() == runtime.identity(),
          "runtime-input identity changed during strict import");
  auto importedView = take(imported.dataflow.view());
  require(imported.workload.spatial() &&
              imported.workload.spatial()->launchRef ==
                  onlyLaunch(importedView),
          "workload launch did not resolve through the recovered owner");
}

} // namespace

int main() {
  storedSpatialInputsRecoverTheirSoleDataflowOwner();
  return EXIT_SUCCESS;
}
