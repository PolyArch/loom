#include "ExecutionMatrixFixtureSupport.h"

#include "Dataflow/IR/DataflowDialect.h"
#include "Dataflow/IR/DataflowOps.h"
#include "Dataflow/IR/OperationSchemaCodec.h"
#include "Fabric/IR/FabricDialect.h"
#include "Mapping/IR/MappingDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Parser/Parser.h"

#include <optional>
#include <utility>

namespace loom::system_test {
namespace {

using deployment::test::fail;
using deployment::test::require;

template <typename T> T take(llvm::StringRef test, llvm::Expected<T> value) {
  if (!value)
    fail(test, llvm::toString(value.takeError()));
  return std::move(*value);
}

std::unique_ptr<mlir::MLIRContext> makeContextImpl() {
  mlir::DialectRegistry registry;
  registry.insert<::dataflow::DataflowDialect, ::fabric::FabricDialect,
                  ::mapping::MappingDialect, mlir::arith::ArithDialect,
                  mlir::DLTIDialect, mlir::func::FuncDialect,
                  mlir::LLVM::LLVMDialect, mlir::memref::MemRefDialect>();
  return std::make_unique<mlir::MLIRContext>(
      registry, mlir::MLIRContext::Threading::DISABLED);
}

} // namespace

std::unique_ptr<mlir::MLIRContext> makeContext() { return makeContextImpl(); }

dataflow::CanonicalDataflowArtifact buildCanonicalApplication(
    llvm::StringRef test, mlir::MLIRContext &context, bool paired) {
  if (paired) {
    auto source = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 64>>} {
  dataflow.graph private @project(%start: none) -> ()
      attributes {input_segments = array<i32: 0, 0, 0>,
                  result_segments = array<i32: 0, 0, 0>} {
    %value = dataflow.constant %start {const_value = 7 : i32} : i32
    %sync0:2 = dataflow.sync %start, %value : (none, i32) -> (none, i32)
    %sync1:2 = dataflow.sync %sync0#0, %sync0#1 : (none, i32) -> (none, i32)
    %sync2:2 = dataflow.sync %sync1#0, %sync1#1 : (none, i32) -> (none, i32)
    %sync3:2 = dataflow.sync %sync2#0, %sync2#1 : (none, i32) -> (none, i32)
    %sync4:2 = dataflow.sync %sync3#0, %sync3#1 : (none, i32) -> (none, i32)
    dataflow.graph.return values() streams() memories()
        complete(%sync4#0 : none)
  }
  dataflow.thread private @project_thread domain(#dataflow.thread_domain<dense>)()
      ctrl (%ctrl: none) {
    %done = dataflow.graph.launch @project deps(%ctrl) values()
        stream_inputs() memories() stream_outputs()
        : (none) -> none
    dataflow.thread.yield %done : none
  }
  func.func private @host() {
    %project = dataflow.thread.launch @project_thread()
        : () -> !dataflow.thread_token
    return
  }
}
)mlir",
                                                          &context);
    require(test, static_cast<bool>(source),
            "cannot parse the paired execution-matrix Dataflow program");
    return take(test, dataflow::finalizeCanonicalDataflow(*source));
  }
  auto source = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 64>>} {
  dataflow.graph private @project(%start: none) -> i32
      attributes {input_segments = array<i32: 0, 0, 0>,
                  result_segments = array<i32: 0, 1, 0>} {
    %value = dataflow.constant %start {const_value = 7 : i32} : i32
    %retired:2 = dataflow.sync %start, %value
        : (none, i32) -> (none, i32)
    dataflow.graph.return values() streams(%retired#1 : i32) memories()
        complete(%retired#0 : none)
  }
  dataflow.graph private @attention(%start: none, %input: i32) -> ()
      attributes {input_segments = array<i32: 0, 1, 0>,
                  result_segments = array<i32: 0, 0, 0>} {
    %select = dataflow.constant %start {const_value = false} : i1
    %lane:2 = dataflow.demux %select, %input
        : (i1, i32) -> (i32, i32)
    %retired:2 = dataflow.sync %start, %lane#0
        : (none, i32) -> (none, i32)
    dataflow.graph.return values() streams() memories()
        complete(%retired#0 : none)
  }
  dataflow.graph private @project_peer(%start: none) -> i32
      attributes {input_segments = array<i32: 0, 0, 0>,
                  result_segments = array<i32: 1, 0, 0>} {
    %value = dataflow.constant %start {const_value = 11 : i32} : i32
    %retired:2 = dataflow.sync %start, %value
        : (none, i32) -> (none, i32)
    dataflow.graph.return values(%retired#1 : i32) streams() memories()
        complete(%retired#0 : none)
  }
  dataflow.thread private @a_project domain(#dataflow.thread_domain<dense>)(
      %channel: !dataflow.channel<i32>) ctrl (%ctrl: none) {
    %done = dataflow.graph.launch @project deps(%ctrl) values()
        stream_inputs() memories() stream_outputs(%channel)
        : (none, !dataflow.channel<i32>) -> none
    dataflow.thread.yield %done : none
  }
  dataflow.thread private @b_attention domain(#dataflow.thread_domain<dense>)(
      %channel: !dataflow.channel<i32>) ctrl (%ctrl: none) {
    %done = dataflow.graph.launch @attention deps(%ctrl) values()
        stream_inputs(%channel source_map affine_map<() -> ()>) memories()
        stream_outputs() : (none, !dataflow.channel<i32>) -> none
    dataflow.thread.yield %done : none
  }
  dataflow.thread private @c_stats domain(#dataflow.thread_domain<dense>)(
      %channel: !dataflow.channel<i32>) ctrl (%ctrl: none) {
    %done = dataflow.graph.launch @attention deps(%ctrl) values()
        stream_inputs(%channel source_map affine_map<() -> ()>) memories()
        stream_outputs() : (none, !dataflow.channel<i32>) -> none
    dataflow.thread.yield %done : none
  }
  dataflow.thread private @d_project_peer
      domain(#dataflow.thread_domain<dense>)() ctrl (%ctrl: none) {
    %value, %done = dataflow.graph.launch @project_peer deps(%ctrl) values()
        stream_inputs() memories() stream_outputs()
        : (none) -> (i32, none)
    dataflow.thread.yield %done : none
  }
  func.func private @host() {
    %channel = dataflow.channel.create : !dataflow.channel<i32>
    %attention = dataflow.thread.launch @b_attention(%channel)
        : (!dataflow.channel<i32>) -> !dataflow.thread_token
    %peer = dataflow.thread.launch @d_project_peer()
        : () -> !dataflow.thread_token
    %stats = dataflow.thread.launch @c_stats(%channel)
        : (!dataflow.channel<i32>) -> !dataflow.thread_token
    %project = dataflow.thread.launch @a_project(%channel)
        : (!dataflow.channel<i32>) -> !dataflow.thread_token
    return
  }
}
)mlir",
                                                        &context);
  require(test, static_cast<bool>(source),
          "cannot parse the execution-matrix Dataflow program");
  return take(test, dataflow::finalizeCanonicalDataflow(*source));
}

namespace {

dataflow::RootedGraphLaunchRef projectLaunch(
    llvm::StringRef test, const dataflow::CanonicalDataflowProgramView &view,
    bool paired) {
  std::optional<dataflow::RootedGraphLaunchRef> result;
  view.forEachRootedGraphLaunch([&](dataflow::RootedGraphLaunchRef launch) {
    if (paired) {
      require(test, !result.has_value(),
              "paired program has more than one rooted graph launch");
      result = launch;
      return;
    }
    const auto graphRef = take(test, view.resolve(launch));
    const auto graph = take(test, view.resolve(graphRef));
    const auto resultSegments =
        llvm::cast<dataflow::GraphOp>(graph.op).getResultSegmentSizes();
    if (resultSegments[1] != 1)
      return;
    require(test, !result.has_value(),
            "more than one rooted graph has a stream output");
    result = launch;
  });
  require(test, result.has_value(), "no project graph launch was selected");
  return *result;
}

} // namespace

std::pair<ArtifactRootReference, ArtifactRootReference> publishSpatialInputs(
    llvm::StringRef test, const dataflow::CanonicalDataflowArtifact &dataflow,
    ArtifactStore &artifacts, bool paired) {
  const auto view = take(test, dataflow.view());
  sim::SpatialSimulationWorkload workloadDraft{projectLaunch(test, view, paired)};
  if (!paired)
    workloadDraft.observableContract.streamOutputs = {0};
  auto workload =
      take(test, sim::finalizeSimulationWorkload(workloadDraft, view));
  sim::SpatialSimulationRuntimeInputDraft runtimeDraft{workload.identity()};
  auto runtime = take(
      test, sim::finalizeSimulationRuntimeInput(runtimeDraft, workload, view));
  return {take(test, sim::publishSimulationWorkload(workload, artifacts)),
          take(test, sim::publishSimulationRuntimeInput(runtime, artifacts))};
}

} // namespace loom::system_test
