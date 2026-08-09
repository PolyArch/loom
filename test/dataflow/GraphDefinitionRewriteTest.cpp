#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Dataflow/IR/DataflowDialect.h"
#include "Dataflow/Transforms/DataflowRewrite.h"
#include "DataflowRewriteTestSupport.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <optional>

namespace {

[[noreturn]] void fail(llvm::StringRef message) {
  llvm::errs() << "dataflow graph-definition rewrite: " << message << '\n';
  std::exit(EXIT_FAILURE);
}

void require(bool condition, llvm::StringRef message) {
  if (!condition)
    fail(message);
}

template <typename T> T take(llvm::Expected<T> value) {
  if (!value)
    fail(llvm::toString(value.takeError()));
  return std::move(*value);
}

template <typename T> bool isRejected(llvm::Expected<T> value) {
  if (value)
    return false;
  llvm::consumeError(value.takeError());
  return true;
}

mlir::MLIRContext &context() {
  static mlir::MLIRContext *result = [] {
    mlir::DialectRegistry registry;
    registry.insert<dataflow::DataflowDialect, mlir::arith::ArithDialect,
                    mlir::func::FuncDialect>();
    auto *created =
        new mlir::MLIRContext(registry, mlir::MLIRContext::Threading::DISABLED);
    created->loadAllAvailableDialects();
    return created;
  }();
  return *result;
}

dataflow::CanonicalDataflowArtifact finalize(llvm::StringRef source) {
  auto module = mlir::parseSourceString<mlir::ModuleOp>(source, &context());
  if (!module)
    fail("cannot parse fixture");
  return take(dataflow::finalizeCanonicalDataflow(module.get()));
}

dataflow::CanonicalDataflowArtifact repeatedLaunchProgram() {
  return finalize(R"mlir(
module {
  dataflow.graph private @work(%start: none) -> i8
      attributes {input_segments = array<i32: 0, 0, 0>,
                  result_segments = array<i32: 1, 0, 0>} {
    %value = dataflow.constant %start {const_value = 7 : i8} : i8
    %retired:2 = dataflow.sync %start, %value
        : (none, i8) -> (none, i8)
    dataflow.graph.return values(%retired#1 : i8) streams() memories()
        complete(%retired#0 : none)
  }
  dataflow.thread private @worker domain(#dataflow.thread_domain<dense>)()
      ctrl (%ctrl: none) {
    %first_value, %first = dataflow.graph.launch @work deps(%ctrl) values()
        stream_inputs() memories() stream_outputs() : (none) -> (i8, none)
    %second_value, %second = dataflow.graph.launch @work deps(%first) values()
        stream_inputs() memories() stream_outputs() : (none) -> (i8, none)
    dataflow.thread.yield %second : none
  }
  func.func private @host() {
    %token = dataflow.thread.launch @worker()
        : () -> !dataflow.thread_token
    return
  }
}
)mlir");
}

dataflow::CanonicalDataflowArtifact nonIsomorphicProgram() {
  return finalize(R"mlir(
module {
  dataflow.graph private @add(%start: none, %x: i8) -> i8
      attributes {input_segments = array<i32: 1, 0, 0>,
                  result_segments = array<i32: 1, 0, 0>} {
    %value = arith.addi %x, %x : i8
    %retired:2 = dataflow.sync %start, %value
        : (none, i8) -> (none, i8)
    dataflow.graph.return values(%retired#1 : i8) streams() memories()
        complete(%retired#0 : none)
  }
  dataflow.graph private @sub(%start: none, %x: i8) -> i8
      attributes {input_segments = array<i32: 1, 0, 0>,
                  result_segments = array<i32: 1, 0, 0>} {
    %value = arith.subi %x, %x : i8
    %retired:2 = dataflow.sync %start, %value
        : (none, i8) -> (none, i8)
    dataflow.graph.return values(%retired#1 : i8) streams() memories()
        complete(%retired#0 : none)
  }
  dataflow.thread private @worker domain(#dataflow.thread_domain<dense>)(
      %x: i8) ctrl (%ctrl: none) {
    %a, %first = dataflow.graph.launch @add deps(%ctrl) values(%x)
        stream_inputs() memories() stream_outputs()
        : (none, i8) -> (i8, none)
    %b, %second = dataflow.graph.launch @sub deps(%first) values(%a)
        stream_inputs() memories() stream_outputs()
        : (none, i8) -> (i8, none)
    dataflow.thread.yield %second : none
  }
  func.func private @host(%x: i8) {
    %token = dataflow.thread.launch @worker(%x)
        : (i8) -> !dataflow.thread_token
    return
  }
}
)mlir");
}

dataflow::CanonicalDataflowArtifact memoryFormalProgram() {
  return finalize(R"mlir(
module {
  dataflow.graph private @first(%start: none, %mem: memref<4xi8>) -> ()
      attributes {input_segments = array<i32: 0, 0, 1>,
                  result_segments = array<i32: 0, 0, 0>} {
    dataflow.graph.return values() streams() memories()
        complete(%start : none)
  }
  dataflow.graph private @second(%start: none, %mem: memref<4xi8>) -> ()
      attributes {input_segments = array<i32: 0, 0, 1>,
                  result_segments = array<i32: 0, 0, 0>} {
    dataflow.graph.return values() streams() memories()
        complete(%start : none)
  }
  dataflow.thread private @worker domain(#dataflow.thread_domain<dense>)(
      %mem: memref<4xi8>) ctrl (%ctrl: none) {
    %first = dataflow.graph.launch @first deps(%ctrl) values()
        stream_inputs() memories(%mem) stream_outputs()
        : (none, memref<4xi8>) -> none
    %second = dataflow.graph.launch @second deps(%first) values()
        stream_inputs() memories(%mem) stream_outputs()
        : (none, memref<4xi8>) -> none
    dataflow.thread.yield %second : none
  }
  func.func private @host(%mem: memref<4xi8>) {
    %token = dataflow.thread.launch @worker(%mem)
        : (memref<4xi8>) -> !dataflow.thread_token
    return
  }
}
)mlir");
}

std::optional<dataflow::GraphDefinitionSplitRewrite>
findSplit(const dataflow::CanonicalDataflowArtifact &artifact) {
  auto decisions =
      take(dataflow::enumerateFixedDataflowRewriteDecisions(artifact));
  std::optional<dataflow::GraphDefinitionSplitRewrite> found;
  for (const dataflow::DataflowRewriteDecision &decision : decisions) {
    const auto *split =
        std::get_if<dataflow::GraphDefinitionSplitRewrite>(&decision);
    if (!split)
      continue;
    if (found)
      fail("fixture has more than one split decision");
    found = *split;
  }
  return found;
}

std::optional<dataflow::GraphDefinitionMergeRewrite>
findMerge(const dataflow::CanonicalDataflowArtifact &artifact) {
  auto decisions =
      take(dataflow::enumerateFixedDataflowRewriteDecisions(artifact));
  std::optional<dataflow::GraphDefinitionMergeRewrite> found;
  for (const dataflow::DataflowRewriteDecision &decision : decisions) {
    const auto *merge =
        std::get_if<dataflow::GraphDefinitionMergeRewrite>(&decision);
    if (!merge)
      continue;
    if (found)
      fail("fixture has more than one merge decision");
    found = *merge;
  }
  return found;
}

void splitMergeRoundTripIsExact() {
  auto parent = repeatedLaunchProgram();
  auto split = findSplit(parent);
  require(split && split->launches.size() == 1,
          "normalized bipartition was not enumerated");

  auto view = take(parent.view());
  require(view.staticGraphLaunches().size() == 2,
          "fixture does not have two static launches");
  auto noncanonical = *split;
  noncanonical.launches = {view.staticGraphLaunches()[0].ref.entity ==
                                   split->launches.front()
                               ? view.staticGraphLaunches()[1].ref.entity
                               : view.staticGraphLaunches()[0].ref.entity};
  require(
      isRejected(dataflow::materializeDataflowRewrite(parent, noncanonical)),
      "noncanonical bipartition side was accepted");

  auto child = take(dataflow::materializeDataflowRewrite(parent, *split));
  require(child.has_value(), "split produced no child");
  auto childView = take(child->view());
  require(childView.graphs().size() == 2 &&
              childView.staticGraphLaunches().size() == 2 &&
              childView.staticGraphLaunches()[0].callee !=
                  childView.staticGraphLaunches()[1].callee,
          "split did not retain two launches over two definitions");
  auto parentOutcome =
      take(dataflow::test::simulateGraph(parent, view.graphs().front().ref));
  for (const dataflow::CanonicalGraphView &graph : childView.graphs())
    require(take(dataflow::test::simulateGraph(*child, graph.ref)) ==
                parentOutcome,
            "split graph definition changed external observations");

  auto merge = findMerge(*child);
  require(merge.has_value(), "alpha-isomorphic definitions were not merged");
  auto restored = take(dataflow::materializeDataflowRewrite(*child, *merge));
  require(restored && restored->identity() == parent.identity(),
          "merge did not restore the exact parent artifact");
}

void nonIsomorphicGraphsAreRejected() {
  auto artifact = nonIsomorphicProgram();
  require(!findMerge(artifact),
          "non-alpha-isomorphic graph definitions entered the merge domain");
}

void memoryFormalGraphsAreMergeable() {
  auto parent = memoryFormalProgram();
  auto merge = findMerge(parent);
  require(merge.has_value(),
          "memory-formal graphs were not recognized as alpha-isomorphic");

  auto child = take(dataflow::materializeDataflowRewrite(parent, *merge));
  require(child.has_value(), "memory-formal merge produced no child");
  auto view = take(child->view());
  require(view.graphs().size() == 1 && view.staticGraphLaunches().size() == 2 &&
              view.staticGraphLaunches()[0].callee ==
                  view.staticGraphLaunches()[1].callee,
          "memory-formal merge did not share one graph definition");
}

} // namespace

int main() {
  splitMergeRoundTripIsExact();
  nonIsomorphicGraphsAreRejected();
  memoryFormalGraphsAreMergeable();
  return EXIT_SUCCESS;
}
