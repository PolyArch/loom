#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Dataflow/IR/DataflowDialect.h"
#include "Dataflow/IR/DataflowOps.h"
#include "Dataflow/Transforms/DataflowRewrite.h"
#include "DataflowRewriteTestSupport.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <array>
#include <cstdlib>
#include <optional>
#include <string>

namespace {

[[noreturn]] void fail(llvm::StringRef message) {
  llvm::errs() << "dataflow sync rewrite: " << message << '\n';
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

mlir::MLIRContext &context() {
  static mlir::MLIRContext *result = [] {
    mlir::DialectRegistry registry;
    registry.insert<dataflow::DataflowDialect>();
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

dataflow::CanonicalDataflowArtifact directProgram() {
  return finalize(R"mlir(
module {
  dataflow.graph private @direct(
      %start: none, %a: i8, %b: i8, %c: i8, %d: i8) -> i8
      attributes {input_segments = array<i32: 4, 0, 0>,
                  result_segments = array<i32: 1, 0, 0>} {
    %wide:4 = dataflow.sync %a, %b, %c, %d
        : (i8, i8, i8, i8) -> (i8, i8, i8, i8)
    %retired:2 = dataflow.sync %start, %wide#2
        : (none, i8) -> (none, i8)
    dataflow.graph.return values(%retired#1 : i8) streams() memories()
        complete(%retired#0 : none)
  }
}
)mlir");
}

dataflow::CanonicalDataflowArtifact twoLiveResultsProgram() {
  return finalize(R"mlir(
module {
  dataflow.graph private @two_live(
      %start: none, %a: i8, %b: i8, %c: i8, %d: i8) -> (i8, i8)
      attributes {input_segments = array<i32: 4, 0, 0>,
                  result_segments = array<i32: 2, 0, 0>} {
    %wide:4 = dataflow.sync %a, %b, %c, %d
        : (i8, i8, i8, i8) -> (i8, i8, i8, i8)
    %first:2 = dataflow.sync %start, %wide#1
        : (none, i8) -> (none, i8)
    %second:2 = dataflow.sync %first#0, %wide#2
        : (none, i8) -> (none, i8)
    dataflow.graph.return values(%first#1, %second#1 : i8, i8)
        streams() memories() complete(%second#0 : none)
  }
}
)mlir");
}

std::optional<dataflow::SyncRendezvousRewrite>
findSyncDecision(const dataflow::CanonicalDataflowArtifact &artifact,
                 dataflow::SyncRendezvousDirection direction) {
  auto decisions =
      take(dataflow::enumerateFixedDataflowRewriteDecisions(artifact));
  std::optional<dataflow::SyncRendezvousRewrite> found;
  for (const dataflow::DataflowRewriteDecision &decision : decisions) {
    const auto *sync = std::get_if<dataflow::SyncRendezvousRewrite>(&decision);
    if (!sync || sync->direction != direction)
      continue;
    if (found)
      fail("fixture has more than one matching sync decision");
    found = *sync;
  }
  return found;
}

void directTreeRoundTripIsExact() {
  auto parent = directProgram();
  auto forward =
      findSyncDecision(parent, dataflow::SyncRendezvousDirection::DirectToTree);
  require(forward.has_value(), "direct rendezvous was not enumerated");
  auto child = take(dataflow::materializeDataflowRewrite(parent, *forward));
  require(child.has_value(), "direct-to-tree produced no child");

  unsigned binarySyncs = 0;
  unsigned wideSyncs = 0;
  child->module().walk([&](dataflow::SyncOp sync) {
    binarySyncs += sync.getInputs().size() == 2;
    wideSyncs += sync.getInputs().size() > 2;
  });
  require(binarySyncs == 4 && wideSyncs == 0,
          "direct-to-tree did not build three binary nodes plus retirement");
  const std::array<loom::sim::DFGRuntimeArg, 4> args = {
      loom::sim::DFGRuntimeArg{0, "1"}, loom::sim::DFGRuntimeArg{1, "2"},
      loom::sim::DFGRuntimeArg{2, "3"}, loom::sim::DFGRuntimeArg{3, "4"}};
  require(take(dataflow::test::simulateOnlyGraph(parent, args)) ==
              take(dataflow::test::simulateOnlyGraph(*child, args)),
          "direct and tree rendezvous changed external observations");

  auto reverse =
      findSyncDecision(*child, dataflow::SyncRendezvousDirection::TreeToDirect);
  require(reverse.has_value(), "canonical tree was not recognized");
  auto restored = take(dataflow::materializeDataflowRewrite(*child, *reverse));
  require(restored.has_value(), "tree-to-direct produced no child");
  require(restored->identity() == parent.identity(),
          "sync inverse did not restore the exact canonical artifact");
}

void multipleLiveResultsAreRejected() {
  auto artifact = twoLiveResultsProgram();
  require(!findSyncDecision(artifact,
                            dataflow::SyncRendezvousDirection::DirectToTree),
          "sync with two externally live results entered the domain");
}

} // namespace

int main() {
  directTreeRoundTripIsExact();
  multipleLiveResultsAreRejected();
  return EXIT_SUCCESS;
}
