#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Dataflow/IR/DataflowDialect.h"
#include "Dataflow/Transforms/DataflowRewrite.h"
#include "DataflowRewriteTestSupport.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <array>
#include <cstdlib>
#include <optional>

namespace {

[[noreturn]] void fail(llvm::StringRef message) {
  llvm::errs() << "dataflow fanout rewrite: " << message << '\n';
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
                    mlir::LLVM::LLVMDialect>();
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

dataflow::CanonicalDataflowArtifact fanoutProgram() {
  return finalize(R"mlir(
module {
  dataflow.graph private @fanout(
      %start: none, %a: i8, %b: i8) -> (i8, i8, i8)
      attributes {input_segments = array<i32: 2, 0, 0>,
                  result_segments = array<i32: 3, 0, 0>} {
    %sum = arith.addi %a, %b : i8
    %left = arith.muli %sum, %a : i8
    %middle = arith.subi %sum, %b : i8
    %right = arith.xori %sum, %a : i8
    %retired:4 = dataflow.sync %start, %left, %middle, %right
        : (none, i8, i8, i8) -> (none, i8, i8, i8)
    dataflow.graph.return
        values(%retired#1, %retired#2, %retired#3 : i8, i8, i8)
        streams() memories() complete(%retired#0 : none)
  }
}
)mlir");
}

dataflow::CanonicalDataflowArtifact freezeFanoutProgram() {
  return finalize(R"mlir(
module {
  dataflow.graph private @freeze_fanout(
      %start: none, %a: i8, %b: i8) -> (i8, i8)
      attributes {input_segments = array<i32: 2, 0, 0>,
                  result_segments = array<i32: 2, 0, 0>} {
    %selected = llvm.freeze %a : i8
    %left = arith.addi %selected, %b : i8
    %right = arith.subi %selected, %b : i8
    %retired:3 = dataflow.sync %start, %left, %right
        : (none, i8, i8) -> (none, i8, i8)
    dataflow.graph.return values(%retired#1, %retired#2 : i8, i8)
        streams() memories() complete(%retired#0 : none)
  }
}
)mlir");
}

dataflow::CanonicalDataflowArtifact selectorFanoutProgram() {
  return finalize(R"mlir(
module {
  dataflow.graph private @selector_fanout(
      %start: none, %lhs: i1, %rhs: i1) -> ()
      attributes {input_segments = array<i32: 2, 0, 0>,
                  result_segments = array<i32: 0, 0, 0>} {
    %phase = arith.andi %lhs, %rhs : i1
    %control = dataflow.carry %phase, %start, %body : none
    %lanes:2 = dataflow.demux %phase, %control
        : (i1, none) -> (none, none)
    %body = dataflow.sync %lanes#1 : (none) -> none
    dataflow.graph.return values() streams() memories()
        complete(%lanes#0 : none)
  }
}
)mlir");
}

dataflow::CanonicalDataflowArtifact layeredSelectorFanoutProgram() {
  return finalize(R"mlir(
module {
  dataflow.graph private @layered_selector_fanout(
      %start: none, %lhs: i1, %rhs: i1) -> i1
      attributes {input_segments = array<i32: 2, 0, 0>,
                  result_segments = array<i32: 1, 0, 0>} {
    %source = arith.xori %lhs, %rhs : i1
    %phase = arith.andi %source, %rhs : i1
    %control = dataflow.carry %phase, %start, %body : none
    %lanes:2 = dataflow.demux %phase, %control
        : (i1, none) -> (none, none)
    %body = dataflow.sync %lanes#1 : (none) -> none
    %retired:2 = dataflow.sync %lanes#0, %source
        : (none, i1) -> (none, i1)
    dataflow.graph.return values(%retired#1 : i1) streams() memories()
        complete(%retired#0 : none)
  }
}
)mlir");
}

template <typename Op>
dataflow::ActorId actorId(const dataflow::CanonicalDataflowArtifact &artifact) {
  const auto &view = artifact.view();
  for (const dataflow::CanonicalActorView &actor : view.actors())
    if (llvm::isa<Op>(actor.op))
      return actor.ref.entity;
  fail("fixture does not contain the requested actor");
}

std::optional<dataflow::PureComputeFanoutReplicateRewrite>
findReplicate(const dataflow::CanonicalDataflowArtifact &artifact) {
  auto decisions =
      take(dataflow::enumerateFixedDataflowRewriteDecisions(artifact));
  std::optional<dataflow::PureComputeFanoutReplicateRewrite> found;
  for (const dataflow::DataflowRewriteDecision &decision : decisions) {
    const auto *replicate =
        std::get_if<dataflow::PureComputeFanoutReplicateRewrite>(&decision);
    if (!replicate)
      continue;
    if (found)
      fail("fixture has more than one replication decision");
    found = *replicate;
  }
  return found;
}

std::optional<dataflow::PureComputeFanoutFactorRewrite>
findFactor(const dataflow::CanonicalDataflowArtifact &artifact) {
  auto decisions =
      take(dataflow::enumerateFixedDataflowRewriteDecisions(artifact));
  std::optional<dataflow::PureComputeFanoutFactorRewrite> found;
  for (const dataflow::DataflowRewriteDecision &decision : decisions) {
    const auto *factor =
        std::get_if<dataflow::PureComputeFanoutFactorRewrite>(&decision);
    if (!factor)
      continue;
    if (found)
      fail("fixture has more than one factor decision");
    found = *factor;
  }
  return found;
}

void replicateFactorRoundTripIsExact() {
  auto parent = fanoutProgram();
  auto replicate = findReplicate(parent);
  require(replicate.has_value(), "complete fanout was not enumerated");
  auto child = take(dataflow::materializeDataflowRewrite(parent, *replicate));
  require(child.has_value(), "replication produced no child");

  unsigned adds = 0;
  bool everyAddHasOneSink = true;
  child->module().walk([&](mlir::arith::AddIOp add) {
    ++adds;
    everyAddHasOneSink &= add.getResult().hasOneUse();
  });
  require(adds == 3 && everyAddHasOneSink,
          "replication did not create one Compute per canonical sink");
  const std::array<loom::sim::DFGRuntimeArg, 2> args = {
      loom::sim::DFGRuntimeArg{0, "5"}, loom::sim::DFGRuntimeArg{1, "7"}};
  require(take(dataflow::test::simulateOnlyGraph(parent, args)) ==
              take(dataflow::test::simulateOnlyGraph(*child, args)),
          "fanout replication changed external observations");

  auto factor = findFactor(*child);
  require(factor && factor->replicas.size() == 3,
          "complete replica group was not enumerated");
  auto incomplete = *factor;
  incomplete.replicas.pop_back();
  require(isRejected(dataflow::materializeDataflowRewrite(*child, incomplete)),
          "proper replica subset was accepted");

  auto restored = take(dataflow::materializeDataflowRewrite(*child, *factor));
  require(restored && restored->identity() == parent.identity(),
          "factor did not restore the exact parent artifact");
}

void nondeterministicComputeIsRejected() {
  auto artifact = freezeFanoutProgram();
  require(!findReplicate(artifact),
          "nondeterministic freeze entered the fanout domain");
}

void selectorFanoutPreservesCompletionProof() {
  auto parent = selectorFanoutProgram();
  auto replicate = findReplicate(parent);
  require(replicate.has_value(),
          "deterministic selector fanout was not enumerated");
  auto child = take(dataflow::materializeDataflowRewrite(parent, *replicate));
  require(child.has_value(), "selector fanout replication produced no child");

  auto factor = findFactor(*child);
  require(factor && factor->replicas.size() == 2,
          "selector replica group was not enumerated");
  auto restored = take(dataflow::materializeDataflowRewrite(*child, *factor));
  require(restored && restored->identity() == parent.identity(),
          "selector factoring did not restore the exact parent artifact");
}

void layeredSelectorFanoutPreservesCompletionProof() {
  auto parent = layeredSelectorFanoutProgram();
  auto phaseChild = take(dataflow::materializeDataflowRewrite(
      parent, dataflow::PureComputeFanoutReplicateRewrite{
                  actorId<mlir::arith::AndIOp>(parent)}));
  require(phaseChild.has_value(), "selector replication produced no child");

  auto sourceChild = take(dataflow::materializeDataflowRewrite(
      *phaseChild, dataflow::PureComputeFanoutReplicateRewrite{
                       actorId<mlir::arith::XOrIOp>(*phaseChild)}));
  require(sourceChild.has_value(),
          "operand replication invalidated selector correspondence");
}

void replicatedStreamBoundPreservesConditionalRetirement() {
  auto parent = finalize(R"mlir(
module {
  dataflow.graph private @gated_bound_fanout(
      %start: none, %lower: i16, %upper: i16, %step: i16,
      %memory: memref<4xi16>) -> ()
      attributes {input_segments = array<i32: 3, 0, 1>,
                  result_segments = array<i32: 0, 0, 0>} {
    %bound = arith.addi %upper, %lower : i16
    %seed = dataflow.constant %start {const_value = 3 : i16} : i16
    %iv, %phase = dataflow.stream %lower, %bound, %step step add while slt : i16
    %issue = dataflow.invariant %phase, %start : none
    %payload = dataflow.invariant %phase, %seed : i16
    %after_cond, %after_value = dataflow.gate %phase, %payload : i16
    %final:2 = dataflow.demux %after_cond, %after_value
        : (i1, i16) -> (i16, i16)
    %collected = dataflow.carry %phase, %start, %written : none
    %issue_lane:2 = dataflow.demux %phase, %issue : (i1, none) -> (none, none)
    %done_lane:2 = dataflow.demux %phase, %collected : (i1, none) -> (none, none)
    %index = arith.index_cast %iv : i16 to index
    %written = dataflow.store %memory[%index] %after_value %issue_lane#1
        : memref<4xi16>
    %nonempty = arith.cmpi slt, %lower, %bound : i16
    %branches:2 = dataflow.demux %nonempty, %issue_lane#0
        : (i1, none) -> (none, none)
    %gated:2 = dataflow.sync %branches#1, %final#0 : (none, i16) -> (none, i16)
    %retired = dataflow.mux %nonempty, %branches#0, %gated#0
        : (i1, none, none) -> none
    dataflow.graph.return values() streams() memories()
        complete(%retired, %done_lane#0 : none, none)
  }
}
)mlir");
  auto child = take(dataflow::materializeDataflowRewrite(
      parent, dataflow::PureComputeFanoutReplicateRewrite{
                  actorId<mlir::arith::AddIOp>(parent)}));
  require(child.has_value(), "stream-bound replication produced no candidate");
  for (const char *upper : {"0", "4"}) {
    loom::sim::DFGSimulationOptions options;
    options.args = {{0, "0"}, {1, upper}, {2, "1"}};
    options.memories = {{3, 0, "9,9,9,9"}};
    const auto simulate = [&](const auto &artifact) {
      options.graphName =
          mlir::cast<dataflow::GraphOp>(artifact.view().graphs().front().op)
              .getSymName()
              .str();
      return take(loom::sim::simulateDataflowGraph(artifact.module(), options));
    };
    auto before = simulate(parent);
    auto after = simulate(*child);
    const llvm::SmallVector<std::string> expected(
        4, llvm::StringRef(upper) == "0" ? "i16:9" : "i16:3");
    require(
        before.status == "pass" && after.status == "pass" &&
            before.finalMemoryState.at("arg3") == expected &&
            before.finalMemoryState == after.finalMemoryState,
        "replicated gate bounds changed empty/nonempty retirement or writes");
  }

  auto mismatched = mlir::OwningOpRef<mlir::ModuleOp>(parent.module().clone());
  mismatched->walk([&](mlir::arith::CmpIOp predicate) {
    auto graph = predicate->getParentOfType<dataflow::GraphOp>();
    predicate->setOperand(1, graph.getBody().front().getArgument(3));
  });
  require(isRejected(dataflow::finalizeCanonicalDataflow(mismatched.get())),
          "an unrelated gate bound acquired a conditional completion proof");
}

void completionPhaseSplitPreservesMemoryAndTermination() {
  auto parent = finalize(R"mlir(
module {
  dataflow.graph private @completion_fanout(
      %start: none, %lower: i16, %upper: i16, %step: i16,
      %memory: memref<4xi16>) -> ()
      attributes {input_segments = array<i32: 3, 0, 1>,
                  result_segments = array<i32: 0, 0, 0>} {
    %iv, %phase = dataflow.stream %lower, %upper, %step step add while slt : i16
    %issue = dataflow.invariant %phase, %start : none
    %collected = dataflow.carry %phase, %start, %written : none
    %issue_lane:2 = dataflow.demux %phase, %issue : (i1, none) -> (none, none)
    %done_lane:2 = dataflow.demux %phase, %collected : (i1, none) -> (none, none)
    %index = arith.index_cast %iv : i16 to index
    %written = dataflow.store %memory[%index] %iv %issue_lane#1 : memref<4xi16>
    dataflow.graph.return values() streams() memories()
        complete(%issue_lane#0, %done_lane#0 : none, none)
  }
}
)mlir");
  auto decisions =
      take(dataflow::enumerateFixedDataflowRewriteDecisions(parent));
  std::optional<dataflow::DataflowRewriteDecision> selected;
  for (const auto &decision : decisions)
    if (std::holds_alternative<dataflow::StreamCompletionPhaseSplitRewrite>(
            decision))
      selected = decision;
  require(selected.has_value(), "completion phase split was not enumerated");
  auto child = take(dataflow::materializeDataflowRewrite(parent, *selected));
  require(child.has_value(), "completion phase split produced no candidate");
  for (const char *upper : {"0", "4"}) {
    loom::sim::DFGSimulationOptions options;
    options.args = {{0, "0"}, {1, upper}, {2, "1"}};
    options.memories = {{3, 0, "9,9,9,9"}};
    const auto simulate =
        [&](const dataflow::CanonicalDataflowArtifact &artifact) {
          options.graphName =
              mlir::cast<dataflow::GraphOp>(artifact.view().graphs().front().op)
                  .getSymName()
                  .str();
          return take(
              loom::sim::simulateDataflowGraph(artifact.module(), options));
        };
    auto before = simulate(parent);
    auto after = simulate(*child);
    require(before.status == "pass" && after.status == "pass" &&
                before.finalMemoryState == after.finalMemoryState,
            "phase split changed memory or activation termination");
  }
  for (const auto &decision :
       take(dataflow::enumerateFixedDataflowRewriteDecisions(*child)))
    require(
        !std::holds_alternative<dataflow::StreamCompletionPhaseSplitRewrite>(
            decision),
        "completion phase splitting did not exhaust its matched domain");
}

} // namespace

int main() {
  replicateFactorRoundTripIsExact();
  nondeterministicComputeIsRejected();
  selectorFanoutPreservesCompletionProof();
  layeredSelectorFanoutPreservesCompletionProof();
  replicatedStreamBoundPreservesConditionalRetirement();
  completionPhaseSplitPreservesMemoryAndTermination();
  return EXIT_SUCCESS;
}
