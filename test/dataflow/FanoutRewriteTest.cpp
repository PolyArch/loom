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
  auto view = take(artifact.view());
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

} // namespace

int main() {
  replicateFactorRoundTripIsExact();
  nondeterministicComputeIsRejected();
  selectorFanoutPreservesCompletionProof();
  layeredSelectorFanoutPreservesCompletionProof();
  return EXIT_SUCCESS;
}
