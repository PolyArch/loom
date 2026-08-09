#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Dataflow/IR/DataflowDialect.h"
#include "Dataflow/IR/DataflowOps.h"
#include "Dataflow/Transforms/DataflowRewrite.h"
#include "DataflowRewriteTestSupport.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
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
  llvm::errs() << "dataflow cardinality rewrite: " << message << '\n';
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
    registry.insert<dataflow::DataflowDialect, mlir::arith::ArithDialect>();
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

dataflow::CanonicalDataflowArtifact scalarThenParallelize() {
  return finalize(R"mlir(
module {
  dataflow.graph private @scalar_then_parallelize(
      %start: none) -> (i8, i1)
      attributes {input_segments = array<i32: 0, 0, 0>,
                  result_segments = array<i32: 0, 2, 0>} {
    %zero = dataflow.constant %start {const_value = 0 : i8} : i8
    %four = dataflow.constant %start {const_value = 4 : i8} : i8
    %one = dataflow.constant %start {const_value = 1 : i8} : i8
    %data_in, %phase = dataflow.stream %zero, %four, %one
        step add while ult : i8
    %sum = arith.addi %data_in, %data_in : i8
    %vector, %mask, %group_phase =
      dataflow.parallelize %sum, %phase
        : (i8, i1) -> (vector<4xi8>, vector<4xi1>, i1)
    %data, %scalar_phase =
      dataflow.serialize %vector, %mask, %group_phase
        : (vector<4xi8>, vector<4xi1>, i1) -> (i8, i1)
    %units = dataflow.invariant %scalar_phase, %start : none
    %close:2 = dataflow.demux %scalar_phase, %units
        : (i1, none) -> (none, none)
    dataflow.graph.return values()
        streams(%data, %scalar_phase : i8, i1)
        memories() complete(%close#0 : none)
  }
}
)mlir");
}

dataflow::CanonicalDataflowArtifact vectorThenSerialize() {
  return finalize(R"mlir(
module {
  dataflow.graph private @vector_then_serialize(
      %start: none) -> (i8, i1)
      attributes {input_segments = array<i32: 0, 0, 0>,
                  result_segments = array<i32: 0, 2, 0>} {
    %zero = dataflow.constant %start {const_value = 0 : i8} : i8
    %four = dataflow.constant %start {const_value = 4 : i8} : i8
    %one = dataflow.constant %start {const_value = 1 : i8} : i8
    %data_in, %phase = dataflow.stream %zero, %four, %one
        step add while ult : i8
    %lhs, %mask, %group_phase =
      dataflow.parallelize %data_in, %phase
        : (i8, i1) -> (vector<4xi8>, vector<4xi1>, i1)
    %rhs, %unused_mask, %unused_group =
      dataflow.parallelize %data_in, %phase
        : (i8, i1) -> (vector<4xi8>, vector<4xi1>, i1)
    %sum = arith.addi %lhs, %rhs : vector<4xi8>
    %data, %scalar_phase =
      dataflow.serialize %sum, %mask, %group_phase
        : (vector<4xi8>, vector<4xi1>, i1) -> (i8, i1)
    %units = dataflow.invariant %scalar_phase, %start : none
    %close:2 = dataflow.demux %scalar_phase, %units
        : (i1, none) -> (none, none)
    dataflow.graph.return values()
        streams(%data, %scalar_phase : i8, i1) memories()
        complete(%close#0 : none)
  }
}
)mlir");
}

dataflow::CanonicalDataflowArtifact sidebandEscape() {
  return finalize(R"mlir(
module {
  dataflow.graph private @sideband_escape(
      %start: none)
      -> (vector<4xi8>, vector<4xi1>, i1, vector<4xi1>, i1)
      attributes {input_segments = array<i32: 0, 0, 0>,
                  result_segments = array<i32: 0, 5, 0>} {
    %zero = dataflow.constant %start {const_value = 0 : i8} : i8
    %four = dataflow.constant %start {const_value = 4 : i8} : i8
    %one = dataflow.constant %start {const_value = 1 : i8} : i8
    %data_in, %phase = dataflow.stream %zero, %four, %one
        step add while ult : i8
    %left, %left_mask, %left_group =
      dataflow.parallelize %data_in, %phase
        : (i8, i1) -> (vector<4xi8>, vector<4xi1>, i1)
    %right, %right_mask, %right_group =
      dataflow.parallelize %data_in, %phase
        : (i8, i1) -> (vector<4xi8>, vector<4xi1>, i1)
    %sum = arith.addi %left, %right : vector<4xi8>
    %left_units = dataflow.invariant %left_group, %start : none
    %left_close:2 = dataflow.demux %left_group, %left_units
        : (i1, none) -> (none, none)
    %right_units = dataflow.invariant %right_group, %start : none
    %right_close:2 = dataflow.demux %right_group, %right_units
        : (i1, none) -> (none, none)
    dataflow.graph.return values()
        streams(%sum, %left_mask, %left_group, %right_mask, %right_group
          : vector<4xi8>, vector<4xi1>, i1, vector<4xi1>, i1)
        memories() complete(%left_close#0, %right_close#0 : none, none)
  }
}
)mlir");
}

std::optional<dataflow::ElementwiseCardinalityCommuteRewrite>
findDecision(const dataflow::CanonicalDataflowArtifact &artifact,
             dataflow::CardinalityCommuteDirection direction,
             std::optional<std::size_t> adapterCount = std::nullopt) {
  auto decisions =
      take(dataflow::enumerateFixedDataflowRewriteDecisions(artifact));
  std::optional<dataflow::ElementwiseCardinalityCommuteRewrite> found;
  for (const dataflow::DataflowRewriteDecision &decision : decisions) {
    const auto *commute =
        std::get_if<dataflow::ElementwiseCardinalityCommuteRewrite>(&decision);
    if (!commute || commute->direction != direction ||
        (adapterCount && commute->adapters.size() != *adapterCount))
      continue;
    if (found)
      fail("fixture has more than one cardinality decision");
    found = *commute;
  }
  return found;
}

void parallelizeShellRoundTrips() {
  auto parent = scalarThenParallelize();
  auto inside = findDecision(
      parent, dataflow::CardinalityCommuteDirection::MoveInside, 1);
  require(inside.has_value(), "result parallelize shell was not enumerated");
  auto child = take(dataflow::materializeDataflowRewrite(parent, *inside));
  require(child.has_value(), "MoveInside produced no child");

  unsigned scalarAdds = 0;
  unsigned vectorAdds = 0;
  unsigned parallelizes = 0;
  child->module().walk([&](mlir::arith::AddIOp add) {
    scalarAdds += add.getType().isInteger(8);
    vectorAdds += llvm::isa<mlir::VectorType>(add.getType());
  });
  child->module().walk([&](dataflow::ParallelizeOp) { ++parallelizes; });
  require(scalarAdds == 0 && vectorAdds == 1 && parallelizes == 2,
          "MoveInside did not build operand adapters and vector Compute");
  require(take(dataflow::test::simulateOnlyGraph(parent)) ==
              take(dataflow::test::simulateOnlyGraph(*child)),
          "parallelize commute changed external observations");

  auto outside = findDecision(
      *child, dataflow::CardinalityCommuteDirection::MoveOutside, 2);
  require(outside.has_value(), "operand parallelize shell was not enumerated");
  auto restored = take(dataflow::materializeDataflowRewrite(*child, *outside));
  require(restored && restored->identity() == parent.identity(),
          "parallelize shell inverse did not restore the parent");
}

void serializeShellRoundTrips() {
  auto parent = vectorThenSerialize();
  auto outside = findDecision(
      parent, dataflow::CardinalityCommuteDirection::MoveOutside, 1);
  require(outside.has_value(), "result serialize shell was not enumerated");
  auto child = take(dataflow::materializeDataflowRewrite(parent, *outside));
  require(child.has_value(), "serialize MoveOutside produced no child");

  unsigned scalarAdds = 0;
  unsigned serializes = 0;
  child->module().walk([&](mlir::arith::AddIOp add) {
    scalarAdds += add.getType().isInteger(8);
  });
  child->module().walk([&](dataflow::SerializeOp) { ++serializes; });
  require(scalarAdds == 1 && serializes == 2,
          "MoveOutside did not build operand serializes and scalar Compute");
  require(take(dataflow::test::simulateOnlyGraph(parent)) ==
              take(dataflow::test::simulateOnlyGraph(*child)),
          "serialize commute changed external observations");

  auto inside = findDecision(
      *child, dataflow::CardinalityCommuteDirection::MoveInside, 2);
  require(inside.has_value(), "operand serialize shell was not enumerated");
  auto restored = take(dataflow::materializeDataflowRewrite(*child, *inside));
  require(restored && restored->identity() == parent.identity(),
          "serialize shell inverse did not restore the parent");
}

void nonRepresentativeSidebandEscapeIsRejected() {
  auto artifact = sidebandEscape();
  require(!findDecision(artifact,
                        dataflow::CardinalityCommuteDirection::MoveOutside),
          "incomplete adapter shell entered the decision domain");
}

} // namespace

int main() {
  parallelizeShellRoundTrips();
  serializeShellRoundTrips();
  nonRepresentativeSidebandEscapeIsRejected();
  return EXIT_SUCCESS;
}
