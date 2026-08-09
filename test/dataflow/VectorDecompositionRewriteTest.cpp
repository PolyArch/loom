#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Dataflow/IR/DataflowDialect.h"
#include "Dataflow/IR/OperationSchema.h"
#include "Dataflow/Transforms/DataflowRewrite.h"
#include "DataflowRewriteTestSupport.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <array>
#include <cstdlib>

namespace {

[[noreturn]] void fail(llvm::StringRef message) {
  llvm::errs() << "dataflow vector decomposition rewrite: " << message << '\n';
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

dataflow::CanonicalDataflowArtifact vectorProgram() {
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  dataflow.graph private @vector_add(
      %start: none, %lhs: vector<4xi8>, %rhs: vector<4xi8>)
      -> vector<4xi8>
      attributes {input_segments = array<i32: 2, 0, 0>,
                  result_segments = array<i32: 1, 0, 0>} {
    %sum = arith.addi %lhs, %rhs : vector<4xi8>
    %retired:2 = dataflow.sync %start, %sum
        : (none, vector<4xi8>) -> (none, vector<4xi8>)
    dataflow.graph.return values(%retired#1 : vector<4xi8>)
        streams() memories() complete(%retired#0 : none)
  }
}
)mlir",
                                                        &context());
  if (!module)
    fail("cannot parse fixture");
  return take(dataflow::finalizeCanonicalDataflow(module.get()));
}

void chunkAndScalarizePreserveExternalObservations() {
  auto parent = vectorProgram();
  auto view = take(parent.view());
  auto compute = llvm::find_if(view.actors(),
                               [](const dataflow::CanonicalActorView &actor) {
                                 return dataflow::operationSchemaOf(actor.op) ==
                                        dataflow::OperationSchemaId::ArithAddI;
                               });
  require(compute != view.actors().end(), "vector Compute actor is absent");
  auto decisions =
      take(dataflow::enumerateElementwiseVectorDecompositionDecisions(
          parent, compute->ref));

  const std::array<loom::sim::DFGRuntimeArg, 2> args = {
      loom::sim::DFGRuntimeArg{0, "67305985"},
      loom::sim::DFGRuntimeArg{1, "134678021"}};
  auto expected = take(dataflow::test::simulateOnlyGraph(parent, args));
  bool sawChunk = false;
  bool sawScalarize = false;
  for (const dataflow::DataflowRewriteDecision &decision : decisions) {
    sawChunk |= std::holds_alternative<dataflow::ElementwiseVectorChunkRewrite>(
        decision);
    sawScalarize |=
        std::holds_alternative<dataflow::ElementwiseVectorScalarizeRewrite>(
            decision);
    auto child = take(dataflow::materializeDataflowRewrite(parent, decision));
    require(child.has_value(), "legal decomposition produced no child");
    require(take(dataflow::test::simulateOnlyGraph(*child, args)) == expected,
            "vector decomposition changed external observations");
  }
  require(sawChunk && sawScalarize,
          "fixture did not cover both decomposition modes");
}

} // namespace

int main() {
  chunkAndScalarizePreserveExternalObservations();
  return EXIT_SUCCESS;
}
