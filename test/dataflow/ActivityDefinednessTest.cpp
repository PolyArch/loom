#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Dataflow/IR/DataflowDialect.h"
#include "Dataflow/IR/DataflowOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <utility>

namespace {

[[noreturn]] void fail(llvm::StringRef message) {
  llvm::errs() << "activity definedness test: " << message << '\n';
  std::exit(1);
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

dataflow::CanonicalActorView
actor(const dataflow::CanonicalDataflowProgramView &view,
      dataflow::OperationSchemaId schema) {
  for (const dataflow::CanonicalActorView &candidate : view.actors()) {
    if (dataflow::requireOperationSchema(candidate.op) != schema)
      continue;
    return candidate;
  }
  fail("fixture has no requested actor");
}

dataflow::CanonicalActorView
actorWithResultCount(const dataflow::CanonicalDataflowProgramView &view,
                     dataflow::OperationSchemaId schema, unsigned resultCount) {
  for (const dataflow::CanonicalActorView &candidate : view.actors())
    if (dataflow::requireOperationSchema(candidate.op) == schema &&
        candidate.op->getNumResults() == resultCount)
      return candidate;
  fail("fixture has no requested actor shape");
}

dataflow::CanonicalActorView
floatComparison(const dataflow::CanonicalDataflowProgramView &view,
                mlir::arith::FastMathFlags flags) {
  for (const dataflow::CanonicalActorView &candidate : view.actors()) {
    auto comparison = llvm::dyn_cast<mlir::arith::CmpFOp>(candidate.op);
    if (comparison && comparison.getFastmath() == flags)
      return candidate;
  }
  fail("fixture has no requested floating comparison");
}

dataflow::CanonicalDataflowArtifact buildProgram(mlir::MLIRContext &context) {
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 64>>} {
  dataflow.graph private @facts(%start: none, %input: i1)
      -> (i1, i1, i1, i1, i1, i1, i1, i1)
      attributes {input_segments = array<i32: 1, 0, 0>,
                  result_segments = array<i32: 0, 8, 0>} {
    %one = dataflow.constant %start {const_value = true} : i1
    %zero = dataflow.constant %start {const_value = false} : i1
    %and = arith.andi %one, %zero : i1
    %missing = arith.divui %one, %one : i1
    %poison = ub.poison : i1
    %frozen = llvm.freeze %poison : i1
    %zero_f = arith.constant 0.0 : f32
    %nan_f = arith.constant 0x7FC00000 : f32
    %inf_f = arith.constant 0x7F800000 : f32
    %strict_cmp = arith.cmpf uno, %nan_f, %zero_f : f32
    %nnan_cmp = arith.cmpf uno, %nan_f, %zero_f fastmath<nnan> : f32
    %ninf_cmp = arith.cmpf ogt, %inf_f, %zero_f fastmath<ninf> : f32
    %done:8 = dataflow.sync %start, %one, %and, %missing, %frozen,
        %strict_cmp, %nnan_cmp, %ninf_cmp
        : (none, i1, i1, i1, i1, i1, i1, i1)
          -> (none, i1, i1, i1, i1, i1, i1, i1)
    dataflow.graph.return values() streams(
        %done#1, %done#2, %done#2, %done#3, %done#4, %done#5, %done#6,
        %done#7 : i1, i1, i1, i1, i1, i1, i1, i1)
        memories() complete(%done#0 : none)
  }

  dataflow.graph private @cycle(%start: none, %phase: i1) -> ()
      attributes {input_segments = array<i32: 1, 0, 0>,
                  result_segments = array<i32: 0, 0, 0>} {
    %cycle_a = dataflow.sync %cycle_b : (i1) -> i1
    %cycle_b = dataflow.sync %cycle_a : (i1) -> i1
    %control = dataflow.carry %phase, %start, %body : none
    %lanes:2 = dataflow.demux %phase, %control
        : (i1, none) -> (none, none)
    %body = dataflow.sync %lanes#1 : (none) -> none
    dataflow.graph.return values() streams() memories()
        complete(%lanes#0 : none)
  }

  dataflow.graph private @adapter_results(%start: none)
      -> (i8, i1)
      attributes {input_segments = array<i32: 0, 0, 0>,
                  result_segments = array<i32: 0, 2, 0>} {
    %zero = dataflow.constant %start {const_value = 0 : i8} : i8
    %one = dataflow.constant %start {const_value = 1 : i8} : i8
    %item, %phase = dataflow.stream %zero, %one, %one
        step add while ult : i8
    %unproven = arith.divui %item, %item : i8
    %vector, %mask, %group_phase =
      dataflow.parallelize %unproven, %phase
        : (i8, i1) -> (vector<4xi8>, vector<4xi1>, i1)
    %scalar, %scalar_phase =
      dataflow.serialize %vector, %mask, %group_phase
        : (vector<4xi8>, vector<4xi1>, i1) -> (i8, i1)
    %units = dataflow.invariant %scalar_phase, %start : none
    %close:2 = dataflow.demux %scalar_phase, %units
        : (i1, none) -> (none, none)
    dataflow.graph.return values() streams(%scalar, %scalar_phase : i8, i1)
        memories() complete(%close#0 : none)
  }

  dataflow.graph private @fixed_point(%start: none) -> i1
      attributes {input_segments = array<i32: 0, 0, 0>,
                  result_segments = array<i32: 1, 0, 0>} {
    %seed = dataflow.constant %start {const_value = true} : i1
    %hop0 = arith.andi %seed, %seed : i1
    %hop1 = arith.andi %hop0, %hop0 : i1
    %hop2 = arith.andi %hop1, %hop1 : i1
    %hop3 = arith.andi %hop2, %hop2 : i1
    %hop4 = arith.andi %hop3, %hop3 : i1
    %hop5 = arith.andi %hop4, %hop4 : i1
    %hop6 = arith.andi %hop5, %hop5 : i1
    %hop7 = arith.andi %hop6, %hop6 : i1
    %done:2 = dataflow.sync %start, %hop7
        : (none, i1) -> (none, i1)
    dataflow.graph.return values(%done#1 : i1) streams() memories()
        complete(%done#0 : none)
  }
}
)mlir",
                                                        &context);
  if (!module)
    fail("cannot parse fixture");
  return take(dataflow::finalizeCanonicalDataflow(*module));
}

void canonicalGraphOwnsLeastFixedPoint() {
  mlir::DialectRegistry registry;
  registry.insert<dataflow::DataflowDialect, mlir::arith::ArithDialect,
                  mlir::DLTIDialect, mlir::func::FuncDialect,
                  mlir::LLVM::LLVMDialect, mlir::ub::UBDialect>();
  mlir::MLIRContext context(registry, mlir::MLIRContext::Threading::DISABLED);
  auto artifact = buildProgram(context);
  const auto view = take(artifact.view());

  using Fact = dataflow::ActivityDefinedness;
  const auto constant =
      actor(view, dataflow::OperationSchemaId::DataflowConstant);
  const auto bitwise = actor(view, dataflow::OperationSchemaId::ArithAndI);
  const auto missing = actor(view, dataflow::OperationSchemaId::ArithDivUI);
  const auto poison = actor(view, dataflow::OperationSchemaId::UBPoison);
  const auto frozen = actor(view, dataflow::OperationSchemaId::LLVMFreeze);
  const auto strictCmp =
      floatComparison(view, mlir::arith::FastMathFlags::none);
  const auto nnanCmp = floatComparison(view, mlir::arith::FastMathFlags::nnan);
  const auto ninfCmp = floatComparison(view, mlir::arith::FastMathFlags::ninf);
  const auto sync =
      actorWithResultCount(view, dataflow::OperationSchemaId::DataflowSync, 8);
  const auto cycle =
      actorWithResultCount(view, dataflow::OperationSchemaId::DataflowSync, 1);
  const auto parallelize =
      actor(view, dataflow::OperationSchemaId::DataflowParallelize);
  const auto serialize =
      actor(view, dataflow::OperationSchemaId::DataflowSerialize);
  const auto fixedPoint =
      actorWithResultCount(view, dataflow::OperationSchemaId::DataflowSync, 2);

  auto actorPosition = [&](dataflow::ActorRef ref) {
    for (auto [position, candidate] : llvm::enumerate(view.actors()))
      if (candidate.ref == ref)
        return position;
    fail("fixture actor is absent from canonical order");
  };
  bool hasBackwardDependency = false;
  for (auto [consumerPosition, candidate] : llvm::enumerate(view.actors())) {
    if (candidate.graph != fixedPoint.graph)
      continue;
    for (unsigned ordinal = 0; ordinal < candidate.op->getNumOperands();
         ++ordinal) {
      auto producer = take(view.graphProducer(
          dataflow::ActorTokenOperandRef{candidate.ref, ordinal}));
      const auto *result =
          std::get_if<dataflow::ActorTokenResultRef>(&producer);
      if (result && actorPosition(result->actor) > consumerPosition)
        hasBackwardDependency = true;
    }
  }
  require(hasBackwardDependency,
          "fixed-point fixture does not require a second canonical sweep");
  require(take(view.activityDefinedness(dataflow::ActorTokenResultRef{
              fixedPoint.ref, 1})) == Fact::AlwaysDefined,
          "least fixed point stopped before a backward dependency converged");

  require(take(view.activityDefinedness(dataflow::ActorTokenResultRef{
              constant.ref, 0})) == Fact::AlwaysDefined,
          "canonical defined constant was not a proof seed");
  require(take(view.activityDefinedness(dataflow::GraphIngressTokenRef{
              dataflow::GraphValueInputTokenRef{cycle.graph, 0}})) ==
              Fact::Unproven,
          "graph input was promoted without a graph proof");
  require(take(view.activityDefinedness(dataflow::ActorTokenResultRef{
              bitwise.ref, 0})) == Fact::AlwaysDefined,
          "registered monotone transfer did not reach its result");
  require(take(view.activityDefinedness(dataflow::ActorTokenResultRef{
              sync.ref, 2})) == Fact::AlwaysDefined,
          "result-wise sync transfer depended on an unrelated operand");
  require(take(view.activityDefinedness(
              dataflow::ActorTokenResultRef{poison.ref, 0})) == Fact::Unproven,
          "explicit poison acquired an activity proof");
  require(take(view.activityDefinedness(dataflow::ActorTokenResultRef{
              frozen.ref, 0})) == Fact::AlwaysDefined,
          "registered freeze transfer did not prove its result");
  require(take(view.activityDefinedness(
              dataflow::ActorTokenResultRef{missing.ref, 0})) == Fact::Unproven,
          "missing transfer relation was treated as identity");
  require(take(view.activityDefinedness(dataflow::ActorTokenResultRef{
              strictCmp.ref, 0})) == Fact::AlwaysDefined,
          "strict floating comparison did not preserve defined operands");
  require(take(view.activityDefinedness(
              dataflow::ActorTokenResultRef{nnanCmp.ref, 0})) == Fact::Unproven,
          "nnan comparison promoted a defined NaN operand");
  require(take(view.activityDefinedness(
              dataflow::ActorTokenResultRef{ninfCmp.ref, 0})) == Fact::Unproven,
          "ninf comparison promoted a defined infinity operand");
  require(take(view.activityDefinedness(
              dataflow::ActorTokenResultRef{cycle.ref, 0})) == Fact::Unproven,
          "unseeded cycle escaped the least fixed point");
  require(take(view.activityDefinedness(dataflow::ActorTokenResultRef{
              parallelize.ref, 0})) == Fact::Unproven,
          "parallelize data result ignored its data dependency");
  require(take(view.activityDefinedness(dataflow::ActorTokenResultRef{
              parallelize.ref, 1})) == Fact::AlwaysDefined,
          "parallelize mask depended on unrelated scalar data");
  require(take(view.activityDefinedness(dataflow::ActorTokenResultRef{
              parallelize.ref, 2})) == Fact::AlwaysDefined,
          "parallelize phase depended on unrelated scalar data");
  require(take(view.activityDefinedness(dataflow::ActorTokenResultRef{
              serialize.ref, 0})) == Fact::Unproven,
          "serialize data result ignored its vector dependency");
  require(take(view.activityDefinedness(dataflow::ActorTokenResultRef{
              serialize.ref, 1})) == Fact::AlwaysDefined,
          "serialize phase depended on unrelated vector data");
}

} // namespace

int main() {
  canonicalGraphOwnsLeastFixedPoint();
  llvm::outs() << "activity definedness tests passed\n";
  return 0;
}
