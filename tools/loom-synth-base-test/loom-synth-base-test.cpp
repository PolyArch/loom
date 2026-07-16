// CLI helper for lit tests: drives the canonical synthesis entrypoint and
// candidate-ranking helpers.
//
// Usage:
//   loom-synth-base-test --list-strategies
//   loom-synth-base-test --list-failure-reasons
//   loom-synth-base-test --synthesize-empty <strategy>
//
// Output formats (one per line):
//   --list-strategies         -> the selectable canonical strategy names.
//   --list-failure-reasons    -> the 11 SynthFailureReason snake_case
//                                strings in enum order. The success
//                                sentinel `None` prints as `none` so
//                                the line is unambiguous in lit
//                                checks; every other value matches
//                                `failureReasonString` verbatim.
//   --synthesize-empty <strategy>
//                             -> `result: success=<bool> reason=<str>`
//                                followed by `note: <text>` lines for every
//                                entry in `SynthResult.notes`.
//
// All command-line modes use a default-constructed `SynthConfig`. The
// `--synthesize-empty` runs on an empty `SynthInputs` (groupName `t`, no
// functions) against a fresh `MLIRContext`.

#include "Common/SynthConfig.h"
#include "Dataflow/IR/DataflowDialect.h"
#include "Fabric/IR/FabricDialect.h"
#include "Fabric/IR/FabricTypes.h"
#include "Fabric/Tech/Synthesizer/Synthesizer.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

static ::llvm::cl::opt<bool> listStrategies(
    "list-strategies",
    ::llvm::cl::desc("Print canonical strategy names, one per line"),
    ::llvm::cl::init(false));

static ::llvm::cl::opt<bool> listFailureReasons(
    "list-failure-reasons",
    ::llvm::cl::desc("Print all SynthFailureReason snake_case strings in "
                     "enum order, one per line"),
    ::llvm::cl::init(false));

static ::llvm::cl::opt<std::string> synthesizeEmptyStrategy(
    "synthesize-empty",
    ::llvm::cl::desc("Run the canonical synthesis entrypoint on empty "
                     "SynthInputs with the named strategy"),
    ::llvm::cl::init(""));

static ::llvm::cl::opt<bool> capabilityTieBreak(
    "capability-tiebreak",
    ::llvm::cl::desc("Exercise deterministic capability-aware candidate "
                     "ranking"),
    ::llvm::cl::init(false));

static ::llvm::cl::opt<bool> synthesizeFixedVector(
    "synthesize-fixed-vector",
    ::llvm::cl::desc("Synthesize two identical fixed-vector configured "
                     "functions through the canonical anchor strategy"),
    ::llvm::cl::init(false));

namespace {

constexpr ::llvm::StringRef kKnownStrategies[] = {
    "anchor",
};

// Enum-order list of every SynthFailureReason (including None so the
// helper exercises the full switch). Kept in lockstep with the enum
// declaration in `Synthesizer.h`; a new value would fail to compile
// here under -Wswitch via the corresponding switch in
// `failureReasonString`.
constexpr ::loom::fabric::tech::SynthFailureReason kAllFailureReasons[] = {
    ::loom::fabric::tech::SynthFailureReason::None,
    ::loom::fabric::tech::SynthFailureReason::CrossShareGroup,
    ::loom::fabric::tech::SynthFailureReason::TopologyMismatch,
    ::loom::fabric::tech::SynthFailureReason::FeedbackAlignConflict,
    ::loom::fabric::tech::SynthFailureReason::Timeout,
    ::loom::fabric::tech::SynthFailureReason::ResourceExhausted,
    ::loom::fabric::tech::SynthFailureReason::UnsupportedOp,
    ::loom::fabric::tech::SynthFailureReason::InvalidInput,
    ::loom::fabric::tech::SynthFailureReason::VerifierFailed,
    ::loom::fabric::tech::SynthFailureReason::SymbolConflict,
    ::loom::fabric::tech::SynthFailureReason::ConfigParseFailed,
};

void printStrategies() {
  for (::llvm::StringRef s : kKnownStrategies)
    ::llvm::outs() << s << "\n";
}

void printFailureReasons() {
  for (::loom::fabric::tech::SynthFailureReason r : kAllFailureReasons) {
    ::llvm::StringRef name = ::loom::fabric::tech::failureReasonString(r);
    if (name.empty()) {
      // `None` round-trips as the empty string by design (so callers
      // can splat it into the `loom.synth_failed` attribute on success
      // without a bogus token). The helper prints a stable placeholder
      // so every line of `--list-failure-reasons` is non-empty.
      ::llvm::outs() << "none\n";
    } else {
      ::llvm::outs() << name << "\n";
    }
  }
}

int doSynthesizeEmpty(::llvm::StringRef strategy) {
  ::loom::SynthConfig cfg;
  cfg.strategy = strategy.str();

  ::mlir::DialectRegistry registry;
  registry.insert<::mlir::func::FuncDialect, ::fabric::FabricDialect,
                  ::dataflow::DataflowDialect>();
  ::mlir::MLIRContext ctx(registry);
  ctx.loadAllAvailableDialects();

  ::llvm::SmallVector<::fabric::ConfiguredFunction, 0> noFunctions;
  ::loom::fabric::tech::SynthInputs inputs{
      /*groupName=*/::llvm::StringRef("t"),
      /*functions=*/::llvm::ArrayRef<::fabric::ConfiguredFunction>(noFunctions),
      /*context=*/&ctx,
  };

  auto result = ::loom::fabric::tech::synthesize(cfg, inputs);
  ::llvm::StringRef reason =
      ::loom::fabric::tech::failureReasonString(result.failureReason);
  ::llvm::outs() << "result: success=" << (result.success() ? "true" : "false")
                 << " reason=" << (reason.empty() ? "none" : reason) << "\n";
  for (const std::string &n : result.notes)
    ::llvm::outs() << "note: " << n << "\n";
  return 0;
}

int doCapabilityTieBreak() {
  ::loom::fabric::tech::SynthCandidateScore smallerExtra;
  smallerExtra.hardwareCost = 10.0;
  smallerExtra.capability.encodingCount = 3;
  smallerExtra.capability.extraCapabilityCount = 1;
  smallerExtra.deterministicOrder = 1;

  ::loom::fabric::tech::SynthCandidateScore largerExtra = smallerExtra;
  largerExtra.capability.encodingCount = 4;
  largerExtra.capability.extraCapabilityCount = 2;
  largerExtra.deterministicOrder = 0;

  ::loom::fabric::tech::SynthCandidateScore lowerCost = largerExtra;
  lowerCost.hardwareCost = 9.0;

  ::llvm::outs() << "equal_cost_prefers_less_extra="
                 << (::loom::fabric::tech::preferSynthCandidate(smallerExtra,
                                                                largerExtra)
                         ? "true"
                         : "false")
                 << "\n";
  ::llvm::outs() << "lower_cost_precedes_extra_metric="
                 << (::loom::fabric::tech::preferSynthCandidate(lowerCost,
                                                                smallerExtra)
                         ? "true"
                         : "false")
                 << "\n";
  return 0;
}

int doFixedVectorSynthesis() {
  ::loom::SynthConfig cfg;

  ::mlir::DialectRegistry registry;
  registry.insert<::mlir::arith::ArithDialect, ::mlir::func::FuncDialect,
                  ::fabric::FabricDialect, ::dataflow::DataflowDialect>();
  ::mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  auto i32 = ::mlir::IntegerType::get(&context, 32);
  auto vector = ::mlir::VectorType::get({4}, i32);
  ::fabric::ConfiguredFunction function;
  function.inputs.push_back({0, vector});
  function.inputs.push_back({1, vector});
  ::fabric::ConfiguredFunctionNode node;
  node.operationName = "arith.addi";
  node.functionType =
      ::mlir::FunctionType::get(&context, {vector, vector}, {vector});
  node.attributes = ::mlir::DictionaryAttr::get(&context);
  node.operands.push_back(::fabric::ConfiguredValue::input(0));
  node.operands.push_back(::fabric::ConfiguredValue::input(1));
  function.nodes.push_back(std::move(node));
  function.outputs.push_back(
      {0, vector, ::fabric::ConfiguredValue::nodeResult(0, 0)});

  ::llvm::SmallVector<::fabric::ConfiguredFunction, 2> functions = {function,
                                                                    function};
  ::loom::fabric::tech::SynthInputs inputs{
      /*groupName=*/"fixed_vector",
      /*functions=*/functions,
      /*context=*/&context,
  };
  auto result = ::loom::fabric::tech::synthesize(cfg, inputs);
  if (!result.success()) {
    ::llvm::outs() << "synthesis=failed reason="
                   << ::loom::fabric::tech::failureReasonString(
                          result.failureReason)
                   << "\n";
    for (const std::string &note : result.notes)
      ::llvm::outs() << "note=" << note << "\n";
    return 1;
  }

  ::fabric::FuOp fu;
  result.wrapper->walk([&](::fabric::FuOp candidate) { fu = candidate; });
  if (!fu)
    return 1;
  auto inputType = ::mlir::dyn_cast<::fabric::BitsType>(
      fu.getBody().front().getArgument(0).getType());
  if (!inputType)
    return 1;
  std::size_t covered = 0;
  for (const auto &witness : result.coverage.witnesses)
    covered += witness.has_value();
  ::llvm::outs() << "synthesis=success\n"
                 << "input_width=" << inputType.getWidth() << "\n"
                 << "encodings=" << ::fabric::getValidSemanticEncodingCount(fu)
                 << "\n"
                 << "covered=" << covered << "\n";
  return 0;
}

} // namespace

int main(int argc, char **argv) {
  ::llvm::cl::ParseCommandLineOptions(
      argc, argv,
      "loom-synth-base-test: drive canonical synthesis and failure-reason "
      "string mapping from lit tests\n");

  bool didSomething = false;
  if (listStrategies.getValue()) {
    printStrategies();
    didSomething = true;
  }
  if (listFailureReasons.getValue()) {
    printFailureReasons();
    didSomething = true;
  }
  if (!synthesizeEmptyStrategy.getValue().empty()) {
    int rc = doSynthesizeEmpty(synthesizeEmptyStrategy.getValue());
    if (rc != 0)
      return rc;
    didSomething = true;
  }
  if (capabilityTieBreak.getValue()) {
    int rc = doCapabilityTieBreak();
    if (rc != 0)
      return rc;
    didSomething = true;
  }
  if (synthesizeFixedVector.getValue()) {
    int rc = doFixedVectorSynthesis();
    if (rc != 0)
      return rc;
    didSomething = true;
  }
  if (!didSomething) {
    ::llvm::errs() << "error: one of --list-strategies / "
                      "--list-failure-reasons / --synthesize-empty / "
                      "--capability-tiebreak / --synthesize-fixed-vector is "
                      "required\n";
    return 1;
  }
  return 0;
}
