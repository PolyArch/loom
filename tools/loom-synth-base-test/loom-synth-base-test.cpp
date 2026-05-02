// CLI helper for lit tests: drives the Synthesizer base + factory glue
// added in this task. The tool is intentionally tiny -- it never builds
// IR -- because the strategies it dispatches through are all stubs at
// this point.
//
// Usage:
//   loom-synth-base-test --list-strategies
//   loom-synth-base-test --list-failure-reasons
//   loom-synth-base-test --make <strategy>
//
// Output formats (one per line):
//   --list-strategies         -> the four canonical strategy names
//                                (lexical: anchor, mcs, incremental,
//                                 incremental_random) in spec order.
//   --list-failure-reasons    -> the 13 SynthFailureReason snake_case
//                                strings in enum order. The success
//                                sentinel `None` prints as `none` so
//                                the line is unambiguous in lit
//                                checks; every other value matches
//                                `failureReasonString` verbatim.
//   --make <strategy>         -> `result: success=<bool> reason=<str>`
//                                followed by `note: <text>` lines for
//                                every entry in `SynthResult.notes`.
//                                For an unknown strategy the line is
//                                `factory: nullptr` instead.
//
// All command-line modes use a default-constructed `SynthConfig`. The
// `--make` paths run on an empty `SynthInputs` (groupName `t`, no
// subgraphs) against a fresh `MLIRContext`; this is sufficient to
// observe the stub's failure path because no current strategy inspects
// inputs before reporting.

#include "Common/SynthConfig.h"
#include "Dataflow/IR/DataflowDialect.h"
#include "Fabric/IR/FabricDialect.h"
#include "Fabric/Tech/Synthesizer/Synthesizer.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"

#include <memory>
#include <string>

static ::llvm::cl::opt<bool> listStrategies(
    "list-strategies",
    ::llvm::cl::desc("Print the four canonical strategy names, one per line"),
    ::llvm::cl::init(false));

static ::llvm::cl::opt<bool> listFailureReasons(
    "list-failure-reasons",
    ::llvm::cl::desc("Print all SynthFailureReason snake_case strings in "
                     "enum order, one per line"),
    ::llvm::cl::init(false));

static ::llvm::cl::opt<std::string>
    makeStrategy("make",
                 ::llvm::cl::desc("Construct the named strategy via "
                                  "makeSynthesizer and run it on empty "
                                  "SynthInputs. Print one `result:` line "
                                  "plus `note:` lines, or `factory: "
                                  "nullptr` on an unknown name."),
                 ::llvm::cl::init(""));

namespace {

constexpr ::llvm::StringRef kKnownStrategies[] = {
    "anchor",
    "mcs",
    "incremental",
    "incremental_random",
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

int doMake(::llvm::StringRef strategy) {
  ::loom::SynthConfig cfg;
  auto synth = ::loom::fabric::tech::makeSynthesizer(strategy, cfg);
  if (!synth) {
    ::llvm::outs() << "factory: nullptr\n";
    return 0;
  }

  ::mlir::DialectRegistry registry;
  registry.insert<::mlir::func::FuncDialect, ::fabric::FabricDialect,
                  ::dataflow::DataflowDialect>();
  ::mlir::MLIRContext ctx(registry);
  ctx.loadAllAvailableDialects();

  ::llvm::SmallVector<::dataflow::SubgraphOp, 0> noSubgraphs;
  ::loom::fabric::tech::SynthInputs inputs{
      /*groupName=*/::llvm::StringRef("t"),
      /*subgraphs=*/::llvm::ArrayRef<::dataflow::SubgraphOp>(noSubgraphs),
      /*config=*/cfg,
      /*context=*/&ctx,
  };

  auto result = synth->run(inputs);
  ::llvm::StringRef reason =
      ::loom::fabric::tech::failureReasonString(result.failureReason);
  ::llvm::outs() << "result: success=" << (result.success() ? "true" : "false")
                 << " reason=" << (reason.empty() ? "none" : reason) << "\n";
  for (const std::string &n : result.notes)
    ::llvm::outs() << "note: " << n << "\n";
  return 0;
}

} // namespace

int main(int argc, char **argv) {
  ::llvm::cl::ParseCommandLineOptions(
      argc, argv,
      "loom-synth-base-test: drive Synthesizer factory + failure-reason "
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
  if (!makeStrategy.getValue().empty()) {
    int rc = doMake(makeStrategy.getValue());
    if (rc != 0)
      return rc;
    didSomething = true;
  }
  if (!didSomething) {
    ::llvm::errs() << "error: one of --list-strategies / "
                      "--list-failure-reasons / --make is required\n";
    return 1;
  }
  return 0;
}
