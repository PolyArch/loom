// CLI helper for lit tests: parse an MLIR module containing configured
// functions grouped by `loom.synth_group`, run the
// `loom-synthesize-configured-functions` pass with a supplied SynthConfig,
// and print three stable artifacts FileCheck can lock onto:
//
//   1. The synthesized FU IR for every `func.func` in the post-pass
//      module that carries `loom.synthesized_for`. Each function is
//      printed verbatim using MLIR's standard Op printer.
//   2. The canonical `synth-stat` line per group, captured from the
//      pass's `dump-stats=true` remarks.
//   3. A `wallclock_us=<N>` line measuring the wall-time spent inside
//      `PassManager::run` (microseconds).
//
// Usage:
//   loom-synth-fu-dump <input.mlir>
//   loom-synth-fu-dump --config=<path.yaml> <input.mlir>
//   loom-synth-fu-dump --configured-feedback -
//   loom-synth-fu-dump --print-ir=false --print-stats=false <input.mlir>
//   loom-synth-fu-dump --quiet <input.mlir>
//
// The helper drives the pass directly via `mlir::PassManager`; it does
// not shell out to `loom`. Diagnostics emitted by the pass are captured
// via a `ScopedDiagnosticHandler` so we can format them in our output
// without polluting stderr (and so wall-time measurement is not
// confounded by stream-flush latency in CHECK pipelines).

#include "Dataflow/IR/DataflowDialect.h"
#include "Dataflow/IR/DataflowOps.h"
#include "Fabric/IR/ConfiguredFunction.h"
#include "Fabric/IR/FabricDialect.h"
#include "Fabric/IR/FabricOps.h"
#include "Fabric/Tech/Passes.h"
#include "Fabric/Tech/Synthesizer/Synthesizer.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

#include <chrono>
#include <string>
#include <vector>

namespace {

// CLI options. Defaults match the task spec: print IR + stats + wallclock,
// do not escalate failures, no config (built-in defaults).
::llvm::cl::opt<std::string>
    inputPath(::llvm::cl::Positional,
              ::llvm::cl::desc("<input MLIR file or '-' for stdin>"),
              ::llvm::cl::Required);

::llvm::cl::opt<bool> configuredFeedback(
    "configured-feedback",
    ::llvm::cl::desc("Synthesize a cyclic ConfiguredFunction carry anchor"),
    ::llvm::cl::init(false));

::llvm::cl::opt<std::string> configPath(
    "config",
    ::llvm::cl::desc("Path to a SynthConfig YAML/TOML file. Empty = defaults."),
    ::llvm::cl::init(""));

::llvm::cl::opt<bool> failAsError(
    "fail-as-error",
    ::llvm::cl::desc(
        "Escalate per-group synthesis warnings to errors (default false)"),
    ::llvm::cl::init(false));

::llvm::cl::opt<bool> printIr(
    "print-ir",
    ::llvm::cl::desc("Print the synthesized FU IR per group (default true)"),
    ::llvm::cl::init(true));

::llvm::cl::opt<bool> printStats(
    "print-stats",
    ::llvm::cl::desc(
        "Print the canonical `synth-stat` line per group (default true)"),
    ::llvm::cl::init(true));

::llvm::cl::opt<bool> printWallclock(
    "print-wallclock",
    ::llvm::cl::desc("Print the `wallclock_us=<N>` line (default true)"),
    ::llvm::cl::init(true));

::llvm::cl::opt<bool> quiet(
    "quiet",
    ::llvm::cl::desc("Suppress non-essential output; stats are still printed"),
    ::llvm::cl::init(false));

// Diagnostic captured during the pass run. We retain just the message
// text for `synth-stat` lines and the severity for non-stat
// diagnostics so the helper can mirror them on stderr (when not in
// `--quiet` mode) without rerunning the pass.
struct CapturedDiagnostic {
  ::mlir::DiagnosticSeverity severity;
  std::string message;
};

// Returns true iff `text` looks like a canonical `synth-stat` line.
// The pass embeds the literal `synth-stat ` prefix at the start of the
// remark text; we detect it as a substring because MLIR's diagnostic
// handler may prepend location text.
bool isSynthStatLine(::llvm::StringRef text) {
  return text.find("synth-stat ") != ::llvm::StringRef::npos;
}

// Slice `text` from the first occurrence of `synth-stat ` to the end of
// the string (trimming trailing whitespace). Returns an empty
// `StringRef` if the marker is absent.
::llvm::StringRef extractSynthStatPayload(::llvm::StringRef text) {
  size_t pos = text.find("synth-stat ");
  if (pos == ::llvm::StringRef::npos)
    return {};
  ::llvm::StringRef tail = text.substr(pos);
  return tail.rtrim();
}

// Convert MLIR severity to a stable lowercase string for non-stat
// diagnostic mirror lines. Mirrors `lit`-friendly severity names.
::llvm::StringRef severityName(::mlir::DiagnosticSeverity sev) {
  switch (sev) {
  case ::mlir::DiagnosticSeverity::Error:
    return "error";
  case ::mlir::DiagnosticSeverity::Warning:
    return "warning";
  case ::mlir::DiagnosticSeverity::Remark:
    return "remark";
  case ::mlir::DiagnosticSeverity::Note:
    return "note";
  }
  return "diag";
}

int runConfiguredFeedback(::mlir::MLIRContext &context) {
  auto i1 = ::mlir::IntegerType::get(&context, 1);
  auto i32 = ::mlir::IntegerType::get(&context, 32);

  ::fabric::ConfiguredFunction function;
  function.inputs.push_back({0, i1});
  function.inputs.push_back({1, i32});

  ::fabric::ConfiguredFunctionNode carry;
  carry.fabricResource = 0;
  carry.operationName = ::dataflow::CarryOp::getOperationName().str();
  carry.functionType =
      ::mlir::FunctionType::get(&context, {i1, i32, i32}, {i32});
  carry.attributes = ::mlir::DictionaryAttr::get(&context);
  carry.operands.push_back(::fabric::ConfiguredValue::input(0));
  carry.operands.push_back(::fabric::ConfiguredValue::input(1));
  carry.operands.push_back(::fabric::ConfiguredValue::nodeResult(0, 0));
  function.nodes.push_back(std::move(carry));
  function.outputs.push_back(
      {0, i32, ::fabric::ConfiguredValue::nodeResult(0, 0)});

  ::llvm::SmallVector<::fabric::ConfiguredFunction, 1> functions;
  functions.push_back(std::move(function));
  ::loom::SynthConfig config;
  ::loom::fabric::tech::SynthInputs inputs{
      /*groupName=*/"feedback",
      /*functions=*/functions,
      /*context=*/&context,
  };

  auto start = std::chrono::steady_clock::now();
  ::loom::fabric::tech::SynthResult result =
      ::loom::fabric::tech::synthesize(config, inputs);
  auto stop = std::chrono::steady_clock::now();
  if (!result.success()) {
    ::llvm::errs() << "loom-synth-fu-dump: configured-feedback failed: "
                   << ::loom::fabric::tech::failureReasonString(
                          result.failureReason)
                   << "\n";
    for (const std::string &note : result.notes)
      ::llvm::errs() << "loom-synth-fu-dump: note: " << note << "\n";
    return 1;
  }

  ::llvm::outs() << "configured-feedback: success\n";
  if (printIr.getValue() && !quiet.getValue()) {
    result.wrapper->print(::llvm::outs());
    ::llvm::outs() << "\n";
  }
  if (printStats.getValue())
    ::llvm::outs() << "// no synth-stat lines emitted\n";
  if (printWallclock.getValue()) {
    auto wallUs =
        std::chrono::duration_cast<std::chrono::microseconds>(stop - start)
            .count();
    ::llvm::outs() << "wallclock_us=" << wallUs << "\n";
  }
  return 0;
}

} // namespace

int main(int argc, char **argv) {
  ::llvm::cl::ParseCommandLineOptions(
      argc, argv,
      "loom-synth-fu-dump: run loom-synthesize-configured-functions on the "
      "input module and print the synthesized FUs, the canonical "
      "synth-stat lines, and the wall-clock cost.\n");

  // Register every standard MLIR dialect plus the Loom dialects used by
  // configured-function fixtures and synthesized outputs.
  ::mlir::DialectRegistry registry;
  ::mlir::registerAllDialects(registry);
  registry.insert<::fabric::FabricDialect, ::dataflow::DataflowDialect>();
  ::mlir::MLIRContext ctx(registry);
  ctx.loadAllAvailableDialects();

  if (configuredFeedback.getValue())
    return runConfiguredFeedback(ctx);

  // Parse the input module from file or stdin.
  auto bufOrErr = ::llvm::MemoryBuffer::getFileOrSTDIN(inputPath.getValue());
  if (auto ec = bufOrErr.getError()) {
    ::llvm::errs() << "loom-synth-fu-dump: error: failed to read \""
                   << inputPath.getValue() << "\": " << ec.message() << "\n";
    return 1;
  }
  ::llvm::SourceMgr sm;
  sm.AddNewSourceBuffer(std::move(*bufOrErr), ::llvm::SMLoc());
  ::mlir::OwningOpRef<::mlir::ModuleOp> mod =
      ::mlir::parseSourceFile<::mlir::ModuleOp>(sm, &ctx);
  if (!mod) {
    ::llvm::errs() << "loom-synth-fu-dump: error: parse failed\n";
    return 1;
  }

  // Capture diagnostics emitted by the pass. We extract `synth-stat`
  // lines for the dedicated stats section and forward the rest to
  // stderr (mirroring `loom`'s behavior) unless `--quiet` is set.
  std::vector<CapturedDiagnostic> diagnostics;
  ::mlir::ScopedDiagnosticHandler diagHandler(&ctx, [&](::mlir::Diagnostic &d) {
    std::string text;
    ::llvm::raw_string_ostream os(text);
    os << d;
    os.flush();
    diagnostics.push_back({d.getSeverity(), std::move(text)});
    // Returning `success()` marks the diagnostic as fully handled
    // so MLIR's default printer does not also dump it.
    return ::mlir::success();
  });

  // Build a single-pass PassManager around the synthesizer pass.
  ::mlir::PassManager pm(&ctx);
  pm.addPass(::fabric::createSynthesizeConfiguredFunctionsPass(
      configPath.getValue(), failAsError.getValue(),
      /*dumpStats=*/printStats.getValue()));

  // Time only the pass invocation. Module parsing, dialect registration,
  // and post-pass printing are excluded so the perf budget is on
  // synthesis itself.
  auto t0 = std::chrono::steady_clock::now();
  ::mlir::LogicalResult passResult = pm.run(*mod);
  auto t1 = std::chrono::steady_clock::now();
  auto wallUs =
      std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

  // Mirror non-stat diagnostics on stderr unless suppressed. The stat
  // lines are reserved for the stdout `synth-stat` block below so that
  // FileCheck pipelines can pin them deterministically.
  if (!quiet.getValue()) {
    for (const auto &diag : diagnostics) {
      if (diag.severity == ::mlir::DiagnosticSeverity::Remark &&
          isSynthStatLine(diag.message))
        continue;
      ::llvm::errs() << "loom-synth-fu-dump: " << severityName(diag.severity)
                     << ": " << diag.message << "\n";
    }
  }

  // Print the synthesized FU IR for each wrapper. Using the
  // `loom.synthesized_for` attribute means we pick up exactly the
  // outputs the pass appended (and ignore pre-existing helper
  // ops). Wrappers are printed in their final module order, which is
  // lexically sorted by group name per the pass's splice rule. Both
  // canonical fabric.module outputs carrying the marker attribute are
  // printed.
  if (printIr.getValue() && !quiet.getValue()) {
    bool any = false;
    for (::mlir::Operation &op : mod->getBody()->getOperations()) {
      if (!op.hasAttr("loom.synthesized_for"))
        continue;
      if (!::mlir::isa<::fabric::ModuleOp, ::mlir::func::FuncOp>(op))
        continue;
      if (!any) {
        ::llvm::outs() << "// --- synthesized FUs ---\n";
        any = true;
      }
      op.print(::llvm::outs());
      ::llvm::outs() << "\n";
    }
    if (!any)
      ::llvm::outs() << "// no synthesized FUs in module\n";
  }

  // Print the captured `synth-stat` block. One line per group, in the
  // order the pass emitted them (which is lexically sorted group order
  // per the pass's determinism rules).
  if (printStats.getValue()) {
    bool any = false;
    for (const auto &diag : diagnostics) {
      if (diag.severity != ::mlir::DiagnosticSeverity::Remark)
        continue;
      if (!isSynthStatLine(diag.message))
        continue;
      if (!any) {
        ::llvm::outs() << "// --- synth stats ---\n";
        any = true;
      }
      ::llvm::outs() << extractSynthStatPayload(diag.message) << "\n";
    }
    if (!any)
      ::llvm::outs() << "// no synth-stat lines emitted\n";
  }

  if (printWallclock.getValue())
    ::llvm::outs() << "wallclock_us=" << wallUs << "\n";

  // The helper exits zero on a successful pass run even when individual
  // groups failed (the pass communicates per-group failures via
  // `loom.synth_failed`, not via the pass return code) unless
  // `--fail-as-error` was supplied, in which case we honor the pass's
  // error signal.
  if (::mlir::failed(passResult) && failAsError.getValue())
    return 2;
  return 0;
}
