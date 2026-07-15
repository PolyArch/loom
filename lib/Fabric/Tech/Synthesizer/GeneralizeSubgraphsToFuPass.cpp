// Synthesize one fabric FU per `loom.synth_group` from functions containing
// one dataflow.subgraph. Strategy execution may run in parallel, while
// deterministic validation, diagnostics, and module mutation remain serial.

#include "Fabric/Tech/Passes.h"

#include "Common/SynthConfig.h"
#include "Dataflow/IR/DataflowDialect.h"
#include "Dataflow/IR/DataflowOps.h"
#include "Fabric/IR/ConfiguredFunction.h"
#include "Fabric/IR/FabricDialect.h"
#include "Fabric/IR/FabricOps.h"
#include "Fabric/Tech/ConfiguredFunctionAdapters.h"
#include "Fabric/Tech/Synthesizer/Anchor.h"
#include "Fabric/Tech/Synthesizer/CostModel.h"
#include "Fabric/Tech/Synthesizer/CoverageVerifier.h"
#include "Fabric/Tech/Synthesizer/Parallel.h"
#include "Fabric/Tech/Synthesizer/Synthesizer.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cctype>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace {

// One synth group's canonical functions and parent operations.
struct SynthGroup {
  std::string name;
  ::llvm::SmallVector<::fabric::ConfiguredFunction, 4> functions;
  ::llvm::SmallVector<::mlir::func::FuncOp, 4> parents;
};

struct ValidatedInput {
  ::mlir::func::FuncOp parent;
  ::fabric::ConfiguredFunction function;
};

// Sanitize a group name into a symbol-safe token: replace any character
// outside `[A-Za-z0-9_]` with `_`. Matches the spec rule for
// `@fu_<sanitized(group)>`.
static std::string sanitizeGroupName(::llvm::StringRef name) {
  std::string out;
  out.reserve(name.size());
  for (char c : name) {
    bool ok = (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') ||
              (c >= '0' && c <= '9') || c == '_';
    out.push_back(ok ? c : '_');
  }
  return out;
}

// Construct the canonical wrapper symbol name for a group.
static std::string wrapperNameFor(::llvm::StringRef groupName) {
  std::string out = "fu_";
  out += sanitizeGroupName(groupName);
  return out;
}

// Side-channel paired with each per-group `SynthResult`. Carries the
// printed IR text of a successful wrapper so the main thread can
// re-parse it under the user's `MLIRContext` before splicing. Empty
// `wrapperIR` means either failure or that the worker did not produce
// a printable wrapper. See `runOnOperation`'s splice loop for details.
struct WorkerHandoff {
  ::loom::fabric::tech::SynthResult result;
  std::string wrapperIR;
};

// Run the one externally selectable canonical synthesis path.
static ::loom::fabric::tech::SynthResult
runCanonicalSynthesis(const ::loom::SynthConfig &cfg,
                      const ::loom::fabric::tech::SynthInputs &inputs) {
  using ::loom::fabric::tech::makeSynthesizer;
  using ::loom::fabric::tech::SynthResult;

  auto primary = makeSynthesizer(cfg.strategy, cfg);
  if (!primary) {
    SynthResult r;
    r.failureReason = ::loom::fabric::tech::SynthFailureReason::InvalidInput;
    std::string note;
    {
      ::llvm::raw_string_ostream os(note);
      os << "unknown strategy '" << cfg.strategy << "'";
    }
    r.notes.push_back(std::move(note));
    return r;
  }

  return primary->run(inputs);
}

// Walk a fabric.fu body and count its top-level fabric.op / mux /
// demux operations. Returned as a triple in the order printed by
// `synth-stat` (op / mux / demux).
struct NodeCounts {
  unsigned ops = 0;
  unsigned muxes = 0;
  unsigned demuxes = 0;
};
static NodeCounts countFuBodyNodes(::fabric::FuOp fu) {
  NodeCounts c;
  if (!fu)
    return c;
  for (::mlir::Operation &op : fu.getBody().front().getOperations()) {
    if (::mlir::isa<::fabric::OpOp>(op))
      ++c.ops;
    else if (::mlir::isa<::fabric::MuxOp>(op))
      ++c.muxes;
    else if (::mlir::isa<::fabric::DemuxOp>(op))
      ++c.demuxes;
  }
  return c;
}

// Locate the inner fabric.fu inside a wrapper fabric.module. The
// wrapper is required to contain exactly one fabric.fu (transitively,
// nested inside a fabric.pe); returns null if the wrapper is missing
// or malformed.
static ::fabric::FuOp findInnerFu(::fabric::ModuleOp wrapper) {
  if (!wrapper)
    return nullptr;
  ::fabric::FuOp found;
  wrapper.walk([&](::fabric::FuOp fu) {
    if (!found)
      found = fu;
  });
  return found;
}

static void enforceCanonicalSynthesisGate(
    ::loom::fabric::tech::SynthResult &result,
    ::llvm::ArrayRef<::fabric::ConfiguredFunction> inputs,
    const ::loom::SynthConfig &cfg) {
  using ::loom::fabric::tech::SynthFailureReason;
  if (!result.success())
    return;

  ::fabric::FuOp fu = findInnerFu(result.wrapper.get());
  if (!fu || ::mlir::failed(::mlir::verify(result.wrapper.get()))) {
    result.wrapper = nullptr;
    result.failureReason = SynthFailureReason::VerifierFailed;
    result.notes.push_back(
        "canonical synthesis gate: wrapper or FU verification failed");
    return;
  }

  ::loom::fabric::tech::CoverageVerifier verifier(cfg);
  result.coverage = verifier.verify(fu, inputs);
  if (!result.coverage.allCovered()) {
    result.wrapper = nullptr;
    result.failureReason = SynthFailureReason::VerifierFailed;
    result.notes.push_back(
        "canonical synthesis gate: explicit encodings do not cover every "
        "input function");
    return;
  }
  result.capability =
      ::loom::fabric::tech::measureCapability(fu, result.coverage);
}

class GeneralizeSubgraphsToFuPass
    : public ::mlir::PassWrapper<GeneralizeSubgraphsToFuPass,
                                 ::mlir::OperationPass<::mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GeneralizeSubgraphsToFuPass)

  GeneralizeSubgraphsToFuPass() = default;
  GeneralizeSubgraphsToFuPass(const GeneralizeSubgraphsToFuPass &other)
      : ::mlir::PassWrapper<GeneralizeSubgraphsToFuPass,
                            ::mlir::OperationPass<::mlir::ModuleOp>>(other) {
    configPath = other.configPath;
    failAsError = other.failAsError;
    dumpStats = other.dumpStats;
  }
  GeneralizeSubgraphsToFuPass(std::string path, bool failAsErr, bool dump) {
    configPath = std::move(path);
    failAsError = failAsErr;
    dumpStats = dump;
  }

  ::llvm::StringRef getArgument() const final {
    return "loom-generalize-subgraphs-to-fu";
  }
  ::llvm::StringRef getDescription() const final {
    return "Synthesize one fabric.fu per loom.synth_group out of the "
           "input dataflow.subgraphs and append it (wrapped in a func.func) "
           "to the module.";
  }

  void getDependentDialects(::mlir::DialectRegistry &registry) const final {
    registry.insert<::mlir::func::FuncDialect, ::dataflow::DataflowDialect,
                    ::fabric::FabricDialect, ::mlir::arith::ArithDialect,
                    ::mlir::LLVM::LLVMDialect, ::mlir::math::MathDialect>();
  }

  Option<std::string> configPath{
      *this, "config",
      ::llvm::cl::desc(
          "Path to a YAML or TOML SynthConfig file. Empty selects defaults."),
      ::llvm::cl::init("")};
  Option<bool> failAsError{
      *this, "fail-as-error",
      ::llvm::cl::desc(
          "Escalate per-group synthesis warnings to errors (default false)."),
      ::llvm::cl::init(false)};
  Option<bool> dumpStats{
      *this, "dump-stats",
      ::llvm::cl::desc("Print one canonical `synth-stat` remark per group "
                       "(default false)."),
      ::llvm::cl::init(false)};

  void runOnOperation() final;

private:
  // Per-func.func input validation. Populates `valid` (one entry per
  // surviving function) and annotates rejected functions with
  // `loom.synth_failed = "invalid_input"`.
  void validateFunctions(::mlir::ModuleOp module,
                         ::llvm::SmallVectorImpl<ValidatedInput> &valid);

  // Bucket surviving (function, subgraph) pairs by the parent
  // func.func's `loom.synth_group` attribute (default == `"default"`).
  // The returned vector is sorted lexically by group name.
  ::llvm::SmallVector<SynthGroup, 4>
  collectGroups(::llvm::ArrayRef<ValidatedInput> valid);

  // Detect symbol conflict / idempotent re-synth for a group before
  // strategy invocation. Returns true iff the splice loop should
  // still try to synthesize this group; false means the group has
  // been handled by either the symbol-conflict failure path or the
  // idempotent-skip remark.
  bool prepareSymbolSlot(::mlir::ModuleOp module, const SynthGroup &group,
                         const ::loom::SynthConfig &cfg);

  // Emit the canonical `synth-stat` remark for one group.
  void emitSynthStat(::mlir::ModuleOp module, const SynthGroup &group,
                     const ::loom::fabric::tech::SynthResult &result,
                     const ::loom::SynthConfig &cfg);

  // Annotate every input function in the group with
  // `loom.synth_failed = <reason>`.
  void annotateGroupFailure(const SynthGroup &group, ::llvm::StringRef reason);
};

} // namespace

void GeneralizeSubgraphsToFuPass::validateFunctions(
    ::mlir::ModuleOp module, ::llvm::SmallVectorImpl<ValidatedInput> &valid) {
  auto *ctx = &getContext();
  auto invalidAttr = ::mlir::StringAttr::get(
      ctx, ::loom::fabric::tech::failureReasonString(
               ::loom::fabric::tech::SynthFailureReason::InvalidInput));

  for (auto func : module.getOps<::mlir::func::FuncOp>()) {
    // Skip the synthesizer's own previous-run output -- wrapper
    // func.funcs tagged with `loom.synthesized_for` are not inputs to
    // the current invocation. Symbol-conflict precheck still inspects
    // them later via the SymbolTable.
    if (func->hasAttr("loom.synthesized_for"))
      continue;
    // Private-visibility func.funcs are also treated as bystanders
    // (helper / library symbols), regardless of whether they contain a
    // fabric.fu. This keeps the symbol-conflict precheck able to see
    // them while still excluding them from input validation.
    if (func.isPrivate())
      continue;

    ::llvm::SmallVector<::dataflow::SubgraphOp, 2> subgraphs;
    func.walk([&](::dataflow::SubgraphOp sg) { subgraphs.push_back(sg); });

    // The remaining functions are user inputs. Each must contain
    // exactly one dataflow.subgraph in its body; zero or many is
    // `invalid_input`.
    if (subgraphs.size() != 1) {
      func->setAttr("loom.synth_failed", invalidAttr);
      ::mlir::InFlightDiagnostic diag =
          func.emitWarning()
          << "func.func @" << func.getName() << ": invalid_input "
          << "(expected exactly one dataflow.subgraph in body, got "
          << subgraphs.size() << ")";
      (void)diag;
      continue;
    }

    ::fabric::ConfiguredFunction function;
    std::string adapterError;
    if (::mlir::failed(::fabric::configuredFunctionFromSubgraph(
            subgraphs.front(), function, adapterError))) {
      func->setAttr("loom.synth_failed", invalidAttr);
      func.emitWarning() << "func.func @" << func.getName()
                         << ": invalid_input (" << adapterError << ")";
      continue;
    }
    valid.push_back({func, std::move(function)});
  }
}

::llvm::SmallVector<SynthGroup, 4> GeneralizeSubgraphsToFuPass::collectGroups(
    ::llvm::ArrayRef<ValidatedInput> valid) {
  // Use std::map for lexically-sorted iteration; the spec requires
  // groups to be processed in lexical name order both for parallel
  // dispatch results and for the splice loop.
  std::map<std::string, SynthGroup> table;
  for (const ValidatedInput &input : valid) {
    auto func = input.parent;
    std::string name = "default";
    if (auto attr = func->getAttrOfType<::mlir::StringAttr>("loom.synth_group"))
      name = attr.getValue().str();
    auto &slot = table[name];
    if (slot.name.empty())
      slot.name = name;
    slot.functions.push_back(input.function);
    slot.parents.push_back(func);
  }
  ::llvm::SmallVector<SynthGroup, 4> groups;
  groups.reserve(table.size());
  for (auto &entry : table)
    groups.push_back(std::move(entry.second));
  return groups;
}

// Validate that a marker-tagged wrapper module is a real synthesized
// wrapper. Performs three checks in order:
//   Body shape: fabric.module body is exactly `[fabric.pe,
//       fabric.yield]` (one fabric.pe plus the module's terminating
//       fabric.yield; no extra ops); the fabric.pe body is exactly
//       `[fabric.fu]` (one inner FU; no other compute member).
//   Inner verifier: the inner fabric.fu passes its own verifier
//       (FuOp::verify and any nested op verifiers reachable from
//       `mlir::verify`).
//   Signature match: the wrapper's signature (operand types + FU
//       result types) matches the expected signature derived from
//       the canonical input functions via `collectWrapperPorts`.
//
// Returns an empty string on success. Returns a deterministic failure
// reason note (suitable for an attached note on the diag) when any
// check fails. The caller treats any non-empty return as a
// `symbol_conflict` failure.
static std::string
validateMarkerWrapper(::fabric::ModuleOp existingModule,
                      ::llvm::StringRef symbolName, ::llvm::StringRef groupName,
                      ::llvm::ArrayRef<::fabric::ConfiguredFunction> functions,
                      ::mlir::MLIRContext *ctx,
                      const ::loom::SynthConfig &cfg) {
  std::string note;
  ::llvm::raw_string_ostream os(note);

  // Body shape: fabric.module body must contain exactly one fabric.pe
  // plus the module's fabric.yield terminator; the fabric.pe body must
  // contain exactly one fabric.fu.
  if (existingModule.getBody().empty()) {
    os << "symbol_conflict (existing @" << symbolName
       << " tagged loom.synthesized_for=\"" << groupName
       << "\" but body shape is malformed: expected fabric.module body "
          "[fabric.pe, fabric.yield], found empty body "
          "[wrapper-body-shape])";
    return os.str();
  }
  ::mlir::Block &moduleEntry = existingModule.getBody().front();
  unsigned peCount = 0;
  ::fabric::PeOp innerPe;
  for (::mlir::Operation &op : moduleEntry.getOperations()) {
    if (auto pe = ::mlir::dyn_cast<::fabric::PeOp>(op)) {
      ++peCount;
      if (!innerPe)
        innerPe = pe;
    }
  }
  ::mlir::Operation *moduleTerminator = moduleEntry.getTerminator();
  bool moduleTerminatorIsYield =
      moduleTerminator && ::mlir::isa<::fabric::YieldOp>(moduleTerminator);
  unsigned moduleNumOps = 0;
  for (::mlir::Operation &op : moduleEntry.getOperations()) {
    (void)op;
    ++moduleNumOps;
  }
  if (peCount != 1 || !moduleTerminatorIsYield || moduleNumOps != 2) {
    os << "symbol_conflict (existing @" << symbolName
       << " tagged loom.synthesized_for=\"" << groupName
       << "\" but body shape is malformed: expected fabric.module body "
          "[fabric.pe, fabric.yield], found "
       << peCount << " fabric.pe op(s) [wrapper-body-shape])";
    return os.str();
  }
  ::mlir::Block &peEntry = innerPe.getBody().front();
  unsigned fuCount = 0;
  ::fabric::FuOp innerFu;
  unsigned peNumOps = 0;
  for (::mlir::Operation &op : peEntry.getOperations()) {
    ++peNumOps;
    if (auto fu = ::mlir::dyn_cast<::fabric::FuOp>(op)) {
      ++fuCount;
      if (!innerFu)
        innerFu = fu;
    }
  }
  if (fuCount != 1 || peNumOps != 1) {
    os << "symbol_conflict (existing @" << symbolName
       << " tagged loom.synthesized_for=\"" << groupName
       << "\" but body shape is malformed: expected fabric.pe body "
          "[fabric.fu], found "
       << fuCount << " fabric.fu op(s) [wrapper-body-shape])";
    return os.str();
  }

  // Inner verifier: verify the inner fabric.fu in isolation. Capture
  // diagnostics through a ScopedDiagnosticHandler so the verifier's
  // error output does not leak to stderr; surface them as part of the
  // attached note instead. Multiple diagnostics are joined with `; `
  // for a single deterministic line.
  ::llvm::SmallVector<std::string, 2> diagMsgs;
  {
    ::mlir::ScopedDiagnosticHandler capture(ctx, [&](::mlir::Diagnostic &d) {
      diagMsgs.push_back(d.str());
      return ::mlir::success();
    });
    if (::mlir::failed(::mlir::verify(innerFu))) {
      std::string joined;
      for (auto [i, m] : ::llvm::enumerate(diagMsgs)) {
        if (i)
          joined += "; ";
        joined += m;
      }
      os << "symbol_conflict (existing @" << symbolName
         << " tagged loom.synthesized_for=\"" << groupName
         << "\" but inner fabric.fu fails verification: "
         << (joined.empty() ? std::string("(no diagnostic)") : joined)
         << " [inner-fu-verifier])";
      return os.str();
    }
  }

  // Signature match: the expected signature is the physical lift of the
  // canonical function boundaries. The wrapper's input
  // surface lives on the fabric.module's entry-block argument types;
  // the wrapper's "result" surface (per the synthesizer contract) is
  // the inner fabric.fu's result type list, since fabric.module itself
  // declares no SSA results.
  auto portsOpt = ::loom::fabric::tech::collectWrapperPorts(functions, ctx);
  if (!portsOpt.has_value()) {
    os << "symbol_conflict (existing @" << symbolName
       << " tagged loom.synthesized_for=\"" << groupName
       << "\" but expected signature could not be derived from the input "
          "functions (boundary types are not lift-able) "
          "[signature-mismatch])";
    return os.str();
  }
  ::mlir::FunctionType actual = existingModule.getFunctionType();
  ::llvm::ArrayRef<::mlir::Type> actualInputs = actual.getInputs();
  ::llvm::SmallVector<::mlir::Type, 4> actualResults(
      innerFu.getOutputs().getTypes().begin(),
      innerFu.getOutputs().getTypes().end());
  ::llvm::ArrayRef<::mlir::Type> expectedInputs(portsOpt->inputs);
  ::llvm::ArrayRef<::mlir::Type> expectedResults(portsOpt->outputs);
  bool inputsMatch = actualInputs.size() == expectedInputs.size() &&
                     std::equal(actualInputs.begin(), actualInputs.end(),
                                expectedInputs.begin());
  bool resultsMatch = actualResults.size() == expectedResults.size() &&
                      std::equal(actualResults.begin(), actualResults.end(),
                                 expectedResults.begin());
  if (!inputsMatch || !resultsMatch) {
    auto printTypes = [](::llvm::raw_string_ostream &s,
                         ::llvm::ArrayRef<::mlir::Type> ts) {
      s << "(";
      for (auto [i, t] : ::llvm::enumerate(ts)) {
        if (i)
          s << ", ";
        t.print(s);
      }
      s << ")";
    };
    std::string expectedStr;
    std::string actualStr;
    {
      ::llvm::raw_string_ostream eo(expectedStr);
      printTypes(eo, expectedInputs);
      eo << " -> ";
      printTypes(eo, expectedResults);
    }
    {
      ::llvm::raw_string_ostream ao(actualStr);
      printTypes(ao, actualInputs);
      ao << " -> ";
      printTypes(ao, actualResults);
    }
    os << "symbol_conflict (existing @" << symbolName
       << " tagged loom.synthesized_for=\"" << groupName
       << "\" but signature mismatch: expected " << expectedStr << ", got "
       << actualStr << " [signature-mismatch])";
    return os.str();
  }

  ::loom::fabric::tech::CoverageVerifier coverageVerifier(cfg);
  auto coverage = coverageVerifier.verify(innerFu, functions);
  if (!coverage.allCovered()) {
    os << "symbol_conflict (existing @" << symbolName
       << " tagged loom.synthesized_for=\"" << groupName
       << "\" but semantic coverage failed: explicit valid encodings do not "
          "cover every input function [semantic-coverage])";
    return os.str();
  }

  return std::string();
}

bool GeneralizeSubgraphsToFuPass::prepareSymbolSlot(
    ::mlir::ModuleOp module, const SynthGroup &group,
    const ::loom::SynthConfig &cfg) {
  std::string symbolName = wrapperNameFor(group.name);
  auto *existing = ::mlir::SymbolTable::lookupSymbolIn(module, symbolName);
  if (!existing)
    return true;

  auto existingModule = ::mlir::dyn_cast<::fabric::ModuleOp>(existing);
  if (existingModule) {
    if (auto tag = existingModule->getAttrOfType<::mlir::StringAttr>(
            "loom.synthesized_for")) {
      if (tag.getValue() == group.name) {
        // Marker-tagged wrapper. Validate that it is a real synthesized
        // wrapper (body shape, inner fabric.fu verifier, signature
        // match) before honoring it as idempotent. A failed check is
        // reported as a `symbol_conflict` so the user can resolve the
        // malformed wrapper rather than silently accepting it as a
        // no-op.
        std::string failureNote = validateMarkerWrapper(
            existingModule, symbolName, group.name,
            ::llvm::ArrayRef<::fabric::ConfiguredFunction>(group.functions),
            &getContext(), cfg);
        if (failureNote.empty()) {
          ::mlir::InFlightDiagnostic diag =
              module.emitRemark()
              << "loom-generalize-subgraphs-to-fu: group \"" << group.name
              << "\": skipping idempotent re-synth (existing @" << symbolName
              << " tagged loom.synthesized_for=\"" << group.name << "\")";
          (void)diag;
          return false;
        }
        // Validation failed: surface as a symbol_conflict.
        annotateGroupFailure(
            group,
            ::loom::fabric::tech::failureReasonString(
                ::loom::fabric::tech::SynthFailureReason::SymbolConflict));
        ::mlir::InFlightDiagnostic diag =
            failAsError ? module.emitError() : module.emitWarning();
        diag << "loom-generalize-subgraphs-to-fu: group \"" << group.name
             << "\": " << failureNote;
        if (failAsError)
          signalPassFailure();
        return false;
      }
    }
  }

  // Conflict: a non-synthesizer-owned symbol already exists at the
  // wrapper's name. Fail the group without invoking the strategy.
  annotateGroupFailure(
      group, ::loom::fabric::tech::failureReasonString(
                 ::loom::fabric::tech::SynthFailureReason::SymbolConflict));
  ::mlir::InFlightDiagnostic diag =
      failAsError ? module.emitError() : module.emitWarning();
  diag << "loom-generalize-subgraphs-to-fu: group \"" << group.name
       << "\": symbol_conflict (top-level symbol @" << symbolName
       << " already exists and is not tagged loom.synthesized_for=\""
       << group.name << "\")";
  if (failAsError)
    signalPassFailure();
  return false;
}

void GeneralizeSubgraphsToFuPass::annotateGroupFailure(
    const SynthGroup &group, ::llvm::StringRef reason) {
  auto *ctx = &getContext();
  auto attr = ::mlir::StringAttr::get(ctx, reason);
  for (auto func : group.parents)
    func->setAttr("loom.synth_failed", attr);
}

// Note: synth-stat emission lives inline at the end of the per-group
// splice loop in `runOnOperation` so the snapshot reads `result.wrapper`
// before the splice releases it. The class still declares
// `emitSynthStat` for future extensibility (e.g. multi-line per-strategy
// stats) but the canonical one-line format is built inline.
void GeneralizeSubgraphsToFuPass::emitSynthStat(
    ::mlir::ModuleOp /*module*/, const SynthGroup & /*group*/,
    const ::loom::fabric::tech::SynthResult & /*result*/,
    const ::loom::SynthConfig & /*cfg*/) {
  // Intentionally empty; replaced by inline snapshot in runOnOperation.
}

void GeneralizeSubgraphsToFuPass::runOnOperation() {
  ::mlir::ModuleOp module = getOperation();
  auto *ctx = &getContext();

  // Load config.
  ::loom::SynthConfig cfg;
  if (!configPath.empty()) {
    auto loaded = ::loom::loadSynthConfig(configPath);
    if (!loaded) {
      // Annotate every input func.func with config_parse_failed so
      // downstream tooling sees the failure even if the parent driver
      // swallows the diagnostic. Then signal pass failure.
      auto failedAttr = ::mlir::StringAttr::get(
          ctx,
          ::loom::fabric::tech::failureReasonString(
              ::loom::fabric::tech::SynthFailureReason::ConfigParseFailed));
      for (auto func : module.getOps<::mlir::func::FuncOp>()) {
        if (func->hasAttr("loom.synthesized_for"))
          continue;
        func->setAttr("loom.synth_failed", failedAttr);
      }
      ::mlir::InFlightDiagnostic diag =
          module.emitError()
          << "loom-generalize-subgraphs-to-fu: config_parse_failed: "
          << ::llvm::toString(loaded.takeError());
      (void)diag;
      return signalPassFailure();
    }
    cfg = *loaded;
  }

  // Validate input functions.
  ::llvm::SmallVector<ValidatedInput, 4> valid;
  validateFunctions(module, valid);

  // Empty input: emit a `no synth groups` remark and return. We
  // surface this even when there are invalid functions present (per
  // the spec, "empty input" is the absence of synthesizable subgraphs;
  // invalid functions are a separate failure path already annotated).
  if (valid.empty()) {
    module.emitRemark() << "loom-generalize-subgraphs-to-fu: no synth groups";
    return;
  }

  // Collect & sort groups lexically.
  auto groups = collectGroups(valid);
  if (groups.empty()) {
    module.emitRemark() << "loom-generalize-subgraphs-to-fu: no synth groups";
    return;
  }

  // Symbol-name precheck: resolve symbol collisions / idempotent
  // re-synth BEFORE strategy work. Functions that fail the precheck
  // are removed from the strategy queue. This is a deliberate
  // strict-superset of the spec's pseudocode (which only checks on
  // success); doing it up front avoids wasted strategy work and lets
  // us test the symbol-conflict / idempotent paths before the real
  // strategies land.
  ::llvm::SmallVector<SynthGroup, 4> queued;
  queued.reserve(groups.size());
  for (auto &g : groups) {
    if (prepareSymbolSlot(module, g, cfg))
      queued.push_back(std::move(g));
  }
  if (queued.empty())
    return;

  // Parallel-dispatch synthesis. Each worker runs canonical Anchor synthesis
  // on its group's ConfiguredFunctions in a thread-local scratch MLIRContext.
  // The strategy never sees
  // the user's MLIRContext: that would race on `StorageUniquer`,
  // attribute interning, and type uniquing as soon as a real strategy
  // begins building IR. To hand the wrapper back to the main thread
  // safely, the worker prints the wrapper to text under its scratch
  // context and discards both the wrapper and the scratch context
  // before returning. The main thread re-parses that text under the
  // user's context below.
  ::loom::fabric::tech::WorkerPool pool(
      cfg.parallelismCrossGroup ? cfg.parallelismWorkers : 1u);

  // Per-group canonical input bundle borrowed by a worker.
  struct WorkerJob {
    ::llvm::StringRef groupName;
    ::llvm::ArrayRef<::fabric::ConfiguredFunction> functions;
  };
  ::llvm::SmallVector<WorkerJob, 4> jobs;
  jobs.reserve(queued.size());
  for (auto &g : queued)
    jobs.push_back(
        WorkerJob{::llvm::StringRef(g.name),
                  ::llvm::ArrayRef<::fabric::ConfiguredFunction>(g.functions)});

  auto handoffs = pool.parallelMap<WorkerJob, WorkerHandoff>(
      ::llvm::ArrayRef<WorkerJob>(jobs), [&cfg](const WorkerJob &job) {
        // Thread-local scratch context. Required dialects must be
        // loaded explicitly: scratch contexts do not inherit anything
        // from the parent pass's context.
        ::mlir::MLIRContext scratch;
        ::mlir::DialectRegistry registry;
        registry.insert<::mlir::func::FuncDialect, ::dataflow::DataflowDialect,
                        ::fabric::FabricDialect, ::mlir::arith::ArithDialect,
                        ::mlir::LLVM::LLVMDialect, ::mlir::math::MathDialect>();
        scratch.appendDialectRegistry(registry);
        scratch.loadAllAvailableDialects();

        ::loom::fabric::tech::SynthInputs in{job.groupName, job.functions, cfg,
                                             &scratch};
        ::loom::fabric::tech::SynthResult res = runCanonicalSynthesis(cfg, in);
        enforceCanonicalSynthesisGate(res, job.functions, cfg);

        WorkerHandoff out;
        if (res.success()) {
          // Print the wrapper into the worker's local string under the
          // scratch context. `useLocalScope` keeps SSA numbering
          // self-contained; `assumeVerified` skips a redundant verify.
          ::llvm::raw_string_ostream os(out.wrapperIR);
          ::mlir::OpPrintingFlags flags;
          flags.useLocalScope().assumeVerified();
          res.wrapper->print(os, flags);
          os.flush();
          // Drop the scratch-context-owned wrapper before returning.
          // The main thread will re-parse `wrapperIR` into the user's
          // context. We preserve the rest of `res` (failureReason,
          // coverage, notes) verbatim.
          res.wrapper = nullptr;
        }
        out.result = std::move(res);
        // `scratch` goes out of scope here; any attribute / type
        // pointers that lived in it are now invalid. `out` carries
        // only POD-ish data plus the printed text.
        return out;
      });

  // Serial splice in lexical group order. This is the only place that
  // mutates the user's module after input validation. Cross-context
  // handoff happens here: each worker handed us the wrapper as printed
  // text (from a now-destroyed scratch context); we re-parse it under
  // the user's context so all attribute/type uniquing happens on the
  // pass's owning thread.
  ::mlir::OpBuilder modBuilder(module.getBody(), module.getBody()->end());
  for (size_t i = 0; i < queued.size(); ++i) {
    SynthGroup &group = queued[i];
    WorkerHandoff &handoff = handoffs[i];
    ::loom::fabric::tech::SynthResult &result = handoff.result;

    // Re-home the wrapper into the user's context if the worker
    // produced one. Parse failures here become a `verifier_failed`
    // demotion since the worker already produced verified IR but the
    // round-trip lost something.
    if (result.failureReason ==
            ::loom::fabric::tech::SynthFailureReason::None &&
        !handoff.wrapperIR.empty()) {
      ::mlir::OwningOpRef<::mlir::ModuleOp> parsed =
          ::mlir::parseSourceString<::mlir::ModuleOp>(handoff.wrapperIR, ctx);
      ::fabric::ModuleOp reHomed;
      if (parsed) {
        for (auto fn : parsed->getOps<::fabric::ModuleOp>()) {
          reHomed = fn;
          break;
        }
      }
      if (reHomed) {
        reHomed->remove();
        result.wrapper = ::mlir::OwningOpRef<::fabric::ModuleOp>(reHomed);
      } else {
        result.failureReason =
            ::loom::fabric::tech::SynthFailureReason::VerifierFailed;
        result.notes.push_back(
            "internal: failed to re-parse worker-built wrapper into the "
            "user's MLIRContext");
      }
    }

    // Capture a snapshot of the synth-stat metrics *before* the splice
    // releases the wrapper from `result.wrapper`. After release the
    // OwningOpRef is null, which would zero out the FU lookup that
    // emitSynthStat needs.
    bool snapshotSuccess = result.success();
    double snapshotCost = 0.0;
    unsigned snapshotCovered = 0;
    unsigned snapshotEncodings = 0;
    unsigned snapshotCoveredEncodings = 0;
    unsigned snapshotExtraCapability = 0;
    NodeCounts snapshotCounts;
    if (snapshotSuccess) {
      auto innerFu = findInnerFu(result.wrapper.get());
      if (innerFu) {
        ::loom::fabric::tech::CostModel cm(cfg);
        snapshotCost = cm.evaluate(innerFu);
        snapshotCounts = countFuBodyNodes(innerFu);
        snapshotEncodings = result.capability.encodingCount;
        snapshotCoveredEncodings = result.capability.coveredEncodingCount;
        snapshotExtraCapability = result.capability.extraCapabilityCount;
      }
      for (const auto &witness : result.coverage.witnesses)
        if (witness)
          ++snapshotCovered;
    }

    if (result.success()) {
      // Tag the wrapper with `loom.synthesized_for = "<group>"` so
      // future re-runs detect this as an idempotent slot.
      ::fabric::ModuleOp wrapper = result.wrapper.get();
      wrapper->setAttr("loom.synthesized_for",
                       ::mlir::StringAttr::get(ctx, group.name));
      // Splice into the module body. `release` transfers ownership
      // from the OwningOpRef to the module.
      ::mlir::Operation *raw = result.wrapper.release();
      modBuilder.insert(raw);
    } else {
      ::llvm::StringRef reason =
          ::loom::fabric::tech::failureReasonString(result.failureReason);
      auto attr = ::mlir::StringAttr::get(ctx, reason);
      for (::mlir::func::FuncOp parent : group.parents)
        parent->setAttr("loom.synth_failed", attr);

      ::mlir::InFlightDiagnostic diag =
          failAsError ? module.emitError() : module.emitWarning();
      diag << "loom-generalize-subgraphs-to-fu: group \"" << group.name
           << "\": synthesis failed: " << reason;
      for (const std::string &note : result.notes)
        diag.attachNote() << note;
      if (failAsError)
        signalPassFailure();
    }

    if (dumpStats) {
      // Build the canonical synth-stat line out of the pre-splice
      // snapshot so success cases retain their cost / coverage / node
      // metrics even though `result.wrapper` has been released.
      std::string line;
      ::llvm::raw_string_ostream os(line);
      os << "synth-stat group=" << group.name << " strategy=" << cfg.strategy
         << " reason=";
      if (snapshotSuccess)
        os << "success";
      else
        os << ::loom::fabric::tech::failureReasonString(result.failureReason);
      unsigned m = static_cast<unsigned>(group.functions.size());
      os << " cost=" << snapshotCost << " covered=" << snapshotCovered << "/"
         << m << " nodes=" << snapshotCounts.ops << "/" << snapshotCounts.muxes
         << "/" << snapshotCounts.demuxes << " encodings=" << snapshotEncodings
         << " covered_encodings=" << snapshotCoveredEncodings
         << " extra_capability=" << snapshotExtraCapability;
      module.emitRemark() << os.str();
    }
  }
}

namespace fabric {

// Header declares this in a different namespace; keep both in sync.
::std::unique_ptr<::mlir::Pass>
createGeneralizeSubgraphsToFuPass(std::string configPath, bool failAsError,
                                  bool dumpStats) {
  return std::make_unique<GeneralizeSubgraphsToFuPass>(std::move(configPath),
                                                       failAsError, dumpStats);
}

void registerFabricTechSynthesizerPasses() {
  ::mlir::registerPass([] { return createGeneralizeSubgraphsToFuPass(); });
}

} // namespace fabric
