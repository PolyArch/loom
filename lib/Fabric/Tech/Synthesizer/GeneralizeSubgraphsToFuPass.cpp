// Top-level entry point for `loom-generalize-subgraphs-to-fu`.
//
// The pass implements the inverse of `loom-enumerate-fu-subgraphs`:
// given a set of `dataflow.subgraph` instances grouped by the
// `loom.synth_group` attribute on their enclosing `func.func`, it
// dispatches per-group synthesis through a `SynthConfig`-driven
// strategy factory and splices the resulting wrapper `func.func`s into
// the user's module in lexical group-name order.
//
// Strategies returned by `makeSynthesizer` may be stubs (real
// implementations are wired in by their respective TUs); the pass exists
// to wire up:
//   * input validation -- each input func.func must contain exactly one
//     `dataflow.subgraph`; zero or many subgraphs -> `invalid_input`.
//   * group collection by `loom.synth_group` (default = "default").
//   * symbol-name precheck so pre-existing `@fu_<sanitized(group)>`
//     symbols are detected before strategy work runs:
//       - tagged with `loom.synthesized_for == group` -> idempotent
//         re-synth; emit a `remark` and skip the group entirely.
//       - otherwise -> annotate every input function in the group with
//         `loom.synth_failed = "symbol_conflict"`, emit a warning, and
//         skip strategy invocation.
//     This is a deliberate strict-superset of the spec pseudocode
//     (which runs the precheck only on success): doing the precheck
//     up front avoids wasted strategy work and is testable today
//     against the failing stub.
//   * parallel-dispatched synthesis via `WorkerPool::parallelMap` with
//     `cfg.parallelismWorkers` workers (worker count == 1 falls back to
//     inline execution). Each worker runs `run_with_fallback`: invoke
//     the primary strategy; on failure, walk `cfg.fallbackChain` in
//     order; return the most informative failure reason.
//   * serial splice in lexical group order (per
//     `Determinism rules`); MLIR mutation is never parallel.
//   * `dump-stats=true` emits one canonical `synth-stat` remark per
//     group so lit tests can assert against `cost / coverage / nodes`
//     without parsing IR.
//   * config-load failure aborts the pass, annotating every input
//     function with `loom.synth_failed = "config_parse_failed"`.
//
// Spec source: `docs/spec-generalize-subgraphs-to-fu.md`, sections
// `Pass`, `Acceptance criteria for the pass`, `Failure handling`, and
// `Determinism rules`.

#include "Fabric/Tech/Passes.h"

#include "Common/SynthConfig.h"
#include "Dataflow/IR/DataflowDialect.h"
#include "Dataflow/IR/DataflowOps.h"
#include "Fabric/IR/FabricDialect.h"
#include "Fabric/IR/FabricOps.h"
#include "Fabric/Tech/Synthesizer/Anchor.h"
#include "Fabric/Tech/Synthesizer/CostModel.h"
#include "Fabric/Tech/Synthesizer/FailureDiagnostic.h"
#include "Fabric/Tech/Synthesizer/Parallel.h"
#include "Fabric/Tech/Synthesizer/Synthesizer.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
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

// One synth group's pre-splice bookkeeping. We carry the lexical name
// (used as a stable sort key), the input subgraphs, and the parent
// func.func of each subgraph (for failure annotation).
struct SynthGroup {
  std::string name;
  ::llvm::SmallVector<::dataflow::SubgraphOp, 4> subgraphs;
  ::llvm::SmallVector<::mlir::func::FuncOp, 4> parents;
};

// Sanitize a group name into a symbol-safe token: replace any character
// outside `[A-Za-z0-9_]` with `_`. Matches the spec rule for
// `@fu_<sanitized(group)>`.
static std::string sanitizeGroupName(::llvm::StringRef name) {
  std::string out;
  out.reserve(name.size());
  for (char c : name) {
    bool ok =
        (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') ||
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

// Run the configured primary strategy and walk `fallbackChain` on
// failure. Returns the *primary's* failure reason on full failure (per
// the spec: "most informative: primary's failure reason"). The
// returned `SynthResult::wrapper` is owned by the caller via
// `OwningOpRef`.
static ::loom::fabric::tech::SynthResult
runWithFallback(const ::loom::SynthConfig &cfg,
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

  SynthResult primaryResult = primary->run(inputs);
  if (primaryResult.success())
    return primaryResult;

  for (const std::string &fallback : cfg.fallbackChain) {
    auto strat = makeSynthesizer(fallback, cfg);
    if (!strat)
      continue;
    SynthResult r = strat->run(inputs);
    if (r.success())
      return r;
  }
  return primaryResult;
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

// Locate the inner fabric.fu inside a wrapper func.func. The wrapper
// is required to contain exactly one fabric.fu in its body; returns
// null if the wrapper is missing or malformed.
static ::fabric::FuOp findInnerFu(::mlir::func::FuncOp wrapper) {
  if (!wrapper)
    return nullptr;
  if (wrapper.getBody().empty())
    return nullptr;
  for (::mlir::Operation &op : wrapper.getBody().front().getOperations())
    if (auto fu = ::mlir::dyn_cast<::fabric::FuOp>(op))
      return fu;
  return nullptr;
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
                    ::mlir::math::MathDialect>();
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
  void validateFunctions(
      ::mlir::ModuleOp module,
      ::llvm::SmallVectorImpl<::std::pair<::mlir::func::FuncOp,
                                          ::dataflow::SubgraphOp>> &valid);

  // Bucket surviving (function, subgraph) pairs by the parent
  // func.func's `loom.synth_group` attribute (default == `"default"`).
  // The returned vector is sorted lexically by group name.
  ::llvm::SmallVector<SynthGroup, 4> collectGroups(
      ::llvm::ArrayRef<::std::pair<::mlir::func::FuncOp,
                                   ::dataflow::SubgraphOp>> valid);

  // Detect symbol conflict / idempotent re-synth for a group before
  // strategy invocation. Returns true iff the splice loop should
  // still try to synthesize this group; false means the group has
  // been handled by either the symbol-conflict failure path or the
  // idempotent-skip remark.
  bool prepareSymbolSlot(::mlir::ModuleOp module, const SynthGroup &group);

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
    ::mlir::ModuleOp module,
    ::llvm::SmallVectorImpl<
        ::std::pair<::mlir::func::FuncOp, ::dataflow::SubgraphOp>> &valid) {
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
    func.walk(
        [&](::dataflow::SubgraphOp sg) { subgraphs.push_back(sg); });

    // The remaining functions are user inputs. Each must contain
    // exactly one dataflow.subgraph in its body; zero or many is
    // `invalid_input`.
    if (subgraphs.size() != 1) {
      func->setAttr("loom.synth_failed", invalidAttr);
      ::mlir::InFlightDiagnostic diag = func.emitWarning()
          << "func.func @" << func.getName() << ": invalid_input "
          << "(expected exactly one dataflow.subgraph in body, got "
          << subgraphs.size() << ")";
      (void)diag;
      continue;
    }

    valid.emplace_back(func, subgraphs.front());
  }
}

::llvm::SmallVector<SynthGroup, 4> GeneralizeSubgraphsToFuPass::collectGroups(
    ::llvm::ArrayRef<::std::pair<::mlir::func::FuncOp,
                                 ::dataflow::SubgraphOp>> valid) {
  // Use std::map for lexically-sorted iteration; the spec requires
  // groups to be processed in lexical name order both for parallel
  // dispatch results and for the splice loop.
  std::map<std::string, SynthGroup> table;
  for (const auto &pair : valid) {
    auto func = pair.first;
    auto sg = pair.second;
    std::string name = "default";
    if (auto attr = func->getAttrOfType<::mlir::StringAttr>("loom.synth_group"))
      name = attr.getValue().str();
    auto &slot = table[name];
    if (slot.name.empty())
      slot.name = name;
    slot.subgraphs.push_back(sg);
    slot.parents.push_back(func);
  }
  ::llvm::SmallVector<SynthGroup, 4> groups;
  groups.reserve(table.size());
  for (auto &entry : table)
    groups.push_back(std::move(entry.second));
  return groups;
}

// Validate that a marker-tagged wrapper function is a real synthesized
// wrapper. Performs three checks in order:
//   B1: body is exactly `[fabric.fu, func.return]` (one fabric.fu plus
//       a func.return terminator; no extra ops).
//   B2: the inner fabric.fu passes its own verifier (FuOp::verify and
//       any nested op verifiers reachable from `mlir::verify`).
//   B3: the wrapper's signature (operand types + result types) matches
//       the expected signature derived from the input subgraphs via
//       `collectWrapperPorts`.
//
// Returns an empty string on success. Returns a deterministic failure
// reason note (suitable for an attached note on the diag) when any of
// B1/B2/B3 fails. The caller treats any non-empty return as a
// `symbol_conflict` failure.
static std::string validateMarkerWrapper(
    ::mlir::func::FuncOp existingFunc, ::llvm::StringRef symbolName,
    ::llvm::StringRef groupName,
    ::llvm::ArrayRef<::dataflow::SubgraphOp> subgraphs,
    ::mlir::MLIRContext *ctx) {
  std::string note;
  ::llvm::raw_string_ostream os(note);

  // B1: body must contain exactly one fabric.fu plus a func.return
  // terminator. Empty body / no terminator / extra ops / no fabric.fu
  // are all malformed.
  if (existingFunc.getBody().empty()) {
    os << "symbol_conflict (existing @" << symbolName
       << " tagged loom.synthesized_for=\"" << groupName
       << "\" but body shape is malformed: expected exactly one fabric.fu, "
          "found 0 [B1])";
    return os.str();
  }
  ::mlir::Block &entry = existingFunc.getBody().front();
  unsigned fuCount = 0;
  ::fabric::FuOp innerFu;
  for (::mlir::Operation &op : entry.getOperations()) {
    if (auto fu = ::mlir::dyn_cast<::fabric::FuOp>(op)) {
      ++fuCount;
      if (!innerFu)
        innerFu = fu;
    }
  }
  ::mlir::Operation *terminator = entry.getTerminator();
  bool terminatorIsReturn =
      terminator && ::mlir::isa<::mlir::func::ReturnOp>(terminator);
  // Body must be exactly two ops: the fabric.fu and the func.return.
  unsigned numOps = 0;
  for (::mlir::Operation &op : entry.getOperations()) {
    (void)op;
    ++numOps;
  }
  if (fuCount != 1 || !terminatorIsReturn || numOps != 2) {
    os << "symbol_conflict (existing @" << symbolName
       << " tagged loom.synthesized_for=\"" << groupName
       << "\" but body shape is malformed: expected exactly one fabric.fu, "
          "found "
       << fuCount << " [B1])";
    return os.str();
  }

  // B2: verify the inner fabric.fu in isolation. Capture diagnostics
  // through a ScopedDiagnosticHandler so the verifier's error output
  // does not leak to stderr; surface them as part of the attached note
  // instead. Multiple diagnostics are joined with `; ` for a single
  // deterministic line.
  ::llvm::SmallVector<std::string, 2> diagMsgs;
  {
    ::mlir::ScopedDiagnosticHandler capture(
        ctx, [&](::mlir::Diagnostic &d) {
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
         << " [B2])";
      return os.str();
    }
  }

  // B3: signature match. The expected signature is the lift of the
  // input subgraphs' block-arg types (-> wrapper inputs) and yield
  // operand types (-> wrapper outputs) to fabric.bits<N>.
  auto portsOpt =
      ::loom::fabric::tech::collectWrapperPorts(subgraphs, ctx);
  if (!portsOpt.has_value()) {
    os << "symbol_conflict (existing @" << symbolName
       << " tagged loom.synthesized_for=\"" << groupName
       << "\" but expected signature could not be derived from the input "
          "subgraphs (block-arg / yield types not lift-able) [B3])";
    return os.str();
  }
  ::mlir::FunctionType actual = existingFunc.getFunctionType();
  ::llvm::ArrayRef<::mlir::Type> actualInputs = actual.getInputs();
  ::llvm::ArrayRef<::mlir::Type> actualResults = actual.getResults();
  ::llvm::ArrayRef<::mlir::Type> expectedInputs(portsOpt->inputs);
  ::llvm::ArrayRef<::mlir::Type> expectedResults(portsOpt->outputs);
  bool inputsMatch =
      actualInputs.size() == expectedInputs.size() &&
      std::equal(actualInputs.begin(), actualInputs.end(),
                 expectedInputs.begin());
  bool resultsMatch =
      actualResults.size() == expectedResults.size() &&
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
       << actualStr << " [B3])";
    return os.str();
  }

  return std::string();
}

bool GeneralizeSubgraphsToFuPass::prepareSymbolSlot(
    ::mlir::ModuleOp module, const SynthGroup &group) {
  std::string symbolName = wrapperNameFor(group.name);
  auto *existing =
      ::mlir::SymbolTable::lookupSymbolIn(module, symbolName);
  if (!existing)
    return true;

  auto existingFunc = ::mlir::dyn_cast<::mlir::func::FuncOp>(existing);
  if (existingFunc) {
    if (auto tag = existingFunc->getAttrOfType<::mlir::StringAttr>(
            "loom.synthesized_for")) {
      if (tag.getValue() == group.name) {
        // Marker-tagged wrapper. Validate that it is a real synthesized
        // wrapper (B1/B2/B3) before honoring it as idempotent. A failed
        // check is reported as a `symbol_conflict` so the user can
        // resolve the malformed wrapper rather than silently accepting
        // it as a no-op.
        std::string failureNote = validateMarkerWrapper(
            existingFunc, symbolName, group.name,
            ::llvm::ArrayRef<::dataflow::SubgraphOp>(group.subgraphs),
            &getContext());
        if (failureNote.empty()) {
          ::mlir::InFlightDiagnostic diag = module.emitRemark()
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
          ctx, ::loom::fabric::tech::failureReasonString(
                   ::loom::fabric::tech::SynthFailureReason::ConfigParseFailed));
      for (auto func : module.getOps<::mlir::func::FuncOp>()) {
        if (func->hasAttr("loom.synthesized_for"))
          continue;
        func->setAttr("loom.synth_failed", failedAttr);
      }
      ::mlir::InFlightDiagnostic diag = module.emitError()
          << "loom-generalize-subgraphs-to-fu: config_parse_failed: "
          << ::llvm::toString(loaded.takeError());
      (void)diag;
      return signalPassFailure();
    }
    cfg = *loaded;
  }

  // Validate input functions.
  ::llvm::SmallVector<
      ::std::pair<::mlir::func::FuncOp, ::dataflow::SubgraphOp>, 4>
      valid;
  validateFunctions(module, valid);

  // Empty input: emit a `no synth groups` remark and return. We
  // surface this even when there are invalid functions present (per
  // the spec, "empty input" is the absence of synthesizable subgraphs;
  // invalid functions are a separate failure path already annotated).
  if (valid.empty()) {
    module.emitRemark()
        << "loom-generalize-subgraphs-to-fu: no synth groups";
    return;
  }

  // Collect & sort groups lexically.
  auto groups = collectGroups(valid);
  if (groups.empty()) {
    module.emitRemark()
        << "loom-generalize-subgraphs-to-fu: no synth groups";
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
    if (prepareSymbolSlot(module, g))
      queued.push_back(std::move(g));
  }
  if (queued.empty())
    return;

  // Parallel-dispatch synthesis. Each worker runs the primary
  // strategy (and the fallback chain on failure) on its group's input
  // subgraphs in a thread-local *scratch* MLIRContext (per the spec
  // rule "MLIR mutation is never parallel"). The strategy never sees
  // the user's MLIRContext: that would race on `StorageUniquer`,
  // attribute interning, and type uniquing as soon as a real strategy
  // begins building IR. To hand the wrapper back to the main thread
  // safely, the worker prints the wrapper to text under its scratch
  // context and discards both the wrapper and the scratch context
  // before returning. The main thread re-parses that text under the
  // user's context below.
  ::loom::fabric::tech::WorkerPool pool(
      cfg.parallelismCrossGroup ? cfg.parallelismWorkers : 1u);

  // Per-group input bundle that is cheap to copy into a worker. We
  // pass the raw SubgraphOp slice + group name; the worker constructs
  // its own scratch MLIRContext and rebuilds a `SynthInputs` rooted in
  // it before invoking the strategy.
  struct WorkerJob {
    ::llvm::StringRef groupName;
    ::llvm::ArrayRef<::dataflow::SubgraphOp> subgraphs;
  };
  ::llvm::SmallVector<WorkerJob, 4> jobs;
  jobs.reserve(queued.size());
  for (auto &g : queued)
    jobs.push_back(WorkerJob{::llvm::StringRef(g.name),
                             ::llvm::ArrayRef<::dataflow::SubgraphOp>(
                                 g.subgraphs)});

  auto handoffs = pool.parallelMap<WorkerJob, WorkerHandoff>(
      ::llvm::ArrayRef<WorkerJob>(jobs),
      [&cfg](const WorkerJob &job) {
        // Thread-local scratch context. Required dialects must be
        // loaded explicitly: scratch contexts do not inherit anything
        // from the parent pass's context.
        ::mlir::MLIRContext scratch;
        ::mlir::DialectRegistry registry;
        registry.insert<::mlir::func::FuncDialect,
                        ::dataflow::DataflowDialect,
                        ::fabric::FabricDialect,
                        ::mlir::arith::ArithDialect,
                        ::mlir::math::MathDialect>();
        scratch.appendDialectRegistry(registry);
        scratch.loadAllAvailableDialects();

        ::loom::fabric::tech::SynthInputs in{
            job.groupName, job.subgraphs, cfg, &scratch};
        ::loom::fabric::tech::SynthResult res = runWithFallback(cfg, in);

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
    if (result.failureReason == ::loom::fabric::tech::SynthFailureReason::None
        && !handoff.wrapperIR.empty()) {
      ::mlir::OwningOpRef<::mlir::ModuleOp> parsed =
          ::mlir::parseSourceString<::mlir::ModuleOp>(handoff.wrapperIR, ctx);
      ::mlir::func::FuncOp reHomed;
      if (parsed) {
        for (auto fn : parsed->getOps<::mlir::func::FuncOp>()) {
          reHomed = fn;
          break;
        }
      }
      if (reHomed) {
        reHomed->remove();
        result.wrapper = ::mlir::OwningOpRef<::mlir::func::FuncOp>(reHomed);
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
    NodeCounts snapshotCounts;
    if (snapshotSuccess) {
      auto innerFu = findInnerFu(result.wrapper.get());
      if (innerFu) {
        ::loom::fabric::tech::CostModel cm(cfg);
        snapshotCost = cm.evaluate(innerFu);
        snapshotCounts = countFuBodyNodes(innerFu);
      }
      snapshotCovered = static_cast<unsigned>(group.subgraphs.size());
    }

    if (result.success()) {
      // Tag the wrapper with `loom.synthesized_for = "<group>"` so
      // future re-runs detect this as an idempotent slot.
      ::mlir::func::FuncOp wrapper = result.wrapper.get();
      wrapper->setAttr("loom.synthesized_for",
                       ::mlir::StringAttr::get(ctx, group.name));
      // Splice into the module body. `release` transfers ownership
      // from the OwningOpRef to the module.
      ::mlir::Operation *raw = result.wrapper.release();
      modBuilder.insert(raw);
    } else {
      ::loom::fabric::tech::annotateAndDiagnoseGroupFailure(
          module, group.name,
          ::llvm::ArrayRef<::mlir::func::FuncOp>(group.parents),
          result.failureReason,
          ::llvm::ArrayRef<std::string>(result.notes.begin(),
                                        result.notes.end()),
          failAsError);
      if (failAsError)
        signalPassFailure();
    }

    if (dumpStats) {
      // Build the canonical synth-stat line out of the pre-splice
      // snapshot so success cases retain their cost / coverage / node
      // metrics even though `result.wrapper` has been released.
      std::string line;
      ::llvm::raw_string_ostream os(line);
      os << "synth-stat group=" << group.name
         << " strategy=" << cfg.strategy << " reason=";
      if (snapshotSuccess)
        os << "success";
      else
        os << ::loom::fabric::tech::failureReasonString(result.failureReason);
      unsigned m = static_cast<unsigned>(group.subgraphs.size());
      os << " cost=" << snapshotCost << " covered=" << snapshotCovered
         << "/" << m << " nodes=" << snapshotCounts.ops << "/"
         << snapshotCounts.muxes << "/" << snapshotCounts.demuxes;
      module.emitRemark() << os.str();
    }
  }
}

namespace fabric {

// Header declares this in a different namespace; keep both in sync.
::std::unique_ptr<::mlir::Pass>
createGeneralizeSubgraphsToFuPass(std::string configPath, bool failAsError,
                                  bool dumpStats) {
  return std::make_unique<GeneralizeSubgraphsToFuPass>(
      std::move(configPath), failAsError, dumpStats);
}

void registerFabricTechSynthesizerPasses() {
  ::mlir::registerPass([] { return createGeneralizeSubgraphsToFuPass(); });
}

} // namespace fabric
