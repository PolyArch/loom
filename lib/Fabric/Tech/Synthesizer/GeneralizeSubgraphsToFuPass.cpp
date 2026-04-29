// Top-level entry point for `loom-generalize-subgraphs-to-fu`.
//
// The pass implements the inverse of `loom-enumerate-fu-subgraphs`:
// given a set of `dataflow.subgraph` instances grouped by the
// `loom.synth_group` attribute on their enclosing `func.func`, it
// dispatches per-group synthesis through a `SynthConfig`-driven
// strategy factory and splices the resulting wrapper `func.func`s into
// the user's module in lexical group-name order.
//
// Strategies returned by `makeSynthesizer` are still stubs at this
// point (T10+ replace them with real implementations); the pass exists
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
#include "Fabric/Tech/Synthesizer/CostModel.h"
#include "Fabric/Tech/Synthesizer/Parallel.h"
#include "Fabric/Tech/Synthesizer/Synthesizer.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
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
    // Pre-existing fabric.fu wrappers (a hand-written library or
    // output from another tool) are also bystanders rather than
    // inputs. Leaving such functions untouched lets the
    // symbol-conflict precheck see them.
    bool hasFabricFu = false;
    func.walk([&](::fabric::FuOp) { hasFabricFu = true; });
    if (hasFabricFu)
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
        ::mlir::InFlightDiagnostic diag = module.emitRemark()
            << "loom-generalize-subgraphs-to-fu: group \"" << group.name
            << "\": skipping idempotent re-synth (existing @" << symbolName
            << " tagged loom.synthesized_for=\"" << group.name << "\")";
        (void)diag;
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

void GeneralizeSubgraphsToFuPass::emitSynthStat(
    ::mlir::ModuleOp module, const SynthGroup &group,
    const ::loom::fabric::tech::SynthResult &result,
    const ::loom::SynthConfig &cfg) {
  // Canonical synth-stat line:
  //   synth-stat group=<name> strategy=<s> reason=<r> cost=<f>
  //              covered=<n>/<m> nodes=<n_op>/<n_mux>/<n_demux>
  std::string line;
  ::llvm::raw_string_ostream os(line);
  os << "synth-stat group=" << group.name << " strategy=" << cfg.strategy
     << " reason=";
  if (result.success())
    os << "success";
  else
    os << ::loom::fabric::tech::failureReasonString(result.failureReason);

  unsigned m = static_cast<unsigned>(group.subgraphs.size());
  if (result.success()) {
    auto innerFu = findInnerFu(result.wrapper.get());
    double cost = 0.0;
    if (innerFu) {
      ::loom::fabric::tech::CostModel cm(cfg);
      cost = cm.evaluate(innerFu);
    }
    unsigned covered = 0;
    if (result.coverage.allCovered()) {
      covered = m;
    } else {
      for (const auto &slot : result.coverage.matchIndex)
        if (slot.has_value())
          ++covered;
    }
    NodeCounts nc = countFuBodyNodes(innerFu);
    os << " cost=" << cost << " covered=" << covered << "/" << m
       << " nodes=" << nc.ops << "/" << nc.muxes << "/" << nc.demuxes;
  } else {
    os << " cost=0 covered=0/" << m << " nodes=0/0/0";
  }

  module.emitRemark() << os.str();
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
  // subgraphs and produces a SynthResult.
  ::loom::fabric::tech::WorkerPool pool(
      cfg.parallelismCrossGroup ? cfg.parallelismWorkers : 1u);

  // Build a vector of SynthInputs that matches `queued` order. We
  // store raw SubgraphOp arrays separately because SynthInputs holds
  // an ArrayRef; the ArrayRef must outlive the worker.
  ::llvm::SmallVector<::loom::fabric::tech::SynthInputs, 4> inputs;
  inputs.reserve(queued.size());
  for (auto &g : queued) {
    inputs.push_back(::loom::fabric::tech::SynthInputs{
        ::llvm::StringRef(g.name),
        ::llvm::ArrayRef<::dataflow::SubgraphOp>(g.subgraphs),
        cfg, ctx});
  }

  auto results = pool.parallelMap<::loom::fabric::tech::SynthInputs,
                                  ::loom::fabric::tech::SynthResult>(
      ::llvm::ArrayRef<::loom::fabric::tech::SynthInputs>(inputs),
      [&cfg](const ::loom::fabric::tech::SynthInputs &in) {
        return runWithFallback(cfg, in);
      });

  // Serial splice in lexical group order. This is the only place that
  // mutates the user's module after input validation.
  ::mlir::OpBuilder modBuilder(module.getBody(), module.getBody()->end());
  for (size_t i = 0; i < queued.size(); ++i) {
    SynthGroup &group = queued[i];
    ::loom::fabric::tech::SynthResult &result = results[i];

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
      annotateGroupFailure(
          group, ::loom::fabric::tech::failureReasonString(
                     result.failureReason));
      ::mlir::InFlightDiagnostic diag =
          failAsError ? module.emitError() : module.emitWarning();
      diag << "loom-generalize-subgraphs-to-fu: group \"" << group.name
           << "\": synthesis failed: "
           << ::loom::fabric::tech::failureReasonString(result.failureReason);
      for (const std::string &n : result.notes)
        diag.attachNote() << n;
      if (failAsError)
        signalPassFailure();
    }

    if (dumpStats)
      emitSynthStat(module, group, result, cfg);
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
