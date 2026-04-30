// IncrementalRandom strategy: parallel multi-restart wrapper around
// `IncrementalSynthesizer`.
//
// Synthesis proceeds as follows:
//
//   1. Build `cfg.incrementalRandomRestarts` independent random
//      permutations of the input subgraph list. The PRNG is a
//      `std::mt19937_64` seeded with `cfg.incrementalRandomSeed`, so the
//      same seed always produces the same set of permutations regardless
//      of host or worker count. When
//      `cfg.incrementalRandomInputOrderHeuristic` is not
//      `random_seeded`, the first permutation is overridden with the
//      deterministic ordering driven by that heuristic
//      (`largest_first` / `smallest_first`); the remaining `restarts - 1`
//      permutations stay random.
//   2. Run all permutations in parallel via `WorkerPool::parallelMap`.
//      Each restart constructs its own sub-`MLIRContext` (so concurrent
//      strategy invocations do not race on the outer scratch context's
//      uniquing tables), instantiates an `IncrementalSynthesizer`
//      against that sub-context, and runs the strategy on the permuted
//      input list. On success the wrapper is serialized to text inside
//      the worker; on failure the failure reason is recorded for later
//      majority-vote merging.
//   3. Among successful restarts, score each wrapper with
//      `CostModel::evaluate`, sort by `(cost, permutation_index)`, and
//      pick the lowest. The winning wrapper text is re-parsed into the
//      outer `SynthInputs.context` so the caller gets a wrapper rooted
//      in its own context (same handoff pattern the pass uses to move
//      worker-built wrappers into the user's context).
//   4. If no restart succeeded, the most common failure reason among
//      the failed restarts is reported (ties broken by the first
//      restart that produced that reason). The accompanying notes carry
//      the per-restart diagnostics for debugging.
//
// Spec source: `docs/spec-generalize-subgraphs-to-fu.md`, sections
// "Strategy: incremental_random" and
// "Acceptance criteria (incremental_random)".

#include "Fabric/Tech/Synthesizer/IncrementalRandom.h"

#include "Common/SynthConfig.h"
#include "Dataflow/IR/DataflowDialect.h"
#include "Fabric/IR/FabricDialect.h"
#include "Fabric/IR/FabricOps.h"
#include "Fabric/Tech/Synthesizer/CostModel.h"
#include "Fabric/Tech/Synthesizer/Incremental.h"
#include "Fabric/Tech/Synthesizer/Parallel.h"
#include "Fabric/Tech/Synthesizer/Synthesizer.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace loom::fabric::tech {

namespace {

//===----------------------------------------------------------------------===//
// Per-restart record carried back from the parallel worker. The wrapper
// is serialized to text so it can be reparsed in the caller's context;
// the strategy result fields (failure reason, coverage, notes) are
// preserved verbatim.
//===----------------------------------------------------------------------===//

struct RestartHandoff {
  std::size_t permutationIndex = 0;
  bool succeeded = false;
  // Empty unless `succeeded` is true.
  std::string wrapperIR;
  // Cost computed in the sub-context (cost is a pure function over a
  // wrapper FU, so the value is independent of context identity).
  double cost = 0.0;
  // Carry through failure metadata for majority-vote merging.
  SynthFailureReason failureReason = SynthFailureReason::None;
  ::llvm::SmallVector<::std::string, 4> notes;
  CoverageReport coverage;
};

//===----------------------------------------------------------------------===//
// FU lookup helper (mirrored from the pass / Incremental).
//===----------------------------------------------------------------------===//

::fabric::FuOp innerFuOf(::mlir::func::FuncOp wrapper) {
  if (!wrapper || wrapper.getBody().empty())
    return {};
  for (::mlir::Operation &op : wrapper.getBody().front().getOperations())
    if (auto fu = ::mlir::dyn_cast<::fabric::FuOp>(op))
      return fu;
  return {};
}

//===----------------------------------------------------------------------===//
// Permutation generation. The same seed must always produce the same
// sequence of permutations regardless of input size, restart count, or
// thread interleaving. We therefore generate every permutation up front
// from a single seeded PRNG before dispatching parallel work.
//===----------------------------------------------------------------------===//

using PermVec = ::llvm::SmallVector<unsigned, 8>;

::llvm::SmallVector<PermVec, 8>
buildPermutations(std::size_t n, unsigned restarts, uint64_t seed) {
  ::llvm::SmallVector<PermVec, 8> perms;
  perms.reserve(restarts);
  PermVec base;
  base.reserve(n);
  for (unsigned i = 0; i < n; ++i)
    base.push_back(i);
  // Single seeded PRNG drives every permutation in turn so the sequence
  // is stable across runs on the same seed and input size.
  std::mt19937_64 rng(seed);
  for (unsigned r = 0; r < restarts; ++r) {
    PermVec perm = base;
    std::shuffle(perm.begin(), perm.end(), rng);
    perms.push_back(std::move(perm));
  }
  return perms;
}

// Replace the first restart's permutation with a deterministic ordering
// driven by `cfg.incrementalRandomInputOrderHeuristic`. The remaining
// `restarts - 1` permutations stay random so the explore-exploit tradeoff
// of multi-restart synthesis is preserved (the deterministic seed gives
// the cost-rank floor a known starting point; random restarts still
// explore alternatives).
//
// "random_seeded" leaves the all-random permutation set unchanged.
void applyFirstPermutationHeuristic(
    ::llvm::SmallVectorImpl<PermVec> &perms,
    ::llvm::ArrayRef<::dataflow::SubgraphOp> inputs,
    ::llvm::StringRef heuristic, uint64_t seed) {
  if (perms.empty())
    return;
  if (heuristic.empty() || heuristic == "random_seeded")
    return;
  ::llvm::SmallVector<unsigned, 8> ordered =
      sortInputsByOrderHeuristic(inputs, heuristic, seed);
  // sortInputsByOrderHeuristic returns indices sized to `inputs`; the
  // first restart's permutation also has that size by construction.
  PermVec first;
  first.reserve(ordered.size());
  for (unsigned i : ordered)
    first.push_back(i);
  perms[0] = std::move(first);
}

//===----------------------------------------------------------------------===//
// Set up a self-contained sub-MLIRContext. Required dialects must be
// loaded explicitly; sub-contexts inherit nothing from their parent.
//===----------------------------------------------------------------------===//

void loadStandardDialects(::mlir::MLIRContext &ctx) {
  ::mlir::DialectRegistry registry;
  registry.insert<::mlir::func::FuncDialect, ::dataflow::DataflowDialect,
                  ::fabric::FabricDialect, ::mlir::arith::ArithDialect,
                  ::mlir::math::MathDialect>();
  ctx.appendDialectRegistry(registry);
  ctx.loadAllAvailableDialects();
}

//===----------------------------------------------------------------------===//
// Run one restart in a fresh sub-context. The closure is small enough to
// inline at the parallelMap call site, but pulling it out keeps the body
// of `run` focused on orchestration.
//===----------------------------------------------------------------------===//

RestartHandoff runOneRestart(const ::loom::SynthConfig &cfg,
                             ::llvm::StringRef groupName,
                             ::llvm::ArrayRef<::dataflow::SubgraphOp> base,
                             const PermVec &perm,
                             std::size_t permutationIndex) {
  RestartHandoff out;
  out.permutationIndex = permutationIndex;

  ::llvm::SmallVector<::dataflow::SubgraphOp, 8> permuted;
  permuted.reserve(perm.size());
  for (unsigned idx : perm) {
    if (idx >= base.size()) {
      out.failureReason = SynthFailureReason::InvalidInput;
      out.notes.push_back(
          "incremental_random: permutation index out of range");
      return out;
    }
    permuted.push_back(base[idx]);
  }

  ::mlir::MLIRContext subContext;
  loadStandardDialects(subContext);

  // Each restart needs an independent inner cfg copy so any stateful
  // tracking inside `IncrementalSynthesizer` does not bleed across
  // restarts. SynthConfig is plain data, so a value copy is enough.
  ::loom::SynthConfig subCfg = cfg;

  IncrementalSynthesizer inner(subCfg);
  SynthInputs subInputs{groupName,
                        ::llvm::ArrayRef<::dataflow::SubgraphOp>(permuted),
                        subCfg, &subContext};
  SynthResult res = inner.run(subInputs);

  out.failureReason = res.failureReason;
  out.notes = std::move(res.notes);
  out.coverage = std::move(res.coverage);

  if (res.success() && res.wrapper) {
    CostModel cm(subCfg);
    if (auto fu = innerFuOf(res.wrapper.get()))
      out.cost = cm.evaluate(fu);

    // Serialize the winning sub-context wrapper to text so the caller
    // can reparse it into the outer scratch context. `useLocalScope`
    // keeps SSA numbering self-contained; `assumeVerified` skips a
    // redundant verify pass we already performed inside Incremental.
    ::llvm::raw_string_ostream os(out.wrapperIR);
    ::mlir::OpPrintingFlags flags;
    flags.useLocalScope().assumeVerified();
    res.wrapper->print(os, flags);
    os.flush();
    out.succeeded = true;
  }

  // `subContext` falls out of scope here. Any attribute / type pointers
  // owned by it are now invalid; we already copied the wrapper into
  // text and the cost into a plain double, so nothing in `out`
  // references the sub-context any more.
  return out;
}

//===----------------------------------------------------------------------===//
// Majority-vote failure reason among unsuccessful restarts. Ties broken
// by the lowest permutation index that produced that reason so the
// reported reason is reproducible across runs.
//===----------------------------------------------------------------------===//

SynthFailureReason
mergeFailureReasons(::llvm::ArrayRef<RestartHandoff> handoffs) {
  if (handoffs.empty())
    return SynthFailureReason::TopologyMismatch;
  // Tally per closed-enum value.
  unsigned counts[256] = {0};
  std::size_t firstIdx[256];
  for (auto &slot : firstIdx)
    slot = static_cast<std::size_t>(-1);
  unsigned totalFailures = 0;
  for (const auto &h : handoffs) {
    if (h.succeeded)
      continue;
    auto code = static_cast<uint8_t>(h.failureReason);
    if (counts[code] == 0)
      firstIdx[code] = h.permutationIndex;
    ++counts[code];
    ++totalFailures;
  }
  if (totalFailures == 0)
    return SynthFailureReason::TopologyMismatch;
  unsigned bestCount = 0;
  std::size_t bestIdx = static_cast<std::size_t>(-1);
  uint8_t bestCode = static_cast<uint8_t>(SynthFailureReason::TopologyMismatch);
  for (unsigned code = 0; code < 256; ++code) {
    if (counts[code] == 0)
      continue;
    if (counts[code] > bestCount ||
        (counts[code] == bestCount && firstIdx[code] < bestIdx)) {
      bestCount = counts[code];
      bestIdx = firstIdx[code];
      bestCode = static_cast<uint8_t>(code);
    }
  }
  return static_cast<SynthFailureReason>(bestCode);
}

} // namespace

//===----------------------------------------------------------------------===//
// IncrementalRandomSynthesizer.
//===----------------------------------------------------------------------===//

IncrementalRandomSynthesizer::IncrementalRandomSynthesizer(
    const ::loom::SynthConfig &c)
    : cfg(c) {}

SynthResult IncrementalRandomSynthesizer::run(const SynthInputs &inputs) {
  SynthResult result;

  if (inputs.subgraphs.empty()) {
    result.failureReason = SynthFailureReason::InvalidInput;
    result.notes.push_back(
        "incremental_random: no input subgraphs in synth group");
    return result;
  }
  if (!inputs.context) {
    result.failureReason = SynthFailureReason::InvalidInput;
    result.notes.push_back(
        "incremental_random: missing scratch MLIRContext");
    return result;
  }

  unsigned restarts = cfg.incrementalRandomRestarts;
  if (restarts == 0) {
    // Treat 0 restarts as a config error rather than vacuous success.
    // The pass-level config loader normally clamps this to >= 1.
    result.failureReason = SynthFailureReason::InvalidInput;
    result.notes.push_back(
        "incremental_random: incrementalRandomRestarts must be >= 1");
    return result;
  }

  // 1. Build all permutations up front from a single seeded PRNG so the
  // permutation set is independent of thread interleaving.
  auto perms = buildPermutations(inputs.subgraphs.size(), restarts,
                                 cfg.incrementalRandomSeed);

  // When the configured heuristic is not `random_seeded`, override the
  // first restart's permutation with the deterministic ordering driven
  // by that heuristic. The remaining restarts stay random.
  applyFirstPermutationHeuristic(perms, inputs.subgraphs,
                                 cfg.incrementalRandomInputOrderHeuristic,
                                 cfg.incrementalRandomSeed);

  // 2. Dispatch restarts in parallel. WorkerPool with `parallelismWorkers`
  // honours `0 == auto` and `1 == inline` (used by tests for
  // single-thread determinism).
  WorkerPool pool(cfg.parallelismWorkers);

  struct WorkerJob {
    std::size_t permutationIndex = 0;
    const PermVec *perm = nullptr;
  };
  ::llvm::SmallVector<WorkerJob, 8> jobs;
  jobs.reserve(perms.size());
  for (std::size_t i = 0; i < perms.size(); ++i)
    jobs.push_back(WorkerJob{i, &perms[i]});

  ::llvm::StringRef groupName = inputs.groupName;
  ::llvm::ArrayRef<::dataflow::SubgraphOp> base = inputs.subgraphs;
  const ::loom::SynthConfig &cfgRef = cfg;

  auto handoffs = pool.parallelMap<WorkerJob, RestartHandoff>(
      ::llvm::ArrayRef<WorkerJob>(jobs),
      [&cfgRef, groupName, base](const WorkerJob &job) {
        return runOneRestart(cfgRef, groupName, base, *job.perm,
                             job.permutationIndex);
      });

  // 3. Filter successful restarts and pick the lowest-cost one. Ties
  // broken by the lowest permutation index so the choice is
  // reproducible across runs.
  ::llvm::SmallVector<const RestartHandoff *, 8> successful;
  successful.reserve(handoffs.size());
  for (const RestartHandoff &h : handoffs)
    if (h.succeeded)
      successful.push_back(&h);

  if (successful.empty()) {
    result.failureReason = mergeFailureReasons(handoffs);
    result.notes.push_back("incremental_random: all restarts failed");
    // Surface the per-restart notes for debugging. Tests routinely
    // scan diagnostics for the inner reason; preserving them keeps the
    // failure path informative without inventing new wording.
    for (const RestartHandoff &h : handoffs) {
      if (h.succeeded)
        continue;
      for (const ::std::string &n : h.notes)
        result.notes.push_back(n);
    }
    return result;
  }

  std::stable_sort(successful.begin(), successful.end(),
                   [](const RestartHandoff *a, const RestartHandoff *b) {
                     if (a->cost != b->cost)
                       return a->cost < b->cost;
                     return a->permutationIndex < b->permutationIndex;
                   });
  const RestartHandoff *best = successful.front();

  // 4. Reparse the winning wrapper into the outer scratch context. This
  // mirrors the pass's outer-scratch -> user-context handoff: the
  // wrapper text was printed under a sub-context whose lifetime ends
  // inside `runOneRestart`, so we materialize the wrapper anew under
  // `inputs.context`.
  ::mlir::OwningOpRef<::mlir::ModuleOp> parsed =
      ::mlir::parseSourceString<::mlir::ModuleOp>(best->wrapperIR,
                                                  inputs.context);
  if (!parsed) {
    result.failureReason = SynthFailureReason::VerifierFailed;
    result.notes.push_back(
        "incremental_random: failed to reparse winning wrapper into "
        "outer scratch MLIRContext");
    return result;
  }
  ::mlir::func::FuncOp reHomed;
  for (auto fn : parsed->getOps<::mlir::func::FuncOp>()) {
    reHomed = fn;
    break;
  }
  if (!reHomed) {
    result.failureReason = SynthFailureReason::VerifierFailed;
    result.notes.push_back(
        "incremental_random: reparsed module contained no func.func");
    return result;
  }
  reHomed->remove();
  result.wrapper = ::mlir::OwningOpRef<::mlir::func::FuncOp>(reHomed);
  result.coverage = best->coverage;
  // Preserve the winning restart's notes so callers can trace which
  // permutation index was chosen on success.
  for (const ::std::string &n : best->notes)
    result.notes.push_back(n);
  ::std::string winnerNote;
  {
    ::llvm::raw_string_ostream os(winnerNote);
    os << "incremental_random: chose permutation " << best->permutationIndex
       << " of " << restarts << " (cost=" << best->cost << ")";
  }
  result.notes.push_back(std::move(winnerNote));
  return result;
}

} // namespace loom::fabric::tech
