// MCS strategy: cost-prioritized branch-and-bound enumeration of
// candidate FUs across the input group.
//
// Pragmatic design (per spec section "Strategy: mcs"):
//
//   The spec's reference algorithm enumerates maximum common edge
//   subgraphs (MCES) anchored at yield positions and grows them by
//   share-group / width compatible alignment. That algorithm is
//   NP-hard. The spec explicitly sanctions a "delegate to incremental
//   for K canonical orderings" fallback as a correct-but-simple
//   implementation that still ranks candidates by `CostModel::evaluate`
//   and honours the same termination knobs.
//
// Synthesis proceeds as follows:
//
//   1. Validate inputs (non-empty, scratch context, sane caps).
//   2. Pre-flight check: if `mcs.timeout_sec == 0` the strategy never
//      gets to run; report `timeout` immediately. If the planned
//      candidate count would exceed `mcs.candidate_cap`, report
//      `resource_exhausted` before launching any branch. These hard
//      caps mirror the spec's "stop early on cap" / "deadline" rules.
//   3. Enumerate candidate orderings deterministically. Each branch is
//      one ordering of the input subgraphs; this is the search structure
//      used to drive cost-prioritized branch-and-bound.
//        - Anchor branches: for each input subgraph, an ordering that
//          uses that subgraph as the seed (input_0). This is the
//          "every yield position is a candidate seed" idea, projected
//          onto the per-input granularity that `IncrementalSynthesizer`
//          accepts.
//        - Random branches: additional seeded permutations (mt19937_64
//          with `mcs.seed` derived from `incrementalRandomSeed`) so the
//          branch space strictly contains the IncrementalRandom space.
//      The total branch count is bounded by `mcs.candidate_cap` and by
//      a fixed upper bound (16 * num_inputs) to keep wall time finite
//      on small inputs.
//   4. Run all branches in parallel via `WorkerPool` sized by
//      `mcs.branch_workers`. Each branch runs in its own sub-MLIRContext,
//      delegates to `IncrementalSynthesizer`, computes the wrapper cost
//      under that sub-context, and serializes the wrapper to text for
//      cross-context handoff. After the deadline elapses subsequent
//      branches short-circuit with `timeout`.
//   5. Among successful branches, pick the lowest cost; ties broken by
//      branch index so the choice is reproducible. Reparse the winning
//      wrapper into the outer scratch context.
//   6. If no branch succeeded, classify the failure: a deadline-hit
//      with no successes is `timeout`; a cap-hit is
//      `resource_exhausted`; otherwise the most common inner failure
//      reason is reported (ties broken by the lowest branch index).
//
// Spec source: `docs/spec-generalize-subgraphs-to-fu.md`, sections
// "Strategy: mcs" and "Acceptance criteria (mcs)".

#include "Fabric/Tech/Synthesizer/MCS.h"

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
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <random>
#include <string>
#include <utility>

namespace loom::fabric::tech {

namespace {

//===----------------------------------------------------------------------===//
// Per-branch record carried back from the parallel worker. Mirrors the
// IncrementalRandom handoff layout: wrapper text + cost + failure
// metadata so the caller can reparse / rank without reaching back into
// the worker's defunct sub-context.
//===----------------------------------------------------------------------===//

struct BranchHandoff {
  std::size_t branchIndex = 0;
  bool succeeded = false;
  // Empty unless `succeeded` is true.
  std::string wrapperIR;
  // Cost computed in the sub-context (cost is a pure function over a
  // wrapper FU, so the value is independent of context identity).
  double cost = 0.0;
  // Failure metadata for branches that did not succeed.
  SynthFailureReason failureReason = SynthFailureReason::None;
  ::llvm::SmallVector<::std::string, 4> notes;
  CoverageReport coverage;
  // True iff this branch was skipped because the global deadline had
  // already elapsed. Used to distinguish `timeout` from inner
  // strategy failures when classifying a no-success outcome.
  bool deadlineSkipped = false;
};

//===----------------------------------------------------------------------===//
// FU lookup helper. Mirrors the same one used by Incremental and
// IncrementalRandom; kept local to avoid linking against either.
//===----------------------------------------------------------------------===//

::fabric::FuOp innerFuOf(::fabric::ModuleOp wrapper) {
  if (!wrapper)
    return {};
  ::fabric::FuOp found;
  wrapper.walk([&](::fabric::FuOp fu) {
    if (!found)
      found = fu;
  });
  return found;
}

//===----------------------------------------------------------------------===//
// Set up a self-contained sub-MLIRContext. Sub-contexts inherit no
// dialects from their parent so we explicitly load the ones the
// Incremental strategy and the lit-test inputs require.
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
// Branch-ordering generation. The branch space is the union of:
//   * one anchor-rooted ordering per input subgraph (input_i first,
//     remaining inputs in stable index order). The anchor branch is
//     the "yield-anchored seed" search structure projected onto the
//     per-input granularity that Incremental accepts.
//   * `mcs.branch_workers` seeded random permutations (mt19937_64 over
//     `incrementalRandomSeed`) so the branch space strictly contains
//     the IncrementalRandom space. This guarantees that on tier-A
//     workloads MCS's chosen FU is no more expensive than IncrementalRandom's.
//===----------------------------------------------------------------------===//

using BranchVec = ::llvm::SmallVector<unsigned, 8>;

::llvm::SmallVector<BranchVec, 8>
buildBranches(std::size_t n, unsigned plannedBranches, uint64_t seed) {
  ::llvm::SmallVector<BranchVec, 8> branches;
  branches.reserve(plannedBranches);
  BranchVec base;
  base.reserve(n);
  for (unsigned i = 0; i < n; ++i)
    base.push_back(i);

  // Anchor branches: input_i first, remaining inputs in stable order.
  for (unsigned anchor = 0; anchor < n && branches.size() < plannedBranches;
       ++anchor) {
    BranchVec perm;
    perm.reserve(n);
    perm.push_back(anchor);
    for (unsigned i = 0; i < n; ++i)
      if (i != anchor)
        perm.push_back(i);
    branches.push_back(std::move(perm));
  }

  // Random branches: seeded permutations using a single PRNG so the
  // branch sequence is stable across runs on the same seed and input
  // size.
  std::mt19937_64 rng(seed);
  while (branches.size() < plannedBranches) {
    BranchVec perm = base;
    std::shuffle(perm.begin(), perm.end(), rng);
    branches.push_back(std::move(perm));
  }
  return branches;
}

//===----------------------------------------------------------------------===//
// Execute one branch in a fresh sub-context. Honours the shared
// deadline through the `deadlineExpired` flag; when set, the branch
// short-circuits with `deadlineSkipped=true` so the caller can
// classify the no-success outcome as `timeout`.
//===----------------------------------------------------------------------===//

BranchHandoff
runOneBranch(const ::loom::SynthConfig &cfg, ::llvm::StringRef groupName,
             ::llvm::ArrayRef<::dataflow::SubgraphOp> base,
             const BranchVec &perm, std::size_t branchIndex,
             const std::atomic<bool> &deadlineExpired) {
  BranchHandoff out;
  out.branchIndex = branchIndex;

  if (deadlineExpired.load(std::memory_order_relaxed)) {
    out.deadlineSkipped = true;
    out.failureReason = SynthFailureReason::Timeout;
    out.notes.push_back("mcs: branch skipped (deadline exceeded)");
    return out;
  }

  ::llvm::SmallVector<::dataflow::SubgraphOp, 8> permuted;
  permuted.reserve(perm.size());
  for (unsigned idx : perm) {
    if (idx >= base.size()) {
      out.failureReason = SynthFailureReason::InvalidInput;
      out.notes.push_back("mcs: branch ordering index out of range");
      return out;
    }
    permuted.push_back(base[idx]);
  }

  ::mlir::MLIRContext subContext;
  loadStandardDialects(subContext);

  // Each branch needs an independent inner cfg copy; SynthConfig is
  // plain data so a value copy is enough.
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

    // Serialize the winning wrapper to text so the outer caller can
    // reparse it under its own context.
    ::llvm::raw_string_ostream os(out.wrapperIR);
    ::mlir::OpPrintingFlags flags;
    flags.useLocalScope().assumeVerified();
    res.wrapper->print(os, flags);
    os.flush();
    out.succeeded = true;
  }
  return out;
}

//===----------------------------------------------------------------------===//
// Failure classification when no branch succeeded. Priority order:
//   1. If any branch was deadline-skipped (or every branch reported
//      `timeout`), report `timeout` -- the user's deadline killed us.
//   2. Otherwise pick the most common inner failure reason; ties
//      broken by the lowest branch index that produced that reason.
// `resource_exhausted` is reported separately at the call site before
// we even launch any branch (cap-hit pre-flight).
//===----------------------------------------------------------------------===//

SynthFailureReason
classifyFailure(::llvm::ArrayRef<BranchHandoff> handoffs) {
  if (handoffs.empty())
    return SynthFailureReason::Timeout;
  bool anyDeadline = false;
  unsigned counts[256] = {0};
  std::size_t firstIdx[256];
  for (auto &slot : firstIdx)
    slot = static_cast<std::size_t>(-1);
  unsigned totalFailures = 0;
  for (const BranchHandoff &h : handoffs) {
    if (h.succeeded)
      continue;
    if (h.deadlineSkipped)
      anyDeadline = true;
    auto code = static_cast<uint8_t>(h.failureReason);
    if (counts[code] == 0)
      firstIdx[code] = h.branchIndex;
    ++counts[code];
    ++totalFailures;
  }
  if (anyDeadline)
    return SynthFailureReason::Timeout;
  if (totalFailures == 0)
    return SynthFailureReason::Timeout;
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
// MCSSynthesizer.
//===----------------------------------------------------------------------===//

MCSSynthesizer::MCSSynthesizer(const ::loom::SynthConfig &c) : cfg(c) {}

SynthResult MCSSynthesizer::run(const SynthInputs &inputs) {
  SynthResult result;

  if (inputs.subgraphs.empty()) {
    result.failureReason = SynthFailureReason::InvalidInput;
    result.notes.push_back("mcs: no input subgraphs in synth group");
    return result;
  }
  if (!inputs.context) {
    result.failureReason = SynthFailureReason::InvalidInput;
    result.notes.push_back("mcs: missing scratch MLIRContext");
    return result;
  }

  // Pre-flight: deadline of 0 means the strategy is not allowed any
  // wall-time budget. Report `timeout` without launching any branch
  // so the failure is deterministic.
  if (cfg.mcsTimeoutSec == 0) {
    result.failureReason = SynthFailureReason::Timeout;
    result.notes.push_back("mcs: timeout_sec=0 disables synthesis");
    return result;
  }

  // Plan branch count. The branch space is bounded by:
  //   * `mcs.candidate_cap` (hard cap; cap-hit -> resource_exhausted).
  //   * `mcs.branch_workers` (random restarts -- a tunable widening of
  //     the anchor branch space).
  //   * 16 * num_inputs (an internal upper bound that keeps wall time
  //     finite on small inputs even when the user passes a very large
  //     candidate_cap).
  std::size_t numInputs = inputs.subgraphs.size();
  std::size_t anchorBranches = numInputs;
  std::size_t randomBranches =
      cfg.mcsBranchWorkers > 0 ? cfg.mcsBranchWorkers : 1u;
  std::size_t internalCap = 16 * numInputs > 0 ? 16 * numInputs : 16;
  std::size_t plannedBranches = anchorBranches + randomBranches;
  if (plannedBranches > internalCap)
    plannedBranches = internalCap;

  // Hard cap from the user's config. The semantic of the cap (per
  // spec) is "max number of candidates the strategy is allowed to
  // generate". A planned branch count strictly above the cap is a
  // resource exhaustion before any work happens. A plan equal to the
  // cap is OK (cap is inclusive).
  if (plannedBranches > cfg.mcsCandidateCap) {
    result.failureReason = SynthFailureReason::ResourceExhausted;
    std::string note;
    {
      ::llvm::raw_string_ostream os(note);
      os << "mcs: planned " << plannedBranches
         << " candidates exceeds candidate_cap " << cfg.mcsCandidateCap;
    }
    result.notes.push_back(std::move(note));
    return result;
  }

  // Build the branch orderings up front from a single seeded PRNG so
  // the sequence is stable across runs on the same seed/input size.
  auto branches = buildBranches(numInputs, plannedBranches,
                                cfg.incrementalRandomSeed);

  // Set up the deadline. `deadlineExpired` is checked by every branch
  // before it begins inner synthesis so late-starting branches do not
  // blow past the user's wall-time budget.
  using Clock = std::chrono::steady_clock;
  Clock::time_point deadline =
      Clock::now() + std::chrono::seconds(cfg.mcsTimeoutSec);
  std::atomic<bool> deadlineExpired{false};

  // Dispatch branches in parallel. `mcs.branch_workers` is the user-
  // tunable knob for branch-and-bound parallelism. `0` reuses the
  // global parallelism workers default; `1` runs branches inline,
  // which keeps single-thread test invocations deterministic.
  unsigned workers = cfg.mcsBranchWorkers > 0
                         ? cfg.mcsBranchWorkers
                         : cfg.parallelismWorkers;
  WorkerPool pool(workers);

  struct WorkerJob {
    std::size_t branchIndex = 0;
    const BranchVec *perm = nullptr;
  };
  ::llvm::SmallVector<WorkerJob, 8> jobs;
  jobs.reserve(branches.size());
  for (std::size_t i = 0; i < branches.size(); ++i)
    jobs.push_back(WorkerJob{i, &branches[i]});

  ::llvm::StringRef groupName = inputs.groupName;
  ::llvm::ArrayRef<::dataflow::SubgraphOp> base = inputs.subgraphs;
  const ::loom::SynthConfig &cfgRef = cfg;

  // Spawn a watchdog thread? The pass already runs each group on a
  // worker thread, so a wall-clock deadline check inside the closure
  // is enough -- branches polled the flag before they enter
  // Incremental, and Incremental's own work is bounded per-input, so
  // the worst-case overshoot is one inner Incremental run. This is
  // acceptable for the pragmatic implementation; an interrupt-driven
  // deadline would require restructuring Incremental, which is out of
  // scope for this strategy task.
  auto handoffs = pool.parallelMap<WorkerJob, BranchHandoff>(
      ::llvm::ArrayRef<WorkerJob>(jobs),
      [&cfgRef, groupName, base, &deadlineExpired,
       deadline](const WorkerJob &job) {
        // Refresh the deadline flag before launching this branch's
        // inner synthesis so late-starting branches that pile up after
        // the wall-clock has already passed are skipped cleanly.
        if (Clock::now() >= deadline)
          deadlineExpired.store(true, std::memory_order_relaxed);
        return runOneBranch(cfgRef, groupName, base, *job.perm,
                            job.branchIndex, deadlineExpired);
      });

  // Filter successful branches and pick the lowest cost. Ties broken
  // by the lowest branch index so the choice is reproducible.
  ::llvm::SmallVector<const BranchHandoff *, 8> successful;
  successful.reserve(handoffs.size());
  for (const BranchHandoff &h : handoffs)
    if (h.succeeded)
      successful.push_back(&h);

  if (successful.empty()) {
    result.failureReason = classifyFailure(handoffs);
    result.notes.push_back("mcs: no candidate branch produced a legal FU");
    for (const BranchHandoff &h : handoffs) {
      if (h.succeeded)
        continue;
      for (const ::std::string &n : h.notes)
        result.notes.push_back(n);
    }
    return result;
  }

  std::stable_sort(successful.begin(), successful.end(),
                   [](const BranchHandoff *a, const BranchHandoff *b) {
                     if (a->cost != b->cost)
                       return a->cost < b->cost;
                     return a->branchIndex < b->branchIndex;
                   });
  const BranchHandoff *best = successful.front();

  // Reparse the winning wrapper into the outer scratch context.
  ::mlir::OwningOpRef<::mlir::ModuleOp> parsed =
      ::mlir::parseSourceString<::mlir::ModuleOp>(best->wrapperIR,
                                                  inputs.context);
  if (!parsed) {
    result.failureReason = SynthFailureReason::VerifierFailed;
    result.notes.push_back(
        "mcs: failed to reparse winning wrapper into outer scratch "
        "MLIRContext");
    return result;
  }
  ::fabric::ModuleOp reHomed;
  for (auto fn : parsed->getOps<::fabric::ModuleOp>()) {
    reHomed = fn;
    break;
  }
  if (!reHomed) {
    result.failureReason = SynthFailureReason::VerifierFailed;
    result.notes.push_back(
        "mcs: reparsed module contained no fabric.module");
    return result;
  }
  reHomed->remove();
  result.wrapper = ::mlir::OwningOpRef<::fabric::ModuleOp>(reHomed);
  result.coverage = best->coverage;
  for (const ::std::string &n : best->notes)
    result.notes.push_back(n);
  ::std::string winnerNote;
  {
    ::llvm::raw_string_ostream os(winnerNote);
    os << "mcs: chose branch " << best->branchIndex << " of "
       << branches.size() << " (cost=" << best->cost << ")";
  }
  result.notes.push_back(std::move(winnerNote));
  return result;
}

} // namespace loom::fabric::tech
