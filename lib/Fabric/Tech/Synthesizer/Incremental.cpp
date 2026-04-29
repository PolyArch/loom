// Incremental strategy: left-fold over input subgraphs.
//
// Synthesis proceeds as follows:
//
//   1. Sort the input subgraphs by `incremental.input_order_heuristic`
//      (largest_first / smallest_first / random_seeded; ties broken by
//      lexical parent func.func name).
//   2. Build a trivial FU from input_0 by reusing the anchor strategy on
//      a single-input bundle (anchor's lock-step BFS over a one-element
//      peer set degenerates to a 1:1 mirror of the body's topology;
//      every position becomes a fabric.op with `op_list = [op_name]`).
//   3. For each subsequent input subgraph:
//        a. Ask `CoverageVerifier` whether the current FU already covers
//           it; if yes, no-op and continue.
//        b. Otherwise enumerate candidate FUs through three generators
//           in `IncrementalExtensions.{h,cpp}`:
//             - `widenOplistCandidates`: op-list widening within a
//               share group + width.
//             - `insertMuxDemuxCandidates`: tier B baseline mux/demux
//               insert (sg-extends-FU and FU-extends-sg cases).
//             - `structuralExtendCandidates`: tier C hook (returns an
//               empty set today; the follow-up task wires it in).
//        c. Filter candidates to those that pass MLIR's verifier (and
//           back-cover all previously folded subgraphs when
//           `coverage_verify_each_attempt` is true).
//        d. Rank legal candidates by `CostModel::evaluate`; ties broken
//           by a stable structural hash of the candidate's printed
//           form. Lowest-cost legal candidate wins; the others are
//           dropped.
//   4. Optional final coverage verification.
//
// Spec source: `docs/spec-generalize-subgraphs-to-fu.md`, sections
// "Strategy: incremental" and "Acceptance criteria (incremental)".

#include "Fabric/Tech/Synthesizer/Incremental.h"

#include "Common/SynthConfig.h"
#include "Fabric/IR/FabricOps.h"
#include "Fabric/Tech/Synthesizer/Alignment.h"
#include "Fabric/Tech/Synthesizer/Anchor.h"
#include "Fabric/Tech/Synthesizer/CostModel.h"
#include "Fabric/Tech/Synthesizer/CoverageVerifier.h"
#include "IncrementalExtensions.h"
#include "Fabric/Tech/Synthesizer/Synthesizer.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace loom::fabric::tech {

namespace {

//===----------------------------------------------------------------------===//
// FU lookup + symbol-name helpers (mirrored from the pass top level so
// the incremental loop is self-contained on these read-only utilities).
//===----------------------------------------------------------------------===//

::fabric::FuOp innerFuOf(::mlir::func::FuncOp wrapper) {
  if (!wrapper || wrapper.getBody().empty())
    return {};
  for (::mlir::Operation &op : wrapper.getBody().front().getOperations())
    if (auto fu = ::mlir::dyn_cast<::fabric::FuOp>(op))
      return fu;
  return {};
}

::std::string sanitizeGroup(::llvm::StringRef name) {
  ::std::string out = "fu_";
  for (char c : name) {
    bool ok = (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') ||
              (c >= '0' && c <= '9') || c == '_';
    out.push_back(ok ? c : '_');
  }
  return out;
}

//===----------------------------------------------------------------------===//
// Cost-tie structural-id hash. Used to break CostModel ties
// deterministically.
//===----------------------------------------------------------------------===//

uint64_t structuralIdOf(::mlir::func::FuncOp wrapper) {
  if (!wrapper)
    return 0;
  ::std::string text;
  ::llvm::raw_string_ostream os(text);
  ::mlir::OpPrintingFlags flags;
  flags.useLocalScope().assumeVerified();
  wrapper->print(os, flags);
  os.flush();
  return static_cast<uint64_t>(::llvm::hash_value(::llvm::StringRef(text)));
}

//===----------------------------------------------------------------------===//
// Subgraph-ordering heuristics.
//===----------------------------------------------------------------------===//

unsigned subgraphNodeCount(::dataflow::SubgraphOp sg) {
  if (!sg)
    return 0;
  ::mlir::Block &body = sg.getBody().front();
  unsigned n = 0;
  for (::mlir::Operation &op : body.without_terminator()) {
    (void)op;
    ++n;
  }
  return n;
}

::llvm::StringRef parentFuncName(::dataflow::SubgraphOp sg) {
  if (!sg)
    return {};
  if (auto func = sg->getParentOfType<::mlir::func::FuncOp>())
    return func.getName();
  return {};
}

::llvm::SmallVector<unsigned, 8>
sortInputs(::llvm::ArrayRef<::dataflow::SubgraphOp> inputs,
           ::llvm::StringRef heuristic, uint64_t seed) {
  ::llvm::SmallVector<unsigned, 8> idx;
  idx.reserve(inputs.size());
  for (unsigned i = 0; i < inputs.size(); ++i)
    idx.push_back(i);

  if (heuristic == "smallest_first") {
    ::std::stable_sort(idx.begin(), idx.end(),
                       [&](unsigned a, unsigned b) {
                         unsigned na = subgraphNodeCount(inputs[a]);
                         unsigned nb = subgraphNodeCount(inputs[b]);
                         if (na != nb)
                           return na < nb;
                         return parentFuncName(inputs[a]) <
                                parentFuncName(inputs[b]);
                       });
  } else if (heuristic == "random_seeded") {
    // Deterministic pseudo-random shuffle: the determinism rule (same
    // seed -> same permutation) must hold across runs. The seed comes
    // from `cfg.incrementalRandomSeed` (shared with incremental_random
    // for consistency).
    ::std::mt19937_64 rng(seed);
    ::std::shuffle(idx.begin(), idx.end(), rng);
  } else {
    // Default: largest_first. Largest body first, ties broken by
    // lexical func name.
    ::std::stable_sort(idx.begin(), idx.end(),
                       [&](unsigned a, unsigned b) {
                         unsigned na = subgraphNodeCount(inputs[a]);
                         unsigned nb = subgraphNodeCount(inputs[b]);
                         if (na != nb)
                           return na > nb;
                         return parentFuncName(inputs[a]) <
                                parentFuncName(inputs[b]);
                       });
  }
  return idx;
}

//===----------------------------------------------------------------------===//
// Trivial FU build: delegate to the anchor strategy on a single-input
// bundle. Anchor's lock-step BFS over a one-element peer set degenerates
// to a 1:1 mirror of the subgraph body, which is exactly the spec's
// trivial-FU definition.
//===----------------------------------------------------------------------===//

::mlir::OwningOpRef<::mlir::func::FuncOp>
buildTrivialFu(const ::loom::SynthConfig &cfg, ::mlir::MLIRContext *ctx,
               ::llvm::StringRef groupName, ::dataflow::SubgraphOp first,
               SynthFailureReason &reason,
               ::llvm::SmallVectorImpl<::std::string> &notes) {
  reason = SynthFailureReason::None;

  // Tier-C inputs (back-edge in the body) cannot be built via the
  // anchor strategy: anchor's lock-step BFS rejects BackEdge sources
  // as `topology_mismatch`. Detect tier-C up front and route to the
  // tier-C-aware mirror builder.
  if (first && !backEdges(first).empty()) {
    auto wrapper = detail::buildTrivialFuTierC(ctx, groupName, first);
    if (!wrapper) {
      reason = SynthFailureReason::TopologyMismatch;
      notes.push_back(
          "incremental: tier-C trivial FU build failed for first input");
    }
    return wrapper;
  }

  AnchorSynthesizer anchor(cfg);
  ::dataflow::SubgraphOp arr[1] = {first};
  SynthInputs in{groupName, ::llvm::ArrayRef<::dataflow::SubgraphOp>(arr), cfg,
                 ctx};
  SynthResult r = anchor.run(in);
  if (!r.success()) {
    reason = r.failureReason != SynthFailureReason::None
                 ? r.failureReason
                 : SynthFailureReason::TopologyMismatch;
    for (auto &n : r.notes)
      notes.push_back(std::move(n));
    return {};
  }
  return std::move(r.wrapper);
}

//===----------------------------------------------------------------------===//
// Verifier + back-coverage filter.
//===----------------------------------------------------------------------===//

bool runMlirVerify(::mlir::func::FuncOp wrapper) {
  if (!wrapper)
    return false;
  return ::mlir::succeeded(::mlir::verify(wrapper));
}

bool backCovers(::mlir::func::FuncOp wrapper,
                ::llvm::ArrayRef<::dataflow::SubgraphOp> covered,
                const ::loom::SynthConfig &cfg) {
  ::fabric::FuOp fu = innerFuOf(wrapper);
  if (!fu)
    return false;
  CoverageVerifier verifier(cfg);
  CoverageReport report = verifier.verify(fu, covered);
  return report.allCovered();
}

//===----------------------------------------------------------------------===//
// Lowest-cost legal candidate, with stable tie-breaker.
//===----------------------------------------------------------------------===//

struct ScoredCandidate {
  ::mlir::OwningOpRef<::mlir::func::FuncOp> wrapper;
  double cost = 0.0;
  uint64_t structuralId = 0;
};

::std::optional<ScoredCandidate>
pickBest(::llvm::SmallVector<::mlir::OwningOpRef<::mlir::func::FuncOp>, 8>
             &legal,
         const ::loom::SynthConfig &cfg) {
  if (legal.empty())
    return std::nullopt;
  ::std::vector<ScoredCandidate> scored;
  scored.reserve(legal.size());
  CostModel cost(cfg);
  for (auto &w : legal) {
    if (!w)
      continue;
    ::fabric::FuOp fu = innerFuOf(w.get());
    if (!fu)
      continue;
    ScoredCandidate sc;
    sc.cost = cost.evaluate(fu);
    sc.structuralId = structuralIdOf(w.get());
    sc.wrapper = std::move(w);
    scored.push_back(std::move(sc));
  }
  legal.clear();
  if (scored.empty())
    return std::nullopt;
  ::std::stable_sort(scored.begin(), scored.end(),
                     [](const ScoredCandidate &a, const ScoredCandidate &b) {
                       if (a.cost != b.cost)
                         return a.cost < b.cost;
                       return a.structuralId < b.structuralId;
                     });
  return std::move(scored.front());
}

//===----------------------------------------------------------------------===//
// is_covered: shorthand around CoverageVerifier for one sg.
//===----------------------------------------------------------------------===//

bool isCovered(::mlir::func::FuncOp wrapper, ::dataflow::SubgraphOp sg,
               const ::loom::SynthConfig &cfg) {
  ::fabric::FuOp fu = innerFuOf(wrapper);
  if (!fu)
    return false;
  CoverageVerifier verifier(cfg);
  ::dataflow::SubgraphOp arr[1] = {sg};
  CoverageReport report =
      verifier.verify(fu, ::llvm::ArrayRef<::dataflow::SubgraphOp>(arr));
  return report.allCovered();
}

} // namespace

//===----------------------------------------------------------------------===//
// IncrementalSynthesizer.
//===----------------------------------------------------------------------===//

IncrementalSynthesizer::IncrementalSynthesizer(const ::loom::SynthConfig &c)
    : cfg(c) {}

SynthResult IncrementalSynthesizer::run(const SynthInputs &inputs) {
  SynthResult result;

  if (inputs.subgraphs.empty()) {
    result.failureReason = SynthFailureReason::InvalidInput;
    result.notes.push_back("incremental: no input subgraphs in synth group");
    return result;
  }
  if (!inputs.context) {
    result.failureReason = SynthFailureReason::InvalidInput;
    result.notes.push_back("incremental: missing scratch MLIRContext");
    return result;
  }

  // 1. Sort inputs by configured heuristic.
  auto sortedIdx = sortInputs(inputs.subgraphs,
                              cfg.incrementalInputOrderHeuristic,
                              cfg.incrementalRandomSeed);
  ::llvm::SmallVector<::dataflow::SubgraphOp, 8> ordered;
  ordered.reserve(sortedIdx.size());
  for (unsigned i : sortedIdx)
    ordered.push_back(inputs.subgraphs[i]);

  // 2. Build trivial FU from the first input.
  SynthFailureReason trivialReason = SynthFailureReason::None;
  ::llvm::SmallVector<::std::string, 4> trivialNotes;
  ::mlir::OwningOpRef<::mlir::func::FuncOp> wrapper =
      buildTrivialFu(cfg, inputs.context, inputs.groupName, ordered.front(),
                     trivialReason, trivialNotes);
  for (auto &n : trivialNotes)
    result.notes.push_back(std::move(n));
  if (!wrapper) {
    result.failureReason = trivialReason != SynthFailureReason::None
                               ? trivialReason
                               : SynthFailureReason::TopologyMismatch;
    return result;
  }
  ::llvm::SmallVector<::dataflow::SubgraphOp, 8> covered;
  covered.push_back(ordered.front());

  // 3. Fold each subsequent input.
  for (size_t i = 1; i < ordered.size(); ++i) {
    ::dataflow::SubgraphOp sg = ordered[i];
    if (isCovered(wrapper.get(), sg, cfg)) {
      covered.push_back(sg);
      continue;
    }

    // Generate candidates via the extension hooks.
    ::llvm::SmallVector<::mlir::OwningOpRef<::mlir::func::FuncOp>, 8>
        candidates;
    {
      auto wide = detail::widenOplistCandidates(wrapper.get(), sg);
      for (auto &c : wide)
        candidates.push_back(std::move(c));
    }
    {
      auto mux = detail::insertMuxDemuxCandidates(wrapper.get(), sg);
      for (auto &c : mux)
        candidates.push_back(std::move(c));
    }
    if (detail::hasBackEdgeInDiff(wrapper.get(), sg)) {
      auto se = detail::structuralExtendCandidates(wrapper.get(), sg, cfg);
      for (auto &c : se)
        candidates.push_back(std::move(c));
    }

    // Filter.
    ::llvm::SmallVector<::mlir::OwningOpRef<::mlir::func::FuncOp>, 8> legal;
    for (auto &c : candidates) {
      if (!c)
        continue;
      if (!runMlirVerify(c.get()))
        continue;
      if (cfg.incrementalCoverageVerifyEachAttempt) {
        ::llvm::SmallVector<::dataflow::SubgraphOp, 8> all(covered.begin(),
                                                           covered.end());
        all.push_back(sg);
        if (!backCovers(c.get(), all, cfg))
          continue;
      }
      legal.push_back(std::move(c));
    }

    if (legal.empty()) {
      // Distinguish tier-C feedback_align_conflict from a generic
      // topology mismatch so the on-IR `loom.synth_failed` attribute
      // matches the spec's failure-reason enumeration.
      SynthFailureReason classified = SynthFailureReason::TopologyMismatch;
      if (detail::hasBackEdgeInDiff(wrapper.get(), sg)) {
        if (auto reason =
                detail::classifyTierCConflict(wrapper.get(), sg, cfg))
          classified = *reason;
      }
      result.failureReason = classified;
      ::std::string note;
      {
        ::llvm::raw_string_ostream os(note);
        os << "incremental: no legal extension for input " << i;
      }
      result.notes.push_back(std::move(note));
      return result;
    }

    auto best = pickBest(legal, cfg);
    if (!best.has_value()) {
      result.failureReason = SynthFailureReason::TopologyMismatch;
      result.notes.push_back("incremental: no legal extension after ranking");
      return result;
    }
    wrapper = std::move(best->wrapper);
    covered.push_back(sg);
  }

  result.wrapper = std::move(wrapper);
  // Defensive: keep the wrapper symbol name canonical even though
  // buildTrivialFu (via the anchor path) already does so.
  if (result.wrapper) {
    ::std::string sym = sanitizeGroup(inputs.groupName);
    if (result.wrapper->getName() != sym)
      result.wrapper->setName(sym);
  }
  return result;
}

} // namespace loom::fabric::tech
