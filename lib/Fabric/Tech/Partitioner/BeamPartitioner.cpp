#include "Fabric/Tech/Partitioner/BeamPartitioner.h"

#include "Dataflow/IR/DataflowOps.h"
#include "Fabric/IR/FabricOps.h"
#include "Fabric/Tech/Partitioner/CandidateCache.h"
#include "Fabric/Tech/Partitioner/CostModel.h"
#include "PartitionerCommon.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <algorithm>
#include <string>
#include <utility>

namespace fabric {

namespace {

// One in-flight beam state. Each successor is produced by extending a
// parent state with a single op decision. States are ranked by cost, then
// by (|blocks_with_template| DESC, root program position ASC, structural
// key ASC) so the comparison is byte-stable across runs and thread counts.
struct BeamState {
  ::llvm::SmallVector<PendingBlock> blocks;
  // Ops currently bound into a block with tpl != nullptr; participates in
  // the cycle-detection ReachMatrix.
  ::llvm::DenseMap<::mlir::Operation *, unsigned> opToBlock;
  // Ops that have been placed into any block (bound or graph-level). Used
  // to short-circuit future op visits.
  ::llvm::DenseSet<::mlir::Operation *> covered;
  // Per-state reachability matrix over bound blocks only.
  ReachMatrix reach;
  // Accepted blocks in chronological order, with their root program
  // position. The final PartitionResult is emitted in body-program-order
  // for deterministic block numbering, mirroring ListPartitioner.
  struct Accepted {
    unsigned rootPosition;
    ::llvm::SmallVector<::mlir::Operation *> ops;
    const FuTemplate *tpl;
  };
  ::llvm::SmallVector<Accepted> accepted;

  // Cached cost of the current partition under the cost model.
  double cost = 0.0;
  // Tie-breakers, recomputed on each successor expansion.
  unsigned blocksWithTemplate = 0;
  unsigned lastRootPosition = ~0u;
  std::string structuralKey;
};

// Edge-collection helper used to feed addBlockToReach when accepting a new
// bound block. Mirrors the per-algorithm helper in the greedy / list paths.
void computeEdgesFor(const ::llvm::ArrayRef<::mlir::Operation *> ops,
                     const ::llvm::DenseMap<::mlir::Operation *, unsigned>
                         &opToBlock,
                     ::llvm::DenseSet<unsigned> &outB,
                     ::llvm::DenseSet<unsigned> &inB) {
  ::llvm::DenseSet<::mlir::Operation *> inCand;
  for (::mlir::Operation *op : ops)
    inCand.insert(op);
  for (::mlir::Operation *op : ops) {
    for (::mlir::Value res : op->getResults())
      for (::mlir::Operation *user : res.getUsers()) {
        if (inCand.contains(user))
          continue;
        auto it = opToBlock.find(user);
        if (it != opToBlock.end())
          outB.insert(it->second);
      }
    for (::mlir::Value v : op->getOperands()) {
      ::mlir::Operation *def = v.getDefiningOp();
      if (!def || inCand.contains(def))
        continue;
      auto it = opToBlock.find(def);
      if (it != opToBlock.end())
        inB.insert(it->second);
    }
  }
}

// Append the textual id of one block's template to a structural-key
// builder. nullptr is encoded as `-`; otherwise the template's stable id
// is rendered as a base-10 unsigned. Block boundaries are separated by
// a semicolon so the resulting key is a deterministic byte string.
void appendBlockKey(std::string &out, const FuTemplate *tpl) {
  if (!out.empty())
    out.push_back(';');
  if (tpl == nullptr) {
    out.push_back('-');
  } else {
    char buf[32];
    int len = std::snprintf(buf, sizeof(buf), "%u", tpl->id);
    if (len > 0)
      out.append(buf, static_cast<size_t>(len));
  }
}

// Recompute every per-state cached field after the state's blocks vector
// has been updated. Doing this once per successor keeps the cost / key
// values consistent with the underlying partition.
void refreshDerived(BeamState &s, const TemplateLibrary &lib,
                    const ::loom::ResolvedFabricTechMapConfig &cfg,
                    unsigned justAcceptedRootPosition) {
  s.cost = computePendingCost(s.blocks, lib, cfg);
  s.blocksWithTemplate = 0;
  for (const PendingBlock &b : s.blocks)
    if (b.tpl != nullptr)
      ++s.blocksWithTemplate;
  s.lastRootPosition = justAcceptedRootPosition;
  s.structuralKey.clear();
  for (const PendingBlock &b : s.blocks)
    appendBlockKey(s.structuralKey, b.tpl);
}

// Strict weak ordering used to rank successors. Smaller is preferred:
// lower cost first, then more bound blocks, then earlier root program
// position, then smaller structural key (lexicographically).
struct StateLess {
  bool operator()(const BeamState &a, const BeamState &b) const {
    if (a.cost != b.cost)
      return a.cost < b.cost;
    if (a.blocksWithTemplate != b.blocksWithTemplate)
      return a.blocksWithTemplate > b.blocksWithTemplate;
    if (a.lastRootPosition != b.lastRootPosition)
      return a.lastRootPosition < b.lastRootPosition;
    return a.structuralKey < b.structuralKey;
  }
};

// Append one accepted block to a beam state. `tpl == nullptr` means the
// op stays at graph level: the block is recorded in `blocks` (so the
// materializer can leave it in place) but not enrolled in the
// cycle-detection bookkeeping. Either way the op set is added to the
// state's `covered` view so future ops in the visit order do not pull
// it back into another candidate.
void acceptBlockInState(BeamState &s,
                        ::llvm::SmallVector<::mlir::Operation *> ops,
                        const FuTemplate *tpl, unsigned rootPosition,
                        const TemplateLibrary &lib,
                        const ::loom::ResolvedFabricTechMapConfig &cfg) {
  unsigned newId = static_cast<unsigned>(s.blocks.size());
  PendingBlock pb;
  pb.ops = ops;
  pb.tpl = tpl;
  if (tpl != nullptr) {
    ::llvm::DenseSet<unsigned> outB;
    ::llvm::DenseSet<unsigned> inB;
    computeEdgesFor(pb.ops, s.opToBlock, outB, inB);
    for (::mlir::Operation *op : pb.ops) {
      s.opToBlock[op] = newId;
      s.covered.insert(op);
    }
    s.blocks.push_back(std::move(pb));
    addBlockToReach(newId, outB, inB, s.reach);
  } else {
    for (::mlir::Operation *op : pb.ops)
      s.covered.insert(op);
    s.blocks.push_back(std::move(pb));
  }

  BeamState::Accepted acc;
  acc.rootPosition = rootPosition;
  acc.ops = std::move(ops);
  acc.tpl = tpl;
  s.accepted.push_back(std::move(acc));

  refreshDerived(s, lib, cfg, rootPosition);
}

// Position lookup keyed by op pointer. Built once over the graph body so
// the beam can score successors using a stable program-position
// tiebreaker.
::llvm::DenseMap<::mlir::Operation *, unsigned>
buildPositionTable(::dataflow::GraphOp graph) {
  ::llvm::DenseMap<::mlir::Operation *, unsigned> position;
  ::mlir::Block &body = graph.getBody().front();
  ::mlir::Operation *terminator = body.getTerminator();
  unsigned next = 0;
  for (::mlir::Operation &op : body) {
    if (&op == terminator)
      continue;
    position[&op] = next++;
  }
  return position;
}

// One unit of work: expand a parent state with a candidate covering
// `root`. The candidate is described by the op set and (optional)
// template; admissibility checks happen here. Returns true if the
// candidate was admissible and a successor was appended to `out`.
bool tryExpand(const BeamState &parent, ::mlir::Operation *root,
               ::llvm::SmallVector<::mlir::Operation *> ops,
               const FuTemplate *tpl, unsigned rootPosition,
               const TemplateLibrary &lib,
               const ::loom::ResolvedFabricTechMapConfig &cfg,
               ::llvm::SmallVector<BeamState, 0> &out) {
  if (ops.empty())
    return false;
  for (::mlir::Operation *op : ops)
    if (parent.covered.contains(op))
      return false;
  if (tpl != nullptr &&
      wouldFormMultiBlockCycle(ops, parent.blocks, parent.opToBlock,
                               parent.reach))
    return false;

  BeamState child = parent;
  acceptBlockInState(child, std::move(ops), tpl, rootPosition, lib, cfg);
  out.push_back(std::move(child));
  (void)root;
  return true;
}

// Final acyclicity sweep over a finished state. The incremental cycle
// check during search should already prevent any multi-block cycle, but
// this catches and softly recovers from any latent regression by
// unbinding ops from a violating block. Mirrors the defensive sweep in
// the greedy / list partitioners.
void enforceAcyclicity(BeamState &s) {
  auto edges = collectBlockEdges(s.blocks, s.opToBlock);
  ReachMatrix verify;
  verify.rebuild(static_cast<unsigned>(s.blocks.size()), edges);
  for (unsigned i = 0; i < s.blocks.size(); ++i) {
    if (s.blocks[i].tpl == nullptr)
      continue;
    for (unsigned d : edges[i]) {
      if (d == i)
        continue;
      if (d < verify.rows.size() && i < verify.rows[d].size() &&
          verify.rows[d].test(i)) {
        s.blocks[i].tpl = nullptr;
        for (::mlir::Operation *op : s.blocks[i].ops)
          s.opToBlock.erase(op);
        for (auto &acc : s.accepted) {
          if (acc.ops.size() != s.blocks[i].ops.size())
            continue;
          bool same = true;
          for (size_t k = 0; k < acc.ops.size(); ++k) {
            if (acc.ops[k] != s.blocks[i].ops[k]) {
              same = false;
              break;
            }
          }
          if (same) {
            acc.tpl = nullptr;
            break;
          }
        }
        break;
      }
    }
  }
}

} // namespace

PartitionResult BeamPartitioner::run(
    ::dataflow::GraphOp graph, const TemplateLibrary &lib,
    const ::loom::ResolvedFabricTechMapConfig &cfg) {
  // Build a candidate cache once. Worker thread count is taken from the
  // tech-map config so single-threaded and multi-threaded runs share the
  // same downstream search path.
  CandidateCache cache = CandidateCache::build(graph, lib, cfg.threads);

  ::llvm::DenseMap<::mlir::Operation *, unsigned> position =
      buildPositionTable(graph);

  // Visitation order: yield-driven reverse topo, identical to
  // GreedyPartitioner. Ops feeding the yield first; orphan ops appended
  // in body program order. The beam search visits the same sequence so
  // beam_width=1 is a strict superset of greedy.
  ::llvm::SmallVector<::mlir::Operation *> visit = reverseTopoOrder(graph);

  // Beam width 0 collapses to 1 (degenerate config); otherwise honor the
  // configured width. Capped at 1 below to avoid any surprise from a
  // future config that sets beamWidth = 0.
  unsigned width = cfg.beamWidth == 0 ? 1u : cfg.beamWidth;

  // Initial beam: a single empty state.
  ::llvm::SmallVector<BeamState, 0> beam;
  beam.emplace_back();

  for (::mlir::Operation *root : visit) {
    unsigned rootPos = ~0u;
    if (auto it = position.find(root); it != position.end())
      rootPos = it->second;

    bool fabricSupported =
        ::fabric::isFabricOpSupported(root->getName().getStringRef());
    ::llvm::ArrayRef<unsigned> tplIds = cache.templatesForOp(root);

    ::llvm::SmallVector<BeamState, 0> nextBeam;
    nextBeam.reserve(beam.size() * 4);

    for (const BeamState &parent : beam) {
      if (parent.covered.contains(root)) {
        // Already covered by an earlier multi-op fusion in this state.
        // Forward the state unchanged; no decision is taken at this op.
        nextBeam.push_back(parent);
        continue;
      }

      // Enumerate admissible candidates: every multi-op chain whose root
      // op kind matches, plus the single-op shortcut. Each successful
      // expansion appends one successor to nextBeam.
      ::llvm::SmallVector<BeamState, 0> localSuccessors;
      if (fabricSupported) {
        for (unsigned id : tplIds) {
          const FuTemplate &tpl = lib.templates()[id];
          if (tpl.bodyOpCount == 0)
            continue;
          if (tpl.rootOpName != root->getName().getStringRef())
            continue;
          ::llvm::SmallVector<::mlir::Operation *> ops;
          if (tpl.bodyOpCount == 1) {
            ops.push_back(root);
          } else {
            ops = collectMultiOpCandidate(root, tpl);
            if (ops.empty())
              continue;
          }
          tryExpand(parent, root, std::move(ops), &tpl, rootPos, lib, cfg,
                    localSuccessors);
        }
      }

      // Fall-back successor: only added when no fabric template covers
      // the op. The op gets a tpl=nullptr block (graph-level) so the
      // materializer leaves it in place. Always-on inclusion of this
      // successor would let the cost model trivially favour zero
      // coverage, since the formula penalizes |blocks_with_template|.
      if (localSuccessors.empty()) {
        ::llvm::SmallVector<::mlir::Operation *> ops{root};
        BeamState child = parent;
        acceptBlockInState(child, std::move(ops), nullptr, rootPos, lib,
                           cfg);
        localSuccessors.push_back(std::move(child));
      }

      for (BeamState &succ : localSuccessors)
        nextBeam.push_back(std::move(succ));
    }

    // Stable-sort successors by (cost ASC, |bound| DESC, root pos ASC,
    // structural key ASC). std::stable_sort plus the strict-weak comparator
    // gives byte-identical orderings across thread counts as long as the
    // scoring keys themselves are deterministic, which they are: cost is
    // a function of partition structure, blocksWithTemplate is a count,
    // root position is an integer keyed by graph body order, structural
    // key is a stable string of template ids.
    std::stable_sort(nextBeam.begin(), nextBeam.end(), StateLess{});

    if (nextBeam.size() > width)
      nextBeam.resize(width);

    beam = std::move(nextBeam);
    if (beam.empty()) {
      // Should not happen because every parent yields at least one
      // successor (either an admissible candidate or the fall-back
      // singleton); guard defensively so the search never produces an
      // empty result.
      beam.emplace_back();
    }
  }

  // Pick the lowest-cost surviving state under the same comparator so a
  // partition tie is resolved identically to the per-step ordering.
  std::stable_sort(beam.begin(), beam.end(), StateLess{});
  BeamState best = std::move(beam.front());

  enforceAcyclicity(best);

  // Sort accepted blocks by their earliest member's program position so
  // the resulting PartitionResult is body-program-order. This mirrors
  // ListPartitioner's externally-visible ordering and keeps lit-test
  // expectations uniform across algorithms.
  for (auto &acc : best.accepted) {
    unsigned earliest = ~0u;
    for (::mlir::Operation *op : acc.ops) {
      auto it = position.find(op);
      if (it != position.end() && it->second < earliest)
        earliest = it->second;
    }
    acc.rootPosition = earliest;
  }
  std::stable_sort(best.accepted.begin(), best.accepted.end(),
                   [](const BeamState::Accepted &a,
                      const BeamState::Accepted &b) {
                     return a.rootPosition < b.rootPosition;
                   });

  PartitionResult result;
  result.blocks.reserve(best.accepted.size());
  for (unsigned i = 0; i < best.accepted.size(); ++i) {
    Block b;
    b.id = i;
    b.ops = std::move(best.accepted[i].ops);
    b.tpl = best.accepted[i].tpl;
    result.blocks.push_back(std::move(b));
  }
  return result;
}

} // namespace fabric
