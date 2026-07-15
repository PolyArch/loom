#include "Fabric/Tech/Partitioner/GreedyPartitioner.h"

#include "Dataflow/IR/DataflowOps.h"
#include "Fabric/IR/FabricOps.h"
#include "Fabric/Tech/Partitioner/CandidateCache.h"
#include "Fabric/Tech/Partitioner/CostModel.h"
#include "PartitionerCommon.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <utility>

namespace fabric {

PartitionResult
GreedyPartitioner::run(::dataflow::GraphOp graph, const TemplateLibrary &lib,
                       const ::loom::ResolvedFabricTechMapConfig &cfg) {
  // Build a candidate cache once. Worker thread count is taken from the
  // tech-map config so single-threaded and multi-threaded runs share the
  // same downstream search path.
  CandidateCache cache = CandidateCache::build(graph, lib, cfg.threads);

  // Visitation order: yield-driven reverse topo. Ops feeding the yield
  // first; orphan ops appended in body program order. Tie-breaks on root
  // program position are realized implicitly by iterating ops in this
  // single fixed order.
  ::llvm::SmallVector<::mlir::Operation *> visit = reverseTopoOrder(graph);

  // Detect feedback edges in the dataflow.graph body. `dataflow.graph` is
  // a graph-region, so an op may consume an SSA value defined later in
  // program order. When such a back-edge exists, the partitioner can no
  // longer assume that wrapping each op in its own block produces a
  // forward-only block graph -- a multi-block cycle CAN form even from
  // singleton blocks. In that case the singleton fast path in
  // `addSingletonBlockToReach` is unsound; fall back to the full
  // transitive-closure update so `wouldFormMultiBlockCycle` keeps
  // catching all such cycles.
  bool hasFeedback = false;
  {
    ::llvm::DenseMap<::mlir::Operation *, unsigned> programOrder;
    {
      unsigned idx = 0;
      for (::mlir::Operation &op : graph.getBody().front().getOperations())
        programOrder[&op] = idx++;
    }
    for (::mlir::Operation &op : graph.getBody().front().getOperations()) {
      auto useIt = programOrder.find(&op);
      if (useIt == programOrder.end())
        continue;
      unsigned useIdx = useIt->second;
      for (::mlir::Value v : op.getOperands()) {
        ::mlir::Operation *def = v.getDefiningOp();
        if (!def)
          continue;
        auto defIt = programOrder.find(def);
        if (defIt == programOrder.end())
          continue;
        if (defIt->second > useIdx) {
          hasFeedback = true;
          break;
        }
      }
      if (hasFeedback)
        break;
    }
  }

  ::llvm::SmallVector<PendingBlock> blocks;
  // Bound-only op->block map: drives cycle detection over the bound-block
  // reachability matrix. Unbound (tpl == nullptr) ops are NOT in here,
  // matching the historical greedy semantics.
  ::llvm::DenseMap<::mlir::Operation *, unsigned> opToBlock;
  // Full op->block map: includes both bound and unbound ops. Mirrors the
  // CostModel's view of "any op in any block" so the running cost tallies
  // and `computeAcceptDelta` agree with `computeCost`.
  ::llvm::DenseMap<::mlir::Operation *, unsigned> opToBlockAll;
  ReachMatrix reach;

  // Running cost tallies, kept in sync with `computeCost(blocks)` via
  // `computeAcceptDelta`. See the end-of-run drift assertion below.
  unsigned blocksWithTemplate = 0;
  unsigned crossEdges = 0;
  double densitySum = 0.0;
  unsigned densityCount = 0;

  auto evalCostFromTallies = [&](const AcceptDelta &d) -> double {
    unsigned b =
        blocksWithTemplate + static_cast<unsigned>(d.blocksWithTemplate);
    int xe = static_cast<int>(crossEdges) + d.crossEdges;
    double dn = densitySum + d.densityNumerator;
    unsigned dc = densityCount + d.densityCount;
    double avgDensity = dc == 0 ? 0.0 : dn / static_cast<double>(dc);
    return cfg.alpha * static_cast<double>(b) +
           cfg.beta * static_cast<double>(xe) - cfg.gamma * avgDensity;
  };

  // For each op in visit order, try to grow a candidate covering it. If
  // multiple candidates are admissible, pick the one that minimizes the
  // partition's total cost; ties resolved by larger block size, then by
  // smaller template id, then by program position of the root op.
  for (::mlir::Operation *root : visit) {
    if (opToBlockAll.contains(root))
      continue;

    // Skip ops that can never be wrapped in a dataflow.subgraph (the
    // verifier would reject them). They still get a Block (tpl=nullptr)
    // so the Materializer leaves them at graph level.
    bool fabricSupported =
        ::fabric::isFabricOpSupported(root->getName().getStringRef());

    ::llvm::ArrayRef<unsigned> tplIds = cache.templatesForOp(root);

    // Helper: compute the direct out / in block edges of a candidate,
    // restricted to currently-bound blocks (tpl != nullptr; only those
    // participate in the reachability matrix).
    auto computeEdges = [&](::llvm::ArrayRef<::mlir::Operation *> ops,
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
    };

    auto acceptBlock = [&](::llvm::SmallVector<::mlir::Operation *> ops,
                           const FuTemplate *tpl, const AcceptDelta &delta) {
      unsigned newId = static_cast<unsigned>(blocks.size());
      PendingBlock pb;
      pb.ops = ops;
      pb.tpl = tpl;
      if (tpl != nullptr) {
        ::llvm::DenseSet<unsigned> outB, inB;
        computeEdges(pb.ops, outB, inB);
        for (::mlir::Operation *op : pb.ops) {
          opToBlock[op] = newId;
          opToBlockAll[op] = newId;
        }
        bool isSingleton = pb.ops.size() == 1;
        blocks.push_back(std::move(pb));
        if (isSingleton && !hasFeedback) {
          // Forward-only body + single-op block => the new block's row
          // only needs a self-bit. The cheap direct cycle test inside
          // `wouldFormMultiBlockCycle` (`if (inBlocks.contains(out))`)
          // still catches any immediate-neighbour cycle. The closure-
          // walk path only matters for transitive cycles between two or
          // more already-bound multi-op blocks, whose rows were
          // populated by their own (full) `addBlockToReach` calls.
          //
          // When the body has feedback edges (graph-region semantics) a
          // singleton CAN participate in a multi-block cycle through a
          // back-edge in the SSA graph; fall back to the full update so
          // future cycle queries still see the correct closure.
          addSingletonBlockToReach(newId, reach);
        } else {
          addBlockToReach(newId, outB, inB, reach);
        }
      } else {
        // Graph-level (unbound) ops do not participate in inter-block
        // reachability; we keep them in the result so the materializer
        // can leave them in place, but skip the cycle bookkeeping. They
        // DO participate in the cost-model's view of "ops in some block",
        // so they still join `opToBlockAll`.
        for (::mlir::Operation *op : pb.ops)
          opToBlockAll[op] = newId;
        blocks.push_back(std::move(pb));
      }

      blocksWithTemplate += static_cast<unsigned>(delta.blocksWithTemplate);
      crossEdges = static_cast<unsigned>(static_cast<int>(crossEdges) +
                                         delta.crossEdges);
      densitySum += delta.densityNumerator;
      densityCount += delta.densityCount;
    };

    // Enumerate candidates that match the root, cover unassigned ops, and
    // do not introduce an immediate cycle.
    struct Candidate {
      ::llvm::SmallVector<::mlir::Operation *> ops;
      const FuTemplate *tpl;
      unsigned tplId;
    };
    ::llvm::SmallVector<Candidate, 4> admissible;

    auto tryAdmit = [&](::llvm::SmallVector<::mlir::Operation *> ops,
                        const FuTemplate *tpl, unsigned tplId) {
      if (ops.empty())
        return;
      for (::mlir::Operation *op : ops)
        if (opToBlockAll.contains(op))
          return;
      if (wouldFormMultiBlockCycle(ops, blocks, opToBlock, reach))
        return;
      admissible.push_back({std::move(ops), tpl, tplId});
    };

    if (fabricSupported) {
      // Walk template ids in sorted order (CandidateCache guarantees that).
      for (unsigned id : tplIds) {
        const FuTemplate &tpl = lib.templates()[id];
        if (tpl.bodyOpCount == 0)
          continue;
        if (tpl.rootOpName != root->getName().getStringRef())
          continue;
        ::llvm::SmallVector<::mlir::Operation *> ops;
        if (tpl.bodyOpCount == 1) {
          // Single-op shortcut: name match is sufficient given the cache
          // already filtered by rootOpName.
          ops.push_back(root);
        } else {
          ops = collectMultiOpCandidate(root, tpl);
          if (ops.empty())
            continue;
        }
        tryAdmit(std::move(ops), &tpl, id);
      }
    }

    // Select the lowest-cost admissible candidate.
    if (admissible.empty()) {
      // Fall back to a singleton block. Bind a template only if the
      // cache reports one and accepting that singleton would not form a
      // multi-block cycle. Ops left unbound stay at graph level.
      const FuTemplate *chosen = nullptr;
      if (fabricSupported && !tplIds.empty()) {
        for (unsigned id : tplIds) {
          const FuTemplate &t = lib.templates()[id];
          if (t.bodyOpCount == 1 &&
              t.rootOpName == root->getName().getStringRef()) {
            ::llvm::SmallVector<::mlir::Operation *> ops{root};
            if (!wouldFormMultiBlockCycle(ops, blocks, opToBlock, reach)) {
              chosen = &t;
              break;
            }
          }
        }
      }
      ::llvm::SmallVector<::mlir::Operation *> ops{root};
      AcceptDelta delta = computeAcceptDelta(ops, chosen, lib, opToBlockAll);
      acceptBlock(std::move(ops), chosen, delta);
    } else if (admissible.size() == 1) {
      // Deterministic choice with a single survivor: skip scoring entirely
      // and accept directly. The four-tally update still runs so the
      // running cost stays in sync with the partition state.
      Candidate &c = admissible.front();
      AcceptDelta delta = computeAcceptDelta(c.ops, c.tpl, lib, opToBlockAll);
      acceptBlock(std::move(c.ops), c.tpl, delta);
    } else {
      // Score every admissible candidate. Tie-breaking order:
      // lower cost > larger size > smaller tplId > program order (which
      // is implicit since admissible[] preserves the cache's sorted order).
      bool haveBest = false;
      double bestCost = 0.0;
      unsigned bestSize = 0;
      unsigned bestTplId = 0;
      unsigned bestIdx = 0;
      AcceptDelta bestDelta;
      for (unsigned i = 0; i < admissible.size(); ++i) {
        const Candidate &c = admissible[i];
        AcceptDelta delta = computeAcceptDelta(c.ops, c.tpl, lib, opToBlockAll);
        double cost = evalCostFromTallies(delta);
        unsigned sz = static_cast<unsigned>(c.ops.size());
        auto better = [&]() {
          if (!haveBest)
            return true;
          if (cost < bestCost)
            return true;
          if (cost > bestCost)
            return false;
          if (sz != bestSize)
            return sz > bestSize;
          if (c.tplId != bestTplId)
            return c.tplId < bestTplId;
          return false;
        };
        if (better()) {
          haveBest = true;
          bestCost = cost;
          bestSize = sz;
          bestTplId = c.tplId;
          bestIdx = i;
          bestDelta = delta;
        }
      }
      Candidate &c = admissible[bestIdx];
      acceptBlock(std::move(c.ops), c.tpl, bestDelta);
    }
  }

  // End-of-run sanity check on inter-block acyclicity over bound blocks.
  // The incremental cycle check should already have prevented any
  // multi-block cycle from forming; this loop unbinds any pair that
  // somehow still participates in one (defensive; never observed in
  // tests, but left so a regression would degrade gracefully rather
  // than emit invalid IR).
  {
    auto edges = collectBlockEdges(blocks, opToBlock);
    ReachMatrix verify;
    verify.rebuild(static_cast<unsigned>(blocks.size()), edges);
    for (unsigned i = 0; i < blocks.size(); ++i) {
      if (blocks[i].tpl == nullptr)
        continue;
      for (unsigned d : edges[i]) {
        if (d == i)
          continue;
        if (d < verify.rows.size() && i < verify.rows[d].size() &&
            verify.rows[d].test(i)) {
          // Unbind: block becomes graph-level. Keep the running tallies
          // in sync so the end-of-run drift assertion stays meaningful.
          // Cross-edges are unaffected (the ops remain in
          // `opToBlockAll`, still grouped under the same block id), but
          // the bound-block / density terms shed their contribution.
          const FuTemplate *tpl = blocks[i].tpl;
          unsigned cap = 1;
          for (const FuTemplate &t : lib.templates()) {
            if (t.rootOpName == tpl->rootOpName && t.bodyOpCount > cap)
              cap = t.bodyOpCount;
          }
          double num = static_cast<double>(blocks[i].ops.size()) /
                       static_cast<double>(cap);
          if (blocksWithTemplate > 0)
            --blocksWithTemplate;
          densitySum -= num;
          if (densityCount > 0)
            --densityCount;
          blocks[i].tpl = nullptr;
          for (::mlir::Operation *op : blocks[i].ops)
            opToBlock.erase(op);
          break;
        }
      }
    }
  }

#ifndef NDEBUG
  // Drift check: the running tallies must agree with `computeCost`'s
  // recomputation on the final partition. If they disagree, the
  // incremental delta path drifted from the cost model and produced a
  // different ranking somewhere upstream.
  {
    PartitionResult mirror;
    mirror.blocks.reserve(blocks.size());
    for (unsigned i = 0; i < blocks.size(); ++i) {
      Block b;
      b.id = i;
      b.ops.append(blocks[i].ops.begin(), blocks[i].ops.end());
      b.tpl = blocks[i].tpl;
      mirror.blocks.push_back(std::move(b));
    }
    double full = computeCost(mirror, lib, cfg);
    AcceptDelta zero;
    double tally = evalCostFromTallies(zero);
    if (std::abs(full - tally) > 1e-9 * (1.0 + std::abs(full)))
      llvm_unreachable("greedy: incremental cost drifted from full cost");
  }
#endif

  // Convert pending blocks to the public PartitionResult.
  PartitionResult result;
  result.blocks.reserve(blocks.size());
  for (unsigned i = 0; i < blocks.size(); ++i) {
    Block b;
    b.id = i;
    b.ops = std::move(blocks[i].ops);
    b.tpl = blocks[i].tpl;
    result.blocks.push_back(std::move(b));
  }
  return result;
}

} // namespace fabric
