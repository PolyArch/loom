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
#include <utility>

namespace fabric {

PartitionResult GreedyPartitioner::run(::dataflow::GraphOp graph,
                                       const TemplateLibrary &lib,
                                       const ::loom::TechMapConfig &cfg) {
  // Build a candidate cache once. Worker thread count is taken from the
  // tech-map config so single-threaded and multi-threaded runs share the
  // same downstream search path.
  CandidateCache cache = CandidateCache::build(graph, lib, cfg.threads);

  // Visitation order: yield-driven reverse topo. Ops feeding the yield
  // first; orphan ops appended in body program order. Tie-breaks on root
  // program position are realized implicitly by iterating ops in this
  // single fixed order.
  ::llvm::SmallVector<::mlir::Operation *> visit = reverseTopoOrder(graph);

  ::llvm::SmallVector<PendingBlock> blocks;
  ::llvm::DenseMap<::mlir::Operation *, unsigned> opToBlock;
  ReachMatrix reach;

  // For each op in visit order, try to grow a candidate covering it. If
  // multiple candidates are admissible, pick the one that minimizes the
  // partition's total cost; ties resolved by larger block size, then by
  // smaller template id, then by program position of the root op.
  for (::mlir::Operation *root : visit) {
    if (opToBlock.contains(root))
      continue;

    // Skip ops that can never be wrapped in a dataflow.subgraph (the
    // verifier would reject them). They still get a Block (tpl=nullptr)
    // so the Materializer leaves them at graph level.
    bool fabricSupported =
        ::fabric::isFabricOpSupported(root->getName().getStringRef());

    ::llvm::ArrayRef<unsigned> tplIds = cache.templatesForOp(root);

    // Best candidate state during this round.
    bool haveBest = false;
    double bestCost = 0.0;
    ::llvm::SmallVector<::mlir::Operation *> bestOps;
    const FuTemplate *bestTpl = nullptr;
    unsigned bestSize = 0;
    unsigned bestTplId = 0;

    auto consider = [&](::llvm::SmallVector<::mlir::Operation *> ops,
                        const FuTemplate *tpl, unsigned tplId) {
      if (ops.empty())
        return;

      // (a) every op must currently be uncovered.
      for (::mlir::Operation *op : ops)
        if (opToBlock.contains(op))
          return;

      // (c) cycle check.
      if (wouldFormMultiBlockCycle(ops, blocks, opToBlock, reach))
        return;

      // Tentatively materialize as a pending block and score.
      blocks.emplace_back();
      blocks.back().ops = ops;
      blocks.back().tpl = tpl;
      double cost = computePendingCost(blocks, lib, cfg);
      blocks.pop_back();

      unsigned sz = static_cast<unsigned>(ops.size());
      auto better = [&]() {
        if (!haveBest)
          return true;
        if (cost < bestCost)
          return true;
        if (cost > bestCost)
          return false;
        if (sz != bestSize)
          return sz > bestSize;
        if (tplId != bestTplId)
          return tplId < bestTplId;
        return false;
      };
      if (better()) {
        haveBest = true;
        bestCost = cost;
        bestOps = std::move(ops);
        bestTpl = tpl;
        bestSize = sz;
        bestTplId = tplId;
      }
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
        consider(std::move(ops), &tpl, id);
      }
    }

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
                           const FuTemplate *tpl) {
      unsigned newId = static_cast<unsigned>(blocks.size());
      PendingBlock pb;
      pb.ops = std::move(ops);
      pb.tpl = tpl;
      if (tpl != nullptr) {
        ::llvm::DenseSet<unsigned> outB, inB;
        computeEdges(pb.ops, outB, inB);
        for (::mlir::Operation *op : pb.ops)
          opToBlock[op] = newId;
        blocks.push_back(std::move(pb));
        addBlockToReach(newId, outB, inB, reach);
      } else {
        // Graph-level (unbound) ops do not participate in inter-block
        // reachability; we keep them in the result so the materializer
        // can leave them in place, but skip the cycle bookkeeping.
        blocks.push_back(std::move(pb));
      }
    };

    if (haveBest) {
      acceptBlock(std::move(bestOps), bestTpl);
    } else {
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
      acceptBlock(std::move(ops), chosen);
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
          blocks[i].tpl = nullptr;
          for (::mlir::Operation *op : blocks[i].ops)
            opToBlock.erase(op);
          break;
        }
      }
    }
  }

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
