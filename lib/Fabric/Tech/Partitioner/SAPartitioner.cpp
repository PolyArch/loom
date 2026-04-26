#include "Fabric/Tech/Partitioner/SAPartitioner.h"

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
#include <cmath>
#include <random>
#include <utility>

namespace fabric {

namespace {

// Direct out / in block edges of a candidate, restricted to currently-bound
// blocks (only blocks with `tpl != nullptr` participate in the reachability
// matrix; unbound blocks are graph-level and never produce inter-block
// edges).
void computeEdgesForCandidate(
    ::llvm::ArrayRef<::mlir::Operation *> ops,
    const ::llvm::DenseMap<::mlir::Operation *, unsigned> &opToBlock,
    ::llvm::DenseSet<unsigned> &outB, ::llvm::DenseSet<unsigned> &inB) {
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

// Build a partition by greedy seed-and-grow over `visit`, restricted to ops
// listed as uncovered. Ops already bound in `blocks` / `opToBlock` are
// preserved untouched. The function only fills bindings for ops whose
// pointer is in `uncovered`.
//
// This is the core "greedy fill" that both the initial seed and the
// neighbor-rebuild step share. The full-search candidate cost ranking is
// retained to keep parity with GreedyPartitioner.
void greedyFillUncovered(
    const ::llvm::SmallVector<::mlir::Operation *> &visit,
    const ::llvm::DenseSet<::mlir::Operation *> &uncovered,
    const TemplateLibrary &lib, const CandidateCache &cache,
    const ::loom::TechMapConfig &cfg,
    ::llvm::SmallVector<PendingBlock> &blocks,
    ::llvm::DenseMap<::mlir::Operation *, unsigned> &opToBlock,
    ReachMatrix &reach) {
  for (::mlir::Operation *root : visit) {
    if (!uncovered.contains(root))
      continue;
    if (opToBlock.contains(root))
      continue;

    bool fabricSupported =
        ::fabric::isFabricOpSupported(root->getName().getStringRef());

    ::llvm::ArrayRef<unsigned> tplIds = cache.templatesForOp(root);

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
      // Every op must currently be uncovered AND in the uncovered worklist
      // (so we never pull a previously-bound op back into a new block).
      for (::mlir::Operation *op : ops) {
        if (opToBlock.contains(op))
          return;
        if (!uncovered.contains(op))
          return;
      }
      if (wouldFormMultiBlockCycle(ops, blocks, opToBlock, reach))
        return;

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
        consider(std::move(ops), &tpl, id);
      }
    }

    auto acceptBlock = [&](::llvm::SmallVector<::mlir::Operation *> ops,
                           const FuTemplate *tpl) {
      unsigned newId = static_cast<unsigned>(blocks.size());
      PendingBlock pb;
      pb.ops = std::move(ops);
      pb.tpl = tpl;
      if (tpl != nullptr) {
        ::llvm::DenseSet<unsigned> outB, inB;
        computeEdgesForCandidate(pb.ops, opToBlock, outB, inB);
        for (::mlir::Operation *op : pb.ops)
          opToBlock[op] = newId;
        blocks.push_back(std::move(pb));
        addBlockToReach(newId, outB, inB, reach);
      } else {
        blocks.push_back(std::move(pb));
      }
    };

    if (haveBest) {
      acceptBlock(std::move(bestOps), bestTpl);
    } else {
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
}

// Scrub blocks that were reset to empty by the tear-down step so block ids
// remain dense. Updates `opToBlock` to reference the new ids and rebuilds
// the reachability matrix from scratch.
void compactBlocks(::llvm::SmallVector<PendingBlock> &blocks,
                   ::llvm::DenseMap<::mlir::Operation *, unsigned> &opToBlock,
                   ReachMatrix &reach) {
  ::llvm::SmallVector<PendingBlock> compact;
  compact.reserve(blocks.size());
  ::llvm::SmallVector<int> remap(blocks.size(), -1);
  for (unsigned i = 0; i < blocks.size(); ++i) {
    if (blocks[i].ops.empty())
      continue;
    remap[i] = static_cast<int>(compact.size());
    compact.push_back(std::move(blocks[i]));
  }
  blocks = std::move(compact);

  // Rebuild opToBlock to the new id space.
  opToBlock.clear();
  for (unsigned i = 0; i < blocks.size(); ++i) {
    if (blocks[i].tpl == nullptr)
      continue;
    for (::mlir::Operation *op : blocks[i].ops)
      opToBlock[op] = i;
  }

  auto edges = collectBlockEdges(blocks, opToBlock);
  reach.rebuild(static_cast<unsigned>(blocks.size()), edges);
}

// Convert pending blocks to the public PartitionResult.
PartitionResult finalize(::llvm::SmallVector<PendingBlock> blocks) {
  PartitionResult result;
  result.blocks.reserve(blocks.size());
  for (unsigned i = 0; i < blocks.size(); ++i) {
    if (blocks[i].ops.empty())
      continue;
    Block b;
    b.id = static_cast<unsigned>(result.blocks.size());
    b.ops = std::move(blocks[i].ops);
    b.tpl = blocks[i].tpl;
    result.blocks.push_back(std::move(b));
  }
  return result;
}

// Snapshot of search state, used to roll back rejected SA neighbors.
struct State {
  ::llvm::SmallVector<PendingBlock> blocks;
  ::llvm::DenseMap<::mlir::Operation *, unsigned> opToBlock;
  ReachMatrix reach;
  double cost = 0.0;
};

// Compute the full uncovered set for the initial seed: every non-terminator
// op in the graph body.
::llvm::DenseSet<::mlir::Operation *>
allOpsAsUncovered(::dataflow::GraphOp graph) {
  ::llvm::DenseSet<::mlir::Operation *> uncovered;
  ::mlir::Block &body = graph.getBody().front();
  ::mlir::Operation *terminator = body.getTerminator();
  for (::mlir::Operation &op : body) {
    if (&op == terminator)
      continue;
    uncovered.insert(&op);
  }
  return uncovered;
}

} // namespace

PartitionResult SAPartitioner::run(::dataflow::GraphOp graph,
                                   const TemplateLibrary &lib,
                                   const ::loom::TechMapConfig &cfg) {
  // Build the candidate cache once; it is read-only during the SA loop.
  CandidateCache cache = CandidateCache::build(graph, lib, cfg.threads);

  // Fixed visitation order, shared across the seed and every neighbor.
  ::llvm::SmallVector<::mlir::Operation *> visit = reverseTopoOrder(graph);

  // Seed: greedy seed-and-grow over all ops.
  State current;
  {
    auto uncovered = allOpsAsUncovered(graph);
    greedyFillUncovered(visit, uncovered, lib, cache, cfg, current.blocks,
                        current.opToBlock, current.reach);
    current.cost = computePendingCost(current.blocks, lib, cfg);
  }

  State best = current;

  // No annealing requested, skip straight to the seed.
  if (cfg.saSteps == 0)
    return finalize(std::move(best.blocks));

  // Single deterministic PRNG, advanced strictly serially. Multi-threading
  // only happens inside the candidate cache (which does not consult this
  // PRNG); the SA acceptance loop and the neighbor selection step are
  // single-threaded.
  std::mt19937_64 rng(cfg.saSeed);

  // Initial temperature: at least 1.0, scaled to the seed's |cost| so the
  // exp(...) acceptance probability is meaningful from the first step.
  double T0 = std::max(1.0, std::fabs(current.cost));

  for (unsigned step = 0; step < cfg.saSteps; ++step) {
    if (current.blocks.empty())
      break;

    // Pick a block uniformly at random.
    std::uniform_int_distribution<size_t> blockDist(
        0, current.blocks.size() - 1);
    size_t victim = blockDist(rng);

    // Build a neighbor by tearing down `victim`: mark its ops uncovered and
    // re-run the greedy fill restricted to those ops. The remaining blocks
    // stay bound (their cycle bookkeeping is preserved) so other ops keep
    // their assignment for free.
    State neighbor = current;
    ::llvm::DenseSet<::mlir::Operation *> uncovered;
    uncovered.reserve(neighbor.blocks[victim].ops.size());
    for (::mlir::Operation *op : neighbor.blocks[victim].ops) {
      uncovered.insert(op);
      neighbor.opToBlock.erase(op);
    }
    // Reset the victim block in place. Compact afterwards so its slot is
    // reclaimed and the reach matrix shrinks; this keeps the marginal cost
    // stable across iterations regardless of how many tear-downs happened.
    neighbor.blocks[victim].ops.clear();
    neighbor.blocks[victim].tpl = nullptr;
    compactBlocks(neighbor.blocks, neighbor.opToBlock, neighbor.reach);

    greedyFillUncovered(visit, uncovered, lib, cache, cfg, neighbor.blocks,
                        neighbor.opToBlock, neighbor.reach);
    neighbor.cost = computePendingCost(neighbor.blocks, lib, cfg);

    bool accept = false;
    if (neighbor.cost < current.cost) {
      accept = true;
    } else {
      double T = T0 * std::pow(0.95, static_cast<double>(step));
      if (T <= 0.0) {
        accept = false;
      } else {
        double prob = std::exp((current.cost - neighbor.cost) / T);
        std::uniform_real_distribution<double> u(0.0, 1.0);
        double r = u(rng);
        accept = r < prob;
      }
    }

    if (accept)
      current = std::move(neighbor);

    if (current.cost < best.cost)
      best = current;
  }

  return finalize(std::move(best.blocks));
}

} // namespace fabric
