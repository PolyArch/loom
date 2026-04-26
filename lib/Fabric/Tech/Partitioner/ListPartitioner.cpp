#include "Fabric/Tech/Partitioner/ListPartitioner.h"

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
#include <queue>
#include <utility>
#include <vector>

namespace fabric {

namespace {

// Compute priority for an op:
//   priority(op) = max_template_size_for_root_op_kind * 100 - fanout(op)
// where fanout(op) is the number of users that live in the same graph body.
// Larger priorities are dequeued first; ties are broken by smaller program
// position (stable). The pointer is the last-resort tiebreaker; it is never
// observable since program positions are unique within a block.
int computePriority(::mlir::Operation *op, const TemplateLibrary &lib,
                    ::mlir::Block *body) {
  ::llvm::StringRef name = op->getName().getStringRef();
  unsigned maxTplSize = 0;
  if (::fabric::isFabricOpSupported(name)) {
    for (unsigned id : lib.templatesByRootOp(name)) {
      const FuTemplate &t = lib.templates()[id];
      if (t.bodyOpCount > maxTplSize)
        maxTplSize = t.bodyOpCount;
    }
  }

  unsigned fanout = 0;
  for (::mlir::Value res : op->getResults()) {
    for (::mlir::Operation *user : res.getUsers()) {
      if (user->getBlock() == body)
        ++fanout;
    }
  }

  return static_cast<int>(maxTplSize) * 100 - static_cast<int>(fanout);
}

// Priority queue entry. Comparator orders by priority DESC, then by program
// position ASC. Pointer breaker keeps `std::priority_queue` strict-weak even
// when both priority and position match (which never happens in well-formed
// IR, since position is unique inside a block).
struct PQEntry {
  int priority;
  unsigned position;
  ::mlir::Operation *op;
};

struct PQEntryLess {
  // std::priority_queue is a max-heap, so `a < b` means a is less preferred
  // (and gets popped after b). We want highest priority first, then
  // smallest position first.
  bool operator()(const PQEntry &a, const PQEntry &b) const {
    if (a.priority != b.priority)
      return a.priority < b.priority;
    if (a.position != b.position)
      return a.position > b.position;
    // Final disambiguation: pointer compare. Not observable in practice
    // because (priority, position) is already unique per op within a block.
    return std::less<::mlir::Operation *>{}(a.op, b.op);
  }
};

} // namespace

PartitionResult ListPartitioner::run(::dataflow::GraphOp graph,
                                     const TemplateLibrary &lib,
                                     const ::loom::TechMapConfig &cfg) {
  // Build a candidate cache once. Worker thread count is taken from the
  // tech-map config so single-threaded and multi-threaded runs share the
  // same downstream search path.
  CandidateCache cache = CandidateCache::build(graph, lib, cfg.threads);

  ::mlir::Block &body = graph.getBody().front();
  ::mlir::Operation *terminator = body.getTerminator();

  // Assign program positions in body order so the priority-queue comparator
  // can break ties deterministically. Positions are zero-based and dense
  // over non-terminator body ops.
  ::llvm::DenseMap<::mlir::Operation *, unsigned> position;
  unsigned nextPos = 0;
  for (::mlir::Operation &op : body) {
    if (&op == terminator)
      continue;
    position[&op] = nextPos++;
  }

  // Seed the priority queue. Priorities are structural (template library +
  // graph topology), both fixed for the duration of partitioning, so we
  // never need to recompute or re-push entries.
  std::priority_queue<PQEntry, std::vector<PQEntry>, PQEntryLess> queue;
  for (::mlir::Operation &op : body) {
    if (&op == terminator)
      continue;
    PQEntry e;
    e.priority = computePriority(&op, lib, &body);
    e.position = position.lookup(&op);
    e.op = &op;
    queue.push(e);
  }

  ::llvm::SmallVector<PendingBlock> blocks;
  ::llvm::DenseMap<::mlir::Operation *, unsigned> opToBlock;
  // `covered` tracks ops that have been consumed by an accepted multi-op
  // candidate before their turn at the head of the queue arrived. Such ops
  // are skipped on dequeue.
  ::llvm::DenseSet<::mlir::Operation *> covered;
  ReachMatrix reach;

  // To stitch together accepted blocks in body program order at the end
  // (matching greedy's externally-visible block ordering for shared lit
  // tests), we collect (rootPosition, ops, tpl) and sort once.
  struct Accepted {
    unsigned rootPosition;
    ::llvm::SmallVector<::mlir::Operation *> ops;
    const FuTemplate *tpl;
  };
  ::llvm::SmallVector<Accepted> accepted;

  // Edge collection helper, identical in spirit to the one in greedy.
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
    pb.ops = ops;
    pb.tpl = tpl;
    if (tpl != nullptr) {
      ::llvm::DenseSet<unsigned> outB, inB;
      computeEdges(pb.ops, outB, inB);
      for (::mlir::Operation *op : pb.ops) {
        opToBlock[op] = newId;
        covered.insert(op);
      }
      blocks.push_back(std::move(pb));
      addBlockToReach(newId, outB, inB, reach);
    } else {
      // Graph-level (unbound) ops do not participate in inter-block
      // reachability; we still mark them covered so the queue does not
      // revisit them.
      for (::mlir::Operation *op : pb.ops)
        covered.insert(op);
      blocks.push_back(std::move(pb));
    }

    // Preserve a body-program-order externally-visible block sequence: key
    // each accepted block by its earliest member's program position.
    Accepted acc;
    acc.rootPosition = ~0u;
    for (::mlir::Operation *op : ops) {
      auto it = position.find(op);
      if (it != position.end() && it->second < acc.rootPosition)
        acc.rootPosition = it->second;
    }
    acc.ops = std::move(ops);
    acc.tpl = tpl;
    accepted.push_back(std::move(acc));
  };

  while (!queue.empty()) {
    PQEntry top = queue.top();
    queue.pop();
    ::mlir::Operation *root = top.op;
    if (covered.contains(root))
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
        if (covered.contains(op))
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
          ops.push_back(root);
        } else {
          ops = collectMultiOpCandidate(root, tpl);
          if (ops.empty())
            continue;
        }
        consider(std::move(ops), &tpl, id);
      }
    }

    if (haveBest) {
      acceptBlock(std::move(bestOps), bestTpl);
    } else {
      // Fall back to a singleton block. Bind a template only if the cache
      // reports one and accepting that singleton would not form a
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
  // Mirrors greedy's defensive sweep: should never trigger if the
  // incremental check did its job.
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
          // Mirror the unbinding into the body-order projection so the
          // PartitionResult exposes the same template assignment.
          for (auto &acc : accepted) {
            if (acc.ops.size() == blocks[i].ops.size()) {
              bool same = true;
              for (size_t k = 0; k < acc.ops.size(); ++k) {
                if (acc.ops[k] != blocks[i].ops[k]) {
                  same = false;
                  break;
                }
              }
              if (same) {
                acc.tpl = nullptr;
                break;
              }
            }
          }
          break;
        }
      }
    }
  }

  // Sort accepted blocks by their earliest member's program position so
  // the resulting PartitionResult mirrors body program order. This keeps
  // downstream materialization deterministic and matches the way greedy
  // emits its blocks (greedy iterates body in reverse-topo and accepts
  // sequentially; the materializer in turn sorts blocks by first-op
  // program position when emitting). Sorting here makes List's output
  // robust even when priority-queue order differs from body order.
  std::sort(accepted.begin(), accepted.end(),
            [](const Accepted &a, const Accepted &b) {
              return a.rootPosition < b.rootPosition;
            });

  PartitionResult result;
  result.blocks.reserve(accepted.size());
  for (unsigned i = 0; i < accepted.size(); ++i) {
    Block b;
    b.id = i;
    b.ops = std::move(accepted[i].ops);
    b.tpl = accepted[i].tpl;
    result.blocks.push_back(std::move(b));
  }
  return result;
}

} // namespace fabric
