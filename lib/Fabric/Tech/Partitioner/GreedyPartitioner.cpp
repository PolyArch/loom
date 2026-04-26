#include "Fabric/Tech/Partitioner/GreedyPartitioner.h"

#include "Dataflow/IR/DataflowOps.h"
#include "Fabric/IR/FabricOps.h"
#include "Fabric/Tech/Partitioner/CandidateCache.h"
#include "Fabric/Tech/Partitioner/CostModel.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <string>
#include <utility>

namespace fabric {

namespace {

// Print one attribute to a canonical string suitable for value-equality
// comparison. Mirrors the helper inside SubgraphMatcher.cpp; duplicated here
// to keep that header free of MLIR raw_ostream includes.
std::string canonAttr(::mlir::Attribute a) {
  std::string s;
  ::llvm::raw_string_ostream os(s);
  a.print(os);
  return s;
}

// Strip `loom.*` annotations, then return a sorted vector of (key, canonical
// value) pairs. Used to compare two ops' attribute sets while ignoring
// loom-internal metadata.
::llvm::SmallVector<std::pair<std::string, std::string>, 4>
stripLoomAttrs(::mlir::ArrayRef<::mlir::NamedAttribute> attrs) {
  ::llvm::SmallVector<std::pair<std::string, std::string>, 4> out;
  for (::mlir::NamedAttribute na : attrs) {
    auto key = na.getName().getValue();
    if (key.starts_with("loom."))
      continue;
    out.emplace_back(key.str(), canonAttr(na.getValue()));
  }
  std::sort(out.begin(), out.end());
  return out;
}

// Compute a reverse-topological visitation order for the body of `graph`,
// driven by the yield: ops whose results feed the yield come first. Ops not
// reachable from the yield (e.g. side-effecting stores not feeding any
// output) are appended at the end in graph body program order, so every op
// is still visited exactly once.
::llvm::SmallVector<::mlir::Operation *>
reverseTopoOrder(::dataflow::GraphOp graph) {
  ::mlir::Block &body = graph.getBody().front();
  ::mlir::Operation *terminator = body.getTerminator();

  ::llvm::SmallVector<::mlir::Operation *> order;
  ::llvm::DenseSet<::mlir::Operation *> seen;

  // Worklist seeded from yield's operand defs, in operand order.
  ::llvm::SmallVector<::mlir::Operation *> worklist;
  if (terminator) {
    for (::mlir::Value v : terminator->getOperands()) {
      ::mlir::Operation *def = v.getDefiningOp();
      if (def && def->getBlock() == &body && !seen.contains(def)) {
        seen.insert(def);
        worklist.push_back(def);
      }
    }
  }

  // BFS along the operand chain. Body ops in worklist[i] are appended to
  // `order` then their operand-def ops (still in body) are pushed to the
  // worklist. Because we always pop from the back of the worklist, ops feed
  // each other's operands strictly upstream of the current frontier.
  while (!worklist.empty()) {
    ::mlir::Operation *op = worklist.pop_back_val();
    order.push_back(op);
    for (::mlir::Value v : op->getOperands()) {
      ::mlir::Operation *def = v.getDefiningOp();
      if (!def || def->getBlock() != &body)
        continue;
      if (seen.contains(def))
        continue;
      seen.insert(def);
      worklist.push_back(def);
    }
  }

  // Catch ops that weren't reachable from the yield (they may still need
  // to be partitioned; e.g. a side-effect chain that ultimately ties back
  // to a memref). Append in body program order for determinism.
  for (::mlir::Operation &op : body) {
    if (&op == terminator)
      continue;
    if (seen.contains(&op))
      continue;
    seen.insert(&op);
    order.push_back(&op);
  }
  return order;
}

// One pending block under construction during greedy search.
struct PendingBlock {
  ::llvm::SmallVector<::mlir::Operation *> ops;
  const FuTemplate *tpl = nullptr;
};

// Match a multi-op template against a candidate rooted at `root`.
//
// We require the template's body to be a linear chain along operand[0]:
// T[N-1] is the root, T[i-1] = T[i].operand[0].defining_op. The user
// candidate must mirror this exact structure with the same op names, the
// same non-loom attribute dictionary at each step, and SSA wiring such that
// any internal operand reference (T[i].operand[p] producer is some T[j]
// with j < i) is realized by C[i].operand[p] == C[j]. External operands
// (block arguments of the template) can be any value at the corresponding
// position in the user op, including block args of the enclosing
// dataflow.graph or results of ops outside the candidate.
//
// Returns the candidate's ops in body program order (T[0]..T[N-1]) on
// success, or an empty vector on failure.
::llvm::SmallVector<::mlir::Operation *>
collectMultiOpCandidate(::mlir::Operation *root, const FuTemplate &tpl) {
  ::llvm::SmallVector<::mlir::Operation *> empty;
  unsigned n = tpl.bodyOpCount;
  if (n == 0)
    return empty;

  // Gather template body ops in program order.
  ::llvm::SmallVector<::mlir::Operation *> tplOps;
  // SubgraphOp::getBody() is not const-qualified; copy the op handle so we
  // can call non-const accessors without losing const-correctness elsewhere.
  ::dataflow::SubgraphOp tplSubgraph = tpl.subgraph;
  ::mlir::Block &tplBody = tplSubgraph.getBody().front();
  for (::mlir::Operation &op : tplBody.without_terminator())
    tplOps.push_back(&op);
  if (tplOps.size() != n)
    return empty;
  if (tplOps.empty())
    return empty;

  // The template's "root" is the op feeding the yield's first operand.
  ::mlir::Operation *yieldOp = tplBody.getTerminator();
  if (!yieldOp || yieldOp->getNumOperands() == 0)
    return empty;
  ::mlir::Operation *tplRoot = yieldOp->getOperand(0).getDefiningOp();
  if (tplRoot != tplOps.back())
    return empty; // we only handle templates whose root is the last body op

  // Walk back along operand[0] in the template to recover the canonical
  // chain T[N-1] = root -> T[N-2] -> ... -> T[0]. We only support
  // templates where this operand[0] backbone covers every body op; that
  // covers the enumerator's current output (linear chains).
  ::llvm::SmallVector<::mlir::Operation *> tplChain(n, nullptr);
  tplChain[n - 1] = tplRoot;
  for (unsigned i = n - 1; i > 0; --i) {
    ::mlir::Operation *cur = tplChain[i];
    if (!cur || cur->getNumOperands() == 0)
      return empty;
    ::mlir::Operation *prev = cur->getOperand(0).getDefiningOp();
    if (!prev || prev->getBlock() != &tplBody)
      return empty;
    tplChain[i - 1] = prev;
  }

  // Verify the chain covers every template body op exactly once.
  ::llvm::DenseSet<::mlir::Operation *> chainSet;
  for (::mlir::Operation *op : tplChain)
    chainSet.insert(op);
  if (chainSet.size() != n)
    return empty;
  for (::mlir::Operation *op : tplOps)
    if (!chainSet.contains(op))
      return empty;

  // Build the user-side chain by mirroring the operand[0] walk on `root`.
  ::llvm::SmallVector<::mlir::Operation *> usrChain(n, nullptr);
  usrChain[n - 1] = root;
  ::mlir::Block *userBody = root->getBlock();
  for (unsigned i = n - 1; i > 0; --i) {
    ::mlir::Operation *cur = usrChain[i];
    if (!cur || cur->getNumOperands() == 0)
      return empty;
    ::mlir::Operation *prev = cur->getOperand(0).getDefiningOp();
    if (!prev || prev->getBlock() != userBody)
      return empty;
    usrChain[i - 1] = prev;
  }
  // The user chain must cover N distinct ops (no repeats).
  ::llvm::DenseSet<::mlir::Operation *> usrSet;
  for (::mlir::Operation *op : usrChain)
    usrSet.insert(op);
  if (usrSet.size() != n)
    return empty;

  // Position lookup for template-body internal refs.
  ::llvm::DenseMap<::mlir::Operation *, unsigned> tplPos;
  for (unsigned i = 0; i < n; ++i)
    tplPos[tplChain[i]] = i;

  // Per-step structural compare: name, attrs, and SSA wiring.
  for (unsigned i = 0; i < n; ++i) {
    ::mlir::Operation *T = tplChain[i];
    ::mlir::Operation *C = usrChain[i];
    if (T->getName() != C->getName())
      return empty;
    if (T->getNumOperands() != C->getNumOperands())
      return empty;
    if (T->getNumResults() != C->getNumResults())
      return empty;
    if (stripLoomAttrs(T->getAttrs()) != stripLoomAttrs(C->getAttrs()))
      return empty;
    for (unsigned p = 0; p < T->getNumOperands(); ++p) {
      ::mlir::Value tv = T->getOperand(p);
      ::mlir::Value cv = C->getOperand(p);
      ::mlir::Operation *tdef = tv.getDefiningOp();
      if (tdef && tplPos.contains(tdef)) {
        // Internal template edge: the same wiring must hold for the user
        // candidate. cv must come from usrChain[tplPos[tdef]] at the same
        // result position.
        unsigned j = tplPos[tdef];
        unsigned resIdx = ::llvm::cast<::mlir::OpResult>(tv).getResultNumber();
        ::mlir::Operation *cdef = cv.getDefiningOp();
        if (cdef != usrChain[j])
          return empty;
        unsigned cResIdx =
            ::llvm::cast<::mlir::OpResult>(cv).getResultNumber();
        if (cResIdx != resIdx)
          return empty;
      } else {
        // External operand at this position in the template (block arg or
        // outside-body def). The user candidate's corresponding operand
        // must NOT be produced by another op in the candidate, otherwise
        // the wiring would diverge.
        ::mlir::Operation *cdef = cv.getDefiningOp();
        if (cdef && usrSet.contains(cdef))
          return empty;
      }
    }
  }

  ::llvm::SmallVector<::mlir::Operation *> result;
  result.reserve(n);
  for (unsigned i = 0; i < n; ++i)
    result.push_back(usrChain[i]);
  return result;
}

// Compact reachability matrix used for cycle detection over the inter-block
// SSA graph. Each block id maps to a BitVector of blocks reachable from
// it via accepted SSA edges. Resized lazily as new blocks join.
struct ReachMatrix {
  // For block id i, rows[i] is the set of block ids reachable from i
  // (including i itself).
  ::llvm::SmallVector<::llvm::BitVector> rows;

  void ensureSize(unsigned n) {
    while (rows.size() < n)
      rows.emplace_back();
    for (auto &r : rows)
      if (r.size() < n)
        r.resize(n);
  }

  // After contracting some old block ids, rebuild from `edges` so the
  // matrix reflects the current condensation graph. `edges[i]` lists
  // direct out-edges of block i.
  void rebuild(unsigned n,
               const ::llvm::SmallVector<::llvm::SmallVector<unsigned>> &edges) {
    rows.clear();
    rows.resize(n);
    for (unsigned i = 0; i < n; ++i) {
      rows[i].resize(n);
      rows[i].set(i);
    }
    // Floyd-Warshall-style transitive closure. For block counts in our
    // typical workloads this is well within budget.
    for (unsigned i = 0; i < n; ++i)
      for (unsigned d : edges[i])
        rows[i].set(d);
    bool changed = true;
    while (changed) {
      changed = false;
      for (unsigned i = 0; i < n; ++i) {
        for (unsigned k : edges[i]) {
          ::llvm::BitVector before = rows[i];
          rows[i] |= rows[k];
          if (rows[i] != before)
            changed = true;
        }
      }
    }
  }
};

// Build the per-block direct out-edge list from the current partition
// state. Only blocks whose op set is tracked in `opToBlock` (i.e. blocks
// with a bound template, since tpl-null blocks stay at graph level and
// must not influence inter-block reachability) contribute edges.
::llvm::SmallVector<::llvm::SmallVector<unsigned>>
collectBlockEdges(const ::llvm::SmallVector<PendingBlock> &blocks,
                  const ::llvm::DenseMap<::mlir::Operation *, unsigned>
                      &opToBlock) {
  ::llvm::SmallVector<::llvm::SmallVector<unsigned>> edges(blocks.size());
  for (unsigned bi = 0; bi < blocks.size(); ++bi) {
    if (blocks[bi].tpl == nullptr)
      continue;
    ::llvm::DenseSet<unsigned> dst;
    for (::mlir::Operation *op : blocks[bi].ops) {
      for (::mlir::Value res : op->getResults()) {
        for (::mlir::Operation *user : res.getUsers()) {
          auto it = opToBlock.find(user);
          if (it == opToBlock.end())
            continue;
          if (it->second == bi)
            continue;
          dst.insert(it->second);
        }
      }
    }
    for (unsigned d : dst)
      edges[bi].push_back(d);
    std::sort(edges[bi].begin(), edges[bi].end());
  }
  return edges;
}

// Check whether accepting `cand` (its ops) under the current partition
// would create a multi-block SSA cycle. The candidate has not yet been
// added to `blocks`; we tentatively assign it id `blocks.size()` and
// inspect direct out / in edges.
bool wouldFormMultiBlockCycle(
    const ::llvm::SmallVector<::mlir::Operation *> &candOps,
    const ::llvm::SmallVector<PendingBlock> &blocks,
    const ::llvm::DenseMap<::mlir::Operation *, unsigned> &opToBlock,
    const ReachMatrix &reach) {
  ::llvm::DenseSet<::mlir::Operation *> inCand;
  for (::mlir::Operation *op : candOps)
    inCand.insert(op);

  // Outgoing block edges from the candidate: any user outside the
  // candidate whose op is already in some block.
  ::llvm::DenseSet<unsigned> outBlocks;
  for (::mlir::Operation *op : candOps) {
    for (::mlir::Value res : op->getResults()) {
      for (::mlir::Operation *user : res.getUsers()) {
        if (inCand.contains(user))
          continue;
        auto it = opToBlock.find(user);
        if (it == opToBlock.end())
          continue;
        outBlocks.insert(it->second);
      }
    }
  }

  // Incoming block edges to the candidate: any operand whose def is in
  // some block (and not in the candidate).
  ::llvm::DenseSet<unsigned> inBlocks;
  for (::mlir::Operation *op : candOps) {
    for (::mlir::Value v : op->getOperands()) {
      ::mlir::Operation *def = v.getDefiningOp();
      if (!def || inCand.contains(def))
        continue;
      auto it = opToBlock.find(def);
      if (it == opToBlock.end())
        continue;
      inBlocks.insert(it->second);
    }
  }

  // A multi-block cycle would form iff some block B is both reachable
  // from the candidate (B in outBlocks's transitive closure) AND can
  // reach the candidate (B reaches some block in inBlocks). Equivalent
  // formulation: some predecessor block P (in inBlocks's reverse closure)
  // is also a successor of the candidate. Using the forward-reach matrix:
  // for any out-block O, if O can reach any in-block I, that closes the
  // cycle. Special case: if outBlocks and inBlocks share any block id,
  // that's an immediate length-2 cycle.
  for (unsigned out : outBlocks) {
    if (inBlocks.contains(out))
      return true;
    if (out >= reach.rows.size())
      continue;
    for (unsigned in : inBlocks) {
      if (in < reach.rows[out].size() && reach.rows[out].test(in))
        return true;
    }
  }
  (void)blocks;
  return false;
}

// Update the reach matrix incrementally after adding the candidate as a
// new block with id `newId`. `outBlocks` are direct successors and
// `inBlocks` are direct predecessors. Caller guarantees the addition does
// not introduce a multi-block cycle.
void addBlockToReach(unsigned newId,
                     const ::llvm::DenseSet<unsigned> &outBlocks,
                     const ::llvm::DenseSet<unsigned> &inBlocks,
                     ReachMatrix &reach) {
  unsigned newSize = newId + 1;
  reach.ensureSize(newSize);

  // Forward reach of the new block: itself plus union of out-blocks'
  // forward reach.
  ::llvm::BitVector &row = reach.rows[newId];
  row.resize(newSize);
  row.set(newId);
  for (unsigned out : outBlocks) {
    row.set(out);
    if (out < reach.rows.size()) {
      // Pad to the current size so the OR is well-defined.
      if (reach.rows[out].size() < newSize)
        reach.rows[out].resize(newSize);
      row |= reach.rows[out];
    }
  }

  // Predecessors of the new block now reach everything the new block
  // reaches. Walk backwards through the in-block closure to update all
  // ancestors. Since we maintain the full forward-reach for every block,
  // the easiest correct update is: for each block i, if i can reach any
  // in-block, then i now also reaches everything in `row`.
  for (unsigned i = 0; i < reach.rows.size(); ++i) {
    if (i == newId)
      continue;
    auto &r = reach.rows[i];
    if (r.size() < newSize)
      r.resize(newSize);
    bool reachesIn = false;
    for (unsigned in : inBlocks) {
      if (in < r.size() && r.test(in)) {
        reachesIn = true;
        break;
      }
    }
    if (reachesIn)
      r |= row;
  }
}

// Compute the marginal cost of adding one more pending block. We build a
// PartitionResult mirror with stable block ids and call the standard
// CostModel. The mirror is rebuilt per call, but block counts are small
// during search so this is cheap relative to the structural checks.
double computePendingCost(const ::llvm::SmallVector<PendingBlock> &blocks,
                          const TemplateLibrary &lib,
                          const ::loom::TechMapConfig &cfg) {
  PartitionResult mirror;
  mirror.blocks.reserve(blocks.size());
  for (unsigned i = 0; i < blocks.size(); ++i) {
    Block b;
    b.id = i;
    b.ops.append(blocks[i].ops.begin(), blocks[i].ops.end());
    b.tpl = blocks[i].tpl;
    mirror.blocks.push_back(std::move(b));
  }
  return computeCost(mirror, lib, cfg);
}

} // namespace

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

  // Convert pending blocks to the public PartitionResult. Block ids are
  // assigned in creation order; the Materializer reads them only for
  // identity (no semantic contract on the integer values).
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
