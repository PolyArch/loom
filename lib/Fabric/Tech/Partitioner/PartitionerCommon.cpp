#include "PartitionerCommon.h"

#include "Dataflow/IR/DataflowOps.h"
#include "Fabric/Tech/Partitioner/CostModel.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <string>
#include <utility>

namespace fabric {

std::string canonAttr(::mlir::Attribute a) {
  std::string s;
  ::llvm::raw_string_ostream os(s);
  a.print(os);
  return s;
}

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

  // BFS along the operand chain.
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

  // Catch ops not reachable from the yield. Append in body program order
  // for determinism.
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

::llvm::SmallVector<::mlir::Operation *>
collectMultiOpCandidate(::mlir::Operation *root, const FuTemplate &tpl) {
  ::llvm::SmallVector<::mlir::Operation *> empty;
  unsigned n = tpl.bodyOpCount;
  if (n == 0)
    return empty;

  // Gather template body ops in program order.
  ::llvm::SmallVector<::mlir::Operation *> tplOps;
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
    return empty;

  // Walk back along operand[0] to recover the canonical chain.
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
  ::llvm::DenseSet<::mlir::Operation *> usrSet;
  for (::mlir::Operation *op : usrChain)
    usrSet.insert(op);
  if (usrSet.size() != n)
    return empty;

  // Position lookup for template-body internal refs.
  ::llvm::DenseMap<::mlir::Operation *, unsigned> tplPos;
  for (unsigned i = 0; i < n; ++i)
    tplPos[tplChain[i]] = i;

  // Per-step structural compare.
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

void ReachMatrix::ensureSize(unsigned n) {
  while (rows.size() < n)
    rows.emplace_back();
  for (auto &r : rows)
    if (r.size() < n)
      r.resize(n);
}

void ReachMatrix::rebuild(
    unsigned n,
    const ::llvm::SmallVector<::llvm::SmallVector<unsigned>> &edges) {
  rows.clear();
  rows.resize(n);
  for (unsigned i = 0; i < n; ++i) {
    rows[i].resize(n);
    rows[i].set(i);
  }
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

::llvm::SmallVector<::llvm::SmallVector<unsigned>> collectBlockEdges(
    const ::llvm::SmallVector<PendingBlock> &blocks,
    const ::llvm::DenseMap<::mlir::Operation *, unsigned> &opToBlock) {
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

bool wouldFormMultiBlockCycle(
    const ::llvm::SmallVector<::mlir::Operation *> &candOps,
    const ::llvm::SmallVector<PendingBlock> &blocks,
    const ::llvm::DenseMap<::mlir::Operation *, unsigned> &opToBlock,
    const ReachMatrix &reach) {
  ::llvm::DenseSet<::mlir::Operation *> inCand;
  for (::mlir::Operation *op : candOps)
    inCand.insert(op);

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

void addBlockToReach(unsigned newId,
                     const ::llvm::DenseSet<unsigned> &outBlocks,
                     const ::llvm::DenseSet<unsigned> &inBlocks,
                     ReachMatrix &reach) {
  unsigned newSize = newId + 1;
  reach.ensureSize(newSize);

  ::llvm::BitVector &row = reach.rows[newId];
  row.resize(newSize);
  row.set(newId);
  for (unsigned out : outBlocks) {
    row.set(out);
    if (out < reach.rows.size()) {
      if (reach.rows[out].size() < newSize)
        reach.rows[out].resize(newSize);
      row |= reach.rows[out];
    }
  }

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

bool partitionHasMultiBlockCycle(const PartitionResult &result) {
  // Build the index map from op -> bound-block-id for the bound subset.
  ::llvm::DenseMap<::mlir::Operation *, unsigned> opToBlock;
  for (const Block &b : result.blocks)
    if (b.tpl != nullptr)
      for (::mlir::Operation *op : b.ops)
        opToBlock[op] = b.id;

  // Mirror the bound blocks into a PendingBlock vector indexed by Block::id
  // so collectBlockEdges' indices align with opToBlock values. Unbound
  // (tpl == nullptr) entries get empty placeholders so collectBlockEdges
  // skips them.
  unsigned n = 0;
  for (const Block &b : result.blocks)
    if (b.id + 1 > n)
      n = b.id + 1;
  ::llvm::SmallVector<PendingBlock> blocks(n);
  for (const Block &b : result.blocks) {
    if (b.tpl == nullptr)
      continue;
    PendingBlock pb;
    pb.ops.append(b.ops.begin(), b.ops.end());
    pb.tpl = b.tpl;
    blocks[b.id] = std::move(pb);
  }

  auto edges = collectBlockEdges(blocks, opToBlock);
  ReachMatrix verify;
  verify.rebuild(n, edges);

  // A multi-block SCC of size >= 2 manifests as a pair (i, j) with i != j
  // and reach[i] -> j and reach[j] -> i. Self-reach (i, i) is normal and
  // is excluded by the i != j guard.
  for (unsigned i = 0; i < n; ++i) {
    if (blocks[i].tpl == nullptr)
      continue;
    if (i >= verify.rows.size())
      continue;
    const ::llvm::BitVector &row_i = verify.rows[i];
    for (unsigned j = 0; j < n; ++j) {
      if (j == i)
        continue;
      if (blocks[j].tpl == nullptr)
        continue;
      if (j >= row_i.size())
        continue;
      if (!row_i.test(j))
        continue;
      if (j >= verify.rows.size())
        continue;
      if (i >= verify.rows[j].size())
        continue;
      if (verify.rows[j].test(i))
        return true;
    }
  }
  return false;
}

} // namespace fabric
