#include "PartitionerCommon.h"

#include "Dataflow/IR/DataflowOps.h"
#include "Fabric/Tech/Partitioner/CostModel.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/ArrayRef.h"
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

// VF2-style isomorphism match for the partitioner's multi-op coverage.
//
// The user side is a `dataflow.graph` body; the template side is a
// `dataflow.subgraph` body materialized from one effective FU configuration
// (mux/demux already elided -- pure compute DAG). We anchor the user's
// `root` to the template's "sink" (the op feeding the first yield operand)
// and then expand by walking template body ops in program order, trying to
// bind each one to a user op in `root->getBlock()` whose op kind, arity,
// result-type list, operand-type multiset and attribute dict match.
//
// Operand sources can be:
//   * another body op result (must respect the bijection),
//   * a value external to the template body (block-arg or external def);
//     on the user side this maps to ANY value that is (a) external to the
//     selected user op set and (b) consistent with previous external-source
//     bindings. This permits commutative-operand permutations and also
//     handles patterns where the user's external-input wiring differs from
//     the canonical template wiring.
namespace {

struct PNode {
  ::mlir::Operation *op;
  ::llvm::StringRef opName;
  unsigned numOperands;
  unsigned numResults;
  ::llvm::SmallVector<std::pair<int, unsigned>, 4> operandRefs;
  // operandRefs[p] = (-1, 0)  ==> external value
  // operandRefs[p] = (i, r)   ==> body op i, result number r
  ::llvm::SmallVector<std::string, 2> resultTypeKeys;
  ::llvm::SmallVector<std::string, 4> sortedOperandTypeKeys;
  ::llvm::SmallVector<std::pair<std::string, std::string>, 4> attrKeys;
};

static std::string canonTypeKey(::mlir::Type t) {
  std::string s;
  ::llvm::raw_string_ostream os(s);
  t.print(os);
  return s;
}

// Build the per-template node list, anchored on the sink (yield-feeder),
// in template program order. Returns empty on malformed bodies. The sink
// is guaranteed to be the last entry.
static ::llvm::SmallVector<PNode, 4>
buildTemplateNodes(::dataflow::SubgraphOp tplSubgraph) {
  ::llvm::SmallVector<PNode, 4> out;
  ::mlir::Block &body = tplSubgraph.getBody().front();
  ::llvm::DenseMap<::mlir::Operation *, int> idx;
  int i = 0;
  for (::mlir::Operation &op : body.without_terminator()) {
    idx[&op] = i++;
  }
  out.reserve(i);
  for (::mlir::Operation &op : body.without_terminator()) {
    PNode n;
    n.op = &op;
    n.opName = op.getName().getStringRef();
    n.numOperands = op.getNumOperands();
    n.numResults = op.getNumResults();
    n.operandRefs.reserve(n.numOperands);
    for (::mlir::Value v : op.getOperands()) {
      ::mlir::Operation *def = v.getDefiningOp();
      auto it = (def ? idx.find(def) : idx.end());
      if (it == idx.end()) {
        n.operandRefs.push_back({-1, 0});
      } else {
        unsigned r = ::llvm::cast<::mlir::OpResult>(v).getResultNumber();
        n.operandRefs.push_back({it->second, r});
      }
    }
    n.resultTypeKeys.reserve(n.numResults);
    for (::mlir::Type t : op.getResultTypes())
      n.resultTypeKeys.push_back(canonTypeKey(t));
    n.sortedOperandTypeKeys.reserve(n.numOperands);
    for (::mlir::Type t : op.getOperandTypes())
      n.sortedOperandTypeKeys.push_back(canonTypeKey(t));
    std::sort(n.sortedOperandTypeKeys.begin(), n.sortedOperandTypeKeys.end());
    n.attrKeys = ::fabric::stripLoomAttrs(op.getAttrs());
    out.push_back(std::move(n));
  }
  return out;
}

// Coarse pre-filter: same op kind + arity + types + attrs.
static bool nodeFeaturesCompatible(::mlir::Operation *u, const PNode &t) {
  if (u->getName().getStringRef() != t.opName)
    return false;
  if (u->getNumOperands() != t.numOperands)
    return false;
  if (u->getNumResults() != t.numResults)
    return false;
  ::llvm::SmallVector<std::string, 2> uRes;
  uRes.reserve(u->getNumResults());
  for (::mlir::Type tt : u->getResultTypes())
    uRes.push_back(canonTypeKey(tt));
  if (uRes != t.resultTypeKeys)
    return false;
  ::llvm::SmallVector<std::string, 4> uOp;
  uOp.reserve(u->getNumOperands());
  for (::mlir::Type tt : u->getOperandTypes())
    uOp.push_back(canonTypeKey(tt));
  std::sort(uOp.begin(), uOp.end());
  if (uOp != t.sortedOperandTypeKeys)
    return false;
  if (::fabric::stripLoomAttrs(u->getAttrs()) != t.attrKeys)
    return false;
  return true;
}

struct MatchState {
  // M[i] = user op bound to template node i (or nullptr).
  ::llvm::SmallVector<::mlir::Operation *> M;
  // Reverse lookup.
  ::llvm::DenseSet<::mlir::Operation *> usedUserOps;
  // External-source bindings: a template "external" Value t_ext maps to
  // some user Value u_ext; both maps must stay consistent.
  ::llvm::DenseMap<::mlir::Value, ::mlir::Value> extTplToUsr;
  ::llvm::DenseMap<::mlir::Value, ::mlir::Value> extUsrToTpl;
};

static bool isCommutativeOp(::llvm::StringRef name) {
  return name == "arith.addi" || name == "arith.muli" ||
         name == "arith.andi" || name == "arith.ori" || name == "arith.xori" ||
         name == "arith.addf" || name == "arith.mulf" ||
         name == "arith.minsi" || name == "arith.maxsi" ||
         name == "arith.minui" || name == "arith.maxui" ||
         name == "arith.minimumf" || name == "arith.maximumf";
}

// Verify body-internal operand consistency under operand permutation
// `perm` (perm[user_pos] = template_pos): every template body-ref operand
// must, in the user's mapping, point at M[bIdx] with the right result num.
//
// Graph-region templates may carry back-edges (bIdx > tIdx). For those the
// producer is bound LATER in the search, so `S.M[bIdx]` is still nullptr at
// this point. We DEFER the check: the caller re-runs source consistency over
// every bound node once the bijection is complete (see vf2VerifyAllSources).
static bool topologyConsistentPerm(const MatchState &S,
                                   const ::llvm::SmallVector<PNode, 4> &T,
                                   unsigned tIdx, ::mlir::Operation *usrOp,
                                   ::llvm::ArrayRef<unsigned> perm) {
  const PNode &tN = T[tIdx];
  for (unsigned pu = 0; pu < tN.numOperands; ++pu) {
    unsigned pt = perm[pu];
    auto [bIdx, rNum] = tN.operandRefs[pt];
    if (bIdx < 0)
      continue;
    ::mlir::Operation *expected = S.M[bIdx];
    if (expected == nullptr)
      // Producer not yet bound (back-edge in template program order).
      // Verification is deferred to the end-of-search source check.
      continue;
    ::mlir::Value uVal = usrOp->getOperand(pu);
    ::mlir::Operation *uDef = uVal.getDefiningOp();
    if (uDef != expected)
      return false;
    unsigned uRes = ::llvm::cast<::mlir::OpResult>(uVal).getResultNumber();
    if (uRes != rNum)
      return false;
  }
  return true;
}

// End-of-search verification: every template body-ref operand of every
// bound node must, in the user mapping, point at M[bIdx] with the right
// result num. Required because topologyConsistentPerm defers checks for
// back-edge producers (bIdx > tIdx) that get bound later in the search.
//
// We only verify body-ref operands here; external bindings are already
// committed/checked in bindExternalsPerm at each step.
static bool vf2VerifyAllSources(const MatchState &S,
                                const ::llvm::SmallVector<PNode, 4> &T) {
  for (unsigned i = 0; i < T.size(); ++i) {
    ::mlir::Operation *uOp = S.M[i];
    if (uOp == nullptr)
      return false;
    const PNode &tN = T[i];
    for (unsigned pt = 0; pt < tN.numOperands; ++pt) {
      auto [bIdx, rNum] = tN.operandRefs[pt];
      if (bIdx < 0)
        continue;
      ::mlir::Operation *expected = S.M[bIdx];
      if (expected == nullptr)
        return false;
      // The user-side operand position that mapped to `pt` could be any
      // permutation under commutativity; recover it by scanning the user
      // op's operands for the unique one whose def matches.
      bool ok = false;
      for (unsigned pu = 0; pu < uOp->getNumOperands(); ++pu) {
        ::mlir::Value uVal = uOp->getOperand(pu);
        if (uVal.getDefiningOp() != expected)
          continue;
        unsigned uRes = ::llvm::cast<::mlir::OpResult>(uVal).getResultNumber();
        if (uRes != rNum)
          continue;
        ok = true;
        break;
      }
      if (!ok)
        return false;
    }
  }
  return true;
}

// Tentatively bind external-source operands under `perm`. On success the
// caller is responsible for snapshotting / restoring; on failure rolls
// back any bindings this call added.
static bool bindExternalsPerm(MatchState &S,
                              const ::llvm::SmallVector<PNode, 4> &T,
                              unsigned tIdx, ::mlir::Operation *usrOp,
                              ::llvm::ArrayRef<unsigned> perm) {
  ::llvm::SmallVector<std::pair<::mlir::Value, ::mlir::Value>, 4> added;
  const PNode &tN = T[tIdx];
  for (unsigned pu = 0; pu < tN.numOperands; ++pu) {
    unsigned pt = perm[pu];
    auto [bIdx, rNum] = tN.operandRefs[pt];
    if (bIdx >= 0)
      continue;
    ::mlir::Value tVal = tN.op->getOperand(pt);
    ::mlir::Value uVal = usrOp->getOperand(pu);
    auto it = S.extTplToUsr.find(tVal);
    if (it != S.extTplToUsr.end()) {
      if (it->second != uVal) {
        for (auto &kv : added) {
          S.extTplToUsr.erase(kv.first);
          S.extUsrToTpl.erase(kv.second);
        }
        return false;
      }
    } else {
      auto rev = S.extUsrToTpl.find(uVal);
      if (rev != S.extUsrToTpl.end() && rev->second != tVal) {
        for (auto &kv : added) {
          S.extTplToUsr.erase(kv.first);
          S.extUsrToTpl.erase(kv.second);
        }
        return false;
      }
      S.extTplToUsr[tVal] = uVal;
      S.extUsrToTpl[uVal] = tVal;
      added.push_back({tVal, uVal});
    }
  }
  return true;
}

static bool vf2RecMulti(MatchState &S, const ::llvm::SmallVector<PNode, 4> &T,
                        ::mlir::Block *userBlock, unsigned nextT,
                        unsigned sinkIdx, ::mlir::Operation *sinkUserOp) {
  if (nextT == T.size())
    return vf2VerifyAllSources(S, T);
  const PNode &tN = T[nextT];

  // Iterate user ops in block program order for determinism. When this is
  // the sink template node, the user op is forced to `sinkUserOp`.
  auto tryBind = [&](::mlir::Operation *uOp) -> bool {
    if (S.usedUserOps.contains(uOp))
      return false;
    if (!nodeFeaturesCompatible(uOp, tN))
      return false;
    S.M[nextT] = uOp;
    // Build the operand permutation to try. For commutative ops, iterate
    // all permutations of operand positions; for non-commutative ops, only
    // the identity permutation. The first admissible permutation wins.
    ::llvm::SmallVector<unsigned, 4> perm(tN.numOperands);
    for (unsigned i = 0; i < tN.numOperands; ++i)
      perm[i] = i;
    bool isComm = isCommutativeOp(tN.opName);
    auto savedTplToUsr = S.extTplToUsr;
    auto savedUsrToTpl = S.extUsrToTpl;
    do {
      if (!topologyConsistentPerm(S, T, nextT, uOp, perm))
        goto next_perm;
      if (!bindExternalsPerm(S, T, nextT, uOp, perm))
        goto next_perm;
      S.usedUserOps.insert(uOp);
      if (vf2RecMulti(S, T, userBlock, nextT + 1, sinkIdx, sinkUserOp))
        return true;
      S.usedUserOps.erase(uOp);
      S.extTplToUsr = savedTplToUsr;
      S.extUsrToTpl = savedUsrToTpl;
    next_perm:;
    } while (isComm && std::next_permutation(perm.begin(), perm.end()));
    S.M[nextT] = nullptr;
    return false;
  };

  if (nextT == sinkIdx) {
    if (tryBind(sinkUserOp))
      return true;
    return false;
  }
  for (::mlir::Operation &uOp : *userBlock) {
    if (uOp.hasTrait<::mlir::OpTrait::IsTerminator>())
      continue;
    if (&uOp == sinkUserOp)
      continue;
    if (tryBind(&uOp))
      return true;
  }
  return false;
}

} // namespace

::llvm::SmallVector<::mlir::Operation *>
collectMultiOpCandidate(::mlir::Operation *root, const FuTemplate &tpl) {
  ::llvm::SmallVector<::mlir::Operation *> empty;
  unsigned n = tpl.bodyOpCount;
  if (n == 0)
    return empty;

  ::dataflow::SubgraphOp tplSubgraph = tpl.subgraph;
  ::llvm::SmallVector<PNode, 4> tplNodes = buildTemplateNodes(tplSubgraph);
  if (tplNodes.size() != n)
    return empty;

  // The template's sink is the producer of yield operand 0. Under
  // graph-region semantics the yielded op is not required to be the last
  // body op in textual order, so we look up its index in tplNodes rather
  // than assuming sinkIdx == n - 1.
  ::mlir::Operation *yieldOp = tplSubgraph.getBody().front().getTerminator();
  if (!yieldOp || yieldOp->getNumOperands() == 0)
    return empty;
  ::mlir::Operation *tplSink = yieldOp->getOperand(0).getDefiningOp();
  if (tplSink == nullptr)
    return empty;
  unsigned sinkIdx = n;
  for (unsigned i = 0; i < tplNodes.size(); ++i) {
    if (tplNodes[i].op == tplSink) {
      sinkIdx = i;
      break;
    }
  }
  if (sinkIdx == n)
    return empty;

  // Quick reject: sink op kind / arity / attrs must match `root`.
  if (!nodeFeaturesCompatible(root, tplNodes[sinkIdx]))
    return empty;

  if (n == 1) {
    // Single-op template: name + signature already verified by the cache;
    // just check feature compatibility above (already done).
    return {root};
  }

  MatchState S;
  S.M.assign(n, nullptr);
  // Run generic VF2 with sink forced to `root`.
  if (!vf2RecMulti(S, tplNodes, /*userBlock=*/root->getBlock(), /*nextT=*/0,
                   sinkIdx, /*sinkUserOp=*/root))
    return empty;

  // Build result in template program order (matches the original API's
  // "body program order" contract).
  ::llvm::SmallVector<::mlir::Operation *> result;
  result.reserve(n);
  for (unsigned i = 0; i < n; ++i) {
    if (S.M[i] == nullptr)
      return empty;
    result.push_back(S.M[i]);
  }
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

void addSingletonBlockToReach(unsigned newId, ReachMatrix &reach) {
  unsigned newSize = newId + 1;
  reach.ensureSize(newSize);
  ::llvm::BitVector &row = reach.rows[newId];
  if (row.size() < newSize)
    row.resize(newSize);
  row.set(newId);
}

AcceptDelta
computeAcceptDelta(::llvm::ArrayRef<::mlir::Operation *> ops,
                   const FuTemplate *tpl, const TemplateLibrary &lib,
                   const ::llvm::DenseMap<::mlir::Operation *, unsigned>
                       &opToBlock) {
  AcceptDelta d;

  ::llvm::DenseSet<::mlir::Operation *> inCand;
  for (::mlir::Operation *op : ops)
    inCand.insert(op);

  // Producer-side: each operand of a candidate op whose def is already in
  // some block (and not in the candidate itself) becomes a cross-block edge
  // once the candidate is accepted. Each operand position is counted, in
  // line with `computeCost`'s per-operand-position semantics.
  int delta = 0;
  for (::mlir::Operation *op : ops) {
    for (::mlir::Value v : op->getOperands()) {
      ::mlir::Operation *def = v.getDefiningOp();
      if (!def)
        continue;
      if (inCand.contains(def))
        continue;
      if (opToBlock.find(def) != opToBlock.end())
        ++delta;
    }
  }

  // Consumer-side: each user of a candidate op's result, if that user is
  // already in some block (and not in the candidate), gains a cross-block
  // edge. `Value::getUsers()` iterates per-Use, so a user that consumes the
  // same result through multiple operand positions is correctly counted
  // multiple times — matching the cost model.
  for (::mlir::Operation *op : ops) {
    for (::mlir::Value res : op->getResults()) {
      for (::mlir::Operation *user : res.getUsers()) {
        if (inCand.contains(user))
          continue;
        if (opToBlock.find(user) != opToBlock.end())
          ++delta;
      }
    }
  }
  d.crossEdges = delta;

  if (tpl != nullptr) {
    d.blocksWithTemplate = 1;
    d.densityCount = 1;
    // Cap = max bodyOpCount across templates sharing this rootOpName.
    // Equivalent to `maxTemplateSizeByRoot(lib).lookup(tpl->rootOpName)`,
    // inlined to avoid building the full StringMap for one query.
    unsigned cap = 1;
    for (const FuTemplate &t : lib.templates()) {
      if (t.rootOpName == tpl->rootOpName && t.bodyOpCount > cap)
        cap = t.bodyOpCount;
    }
    d.densityNumerator =
        static_cast<double>(ops.size()) / static_cast<double>(cap);
  }
  return d;
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
