#include "Fabric/Tech/SubgraphMatcher.h"

#include "Fabric/Tech/SubgraphEnumerator.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
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

//===----------------------------------------------------------------------===//
// Attribute canonicalization helpers (shared with the partitioner).
//===----------------------------------------------------------------------===//

static std::string canonAttr(::mlir::Attribute a) {
  std::string s;
  ::llvm::raw_string_ostream os(s);
  a.print(os);
  return s;
}

static llvm::SmallVector<std::pair<std::string, std::string>, 4>
stripLoomAttrs(::mlir::ArrayRef<::mlir::NamedAttribute> attrs) {
  llvm::SmallVector<std::pair<std::string, std::string>, 4> out;
  for (::mlir::NamedAttribute na : attrs) {
    auto key = na.getName().getValue();
    if (key.starts_with("loom."))
      continue;
    out.emplace_back(key.str(), canonAttr(na.getValue()));
  }
  llvm::sort(out);
  return out;
}

//===----------------------------------------------------------------------===//
// VF2-style isomorphism on dataflow.subgraph bodies.
//
// The user pattern and the template are both `dataflow.subgraph` ops with
// graph-region semantics. Each body op produces SSA results whose users are
// either other body ops or the terminating `dataflow.yield`. The template
// produced by SubgraphEnumerator already has the "reduced" form: mux/demux
// are elided during materialization, leaving a pure compute DAG.
//
// The matcher builds, for each subgraph, a flat node list (ops in body
// program order) and, for each operand, a "source" descriptor:
//   * BodyOp(idx, resultNum): operand is produced by the i-th body op,
//     specifically its `resultNum`-th result.
//   * BlockArg(idx): operand is the i-th block argument of the subgraph.
// A bijection on body ops together with a bijection on block args defines
// an isomorphism iff every operand source is consistently mapped.
//===----------------------------------------------------------------------===//

struct Source {
  enum Kind : uint8_t { BodyOp, BlockArg } kind;
  unsigned idx;        // body-op position OR block-arg position
  unsigned resultNum;  // only meaningful for BodyOp
  bool operator==(const Source &o) const {
    return kind == o.kind && idx == o.idx &&
           (kind == BlockArg || resultNum == o.resultNum);
  }
  bool operator!=(const Source &o) const { return !(*this == o); }
};

struct NodeInfo {
  ::mlir::Operation *op = nullptr;
  ::llvm::StringRef opName;
  unsigned numOperands = 0;
  unsigned numResults = 0;
  // Source for each operand.
  llvm::SmallVector<Source, 4> operands;
  // Result types as canonical strings (used as a coarse pre-filter).
  llvm::SmallVector<std::string, 2> resultTypeKeys;
  // Sorted operand-type key multiset (coarse pre-filter).
  llvm::SmallVector<std::string, 4> sortedOperandTypeKeys;
  // Canonical attribute key/value pairs (already sorted).
  llvm::SmallVector<std::pair<std::string, std::string>, 4> attrKeys;
};

struct GraphView {
  ::dataflow::SubgraphOp sg;
  ::mlir::Block *body = nullptr;
  unsigned numBlockArgs = 0;
  // Body ops in program order.
  llvm::SmallVector<NodeInfo, 8> nodes;
  // Per-value descriptor (source) keyed by SSA Value.
  llvm::DenseMap<::mlir::Value, Source> valueSource;
  // Yield operand sources (in yield-operand order).
  llvm::SmallVector<Source, 4> yieldSources;
  // Block-arg types (canonical strings).
  llvm::SmallVector<std::string, 4> blockArgTypeKeys;
};

static std::string typeKey(::mlir::Type t) {
  std::string s;
  ::llvm::raw_string_ostream os(s);
  t.print(os);
  return s;
}

// Build a GraphView from a SubgraphOp. Returns false on malformed bodies
// (e.g., body op operand referring to a Value not produced inside the
// subgraph and not a block arg).
static bool buildGraphView(::dataflow::SubgraphOp sg, GraphView &gv) {
  gv.sg = sg;
  gv.body = &sg.getBody().front();
  gv.numBlockArgs = gv.body->getNumArguments();
  gv.blockArgTypeKeys.reserve(gv.numBlockArgs);
  for (unsigned i = 0; i < gv.numBlockArgs; ++i) {
    auto a = gv.body->getArgument(i);
    Source s{Source::BlockArg, i, 0};
    gv.valueSource[a] = s;
    gv.blockArgTypeKeys.push_back(typeKey(a.getType()));
  }

  // Index body ops first so we can resolve same-block back/forward edges.
  llvm::DenseMap<::mlir::Operation *, unsigned> opIdx;
  unsigned i = 0;
  for (::mlir::Operation &op : gv.body->without_terminator()) {
    opIdx[&op] = i;
    for (auto [r, res] : llvm::enumerate(op.getResults())) {
      Source s{Source::BodyOp, i, static_cast<unsigned>(r)};
      gv.valueSource[res] = s;
    }
    ++i;
  }
  gv.nodes.resize(i);

  i = 0;
  for (::mlir::Operation &op : gv.body->without_terminator()) {
    NodeInfo &ni = gv.nodes[i];
    ni.op = &op;
    ni.opName = op.getName().getStringRef();
    ni.numOperands = op.getNumOperands();
    ni.numResults = op.getNumResults();
    ni.operands.reserve(ni.numOperands);
    for (::mlir::Value v : op.getOperands()) {
      auto it = gv.valueSource.find(v);
      if (it == gv.valueSource.end())
        return false;
      ni.operands.push_back(it->second);
    }
    ni.resultTypeKeys.reserve(ni.numResults);
    for (::mlir::Type t : op.getResultTypes())
      ni.resultTypeKeys.push_back(typeKey(t));
    ni.sortedOperandTypeKeys.reserve(ni.numOperands);
    for (::mlir::Type t : op.getOperandTypes())
      ni.sortedOperandTypeKeys.push_back(typeKey(t));
    llvm::sort(ni.sortedOperandTypeKeys);
    ni.attrKeys = stripLoomAttrs(op.getAttrs());
    ++i;
  }

  // Yield operands (terminator).
  ::mlir::Operation *term = gv.body->getTerminator();
  if (term) {
    for (::mlir::Value v : term->getOperands()) {
      auto it = gv.valueSource.find(v);
      if (it == gv.valueSource.end())
        return false;
      gv.yieldSources.push_back(it->second);
    }
  }
  return true;
}

// VF2 search state.
struct VF2State {
  const GraphView &U; // user
  const GraphView &T; // template
  // M[u] = t means user node u is mapped to template node t.
  llvm::SmallVector<int, 8> M;
  // M_inv[t] = u or -1.
  llvm::SmallVector<int, 8> Minv;
  // BA[ub] = tb means user block-arg ub maps to template block-arg tb.
  llvm::SmallVector<int, 4> BA;
  // BA_inv[tb] = ub or -1.
  llvm::SmallVector<int, 4> BAinv;

  VF2State(const GraphView &u, const GraphView &t)
      : U(u), T(t),
        M(u.nodes.size(), -1),
        Minv(t.nodes.size(), -1),
        BA(u.numBlockArgs, -1),
        BAinv(t.numBlockArgs, -1) {}
};

// Coarse compatibility filter on (user, template) node pairs. Returns
// false when no consistent bijection between u and t is possible from
// purely-local features. This avoids descending into hopeless branches.
static bool nodeFeaturesCompatible(const NodeInfo &u, const NodeInfo &t) {
  if (u.opName != t.opName)
    return false;
  if (u.numOperands != t.numOperands)
    return false;
  if (u.numResults != t.numResults)
    return false;
  if (u.resultTypeKeys != t.resultTypeKeys)
    return false;
  if (u.sortedOperandTypeKeys != t.sortedOperandTypeKeys)
    return false;
  if (u.attrKeys != t.attrKeys)
    return false;
  return true;
}

// Try to extend the partial bijection by binding user block-arg `ub` to
// template block-arg `tb`. Returns false on contradiction with prior
// bindings.
static bool tryBindBlockArg(VF2State &S, int ub, int tb) {
  if (ub < 0 || tb < 0)
    return false;
  if (S.BA[ub] != -1)
    return S.BA[ub] == tb;
  if (S.BAinv[tb] != -1)
    return S.BAinv[tb] == ub;
  // Block-arg types must agree.
  if (S.U.blockArgTypeKeys[ub] != S.T.blockArgTypeKeys[tb])
    return false;
  S.BA[ub] = tb;
  S.BAinv[tb] = ub;
  return true;
}

static void unbindBlockArg(VF2State &S, int ub) {
  int tb = S.BA[ub];
  if (tb < 0)
    return;
  S.BA[ub] = -1;
  S.BAinv[tb] = -1;
}

// A handful of arith/math op kinds are commutative: their operands may
// be permuted without changing semantics. We allow their VF2 operand
// matching to permute user operand positions to template operand positions.
// Non-commutative ops are matched position-by-position.
static bool isCommutativeOp(::llvm::StringRef name) {
  // Listed once in canonical order so the runtime has a tight static set.
  // arith: addi, muli, andi, ori, xori, addf, mulf, mins/maxs/u-i,
  // minimumf/maximumf, cmpi/cmpf for symmetric predicates only -- but
  // since cmpi predicate is part of the attr key, swapping operands of a
  // non-symmetric predicate would change it; we leave cmp out of this list
  // to be safe.
  return name == "arith.addi" || name == "arith.muli" ||
         name == "arith.andi" || name == "arith.ori" || name == "arith.xori" ||
         name == "arith.addf" || name == "arith.mulf" ||
         name == "arith.minsi" || name == "arith.maxsi" ||
         name == "arith.minui" || name == "arith.maxui" ||
         name == "arith.minimumf" || name == "arith.maximumf";
}

// Try to bind operands of user node `u` to operands of template node `t`
// under permutation `perm` (perm[user_pos] = template_pos). Verifies SSA
// source consistency. On success the partial bijection in `S` is extended
// (and the caller is responsible for snapshotting / restoring); on failure
// the function rolls back any block-arg bindings it added.
static bool tryOperandPermutation(VF2State &S, unsigned u, unsigned t,
                                  ::llvm::ArrayRef<unsigned> perm) {
  const NodeInfo &uN = S.U.nodes[u];
  const NodeInfo &tN = S.T.nodes[t];
  llvm::SmallVector<int, 4> newlyBound;
  for (unsigned pu = 0; pu < uN.numOperands; ++pu) {
    unsigned pt = perm[pu];
    Source us = uN.operands[pu];
    Source ts = tN.operands[pt];
    if (us.kind != ts.kind) {
      for (int ub : newlyBound)
        unbindBlockArg(S, ub);
      return false;
    }
    if (us.kind == Source::BodyOp) {
      if (us.resultNum != ts.resultNum) {
        for (int ub : newlyBound)
          unbindBlockArg(S, ub);
        return false;
      }
      int mappedTo = S.M[us.idx];
      if (mappedTo != -1 && mappedTo != static_cast<int>(ts.idx)) {
        for (int ub : newlyBound)
          unbindBlockArg(S, ub);
        return false;
      }
      int invMappedFrom = S.Minv[ts.idx];
      if (invMappedFrom != -1 && invMappedFrom != static_cast<int>(us.idx)) {
        for (int ub : newlyBound)
          unbindBlockArg(S, ub);
        return false;
      }
    } else {
      int prev = S.BA[us.idx];
      if (!tryBindBlockArg(S, static_cast<int>(us.idx),
                           static_cast<int>(ts.idx))) {
        for (int ub : newlyBound)
          unbindBlockArg(S, ub);
        return false;
      }
      if (prev == -1)
        newlyBound.push_back(static_cast<int>(us.idx));
    }
  }
  return true;
}

// Verify that the operand sources at user node `u` (already mapped to
// template node `t`) are consistent under the current bijection. For a
// commutative op kind, tries operand permutations and commits the first
// admissible one (deterministic by lexicographic permutation order). For
// non-commutative ops, requires positional match.
static bool checkAndBindOperandSources(VF2State &S, unsigned u, unsigned t) {
  const NodeInfo &uN = S.U.nodes[u];
  if (uN.numOperands == 0)
    return true;

  // Identity permutation as the first try.
  llvm::SmallVector<unsigned, 4> perm(uN.numOperands);
  for (unsigned i = 0; i < uN.numOperands; ++i)
    perm[i] = i;

  if (!isCommutativeOp(uN.opName)) {
    return tryOperandPermutation(S, u, t, perm);
  }

  // Snapshot the partial bijection so we can roll back between attempts.
  auto savedBA = S.BA;
  auto savedBAinv = S.BAinv;
  do {
    if (tryOperandPermutation(S, u, t, perm))
      return true;
    S.BA = savedBA;
    S.BAinv = savedBAinv;
  } while (std::next_permutation(perm.begin(), perm.end()));
  return false;
}

// Verify that the operand sources at user node `u` (with M[u] bound) are
// consistent under the current bijection. Tries operand permutations for
// commutative ops; otherwise requires positional match.
static bool sourcesConsistent(const VF2State &S, unsigned u) {
  const NodeInfo &uN = S.U.nodes[u];
  int tIdx = S.M[u];
  if (tIdx < 0)
    return false;
  const NodeInfo &tN = S.T.nodes[tIdx];
  if (uN.numOperands != tN.numOperands)
    return false;
  // For non-commutative ops, positional check.
  auto checkPosition = [&](::llvm::ArrayRef<unsigned> perm) -> bool {
    for (unsigned pu = 0; pu < uN.numOperands; ++pu) {
      unsigned pt = perm[pu];
      Source us = uN.operands[pu];
      Source ts = tN.operands[pt];
      if (us.kind != ts.kind)
        return false;
      if (us.kind == Source::BodyOp) {
        if (us.resultNum != ts.resultNum)
          return false;
        int mappedTo = S.M[us.idx];
        if (mappedTo != static_cast<int>(ts.idx))
          return false;
      } else {
        int boundTo = S.BA[us.idx];
        if (boundTo != static_cast<int>(ts.idx))
          return false;
      }
    }
    return true;
  };
  llvm::SmallVector<unsigned, 4> perm(uN.numOperands);
  for (unsigned i = 0; i < uN.numOperands; ++i)
    perm[i] = i;
  if (!isCommutativeOp(uN.opName))
    return checkPosition(perm);
  do {
    if (checkPosition(perm))
      return true;
  } while (std::next_permutation(perm.begin(), perm.end()));
  return false;
}

// Check that the yield operand bijection is consistent.
static bool yieldConsistent(const VF2State &S) {
  if (S.U.yieldSources.size() != S.T.yieldSources.size())
    return false;
  for (auto [us, ts] : llvm::zip(S.U.yieldSources, S.T.yieldSources)) {
    if (us.kind != ts.kind)
      return false;
    if (us.kind == Source::BodyOp) {
      if (us.resultNum != ts.resultNum)
        return false;
      int mappedTo = S.M[us.idx];
      if (mappedTo != static_cast<int>(ts.idx))
        return false;
    } else {
      int boundTo = S.BA[us.idx];
      if (boundTo != static_cast<int>(ts.idx))
        return false;
    }
  }
  return true;
}

// Recursive VF2 backtracking. We pick the next user node in body program
// order (lowest unmapped index) and try every template candidate of
// matching coarse features in template program order; for each candidate
// we tentatively bind, run a local consistency check on operand sources,
// and recurse. On full bijection we additionally check the yield wiring.
static bool vf2Match(VF2State &S, unsigned nextU) {
  if (nextU == S.U.nodes.size()) {
    // All user nodes mapped. Re-validate operand sources end-to-end and the
    // yield wiring (newly-bound block args during descent could otherwise
    // have been left unverified along some path).
    for (unsigned u = 0; u < S.U.nodes.size(); ++u)
      if (!sourcesConsistent(S, u))
        return false;
    return yieldConsistent(S);
  }

  const NodeInfo &uN = S.U.nodes[nextU];
  for (unsigned t = 0; t < S.T.nodes.size(); ++t) {
    if (S.Minv[t] != -1)
      continue;
    if (!nodeFeaturesCompatible(uN, S.T.nodes[t]))
      continue;

    // Save state.
    llvm::SmallVector<int, 4> savedBA = S.BA;
    llvm::SmallVector<int, 4> savedBAinv = S.BAinv;

    S.M[nextU] = t;
    S.Minv[t] = nextU;
    if (checkAndBindOperandSources(S, nextU, t)) {
      if (vf2Match(S, nextU + 1))
        return true;
    }
    // Roll back.
    S.M[nextU] = -1;
    S.Minv[t] = -1;
    S.BA = std::move(savedBA);
    S.BAinv = std::move(savedBAinv);
  }
  return false;
}

static bool isomorphicViews(const GraphView &U, const GraphView &T) {
  if (U.numBlockArgs != T.numBlockArgs)
    return false;
  if (U.nodes.size() != T.nodes.size())
    return false;
  if (U.yieldSources.size() != T.yieldSources.size())
    return false;
  // Block-arg type multisets must match (we permute over block args).
  llvm::SmallVector<std::string, 4> ua = U.blockArgTypeKeys;
  llvm::SmallVector<std::string, 4> ta = T.blockArgTypeKeys;
  llvm::sort(ua);
  llvm::sort(ta);
  if (ua != ta)
    return false;
  // Op-name multiset must match too.
  llvm::SmallVector<std::string, 8> uns, tns;
  uns.reserve(U.nodes.size());
  tns.reserve(T.nodes.size());
  for (const NodeInfo &n : U.nodes) uns.push_back(n.opName.str());
  for (const NodeInfo &n : T.nodes) tns.push_back(n.opName.str());
  llvm::sort(uns);
  llvm::sort(tns);
  if (uns != tns)
    return false;

  VF2State S(U, T);
  return vf2Match(S, 0);
}

} // namespace

bool subgraphsIsomorphic(::dataflow::SubgraphOp user,
                         ::dataflow::SubgraphOp tpl) {
  if (!user || !tpl)
    return false;
  // Quick reject on subgraph-level signature.
  if (user.getInputs().size() != tpl.getInputs().size())
    return false;
  if (user.getResultTypes().size() != tpl.getResultTypes().size())
    return false;
  GraphView U, T;
  if (!buildGraphView(user, U))
    return false;
  if (!buildGraphView(tpl, T))
    return false;
  // Result-type multiset must match.
  llvm::SmallVector<std::string, 4> ur, tr;
  for (::mlir::Type t : user.getResultTypes()) ur.push_back(typeKey(t));
  for (::mlir::Type t : tpl.getResultTypes()) tr.push_back(typeKey(t));
  llvm::sort(ur);
  llvm::sort(tr);
  if (ur != tr)
    return false;
  return isomorphicViews(U, T);
}

bool subgraphsStructurallyEqual(::dataflow::SubgraphOp a,
                                ::dataflow::SubgraphOp b) {
  return subgraphsIsomorphic(a, b);
}

FuMatchResult mapPatternToFu(::dataflow::SubgraphOp pattern, FuOp fu,
                             ::mlir::ModuleOp tempModule) {
  ::llvm::StringRef unsupported;
  auto cands = enumerateFuSubgraphs(fu, tempModule, "match_tmp", &unsupported);
  FuMatchResult r;
  for (auto &c : cands) {
    if (subgraphsIsomorphic(pattern, c.subgraph)) {
      r.matched = true;
      r.fu = fu;
      r.configDescription = c.configDescription;
      r.swConfigsByOp = std::move(c.swConfigsByOp);
      break;
    }
  }
  return r;
}

} // namespace fabric
