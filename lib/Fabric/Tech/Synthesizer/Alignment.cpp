#include "Fabric/Tech/Synthesizer/Alignment.h"

#include "Common/HwShareGroup.h"
#include "Fabric/Tech/SubgraphGraphView.h"

#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstdint>
#include <utility>

namespace loom::fabric::tech {

namespace gv = ::loom::fabric::tech::detail;

//===----------------------------------------------------------------------===//
// Source equality + hash.
//===----------------------------------------------------------------------===//

bool Source::operator==(const Source &o) const {
  if (kind != o.kind)
    return false;
  switch (kind) {
  case BlockArg:
    return argIndex == o.argIndex;
  case BodyOp:
  case BackEdge:
    return op == o.op && resultIndex == o.resultIndex;
  }
  return false;
}

::llvm::hash_code hash_value(const Source &s) {
  switch (s.kind) {
  case Source::BlockArg:
    return ::llvm::hash_combine(static_cast<unsigned>(s.kind), s.argIndex);
  case Source::BodyOp:
  case Source::BackEdge:
    return ::llvm::hash_combine(static_cast<unsigned>(s.kind),
                                reinterpret_cast<uintptr_t>(s.op),
                                s.resultIndex);
  }
  return ::llvm::hash_value(static_cast<unsigned>(s.kind));
}

bool NodeSignature::operator==(const NodeSignature &o) const {
  return op == o.op && shareGroup == o.shareGroup && bitwidth == o.bitwidth &&
         arity == o.arity && operandKinds == o.operandKinds &&
         structuralHash == o.structuralHash;
}

//===----------------------------------------------------------------------===//
// Helpers shared between yieldAnchors / operandSource / backEdges.
//===----------------------------------------------------------------------===//

namespace {

// Returns the parent dataflow.subgraph of `op` if `op` is in a subgraph
// body; nullptr otherwise. `op` may be the yield terminator itself.
static ::dataflow::SubgraphOp parentSubgraph(::mlir::Operation *op) {
  if (!op)
    return {};
  ::mlir::Block *block = op->getBlock();
  if (!block)
    return {};
  ::mlir::Operation *parent = block->getParentOp();
  return ::llvm::dyn_cast_or_null<::dataflow::SubgraphOp>(parent);
}

// Maps every body op (excluding the terminator) of `sg` to its 0-based
// position in the body's program order. The yield terminator gets the
// sentinel position `numBodyOps` (so consumers in yield are after every
// body op and never produce back-edges from yield).
struct BodyPositions {
  ::llvm::DenseMap<::mlir::Operation *, unsigned> opIdx;
  unsigned numBodyOps = 0;
};

static BodyPositions collectBodyPositions(::dataflow::SubgraphOp sg) {
  BodyPositions bp;
  if (!sg)
    return bp;
  ::mlir::Block &body = sg.getBody().front();
  unsigned i = 0;
  for (::mlir::Operation &op : body.without_terminator()) {
    bp.opIdx[&op] = i;
    ++i;
  }
  bp.numBodyOps = i;
  return bp;
}

// Convert an arbitrary `mlir::Value` produced inside `sg`'s body to a
// public Source. The `consumerIdx` is the consumer's body position, used
// to flag back-edges (producer textual index >= consumer textual index
// means the value is defined after the consumer, i.e. a graph-region
// back-edge). Yield operands set `consumerIdx` to `numBodyOps` to avoid
// false back-edge classification.
static Source classifyValueAsSource(::mlir::Value v, const BodyPositions &bp) {
  Source s;
  if (auto barg = ::llvm::dyn_cast<::mlir::BlockArgument>(v)) {
    s.kind = Source::BlockArg;
    s.argIndex = barg.getArgNumber();
    return s;
  }
  auto opRes = ::llvm::cast<::mlir::OpResult>(v);
  ::mlir::Operation *producer = opRes.getOwner();
  s.op = producer;
  s.resultIndex = opRes.getResultNumber();
  s.kind = Source::BodyOp; // default; promoted to BackEdge by caller
  (void)bp;
  return s;
}

// Promote a BodyOp source to BackEdge iff the producer's textual
// position in the body is >= the consumer's position. This is a
// strict superset of true SCC back-edges and is sufficient for
// placeholder reservation during FU emission. The matcher treats
// every body-internal SSA value the same way (its valueSource map is
// direction-agnostic); the synthesizer surfaces this approximation
// only so back-edge endpoints can be reserved as placeholders.
// Yield-operand callers pass `consumerIdx = numBodyOps`, which never
// triggers promotion.
static void maybePromoteToBackEdge(Source &s, unsigned consumerIdx,
                                   const BodyPositions &bp) {
  if (s.kind != Source::BodyOp)
    return;
  auto it = bp.opIdx.find(s.op);
  if (it == bp.opIdx.end())
    return;
  unsigned prodIdx = it->second;
  if (prodIdx >= consumerIdx)
    s.kind = Source::BackEdge;
}

} // namespace

//===----------------------------------------------------------------------===//
// Public API.
//===----------------------------------------------------------------------===//

::llvm::SmallVector<Source, 4>
yieldAnchors(::dataflow::SubgraphOp sg) {
  ::llvm::SmallVector<Source, 4> out;
  if (!sg)
    return out;
  BodyPositions bp = collectBodyPositions(sg);
  ::mlir::Block &body = sg.getBody().front();
  ::mlir::Operation *term = body.getTerminator();
  if (!term)
    return out;
  out.reserve(term->getNumOperands());
  for (::mlir::Value v : term->getOperands()) {
    Source s = classifyValueAsSource(v, bp);
    // Yield is the terminator: every BodyOp producer is necessarily
    // earlier in textual order. Pass consumerIdx = numBodyOps so the
    // BackEdge promotion never triggers from a yield context.
    maybePromoteToBackEdge(s, bp.numBodyOps, bp);
    out.push_back(s);
  }
  return out;
}

Source operandSource(::mlir::Operation *consumer, unsigned operandIdx) {
  Source s;
  if (!consumer)
    return s;
  ::dataflow::SubgraphOp sg = parentSubgraph(consumer);
  if (!sg) {
    // Consumer is not inside a dataflow.subgraph body: classify on
    // operand identity alone, without back-edge promotion.
    ::mlir::Value v = consumer->getOperand(operandIdx);
    BodyPositions empty;
    return classifyValueAsSource(v, empty);
  }
  BodyPositions bp = collectBodyPositions(sg);
  ::mlir::Value v = consumer->getOperand(operandIdx);
  s = classifyValueAsSource(v, bp);
  // Decide consumer's textual position. The yield terminator gets the
  // sentinel `numBodyOps` so its operands are never back-edges.
  unsigned consumerIdx = bp.numBodyOps;
  auto it = bp.opIdx.find(consumer);
  if (it != bp.opIdx.end())
    consumerIdx = it->second;
  maybePromoteToBackEdge(s, consumerIdx, bp);
  return s;
}

::llvm::DenseSet<::std::pair<::mlir::Operation *, unsigned>>
backEdges(::dataflow::SubgraphOp sg) {
  ::llvm::DenseSet<::std::pair<::mlir::Operation *, unsigned>> out;
  if (!sg)
    return out;
  BodyPositions bp = collectBodyPositions(sg);
  ::mlir::Block &body = sg.getBody().front();
  for (::mlir::Operation &op : body.without_terminator()) {
    auto it = bp.opIdx.find(&op);
    if (it == bp.opIdx.end())
      continue;
    unsigned consumerIdx = it->second;
    for (unsigned i = 0, n = op.getNumOperands(); i < n; ++i) {
      ::mlir::Value v = op.getOperand(i);
      Source s = classifyValueAsSource(v, bp);
      maybePromoteToBackEdge(s, consumerIdx, bp);
      if (s.kind == Source::BackEdge)
        out.insert({&op, i});
    }
  }
  // The yield terminator is positioned after every body op, so its
  // operands are by construction never classified as back-edges.
  // We deliberately do not iterate the terminator here.
  return out;
}

//===----------------------------------------------------------------------===//
// signatureOf.
//===----------------------------------------------------------------------===//

namespace {

// Best-effort bit-width extraction matching the rest of synth's cost
// model semantics: integer / float / index width when known, else 0.
static unsigned bitwidthOfType(::mlir::Type t) {
  if (auto it = ::llvm::dyn_cast<::mlir::IntegerType>(t))
    return it.getWidth();
  if (auto ft = ::llvm::dyn_cast<::mlir::FloatType>(t))
    return ft.getWidth();
  if (::llvm::isa<::mlir::IndexType>(t))
    return 64;
  return 0;
}

// Operand-kind sequence for a body op. For commutative ops (per the
// matcher's static `isCommutativeOp` set) the kinds are sorted so that
// `arith.addi %a, %b` and `arith.addi %b, %a` produce identical
// signatures. For non-commutative ops the textual operand order is
// preserved -- positional matching depends on it.
static ::llvm::SmallVector<Source::Kind, 4>
canonicalOperandKinds(::mlir::Operation *op) {
  ::llvm::SmallVector<Source::Kind, 4> kinds;
  kinds.reserve(op->getNumOperands());
  for (unsigned i = 0, n = op->getNumOperands(); i < n; ++i) {
    Source s = operandSource(op, i);
    kinds.push_back(s.kind);
  }
  if (gv::isCommutativeOp(op->getName().getStringRef())) {
    // Sort by enum value so the canonical order is purely structural.
    std::sort(kinds.begin(), kinds.end(),
              [](Source::Kind a, Source::Kind b) {
                return static_cast<unsigned>(a) < static_cast<unsigned>(b);
              });
  }
  return kinds;
}

// Compute the structural hash for a body-op source. Combines op name,
// share-group index (or sentinel -1 for singletons), bitwidth, arity,
// and the canonicalized operand-kind sequence. The hash never depends
// on operand identity (Source.op pointer / argIndex) -- the matcher's
// VF2 search resolves identity through the bijection, not the hash.
static uint64_t structuralHashFor(::llvm::StringRef opName,
                                  ::std::optional<::std::size_t> shareGroup,
                                  unsigned bitwidth, unsigned arity,
                                  ::llvm::ArrayRef<Source::Kind> kinds) {
  ::llvm::hash_code h = ::llvm::hash_combine(opName);
  uint64_t sgTag = shareGroup.has_value() ? static_cast<uint64_t>(*shareGroup)
                                          : ~static_cast<uint64_t>(0);
  h = ::llvm::hash_combine(h, sgTag);
  h = ::llvm::hash_combine(h, bitwidth);
  h = ::llvm::hash_combine(h, arity);
  for (Source::Kind k : kinds)
    h = ::llvm::hash_combine(h, static_cast<unsigned>(k));
  return static_cast<uint64_t>(h);
}

} // namespace

NodeSignature signatureOf(Source s) {
  NodeSignature ns;
  if (s.kind == Source::BlockArg) {
    // Block-arg "signatures" exist only so callers can compare anchors
    // uniformly. They never collide with body-op signatures (the kind
    // tag is folded into the structural hash).
    ns.op = ::llvm::StringRef();
    ns.shareGroup.reset();
    ns.bitwidth = 0;
    ns.arity = 0;
    ns.operandKinds.clear();
    ns.structuralHash = static_cast<uint64_t>(::llvm::hash_combine(
        static_cast<unsigned>(s.kind), s.argIndex));
    return ns;
  }
  // BodyOp / BackEdge: identical signature shape -- the textual-
  // position rule distinguishes them via the Source::kind tag, but
  // the producer's structural identity is the same in both cases.
  ::mlir::Operation *op = s.op;
  if (!op) {
    // Defensive: unresolved Source.
    ns.structuralHash = static_cast<uint64_t>(::llvm::hash_combine(
        static_cast<unsigned>(s.kind)));
    return ns;
  }
  ns.op = op->getName().getStringRef();
  ns.shareGroup = ::loom::common::findShareGroup(ns.op);
  // Bitwidth comes from the *result* the source names; multi-result ops
  // (e.g. dataflow.stream produces (index, rwc)) need the per-result
  // type so the signature distinguishes the index port from the rwc
  // port.
  if (s.resultIndex < op->getNumResults())
    ns.bitwidth = bitwidthOfType(op->getResult(s.resultIndex).getType());
  ns.arity = op->getNumOperands();
  ns.operandKinds = canonicalOperandKinds(op);
  ns.structuralHash = structuralHashFor(ns.op, ns.shareGroup, ns.bitwidth,
                                        ns.arity, ns.operandKinds);
  return ns;
}

} // namespace loom::fabric::tech
