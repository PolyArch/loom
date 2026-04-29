#ifndef LOOM_FABRIC_TECH_SYNTHESIZER_ALIGNMENT_H
#define LOOM_FABRIC_TECH_SYNTHESIZER_ALIGNMENT_H

// Alignment facade over `dataflow.subgraph` bodies for the synthesizer's
// anchor / mcs / incremental / incremental_random strategies.
//
// This header is intentionally a thin wrapper around the matcher's
// `GraphView` data model (see `Fabric/Tech/SubgraphGraphView.h`) so that
// synthesis and matching agree byte-for-byte on:
//
//   * how a value's producer is named (`Source`: BodyOp / BlockArg /
//     BackEdge),
//   * which arith / math op kinds get commutative-operand normalization,
//   * how multi-result body ops (e.g. `dataflow.stream`) are addressed,
//   * which graph-region in-body uses are classified as back-edges
//     (via the textual-position back-edge rule documented on
//     `backEdges`; a strict superset of true SCC back-edges).
//
// Spec source: `docs/spec-generalize-subgraphs-to-fu.md`, section
// "Sub-algorithms shared by strategies > Alignment".
//
// Threading: every function below is read-only over the input MLIR
// objects and stateless; safe to call from any worker thread provided
// the underlying subgraph is not being mutated concurrently.

#include "Dataflow/IR/DataflowOps.h"

#include "mlir/IR/Operation.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>

namespace loom::fabric::tech {

// Source descriptor: how a value is produced inside a subgraph.
//
//   BodyOp:    `op->getResult(resultIndex)` for an op in the subgraph
//              body; `op` is the producing op pointer.
//   BlockArg:  the subgraph's `argIndex`-th block argument (live-in
//              from the enclosing function).
//   BackEdge:  in a graph-region body, a value produced by an op that
//              also consumes (transitively) one of the consumer's
//              results -- identified by the textual-position rule
//              jointly with the matcher's `GraphView`. `op` and
//              `resultIndex` name the producing op / result, exactly as
//              for `BodyOp`. Strategies use the kind to decide whether
//              the consuming edge is reserved as a placeholder (and
//              resolved at back-edge emission) versus walked normally.
struct Source {
  enum Kind { BodyOp, BlockArg, BackEdge } kind;
  ::mlir::Operation *op = nullptr; // BodyOp / BackEdge
  unsigned resultIndex = 0;        // BodyOp / BackEdge
  unsigned argIndex = 0;           // BlockArg
  bool operator==(const Source &o) const;
  bool operator!=(const Source &o) const { return !(*this == o); }
};

// Hash combiner for Source. Stable across runs (built on
// `llvm::hash_value` over the type-erased fields and the kind tag).
::llvm::hash_code hash_value(const Source &s);

// Per-source signature collapsing op identity, share-group id,
// bit-width, arity, and the *kinds* of operand sources (NOT operand
// identity). Two subgraph positions are alignable iff their signatures
// match AND their per-operand source kinds line up; commutative
// operand permutations are normalized inside `signatureOf` so the same
// SSA shape -- regardless of textual operand order -- yields one
// signature.
//
// `op` borrows from the MLIR registered-op-name interning pool, so it
// outlives any single pass invocation. NodeSignature is trivially
// copyable and safe to cache across thread boundaries.
struct NodeSignature {
  ::llvm::StringRef op;
  ::std::optional<::std::size_t> shareGroup;
  unsigned bitwidth = 0;
  unsigned arity = 0;
  ::llvm::SmallVector<Source::Kind, 4> operandKinds;
  uint64_t structuralHash = 0; // stable, deterministic
  bool operator==(const NodeSignature &o) const;
};

// Build a NodeSignature for the producer named by `s`. Only meaningful
// when `s.kind == BodyOp` or `s.kind == BackEdge`; for `BlockArg` the
// returned signature has empty `op`, zero arity and bitwidth, and a
// hash derived from the arg index + kind tag (so block-arg positions
// remain distinguishable but never collide with body-op signatures).
NodeSignature signatureOf(Source s);

// Yield anchors: the ordered list of `Source` descriptors for the
// `dataflow.yield`'s operands. This is the canonical entry point for
// anchor / mcs / incremental alignment: every strategy walks its
// subgraphs starting from these anchor sources.
::llvm::SmallVector<Source, 4>
yieldAnchors(::dataflow::SubgraphOp sg);

// Resolve op operand `operandIdx` to a Source. Handles:
//   * `BlockArg`: when the operand is the subgraph's block argument;
//   * `BodyOp`:   when the operand is produced by another body op
//                 (also captures multi-result by setting `resultIndex`);
//   * `BackEdge`: when the producer is downstream of the consumer in
//                 the body's textual order. Identified by a textual-
//                 position approximation (see `backEdges` below) -- a
//                 strict superset of true SCC back-edges that is
//                 sufficient for the synthesizer's placeholder-
//                 reservation needs. Only meaningful in graph-region
//                 bodies; in DAG-only bodies this never returns
//                 `BackEdge`.
//
// `consumer` must be an op inside `dataflow.subgraph`'s body or its
// `dataflow.yield` terminator (handled identically). Calling on an op
// outside the subgraph body is undefined.
Source operandSource(::mlir::Operation *consumer, unsigned operandIdx);

// Lightweight back-edge identification: returns the set of
// `(consumer, operandIdx)` pairs whose `operandSource` would return
// `BackEdge`. The current implementation uses a textual-position rule
// (an operand whose producer's body index is `>=` the consumer's index
// is classified as a back-edge); this is a strict superset of the true
// SCC back-edge set and is sufficient for placeholder reservation
// during FU emission. A future pass that needs exact SCC membership
// (e.g. for cycle-length reasoning) can replace this with a Tarjan
// pre-pass without changing the public API. The matcher itself does
// not need to make this distinction (its value-source map is direction-
// agnostic), so synth and matcher remain in lockstep.
::llvm::DenseSet<::std::pair<::mlir::Operation *, unsigned>>
backEdges(::dataflow::SubgraphOp sg);

} // namespace loom::fabric::tech

#endif // LOOM_FABRIC_TECH_SYNTHESIZER_ALIGNMENT_H
