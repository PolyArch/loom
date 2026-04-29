#ifndef FABRIC_TECH_SUBGRAPHGRAPHVIEW_H
#define FABRIC_TECH_SUBGRAPHGRAPHVIEW_H

// Internal shared GraphView model over `dataflow.subgraph` bodies.
//
// This header was extracted from the file-local helpers of
// `lib/Fabric/Tech/SubgraphMatcher.cpp` so that the synthesizer's
// Alignment facade and the matcher's VF2 search agree byte-for-byte on
// node ordering, source descriptors, commutative-operand canonicalization,
// and (downstream) SCC back-edge identification. The matcher continues
// to consume these primitives through this header; the alignment facade
// (include/Fabric/Tech/Synthesizer/Alignment.h) wraps them with the
// public Source / NodeSignature API mandated by the spec
// (`docs/spec-generalize-subgraphs-to-fu.md`, section "Alignment").
//
// The names live in `loom::fabric::tech::detail` to make it clear they
// are an implementation detail shared between the matcher and the
// synthesizer: callers outside those two areas should prefer the public
// Alignment API.

#include "Dataflow/IR/DataflowOps.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <string>
#include <utility>

namespace loom::fabric::tech::detail {

// Source descriptor for a value flowing inside a subgraph body. Two
// kinds are tracked at build time:
//   * BodyOp(idx, resultNum): produced by the i-th body op (in textual
//     program order), specifically its `resultNum`-th result.
//   * BlockArg(idx): forwarded from the i-th block argument of the
//     subgraph's entry block.
// Back-edge classification (graph-region bodies only) is layered on
// top by callers via SCC analysis; it is not a separate kind here so
// that the matcher's existing source-resolution model is unchanged.
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

// Per-body-op cached metadata: op pointer, name, arity, result types
// (canonical strings), and the per-operand Source vector.
struct NodeInfo {
  ::mlir::Operation *op = nullptr;
  ::llvm::StringRef opName;
  unsigned numOperands = 0;
  unsigned numResults = 0;
  // Source for each operand.
  ::llvm::SmallVector<Source, 4> operands;
  // Result types as canonical strings (used as a coarse pre-filter
  // by the matcher).
  ::llvm::SmallVector<std::string, 2> resultTypeKeys;
  // Sorted operand-type key multiset (coarse pre-filter).
  ::llvm::SmallVector<std::string, 4> sortedOperandTypeKeys;
  // Canonical attribute key/value pairs (already sorted).
  ::llvm::SmallVector<std::pair<std::string, std::string>, 4> attrKeys;
};

// Cached structural view over a `dataflow.subgraph`. Built once per
// subgraph and shared between the matcher and the alignment facade.
struct GraphView {
  ::dataflow::SubgraphOp sg;
  ::mlir::Block *body = nullptr;
  unsigned numBlockArgs = 0;
  // Body ops in program order.
  ::llvm::SmallVector<NodeInfo, 8> nodes;
  // Per-value descriptor (source) keyed by SSA Value.
  ::llvm::DenseMap<::mlir::Value, Source> valueSource;
  // Yield operand sources (in yield-operand order).
  ::llvm::SmallVector<Source, 4> yieldSources;
  // Block-arg types (canonical strings).
  ::llvm::SmallVector<std::string, 4> blockArgTypeKeys;
};

// Canonical printer for a single MLIR Type. Inlined into a string so
// the result can participate in equality comparisons / hashing without
// keeping the originating MLIR objects alive.
std::string typeKey(::mlir::Type t);

// Canonical printer for a single MLIR Attribute. Same contract as
// typeKey: result is a stable string suitable for ordering / hashing.
std::string canonAttr(::mlir::Attribute a);

// Returns the named attribute list of `attrs` with any `loom.*` keys
// elided, each value canonicalized via `canonAttr`, sorted by key.
::llvm::SmallVector<std::pair<std::string, std::string>, 4>
stripLoomAttrs(::mlir::ArrayRef<::mlir::NamedAttribute> attrs);

// Build a GraphView from a SubgraphOp. Returns false on malformed
// bodies (e.g., body op operand referring to a Value not produced
// inside the subgraph and not a block arg).
bool buildGraphView(::dataflow::SubgraphOp sg, GraphView &gv);

// Whether `name` is one of the commutative arith / math op kinds the
// matcher and the alignment facade jointly normalize over. Listed in
// canonical order to keep the runtime set tight.
bool isCommutativeOp(::llvm::StringRef name);

} // namespace loom::fabric::tech::detail

#endif // FABRIC_TECH_SUBGRAPHGRAPHVIEW_H
