// Anchor strategy: lock-step BFS from yield anchors. Handles tier-A
// topology-isomorphic input groups by walking every input subgraph in
// parallel from its `dataflow.yield` operands, classifying each
// position as `BlockArg` / `BodyOp` / `BackEdge`, and emitting one
// `fabric.op` per body-op position. When all peer ops at a position
// share one hardware-share group the strategy emits a single fabric.op
// whose `op_list` is the sorted union of observed op names; when peers
// disagree on share group and `SynthConfig.anchorAllowIntraPositionMux`
// is true, the strategy emits one fabric.op per share-group bucket and
// joins them through a fresh `fabric.mux`. Otherwise the strategy
// returns `cross_share_group`.
//
// Spec source: `docs/spec-generalize-subgraphs-to-fu.md`, sections
// "Strategy: anchor (tier A by default)" and "Acceptance criteria
// (anchor)".

#include "Fabric/Tech/Synthesizer/Anchor.h"

#include "Common/HwShareGroup.h"
#include "Common/IndexWidth.h"
#include "Common/SynthConfig.h"
#include "Fabric/IR/FabricOps.h"
#include "Fabric/IR/FabricTypes.h"
#include "Fabric/Tech/Synthesizer/Alignment.h"
#include "Fabric/Tech/Synthesizer/CostModel.h"
#include "Fabric/Tech/Synthesizer/HwParams.h"
#include "Fabric/Tech/Synthesizer/Synthesizer.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace loom::fabric::tech {

namespace {

//===----------------------------------------------------------------------===//
// Type lifting helpers.
//===----------------------------------------------------------------------===//
//
// Input subgraph block-arg types are software types (iN / fN / index /
// i1). The synthesized FU's signature exposes `fabric.bits<N>` ports
// only. Mirror the forward direction's bit-width assignments so the
// inverse lifting agrees with `SubgraphEnumerator::computePortLiftMap`
// and `liftFor`.

// Bit width of an MLIR software type expressible as a fabric.bits<N>
// port. Returns std::nullopt when the type cannot be lifted (caller
// treats as a topology mismatch). NoneType (e.g. dataflow.constant's
// ctrl input) lifts to bits<0> -- a legitimate, zero-width port.
::std::optional<unsigned> tryBitWidthOf(::mlir::Type t) {
  if (auto i = ::llvm::dyn_cast<::mlir::IntegerType>(t))
    return i.getWidth();
  if (auto f = ::llvm::dyn_cast<::mlir::FloatType>(t))
    return f.getWidth();
  if (::llvm::isa<::mlir::IndexType>(t))
    return ::loom::getIndexWidth();
  if (::llvm::isa<::mlir::NoneType>(t))
    return 0u;
  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// Worker-side state for one anchor synthesis run.
//===----------------------------------------------------------------------===//

// Identity of a "peer set" -- one Source descriptor per input
// subgraph for the same lock-step BFS position. `PeerKey` collapses
// the per-input pointer/index identities into a hashable value so the
// `visited` map can dedup DAG fanout.
struct PeerKey {
  ::llvm::SmallVector<Source, 4> peers;
  bool operator==(const PeerKey &o) const { return peers == o.peers; }
};

struct PeerKeyInfo {
  static PeerKey getEmptyKey() {
    PeerKey k;
    Source s;
    s.kind = Source::BlockArg;
    s.argIndex = ~0u;
    k.peers.push_back(s);
    return k;
  }
  static PeerKey getTombstoneKey() {
    PeerKey k;
    Source s;
    s.kind = Source::BlockArg;
    s.argIndex = ~0u - 1;
    k.peers.push_back(s);
    return k;
  }
  static unsigned getHashValue(const PeerKey &k) {
    ::llvm::hash_code h = ::llvm::hash_value(static_cast<unsigned>(0));
    for (const Source &s : k.peers)
      h = ::llvm::hash_combine(h, hash_value(s));
    return static_cast<unsigned>(h);
  }
  static bool isEqual(const PeerKey &a, const PeerKey &b) { return a == b; }
};

// A fabric.bits<N> Value already emitted for a peer set.
struct EmittedSlot {
  ::mlir::Value value;
};

// Per-position bucket: every input subgraph contributes one Source.
using PeerVec = ::llvm::SmallVector<Source, 4>;

// Helper: build a PeerKey from a PeerVec.
PeerKey keyOf(const PeerVec &v) {
  PeerKey k;
  k.peers.assign(v.begin(), v.end());
  return k;
}

// "Op key" for a peer set that ignores the per-anchor resultIndex.
// Multiple anchors may name distinct results of the same source op
// (e.g. `dataflow.stream`'s `idx` at #0 and `rwc` at #1); they all
// share one fabric.op emission and only differ in result projection.
PeerKey opKeyOf(const PeerVec &v) {
  PeerKey k;
  k.peers.reserve(v.size());
  for (const Source &s : v) {
    Source proj = s;
    if (s.kind == Source::BodyOp || s.kind == Source::BackEdge)
      proj.resultIndex = 0;
    k.peers.push_back(proj);
  }
  return k;
}

//===----------------------------------------------------------------------===//
// Wrapper-port assignment (block-arg identity).
//===----------------------------------------------------------------------===//
//
// All inputs are tier-A: their subgraphs share one DAG topology, so
// they also share one block-arg shape. The wrapper exposes one input
// port per block-arg index of the canonical (input #0) subgraph, with
// the bit-width drawn from the union of observed widths (which must
// agree, otherwise the position is a `topology_mismatch`).

// Compute the wrapper's input ports: one entry per block-arg index of
// input subgraphs (all subgraphs must agree on the per-index
// bit-width). Used by the AnchorSynthesizer to size the inner FU's
// entry block before BFS materialization.
struct WrapperInputSlot {
  unsigned argIndex;
  unsigned bitwidth;
};

::std::optional<::llvm::SmallVector<WrapperInputSlot, 4>>
collectWrapperInputSlots(::llvm::ArrayRef<::dataflow::SubgraphOp> sgs) {
  ::llvm::SmallVector<WrapperInputSlot, 4> ports;
  if (sgs.empty())
    return ports;
  ::dataflow::SubgraphOp first = sgs.front();
  if (!first)
    return std::nullopt;
  ::mlir::Block &fb = first.getBody().front();
  unsigned na = fb.getNumArguments();
  // Tier A precondition: every subgraph has the same arg count.
  for (auto sg : sgs)
    if (sg.getBody().front().getNumArguments() != na)
      return std::nullopt;
  ports.reserve(na);
  for (unsigned i = 0; i < na; ++i) {
    auto firstBw = tryBitWidthOf(fb.getArgument(i).getType());
    if (!firstBw.has_value())
      return std::nullopt;
    unsigned bw = *firstBw;
    for (auto sg : sgs) {
      ::mlir::Block &b = sg.getBody().front();
      auto other = tryBitWidthOf(b.getArgument(i).getType());
      if (!other.has_value() || *other != bw)
        return std::nullopt;
    }
    WrapperInputSlot p;
    p.argIndex = i;
    p.bitwidth = bw;
    ports.push_back(p);
  }
  return ports;
}

// Compute the wrapper's expected yield-side bit-widths: one entry per
// `dataflow.yield` operand of the canonical (input #0) subgraph. All
// inputs must already agree on yield arity (caller's precondition);
// per-position width uniformity is checked lazily at peer materialization
// time inside the BFS, so here we trust the canonical subgraph's widths.
::std::optional<::llvm::SmallVector<unsigned, 4>>
collectWrapperOutputWidths(::llvm::ArrayRef<::dataflow::SubgraphOp> sgs) {
  ::llvm::SmallVector<unsigned, 4> widths;
  if (sgs.empty())
    return widths;
  ::dataflow::SubgraphOp first = sgs.front();
  if (!first)
    return std::nullopt;
  ::mlir::Block &fb = first.getBody().front();
  ::mlir::Operation *yield = fb.getTerminator();
  if (!yield)
    return std::nullopt;
  unsigned ar = yield->getNumOperands();
  // Tier A precondition: every subgraph has the same yield arity.
  for (auto sg : sgs) {
    ::mlir::Operation *y = sg.getBody().front().getTerminator();
    if (!y || y->getNumOperands() != ar)
      return std::nullopt;
  }
  widths.reserve(ar);
  for (unsigned i = 0; i < ar; ++i) {
    auto bw = tryBitWidthOf(yield->getOperand(i).getType());
    if (!bw.has_value())
      return std::nullopt;
    widths.push_back(*bw);
  }
  return widths;
}

//===----------------------------------------------------------------------===//
// Source classification helpers.
//===----------------------------------------------------------------------===//

// Validate that all peers at a position share one Source::Kind. A
// mismatch is a topology violation (e.g. one input has BlockArg here
// while another has BodyOp).
bool peersUniformKind(const PeerVec &peers) {
  if (peers.empty())
    return false;
  Source::Kind k = peers.front().kind;
  for (const Source &s : peers)
    if (s.kind != k)
      return false;
  return true;
}

// Bit-width of the value the source names. For BlockArg: width of the
// block argument; for BodyOp / BackEdge: width of the named result.
// Returns std::nullopt when the type cannot be lifted to fabric.bits<N>;
// returns Some(0) for legitimate zero-width values (NoneType ctrl tokens).
::std::optional<unsigned> widthOfSource(const Source &s,
                                        ::dataflow::SubgraphOp sg) {
  if (s.kind == Source::BlockArg) {
    if (s.argIndex >= sg.getBody().front().getNumArguments())
      return std::nullopt;
    return tryBitWidthOf(sg.getBody().front().getArgument(s.argIndex).getType());
  }
  if (!s.op || s.resultIndex >= s.op->getNumResults())
    return std::nullopt;
  return tryBitWidthOf(s.op->getResult(s.resultIndex).getType());
}

// Validate that all peers at a body-op position name a result with the
// same bit-width. Mismatch -> topology_mismatch.
bool peersUniformWidth(const PeerVec &peers,
                       ::llvm::ArrayRef<::dataflow::SubgraphOp> sgs,
                       unsigned &widthOut) {
  unsigned w = 0;
  for (auto [i, s] : ::llvm::enumerate(peers)) {
    auto cur = widthOfSource(s, sgs[i]);
    if (!cur.has_value())
      return false;
    if (i == 0)
      w = *cur;
    else if (*cur != w)
      return false;
  }
  widthOut = w;
  return true;
}

// Validate that all peers at a body-op position share one operand
// arity. (Tier A precondition: same DAG topology means same per-node
// arity.)
bool peersUniformArity(const PeerVec &peers, unsigned &arityOut) {
  if (peers.empty())
    return false;
  unsigned ar = peers.front().op ? peers.front().op->getNumOperands() : 0;
  for (const Source &s : peers) {
    if (!s.op)
      return false;
    if (s.op->getNumOperands() != ar)
      return false;
  }
  arityOut = ar;
  return true;
}

//===----------------------------------------------------------------------===//
// fabric.op emission.
//===----------------------------------------------------------------------===//

// Build a length-1 ArrayAttr wrapping an empty DictionaryAttr; the
// canonical form for `hw_params` when the inner op kind has no
// configurable axis (per spec "hw_params policy"). Tier A's anchor
// strategy only handles fixed-arity ops (addi/subi/muli/etc.) without
// configurable axes, so this is the universal `hw_params` value.
::mlir::ArrayAttr emptyHwParams(::mlir::MLIRContext *ctx) {
  auto emptyDict = ::mlir::DictionaryAttr::get(ctx, {});
  ::llvm::SmallVector<::mlir::Attribute, 1> outer{emptyDict};
  return ::mlir::ArrayAttr::get(ctx, outer);
}

// Build the sorted-union op_list ArrayAttr for a single share group.
// Sort key is the op-name string; the spec requires lexical order so
// the canonical printed form is deterministic.
::mlir::ArrayAttr sortedOpListFor(const ::std::set<::std::string> &names,
                                  ::mlir::MLIRContext *ctx) {
  ::llvm::SmallVector<::mlir::Attribute, 4> attrs;
  attrs.reserve(names.size());
  for (const ::std::string &n : names)
    attrs.push_back(::mlir::FlatSymbolRefAttr::get(ctx, n));
  return ::mlir::ArrayAttr::get(ctx, attrs);
}

// Construct one `fabric.op` instance with the given op_list, hw_params,
// inputs, and result types. The op is emitted at the builder's current
// insertion point.
::fabric::OpOp emitFabricOp(::mlir::OpBuilder &builder, ::mlir::Location loc,
                            ::mlir::ArrayAttr opList,
                            ::mlir::ArrayAttr hwParams,
                            ::mlir::ValueRange operands,
                            ::mlir::TypeRange resultTypes) {
  ::mlir::OperationState state(loc, ::fabric::OpOp::getOperationName());
  state.addOperands(operands);
  state.addTypes(resultTypes);
  state.addAttribute("op_list", opList);
  if (hwParams)
    state.addAttribute("hw_params", hwParams);
  ::mlir::Operation *raw = builder.create(state);
  return ::mlir::cast<::fabric::OpOp>(raw);
}

// Construct one `fabric.mux` over `arms` with the shared bits type.
::fabric::MuxOp emitFabricMux(::mlir::OpBuilder &builder, ::mlir::Location loc,
                              ::mlir::ValueRange arms, ::mlir::Type bits) {
  ::mlir::OperationState state(loc, ::fabric::MuxOp::getOperationName());
  state.addOperands(arms);
  state.addTypes({bits});
  ::mlir::Operation *raw = builder.create(state);
  return ::mlir::cast<::fabric::MuxOp>(raw);
}

//===----------------------------------------------------------------------===//
// Per-position decision: single-share-group vs cross-share-group merge.
//===----------------------------------------------------------------------===//

// Key for a share-group bucket. A multi-member group is identified by
// its index in `loom::common::hwShareGroups()`; a singleton op (any op
// not in any multi-member group) is identified by its own op name so
// that distinct singletons at the same anchor position never collapse
// into one bucket. Bucket order is deterministic: multi-member groups
// (sorted by index) come first, then singletons sorted by op name.
struct BucketKey {
  bool isSingleton = false;
  ::std::size_t groupIndex = 0; // valid iff !isSingleton
  ::std::string singletonName; // valid iff isSingleton

  static BucketKey forGroup(::std::size_t idx) {
    BucketKey k;
    k.isSingleton = false;
    k.groupIndex = idx;
    return k;
  }
  static BucketKey forSingleton(::llvm::StringRef name) {
    BucketKey k;
    k.isSingleton = true;
    k.singletonName = name.str();
    return k;
  }
  static BucketKey forName(::llvm::StringRef name) {
    if (auto idx = ::loom::common::findShareGroup(name))
      return forGroup(*idx);
    return forSingleton(name);
  }

  bool operator<(const BucketKey &o) const {
    // Multi-member groups (isSingleton == false) sort before singletons
    // so deterministic ordering is stable across all bucket sets.
    if (isSingleton != o.isSingleton)
      return !isSingleton; // false < true
    if (!isSingleton)
      return groupIndex < o.groupIndex;
    return singletonName < o.singletonName;
  }
};

// One layout candidate for a body-op position. `useMux == false` means
// a single fabric.op merging every peer's op name into one op_list
// (must all share one share group). `useMux == true` means one
// fabric.op per share-group bucket fed from the same operands and
// joined by a fresh fabric.mux. The CostModel evaluates the resulting
// sub-FU and the lowest-cost legal candidate wins.
struct OpNodeDecision {
  bool useMux = false;
  // Buckets[k] = the sorted set of op names belonging to share-group k.
  // Each multi-member group (e.g. arith.addi/subi) contributes one
  // entry; each distinct singleton contributes its own entry. For
  // useMux == false the map has exactly one entry.
  ::std::map<BucketKey, ::std::set<::std::string>> buckets;
  // Parallel map: per-bucket source ops the union came from. The hw_params
  // synthesizer scans these for observed-attribute axes (predicate,
  // step_op, cont_cond, const_hex_value, bitmask).
  ::std::map<BucketKey, ::llvm::SmallVector<::mlir::Operation *, 4>>
      bucketPeerOps;
};

// Group peers by share-group key. Each multi-member group collapses
// its members under one key; each distinct singleton receives its own
// key (so two distinct singletons at the same anchor position never
// land in one bucket).
::std::map<BucketKey, ::std::set<::std::string>>
bucketBySharegroup(const PeerVec &peers) {
  ::std::map<BucketKey, ::std::set<::std::string>> out;
  for (const Source &s : peers) {
    if (!s.op)
      continue;
    ::llvm::StringRef name = s.op->getName().getStringRef();
    out[BucketKey::forName(name)].insert(name.str());
  }
  return out;
}

// Parallel collector to bucketBySharegroup that retains the source ops
// per bucket (so the hw_params synthesis can scan their attributes).
::std::map<BucketKey, ::llvm::SmallVector<::mlir::Operation *, 4>>
bucketPeerOpsBySharegroup(const PeerVec &peers) {
  ::std::map<BucketKey, ::llvm::SmallVector<::mlir::Operation *, 4>> out;
  for (const Source &s : peers) {
    if (!s.op)
      continue;
    ::llvm::StringRef name = s.op->getName().getStringRef();
    out[BucketKey::forName(name)].push_back(s.op);
  }
  return out;
}

// Decide the cheapest legal layout for a body-op peer set. Returns
// std::nullopt when no legal layout exists (e.g. cross-share-group
// peers and `allowMux == false`).
::std::optional<OpNodeDecision>
decideOpNode(const PeerVec &peers, bool allowMux,
             const AreaWeights &weights, unsigned bw) {
  auto buckets = bucketBySharegroup(peers);
  if (buckets.empty())
    return std::nullopt;

  // Each singleton has its own BucketKey, so any singleton bucket holds
  // exactly one op name; OpOp::verify's "singletons must occupy
  // fabric.op alone" rule is therefore satisfied by construction. No
  // additional bucket-level filtering is needed here.

  auto peerOps = bucketPeerOpsBySharegroup(peers);
  if (buckets.size() == 1) {
    OpNodeDecision d;
    d.useMux = false;
    d.buckets = std::move(buckets);
    d.bucketPeerOps = std::move(peerOps);
    return d;
  }
  // More than one bucket: cross-share-group (multi-member groups vs
  // each other, or distinct singletons against any other bucket). Only
  // legal when the user opted into intra-position muxing.
  if (!allowMux)
    return std::nullopt;
  OpNodeDecision d;
  d.useMux = true;
  d.buckets = std::move(buckets);
  d.bucketPeerOps = std::move(peerOps);
  // Cost ranking is trivial here -- there is only one cross-share-group
  // layout for tier A: one fabric.op per bucket + one fabric.mux
  // joining them. Future strategies may produce alternative layouts
  // (e.g. per-arm demuxing); the explicit cost call still belongs to
  // the spec contract so we keep the AreaWeights / bw parameters even
  // though the comparison is degenerate.
  (void)weights;
  (void)bw;
  return d;
}

//===----------------------------------------------------------------------===//
// Anchor BFS state.
//===----------------------------------------------------------------------===//

// All fabric.op results for one BodyOp peer set, kept around so that
// multiple anchors naming distinct resultIndex of the same source op
// share one fabric.op emission.
struct EmittedOp {
  ::llvm::SmallVector<::mlir::Value, 2> results;
};

struct AnchorState {
  AnchorState(::mlir::MLIRContext *c, ::mlir::Location l)
      : ctx(c), loc(l) {}
  ::mlir::MLIRContext *ctx = nullptr;
  ::mlir::Location loc;
  // Per-input subgraph copy held by AnchorSynthesizer::run.
  ::llvm::ArrayRef<::dataflow::SubgraphOp> sgs;
  // The wrapper (a func.func) and the inner fabric.fu.
  ::mlir::OpBuilder *bodyBuilder = nullptr;
  ::fabric::FuOp fu;
  // Wrapper-input port -> entry-block argument value of the inner FU.
  ::llvm::SmallVector<::mlir::Value, 4> portValues;
  // Cache: peer set -> emitted Value. Dedups DAG fanout.
  ::llvm::DenseMap<PeerKey, EmittedSlot, PeerKeyInfo> visited;
  // Cache keyed on the resultIndex-stripped peer set, holding the full
  // fabric.op result list. Lets distinct (BodyOp,resultIndex) anchors
  // share one fabric.op emission.
  ::llvm::DenseMap<PeerKey, EmittedOp, PeerKeyInfo> emittedOps;
  // Diagnostic notes accumulated during the run.
  ::llvm::SmallVector<::std::string, 4> notes;
};

// Forward declarations.
struct EmitOutcome {
  bool ok = false;
  SynthFailureReason reason = SynthFailureReason::None;
  ::mlir::Value value;
};

EmitOutcome materializePeers(AnchorState &st, const PeerVec &peers,
                             const ::loom::SynthConfig &cfg);

// Compute hw_params for the bucket whose op-name set is `names` and
// whose source-side peer ops are `peerOps`. The merged op_list within
// one bucket has at most one symbol per share-group entry; the helper
// chooses the merged op name as the lexically smallest name in `names`,
// which is the same name used to drive the enumerator's primary flavor
// detection. (Within a share-group all members share the same
// configurable axes by construction.)
::mlir::ArrayAttr
hwParamsForBucket(::mlir::MLIRContext *ctx,
                  const ::std::set<::std::string> &names,
                  ::llvm::ArrayRef<::mlir::Operation *> peerOps) {
  if (names.empty())
    return emptyHwParams(ctx);
  ::llvm::StringRef merged(*names.begin());
  return buildHwParamsUnion(ctx, merged, peerOps);
}

// Outcome of emitting one body-op position; returns ALL fabric.op
// result values (lifted to fabric.bits<N>) so callers can project the
// resultIndex of interest.
struct EmitOpOutcome {
  bool ok = false;
  SynthFailureReason reason = SynthFailureReason::None;
  ::llvm::SmallVector<::mlir::Value, 2> results;
};

// Compute the lifted (fabric.bits<N>) types for every result of the
// peers' source ops. Tier A: peers must agree on result count and per-
// index bit-width. Returns std::nullopt on a mismatch.
::std::optional<::llvm::SmallVector<::mlir::Type, 2>>
liftedResultTypesForPeers(::mlir::MLIRContext *ctx, const PeerVec &peers) {
  if (peers.empty() || !peers.front().op)
    return std::nullopt;
  unsigned nr = peers.front().op->getNumResults();
  ::llvm::SmallVector<unsigned, 2> bws;
  bws.reserve(nr);
  for (unsigned i = 0; i < nr; ++i) {
    auto bw =
        tryBitWidthOf(peers.front().op->getResult(i).getType());
    if (!bw.has_value())
      return std::nullopt;
    bws.push_back(*bw);
  }
  for (const Source &s : peers) {
    if (!s.op || s.op->getNumResults() != nr)
      return std::nullopt;
    for (unsigned i = 0; i < nr; ++i) {
      auto bw = tryBitWidthOf(s.op->getResult(i).getType());
      if (!bw.has_value() || *bw != bws[i])
        return std::nullopt;
    }
  }
  ::llvm::SmallVector<::mlir::Type, 2> out;
  out.reserve(nr);
  for (unsigned w : bws)
    out.push_back(::fabric::BitsType::get(ctx, w));
  return out;
}

// Emit a fabric.op (or per-bucket fabric.op + fabric.mux) for a body-op
// peer set whose operands have already been materialized into
// `operandValues`. Returns the FULL list of result values from the
// emitted fabric.op (or a single mux output for the cross-share-group
// case, which is single-result by construction).
EmitOpOutcome
emitBodyOpPositionMulti(AnchorState &st, const OpNodeDecision &decision,
                        ::mlir::ValueRange operandValues,
                        ::llvm::ArrayRef<::mlir::Type> resultTypes) {
  EmitOpOutcome r;
  if (!decision.useMux) {
    // Single share-group: one fabric.op with the sorted union.
    const auto &kv = *decision.buckets.begin();
    ::mlir::ArrayAttr opList = sortedOpListFor(kv.second, st.ctx);
    auto peerIt = decision.bucketPeerOps.find(kv.first);
    ::llvm::ArrayRef<::mlir::Operation *> peerOps =
        peerIt != decision.bucketPeerOps.end()
            ? ::llvm::ArrayRef<::mlir::Operation *>(peerIt->second)
            : ::llvm::ArrayRef<::mlir::Operation *>();
    ::mlir::ArrayAttr hwParams =
        hwParamsForBucket(st.ctx, kv.second, peerOps);
    auto op = emitFabricOp(*st.bodyBuilder, st.loc, opList, hwParams,
                           operandValues, resultTypes);
    r.ok = true;
    r.results.assign(op.getOutputs().begin(), op.getOutputs().end());
    return r;
  }
  // Cross-share-group with intra-position mux: per spec, only the
  // primary (single-result) output is muxed. Multi-result merging
  // across share groups is out of scope for tier A.
  if (resultTypes.size() != 1) {
    r.reason = SynthFailureReason::TopologyMismatch;
    return r;
  }
  ::llvm::SmallVector<::mlir::Value, 4> arms;
  arms.reserve(decision.buckets.size());
  for (const auto &kv : decision.buckets) {
    ::mlir::ArrayAttr opList = sortedOpListFor(kv.second, st.ctx);
    auto peerIt = decision.bucketPeerOps.find(kv.first);
    ::llvm::ArrayRef<::mlir::Operation *> peerOps =
        peerIt != decision.bucketPeerOps.end()
            ? ::llvm::ArrayRef<::mlir::Operation *>(peerIt->second)
            : ::llvm::ArrayRef<::mlir::Operation *>();
    ::mlir::ArrayAttr hwParams =
        hwParamsForBucket(st.ctx, kv.second, peerOps);
    auto opOp = emitFabricOp(*st.bodyBuilder, st.loc, opList, hwParams,
                             operandValues, resultTypes);
    arms.push_back(opOp.getOutputs()[0]);
  }
  auto mux = emitFabricMux(*st.bodyBuilder, st.loc, arms, resultTypes[0]);
  r.ok = true;
  r.results.push_back(mux.getOutput());
  return r;
}

// Anchor's lock-step BFS is tier A only: it requires every input to share
// the same DAG topology, with no graph-region back-edges. Tier C SCC
// handling (back-edge alignment, fabric.op[@dataflow.carry] emission) is
// the Incremental strategy's job. If the BFS reaches a BackEdge source
// here it means the caller fed us a tier-C input by mistake; report
// `TopologyMismatch` rather than emitting a verifier-violating placeholder
// inside the FU body. (FuOp::verify rejects everything that is not
// fabric.op / fabric.mux / fabric.demux, so emitting an
// `unrealized_conversion_cast` here would unconditionally fail
// verification once executed.)
EmitOutcome reserveBackEdgePlaceholder(AnchorState & /*st*/,
                                       unsigned /*bw*/) {
  EmitOutcome r;
  r.ok = false;
  r.reason = SynthFailureReason::TopologyMismatch;
  return r;
}

EmitOutcome materializePeers(AnchorState &st, const PeerVec &peers,
                             const ::loom::SynthConfig &cfg) {
  EmitOutcome out;

  // Dedup: the same peer set may be reached from multiple parents
  // (DAG fanout). Reuse the previously emitted Value verbatim.
  PeerKey key = keyOf(peers);
  auto it = st.visited.find(key);
  if (it != st.visited.end()) {
    out.ok = true;
    out.value = it->second.value;
    return out;
  }

  if (!peersUniformKind(peers)) {
    out.reason = SynthFailureReason::TopologyMismatch;
    return out;
  }

  Source::Kind kind = peers.front().kind;
  if (kind == Source::BlockArg) {
    // All peers must agree on the block-arg index for tier A. Any
    // mismatch is a topology error.
    unsigned idx = peers.front().argIndex;
    for (const Source &s : peers)
      if (s.argIndex != idx) {
        out.reason = SynthFailureReason::TopologyMismatch;
        return out;
      }
    if (idx >= st.portValues.size()) {
      out.reason = SynthFailureReason::TopologyMismatch;
      return out;
    }
    out.ok = true;
    out.value = st.portValues[idx];
    st.visited[key] = EmittedSlot{out.value};
    return out;
  }

  if (kind == Source::BackEdge) {
    unsigned bw = 0;
    if (!peersUniformWidth(peers, st.sgs, bw)) {
      out.reason = SynthFailureReason::TopologyMismatch;
      return out;
    }
    EmitOutcome ph = reserveBackEdgePlaceholder(st, bw);
    if (!ph.ok)
      return ph;
    st.visited[key] = EmittedSlot{ph.value};
    return ph;
  }

  // BodyOp: validate share-group, bit-width and arity uniformity.
  unsigned bw = 0;
  if (!peersUniformWidth(peers, st.sgs, bw)) {
    out.reason = SynthFailureReason::TopologyMismatch;
    return out;
  }
  unsigned arity = 0;
  if (!peersUniformArity(peers, arity)) {
    out.reason = SynthFailureReason::TopologyMismatch;
    return out;
  }

  // The peer set may have been emitted before under a different
  // resultIndex (e.g. yield #0 saw `dataflow.stream#0`, yield #1 saw
  // `dataflow.stream#1`). Reuse the same fabric.op by projecting the
  // requested resultIndex from the cached result list.
  PeerKey opKey = opKeyOf(peers);
  auto reuse = st.emittedOps.find(opKey);
  if (reuse != st.emittedOps.end()) {
    unsigned ri = peers.front().resultIndex;
    if (ri >= reuse->second.results.size()) {
      out.reason = SynthFailureReason::TopologyMismatch;
      return out;
    }
    out.ok = true;
    out.value = reuse->second.results[ri];
    st.visited[key] = EmittedSlot{out.value};
    return out;
  }

  // Pre-decision: distinct share-groups when intra-position muxing is
  // disabled is a `cross_share_group` failure (distinct from
  // `topology_mismatch`).
  auto buckets = bucketBySharegroup(peers);
  if (buckets.size() > 1 && !cfg.anchorAllowIntraPositionMux) {
    out.reason = SynthFailureReason::CrossShareGroup;
    return out;
  }

  // Recurse into operands. Tier A constraint: each peer.op shares the
  // same arity, so we can lock-step iterate operand index `i`.
  ::llvm::SmallVector<::mlir::Value, 4> operandValues;
  operandValues.reserve(arity);
  for (unsigned i = 0; i < arity; ++i) {
    PeerVec child;
    child.reserve(peers.size());
    for (const Source &s : peers) {
      Source childSrc = operandSource(s.op, i);
      child.push_back(childSrc);
    }
    EmitOutcome co = materializePeers(st, child, cfg);
    if (!co.ok) {
      out.reason = co.reason;
      return out;
    }
    operandValues.push_back(co.value);
  }

  AreaWeights weights;
  weights.muxPenalty = cfg.costMuxPenalty;
  weights.demuxPenalty = cfg.costDemuxPenalty;
  weights.carryPenalty = cfg.costCarryPenalty;
  auto decision = decideOpNode(peers, cfg.anchorAllowIntraPositionMux,
                               weights, bw);
  if (!decision.has_value()) {
    if (buckets.size() > 1)
      out.reason = SynthFailureReason::CrossShareGroup;
    else
      out.reason = SynthFailureReason::TopologyMismatch;
    return out;
  }
  // Compute the lifted result types from the source-side ops so multi-
  // result body ops (e.g. dataflow.stream's idx + rwc) are emitted with
  // matching structural fabric.op shape.
  auto resultTypesOpt = liftedResultTypesForPeers(st.ctx, peers);
  if (!resultTypesOpt.has_value()) {
    out.reason = SynthFailureReason::TopologyMismatch;
    return out;
  }
  EmitOpOutcome emitted = emitBodyOpPositionMulti(st, *decision,
                                                  operandValues,
                                                  *resultTypesOpt);
  if (!emitted.ok) {
    out.reason = emitted.reason != SynthFailureReason::None
                     ? emitted.reason
                     : SynthFailureReason::TopologyMismatch;
    return out;
  }
  unsigned ri = peers.front().resultIndex;
  if (ri >= emitted.results.size()) {
    out.reason = SynthFailureReason::TopologyMismatch;
    return out;
  }
  EmittedOp slot;
  slot.results.assign(emitted.results.begin(), emitted.results.end());
  st.emittedOps[opKey] = slot;
  out.ok = true;
  out.value = emitted.results[ri];
  st.visited[key] = EmittedSlot{out.value};
  return out;
}

//===----------------------------------------------------------------------===//
// Wrapper builder.
//===----------------------------------------------------------------------===//

// Wrapper symbol name follows the spec: `fu_<sanitized(group)>` with
// sanitization replacing non-`[A-Za-z0-9_]` characters by `_`.
::std::string wrapperName(::llvm::StringRef groupName) {
  ::std::string out = "fu_";
  for (char c : groupName) {
    bool ok = (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') ||
              (c >= '0' && c <= '9') || c == '_';
    out.push_back(ok ? c : '_');
  }
  return out;
}

} // namespace

//===----------------------------------------------------------------------===//
// AnchorSynthesizer.
//===----------------------------------------------------------------------===//

AnchorSynthesizer::AnchorSynthesizer(const ::loom::SynthConfig &c) : cfg(c) {}

SynthResult AnchorSynthesizer::run(const SynthInputs &inputs) {
  SynthResult result;

  if (inputs.subgraphs.empty()) {
    result.failureReason = SynthFailureReason::InvalidInput;
    result.notes.push_back("anchor: no input subgraphs in synth group");
    return result;
  }
  if (!inputs.context) {
    result.failureReason = SynthFailureReason::InvalidInput;
    result.notes.push_back("anchor: missing scratch MLIRContext");
    return result;
  }

  ::mlir::MLIRContext *ctx = inputs.context;
  ::mlir::Location loc = ::mlir::UnknownLoc::get(ctx);

  // Yield-arity uniformity is the first hard precondition; per spec
  // anchor's idea is "lock-step BFS from yield anchors", so disagreeing
  // arities short-circuit to topology_mismatch.
  ::llvm::SmallVector<::llvm::SmallVector<Source, 4>, 4> perInputAnchors;
  perInputAnchors.reserve(inputs.subgraphs.size());
  unsigned arity = 0;
  for (auto [i, sg] : ::llvm::enumerate(inputs.subgraphs)) {
    auto anchors = yieldAnchors(sg);
    if (i == 0)
      arity = static_cast<unsigned>(anchors.size());
    else if (anchors.size() != arity) {
      result.failureReason = SynthFailureReason::TopologyMismatch;
      result.notes.push_back("anchor: input subgraphs disagree on yield arity");
      return result;
    }
    perInputAnchors.push_back(std::move(anchors));
  }
  if (arity == 0) {
    result.failureReason = SynthFailureReason::TopologyMismatch;
    result.notes.push_back("anchor: empty yield list (no anchor positions)");
    return result;
  }

  // Wrapper input ports: one per block-arg index, widths agreed across
  // all inputs. Disagreement is a topology mismatch (a block-arg
  // mapped to wider/narrower types in different inputs cannot be
  // realized as a single fabric.bits<N> port).
  auto portsOpt = collectWrapperInputSlots(inputs.subgraphs);
  if (!portsOpt.has_value()) {
    result.failureReason = SynthFailureReason::TopologyMismatch;
    result.notes.push_back(
        "anchor: input subgraphs disagree on block-arg shape or width");
    return result;
  }
  auto &ports = *portsOpt;

  // Build the wrapper fabric.module + inner fabric.pe + fabric.fu skeleton.
  // Two type lanes:
  //   * "inner" types: the actual lifted bit-widths per input slot and per
  //     yield slot. Used for FU body block-arg types (input lane) and FU
  //     body yield value types (output lane).
  //   * "outer" types: the uniform PE port width W = max(all lifted widths)
  //     applied to every PE input port, every PE result port, and every
  //     FU outer input/result type. Bridging to inner widths happens at
  //     the FU boundary via the input-side `to <inner-type>` truncation
  //     and the output-side `to <outer-type>` widening on fabric.yield.

  // Per-slot lifted (inner) input types.
  ::llvm::SmallVector<::mlir::Type, 4> innerInputTypes;
  innerInputTypes.reserve(ports.size());
  for (const WrapperInputSlot &p : ports)
    innerInputTypes.push_back(::fabric::BitsType::get(ctx, p.bitwidth));

  // Per-slot lifted (inner) output types come from first subgraph's
  // yield operand widths. Cross-input yield-width uniformity is
  // enforced lazily by peersUniformWidth() during BFS, so we trust the
  // first input's width as the wrapper signature here.
  ::dataflow::SubgraphOp first = inputs.subgraphs.front();
  ::llvm::SmallVector<::mlir::Type, 4> innerResultTypes;
  innerResultTypes.reserve(arity);
  ::mlir::Block &fb0 = first.getBody().front();
  ::mlir::Operation *yield0 = fb0.getTerminator();
  if (!yield0 || yield0->getNumOperands() != arity) {
    result.failureReason = SynthFailureReason::TopologyMismatch;
    result.notes.push_back("anchor: first input yield arity mismatch");
    return result;
  }
  for (unsigned i = 0; i < arity; ++i) {
    auto bw = tryBitWidthOf(yield0->getOperand(i).getType());
    if (!bw.has_value()) {
      result.failureReason = SynthFailureReason::TopologyMismatch;
      result.notes.push_back("anchor: yield operand has unsupported type");
      return result;
    }
    innerResultTypes.push_back(::fabric::BitsType::get(ctx, *bw));
  }

  // Pick uniform W = max over all lifted input AND output widths. The
  // PE-uniform-width invariant constrains the PE port-list types and
  // the FU outer port types to a single bits<W>. When all widths are
  // 0, W = 0 (legitimate for none-only signatures).
  unsigned uniformW = 0;
  for (const WrapperInputSlot &p : ports)
    uniformW = std::max(uniformW, p.bitwidth);
  for (::mlir::Type t : innerResultTypes) {
    if (auto bt = ::llvm::dyn_cast<::fabric::BitsType>(t))
      uniformW = std::max(uniformW, bt.getWidth());
  }
  ::mlir::Type uniformBits = ::fabric::BitsType::get(ctx, uniformW);

  // PE/FU outer (port-uniform) types.
  ::llvm::SmallVector<::mlir::Type, 4> wrapperInputTypes(ports.size(),
                                                          uniformBits);
  ::llvm::SmallVector<::mlir::Type, 4> wrapperResultTypes(arity, uniformBits);

  ::std::string symName = wrapperName(inputs.groupName);

  // 1. Build the outer fabric.module. The module declares the inputs as
  // entry-block arguments and carries no SSA results (the inner
  // fabric.pe owns the visible results). The module's body terminator
  // is a zero-operand fabric.yield.
  ::llvm::SmallVector<::mlir::Type, 0> moduleResultTypes;
  auto moduleFuncType = ::mlir::FunctionType::get(
      ctx, wrapperInputTypes, ::mlir::TypeRange(moduleResultTypes));
  ::mlir::OperationState moduleState(
      loc, ::fabric::ModuleOp::getOperationName());
  moduleState.addAttribute(
      "sym_name", ::mlir::StringAttr::get(ctx, symName));
  moduleState.addAttribute("function_type",
                           ::mlir::TypeAttr::get(moduleFuncType));
  ::mlir::Region *moduleRegion = moduleState.addRegion();
  ::mlir::Block *moduleEntry = new ::mlir::Block();
  moduleRegion->push_back(moduleEntry);
  ::llvm::SmallVector<::mlir::Location, 4> moduleArgLocs(
      wrapperInputTypes.size(), loc);
  moduleEntry->addArguments(wrapperInputTypes, moduleArgLocs);
  ::mlir::OpBuilder topBuilder(ctx);
  ::mlir::Operation *rawModule = topBuilder.create(moduleState);
  auto wrapper = ::mlir::cast<::fabric::ModuleOp>(rawModule);

  // 2. Build the inner fabric.pe. Spatial schedule, anonymous form: PE
  // operands are the module's entry-block arguments, PE results match
  // the FU's results (or, when the FU has no results, a single bits<W>
  // placeholder so PE's L>=1 invariant is satisfied).
  ::mlir::OpBuilder moduleBuilder(moduleEntry, moduleEntry->end());
  ::llvm::SmallVector<::mlir::Type, 4> peResultTypes(wrapperResultTypes);
  if (peResultTypes.empty()) {
    // FU has zero results. PE still needs L>=1; declare one bits<W>
    // output port that the FU does not drive. Width derived from the
    // module's input width (uniform W) when present, else 0 (which is
    // a verifier error for empty-input wrappers, but anchor already
    // requires K>=1 above).
    unsigned w = wrapperInputTypes.empty()
                     ? 0u
                     : ::llvm::cast<::fabric::BitsType>(wrapperInputTypes[0])
                           .getWidth();
    peResultTypes.push_back(::fabric::BitsType::get(ctx, w));
  }
  ::mlir::OperationState peState(loc, ::fabric::PeOp::getOperationName());
  peState.addOperands(::mlir::ValueRange(moduleEntry->getArguments()));
  peState.addTypes(peResultTypes);
  peState.addAttribute(
      "schedule",
      ::fabric::ScheduleAttr::get(ctx, ::fabric::Schedule::Spatial));
  ::mlir::Region *peRegion = peState.addRegion();
  ::mlir::Block *peEntry = new ::mlir::Block();
  peRegion->push_back(peEntry);
  ::llvm::SmallVector<::mlir::Location, 4> peArgLocs(wrapperInputTypes.size(),
                                                     loc);
  peEntry->addArguments(wrapperInputTypes, peArgLocs);
  ::mlir::Operation *rawPe = moduleBuilder.create(peState);
  auto pe = ::mlir::cast<::fabric::PeOp>(rawPe);
  (void)pe;

  // 3. Build the inner fabric.fu inside the PE body. FU operands are
  // the PE's block arguments (outer bits<W>); FU outer result types are
  // also bits<W>. FU body block-arg types are the per-slot inner widths
  // (input-side truncation when inner < outer is handled by the FU
  // boundary `to <inner-type>` clause).
  ::mlir::OpBuilder peBodyBuilder(peEntry, peEntry->end());
  ::mlir::OperationState fuState(loc, ::fabric::FuOp::getOperationName());
  fuState.addOperands(::mlir::ValueRange(peEntry->getArguments()));
  fuState.addTypes(wrapperResultTypes);
  ::mlir::Region *fuRegion = fuState.addRegion();
  ::mlir::Block *fuEntry = new ::mlir::Block();
  fuRegion->push_back(fuEntry);
  ::llvm::SmallVector<::mlir::Location, 4> fuArgLocs(innerInputTypes.size(),
                                                     loc);
  fuEntry->addArguments(innerInputTypes, fuArgLocs);
  ::mlir::Operation *rawFu = peBodyBuilder.create(fuState);
  auto fu = ::mlir::cast<::fabric::FuOp>(rawFu);

  // 4. Build the FU body using the BFS algorithm. Anchor strategy is
  // self-contained: every node it emits inside the body lives at the
  // top level (no nested control flow), so a single OpBuilder anchored
  // at the FU entry block suffices.
  ::mlir::OpBuilder bodyBuilder(fuEntry, fuEntry->end());

  AnchorState st(ctx, loc);
  st.sgs = inputs.subgraphs;
  st.bodyBuilder = &bodyBuilder;
  st.fu = fu;
  st.portValues.reserve(fuEntry->getNumArguments());
  for (auto a : fuEntry->getArguments())
    st.portValues.push_back(a);

  // 5. Walk yield anchors in order, materializing each peer set. The
  // BFS dedups DAG fanout via `visited`, so emitting in anchor order
  // does not duplicate inner ops shared across multiple yield outputs.
  ::llvm::SmallVector<::mlir::Value, 4> yieldValues;
  yieldValues.reserve(arity);
  for (unsigned k = 0; k < arity; ++k) {
    PeerVec peers;
    peers.reserve(inputs.subgraphs.size());
    for (auto &perInput : perInputAnchors)
      peers.push_back(perInput[k]);
    EmitOutcome out = materializePeers(st, peers, cfg);
    if (!out.ok) {
      // Drop the partially built wrapper deterministically. The
      // worker-local context goes out of scope in the caller.
      wrapper.erase();
      result.failureReason = out.reason != SynthFailureReason::None
                                 ? out.reason
                                 : SynthFailureReason::TopologyMismatch;
      for (auto &n : st.notes)
        result.notes.push_back(std::move(n));
      return result;
    }
    yieldValues.push_back(out.value);
  }

  // 6. Emit the FU's terminator fabric.yield (matches the FU result
  // arity, which may be zero) and the module-level fabric.yield (zero
  // operands; fabric.module declares no SSA results).
  //
  // For each yield value `k`, when the inner bit-width is narrower than
  // the FU outer width W, attach the per-value `to <outer>` clause via
  // the `declared_types` array attribute. This drives the FU output
  // boundary widening (low-bit-aligned, high bits zero-filled).
  ::mlir::OperationState fuYieldState(loc,
                                      ::fabric::YieldOp::getOperationName());
  fuYieldState.addOperands(yieldValues);
  if (!yieldValues.empty()) {
    ::llvm::SmallVector<::mlir::Attribute, 4> declaredAttrs;
    declaredAttrs.reserve(yieldValues.size());
    for (auto [k, v] : ::llvm::enumerate(yieldValues)) {
      // The declared destination is always the FU's outer result type
      // (uniform bits<W>). When inner == outer the verifier still
      // accepts the redundant annotation, but to keep the printed IR
      // clean we record the declared type only when it actually
      // differs from the inner SSA type.
      ::mlir::Type declared = wrapperResultTypes[k];
      declaredAttrs.push_back(::mlir::TypeAttr::get(declared));
    }
    bool anyWidens = false;
    for (auto [k, v] : ::llvm::enumerate(yieldValues)) {
      if (v.getType() != wrapperResultTypes[k]) {
        anyWidens = true;
        break;
      }
    }
    if (anyWidens)
      fuYieldState.addAttribute(
          "declared_types", ::mlir::ArrayAttr::get(ctx, declaredAttrs));
  }
  bodyBuilder.create(fuYieldState);

  ::mlir::OperationState moduleYieldState(
      loc, ::fabric::YieldOp::getOperationName());
  moduleBuilder.create(moduleYieldState);

  // 7. Run the MLIR verifier on the wrapper. Any verifier failure
  // demotes the result to verifier_failed (no IR is appended).
  if (::mlir::failed(::mlir::verify(wrapper))) {
    wrapper.erase();
    result.failureReason = SynthFailureReason::VerifierFailed;
    result.notes.push_back("anchor: synthesized FU failed MLIR verifier");
    return result;
  }

  // Wrap into an OwningOpRef so the worker-side thread retains
  // ownership until the main thread re-homes it.
  ::mlir::OwningOpRef<::fabric::ModuleOp> owned(wrapper);

  result.wrapper = std::move(owned);
  for (auto &n : st.notes)
    result.notes.push_back(std::move(n));
  return result;
}

//===----------------------------------------------------------------------===//
// Public helpers (callable from any TU that links MLIRFabricTechSynthesizer).
//===----------------------------------------------------------------------===//

::std::optional<WrapperPorts>
collectWrapperPorts(::llvm::ArrayRef<::dataflow::SubgraphOp> sgs,
                    ::mlir::MLIRContext *ctx) {
  if (!ctx)
    return ::std::nullopt;
  auto inputSlots = collectWrapperInputSlots(sgs);
  if (!inputSlots.has_value())
    return ::std::nullopt;
  auto outputWidths = collectWrapperOutputWidths(sgs);
  if (!outputWidths.has_value())
    return ::std::nullopt;
  WrapperPorts ports;
  ports.inputs.reserve(inputSlots->size());
  for (const WrapperInputSlot &p : *inputSlots)
    ports.inputs.push_back(::fabric::BitsType::get(ctx, p.bitwidth));
  ports.outputs.reserve(outputWidths->size());
  for (unsigned w : *outputWidths)
    ports.outputs.push_back(::fabric::BitsType::get(ctx, w));
  return ports;
}

} // namespace loom::fabric::tech
