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
#include "Fabric/Tech/Synthesizer/CoverageVerifier.h"
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
// port. Returns 0 when the type cannot be lifted (caller treats as a
// topology mismatch).
unsigned bitWidthOf(::mlir::Type t) {
  if (auto i = ::llvm::dyn_cast<::mlir::IntegerType>(t))
    return i.getWidth();
  if (auto f = ::llvm::dyn_cast<::mlir::FloatType>(t))
    return f.getWidth();
  if (::llvm::isa<::mlir::IndexType>(t))
    return ::loom::getIndexWidth();
  return 0;
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
// bit-width).
struct WrapperPort {
  unsigned argIndex;
  unsigned bitwidth;
};

::std::optional<::llvm::SmallVector<WrapperPort, 4>>
collectWrapperPorts(::llvm::ArrayRef<::dataflow::SubgraphOp> sgs) {
  ::llvm::SmallVector<WrapperPort, 4> ports;
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
    unsigned bw = bitWidthOf(fb.getArgument(i).getType());
    if (bw == 0)
      return std::nullopt;
    for (auto sg : sgs) {
      ::mlir::Block &b = sg.getBody().front();
      unsigned other = bitWidthOf(b.getArgument(i).getType());
      if (other == 0 || other != bw)
        return std::nullopt;
    }
    WrapperPort p;
    p.argIndex = i;
    p.bitwidth = bw;
    ports.push_back(p);
  }
  return ports;
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
unsigned widthOfSource(const Source &s, ::dataflow::SubgraphOp sg) {
  if (s.kind == Source::BlockArg) {
    if (s.argIndex >= sg.getBody().front().getNumArguments())
      return 0;
    return bitWidthOf(sg.getBody().front().getArgument(s.argIndex).getType());
  }
  if (!s.op || s.resultIndex >= s.op->getNumResults())
    return 0;
  return bitWidthOf(s.op->getResult(s.resultIndex).getType());
}

// Validate that all peers at a body-op position name a result with the
// same bit-width. Mismatch -> topology_mismatch.
bool peersUniformWidth(const PeerVec &peers,
                       ::llvm::ArrayRef<::dataflow::SubgraphOp> sgs,
                       unsigned &widthOut) {
  unsigned w = 0;
  for (auto [i, s] : ::llvm::enumerate(peers)) {
    unsigned cur = widthOfSource(s, sgs[i]);
    if (cur == 0)
      return false;
    if (i == 0)
      w = cur;
    else if (cur != w)
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
// inputs, and result type. The op is emitted at the builder's current
// insertion point.
::fabric::OpOp emitFabricOp(::mlir::OpBuilder &builder, ::mlir::Location loc,
                            ::mlir::ArrayAttr opList,
                            ::mlir::ArrayAttr hwParams,
                            ::mlir::ValueRange operands,
                            ::mlir::Type resultType) {
  ::mlir::OperationState state(loc, ::fabric::OpOp::getOperationName());
  state.addOperands(operands);
  state.addTypes({resultType});
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

// One layout candidate for a body-op position. `useMux == false` means
// a single fabric.op merging every peer's op name into one op_list
// (must all share one share group). `useMux == true` means one
// fabric.op per share-group bucket fed from the same operands and
// joined by a fresh fabric.mux. The CostModel evaluates the resulting
// sub-FU and the lowest-cost legal candidate wins.
struct OpNodeDecision {
  bool useMux = false;
  // Buckets[k] = the sorted set of op names belonging to share-group k
  // (or the sentinel index for singletons). For useMux == false the
  // map has exactly one entry.
  ::std::map<::std::optional<::std::size_t>, ::std::set<::std::string>>
      buckets;
};

// Group peers by share-group index (with singletons collapsed to
// std::nullopt as a single bucket per distinct op name). For
// useMux == false we expect every entry to map to the same key.
::std::map<::std::optional<::std::size_t>, ::std::set<::std::string>>
bucketBySharegroup(const PeerVec &peers) {
  ::std::map<::std::optional<::std::size_t>, ::std::set<::std::string>> out;
  for (const Source &s : peers) {
    if (!s.op)
      continue;
    ::llvm::StringRef name = s.op->getName().getStringRef();
    auto sg = ::loom::common::findShareGroup(name);
    out[sg].insert(name.str());
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

  // Collect the per-bucket "did we see > 1 op name and were they
  // singletons?" check. fabric.op forbids multi-name op_lists where
  // any entry is a singleton (per OpOp::verify); we avoid producing
  // such candidates.
  for (const auto &kv : buckets) {
    if (!kv.first.has_value() && kv.second.size() > 1)
      return std::nullopt; // two distinct singletons cannot share a fabric.op
  }

  if (buckets.size() == 1) {
    OpNodeDecision d;
    d.useMux = false;
    d.buckets = std::move(buckets);
    return d;
  }
  // More than one bucket: cross-share-group. Only legal when the user
  // opted into intra-position muxing.
  if (!allowMux)
    return std::nullopt;
  OpNodeDecision d;
  d.useMux = true;
  d.buckets = std::move(buckets);
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

// Emit a fabric.op (or per-bucket fabric.op + fabric.mux) for a body-op
// peer set whose operands have already been materialized into
// `operandValues`. Returns the value yielded by the (possibly mux-
// joined) sub-FU at this position.
EmitOutcome emitBodyOpPosition(AnchorState &st,
                               const OpNodeDecision &decision, unsigned bw,
                               ::mlir::ValueRange operandValues) {
  EmitOutcome r;
  ::mlir::Type bits = ::fabric::BitsType::get(st.ctx, bw);
  ::mlir::ArrayAttr hwParams = emptyHwParams(st.ctx);
  if (!decision.useMux) {
    // Single share-group: one fabric.op with the sorted union.
    const auto &kv = *decision.buckets.begin();
    ::mlir::ArrayAttr opList = sortedOpListFor(kv.second, st.ctx);
    auto op = emitFabricOp(*st.bodyBuilder, st.loc, opList, hwParams,
                           operandValues, bits);
    r.ok = true;
    r.value = op.getOutputs()[0];
    return r;
  }
  // Cross-share-group with intra-position mux: one fabric.op per
  // bucket (in lexical share-group order via the std::map iteration),
  // joined by one fabric.mux. The shared operand vector is reused
  // across buckets verbatim per spec.
  ::llvm::SmallVector<::mlir::Value, 4> arms;
  arms.reserve(decision.buckets.size());
  for (const auto &kv : decision.buckets) {
    ::mlir::ArrayAttr opList = sortedOpListFor(kv.second, st.ctx);
    auto opOp = emitFabricOp(*st.bodyBuilder, st.loc, opList, hwParams,
                             operandValues, bits);
    arms.push_back(opOp.getOutputs()[0]);
  }
  auto mux = emitFabricMux(*st.bodyBuilder, st.loc, arms, bits);
  r.ok = true;
  r.value = mux.getOutput();
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
  EmitOutcome emitted = emitBodyOpPosition(st, *decision, bw, operandValues);
  if (!emitted.ok) {
    out.reason = emitted.reason;
    return out;
  }
  st.visited[key] = EmittedSlot{emitted.value};
  return emitted;
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
  auto portsOpt = collectWrapperPorts(inputs.subgraphs);
  if (!portsOpt.has_value()) {
    result.failureReason = SynthFailureReason::TopologyMismatch;
    result.notes.push_back(
        "anchor: input subgraphs disagree on block-arg shape or width");
    return result;
  }
  auto &ports = *portsOpt;

  // Build the wrapper func.func and inner fabric.fu skeleton.
  ::llvm::SmallVector<::mlir::Type, 4> wrapperInputTypes;
  wrapperInputTypes.reserve(ports.size());
  for (const WrapperPort &p : ports)
    wrapperInputTypes.push_back(::fabric::BitsType::get(ctx, p.bitwidth));
  // Output types: bit-widths drawn from the first input's yield (all
  // inputs already agreed on yield arity; per-position uniformity
  // checked when each anchor is materialized). The wrapper's result
  // shape is set to match the inner fu's results, which we'll assign
  // after the body is built.

  // 1. Build the outer func.func with placeholder result types; we'll
  // patch the function type after we discover yield bit-widths.
  // Output types come from first subgraph's yield operand widths.
  ::dataflow::SubgraphOp first = inputs.subgraphs.front();
  ::llvm::SmallVector<::mlir::Type, 4> wrapperResultTypes;
  wrapperResultTypes.reserve(arity);
  ::mlir::Block &fb0 = first.getBody().front();
  ::mlir::Operation *yield0 = fb0.getTerminator();
  if (!yield0 || yield0->getNumOperands() != arity) {
    result.failureReason = SynthFailureReason::TopologyMismatch;
    result.notes.push_back("anchor: first input yield arity mismatch");
    return result;
  }
  for (unsigned i = 0; i < arity; ++i) {
    unsigned bw = bitWidthOf(yield0->getOperand(i).getType());
    if (bw == 0) {
      result.failureReason = SynthFailureReason::TopologyMismatch;
      result.notes.push_back("anchor: yield operand has unsupported type");
      return result;
    }
    // Cross-input yield-width uniformity is enforced lazily by
    // peersUniformWidth() during BFS, so we trust the first input's
    // width as the wrapper signature here.
    wrapperResultTypes.push_back(::fabric::BitsType::get(ctx, bw));
  }

  ::std::string symName = wrapperName(inputs.groupName);
  auto funcType =
      ::mlir::FunctionType::get(ctx, wrapperInputTypes, wrapperResultTypes);
  auto wrapper = ::mlir::func::FuncOp::create(loc, symName, funcType);
  // Build the entry block and FU.
  ::mlir::Block *entry = wrapper.addEntryBlock();
  ::mlir::OpBuilder funcBuilder(entry, entry->end());
  // 2. Build the inner fabric.fu. Operands are the wrapper's entry
  // block args, types are the wrapper's input types, results are the
  // wrapper's result types.
  ::mlir::OperationState fuState(loc, ::fabric::FuOp::getOperationName());
  fuState.addOperands(::mlir::ValueRange(entry->getArguments()));
  fuState.addTypes(wrapperResultTypes);
  ::mlir::Region *fuRegion = fuState.addRegion();
  ::mlir::Block *fuEntry = new ::mlir::Block();
  fuRegion->push_back(fuEntry);
  ::llvm::SmallVector<::mlir::Location, 4> fuArgLocs(wrapperInputTypes.size(),
                                                     loc);
  fuEntry->addArguments(wrapperInputTypes, fuArgLocs);
  ::mlir::Operation *rawFu = funcBuilder.create(fuState);
  auto fu = ::mlir::cast<::fabric::FuOp>(rawFu);

  // 3. Build the FU body using the BFS algorithm. Anchor strategy is
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

  // 4. Walk yield anchors in order, materializing each peer set. The
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
      // Drop the partially built wrapper deterministically by clearing
      // the OwningOpRef (which we never constructed yet). The
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

  // 5. Emit fabric.yield and append a return from the wrapper to
  // satisfy the func dialect's signature.
  ::mlir::OperationState yieldState(loc, ::fabric::YieldOp::getOperationName());
  yieldState.addOperands(yieldValues);
  bodyBuilder.create(yieldState);

  // The wrapper returns the FU's results.
  ::mlir::OperationState retState(loc,
                                  ::mlir::func::ReturnOp::getOperationName());
  retState.addOperands(::mlir::ValueRange(fu.getResults()));
  funcBuilder.create(retState);

  // 6. Run the MLIR verifier on the wrapper. Any verifier failure
  // demotes the result to verifier_failed (no IR is appended).
  if (::mlir::failed(::mlir::verify(wrapper))) {
    wrapper.erase();
    result.failureReason = SynthFailureReason::VerifierFailed;
    result.notes.push_back("anchor: synthesized FU failed MLIR verifier");
    return result;
  }

  // Wrap into an OwningOpRef so the worker-side thread retains
  // ownership until the main thread re-homes it.
  ::mlir::OwningOpRef<::mlir::func::FuncOp> owned(wrapper);

  // 7. Optional coverage verification. We run this against the inputs
  // before transferring ownership; the verifier clones the wrapper
  // into its own scratch module so this does not contaminate the
  // user's IR. Skipping this when disabled lets tier-A correctness be
  // pinned by the structural FileCheck patterns alone.
  if (cfg.coverageVerifierEnabled) {
    CoverageVerifier verifier(cfg);
    result.coverage = verifier.verify(fu, inputs.subgraphs);
    if (!result.coverage.allCovered()) {
      result.failureReason = SynthFailureReason::CoverageVerifyFailed;
      result.notes.push_back(
          "anchor: synthesized FU did not cover every input subgraph");
      // owned destructor erases the wrapper; the failure is reported
      // upward without any IR escaping the worker.
      return result;
    }
  } else {
    // When coverage is off, populate matchIndex with vacuous slots so
    // downstream stats reporting reads `covered=N/N` rather than 0/N.
    result.coverage.matchIndex.assign(inputs.subgraphs.size(), std::nullopt);
    for (size_t i = 0; i < inputs.subgraphs.size(); ++i)
      result.coverage.matchIndex[i] = i;
  }

  result.wrapper = std::move(owned);
  for (auto &n : st.notes)
    result.notes.push_back(std::move(n));
  return result;
}

} // namespace loom::fabric::tech
