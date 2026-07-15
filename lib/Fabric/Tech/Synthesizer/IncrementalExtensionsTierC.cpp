// Tier-C SCC handling for the Incremental left-fold synthesizer.
//
// This translation unit implements two responsibilities:
//
//   1. `buildTrivialFuTierC`: trivial-FU construction for a single
//      tier-C input whose body contains a graph-region back-edge or an
//      explicit state head (`dataflow.carry`, `dataflow.gate`, or
//      `dataflow.invariant`). The Anchor strategy refuses to walk
//      back-edges (lock-step BFS is tier-A only), so Incremental needs
//      its own one-shot mirror builder.
//
//   2. `structuralExtendCandidates`: the tier-C extension hook the main
//      Incremental loop invokes when the fold involves a back-edge or
//      state head in the new input or in the current FU. Per spec
//      section "SCC handling for tier C", the extension:
//        a. Computes flow signatures for every state head in the FU and
//           in `sg`.
//        b. Builds equivalence classes by signature (with transitive
//           closure for N > 2 inputs); fails with
//           `feedback_align_conflict` if any single input contributes
//           more than one head to a class.
//        c. Generates one candidate FU that grafts the new sg's SCC
//           body onto the FU, reusing state heads whose signatures
//           match an existing FU state head, merging carry feedback
//           inputs for matched carries, routing shared branch inputs
//           through outer `fabric.demux` ops, and inserting a
//           `fabric.mux` where the post-state value differs.
//
// Back-edge realization. Both builders use the same
// "build-then-resolve" placeholder scheme that mirrors
// `SubgraphEnumerator`'s graph-region materializer: when emission
// reaches a back-edge consumer before its producer is built, the
// consumer is wired to a fresh `unrealized_conversion_cast` placeholder
// of the back-edge's bit type. After the body walk completes, a
// post-pass walks every placeholder, looks the producer up in the
// per-build value map (or transitively through any chained
// placeholders), `replaceAllUsesWith` the real value, and erases the
// placeholder. Because the cleanup runs *before* `mlir::verify(wrapper)`
// the FU body never escapes this TU containing an
// `unrealized_conversion_cast`, so `FuOp::verify`'s "only fabric.op /
// fabric.mux / fabric.demux" rule is preserved.
//
// Spec source: `docs/spec-generalize-subgraphs-to-fu.md`, sections
// "SCC handling for tier C", "Tier C example (feedback alignment)",
// and "Failure reasons" (`feedback_align_conflict`).

#include "IncrementalExtensions.h"

#include "Common/IndexWidth.h"
#include "Fabric/IR/FabricOps.h"
#include "Fabric/IR/FabricTypes.h"
#include "Fabric/Tech/Synthesizer/Alignment.h"
#include "Fabric/Tech/Synthesizer/HwParams.h"

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
#include "mlir/IR/Region.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <algorithm>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace loom::fabric::tech::detail {

namespace {

//===----------------------------------------------------------------------===//
// Bit-width / type lifting (mirrors Anchor.cpp's bitWidthOf so this TU
// is self-contained).
//===----------------------------------------------------------------------===//

// Returns the lift-target bit width for `t`, or std::nullopt if `t`
// is not expressible as a fabric.bits<N> port. NoneType (e.g.
// dataflow.constant's ctrl input) lifts to a legitimate bits<0>.
::std::optional<unsigned> tryBitWidthOfType(::mlir::Type t) {
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

unsigned bitWidthOfType(::mlir::Type t) {
  auto v = tryBitWidthOfType(t);
  return v.has_value() ? *v : 0u;
}

::mlir::Type bitsOf(::mlir::MLIRContext *ctx, unsigned bw) {
  return ::fabric::BitsType::get(ctx, bw);
}

//===----------------------------------------------------------------------===//
// FU lookup helpers.
//===----------------------------------------------------------------------===//

::fabric::FuOp innerFu(::fabric::ModuleOp wrapper) {
  if (!wrapper)
    return {};
  ::fabric::FuOp found;
  wrapper.walk([&](::fabric::FuOp fu) {
    if (!found)
      found = fu;
  });
  return found;
}

//===----------------------------------------------------------------------===//
// State-head flow signature.
//===----------------------------------------------------------------------===//
//
// Per spec "SCC handling for tier C":
//
//   flow_signature(state_head) = (
//       op_name,
//       data_type,
//       upstream_stream_signature_or_none
//           // present iff cond is produced by a dataflow.stream:
//           //   (index_type, step_op, cont_cond)
//           // otherwise (e.g. cond from arith.cmpi or a block-arg):
//           //   (cond_source_kind, cond_source_op_name)
//   )
//
// We store the signature as a small string so it can be the key of an
// std::map (lexical iteration order keeps emission deterministic). The
// fields are joined by `|` separators that the spec's structural inputs
// never produce.

struct FlowSignature {
  ::std::string text;
  bool operator==(const FlowSignature &o) const { return text == o.text; }
  bool operator<(const FlowSignature &o) const { return text < o.text; }
};

// Encode an MLIR type as a stable, lift-equivalent bit-width string.
// Both software (i32, f32, index) and lifted hardware (fabric.bits<32>)
// types encode to the same canonical form so the flow signature can
// match across the FU/sg side boundary.
::std::string typeString(::mlir::Type t) {
  unsigned bw = bitWidthOfType(t);
  if (bw != 0) {
    ::std::string s = "bw";
    s += ::std::to_string(bw);
    return s;
  }
  if (auto bt = ::llvm::dyn_cast<::fabric::BitsType>(t))
    return "bw" + ::std::to_string(bt.getWidth());
  ::std::string s;
  ::llvm::raw_string_ostream os(s);
  t.print(os);
  os.flush();
  return s;
}

// Compute the cond-source part of a state-head flow signature. The stream
// upstream classification mirrors the carry rule from the spec and is also
// used for gate and invariant heads.
//
// `condValue` is the state head's cond operand.
::std::string condSignatureFromValue(::mlir::Value condValue) {
  ::std::string text;
  if (auto barg = ::llvm::dyn_cast<::mlir::BlockArgument>(condValue)) {
    text += "cond=blockarg";
    text += std::to_string(barg.getArgNumber());
    return text;
  }
  auto opRes = ::llvm::dyn_cast<::mlir::OpResult>(condValue);
  if (!opRes) {
    return "cond=unknown";
  }
  ::mlir::Operation *producer = opRes.getOwner();
  ::llvm::StringRef pname = producer->getName().getStringRef();
  // Match dataflow.stream (sg side) and fabric.op[@dataflow.stream] (FU
  // side) uniformly: both expose `step_op` and `cont_cond`. For
  // fabric.op[@dataflow.stream], the attributes live in
  // `hw_params[0].step_op[0]` / `cont_cond[0]` (per spec hw_params
  // policy). For dataflow.stream the attributes are top-level on the op.
  bool isStream = false;
  ::std::string stepOp, contCond, indexType;
  if (pname == "dataflow.stream") {
    isStream = true;
    if (auto so = producer->getAttrOfType<::mlir::StringAttr>("step_op"))
      stepOp = so.getValue().str();
    if (auto cc = producer->getAttrOfType<::mlir::StringAttr>("cont_cond"))
      contCond = cc.getValue().str();
    if (producer->getNumResults() > 0)
      indexType = typeString(producer->getResult(0).getType());
  } else if (pname == "fabric.op") {
    auto fop = ::mlir::cast<::fabric::OpOp>(producer);
    ::mlir::ArrayAttr opList = fop.getOpList();
    if (!opList.empty()) {
      if (auto sym = ::llvm::dyn_cast<::mlir::FlatSymbolRefAttr>(opList[0]))
        if (sym.getValue() == "dataflow.stream") {
          isStream = true;
          if (auto hp = fop.getHwParamsAttr())
            if (hp.size() == 1)
              if (auto dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(hp[0])) {
                if (auto so = dict.getAs<::mlir::ArrayAttr>("step_op"))
                  if (!so.empty())
                    if (auto sStr =
                            ::llvm::dyn_cast<::mlir::StringAttr>(so[0]))
                      stepOp = sStr.getValue().str();
                if (auto cc = dict.getAs<::mlir::ArrayAttr>("cont_cond"))
                  if (!cc.empty())
                    if (auto cStr =
                            ::llvm::dyn_cast<::mlir::StringAttr>(cc[0]))
                      contCond = cStr.getValue().str();
              }
          if (producer->getNumResults() > 0)
            indexType = typeString(producer->getResult(0).getType());
        }
    }
  }
  if (isStream) {
    text += "cond=stream(";
    text += "index=" + indexType;
    text += ",step_op=" + stepOp;
    text += ",cont_cond=" + contCond;
    text += ")";
  } else {
    text += "cond=op(";
    text += pname.str();
    text += ")";
  }
  return text;
}

enum class StateHeadKind : uint8_t { Carry, Gate, Invariant };

::std::optional<StateHeadKind> stateHeadKind(::llvm::StringRef name) {
  if (name == "dataflow.carry")
    return StateHeadKind::Carry;
  if (name == "dataflow.gate")
    return StateHeadKind::Gate;
  if (name == "dataflow.invariant")
    return StateHeadKind::Invariant;
  return std::nullopt;
}

::std::string stateHeadKindText(StateHeadKind kind) {
  switch (kind) {
  case StateHeadKind::Carry:
    return "dataflow.carry";
  case StateHeadKind::Gate:
    return "dataflow.gate";
  case StateHeadKind::Invariant:
    return "dataflow.invariant";
  }
  return "";
}

struct SgStateHead {
  ::mlir::Operation *op = nullptr;
  StateHeadKind kind = StateHeadKind::Carry;
  FlowSignature sig;
};

struct FuStateHead {
  ::fabric::OpOp op;
  StateHeadKind kind = StateHeadKind::Carry;
  FlowSignature sig;
};

FlowSignature buildStateHeadSignature(StateHeadKind kind,
                                      ::mlir::Value condValue,
                                      ::mlir::Type dataType) {
  FlowSignature sig;
  sig.text.reserve(96);
  sig.text += "op=";
  sig.text += stateHeadKindText(kind);
  sig.text += "|type=";
  sig.text += typeString(dataType);
  sig.text += "|";
  sig.text += condSignatureFromValue(condValue);
  return sig;
}

::std::optional<SgStateHead> describeSgStateHead(::mlir::Operation *op) {
  if (!op)
    return std::nullopt;
  auto kind = stateHeadKind(op->getName().getStringRef());
  if (!kind.has_value() || op->getNumOperands() < 2)
    return std::nullopt;
  SgStateHead head;
  head.op = op;
  head.kind = *kind;
  head.sig = buildStateHeadSignature(*kind, op->getOperand(0),
                                     op->getOperand(1).getType());
  return head;
}

::std::optional<FuStateHead> describeFuStateHead(::fabric::OpOp op) {
  if (!op)
    return std::nullopt;
  ::mlir::ArrayAttr opList = op.getOpList();
  if (opList.empty())
    return std::nullopt;
  auto sym = ::llvm::dyn_cast<::mlir::FlatSymbolRefAttr>(opList[0]);
  if (!sym)
    return std::nullopt;
  auto kind = stateHeadKind(sym.getValue());
  if (!kind.has_value() || op.getInputs().size() < 2)
    return std::nullopt;
  FuStateHead head;
  head.op = op;
  head.kind = *kind;
  head.sig = buildStateHeadSignature(*kind, op.getInputs()[0],
                                     op.getInputs()[1].getType());
  return head;
}

::llvm::SmallVector<SgStateHead, 4>
collectSgStateHeads(::dataflow::SubgraphOp sg) {
  ::llvm::SmallVector<SgStateHead, 4> out;
  if (!sg)
    return out;
  for (::mlir::Operation &raw : sg.getBody().front().without_terminator()) {
    auto head = describeSgStateHead(&raw);
    if (head.has_value())
      out.push_back(*head);
  }
  return out;
}

::llvm::SmallVector<FuStateHead, 4>
collectFuStateHeads(::fabric::FuOp fu) {
  ::llvm::SmallVector<FuStateHead, 4> out;
  if (!fu)
    return out;
  for (::mlir::Operation &raw : fu.getBody().front().without_terminator()) {
    auto op = ::mlir::dyn_cast<::fabric::OpOp>(raw);
    if (!op)
      continue;
    auto head = describeFuStateHead(op);
    if (head.has_value())
      out.push_back(*head);
  }
  return out;
}

//===----------------------------------------------------------------------===//
// Pre-align SCCs (signature heuristic).
//===----------------------------------------------------------------------===//

// PreAlignment is the externally visible result: per FU state head and
// per sg state head, the "class id" identifies which heads are equivalent
// (and therefore must merge in the synthesized FU). Class ids are dense
// integers assigned in lexical signature order so they are stable across
// runs.
struct PreAlignment {
  // For each state-bearing fabric.op in the FU body (in body order),
  // the class id.
  ::llvm::SmallVector<unsigned, 4> fuHeadClass;
  // For each state-bearing op in `sg`'s body (in body order), the class id.
  ::llvm::SmallVector<unsigned, 4> sgHeadClass;
  // Class id -> the FU state-head index that anchors this class.
  ::llvm::SmallVector<::std::optional<unsigned>, 4> classFuAnchor;
  // Class id -> textual signature (only useful for diagnostics).
  ::llvm::SmallVector<FlowSignature, 4> classSignatures;
  bool conflict = false;
  bool hasSharedClass = false;
};

// Returns std::nullopt on `feedback_align_conflict` (at least one input
// has more than one head in the same class).
//
// The default path requires at least one shared state-head class when both
// sides contain state. With cfg.sccFullUnroll enabled, incompatible
// signatures are allowed to proceed to the conservative mirror builder; it
// keeps the new state's slots separate and relies on coverage verification to
// prove the resulting FU still covers all folded inputs.
::std::optional<PreAlignment>
preAlignSccs(::fabric::FuOp fu, ::dataflow::SubgraphOp sg,
             const ::loom::SynthConfig &cfg) {
  PreAlignment pa;

  ::llvm::SmallVector<FuStateHead, 4> fuHeads = collectFuStateHeads(fu);
  ::llvm::SmallVector<SgStateHead, 4> sgHeads = collectSgStateHeads(sg);

  // Class assignment by lexical signature order. std::map keeps
  // iteration deterministic.
  ::std::map<FlowSignature, unsigned> sigToClass;
  ::std::vector<FlowSignature> orderedSignatures;

  auto getOrAssignClass = [&](const FlowSignature &s) -> unsigned {
    auto it = sigToClass.find(s);
    if (it != sigToClass.end())
      return it->second;
    unsigned cid = static_cast<unsigned>(orderedSignatures.size());
    sigToClass[s] = cid;
    orderedSignatures.push_back(s);
    return cid;
  };

  // Per-input head counts for the conflict check.
  ::std::map<unsigned, unsigned> fuHeadsInClass;
  ::std::map<unsigned, unsigned> sgHeadsInClass;

  pa.fuHeadClass.reserve(fuHeads.size());
  for (const FuStateHead &head : fuHeads) {
    unsigned cid = getOrAssignClass(head.sig);
    pa.fuHeadClass.push_back(cid);
    ++fuHeadsInClass[cid];
  }
  pa.sgHeadClass.reserve(sgHeads.size());
  for (const SgStateHead &head : sgHeads) {
    unsigned cid = getOrAssignClass(head.sig);
    pa.sgHeadClass.push_back(cid);
    ++sgHeadsInClass[cid];
  }

  // Conflict: any single side contributes >1 head to one class. For
  // N>2 inputs the same rule applies after transitive closure -- since
  // signatures use string equality, "transitive closure" reduces to
  // "same exact signature", so the per-side check is sufficient.
  for (auto [cid, n] : fuHeadsInClass)
    if (n > 1)
      return std::nullopt;
  for (auto [cid, n] : sgHeadsInClass)
    if (n > 1)
      return std::nullopt;

  pa.classSignatures = ::llvm::SmallVector<FlowSignature, 4>(
      orderedSignatures.begin(), orderedSignatures.end());
  pa.classFuAnchor.assign(pa.classSignatures.size(), std::nullopt);
  for (auto [i, cid] : ::llvm::enumerate(pa.fuHeadClass))
    pa.classFuAnchor[cid] = static_cast<unsigned>(i);
  ::llvm::DenseSet<unsigned> sgClasses(pa.sgHeadClass.begin(),
                                       pa.sgHeadClass.end());
  for (unsigned cid : pa.fuHeadClass)
    if (sgClasses.count(cid)) {
      pa.hasSharedClass = true;
      break;
    }
  if (!cfg.sccFullUnroll && !fuHeads.empty() && !sgHeads.empty() &&
      !pa.hasSharedClass)
    pa.conflict = true;
  return pa;
}

//===----------------------------------------------------------------------===//
// Tier-C-aware FU body builder.
//===----------------------------------------------------------------------===//
//
// The builder walks a sg body in textual order, mirroring each op as
// `fabric.op[@<op_name>]`. Operands are looked up in a value map; missing
// entries (back-edge consumers reached before producers) are filled with
// an `unrealized_conversion_cast` placeholder of the right bit type.
// After the walk, a post-pass rewrites every placeholder use to the real
// value (now in the map) and erases the placeholder.

// Mapping context shared across helpers in this builder.
struct BodyBuildCtx {
  BodyBuildCtx(::mlir::MLIRContext *c, ::mlir::Location l,
               ::mlir::OpBuilder *b)
      : ctx(c), loc(l), builder(b) {}
  ::mlir::MLIRContext *ctx = nullptr;
  ::mlir::Location loc;
  ::mlir::OpBuilder *builder = nullptr;
  // Map from the source-side Value (block arg or body op result) to the
  // FU-side Value (fabric.bits<N> from the entry block or a fabric.op
  // result).
  ::llvm::DenseMap<::mlir::Value, ::mlir::Value> valueMap;
  // Placeholder ops created during the walk (held so the post-pass can
  // erase them after replaceAllUsesWith).
  ::llvm::SmallVector<::mlir::Operation *, 4> placeholderOps;
  // For each placeholder op, the source-side Value it is standing in
  // for. The post-pass uses this to look up the real value in
  // `valueMap` (which by then contains every body op's result).
  ::llvm::DenseMap<::mlir::Operation *, ::mlir::Value> placeholderSource;
  // fabric.op instances emitted while mirroring the source body. Tier-C
  // extension uses this list to distinguish the new private arm from the
  // existing aggregate.
  ::llvm::SmallVector<::mlir::Operation *, 8> emittedOps;
};

// Returns the FU-side value for `srcValue`. Block-args are mapped 1:1
// from the sg's block-arg to the FU entry-block-arg via `valueMap`;
// missing entries (back-edge producer not yet built) yield a fresh
// placeholder.
::mlir::Value lookupOrPlaceholder(BodyBuildCtx &c, ::mlir::Value srcValue) {
  auto it = c.valueMap.find(srcValue);
  if (it != c.valueMap.end())
    return it->second;
  auto bwOpt = tryBitWidthOfType(srcValue.getType());
  if (!bwOpt.has_value())
    return {};
  unsigned bw = *bwOpt;
  ::mlir::Type bits = bitsOf(c.ctx, bw);
  ::mlir::OperationState ph(
      c.loc, ::mlir::UnrealizedConversionCastOp::getOperationName());
  ph.addTypes({bits});
  ::mlir::Operation *raw = c.builder->create(ph);
  c.placeholderOps.push_back(raw);
  c.placeholderSource[raw] = srcValue;
  ::mlir::Value v = raw->getResult(0);
  c.valueMap[srcValue] = v;
  return v;
}

// Build the hw_params attribute appropriate for `opName` from a single
// source-side op. Delegates to the shared `buildHwParamsUnion` helper so
// the trivial-FU path agrees with Anchor on the per-op-kind axes
// (predicate, step_op, cont_cond, const_hex_value, bitmask).
::mlir::ArrayAttr hwParamsFor(::mlir::MLIRContext *ctx,
                              ::llvm::StringRef opName,
                              ::mlir::Operation *srcOp) {
  ::mlir::Operation *peers[1] = {srcOp};
  return buildHwParamsUnion(ctx, opName,
                            ::llvm::ArrayRef<::mlir::Operation *>(peers, 1));
}

::mlir::ArrayAttr opListSingleton(::mlir::MLIRContext *ctx,
                                  ::llvm::StringRef name) {
  ::llvm::SmallVector<::mlir::Attribute, 1> v{
      ::mlir::FlatSymbolRefAttr::get(ctx, name)};
  return ::mlir::ArrayAttr::get(ctx, v);
}

::fabric::OpOp emitFabricOpInBody(BodyBuildCtx &c,
                                  ::llvm::StringRef opName,
                                  ::mlir::ValueRange inputs,
                                  ::mlir::TypeRange resultTypes,
                                  ::mlir::ArrayAttr hwParams) {
  ::mlir::OperationState st(c.loc, ::fabric::OpOp::getOperationName());
  st.addOperands(inputs);
  st.addTypes(resultTypes);
  st.addAttribute("op_list", opListSingleton(c.ctx, opName));
  if (hwParams)
    st.addAttribute("hw_params", hwParams);
  ::mlir::Operation *raw = c.builder->create(st);
  return ::mlir::cast<::fabric::OpOp>(raw);
}

// Resolve every placeholder to a real value. Returns false if a
// placeholder cannot be resolved (which would indicate a builder bug).
bool resolvePlaceholders(BodyBuildCtx &c) {
  for (::mlir::Operation *ph : c.placeholderOps) {
    auto it = c.placeholderSource.find(ph);
    if (it == c.placeholderSource.end())
      return false;
    ::mlir::Value src = it->second;
    auto vit = c.valueMap.find(src);
    if (vit == c.valueMap.end())
      return false;
    ::mlir::Value real = vit->second;
    // Defensive: if `real` is itself a placeholder result (a back-edge
    // chain), look through it.
    while (real.getDefiningOp() &&
           ::mlir::isa<::mlir::UnrealizedConversionCastOp>(
               real.getDefiningOp())) {
      auto rit = c.placeholderSource.find(real.getDefiningOp());
      if (rit == c.placeholderSource.end())
        break;
      auto rvIt = c.valueMap.find(rit->second);
      if (rvIt == c.valueMap.end())
        break;
      ::mlir::Value next = rvIt->second;
      if (next == real)
        break;
      real = next;
    }
    if (real == ph->getResult(0))
      return false;
    ph->getResult(0).replaceAllUsesWith(real);
  }
  for (::mlir::Operation *ph : c.placeholderOps)
    ph->erase();
  c.placeholderOps.clear();
  c.placeholderSource.clear();
  return true;
}

// Returns the lifted (fabric.bits<N>) type list for `srcOp`'s results.
// Any unsupported type returns an empty SmallVector. NoneType lifts to
// a legitimate bits<0>.
::llvm::SmallVector<::mlir::Type, 2>
liftedResultTypes(::mlir::MLIRContext *ctx, ::mlir::Operation *srcOp) {
  ::llvm::SmallVector<::mlir::Type, 2> out;
  out.reserve(srcOp->getNumResults());
  for (::mlir::Value r : srcOp->getResults()) {
    auto bw = tryBitWidthOfType(r.getType());
    if (!bw.has_value())
      return {};
    out.push_back(bitsOf(ctx, *bw));
  }
  return out;
}

// Mirror one sg body op into the FU body. Returns false on failure
// (unsupported op, unsupported operand type, etc.).
bool mirrorBodyOp(BodyBuildCtx &c, ::mlir::Operation *srcOp) {
  ::llvm::StringRef opName = srcOp->getName().getStringRef();
  if (!::fabric::isFabricOpSupported(opName))
    return false;
  ::llvm::SmallVector<::mlir::Value, 4> inputs;
  inputs.reserve(srcOp->getNumOperands());
  for (::mlir::Value v : srcOp->getOperands()) {
    ::mlir::Value mapped = lookupOrPlaceholder(c, v);
    if (!mapped)
      return false;
    inputs.push_back(mapped);
  }
  ::llvm::SmallVector<::mlir::Type, 2> resultTypes =
      liftedResultTypes(c.ctx, srcOp);
  if (resultTypes.size() != srcOp->getNumResults())
    return false;
  ::mlir::ArrayAttr hwParams = hwParamsFor(c.ctx, opName, srcOp);
  ::fabric::OpOp emitted =
      emitFabricOpInBody(c, opName, inputs, resultTypes, hwParams);
  if (!emitted)
    return false;
  c.emittedOps.push_back(emitted.getOperation());
  for (auto [i, r] : ::llvm::enumerate(srcOp->getResults()))
    c.valueMap[r] = emitted->getResult(i);
  return true;
}

//===----------------------------------------------------------------------===//
// Wrapper construction helpers shared between the trivial and the
// extension builders.
//===----------------------------------------------------------------------===//

struct WrapperShell {
  ::mlir::OwningOpRef<::fabric::ModuleOp> wrapper;
  ::fabric::FuOp fu;
  ::mlir::Block *fuEntry = nullptr;
};

// Build a wrapper `fabric.module` containing one anonymous `fabric.pe`
// containing one empty `fabric.fu` skeleton sized for `inputBitWidths`
// and `resultBitWidths`. The caller fills in the FU body.
//
// Layout invariants (see test/fabric/unit/fu/valid.mlir for shape):
//   * fabric.module declares zero SSA results; its body terminator is
//     a zero-operand fabric.yield.
//   * fabric.pe is anonymous-form, spatial-scheduled; its result count
//     mirrors the FU result count when non-empty, otherwise a single
//     bits<W> placeholder is added so PE's L>=1 invariant is satisfied.
//   * The FU's outer operand types match the PE block-arg types; they
//     all share the same bits<W> with the module's input types.
WrapperShell buildShell(::mlir::MLIRContext *ctx, ::llvm::StringRef symName,
                        ::llvm::ArrayRef<unsigned> inputBitWidths,
                        ::llvm::ArrayRef<unsigned> resultBitWidths) {
  WrapperShell ws;
  ::mlir::Location loc = ::mlir::UnknownLoc::get(ctx);
  ::llvm::SmallVector<::mlir::Type, 4> inTypes;
  inTypes.reserve(inputBitWidths.size());
  for (unsigned bw : inputBitWidths)
    inTypes.push_back(bitsOf(ctx, bw));
  ::llvm::SmallVector<::mlir::Type, 4> fuOutTypes;
  fuOutTypes.reserve(resultBitWidths.size());
  for (unsigned bw : resultBitWidths)
    fuOutTypes.push_back(bitsOf(ctx, bw));

  // 1. Build the fabric.module top-level op (zero SSA results).
  ::llvm::SmallVector<::mlir::Type, 0> moduleResultTypes;
  auto moduleFuncType = ::mlir::FunctionType::get(
      ctx, inTypes, ::mlir::TypeRange(moduleResultTypes));
  ::mlir::OperationState moduleState(
      loc, ::fabric::ModuleOp::getOperationName());
  moduleState.addAttribute(
      "sym_name", ::mlir::StringAttr::get(ctx, symName));
  moduleState.addAttribute("function_type",
                           ::mlir::TypeAttr::get(moduleFuncType));
  ::mlir::Region *moduleRegion = moduleState.addRegion();
  ::mlir::Block *moduleEntry = new ::mlir::Block();
  moduleRegion->push_back(moduleEntry);
  ::llvm::SmallVector<::mlir::Location, 4> moduleArgLocs(inTypes.size(), loc);
  moduleEntry->addArguments(inTypes, moduleArgLocs);
  ::mlir::OpBuilder topBuilder(ctx);
  ::mlir::Operation *rawModule = topBuilder.create(moduleState);
  auto wrapper = ::mlir::cast<::fabric::ModuleOp>(rawModule);

  ::mlir::OpBuilder moduleBuilder(moduleEntry, moduleEntry->end());

  // 2. Build the inner fabric.pe (spatial, anonymous form).
  ::llvm::SmallVector<::mlir::Type, 4> peResultTypes(fuOutTypes);
  if (peResultTypes.empty()) {
    unsigned w = inTypes.empty()
                     ? 0u
                     : ::llvm::cast<::fabric::BitsType>(inTypes[0]).getWidth();
    peResultTypes.push_back(bitsOf(ctx, w));
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
  ::llvm::SmallVector<::mlir::Location, 4> peArgLocs(inTypes.size(), loc);
  peEntry->addArguments(inTypes, peArgLocs);
  moduleBuilder.create(peState);

  // 3. Build the inner fabric.fu inside the PE body.
  ::mlir::OpBuilder peBodyBuilder(peEntry, peEntry->end());
  ::mlir::OperationState fuState(loc, ::fabric::FuOp::getOperationName());
  fuState.addOperands(::mlir::ValueRange(peEntry->getArguments()));
  fuState.addTypes(fuOutTypes);
  ::mlir::Region *fuRegion = fuState.addRegion();
  ::mlir::Block *fuEntry = new ::mlir::Block();
  fuRegion->push_back(fuEntry);
  ::llvm::SmallVector<::mlir::Location, 4> fuArgLocs(inTypes.size(), loc);
  fuEntry->addArguments(inTypes, fuArgLocs);
  ::mlir::Operation *rawFu = peBodyBuilder.create(fuState);
  auto fu = ::mlir::cast<::fabric::FuOp>(rawFu);

  // 4. Module-level fabric.yield (zero operands).
  ::mlir::OperationState modYield(loc,
                                  ::fabric::YieldOp::getOperationName());
  moduleBuilder.create(modYield);

  ws.wrapper = ::mlir::OwningOpRef<::fabric::ModuleOp>(wrapper);
  ws.fu = fu;
  ws.fuEntry = fuEntry;
  return ws;
}

// Sanitize a group name into a wrapper symbol per spec
// (`@fu_<sanitized(group)>`).
::std::string sanitize(::llvm::StringRef name) {
  ::std::string out = "fu_";
  for (char c : name) {
    bool ok = (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') ||
              (c >= '0' && c <= '9') || c == '_';
    out.push_back(ok ? c : '_');
  }
  return out;
}

} // namespace

//===----------------------------------------------------------------------===//
// Public tier-C entry points.
//===----------------------------------------------------------------------===//

::mlir::OwningOpRef<::fabric::ModuleOp>
buildTrivialFuTierC(::mlir::MLIRContext *ctx, ::llvm::StringRef groupName,
                    ::dataflow::SubgraphOp first) {
  if (!ctx || !first)
    return {};

  // Collect the wrapper input bit-widths (per sg block-arg) and the
  // wrapper result bit-widths (per yield operand). NoneType (e.g.
  // dataflow.constant's ctrl block-arg) lifts to a legitimate bits<0>.
  ::mlir::Block &sgBody = first.getBody().front();
  ::llvm::SmallVector<unsigned, 4> inputBws;
  inputBws.reserve(sgBody.getNumArguments());
  for (::mlir::BlockArgument a : sgBody.getArguments()) {
    auto bw = tryBitWidthOfType(a.getType());
    if (!bw.has_value())
      return {};
    inputBws.push_back(*bw);
  }
  ::mlir::Operation *yieldTerm = sgBody.getTerminator();
  if (!yieldTerm)
    return {};
  ::llvm::SmallVector<unsigned, 4> resultBws;
  resultBws.reserve(yieldTerm->getNumOperands());
  for (::mlir::Value v : yieldTerm->getOperands()) {
    auto bw = tryBitWidthOfType(v.getType());
    if (!bw.has_value())
      return {};
    resultBws.push_back(*bw);
  }
  WrapperShell ws =
      buildShell(ctx, sanitize(groupName), inputBws, resultBws);

  // Mirror sg's body op-by-op.
  ::mlir::OpBuilder bodyBuilder(ws.fuEntry, ws.fuEntry->end());
  BodyBuildCtx bc(ctx, ::mlir::UnknownLoc::get(ctx), &bodyBuilder);
  for (auto [i, a] : ::llvm::enumerate(sgBody.getArguments()))
    bc.valueMap[a] = ws.fuEntry->getArgument(i);

  for (::mlir::Operation &raw : sgBody.without_terminator()) {
    if (!mirrorBodyOp(bc, &raw))
      return {};
  }

  if (!resolvePlaceholders(bc))
    return {};

  // Emit fabric.yield from the mapped yield operands.
  ::llvm::SmallVector<::mlir::Value, 4> yieldVals;
  yieldVals.reserve(yieldTerm->getNumOperands());
  for (::mlir::Value v : yieldTerm->getOperands()) {
    auto it = bc.valueMap.find(v);
    if (it == bc.valueMap.end())
      return {};
    yieldVals.push_back(it->second);
  }
  ::mlir::OperationState yieldState(bc.loc,
                                    ::fabric::YieldOp::getOperationName());
  yieldState.addOperands(yieldVals);
  bodyBuilder.create(yieldState);

  return std::move(ws.wrapper);
}

namespace {

::fabric::MuxOp emitMuxN(::mlir::OpBuilder &b, ::mlir::Location loc,
                         ::mlir::ValueRange arms, ::mlir::Type bits) {
  ::mlir::OperationState st(loc, ::fabric::MuxOp::getOperationName());
  st.addOperands(arms);
  st.addTypes({bits});
  ::mlir::Operation *raw = b.create(st);
  return ::mlir::cast<::fabric::MuxOp>(raw);
}

::fabric::DemuxOp emitDemux2(::mlir::OpBuilder &builder, ::mlir::Location loc,
                             ::mlir::Value input, ::mlir::Type bits) {
  ::mlir::OperationState state(loc, ::fabric::DemuxOp::getOperationName());
  state.addOperands({input});
  state.addTypes({bits, bits});
  return ::mlir::cast<::fabric::DemuxOp>(builder.create(state));
}

bool routePrivateBranchInputs(
    BodyBuildCtx &bc,
    const ::llvm::DenseSet<::mlir::Operation *> &anchoredFuOps,
    const ::llvm::DenseSet<::mlir::OpOperand *> &branchSpecificUses) {
  ::llvm::DenseSet<::mlir::Operation *> newOps(bc.emittedOps.begin(),
                                               bc.emittedOps.end());
  ::llvm::DenseSet<::mlir::Value> seenSources;
  ::llvm::SmallVector<::mlir::Value, 8> sources;
  for (::mlir::Operation *op : bc.emittedOps) {
    for (::mlir::Value operand : op->getOperands()) {
      if (newOps.contains(operand.getDefiningOp()))
        continue;
      if (seenSources.insert(operand).second)
        sources.push_back(operand);
    }
  }

  for (::mlir::Value source : sources) {
    ::llvm::SmallVector<::mlir::OpOperand *, 8> oldBranchUses;
    ::llvm::SmallVector<::mlir::OpOperand *, 8> newBranchUses;
    bool hasExistingUse = false;
    for (::mlir::OpOperand &use : source.getUses()) {
      ::mlir::Operation *owner = use.getOwner();
      if (newOps.contains(owner)) {
        newBranchUses.push_back(&use);
        continue;
      }
      hasExistingUse = true;
      if ((anchoredFuOps.contains(owner) ||
           ::mlir::isa<::fabric::YieldOp>(owner)) &&
          !branchSpecificUses.contains(&use))
        continue;
      oldBranchUses.push_back(&use);
    }
    if (!hasExistingUse || newBranchUses.empty())
      continue;

    auto bits = ::llvm::dyn_cast<::fabric::BitsType>(source.getType());
    if (!bits)
      return false;
    ::fabric::DemuxOp demux = emitDemux2(*bc.builder, bc.loc, source, bits);
    for (::mlir::OpOperand *use : oldBranchUses)
      use->set(demux.getOutputs()[0]);
    for (::mlir::OpOperand *use : newBranchUses)
      use->set(demux.getOutputs()[1]);
  }
  return true;
}

bool containsValue(::llvm::ArrayRef<::mlir::Value> values, ::mlir::Value v) {
  return llvm::any_of(values, [&](::mlir::Value cur) { return cur == v; });
}

::llvm::SmallVector<::mlir::Value, 4> muxArms(::mlir::Value oldValue) {
  ::llvm::SmallVector<::mlir::Value, 4> arms;
  if (auto mux =
          ::mlir::dyn_cast_or_null<::fabric::MuxOp>(oldValue.getDefiningOp())) {
    for (::mlir::Value in : mux.getInputs())
      arms.push_back(in);
    return arms;
  }
  arms.push_back(oldValue);
  return arms;
}

::mlir::Value mergeWithMux(::mlir::OpBuilder &builder, ::mlir::Location loc,
                           ::mlir::Value oldValue, ::mlir::Value newValue) {
  if (!oldValue || !newValue)
    return {};
  if (oldValue == newValue)
    return oldValue;
  ::llvm::SmallVector<::mlir::Value, 4> arms = muxArms(oldValue);
  if (containsValue(arms, newValue))
    return oldValue;
  arms.push_back(newValue);
  auto bits = ::llvm::dyn_cast<::fabric::BitsType>(oldValue.getType());
  if (!bits || oldValue.getType() != newValue.getType())
    return {};
  return emitMuxN(builder, loc, arms, oldValue.getType()).getOutput();
}

void eraseUnusedMuxProducer(::mlir::Value v) {
  auto mux = ::mlir::dyn_cast_or_null<::fabric::MuxOp>(v.getDefiningOp());
  if (!mux || !mux->use_empty())
    return;
  mux->erase();
}

::mlir::Operation *fuYield(::fabric::FuOp fu) {
  if (!fu)
    return nullptr;
  return fu.getBody().front().getTerminator();
}

::llvm::StringRef firstFabricSymbol(::fabric::OpOp op) {
  if (!op)
    return {};
  ::mlir::ArrayAttr opList = op.getOpList();
  if (opList.empty())
    return {};
  auto sym = ::llvm::dyn_cast<::mlir::FlatSymbolRefAttr>(opList[0]);
  if (!sym)
    return {};
  return sym.getValue();
}

bool mapOpResults(BodyBuildCtx &bc, ::mlir::Operation *sgOp,
                  ::fabric::OpOp fuOp) {
  if (!sgOp || !fuOp)
    return false;
  if (sgOp->getNumResults() != fuOp.getOutputs().size())
    return false;
  for (auto [i, r] : ::llvm::enumerate(sgOp->getResults()))
    bc.valueMap[r] = fuOp.getOutputs()[i];
  return true;
}

void addMatchedCondProducer(
    ::mlir::Operation *sgHead, ::fabric::OpOp fuHead,
    ::llvm::DenseMap<::mlir::Operation *, ::fabric::OpOp> &anchoredOps) {
  if (!sgHead || !fuHead || sgHead->getNumOperands() < 1 ||
      fuHead.getInputs().empty())
    return;
  auto sgCond = ::llvm::dyn_cast<::mlir::OpResult>(sgHead->getOperand(0));
  auto fuCond = ::llvm::dyn_cast<::mlir::OpResult>(fuHead.getInputs()[0]);
  if (!sgCond || !fuCond)
    return;
  ::mlir::Operation *sgProducer = sgCond.getOwner();
  auto fuProducer = ::mlir::dyn_cast<::fabric::OpOp>(fuCond.getOwner());
  if (!fuProducer)
    return;
  if (sgProducer->getName().getStringRef() != firstFabricSymbol(fuProducer))
    return;
  anchoredOps[sgProducer] = fuProducer;
}

bool isAnchoredOp(::mlir::Operation *op,
                  const ::llvm::DenseMap<::mlir::Operation *, ::fabric::OpOp>
                      &anchoredOps) {
  return anchoredOps.find(op) != anchoredOps.end();
}

::mlir::OwningOpRef<::fabric::ModuleOp>
buildMirroredTierCCandidate(::fabric::ModuleOp curWrapper,
                            ::dataflow::SubgraphOp sg,
                            const PreAlignment &pa) {
  ::mlir::Operation *clonedRaw = curWrapper->clone();
  auto newWrapper = ::mlir::OwningOpRef<::fabric::ModuleOp>(
      ::mlir::cast<::fabric::ModuleOp>(clonedRaw));
  ::fabric::FuOp newFu = innerFu(newWrapper.get());
  if (!newFu)
    return {};

  ::llvm::SmallVector<FuStateHead, 4> newFuHeads =
      collectFuStateHeads(newFu);
  ::llvm::SmallVector<SgStateHead, 4> sgHeads = collectSgStateHeads(sg);
  ::llvm::DenseMap<::mlir::Operation *, ::fabric::OpOp> matchedHeads;
  ::llvm::DenseMap<::mlir::Operation *, ::fabric::OpOp> anchoredOps;
  for (auto [i, head] : ::llvm::enumerate(sgHeads)) {
    if (i >= pa.sgHeadClass.size())
      return {};
    unsigned cid = pa.sgHeadClass[i];
    if (cid >= pa.classFuAnchor.size())
      return {};
    if (!pa.classFuAnchor[cid].has_value())
      continue;
    unsigned fuIdx = *pa.classFuAnchor[cid];
    if (fuIdx >= newFuHeads.size())
      return {};
    ::fabric::OpOp fuHead = newFuHeads[fuIdx].op;
    matchedHeads[head.op] = fuHead;
    anchoredOps[head.op] = fuHead;
    addMatchedCondProducer(head.op, fuHead, anchoredOps);
  }
  ::llvm::DenseSet<::mlir::Operation *> anchoredFuOps;
  for (auto &kv : anchoredOps)
    anchoredFuOps.insert(kv.second.getOperation());

  ::mlir::Operation *yield = fuYield(newFu);
  if (!yield)
    return {};
  ::mlir::Location loc = ::mlir::UnknownLoc::get(curWrapper.getContext());
  ::mlir::OpBuilder builder(newFu->getContext());
  builder.setInsertionPoint(yield);
  BodyBuildCtx bc(curWrapper.getContext(), loc, &builder);

  ::mlir::Block &sgBody = sg.getBody().front();
  if (sgBody.getNumArguments() != newFu.getBody().front().getNumArguments())
    return {};
  for (auto [i, a] : ::llvm::enumerate(sgBody.getArguments()))
    bc.valueMap[a] = newFu.getBody().front().getArgument(i);

  for (auto &kv : anchoredOps) {
    if (!mapOpResults(bc, kv.first, kv.second))
      return {};
  }

  for (::mlir::Operation &raw : sgBody.without_terminator()) {
    if (isAnchoredOp(&raw, anchoredOps))
      continue;
    if (!mirrorBodyOp(bc, &raw))
      return {};
  }

  if (!resolvePlaceholders(bc))
    return {};

  ::llvm::DenseSet<::mlir::OpOperand *> branchSpecificUses;
  for (const SgStateHead &head : sgHeads) {
    if (head.kind != StateHeadKind::Carry)
      continue;
    auto it = matchedHeads.find(head.op);
    if (it == matchedHeads.end())
      continue;
    ::fabric::OpOp fuCarry = it->second;
    if (head.op->getNumOperands() < 3 || fuCarry.getInputs().size() < 3)
      return {};
    auto mappedBackedge = bc.valueMap.find(head.op->getOperand(2));
    if (mappedBackedge == bc.valueMap.end())
      return {};
    if (fuCarry.getInputs()[2] != mappedBackedge->second)
      branchSpecificUses.insert(&fuCarry->getOpOperand(2));
  }

  ::mlir::Operation *sgYield = sgBody.getTerminator();
  if (!sgYield || sgYield->getNumOperands() != yield->getNumOperands())
    return {};
  for (unsigned i = 0, e = sgYield->getNumOperands(); i < e; ++i) {
    auto mapped = bc.valueMap.find(sgYield->getOperand(i));
    if (mapped == bc.valueMap.end())
      return {};
    if (yield->getOperand(i) != mapped->second)
      branchSpecificUses.insert(&yield->getOpOperand(i));
  }

  builder.setInsertionPoint(yield);
  if (!routePrivateBranchInputs(bc, anchoredFuOps, branchSpecificUses))
    return {};

  builder.setInsertionPoint(yield);
  for (const SgStateHead &head : sgHeads) {
    if (head.kind != StateHeadKind::Carry)
      continue;
    auto it = matchedHeads.find(head.op);
    if (it == matchedHeads.end())
      continue;
    ::fabric::OpOp fuCarry = it->second;
    if (head.op->getNumOperands() < 3 || fuCarry.getInputs().size() < 3)
      return {};
    auto mappedBackedge = bc.valueMap.find(head.op->getOperand(2));
    if (mappedBackedge == bc.valueMap.end())
      return {};
    ::mlir::Value oldFeedback = fuCarry.getInputs()[2];
    ::mlir::Value merged =
        mergeWithMux(builder, loc, oldFeedback, mappedBackedge->second);
    if (!merged)
      return {};
    fuCarry->setOperand(2, merged);
    eraseUnusedMuxProducer(oldFeedback);
  }

  builder.setInsertionPoint(yield);
  for (unsigned i = 0, e = sgYield->getNumOperands(); i < e; ++i) {
    auto mapped = bc.valueMap.find(sgYield->getOperand(i));
    if (mapped == bc.valueMap.end())
      return {};
    ::mlir::Value oldValue = yield->getOperand(i);
    ::mlir::Value newValue = mapped->second;
    if (oldValue == newValue)
      continue;
    ::mlir::Value merged = mergeWithMux(builder, loc, oldValue, newValue);
    if (!merged)
      return {};
    yield->setOperand(i, merged);
    eraseUnusedMuxProducer(oldValue);
  }

  return newWrapper;
}

} // namespace

::llvm::SmallVector<::mlir::OwningOpRef<::fabric::ModuleOp>, 4>
structuralExtendCandidates(::fabric::ModuleOp curWrapper,
                           ::dataflow::SubgraphOp sg,
                           const ::loom::SynthConfig &cfg) {
  ::llvm::SmallVector<::mlir::OwningOpRef<::fabric::ModuleOp>, 4> out;
  if (!curWrapper || !sg)
    return out;
  ::fabric::FuOp fu = innerFu(curWrapper);
  if (!fu)
    return out;

  ::std::optional<PreAlignment> paOpt = preAlignSccs(fu, sg, cfg);
  if (!paOpt.has_value())
    return out;
  PreAlignment &pa = *paOpt;
  if (pa.conflict)
    return out;

  auto cand = buildMirroredTierCCandidate(curWrapper, sg, pa);
  if (cand)
    out.push_back(std::move(cand));
  return out;
}

//===----------------------------------------------------------------------===//
// Tier-C reason hook for the Incremental main loop.
//===----------------------------------------------------------------------===//
//
// When `structuralExtendCandidates` cannot produce a candidate, we want
// the main loop to know whether the failure was a flow-signature
// conflict (`feedback_align_conflict`) or a generic topology mismatch.
// Exposed via `tierCLastFailureReason` so Incremental.cpp can pick the
// right `loom.synth_failed` value without taking a dependency on the
// internal PreAlignment data.

::std::optional<SynthFailureReason>
classifyTierCConflict(::fabric::ModuleOp curWrapper,
                      ::dataflow::SubgraphOp sg,
                      const ::loom::SynthConfig &cfg) {
  if (!curWrapper || !sg)
    return std::nullopt;
  ::fabric::FuOp fu = innerFu(curWrapper);
  if (!fu)
    return std::nullopt;
  ::std::optional<PreAlignment> paOpt = preAlignSccs(fu, sg, cfg);
  // Per-side cardinality conflict (>1 head per class on one side) is
  // the strict reading of the spec's pseudocode.
  if (!paOpt.has_value())
    return SynthFailureReason::FeedbackAlignConflict;
  PreAlignment &pa = *paOpt;
  if (pa.conflict)
    return SynthFailureReason::FeedbackAlignConflict;
  return std::nullopt;
}

} // namespace loom::fabric::tech::detail
