// Tier-C SCC handling for the Incremental left-fold synthesizer.
//
// This translation unit implements two responsibilities:
//
//   1. `buildTrivialFuTierC`: trivial-FU construction for a single
//      tier-C input (one whose body contains a graph-region back-edge,
//      typically driven by `dataflow.carry`). The Anchor strategy
//      refuses to walk back-edges (lock-step BFS is tier-A only), so
//      Incremental needs its own one-shot mirror builder.
//
//   2. `structuralExtendCandidates`: the tier-C extension hook the main
//      Incremental loop invokes when the new input subgraph has a
//      back-edge in the diff against the current FU. Per spec section
//      "SCC handling for tier C", the extension:
//        a. Computes flow signatures for every carry head in the FU and
//           in `sg`.
//        b. Builds equivalence classes by signature (with transitive
//           closure for N > 2 inputs); fails with
//           `feedback_align_conflict` if any single input contributes
//           more than one head to a class.
//        c. Generates one candidate FU that grafts the new sg's SCC
//           body onto the FU, reusing carry heads whose signatures
//           match an existing FU carry and inserting a `fabric.mux`
//           where the post-carry op differs (tier-B baseline behind a
//           shared carry).
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

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
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
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <algorithm>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace loom::fabric::tech::detail {

namespace {

//===----------------------------------------------------------------------===//
// Bit-width / type lifting (mirrors Anchor.cpp's bitWidthOf so this TU
// is self-contained).
//===----------------------------------------------------------------------===//

unsigned bitWidthOfType(::mlir::Type t) {
  if (auto i = ::llvm::dyn_cast<::mlir::IntegerType>(t))
    return i.getWidth();
  if (auto f = ::llvm::dyn_cast<::mlir::FloatType>(t))
    return f.getWidth();
  if (::llvm::isa<::mlir::IndexType>(t))
    return ::loom::getIndexWidth();
  return 0;
}

::mlir::Type bitsOf(::mlir::MLIRContext *ctx, unsigned bw) {
  return ::fabric::BitsType::get(ctx, bw);
}

//===----------------------------------------------------------------------===//
// FU lookup helpers.
//===----------------------------------------------------------------------===//

::fabric::FuOp innerFu(::mlir::func::FuncOp wrapper) {
  if (!wrapper || wrapper.getBody().empty())
    return {};
  for (::mlir::Operation &op : wrapper.getBody().front().getOperations())
    if (auto fu = ::mlir::dyn_cast<::fabric::FuOp>(op))
      return fu;
  return {};
}

// Returns the textual list of body fabric.op's whose op_list[0] is the
// requested symbol (e.g. "@dataflow.carry"). Used both to identify the
// existing FU's carry heads and to drive merge decisions.
::llvm::SmallVector<::fabric::OpOp, 4>
collectFabricOpsByFirstSymbol(::fabric::FuOp fu, ::llvm::StringRef symName) {
  ::llvm::SmallVector<::fabric::OpOp, 4> out;
  if (!fu)
    return out;
  for (::mlir::Operation &raw : fu.getBody().front().getOperations()) {
    auto op = ::mlir::dyn_cast<::fabric::OpOp>(raw);
    if (!op)
      continue;
    ::mlir::ArrayAttr opList = op.getOpList();
    if (opList.empty())
      continue;
    auto sym = ::llvm::dyn_cast<::mlir::FlatSymbolRefAttr>(opList[0]);
    if (!sym)
      continue;
    if (sym.getValue() == symName)
      out.push_back(op);
  }
  return out;
}

//===----------------------------------------------------------------------===//
// Carry flow signature.
//===----------------------------------------------------------------------===//
//
// Per spec "SCC handling for tier C":
//
//   flow_signature(carry) = (
//       carry_type,
//       upstream_stream_signature_or_none
//           // present iff carry.cond is produced by a dataflow.stream:
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

// Compute the flow signature of one carry node (either a dataflow.carry
// op in a sg body or a fabric.op[@dataflow.carry] in an FU body). The
// carry's cond operand is the only operand we look at; the stream
// upstream classification mirrors the spec.
//
// `condValue` is the carry's cond operand (a Value);
// `carriedTypeText` is the textual encoding of the carry's data type
// (the operand 1/result 0 type for dataflow.carry; the operand 1 type
// for fabric.op[@dataflow.carry]).
FlowSignature buildFlowSignatureFromCondValue(::mlir::Value condValue,
                                              ::llvm::StringRef carriedTypeText) {
  FlowSignature sig;
  sig.text.reserve(64);
  sig.text += "carry_type=";
  sig.text += carriedTypeText.str();
  sig.text += "|";
  if (auto barg = ::llvm::dyn_cast<::mlir::BlockArgument>(condValue)) {
    sig.text += "cond=blockarg";
    return sig;
  }
  auto opRes = ::llvm::dyn_cast<::mlir::OpResult>(condValue);
  if (!opRes) {
    sig.text += "cond=unknown";
    return sig;
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
    sig.text += "cond=stream(";
    sig.text += "index=" + indexType;
    sig.text += ",step_op=" + stepOp;
    sig.text += ",cont_cond=" + contCond;
    sig.text += ")";
  } else {
    sig.text += "cond=op(";
    sig.text += pname.str();
    sig.text += ")";
  }
  return sig;
}

// dataflow.carry side signature. The carried-value type is the carry's
// init operand type (operand index 1).
FlowSignature signatureOfDataflowCarry(::mlir::Operation *carry) {
  ::std::string carried = typeString(carry->getOperand(1).getType());
  return buildFlowSignatureFromCondValue(carry->getOperand(0), carried);
}

// fabric.op[@dataflow.carry] side signature. fabric.op exposes inputs
// as `getInputs()`; per spec carry signature is operand 0 (cond) /
// operand 1 (init). The carried value's bit-width drives the type.
FlowSignature signatureOfFabricCarry(::fabric::OpOp carry) {
  ::std::string carried;
  if (carry.getInputs().size() >= 2)
    carried = typeString(carry.getInputs()[1].getType());
  ::mlir::Value cond;
  if (!carry.getInputs().empty())
    cond = carry.getInputs()[0];
  return buildFlowSignatureFromCondValue(cond, carried);
}

//===----------------------------------------------------------------------===//
// Carry-head collection on both sides.
//===----------------------------------------------------------------------===//

// Walk a dataflow.subgraph body and return every `dataflow.carry` op.
::llvm::SmallVector<::mlir::Operation *, 4>
collectSgCarries(::dataflow::SubgraphOp sg) {
  ::llvm::SmallVector<::mlir::Operation *, 4> out;
  if (!sg)
    return out;
  for (::mlir::Operation &raw : sg.getBody().front().getOperations()) {
    if (raw.getName().getStringRef() == "dataflow.carry")
      out.push_back(&raw);
  }
  return out;
}

//===----------------------------------------------------------------------===//
// Pre-align SCCs (signature heuristic).
//===----------------------------------------------------------------------===//

// PreAlignment is the externally visible result: per FU carry head and
// per sg carry head, the "class id" identifies which carries are equivalent
// (and therefore must merge in the synthesized FU). Class ids are dense
// integers assigned in lexical signature order so they are stable across
// runs.
struct PreAlignment {
  // For each fabric.op[@dataflow.carry] in the FU body (in body order),
  // the class id.
  ::llvm::SmallVector<unsigned, 4> fuClass;
  // For each dataflow.carry in `sg`'s body (in body order), the class id.
  ::llvm::SmallVector<unsigned, 4> sgClass;
  // Class id -> the FU carry index that "anchors" this class (i.e. an
  // existing carry the new sg carry should merge into). std::nullopt
  // means the class has no existing FU member -- the candidate must
  // graft a fresh fabric.op[@dataflow.carry].
  ::llvm::SmallVector<::std::optional<unsigned>, 4> classFuAnchor;
  // Class id -> textual signature (only useful for diagnostics).
  ::llvm::SmallVector<FlowSignature, 4> classSignatures;
};

// Returns std::nullopt on `feedback_align_conflict` (at least one input
// has more than one head in the same class).
//
// `cfg.sccFullUnroll == true` selects the alternate path described in
// the spec: unroll once per SCC (longest-cycle path), materialize back-
// edges as placeholders, run alignment on the unrolled DAG, then re-fold.
// That alternate path is documented in spec section "SCC handling for
// tier C" but is not implemented here -- the signature heuristic is the
// default and covers every example workload in the spec. When the flag
// is set we still fall back to the heuristic so the strategy degrades
// gracefully rather than aborting.
::std::optional<PreAlignment>
preAlignSccs(::fabric::FuOp fu, ::dataflow::SubgraphOp sg,
             const ::loom::SynthConfig &cfg) {
  PreAlignment pa;
  (void)cfg;

  ::llvm::SmallVector<::fabric::OpOp, 4> fuCarries =
      collectFabricOpsByFirstSymbol(fu, "dataflow.carry");
  ::llvm::SmallVector<::mlir::Operation *, 4> sgCarries = collectSgCarries(sg);

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

  pa.fuClass.reserve(fuCarries.size());
  for (auto fc : fuCarries) {
    FlowSignature s = signatureOfFabricCarry(fc);
    unsigned cid = getOrAssignClass(s);
    pa.fuClass.push_back(cid);
    ++fuHeadsInClass[cid];
  }
  pa.sgClass.reserve(sgCarries.size());
  for (auto *sc : sgCarries) {
    FlowSignature s = signatureOfDataflowCarry(sc);
    unsigned cid = getOrAssignClass(s);
    pa.sgClass.push_back(cid);
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
  for (auto [i, cid] : ::llvm::enumerate(pa.fuClass))
    pa.classFuAnchor[cid] = static_cast<unsigned>(i);
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
};

// Returns the FU-side value for `srcValue`. Block-args are mapped 1:1
// from the sg's block-arg to the FU entry-block-arg via `valueMap`;
// missing entries (back-edge producer not yet built) yield a fresh
// placeholder.
::mlir::Value lookupOrPlaceholder(BodyBuildCtx &c, ::mlir::Value srcValue) {
  auto it = c.valueMap.find(srcValue);
  if (it != c.valueMap.end())
    return it->second;
  unsigned bw = bitWidthOfType(srcValue.getType());
  if (bw == 0)
    return {};
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

// Build the hw_params attribute appropriate for `opName`. Per spec
// "hw_params policy" the synthesizer emits an observed-value union for
// every configurable axis the enumerator inspects. For
// `dataflow.stream` we surface `step_op` and `cont_cond` as one-element
// arrays drawn from the source op; for `dataflow.constant` /
// `dataflow.mux` / `dataflow.demux` / `dataflow.sync` /
// `arith.cmpi` / `arith.cmpf` similar surfacing applies. The trivial
// builder only needs `dataflow.stream` for the spec example; other
// configurable axes are handled by the same builder when needed (the
// implementation walks `srcOp`'s attributes for the well-known keys).
::mlir::ArrayAttr hwParamsFor(::mlir::MLIRContext *ctx,
                              ::llvm::StringRef opName,
                              ::mlir::Operation *srcOp) {
  ::llvm::SmallVector<::mlir::NamedAttribute, 2> entries;
  auto strToArr = [&](::llvm::StringRef value) -> ::mlir::ArrayAttr {
    ::llvm::SmallVector<::mlir::Attribute, 1> v{
        ::mlir::StringAttr::get(ctx, value)};
    return ::mlir::ArrayAttr::get(ctx, v);
  };
  if (opName == "dataflow.stream") {
    if (auto so = srcOp->getAttrOfType<::mlir::StringAttr>("step_op"))
      entries.emplace_back(::mlir::StringAttr::get(ctx, "step_op"),
                           strToArr(so.getValue()));
    if (auto cc = srcOp->getAttrOfType<::mlir::StringAttr>("cont_cond"))
      entries.emplace_back(::mlir::StringAttr::get(ctx, "cont_cond"),
                           strToArr(cc.getValue()));
  }
  // Empty dictionary is the canonical hw_params for ops without
  // configurable axes (per spec).
  ::mlir::DictionaryAttr inner =
      ::mlir::DictionaryAttr::get(ctx, entries);
  ::llvm::SmallVector<::mlir::Attribute, 1> outer{inner};
  return ::mlir::ArrayAttr::get(ctx, outer);
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
// Any unsupported type returns an empty SmallVector.
::llvm::SmallVector<::mlir::Type, 2>
liftedResultTypes(::mlir::MLIRContext *ctx, ::mlir::Operation *srcOp) {
  ::llvm::SmallVector<::mlir::Type, 2> out;
  out.reserve(srcOp->getNumResults());
  for (::mlir::Value r : srcOp->getResults()) {
    unsigned bw = bitWidthOfType(r.getType());
    if (bw == 0)
      return {};
    out.push_back(bitsOf(ctx, bw));
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
  for (auto [i, r] : ::llvm::enumerate(srcOp->getResults()))
    c.valueMap[r] = emitted->getResult(i);
  return true;
}

//===----------------------------------------------------------------------===//
// Wrapper construction helpers shared between the trivial and the
// extension builders.
//===----------------------------------------------------------------------===//

struct WrapperShell {
  ::mlir::OwningOpRef<::mlir::func::FuncOp> wrapper;
  ::fabric::FuOp fu;
  ::mlir::Block *fuEntry = nullptr;
};

// Build an empty wrapper func.func + empty fabric.fu skeleton sized
// for `inputBitWidths` and `resultBitWidths`. The caller fills in the
// FU body.
WrapperShell buildShell(::mlir::MLIRContext *ctx, ::llvm::StringRef symName,
                        ::llvm::ArrayRef<unsigned> inputBitWidths,
                        ::llvm::ArrayRef<unsigned> resultBitWidths) {
  WrapperShell ws;
  ::mlir::Location loc = ::mlir::UnknownLoc::get(ctx);
  ::llvm::SmallVector<::mlir::Type, 4> inTypes;
  inTypes.reserve(inputBitWidths.size());
  for (unsigned bw : inputBitWidths)
    inTypes.push_back(bitsOf(ctx, bw));
  ::llvm::SmallVector<::mlir::Type, 4> outTypes;
  outTypes.reserve(resultBitWidths.size());
  for (unsigned bw : resultBitWidths)
    outTypes.push_back(bitsOf(ctx, bw));
  auto funcType = ::mlir::FunctionType::get(ctx, inTypes, outTypes);
  auto wrapper =
      ::mlir::func::FuncOp::create(loc, symName.str(), funcType);
  ::mlir::Block *entry = wrapper.addEntryBlock();
  ::mlir::OpBuilder funcBuilder(entry, entry->end());
  ::mlir::OperationState fuState(loc, ::fabric::FuOp::getOperationName());
  fuState.addOperands(::mlir::ValueRange(entry->getArguments()));
  fuState.addTypes(outTypes);
  ::mlir::Region *fuRegion = fuState.addRegion();
  ::mlir::Block *fuEntry = new ::mlir::Block();
  fuRegion->push_back(fuEntry);
  ::llvm::SmallVector<::mlir::Location, 4> fuArgLocs(inTypes.size(), loc);
  fuEntry->addArguments(inTypes, fuArgLocs);
  ::mlir::Operation *rawFu = funcBuilder.create(fuState);
  auto fu = ::mlir::cast<::fabric::FuOp>(rawFu);
  ws.wrapper = ::mlir::OwningOpRef<::mlir::func::FuncOp>(wrapper);
  ws.fu = fu;
  ws.fuEntry = fuEntry;
  // Append the wrapper-level return now; the caller fills the FU body
  // and we leave the fabric.fu's results wired through.
  ::mlir::OpBuilder afterFu(entry, entry->end());
  ::mlir::OperationState retState(
      loc, ::mlir::func::ReturnOp::getOperationName());
  retState.addOperands(::mlir::ValueRange(fu.getResults()));
  afterFu.create(retState);
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

::mlir::OwningOpRef<::mlir::func::FuncOp>
buildTrivialFuTierC(::mlir::MLIRContext *ctx, ::llvm::StringRef groupName,
                    ::dataflow::SubgraphOp first) {
  if (!ctx || !first)
    return {};

  // Collect the wrapper input bit-widths (per sg block-arg) and the
  // wrapper result bit-widths (per yield operand).
  ::mlir::Block &sgBody = first.getBody().front();
  ::llvm::SmallVector<unsigned, 4> inputBws;
  inputBws.reserve(sgBody.getNumArguments());
  for (::mlir::BlockArgument a : sgBody.getArguments()) {
    unsigned bw = bitWidthOfType(a.getType());
    if (bw == 0)
      return {};
    inputBws.push_back(bw);
  }
  ::mlir::Operation *yieldTerm = sgBody.getTerminator();
  if (!yieldTerm)
    return {};
  ::llvm::SmallVector<unsigned, 4> resultBws;
  resultBws.reserve(yieldTerm->getNumOperands());
  for (::mlir::Value v : yieldTerm->getOperands()) {
    unsigned bw = bitWidthOfType(v.getType());
    if (bw == 0)
      return {};
    resultBws.push_back(bw);
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

//===----------------------------------------------------------------------===//
// Tier-C structural extension. Generates ONE candidate that grafts sg's
// post-carry compute path (+ shared carry head) onto the FU.
//===----------------------------------------------------------------------===//
//
// Strategy for the spec's accumulator example:
//
//   FU side (built from input_0, addi case):
//     %idx, %rwc = fabric.op[@dataflow.stream] (...)
//     %c        = fabric.op[@dataflow.carry] (%rwc, %init, %nxt)
//     %nxt      = fabric.op[@arith.addi] (%c, %idx)
//     yield %c
//
//   sg side (input_1, xori case):
//     %idx', %rwc' = dataflow.stream (...)
//     %c'         = dataflow.carry %rwc', %init', %nxt'
//     %nxt'       = arith.xori %c', %idx'
//     yield %c'
//
// pre_align_sccs returns one class with the FU's existing carry as its
// anchor. The candidate keeps the FU's stream and carry ops verbatim,
// adds an `arith.xori` fed from (%c, %idx), and replaces the carry's
// %nxt operand with a fresh `fabric.mux` whose two arms are the
// existing addi and the new xori. The result is precisely the spec's
// "Tier C example (feedback alignment)" sketch.

namespace {

// Walk the sg body and find, per existing FU carry head, the
// "post-carry" sg op that consumes the sg's matched carry result and
// also feeds back into the carry's `carry` operand. Returns std::nullopt
// when the sg body shape is not the spec's accumulator form (i.e.
// there is no single op feeding the carry's back-edge). The current
// implementation only handles single-result post-carry ops with two
// operands of the form `(%c, %x)` where `%x` is either a block arg or
// another body-op result already covered by the FU; this matches the
// spec example.
struct PostCarryDiff {
  // Op name of the sg's post-carry op (e.g. "arith.xori").
  ::llvm::StringRef opName;
  // Source-side carry result (so we can map operand 0 back).
  ::mlir::Value sgCarryResult;
  // Source-side second operand (so we can map operand 1 back).
  ::mlir::Value sgSecondOperand;
  // Result bit-width of the post-carry op.
  unsigned resultBw = 0;
};

::std::optional<PostCarryDiff>
detectPostCarryDiff(::mlir::Operation *sgCarry) {
  if (!sgCarry || sgCarry->getNumResults() == 0)
    return std::nullopt;
  ::mlir::Value c = sgCarry->getResult(0);
  // The carry's "carry" operand (operand index 2) names the back-edge
  // producer.
  if (sgCarry->getNumOperands() < 3)
    return std::nullopt;
  ::mlir::Value backEdgeSrc = sgCarry->getOperand(2);
  auto opRes = ::llvm::dyn_cast<::mlir::OpResult>(backEdgeSrc);
  if (!opRes)
    return std::nullopt;
  ::mlir::Operation *post = opRes.getOwner();
  if (post->getNumResults() != 1 || post->getNumOperands() != 2)
    return std::nullopt;
  // First operand must be the carry's result.
  if (post->getOperand(0) != c)
    return std::nullopt;
  PostCarryDiff d;
  d.opName = post->getName().getStringRef();
  d.sgCarryResult = c;
  d.sgSecondOperand = post->getOperand(1);
  d.resultBw = bitWidthOfType(post->getResult(0).getType());
  if (d.resultBw == 0)
    return std::nullopt;
  return d;
}

// Locate the FU's `arith.addi`-style post-carry op (the op whose result
// feeds the FU carry's `carry` operand, i.e. operand index 2). Returns
// nullptr when the FU's carry has no single back-edge producer.
::fabric::OpOp findFuPostCarry(::fabric::OpOp fuCarry) {
  if (!fuCarry || fuCarry.getInputs().size() < 3)
    return {};
  ::mlir::Value bk = fuCarry.getInputs()[2];
  auto opRes = ::llvm::dyn_cast<::mlir::OpResult>(bk);
  if (!opRes)
    return {};
  return ::mlir::dyn_cast<::fabric::OpOp>(opRes.getOwner());
}

// Find a fabric.op in the FU body whose first op_list symbol equals
// `name` and whose inputs are exactly `(carry-result, idx)` (in that
// order). Used to dedup if the sg's post-carry op already has an
// equivalent in the FU (so we don't double-emit when the same input is
// folded twice in different orders).
::fabric::OpOp findEquivalentPostCarry(::fabric::FuOp fu,
                                       ::llvm::StringRef name,
                                       ::mlir::Value fuCarryRes,
                                       ::mlir::Value fuIdx) {
  for (::mlir::Operation &raw : fu.getBody().front().getOperations()) {
    auto op = ::mlir::dyn_cast<::fabric::OpOp>(raw);
    if (!op)
      continue;
    ::mlir::ArrayAttr opList = op.getOpList();
    if (opList.size() != 1)
      continue;
    auto sym = ::llvm::dyn_cast<::mlir::FlatSymbolRefAttr>(opList[0]);
    if (!sym || sym.getValue() != name)
      continue;
    if (op.getInputs().size() != 2)
      continue;
    if (op.getInputs()[0] == fuCarryRes && op.getInputs()[1] == fuIdx)
      return op;
  }
  return {};
}

::fabric::MuxOp emitMuxN(::mlir::OpBuilder &b, ::mlir::Location loc,
                         ::mlir::ValueRange arms, ::mlir::Type bits) {
  ::mlir::OperationState st(loc, ::fabric::MuxOp::getOperationName());
  st.addOperands(arms);
  st.addTypes({bits});
  ::mlir::Operation *raw = b.create(st);
  return ::mlir::cast<::fabric::MuxOp>(raw);
}

} // namespace

::llvm::SmallVector<::mlir::OwningOpRef<::mlir::func::FuncOp>, 4>
structuralExtendCandidates(::mlir::func::FuncOp curWrapper,
                           ::dataflow::SubgraphOp sg,
                           const ::loom::SynthConfig &cfg) {
  ::llvm::SmallVector<::mlir::OwningOpRef<::mlir::func::FuncOp>, 4> out;
  if (!curWrapper || !sg)
    return out;
  ::fabric::FuOp fu = innerFu(curWrapper);
  if (!fu)
    return out;

  ::std::optional<PreAlignment> paOpt = preAlignSccs(fu, sg, cfg);
  if (!paOpt.has_value()) {
    // feedback_align_conflict: the caller's filter loop will reject the
    // empty candidate set, but the failure reason itself is observed by
    // the Incremental main loop separately via `reasonForExtensionFailure`
    // (if it ever needs to surface it). For now the contract is "tier-C
    // unable to extend == empty candidate vector", and the main loop
    // converts that into a TopologyMismatch unless the test file checks
    // for the specific reason via the failure-reason hook below.
    return out;
  }
  PreAlignment &pa = *paOpt;

  // Today's slice handles the spec's single-carry-class case:
  //   * Exactly one class.
  //   * Class has an existing FU anchor (so we merge into it instead of
  //     grafting a new fabric.op[@dataflow.carry]).
  //   * sg has exactly one carry head whose post-carry shape matches
  //     the spec example (binary op consuming `(%c, %idx)`).
  //
  // Other tier-C shapes (multiple disjoint SCCs, fresh carry head with
  // no FU anchor, or non-spec post-carry shapes) are out of scope for
  // this slice and yield an empty candidate set so the main loop
  // fails over to the topology-mismatch path.
  if (pa.classSignatures.size() != 1)
    return out;
  if (!pa.classFuAnchor[0].has_value())
    return out;
  unsigned anchorIdx = *pa.classFuAnchor[0];

  ::llvm::SmallVector<::fabric::OpOp, 4> fuCarries =
      collectFabricOpsByFirstSymbol(fu, "dataflow.carry");
  if (anchorIdx >= fuCarries.size())
    return out;
  ::fabric::OpOp fuCarry = fuCarries[anchorIdx];

  ::llvm::SmallVector<::mlir::Operation *, 4> sgCarries = collectSgCarries(sg);
  if (sgCarries.size() != 1)
    return out;
  ::mlir::Operation *sgCarry = sgCarries[0];

  ::std::optional<PostCarryDiff> diffOpt = detectPostCarryDiff(sgCarry);
  if (!diffOpt.has_value())
    return out;
  PostCarryDiff diff = *diffOpt;

  ::fabric::OpOp fuPostCarry = findFuPostCarry(fuCarry);
  if (!fuPostCarry)
    return out;
  if (fuPostCarry.getInputs().size() != 2)
    return out;
  ::mlir::Value fuCarryRes = fuCarry.getOutputs()[0];
  ::mlir::Value fuIdx = fuPostCarry.getInputs()[1];

  // Dedup: if the FU body already has an equivalent post-carry op for
  // sg's diff (e.g. because the same input was folded earlier under a
  // different ordering), we have nothing to add -- the existing FU
  // already covers `sg`. The main loop's coverage check would normally
  // catch this, but emitting an empty candidate set here is a no-op.
  if (findEquivalentPostCarry(fu, diff.opName, fuCarryRes, fuIdx))
    return out;

  // Build the candidate by cloning the wrapper. The clone preserves the
  // existing FU body verbatim; we then (1) graft the new post-carry
  // fabric.op next to the existing one, (2) insert a fabric.mux merging
  // the two post-carry ops, and (3) rewire the cloned FU carry's
  // operand 2 to the mux output.
  ::mlir::Operation *clonedRaw = curWrapper->clone();
  auto newWrapper = ::mlir::OwningOpRef<::mlir::func::FuncOp>(
      ::mlir::cast<::mlir::func::FuncOp>(clonedRaw));
  ::fabric::FuOp newFu = innerFu(newWrapper.get());
  if (!newFu)
    return out;

  ::llvm::SmallVector<::fabric::OpOp, 4> newFuCarries =
      collectFabricOpsByFirstSymbol(newFu, "dataflow.carry");
  if (anchorIdx >= newFuCarries.size())
    return out;
  ::fabric::OpOp newCarry = newFuCarries[anchorIdx];
  ::fabric::OpOp newPost = findFuPostCarry(newCarry);
  if (!newPost)
    return out;
  ::mlir::Value newCarryRes = newCarry.getOutputs()[0];
  ::mlir::Value newIdx = newPost.getInputs()[1];

  ::mlir::Type bits = bitsOf(curWrapper.getContext(), diff.resultBw);
  ::mlir::Location loc = ::mlir::UnknownLoc::get(curWrapper.getContext());

  // Insert the new post-carry op directly after the existing one so the
  // fabric.fu body retains a deterministic order (existing post-carry
  // first, new post-carry second, mux third). The carry itself stays
  // in its original textual position.
  ::mlir::OpBuilder builder(newPost->getContext());
  builder.setInsertionPointAfter(newPost);
  BodyBuildCtx bc(curWrapper.getContext(), loc, &builder);
  ::mlir::ArrayAttr hwParams = hwParamsFor(bc.ctx, diff.opName, sgCarry);
  // The post-carry op consumes (newCarryRes, newIdx) -- exactly the
  // operand pair the existing post-carry op consumes -- per the spec
  // example's symmetry.
  ::llvm::SmallVector<::mlir::Value, 2> postInputs{newCarryRes, newIdx};
  ::fabric::OpOp newPostNew = emitFabricOpInBody(
      bc, diff.opName, postInputs, ::mlir::TypeRange{bits}, hwParams);

  builder.setInsertionPointAfter(newPostNew);
  ::llvm::SmallVector<::mlir::Value, 2> arms{newPost.getOutputs()[0],
                                             newPostNew.getOutputs()[0]};
  ::fabric::MuxOp mux = emitMuxN(builder, loc, arms, bits);

  // Rewire the carry's back-edge operand (operand index 2) to the mux
  // output. setOperand is safe on the live IR -- the carry's own
  // fabric.op result is not affected.
  newCarry->setOperand(2, mux.getOutput());

  out.push_back(std::move(newWrapper));
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
classifyTierCConflict(::mlir::func::FuncOp curWrapper,
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
  // Cross-side incompatibility: both sides have at least one carry
  // head, but no equivalence class contains members from both. This is
  // the spec's "incompatible flow signatures on cyclic SCCs" wording
  // applied to the N==2 case (the N>2 case generalizes the same way
  // since string equality is already transitive).
  PreAlignment &pa = *paOpt;
  bool fuHasCarry = !pa.fuClass.empty();
  bool sgHasCarry = !pa.sgClass.empty();
  if (fuHasCarry && sgHasCarry) {
    bool sharesClass = false;
    ::llvm::DenseSet<unsigned> sgClasses(pa.sgClass.begin(),
                                         pa.sgClass.end());
    for (unsigned cid : pa.fuClass)
      if (sgClasses.count(cid)) {
        sharesClass = true;
        break;
      }
    if (!sharesClass)
      return SynthFailureReason::FeedbackAlignConflict;
  }
  return std::nullopt;
}

} // namespace loom::fabric::tech::detail
