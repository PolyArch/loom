// Implementation of buildHwParamsUnion. See HwParams.h.

#include "Fabric/Tech/Synthesizer/HwParams.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include <set>
#include <string>

namespace loom::fabric::tech {

namespace {

// Build a sorted, deduplicated ArrayAttr of StringAttr from a set of
// strings. The set is std::set so iteration order is already lexical.
::mlir::ArrayAttr stringSetToArray(::mlir::MLIRContext *ctx,
                                   const ::std::set<::std::string> &s) {
  ::llvm::SmallVector<::mlir::Attribute, 4> v;
  v.reserve(s.size());
  for (const ::std::string &x : s)
    v.push_back(::mlir::StringAttr::get(ctx, x));
  return ::mlir::ArrayAttr::get(ctx, v);
}

// Render an APInt as a 0x... hex string with leading zeros stripped (but
// the literal "0x0" is preserved for zero). The width drives how many
// hex characters are needed for the unsigned value; we use the smaller
// of the bit width or the value's significant bits.
::std::string apintToHexString(const ::llvm::APInt &v) {
  ::llvm::SmallString<32> hex;
  v.toString(hex, /*Radix=*/16, /*Signed=*/false, /*formatAsCLiteral=*/false);
  // Lower-case to match the canonical form `0xdeadbeef` used in tests.
  for (char &c : hex)
    if (c >= 'A' && c <= 'F')
      c = static_cast<char>(c - 'A' + 'a');
  ::std::string out = "0x";
  out += hex.c_str();
  return out;
}

// Encode a peer's `dataflow.constant` `const_value` attribute as the
// canonical hex string the enumerator's parseConstHex consumes.
// Returns std::nullopt when the attribute is unsupported.
::std::optional<::std::string>
encodeConstHex(::mlir::Attribute attr) {
  if (auto i = ::llvm::dyn_cast<::mlir::IntegerAttr>(attr))
    return apintToHexString(i.getValue());
  if (auto f = ::llvm::dyn_cast<::mlir::FloatAttr>(attr)) {
    ::llvm::APInt bits = f.getValue().bitcastToAPInt();
    return apintToHexString(bits);
  }
  if (auto s = ::llvm::dyn_cast<::mlir::StringAttr>(attr)) {
    ::llvm::StringRef body = s.getValue();
    if (body.starts_with("0x") || body.starts_with("0X"))
      return body.str();
    return ::std::string("0x") + body.str();
  }
  return std::nullopt;
}

// Build hw_params for arith.cmpi / arith.cmpf by collecting each peer's
// `predicate` attribute (an enum-backed IntegerAttr) and stringifying it.
::mlir::ArrayAttr
buildPredicateUnion(::mlir::MLIRContext *ctx, ::llvm::StringRef opName,
                    ::llvm::ArrayRef<::mlir::Operation *> peers) {
  ::std::set<::std::string> values;
  for (::mlir::Operation *p : peers) {
    if (!p)
      continue;
    auto attr = p->getAttr("predicate");
    if (!attr)
      continue;
    if (opName == "arith.cmpi") {
      auto ia = ::llvm::dyn_cast<::mlir::IntegerAttr>(attr);
      if (!ia)
        continue;
      auto pred = static_cast<::mlir::arith::CmpIPredicate>(ia.getInt());
      values.insert(::mlir::arith::stringifyCmpIPredicate(pred).str());
    } else { // arith.cmpf
      auto ia = ::llvm::dyn_cast<::mlir::IntegerAttr>(attr);
      if (!ia)
        continue;
      auto pred = static_cast<::mlir::arith::CmpFPredicate>(ia.getInt());
      values.insert(::mlir::arith::stringifyCmpFPredicate(pred).str());
    }
  }
  if (values.empty())
    return {};
  ::llvm::SmallVector<::mlir::NamedAttribute, 1> entries{
      ::mlir::NamedAttribute(::mlir::StringAttr::get(ctx, "predicate"),
                             stringSetToArray(ctx, values))};
  ::mlir::DictionaryAttr inner = ::mlir::DictionaryAttr::get(ctx, entries);
  ::llvm::SmallVector<::mlir::Attribute, 1> outer{inner};
  return ::mlir::ArrayAttr::get(ctx, outer);
}

// Build hw_params for dataflow.stream by collecting each peer's
// `step_op` and `cont_cond` string attributes.
::mlir::ArrayAttr
buildStreamUnion(::mlir::MLIRContext *ctx,
                 ::llvm::ArrayRef<::mlir::Operation *> peers) {
  ::std::set<::std::string> stepOps, contConds;
  for (::mlir::Operation *p : peers) {
    if (!p)
      continue;
    if (auto so = p->getAttrOfType<::mlir::StringAttr>("step_op"))
      stepOps.insert(so.getValue().str());
    if (auto cc = p->getAttrOfType<::mlir::StringAttr>("cont_cond"))
      contConds.insert(cc.getValue().str());
  }
  ::llvm::SmallVector<::mlir::NamedAttribute, 2> entries;
  if (!contConds.empty())
    entries.emplace_back(::mlir::StringAttr::get(ctx, "cont_cond"),
                         stringSetToArray(ctx, contConds));
  if (!stepOps.empty())
    entries.emplace_back(::mlir::StringAttr::get(ctx, "step_op"),
                         stringSetToArray(ctx, stepOps));
  ::mlir::DictionaryAttr inner = ::mlir::DictionaryAttr::get(ctx, entries);
  ::llvm::SmallVector<::mlir::Attribute, 1> outer{inner};
  return ::mlir::ArrayAttr::get(ctx, outer);
}

// Build hw_params for dataflow.constant by encoding each peer's
// const_value attribute as a hex string.
::mlir::ArrayAttr
buildConstantUnion(::mlir::MLIRContext *ctx,
                   ::llvm::ArrayRef<::mlir::Operation *> peers) {
  ::std::set<::std::string> values;
  for (::mlir::Operation *p : peers) {
    if (!p)
      continue;
    auto cv = p->getAttr("const_value");
    if (!cv)
      continue;
    auto enc = encodeConstHex(cv);
    if (!enc.has_value())
      continue;
    values.insert(*enc);
  }
  if (values.empty())
    return {};
  ::llvm::SmallVector<::mlir::NamedAttribute, 1> entries{
      ::mlir::NamedAttribute(::mlir::StringAttr::get(ctx, "const_hex_value"),
                             stringSetToArray(ctx, values))};
  ::mlir::DictionaryAttr inner = ::mlir::DictionaryAttr::get(ctx, entries);
  ::llvm::SmallVector<::mlir::Attribute, 1> outer{inner};
  return ::mlir::ArrayAttr::get(ctx, outer);
}

// Build hw_params for variadic dataflow.{sync,mux,demux} ops. The
// emitted bitmask reflects the structural width of each peer:
//   * sync: M = numOperands  -> bitmask = "1" * M  (every operand active)
//   * mux:  M = numOperands - 1 (sel + M data); bitmask = "1" * M
//   * demux: M = numResults; bitmask = "1" * M
::mlir::ArrayAttr
buildBitmaskUnion(::mlir::MLIRContext *ctx, ::llvm::StringRef opName,
                  ::llvm::ArrayRef<::mlir::Operation *> peers) {
  ::std::set<::std::string> values;
  for (::mlir::Operation *p : peers) {
    if (!p)
      continue;
    unsigned m = 0;
    if (opName == "dataflow.sync") {
      m = p->getNumOperands();
    } else if (opName == "dataflow.mux") {
      // mux operands: sel + N data. M is the data port count.
      unsigned n = p->getNumOperands();
      m = n > 0 ? n - 1 : 0;
    } else if (opName == "dataflow.demux") {
      m = p->getNumResults();
    } else {
      continue;
    }
    if (m == 0)
      continue;
    values.insert(::std::string(m, '1'));
  }
  if (values.empty())
    return {};
  ::llvm::SmallVector<::mlir::NamedAttribute, 1> entries{
      ::mlir::NamedAttribute(::mlir::StringAttr::get(ctx, "bitmask"),
                             stringSetToArray(ctx, values))};
  ::mlir::DictionaryAttr inner = ::mlir::DictionaryAttr::get(ctx, entries);
  ::llvm::SmallVector<::mlir::Attribute, 1> outer{inner};
  return ::mlir::ArrayAttr::get(ctx, outer);
}

// Canonical "no configurable axis" hw_params: a length-1 array wrapping
// an empty dictionary.
::mlir::ArrayAttr emptyHwParams(::mlir::MLIRContext *ctx) {
  auto emptyDict = ::mlir::DictionaryAttr::get(ctx, {});
  ::llvm::SmallVector<::mlir::Attribute, 1> outer{emptyDict};
  return ::mlir::ArrayAttr::get(ctx, outer);
}

} // namespace

::mlir::ArrayAttr
buildHwParamsUnion(::mlir::MLIRContext *ctx, ::llvm::StringRef opName,
                   ::llvm::ArrayRef<::mlir::Operation *> peers) {
  if (opName == "arith.cmpi" || opName == "arith.cmpf") {
    if (auto a = buildPredicateUnion(ctx, opName, peers))
      return a;
    return emptyHwParams(ctx);
  }
  if (opName == "dataflow.stream") {
    return buildStreamUnion(ctx, peers);
  }
  if (opName == "dataflow.constant") {
    if (auto a = buildConstantUnion(ctx, peers))
      return a;
    return emptyHwParams(ctx);
  }
  if (opName == "dataflow.sync" || opName == "dataflow.mux" ||
      opName == "dataflow.demux") {
    if (auto a = buildBitmaskUnion(ctx, opName, peers))
      return a;
    return emptyHwParams(ctx);
  }
  // No configurable axis the enumerator inspects -> [{}].
  return emptyHwParams(ctx);
}

} // namespace loom::fabric::tech
