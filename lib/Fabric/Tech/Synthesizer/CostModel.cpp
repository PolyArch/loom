#include "Fabric/Tech/Synthesizer/CostModel.h"

#include "Common/HwShareGroup.h"
#include "Common/SynthConfig.h"
#include "Fabric/IR/FabricOps.h"
#include "Fabric/IR/FabricTypes.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"

#include <cstddef>
#include <optional>

namespace loom::fabric::tech {

namespace {

// 13-row baseUnit table from the spec, keyed by a representative op name
// belonging to each share group. All ops in the same share group return
// the same baseUnit (a share group is always assigned a single area cost
// in this analytic model). The table is built once and indexed by the
// `findShareGroup` result for fast O(1) lookups at evaluation time.
//
// Singleton fallback: any op name absent from `hwShareGroups()` (i.e.
// `findShareGroup(name) == nullopt`) maps to baseUnit = 1.0.
//
// Order of rows follows the spec table verbatim. Multiple share groups
// that share the same baseUnit value (e.g. divsi/remsi vs divui/remui)
// each get a representative entry so every group index is covered.
struct BaseUnitEntry {
  ::llvm::StringRef representative;
  double baseUnit;
};

::llvm::ArrayRef<BaseUnitEntry> baseUnitTable() {
  static const BaseUnitEntry kEntries[] = {
      {"arith.addi", 1.0},        // arith.addi/subi
      {"arith.andi", 0.5},        // arith.andi/ori/xori
      {"arith.shli", 1.5},        // arith.shli/shrsi/shrui
      {"arith.minsi", 1.0},       // arith.minsi/maxsi
      {"arith.minui", 1.0},       // arith.minui/maxui
      {"arith.divsi", 8.0},       // arith.divsi/remsi
      {"arith.divui", 8.0},       // arith.divui/remui
      {"arith.addf", 4.0},        // arith.addf/subf
      {"arith.divf", 12.0},       // arith.divf/remf
      {"arith.minimumf", 3.0},    // arith.minimumf/maximumf
      {"arith.sitofp", 3.0},      // arith.sitofp/uitofp
      {"arith.fptosi", 3.0},      // arith.fptosi/fptoui
      {"math.sin", 16.0},         // math.sin/cos
      {"math.sinh", 16.0},        // math.sinh/cosh
      {"math.tanh", 16.0},        // math.tanh/erf
      {"math.exp", 12.0},         // math.exp/exp2/expm1
      {"math.log", 12.0},         // math.log/log2/log10/log1p
      {"math.sqrt", 8.0},         // math.sqrt/rsqrt
      {"math.floor", 2.0},        // math.floor/ceil/round/trunc/roundeven
  };
  return ::llvm::ArrayRef<BaseUnitEntry>(kEntries);
}

// Lookup: share-group index -> baseUnit. Built lazily from `baseUnitTable`
// using `findShareGroup` on each representative. Cached in a function-local
// static. Singleton fallback (nullopt index) returns 1.0 from `baseUnitFor`.
const ::llvm::DenseMap<size_t, double> &baseUnitByGroupIndex() {
  static const ::llvm::DenseMap<size_t, double> kMap = []() {
    ::llvm::DenseMap<size_t, double> m;
    for (const BaseUnitEntry &e : baseUnitTable()) {
      auto idx = ::loom::common::findShareGroup(e.representative);
      if (idx)
        m.try_emplace(*idx, e.baseUnit);
    }
    return m;
  }();
  return kMap;
}

// Bitwidth of a fabric.bits<N> result. fabric.op / mux / demux are
// constrained to `Fabric_BitsType` ports inside an `fabric.fu`, so the
// cast is always safe; if a future op grows non-bits result types, we
// fall back to width 0 to avoid undefined cost contributions.
unsigned bitWidthOfType(::mlir::Type t) {
  if (auto bits = ::llvm::dyn_cast<::fabric::BitsType>(t))
    return bits.getWidth();
  return 0;
}

// First-result bitwidth for an OpOp / MuxOp / DemuxOp. fabric.op may have
// any number of result ports; spec uses `bitwidthOf(op)` which is the
// op's representative width (all ports in a fabric.op share the same
// bitwidth in practice). For DemuxOp and MuxOp, all ports share one
// fabric type by SameOperandsAndResultType.
unsigned firstResultWidth(::mlir::Operation *op) {
  if (op->getNumResults() == 0)
    return 0;
  return bitWidthOfType(op->getResult(0).getType());
}

// op_list[0] symbol value, e.g. "arith.addi". Returns empty StringRef
// when op_list is empty (which a verified fabric.op cannot reach, but
// we guard so evaluate() never crashes on malformed IR).
::llvm::StringRef firstOpListSymbol(::fabric::OpOp op) {
  ::mlir::ArrayAttr opList = op.getOpList();
  if (opList.empty())
    return {};
  auto sym = ::llvm::dyn_cast<::mlir::FlatSymbolRefAttr>(opList[0]);
  if (!sym)
    return {};
  return sym.getValue();
}

} // namespace

double baseUnitFor(::std::optional<::std::size_t> shareGroupIndex) {
  if (!shareGroupIndex)
    return 1.0; // singleton fallback per spec.
  const auto &m = baseUnitByGroupIndex();
  auto it = m.find(*shareGroupIndex);
  if (it == m.end())
    return 1.0; // group present in hwShareGroups() but absent from spec table.
  return it->second;
}

CostModel::CostModel(const ::loom::SynthConfig &config) {
  weights.muxPenalty   = config.costMuxPenalty;
  weights.demuxPenalty = config.costDemuxPenalty;
  weights.carryPenalty = config.costCarryPenalty;
}

double CostModel::evaluate(::fabric::FuOp fu) const {
  double total = 0.0;

  // Iterate the FU's body block. Spec restricts the body to fabric.op /
  // mux / demux plus the fabric.yield terminator, so any other op is
  // ignored (no contribution) — matches the formula's explicit per-op-kind
  // sums.
  ::mlir::Region &body = fu.getBody();
  if (body.empty())
    return total;

  for (::mlir::Operation &raw : body.front()) {
    if (auto opOp = ::llvm::dyn_cast<::fabric::OpOp>(&raw)) {
      ::llvm::StringRef sym = firstOpListSymbol(opOp);
      unsigned bw = firstResultWidth(opOp);
      if (sym == "dataflow.carry") {
        total += weights.carryPenalty * static_cast<double>(bw);
      } else {
        const double base = baseUnitFor(::loom::common::findShareGroup(sym));
        // baseArea(group, bw) = baseUnit[group] * (bw / 32.0)
        total += base * (static_cast<double>(bw) / 32.0);
      }
      continue;
    }
    if (auto mux = ::llvm::dyn_cast<::fabric::MuxOp>(&raw)) {
      const unsigned ports =
          static_cast<unsigned>(mux.getInputs().size());
      const unsigned bw = bitWidthOfType(mux.getOutput().getType());
      total += weights.muxPenalty * static_cast<double>(ports) *
               static_cast<double>(bw);
      continue;
    }
    if (auto demux = ::llvm::dyn_cast<::fabric::DemuxOp>(&raw)) {
      const unsigned ports =
          static_cast<unsigned>(demux.getOutputs().size());
      const unsigned bw = bitWidthOfType(demux.getInput().getType());
      total += weights.demuxPenalty * static_cast<double>(ports) *
               static_cast<double>(bw);
      continue;
    }
    // fabric.yield and any other ops contribute nothing.
  }
  return total;
}

} // namespace loom::fabric::tech
