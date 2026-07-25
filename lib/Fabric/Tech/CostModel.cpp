#include "Fabric/Tech/CostModel.h"

#include "Common/SynthConfig.h"
#include "Fabric/IR/FabricOps.h"
#include "Fabric/IR/FabricTypes.h"
#include "Fabric/IR/ImplementationFamily.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"

namespace loom::fabric::tech {

namespace {

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

} // namespace

double baseUnitFor(::fabric::ImplementationFamilyId family) {
  using Family = ::fabric::ImplementationFamilyId;
  switch (family) {
  case Family::ScalarIntegerAddSub:
  case Family::ScalarIntegerCompareMinMax:
  case Family::ScalarValueSelect:
  case Family::ScalarIntegerCast:
  case Family::ScalarFloatSign:
  case Family::ScalarFloatWidthCast:
  case Family::LoopStream:
  case Family::LoopInvariant:
  case Family::LoopGate:
    return 1.0;
  case Family::ScalarIntegerLogic:
  case Family::ScalarBitReinterpret:
    return 0.5;
  case Family::ScalarIntegerShift:
    return 1.5;
  case Family::ScalarFloatAddSub:
    return 4.0;
  case Family::ScalarFloatCompareMinMax:
  case Family::ScalarIntegerToFloat:
  case Family::ScalarFloatToInteger:
  case Family::ScalarIntegerMultiply:
  case Family::ScalarFloatMultiply:
    return 3.0;
  case Family::ScalarFloatFma:
    return 8.0;
  case Family::LoopCarry:
    return 1.0;
  }
  llvm_unreachable("unregistered implementation family");
}

CostModel::CostModel(const ::loom::SynthConfig &config) {
  weights.muxPenalty = config.costMuxPenalty;
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
      std::optional<::fabric::ImplementationFamilyId> family =
          opOp.getImplementationFamily();
      if (!family)
        continue;
      unsigned bw = firstResultWidth(opOp);
      if (*family == ::fabric::ImplementationFamilyId::LoopCarry) {
        total += weights.carryPenalty * static_cast<double>(bw);
      } else {
        const double base = baseUnitFor(*family);
        // baseArea(group, bw) = baseUnit[group] * (bw / 32.0)
        total += base * (static_cast<double>(bw) / 32.0);
      }
      continue;
    }
    if (auto mux = ::llvm::dyn_cast<::fabric::MuxOp>(&raw)) {
      const unsigned ports = static_cast<unsigned>(mux.getInputs().size());
      const unsigned bw = bitWidthOfType(mux.getOutput().getType());
      total += weights.muxPenalty * static_cast<double>(ports) *
               static_cast<double>(bw);
      continue;
    }
    if (auto demux = ::llvm::dyn_cast<::fabric::DemuxOp>(&raw)) {
      const unsigned ports = static_cast<unsigned>(demux.getOutputs().size());
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
