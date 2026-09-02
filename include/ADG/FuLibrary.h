#ifndef LOOM_ADG_FULIBRARY_H
#define LOOM_ADG_FULIBRARY_H

#include "ADG/Builder.h"
#include "ADG/SpecialMathCapabilityProfile.h"
#include "Dataflow/IR/DataflowEnums.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>

namespace loom::adg {

/// Adds the catalog's scalar ALU FU to one PE. Inputs are data0, data1, and
/// condition, in that order. The helper constructs only ordinary Fabric
/// resources and closes the FU before returning.
llvm::Error addCoreAluFu(PeBuilder &pe, llvm::ArrayRef<PeValue> inputs,
                         ::fabric::ResolvedIndexWidthSet resolvedIndexWidths);

/// Adds scalar multiply, fused and non-fused multiply-add, and local carry
/// recurrence graphs. Inputs are data0, data1, data2, and phase.
llvm::Error addMacFu(PeBuilder &pe, llvm::ArrayRef<PeValue> inputs);

/// Adds two fixed-step stream resources plus carry, invariant, and gate
/// resources. Inputs are data0, data1, data2, and phase. The two stream step
/// kinds identify distinct physical resources and must differ.
llvm::Error addLoopControlFu(PeBuilder &pe, llvm::ArrayRef<PeValue> inputs,
                             ::dataflow::StreamStepKind firstStep,
                             ::dataflow::StreamStepKind secondStep);

/// Transient typed widths for one vector-compute FU expansion. The emitted
/// physical ports and capability records remain the only persistent owner.
struct VectorComputeFuParameters final {
  std::uint32_t outerPayloadBits;
  std::uint32_t vectorPayloadBits;
};

/// Adds the fixed-vector compute FU. Inputs are data0, data1, data2, and
/// vector condition, in that order.
llvm::Error addVectorComputeFu(PeBuilder &pe, llvm::ArrayRef<PeValue> inputs,
                               VectorComputeFuParameters parameters);

/// Transient typed inputs for one vector-structure FU expansion. The emitted
/// Fabric ports and capability records are the only persistent authority.
struct VectorStructuralFuParameters final {
  std::uint32_t outerPayloadBits;
  std::uint32_t vectorPayloadBits;
  std::uint32_t indexPayloadBits;
  ::fabric::FixedVectorSliceAlignMergeParams sliceCapability;
  ::fabric::FixedVectorShuffleParams shuffleCapability;
};

/// Adds fixed-vector leading-slice alignment/merge and shuffle resources.
/// Inputs are two vector/value roles followed by the slice capability's
/// maximum number of dynamic-position roles.
llvm::Error
addVectorStructuralFu(PeBuilder &pe, llvm::ArrayRef<PeValue> inputs,
                      const VectorStructuralFuParameters &parameters);

/// Adds fixed-vector representation and stream-group adapters. Inputs are
/// data/vector, mask, and phase. Results are data/vector, mask, and phase.
llvm::Error addVectorAdapterFu(PeBuilder &pe, llvm::ArrayRef<PeValue> inputs);

/// Adds constant, rendezvous, and runtime token-routing resources. Inputs are
/// selector/control followed by four payload lanes. Four payload lanes are
/// exposed as results.
struct TokenControlFuParameters final {
  std::uint32_t outerPayloadBits;
  std::uint32_t selectorPayloadBits;
};

llvm::Error addTokenControlFu(PeBuilder &pe, llvm::ArrayRef<PeValue> inputs,
                              TokenControlFuParameters parameters);

/// Adds the low-density scalar divide, remainder, and elementary math FU.
/// Inputs are data0 and data1, in that order.
llvm::Error addSpecialMathFu(PeBuilder &pe, llvm::ArrayRef<PeValue> inputs,
                             BuiltinSpecialMathCapabilityProfile profile);

} // namespace loom::adg

#endif // LOOM_ADG_FULIBRARY_H
