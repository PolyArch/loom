#ifndef LOOM_ADG_FULIBRARY_H
#define LOOM_ADG_FULIBRARY_H

#include "ADG/Builder.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

namespace loom::adg {

/// Adds the catalog's scalar ALU FU to one PE. Inputs are data0, data1, and
/// condition, in that order. The helper constructs only ordinary Fabric
/// resources and closes the FU before returning.
llvm::Error addCoreAluFu(PeBuilder &pe, llvm::ArrayRef<PeValue> inputs);

/// Adds scalar multiply, fused and non-fused multiply-add, and local carry
/// recurrence graphs. Inputs are data0, data1, data2, and phase.
llvm::Error addMacFu(PeBuilder &pe, llvm::ArrayRef<PeValue> inputs);

/// Adds the fixed-vector compute FU. Inputs are data0, data1, data2, and
/// vector condition, in that order.
llvm::Error addVectorComputeFu(PeBuilder &pe, llvm::ArrayRef<PeValue> inputs);

/// Adds fixed-vector representation and stream-group adapters. Inputs are
/// data/vector, mask, and phase. Results are data/vector, mask, and phase.
llvm::Error addVectorAdapterFu(PeBuilder &pe, llvm::ArrayRef<PeValue> inputs);

/// Adds constant, rendezvous, and runtime token-routing resources. Inputs are
/// selector/control followed by four payload lanes. Four payload lanes are
/// exposed as results.
llvm::Error addTokenControlFu(PeBuilder &pe, llvm::ArrayRef<PeValue> inputs);

/// Adds the low-density scalar divide, remainder, and elementary math FU.
/// Inputs are data0 and data1, in that order.
llvm::Error addSpecialMathFu(PeBuilder &pe, llvm::ArrayRef<PeValue> inputs);

} // namespace loom::adg

#endif // LOOM_ADG_FULIBRARY_H
