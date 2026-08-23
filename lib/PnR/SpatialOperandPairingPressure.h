#ifndef LOOM_LIB_PNR_SPATIALOPERANDPAIRINGPRESSURE_H
#define LOOM_LIB_PNR_SPATIALOPERANDPAIRINGPRESSURE_H

#include "PnR/SpatialPnrProblem.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>

namespace loom::pnr::detail {

/// Measures the derived shared-ingress pressure of selected Temporal operand
/// attachments. Each group contributes the number of independently produced
/// members beyond its distinct physical ingress count. It is an analytic QoR
/// measure, never a Mapping legality or liveness proof.
llvm::Expected<std::uint64_t> measureSpatialOperandIngressPressure(
    const FrozenSpatialPnrProblem &problem,
    llvm::ArrayRef<PnrIndex> portAttachments,
    llvm::ArrayRef<PnrIndex> registerFifoTransfers);

/// Measures a canonical subset of pairing groups. Candidate transactions use
/// the demand-to-group reverse CSR to update only the affected cone.
llvm::Expected<std::uint64_t> measureSpatialOperandIngressPressure(
    const FrozenSpatialPnrProblem &problem,
    llvm::ArrayRef<PnrIndex> portAttachments,
    llvm::ArrayRef<PnrIndex> registerFifoTransfers,
    llvm::ArrayRef<PnrIndex> pairingGroups);

} // namespace loom::pnr::detail

#endif // LOOM_LIB_PNR_SPATIALOPERANDPAIRINGPRESSURE_H
