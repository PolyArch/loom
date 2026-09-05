#ifndef LOOM_LIB_PNR_HANDSHAKECYCLEDIAGNOSTICS_H
#define LOOM_LIB_PNR_HANDSHAKECYCLEDIAGNOSTICS_H

#include "PnR/SpatialPnrProblem.h"

namespace loom::pnr::detail {

enum class HandshakeCycleOrigin { Candidate, Projection };

/// Both callers translate their compact witness into the frozen graph's
/// numbering. Diagnostics derive contributors from that same immutable index.
void emitHandshakeCycleDiagnostic(
    const FrozenSpatialHandshakeIndex &index, HandshakeCycleOrigin origin,
    llvm::ArrayRef<PnrIndex> frozenWitness,
    llvm::ArrayRef<PnrIndex> activeFragments,
    llvm::ArrayRef<PnrIndex> fragmentRefcounts);

} // namespace loom::pnr::detail

#endif // LOOM_LIB_PNR_HANDSHAKECYCLEDIAGNOSTICS_H
