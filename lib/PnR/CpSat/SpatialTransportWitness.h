#ifndef LOOM_LIB_PNR_CPSAT_SPATIALTRANSPORTWITNESS_H
#define LOOM_LIB_PNR_CPSAT_SPATIALTRANSPORTWITNESS_H

#include "Common/ResolvedPnrPolicy.h"
#include "PnR/PnrIndex.h"

#include "llvm/Support/Error.h"

#include <optional>

namespace loom::pnr {
class SpatialCandidateState;

namespace detail {

struct SpatialTransportWitness final {
  ResolvedPnrViolationKind kind;
  PnrIndex ordinal;
};

/// Selects the first live transport violation in canonical repair order.
llvm::Expected<std::optional<SpatialTransportWitness>>
firstSpatialTransportWitness(const SpatialCandidateState &candidate);

/// Rechecks the same violation after a provisional or committed repair.
llvm::Expected<bool>
spatialTransportWitnessIsLive(const SpatialCandidateState &candidate,
                              SpatialTransportWitness witness);

} // namespace detail
} // namespace loom::pnr

#endif // LOOM_LIB_PNR_CPSAT_SPATIALTRANSPORTWITNESS_H
