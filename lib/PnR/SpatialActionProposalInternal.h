#ifndef LOOM_LIB_PNR_SPATIALACTIONPROPOSALINTERNAL_H
#define LOOM_LIB_PNR_SPATIALACTIONPROPOSALINTERNAL_H

#include "PnR/SpatialAction.h"

namespace loom::pnr::detail {

llvm::Expected<std::optional<SpatialMappingAction>>
proposeCanonicalSpatialAction(const ResolvedPnrActionProposalPolicy &policy,
                              SpatialActionProposalDomain domain,
                              DeterministicPnrRandomStream &proposalStream);

} // namespace loom::pnr::detail

#endif // LOOM_LIB_PNR_SPATIALACTIONPROPOSALINTERNAL_H
