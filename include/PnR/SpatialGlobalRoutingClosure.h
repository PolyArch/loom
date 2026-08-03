#ifndef LOOM_PNR_SPATIALGLOBALROUTINGCLOSURE_H
#define LOOM_PNR_SPATIALGLOBALROUTINGCLOSURE_H

#include "PnR/SpatialActionExecutor.h"

#include "llvm/Support/Error.h"

#include <cstddef>

namespace loom::pnr {

/// Executes one final Global TransportRoutingAction through the ordinary
/// Spatial Action transaction. This owner closes only route, route-capacity,
/// and route-derived tag state; complete final verification additionally
/// requires every other Mapping violation owner.
class SpatialGlobalRoutingClosureScratch final {
public:
  llvm::Error run(SpatialCandidateState &candidate);

  std::size_t retainedStorageBytes() const;

private:
  SpatialActionExecutorScratch actionExecutor_;
};

} // namespace loom::pnr

#endif // LOOM_PNR_SPATIALGLOBALROUTINGCLOSURE_H
