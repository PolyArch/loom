#ifndef LOOM_PNR_SYSTEM_SYSTEMROUTEMIGRATION_H
#define LOOM_PNR_SYSTEM_SYSTEMROUTEMIGRATION_H

#include "PnR/PnrIndex.h"

#include <vector>

namespace loom::pnr {

struct SystemServiceRouteNodeSelection final {
  PnrIndex endpoint = 0;
  PnrIndex parentNode = getInvalidPnrIndex();
  PnrIndex incomingTraversal = getInvalidPnrIndex();
};

struct SystemServiceRouteSinkSelection final {
  PnrIndex terminal = 0;
  PnrIndex node = 0;
};

struct SystemServiceRouteSelection final {
  PnrIndex leg = 0;
  PnrIndex rootEndpoint = getInvalidPnrIndex();
  PnrIndex nodeOffset = 0;
  PnrIndex nodeCount = 0;
  PnrIndex sinkOffset = 0;
  PnrIndex sinkCount = 0;
};

/// Child-indexed route preferences reconstructed from a finalized parent
/// Mapping. Entries in `reroutedLegs` carry empty placeholders; every other
/// route is immutable input to negotiated routing and is reverified.
struct SystemCandidateRouteSeed final {
  std::vector<SystemServiceRouteSelection> routes;
  std::vector<SystemServiceRouteNodeSelection> nodes;
  std::vector<SystemServiceRouteSinkSelection> sinks;
  std::vector<PnrIndex> reroutedLegs;
};

} // namespace loom::pnr

#endif // LOOM_PNR_SYSTEM_SYSTEMROUTEMIGRATION_H
