#ifndef LOOM_LIB_PNR_FROZENCOMPUTEDOMAINS_H
#define LOOM_LIB_PNR_FROZENCOMPUTEDOMAINS_H

#include "Mapping/Artifact.h"
#include "PnR/FrozenRealizationGraph.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <vector>

namespace loom::mapping {
class ValidatedTechMapping;
} // namespace loom::mapping

namespace loom::pnr::detail {

struct FrozenComputeDomains {
  std::vector<FrozenComputeRealization> realizations;
  std::vector<FrozenComputeOccurrence> occurrences;
  std::vector<mapping::FuId> occurrenceFuMemberships;
  std::vector<FrozenPhysicalEndpoint> endpoints;
  std::vector<mapping::TypeKey> endpointCompatibleTypes;
  std::vector<FrozenComputeLocalArc> localArcs;
  std::vector<FrozenImplementationOccurrence> implementationOccurrences;
  std::vector<FrozenPortDemand> portDemands;
  std::vector<PnrIndex> compatibleEndpoints;
};

llvm::Expected<FrozenComputeDomains> buildFrozenComputeDomains(
    const mapping::FabricHardwareView &fabric,
    const mapping::ValidatedTechMapping &mapping,
    llvm::ArrayRef<const mapping::ComputeRealizationDraft *> realizations);

} // namespace loom::pnr::detail

#endif // LOOM_LIB_PNR_FROZENCOMPUTEDOMAINS_H
