#ifndef LOOM_LIB_PNR_FROZENMEMORYDOMAINS_H
#define LOOM_LIB_PNR_FROZENMEMORYDOMAINS_H

#include "Mapping/Artifact.h"
#include "PnR/FrozenRealizationGraph.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <vector>

namespace loom::mapping {
class ValidatedTechMapping;
} // namespace loom::mapping

namespace loom::pnr::detail {

struct FrozenMemoryDomains {
  std::vector<FrozenMemoryRealization> realizations;
  std::vector<FrozenFabricMemoryOccurrence> occurrences;
  std::vector<FrozenMemoryPhysicalEndpoint> endpoints;
  std::vector<mapping::TypeKey> endpointCompatibleTypes;
  std::vector<FrozenMemoryLocalArc> localArcs;
  std::vector<FrozenMemoryImplementationOccurrence> implementationOccurrences;
  std::vector<FrozenMemoryPortDemand> portDemands;
  std::vector<PnrIndex> compatibleEndpoints;
};

llvm::Expected<FrozenMemoryDomains> buildFrozenMemoryDomains(
    const mapping::FabricHardwareView &fabric,
    const mapping::ValidatedTechMapping &mapping,
    llvm::ArrayRef<const mapping::MemoryRealizationDraft *> realizations);

} // namespace loom::pnr::detail

#endif // LOOM_LIB_PNR_FROZENMEMORYDOMAINS_H
