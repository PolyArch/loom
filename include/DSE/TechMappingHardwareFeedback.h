#ifndef LOOM_DSE_TECHMAPPINGHARDWAREFEEDBACK_H
#define LOOM_DSE_TECHMAPPINGHARDWAREFEEDBACK_H

#include "DSE/HardwareDecision.h"
#include "Mapping/Tech/TechMappingHardwareDemand.h"

#include "llvm/Support/Error.h"

#include <vector>

namespace loom::fabric {
class FabricArtifactView;
}

namespace loom::dse {

/// Projects the minimal existing hardware action family that can add supply
/// to an observed compute-context Hall relation. Each domain changes one
/// Temporal PE and offers the smallest increment plus the complete observed
/// deficit when those differ. Mapping and the ordinary hardware generator
/// remain the legality and materialization owners.
llvm::Expected<std::vector<SpatialMicroarchitectureDecisionDomain>>
projectTechMappingComputeContextGrowthDomains(
    const mapping::TechMappingComputeContextHallDeficit &feedback,
    const fabric::FabricArtifactView &module);

} // namespace loom::dse

#endif // LOOM_DSE_TECHMAPPINGHARDWAREFEEDBACK_H
