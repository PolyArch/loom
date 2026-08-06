#ifndef LOOM_DSE_MAPPINGCANDIDATEGENERATOR_H
#define LOOM_DSE_MAPPINGCANDIDATEGENERATOR_H

#include "DSE/CandidateGenerator.h"
#include "PnR/SpatialPnrGenerator.h"

#include <array>

namespace loom::dse {

inline constexpr CandidateGeneratorKind spatialPnrCandidateGeneratorKind(0);

inline constexpr std::array<CandidateGeneratorWorkUnitDescriptor, 10>
    spatialPnrCandidateGeneratorWorkUnits = {{
        {CandidateGeneratorWorkUnitRef(0), "seed_attempt"},
        {CandidateGeneratorWorkUnitRef(1), "assignment_attempt_per_seed"},
        {CandidateGeneratorWorkUnitRef(2), "endpoint_expansion"},
        {CandidateGeneratorWorkUnitRef(3), "negotiation_iteration"},
        {CandidateGeneratorWorkUnitRef(4), "calibration_proposal"},
        {CandidateGeneratorWorkUnitRef(5), "proposal_per_level_base"},
        {CandidateGeneratorWorkUnitRef(6), "proposal_per_movable_decision"},
        {CandidateGeneratorWorkUnitRef(7), "focused_closure_proposal"},
        {CandidateGeneratorWorkUnitRef(8), "exact_repair_region_decision"},
        {CandidateGeneratorWorkUnitRef(9), "exact_repair_solver_call"},
    }};

const CandidateGeneratorDescriptor &spatialPnrCandidateGeneratorDescriptor();
llvm::Error registerSpatialPnrCandidateGenerator();

llvm::Expected<std::vector<CandidateGeneratorInputBinding>>
bindSpatialPnrCandidateGeneratorInputs(
    const ArtifactRootReference &dataflow,
    const ArtifactRootReference &techMapping,
    const ArtifactRootReference &fabric,
    const ArtifactRootReference &constraints);

llvm::Expected<ResolvedCandidateGeneratorBinding>
resolveSpatialPnrCandidateGeneratorBinding(
    const ::loom::pnr::ResolvedPnrConfigView &config);

std::vector<CandidateGeneratorWorkUnitSummary>
spatialPnrCandidateGeneratorWorkSummary(
    const ::loom::pnr::SpatialPnrGenerationAccounting &accounting);

/// Strictly imports one exact D/T/F/C/K binding and invokes the Spatial PnR
/// owner. Import or coupling failures are returned through the owner's Invalid
/// outcome; no partial candidate set is exposed.
::loom::pnr::SpatialPnrGenerationOutcome invokeSpatialPnrCandidateGenerator(
    llvm::ArrayRef<CandidateGeneratorInputBinding> inputBindings,
    const ResolvedCandidateGeneratorBinding &binding,
    const ArtifactStore &store);

} // namespace loom::dse

#endif // LOOM_DSE_MAPPINGCANDIDATEGENERATOR_H
