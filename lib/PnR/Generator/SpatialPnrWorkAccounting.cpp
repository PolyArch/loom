#include "SpatialPnrWorkAccounting.h"

#include "llvm/Support/Error.h"

#include <array>
#include <cstddef>
#include <limits>
#include <system_error>
#include <utility>

using namespace loom::pnr;

llvm::Error loom::pnr::detail::checkedAdd(std::uint64_t amount,
                                          std::uint64_t &target,
                                          llvm::StringRef subject) {
  if (amount > std::numeric_limits<std::uint64_t>::max() - target)
    return llvm::createStringError(
        std::make_error_code(std::errc::value_too_large),
        "Spatial PnR accounting overflow: " + subject);
  target += amount;
  return llvm::Error::success();
}

SpatialPnrWorkLedgerView loom::pnr::detail::canonicalWorkLedger(
    SpatialPnrGenerationAccounting &accounting) {
  std::array<SpatialPnrWorkCounterRef, spatialPnrWorkKindCount> counters{};
  const auto bind = [&](SpatialPnrWorkKind kind, std::uint64_t &planned,
                        std::uint64_t &consumed) {
    counters[static_cast<std::size_t>(kind)] = {&planned, &consumed};
  };
  bind(SpatialPnrWorkKind::SeedAttempt, accounting.plannedSeedAttemptSlots,
       accounting.seedAttemptSlots);
  bind(SpatialPnrWorkKind::InitializerAssignment,
       accounting.plannedInitializerAssignmentAttempts,
       accounting.initializerAssignmentAttempts);
  bind(SpatialPnrWorkKind::EndpointExpansion,
       accounting.plannedEndpointExpansionSlots,
       accounting.endpointExpansionSlots);
  bind(SpatialPnrWorkKind::NegotiationIteration,
       accounting.plannedNegotiationIterationSlots,
       accounting.negotiationIterationSlots);
  bind(SpatialPnrWorkKind::CalibrationProposal,
       accounting.plannedCalibrationProposalSlots,
       accounting.calibrationProposalSlots);
  bind(SpatialPnrWorkKind::AnnealingBaseProposal,
       accounting.plannedAnnealingBaseProposalSlots,
       accounting.annealingBaseProposalSlots);
  bind(SpatialPnrWorkKind::AnnealingMovableProposal,
       accounting.plannedAnnealingMovableProposalSlots,
       accounting.annealingMovableProposalSlots);
  bind(SpatialPnrWorkKind::ExactRepairRegionDecision,
       accounting.plannedExactRepairRegionDecisions,
       accounting.exactRepairRegionDecisions);
  bind(SpatialPnrWorkKind::ExactRepairSolverCall,
       accounting.plannedExactRepairSolverCalls,
       accounting.exactRepairSolverCalls);
  bind(SpatialPnrWorkKind::LocalTransferAdoptionProbe,
       accounting.plannedLocalTransferAdoptionProbes,
       accounting.localTransferAdoptionProbes);
  bind(SpatialPnrWorkKind::FinalClosureAttempt,
       accounting.plannedFinalClosureAttempts, accounting.finalClosureAttempts);
  return SpatialPnrWorkLedgerView(counters);
}

llvm::Error loom::pnr::detail::accumulateRestartAccounting(
    const SpatialPnrGenerationAccounting &source,
    SpatialPnrGenerationAccounting &target) {
#define LOOM_ACCUMULATE_SPATIAL_FIELD(Field, Label)                            \
  if (llvm::Error error = checkedAdd(source.Field, target.Field, Label))       \
  return error
  LOOM_ACCUMULATE_SPATIAL_FIELD(plannedInitializerAssignmentAttempts,
                                "planned initializer assignment attempts");
  LOOM_ACCUMULATE_SPATIAL_FIELD(plannedEndpointExpansionSlots,
                                "planned endpoint expansion slots");
  LOOM_ACCUMULATE_SPATIAL_FIELD(plannedNegotiationIterationSlots,
                                "planned negotiation iteration slots");
  LOOM_ACCUMULATE_SPATIAL_FIELD(plannedCalibrationProposalSlots,
                                "planned calibration proposal slots");
  LOOM_ACCUMULATE_SPATIAL_FIELD(plannedAnnealingBaseProposalSlots,
                                "planned base annealing proposal slots");
  LOOM_ACCUMULATE_SPATIAL_FIELD(plannedAnnealingMovableProposalSlots,
                                "planned movable annealing proposal slots");
  LOOM_ACCUMULATE_SPATIAL_FIELD(plannedExactRepairRegionDecisions,
                                "planned exact repair region decisions");
  LOOM_ACCUMULATE_SPATIAL_FIELD(plannedExactRepairSolverCalls,
                                "planned exact repair solver calls");
  LOOM_ACCUMULATE_SPATIAL_FIELD(plannedLocalTransferAdoptionProbes,
                                "planned local transfer adoption probes");
  LOOM_ACCUMULATE_SPATIAL_FIELD(plannedFinalClosureAttempts,
                                "planned final closure attempts");
  LOOM_ACCUMULATE_SPATIAL_FIELD(plannedSeedAttemptSlots,
                                "planned seed attempt slots");
  LOOM_ACCUMULATE_SPATIAL_FIELD(seedAttemptSlots, "seed attempt slots");
  LOOM_ACCUMULATE_SPATIAL_FIELD(preparedSeeds, "prepared seeds");
  LOOM_ACCUMULATE_SPATIAL_FIELD(initializerAssignmentAttempts,
                                "initializer assignment attempts");
  LOOM_ACCUMULATE_SPATIAL_FIELD(endpointExpansionSlots,
                                "endpoint expansion slots");
  LOOM_ACCUMULATE_SPATIAL_FIELD(negotiationIterationSlots,
                                "negotiation iteration slots");
  LOOM_ACCUMULATE_SPATIAL_FIELD(calibrationProposalSlots,
                                "calibration proposal slots");
  LOOM_ACCUMULATE_SPATIAL_FIELD(annealingBaseProposalSlots,
                                "base annealing proposal slots");
  LOOM_ACCUMULATE_SPATIAL_FIELD(annealingMovableProposalSlots,
                                "movable annealing proposal slots");
  LOOM_ACCUMULATE_SPATIAL_FIELD(annealingAcceptedActions,
                                "annealing accepted Actions");
  LOOM_ACCUMULATE_SPATIAL_FIELD(exactRepairInvocations,
                                "exact repair invocations");
  LOOM_ACCUMULATE_SPATIAL_FIELD(exactRepairRegionDecisions,
                                "exact repair region decisions");
  LOOM_ACCUMULATE_SPATIAL_FIELD(exactRepairSolverCalls,
                                "exact repair solver calls");
  LOOM_ACCUMULATE_SPATIAL_FIELD(localTransferAdoptionProbes,
                                "local transfer adoption probes");
  LOOM_ACCUMULATE_SPATIAL_FIELD(adoptedLocalTransfers,
                                "adopted local transfers");
  LOOM_ACCUMULATE_SPATIAL_FIELD(finalClosureAttempts, "final closure attempts");
  LOOM_ACCUMULATE_SPATIAL_FIELD(finalizedRestarts, "finalized restarts");
  LOOM_ACCUMULATE_SPATIAL_FIELD(publicationSlots, "publication slots");
#undef LOOM_ACCUMULATE_SPATIAL_FIELD
  return llvm::Error::success();
}

llvm::Error loom::pnr::verifySpatialPnrWorkAccounting(
    const SpatialPnrGenerationAccounting &accounting, bool requireClosedWork) {
  const std::array<std::pair<std::uint64_t, std::uint64_t>, 11> counters = {{
      {accounting.plannedSeedAttemptSlots, accounting.seedAttemptSlots},
      {accounting.plannedInitializerAssignmentAttempts,
       accounting.initializerAssignmentAttempts},
      {accounting.plannedEndpointExpansionSlots,
       accounting.endpointExpansionSlots},
      {accounting.plannedNegotiationIterationSlots,
       accounting.negotiationIterationSlots},
      {accounting.plannedCalibrationProposalSlots,
       accounting.calibrationProposalSlots},
      {accounting.plannedAnnealingBaseProposalSlots,
       accounting.annealingBaseProposalSlots},
      {accounting.plannedAnnealingMovableProposalSlots,
       accounting.annealingMovableProposalSlots},
      {accounting.plannedExactRepairRegionDecisions,
       accounting.exactRepairRegionDecisions},
      {accounting.plannedExactRepairSolverCalls,
       accounting.exactRepairSolverCalls},
      {accounting.plannedLocalTransferAdoptionProbes,
       accounting.localTransferAdoptionProbes},
      {accounting.plannedFinalClosureAttempts, accounting.finalClosureAttempts},
  }};
  for (const auto [planned, consumed] : counters) {
    if (consumed > planned)
      return llvm::createStringError(
          std::make_error_code(std::errc::invalid_argument),
          "Spatial PnR consumed work exceeds planned work");
    if (requireClosedWork && planned != consumed)
      return llvm::createStringError(
          std::make_error_code(std::errc::invalid_argument),
          "Spatial PnR completed with admitted work still live");
  }
  return llvm::Error::success();
}
