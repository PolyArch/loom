#ifndef LOOM_PNR_SPATIALPNRGENERATOR_H
#define LOOM_PNR_SPATIALPNRGENERATOR_H

#include "Common/ArtifactStore.h"
#include "Common/ExecutionControl.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Fabric/Identity/FabricPhysicalTiming.h"
#include "Fabric/Identity/FabricRefImport.h"
#include "Mapping/Artifact/MappingArtifact.h"
#include "Mapping/Artifact/MappingConstraintSet.h"
#include "PnR/PnrConfig.h"
#include "PnR/PnrDerivedContext.h"
#include "PnR/PnrGeneration.h"
#include "PnR/SpatialCanonicalSeed.h"
#include "PnR/SpatialPnrProblem.h"

#include "llvm/ADT/StringRef.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace loom::fabric {
struct FabricTopologyQualityReport;
}

namespace loom::pnr {

struct SpatialPnrGenerationAccounting final {
  std::uint64_t plannedSeedAttemptSlots = 0;
  std::uint64_t plannedInitializerAssignmentAttempts = 0;
  std::uint64_t plannedEndpointExpansionSlots = 0;
  std::uint64_t plannedNegotiationIterationSlots = 0;
  std::uint64_t plannedCalibrationProposalSlots = 0;
  std::uint64_t plannedAnnealingBaseProposalSlots = 0;
  std::uint64_t plannedAnnealingMovableProposalSlots = 0;
  std::uint64_t plannedExactRepairRegionDecisions = 0;
  std::uint64_t plannedExactRepairSolverCalls = 0;
  std::uint64_t seedAttemptSlots = 0;
  std::uint64_t preparedSeeds = 0;
  std::uint64_t initializerAssignmentAttempts = 0;
  std::uint64_t endpointExpansionSlots = 0;
  std::uint64_t negotiationIterationSlots = 0;
  std::uint64_t calibrationProposalSlots = 0;
  std::uint64_t annealingBaseProposalSlots = 0;
  std::uint64_t annealingMovableProposalSlots = 0;
  std::uint64_t annealingAcceptedActions = 0;
  std::uint64_t exactRepairInvocations = 0;
  std::uint64_t exactRepairRegionDecisions = 0;
  std::uint64_t exactRepairSolverCalls = 0;
  std::uint64_t finalClosureAttempts = 0;
  std::uint64_t finalizedRestarts = 0;
  std::uint64_t publicationSlots = 0;

  friend bool operator==(const SpatialPnrGenerationAccounting &lhs,
                         const SpatialPnrGenerationAccounting &rhs) {
    return lhs.plannedSeedAttemptSlots == rhs.plannedSeedAttemptSlots &&
           lhs.plannedInitializerAssignmentAttempts ==
               rhs.plannedInitializerAssignmentAttempts &&
           lhs.plannedEndpointExpansionSlots ==
               rhs.plannedEndpointExpansionSlots &&
           lhs.plannedNegotiationIterationSlots ==
               rhs.plannedNegotiationIterationSlots &&
           lhs.plannedCalibrationProposalSlots ==
               rhs.plannedCalibrationProposalSlots &&
           lhs.plannedAnnealingBaseProposalSlots ==
               rhs.plannedAnnealingBaseProposalSlots &&
           lhs.plannedAnnealingMovableProposalSlots ==
               rhs.plannedAnnealingMovableProposalSlots &&
           lhs.plannedExactRepairRegionDecisions ==
               rhs.plannedExactRepairRegionDecisions &&
           lhs.plannedExactRepairSolverCalls ==
               rhs.plannedExactRepairSolverCalls &&
           lhs.seedAttemptSlots == rhs.seedAttemptSlots &&
           lhs.preparedSeeds == rhs.preparedSeeds &&
           lhs.initializerAssignmentAttempts ==
               rhs.initializerAssignmentAttempts &&
           lhs.endpointExpansionSlots == rhs.endpointExpansionSlots &&
           lhs.negotiationIterationSlots == rhs.negotiationIterationSlots &&
           lhs.calibrationProposalSlots == rhs.calibrationProposalSlots &&
           lhs.annealingBaseProposalSlots == rhs.annealingBaseProposalSlots &&
           lhs.annealingMovableProposalSlots ==
               rhs.annealingMovableProposalSlots &&
           lhs.annealingAcceptedActions == rhs.annealingAcceptedActions &&
           lhs.exactRepairInvocations == rhs.exactRepairInvocations &&
           lhs.exactRepairRegionDecisions == rhs.exactRepairRegionDecisions &&
           lhs.exactRepairSolverCalls == rhs.exactRepairSolverCalls &&
           lhs.finalClosureAttempts == rhs.finalClosureAttempts &&
           lhs.finalizedRestarts == rhs.finalizedRestarts &&
           lhs.publicationSlots == rhs.publicationSlots;
  }
};

struct GeneratedSpatialMappings final {
  std::vector<ArtifactRootReference> candidates;
  PnrGenerationTermination termination;
  SpatialPnrGenerationAccounting accounting;
};

/// Exact graph-boundary all-different deficit observed before Spatial search.
/// Input and output counts remain separate because one bidirectional Fabric
/// gateway contributes one value to each directional domain.
struct SpatialGraphBoundaryEndpointHallDeficit final {
  std::uint64_t inputDemandCount = 0;
  std::uint64_t inputEndpointCount = 0;
  std::uint64_t outputDemandCount = 0;
  std::uint64_t outputEndpointCount = 0;

  std::uint64_t requiredBoundaryPairs() const {
    const std::uint64_t inputDeficit =
        inputDemandCount > inputEndpointCount
            ? inputDemandCount - inputEndpointCount
            : 0;
    const std::uint64_t outputDeficit =
        outputDemandCount > outputEndpointCount
            ? outputDemandCount - outputEndpointCount
            : 0;
    return std::max(inputDeficit, outputDeficit);
  }
};

enum class SpatialPnrInfeasibilityProofKind : std::uint8_t {
  FrozenDerivedContext = 0,
  FrozenActiveProblem = 1,
  InitializerRelation = 2,
  GraphBoundaryEndpointHall = 3,
};

struct ProvenInfeasibleSpatialMapping final {
  SpatialPnrGenerationAccounting accounting;
  std::string diagnostic;
  std::optional<SpatialGraphBoundaryEndpointHallDeficit>
      graphBoundaryEndpointHall = std::nullopt;
  SpatialPnrInfeasibilityProofKind proofKind;
};

enum class IncompleteSpatialPnrGenerationReason : std::uint8_t {
  ProofNotEstablished,
  NoPreparedSeed,
  SemanticLimitReached,
};

struct IncompleteSpatialPnrGeneration final {
  IncompleteSpatialPnrGenerationReason reason;
  SpatialPnrGenerationAccounting accounting;
  std::string diagnostic;
};

enum class SpatialPnrInterruptionStage : std::uint8_t {
  InputAdmission,
  FrozenModelConstruction,
  SeedConstruction,
  Annealing,
  ExactRepair,
  FinalClosure,
  CandidateVerification,
  CandidateFinalization,
};

llvm::StringRef
spatialPnrInterruptionStageSpelling(SpatialPnrInterruptionStage stage);

struct SpatialPnrSearchFrontier final {
  std::optional<std::uint32_t> restartOrdinal;
  std::uint64_t seedAttemptSlots = 0;
  std::uint64_t preparedSeeds = 0;
  std::uint64_t initializerAssignmentAttempts = 0;
  std::uint64_t endpointExpansionSlots = 0;
  std::uint64_t negotiationIterationSlots = 0;
  std::uint64_t calibrationProposalSlots = 0;
  std::uint64_t annealingBaseProposalSlots = 0;
  std::uint64_t annealingMovableProposalSlots = 0;
  std::uint64_t exactRepairSolverCalls = 0;
  std::uint64_t finalClosureAttempts = 0;
  std::uint64_t finalizedRestarts = 0;
  std::uint64_t publicationSlots = 0;
};

struct SpatialPnrClosureResidual final {
  std::optional<
      std::array<std::optional<std::uint64_t>, resolvedPnrViolationKindCount>>
      violationValues;
  std::uint64_t retainedCandidates = 0;
};

struct SpatialPnrInterruptionSnapshot final {
  SpatialPnrInterruptionStage stage =
      SpatialPnrInterruptionStage::InputAdmission;
  SpatialPnrSearchFrontier frontier;
  std::optional<std::vector<std::uint64_t>> bestSelectedRank;
  SpatialPnrClosureResidual closureResidual;
  ExecutionResourceStatistics resources;
};

struct InterruptedSpatialPnrGeneration final {
  std::vector<ArtifactRootReference> candidates;
  SpatialPnrGenerationAccounting accounting;
  SpatialPnrInterruptionSnapshot snapshot;
};

enum class UnsupportedSpatialPnrGenerationReason : std::uint8_t {
  RoutingNegotiation,
  ExactRepairCapability,
};

struct UnsupportedSpatialPnrGeneration final {
  UnsupportedSpatialPnrGenerationReason reason;
  SpatialPnrGenerationAccounting accounting;
  std::string diagnostic;
};

enum class InvalidSpatialPnrGenerationReason : std::uint8_t {
  FrozenInput,
};

struct InvalidSpatialPnrGeneration final {
  InvalidSpatialPnrGenerationReason reason;
  SpatialPnrGenerationAccounting accounting;
  std::string diagnostic;
};

enum class InternalSpatialPnrGenerationReason : std::uint8_t {
  FrozenModelConstruction,
  SeedConstruction,
  Annealing,
  ExactRepair,
  FinalClosure,
  CandidateVerification,
  CandidateFinalization,
  AccountingOverflow,
};

struct InternalSpatialPnrGeneration final {
  InternalSpatialPnrGenerationReason reason;
  SpatialPnrGenerationAccounting accounting;
  std::string diagnostic;
};

using SpatialPnrGenerationOutcome =
    std::variant<GeneratedSpatialMappings, ProvenInfeasibleSpatialMapping,
                 IncompleteSpatialPnrGeneration,
                 InterruptedSpatialPnrGeneration,
                 UnsupportedSpatialPnrGeneration, InvalidSpatialPnrGeneration,
                 InternalSpatialPnrGeneration>;

struct SpatialPnrGenerationInputs final {
  const ::dataflow::CanonicalDataflowProgramView &dataflow;
  const ::loom::mapping::TechMappingView &techMapping;
  const ::loom::fabric::FabricArtifactView &fabric;
  const ::loom::fabric::FabricPhysicalTimingProfileView &physicalTiming;
  const ResolvedPnrConfigView &config;
  const ::loom::mapping::SpatialMappingConstraintSetView &constraints;
  const ArtifactStore &store;
  /// Invocation-local execution limit. It changes only physical scheduling;
  /// every fixed restart slot and its canonical reduction remain unchanged.
  std::uint32_t candidateWorkerCount = 1;
  ExecutionControlView executionControl = {};
  const FabricDerivedContextBundle *derivedContexts = nullptr;
  const ::loom::fabric::FabricTopologyQualityReport *topologyQualityDiagnostic =
      nullptr;
  FrozenSpatialPnrProblemHandle preparedActiveProblem = nullptr;
  bool emitTopologyQualityDiagnostic = true;
  /// Plan-derived maximum number of canonical candidate restarts to publish.
  /// Every restart required by ExhaustConfiguredWork still executes. This
  /// publication projection is not part of Mapping identity or legality.
  std::optional<std::uint64_t> maximumCandidatePublications = std::nullopt;
  ExecutionResourceBudget executionBudget = {};
  /// Optional one-shot ordinal-zero seed prepared by a deterministic ranking
  /// pass. The formal owner consumes the handoff and accounts its seed work.
  SpatialPathFinderSeedHandoffHandle preparedCanonicalSeed = nullptr;
};

/// Runs the fixed canonical Spatial restart sequence for one exact D/T/F/C/K
/// invocation. Only independently finalized Mapping references enter the
/// returned canonical candidate set; mutable search state is never exposed.
SpatialPnrGenerationOutcome
generateSpatialMappings(const SpatialPnrGenerationInputs &inputs);

} // namespace loom::pnr

#endif // LOOM_PNR_SPATIALPNRGENERATOR_H
