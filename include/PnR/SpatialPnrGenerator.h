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
#include "PnR/PnrGeneration.h"

#include "llvm/ADT/StringRef.h"

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace loom::pnr {

struct SpatialPnrGenerationAccounting final {
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
    return lhs.seedAttemptSlots == rhs.seedAttemptSlots &&
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

struct ProvenInfeasibleSpatialMapping final {
  SpatialPnrGenerationAccounting accounting;
  std::string diagnostic;
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
};

/// Runs the fixed canonical Spatial restart sequence for one exact D/T/F/C/K
/// invocation. Only independently finalized Mapping references enter the
/// returned canonical candidate set; mutable search state is never exposed.
SpatialPnrGenerationOutcome
generateSpatialMappings(const SpatialPnrGenerationInputs &inputs);

} // namespace loom::pnr

#endif // LOOM_PNR_SPATIALPNRGENERATOR_H
