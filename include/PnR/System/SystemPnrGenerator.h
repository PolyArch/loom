#ifndef LOOM_PNR_SYSTEM_SYSTEMPNRGENERATOR_H
#define LOOM_PNR_SYSTEM_SYSTEMPNRGENERATOR_H

#include "Common/ArtifactStore.h"
#include "Common/ExecutionControl.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Fabric/Artifact/FabricSystemRootView.h"
#include "Fabric/Identity/FabricPhysicalTiming.h"
#include "Mapping/Artifact/SystemMappingConstraintSet.h"
#include "PnR/PnrConfig.h"
#include "PnR/PnrGeneration.h"
#include "PnR/System/SystemPnrDerivedContext.h"
#include "PnR/System/SystemPnrSearchDomain.h"

#include "llvm/ADT/StringRef.h"

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace loom::pnr {

struct SystemPnrGenerationAccounting final {
  std::uint64_t seedAttemptSlots = 0;
  std::uint64_t preparedSeeds = 0;
  std::uint64_t initializerAssignmentAttempts = 0;
  std::uint64_t endpointExpansionSlots = 0;
  std::uint64_t negotiationIterationSlots = 0;
  std::uint64_t calibrationProposalSlots = 0;
  std::uint64_t annealingBaseProposalSlots = 0;
  std::uint64_t annealingMovableProposalSlots = 0;
  std::uint64_t annealingAcceptedActions = 0;
  std::uint64_t mutationOracleVerificationAttempts = 0;
  std::uint64_t exactRepairInvocations = 0;
  std::uint64_t exactRepairRegionDecisions = 0;
  std::uint64_t exactRepairSolverCalls = 0;
  std::uint64_t finalClosureAttempts = 0;
  std::uint64_t finalVerificationAttempts = 0;
  std::uint64_t finalizedRestarts = 0;
  std::uint64_t publicationSlots = 0;

  friend bool operator==(const SystemPnrGenerationAccounting &lhs,
                         const SystemPnrGenerationAccounting &rhs) {
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
           lhs.mutationOracleVerificationAttempts ==
               rhs.mutationOracleVerificationAttempts &&
           lhs.exactRepairInvocations == rhs.exactRepairInvocations &&
           lhs.exactRepairRegionDecisions == rhs.exactRepairRegionDecisions &&
           lhs.exactRepairSolverCalls == rhs.exactRepairSolverCalls &&
           lhs.finalClosureAttempts == rhs.finalClosureAttempts &&
           lhs.finalVerificationAttempts == rhs.finalVerificationAttempts &&
           lhs.finalizedRestarts == rhs.finalizedRestarts &&
           lhs.publicationSlots == rhs.publicationSlots;
  }
};

struct GeneratedSystemMappings final {
  std::vector<ArtifactRootReference> candidates;
  PnrGenerationTermination termination;
  SystemPnrGenerationAccounting accounting;
};

struct ProvenInfeasibleSystemMapping final {
  SystemPnrGenerationAccounting accounting;
  std::string diagnostic;
};

enum class IncompleteSystemPnrGenerationReason : std::uint8_t {
  ProofNotEstablished,
  SemanticLimitReached,
};

struct IncompleteSystemPnrGeneration final {
  IncompleteSystemPnrGenerationReason reason;
  SystemPnrGenerationAccounting accounting;
  std::string diagnostic;
};

enum class SystemPnrInterruptionStage : std::uint8_t {
  InputAdmission,
  FrozenModelConstruction,
  CandidateInitialization,
  Annealing,
  FinalClosure,
  CandidateVerification,
  CandidateFinalization,
};

llvm::StringRef
systemPnrInterruptionStageSpelling(SystemPnrInterruptionStage stage);

struct SystemPnrSearchFrontier final {
  std::optional<std::uint32_t> restartOrdinal;
  std::uint64_t seedAttemptSlots = 0;
  std::uint64_t preparedSeeds = 0;
  std::uint64_t initializerAssignmentAttempts = 0;
  std::uint64_t endpointExpansionSlots = 0;
  std::uint64_t negotiationIterationSlots = 0;
  std::uint64_t calibrationProposalSlots = 0;
  std::uint64_t annealingBaseProposalSlots = 0;
  std::uint64_t annealingMovableProposalSlots = 0;
  std::uint64_t mutationOracleVerificationAttempts = 0;
  std::uint64_t finalClosureAttempts = 0;
  std::uint64_t finalVerificationAttempts = 0;
  std::uint64_t finalizedRestarts = 0;
  std::uint64_t publicationSlots = 0;
};

struct SystemPnrClosureResidual final {
  std::optional<
      std::array<std::optional<std::uint64_t>, resolvedPnrViolationKindCount>>
      violationValues;
  std::uint64_t retainedCandidates = 0;
};

struct SystemPnrInterruptionSnapshot final {
  SystemPnrInterruptionStage stage = SystemPnrInterruptionStage::InputAdmission;
  SystemPnrSearchFrontier frontier;
  std::optional<std::vector<std::uint64_t>> bestSelectedRank;
  SystemPnrClosureResidual closureResidual;
  ExecutionResourceStatistics resources;
};

struct InterruptedSystemPnrGeneration final {
  std::vector<ArtifactRootReference> candidates;
  SystemPnrGenerationAccounting accounting;
  SystemPnrInterruptionSnapshot snapshot;
};

enum class InvalidSystemPnrGenerationReason : std::uint8_t {
  FrozenInput,
};

struct InvalidSystemPnrGeneration final {
  InvalidSystemPnrGenerationReason reason;
  SystemPnrGenerationAccounting accounting;
  std::string diagnostic;
};

enum class InternalSystemPnrGenerationReason : std::uint8_t {
  FrozenModelConstruction,
  CandidateInitialization,
  Annealing,
  FinalClosure,
  CandidateVerification,
  CandidateFinalization,
  AccountingOverflow,
};

struct InternalSystemPnrGeneration final {
  InternalSystemPnrGenerationReason reason;
  SystemPnrGenerationAccounting accounting;
  std::string diagnostic;
};

using SystemPnrGenerationOutcome =
    std::variant<GeneratedSystemMappings, ProvenInfeasibleSystemMapping,
                 IncompleteSystemPnrGeneration, InterruptedSystemPnrGeneration,
                 InvalidSystemPnrGeneration, InternalSystemPnrGeneration>;

struct SystemPnrGenerationInputs final {
  const ::dataflow::CanonicalDataflowProgramView &dataflow;
  const ::loom::fabric::FabricSystemRootView &fabric;
  llvm::ArrayRef<::loom::fabric::FabricPhysicalTimingProfileView>
      physicalTimingProfiles;
  const SystemPnrSearchDomainView &searchDomain;
  const ResolvedPnrConfigView &config;
  const ::loom::mapping::FinalizedSystemMappingConstraintSet &constraints;
  const ArtifactStore &store;
  ExecutionControlView executionControl = {};
  const SystemStaticContext *staticContext = nullptr;
  const SystemActiveContext *activeContext = nullptr;
};

/// Runs the canonical System PnR invocation for one exact D/F/R/H/C/K
/// binding. Only an independently verified and finalized
/// SystemMapping may enter the returned candidate set.
SystemPnrGenerationOutcome
generateSystemMappings(const SystemPnrGenerationInputs &inputs);

} // namespace loom::pnr

#endif // LOOM_PNR_SYSTEM_SYSTEMPNRGENERATOR_H
