#ifndef LOOM_MAPPING_TECH_TECHMAPPINGGENERATOR_H
#define LOOM_MAPPING_TECH_TECHMAPPINGGENERATOR_H

#include "Common/ExecutionControl.h"
#include "Mapping/Artifact/MappingArtifact.h"
#include "Mapping/Tech/TechMappingConfig.h"
#include "Mapping/Tech/TechMappingHardwareDemand.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace loom::mapping {

enum class TechMappingGenerationTermination : std::uint8_t {
  SearchExhausted,
  SemanticLimitReached,
};

struct TechMappingGenerationAccounting final {
  std::uint64_t matchRowAttempts = 0;
  std::uint64_t matchRowFirstVisits = 0;
  std::uint64_t matchRowCursorResumptions = 0;
  std::uint64_t matchRowReplayVisits = 0;
  std::uint64_t memoryRowFrontierLimits = 0;
  std::uint64_t partialCoverExpansions = 0;
  std::uint64_t constructiveCoverSearchInvocations = 0;
  std::uint64_t constructiveCoverCompletedChecks = 0;
  std::uint64_t constructiveCoverPublications = 0;
  std::uint64_t computeContextProjectionWork = 0;
  std::uint64_t computeContextMatchingChecks = 0;
  std::uint64_t computeContextRejectedChecks = 0;
  std::uint64_t computeContextMatchingWork = 0;
  std::uint64_t memorySupplyProjectionWork = 0;
  std::uint64_t memorySupplyChecks = 0;
  std::uint64_t memorySupplyPartialChecks = 0;
  std::uint64_t memorySupplyFullChecks = 0;
  std::uint64_t memorySupplyRejectedChecks = 0;
  std::uint64_t memorySupplyEmptyDomainRejections = 0;
  std::uint64_t memorySupplyExclusiveResourceRejections = 0;
  std::uint64_t memorySupplySpatialPortRejections = 0;
  std::uint64_t memorySupplyTemporalIngressRejections = 0;
  std::uint64_t memorySupplyInternalConnectionRejections = 0;
  std::uint64_t memorySupplyResidentCapacityRejections = 0;
  std::uint64_t memorySupplyJointAssignmentRejections = 0;
  std::uint64_t memorySupplySearchWork = 0;
  std::uint64_t candidateEvaluations = 0;
  std::uint64_t publicationSlots = 0;

  friend bool operator==(const TechMappingGenerationAccounting &lhs,
                         const TechMappingGenerationAccounting &rhs) {
    return lhs.matchRowAttempts == rhs.matchRowAttempts &&
           lhs.matchRowFirstVisits == rhs.matchRowFirstVisits &&
           lhs.matchRowCursorResumptions == rhs.matchRowCursorResumptions &&
           lhs.matchRowReplayVisits == rhs.matchRowReplayVisits &&
           lhs.memoryRowFrontierLimits == rhs.memoryRowFrontierLimits &&
           lhs.partialCoverExpansions == rhs.partialCoverExpansions &&
           lhs.constructiveCoverSearchInvocations ==
               rhs.constructiveCoverSearchInvocations &&
           lhs.constructiveCoverCompletedChecks ==
               rhs.constructiveCoverCompletedChecks &&
           lhs.constructiveCoverPublications ==
               rhs.constructiveCoverPublications &&
           lhs.computeContextProjectionWork ==
               rhs.computeContextProjectionWork &&
           lhs.computeContextMatchingChecks ==
               rhs.computeContextMatchingChecks &&
           lhs.computeContextRejectedChecks ==
               rhs.computeContextRejectedChecks &&
           lhs.computeContextMatchingWork == rhs.computeContextMatchingWork &&
           lhs.memorySupplyProjectionWork == rhs.memorySupplyProjectionWork &&
           lhs.memorySupplyChecks == rhs.memorySupplyChecks &&
           lhs.memorySupplyPartialChecks == rhs.memorySupplyPartialChecks &&
           lhs.memorySupplyFullChecks == rhs.memorySupplyFullChecks &&
           lhs.memorySupplyRejectedChecks == rhs.memorySupplyRejectedChecks &&
           lhs.memorySupplyEmptyDomainRejections ==
               rhs.memorySupplyEmptyDomainRejections &&
           lhs.memorySupplyExclusiveResourceRejections ==
               rhs.memorySupplyExclusiveResourceRejections &&
           lhs.memorySupplySpatialPortRejections ==
               rhs.memorySupplySpatialPortRejections &&
           lhs.memorySupplyTemporalIngressRejections ==
               rhs.memorySupplyTemporalIngressRejections &&
           lhs.memorySupplyInternalConnectionRejections ==
               rhs.memorySupplyInternalConnectionRejections &&
           lhs.memorySupplyResidentCapacityRejections ==
               rhs.memorySupplyResidentCapacityRejections &&
           lhs.memorySupplyJointAssignmentRejections ==
               rhs.memorySupplyJointAssignmentRejections &&
           lhs.memorySupplySearchWork == rhs.memorySupplySearchWork &&
           lhs.candidateEvaluations == rhs.candidateEvaluations &&
           lhs.publicationSlots == rhs.publicationSlots;
  }
};

struct TechMappingGenerationFeedback final {
  std::optional<TechMappingComputeContextHallDeficit> computeContextHall;
};

struct GeneratedTechMappings final {
  std::vector<ArtifactRootReference> candidates;
  TechMappingGenerationTermination termination;
  TechMappingGenerationAccounting accounting;
  TechMappingGenerationFeedback feedback = {};
};

struct ProvenInfeasibleTechMapping final {
  TechMappingGenerationAccounting accounting;
  TechMappingGenerationFeedback feedback = {};
};

enum class IncompleteTechMappingGenerationReason : std::uint8_t {
  ProofNotEstablished,
};

struct IncompleteTechMappingGeneration final {
  IncompleteTechMappingGenerationReason reason;
  TechMappingGenerationAccounting accounting;
  TechMappingGenerationFeedback feedback = {};
};

enum class TechMappingInterruptionStage : std::uint8_t {
  InputAdmission,
  MatchRowDerivation,
  CoverSearch,
  CandidateFinalization,
};

struct TechMappingSearchFrontier final {
  std::uint64_t matchRowAttempts = 0;
  std::uint64_t partialCoverExpansions = 0;
  std::uint64_t candidateEvaluations = 0;
  std::uint64_t publicationSlots = 0;
};

struct TechMappingClosureResidual final {
  std::uint64_t uncoveredActors = 0;
  std::uint64_t retainedCandidates = 0;
};

struct TechMappingInterruptionSnapshot final {
  TechMappingInterruptionStage stage =
      TechMappingInterruptionStage::InputAdmission;
  TechMappingSearchFrontier frontier;
  std::optional<std::uint64_t> bestCanonicalRank;
  TechMappingClosureResidual closureResidual;
  ExecutionResourceStatistics resources;
};

llvm::StringRef
techMappingInterruptionStageSpelling(TechMappingInterruptionStage stage);

struct InterruptedTechMappingGeneration final {
  std::vector<ArtifactRootReference> candidates;
  TechMappingGenerationAccounting accounting;
  TechMappingInterruptionSnapshot snapshot;
  TechMappingGenerationFeedback feedback = {};
};

enum class InvalidTechMappingGenerationReason : std::uint8_t {
  EmptyGraphCover,
  ForeignGraphReference,
  UnresolvedGraphReference,
  NonCanonicalGraphCover,
  GraphCoverHasNoActors,
};

struct InvalidTechMappingGeneration final {
  InvalidTechMappingGenerationReason reason;
  TechMappingGenerationAccounting accounting;
  std::string diagnostic;
  TechMappingGenerationFeedback feedback = {};
};

enum class InternalTechMappingGenerationReason : std::uint8_t {
  MatchRowDerivationFailed,
  CandidateFinalizationFailed,
};

struct InternalTechMappingGeneration final {
  InternalTechMappingGenerationReason reason;
  TechMappingGenerationAccounting accounting;
  std::string diagnostic;
  TechMappingGenerationFeedback feedback = {};
};

using TechMappingGenerationOutcome =
    std::variant<GeneratedTechMappings, ProvenInfeasibleTechMapping,
                 IncompleteTechMappingGeneration,
                 InterruptedTechMappingGeneration, InvalidTechMappingGeneration,
                 InternalTechMappingGeneration>;

struct TechMappingGenerationInputs final {
  const ::dataflow::CanonicalDataflowProgramView &dataflow;
  llvm::ArrayRef<::dataflow::GraphRef> covers;
  const ::loom::fabric::FabricArtifactView &fabric;
  const ResolvedTechMappingConfigView &config;
  const ArtifactStore &store;
  ExecutionControlView executionControl = {};
};

TechMappingGenerationOutcome
generateTechMappings(const TechMappingGenerationInputs &inputs);

enum class TechMappingCandidateEnumerationControl : std::uint8_t {
  Continue,
  Stop,
};

struct TechMappingCandidateEnumerationResult final {
  TechMappingGenerationTermination termination;
  TechMappingGenerationAccounting accounting;
  std::uint64_t visitedCandidates = 0;
  std::optional<TechMappingInterruptionSnapshot> interruption;
  TechMappingGenerationFeedback feedback = {};
};

/// Materializes and visits the canonical TechMapping stream until its domain,
/// semantic evaluation bound, or the invocation-local visitor stops it. A
/// caller may use a rejected exact candidate as an ephemeral no-good and ask
/// for the next realization; no feedback state is persisted in Mapping.
llvm::Expected<TechMappingCandidateEnumerationResult>
enumerateTechMappingCandidates(
    const TechMappingGenerationInputs &inputs,
    llvm::function_ref<llvm::Expected<TechMappingCandidateEnumerationControl>(
        const ArtifactRootReference &)>
        visitor);

} // namespace loom::mapping

#endif // LOOM_MAPPING_TECH_TECHMAPPINGGENERATOR_H
