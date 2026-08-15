#ifndef LOOM_MAPPING_TECH_TECHMAPPINGGENERATOR_H
#define LOOM_MAPPING_TECH_TECHMAPPINGGENERATOR_H

#include "Common/ExecutionControl.h"
#include "Mapping/Artifact/MappingArtifact.h"
#include "Mapping/Tech/TechMappingConfig.h"

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
  std::uint64_t partialCoverExpansions = 0;
  std::uint64_t candidateEvaluations = 0;
  std::uint64_t publicationSlots = 0;

  friend bool operator==(const TechMappingGenerationAccounting &lhs,
                         const TechMappingGenerationAccounting &rhs) {
    return lhs.matchRowAttempts == rhs.matchRowAttempts &&
           lhs.matchRowFirstVisits == rhs.matchRowFirstVisits &&
           lhs.matchRowCursorResumptions == rhs.matchRowCursorResumptions &&
           lhs.matchRowReplayVisits == rhs.matchRowReplayVisits &&
           lhs.partialCoverExpansions == rhs.partialCoverExpansions &&
           lhs.candidateEvaluations == rhs.candidateEvaluations &&
           lhs.publicationSlots == rhs.publicationSlots;
  }
};

struct GeneratedTechMappings final {
  std::vector<ArtifactRootReference> candidates;
  TechMappingGenerationTermination termination;
  TechMappingGenerationAccounting accounting;
};

struct ProvenInfeasibleTechMapping final {
  TechMappingGenerationAccounting accounting;
};

enum class IncompleteTechMappingGenerationReason : std::uint8_t {
  ProofNotEstablished,
};

struct IncompleteTechMappingGeneration final {
  IncompleteTechMappingGenerationReason reason;
  TechMappingGenerationAccounting accounting;
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
};

enum class InternalTechMappingGenerationReason : std::uint8_t {
  MatchRowDerivationFailed,
  CandidateFinalizationFailed,
};

struct InternalTechMappingGeneration final {
  InternalTechMappingGenerationReason reason;
  TechMappingGenerationAccounting accounting;
  std::string diagnostic;
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
