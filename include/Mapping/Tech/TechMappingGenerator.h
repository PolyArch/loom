#ifndef LOOM_MAPPING_TECH_TECHMAPPINGGENERATOR_H
#define LOOM_MAPPING_TECH_TECHMAPPINGGENERATOR_H

#include "Mapping/Artifact/MappingArtifact.h"
#include "Mapping/Tech/TechMappingConfig.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
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
  std::uint64_t partialCoverExpansions = 0;
  std::uint64_t publicationSlots = 0;

  friend bool operator==(const TechMappingGenerationAccounting &lhs,
                         const TechMappingGenerationAccounting &rhs) {
    return lhs.matchRowAttempts == rhs.matchRowAttempts &&
           lhs.partialCoverExpansions == rhs.partialCoverExpansions &&
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
                 IncompleteTechMappingGeneration, InvalidTechMappingGeneration,
                 InternalTechMappingGeneration>;

struct TechMappingGenerationInputs final {
  const ::dataflow::CanonicalDataflowProgramView &dataflow;
  llvm::ArrayRef<::dataflow::GraphRef> covers;
  const ::loom::fabric::FabricArtifactView &fabric;
  const ResolvedTechMappingConfigView &config;
  const ArtifactStore &store;
};

TechMappingGenerationOutcome
generateTechMappings(const TechMappingGenerationInputs &inputs);

} // namespace loom::mapping

#endif // LOOM_MAPPING_TECH_TECHMAPPINGGENERATOR_H
