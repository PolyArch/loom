#ifndef LOOM_PNR_SYSTEM_SYSTEMPNRGENERATOR_H
#define LOOM_PNR_SYSTEM_SYSTEMPNRGENERATOR_H

#include "Common/ArtifactStore.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Fabric/Artifact/FabricSystemRootView.h"
#include "Mapping/Artifact/SystemMappingConstraintSet.h"
#include "PnR/PnrConfig.h"
#include "PnR/System/SystemPnrSearchDomain.h"

#include <cstdint>
#include <string>
#include <variant>
#include <vector>

namespace loom::pnr {

struct SystemPnrGenerationAccounting final {
  std::uint64_t initializerAssignmentAttempts = 0;
  std::uint64_t endpointExpansionSlots = 0;
  std::uint64_t finalVerificationAttempts = 0;
  std::uint64_t publicationSlots = 0;

  friend bool operator==(const SystemPnrGenerationAccounting &lhs,
                         const SystemPnrGenerationAccounting &rhs) {
    return lhs.initializerAssignmentAttempts ==
               rhs.initializerAssignmentAttempts &&
           lhs.endpointExpansionSlots == rhs.endpointExpansionSlots &&
           lhs.finalVerificationAttempts == rhs.finalVerificationAttempts &&
           lhs.publicationSlots == rhs.publicationSlots;
  }
};

struct GeneratedSystemMappings final {
  std::vector<ArtifactRootReference> candidates;
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
  CandidateVerification,
  CandidateFinalization,
};

struct InternalSystemPnrGeneration final {
  InternalSystemPnrGenerationReason reason;
  SystemPnrGenerationAccounting accounting;
  std::string diagnostic;
};

using SystemPnrGenerationOutcome =
    std::variant<GeneratedSystemMappings, ProvenInfeasibleSystemMapping,
                 IncompleteSystemPnrGeneration, InvalidSystemPnrGeneration,
                 InternalSystemPnrGeneration>;

struct SystemPnrGenerationInputs final {
  const ::dataflow::CanonicalDataflowProgramView &dataflow;
  const ::loom::fabric::FabricSystemRootView &fabric;
  const SystemPnrSearchDomainView &searchDomain;
  const ResolvedPnrConfigView &config;
  const ::loom::mapping::FinalizedSystemMappingConstraintSet &constraints;
  const ArtifactStore &store;
};

/// Runs the canonical hierarchical or flat System PnR invocation for one exact
/// D/F/R/H/C/K binding. Only an independently verified and finalized
/// SystemMapping may enter the returned candidate set.
SystemPnrGenerationOutcome
generateSystemMappings(const SystemPnrGenerationInputs &inputs);

} // namespace loom::pnr

#endif // LOOM_PNR_SYSTEM_SYSTEMPNRGENERATOR_H
