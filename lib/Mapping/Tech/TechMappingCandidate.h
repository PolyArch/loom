#ifndef LOOM_LIB_MAPPING_TECH_TECHMAPPINGCANDIDATE_H
#define LOOM_LIB_MAPPING_TECH_TECHMAPPINGCANDIDATE_H

#include "Mapping/Artifact/SpatialPhysicalDemandProjection.h"
#include "Mapping/Tech/TechMappingGenerator.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <variant>
#include <vector>

namespace loom::mapping::detail {

using TechMatchRealization =
    std::variant<TechComputeRealizationView, TechMemoryRealizationView>;

struct TechMatchRow final {
  std::vector<std::uint8_t> key;
  std::vector<std::size_t> actorSlots;
  TechMatchRealization realization;
  /// Invocation-local ordinals into TechMatchDomain's resident-context
  /// supply. Empty for memory rows and for compute rows with no physical root.
  std::vector<std::size_t> computeContextValues;
  /// Immutable active-demand projection for a memory row. This is derived
  /// once per invocation and never contains an occurrence selection.
  std::optional<SpatialMemoryOccurrenceDemandView> memoryOccurrenceDemand;
};

struct TechMatchDomain final {
  std::vector<::dataflow::ActorRef> actors;
  std::vector<TechMatchRow> rows;
  std::size_t computeContextValueCount = 0;
  std::vector<::loom::fabric::InstructionContextRef> computeContexts;
  bool exhausted = true;
  bool interrupted = false;
};

struct TechMatchMemoryDomainBucket final {
  ::fabric::Schedule schedule = ::fabric::Schedule::Spatial;
  std::uint64_t actorCount = 0;
  std::uint64_t occurrenceDomainWidth = 0;
  std::uint64_t rowCount = 0;
};

struct TechMatchDomainStatistics final {
  std::uint64_t rowCount = 0;
  std::uint64_t computeRowCount = 0;
  std::uint64_t memoryRowCount = 0;
  std::vector<TechMatchMemoryDomainBucket> memoryBuckets;
};

TechMatchDomainStatistics
summarizeTechMatchDomain(const TechMatchDomain &domain);

llvm::Expected<TechMatchDomain> deriveTechMatchDomain(
    const TechMappingGenerationInputs &inputs,
    llvm::ArrayRef<::dataflow::CanonicalActorView> selectedActors,
    TechMappingGenerationAccounting &accounting);

llvm::Expected<ArtifactRootReference>
materializeTechMappingCandidate(const TechMappingGenerationInputs &inputs,
                                llvm::ArrayRef<const TechMatchRow *> rows);

struct TechCoverSearchResult final {
  std::vector<std::vector<const TechMatchRow *>> covers;
  bool exhausted = true;
  bool interrupted = false;
  TechMappingGenerationFeedback feedback = {};
};

TechCoverSearchResult
searchTechMatchCovers(const TechMatchDomain &domain,
                      const ResolvedTechMappingConfigView &config,
                      TechMappingGenerationAccounting &accounting,
                      ExecutionControlView executionControl = {});

TechCoverSearchResult searchTechMatchCovers(
    const TechMatchDomain &domain, const ResolvedTechMappingConfigView &config,
    TechMappingGenerationAccounting &accounting, std::uint64_t coverLimit,
    ExecutionControlView executionControl = {});

} // namespace loom::mapping::detail

#endif // LOOM_LIB_MAPPING_TECH_TECHMAPPINGCANDIDATE_H
