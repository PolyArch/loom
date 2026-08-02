#ifndef LOOM_LIB_MAPPING_TECH_TECHMAPPINGCANDIDATE_H
#define LOOM_LIB_MAPPING_TECH_TECHMAPPINGCANDIDATE_H

#include "Mapping/Tech/TechMappingGenerator.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstddef>
#include <cstdint>
#include <variant>
#include <vector>

namespace loom::mapping::detail {

using TechMatchRealization =
    std::variant<TechComputeRealizationView, TechMemoryRealizationView>;

struct TechMatchRow final {
  std::vector<std::uint8_t> key;
  std::vector<std::size_t> actorSlots;
  TechMatchRealization realization;
};

struct TechMatchDomain final {
  std::vector<::dataflow::ActorRef> actors;
  std::vector<TechMatchRow> rows;
  bool exhausted = true;
};

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
};

TechCoverSearchResult
searchTechMatchCovers(const TechMatchDomain &domain,
                      const ResolvedTechMappingConfigView &config,
                      TechMappingGenerationAccounting &accounting);

} // namespace loom::mapping::detail

#endif // LOOM_LIB_MAPPING_TECH_TECHMAPPINGCANDIDATE_H
