#ifndef LOOM_DSE_TECHMAPPINGCOMPOSEDSUPPLY_H
#define LOOM_DSE_TECHMAPPINGCOMPOSEDSUPPLY_H

#include "ADG/BuiltinDescriptor.h"
#include "Common/Artifact.h"
#include "Common/ArtifactStore.h"
#include "DSE/FabricTemplateCandidateGenerator.h"
#include "Mapping/Tech/TechMappingHardwareDemand.h"

#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>

namespace loom::dse {

/// One composite FU template proposed for an exact observed compute-context
/// Hall deficit, mined from the canonical Dataflow of the very candidates
/// TechMapping refused.
///
/// The proposal names the template only by the Dataflow it was mined from and
/// its canonical shape key, which is all the hardware-template configuration
/// carries. Re-mining that Dataflow with the same request reproduces the
/// shape, and a key it does not reproduce stays a typed rejection at the
/// generator; the miner and the canonical capability derivation therefore
/// remain the only owners of FU structure.
struct MinedCompositeFuProposal final {
  MinedCompositeFuSelection selection;
  /// Spatial PE sites the proposal asks to receive an occurrence.
  std::uint32_t occurrences = 0;
  /// Actors one realization of the template binds. This is what makes it
  /// composite: the cover spends one realization where it previously spent
  /// this many.
  std::uint64_t actorsPerRealization = 0;
  /// Least number of distinct actors any node position of the shape binds.
  std::uint64_t support = 0;
  /// Whether the bounded search stopped growing before its node bound. A
  /// bounded search still proposes what it found; the flag is the evidence
  /// that a wider template might exist.
  bool bounded = false;
  /// Wall time the bounded search spent. Mining runs inside a compile budget
  /// that the rest of the invocation also spends, so what it costs is part of
  /// the observation rather than an afterthought.
  std::uint64_t searchMilliseconds = 0;
};

/// Proposes the composite FU template that answers one deficient relation, or
/// reports that the refused software offers none.
///
/// The deficit says how many more compatible context values the cover needs.
/// One composite occurrence answers `actorsPerRealization` of the demand per
/// realization it covers, so the proposal asks for the sites that deficit
/// needs, bounded by the target's Spatial PE count. Selection walks the mined
/// rank and takes the first candidate whose boundary fits a PE of the target
/// and whose capability the canonical derivation admits; an absent result
/// means no mined shape does, which is an ordinary observation the caller
/// retreats from.
llvm::Expected<std::optional<MinedCompositeFuProposal>>
proposeMinedCompositeFuSupply(
    const ArtifactRootReference &dataflow,
    const mapping::TechMappingComputeContextHallDeficit &feedback,
    const loom::adg::BuiltinTargetScale &scale, const ArtifactStore &store);

} // namespace loom::dse

#endif // LOOM_DSE_TECHMAPPINGCOMPOSEDSUPPLY_H
