#ifndef LOOM_DSE_COMPOSITEFUSUPPLY_H
#define LOOM_DSE_COMPOSITEFUSUPPLY_H

#include "ADG/BuiltinDescriptor.h"
#include "Common/Artifact.h"
#include "Common/ArtifactStore.h"
#include "DSE/FabricTemplateCandidateGenerator.h"

#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>

namespace loom::dse {

/// One composite FU template proposed for an exact observed actor demand,
/// mined from the canonical Dataflow of the very software that raised it.
///
/// Two observations raise that demand and both spend this one selection path.
/// A compute-context Hall deficit raises it from the cover side: the cover
/// refused candidates it has no compatible context value for. A verified
/// parent whose mapped replay is far behind its own dataflow oracle raises it
/// from the measured side: the deployed loop spends its interval on actors,
/// most of them control, that one realization could bind together. Both ask
/// the same question, how to spend fewer realizations on the same actors, so
/// both consume the same miner, the same rank, and the same admission owner.
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
  /// Largest node count whose level the search enumerated completely. A
  /// bounded search over a whole-layer graph reports a small number here, and
  /// that number rather than the node bound is what the proposal actually
  /// searched.
  std::uint32_t exploredActorCount = 0;
  /// Wall time the bounded search spent. Mining runs inside a compile budget
  /// that the rest of the invocation also spends, so what it costs is part of
  /// the observation rather than an afterthought.
  std::uint64_t searchMilliseconds = 0;
};

/// Actors of one canonical Dataflow by registered kind. A composite FU can
/// bind only token-plane actors, so the census is both the trigger evidence
/// for a control-dense deployment and the demand a proposal is sized against.
struct CanonicalDataflowActorCensus final {
  std::uint64_t computeActors = 0;
  std::uint64_t controlActors = 0;
  std::uint64_t memoryActors = 0;

  std::uint64_t totalActors() const {
    return computeActors + controlActors + memoryActors;
  }
};

/// Counts the actors of one canonical Dataflow by their registered operation
/// schema kind. The Dataflow operation schema registry remains the only owner
/// of which kind an actor has.
llvm::Expected<CanonicalDataflowActorCensus>
censusCanonicalDataflowActors(const ArtifactRootReference &dataflow,
                              const ArtifactStore &store);

/// Proposes the composite FU template that answers one observed actor demand,
/// or reports that the software offers none.
///
/// `actorDemand` is how many actors the caller needs the supply to stop
/// spending one realization each on. One composite occurrence answers
/// `actorsPerRealization` of it per realization it covers, so the proposal
/// asks for the sites that demand needs, bounded by the target's Spatial PE
/// count. Selection walks the mined rank and takes the first candidate whose
/// boundary fits a PE of the target and whose capability the canonical
/// derivation admits; an absent result means no mined shape does, which is an
/// ordinary observation the caller retreats from.
llvm::Expected<std::optional<MinedCompositeFuProposal>>
proposeMinedCompositeFuSupply(const ArtifactRootReference &dataflow,
                              std::uint64_t actorDemand,
                              const loom::adg::BuiltinTargetScale &scale,
                              const ArtifactStore &store);

} // namespace loom::dse

#endif // LOOM_DSE_COMPOSITEFUSUPPLY_H
