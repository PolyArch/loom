#ifndef LOOM_DSE_COMPOSITEFUSUPPLY_H
#define LOOM_DSE_COMPOSITEFUSUPPLY_H

#include "ADG/BuiltinDescriptor.h"
#include "Common/Artifact.h"
#include "Common/ArtifactStore.h"
#include "DSE/FabricTemplateCandidateGenerator.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

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
};

/// What the one selection path did. An absent proposal has four different
/// owners and a caller cannot act on, or report, the difference without them:
/// software that mines no shape at all, shapes every one of which presents a
/// wider boundary than a PE, shapes whose operation families no inverse
/// capability policy admits, and shapes the FU model cannot hold because their
/// internal relation closes a loop recurrence. The capability refusal names a
/// missing Fabric policy rather than a missing opportunity, so it is recorded
/// with its exact reason.
struct MinedCompositeFuSupplyOutcome final {
  std::optional<MinedCompositeFuProposal> proposal;
  /// Shapes the miner reported for this software, by node count. The level a
  /// shape was reported at is what says whether the search reached the wide
  /// recurring shapes or only the narrow ones, so the account is kept per
  /// level and the total is derived from it.
  std::vector<std::uint64_t> minedCandidatesByActorCount;
  /// Shapes the miner withheld because their FU boundary exceeds what a PE
  /// presents. The mining request carries that width, so these never reach the
  /// rank; the count is what the software recurs on but no PE can hold.
  std::uint64_t boundaryRefusedCount = 0;
  /// Reported shapes the canonical capability derivation refused.
  std::uint64_t capabilityRefusedCount = 0;
  /// The first such refusal's exact typed reason, which names the operation
  /// family whose inverse policy is missing.
  std::string firstCapabilityRefusal;
  /// Reported shapes no FU can hold in this profile. Mining is a relation over
  /// software, so a shape that closes a loop recurrence is an ordinary thing
  /// to find and a different observation from a family the Fabric cannot yet
  /// express.
  std::uint64_t recurrenceRefusedCount = 0;
  /// Whether the bounded search stopped growing before its node bound. A
  /// bounded search still proposes what it found; the flag is the evidence
  /// that a wider template might exist.
  bool bounded = false;
  /// Largest node count whose level the search enumerated completely. A
  /// bounded search over a whole-layer graph reports a small number here, and
  /// that number rather than the node bound is what the search covered.
  std::uint32_t exploredActorCount = 0;
  /// Wall time the bounded search spent. Mining runs inside a compile budget
  /// that the rest of the invocation also spends, so what it costs is part of
  /// the observation rather than an afterthought.
  std::uint64_t searchMilliseconds = 0;

  std::uint64_t minedCandidateCount() const {
    std::uint64_t total = 0;
    for (std::uint64_t reported : minedCandidatesByActorCount)
      total += reported;
    return total;
  }
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
/// or reports that the software offers none and why.
///
/// `actorDemand` is how many actors the caller needs the supply to stop
/// spending one realization each on. One composite occurrence answers
/// `actorsPerRealization` of it per realization it covers, so the proposal
/// asks for the sites that demand needs, bounded by the target's Spatial PE
/// count. Every ranked candidate already fits the boundary a PE presents,
/// because the mining request carries it, so selection walks the rank and takes
/// the first candidate whose capability the canonical derivation admits; an
/// absent proposal is an ordinary observation the caller retreats from.
llvm::Expected<MinedCompositeFuSupplyOutcome>
proposeMinedCompositeFuSupply(const ArtifactRootReference &dataflow,
                              std::uint64_t actorDemand,
                              const loom::adg::BuiltinTargetScale &scale,
                              const ArtifactStore &store);

/// Records one selection walk into a diagnostic record. Both observations that
/// raise a demand report the same fields, so the field names stay with the
/// outcome that defines them.
void describeMinedCompositeFuSupply(
    llvm::json::Object &fields, const MinedCompositeFuSupplyOutcome &supply);

} // namespace loom::dse

#endif // LOOM_DSE_COMPOSITEFUSUPPLY_H
