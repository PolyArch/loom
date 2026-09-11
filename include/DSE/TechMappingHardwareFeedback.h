#ifndef LOOM_DSE_TECHMAPPINGHARDWAREFEEDBACK_H
#define LOOM_DSE_TECHMAPPINGHARDWAREFEEDBACK_H

#include "DSE/HardwareDecision.h"
#include "Mapping/Tech/TechMappingHardwareDemand.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>
#include <vector>

namespace loom::fabric {
class FabricArtifactView;
}

namespace loom::dse {

/// Projects the minimal existing hardware action family that can add supply
/// to an observed compute-context Hall relation. Each domain changes one
/// Temporal PE and offers the smallest increment plus the complete observed
/// deficit when those differ. Mapping and the ordinary hardware generator
/// remain the legality and materialization owners.
llvm::Expected<std::vector<SpatialMicroarchitectureDecisionDomain>>
projectTechMappingComputeContextGrowthDomains(
    const mapping::TechMappingComputeContextHallDeficit &feedback,
    const fabric::FabricArtifactView &module);

/// One observation of a compute-context Hall relation, reduced to what a
/// continuation proof needs: how far the relation is from admissible, how much
/// demand it carries, and how much context supply answers that demand.
struct TechMappingComputeContextHallProgress final {
  std::uint64_t deficit = 0;
  std::uint64_t demand = 0;
  std::uint64_t contexts = 0;
};

TechMappingComputeContextHallProgress
observeTechMappingComputeContextHallProgress(
    const mapping::TechMappingComputeContextHallDeficit &feedback);

/// Whether the growth applied between the two observations bought nothing.
/// The closure is atomic and always closes the complete observed deficit, so
/// a relation that reappears with the same deficit while its demand and its
/// context supply both grew by that same amount has spent the new contexts on
/// new demand: every added context let the cover admit one more realization.
/// Another child of the same kind cannot close such a relation, so this is the
/// one owner that decides a context-growth sequence is not a continuation
/// proof. Both the hardware reopen chain and the qualification search consult
/// it; neither restates the rule.
bool techMappingComputeContextHallGrowthStagnates(
    const TechMappingComputeContextHallProgress &previous,
    const TechMappingComputeContextHallProgress &current);

/// The two typed supply sources a compute-context Hall deficit admits.
/// Temporal instruction-store growth closes the complete observed relation in
/// one atomic decision and rebases the parent Mapping layers. Spatial FU
/// occurrence growth keeps loop-carried work unserialized because a
/// Spatial-schedule PE hosts one resident instruction per context and never
/// rotates, but it changes PE-internal structure, reopens every Mapping layer,
/// and can only close a relation whose deficit one PE's resident contexts
/// already cover.
enum class TechMappingComputeContextGrowthDirection : std::uint8_t {
  SpatialFuOccurrence,
  TemporalInstructionStore,
};

llvm::StringRef techMappingComputeContextGrowthDirectionSpelling(
    TechMappingComputeContextGrowthDirection direction);

/// One exact Spatial FU occurrence growth decision. It replaces the whole FU
/// inventory of `decision.target` with its current occurrences plus one clone
/// of an existing occurrence of `capability`, and it is admitted only when
/// that single change makes the complete observed relation admissible.
/// `addedContextCount` is the target's resident-context count, which is
/// exactly the supply the decision makes compatible with every demand group
/// that admits that capability.
struct TechMappingComputeContextSpatialFuGrowth final {
  ChangeFuInventory decision;
  loom::fabric::FabricFuCapabilityTemplateRef capability;
  std::uint64_t addedContextCount = 0;
};

struct TechMappingComputeContextJointGrowthPlan final {
  TechMappingComputeContextGrowthDirection direction =
      TechMappingComputeContextGrowthDirection::TemporalInstructionStore;
  /// Temporal direction only: the atomic minimal instruction-store closure.
  std::vector<ResizeInstructionStore> decisions;
  std::uint64_t addedContextCount = 0;
  /// Spatial direction only: the one decision that closes the complete
  /// observed relation. The microarchitecture vocabulary changes one PE's FU
  /// inventory per child Module, and a Hall closure child must be atomic, so
  /// a relation no single admissible Spatial PE can close is left to the
  /// Temporal direction instead of being approached one PE at a time.
  std::optional<TechMappingComputeContextSpatialFuGrowth> spatialFuGrowth;
  /// Structural bound of the Spatial direction: the resident contexts the
  /// exact parent Module's admissible Spatial PEs other than the one this
  /// plan selects could still make compatible, and the part of the observed
  /// deficit that bound cannot reach. The bound is an upper estimate of the
  /// Spatial supply, so the residual is a lower estimate of the demand
  /// instruction stores must cover.
  std::uint64_t spatialFuContextSupplyBound = 0;
  std::uint64_t spatialFuUnclosedDeficit = 0;
};

/// Chooses the growth direction for one exact observed Hall relation and
/// sizes the chosen direction.
///
/// `preferTemporalInstructionStore` carries the reopen chain's evidence. It
/// holds while the structurally local, atomic instruction-store closure has
/// not yet been shown to produce an unmappable child, and the plan then closes
/// the exact relation with minimum total new Temporal context capacity. A
/// chain that withdraws the preference asks for the Spatial FU occurrence
/// supply instead; the owner offers it only when one admissible Spatial PE
/// makes the complete observed relation admissible, and otherwise falls back
/// to the Temporal closure with the bound recording why. The returned
/// parent-scoped PE resizes form one atomic kind-14 ResizeInstructionStores
/// decision; they must not be rebound through intermediate child identities.
///
/// A relation whose compatible contexts lie on no Temporal PE admits no
/// instruction-store supply at all, and the owner then searches the Spatial
/// direction whatever the preference. An absent result is the typed refusal
/// that neither direction has supply for this relation; it is an ordinary
/// observation the caller retreats from, not a malformed feedback.
llvm::Expected<std::optional<TechMappingComputeContextJointGrowthPlan>>
projectTechMappingComputeContextJointGrowthPlan(
    const mapping::TechMappingComputeContextHallDeficit &feedback,
    const fabric::FabricArtifactView &module,
    bool preferTemporalInstructionStore = true);

} // namespace loom::dse

#endif // LOOM_DSE_TECHMAPPINGHARDWAREFEEDBACK_H
