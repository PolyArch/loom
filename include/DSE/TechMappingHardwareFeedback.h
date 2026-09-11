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

/// The two typed supply sources a compute-context Hall deficit admits. A
/// Spatial-schedule PE hosts one resident instruction per context and never
/// rotates, so moving demand onto Spatial FU occurrences keeps loop-carried
/// work unserialized. Temporal instruction stores remain the supply for
/// demand that no Spatial PE of the exact parent Module can host.
enum class TechMappingComputeContextGrowthDirection : std::uint8_t {
  SpatialFuOccurrence,
  TemporalInstructionStore,
};

llvm::StringRef techMappingComputeContextGrowthDirectionSpelling(
    TechMappingComputeContextGrowthDirection direction);

/// One exact Spatial FU occurrence growth step. `decision` replaces the whole
/// FU inventory of `decision.target` with its current occurrences plus one
/// clone of an existing occurrence of `capability`. `addedContextCount` is the
/// target's resident-context count, which is exactly the supply the step makes
/// compatible with every demand group that admits that capability.
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
  /// Spatial direction only. The microarchitecture decision vocabulary
  /// changes one PE's FU inventory per child Module, so the plan carries the
  /// one step that most increases the observed matching; the reopen chain
  /// re-observes the Hall relation and takes the next step.
  std::optional<TechMappingComputeContextSpatialFuGrowth> spatialFuGrowth;
  /// Structural bound of the Spatial direction: the resident contexts the
  /// exact parent Module's admissible Spatial PEs could still make compatible
  /// after this step, and the part of the observed deficit that bound cannot
  /// reach. The bound is an upper estimate of remaining Spatial supply, so
  /// the residual is a lower estimate of the demand instruction stores must
  /// cover. It is zero exactly when no admissible Spatial PE improves the
  /// observed relation, and the residual is then the complete deficit: that
  /// is why the Temporal direction takes over.
  std::uint64_t spatialFuContextSupplyBound = 0;
  std::uint64_t spatialFuUnclosedDeficit = 0;
};

/// Chooses the growth direction for one exact observed Hall relation and
/// sizes the chosen direction. The Spatial direction is preferred whenever an
/// admissible Spatial FU occurrence strictly increases the observed maximum
/// matching; otherwise the plan closes the exact relation with minimum total
/// new Temporal context capacity. The returned parent-scoped PE resizes form
/// one atomic kind-14 ResizeInstructionStores decision; they must not be
/// rebound through intermediate child identities.
llvm::Expected<TechMappingComputeContextJointGrowthPlan>
projectTechMappingComputeContextJointGrowthPlan(
    const mapping::TechMappingComputeContextHallDeficit &feedback,
    const fabric::FabricArtifactView &module);

} // namespace loom::dse

#endif // LOOM_DSE_TECHMAPPINGHARDWAREFEEDBACK_H
