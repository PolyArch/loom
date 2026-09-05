#ifndef LOOM_DSE_JOINTMAPPINGMIGRATION_H
#define LOOM_DSE_JOINTMAPPINGMIGRATION_H

#include "DSE/JointDesignExploration.h"
#include "PnR/System/SystemMappingMigration.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace loom {
class ArtifactStore;
}

namespace loom::dse {

enum class JointMappingRebaseFailureReason : std::uint8_t {
  MissingParentFrontier,
  MissingImpactProjection,
  ImpactRequiresColdFallback,
  ModuleCorrespondence,
  TechImpactReopened,
  SpatialImpactReopened,
  TechRebaseRejected,
  SpatialRebaseRejected,
};

struct JointMappingRebaseFailure final {
  JointMappingRebaseFailureReason reason;
  std::optional<ArtifactRootReference> parent;
  std::string diagnostic;
};

struct JointMappingRebaseAccounting final {
  std::uint64_t parentTechMappings = 0;
  std::uint64_t parentSpatialMappings = 0;
  std::uint64_t preservedTechMappings = 0;
  std::uint64_t preservedSpatialMappings = 0;
  std::uint64_t repairedTechMappings = 0;
  std::uint64_t repairedSpatialMappings = 0;
  std::uint64_t invalidatedTechMappings = 0;
  std::uint64_t invalidatedSpatialMappings = 0;
  std::uint64_t parentTechDecisions = 0;
  std::uint64_t parentSpatialDecisions = 0;
  std::uint64_t preservedTechDecisions = 0;
  std::uint64_t preservedSpatialDecisions = 0;
  std::uint64_t reopenedTechDecisions = 0;
  std::uint64_t reopenedSpatialDecisions = 0;
  std::uint64_t repairedTechDecisions = 0;
  std::uint64_t repairedSpatialDecisions = 0;
  std::uint64_t invalidationRootCount = 0;
  std::uint64_t invalidationConeDecisionCount = 0;
  std::uint64_t parentRouteNodeCount = 0;
  std::uint64_t preservedRouteNodeCount = 0;
  std::uint64_t reopenedRouteNodeCount = 0;
  std::uint64_t repairedRouteNodeCount = 0;
  std::uint64_t parentServiceLegCount = 0;
  std::uint64_t preservedServiceLegCount = 0;
  std::uint64_t reopenedServiceLegCount = 0;
  std::uint64_t parentThreadBindingCount = 0;
  std::uint64_t preservedThreadBindingCount = 0;
  std::uint64_t reopenedThreadBindingCount = 0;
  std::uint64_t parentGraphBindingCount = 0;
  std::uint64_t preservedGraphBindingCount = 0;
  std::uint64_t reopenedGraphBindingCount = 0;
  std::uint64_t parentResourceUseCount = 0;
  std::uint64_t preservedResourceUseCount = 0;
  std::uint64_t reopenedResourceUseCount = 0;
  std::uint64_t parentServiceRealizationCount = 0;
  std::uint64_t preservedServiceRealizationCount = 0;
  std::uint64_t reopenedServiceRealizationCount = 0;
};

enum class JointMappingReuseDisposition : std::uint8_t {
  Preserved,
  LocalRepair,
  ColdFallback,
};

llvm::StringRef
jointMappingReuseDispositionSpelling(JointMappingReuseDisposition disposition);

/// Verifies that every parent object is partitioned exactly once and that the
/// invalidation cone equals the typed reopened and repaired decision domain.
llvm::Error validateJointMappingRebaseAccounting(
    const JointMappingRebaseAccounting &accounting);

/// Projects the canonical unique hardware-root inventory charged to one
/// mutation lineage. Composed impacts use the same cold-fallback aggregation
/// as rebase accounting; durable evidence must reuse this owner.
std::uint64_t projectJointHardwareInvalidationRootCount(
    llvm::ArrayRef<struct HardwareImpactProjection> impacts);

struct JointMappingRebaseResult final {
  JointDesignMappingSeed seed;
  JointMappingRebaseAccounting accounting;
  std::vector<JointMappingRebaseFailure> failures;
  JointMappingReuseDisposition disposition =
      JointMappingReuseDisposition::ColdFallback;
};

llvm::StringRef
jointMappingRebaseFailureReasonSpelling(JointMappingRebaseFailureReason reason);

llvm::Expected<std::vector<ArtifactRootReference>>
resolveJointSpatialMappingFrontier(const JointDesignExplorationPlan &plan,
                                   const JointDesignExecution &execution);

/// Projects typed System hardware roots through one exact parent Mapping.
/// Execution-root changes reopen only the Dataflow roots bound to affected
/// AccCores. Transport and service changes conservatively reopen every root
/// because their routes may serve any binding. The projection is the shared
/// owner used by both migration-seed construction and cone accounting.
std::vector<::dataflow::RootThreadLaunchRef> projectJointSystemReopenRoots(
    const mapping::SystemMappingView &parentMapping,
    llvm::ArrayRef<struct HardwareImpactProjection> impacts);

/// Rebases each parent Mapping independently. A rejected Mapping enters the
/// child Generate domain while successfully rebased siblings remain exact
/// plan inputs. A selected parent scope restricts System binding accounting to
/// the one Mapping for which a migration seed will be materialized. The
/// returned roots are child-owned preferences, never proofs.
llvm::Expected<JointMappingRebaseResult> rebaseJointMappingFrontier(
    const JointDesignExplorationPlan &parentPlan,
    const JointDesignExecution &parentExecution,
    const ArtifactRootReference &childSystem,
    llvm::ArrayRef<pnr::SystemModuleCorrespondence> moduleCorrespondences,
    llvm::ArrayRef<struct HardwareImpactProjection> impacts,
    const ArtifactStore &artifacts,
    std::optional<ArtifactRootReference> selectedParentMapping = std::nullopt);

} // namespace loom::dse

#endif // LOOM_DSE_JOINTMAPPINGMIGRATION_H
