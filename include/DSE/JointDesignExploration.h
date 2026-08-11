#ifndef LOOM_DSE_JOINTDESIGNEXPLORATION_H
#define LOOM_DSE_JOINTDESIGNEXPLORATION_H

#include "Config/ResolvedConfig.h"
#include "DSE/JointDesignPolicy.h"
#include "DSE/PlanExecutor.h"
#include "DSE/Promotion.h"
#include "Fabric/Identity/FabricRefs.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <optional>
#include <utility>
#include <variant>
#include <vector>

namespace loom {
class ArtifactStore;
class BlobStore;
}

namespace loom::dse {

struct JointDesignPlanPair final {
  JointDesignPair pair;
  PlanOutputRef systemMappings;
};

struct JointDesignExplorationPlan final {
  ResolvedConfig resolvedConfig;
  BoundedJointFrontier frontier;
  std::vector<JointDesignPlanPair> pairOutputs;
};

/// Builds one ordinary finite Generate plan. Each explicit software/System
/// pair traverses root-complete TechMapping, SpatialMapping, and SystemMapping.
llvm::Expected<JointDesignExplorationPlan> buildJointDesignExplorationPlan(
    JointDesignInputs inputs, const JointDesignPolicy &policy,
    const ResolvedConfig &baseConfig, const ArtifactStore &artifactStore);

struct JointMappedPair final {
  JointDesignPair pair;
  std::vector<ArtifactRootReference> systemMappings;
};

struct JointDesignExecution final {
  DsePlanExecutionResult planExecution;
  std::vector<JointMappedPair> mappedPairs;
};

/// Executes or resumes the exact plan through the shared Journal and
/// scheduler. Missing Mapping support remains the underlying typed incomplete
/// plan outcome; this layer never substitutes an estimate.
llvm::Expected<JointDesignExecution> executeJointDesignExploration(
    const JointDesignExplorationPlan &plan, const DseRunClosure &closure,
    ExecutionJournal &journal, SiteScheduler &scheduler,
    const PlanExecutionPolicy &executionPolicy,
    const ArtifactStore &artifactStore, const BlobStore &blobStore);

/// One workload member's already gate-qualified Mapping set. The existing
/// Promotion contract remains the sole correctness and acceleration gate
/// authority.
struct JointMemberPromotion final {
  ArtifactRootReference software;
  CompletedSelection promotion;
};

struct JointEligibleSystem final {
  ArtifactRootReference system;
  std::vector<ArtifactRootReference> acceptedMappings;
};

struct JointSystemMissingMember final {
  ArtifactRootReference system;
  ArtifactRootReference member;
};

struct JointSystemUnusedAccCore final {
  ArtifactRootReference system;
  fabric::AccCoreOccurrenceRef accCore;
};

using JointSystemGateOutcome =
    std::variant<JointEligibleSystem, JointSystemMissingMember,
                 JointSystemUnusedAccCore>;

struct JointDesignSelection final {
  std::vector<ArtifactRootReference> selectedSystems;
  std::vector<ArtifactRootReference> acceptedMappings;
  std::vector<ArtifactRootReference> satisfiedEvidence;
  std::vector<JointSystemGateOutcome> systemOutcomes;
};

struct JointDesignNoFeasibleSystem final {
  std::vector<ArtifactRootReference> satisfiedEvidence;
  std::vector<JointSystemGateOutcome> systemOutcomes;
};

using JointDesignSelectionOutcome =
    std::variant<JointDesignSelection, JointDesignNoFeasibleSystem>;

/// Applies member-local Promotion results and AccCore-use coverage before the
/// existing aggregate candidate selection policy.
llvm::Expected<JointDesignSelectionOutcome> selectJointDesignSystems(
    llvm::ArrayRef<ArtifactRootReference> systems,
    llvm::ArrayRef<JointMemberPromotion> memberPromotions,
    llvm::ArrayRef<CandidateObjectiveVector> systemObjectives,
    const CandidateSelectionPolicy &selection,
    const ObjectiveProgram *objectiveProgram,
    const ArtifactStore &artifactStore);

} // namespace loom::dse

#endif // LOOM_DSE_JOINTDESIGNEXPLORATION_H
