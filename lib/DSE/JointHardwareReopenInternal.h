#ifndef LOOM_DSE_JOINTHARDWAREREOPENINTERNAL_H
#define LOOM_DSE_JOINTHARDWAREREOPENINTERNAL_H

#include "DSE/ExecutionJournal.h"
#include "DSE/HardwareDecision.h"
#include "DSE/JointHardwareReopen.h"
#include "DSE/ResolvedConfigView.h"
#include "DSE/SiteScheduler.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Mapping/Artifact/SpatialMappingHardwareDemand.h"
#include "Mapping/Artifact/SystemMappingHardwareDemand.h"
#include "Mapping/Tech/TechMappingHardwareDemand.h"
#include "PnR/System/SystemMappingMigration.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>
#include <vector>

namespace loom::dse::joint_reopen_detail {

llvm::Error invalid(const llvm::Twine &message);

struct JointSoftwareCoverage final {
  std::uint64_t acceleratedRootCount = 0;
  std::uint64_t graphCount = 0;
  std::uint64_t actorCount = 0;
};

struct TechHardwareFeedbackObservation final {
  ArtifactRootReference module;
  mapping::TechMappingComputeContextHallDeficit feedback;
};

struct SpatialHardwareFeedbackObservation final {
  mapping::SpatialGraphBoundaryEndpointHallDeficit feedback;
};

struct SystemHardwareFeedbackObservation final {
  mapping::SystemAccCoreCapacityPressure feedback;
};

struct HardwareRecipeGrowth final {
  ResolvedConfig config;
  std::optional<ArtifactRootReference> accCoreParent;
  std::optional<ArtifactRootReference> accCoreTargetModule;
  std::optional<ArtifactRootReference> techModule;
  std::vector<ResizeInstructionStore> instructionStoreResizes;
  std::optional<ResizeFifo> fifoResize;
  std::optional<ChangeFifoBypassCapability> fifoBypassChange;
  std::optional<ChangeFifoQueueDiscipline> fifoDisciplineChange;
  std::optional<ChangeTemporalOperandBufferMode> operandBufferModeChange;
  std::optional<ResizeTemporalOperandBuffer> operandBufferResize;
  std::optional<SpatialMicroarchitectureDecisionDomain> moduleDecision;
  std::optional<SpatialTopologyDecisionDomain> topologyDecision;
  std::uint64_t resizedInstructionStoreCount = 0;
  std::uint64_t maximumInstructionStoreCapacity = 0;
  std::uint64_t addedContexts = 0;
  std::uint64_t resultingContexts = 0;
  std::uint64_t addedGateways = 0;
  std::uint64_t resultingGateways = 0;
  std::uint64_t addedAccCores = 0;
  std::uint64_t resultingAccCores = 0;
  bool uniformContextGrowth = false;
};

struct MaterializedHardwareCandidate final {
  ArtifactRootReference reference;
  ResolvedConfig config;
  std::optional<pnr::SystemExecutionBindingCorrespondence>
      executionBindingCorrespondence;
  std::vector<pnr::SystemModuleCorrespondence> moduleCorrespondences;
  std::optional<HardwareImpactProjection> mappingImpact;
  std::uint64_t resizedInstructionStoreCount = 0;
  std::uint64_t maximumInstructionStoreCapacity = 0;
  std::uint64_t addedContexts = 0;
  std::uint64_t resultingContexts = 0;
  std::uint64_t addedGateways = 0;
  std::uint64_t resultingGateways = 0;
  std::uint64_t addedAccCores = 0;
  std::uint64_t resultingAccCores = 0;
  std::optional<JointDesignInvocationManifestReference> constructionInvocation;
};

struct TechGateExecution final {
  JointDesignExecution execution;
  std::vector<ArtifactRootReference> techMappings;
  bool coversRequiredGraphs = false;
};

struct FinalizedMappingHardwareAttempt final {
  ArtifactRootReference system;
  JointDesignExecution execution;
};

struct FinalizedMappingHardwareSpectrum final {
  /// Every child System that entered ordinary Mapping. Failed and incomplete
  /// attempts remain necessary promotion provenance even though only a
  /// verified Mapping can re-enter bounded-quality selection.
  std::vector<FinalizedMappingHardwareAttempt> attempts;
  std::vector<JointDesignInvocationManifestReference> invocations;
  std::uint64_t attemptedSystems = 0;
  bool incomplete = false;
};

llvm::Expected<JointSoftwareCoverage>
projectJointSoftwareCoverage(const JointDesignExplorationPlan &plan,
                             const ArtifactStore &artifacts);

bool dispatchDeadlineReached(const PlanExecutionPolicy &policy);

llvm::Expected<PlanExecutionPolicy>
fairBoundedQualityPlanPolicy(const PlanExecutionPolicy &base,
                             std::uint64_t remainingPlanCount);

std::size_t mappingCount(const JointDesignExecution &execution);

void canonicalizeRoots(std::vector<ArtifactRootReference> &roots);

llvm::Expected<std::vector<ArtifactRootReference>>
boundTechMappingFrontierForRepair(
    llvm::ArrayRef<ArtifactRootReference> candidates, std::uint64_t limit,
    const ArtifactStore &artifacts);

std::vector<ArtifactRootReference>
mappingRoots(const JointDesignExecution &execution);

void mergeMappedPairs(JointDesignExecution &target,
                      const JointDesignExecution &source);

std::optional<ArtifactRootReference>
firstMapping(const JointDesignExecution &execution);

llvm::Error recordJointAttempt(
    std::vector<JointDesignAttemptRecord> &records, std::uint64_t planOrdinal,
    const ArtifactRootReference &fallbackSystem,
    const JointDesignExecution &execution,
    std::optional<ArtifactRootReference> hardwarePromotionParentSystem = {});

llvm::Error bindImmutableSpatialMappingFrontier(
    JointDesignExplorationPlan &plan,
    llvm::ArrayRef<ArtifactRootReference> spatialMappings,
    const ArtifactStore &artifacts);

llvm::Error bindCheckpointSystemMappingMigrationSeed(
    JointDesignExplorationPlan &plan,
    const ArtifactRootReference &migrationSeed, const ArtifactStore &artifacts);

llvm::Error bindFinalizedSystemMappingMigrationSeed(
    JointDesignExplorationPlan &plan,
    const ArtifactRootReference &migrationSeed, const ArtifactStore &artifacts);

llvm::Expected<pnr::SystemMappingMigrationContext>
deriveSystemMappingMigrationContext(const JointDesignExplorationPlan &plan);

llvm::Expected<TechGateExecution>
executeTechGate(const JointDesignExplorationPlan &plan,
                llvm::ArrayRef<ArtifactRootReference> evidence,
                const JointHardwareReopenRequest &request,
                SiteScheduler &scheduler, const ArtifactStore &artifacts,
                const BlobStore &blobs,
                const PlanExecutionPolicy &executionPolicy);

llvm::Expected<std::optional<TechHardwareFeedbackObservation>>
selectTechHardwareFeedback(const JointDesignExecution &execution,
                           const ArtifactStore &artifacts);

llvm::Expected<std::optional<SpatialHardwareFeedbackObservation>>
selectSpatialHardwareFeedback(const JointDesignExecution &execution,
                              const ArtifactStore &artifacts);

llvm::Expected<std::optional<SystemHardwareFeedbackObservation>>
selectSystemHardwareFeedback(const JointDesignExecution &execution,
                             const ArtifactStore &artifacts);

llvm::Expected<HardwareRecipeGrowth> deriveHardwareRecipeGrowth(
    const ResolvedConfig &baseConfig,
    const std::optional<TechHardwareFeedbackObservation> &techObservation,
    const std::optional<SpatialHardwareFeedbackObservation> &spatialObservation,
    const std::optional<SystemHardwareFeedbackObservation> &systemObservation,
    const ArtifactStore &artifacts);

llvm::Expected<HardwareRecipeGrowth> deriveUniformTechHardwareRecipeGrowth(
    const ResolvedConfig &baseConfig,
    const TechHardwareFeedbackObservation &observation,
    const ArtifactStore &artifacts);

llvm::Expected<MaterializedHardwareCandidate> materializeHardwareRecipeGrowth(
    HardwareRecipeGrowth growth, llvm::ArrayRef<ArtifactRootReference> evidence,
    const JointHardwareReopenRequest &request, SiteScheduler &scheduler,
    const ArtifactStore &artifacts, const BlobStore &blobs);

llvm::Expected<MaterializedHardwareCandidate>
materializeTypedModuleSystemGrowth(HardwareRecipeGrowth growth,
                                   const ArtifactRootReference &parentSystem,
                                   const ArtifactStore &artifacts,
                                   const BlobStore &blobs);

llvm::Expected<MaterializedHardwareCandidate>
materializeTypedAccCoreGrowth(HardwareRecipeGrowth growth,
                              const ArtifactStore &artifacts,
                              const BlobStore &blobs);

llvm::Expected<FinalizedMappingHardwareSpectrum>
exploreFinalizedMappingHardwareSpectrum(
    const JointDesignPolicy &policy, const JointDesignExplorationPlan &plan,
    const JointDesignExecution &parentExecution,
    llvm::ArrayRef<ArtifactRootReference> evidence,
    const JointHardwareReopenRequest &request, SiteScheduler &scheduler,
    const ArtifactStore &artifacts, const BlobStore &blobs,
    const PlanExecutionPolicy *executionPolicy);

llvm::Expected<std::optional<JointDesignExecution>> tryHardwareFeedbackReopen(
    const JointDesignPolicy &policy, const JointDesignExplorationPlan &plan,
    const JointDesignExecution &failedExecution,
    std::optional<JointDesignExecution> &lastFailedExecution,
    std::uint64_t planOrdinal,
    std::vector<JointDesignAttemptRecord> &attemptRecords,
    JointDesignExecutionSummary &accounting,
    std::vector<JointDesignInvocationManifestReference> &encounteredInvocations,
    llvm::ArrayRef<ArtifactRootReference> evidence,
    const JointHardwareReopenRequest &request, SiteScheduler &scheduler,
    const ArtifactStore &artifacts, const BlobStore &blobs,
    std::optional<ArtifactRootReference> hardwarePromotionParentSystem,
    const PlanExecutionPolicy *executionPolicy);

} // namespace loom::dse::joint_reopen_detail

#endif
