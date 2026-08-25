#ifndef LOOM_PNR_SYSTEM_SYSTEMMAPPINGMIGRATION_H
#define LOOM_PNR_SYSTEM_SYSTEMMAPPINGMIGRATION_H

#include "Common/Artifact.h"
#include "Common/ComponentViewDigest.h"
#include "Dataflow/IR/DataflowStructuralRefs.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/Identity/FabricRefs.h"
#include "Mapping/Artifact/SystemMappingArtifact.h"
#include "Mapping/Artifact/SystemMappingHardwareDemand.h"
#include "Mapping/Artifact/SystemPresburger.h"
#include "PnR/PnrIndex.h"
#include "PnR/System/SystemRouteMigration.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>
#include <variant>
#include <vector>

namespace loom {
class ArtifactStore;
class BlobStore;
} // namespace loom

namespace loom::pnr {

class FrozenSystemPnrProblem;

inline constexpr ArtifactSchemaDescriptor
    systemMappingCheckpointMigrationSeedArtifactSchema{
        "loom.pnr.system_mapping_checkpoint_migration_seed",
        SchemaVersion{5, 0}};

inline constexpr ArtifactSchemaDescriptor
    systemMappingFinalizedMigrationSeedArtifactSchema{
        "loom.pnr.system_mapping_finalized_migration_seed",
        SchemaVersion{5, 0}};

class SystemMappingMigrationContext final {
public:
  static llvm::Expected<SystemMappingMigrationContext>
  get(ArtifactRootReference childConstraints,
      std::vector<ArtifactRootReference> spatialMappings,
      ComponentViewDigest resolvedPnrConfigDigest);

  const ArtifactRootReference &childConstraints() const {
    return childConstraints_;
  }
  llvm::ArrayRef<ArtifactRootReference> spatialMappings() const {
    return spatialMappings_;
  }
  const ComponentViewDigest &resolvedPnrConfigDigest() const {
    return resolvedPnrConfigDigest_;
  }

private:
  SystemMappingMigrationContext(
      ArtifactRootReference childConstraints,
      std::vector<ArtifactRootReference> spatialMappings,
      ComponentViewDigest resolvedPnrConfigDigest)
      : childConstraints_(std::move(childConstraints)),
        spatialMappings_(std::move(spatialMappings)),
        resolvedPnrConfigDigest_(resolvedPnrConfigDigest) {}

  ArtifactRootReference childConstraints_;
  std::vector<ArtifactRootReference> spatialMappings_;
  ComponentViewDigest resolvedPnrConfigDigest_;
};

enum class ResourceTimeTransitionStatus : std::uint8_t {
  Verified,
  Unsupported,
  ProofNotEstablished,
  CancelledOrTimeout,
};

llvm::StringRef
resourceTimeTransitionStatusSpelling(ResourceTimeTransitionStatus status);

/// The compiler-owned event boundary that makes a finite Mapping transition
/// causally selectable. A completion safe point is derived from a canonical
/// Dataflow completion event; an explicit safe point is a compiler artifact
/// carrying an event that the compiler proved quiescent.
enum class ResourceTimeSafePointKind : std::uint8_t {
  Completion,
  Explicit,
};

llvm::StringRef
resourceTimeSafePointKindSpelling(ResourceTimeSafePointKind kind);

struct ResourceTimeSafePointReference final {
  ArtifactRootReference artifact;
  ResourceTimeSafePointKind kind = ResourceTimeSafePointKind::Explicit;
};

/// One typed endpoint of a resource-time edge. Deployment is absent only for
/// an incomplete candidate edge; a verified edge always names both exact
/// Deployment closures alongside their SystemMappings.
struct ResourceTimeTransitionEndpointReference final {
  ArtifactRootReference mapping;
  std::optional<ArtifactRootReference> deployment;

  friend bool operator==(const ResourceTimeTransitionEndpointReference &lhs,
                         const ResourceTimeTransitionEndpointReference &rhs) {
    return lhs.mapping == rhs.mapping && lhs.deployment == rhs.deployment;
  }
  friend bool operator!=(const ResourceTimeTransitionEndpointReference &lhs,
                         const ResourceTimeTransitionEndpointReference &rhs) {
    return !(lhs == rhs);
  }
};

enum class ResourceTimeReadinessKind : std::uint8_t {
  Completion,
  FifoToken,
};

llvm::StringRef
resourceTimeReadinessKindSpelling(ResourceTimeReadinessKind kind);

enum class ResourceTimeConcurrencyBoundStatus : std::uint8_t {
  Exact,
  ProofNotEstablished,
};

llvm::StringRef resourceTimeConcurrencyBoundStatusSpelling(
    ResourceTimeConcurrencyBoundStatus status);

/// One region's explicit allocation at a compiler-known resource-time state.
/// The region identity is owned by Canonical Dataflow; physical owners are
/// borrowed from the existing Fabric/System mapping domains.
struct ResourceTimeRegionAllocation final {
  ::dataflow::RootThreadLaunchRef region;
  std::vector<::loom::fabric::FabricPhysicalOccurrenceOwnerRef> resources;
};

/// Finite, compiler-precomputed transition between two already verified
/// SystemMappings. This is a migration contract, not a second Mapping
/// legality model and not an online DSE/PnR request.
struct ResourceTimeTransition final {
  ::dataflow::EventFamilyKey trigger;
  /// A compiler-known completion or safe-point root. Runtime preemption is
  /// outside this contract. An incomplete candidate edge may retain no safe
  /// point rather than fabricating one.
  std::optional<ResourceTimeSafePointReference> safePoint;
  ResourceTimeTransitionEndpointReference parent;
  ResourceTimeTransitionEndpointReference child;
  std::vector<ResourceTimeRegionAllocation> beforeActive;
  std::vector<ResourceTimeRegionAllocation> afterActive;
  std::vector<ArtifactRootReference> beforeLiveWork;
  std::vector<ArtifactRootReference> afterLiveWork;
  std::optional<ArtifactRootReference> tokenLiveStateCorrespondence;
  std::optional<ComponentViewDigest> resourceDeltaDigest;
  std::optional<ComponentViewDigest> configurationDeltaDigest;
  std::optional<ComponentViewDigest> routeDeltaDigest;
  std::optional<std::uint64_t> migrationTimePicoseconds;
  ResourceTimeTransitionStatus status =
      ResourceTimeTransitionStatus::ProofNotEstablished;
};

struct ResourceTimeTransitionSequence final {
  std::vector<ResourceTimeTransition> transitions;
};

struct ResourceTimeRegionPrerequisite final {
  ::dataflow::RootThreadLaunchRef region;
  ResourceTimeReadinessKind readiness = ResourceTimeReadinessKind::Completion;
};

/// One explicit execution interval in a compiler-precomputed schedule. The
/// interval is evidence about a chosen schedule, not a promise that Loom can
/// preempt an in-flight kernel. `FifoToken` records the distinct case where a
/// consumer is released by a token before the producer region completes.
struct ResourceTimeRegionExecution final {
  ::dataflow::RootThreadLaunchRef region;
  std::vector<ResourceTimeRegionPrerequisite> prerequisites;
  std::uint64_t readyPicoseconds = 0;
  std::uint64_t startPicoseconds = 0;
  std::uint64_t completionPicoseconds = 0;
};

/// One event-relative snapshot of the active set. Mapping references are
/// explicit so a witness cannot silently reuse one mapping for a different
/// resource-time state.
struct ResourceTimeScheduleState final {
  ArtifactRootReference mapping;
  ::dataflow::EventFamilyKey event;
  std::uint64_t timePicoseconds = 0;
  std::vector<ResourceTimeRegionAllocation> active;
};

/// A finite schedule alternative whose Mapping transitions have already been
/// selected. It is validated independently of endpoint classification.
struct ResourceTimeScheduleScenario final {
  std::vector<ResourceTimeRegionExecution> executions;
  std::vector<ResourceTimeScheduleState> states;
  ResourceTimeTransitionSequence transitions;
  std::uint64_t makespanPicoseconds = 0;
};

/// Invocation-local witness for comparing resource-time alternatives. The
/// witness carries no MaxSpatial/MaxTemporal label: those labels remain owned
/// by the verified SystemMapping spectrum classifier.
struct ResourceTimeScheduleWitness final {
  std::vector<::dataflow::RootThreadLaunchRef> regions;
  std::vector<ResourceTimeScheduleScenario> scenarios;
  std::uint64_t minimumConcurrentRegions = 0;
  std::uint64_t maximumConcurrentRegions = 0;
  ResourceTimeConcurrencyBoundStatus concurrencyBoundStatus =
      ResourceTimeConcurrencyBoundStatus::ProofNotEstablished;
};

/// Checks only the transition contract's structural invariants. Mapping and
/// Deployment legality remain owned by their existing import/verifier paths.
llvm::Error
validateResourceTimeTransition(const ResourceTimeTransition &transition);

/// Independently imports both Mapping/Deployment endpoints and verifies the
/// exact event-relative allocation evidence. Structural validation alone can
/// describe an incomplete candidate edge; only this closure check authorizes
/// a preverified edge for later runtime selection.
llvm::Error
verifyResourceTimeTransitionClosure(const ResourceTimeTransition &transition,
                                    const ArtifactStore &artifacts,
                                    const BlobStore &blobs);

llvm::Error validateResourceTimeTransitionSequence(
    const ResourceTimeTransitionSequence &sequence);

llvm::Error
validateResourceTimeScheduleWitness(const ResourceTimeScheduleWitness &witness);

struct SystemAccCoreCorrespondence final {
  ::loom::fabric::AccCoreOccurrenceRef parent;
  ::loom::fabric::AccCoreOccurrenceRef child;
};

/// Exact imported-Module lineage for the AccCores retained by one System
/// transformation. Identity entries express resource-only System changes;
/// non-identity entries must come from the typed Module transformation that
/// produced the child attachment.
struct SystemModuleCorrespondence final {
  ArtifactRootReference parent;
  ArtifactRootReference child;

  friend bool operator==(const SystemModuleCorrespondence &lhs,
                         const SystemModuleCorrespondence &rhs) {
    return lhs.parent == rhs.parent && lhs.child == rhs.child;
  }
};

/// Invocation-local proof input for rebasing execution bindings. Hardware DSE
/// derives this relation from exact typed parent/child lineage. PnR validates
/// every selected target against the frozen child problem before use.
class SystemExecutionBindingCorrespondence final {
public:
  static llvm::Expected<SystemExecutionBindingCorrespondence>
  get(ArtifactRootReference parentSystem, ArtifactRootReference childSystem,
      std::vector<::loom::fabric::FabricSystemEntityCorrespondence> entities,
      std::vector<::loom::fabric::FabricSystemTransferPatternCorrespondence>
          transferPatterns,
      std::vector<SystemModuleCorrespondence> modules,
      const ArtifactStore &store);

  /// Derives the identity correspondence of one immutable System. This is the
  /// schedule-preserving case: the Fabric does not change, so no serialized
  /// hardware lineage or copied reference table is required from a caller.
  static llvm::Expected<SystemExecutionBindingCorrespondence>
  getIdentity(const ArtifactRootReference &system, const ArtifactStore &store);

  const ArtifactRootReference &parentSystem() const { return parentSystem_; }
  const ArtifactRootReference &childSystem() const { return childSystem_; }
  llvm::ArrayRef<SystemAccCoreCorrespondence> accCores() const {
    return accCores_;
  }
  llvm::ArrayRef<::loom::fabric::FabricSystemEntityCorrespondence>
  entities() const {
    return entities_;
  }
  llvm::ArrayRef<::loom::fabric::FabricSystemTransferPatternCorrespondence>
  transferPatterns() const {
    return transferPatterns_;
  }
  llvm::ArrayRef<SystemModuleCorrespondence> modules() const {
    return modules_;
  }

private:
  SystemExecutionBindingCorrespondence(
      ArtifactRootReference parentSystem, ArtifactRootReference childSystem,
      std::vector<::loom::fabric::FabricSystemEntityCorrespondence> entities,
      std::vector<::loom::fabric::FabricSystemTransferPatternCorrespondence>
          transferPatterns,
      std::vector<SystemModuleCorrespondence> modules,
      std::vector<SystemAccCoreCorrespondence> accCores)
      : parentSystem_(std::move(parentSystem)),
        childSystem_(std::move(childSystem)), entities_(std::move(entities)),
        transferPatterns_(std::move(transferPatterns)),
        modules_(std::move(modules)), accCores_(std::move(accCores)) {}

  ArtifactRootReference parentSystem_;
  ArtifactRootReference childSystem_;
  std::vector<::loom::fabric::FabricSystemEntityCorrespondence> entities_;
  std::vector<::loom::fabric::FabricSystemTransferPatternCorrespondence>
      transferPatterns_;
  std::vector<SystemModuleCorrespondence> modules_;
  std::vector<SystemAccCoreCorrespondence> accCores_;
};

class FinalizedSystemMappingMigrationSeed final {
public:
  const ArtifactRootReference &reference() const { return reference_; }
  const ::loom::mapping::FinalizedSystemMapping &parentMapping() const {
    return parentMapping_;
  }
  const SystemExecutionBindingCorrespondence &correspondence() const {
    return correspondence_;
  }
  const SystemMappingMigrationContext &context() const { return context_; }
  llvm::ArrayRef<::dataflow::RootThreadLaunchRef> reopenedRoots() const {
    return reopenedRoots_;
  }

private:
  FinalizedSystemMappingMigrationSeed(
      ArtifactRootReference reference,
      ::loom::mapping::FinalizedSystemMapping parentMapping,
      SystemExecutionBindingCorrespondence correspondence,
      SystemMappingMigrationContext context,
      std::vector<::dataflow::RootThreadLaunchRef> reopenedRoots)
      : reference_(std::move(reference)),
        parentMapping_(std::move(parentMapping)),
        correspondence_(std::move(correspondence)),
        context_(std::move(context)), reopenedRoots_(std::move(reopenedRoots)) {
  }

  ArtifactRootReference reference_;
  ::loom::mapping::FinalizedSystemMapping parentMapping_;
  SystemExecutionBindingCorrespondence correspondence_;
  SystemMappingMigrationContext context_;
  std::vector<::dataflow::RootThreadLaunchRef> reopenedRoots_;

  friend llvm::Expected<FinalizedSystemMappingMigrationSeed>
  finalizeSystemMappingMigrationSeed(
      const ArtifactRootReference &,
      const SystemExecutionBindingCorrespondence &,
      const SystemMappingMigrationContext &, const ArtifactStore &);
  friend llvm::Expected<FinalizedSystemMappingMigrationSeed>
  finalizeSystemMappingMigrationSeed(
      const ArtifactRootReference &,
      const SystemExecutionBindingCorrespondence &,
      const SystemMappingMigrationContext &,
      llvm::ArrayRef<::dataflow::RootThreadLaunchRef>, const ArtifactStore &);
  friend llvm::Expected<FinalizedSystemMappingMigrationSeed>
  importSystemMappingMigrationSeed(const ArtifactRootReference &,
                                   const ArtifactStore &);
};

llvm::Expected<FinalizedSystemMappingMigrationSeed>
finalizeSystemMappingMigrationSeed(
    const ArtifactRootReference &parentMapping,
    const SystemExecutionBindingCorrespondence &correspondence,
    const SystemMappingMigrationContext &context, const ArtifactStore &store);

/// Same preserve-first seed with a typed schedule invalidation root set. Only
/// execution decisions owned by these roots and their System service/route
/// dependency cone are released; all remaining choices stay preferences and
/// the generated child still passes the ordinary independent verifier.
llvm::Expected<FinalizedSystemMappingMigrationSeed>
finalizeSystemMappingMigrationSeed(
    const ArtifactRootReference &parentMapping,
    const SystemExecutionBindingCorrespondence &correspondence,
    const SystemMappingMigrationContext &context,
    llvm::ArrayRef<::dataflow::RootThreadLaunchRef> reopenedRoots,
    const ArtifactStore &store);

llvm::Expected<FinalizedSystemMappingMigrationSeed>
importSystemMappingMigrationSeed(const ArtifactRootReference &reference,
                                 const ArtifactStore &store);

/// Durable preserve-first seed assembled from a PnR-owned incomplete
/// execution-binding checkpoint and exact typed hardware correspondence.
/// Its selections remain preferences; child legality is rebuilt independently.
class FinalizedSystemMappingCheckpointMigrationSeed final {
public:
  const ArtifactRootReference &reference() const { return reference_; }
  const ::loom::mapping::FinalizedSystemExecutionBindingCheckpoint &
  checkpoint() const {
    return checkpoint_;
  }
  const SystemExecutionBindingCorrespondence &correspondence() const {
    return correspondence_;
  }
  const SystemMappingMigrationContext &context() const { return context_; }
  ::loom::fabric::AccCoreOccurrenceRef reopenedParentAccCore() const {
    return reopenedParentAccCore_;
  }

private:
  FinalizedSystemMappingCheckpointMigrationSeed(
      ArtifactRootReference reference,
      ::loom::mapping::FinalizedSystemExecutionBindingCheckpoint checkpoint,
      SystemExecutionBindingCorrespondence correspondence,
      SystemMappingMigrationContext context,
      ::loom::fabric::AccCoreOccurrenceRef reopenedParentAccCore)
      : reference_(std::move(reference)), checkpoint_(std::move(checkpoint)),
        correspondence_(std::move(correspondence)),
        context_(std::move(context)),
        reopenedParentAccCore_(reopenedParentAccCore) {}

  ArtifactRootReference reference_;
  ::loom::mapping::FinalizedSystemExecutionBindingCheckpoint checkpoint_;
  SystemExecutionBindingCorrespondence correspondence_;
  SystemMappingMigrationContext context_;
  ::loom::fabric::AccCoreOccurrenceRef reopenedParentAccCore_;

  friend llvm::Expected<FinalizedSystemMappingCheckpointMigrationSeed>
  finalizeSystemMappingCheckpointMigrationSeed(
      const ArtifactRootReference &,
      const SystemExecutionBindingCorrespondence &,
      const SystemMappingMigrationContext &,
      ::loom::fabric::AccCoreOccurrenceRef, const ArtifactStore &);
  friend llvm::Expected<FinalizedSystemMappingCheckpointMigrationSeed>
  importSystemMappingCheckpointMigrationSeed(const ArtifactRootReference &,
                                             const ArtifactStore &);
};

llvm::Expected<FinalizedSystemMappingCheckpointMigrationSeed>
finalizeSystemMappingCheckpointMigrationSeed(
    const ArtifactRootReference &checkpoint,
    const SystemExecutionBindingCorrespondence &correspondence,
    const SystemMappingMigrationContext &context,
    ::loom::fabric::AccCoreOccurrenceRef reopenedParentAccCore,
    const ArtifactStore &store);

llvm::Expected<FinalizedSystemMappingCheckpointMigrationSeed>
importSystemMappingCheckpointMigrationSeed(
    const ArtifactRootReference &reference, const ArtifactStore &store);

enum class SystemMappingMigrationFallbackReason : std::uint8_t {
  ParentMappingDataflowMismatch,
  ParentMappingFabricMismatch,
  ChildFabricMismatch,
  MissingThreadBinding,
  AmbiguousThreadBinding,
  UnmatchedAccCore,
  MissingGraphBinding,
  AmbiguousGraphBinding,
  UnmatchedSpatialMapping,
  EmptyReopenScope,
  ChildRebaseRejected,
  ChildInitializerRejected,
};

struct SystemMappingMigrationFallback final {
  SystemMappingMigrationFallbackReason reason;
};

struct SystemMappingMigrationProjection final {
  std::vector<PnrIndex> fixedChoices;
  std::vector<PnrIndex> releasedChoices;
  std::optional<SystemCandidateRouteSeed> routeSeed;
  std::uint64_t preservedThreadBindings = 0;
  std::uint64_t preservedGraphBindings = 0;
  std::uint64_t preservedServiceLegs = 0;
  std::uint64_t reopenedServiceLegs = 0;
};

using SystemMappingMigrationProjectionOutcome =
    std::variant<SystemMappingMigrationProjection,
                 SystemMappingMigrationFallback>;

/// Rebases only finalized execution-binding choices. The result is one
/// preserve-first initializer seed, never a hard constraint or legality proof.
SystemMappingMigrationProjectionOutcome projectSystemMappingMigrationSeed(
    const FinalizedSystemMappingMigrationSeed &seed,
    const FrozenSystemPnrProblem &childProblem);

SystemMappingMigrationProjectionOutcome projectSystemMappingMigrationSeed(
    const FinalizedSystemMappingCheckpointMigrationSeed &seed,
    const FrozenSystemPnrProblem &childProblem);

llvm::StringRef systemMappingMigrationFallbackReasonSpelling(
    SystemMappingMigrationFallbackReason reason);

} // namespace loom::pnr

#endif // LOOM_PNR_SYSTEM_SYSTEMMAPPINGMIGRATION_H
