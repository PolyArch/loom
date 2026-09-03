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
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <optional>
#include <string>
#include <system_error>
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

/// Exact lower-Mapping and System-binding partition induced by a typed root
/// reopen set. A lower Mapping used by any preserved graph binding belongs to
/// the preserved side, even when a reopened binding also names it; only
/// mappings used exclusively by reopened roots belong to the reopened side.
/// A graph definition reached from both root sets is likewise preserved.
/// This is the canonical projection for repair execution and derived product
/// evidence.
struct SystemMappingMigrationConePartition final {
  std::vector<::dataflow::RootThreadLaunchRef> reopenedRoots;
  std::vector<::dataflow::GraphRef> preservedGraphs;
  std::vector<::dataflow::GraphRef> reopenedGraphs;
  std::vector<ArtifactRootReference> preservedTechMappings;
  std::vector<ArtifactRootReference> reopenedTechMappings;
  std::vector<ArtifactRootReference> preservedSpatialMappings;
  std::vector<ArtifactRootReference> reopenedSpatialMappings;
  std::uint64_t preservedThreadBindings = 0;
  std::uint64_t reopenedThreadBindings = 0;
  std::uint64_t preservedGraphBindings = 0;
  std::uint64_t reopenedGraphBindings = 0;

  std::uint64_t preservedSystemBindings() const {
    return preservedThreadBindings + preservedGraphBindings;
  }
  std::uint64_t reopenedSystemBindings() const {
    return reopenedThreadBindings + reopenedGraphBindings;
  }
  bool admitsReplacementGraphs(
      llvm::ArrayRef<::dataflow::GraphRef> coveredGraphs) const;
};

llvm::Expected<SystemMappingMigrationConePartition>
projectSystemMappingMigrationConePartition(
    const ::loom::mapping::SystemMappingView &mapping,
    llvm::ArrayRef<::dataflow::RootThreadLaunchRef> reopenedRoots,
    const ArtifactStore &store);

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

/// The closed set of persistent live-state classes a Canonical Dataflow may
/// carry across a completion safe point. `LogicalMemory` has a correspondence
/// owner (`ResourceTimeLogicalMemoryCorrespondence`); `OrderedChannel` and
/// `DynamicWork` state has no migration owner and is a typed refusal.
enum class ResourceTimeLiveStateClass : std::uint8_t {
  LogicalMemory,
  OrderedChannel,
  DynamicWork,
};

llvm::StringRef
resourceTimeLiveStateClassSpelling(ResourceTimeLiveStateClass stateClass);

/// How one retained live-state owner crosses the edge. `RetainedInPlace`
/// names identical physical targets at exact zero cost. `Copied` names one
/// complete, statically bounded logical-memory object moved between distinct
/// equal-extent targets by the selected runtime provider.
enum class ResourceTimeLiveStateMigration : std::uint8_t {
  RetainedInPlace,
  Copied,
};

llvm::StringRef
resourceTimeLiveStateMigrationSpelling(ResourceTimeLiveStateMigration migration);

/// Typed correspondence of one logical memory between the parent and child
/// endpoints. The binding digests are derived by the finalizer from the exact
/// SystemMapping memory targets of each endpoint; an authored record cannot
/// earn `Verified`.
struct ResourceTimeLogicalMemoryCorrespondence final {
  ::dataflow::LogicalMemoryRootRef memory;
  ComponentViewDigest parentBinding;
  ComponentViewDigest childBinding;
  ResourceTimeLiveStateMigration migration =
      ResourceTimeLiveStateMigration::RetainedInPlace;
  std::uint64_t migrationTimePicoseconds = 0;

  friend bool operator==(const ResourceTimeLogicalMemoryCorrespondence &lhs,
                         const ResourceTimeLogicalMemoryCorrespondence &rhs) {
    return lhs.memory == rhs.memory &&
           lhs.parentBinding == rhs.parentBinding &&
           lhs.childBinding == rhs.childBinding &&
           lhs.migration == rhs.migration &&
           lhs.migrationTimePicoseconds == rhs.migrationTimePicoseconds;
  }
  friend bool operator!=(const ResourceTimeLogicalMemoryCorrespondence &lhs,
                         const ResourceTimeLogicalMemoryCorrespondence &rhs) {
    return !(lhs == rhs);
  }
};

/// One occurrence-qualified local-memory target selected for a logical
/// memory. The Module-local service region and offset are qualified by the
/// concrete System AccCore occurrence; SpatialMapping identity is deliberately
/// absent because changing a Mapping does not move physical memory.
struct ResourceTimeSpatialMemoryTarget final {
  ::loom::fabric::AccCoreOccurrenceRef accCore;
  ::loom::mapping::SpatialMemoryIntervalView interval;
  ::loom::mapping::SpatialMemoryLocalRegionView region;
};

/// Exact physical target used by the live-state copy executor. System memory
/// targets already carry their complete service-region and transform path;
/// local targets require the occurrence-qualified context above.
using ResourceTimeMemoryTarget =
    std::variant<ResourceTimeSpatialMemoryTarget,
                 ::loom::mapping::SystemMemoryRegionElementView>;

/// Removable projection from one verified SystemMapping for a particular
/// logical memory and root subset. The digest is the value retained by the
/// transition record; the target list is rederived for runtime preparation.
struct ResourceTimeLogicalMemoryBindingProjection final {
  ::dataflow::LogicalMemoryRootRef memory;
  std::optional<std::uint64_t> byteCount;
  std::vector<ResourceTimeMemoryTarget> targets;
  ComponentViewDigest digest;
};

llvm::Expected<ResourceTimeLogicalMemoryBindingProjection>
projectResourceTimeLogicalMemoryBinding(
    const ::loom::mapping::FinalizedSystemMapping &mapping,
    ::dataflow::LogicalMemoryRootRef memory,
    llvm::ArrayRef<::dataflow::RootThreadLaunchRef> roots,
    const ArtifactStore &artifacts);

llvm::Expected<std::vector<std::uint8_t>>
canonicalResourceTimeMemoryTargetBytes(const ResourceTimeMemoryTarget &target);

/// Typed refusal of a transition proof. Every reason is an auditable
/// negative: the edge stays `ProofNotEstablished` and the reason names the
/// missing owner rather than a generic unsupported state.
enum class ResourceTimeTransitionRefusalReason : std::uint8_t {
  OrderedChannelState,
  DynamicWorkState,
  LogicalMemoryUnbound,
  LogicalMemoryExtentUnknown,
  LogicalMemoryCopyShapeUnsupported,
  LogicalMemoryReinitialized,
  HardwareBindingChanged,
  RuntimeTransitionCapabilityUnavailable,
  CompletionFrontierInadmissible,
};

llvm::StringRef resourceTimeTransitionRefusalReasonSpelling(
    ResourceTimeTransitionRefusalReason reason);

class ResourceTimeTransitionRefusal final
    : public llvm::ErrorInfo<ResourceTimeTransitionRefusal> {
public:
  static char ID;

  ResourceTimeTransitionRefusal(ResourceTimeTransitionRefusalReason reason,
                                std::string message)
      : reason_(reason), message_(std::move(message)) {}

  ResourceTimeTransitionRefusalReason reason() const { return reason_; }
  void log(llvm::raw_ostream &stream) const override;
  std::error_code convertToErrorCode() const override;

private:
  ResourceTimeTransitionRefusalReason reason_;
  std::string message_;
};

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
  /// Canonical roots already complete immediately before `trigger`. Together
  /// with the one completing active root, this is the exact completed subset
  /// at a completion-only safe point. Remaining roots may start under the
  /// child Mapping after the edge is selected.
  std::vector<::dataflow::RootThreadLaunchRef> completedBefore;
  /// Derived live-state correspondence of every logical memory that crosses
  /// the edge, one record per Canonical Dataflow memory root. Channel-typed
  /// and DynamicWork state cannot cross a verified edge.
  std::vector<ResourceTimeLogicalMemoryCorrespondence> logicalMemories;
  std::optional<ComponentViewDigest> resourceDeltaDigest;
  std::optional<ComponentViewDigest> configurationDeltaDigest;
  std::optional<ComponentViewDigest> routeDeltaDigest;
  /// Exact cost of programming the changed Deployment configuration, derived
  /// from exact changed words and the bound runtime provider's cost model.
  /// This remains distinct from live-state migration.
  std::optional<std::uint64_t> reprogrammingTimePicoseconds;
  /// Exact live-state migration cost: the sum over `logicalMemories`, using
  /// the bound runtime provider's copy model for `Copied` records. A
  /// retained-in-place edge establishes exact zero without fabricating work.
  std::optional<std::uint64_t> migrationTimePicoseconds;
  ResourceTimeTransitionStatus status =
      ResourceTimeTransitionStatus::ProofNotEstablished;
};

/// One exact child configuration image and the payload words that differ
/// from the same ConfigurationABI unit at the parent endpoint. Word ordinals
/// are interpreted only through the child image's ConfigurationABI layout.
struct ResourceTimeConfigurationImageDelta final {
  ArtifactRootReference childImage;
  std::vector<std::uint64_t> changedWordOrdinals;
};

/// One executable logical-memory copy rederived from a verified edge. The
/// provider reads the complete source object at commit time, so preparation
/// does not snapshot producer state before the safe point.
struct ResourceTimeLogicalMemoryCopyPlan final {
  ::dataflow::LogicalMemoryRootRef memory;
  std::uint64_t byteCount = 0;
  ResourceTimeMemoryTarget source;
  ResourceTimeMemoryTarget destination;
};

/// Removable execution projection for one exact preverified transition. The
/// persistent edge retains only correspondence digests and exact costs; the
/// provider target and changed-word lists are independently rederived from
/// its Mapping and Deployment endpoints.
struct ResourceTimeTransitionExecutionPlan final {
  std::vector<ResourceTimeConfigurationImageDelta> configurationImages;
  std::vector<ResourceTimeLogicalMemoryCopyPlan> logicalMemoryCopies;
  std::uint64_t reprogrammingTimePicoseconds = 0;
  std::uint64_t migrationTimePicoseconds = 0;
};

llvm::Expected<ResourceTimeTransitionExecutionPlan>
deriveResourceTimeTransitionExecutionPlan(
    const ResourceTimeTransition &transition, const ArtifactStore &artifacts,
    const BlobStore &blobs);

struct ResourceTimeTransitionSequence final {
  std::vector<ResourceTimeTransition> transitions;
};

/// One finite compiler-owned catalog of preverified Mapping states and safe-
/// point edges. Runtime may select only an edge in this catalog. Endpoint and
/// edge order is provenance, not identity or a runtime priority rule.
struct ResourceTimeTransitionGraph final {
  ResourceTimeTransitionEndpointReference entry;
  std::vector<ResourceTimeTransitionEndpointReference> endpoints;
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

/// Derives the resource, Deployment-configuration, and Mapping-route deltas
/// and the logical-memory live-state correspondence from exact endpoint
/// closures, marks the edge verified, and immediately replays the independent
/// closure verifier. The draft must carry an exact compiler-owned completion
/// safe point; the completion prefix may be nonterminal, but no in-flight
/// region may cross the edge. Logical memories bound to identical physical
/// targets at both endpoints are retained in place at exact zero cost;
/// relocated memories, channel-typed state, DynamicWork, and changed hardware
/// programming are typed `ResourceTimeTransitionRefusal` values. This
/// function neither discovers a Mapping nor invents live-state evidence.
llvm::Expected<ResourceTimeTransition>
finalizeResourceTimeTransition(ResourceTimeTransition draft,
                               const ArtifactStore &artifacts,
                               const BlobStore &blobs);

llvm::Error validateResourceTimeTransitionSequence(
    const ResourceTimeTransitionSequence &sequence);

/// Checks graph closure and exact endpoint membership without importing
/// Mapping or Deployment artifacts.
llvm::Error
validateResourceTimeTransitionGraph(const ResourceTimeTransitionGraph &graph);

/// Strictly imports every graph endpoint, proves one canonical root scope,
/// independently verifies every edge, and proves that every edge has a
/// monotonically realizable completion frontier from `entry`.
llvm::Error
verifyResourceTimeTransitionGraph(const ResourceTimeTransitionGraph &graph,
                                  const ArtifactStore &artifacts,
                                  const BlobStore &blobs);

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

/// Composes consecutive exact System transformation lineages. Entities,
/// transfer patterns, and imported Modules that disappear in the second
/// child are omitted; surviving entries remain one-to-one and are revalidated
/// against the original parent and final child Systems.
llvm::Expected<SystemExecutionBindingCorrespondence>
composeSystemExecutionBindingCorrespondence(
    const SystemExecutionBindingCorrespondence &first,
    const SystemExecutionBindingCorrespondence &second,
    const ArtifactStore &store);

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
