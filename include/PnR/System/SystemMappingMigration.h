#ifndef LOOM_PNR_SYSTEM_SYSTEMMAPPINGMIGRATION_H
#define LOOM_PNR_SYSTEM_SYSTEMMAPPINGMIGRATION_H

#include "Common/Artifact.h"
#include "Dataflow/IR/DataflowStructuralRefs.h"
#include "Fabric/Identity/FabricRefs.h"
#include "Mapping/Artifact/SystemMappingArtifact.h"
#include "Mapping/Artifact/SystemMappingHardwareDemand.h"
#include "Mapping/Artifact/SystemPresburger.h"
#include "PnR/PnrIndex.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <variant>
#include <vector>

namespace loom::pnr {

class FrozenSystemPnrProblem;

inline constexpr ArtifactSchemaDescriptor
    systemMappingCheckpointMigrationSeedArtifactSchema{
        "loom.pnr.system_mapping_checkpoint_migration_seed",
        SchemaVersion{2, 0}};

struct SystemAccCoreCorrespondence final {
  ::loom::fabric::AccCoreOccurrenceRef parent;
  ::loom::fabric::AccCoreOccurrenceRef child;
};

/// Invocation-local proof input for rebasing execution bindings. Hardware DSE
/// derives this relation from exact typed parent/child lineage. PnR validates
/// every selected target against the frozen child problem before use.
class SystemExecutionBindingCorrespondence final {
public:
  static llvm::Expected<SystemExecutionBindingCorrespondence>
  get(ArtifactRootReference parentSystem, ArtifactRootReference childSystem,
      std::vector<SystemAccCoreCorrespondence> accCores,
      const ArtifactStore &store);

  const ArtifactRootReference &parentSystem() const { return parentSystem_; }
  const ArtifactRootReference &childSystem() const { return childSystem_; }
  llvm::ArrayRef<SystemAccCoreCorrespondence> accCores() const {
    return accCores_;
  }

private:
  SystemExecutionBindingCorrespondence(
      ArtifactRootReference parentSystem, ArtifactRootReference childSystem,
      std::vector<SystemAccCoreCorrespondence> accCores)
      : parentSystem_(std::move(parentSystem)),
        childSystem_(std::move(childSystem)), accCores_(std::move(accCores)) {}

  ArtifactRootReference parentSystem_;
  ArtifactRootReference childSystem_;
  std::vector<SystemAccCoreCorrespondence> accCores_;
};

struct FinalizedSystemMappingMigrationSeed final {
  const ::loom::mapping::FinalizedSystemMapping &parentMapping;
  const SystemExecutionBindingCorrespondence &correspondence;
};

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
  ::loom::fabric::AccCoreOccurrenceRef reopenedParentAccCore() const {
    return reopenedParentAccCore_;
  }

private:
  FinalizedSystemMappingCheckpointMigrationSeed(
      ArtifactRootReference reference,
      ::loom::mapping::FinalizedSystemExecutionBindingCheckpoint checkpoint,
      SystemExecutionBindingCorrespondence correspondence,
      ::loom::fabric::AccCoreOccurrenceRef reopenedParentAccCore)
      : reference_(std::move(reference)), checkpoint_(std::move(checkpoint)),
        correspondence_(std::move(correspondence)),
        reopenedParentAccCore_(reopenedParentAccCore) {}

  ArtifactRootReference reference_;
  ::loom::mapping::FinalizedSystemExecutionBindingCheckpoint checkpoint_;
  SystemExecutionBindingCorrespondence correspondence_;
  ::loom::fabric::AccCoreOccurrenceRef reopenedParentAccCore_;

  friend llvm::Expected<FinalizedSystemMappingCheckpointMigrationSeed>
  finalizeSystemMappingCheckpointMigrationSeed(
      const ArtifactRootReference &,
      const SystemExecutionBindingCorrespondence &,
      ::loom::fabric::AccCoreOccurrenceRef, const ArtifactStore &);
  friend llvm::Expected<FinalizedSystemMappingCheckpointMigrationSeed>
  importSystemMappingCheckpointMigrationSeed(const ArtifactRootReference &,
                                             const ArtifactStore &);
};

llvm::Expected<FinalizedSystemMappingCheckpointMigrationSeed>
finalizeSystemMappingCheckpointMigrationSeed(
    const ArtifactRootReference &checkpoint,
    const SystemExecutionBindingCorrespondence &correspondence,
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
  ChildInitializerRejected,
};

struct SystemMappingMigrationFallback final {
  SystemMappingMigrationFallbackReason reason;
};

struct SystemMappingMigrationProjection final {
  std::vector<PnrIndex> fixedChoices;
  std::vector<PnrIndex> releasedChoices;
  std::uint64_t preservedThreadBindings = 0;
  std::uint64_t preservedGraphBindings = 0;
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
