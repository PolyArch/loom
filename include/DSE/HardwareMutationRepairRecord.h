#ifndef LOOM_DSE_HARDWAREMUTATIONREPAIRRECORD_H
#define LOOM_DSE_HARDWAREMUTATIONREPAIRRECORD_H

#include "Common/Artifact.h"
#include "DSE/HardwareDecision.h"
#include "DSE/JointDesignExploration.h"
#include "DSE/JointHardwareReopen.h"
#include "DSE/JointMappingMigration.h"
#include "Mapping/Artifact/SystemMappingArtifact.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

namespace loom {
class ArtifactStore;
} // namespace loom

namespace loom::dse {

inline constexpr ArtifactSchemaDescriptor hardwareMutationRepairRecordSchema{
    "loom.dse.hardware_mutation_repair_record", SchemaVersion{2, 0}};

/// The affected Mapping cones of one typed hardware mutation component.
/// Parent and Module correspondence are derived on strict import from the
/// canonical candidate-decision lineage rather than serialized twice.
struct HardwareMutationImpactRecord final {
  ArtifactRootReference parent;
  std::optional<ArtifactRootReference> child;
  std::vector<loom::fabric::FabricModuleEntityCorrespondence> moduleEntities;
  HardwareMutationFamily family = HardwareMutationFamily::SpatialTopology;
  HardwareMutationLocality locality = HardwareMutationLocality::Unchanged;
  TechMappingImpactProjection tech;
  SpatialMappingImpactProjection spatial;
  SystemMappingImpactProjection system;
};

/// Provider dispatch, journal replay, wall time, and independent verifier
/// accounting of one side of the paired repair.
struct HardwareMutationRepairSideRecord final {
  std::vector<ArtifactRootReference> mappings;
  std::uint64_t techMappingInvocations = 0;
  std::uint64_t spatialPnrInvocations = 0;
  std::uint64_t systemPnrInvocations = 0;
  std::uint64_t techMappingDispatches = 0;
  std::uint64_t spatialPnrDispatches = 0;
  std::uint64_t systemPnrDispatches = 0;
  std::uint64_t techMappingJournalReplays = 0;
  std::uint64_t spatialPnrJournalReplays = 0;
  std::uint64_t systemPnrJournalReplays = 0;
  std::uint64_t executionWallTimeNanoseconds = 0;
  mapping::SystemMappingImportSessionStatistics verification;
};

struct HardwareMutationRepairQualityObservation final {
  ArtifactRootReference candidate;
  std::vector<std::uint64_t> objectiveCodes;
  std::optional<JointDesignQualityIncompleteReason> incompleteReason;
};

/// Durable per-family evidence of one typed hardware mutation repair: the
/// exact parent Mapping and System, the child System, every canonical
/// candidate-decision edge and its derived component impact cone, the typed
/// reuse dispositions with their rebase accounting and failures, the
/// independently verified cold and preserve-first Mapping roots with their
/// dispatch and verifier accounting, and the quality observations of the
/// preserve-first execution. The executor publishes it; a test summary or
/// debug event is not a substitute.
struct HardwareMutationRepairRecord final {
  ArtifactRootReference parentMapping;
  ArtifactRootReference parentSystem;
  ArtifactRootReference childSystem;
  std::vector<HardwareMutationDecisionLineage> decisionLineage;
  std::vector<HardwareMutationImpactRecord> impacts;
  JointMappingReuseDisposition mappingReuseDisposition =
      JointMappingReuseDisposition::ColdFallback;
  JointSystemMappingReuseDisposition systemMappingReuseDisposition =
      JointSystemMappingReuseDisposition::ColdFallback;
  std::vector<JointMappingRebaseFailure> rebaseFailures;
  JointMappingRebaseAccounting accounting;
  std::optional<HardwareMutationRepairSideRecord> cold;
  HardwareMutationRepairSideRecord incremental;
  std::vector<HardwareMutationRepairQualityObservation> qualityObservations;
};

class FinalizedHardwareMutationRepairRecord final {
public:
  const ArtifactRootReference &reference() const { return reference_; }
  const HardwareMutationRepairRecord &record() const { return record_; }
  const CanonicalSemanticBytes &canonicalBytes() const {
    return canonicalBytes_;
  }

private:
  FinalizedHardwareMutationRepairRecord(ArtifactRootReference reference,
                                        HardwareMutationRepairRecord record,
                                        CanonicalSemanticBytes canonicalBytes)
      : reference_(std::move(reference)), record_(std::move(record)),
        canonicalBytes_(std::move(canonicalBytes)) {}

  ArtifactRootReference reference_;
  HardwareMutationRepairRecord record_;
  CanonicalSemanticBytes canonicalBytes_;

  friend llvm::Expected<FinalizedHardwareMutationRepairRecord>
  publishHardwareMutationRepairRecord(const JointHardwareMutationRepair &,
                                      const ArtifactStore &);
  friend llvm::Expected<FinalizedHardwareMutationRepairRecord>
  importHardwareMutationRepairRecord(const ArtifactRootReference &,
                                     const ArtifactStore &);
};

/// Projects one executed repair onto its durable record and publishes it.
/// The published bytes are re-imported strictly before the reference is
/// returned.
llvm::Expected<FinalizedHardwareMutationRepairRecord>
publishHardwareMutationRepairRecord(const JointHardwareMutationRepair &repair,
                                    const ArtifactStore &artifacts);

llvm::Expected<FinalizedHardwareMutationRepairRecord>
importHardwareMutationRepairRecord(const ArtifactRootReference &reference,
                                   const ArtifactStore &artifacts);

} // namespace loom::dse

#endif // LOOM_DSE_HARDWAREMUTATIONREPAIRRECORD_H
