#ifndef LOOM_DSE_RESOURCETIMEADJACENTMAPPINGSELECTION_H
#define LOOM_DSE_RESOURCETIMEADJACENTMAPPINGSELECTION_H

#include "DSE/JointHardwareReopen.h"

#include <cstdint>
#include <optional>
#include <vector>

namespace loom::mapping {
class SystemMappingView;
} // namespace loom::mapping

namespace loom::dse::joint_reopen_detail {

struct ResourceTimePartitionMappingSelection final {
  std::optional<ArtifactRootReference> mapping;
  std::optional<ResourceTimeSpectrumFunnelResult> spectrum;
  std::vector<ArtifactRootReference> eligibleMappings;
  std::vector<DsePlanIncompleteReason> executionIncompleteReasons;
};

/// The eligible Mapping ledger of one adjacent-repair side: the generated
/// Mappings that realize the exact partition intent and, when a parent is
/// required, preserve its cone-external System selections. The counters are
/// the structural reason an empty ledger is empty.
struct ResourceTimePartitionEligibility final {
  std::vector<ArtifactRootReference> eligibleMappings;
  std::uint64_t partitionMatchingCandidates = 0;
  std::uint64_t preservationMatchingCandidates = 0;
};

/// Projects that ledger. This is the sole owner of eligibility, so a
/// transition that reports a selected Mapping and a transition that reports
/// only its candidate ledger agree by construction. It imports Mappings and
/// compares partitions; it runs no Mapping verifier.
llvm::Expected<ResourceTimePartitionEligibility>
projectResourceTimePartitionEligibility(
    const JointDesignExecution &execution,
    const ArtifactRootReference &dataflowReference,
    const ArtifactRootReference &fabricReference,
    llvm::ArrayRef<pnr::SystemBindingPartitionIntent> partitions,
    llvm::ArrayRef<::dataflow::RootThreadLaunchRef> reopenedRoots,
    const mapping::SystemMappingView *requiredParentMapping,
    const ArtifactStore &artifacts);

llvm::Expected<ResourceTimePartitionMappingSelection>
selectResourceTimePartitionMapping(
    JointDesignExecution &execution,
    const ArtifactRootReference &dataflowReference,
    const ArtifactRootReference &fabricReference,
    llvm::ArrayRef<pnr::SystemBindingPartitionIntent> partitions,
    llvm::ArrayRef<::dataflow::RootThreadLaunchRef> reopenedRoots,
    const mapping::SystemMappingView *requiredParentMapping,
    llvm::ArrayRef<DsePlanIncompleteReason> prerequisiteIncompleteReasons,
    PreMappingSpectrumEndpoint spectrumEndpoint,
    JointResourceTimeMappingRepairSide side,
    JointResourceTimeMappingVerifier mappingVerifier,
    const ArtifactStore &artifacts);

} // namespace loom::dse::joint_reopen_detail

#endif // LOOM_DSE_RESOURCETIMEADJACENTMAPPINGSELECTION_H
