#ifndef LOOM_DSE_RESOURCETIMEADJACENTMAPPINGSELECTION_H
#define LOOM_DSE_RESOURCETIMEADJACENTMAPPINGSELECTION_H

#include "DSE/JointHardwareReopen.h"

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
