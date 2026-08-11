#ifndef LOOM_DSE_HARDWAREIMPLEMENTATIONEVALUATIONACQUISITION_H
#define LOOM_DSE_HARDWAREIMPLEMENTATIONEVALUATIONACQUISITION_H

#include "DSE/EvidenceObligationSetConfig.h"
#include "DSE/PromotionAcquisition.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <optional>
#include <vector>

namespace loom {
class ArtifactStore;
class BlobStore;
struct ResolvedConfig;
} // namespace loom

namespace loom::dse {

inline constexpr PromotionAcquisitionKind
    hardwareImplementationEvaluationPromotionAcquisitionKind(3);

const PromotionAcquisitionDescriptor &
hardwareImplementationEvaluationPromotionAcquisitionDescriptor();
llvm::Error registerHardwareImplementationEvaluationPromotionAcquisition();

llvm::Expected<std::vector<PromotionAcquisitionInputBinding>>
bindHardwareImplementationEvaluationPromotionInputs(
    llvm::ArrayRef<ArtifactRootReference> hardwareImplementations);

llvm::Expected<ResolvedPromotionAcquisitionBinding>
resolveHardwareImplementationEvaluationPromotionAcquisitionBinding(
    const ResolvedEvidenceObligationSetConfigView &config);

llvm::Expected<EvidenceObligationTemplate>
prepareOpenRoadStaticFpaEvidenceObligationTemplate(
    const ArtifactRootReference &prototypeHardwareImplementation,
    llvm::ArrayRef<evaluation::EvaluationCondition> conditions,
    llvm::ArrayRef<evaluation::MetricKind> metrics,
    std::optional<CalibrationPartitionRole> calibrationPartitionRole,
    const ResolvedConfig &config, const ArtifactStore &artifactStore,
    const BlobStore &blobStore);

} // namespace loom::dse

#endif // LOOM_DSE_HARDWAREIMPLEMENTATIONEVALUATIONACQUISITION_H
