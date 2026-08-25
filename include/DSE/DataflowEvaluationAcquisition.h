#ifndef LOOM_DSE_DATAFLOWEVALUATIONACQUISITION_H
#define LOOM_DSE_DATAFLOWEVALUATIONACQUISITION_H

#include "DSE/EvidenceObligationSetConfig.h"
#include "DSE/PromotionAcquisition.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <vector>

namespace loom {
class ArtifactStore;
class BlobStore;
struct ResolvedConfig;
} // namespace loom

namespace loom::dse {

const PromotionAcquisitionDescriptor &
dataflowEvaluationPromotionAcquisitionDescriptor();
llvm::Error registerDataflowEvaluationPromotionAcquisition();

llvm::Expected<std::vector<PromotionAcquisitionInputBinding>>
bindDataflowEvaluationPromotionInputs(
    llvm::ArrayRef<ArtifactRootReference> canonicalDataflowPrograms,
    const ArtifactRootReference &structuredParent,
    const ArtifactRootReference &fabric, const ArtifactRootReference &workload,
    const ArtifactRootReference &runtimeInput);

llvm::Expected<ResolvedPromotionAcquisitionBinding>
resolveDataflowEvaluationPromotionAcquisitionBinding(
    const ResolvedEvidenceObligationSetConfigView &config);

llvm::Expected<EvidenceObligationTemplate>
prepareCanonicalDataflowFabricAnalyticEvidenceObligationTemplate(
    const ArtifactRootReference &prototypeCandidate,
    const ArtifactRootReference &fabric, const ResolvedConfig &config,
    const ArtifactStore &store, const BlobStore &blobs);

llvm::Expected<EvidenceObligationTemplate>
prepareCanonicalDataflowFunctionalEvidenceObligationTemplate(
    const ArtifactRootReference &prototypeCandidate,
    const ArtifactRootReference &structuredParent,
    const ArtifactRootReference &workload,
    const ArtifactRootReference &runtimeInput, const ResolvedConfig &config,
    const ArtifactStore &store, const BlobStore &blobs);

} // namespace loom::dse

#endif // LOOM_DSE_DATAFLOWEVALUATIONACQUISITION_H
