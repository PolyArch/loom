#ifndef LOOM_DSE_STRUCTUREDEVALUATIONACQUISITION_H
#define LOOM_DSE_STRUCTUREDEVALUATIONACQUISITION_H

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
structuredEvaluationPromotionAcquisitionDescriptor();
llvm::Error registerStructuredEvaluationPromotionAcquisition();

llvm::Expected<std::vector<PromotionAcquisitionInputBinding>>
bindStructuredEvaluationPromotionInputs(
    llvm::ArrayRef<ArtifactRootReference> structuredPrograms,
    const ArtifactRootReference &fabric, const ArtifactRootReference &workload,
    const ArtifactRootReference &runtimeInput);

llvm::Expected<ResolvedPromotionAcquisitionBinding>
resolveStructuredEvaluationPromotionAcquisitionBinding(
    const ResolvedEvidenceObligationSetConfigView &config);

llvm::Expected<EvidenceObligationTemplate>
prepareStructuredFabricAnalyticEvidenceObligationTemplate(
    const ArtifactRootReference &prototypeCandidate,
    const ArtifactRootReference &fabric, const ArtifactRootReference &workload,
    const ArtifactRootReference &runtimeInput, const ResolvedConfig &config,
    const ArtifactStore &store, const BlobStore &blobs);

llvm::Expected<EvidenceObligationTemplate>
prepareStructuredProgramFunctionalEvidenceObligationTemplate(
    const ArtifactRootReference &prototypeCandidate,
    const ArtifactRootReference &workload,
    const ArtifactRootReference &runtimeInput, const ResolvedConfig &config,
    const ArtifactStore &store, const BlobStore &blobs);

} // namespace loom::dse

#endif // LOOM_DSE_STRUCTUREDEVALUATIONACQUISITION_H
