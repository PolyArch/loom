#ifndef LOOM_DSE_SPATIALMAPPINGEVALUATIONACQUISITION_H
#define LOOM_DSE_SPATIALMAPPINGEVALUATIONACQUISITION_H

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
spatialMappingEvaluationPromotionAcquisitionDescriptor();
llvm::Error registerSpatialMappingEvaluationPromotionAcquisition();

llvm::Expected<std::vector<PromotionAcquisitionInputBinding>>
bindSpatialMappingEvaluationPromotionInputs(
    llvm::ArrayRef<ArtifactRootReference> spatialMappings,
    llvm::ArrayRef<ArtifactRootReference> canonicalDataflowPrograms,
    const ArtifactRootReference &fabric, const ArtifactRootReference &workload,
    const ArtifactRootReference &runtimeInput);

llvm::Expected<ResolvedPromotionAcquisitionBinding>
resolveSpatialMappingEvaluationPromotionAcquisitionBinding(
    const ResolvedEvidenceObligationSetConfigView &config);

llvm::Expected<EvidenceObligationTemplate>
prepareCgraSimulationEvidenceObligationTemplate(
    const ArtifactRootReference &prototypeDataflow,
    const ArtifactRootReference &fabric,
    const ArtifactRootReference &prototypeSpatialMapping,
    const ArtifactRootReference &workload,
    const ArtifactRootReference &runtimeInput, const ResolvedConfig &config,
    const ArtifactStore &store, const BlobStore &blobs);

} // namespace loom::dse

#endif // LOOM_DSE_SPATIALMAPPINGEVALUATIONACQUISITION_H
