#ifndef LOOM_RUNTIME_GEM5SYSTEMEXECUTION_H
#define LOOM_RUNTIME_GEM5SYSTEMEXECUTION_H

#include "Evaluation/ModelProvider.h"

namespace loom::runtime {

/// Prepares the exact gem5 System invocation selected by model kind 17, 18,
/// or 19. The model descriptor remains the sole engine selector.
llvm::Expected<evaluation::EvaluationModelProviderPreparation>
prepareGem5SystemInvocation(
    const evaluation::EvaluationRequest &request,
    const evaluation::CaseArtifactResolution &resolution,
    const ArtifactStore &artifacts, const BlobStore &blobs,
    const external_tool::ExternalToolPreparationContext &context);

/// Strictly imports one completed gem5 invocation and publishes its
/// Deployment-owned SystemSimulationExecution.
llvm::Expected<evaluation::EvaluationModelResult>
importGem5SystemInvocation(
    const evaluation::EvaluationRequest &request,
    const evaluation::CaseArtifactResolution &resolution,
    const external_tool::PreparedExternalToolInvocation &prepared,
    const ArtifactStore &artifacts, const BlobStore &blobs);

} // namespace loom::runtime

#endif // LOOM_RUNTIME_GEM5SYSTEMEXECUTION_H
