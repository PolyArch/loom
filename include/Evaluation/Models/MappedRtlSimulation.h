#ifndef LOOM_EVALUATION_MODELS_MAPPEDRTLSIMULATION_H
#define LOOM_EVALUATION_MODELS_MAPPEDRTLSIMULATION_H

#include "Evaluation/Models/MappedRtlSimulationConfig.h"
#include "Evaluation/Request.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

namespace loom {
class ArtifactStore;
class BlobStore;
} // namespace loom

namespace loom::evaluation::models {

inline constexpr llvm::StringLiteral mappedRtlSimulatorSemanticIdentity =
    "loom.mapped_rtl.simulator.v2";

struct MappedRtlSimulationConfiguration final {
  MappedRtlSimulatorBinding providerBinding;
};

/// Registers the production Evaluation catalog that owns case kind 12 and
/// model kind 21. Repeated registration is idempotent.
llvm::Error registerMappedRtlSimulationModel();

EvaluationModelDescriptorRef mappedRtlSimulatorModelDescriptorRef();
CaseSubjectRoleRef mappedRtlHardwareImplementationSubjectRole();
CaseSubjectRoleRef mappedRtlDeploymentSubjectRole();

/// The exact descriptor-owned ResolvedConfig view contract used by the
/// production model registry.
const ResolvedModelConfigViewContract &mappedRtlSimulationConfigViewContract();

/// Validates one exact request and projects the sole provider configuration.
llvm::Expected<MappedRtlSimulationConfiguration>
projectMappedRtlSimulationConfiguration(
    const EvaluationRequest &request, const CaseArtifactResolution &resolution,
    const ArtifactStore &artifacts, const BlobStore &blobs);

/// Projects the provider configuration after the Evaluation facade has
/// verified the exact Request against the same resolution and stores.
llvm::Expected<MappedRtlSimulationConfiguration>
projectVerifiedMappedRtlSimulationConfiguration(
    const EvaluationRequest &request);

} // namespace loom::evaluation::models

#endif // LOOM_EVALUATION_MODELS_MAPPEDRTLSIMULATION_H
