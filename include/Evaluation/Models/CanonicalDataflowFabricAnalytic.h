#ifndef LOOM_EVALUATION_MODELS_CANONICALDATAFLOWFABRICANALYTIC_H
#define LOOM_EVALUATION_MODELS_CANONICALDATAFLOWFABRICANALYTIC_H

#include "Common/Artifact.h"
#include "Evaluation/Case.h"
#include "Evaluation/Request.h"

#include "llvm/Support/Error.h"

#include <cstdint>

namespace loom {
class ArtifactStore;
struct ResolvedConfig;
} // namespace loom

namespace loom::evaluation::models {

struct PreparedCanonicalDataflowFabricEvaluation final {
  EvaluationRequest request;
  CaseArtifactResolution resolution;
  CaseSubjectRoleRef candidateRole;
};

llvm::Error registerCanonicalDataflowFabricAnalyticModel();

EvaluationModelDescriptorRef
canonicalDataflowFabricAnalyticModelDescriptorRef();
CaseSubjectRoleRef canonicalDataflowFabricAnalyticCandidateRole();
CaseSubjectRoleRef canonicalDataflowFabricAnalyticFabricRole();

llvm::Expected<std::int64_t>
canonicalDataflowFabricAnalyticMetricQuantumBase10Exponent(MetricKind metric);

llvm::Expected<CaseArtifactResolution>
resolveCanonicalDataflowFabricEvaluationCase(
    const ::loom::ArtifactRootReference &canonicalDataflow,
    const ::loom::ArtifactRootReference &fabric,
    const ::loom::ArtifactStore &artifactStore);

llvm::Expected<CaseArtifactResolution>
resolveCanonicalDataflowFabricEvaluationCases(
    llvm::ArrayRef<::loom::ArtifactRootReference> canonicalDataflowPrograms,
    const ::loom::ArtifactRootReference &fabric,
    const ::loom::ArtifactStore &artifactStore);

llvm::Expected<PreparedCanonicalDataflowFabricEvaluation>
prepareCanonicalDataflowFabricEvaluation(
    const ::loom::ArtifactRootReference &canonicalDataflow,
    const ::loom::ArtifactRootReference &fabric,
    const ::loom::ResolvedConfig &config,
    const ::loom::ArtifactStore &artifactStore);

} // namespace loom::evaluation::models

#endif // LOOM_EVALUATION_MODELS_CANONICALDATAFLOWFABRICANALYTIC_H
