#ifndef LOOM_EVALUATION_MODELS_STRUCTUREDFABRICANALYTIC_H
#define LOOM_EVALUATION_MODELS_STRUCTUREDFABRICANALYTIC_H

#include "Common/Artifact.h"
#include "Evaluation/Case.h"
#include "Evaluation/Request.h"
#include "Frontend/IR/StructuredProgramArtifact.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>
#include <vector>

namespace loom {
class ArtifactStore;
struct ResolvedConfig;
} // namespace loom

namespace dataflow {
class CanonicalDataflowArtifact;
}

namespace loom::fabric {
class FinalizedFabricRoot;
}

namespace loom::frontend {
class StructuredProgramCandidate;
}

namespace loom::sim {
struct NativeStructuredProgramObservations;
}

namespace loom::evaluation::models {

struct PreparedStructuredFabricEvaluation final {
  EvaluationRequest request;
  CaseArtifactResolution resolution;
  CaseSubjectRoleRef candidateRole;
};

/// Registers the exact low-fidelity StructuredProgram/Fabric analytic model.
/// Repeated registration in one process is a no-op.
llvm::Error registerStructuredFabricAnalyticModel();

/// Constructs and publishes the complete low-confidence metric request for the
/// exact S/F pair. The returned resolution is the complete transient case
/// closure used to execute and validate the resulting Evidence.
llvm::Expected<PreparedStructuredFabricEvaluation>
prepareStructuredFabricEvaluation(
    const ::loom::ArtifactRootReference &structuredProgram,
    const ::loom::ArtifactRootReference &fabric,
    const ::loom::ArtifactRootReference &workload,
    const ::loom::ArtifactRootReference &runtimeInput,
    const ::loom::ResolvedConfig &config,
    const ::loom::ArtifactStore &artifactStore);

/// Invocation-local typed input used to derive workload-aware metrics without
/// adding a profile Artifact or copying source coverage into a candidate.
struct StructuredFabricAnalyticInvocation final {
  const ::loom::ArtifactRootReference &workload;
  const ::loom::ArtifactRootReference &runtimeInput;
  const ::loom::frontend::StructuredProgramCandidate &sourceProgram;
  const ::loom::sim::NativeStructuredProgramObservations &sourceObservations;
};

/// One exact candidate projection within the invocation. `sourceScope` and
/// `canonicalDataflow` are both absent only for the unmodified source
/// baseline; a Spatial candidate requires both.
struct StructuredFabricAnalyticCandidateProjection final {
  const ::loom::frontend::StructuredProgramCandidate &candidate;
  const ::dataflow::CanonicalDataflowArtifact *canonicalDataflow = nullptr;
  std::optional<::loom::frontend::StructuredEntityRef> sourceScope;
};

/// One descriptor-owned, invocation-local applicability projection for an
/// exact Structured ownership scope. This is derived from the source program
/// and its exact native workload observations; it is neither persistent nor a
/// second source-profile authority.
struct StructuredScopeActivityProjection final {
  ::loom::frontend::StructuredEntityRef scope;
  std::uint64_t dynamicActivations = 0;
};

/// Projects dynamic activation counts for exact source scopes in caller order.
/// The projection validates the complete block-observation correspondence once
/// and performs no candidate materialization, ranking, or target admission.
llvm::Expected<std::vector<StructuredScopeActivityProjection>>
projectStructuredScopeActivity(
    const ::loom::frontend::StructuredProgramCandidate &sourceProgram,
    const ::loom::sim::NativeStructuredProgramObservations &sourceObservations,
    llvm::ArrayRef<::loom::frontend::StructuredEntityRef> scopes);

/// Primes the removable model-result cache from already finalized owner views.
/// The full provider remains the oracle on a miss; this function only avoids
/// re-importing and mechanically re-lowering an exact candidate that the
/// caller has just finalized.
llvm::Error primeStructuredFabricAnalyticResult(
    const ::loom::ArtifactRootReference &structuredProgramReference,
    const StructuredFabricAnalyticCandidateProjection &candidate,
    const StructuredFabricAnalyticInvocation &invocation,
    const ::loom::fabric::FinalizedFabricRoot &fabric,
    const ::loom::ResolvedConfig &config,
    const ::loom::ArtifactStore &artifactStore);

} // namespace loom::evaluation::models

#endif // LOOM_EVALUATION_MODELS_STRUCTUREDFABRICANALYTIC_H
