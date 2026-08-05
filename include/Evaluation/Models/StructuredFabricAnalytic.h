#ifndef LOOM_EVALUATION_MODELS_STRUCTUREDFABRICANALYTIC_H
#define LOOM_EVALUATION_MODELS_STRUCTUREDFABRICANALYTIC_H

#include "Common/Artifact.h"
#include "Evaluation/Case.h"
#include "Evaluation/Request.h"
#include "Frontend/IR/StructuredProgramArtifact.h"
#include "Frontend/Lowering/CanonicalDataflowLowering.h"

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
struct StructuredBlockActivityLineage;
class StructuredProgramCandidate;
} // namespace loom::frontend

namespace loom::sim {
class CanonicalSimulationRuntimeInput;
class CanonicalSimulationWorkload;
struct NativeStructuredProgramObservations;
} // namespace loom::sim

namespace loom::evaluation::models {

struct PreparedStructuredFabricEvaluation final {
  EvaluationRequest request;
  CaseArtifactResolution resolution;
  CaseSubjectRoleRef candidateRole;
};

/// One invocation-local resolution of the exact source inputs, Fabric closure,
/// and finite Structured candidate set. It is derived from published roots,
/// carries no semantic identity, and may be discarded and rebuilt at any time.
class StructuredFabricAnalyticRequestContext final {
public:
  const ArtifactRootReference &fabric() const { return fabric_; }
  const ArtifactRootReference &workload() const { return workload_; }
  const ArtifactRootReference &runtimeInput() const { return runtimeInput_; }
  const CaseArtifactResolution &caseResolution() const { return resolution_; }

private:
  StructuredFabricAnalyticRequestContext(
      std::vector<ArtifactRootReference> candidates,
      ArtifactRootReference fabric, ArtifactRootReference workload,
      ArtifactRootReference runtimeInput, CaseArtifactResolution resolution)
      : candidates_(std::move(candidates)), fabric_(std::move(fabric)),
        workload_(std::move(workload)), runtimeInput_(std::move(runtimeInput)),
        resolution_(std::move(resolution)) {}

  std::vector<ArtifactRootReference> candidates_;
  ArtifactRootReference fabric_;
  ArtifactRootReference workload_;
  ArtifactRootReference runtimeInput_;
  CaseArtifactResolution resolution_;

  friend llvm::Expected<StructuredFabricAnalyticRequestContext>
  prepareStructuredFabricAnalyticInvocation(
      llvm::ArrayRef<ArtifactRootReference>, const ArtifactRootReference &,
      const ArtifactRootReference &, const ArtifactRootReference &,
      const ArtifactStore &);
  friend llvm::Expected<PreparedStructuredFabricEvaluation>
  prepareStructuredFabricEvaluation(
      const ArtifactRootReference &,
      const StructuredFabricAnalyticRequestContext &, const ResolvedConfig &,
      const ArtifactStore &);
};

/// Registers the exact low-fidelity StructuredProgram/Fabric analytic model.
/// Repeated registration in one process is a no-op.
llvm::Error registerStructuredFabricAnalyticModel();

EvaluationModelDescriptorRef structuredFabricAnalyticModelDescriptorRef();
CaseSubjectRoleRef structuredFabricAnalyticCandidateRole();
CaseSubjectRoleRef structuredFabricAnalyticFabricRole();

/// Exact decimal quantum used by this model for one supported MetricKind.
/// DSE objective normalization consumes this projection rather than copying
/// model-private unit scaling.
llvm::Expected<std::int64_t>
structuredFabricAnalyticMetricQuantumBase10Exponent(MetricKind metric);

/// Resolves the already published candidates, immutable workload, runtime
/// input, and Fabric closure shared by one finite central DSE candidate set.
/// Every candidate remains identified only by its exact ArtifactRootReference.
llvm::Expected<StructuredFabricAnalyticRequestContext>
prepareStructuredFabricAnalyticInvocation(
    llvm::ArrayRef<ArtifactRootReference> structuredPrograms,
    const ::loom::ArtifactRootReference &fabric,
    const ::loom::ArtifactRootReference &workload,
    const ::loom::ArtifactRootReference &runtimeInput,
    const ::loom::ArtifactStore &artifactStore);

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

/// Constructs the same exact Request using roots already resolved for this
/// invocation. A candidate outside the finite resolved set is rejected.
llvm::Expected<PreparedStructuredFabricEvaluation>
prepareStructuredFabricEvaluation(
    const ::loom::ArtifactRootReference &structuredProgram,
    const StructuredFabricAnalyticRequestContext &invocation,
    const ::loom::ResolvedConfig &config,
    const ::loom::ArtifactStore &artifactStore);

/// Invocation-local typed input used to derive workload-aware metrics without
/// adding a profile Artifact or copying source coverage into a candidate.
struct StructuredFabricAnalyticInvocation final {
  const ::loom::ArtifactRootReference &workload;
  const ::loom::ArtifactRootReference &runtimeInput;
  const ::loom::sim::CanonicalSimulationWorkload &simulationWorkload;
  const ::loom::sim::CanonicalSimulationRuntimeInput &simulationRuntimeInput;
  const ::loom::frontend::StructuredProgramCandidate &sourceProgram;
  const ::loom::sim::NativeStructuredProgramObservations &sourceObservations;
};

/// One exact complete-candidate projection within the invocation. The
/// Dataflow owner and every Structured-to-graph relation are absent together
/// for a candidate with no Spatial ownership. A nonempty
/// `blockActivityLineage` mechanically projects an activity-preserving parent
/// execution onto a just-materialized child. A transformation that repartitions
/// dynamic activity publishes no lineage and must supply `observations`, the
/// removable result of executing the exact candidate. The provider derives
/// exact observations on a cache miss.
struct StructuredFabricAnalyticCandidateProjection final {
  const ::loom::frontend::StructuredProgramCandidate &candidate;
  const ::dataflow::CanonicalDataflowArtifact *canonicalDataflow = nullptr;
  llvm::ArrayRef<::loom::lowering::StructuredSpatialGraphProjection>
      spatialGraphs = {};
  llvm::ArrayRef<::loom::frontend::StructuredBlockActivityLineage>
      blockActivityLineage = {};
  const ::loom::sim::NativeStructuredProgramObservations *observations =
      nullptr;
};

/// One descriptor-owned, invocation-local applicability projection for an
/// exact Structured ownership scope. This is derived from the source program
/// and its exact native workload observations; it is neither persistent nor a
/// second source-profile authority.
struct StructuredScopeActivityProjection final {
  ::loom::frontend::StructuredEntityRef scope;
  std::uint64_t dynamicActivations = 0;
  std::uint64_t dynamicLeafExecutions = 0;
};

/// Projects dynamic activation and executable-leaf counts for exact source
/// scopes in caller order. The projection validates the complete
/// block-observation correspondence once and performs no candidate
/// materialization, ranking, or target admission.
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
