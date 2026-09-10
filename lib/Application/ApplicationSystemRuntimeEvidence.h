#ifndef LOOM_APPLICATION_APPLICATIONSYSTEMRUNTIMEEVIDENCE_H
#define LOOM_APPLICATION_APPLICATIONSYSTEMRUNTIMEEVIDENCE_H

#include "Application/SystemQor.h"

namespace loom::application::detail {

struct ApplicationSystemRuntimeEvidenceContext final {
  ArtifactRootReference sourceProgram;
  ArtifactRootReference sourceWorkload;
  ArtifactRootReference sourceRuntimeInput;
  ArtifactRootReference mapping;
};

/// Re-derives both native activation inputs from the source invocation and
/// validates the exact selected Mapping, host-only target, and paired runs.
llvm::Expected<std::uint64_t> resolveApplicationSystemRuntimeEvidenceJoin(
    llvm::ArrayRef<ArtifactRootReference> evidence,
    const ApplicationSystemRuntimeEvidenceContext &context,
    const ArtifactStore &artifacts, const BlobStore &blobs);

struct ImportedApplicationSystemRun final {
  sim::CanonicalSimulationExecution execution;
  evaluation::EvaluationRequest request;
  ArtifactRootReference gem5Binding;
  ApplicationSystemRunMeasurement measurement;
};

/// Independently imports the complete native System execution and its Runtime
/// Evidence. The caller owns the expected Deployment/input pair and any
/// additional product oracle obligation.
llvm::Expected<ImportedApplicationSystemRun>
importApplicationSystemRun(const ApplicationSystemRunEvidence &roots,
                           const ArtifactRootReference &expectedDeployment,
                           const ArtifactRootReference &expectedWorkload,
                           const ArtifactRootReference &expectedInput,
                           const evaluation::CaseArtifactResolution &resolution,
                           const ArtifactStore &artifacts,
                           const BlobStore &blobs);

/// Requires one exact machine and observation contract, a host-only baseline,
/// and matching source computation boundaries. Returns whether their complete
/// functional observations agree; malformed observation pairs remain errors.
llvm::Expected<bool>
compareApplicationSystemRuns(const ImportedApplicationSystemRun &host,
                             const ImportedApplicationSystemRun &candidate);

} // namespace loom::application::detail

#endif // LOOM_APPLICATION_APPLICATIONSYSTEMRUNTIMEEVIDENCE_H
