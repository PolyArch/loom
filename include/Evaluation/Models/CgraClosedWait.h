#ifndef LOOM_EVALUATION_MODELS_CGRACLOSEDWAIT_H
#define LOOM_EVALUATION_MODELS_CGRACLOSEDWAIT_H

#include "Common/Artifact.h"
#include "Evaluation/Finding.h"
#include "Simulator/CgraClosedWaitCertificate.h"

#include "llvm/Support/Error.h"

#include <string>
#include <utility>

namespace loom {
class ArtifactStore;
class BlobStore;
} // namespace loom

namespace loom::evaluation {
class EvaluationEvidence;
}

namespace loom::evaluation::models {

inline constexpr FindingKind CgraClosedWait{1};

/// Registers the sole Finding owner for a proven CGRA closed-wait terminal.
llvm::Error registerCgraClosedWaitFinding();

/// Strictly imported and deterministically replayed Evidence carrier for one
/// CGRA closed-wait terminal. The certificate is reachable only through the
/// Evidence's own output binding and Halted terminal; callers cannot replace
/// it independently or author a structurally valid witness in lieu of a model
/// execution.
class VerifiedCgraClosedWaitEvidence final {
public:
  const ArtifactRootReference &evidence() const { return evidence_; }
  const ArtifactRootReference &request() const { return request_; }
  const ArtifactRootReference &execution() const { return execution_; }
  const sim::CgraClosedWaitCertificate &certificate() const {
    return certificate_;
  }
  const sim::CgraClosedWaitCertificateDigest &certificateDigest() const {
    return certificateDigest_;
  }

private:
  VerifiedCgraClosedWaitEvidence(
      ArtifactRootReference evidence, ArtifactRootReference request,
      ArtifactRootReference execution,
      sim::CgraClosedWaitCertificate certificate,
      sim::CgraClosedWaitCertificateDigest certificateDigest)
      : evidence_(std::move(evidence)), request_(std::move(request)),
        execution_(std::move(execution)), certificate_(std::move(certificate)),
        certificateDigest_(std::move(certificateDigest)) {}

  ArtifactRootReference evidence_;
  ArtifactRootReference request_;
  ArtifactRootReference execution_;
  sim::CgraClosedWaitCertificate certificate_;
  sim::CgraClosedWaitCertificateDigest certificateDigest_;

  friend llvm::Expected<VerifiedCgraClosedWaitEvidence>
  importVerifiedCgraClosedWaitEvidence(const ArtifactRootReference &,
                                       const ArtifactStore &,
                                       const BlobStore &);
};

/// Strictly imports Evidence, its exact CGRA Request, its sole execution
/// output, and that execution's typed closed-wait terminal as one inseparable
/// carrier, then reconstructs and replays the exact deterministic Request and
/// requires byte-identical Evidence identity. A foreign, missing, retired,
/// stopped, authored, or tampered execution fails closed.
llvm::Expected<VerifiedCgraClosedWaitEvidence>
importVerifiedCgraClosedWaitEvidence(
    const ArtifactRootReference &evidence, const ArtifactStore &artifactStore,
    const BlobStore &blobStore);

enum class CgraSimulationEvidenceTerminal : std::uint8_t {
  Retired,
  ClosedWait,
};

/// Classifies a strictly imported Completed CGRA Evidence by jointly checking
/// its mandatory finding result and sole SimulationExecution terminal. This is
/// the shared consumer boundary: Completed means the model fulfilled the
/// Request, not that the mapped graph retired.
llvm::Expected<CgraSimulationEvidenceTerminal>
classifyCompletedCgraSimulationEvidence(
    const EvaluationEvidence &evidence,
    const CaseArtifactResolution &resolution,
    const ArtifactStore &artifactStore, const BlobStore &blobStore);

} // namespace loom::evaluation::models

#endif // LOOM_EVALUATION_MODELS_CGRACLOSEDWAIT_H
