#ifndef LOOM_APPLICATION_ACTIVATIONDECISION_H
#define LOOM_APPLICATION_ACTIVATIONDECISION_H

#include "Common/Artifact.h"
#include "Common/BlobDigest.h"
#include "Common/ComponentViewDigest.h"
#include "DSE/JointDesignExploration.h"
#include "DSE/ResourceTimeFrontier.h"
#include "Frontend/IR/StructuredProgramArtifact.h"
#include "Simulator/SourceBackedDfgValidation.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace loom {
class ArtifactStore;
class BlobStore;
} // namespace loom

namespace loom::application {

enum class ApplicationPairDecisionDisposition : std::uint8_t;

inline constexpr ArtifactSchemaDescriptor applicationActivationDecisionSchema{
    "loom.application.activation_decision", SchemaVersion{2, 0}};

enum class ApplicationActivationDecisionErrorReason : std::uint8_t {
  ForeignSchema,
  MalformedEncoding,
  NonCanonicalEncoding,
  DependencyMismatch,
  InvocationMismatch,
  PlanningMismatch,
  ScheduleMismatch,
  MappingMismatch,
  EvidenceMismatch,
  HardwareMutationRepairMismatch,
};

class ApplicationActivationDecisionError final
    : public llvm::ErrorInfo<ApplicationActivationDecisionError> {
public:
  static char ID;

  ApplicationActivationDecisionError(
      ApplicationActivationDecisionErrorReason reason, std::string message)
      : reason_(reason), message_(std::move(message)) {}

  ApplicationActivationDecisionErrorReason reason() const { return reason_; }
  void log(llvm::raw_ostream &stream) const override;
  std::error_code convertToErrorCode() const override;

private:
  ApplicationActivationDecisionErrorReason reason_;
  std::string message_;
};

/// Exact inputs to the selected pre-Mapping candidate identity. Mutable
/// evaluation, ranking, and disposition fields are intentionally absent.
struct ApplicationActivationPlanningPreimage final {
  ArtifactRootReference structuredProgram;
  ArtifactRootReference canonicalDataflow;
  std::vector<frontend::StructuredEntityRef> ownedProtocolRoots;
  ComponentViewDigest projectionIdentity;
  ComponentViewDigest frontierPolicyDigest;
};

/// Direct immutable dependencies named by one activation decision. Recursive
/// family closures remain owned by their Artifact and Evidence importers.
struct ApplicationActivationDecisionDependencyProjection final {
  std::vector<ArtifactRootReference> artifacts;
  std::vector<BlobDigest> blobs;
};

struct ApplicationActivationDecisionDraft final {
  ArtifactRootReference sourceProgram;
  ArtifactRootReference fabric;
  ArtifactRootReference workload;
  ArtifactRootReference runtimeInput;
  std::vector<sim::SourceBackedDfgReplayCaseReference> sourceBackedReplayCases;
  dse::JointDesignInvocationManifestReference dseInvocation;
  std::vector<dse::JointDesignInvocationManifestReference>
      supportingDseInvocations;
  ApplicationActivationPlanningPreimage planning;
  ComponentViewDigest selectedCandidateIdentity;
  std::uint64_t selectedPlanOrdinal = 0;
  std::vector<dse::ResourceTimeScheduleHint> selectedScheduleHints;
  ArtifactRootReference selectedSystem;
  ArtifactRootReference selectedMapping;
  ApplicationPairDecisionDisposition disposition;
  std::vector<ArtifactRootReference> runtimeEvidence;
  std::vector<ArtifactRootReference> oracleEvidence;
  std::optional<ArtifactRootReference> selectedHardwareMutationRepairRecord;
  std::vector<ArtifactRootReference> hardwareMutationRepairRecords;
};

/// Immutable owner of the exact application choice which is eligible for
/// Deployment activation, including the evaluated hardware-mutation repair
/// inventory and the nullable exact repair that selected its SystemMapping.
/// Candidate identity and schedule-hint digests are derived from their
/// complete preimages and never encoded as second owners.
class ApplicationActivationDecision final {
public:
  static llvm::Expected<ApplicationActivationDecision>
  get(ApplicationActivationDecisionDraft draft, const ArtifactStore &artifacts,
      const BlobStore &blobs);

  const ArtifactRootReference &sourceProgram() const { return sourceProgram_; }
  const ArtifactRootReference &fabric() const { return fabric_; }
  const ArtifactRootReference &workload() const { return workload_; }
  const ArtifactRootReference &runtimeInput() const { return runtimeInput_; }
  llvm::ArrayRef<sim::SourceBackedDfgReplayCaseReference>
  sourceBackedReplayCases() const {
    return sourceBackedReplayCases_;
  }
  const dse::JointDesignInvocationManifestReference &dseInvocation() const {
    return dseInvocation_;
  }
  llvm::ArrayRef<dse::JointDesignInvocationManifestReference>
  supportingDseInvocations() const {
    return supportingDseInvocations_;
  }
  const ApplicationActivationPlanningPreimage &planning() const {
    return planning_;
  }
  const ComponentViewDigest &selectedCandidateIdentity() const {
    return selectedCandidateIdentity_;
  }
  std::uint64_t selectedPlanOrdinal() const { return selectedPlanOrdinal_; }
  llvm::ArrayRef<dse::ResourceTimeScheduleHint> selectedScheduleHints() const {
    return selectedScheduleHints_;
  }
  const ArtifactRootReference &selectedSystem() const {
    return selectedSystem_;
  }
  const ArtifactRootReference &selectedMapping() const {
    return selectedMapping_;
  }
  ApplicationPairDecisionDisposition disposition() const {
    return disposition_;
  }
  llvm::ArrayRef<ArtifactRootReference> runtimeEvidence() const {
    return runtimeEvidence_;
  }
  llvm::ArrayRef<ArtifactRootReference> oracleEvidence() const {
    return oracleEvidence_;
  }
  const std::optional<ArtifactRootReference> &
  selectedHardwareMutationRepairRecord() const {
    return selectedHardwareMutationRepairRecord_;
  }
  llvm::ArrayRef<ArtifactRootReference> hardwareMutationRepairRecords() const {
    return hardwareMutationRepairRecords_;
  }
  const CanonicalSemanticBytes &canonicalBytes() const {
    return canonicalBytes_;
  }

private:
  ApplicationActivationDecision(ApplicationActivationDecisionDraft draft,
                                CanonicalSemanticBytes canonicalBytes)
      : sourceProgram_(std::move(draft.sourceProgram)),
        fabric_(std::move(draft.fabric)), workload_(std::move(draft.workload)),
        runtimeInput_(std::move(draft.runtimeInput)),
        sourceBackedReplayCases_(std::move(draft.sourceBackedReplayCases)),
        dseInvocation_(std::move(draft.dseInvocation)),
        supportingDseInvocations_(std::move(draft.supportingDseInvocations)),
        planning_(std::move(draft.planning)),
        selectedCandidateIdentity_(draft.selectedCandidateIdentity),
        selectedPlanOrdinal_(draft.selectedPlanOrdinal),
        selectedScheduleHints_(std::move(draft.selectedScheduleHints)),
        selectedSystem_(std::move(draft.selectedSystem)),
        selectedMapping_(std::move(draft.selectedMapping)),
        disposition_(draft.disposition),
        runtimeEvidence_(std::move(draft.runtimeEvidence)),
        oracleEvidence_(std::move(draft.oracleEvidence)),
        selectedHardwareMutationRepairRecord_(
            std::move(draft.selectedHardwareMutationRepairRecord)),
        hardwareMutationRepairRecords_(
            std::move(draft.hardwareMutationRepairRecords)),
        canonicalBytes_(std::move(canonicalBytes)) {}

  ArtifactRootReference sourceProgram_;
  ArtifactRootReference fabric_;
  ArtifactRootReference workload_;
  ArtifactRootReference runtimeInput_;
  std::vector<sim::SourceBackedDfgReplayCaseReference> sourceBackedReplayCases_;
  dse::JointDesignInvocationManifestReference dseInvocation_;
  std::vector<dse::JointDesignInvocationManifestReference>
      supportingDseInvocations_;
  ApplicationActivationPlanningPreimage planning_;
  ComponentViewDigest selectedCandidateIdentity_;
  std::uint64_t selectedPlanOrdinal_ = 0;
  std::vector<dse::ResourceTimeScheduleHint> selectedScheduleHints_;
  ArtifactRootReference selectedSystem_;
  ArtifactRootReference selectedMapping_;
  ApplicationPairDecisionDisposition disposition_;
  std::vector<ArtifactRootReference> runtimeEvidence_;
  std::vector<ArtifactRootReference> oracleEvidence_;
  std::optional<ArtifactRootReference> selectedHardwareMutationRepairRecord_;
  std::vector<ArtifactRootReference> hardwareMutationRepairRecords_;
  CanonicalSemanticBytes canonicalBytes_;
};

class FinalizedApplicationActivationDecision final {
public:
  const ArtifactRootReference &reference() const { return reference_; }
  const ApplicationActivationDecision &decision() const { return decision_; }

private:
  FinalizedApplicationActivationDecision(ArtifactRootReference reference,
                                         ApplicationActivationDecision decision)
      : reference_(std::move(reference)), decision_(std::move(decision)) {}

  ArtifactRootReference reference_;
  ApplicationActivationDecision decision_;

  friend llvm::Expected<FinalizedApplicationActivationDecision>
  publishApplicationActivationDecision(ApplicationActivationDecision,
                                       const ArtifactStore &);
  friend llvm::Expected<FinalizedApplicationActivationDecision>
  importApplicationActivationDecision(const ArtifactRootReference &,
                                      const ArtifactStore &, const BlobStore &);
};

llvm::Expected<FinalizedApplicationActivationDecision>
publishApplicationActivationDecision(ApplicationActivationDecision decision,
                                     const ArtifactStore &artifacts);

llvm::Expected<FinalizedApplicationActivationDecision>
importApplicationActivationDecision(const ArtifactRootReference &reference,
                                    const ArtifactStore &artifacts,
                                    const BlobStore &blobs);

/// Replays the embedded InvocationManifest and Evidence dependency owners,
/// then returns the complete direct closure needed to re-import this decision.
llvm::Expected<ApplicationActivationDecisionDependencyProjection>
projectApplicationActivationDecisionDependencies(
    const ApplicationActivationDecision &decision,
    const ArtifactStore &artifacts, const BlobStore &blobs);

} // namespace loom::application

#endif // LOOM_APPLICATION_ACTIVATIONDECISION_H
