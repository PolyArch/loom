#ifndef LOOM_APPLICATION_RUNTIMEMANIFEST_H
#define LOOM_APPLICATION_RUNTIMEMANIFEST_H

#include "Common/Artifact.h"
#include "Common/ComponentViewDigest.h"
#include "PnR/System/SystemMappingMigration.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <array>
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

inline constexpr ArtifactSchemaDescriptor applicationRuntimeManifestSchema{
    "loom.application.runtime_manifest", SchemaVersion{1, 0}};

enum class ApplicationRuntimeManifestErrorReason : std::uint8_t {
  ForeignSchema,
  MalformedEncoding,
  NonCanonicalEncoding,
  PairIdentityMismatch,
  PairDecisionIncomplete,
  MappingMismatch,
  DeploymentMismatch,
  RuntimeEvidenceMismatch,
  TransitionGraphMismatch,
};

class ApplicationRuntimeManifestError final
    : public llvm::ErrorInfo<ApplicationRuntimeManifestError> {
public:
  static char ID;

  ApplicationRuntimeManifestError(ApplicationRuntimeManifestErrorReason reason,
                                  std::string message)
      : reason_(reason), message_(std::move(message)) {}

  ApplicationRuntimeManifestErrorReason reason() const { return reason_; }
  void log(llvm::raw_ostream &stream) const override;
  std::error_code convertToErrorCode() const override;

private:
  ApplicationRuntimeManifestErrorReason reason_;
  std::string message_;
};

struct ApplicationRuntimeManifestDraft final {
  ArtifactRootReference sourceProgram;
  ArtifactRootReference fabric;
  ArtifactRootReference workload;
  ArtifactRootReference runtimeInput;
  ComponentViewDigest pairIdentity;
  std::array<std::uint8_t, 32> invocationRunKey;
  ApplicationPairDecisionDisposition pairDisposition;
  ComponentViewDigest selectedCandidateIdentity;
  std::uint64_t selectedPlanOrdinal = 0;
  std::vector<ComponentViewDigest> selectedScheduleHintDigests;
  ArtifactRootReference selectedSystem;
  ArtifactRootReference selectedMapping;
  ArtifactRootReference deployment;
  std::vector<ArtifactRootReference> runtimeRequestDependencies;
  std::vector<ArtifactRootReference> runtimeEvidence;
  std::vector<ArtifactRootReference> oracleEvidence;
  std::optional<pnr::ResourceTimeTransitionGraph> transitionGraph;
};

class ApplicationRuntimeManifest final {
public:
  static llvm::Expected<ApplicationRuntimeManifest>
  get(ApplicationRuntimeManifestDraft draft, const ArtifactStore &artifacts,
      const BlobStore &blobs);

  const ArtifactRootReference &sourceProgram() const { return sourceProgram_; }
  const ArtifactRootReference &fabric() const { return fabric_; }
  const ArtifactRootReference &workload() const { return workload_; }
  const ArtifactRootReference &runtimeInput() const { return runtimeInput_; }
  const ComponentViewDigest &pairIdentity() const { return pairIdentity_; }
  const std::array<std::uint8_t, 32> &invocationRunKey() const {
    return invocationRunKey_;
  }
  ApplicationPairDecisionDisposition pairDisposition() const {
    return pairDisposition_;
  }
  const ComponentViewDigest &selectedCandidateIdentity() const {
    return selectedCandidateIdentity_;
  }
  std::uint64_t selectedPlanOrdinal() const { return selectedPlanOrdinal_; }
  llvm::ArrayRef<ComponentViewDigest> selectedScheduleHintDigests() const {
    return selectedScheduleHintDigests_;
  }
  const ArtifactRootReference &selectedSystem() const {
    return selectedSystem_;
  }
  const ArtifactRootReference &selectedMapping() const {
    return selectedMapping_;
  }
  const ArtifactRootReference &deployment() const { return deployment_; }
  llvm::ArrayRef<ArtifactRootReference> runtimeRequestDependencies() const {
    return runtimeRequestDependencies_;
  }
  llvm::ArrayRef<ArtifactRootReference> runtimeEvidence() const {
    return runtimeEvidence_;
  }
  llvm::ArrayRef<ArtifactRootReference> oracleEvidence() const {
    return oracleEvidence_;
  }
  const std::optional<pnr::ResourceTimeTransitionGraph> &
  transitionGraph() const {
    return transitionGraph_;
  }
  const CanonicalSemanticBytes &canonicalBytes() const {
    return canonicalBytes_;
  }

private:
  ApplicationRuntimeManifest(ApplicationRuntimeManifestDraft draft,
                             CanonicalSemanticBytes canonicalBytes)
      : sourceProgram_(std::move(draft.sourceProgram)),
        fabric_(std::move(draft.fabric)), workload_(std::move(draft.workload)),
        runtimeInput_(std::move(draft.runtimeInput)),
        pairIdentity_(draft.pairIdentity),
        invocationRunKey_(draft.invocationRunKey),
        pairDisposition_(draft.pairDisposition),
        selectedCandidateIdentity_(draft.selectedCandidateIdentity),
        selectedPlanOrdinal_(draft.selectedPlanOrdinal),
        selectedScheduleHintDigests_(
            std::move(draft.selectedScheduleHintDigests)),
        selectedSystem_(std::move(draft.selectedSystem)),
        selectedMapping_(std::move(draft.selectedMapping)),
        deployment_(std::move(draft.deployment)),
        runtimeRequestDependencies_(
            std::move(draft.runtimeRequestDependencies)),
        runtimeEvidence_(std::move(draft.runtimeEvidence)),
        oracleEvidence_(std::move(draft.oracleEvidence)),
        transitionGraph_(std::move(draft.transitionGraph)),
        canonicalBytes_(std::move(canonicalBytes)) {}

  ArtifactRootReference sourceProgram_;
  ArtifactRootReference fabric_;
  ArtifactRootReference workload_;
  ArtifactRootReference runtimeInput_;
  ComponentViewDigest pairIdentity_;
  std::array<std::uint8_t, 32> invocationRunKey_;
  ApplicationPairDecisionDisposition pairDisposition_;
  ComponentViewDigest selectedCandidateIdentity_;
  std::uint64_t selectedPlanOrdinal_ = 0;
  std::vector<ComponentViewDigest> selectedScheduleHintDigests_;
  ArtifactRootReference selectedSystem_;
  ArtifactRootReference selectedMapping_;
  ArtifactRootReference deployment_;
  std::vector<ArtifactRootReference> runtimeRequestDependencies_;
  std::vector<ArtifactRootReference> runtimeEvidence_;
  std::vector<ArtifactRootReference> oracleEvidence_;
  std::optional<pnr::ResourceTimeTransitionGraph> transitionGraph_;
  CanonicalSemanticBytes canonicalBytes_;
};

class FinalizedApplicationRuntimeManifest final {
public:
  const ArtifactRootReference &reference() const { return reference_; }
  const ApplicationRuntimeManifest &manifest() const { return manifest_; }

private:
  FinalizedApplicationRuntimeManifest(ArtifactRootReference reference,
                                      ApplicationRuntimeManifest manifest)
      : reference_(std::move(reference)), manifest_(std::move(manifest)) {}

  ArtifactRootReference reference_;
  ApplicationRuntimeManifest manifest_;

  friend llvm::Expected<FinalizedApplicationRuntimeManifest>
  publishApplicationRuntimeManifest(ApplicationRuntimeManifest,
                                    const ArtifactStore &);
  friend llvm::Expected<FinalizedApplicationRuntimeManifest>
  importApplicationRuntimeManifest(const ArtifactRootReference &,
                                   const ArtifactStore &, const BlobStore &);
};

llvm::Expected<ComponentViewDigest>
deriveApplicationPairIdentity(const ArtifactRootReference &sourceProgram,
                              const ArtifactRootReference &fabric,
                              const ArtifactRootReference &workload,
                              const ArtifactRootReference &runtimeInput);

std::string
serializeApplicationRuntimeManifest(const ApplicationRuntimeManifest &manifest);

llvm::Expected<FinalizedApplicationRuntimeManifest>
publishApplicationRuntimeManifest(ApplicationRuntimeManifest manifest,
                                  const ArtifactStore &artifacts);

llvm::Expected<FinalizedApplicationRuntimeManifest>
importApplicationRuntimeManifest(const ArtifactRootReference &reference,
                                 const ArtifactStore &artifacts,
                                 const BlobStore &blobs);

} // namespace loom::application

#endif // LOOM_APPLICATION_RUNTIMEMANIFEST_H
