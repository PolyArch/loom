#ifndef LOOM_DSE_INVOCATIONMANIFEST_H
#define LOOM_DSE_INVOCATIONMANIFEST_H

#include "Common/Artifact.h"
#include "Common/BlobDigest.h"
#include "DSE/ExternalToolWorkLedger.h"
#include "DSE/Plan.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace loom {
class ArtifactStore;
class BlobStore;
struct ResolvedConfig;
} // namespace loom

namespace loom::dse {

class ExecutionJournal;
class ResolvedDseConfigView;
enum class CampaignAdmissionFailureReason : std::uint32_t;

class DseProducerSemanticBuildIdentity final {
public:
  static llvm::Expected<DseProducerSemanticBuildIdentity>
  get(llvm::StringRef spelling);

  llvm::StringRef spelling() const { return spelling_; }

  friend bool operator==(const DseProducerSemanticBuildIdentity &lhs,
                         const DseProducerSemanticBuildIdentity &rhs) {
    return lhs.spelling_ == rhs.spelling_;
  }

private:
  explicit DseProducerSemanticBuildIdentity(std::string spelling)
      : spelling_(std::move(spelling)) {}

  std::string spelling_;
};

class DseRunKey final {
public:
  using Storage = std::array<std::uint8_t, 32>;
  static constexpr std::size_t byteSize = 32;

  static llvm::Expected<DseRunKey>
  fromBytes(llvm::ArrayRef<std::uint8_t> bytes);

  const Storage &bytes() const { return bytes_; }

  friend bool operator==(const DseRunKey &lhs, const DseRunKey &rhs) {
    return lhs.bytes_ == rhs.bytes_;
  }
  friend bool operator!=(const DseRunKey &lhs, const DseRunKey &rhs) {
    return !(lhs == rhs);
  }

private:
  explicit DseRunKey(Storage bytes) : bytes_(bytes) {}

  Storage bytes_;

  friend class DseRunClosure;
};

class DseRunClosure final {
public:
  static llvm::Expected<DseRunClosure>
  get(DseProducerSemanticBuildIdentity producer,
      llvm::ArrayRef<ArtifactRootReference> semanticInputs,
      const ResolvedConfig &resolvedConfig,
      llvm::ArrayRef<ArtifactRootReference> preexistingEvidence,
      const ArtifactStore &artifactStore);

  const DseProducerSemanticBuildIdentity &producer() const { return producer_; }
  llvm::ArrayRef<ArtifactRootReference> semanticInputs() const {
    return semanticInputs_;
  }
  const ArtifactIdentity &resolvedConfigIdentity() const {
    return resolvedConfigIdentity_;
  }
  llvm::ArrayRef<ArtifactRootReference> preexistingEvidence() const {
    return preexistingEvidence_;
  }
  const DseRunKey &runKey() const { return runKey_; }

private:
  DseRunClosure(DseProducerSemanticBuildIdentity producer,
                std::vector<ArtifactRootReference> semanticInputs,
                ArtifactIdentity resolvedConfigIdentity,
                std::vector<ArtifactRootReference> preexistingEvidence,
                DseRunKey runKey)
      : producer_(std::move(producer)),
        semanticInputs_(std::move(semanticInputs)),
        resolvedConfigIdentity_(std::move(resolvedConfigIdentity)),
        preexistingEvidence_(std::move(preexistingEvidence)),
        runKey_(std::move(runKey)) {}

  DseProducerSemanticBuildIdentity producer_;
  std::vector<ArtifactRootReference> semanticInputs_;
  ArtifactIdentity resolvedConfigIdentity_;
  std::vector<ArtifactRootReference> preexistingEvidence_;
  DseRunKey runKey_;
};

struct InvocationOccurrenceRef final {
  DseRunKey runKey;
  std::uint64_t occurrenceOrdinal = 0;

  friend bool operator==(const InvocationOccurrenceRef &lhs,
                         const InvocationOccurrenceRef &rhs) {
    return lhs.runKey == rhs.runKey &&
           lhs.occurrenceOrdinal == rhs.occurrenceOrdinal;
  }
  friend bool operator!=(const InvocationOccurrenceRef &lhs,
                         const InvocationOccurrenceRef &rhs) {
    return !(lhs == rhs);
  }
};

/// Durable journal-owned binding between one occurrence and its canonical
/// manifest Blob. Content import alone cannot construct this receipt.
class InvocationManifestReceipt final {
public:
  const InvocationOccurrenceRef &occurrence() const { return occurrence_; }
  const BlobDigest &manifest() const { return manifest_; }

  friend bool operator==(const InvocationManifestReceipt &lhs,
                         const InvocationManifestReceipt &rhs) {
    return lhs.occurrence_ == rhs.occurrence_ && lhs.manifest_ == rhs.manifest_;
  }

private:
  InvocationManifestReceipt(InvocationOccurrenceRef occurrence,
                            BlobDigest manifest)
      : occurrence_(std::move(occurrence)), manifest_(std::move(manifest)) {}

  InvocationOccurrenceRef occurrence_;
  BlobDigest manifest_;

  friend class ExecutionJournal;
};

struct InvocationCompletedSelection final {
  std::vector<ArtifactRootReference> selected;
  std::vector<ArtifactRootReference> satisfiedEvidence;
};

struct InvocationCompletedNoFeasibleCandidate final {
  std::vector<ArtifactRootReference> satisfiedEvidence;
};

struct InvocationIncomplete final {
  std::uint64_t planNodeOrdinal = 0;
  DsePlanIncompleteReason reason;
  std::vector<EvidenceObligationTemplateRef> unsatisfiedObligations;
  std::vector<ArtifactRootReference> retainedArtifacts;
  std::vector<ArtifactRootReference> retainedEvidence;
};

using InvocationControllerOutcome =
    std::variant<InvocationCompletedSelection,
                 InvocationCompletedNoFeasibleCandidate, InvocationIncomplete>;

struct InvocationGenerateRecord final {
  bool completed = false;
  GenerateInvocationRecord invocation;
  GenerateInvocationWorkSummary workSummary;
};

struct PlanNodeOperationalObservation final {
  std::uint64_t planNodeOrdinal = 0;
  std::uint64_t activeWallTimeNanoseconds = 0;
  std::uint64_t processCpuTimeNanoseconds = 0;

  friend bool operator==(const PlanNodeOperationalObservation &lhs,
                         const PlanNodeOperationalObservation &rhs) {
    return lhs.planNodeOrdinal == rhs.planNodeOrdinal &&
           lhs.activeWallTimeNanoseconds == rhs.activeWallTimeNanoseconds &&
           lhs.processCpuTimeNanoseconds == rhs.processCpuTimeNanoseconds;
  }
  friend bool operator!=(const PlanNodeOperationalObservation &lhs,
                         const PlanNodeOperationalObservation &rhs) {
    return !(lhs == rhs);
  }
};

struct InvocationOperationalObservations final {
  std::uint64_t totalActiveWallTimeNanoseconds = 0;
  std::uint64_t totalProcessCpuTimeNanoseconds = 0;
  std::uint64_t peakResidentBytes = 0;
  std::uint64_t requestedWorkerCount = 0;
  std::uint64_t availableLogicalCpuCount = 0;
  std::vector<PlanNodeOperationalObservation> planNodes;

  friend bool operator==(const InvocationOperationalObservations &lhs,
                         const InvocationOperationalObservations &rhs) {
    return lhs.totalActiveWallTimeNanoseconds ==
               rhs.totalActiveWallTimeNanoseconds &&
           lhs.totalProcessCpuTimeNanoseconds ==
               rhs.totalProcessCpuTimeNanoseconds &&
           lhs.peakResidentBytes == rhs.peakResidentBytes &&
           lhs.requestedWorkerCount == rhs.requestedWorkerCount &&
           lhs.availableLogicalCpuCount == rhs.availableLogicalCpuCount &&
           lhs.planNodes == rhs.planNodes;
  }
  friend bool operator!=(const InvocationOperationalObservations &lhs,
                         const InvocationOperationalObservations &rhs) {
    return !(lhs == rhs);
  }
};

class InvocationManifest final {
public:
  static constexpr llvm::StringLiteral schemaIdentity =
      "loom.dse.invocation_manifest";
  static constexpr SchemaVersion schemaVersion{1, 6};

  static llvm::Expected<InvocationManifest>
  get(DseRunClosure closure, std::uint64_t occurrenceOrdinal,
      std::optional<InvocationOccurrenceRef> resumedFrom,
      const ResolvedConfig &resolvedConfig,
      const DsePlanGenerateInvocationRecords &generateRecords,
      InvocationControllerOutcome outcome, const ArtifactStore &artifactStore,
      const BlobStore &blobStore,
      std::optional<InvocationOperationalObservations> operationalObservations =
          std::nullopt,
      std::optional<InvocationExternalToolWorkLedger> externalToolWork =
          std::nullopt,
      std::optional<CampaignAdmissionFailureReason> campaignAdmissionFailure =
          std::nullopt);

  const InvocationOccurrenceRef &occurrence() const { return occurrence_; }
  const DseRunClosure &closure() const { return closure_; }
  const std::optional<InvocationOccurrenceRef> &resumedFrom() const {
    return resumedFrom_;
  }
  llvm::ArrayRef<std::uint8_t> resolvedDseConfigViewDescriptorBytes() const {
    return resolvedDseConfigViewDescriptorBytes_;
  }
  const ComponentViewDigest &resolvedDseConfigViewDigest() const {
    return resolvedDseConfigViewDigest_;
  }
  llvm::ArrayRef<InvocationGenerateRecord> generateRecords() const {
    return generateRecords_;
  }
  const InvocationControllerOutcome &outcome() const { return outcome_; }
  const std::optional<InvocationOperationalObservations> &
  operationalObservations() const {
    return operationalObservations_;
  }
  const InvocationExternalToolWorkLedger &externalToolWork() const {
    return externalToolWork_;
  }
  const std::optional<CampaignAdmissionFailureReason> &
  campaignAdmissionFailure() const {
    return campaignAdmissionFailure_;
  }
  llvm::ArrayRef<std::uint8_t> canonicalBytes() const {
    return canonicalBytes_;
  }

private:
  InvocationManifest(
      InvocationOccurrenceRef occurrence, DseRunClosure closure,
      std::optional<InvocationOccurrenceRef> resumedFrom,
      std::vector<std::uint8_t> resolvedDseConfigViewDescriptorBytes,
      ComponentViewDigest resolvedDseConfigViewDigest,
      std::vector<InvocationGenerateRecord> generateRecords,
      InvocationControllerOutcome outcome,
      std::optional<InvocationOperationalObservations> operationalObservations,
      InvocationExternalToolWorkLedger externalToolWork,
      std::optional<CampaignAdmissionFailureReason> campaignAdmissionFailure,
      std::vector<std::uint8_t> canonicalBytes)
      : occurrence_(std::move(occurrence)), closure_(std::move(closure)),
        resumedFrom_(std::move(resumedFrom)),
        resolvedDseConfigViewDescriptorBytes_(
            std::move(resolvedDseConfigViewDescriptorBytes)),
        resolvedDseConfigViewDigest_(resolvedDseConfigViewDigest),
        generateRecords_(std::move(generateRecords)),
        outcome_(std::move(outcome)),
        operationalObservations_(std::move(operationalObservations)),
        externalToolWork_(std::move(externalToolWork)),
        campaignAdmissionFailure_(campaignAdmissionFailure),
        canonicalBytes_(std::move(canonicalBytes)) {}

  InvocationOccurrenceRef occurrence_;
  DseRunClosure closure_;
  std::optional<InvocationOccurrenceRef> resumedFrom_;
  std::vector<std::uint8_t> resolvedDseConfigViewDescriptorBytes_;
  ComponentViewDigest resolvedDseConfigViewDigest_;
  std::vector<InvocationGenerateRecord> generateRecords_;
  InvocationControllerOutcome outcome_;
  std::optional<InvocationOperationalObservations> operationalObservations_;
  InvocationExternalToolWorkLedger externalToolWork_;
  std::optional<CampaignAdmissionFailureReason> campaignAdmissionFailure_;
  std::vector<std::uint8_t> canonicalBytes_;

  friend llvm::Expected<InvocationManifest>
  adoptInvocationManifest(llvm::ArrayRef<std::uint8_t>, const ResolvedConfig &,
                          const ArtifactStore &, const BlobStore &);
};

llvm::Expected<InvocationManifest>
adoptInvocationManifest(llvm::ArrayRef<std::uint8_t> canonicalBytes,
                        const ResolvedConfig &resolvedConfig,
                        const ArtifactStore &artifactStore,
                        const BlobStore &blobStore);

/// Strict content reference to one canonical InvocationManifest. Occurrence
/// allocation and receipt commit remain owned by ExecutionJournal.
class InvocationManifestReference final {
public:
  static llvm::Expected<InvocationManifestReference>
  get(ArtifactRootReference resolvedConfig, BlobDigest blob,
      InvocationOccurrenceRef occurrence, const ArtifactStore &artifacts,
      const BlobStore &blobs);

  const ArtifactRootReference &resolvedConfig() const {
    return resolvedConfig_;
  }
  const BlobDigest &blob() const { return blob_; }
  const InvocationOccurrenceRef &occurrence() const { return occurrence_; }
  const std::optional<InvocationManifestReceipt> &journalReceipt() const {
    return journalReceipt_;
  }

private:
  InvocationManifestReference(ArtifactRootReference resolvedConfig,
                              BlobDigest blob,
                              InvocationOccurrenceRef occurrence)
      : resolvedConfig_(std::move(resolvedConfig)), blob_(std::move(blob)),
        occurrence_(std::move(occurrence)) {}

  ArtifactRootReference resolvedConfig_;
  BlobDigest blob_;
  InvocationOccurrenceRef occurrence_;
  std::optional<InvocationManifestReceipt> journalReceipt_;

  friend llvm::Expected<InvocationManifestReference>
  bindInvocationManifestReceipt(InvocationManifestReference,
                                const InvocationManifestReceipt &);
};

llvm::Expected<InvocationManifestReference>
bindInvocationManifestReceipt(InvocationManifestReference reference,
                              const InvocationManifestReceipt &receipt);

llvm::Expected<InvocationManifestReference> publishInvocationManifest(
    const InvocationManifest &manifest, const ResolvedConfig &resolvedConfig,
    const ArtifactStore &artifacts, const BlobStore &blobs);

llvm::Expected<InvocationManifest>
importInvocationManifest(const InvocationManifestReference &reference,
                         const ArtifactStore &artifacts,
                         const BlobStore &blobs);

llvm::Expected<InvocationControllerOutcome> projectDsePlanInvocationOutcome(
    const ResolvedDseConfigView &view,
    const DsePlanExecutionOutcome &executionOutcome);

/// Publishes the canonical manifest and atomically binds its digest to the
/// active journal occurrence. Hard execution errors have no plan outcome and
/// therefore remain outside this owner transaction.
llvm::Expected<InvocationManifestReference> finalizeDsePlanInvocation(
    DseRunClosure closure, const ResolvedConfig &resolvedConfig,
    const DsePlanExecutionOutcome &executionOutcome, ExecutionJournal &journal,
    const ArtifactStore &artifacts, const BlobStore &blobs,
    std::optional<CampaignAdmissionFailureReason> campaignAdmissionFailure =
        std::nullopt);

} // namespace loom::dse

#endif // LOOM_DSE_INVOCATIONMANIFEST_H
