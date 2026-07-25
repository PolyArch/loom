#ifndef LOOM_EVALUATION_EVIDENCE_H
#define LOOM_EVALUATION_EVIDENCE_H

#include "Evaluation/Request.h"

#include "Common/Artifact.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace loom::evaluation {

enum class OutcomeReason : std::uint8_t {
  RuntimeCapabilityUnavailable,
  ToolFailure,
  AdapterFailure,
  InfrastructureFailure,
  ExternalCancellation,
  ExecutionLimitReached,
};

llvm::StringRef toString(OutcomeReason reason);
llvm::Expected<OutcomeReason> parseOutcomeReason(llvm::StringRef spelling);

struct ModelOutputBinding {
  ModelOutputSlotRef slot;
  std::vector<ArtifactRootReference> artifacts;

  friend bool operator==(const ModelOutputBinding &lhs,
                         const ModelOutputBinding &rhs) {
    return lhs.slot == rhs.slot && lhs.artifacts == rhs.artifacts;
  }
};

/// Exact Evidence context supplied to the selected Finding occurrence owner.
class FindingOccurrenceContext {
public:
  FindingOccurrenceContext(const EvaluationRequest &request,
                           FindingRequestOrdinal findingRequestOrdinal,
                           llvm::ArrayRef<ModelOutputBinding> outputBindings,
                           const CaseArtifactResolution &resolution,
                           const ArtifactStore &artifactStore)
      : request_(request), findingRequestOrdinal_(findingRequestOrdinal),
        outputBindings_(outputBindings), resolution_(resolution),
        artifactStore_(artifactStore) {}

  const EvaluationRequest &request() const { return request_; }
  FindingRequestOrdinal findingRequestOrdinal() const {
    return findingRequestOrdinal_;
  }
  llvm::ArrayRef<ModelOutputBinding> outputBindings() const {
    return outputBindings_;
  }
  const CaseArtifactResolution &resolution() const { return resolution_; }
  const ArtifactStore &artifactStore() const { return artifactStore_; }
  const ArtifactRootReference *resolveOutput(ModelOutputSlotRef slot,
                                             std::uint64_t ordinal) const;

private:
  const EvaluationRequest &request_;
  FindingRequestOrdinal findingRequestOrdinal_;
  llvm::ArrayRef<ModelOutputBinding> outputBindings_;
  const CaseArtifactResolution &resolution_;
  const ArtifactStore &artifactStore_;
};

class FindingOccurrence {
public:
  template <typename T> static FindingOccurrence get(T occurrence) {
    return FindingOccurrence(OwnerValue::get(std::move(occurrence)));
  }

  template <typename T> const std::decay_t<T> *getIf() const {
    return occurrence_.getIf<T>();
  }

  static llvm::Expected<FindingOccurrence>
  decode(const FindingOccurrenceCodec &codec,
         llvm::ArrayRef<std::uint8_t> canonicalPayload,
         const FindingOccurrenceContext &context);

  llvm::Error canonicalize(const FindingOccurrenceCodec &codec,
                           const FindingOccurrenceContext &context);
  std::string canonicalHex() const;

  friend bool operator==(const FindingOccurrence &lhs,
                         const FindingOccurrence &rhs) {
    return lhs.canonicalPayload_ == rhs.canonicalPayload_;
  }
  friend bool operator<(const FindingOccurrence &lhs,
                        const FindingOccurrence &rhs) {
    return lhs.canonicalPayload_ < rhs.canonicalPayload_;
  }

private:
  explicit FindingOccurrence(OwnerValue occurrence)
      : occurrence_(std::move(occurrence)) {}
  FindingOccurrence(OwnerValue occurrence,
                    std::vector<std::uint8_t> canonicalPayload)
      : occurrence_(std::move(occurrence)),
        canonicalPayload_(std::move(canonicalPayload)) {}

  OwnerValue occurrence_;
  std::vector<std::uint8_t> canonicalPayload_;
};

struct AbsentFinding {};

struct PresentFinding {
  std::vector<FindingOccurrence> occurrences;
};

struct NotApplicableFinding {
  NotApplicableReason reason;
};

using FindingResultValue =
    std::variant<AbsentFinding, PresentFinding, NotApplicableFinding>;

struct FindingResult {
  FindingResultValue result;
};

struct MetricResult {
  UncertaintyKind uncertainty;
  MetricObservationValue observation;
  std::vector<ModelInputSlotRef> calibrationInputSlots;
};

struct CompletedEvidence {
  std::vector<MetricResult> metricResults;
  std::vector<FindingResult> findingResults;
};

struct UnsupportedEvidence {
  OutcomeReason reason;
};

struct ExecutionFailedEvidence {
  OutcomeReason reason;
};

struct CancelledOrTimeoutEvidence {
  OutcomeReason reason;
};

using EvaluationEvidenceOutcome =
    std::variant<CompletedEvidence, UnsupportedEvidence,
                 ExecutionFailedEvidence, CancelledOrTimeoutEvidence>;

EvidenceOutcomeKind outcomeKind(const EvaluationEvidenceOutcome &outcome);

class EvaluationEvidence {
public:
  static const ArtifactSchemaDescriptor artifactSchema;

  static llvm::Expected<EvaluationEvidence>
  get(const EvaluationRequest &request,
      std::vector<ModelOutputBinding> outputBindings,
      EvaluationEvidenceOutcome outcome,
      const CaseArtifactResolution &resolution,
      const ArtifactStore &artifactStore);

  const ArtifactRootReference &requestRef() const { return requestRef_; }
  llvm::ArrayRef<ModelOutputBinding> outputBindings() const {
    return outputBindings_;
  }
  EvidenceOutcomeKind outcomeKind() const {
    return evaluation::outcomeKind(outcome_);
  }
  const EvaluationEvidenceOutcome &outcome() const { return outcome_; }

private:
  EvaluationEvidence(ArtifactRootReference requestRef,
                     std::vector<ModelOutputBinding> outputBindings,
                     EvaluationEvidenceOutcome outcome)
      : requestRef_(std::move(requestRef)),
        outputBindings_(std::move(outputBindings)),
        outcome_(std::move(outcome)) {}

  ArtifactRootReference requestRef_;
  std::vector<ModelOutputBinding> outputBindings_;
  EvaluationEvidenceOutcome outcome_;
};

CanonicalSemanticBytes
canonicalEvaluationEvidenceBytes(const EvaluationEvidence &evidence);
std::string serializeEvaluationEvidence(const EvaluationEvidence &evidence);
llvm::Expected<EvaluationEvidence>
parseEvaluationEvidence(llvm::StringRef json,
                        const CaseArtifactResolution &resolution,
                        const ArtifactStore &artifactStore);
ArtifactIdentity evaluationEvidenceIdentity(const EvaluationEvidence &evidence);
ArtifactRootReference
evaluationEvidenceReference(const EvaluationEvidence &evidence);
llvm::Expected<ArtifactRootReference>
publishEvaluationEvidence(const EvaluationEvidence &evidence,
                          const ArtifactStore &artifactStore);
llvm::Expected<EvaluationEvidence>
importEvaluationEvidence(const ArtifactRootReference &reference,
                         const CaseArtifactResolution &resolution,
                         const ArtifactStore &artifactStore);

} // namespace loom::evaluation

#endif // LOOM_EVALUATION_EVIDENCE_H
