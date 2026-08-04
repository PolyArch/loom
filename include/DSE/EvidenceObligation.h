#ifndef LOOM_DSE_EVIDENCEOBLIGATION_H
#define LOOM_DSE_EVIDENCEOBLIGATION_H

#include "Evaluation/Request.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

namespace loom {
class ArtifactStore;
}

namespace loom::dse {

class EvidenceObligationTemplateRef final {
public:
  explicit constexpr EvidenceObligationTemplateRef(std::uint32_t ordinal)
      : ordinal_(ordinal) {}
  constexpr std::uint32_t ordinal() const { return ordinal_; }
  friend constexpr bool operator==(EvidenceObligationTemplateRef lhs,
                                   EvidenceObligationTemplateRef rhs) {
    return lhs.ordinal_ == rhs.ordinal_;
  }

private:
  std::uint32_t ordinal_;
};

class EvidenceAcquisitionInputSlotRef final {
public:
  explicit constexpr EvidenceAcquisitionInputSlotRef(std::uint32_t ordinal)
      : ordinal_(ordinal) {}
  constexpr std::uint32_t ordinal() const { return ordinal_; }
  friend constexpr bool operator==(EvidenceAcquisitionInputSlotRef lhs,
                                   EvidenceAcquisitionInputSlotRef rhs) {
    return lhs.ordinal_ == rhs.ordinal_;
  }
  friend constexpr bool operator!=(EvidenceAcquisitionInputSlotRef lhs,
                                   EvidenceAcquisitionInputSlotRef rhs) {
    return !(lhs == rhs);
  }

private:
  std::uint32_t ordinal_;
};

enum class CalibrationPartitionRole : std::uint32_t {
  Training = 0,
  Validation = 1,
  HeldOut = 2,
};

struct InputSubjectBinding final {
  evaluation::CaseSubjectRoleRef role;
  EvidenceAcquisitionInputSlotRef inputSlot;
};

struct EvidenceAcquisitionInputBinding final {
  EvidenceAcquisitionInputSlotRef slot;
  std::vector<ArtifactRootReference> artifacts;
};

struct MetricRequestTemplate final {
  evaluation::MetricQuery query;
  std::vector<evaluation::EvaluationCondition> conditions;
};

struct FindingRequestTemplate final {
  evaluation::FindingQuery query;
  std::vector<evaluation::EvaluationCondition> conditions;
};

/// Candidate-independent request shape. Dynamic subject roles are named only
/// by the candidate role or an acquisition-policy-local input slot. All model,
/// workload, runtime-input, condition, query, and fixed-subject facts remain
/// exact and immutable.
class EvidenceObligationTemplate final {
public:
  static llvm::Expected<EvidenceObligationTemplate>
  get(const evaluation::EvaluationRequest &prototype,
      evaluation::CaseSubjectRoleRef candidateRole,
      std::vector<InputSubjectBinding> inputSubjectBindings,
      std::optional<CalibrationPartitionRole> calibrationPartitionRole =
          std::nullopt);

  evaluation::CaseSubjectRoleRef candidateRole() const {
    return candidateRole_;
  }
  llvm::ArrayRef<InputSubjectBinding> inputSubjectBindings() const {
    return inputSubjectBindings_;
  }
  std::optional<CalibrationPartitionRole> calibrationPartitionRole() const {
    return calibrationPartitionRole_;
  }
  llvm::ArrayRef<std::uint8_t> canonicalBytes() const {
    return canonicalBytes_;
  }
  const evaluation::ResolvedModelBinding &modelBinding() const {
    return modelBinding_;
  }
  llvm::ArrayRef<MetricRequestTemplate> metricRequests() const {
    return metricRequests_;
  }
  llvm::ArrayRef<FindingRequestTemplate> findingRequests() const {
    return findingRequests_;
  }

private:
  EvidenceObligationTemplate(
      evaluation::ResolvedModelBinding modelBinding,
      std::vector<evaluation::CaseRoleBinding> fixedSubjectBindings,
      std::optional<ArtifactRootReference> workload,
      std::optional<ArtifactRootReference> runtimeInput,
      std::vector<evaluation::EvaluationCondition> baseConditions,
      std::vector<MetricRequestTemplate> metricRequests,
      std::vector<FindingRequestTemplate> findingRequests,
      evaluation::CaseSubjectRoleRef candidateRole,
      std::vector<InputSubjectBinding> inputSubjectBindings,
      std::optional<CalibrationPartitionRole> calibrationPartitionRole,
      std::vector<std::uint8_t> canonicalBytes)
      : modelBinding_(std::move(modelBinding)),
        fixedSubjectBindings_(std::move(fixedSubjectBindings)),
        workload_(std::move(workload)), runtimeInput_(std::move(runtimeInput)),
        baseConditions_(std::move(baseConditions)),
        metricRequests_(std::move(metricRequests)),
        findingRequests_(std::move(findingRequests)),
        candidateRole_(candidateRole),
        inputSubjectBindings_(std::move(inputSubjectBindings)),
        calibrationPartitionRole_(calibrationPartitionRole),
        canonicalBytes_(std::move(canonicalBytes)) {}

  evaluation::ResolvedModelBinding modelBinding_;
  std::vector<evaluation::CaseRoleBinding> fixedSubjectBindings_;
  std::optional<ArtifactRootReference> workload_;
  std::optional<ArtifactRootReference> runtimeInput_;
  std::vector<evaluation::EvaluationCondition> baseConditions_;
  std::vector<MetricRequestTemplate> metricRequests_;
  std::vector<FindingRequestTemplate> findingRequests_;
  evaluation::CaseSubjectRoleRef candidateRole_;
  std::vector<InputSubjectBinding> inputSubjectBindings_;
  std::optional<CalibrationPartitionRole> calibrationPartitionRole_;
  std::vector<std::uint8_t> canonicalBytes_;

  friend llvm::Expected<EvidenceObligationTemplate>
  adoptEvidenceObligationTemplate(llvm::ArrayRef<std::uint8_t> bytes);
  friend llvm::Expected<evaluation::EvaluationRequest>
  instantiateEvidenceObligation(
      const EvidenceObligationTemplate &obligation,
      const ArtifactRootReference &candidate,
      llvm::ArrayRef<EvidenceAcquisitionInputBinding> inputBindings,
      std::uint64_t replicateIndex,
      const evaluation::CaseArtifactResolution &resolution,
      const ArtifactStore &artifactStore);
};

llvm::Expected<EvidenceObligationTemplate>
adoptEvidenceObligationTemplate(llvm::ArrayRef<std::uint8_t> bytes);

llvm::Expected<evaluation::EvaluationRequest> instantiateEvidenceObligation(
    const EvidenceObligationTemplate &obligation,
    const ArtifactRootReference &candidate,
    llvm::ArrayRef<EvidenceAcquisitionInputBinding> inputBindings,
    std::uint64_t replicateIndex,
    const evaluation::CaseArtifactResolution &resolution,
    const ArtifactStore &artifactStore);

} // namespace loom::dse

#endif // LOOM_DSE_EVIDENCEOBLIGATION_H
