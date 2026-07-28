#include "DSE/Promotion.h"

#include "Common/ArtifactLocalReference.h"
#include "Common/ArtifactStore.h"
#include "Evaluation/NumericValue.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <utility>

namespace loom::dse {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "dse_promotion_invalid: " + message);
}

bool matchesSchema(const ArtifactRootReference &reference,
                   const ArtifactSchemaDescriptor &schema) {
  return reference.schemaIdentity == schema.identity &&
         reference.schemaVersion == schema.version;
}

bool containsCandidate(const CandidateSet &candidateSet,
                       const ArtifactRootReference &candidate) {
  return std::binary_search(candidateSet.candidates().begin(),
                            candidateSet.candidates().end(), candidate,
                            artifactRootReferenceLess);
}

bool equalModelBinding(const evaluation::ResolvedModelBinding &lhs,
                       const evaluation::ResolvedModelBinding &rhs) {
  return lhs.descriptorRef() == rhs.descriptorRef() &&
         llvm::equal(lhs.inputBindings(), rhs.inputBindings()) &&
         lhs.resolvedModelConfig().digest() ==
             rhs.resolvedModelConfig().digest() &&
         llvm::equal(lhs.resolvedModelConfig().canonicalViewBytes(),
                     rhs.resolvedModelConfig().canonicalViewBytes());
}

bool equalSubjectShapeExceptCandidate(
    const evaluation::EvaluationRequest &lhs,
    const evaluation::EvaluationRequest &rhs,
    evaluation::CaseSubjectRoleRef candidateRole) {
  llvm::ArrayRef<evaluation::CaseRoleBinding> lhsBindings =
      lhs.subjectBindings().roleBindings();
  llvm::ArrayRef<evaluation::CaseRoleBinding> rhsBindings =
      rhs.subjectBindings().roleBindings();
  if (lhsBindings.size() != rhsBindings.size())
    return false;
  for (std::size_t index = 0; index < lhsBindings.size(); ++index) {
    if (lhsBindings[index].role != rhsBindings[index].role)
      return false;
    if (lhsBindings[index].role == candidateRole) {
      if (lhsBindings[index].subjects.size() != 1 ||
          rhsBindings[index].subjects.size() != 1)
        return false;
      continue;
    }
    if (lhsBindings[index].subjects != rhsBindings[index].subjects)
      return false;
  }
  return true;
}

bool sameObligationShape(const evaluation::EvaluationRequest &lhs,
                         const evaluation::EvaluationRequest &rhs,
                         evaluation::CaseSubjectRoleRef candidateRole) {
  return equalSubjectShapeExceptCandidate(lhs, rhs, candidateRole) &&
         lhs.workload() == rhs.workload() &&
         lhs.runtimeInput() == rhs.runtimeInput() &&
         llvm::equal(lhs.baseConditions(), rhs.baseConditions()) &&
         llvm::equal(lhs.metricRequests(), rhs.metricRequests()) &&
         llvm::equal(lhs.findingRequests(), rhs.findingRequests()) &&
         equalModelBinding(lhs.modelBinding(), rhs.modelBinding()) &&
         lhs.replicateIndex() == rhs.replicateIndex();
}

int compareMetricValue(const evaluation::MetricValue &lhs,
                       const evaluation::MetricValue &rhs) {
  assert(lhs.index() == rhs.index() &&
         "Metric value kinds were validated before sorting");
  if (const auto *lhsInteger = std::get_if<evaluation::IntegerValue>(&lhs)) {
    const auto rhsInteger = std::get<evaluation::IntegerValue>(rhs);
    if (lhsInteger->value() == rhsInteger.value())
      return 0;
    return lhsInteger->value() < rhsInteger.value() ? -1 : 1;
  }
  return evaluation::compareDecimalValue(
      std::get<evaluation::DecimalValue>(lhs),
      std::get<evaluation::DecimalValue>(rhs));
}

struct RankedCandidate final {
  ArtifactRootReference candidate;
  evaluation::MetricValue value;
  ArtifactRootReference evidence;
};

} // namespace

llvm::Expected<CandidateSet>
CandidateSet::get(const ArtifactSchemaDescriptor &schema,
                  llvm::ArrayRef<ArtifactRootReference> candidates) {
  std::vector<ArtifactRootReference> canonical(candidates.begin(),
                                               candidates.end());
  for (const ArtifactRootReference &candidate : canonical)
    if (!matchesSchema(candidate, schema))
      return invalid("candidate does not match the selected-set schema");
  llvm::sort(canonical, artifactRootReferenceLess);
  canonical.erase(std::unique(canonical.begin(), canonical.end()),
                  canonical.end());
  return CandidateSet(schema, std::move(canonical));
}

llvm::Expected<PromotionOutcome>
promoteMetricTopK(const CandidateSet &candidateSet,
                  evaluation::CaseSubjectRoleRef candidateRole,
                  llvm::ArrayRef<PromotionEvidence> evidence,
                  const PointMetricTopKSelection &selection,
                  const ArtifactStore &artifactStore) {
  if (selection.k == 0)
    return invalid("TopK requires positive k");
  if (candidateSet.candidates().empty())
    return PromotionOutcome{CompletedNoFeasibleCandidate{}};

  std::map<ArtifactRootReference, RankedCandidate,
           decltype(&artifactRootReferenceLess)>
      ranked(&artifactRootReferenceLess);
  std::map<ArtifactRootReference, const PromotionEvidence *,
           decltype(&artifactRootReferenceLess)>
      records(&artifactRootReferenceLess);
  std::vector<ArtifactRootReference> retainedEvidence;

  for (const PromotionEvidence &record : evidence) {
    if (record.evidence.requestRef() !=
        evaluation::evaluationRequestReference(record.request))
      return invalid("Evidence does not reference its supplied Request");
    llvm::ArrayRef<ArtifactRootReference> subjects =
        record.request.subjectBindings().subjects(candidateRole);
    if (subjects.size() != 1)
      return invalid("candidate role is not bound to exactly one Artifact");
    const ArtifactRootReference &candidate = subjects.front();
    if (!containsCandidate(candidateSet, candidate))
      return invalid("Evidence names a candidate outside the input set");
    if (!records.emplace(candidate, &record).second)
      return invalid("candidate has duplicate Evidence obligations");
  }

  const evaluation::EvaluationRequest *obligationShape = nullptr;
  std::optional<std::pair<IncompleteSelectionReason, ArtifactRootReference>>
      incomplete;
  std::optional<std::size_t> metricValueKind;
  for (const ArtifactRootReference &candidate : candidateSet.candidates()) {
    auto recordIt = records.find(candidate);
    if (recordIt == records.end()) {
      if (!incomplete)
        incomplete = std::make_pair(IncompleteSelectionReason::MissingEvidence,
                                    candidate);
      continue;
    }
    const PromotionEvidence &record = *recordIt->second;
    if (obligationShape &&
        !sameObligationShape(*obligationShape, record.request, candidateRole))
      return invalid("candidate Evidence obligations are not same-shaped");
    obligationShape = &record.request;

    auto evidenceReference =
        evaluation::publishEvaluationEvidence(record.evidence, artifactStore);
    if (!evidenceReference)
      return evidenceReference.takeError();
    retainedEvidence.push_back(*evidenceReference);

    const auto *completed =
        std::get_if<evaluation::CompletedEvidence>(&record.evidence.outcome());
    if (!completed) {
      IncompleteSelectionReason reason =
          IncompleteSelectionReason::UnsupportedEvidence;
      switch (record.evidence.outcomeKind()) {
      case evaluation::EvidenceOutcomeKind::Completed:
        llvm_unreachable("Completed outcome variant mismatch");
      case evaluation::EvidenceOutcomeKind::Unsupported:
        break;
      case evaluation::EvidenceOutcomeKind::ExecutionFailed:
        reason = IncompleteSelectionReason::ExecutionFailedEvidence;
        break;
      case evaluation::EvidenceOutcomeKind::CancelledOrTimeout:
        reason = IncompleteSelectionReason::CancelledOrTimeoutEvidence;
        break;
      }
      if (!incomplete)
        incomplete = std::make_pair(reason, candidate);
      continue;
    }

    const std::uint64_t ordinal = selection.metricRequest.ordinal();
    if (ordinal >= completed->metricResults.size() ||
        !record.request.resolve(selection.metricRequest))
      return invalid("TopK metric ordinal is outside the Request");
    const auto *point = std::get_if<evaluation::PointObservation>(
        &completed->metricResults[ordinal].observation);
    if (!point) {
      if (!incomplete)
        incomplete = std::make_pair(
            IncompleteSelectionReason::NonComparableEvidence, candidate);
      continue;
    }
    if (metricValueKind && *metricValueKind != point->value.index())
      return invalid("comparable Metric results use different value kinds");
    metricValueKind = point->value.index();
    ranked.emplace(candidate, RankedCandidate{candidate, point->value,
                                              *evidenceReference});
  }

  llvm::sort(retainedEvidence, artifactRootReferenceLess);
  retainedEvidence.erase(
      std::unique(retainedEvidence.begin(), retainedEvidence.end()),
      retainedEvidence.end());
  if (incomplete) {
    return PromotionOutcome{IncompleteSelection{
        incomplete->first, incomplete->second, std::move(retainedEvidence)}};
  }

  std::vector<RankedCandidate> ordered;
  ordered.reserve(ranked.size());
  for (const auto &[candidate, record] : ranked) {
    (void)candidate;
    ordered.push_back(record);
  }
  llvm::sort(ordered,
             [&](const RankedCandidate &lhs, const RankedCandidate &rhs) {
               const int comparison = compareMetricValue(lhs.value, rhs.value);
               if (comparison != 0)
                 return selection.direction == ObjectiveDirection::Minimize
                            ? comparison < 0
                            : comparison > 0;
               return artifactRootReferenceLess(lhs.candidate, rhs.candidate);
             });

  const std::size_t selectedCount = static_cast<std::size_t>(
      std::min<std::uint64_t>(selection.k, ordered.size()));
  std::vector<ArtifactRootReference> selected;
  selected.reserve(selectedCount);
  for (std::size_t index = 0; index < selectedCount; ++index)
    selected.push_back(ordered[index].candidate);
  llvm::sort(selected, artifactRootReferenceLess);
  return PromotionOutcome{
      CompletedSelection{std::move(selected), std::move(retainedEvidence)}};
}

} // namespace loom::dse
