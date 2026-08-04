#include "DSE/Promotion.h"

#include "Common/ArtifactLocalReference.h"
#include "Common/ArtifactStore.h"
#include "Evaluation/NumericValue.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
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

std::uint64_t signedMagnitude(std::int64_t value) {
  if (value >= 0)
    return static_cast<std::uint64_t>(value);
  return static_cast<std::uint64_t>(-(value + 1)) + 1;
}

ResolvedObjectiveScalar objectiveScalar(const evaluation::MetricValue &value) {
  if (const auto *integer = std::get_if<evaluation::IntegerValue>(&value))
    return resolvedObjectiveInteger(signedMagnitude(integer->value()),
                                    integer->value() < 0);
  const auto decimal = std::get<evaluation::DecimalValue>(value);
  return resolvedObjectiveDecimal(decimal.coefficient(),
                                  decimal.base10Exponent());
}

int compareMetricValues(const evaluation::MetricValue &lhs,
                        const evaluation::MetricValue &rhs) {
  assert(lhs.index() == rhs.index() &&
         "metric value kinds must be validated before comparison");
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

bool metricValueLess(const evaluation::MetricValue &lhs,
                     const evaluation::MetricValue &rhs) {
  if (lhs.index() != rhs.index())
    return lhs.index() < rhs.index();
  return compareMetricValues(lhs, rhs) < 0;
}

bool atomLess(const QualityGateAtom &lhs, const QualityGateAtom &rhs) {
  if (lhs.index() != rhs.index())
    return lhs.index() < rhs.index();
  if (const auto *left = std::get_if<MetricGate>(&lhs)) {
    const auto &right = std::get<MetricGate>(rhs);
    if (left->evidenceObligationTemplate != right.evidenceObligationTemplate)
      return left->evidenceObligationTemplate <
             right.evidenceObligationTemplate;
    if (left->metricRequest.ordinal() != right.metricRequest.ordinal())
      return left->metricRequest.ordinal() < right.metricRequest.ordinal();
    if (left->comparator != right.comparator)
      return left->comparator < right.comparator;
    return metricValueLess(left->threshold, right.threshold);
  }
  const auto &left = std::get<FindingGate>(lhs);
  const auto &right = std::get<FindingGate>(rhs);
  if (left.evidenceObligationTemplate != right.evidenceObligationTemplate)
    return left.evidenceObligationTemplate < right.evidenceObligationTemplate;
  if (left.findingRequest.ordinal() != right.findingRequest.ordinal())
    return left.findingRequest.ordinal() < right.findingRequest.ordinal();
  return left.requiredState < right.requiredState;
}

bool atomEqual(const QualityGateAtom &lhs, const QualityGateAtom &rhs) {
  return !atomLess(lhs, rhs) && !atomLess(rhs, lhs);
}

bool clauseLess(const QualityGateClause &lhs, const QualityGateClause &rhs) {
  return std::lexicographical_compare(lhs.atoms.begin(), lhs.atoms.end(),
                                      rhs.atoms.begin(), rhs.atoms.end(),
                                      atomLess);
}

bool validComparator(MetricGateComparator comparator) {
  return static_cast<std::uint32_t>(comparator) <=
         static_cast<std::uint32_t>(MetricGateComparator::GT);
}

bool validFindingState(RequiredFindingState state) {
  return static_cast<std::uint32_t>(state) <=
         static_cast<std::uint32_t>(RequiredFindingState::Absent);
}

bool provesAll(MetricGateComparator comparator, int lowerToThreshold,
               int upperToThreshold) {
  switch (comparator) {
  case MetricGateComparator::LT:
    return upperToThreshold < 0;
  case MetricGateComparator::LE:
    return upperToThreshold <= 0;
  case MetricGateComparator::EQ:
    return lowerToThreshold == 0 && upperToThreshold == 0;
  case MetricGateComparator::NE:
    return upperToThreshold < 0 || lowerToThreshold > 0;
  case MetricGateComparator::GE:
    return lowerToThreshold >= 0;
  case MetricGateComparator::GT:
    return lowerToThreshold > 0;
  }
  llvm_unreachable("unknown metric gate comparator");
}

bool provesNone(MetricGateComparator comparator, int lowerToThreshold,
                int upperToThreshold) {
  switch (comparator) {
  case MetricGateComparator::LT:
    return lowerToThreshold >= 0;
  case MetricGateComparator::LE:
    return lowerToThreshold > 0;
  case MetricGateComparator::EQ:
    return upperToThreshold < 0 || lowerToThreshold > 0;
  case MetricGateComparator::NE:
    return lowerToThreshold == 0 && upperToThreshold == 0;
  case MetricGateComparator::GE:
    return upperToThreshold < 0;
  case MetricGateComparator::GT:
    return upperToThreshold <= 0;
  }
  llvm_unreachable("unknown metric gate comparator");
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

struct EvidenceKey final {
  ArtifactRootReference candidate;
  std::uint32_t obligationTemplate;
};

struct EvidenceKeyLess final {
  bool operator()(const EvidenceKey &lhs, const EvidenceKey &rhs) const {
    if (artifactRootReferenceLess(lhs.candidate, rhs.candidate))
      return true;
    if (artifactRootReferenceLess(rhs.candidate, lhs.candidate))
      return false;
    return lhs.obligationTemplate < rhs.obligationTemplate;
  }
};

using PromotionEvidenceRecords =
    std::map<EvidenceKey, const PromotionEvidence *, EvidenceKeyLess>;

llvm::Expected<PromotionEvidenceRecords>
indexPromotionEvidence(const CandidateSet &candidateSet,
                       evaluation::CaseSubjectRoleRef candidateRole,
                       llvm::ArrayRef<PromotionEvidence> evidence) {
  PromotionEvidenceRecords records;
  std::map<std::uint32_t, const evaluation::EvaluationRequest *>
      obligationShapes;
  for (const PromotionEvidence &record : evidence) {
    if (record.evidence.requestRef() !=
        evaluation::evaluationRequestReference(record.request))
      return invalid("Evidence does not reference its supplied Request");
    const llvm::ArrayRef<ArtifactRootReference> subjects =
        record.request.subjectBindings().subjects(candidateRole);
    if (subjects.size() != 1)
      return invalid("candidate role is not bound to exactly one Artifact");
    if (!containsCandidate(candidateSet, subjects.front()))
      return invalid("Evidence names a candidate outside the input set");
    if (!records
             .emplace(EvidenceKey{subjects.front(), record.obligationTemplate},
                      &record)
             .second)
      return invalid("candidate has duplicate Evidence obligations");
    const auto shape = obligationShapes.find(record.obligationTemplate);
    if (shape == obligationShapes.end()) {
      obligationShapes.emplace(record.obligationTemplate, &record.request);
    } else if (!sameObligationShape(*shape->second, record.request,
                                    candidateRole)) {
      return invalid("candidate Evidence obligations are not same-shaped");
    }
  }
  return records;
}

llvm::Expected<std::vector<ArtifactRootReference>>
publishPromotionEvidence(const PromotionEvidenceRecords &records,
                         const ArtifactStore &artifactStore) {
  std::vector<ArtifactRootReference> references;
  references.reserve(records.size());
  for (const auto &[key, record] : records) {
    (void)key;
    auto reference =
        evaluation::publishEvaluationEvidence(record->evidence, artifactStore);
    if (!reference)
      return reference.takeError();
    references.push_back(*reference);
  }
  llvm::sort(references, artifactRootReferenceLess);
  references.erase(std::unique(references.begin(), references.end()),
                   references.end());
  return references;
}

IncompleteSelectionReason
incompleteReason(const evaluation::EvaluationEvidence &value) {
  switch (value.outcomeKind()) {
  case evaluation::EvidenceOutcomeKind::Completed:
    llvm_unreachable("completed Evidence has no incomplete reason");
  case evaluation::EvidenceOutcomeKind::Unsupported:
    return IncompleteSelectionReason::UnsupportedEvidence;
  case evaluation::EvidenceOutcomeKind::ExecutionFailed:
    return IncompleteSelectionReason::ExecutionFailedEvidence;
  case evaluation::EvidenceOutcomeKind::CancelledOrTimeout:
    return IncompleteSelectionReason::CancelledOrTimeoutEvidence;
  }
  llvm_unreachable("unknown Evidence outcome");
}

llvm::Expected<ObjectiveVector>
deriveObjectiveVector(const ArtifactRootReference &candidate,
                      const PromotionEvidenceRecords &records,
                      const ObjectiveProgram &objectiveProgram) {
  std::vector<EvaluationMetricObjectiveValue> metrics;
  auto record = records.lower_bound(EvidenceKey{candidate, 0});
  while (record != records.end() && record->first.candidate == candidate) {
    const PromotionEvidence &evidenceRecord = *record->second;
    const auto *completed = std::get_if<evaluation::CompletedEvidence>(
        &evidenceRecord.evidence.outcome());
    if (completed) {
      for (std::size_t ordinal = 0; ordinal != completed->metricResults.size();
           ++ordinal) {
        const auto *point = std::get_if<evaluation::PointObservation>(
            &completed->metricResults[ordinal].observation);
        if (!point)
          continue;
        metrics.push_back({record->first.obligationTemplate,
                           static_cast<std::uint64_t>(ordinal),
                           objectiveScalar(point->value)});
      }
    }
    ++record;
  }
  ObjectiveVector vector = objectiveProgram.makeVector();
  if (llvm::Error error = objectiveProgram.evaluate({{}, {}, metrics}, vector))
    return std::move(error);
  return vector;
}

} // namespace

llvm::Expected<QualityGatePolicy>
QualityGatePolicy::get(std::vector<QualityGateClause> clauses) {
  std::size_t atomCount = 0;
  for (QualityGateClause &clause : clauses) {
    if (clause.atoms.empty())
      return invalid("quality gate contains an empty clause");
    for (const QualityGateAtom &atom : clause.atoms) {
      if (const auto *metric = std::get_if<MetricGate>(&atom)) {
        if (!validComparator(metric->comparator))
          return invalid("quality gate metric comparator is unknown");
      } else if (!validFindingState(
                     std::get<FindingGate>(atom).requiredState)) {
        return invalid("quality gate finding state is unknown");
      }
    }
    llvm::sort(clause.atoms, atomLess);
    clause.atoms.erase(
        std::unique(clause.atoms.begin(), clause.atoms.end(), atomEqual),
        clause.atoms.end());
    if (clause.atoms.size() >
        std::numeric_limits<std::size_t>::max() - atomCount)
      return invalid("quality gate atom count overflows size_t");
    atomCount += clause.atoms.size();
  }
  llvm::sort(clauses, clauseLess);
  clauses.erase(std::unique(clauses.begin(), clauses.end(),
                            [](const QualityGateClause &lhs,
                               const QualityGateClause &rhs) {
                              return !clauseLess(lhs, rhs) &&
                                     !clauseLess(rhs, lhs);
                            }),
                clauses.end());
  atomCount = 0;
  for (const QualityGateClause &clause : clauses)
    atomCount += clause.atoms.size();
  return QualityGatePolicy(std::move(clauses), atomCount);
}

llvm::Expected<GateTruth> evaluateMetricGate(
    evaluation::MetricKind metric, const evaluation::MetricResult &result,
    MetricGateComparator comparator, evaluation::MetricValue threshold) {
  if (!validComparator(comparator))
    return invalid("metric gate comparator is unknown");
  if (llvm::Error error = evaluation::validateMetricObservationValue(
          metric, result.uncertainty, result.observation))
    return std::move(error);
  if (llvm::Error error = evaluation::validateMetricObservationValue(
          metric, evaluation::UncertaintyKind::ExactWithinModel,
          evaluation::PointObservation{threshold}))
    return std::move(error);
  if (std::holds_alternative<evaluation::NotApplicableObservation>(
          result.observation))
    return GateTruth::Indeterminate;

  std::optional<evaluation::MetricValue> lower;
  std::optional<evaluation::MetricValue> upper;
  if (const auto *point =
          std::get_if<evaluation::PointObservation>(&result.observation)) {
    lower = point->value;
    upper = point->value;
  } else if (const auto *interval =
                 std::get_if<evaluation::IntervalObservation>(
                     &result.observation)) {
    lower = interval->lower;
    upper = interval->upper;
  } else {
    const auto &censored =
        std::get<evaluation::CensoredObservation>(result.observation);
    lower = censored.lower;
    upper = censored.upper;
  }
  if ((lower && lower->index() != threshold.index()) ||
      (upper && upper->index() != threshold.index()))
    return invalid("metric gate threshold has the wrong value kind");

  const int lowerComparison =
      lower ? compareMetricValues(*lower, threshold) : -1;
  const int upperComparison =
      upper ? compareMetricValues(*upper, threshold) : 1;
  const bool all =
      lower && upper
          ? provesAll(comparator, lowerComparison, upperComparison)
          : (lower &&
             (comparator == MetricGateComparator::GE   ? lowerComparison >= 0
              : comparator == MetricGateComparator::GT ? lowerComparison > 0
              : comparator == MetricGateComparator::NE ? lowerComparison > 0
                                                       : false)) ||
                (upper &&
                 (comparator == MetricGateComparator::LT ? upperComparison < 0
                  : comparator == MetricGateComparator::LE
                      ? upperComparison <= 0
                  : comparator == MetricGateComparator::NE ? upperComparison < 0
                                                           : false));
  if (all)
    return GateTruth::DefinitelyTrue;

  const bool none =
      lower && upper
          ? provesNone(comparator, lowerComparison, upperComparison)
          : (lower &&
             (comparator == MetricGateComparator::LT   ? lowerComparison >= 0
              : comparator == MetricGateComparator::LE ? lowerComparison > 0
              : comparator == MetricGateComparator::EQ ? lowerComparison > 0
                                                       : false)) ||
                (upper &&
                 (comparator == MetricGateComparator::GE ? upperComparison < 0
                  : comparator == MetricGateComparator::GT
                      ? upperComparison <= 0
                  : comparator == MetricGateComparator::EQ ? upperComparison < 0
                                                           : false));
  if (none)
    return GateTruth::DefinitelyFalse;
  return GateTruth::Indeterminate;
}

GateTruth evaluateFindingGate(const evaluation::FindingResult &result,
                              RequiredFindingState requiredState) {
  if (std::holds_alternative<evaluation::NotApplicableFinding>(result.result))
    return GateTruth::Indeterminate;
  const bool present =
      std::holds_alternative<evaluation::PresentFinding>(result.result);
  const bool requiredPresent = requiredState == RequiredFindingState::Present;
  return present == requiredPresent ? GateTruth::DefinitelyTrue
                                    : GateTruth::DefinitelyFalse;
}

llvm::Expected<GateTruth>
evaluateQualityGate(const QualityGatePolicy &policy,
                    llvm::ArrayRef<GateTruth> atomTruths) {
  if (atomTruths.size() != policy.atomCount())
    return invalid("quality gate truth count does not match the policy");
  if (llvm::is_contained(atomTruths, GateTruth::Indeterminate))
    return GateTruth::Indeterminate;

  std::size_t ordinal = 0;
  for (const QualityGateClause &clause : policy.clauses()) {
    bool clauseTrue = false;
    for (std::size_t atom = 0; atom < clause.atoms.size(); ++atom)
      clauseTrue |= atomTruths[ordinal++] == GateTruth::DefinitelyTrue;
    if (!clauseTrue)
      return GateTruth::DefinitelyFalse;
  }
  return GateTruth::DefinitelyTrue;
}

llvm::StringRef toString(IncompleteSelectionReason reason) {
  switch (reason) {
  case IncompleteSelectionReason::MissingEvidence:
    return "missing_evidence";
  case IncompleteSelectionReason::UnsupportedEvidence:
    return "unsupported_evidence";
  case IncompleteSelectionReason::ExecutionFailedEvidence:
    return "execution_failed_evidence";
  case IncompleteSelectionReason::CancelledOrTimeoutEvidence:
    return "cancelled_or_timeout_evidence";
  case IncompleteSelectionReason::NonComparableEvidence:
    return "non_comparable_evidence";
  case IncompleteSelectionReason::ObjectiveUnavailable:
    return "objective_unavailable";
  }
  llvm_unreachable("unknown IncompleteSelectionReason");
}

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

llvm::Expected<std::vector<ArtifactRootReference>> applyCandidateSelection(
    const CandidateSet &candidateSet,
    llvm::ArrayRef<ArtifactRootReference> gateQualifiedCandidates,
    llvm::ArrayRef<CandidateObjectiveVector> objectives,
    const CandidateSelectionPolicy &selection,
    const ObjectiveProgram *objectiveProgram) {
  std::vector<ArtifactRootReference> eligible(gateQualifiedCandidates.begin(),
                                              gateQualifiedCandidates.end());
  llvm::sort(eligible, artifactRootReferenceLess);
  eligible.erase(std::unique(eligible.begin(), eligible.end()), eligible.end());
  for (const ArtifactRootReference &candidate : eligible)
    if (!containsCandidate(candidateSet, candidate))
      return invalid("gate-qualified candidate is outside the input set");

  if (std::holds_alternative<AllPassingSelection>(selection))
    return eligible;
  if (!objectiveProgram)
    return invalid("objective selection has no ObjectiveProgram");

  std::map<ArtifactRootReference, const CandidateObjectiveVector *,
           decltype(&artifactRootReferenceLess)>
      objectiveByCandidate(&artifactRootReferenceLess);
  for (const CandidateObjectiveVector &objective : objectives) {
    if (!containsCandidate(candidateSet, objective.candidate))
      return invalid(
          "objective vector names a candidate outside the input set");
    if (!objectiveByCandidate.emplace(objective.candidate, &objective).second)
      return invalid("candidate has duplicate objective vectors");
  }

  struct SelectionRecord final {
    ArtifactRootReference candidate;
    const ObjectiveVector *objective;
    std::vector<std::uint8_t> candidateKey;
  };
  std::vector<SelectionRecord> records;
  records.reserve(eligible.size());
  for (const ArtifactRootReference &candidate : eligible) {
    const auto objective = objectiveByCandidate.find(candidate);
    if (objective == objectiveByCandidate.end())
      return invalid("gate-qualified candidate has no objective vector");
    records.push_back({candidate, &objective->second->objective,
                       encodeArtifactRootReference(candidate)});
  }

  if (const auto *topK = std::get_if<TopKSelection>(&selection)) {
    if (topK->k == 0)
      return invalid("TopK requires positive k");
    if (topK->totalOrdering >= objectiveProgram->totalOrderingCount())
      return invalid("TopK total ordering reference is out of range");
    for (const SelectionRecord &record : records) {
      auto validated = objectiveProgram->compareTotalOrdering(
          *record.objective, record.candidateKey, *record.objective,
          record.candidateKey, topK->totalOrdering);
      if (!validated)
        return validated.takeError();
    }
    const std::size_t selectedCount = static_cast<std::size_t>(
        std::min<std::uint64_t>(topK->k, records.size()));
    std::partial_sort(
        records.begin(), records.begin() + selectedCount, records.end(),
        [&](const SelectionRecord &lhs, const SelectionRecord &rhs) {
          return llvm::cantFail(objectiveProgram->compareTotalOrdering(
                     *lhs.objective, lhs.candidateKey, *rhs.objective,
                     rhs.candidateKey, topK->totalOrdering)) < 0;
        });
    eligible.clear();
    eligible.reserve(selectedCount);
    for (std::size_t index = 0; index < selectedCount; ++index)
      eligible.push_back(records[index].candidate);
    llvm::sort(eligible, artifactRootReferenceLess);
    return eligible;
  }

  const auto &pareto = std::get<ParetoSelection>(selection);
  if (pareto.objectiveDimensions.empty())
    return invalid("Pareto dimension set is empty");
  if (!llvm::is_sorted(pareto.objectiveDimensions) ||
      std::adjacent_find(pareto.objectiveDimensions.begin(),
                         pareto.objectiveDimensions.end()) !=
          pareto.objectiveDimensions.end())
    return invalid("Pareto dimensions are not canonical");
  for (const SelectionRecord &record : records) {
    auto validated = objectiveProgram->comparePareto(
        *record.objective, *record.objective, pareto.objectiveDimensions);
    if (!validated)
      return validated.takeError();
  }

  std::vector<bool> dominated(records.size(), false);
  for (std::size_t candidate = 0; candidate < records.size(); ++candidate) {
    for (std::size_t challenger = 0; challenger < records.size();
         ++challenger) {
      if (candidate == challenger)
        continue;
      const ParetoRelation relation =
          llvm::cantFail(objectiveProgram->comparePareto(
              *records[challenger].objective, *records[candidate].objective,
              pareto.objectiveDimensions));
      if (relation == ParetoRelation::Dominates) {
        dominated[candidate] = true;
        break;
      }
    }
  }
  eligible.clear();
  for (std::size_t index = 0; index < records.size(); ++index)
    if (!dominated[index])
      eligible.push_back(records[index].candidate);
  llvm::sort(eligible, artifactRootReferenceLess);
  return eligible;
}

llvm::Expected<CandidateObjectiveRankingOutcome> rankCandidatesByObjective(
    const CandidateSet &candidateSet,
    evaluation::CaseSubjectRoleRef candidateRole,
    llvm::ArrayRef<PromotionEvidence> evidence,
    llvm::ArrayRef<std::uint32_t> objectiveObligationTemplates,
    std::uint32_t totalOrdering, const ObjectiveProgram &objectiveProgram,
    const ArtifactStore &artifactStore) {
  if (objectiveObligationTemplates.empty())
    return invalid("objective ranking requires an Evidence obligation");
  if (!llvm::is_sorted(objectiveObligationTemplates) ||
      std::adjacent_find(objectiveObligationTemplates.begin(),
                         objectiveObligationTemplates.end()) !=
          objectiveObligationTemplates.end())
    return invalid("objective Evidence obligations are not canonical");
  if (totalOrdering >= objectiveProgram.totalOrderingCount())
    return invalid("objective ranking total ordering is out of range");

  auto indexed = indexPromotionEvidence(candidateSet, candidateRole, evidence);
  if (!indexed)
    return indexed.takeError();
  auto retained = publishPromotionEvidence(*indexed, artifactStore);
  if (!retained)
    return retained.takeError();

  struct RankingRecord final {
    ArtifactRootReference candidate;
    ObjectiveVector objective;
    std::vector<std::uint8_t> candidateKey;
  };
  std::vector<RankingRecord> ranking;
  ranking.reserve(candidateSet.candidates().size());
  for (const ArtifactRootReference &candidate : candidateSet.candidates()) {
    for (std::uint32_t obligation : objectiveObligationTemplates) {
      const auto found = indexed->find(EvidenceKey{candidate, obligation});
      if (found == indexed->end())
        return CandidateObjectiveRankingOutcome{
            IncompleteSelection{IncompleteSelectionReason::MissingEvidence,
                                candidate, std::move(*retained)}};
      if (found->second->evidence.outcomeKind() !=
          evaluation::EvidenceOutcomeKind::Completed)
        return CandidateObjectiveRankingOutcome{
            IncompleteSelection{incompleteReason(found->second->evidence),
                                candidate, std::move(*retained)}};
    }

    auto objective =
        deriveObjectiveVector(candidate, *indexed, objectiveProgram);
    if (!objective) {
      bool unavailable = false;
      llvm::Error remaining = llvm::handleErrors(
          objective.takeError(),
          [&](const ObjectiveUnavailableError &) -> llvm::Error {
            unavailable = true;
            return llvm::Error::success();
          });
      if (remaining)
        return std::move(remaining);
      if (unavailable)
        return CandidateObjectiveRankingOutcome{
            IncompleteSelection{IncompleteSelectionReason::ObjectiveUnavailable,
                                candidate, std::move(*retained)}};
      llvm_unreachable("handled objective error had no classification");
    }
    ranking.push_back({candidate, std::move(*objective),
                       encodeArtifactRootReference(candidate)});
  }

  for (const RankingRecord &record : ranking) {
    auto validated = objectiveProgram.compareTotalOrdering(
        record.objective, record.candidateKey, record.objective,
        record.candidateKey, totalOrdering);
    if (!validated)
      return validated.takeError();
  }
  llvm::sort(ranking, [&](const RankingRecord &lhs, const RankingRecord &rhs) {
    return llvm::cantFail(objectiveProgram.compareTotalOrdering(
               lhs.objective, lhs.candidateKey, rhs.objective, rhs.candidateKey,
               totalOrdering)) < 0;
  });

  std::vector<ArtifactRootReference> rankedCandidates;
  rankedCandidates.reserve(ranking.size());
  for (RankingRecord &record : ranking)
    rankedCandidates.push_back(std::move(record.candidate));
  return CandidateObjectiveRankingOutcome{CompletedCandidateObjectiveRanking{
      std::move(rankedCandidates), std::move(*retained)}};
}

llvm::Expected<PromotionOutcome>
promoteCandidates(const CandidateSet &candidateSet,
                  evaluation::CaseSubjectRoleRef candidateRole,
                  llvm::ArrayRef<PromotionEvidence> evidence,
                  const QualityGatePolicy &qualityGate,
                  const CandidateSelectionPolicy &selection,
                  const ObjectiveProgram *objectiveProgram,
                  const ArtifactStore &artifactStore) {
  if (candidateSet.candidates().empty())
    return PromotionOutcome{CompletedNoFeasibleCandidate{}};
  if (!std::holds_alternative<AllPassingSelection>(selection) &&
      !objectiveProgram)
    return invalid("objective selection has no ObjectiveProgram");

  auto indexed = indexPromotionEvidence(candidateSet, candidateRole, evidence);
  if (!indexed)
    return indexed.takeError();
  const PromotionEvidenceRecords &records = *indexed;
  auto retained = publishPromotionEvidence(records, artifactStore);
  if (!retained)
    return retained.takeError();
  std::vector<ArtifactRootReference> retainedEvidence = std::move(*retained);

  std::optional<std::pair<IncompleteSelectionReason, ArtifactRootReference>>
      incomplete;
  std::vector<ArtifactRootReference> gateQualified;
  for (const ArtifactRootReference &candidate : candidateSet.candidates()) {
    std::vector<GateTruth> truths;
    truths.reserve(qualityGate.atomCount());
    bool missingResult = false;
    for (const QualityGateClause &clause : qualityGate.clauses()) {
      for (const QualityGateAtom &atom : clause.atoms) {
        const std::uint32_t obligation = std::visit(
            [](const auto &gate) { return gate.evidenceObligationTemplate; },
            atom);
        const auto found = records.find(EvidenceKey{candidate, obligation});
        if (found == records.end()) {
          if (!incomplete)
            incomplete = std::make_pair(
                IncompleteSelectionReason::MissingEvidence, candidate);
          missingResult = true;
          continue;
        }
        const PromotionEvidence &record = *found->second;
        const auto *completed = std::get_if<evaluation::CompletedEvidence>(
            &record.evidence.outcome());
        if (!completed) {
          if (!incomplete)
            incomplete =
                std::make_pair(incompleteReason(record.evidence), candidate);
          missingResult = true;
          continue;
        }
        if (const auto *metric = std::get_if<MetricGate>(&atom)) {
          const evaluation::MetricRequest *request =
              record.request.resolve(metric->metricRequest);
          if (!request || metric->metricRequest.ordinal() >=
                              completed->metricResults.size())
            return invalid(
                "quality gate metric ordinal is outside the Request");
          auto truth = evaluateMetricGate(
              request->query().metric,
              completed->metricResults[metric->metricRequest.ordinal()],
              metric->comparator, metric->threshold);
          if (!truth)
            return truth.takeError();
          truths.push_back(*truth);
        } else {
          const auto &finding = std::get<FindingGate>(atom);
          if (!record.request.resolve(finding.findingRequest) ||
              finding.findingRequest.ordinal() >=
                  completed->findingResults.size())
            return invalid(
                "quality gate finding ordinal is outside the Request");
          truths.push_back(evaluateFindingGate(
              completed->findingResults[finding.findingRequest.ordinal()],
              finding.requiredState));
        }
      }
    }
    if (missingResult)
      continue;
    auto gateResult = evaluateQualityGate(qualityGate, truths);
    if (!gateResult)
      return gateResult.takeError();
    if (*gateResult == GateTruth::Indeterminate) {
      if (!incomplete)
        incomplete = std::make_pair(
            IncompleteSelectionReason::NonComparableEvidence, candidate);
      continue;
    }
    if (*gateResult == GateTruth::DefinitelyTrue)
      gateQualified.push_back(candidate);
  }
  if (incomplete)
    return PromotionOutcome{IncompleteSelection{
        incomplete->first, incomplete->second, std::move(retainedEvidence)}};

  std::vector<CandidateObjectiveVector> objectives;
  if (!std::holds_alternative<AllPassingSelection>(selection)) {
    objectives.reserve(gateQualified.size());
    for (const ArtifactRootReference &candidate : gateQualified) {
      auto vector =
          deriveObjectiveVector(candidate, records, *objectiveProgram);
      if (!vector) {
        bool unavailable = false;
        llvm::Error remaining = llvm::handleErrors(
            vector.takeError(),
            [&](const ObjectiveUnavailableError &) -> llvm::Error {
              unavailable = true;
              return llvm::Error::success();
            });
        if (remaining)
          return std::move(remaining);
        if (unavailable)
          return PromotionOutcome{IncompleteSelection{
              IncompleteSelectionReason::ObjectiveUnavailable, candidate,
              std::move(retainedEvidence)}};
        llvm_unreachable("handled objective error had no classification");
      }
      objectives.push_back({candidate, std::move(*vector)});
    }
  }

  auto selected = applyCandidateSelection(
      candidateSet, gateQualified, objectives, selection, objectiveProgram);
  if (!selected)
    return selected.takeError();
  if (selected->empty())
    return PromotionOutcome{
        CompletedNoFeasibleCandidate{std::move(retainedEvidence)}};
  return PromotionOutcome{
      CompletedSelection{std::move(*selected), std::move(retainedEvidence)}};
}

} // namespace loom::dse
