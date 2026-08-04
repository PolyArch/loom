#include "DSE/Promotion.h"

#include "Common/Artifact.h"
#include "Common/ResolvedPnrPolicy.h"
#include "Evaluation/Metric.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace loom;
using namespace loom::dse;
using namespace loom::evaluation;

[[noreturn]] void fail(llvm::StringRef message) {
  llvm::errs() << "DSE promotion test: " << message << '\n';
  std::exit(1);
}

void require(bool condition, llvm::StringRef message) {
  if (!condition)
    fail(message);
}

template <typename T> T take(llvm::Expected<T> value) {
  if (!value)
    fail(llvm::toString(value.takeError()));
  return std::move(*value);
}

void requireSuccess(llvm::Error error) {
  if (error)
    fail(llvm::toString(std::move(error)));
}

ArtifactRootReference makeReference(std::uint8_t fill) {
  std::array<std::uint8_t, ArtifactIdentity::byteSize> bytes{};
  bytes.fill(fill);
  return {"loom.test.promotion_candidate", SchemaVersion{1, 0},
          take(ArtifactIdentity::fromBytes(bytes))};
}

void metricGateUsesRepresentedSetProof() {
  MetricResult interval{UncertaintyKind::Bounded,
                        IntervalObservation{IntegerValue(0), IntegerValue(30)},
                        {}};
  require(take(evaluateMetricGate(
              MetricKind::CycleCount, interval, MetricGateComparator::LT,
              IntegerValue(40))) == GateTruth::DefinitelyTrue,
          "an interval wholly below the threshold was not proven true");
  require(take(evaluateMetricGate(
              MetricKind::CycleCount, interval, MetricGateComparator::LT,
              IntegerValue(20))) == GateTruth::Indeterminate,
          "a straddling interval did not remain indeterminate");
  require(take(evaluateMetricGate(
              MetricKind::CycleCount, interval, MetricGateComparator::GT,
              IntegerValue(40))) == GateTruth::DefinitelyFalse,
          "an interval wholly below a greater-than threshold was not false");

  MetricResult censored{
      UncertaintyKind::Bounded,
      CensoredObservation{MetricValue{IntegerValue(25)}, std::nullopt,
                          CensoredReason::SubjectDidNotComplete},
      {}};
  require(take(evaluateMetricGate(
              MetricKind::CycleCount, censored, MetricGateComparator::GE,
              IntegerValue(20))) == GateTruth::DefinitelyTrue,
          "a lower-censored set did not prove its lower-bound gate");
  require(take(evaluateMetricGate(
              MetricKind::CycleCount, censored, MetricGateComparator::LT,
              IntegerValue(30))) == GateTruth::Indeterminate,
          "an unbounded censored set was treated as a point");
}

void indeterminateAtomPrecedesBooleanSelection() {
  QualityGateClause clause;
  clause.atoms = {
      MetricGate{0, MetricRequestOrdinal(0), MetricGateComparator::LT,
                 IntegerValue(10)},
      FindingGate{1, FindingRequestOrdinal(0), RequiredFindingState::Absent},
  };
  QualityGatePolicy policy = take(QualityGatePolicy::get({std::move(clause)}));
  require(policy.atomCount() == 2,
          "quality policy did not preserve distinct canonical atoms");
  const std::array<GateTruth, 2> truths = {GateTruth::DefinitelyTrue,
                                           GateTruth::Indeterminate};
  require(take(evaluateQualityGate(policy, truths)) == GateTruth::Indeterminate,
          "a true sibling incorrectly hid an indeterminate obligation");
}

void paretoRetainsEveryNondominatedCandidate() {
  const ResolvedObjectiveCatalogs catalogs = resolvedBuiltinObjectiveCatalogs();
  const ObjectiveProgram program = take(ObjectiveProgram::get(catalogs));
  const ArtifactSchemaDescriptor schema{"loom.test.promotion_candidate",
                                        SchemaVersion{1, 0}};
  const ArtifactRootReference first = makeReference(0x11);
  const ArtifactRootReference second = makeReference(0x22);
  const ArtifactRootReference dominated = makeReference(0x33);
  const CandidateSet candidates =
      take(CandidateSet::get(schema, {dominated, second, first}));

  auto makeObjective = [&](const ArtifactRootReference &candidate,
                           std::uint64_t violation, std::uint64_t traversal) {
    std::vector<std::uint64_t> violations(resolvedPnrViolationKindCount, 0);
    violations[0] = violation;
    ObjectiveVector vector = program.makeVector();
    requireSuccess(program.evaluate({violations, {&traversal, 1}}, vector));
    return CandidateObjectiveVector{candidate, std::move(vector)};
  };
  std::vector<CandidateObjectiveVector> objectives;
  objectives.push_back(makeObjective(first, 0, 5));
  objectives.push_back(makeObjective(second, 5, 0));
  objectives.push_back(makeObjective(dominated, 6, 6));
  const std::array<std::uint32_t, 2> dimensions = {
      0, resolvedPnrViolationKindCount};
  const CandidateSelectionPolicy policy = ParetoSelection{
      std::vector<std::uint32_t>(dimensions.begin(), dimensions.end())};
  const std::vector<ArtifactRootReference> selected =
      take(applyCandidateSelection(candidates, candidates.candidates(),
                                   objectives, policy, &program));
  require(selected == std::vector<ArtifactRootReference>({first, second}),
          "Pareto selection did not return the canonical nondominated set");
}

} // namespace

int main() {
  metricGateUsesRepresentedSetProof();
  indeterminateAtomPrecedesBooleanSelection();
  paretoRetainsEveryNondominatedCandidate();
  return 0;
}
