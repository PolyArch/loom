#include "DSE/Objective.h"

#include "Common/ResolvedPnrPolicy.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace {

[[noreturn]] void fail(llvm::StringRef message) {
  llvm::errs() << "DSE objective test: " << message << '\n';
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

template <typename T>
void requireRejected(llvm::Expected<T> value, llvm::StringRef fragment) {
  if (value)
    fail("expected rejection");
  const std::string message = llvm::toString(value.takeError());
  require(llvm::StringRef(message).contains(fragment), message);
}

void requireRejected(llvm::Error error, llvm::StringRef fragment) {
  if (!error)
    fail("expected rejection");
  const std::string message = llvm::toString(std::move(error));
  require(llvm::StringRef(message).contains(fragment), message);
}

void exactAffineQuantizationIsCheckedAndDirected() {
  loom::ResolvedObjectiveCatalogs catalogs;
  catalogs.dimensions = {
      {loom::ResolvedMappingViolationObjectiveSource{
           loom::ResolvedPnrViolationKind::UnroutedObligation},
       loom::ResolvedObjectiveDirection::Minimize,
       loom::resolvedObjectiveInteger(10), loom::resolvedObjectiveInteger(3), 2,
       5},
      {loom::ResolvedMappingViolationObjectiveSource{
           loom::ResolvedPnrViolationKind::UnroutedObligation},
       loom::ResolvedObjectiveDirection::Maximize,
       loom::resolvedObjectiveInteger(10), loom::resolvedObjectiveInteger(3), 2,
       5},
  };
  const loom::dse::ObjectiveProgram program =
      take(loom::dse::ObjectiveProgram::get(catalogs));
  loom::dse::ObjectiveVector vector = program.makeVector();

  const std::uint64_t values[] = {20};
  requireSuccess(program.evaluate({values, {}, {}}, vector));
  require(vector.codes() == llvm::ArrayRef<std::uint64_t>({1, 2}),
          "directed affine codes are incorrect");

  const std::uint64_t belowOrigin[] = {9};
  requireRejected(program.evaluate({belowOrigin, {}, {}}, vector),
                  "objective_contract_failure");
  const std::uint64_t aboveUpperBound[] = {28};
  requireRejected(program.evaluate({aboveUpperBound, {}, {}}, vector),
                  "objective_contract_failure");
  requireRejected(program.evaluate({{}, {}, {}}, vector),
                  "objective_unavailable");
}

void evaluationDecimalQuantizationIsExactAndSparse() {
  loom::ResolvedObjectiveCatalogs catalogs;
  catalogs.dimensions = {
      {loom::ResolvedEvaluationMetricObjectiveSource{3, 5},
       loom::ResolvedObjectiveDirection::Minimize,
       loom::resolvedObjectiveDecimal(25, -2),
       loom::resolvedObjectiveDecimal(5, -2), 0, 20},
      {loom::ResolvedEvaluationMetricObjectiveSource{7, 11},
       loom::ResolvedObjectiveDirection::Maximize,
       loom::resolvedObjectiveDecimal(0, 0),
       loom::resolvedObjectiveDecimal(1, 1000000000), 0, 1},
  };
  const loom::dse::ObjectiveProgram program =
      take(loom::dse::ObjectiveProgram::get(catalogs));
  loom::dse::ObjectiveVector vector = program.makeVector();
  const loom::dse::EvaluationMetricObjectiveValue metrics[] = {
      {3, 5, loom::resolvedObjectiveDecimal(124, -2)},
      {7, 11, loom::resolvedObjectiveDecimal(1, 1000000000)},
  };
  requireSuccess(program.evaluate({{}, {}, metrics}, vector));
  require(vector.codes() == llvm::ArrayRef<std::uint64_t>({19, 0}),
          "decimal affine quantization rounded or expanded a huge exponent");

  const loom::dse::EvaluationMetricObjectiveValue missing[] = {
      {3, 5, loom::resolvedObjectiveDecimal(124, -2)},
  };
  requireRejected(program.evaluate({{}, {}, missing}, vector),
                  "objective_unavailable");
}

void builtinOrderingEnergyAndParetoUseOneVector() {
  const loom::ResolvedObjectiveCatalogs catalogs =
      loom::resolvedBuiltinObjectiveCatalogs();
  const loom::dse::ObjectiveProgram program =
      take(loom::dse::ObjectiveProgram::get(catalogs));
  std::size_t mappingMeasureCount = 0;
  for (const loom::ResolvedObjectiveDimension &dimension : catalogs.dimensions)
    if (const auto *measure =
            std::get_if<loom::ResolvedMappingMeasureObjectiveSource>(
                &dimension.source))
      mappingMeasureCount = std::max(
          mappingMeasureCount, static_cast<std::size_t>(measure->ordinal) + 1);

  std::vector<std::uint64_t> leftViolations(loom::resolvedPnrViolationKindCount,
                                            0);
  leftViolations[0] = 1;
  const std::vector<std::uint64_t> leftMeasures(mappingMeasureCount, 0);
  loom::dse::ObjectiveVector left = program.makeVector();
  requireSuccess(program.evaluate({leftViolations, leftMeasures, {}}, left));

  std::vector<std::uint64_t> rightViolations(
      loom::resolvedPnrViolationKindCount, 0);
  std::vector<std::uint64_t> rightMeasures(mappingMeasureCount, 0);
  rightMeasures.front() = std::numeric_limits<std::uint64_t>::max();
  loom::dse::ObjectiveVector right = program.makeVector();
  requireSuccess(program.evaluate({rightViolations, rightMeasures, {}}, right));

  const std::uint8_t leftKey[] = {0};
  const std::uint8_t rightKey[] = {1};
  require(
      take(program.compareTotalOrdering(left, leftKey, right, rightKey, 0)) > 0,
      "violation level did not dominate traversal quality");

  const std::uint32_t searchEnergyLevel =
      loom::resolvedBuiltinSpatialPnrPolicy(
          loom::ResolvedProfilePreset::BalancedExplore)
          .objectiveSelection.selectedSearchEnergy;
  const loom::dse::ObjectiveWideValue leftEnergy =
      take(program.weightedLevelValue(left, searchEnergyLevel));
  require(leftEnergy.high == 0 &&
              leftEnergy.low == UINT64_C(281474976710656),
          "search energy did not use the selected fixed weight");
  const loom::dse::ObjectiveSignedDifference delta =
      take(program.signedWeightedLevelDifference(left, right,
                                                 searchEnergyLevel));
  require(delta.sign == loom::dse::ObjectiveDifferenceSign::Negative,
          "energy difference has the wrong sign");

  const std::uint32_t paretoDimensions[] = {
      0, loom::resolvedPnrViolationKindCount};
  require(take(program.comparePareto(left, right, paretoDimensions)) ==
              loom::dse::ParetoRelation::Incomparable,
          "crossing objective dimensions must remain incomparable");

  loom::dse::ObjectiveVector zero = program.makeVector();
  const std::vector<std::uint64_t> zeroMeasures(mappingMeasureCount, 0);
  requireSuccess(program.evaluate({rightViolations, zeroMeasures, {}}, zero));
  require(take(program.comparePareto(zero, left, paretoDimensions)) ==
              loom::dse::ParetoRelation::Dominates,
          "componentwise lower code did not dominate");
  require(take(program.compareTotalOrdering(zero, leftKey, zero, rightKey, 0)) <
              0,
          "equal objective rank did not use the candidate semantic key");
}

void completeDeclaredLevelDomainMustFitUint128() {
  loom::ResolvedObjectiveCatalogs catalogs;
  const std::uint64_t maximum = std::numeric_limits<std::uint64_t>::max();
  catalogs.dimensions = {
      {loom::ResolvedMappingViolationObjectiveSource{
           loom::ResolvedPnrViolationKind::UnroutedObligation},
       loom::ResolvedObjectiveDirection::Minimize,
       loom::resolvedObjectiveInteger(0), loom::resolvedObjectiveInteger(1), 0,
       maximum},
      {loom::ResolvedMappingViolationObjectiveSource{
           loom::ResolvedPnrViolationKind::CapacityOveruse},
       loom::ResolvedObjectiveDirection::Minimize,
       loom::resolvedObjectiveInteger(0), loom::resolvedObjectiveInteger(1), 0,
       maximum},
  };
  catalogs.weightedLevels = {{{{0, maximum}, {1, maximum - 1}}}};
  catalogs.totalOrderings = {{{0}}};
  requireRejected(loom::dse::ObjectiveProgram::get(catalogs),
                  "weighted level domain overflows uint128");
}

void malformedOwnerReferencesFailAtPreflight() {
  loom::ResolvedObjectiveCatalogs stale;
  stale.dimensions = {
      {loom::ResolvedMappingMeasureObjectiveSource{
           std::numeric_limits<std::uint32_t>::max()},
       loom::ResolvedObjectiveDirection::Minimize,
       loom::resolvedObjectiveInteger(0), loom::resolvedObjectiveInteger(1), 0,
       1},
  };
  requireRejected(loom::dse::ObjectiveProgram::get(stale),
                  "Mapping measure source ordinal is out of range");

  stale.dimensions.front().source =
      loom::ResolvedMappingMeasureObjectiveSource{0};
  stale.dimensions.front().direction =
      static_cast<loom::ResolvedObjectiveDirection>(99);
  requireRejected(loom::dse::ObjectiveProgram::get(stale),
                  "objective direction is unknown");
}

void transientCandidateMeasuresDoNotBorrowMappingOrdinals() {
  loom::dse::CandidateMeasureObjectiveCatalogs catalogs;
  catalogs.dimensions = {
      {0, loom::ResolvedObjectiveDirection::Minimize, 0, 100},
      {1, loom::ResolvedObjectiveDirection::Maximize, 0, 100},
  };
  catalogs.weightedLevels = {{{{0, 1}}}, {{{1, 1}}}};
  catalogs.totalOrderings = {{{0, 1}}};
  const loom::dse::ObjectiveProgram program =
      take(loom::dse::ObjectiveProgram::getCandidateMeasures(catalogs));
  loom::dse::ObjectiveVector vector = program.makeVector();
  const std::uint64_t measures[] = {7, 11};
  requireSuccess(program.evaluateCandidateMeasures(measures, vector));
  require(vector.codes() == llvm::ArrayRef<std::uint64_t>({7, 89}),
          "candidate measures changed direction or ordinal ownership");
  const std::uint64_t incomplete[] = {7};
  requireRejected(program.evaluateCandidateMeasures(incomplete, vector),
                  "objective_unavailable");
}

} // namespace

int main() {
  exactAffineQuantizationIsCheckedAndDirected();
  evaluationDecimalQuantizationIsExactAndSparse();
  builtinOrderingEnergyAndParetoUseOneVector();
  completeDeclaredLevelDomainMustFitUint128();
  malformedOwnerReferencesFailAtPreflight();
  transientCandidateMeasuresDoNotBorrowMappingOrdinals();
  return 0;
}
