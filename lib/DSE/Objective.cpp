#include "DSE/Objective.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <system_error>

using namespace loom;
using namespace loom::dse;

namespace {

using Wide = unsigned __int128;

llvm::Error invalid(const llvm::Twine &detail) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      "dse_objective_invalid: " + detail);
}

llvm::Error unavailable(const llvm::Twine &detail) {
  return llvm::createStringError(
      std::make_error_code(std::errc::operation_not_supported),
      "objective_unavailable: " + detail);
}

llvm::Error contractFailure(const llvm::Twine &detail) {
  return llvm::createStringError(
      std::make_error_code(std::errc::result_out_of_range),
      "objective_contract_failure: " + detail);
}

ObjectiveWideValue split(Wide value) {
  return {static_cast<std::uint64_t>(value >> 64),
          static_cast<std::uint64_t>(value)};
}

Wide join(ObjectiveWideValue value) {
  return (static_cast<Wide>(value.high) << 64) | value.low;
}

int compareWide(ObjectiveWideValue lhs, ObjectiveWideValue rhs) {
  if (lhs == rhs)
    return 0;
  return lhs < rhs ? -1 : 1;
}

llvm::Expected<std::uint32_t> checkedU32(std::size_t value,
                                         llvm::StringRef subject) {
  if (value > std::numeric_limits<std::uint32_t>::max())
    return invalid(subject + " exceeds uint32");
  return static_cast<std::uint32_t>(value);
}

} // namespace

llvm::Expected<ObjectiveProgram>
ObjectiveProgram::get(const ResolvedObjectiveCatalogs &catalogs) {
  if (llvm::Error error = validateResolvedObjectiveCatalogs(catalogs))
    return std::move(error);

  ObjectiveProgram result;
  result.dimensions_.reserve(catalogs.dimensions.size());
  for (const ResolvedObjectiveDimension &dimension : catalogs.dimensions) {
    result.dimensions_.push_back({dimension.sourceKind, dimension.sourceOrdinal,
                                  dimension.direction, dimension.origin,
                                  dimension.quantum, dimension.lowerIndex,
                                  dimension.upperIndex});
  }

  const Wide wideMaximum = ~static_cast<Wide>(0);
  for (const ResolvedWeightedObjectiveLevel &level : catalogs.weightedLevels) {
    auto termOffset = checkedU32(result.terms_.size(), "objective term offset");
    if (!termOffset)
      return termOffset.takeError();
    auto termCount = checkedU32(level.terms.size(), "objective term count");
    if (!termCount)
      return termCount.takeError();

    Wide declaredMaximum = 0;
    for (const ResolvedWeightedObjectiveTerm &term : level.terms) {
      const CompiledDimension &dimension = result.dimensions_[term.dimension];
      const std::uint64_t maximumCode =
          dimension.upperIndex - dimension.lowerIndex;
      const Wide product = static_cast<Wide>(maximumCode) * term.weight;
      if (declaredMaximum > wideMaximum - product)
        return invalid("weighted level domain overflows uint128");
      declaredMaximum += product;
      result.terms_.push_back({term.dimension, term.weight});
    }
    result.levels_.push_back({*termOffset, *termCount});
  }

  for (const ResolvedTotalOrdering &ordering : catalogs.totalOrderings) {
    auto levelOffset =
        checkedU32(result.orderingLevels_.size(), "ordering level offset");
    if (!levelOffset)
      return levelOffset.takeError();
    auto levelCount =
        checkedU32(ordering.weightedLevels.size(), "ordering level count");
    if (!levelCount)
      return levelCount.takeError();
    result.orderingLevels_.insert(result.orderingLevels_.end(),
                                  ordering.weightedLevels.begin(),
                                  ordering.weightedLevels.end());
    result.orderings_.push_back({*levelOffset, *levelCount});
  }
  return result;
}

llvm::Error ObjectiveProgram::evaluate(ObjectiveSourceValues sources,
                                       ObjectiveVector &result) const {
  if (result.codes_.size() != dimensions_.size())
    return invalid("ObjectiveVector has the wrong dimension count");

  for (std::size_t ordinal = 0; ordinal != dimensions_.size(); ++ordinal) {
    const CompiledDimension &dimension = dimensions_[ordinal];
    llvm::ArrayRef<std::uint64_t> ownerValues;
    switch (dimension.sourceKind) {
    case ResolvedObjectiveSourceKind::MappingViolation:
      ownerValues = sources.mappingViolations;
      break;
    case ResolvedObjectiveSourceKind::MappingMeasure:
      ownerValues = sources.mappingMeasures;
      break;
    }
    if (dimension.sourceOrdinal >= ownerValues.size())
      return unavailable("required source ordinal is absent");

    const std::uint64_t source = ownerValues[dimension.sourceOrdinal];
    if (source < dimension.origin)
      return contractFailure("source value is below quantization origin");
    const std::uint64_t index = (source - dimension.origin) / dimension.quantum;
    if (index < dimension.lowerIndex || index > dimension.upperIndex)
      return contractFailure("source value is outside quantization bounds");
    result.codes_[ordinal] =
        dimension.direction == ResolvedObjectiveDirection::Minimize
            ? index - dimension.lowerIndex
            : dimension.upperIndex - index;
  }
  return llvm::Error::success();
}

llvm::Expected<ObjectiveWideValue>
ObjectiveProgram::weightedLevelValue(const ObjectiveVector &vector,
                                     std::uint32_t weightedLevel) const {
  if (vector.codes_.size() != dimensions_.size())
    return invalid("ObjectiveVector has the wrong dimension count");
  if (weightedLevel >= levels_.size())
    return invalid("weighted level reference is out of range");

  const Wide wideMaximum = ~static_cast<Wide>(0);
  Wide total = 0;
  const CompiledLevel &level = levels_[weightedLevel];
  for (const CompiledTerm &term :
       llvm::ArrayRef(terms_).slice(level.termOffset, level.termCount)) {
    const CompiledDimension &dimension = dimensions_[term.dimension];
    const std::uint64_t code = vector.codes_[term.dimension];
    if (code > dimension.upperIndex - dimension.lowerIndex)
      return contractFailure("ObjectiveVector code is outside its dimension");
    const Wide product = static_cast<Wide>(code) * term.weight;
    if (total > wideMaximum - product)
      return contractFailure("weighted level accumulation overflows uint128");
    total += product;
  }
  return split(total);
}

llvm::Expected<ObjectiveSignedDifference>
ObjectiveProgram::signedWeightedLevelDifference(
    const ObjectiveVector &left, const ObjectiveVector &right,
    std::uint32_t weightedLevel) const {
  auto leftValue = weightedLevelValue(left, weightedLevel);
  if (!leftValue)
    return leftValue.takeError();
  auto rightValue = weightedLevelValue(right, weightedLevel);
  if (!rightValue)
    return rightValue.takeError();
  const int comparison = compareWide(*leftValue, *rightValue);
  if (comparison == 0)
    return ObjectiveSignedDifference{};
  const Wide leftWide = join(*leftValue);
  const Wide rightWide = join(*rightValue);
  return ObjectiveSignedDifference{
      comparison < 0 ? ObjectiveDifferenceSign::Negative
                     : ObjectiveDifferenceSign::Positive,
      split(comparison < 0 ? rightWide - leftWide : leftWide - rightWide)};
}

llvm::Expected<int> ObjectiveProgram::compareTotalOrdering(
    const ObjectiveVector &left, llvm::ArrayRef<std::uint8_t> leftCandidateKey,
    const ObjectiveVector &right,
    llvm::ArrayRef<std::uint8_t> rightCandidateKey,
    std::uint32_t totalOrdering) const {
  if (totalOrdering >= orderings_.size())
    return invalid("total ordering reference is out of range");
  const CompiledOrdering &ordering = orderings_[totalOrdering];
  for (std::uint32_t level :
       llvm::ArrayRef(orderingLevels_)
           .slice(ordering.levelOffset, ordering.levelCount)) {
    auto leftValue = weightedLevelValue(left, level);
    if (!leftValue)
      return leftValue.takeError();
    auto rightValue = weightedLevelValue(right, level);
    if (!rightValue)
      return rightValue.takeError();
    const int comparison = compareWide(*leftValue, *rightValue);
    if (comparison != 0)
      return comparison;
  }
  if (leftCandidateKey == rightCandidateKey)
    return 0;
  return std::lexicographical_compare(
             leftCandidateKey.begin(), leftCandidateKey.end(),
             rightCandidateKey.begin(), rightCandidateKey.end())
             ? -1
             : 1;
}

llvm::Expected<ParetoRelation> ObjectiveProgram::comparePareto(
    const ObjectiveVector &left, const ObjectiveVector &right,
    llvm::ArrayRef<std::uint32_t> dimensions) const {
  if (left.codes_.size() != dimensions_.size() ||
      right.codes_.size() != dimensions_.size())
    return invalid("ObjectiveVector has the wrong dimension count");
  if (dimensions.empty())
    return invalid("Pareto dimension set is empty");

  bool leftBetter = false;
  bool rightBetter = false;
  std::uint32_t previous = 0;
  bool first = true;
  for (std::uint32_t dimension : dimensions) {
    if (dimension >= dimensions_.size())
      return invalid("Pareto dimension reference is out of range");
    if (!first && dimension <= previous)
      return invalid("Pareto dimensions are not canonical");
    first = false;
    previous = dimension;
    leftBetter |= left.codes_[dimension] < right.codes_[dimension];
    rightBetter |= right.codes_[dimension] < left.codes_[dimension];
  }
  if (leftBetter && rightBetter)
    return ParetoRelation::Incomparable;
  if (leftBetter)
    return ParetoRelation::Dominates;
  if (rightBetter)
    return ParetoRelation::Dominated;
  return ParetoRelation::Equivalent;
}
