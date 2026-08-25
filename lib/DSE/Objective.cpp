#include "DSE/Objective.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <system_error>
#include <tuple>

using namespace loom;
using namespace loom::dse;

char ObjectiveUnavailableError::ID;

void ObjectiveUnavailableError::log(llvm::raw_ostream &stream) const {
  stream << "objective_unavailable: " << detail_;
}

std::error_code ObjectiveUnavailableError::convertToErrorCode() const {
  return std::make_error_code(std::errc::operation_not_supported);
}

namespace {

using Wide = unsigned __int128;
constexpr std::uint64_t decimalLimbBase = UINT64_C(1000000000);

struct SparseDecimalMagnitude final {
  std::map<std::int64_t, Wide> limbs;
};

llvm::Error invalid(const llvm::Twine &detail) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      "dse_objective_invalid: " + detail);
}

llvm::Error unavailable(const llvm::Twine &detail) {
  return llvm::make_error<ObjectiveUnavailableError>(detail.str());
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

Wide signedMagnitude(std::int64_t value) {
  if (value >= 0)
    return static_cast<std::uint64_t>(value);
  return static_cast<std::uint64_t>(-(value + 1)) + 1;
}

void addMonomial(SparseDecimalMagnitude &value, Wide coefficient,
                 std::int64_t exponent) {
  if (coefficient == 0)
    return;
  std::int64_t limb = exponent / 9;
  int remainder = static_cast<int>(exponent % 9);
  if (remainder < 0) {
    remainder += 9;
    --limb;
  }
  std::uint64_t factor = 1;
  for (int index = 0; index != remainder; ++index)
    factor *= 10;
  while (coefficient != 0) {
    const Wide digit = coefficient % decimalLimbBase;
    coefficient /= decimalLimbBase;
    value.limbs[limb++] += digit * factor;
  }
}

void normalize(SparseDecimalMagnitude &value) {
  for (auto current = value.limbs.begin(); current != value.limbs.end();
       ++current) {
    const Wide carry = current->second / decimalLimbBase;
    current->second %= decimalLimbBase;
    if (carry != 0)
      value.limbs[current->first + 1] += carry;
  }
  for (auto current = value.limbs.begin(); current != value.limbs.end();) {
    if (current->second == 0)
      current = value.limbs.erase(current);
    else
      ++current;
  }
}

int compareSparse(SparseDecimalMagnitude left, SparseDecimalMagnitude right) {
  normalize(left);
  normalize(right);
  auto leftIt = left.limbs.rbegin();
  auto rightIt = right.limbs.rbegin();
  while (leftIt != left.limbs.rend() || rightIt != right.limbs.rend()) {
    if (rightIt == right.limbs.rend() ||
        (leftIt != left.limbs.rend() && leftIt->first > rightIt->first))
      return 1;
    if (leftIt == left.limbs.rend() || rightIt->first > leftIt->first)
      return -1;
    if (leftIt->second != rightIt->second)
      return leftIt->second < rightIt->second ? -1 : 1;
    ++leftIt;
    ++rightIt;
  }
  return 0;
}

void addScalar(SparseDecimalMagnitude &positive,
               SparseDecimalMagnitude &negative,
               const ResolvedObjectiveScalar &scalar, bool negate,
               Wide multiplier = 1) {
  bool isNegative = false;
  Wide magnitude = 0;
  std::int64_t exponent = 0;
  if (const auto *integer = std::get_if<ResolvedObjectiveInteger>(&scalar)) {
    isNegative = integer->negative;
    magnitude = integer->magnitude;
  } else {
    const auto &decimal = std::get<ResolvedObjectiveDecimal>(scalar);
    isNegative = decimal.coefficient < 0;
    magnitude = signedMagnitude(decimal.coefficient);
    exponent = decimal.base10Exponent;
  }
  isNegative ^= negate;
  addMonomial(isNegative ? negative : positive, magnitude * multiplier,
              exponent);
}

int compareAffine(const ResolvedObjectiveScalar &source,
                  const ResolvedObjectiveScalar &origin,
                  const ResolvedObjectiveScalar &quantum, Wide multiplier) {
  SparseDecimalMagnitude positive;
  SparseDecimalMagnitude negative;
  addScalar(positive, negative, source, false);
  addScalar(positive, negative, origin, true);
  addScalar(positive, negative, quantum, true, multiplier);
  return compareSparse(std::move(positive), std::move(negative));
}

llvm::Expected<std::uint64_t>
quantize(const ResolvedObjectiveScalar &source,
         const ResolvedObjectiveScalar &originValue,
         const ResolvedObjectiveScalar &quantumValue, std::uint64_t lowerIndex,
         std::uint64_t upperIndex) {
  if (source.index() != originValue.index())
    return contractFailure("source and quantization domains differ");

  if (const auto *sourceInteger =
          std::get_if<ResolvedObjectiveInteger>(&source)) {
    const auto &origin = std::get<ResolvedObjectiveInteger>(originValue);
    const auto &quantum = std::get<ResolvedObjectiveInteger>(quantumValue);
    if (!sourceInteger->negative && !origin.negative && !quantum.negative) {
      if (sourceInteger->magnitude < origin.magnitude)
        return contractFailure("source value is below quantization origin");
      const std::uint64_t index =
          (sourceInteger->magnitude - origin.magnitude) / quantum.magnitude;
      if (index < lowerIndex || index > upperIndex)
        return contractFailure("source value is outside quantization bounds");
      return index;
    }
  }

  if (compareAffine(source, originValue, quantumValue, 0) < 0)
    return contractFailure("source value is below quantization origin");
  const Wide beyondUpper = static_cast<Wide>(upperIndex) + 1;
  if (compareAffine(source, originValue, quantumValue, beyondUpper) >= 0)
    return contractFailure("source value is outside quantization bounds");

  std::uint64_t lower = 0;
  std::uint64_t upper = upperIndex;
  while (lower < upper) {
    const std::uint64_t midpoint =
        lower + (upper - lower) / 2 + (upper - lower) % 2;
    if (compareAffine(source, originValue, quantumValue, midpoint) >= 0)
      lower = midpoint;
    else
      upper = midpoint - 1;
  }
  if (lower < lowerIndex)
    return contractFailure("source value is outside quantization bounds");
  return lower;
}

auto metricKey(const EvaluationMetricObjectiveValue &value) {
  return std::tie(value.evidenceObligationTemplate, value.metricRequestOrdinal);
}

} // namespace

llvm::Expected<ObjectiveProgram>
ObjectiveProgram::get(const ResolvedObjectiveCatalogs &catalogs) {
  if (llvm::Error error = validateResolvedObjectiveCatalogs(catalogs))
    return std::move(error);

  ObjectiveProgram result;
  result.dimensions_.reserve(catalogs.dimensions.size());
  for (const ResolvedObjectiveDimension &dimension : catalogs.dimensions) {
    result.dimensions_.push_back({dimension.source, dimension.direction,
                                  dimension.origin, dimension.quantum,
                                  dimension.lowerIndex, dimension.upperIndex});
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

llvm::Expected<ObjectiveProgram> ObjectiveProgram::getCandidateMeasures(
    const CandidateMeasureObjectiveCatalogs &catalogs) {
  ResolvedObjectiveCatalogs resolved;
  resolved.dimensions.reserve(catalogs.dimensions.size());
  for (const CandidateMeasureObjectiveDimension &dimension :
       catalogs.dimensions) {
    resolved.dimensions.push_back(
        {ResolvedEvaluationMetricObjectiveSource{0, dimension.measureOrdinal},
         dimension.direction, dimension.origin, dimension.quantum,
         dimension.lowerIndex, dimension.upperIndex});
  }
  resolved.weightedLevels = catalogs.weightedLevels;
  resolved.totalOrderings = catalogs.totalOrderings;
  auto program = get(resolved);
  if (!program)
    return program.takeError();
  program->candidateMeasureProgram_ = true;
  return program;
}

llvm::Expected<ObjectiveVector>
ObjectiveProgram::adoptVectorCodes(llvm::ArrayRef<std::uint64_t> codes) const {
  if (codes.size() != dimensions_.size())
    return invalid("recorded ObjectiveVector has the wrong dimension count");
  ObjectiveVector result = makeVector();
  for (std::size_t ordinal = 0; ordinal != codes.size(); ++ordinal) {
    const CompiledDimension &dimension = dimensions_[ordinal];
    if (codes[ordinal] > dimension.upperIndex - dimension.lowerIndex)
      return invalid("recorded ObjectiveVector code is outside its dimension");
    result.codes_[ordinal] = codes[ordinal];
  }
  return result;
}

llvm::Error ObjectiveProgram::evaluate(ObjectiveSourceValues sources,
                                       ObjectiveVector &result) const {
  if (result.codes_.size() != dimensions_.size())
    return invalid("ObjectiveVector has the wrong dimension count");
  if (!llvm::is_sorted(sources.evaluationMetrics,
                       [](const EvaluationMetricObjectiveValue &left,
                          const EvaluationMetricObjectiveValue &right) {
                         return metricKey(left) < metricKey(right);
                       }) ||
      std::adjacent_find(sources.evaluationMetrics.begin(),
                         sources.evaluationMetrics.end(),
                         [](const EvaluationMetricObjectiveValue &left,
                            const EvaluationMetricObjectiveValue &right) {
                           return metricKey(left) == metricKey(right);
                         }) != sources.evaluationMetrics.end())
    return invalid("Evaluation metric objective values are not canonical");

  for (std::size_t ordinal = 0; ordinal != dimensions_.size(); ++ordinal) {
    const CompiledDimension &dimension = dimensions_[ordinal];
    ResolvedObjectiveScalar source = resolvedObjectiveInteger(0);
    if (const auto *violation =
            std::get_if<ResolvedMappingViolationObjectiveSource>(
                &dimension.source)) {
      const std::uint32_t sourceOrdinal =
          static_cast<std::uint32_t>(violation->kind);
      if (sourceOrdinal >= sources.mappingViolations.size())
        return unavailable("required source ordinal is absent");
      source =
          resolvedObjectiveInteger(sources.mappingViolations[sourceOrdinal]);
    } else if (const auto *measure =
                   std::get_if<ResolvedMappingMeasureObjectiveSource>(
                       &dimension.source)) {
      if (measure->ordinal >= sources.mappingMeasures.size())
        return unavailable("required source ordinal is absent");
      source =
          resolvedObjectiveInteger(sources.mappingMeasures[measure->ordinal]);
    } else {
      const auto &metric =
          std::get<ResolvedEvaluationMetricObjectiveSource>(dimension.source);
      EvaluationMetricObjectiveValue key{metric.evidenceObligationTemplate,
                                         metric.metricRequestOrdinal,
                                         resolvedObjectiveInteger(0)};
      auto found =
          llvm::lower_bound(sources.evaluationMetrics, key,
                            [](const EvaluationMetricObjectiveValue &left,
                               const EvaluationMetricObjectiveValue &right) {
                              return metricKey(left) < metricKey(right);
                            });
      if (found == sources.evaluationMetrics.end() ||
          metricKey(*found) != metricKey(key))
        return unavailable("required Evaluation metric is absent");
      source = found->value;
    }
    auto index = quantize(source, dimension.origin, dimension.quantum,
                          dimension.lowerIndex, dimension.upperIndex);
    if (!index)
      return index.takeError();
    result.codes_[ordinal] =
        dimension.direction == ResolvedObjectiveDirection::Minimize
            ? *index - dimension.lowerIndex
            : dimension.upperIndex - *index;
  }
  return llvm::Error::success();
}

llvm::Error ObjectiveProgram::evaluateCandidateMeasures(
    llvm::ArrayRef<std::uint64_t> measures, ObjectiveVector &result) const {
  std::vector<ResolvedObjectiveScalar> scalars;
  scalars.reserve(measures.size());
  for (std::uint64_t measure : measures)
    scalars.push_back(resolvedObjectiveInteger(measure));
  return evaluateCandidateMeasures(scalars, result);
}

llvm::Error ObjectiveProgram::evaluateCandidateMeasures(
    llvm::ArrayRef<ResolvedObjectiveScalar> measures,
    ObjectiveVector &result) const {
  if (!candidateMeasureProgram_)
    return invalid("candidate measures require a candidate-measure program");
  std::vector<EvaluationMetricObjectiveValue> values;
  values.reserve(measures.size());
  for (auto indexed : llvm::enumerate(measures))
    values.push_back({0, indexed.index(), indexed.value()});
  return evaluate({{}, {}, values}, result);
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
