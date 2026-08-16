#include "Common/ResolvedPnrPolicy.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ErrorHandling.h"

#include <algorithm>
#include <limits>
#include <numeric>
#include <system_error>
#include <tuple>
#include <utility>

namespace loom {
namespace {

constexpr std::uint64_t builtinNoProgressIterationLimit = 8;
constexpr std::uint64_t builtinNoProgressTrendWindow = 4;

struct BuiltinLimits final {
  std::uint32_t seeds;
  std::uint64_t assignments;
  std::uint64_t endpointExpansions;
  std::uint64_t negotiations;
  std::uint64_t calibration;
  std::uint64_t levelBase;
  std::uint64_t perMovable;
  ResolvedPnrExactRepairKind repairKind;
  std::uint64_t repairDecisions;
  std::uint64_t solverCalls;
};

constexpr BuiltinLimits limitsFor(ResolvedProfilePreset preset) {
  switch (preset) {
  case ResolvedProfilePreset::ReportOnly:
    return {1, 4096, 16384, 8, 16, 16, 1, ResolvedPnrExactRepairKind::Disabled,
            0, 0};
  case ResolvedProfilePreset::QuickExplore:
    return {2,  16384, 65536, 16, 64, 64, 2, ResolvedPnrExactRepairKind::CpSat,
            64, 128};
  case ResolvedProfilePreset::BalancedExplore:
    return {4,   65536, 262144, 64,
            256, 128,   8,      ResolvedPnrExactRepairKind::CpSat,
            256, 1024};
  case ResolvedProfilePreset::PerformanceExplore:
    return {8,   262144, 1048576, 128,
            512, 256,    16,      ResolvedPnrExactRepairKind::CpSat,
            512, 4096};
  case ResolvedProfilePreset::Implementation:
    return {16,   524288, 2097152, 256,
            1024, 512,    24,      ResolvedPnrExactRepairKind::CpSat,
            1024, 8192};
  case ResolvedProfilePreset::StrictImplementation:
    return {32,   1048576, 4194304, 512,
            2048, 1024,    32,      ResolvedPnrExactRepairKind::CpSat,
            2048, 16384};
  }
  llvm_unreachable("all resolved profile presets are handled");
}

ResolvedPnrTemporaryViolationPolicy allTemporaryViolations() {
  return {{
#define LOOM_MAPPING_VIOLATION(Name, Ordinal, DisplayName, ConfigSpelling)     \
  ResolvedPnrViolationKind::Name,
#include "Common/MappingObjectiveKinds.def"
  }};
}

llvm::Error invalid(const char *detail) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      "resolved_pnr_policy_invalid: %s", detail);
}

bool isReduced(const ResolvedExactRatio &ratio) {
  return ratio.denominator != 0 &&
         std::gcd(ratio.numerator, ratio.denominator) == 1;
}

bool ratioAtMostOne(const ResolvedExactRatio &ratio) {
  return ratio.numerator <= ratio.denominator;
}

bool ratioStrictlyBetweenZeroAndOne(const ResolvedExactRatio &ratio) {
  return ratio.numerator != 0 && ratio.numerator < ratio.denominator;
}

constexpr std::uint32_t mappingMeasureKindCount = 0
#define LOOM_MAPPING_MEASURE(Name, Ordinal, DisplayName) +1
#include "Common/MappingObjectiveKinds.def"
    ;

enum class BuiltinMappingMeasureKind : std::uint32_t {
#define LOOM_MAPPING_MEASURE(Name, Ordinal, DisplayName) Name = Ordinal,
#include "Common/MappingObjectiveKinds.def"
};

auto sourceKey(const ResolvedObjectiveScalarSource &source) {
  if (const auto *violation =
          std::get_if<ResolvedMappingViolationObjectiveSource>(&source))
    return std::make_tuple(std::uint32_t{0},
                           static_cast<std::uint32_t>(violation->kind),
                           std::uint64_t{0});
  if (const auto *measure =
          std::get_if<ResolvedMappingMeasureObjectiveSource>(&source))
    return std::make_tuple(std::uint32_t{1}, measure->ordinal,
                           std::uint64_t{0});
  const auto &metric =
      std::get<ResolvedEvaluationMetricObjectiveSource>(source);
  return std::make_tuple(std::uint32_t{2}, metric.evidenceObligationTemplate,
                         metric.metricRequestOrdinal);
}

auto scalarKey(const ResolvedObjectiveScalar &value) {
  if (const auto *integer = std::get_if<ResolvedObjectiveInteger>(&value))
    return std::make_tuple(std::uint32_t{0}, integer->negative,
                           integer->magnitude, std::int64_t{0},
                           std::int64_t{0});
  const auto &decimal = std::get<ResolvedObjectiveDecimal>(value);
  return std::make_tuple(std::uint32_t{1}, false, std::uint64_t{0},
                         decimal.coefficient, decimal.base10Exponent);
}

bool isCanonicalScalar(const ResolvedObjectiveScalar &value) {
  if (const auto *integer = std::get_if<ResolvedObjectiveInteger>(&value))
    return !integer->negative || (integer->magnitude != 0 &&
                                  integer->magnitude <= (UINT64_C(1) << 63));
  const auto &decimal = std::get<ResolvedObjectiveDecimal>(value);
  if (decimal.coefficient == 0)
    return decimal.base10Exponent == 0;
  return decimal.coefficient % 10 != 0;
}

bool isPositiveScalar(const ResolvedObjectiveScalar &value) {
  if (const auto *integer = std::get_if<ResolvedObjectiveInteger>(&value))
    return !integer->negative && integer->magnitude != 0;
  return std::get<ResolvedObjectiveDecimal>(value).coefficient > 0;
}

auto dimensionKey(const ResolvedObjectiveDimension &dimension) {
  return std::make_tuple(sourceKey(dimension.source), dimension.direction,
                         scalarKey(dimension.origin),
                         scalarKey(dimension.quantum), dimension.lowerIndex,
                         dimension.upperIndex);
}

bool weightedLevelLess(const ResolvedWeightedObjectiveLevel &left,
                       const ResolvedWeightedObjectiveLevel &right) {
  if (left.terms.size() != right.terms.size())
    return left.terms.size() < right.terms.size();
  return std::lexicographical_compare(
      left.terms.begin(), left.terms.end(), right.terms.begin(),
      right.terms.end(),
      [](const ResolvedWeightedObjectiveTerm &a,
         const ResolvedWeightedObjectiveTerm &b) {
        return std::tie(a.dimension, a.weight) <
               std::tie(b.dimension, b.weight);
      });
}

bool totalOrderingLess(const ResolvedTotalOrdering &left,
                       const ResolvedTotalOrdering &right) {
  if (left.weightedLevels.size() != right.weightedLevels.size())
    return left.weightedLevels.size() < right.weightedLevels.size();
  return std::lexicographical_compare(
      left.weightedLevels.begin(), left.weightedLevels.end(),
      right.weightedLevels.begin(), right.weightedLevels.end());
}

} // namespace

llvm::StringRef
resolvedPnrCompletionGoalSpelling(ResolvedPnrCompletionGoal goal) {
  switch (goal) {
  case ResolvedPnrCompletionGoal::ExhaustConfiguredWork:
    return "exhaust_configured_work";
  case ResolvedPnrCompletionGoal::FirstVerifiedCandidate:
    return "first_verified_candidate";
  }
  llvm_unreachable("unknown PnR completion goal");
}

std::optional<ResolvedPnrCompletionGoal>
parseResolvedPnrCompletionGoal(llvm::StringRef spelling) {
  if (spelling == "exhaust_configured_work")
    return ResolvedPnrCompletionGoal::ExhaustConfiguredWork;
  if (spelling == "first_verified_candidate")
    return ResolvedPnrCompletionGoal::FirstVerifiedCandidate;
  return std::nullopt;
}

ResolvedPnrPolicyConfig
resolvedBuiltinSpatialPnrPolicy(ResolvedProfilePreset preset) {
  const BuiltinLimits limits = limitsFor(preset);
  return {{ResolvedPnrInitializerPolicy{limits.seeds, limits.assignments},
           ResolvedPnrActionProposalPolicy{1, 3, 2},
           ResolvedPnrRoutingPolicy{
               limits.endpointExpansions, limits.negotiations,
               builtinNoProgressIterationLimit, builtinNoProgressTrendWindow,
               ResolvedPathFinderPolicy{
                   ResolvedPathFinderPriceKernel::Multiplicative, 1,
                   ResolvedExactRatio{3, 2}, 1}},
           ResolvedPnrAnnealingPolicy{
               limits.calibration, ResolvedExactRatio{3, 4},
               ResolvedExactRatio{4, 5}, 1024, 1, ResolvedExactRatio{19, 20},
               limits.levelBase, limits.perMovable},
           ResolvedPnrExactRepairPolicy{
               limits.repairKind, limits.repairDecisions, limits.solverCalls},
           ResolvedPnrCompletionGoal::ExhaustConfiguredWork},
          ResolvedPnrDeterminismPolicy{
              0, ResolvedPnrPrngProtocol::Sha256SeededXoshiro256StarStar_1_0,
              ResolvedPnrAcceptanceProtocol::ExpNegativeQ64Table_1_0},
          allTemporaryViolations(),
          ResolvedPnrObjectiveSelection{0, 4}};
}

ResolvedPnrPolicyConfig
resolvedBuiltinSystemPnrPolicy(ResolvedProfilePreset preset) {
  ResolvedPnrPolicyConfig policy = resolvedBuiltinSpatialPnrPolicy(preset);
  policy.search.exactRepair =
      ResolvedPnrExactRepairPolicy{ResolvedPnrExactRepairKind::Disabled, 0, 0};
  return policy;
}

ResolvedObjectiveCatalogs resolvedBuiltinObjectiveCatalogs() {
  ResolvedObjectiveCatalogs catalogs;
  catalogs.dimensions.reserve(resolvedPnrViolationKindCount +
                              mappingMeasureKindCount);
  for (std::uint32_t ordinal = 0; ordinal != resolvedPnrViolationKindCount;
       ++ordinal)
    catalogs.dimensions.push_back(
        {ResolvedMappingViolationObjectiveSource{
             static_cast<ResolvedPnrViolationKind>(ordinal)},
         ResolvedObjectiveDirection::Minimize, resolvedObjectiveInteger(0),
         resolvedObjectiveInteger(1), 0,
         std::numeric_limits<std::uint64_t>::max()});
  for (std::uint32_t ordinal = 0; ordinal != mappingMeasureKindCount; ++ordinal)
    catalogs.dimensions.push_back(
        {ResolvedMappingMeasureObjectiveSource{ordinal},
         ResolvedObjectiveDirection::Minimize, resolvedObjectiveInteger(0),
         resolvedObjectiveInteger(1), 0,
         std::numeric_limits<std::uint64_t>::max()});

  ResolvedWeightedObjectiveLevel closure;
  ResolvedWeightedObjectiveLevel traversal;
  ResolvedWeightedObjectiveLevel schedule;
  ResolvedWeightedObjectiveLevel timing;
  ResolvedWeightedObjectiveLevel energy;
  for (std::uint32_t dimension = 0; dimension != resolvedPnrViolationKindCount;
       ++dimension) {
    closure.terms.push_back({dimension, 1});
    energy.terms.push_back({dimension, UINT64_C(281474976710656)});
  }
  traversal.terms.push_back({resolvedPnrViolationKindCount, 1});
  energy.terms.push_back({resolvedPnrViolationKindCount, 1});
  const std::uint32_t scheduleDimension = resolvedPnrViolationKindCount + 1;
  schedule.terms.push_back({scheduleDimension, 1});
  energy.terms.push_back({scheduleDimension, UINT64_C(4294967296)});
  const std::uint32_t timingBegin = scheduleDimension + 1;
  for (std::uint32_t dimension = timingBegin;
       dimension != resolvedPnrViolationKindCount + mappingMeasureKindCount;
       ++dimension) {
    timing.terms.push_back({dimension, 1});
    const bool transport =
        dimension ==
        resolvedPnrViolationKindCount +
            static_cast<std::uint32_t>(
                BuiltinMappingMeasureKind::TransportBitCycleDemand);
    energy.terms.push_back(
        {dimension, transport ? UINT64_C(1) : UINT64_C(4294967296)});
  }
  catalogs.weightedLevels = {std::move(traversal), std::move(schedule),
                             std::move(closure), std::move(timing),
                             std::move(energy)};
  catalogs.totalOrderings.push_back({{2, 3, 1, 0}});
  return catalogs;
}

llvm::Error
validateResolvedObjectiveCatalogs(const ResolvedObjectiveCatalogs &catalogs) {
  for (const ResolvedObjectiveDimension &dimension : catalogs.dimensions) {
    if (const auto *violation =
            std::get_if<ResolvedMappingViolationObjectiveSource>(
                &dimension.source)) {
      if (static_cast<std::uint32_t>(violation->kind) >=
          resolvedPnrViolationKindCount)
        return invalid("Mapping violation source ordinal is out of range");
    } else if (const auto *measure =
                   std::get_if<ResolvedMappingMeasureObjectiveSource>(
                       &dimension.source)) {
      if (measure->ordinal >= mappingMeasureKindCount)
        return invalid("Mapping measure source ordinal is out of range");
    }
    const bool mappingSource =
        !std::holds_alternative<ResolvedEvaluationMetricObjectiveSource>(
            dimension.source);
    if (mappingSource &&
        (!std::holds_alternative<ResolvedObjectiveInteger>(dimension.origin) ||
         !std::holds_alternative<ResolvedObjectiveInteger>(dimension.quantum)))
      return invalid("Mapping objective quantization must use integers");
    if (dimension.origin.index() != dimension.quantum.index())
      return invalid("objective origin and quantum have different domains");
    if (!isCanonicalScalar(dimension.origin) ||
        !isCanonicalScalar(dimension.quantum))
      return invalid("objective quantization scalar is not canonical");
    if (mappingSource &&
        std::get<ResolvedObjectiveInteger>(dimension.origin).negative)
      return invalid("Mapping objective origin must be nonnegative");
    if (!isPositiveScalar(dimension.quantum))
      return invalid("objective quantum must be positive");
    if (static_cast<std::uint32_t>(dimension.direction) >
        static_cast<std::uint32_t>(ResolvedObjectiveDirection::Maximize))
      return invalid("objective direction is unknown");
    if (dimension.lowerIndex > dimension.upperIndex)
      return invalid("objective bounds are reversed");
  }
  if (!std::is_sorted(catalogs.dimensions.begin(), catalogs.dimensions.end(),
                      [](const ResolvedObjectiveDimension &a,
                         const ResolvedObjectiveDimension &b) {
                        return dimensionKey(a) < dimensionKey(b);
                      }) ||
      std::adjacent_find(catalogs.dimensions.begin(), catalogs.dimensions.end(),
                         [](const ResolvedObjectiveDimension &a,
                            const ResolvedObjectiveDimension &b) {
                           return dimensionKey(a) == dimensionKey(b);
                         }) != catalogs.dimensions.end())
    return invalid("objective dimensions are not a canonical unique sequence");

  for (const ResolvedWeightedObjectiveLevel &level : catalogs.weightedLevels) {
    if (level.terms.empty())
      return invalid("weighted objective level is empty");
    std::uint64_t commonDivisor = 0;
    std::uint32_t previous = 0;
    bool first = true;
    for (const ResolvedWeightedObjectiveTerm &term : level.terms) {
      if (term.dimension >= catalogs.dimensions.size())
        return invalid("weighted objective term has an out-of-range dimension");
      if (term.weight == 0)
        return invalid("weighted objective term has zero weight");
      if (!first && term.dimension <= previous)
        return invalid("weighted objective terms are not canonical");
      first = false;
      previous = term.dimension;
      commonDivisor = std::gcd(commonDivisor, term.weight);
    }
    if (commonDivisor != 1)
      return invalid("weighted objective level weights are not reduced");
  }
  if (!std::is_sorted(catalogs.weightedLevels.begin(),
                      catalogs.weightedLevels.end(), weightedLevelLess) ||
      std::adjacent_find(
          catalogs.weightedLevels.begin(), catalogs.weightedLevels.end(),
          [](const ResolvedWeightedObjectiveLevel &a,
             const ResolvedWeightedObjectiveLevel &b) {
            return !weightedLevelLess(a, b) && !weightedLevelLess(b, a);
          }) != catalogs.weightedLevels.end())
    return invalid("weighted objective levels are not canonical and unique");

  for (const ResolvedTotalOrdering &ordering : catalogs.totalOrderings) {
    if (ordering.weightedLevels.empty())
      return invalid("total ordering is empty");
    std::vector<std::uint32_t> seen;
    for (std::uint32_t level : ordering.weightedLevels) {
      if (level >= catalogs.weightedLevels.size())
        return invalid("total ordering has an out-of-range level");
      if (llvm::is_contained(seen, level))
        return invalid("total ordering repeats a level");
      seen.push_back(level);
    }
  }
  if (!std::is_sorted(catalogs.totalOrderings.begin(),
                      catalogs.totalOrderings.end(), totalOrderingLess) ||
      std::adjacent_find(
          catalogs.totalOrderings.begin(), catalogs.totalOrderings.end(),
          [](const ResolvedTotalOrdering &a, const ResolvedTotalOrdering &b) {
            return !totalOrderingLess(a, b) && !totalOrderingLess(b, a);
          }) != catalogs.totalOrderings.end())
    return invalid("total orderings are not canonical and unique");
  return llvm::Error::success();
}

llvm::Error
validateResolvedPathFinderPolicy(const ResolvedPathFinderPolicy &policy) {
  if (policy.presentPressureInitial == 0 ||
      policy.historyPressureIncrement == 0 ||
      !isReduced(policy.presentPressureGrowth) ||
      policy.presentPressureGrowth.numerator <
          policy.presentPressureGrowth.denominator)
    return invalid("PathFinder pressure policy is not canonical");
  return llvm::Error::success();
}

llvm::Error validateResolvedPnrActionProposalPolicy(
    const ResolvedPnrActionProposalPolicy &policy) {
  const std::uint64_t realization = policy.realizationBindingWeight;
  const std::uint64_t transport = policy.transportRoutingWeight;
  const std::uint64_t resource = policy.resourceAllocationWeight;
  if ((realization | transport | resource) == 0)
    return invalid("action proposal weights are all zero");
  if (std::gcd(std::gcd(realization, transport), resource) != 1)
    return invalid("action proposal weights are not reduced");
  if (realization > std::numeric_limits<std::uint64_t>::max() - transport ||
      realization + transport >
          std::numeric_limits<std::uint64_t>::max() - resource)
    return invalid("action proposal weight sum is not representable");
  return llvm::Error::success();
}

llvm::Error
validateResolvedDualStepSchedule(const ResolvedDualStepSchedule &schedule) {
  switch (schedule.kind) {
  case ResolvedDualStepScheduleKind::Constant:
    if (schedule.first == 0 || schedule.second != 0 || schedule.third != 0 ||
        schedule.fourth != 0)
      return invalid("constant dual schedule is not canonical");
    return llvm::Error::success();
  case ResolvedDualStepScheduleKind::GeometricDecay: {
    const ResolvedExactRatio decay{schedule.third, schedule.fourth};
    if (schedule.second == 0 || schedule.first <= schedule.second ||
        !isReduced(decay) || !ratioStrictlyBetweenZeroAndOne(decay))
      return invalid("geometric dual schedule is not canonical");
    return llvm::Error::success();
  }
  case ResolvedDualStepScheduleKind::HarmonicDecay:
    if (schedule.first == 0 || schedule.second == 0 || schedule.third == 0 ||
        schedule.fourth != 0 ||
        schedule.first / schedule.second <= schedule.third)
      return invalid("harmonic dual schedule is not canonical");
    return llvm::Error::success();
  }
  llvm_unreachable("all resolved dual step schedules are handled");
}

llvm::Error validateResolvedDualSubgradientPolicy(
    const ResolvedDualSubgradientPolicy &policy) {
  if (policy.directionKernel ==
      ResolvedDualDirectionKernel::MomentumDeflected) {
    if (!policy.momentum || !isReduced(*policy.momentum) ||
        policy.momentum->numerator >= policy.momentum->denominator)
      return invalid("momentum direction requires canonical beta in [0,1)");
  } else if (policy.momentum) {
    return invalid("inactive momentum is present");
  }
  return validateResolvedDualStepSchedule(policy.stepSchedule);
}

llvm::Error
validateResolvedPnrAnnealingPolicy(const ResolvedPnrAnnealingPolicy &policy) {
  if (policy.calibrationProposalCount == 0 || policy.fallbackTemperature == 0 ||
      policy.minimumTemperature == 0 ||
      !isReduced(policy.positiveDeltaQuantile) ||
      !ratioAtMostOne(policy.positiveDeltaQuantile) ||
      !isReduced(policy.targetInitialAcceptance) ||
      !ratioStrictlyBetweenZeroAndOne(policy.targetInitialAcceptance) ||
      !isReduced(policy.coolingRatio) ||
      !ratioStrictlyBetweenZeroAndOne(policy.coolingRatio) ||
      (policy.proposalsPerLevelBase == 0 &&
       policy.proposalsPerMovableDecision == 0))
    return invalid("annealing policy is not canonical");
  return llvm::Error::success();
}

llvm::Error
validateResolvedPnrPolicyConfig(const ResolvedPnrPolicyConfig &policy,
                                const ResolvedObjectiveCatalogs &catalogs) {
  if (llvm::Error error = validateResolvedObjectiveCatalogs(catalogs))
    return error;
  if (policy.search.initializer.seedAttemptCount == 0 ||
      policy.search.initializer.assignmentAttemptLimitPerSeed == 0)
    return invalid("initializer work limits must be positive");

  if (llvm::Error error =
          validateResolvedPnrActionProposalPolicy(policy.search.actionProposal))
    return error;

  const ResolvedPnrRoutingPolicy &routing = policy.search.routing;
  if (routing.endpointExpansionLimit == 0 ||
      routing.negotiationIterationLimit == 0 ||
      routing.noProgressIterationLimit == 0 ||
      routing.noProgressTrendWindow == 0)
    return invalid("routing work limits must be positive");
  if (routing.noProgressTrendWindow > routing.noProgressIterationLimit ||
      routing.noProgressIterationLimit > routing.negotiationIterationLimit)
    return invalid("routing no-progress limits are not canonical");
  if (const auto *pathFinder =
          std::get_if<ResolvedPathFinderPolicy>(&routing.negotiation)) {
    if (llvm::Error error = validateResolvedPathFinderPolicy(*pathFinder))
      return error;
  } else {
    const auto &dual =
        std::get<ResolvedDualSubgradientPolicy>(routing.negotiation);
    if (llvm::Error error = validateResolvedDualSubgradientPolicy(dual))
      return error;
  }

  if (llvm::Error error =
          validateResolvedPnrAnnealingPolicy(policy.search.annealing))
    return error;
  const ResolvedPnrExactRepairPolicy &repair = policy.search.exactRepair;
  if (repair.kind == ResolvedPnrExactRepairKind::Disabled) {
    if (repair.maxRegionDecisions != 0 || repair.maxSolverCalls != 0)
      return invalid("disabled exact repair has active fields");
  } else if (repair.maxRegionDecisions == 0 || repair.maxSolverCalls == 0) {
    return invalid("CP-SAT exact repair limits must be positive");
  }
  if (static_cast<std::uint32_t>(policy.search.completionGoal) >
      static_cast<std::uint32_t>(
          ResolvedPnrCompletionGoal::FirstVerifiedCandidate))
    return invalid("search completion goal is unknown");

  std::uint32_t previousViolation = 0;
  bool firstViolation = true;
  for (ResolvedPnrViolationKind kind : policy.temporaryViolations.admitted) {
    const std::uint32_t ordinal = static_cast<std::uint32_t>(kind);
    if (!firstViolation && ordinal <= previousViolation)
      return invalid("temporary violations are not canonical");
    firstViolation = false;
    previousViolation = ordinal;
  }

  const ResolvedPnrObjectiveSelection &selection = policy.objectiveSelection;
  if (selection.selectedTotalOrdering >= catalogs.totalOrderings.size() ||
      selection.selectedSearchEnergy >= catalogs.weightedLevels.size())
    return invalid("objective selection is out of range");
  for (ResolvedPnrViolationKind kind : policy.temporaryViolations.admitted) {
    const std::uint32_t sourceOrdinal = static_cast<std::uint32_t>(kind);
    const bool visible = llvm::any_of(
        catalogs.weightedLevels[selection.selectedSearchEnergy].terms,
        [&](const ResolvedWeightedObjectiveTerm &term) {
          const ResolvedObjectiveDimension &dimension =
              catalogs.dimensions[term.dimension];
          const auto *source =
              std::get_if<ResolvedMappingViolationObjectiveSource>(
                  &dimension.source);
          return source &&
                 static_cast<std::uint32_t>(source->kind) == sourceOrdinal &&
                 dimension.direction == ResolvedObjectiveDirection::Minimize &&
                 dimension.origin == resolvedObjectiveInteger(0) &&
                 dimension.quantum == resolvedObjectiveInteger(1);
        });
    if (!visible)
      return invalid("temporary violation is absent from search energy");
  }
  return llvm::Error::success();
}

} // namespace loom
