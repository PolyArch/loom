#ifndef LOOM_COMMON_RESOLVEDPNRPOLICY_H
#define LOOM_COMMON_RESOLVEDPNRPOLICY_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>
#include <variant>
#include <vector>

namespace loom {

struct ResolvedExactRatio final {
  std::uint64_t numerator;
  std::uint64_t denominator;
};

struct ResolvedPnrInitializerPolicy final {
  std::uint32_t seedAttemptCount;
  std::uint64_t assignmentAttemptLimitPerSeed;
};

struct ResolvedPnrActionProposalPolicy final {
  std::uint64_t realizationBindingWeight;
  std::uint64_t transportRoutingWeight;
  std::uint64_t resourceAllocationWeight;
};

enum class ResolvedPathFinderPriceKernel : std::uint32_t {
  Multiplicative,
  Additive,
};

struct ResolvedPathFinderPolicy final {
  ResolvedPathFinderPriceKernel priceKernel;
  std::uint64_t presentPressureInitial;
  ResolvedExactRatio presentPressureGrowth;
  std::uint64_t historyPressureIncrement;
};

enum class ResolvedDualDirectionKernel : std::uint32_t {
  ProjectedSigned,
  PositiveViolationOnly,
  MomentumDeflected,
};

enum class ResolvedDualStepScheduleKind : std::uint32_t {
  Constant,
  GeometricDecay,
  HarmonicDecay,
};

struct ResolvedDualStepSchedule final {
  ResolvedDualStepScheduleKind kind;
  std::uint64_t first;
  std::uint64_t second;
  std::uint64_t third;
  std::uint64_t fourth;
};

struct ResolvedDualSubgradientPolicy final {
  ResolvedDualDirectionKernel directionKernel;
  std::optional<ResolvedExactRatio> momentum;
  ResolvedDualStepSchedule stepSchedule;
};

using ResolvedRoutingNegotiationPolicy =
    std::variant<ResolvedPathFinderPolicy, ResolvedDualSubgradientPolicy>;

struct ResolvedPnrRoutingPolicy final {
  std::uint64_t endpointExpansionLimit;
  std::uint64_t negotiationIterationLimit;
  std::uint64_t noProgressIterationLimit;
  std::uint64_t noProgressTrendWindow;
  ResolvedRoutingNegotiationPolicy negotiation;
};

struct ResolvedPnrAnnealingPolicy final {
  std::uint64_t calibrationProposalCount;
  ResolvedExactRatio positiveDeltaQuantile;
  ResolvedExactRatio targetInitialAcceptance;
  std::uint64_t fallbackTemperature;
  std::uint64_t minimumTemperature;
  ResolvedExactRatio coolingRatio;
  std::uint64_t temperatureLevelLimit;
  std::uint64_t proposalsPerLevelBase;
  std::uint64_t proposalsPerMovableDecision;
};

enum class ResolvedPnrExactRepairKind : std::uint32_t {
  Disabled,
  CpSat,
};

struct ResolvedPnrExactRepairPolicy final {
  ResolvedPnrExactRepairKind kind;
  std::uint64_t maxRegionDecisions;
  std::uint64_t maxSolverCalls;
};

enum class ResolvedPnrCompletionGoal : std::uint32_t {
  ExhaustConfiguredWork,
  FirstVerifiedCandidate,
};

llvm::StringRef
resolvedPnrCompletionGoalSpelling(ResolvedPnrCompletionGoal goal);
std::optional<ResolvedPnrCompletionGoal>
parseResolvedPnrCompletionGoal(llvm::StringRef spelling);

struct ResolvedPnrSearchPolicy final {
  ResolvedPnrInitializerPolicy initializer;
  ResolvedPnrActionProposalPolicy actionProposal;
  ResolvedPnrRoutingPolicy routing;
  ResolvedPnrAnnealingPolicy annealing;
  ResolvedPnrExactRepairPolicy exactRepair;
  ResolvedPnrCompletionGoal completionGoal =
      ResolvedPnrCompletionGoal::ExhaustConfiguredWork;
};

enum class ResolvedPnrPrngProtocol : std::uint32_t {
  Sha256SeededXoshiro256StarStar_1_0,
};

enum class ResolvedPnrAcceptanceProtocol : std::uint32_t {
  ExpNegativeQ64Table_1_0,
};

struct ResolvedPnrDeterminismPolicy final {
  std::uint64_t masterSeed;
  ResolvedPnrPrngProtocol prngProtocol;
  ResolvedPnrAcceptanceProtocol acceptanceProtocol;
};

enum class ResolvedPnrViolationKind : std::uint32_t {
#define LOOM_MAPPING_VIOLATION(Name, Ordinal, DisplayName, ConfigSpelling)     \
  Name = Ordinal,
#include "Common/MappingObjectiveKinds.def"
};

inline constexpr std::uint32_t resolvedPnrViolationKindCount = 0
#define LOOM_MAPPING_VIOLATION(Name, Ordinal, DisplayName, ConfigSpelling) +1
#include "Common/MappingObjectiveKinds.def"
    ;

struct ResolvedPnrTemporaryViolationPolicy final {
  std::vector<ResolvedPnrViolationKind> admitted;
};

struct ResolvedMappingViolationObjectiveSource final {
  ResolvedPnrViolationKind kind;

  friend bool operator==(ResolvedMappingViolationObjectiveSource lhs,
                         ResolvedMappingViolationObjectiveSource rhs) {
    return lhs.kind == rhs.kind;
  }
};

struct ResolvedMappingMeasureObjectiveSource final {
  std::uint32_t ordinal;

  friend bool operator==(ResolvedMappingMeasureObjectiveSource lhs,
                         ResolvedMappingMeasureObjectiveSource rhs) {
    return lhs.ordinal == rhs.ordinal;
  }
};

struct ResolvedEvaluationMetricObjectiveSource final {
  std::uint32_t evidenceObligationTemplate;
  std::uint64_t metricRequestOrdinal;

  friend bool operator==(ResolvedEvaluationMetricObjectiveSource lhs,
                         ResolvedEvaluationMetricObjectiveSource rhs) {
    return lhs.evidenceObligationTemplate == rhs.evidenceObligationTemplate &&
           lhs.metricRequestOrdinal == rhs.metricRequestOrdinal;
  }
};

using ResolvedObjectiveScalarSource =
    std::variant<ResolvedMappingViolationObjectiveSource,
                 ResolvedMappingMeasureObjectiveSource,
                 ResolvedEvaluationMetricObjectiveSource>;

struct ResolvedObjectiveInteger final {
  bool negative;
  std::uint64_t magnitude;

  friend bool operator==(ResolvedObjectiveInteger lhs,
                         ResolvedObjectiveInteger rhs) {
    return lhs.negative == rhs.negative && lhs.magnitude == rhs.magnitude;
  }
};

struct ResolvedObjectiveDecimal final {
  std::int64_t coefficient;
  std::int64_t base10Exponent;

  friend bool operator==(ResolvedObjectiveDecimal lhs,
                         ResolvedObjectiveDecimal rhs) {
    return lhs.coefficient == rhs.coefficient &&
           lhs.base10Exponent == rhs.base10Exponent;
  }
};

using ResolvedObjectiveScalar =
    std::variant<ResolvedObjectiveInteger, ResolvedObjectiveDecimal>;

inline ResolvedObjectiveScalar resolvedObjectiveInteger(std::uint64_t magnitude,
                                                        bool negative = false) {
  return ResolvedObjectiveInteger{negative && magnitude != 0, magnitude};
}

inline ResolvedObjectiveScalar
resolvedObjectiveDecimal(std::int64_t coefficient,
                         std::int64_t base10Exponent) {
  return ResolvedObjectiveDecimal{coefficient, base10Exponent};
}

enum class ResolvedObjectiveDirection : std::uint32_t {
  Minimize,
  Maximize,
};

struct ResolvedObjectiveDimension final {
  ResolvedObjectiveScalarSource source;
  ResolvedObjectiveDirection direction;
  ResolvedObjectiveScalar origin;
  ResolvedObjectiveScalar quantum;
  std::uint64_t lowerIndex;
  std::uint64_t upperIndex;
};

struct ResolvedWeightedObjectiveTerm final {
  std::uint32_t dimension;
  std::uint64_t weight;
};

struct ResolvedWeightedObjectiveLevel final {
  std::vector<ResolvedWeightedObjectiveTerm> terms;
};

struct ResolvedTotalOrdering final {
  std::vector<std::uint32_t> weightedLevels;
};

struct ResolvedObjectiveCatalogs final {
  std::vector<ResolvedObjectiveDimension> dimensions;
  std::vector<ResolvedWeightedObjectiveLevel> weightedLevels;
  std::vector<ResolvedTotalOrdering> totalOrderings;
};

struct ResolvedPnrObjectiveSelection final {
  std::uint32_t selectedTotalOrdering;
  std::uint32_t selectedSearchEnergy;
};

struct ResolvedPnrPolicyConfig final {
  ResolvedPnrSearchPolicy search;
  ResolvedPnrDeterminismPolicy determinism;
  ResolvedPnrTemporaryViolationPolicy temporaryViolations;
  ResolvedPnrObjectiveSelection objectiveSelection;
};

enum class ResolvedProfilePreset : std::uint32_t {
  ReportOnly,
  QuickExplore,
  BalancedExplore,
  PerformanceExplore,
  Implementation,
  StrictImplementation,
};

ResolvedPnrPolicyConfig
resolvedBuiltinSpatialPnrPolicy(ResolvedProfilePreset preset);
ResolvedPnrPolicyConfig
resolvedBuiltinSystemPnrPolicy(ResolvedProfilePreset preset);
ResolvedObjectiveCatalogs resolvedBuiltinObjectiveCatalogs();

llvm::Error
validateResolvedObjectiveCatalogs(const ResolvedObjectiveCatalogs &catalogs);
llvm::Error
validateResolvedPathFinderPolicy(const ResolvedPathFinderPolicy &policy);
llvm::Error validateResolvedPnrActionProposalPolicy(
    const ResolvedPnrActionProposalPolicy &policy);
llvm::Error
validateResolvedDualStepSchedule(const ResolvedDualStepSchedule &schedule);
llvm::Error validateResolvedDualSubgradientPolicy(
    const ResolvedDualSubgradientPolicy &policy);
llvm::Error
validateResolvedPnrAnnealingPolicy(const ResolvedPnrAnnealingPolicy &policy);
llvm::Error
validateResolvedPnrPolicyConfig(const ResolvedPnrPolicyConfig &policy,
                                const ResolvedObjectiveCatalogs &catalogs);

} // namespace loom

#endif // LOOM_COMMON_RESOLVEDPNRPOLICY_H
