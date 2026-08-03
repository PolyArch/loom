#ifndef LOOM_COMMON_RESOLVEDPNRPOLICY_H
#define LOOM_COMMON_RESOLVEDPNRPOLICY_H

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
  ResolvedRoutingNegotiationPolicy negotiation;
  std::optional<std::uint32_t> routeGuidanceBinding;
};

struct ResolvedPnrAnnealingPolicy final {
  std::uint64_t calibrationProposalCount;
  ResolvedExactRatio positiveDeltaQuantile;
  ResolvedExactRatio targetInitialAcceptance;
  std::uint64_t fallbackTemperature;
  std::uint64_t minimumTemperature;
  ResolvedExactRatio coolingRatio;
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

struct ResolvedPnrSearchPolicy final {
  ResolvedPnrInitializerPolicy initializer;
  ResolvedPnrActionProposalPolicy actionProposal;
  ResolvedPnrRoutingPolicy routing;
  ResolvedPnrAnnealingPolicy annealing;
  std::uint64_t focusedClosureProposalLimit;
  ResolvedPnrExactRepairPolicy exactRepair;
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

enum class ResolvedObjectiveSourceKind : std::uint32_t {
  MappingViolation,
  MappingMeasure,
};

enum class ResolvedObjectiveDirection : std::uint32_t {
  Minimize,
  Maximize,
};

struct ResolvedObjectiveDimension final {
  ResolvedObjectiveSourceKind sourceKind;
  std::uint32_t sourceOrdinal;
  ResolvedObjectiveDirection direction;
  std::uint64_t origin;
  std::uint64_t quantum;
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
  std::vector<std::uint32_t> focusedClosureDimensions;
};

struct ResolvedPnrEvaluationBindingSelection final {
  std::uint32_t obligationTemplate;
  std::uint32_t interactionDomain;
};

struct ResolvedPnrPolicyConfig final {
  ResolvedPnrSearchPolicy search;
  ResolvedPnrDeterminismPolicy determinism;
  ResolvedPnrTemporaryViolationPolicy temporaryViolations;
  ResolvedPnrObjectiveSelection objectiveSelection;
  std::vector<ResolvedPnrEvaluationBindingSelection> evaluationBindings;
};

enum class ResolvedProfilePreset : std::uint32_t {
  ReportOnly,
  QuickExplore,
  BalancedExplore,
  PerformanceExplore,
  Implementation,
  StrictImplementation,
};

ResolvedPnrPolicyConfig resolvedBuiltinPnrPolicy(ResolvedProfilePreset preset);
ResolvedObjectiveCatalogs resolvedBuiltinObjectiveCatalogs();

llvm::Error
validateResolvedObjectiveCatalogs(const ResolvedObjectiveCatalogs &catalogs);
llvm::Error
validateResolvedPathFinderPolicy(const ResolvedPathFinderPolicy &policy);
llvm::Error
validateResolvedDualStepSchedule(const ResolvedDualStepSchedule &schedule);
llvm::Error validateResolvedDualSubgradientPolicy(
    const ResolvedDualSubgradientPolicy &policy);
llvm::Error
validateResolvedPnrPolicyConfig(const ResolvedPnrPolicyConfig &policy,
                                const ResolvedObjectiveCatalogs &catalogs);

} // namespace loom

#endif // LOOM_COMMON_RESOLVEDPNRPOLICY_H
