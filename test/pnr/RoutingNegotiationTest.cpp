#include "PnR/RoutingNegotiation.h"

#include "Common/ResolvedPnrPolicy.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <cstdlib>
#include <limits>
#include <string>
#include <system_error>
#include <type_traits>
#include <variant>

using namespace loom::pnr;
using namespace loom;

namespace {

static_assert(std::is_same_v<RouteCost, std::uint64_t>);
static_assert(std::is_same_v<DualPrice, std::uint64_t>);
static_assert(std::is_same_v<DualDirection, std::int64_t>);
static_assert(std::is_same_v<DualStep, std::uint64_t>);
static_assert(maxFiniteRouteCost + 1 == routeCostInfinity);

using Kind = RoutingNegotiationError::Kind;

constexpr std::uint64_t u64Max = std::numeric_limits<std::uint64_t>::max();
constexpr std::int64_t i64Max = std::numeric_limits<std::int64_t>::max();
constexpr std::int64_t i64Min = std::numeric_limits<std::int64_t>::min();
constexpr std::uint64_t two63 = std::uint64_t{1} << 63;

ResolvedPathFinderPolicy pathFinderPolicy(ResolvedPathFinderPriceKernel kernel,
                                          std::uint64_t initial,
                                          std::uint64_t growthNumerator,
                                          std::uint64_t growthDenominator,
                                          std::uint64_t historyIncrement) {
  return {
      kernel, initial, {growthNumerator, growthDenominator}, historyIncrement};
}

ResolvedDualStepSchedule constantStep(std::uint64_t step) {
  return {ResolvedDualStepScheduleKind::Constant, step, 0, 0, 0};
}

ResolvedDualStepSchedule geometricStep(std::uint64_t initial,
                                       std::uint64_t minimum,
                                       std::uint64_t numerator,
                                       std::uint64_t denominator) {
  return {ResolvedDualStepScheduleKind::GeometricDecay, initial, minimum,
          numerator, denominator};
}

ResolvedDualStepSchedule harmonicStep(std::uint64_t numerator,
                                      std::uint64_t offset,
                                      std::uint64_t minimum) {
  return {ResolvedDualStepScheduleKind::HarmonicDecay, numerator, offset,
          minimum, 0};
}

ResolvedDualSubgradientPolicy
directionPolicy(ResolvedDualDirectionKernel kind,
                std::optional<ResolvedExactRatio> momentum = std::nullopt) {
  return {kind, momentum, constantStep(1)};
}

void fail(const char *test, const std::string &message) {
  llvm::errs() << test << ": " << message << '\n';
  std::exit(1);
}

void requireEqual(const char *test, const char *what, std::uint64_t actual,
                  std::uint64_t expected) {
  if (actual != expected)
    fail(test, std::string(what) + ": got " + std::to_string(actual) +
                   ", expected " + std::to_string(expected));
}

void requireEqualSigned(const char *test, const char *what, std::int64_t actual,
                        std::int64_t expected) {
  if (actual != expected)
    fail(test, std::string(what) + ": got " + std::to_string(actual) +
                   ", expected " + std::to_string(expected));
}

template <typename T> T takeValue(const char *test, llvm::Expected<T> result) {
  if (!result)
    fail(test, llvm::toString(result.takeError()));
  return *result;
}

void expectValid(const char *test, llvm::Error error) {
  if (error)
    fail(test, llvm::toString(std::move(error)));
}

void expectFailure(const char *test, const char *what, llvm::Error error,
                   Kind expected) {
  if (!error)
    fail(test, std::string(what) + ": expected a failure");
  bool matched = false;
  llvm::handleAllErrors(
      std::move(error),
      [&](const RoutingNegotiationError &negotiationError) {
        matched = negotiationError.kind() == expected;
      },
      [&](const llvm::StringError &configError) {
        matched = expected == Kind::InvalidPolicy &&
                  configError.convertToErrorCode() ==
                      std::make_error_code(std::errc::invalid_argument);
      });
  if (!matched)
    fail(test, std::string(what) + ": unexpected error kind");
}

template <typename T>
void expectFailure(const char *test, const char *what, llvm::Expected<T> result,
                   Kind expected) {
  if (result)
    fail(test, std::string(what) + ": expected a failure");
  expectFailure(test, what, result.takeError(), expected);
}

// The maximum RouteCost is the A* infinity sentinel: every cost kernel accepts
// the largest finite cost and rejects a mathematical result that reaches it.
void routeCostInfinityBoundary() {
  requireEqual(__func__, "accumulated largest finite",
               takeValue(__func__, accumulateRouteCost(maxFiniteRouteCost, 0)),
               maxFiniteRouteCost);
  expectFailure(__func__, "accumulation reaches infinity",
                accumulateRouteCost(maxFiniteRouteCost, 1),
                Kind::ArithmeticOverflow);

  requireEqual(__func__, "dual arc largest finite",
               takeValue(__func__, dualArcResourceCost(routeCostScale,
                                                       maxFiniteRouteCost -
                                                           routeCostScale)),
               maxFiniteRouteCost);
  expectFailure(__func__, "dual arc reaches infinity",
                dualArcResourceCost(routeCostScale,
                                    maxFiniteRouteCost - routeCostScale + 1),
                Kind::ArithmeticOverflow);

  const ResolvedPathFinderPriceKernel multiplicative =
      ResolvedPathFinderPriceKernel::Multiplicative;
  const ResolvedPathFinderPriceKernel additive =
      ResolvedPathFinderPriceKernel::Additive;
  requireEqual(__func__, "multiplicative largest finite",
               takeValue(__func__, pathFinderResourceCost(
                                       multiplicative, routeCostScale, 0, 1,
                                       maxFiniteRouteCost - routeCostScale)),
               maxFiniteRouteCost);
  expectFailure(__func__, "multiplicative reaches infinity",
                pathFinderResourceCost(multiplicative, routeCostScale, 0, 1,
                                       maxFiniteRouteCost - routeCostScale + 1),
                Kind::ArithmeticOverflow);
  requireEqual(__func__, "additive largest finite",
               takeValue(__func__, pathFinderResourceCost(
                                       additive, routeCostScale, 0, 1,
                                       maxFiniteRouteCost - routeCostScale)),
               maxFiniteRouteCost);
  expectFailure(__func__, "additive reaches infinity",
                pathFinderResourceCost(additive, routeCostScale, 0, 1,
                                       maxFiniteRouteCost - routeCostScale + 1),
                Kind::ArithmeticOverflow);
}

// The shared rounding authorities: truncation toward zero and the ceiling
// present-pressure update.
void deterministicRounding() {
  requireEqualSigned(__func__, "positive truncates down",
                     takeValue(__func__, scaleTowardZero(7, 2, 3)), 4);
  requireEqualSigned(__func__, "negative truncates toward zero",
                     takeValue(__func__, scaleTowardZero(-7, 2, 3)), -4);
  requireEqualSigned(__func__, "minimum value halved",
                     takeValue(__func__, scaleTowardZero(i64Min, 1, 2)),
                     i64Min / 2);
  requireEqualSigned(__func__, "minimum value identity",
                     takeValue(__func__, scaleTowardZero(i64Min, 1, 1)),
                     i64Min);
  expectFailure(__func__, "scale denominator zero", scaleTowardZero(5, 3, 0),
                Kind::InvalidPolicy);

  requireEqual(__func__, "ceiling rounds up",
               takeValue(__func__, ceilMulDiv(5, 3, 2)), 8);
  requireEqual(__func__, "ceiling of an exact quotient",
               takeValue(__func__, ceilMulDiv(4, 2, 2)), 4);
  requireEqual(__func__, "ceiling of the maximum halved",
               takeValue(__func__, ceilMulDiv(u64Max, 1, 2)), two63);
  expectFailure(__func__, "ceiling denominator zero", ceilMulDiv(5, 1, 0),
                Kind::InvalidPolicy);
}

// Dual residuals normalize unlike raw capacities to one sign-preserving scale;
// signed intermediates still fail rather than wrapping.
void signedOverflowBoundaries() {
  requireEqualSigned(__func__, "half-capacity violation",
                     takeValue(__func__, dualResidual(6, 4)),
                     routeCostScale / 2);
  requireEqualSigned(__func__, "half-capacity slack",
                     takeValue(__func__, dualResidual(2, 4)),
                     -static_cast<std::int64_t>(routeCostScale / 2));
  requireEqualSigned(__func__, "balanced zero capacity",
                     takeValue(__func__, dualResidual(0, 0)), 0);
  expectFailure(__func__, "positive use with zero capacity", dualResidual(1, 0),
                Kind::InvalidPolicy);
  expectFailure(__func__, "normalized violation above the maximum",
                dualResidual(u64Max, 1), Kind::ArithmeticOverflow);

  const ResolvedDualSubgradientPolicy momentum = directionPolicy(
      ResolvedDualDirectionKernel::MomentumDeflected, ResolvedExactRatio{1, 2});
  expectFailure(__func__, "deflection above the maximum",
                dualDirectionFromResidual(momentum, i64Max, i64Max),
                Kind::ArithmeticOverflow);
  expectFailure(__func__, "deflection below the minimum",
                dualDirectionFromResidual(momentum, i64Min, i64Min),
                Kind::ArithmeticOverflow);
}

void pathFinderCostVectors() {
  const ResolvedPathFinderPriceKernel multiplicative =
      ResolvedPathFinderPriceKernel::Multiplicative;
  const ResolvedPathFinderPriceKernel additive =
      ResolvedPathFinderPriceKernel::Additive;

  requireEqual(__func__, "half-capacity claim",
               takeValue(__func__, normalizedRouteClaimCost(3, 6)),
               std::uint64_t{1} << 31);
  requireEqual(__func__, "heterogeneous half-capacity claim",
               takeValue(__func__, normalizedRouteClaimCost(4, 8)),
               takeValue(__func__, normalizedRouteClaimCost(32, 64)));
  requireEqual(__func__, "rounded normalized overuse",
               takeValue(__func__, normalizedRouteOveruseCost(4, 3, 6)),
               715827883);

  requireEqual(
      __func__, "full-capacity multiplicative congestion",
      takeValue(__func__, pathFinderResourceCost(multiplicative, routeCostScale,
                                                 routeCostScale, 1, 0)),
      2 * routeCostScale);
  requireEqual(__func__, "full-capacity additive congestion and history",
               takeValue(__func__, pathFinderResourceCost(
                                       additive, routeCostScale, routeCostScale,
                                       1, routeCostScale)),
               3 * routeCostScale);

  const RouteCost oneThird =
      takeValue(__func__, normalizedRouteClaimCost(1, 3));
  requireEqual(__func__, "single-ceiling non-divisible product",
               takeValue(__func__, pathFinderResourceCost(multiplicative,
                                                          oneThird, oneThird, 3,
                                                          3 * routeCostScale)),
               11453246131ULL);
  requireEqual(
      __func__, "widened present-pressure factor",
      takeValue(__func__, pathFinderResourceCost(multiplicative, 1,
                                                 maxFiniteRouteCost, 1, 0)),
      routeCostScale + 1);
  requireEqual(
      __func__, "widened history factor",
      takeValue(__func__, pathFinderResourceCost(multiplicative, 1, 0, 1,
                                                 maxFiniteRouteCost)),
      routeCostScale + 1);

  requireEqual(__func__, "scaled conflict contribution",
               takeValue(__func__, scaledRouteProduct(routeCostScale / 2,
                                                      routeCostScale / 4)),
               routeCostScale / 8);
  requireEqual(
      __func__, "dual one-unit price",
      takeValue(__func__, dualArcResourceCost(routeCostScale, routeCostScale)),
      2 * routeCostScale);

  // Without congestion both kernels charge the generic lower bound.
  requireEqual(
      __func__, "multiplicative uncongested claim",
      takeValue(__func__, pathFinderResourceCost(multiplicative,
                                                 routeCostScale / 2, 0, 3, 0)),
      routeCostScale / 2);
  requireEqual(__func__, "additive uncongested claim",
               takeValue(__func__, pathFinderResourceCost(
                                       additive, routeCostScale / 2, 0, 3, 0)),
               routeCostScale / 2);

  RouteCost arc = 0;
  for (RouteCost term :
       {2 * routeCostScale, 3 * routeCostScale, routeCostScale / 2})
    arc = takeValue(__func__, accumulateRouteCost(arc, term));
  requireEqual(__func__, "accumulated arc", arc,
               5 * routeCostScale + routeCostScale / 2);
}

void pathFinderPressureUpdates() {
  requireEqual(__func__, "proportional history",
               takeValue(__func__, pathFinderHistoryUpdate(routeCostScale, 3,
                                                           routeCostScale / 4)),
               routeCostScale + 3 * (routeCostScale / 4));
  requireEqual(__func__, "no overuse keeps history",
               takeValue(__func__, pathFinderHistoryUpdate(5, 1, 0)), 5);
  expectFailure(__func__, "history above the representable range",
                pathFinderHistoryUpdate(u64Max, 1, 1),
                Kind::ArithmeticOverflow);

  expectValid(__func__,
              validateResolvedPathFinderPolicy(pathFinderPolicy(
                  ResolvedPathFinderPriceKernel::Multiplicative, 1, 3, 2, 1)));
  expectFailure(__func__, "zero initial pressure",
                validateResolvedPathFinderPolicy(pathFinderPolicy(
                    ResolvedPathFinderPriceKernel::Multiplicative, 0, 3, 2, 1)),
                Kind::InvalidPolicy);
  expectFailure(__func__, "zero history increment",
                validateResolvedPathFinderPolicy(pathFinderPolicy(
                    ResolvedPathFinderPriceKernel::Multiplicative, 1, 3, 2, 0)),
                Kind::InvalidPolicy);
  expectFailure(__func__, "decaying growth ratio",
                validateResolvedPathFinderPolicy(pathFinderPolicy(
                    ResolvedPathFinderPriceKernel::Multiplicative, 1, 2, 3, 1)),
                Kind::InvalidPolicy);
  expectFailure(__func__, "unreduced growth ratio",
                validateResolvedPathFinderPolicy(pathFinderPolicy(
                    ResolvedPathFinderPriceKernel::Multiplicative, 1, 4, 2, 1)),
                Kind::InvalidPolicy);
}

void dualDirectionVectors() {
  const ResolvedDualSubgradientPolicy projected =
      directionPolicy(ResolvedDualDirectionKernel::ProjectedSigned);
  requireEqualSigned(
      __func__, "projected keeps slack",
      takeValue(__func__, dualDirectionFromResidual(projected, -4, 0)), -4);

  const ResolvedDualSubgradientPolicy positiveOnly =
      directionPolicy(ResolvedDualDirectionKernel::PositiveViolationOnly);
  requireEqualSigned(
      __func__, "positive-only clamps slack",
      takeValue(__func__, dualDirectionFromResidual(positiveOnly, -4, 0)), 0);
  requireEqualSigned(
      __func__, "positive-only keeps violation",
      takeValue(__func__, dualDirectionFromResidual(positiveOnly, 4, 0)), 4);

  const ResolvedDualSubgradientPolicy momentum = directionPolicy(
      ResolvedDualDirectionKernel::MomentumDeflected, ResolvedExactRatio{1, 2});
  requireEqualSigned(
      __func__, "momentum adds the deflected previous direction",
      takeValue(__func__, dualDirectionFromResidual(momentum, 3, 6)), 6);
  requireEqualSigned(
      __func__, "momentum truncates toward zero",
      takeValue(__func__, dualDirectionFromResidual(momentum, 0, 7)), 3);
  requireEqualSigned(
      __func__, "momentum truncates negatives toward zero",
      takeValue(__func__, dualDirectionFromResidual(momentum, 0, -7)), -3);

  const ResolvedDualSubgradientPolicy zeroMomentum = directionPolicy(
      ResolvedDualDirectionKernel::MomentumDeflected, ResolvedExactRatio{0, 1});
  requireEqualSigned(
      __func__, "zero momentum keeps the current residual",
      takeValue(__func__, dualDirectionFromResidual(zeroMomentum, 5, 99)), 5);
}

// Canonical schedules resolve to their defined steps; degenerate forms that
// reduce to Constant belong to the resolver and are rejected here.
void dualStepScheduleContract() {
  requireEqual(__func__, "constant step",
               takeValue(__func__, dualStepAt(constantStep(7), 5)), 7);

  const ResolvedDualStepSchedule geometric = geometricStep(100, 3, 1, 2);
  requireEqual(__func__, "geometric initial step",
               takeValue(__func__, dualStepAt(geometric, 0)), 100);
  requireEqual(__func__, "geometric truncated step",
               takeValue(__func__, dualStepAt(geometric, 3)), 12);
  requireEqual(__func__, "geometric holds at its floor",
               takeValue(__func__, dualStepAt(geometric, 64)), 3);

  const ResolvedDualStepSchedule harmonic = harmonicStep(100, 10, 2);
  requireEqual(__func__, "harmonic initial step",
               takeValue(__func__, dualStepAt(harmonic, 0)), 10);
  requireEqual(__func__, "harmonic truncated step",
               takeValue(__func__, dualStepAt(harmonic, 1)), 9);
  requireEqual(__func__, "harmonic holds at its floor",
               takeValue(__func__, dualStepAt(harmonic, 1000)), 2);

  expectFailure(__func__, "geometric initial step at the floor",
                validateResolvedDualStepSchedule(geometricStep(3, 3, 1, 2)),
                Kind::InvalidPolicy);
  expectFailure(__func__, "harmonic pinned to its floor",
                validateResolvedDualStepSchedule(harmonicStep(10, 10, 1)),
                Kind::InvalidPolicy);
  expectFailure(__func__, "geometric zero minimum step",
                validateResolvedDualStepSchedule(geometricStep(10, 0, 1, 2)),
                Kind::InvalidPolicy);
  expectFailure(__func__, "geometric growing decay ratio",
                validateResolvedDualStepSchedule(geometricStep(10, 3, 2, 2)),
                Kind::InvalidPolicy);
  expectFailure(__func__, "geometric unreduced decay ratio",
                validateResolvedDualStepSchedule(geometricStep(10, 3, 2, 4)),
                Kind::InvalidPolicy);
  expectFailure(__func__, "harmonic zero offset",
                validateResolvedDualStepSchedule(harmonicStep(1, 0, 1)),
                Kind::InvalidPolicy);
}

// Public kernels reject inputs outside their contract domain, so an invalid
// policy, inactive session state, or absent claim cannot execute.
void publicValidationBoundary() {
  expectFailure(__func__, "constant step zero", dualStepAt(constantStep(0), 0),
                Kind::InvalidPolicy);
  expectFailure(__func__, "noncanonical geometric schedule",
                dualStepAt(geometricStep(3, 3, 1, 2), 1), Kind::InvalidPolicy);
  expectFailure(
      __func__, "momentum beta at one",
      dualDirectionFromResidual(
          directionPolicy(ResolvedDualDirectionKernel::MomentumDeflected,
                          ResolvedExactRatio{1, 1}),
          1, 1),
      Kind::InvalidPolicy);
  expectFailure(
      __func__, "momentum beta denominator zero",
      dualDirectionFromResidual(
          directionPolicy(ResolvedDualDirectionKernel::MomentumDeflected,
                          ResolvedExactRatio{1, 0}),
          1, 1),
      Kind::InvalidPolicy);
  expectFailure(
      __func__, "prior direction outside momentum",
      dualDirectionFromResidual(
          directionPolicy(ResolvedDualDirectionKernel::ProjectedSigned), -4,
          99),
      Kind::InvalidPolicy);
  expectFailure(__func__, "pathfinder absent claim",
                pathFinderResourceCost(
                    ResolvedPathFinderPriceKernel::Multiplicative, 0, 1, 2, 5),
                Kind::InvalidPolicy);
  expectFailure(__func__, "dual arc absent claim", dualArcResourceCost(0, 3),
                Kind::InvalidPolicy);

  // Active policy values reaching a raw kernel are rejected rather than
  // silently degenerating into an identity update.
  expectFailure(__func__, "zero present pressure",
                pathFinderResourceCost(
                    ResolvedPathFinderPriceKernel::Multiplicative, 3, 1, 0, 1),
                Kind::InvalidPolicy);
  expectFailure(__func__, "zero history increment",
                pathFinderHistoryUpdate(2, 0, 4), Kind::InvalidPolicy);
  expectFailure(__func__, "zero dual step", dualPriceUpdate(10, 0, 4),
                Kind::InvalidPolicy);
}

void dualPriceUpdateVectors() {
  requireEqual(__func__, "price rises",
               takeValue(__func__, dualPriceUpdate(10, 3, 4)), 22);
  requireEqual(__func__, "price falls",
               takeValue(__func__, dualPriceUpdate(10, 3, -3)), 1);
  requireEqual(__func__, "projection clamps at zero",
               takeValue(__func__, dualPriceUpdate(10, 3, -4)), 0);
  requireEqual(__func__, "extreme negative product clamps",
               takeValue(__func__, dualPriceUpdate(0, u64Max, i64Min)), 0);
  expectFailure(__func__, "price above the representable range",
                dualPriceUpdate(u64Max, 2, 1), Kind::ArithmeticOverflow);
}

} // namespace

int main() {
  routeCostInfinityBoundary();
  deterministicRounding();
  signedOverflowBoundaries();
  pathFinderCostVectors();
  pathFinderPressureUpdates();
  dualDirectionVectors();
  dualStepScheduleContract();
  publicValidationBoundary();
  dualPriceUpdateVectors();
  return 0;
}
