#include "PnR/RoutingNegotiation.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <cstdlib>
#include <limits>
#include <string>
#include <type_traits>
#include <variant>

using namespace loom::pnr;

namespace {

static_assert(std::is_same_v<RouteCost, std::uint64_t>);
static_assert(std::is_same_v<DualPrice, std::uint64_t>);
static_assert(std::is_same_v<DualDirection, std::int64_t>);
static_assert(std::is_same_v<DualStep, std::uint64_t>);
static_assert(std::variant_size_v<DualDirectionKernel> == 3);
static_assert(std::variant_size_v<DualStepSchedule> == 3);
static_assert(maxFiniteRouteCost + 1 == routeCostInfinity);

using Kind = RoutingNegotiationError::Kind;

constexpr std::uint64_t u64Max = std::numeric_limits<std::uint64_t>::max();
constexpr std::int64_t i64Max = std::numeric_limits<std::int64_t>::max();
constexpr std::int64_t i64Min = std::numeric_limits<std::int64_t>::min();
constexpr std::uint64_t two63 = std::uint64_t{1} << 63;

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
  llvm::handleAllErrors(std::move(error),
                        [&](const RoutingNegotiationError &negotiationError) {
                          matched = negotiationError.kind() == expected;
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

  requireEqual(
      __func__, "dual arc largest finite",
      takeValue(__func__, dualArcResourceCost(1, maxFiniteRouteCost - 1)),
      maxFiniteRouteCost);
  expectFailure(__func__, "dual arc reaches infinity",
                dualArcResourceCost(1, maxFiniteRouteCost),
                Kind::ArithmeticOverflow);

  // X = 1, so the cost is 1 + present pressure in both price kernels.
  const PathFinderPriceKernel multiplicative =
      PathFinderPriceKernel::Multiplicative;
  const PathFinderPriceKernel additive = PathFinderPriceKernel::Additive;
  requireEqual(
      __func__, "multiplicative largest finite",
      takeValue(__func__, pathFinderResourceCost(multiplicative, 1, 0, 0,
                                                 maxFiniteRouteCost - 1, 0)),
      maxFiniteRouteCost);
  expectFailure(
      __func__, "multiplicative reaches infinity",
      pathFinderResourceCost(multiplicative, 1, 0, 0, maxFiniteRouteCost, 0),
      Kind::ArithmeticOverflow);
  requireEqual(
      __func__, "additive largest finite",
      takeValue(__func__, pathFinderResourceCost(additive, 1, 0, 0,
                                                 maxFiniteRouteCost - 1, 0)),
      maxFiniteRouteCost);
  expectFailure(
      __func__, "additive reaches infinity",
      pathFinderResourceCost(additive, 1, 0, 0, maxFiniteRouteCost, 0),
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

// Signed intermediates must fail in both directions instead of wrapping.
void signedOverflowBoundaries() {
  requireEqualSigned(__func__, "largest violation",
                     takeValue(__func__, dualResidual(i64Max, 0)), i64Max);
  requireEqualSigned(__func__, "largest slack",
                     takeValue(__func__, dualResidual(0, two63)), i64Min);
  expectFailure(__func__, "violation above the maximum",
                dualResidual(u64Max, 0), Kind::ArithmeticOverflow);
  expectFailure(__func__, "slack below the minimum", dualResidual(0, u64Max),
                Kind::ArithmeticOverflow);

  const DualDirectionKernel momentum = MomentumDeflectedDirection{1, 2};
  expectFailure(__func__, "deflection above the maximum",
                dualDirectionFromResidual(momentum, i64Max, i64Max),
                Kind::ArithmeticOverflow);
  expectFailure(__func__, "deflection below the minimum",
                dualDirectionFromResidual(momentum, i64Min, i64Min),
                Kind::ArithmeticOverflow);
}

void pathFinderCostVectors() {
  const PathFinderPriceKernel multiplicative =
      PathFinderPriceKernel::Multiplicative;
  const PathFinderPriceKernel additive = PathFinderPriceKernel::Additive;

  // q=3, u=4, cap=6, P=2, H=1 gives X=1.
  requireEqual(__func__, "multiplicative congested claim",
               takeValue(__func__,
                         pathFinderResourceCost(multiplicative, 3, 4, 6, 2, 1)),
               18);
  requireEqual(
      __func__, "additive congested claim",
      takeValue(__func__, pathFinderResourceCost(additive, 3, 4, 6, 2, 1)), 8);

  // Without congestion both kernels charge the generic lower bound.
  requireEqual(__func__, "multiplicative uncongested claim",
               takeValue(__func__, pathFinderResourceCost(multiplicative, 7, 0,
                                                          10, 3, 0)),
               7);
  requireEqual(
      __func__, "additive uncongested claim",
      takeValue(__func__, pathFinderResourceCost(additive, 7, 0, 10, 3, 0)), 7);

  RouteCost arc = 0;
  for (RouteCost term : {18, 8, 7})
    arc = takeValue(__func__, accumulateRouteCost(arc, term));
  requireEqual(__func__, "accumulated arc", arc, 33);
}

void pathFinderPressureUpdates() {
  requireEqual(__func__, "proportional history",
               takeValue(__func__, pathFinderHistoryUpdate(2, 3, 4)), 14);
  requireEqual(__func__, "no overuse keeps history",
               takeValue(__func__, pathFinderHistoryUpdate(5, 1, 0)), 5);
  expectFailure(__func__, "history above the representable range",
                pathFinderHistoryUpdate(u64Max, 1, 1),
                Kind::ArithmeticOverflow);

  expectValid(__func__, validatePathFinderPressurePolicy(
                            PathFinderPressurePolicy{1, 3, 2, 1}));
  expectFailure(
      __func__, "zero initial pressure",
      validatePathFinderPressurePolicy(PathFinderPressurePolicy{0, 3, 2, 1}),
      Kind::InvalidPolicy);
  expectFailure(
      __func__, "zero history increment",
      validatePathFinderPressurePolicy(PathFinderPressurePolicy{1, 3, 2, 0}),
      Kind::InvalidPolicy);
  expectFailure(
      __func__, "decaying growth ratio",
      validatePathFinderPressurePolicy(PathFinderPressurePolicy{1, 2, 3, 1}),
      Kind::InvalidPolicy);
  expectFailure(
      __func__, "unreduced growth ratio",
      validatePathFinderPressurePolicy(PathFinderPressurePolicy{1, 4, 2, 1}),
      Kind::InvalidPolicy);
}

void dualDirectionVectors() {
  const DualDirectionKernel projected = ProjectedSignedDirection{};
  requireEqualSigned(
      __func__, "projected keeps slack",
      takeValue(__func__, dualDirectionFromResidual(projected, -4, 0)), -4);

  const DualDirectionKernel positiveOnly = PositiveViolationOnlyDirection{};
  requireEqualSigned(
      __func__, "positive-only clamps slack",
      takeValue(__func__, dualDirectionFromResidual(positiveOnly, -4, 0)), 0);
  requireEqualSigned(
      __func__, "positive-only keeps violation",
      takeValue(__func__, dualDirectionFromResidual(positiveOnly, 4, 0)), 4);

  const DualDirectionKernel momentum = MomentumDeflectedDirection{1, 2};
  requireEqualSigned(
      __func__, "momentum adds the deflected previous direction",
      takeValue(__func__, dualDirectionFromResidual(momentum, 3, 6)), 6);
  requireEqualSigned(
      __func__, "momentum truncates toward zero",
      takeValue(__func__, dualDirectionFromResidual(momentum, 0, 7)), 3);
  requireEqualSigned(
      __func__, "momentum truncates negatives toward zero",
      takeValue(__func__, dualDirectionFromResidual(momentum, 0, -7)), -3);
}

// Canonical schedules resolve to their defined steps; degenerate forms that
// reduce to Constant belong to the resolver and are rejected here.
void dualStepScheduleContract() {
  requireEqual(__func__, "constant step",
               takeValue(__func__, dualStepAt(ConstantStepSchedule{7}, 5)), 7);

  const DualStepSchedule geometric = GeometricDecayStepSchedule{100, 3, 1, 2};
  requireEqual(__func__, "geometric initial step",
               takeValue(__func__, dualStepAt(geometric, 0)), 100);
  requireEqual(__func__, "geometric truncated step",
               takeValue(__func__, dualStepAt(geometric, 3)), 12);
  requireEqual(__func__, "geometric holds at its floor",
               takeValue(__func__, dualStepAt(geometric, 64)), 3);

  const DualStepSchedule harmonic = HarmonicDecayStepSchedule{100, 10, 2};
  requireEqual(__func__, "harmonic initial step",
               takeValue(__func__, dualStepAt(harmonic, 0)), 10);
  requireEqual(__func__, "harmonic truncated step",
               takeValue(__func__, dualStepAt(harmonic, 1)), 9);
  requireEqual(__func__, "harmonic holds at its floor",
               takeValue(__func__, dualStepAt(harmonic, 1000)), 2);

  expectFailure(
      __func__, "geometric initial step at the floor",
      validateDualStepSchedule(GeometricDecayStepSchedule{3, 3, 1, 2}),
      Kind::InvalidPolicy);
  expectFailure(__func__, "harmonic pinned to its floor",
                validateDualStepSchedule(HarmonicDecayStepSchedule{10, 10, 1}),
                Kind::InvalidPolicy);
  expectFailure(
      __func__, "geometric zero minimum step",
      validateDualStepSchedule(GeometricDecayStepSchedule{10, 0, 1, 2}),
      Kind::InvalidPolicy);
  expectFailure(
      __func__, "geometric growing decay ratio",
      validateDualStepSchedule(GeometricDecayStepSchedule{10, 3, 2, 2}),
      Kind::InvalidPolicy);
  expectFailure(
      __func__, "geometric unreduced decay ratio",
      validateDualStepSchedule(GeometricDecayStepSchedule{10, 3, 2, 4}),
      Kind::InvalidPolicy);
  expectFailure(__func__, "harmonic zero offset",
                validateDualStepSchedule(HarmonicDecayStepSchedule{1, 0, 1}),
                Kind::InvalidPolicy);
}

// Public kernels reject inputs outside their contract domain, so an invalid
// policy, inactive session state, or absent claim cannot execute.
void publicValidationBoundary() {
  expectFailure(__func__, "constant step zero",
                dualStepAt(ConstantStepSchedule{0}, 0), Kind::InvalidPolicy);
  expectFailure(__func__, "noncanonical geometric schedule",
                dualStepAt(GeometricDecayStepSchedule{3, 3, 1, 2}, 1),
                Kind::InvalidPolicy);
  expectFailure(
      __func__, "momentum beta at one",
      dualDirectionFromResidual(MomentumDeflectedDirection{1, 1}, 1, 1),
      Kind::InvalidPolicy);
  expectFailure(
      __func__, "momentum beta denominator zero",
      dualDirectionFromResidual(MomentumDeflectedDirection{1, 0}, 1, 1),
      Kind::InvalidPolicy);
  expectFailure(__func__, "prior direction outside momentum",
                dualDirectionFromResidual(ProjectedSignedDirection{}, -4, 99),
                Kind::InvalidPolicy);
  expectFailure(__func__, "pathfinder absent claim",
                pathFinderResourceCost(PathFinderPriceKernel::Multiplicative, 0,
                                       9, 6, 2, 5),
                Kind::InvalidPolicy);
  expectFailure(__func__, "dual arc absent claim", dualArcResourceCost(0, 3),
                Kind::InvalidPolicy);

  // Active policy values reaching a raw kernel are rejected rather than
  // silently degenerating into an identity update.
  expectFailure(__func__, "zero present pressure",
                pathFinderResourceCost(PathFinderPriceKernel::Multiplicative, 3,
                                       4, 6, 0, 1),
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
