#include "PnR/RoutingNegotiation.h"

#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <limits>
#include <numeric>
#include <utility>

using namespace loom::pnr;

char RoutingNegotiationError::ID;

RoutingNegotiationError::RoutingNegotiationError(Kind kind, std::string message)
    : kind_(kind), message_(std::move(message)) {}

void RoutingNegotiationError::log(llvm::raw_ostream &stream) const {
  stream << message_;
}

std::error_code RoutingNegotiationError::convertToErrorCode() const {
  switch (kind_) {
  case Kind::InvalidPolicy:
    return std::make_error_code(std::errc::invalid_argument);
  case Kind::ArithmeticOverflow:
    return std::make_error_code(std::errc::result_out_of_range);
  }
  llvm_unreachable("invalid routing negotiation error kind");
}

namespace {

template <typename... Parts> std::string renderMessage(Parts &&...parts) {
  std::string message;
  llvm::raw_string_ostream stream(message);
  (stream << ... << parts);
  return message;
}

template <typename... Parts> llvm::Error invalidPolicy(Parts &&...parts) {
  return llvm::make_error<RoutingNegotiationError>(
      RoutingNegotiationError::Kind::InvalidPolicy,
      renderMessage("routing negotiation invalid policy: ", parts...));
}

template <typename... Parts> llvm::Error arithmeticOverflow(Parts &&...parts) {
  return llvm::make_error<RoutingNegotiationError>(
      RoutingNegotiationError::Kind::ArithmeticOverflow,
      renderMessage("routing negotiation arithmetic overflow: ", parts...));
}

// The single checked-math path. Every kernel forms intermediates in 128-bit
// and narrows through these helpers; nothing wraps, saturates, or changes
// representation silently.

llvm::Expected<std::uint64_t> checkedAdd(std::uint64_t lhs, std::uint64_t rhs,
                                         llvm::StringRef operation) {
  const unsigned __int128 sum =
      static_cast<unsigned __int128>(lhs) + static_cast<unsigned __int128>(rhs);
  if (sum > std::numeric_limits<std::uint64_t>::max())
    return arithmeticOverflow(operation, ": ", lhs, " + ", rhs,
                              " is not representable in uint64_t");
  return static_cast<std::uint64_t>(sum);
}

llvm::Expected<std::uint64_t> checkedMultiply(std::uint64_t lhs,
                                              std::uint64_t rhs,
                                              llvm::StringRef operation) {
  const unsigned __int128 product =
      static_cast<unsigned __int128>(lhs) * static_cast<unsigned __int128>(rhs);
  if (product > std::numeric_limits<std::uint64_t>::max())
    return arithmeticOverflow(operation, ": ", lhs, " * ", rhs,
                              " is not representable in uint64_t");
  return static_cast<std::uint64_t>(product);
}

// The single finite-cost narrowing rule. routeCostInfinity is reserved as the
// A* infinity sentinel, so a mathematical result that reaches or exceeds it is
// an overflow rather than a publishable cost; every value leaving this module
// as a RouteCost passes through here.
llvm::Expected<RouteCost> narrowFiniteCost(unsigned __int128 cost,
                                           llvm::StringRef operation) {
  if (cost > maxFiniteRouteCost)
    return arithmeticOverflow(operation, ": exceeds the largest finite cost ",
                              maxFiniteRouteCost);
  return static_cast<RouteCost>(cost);
}

llvm::Expected<std::int64_t> checkedAddSigned(std::int64_t lhs,
                                              std::int64_t rhs,
                                              llvm::StringRef operation) {
  const __int128 sum = static_cast<__int128>(lhs) + static_cast<__int128>(rhs);
  if (sum > std::numeric_limits<std::int64_t>::max() ||
      sum < std::numeric_limits<std::int64_t>::min())
    return arithmeticOverflow(operation, ": ", lhs, " + ", rhs,
                              " is not representable in int64_t");
  return static_cast<std::int64_t>(sum);
}

struct ScaledMagnitude {
  bool negative;
  std::uint64_t magnitude;
};

constexpr std::uint64_t int64MinMagnitude = std::uint64_t{1} << 63;

// Single scale-toward-zero core. Values are decomposed into sign and
// magnitude so the most negative int64_t is handled exactly; the exact
// truncating quotient of value * numerator / denominator is formed in a
// 128-bit intermediate and must fit the caller-selected magnitude limit.
llvm::Expected<ScaledMagnitude>
scaleCore(bool negative, std::uint64_t magnitude, std::uint64_t numerator,
          std::uint64_t denominator, std::uint64_t magnitudeLimit,
          llvm::StringRef operation) {
  if (denominator == 0)
    return invalidPolicy(operation, ": denominator must be positive");
  const unsigned __int128 product =
      static_cast<unsigned __int128>(magnitude) * numerator;
  const unsigned __int128 quotient = product / denominator;
  if (quotient > magnitudeLimit)
    return arithmeticOverflow(operation, ": scaled quotient of ", magnitude,
                              " * ", numerator, " / ", denominator,
                              " is not representable");
  const std::uint64_t result = static_cast<std::uint64_t>(quotient);
  return ScaledMagnitude{negative && result != 0, result};
}

// Nonnegative entry into the same scale-toward-zero core, used by the step
// schedules whose values are uint64_t.
llvm::Expected<std::uint64_t> scaleMagnitude(std::uint64_t value,
                                             std::uint64_t numerator,
                                             std::uint64_t denominator,
                                             llvm::StringRef operation) {
  auto scaled = scaleCore(false, value, numerator, denominator,
                          std::numeric_limits<std::uint64_t>::max(), operation);
  if (!scaled)
    return scaled.takeError();
  return scaled->magnitude;
}

llvm::Error requireAtLeast(std::uint64_t value, std::uint64_t minimum,
                           llvm::StringRef field) {
  if (value >= minimum)
    return llvm::Error::success();
  return invalidPolicy(field, " must be at least ", minimum, ", got ", value);
}

} // namespace

llvm::Error loom::pnr::validatePathFinderPressurePolicy(
    const PathFinderPressurePolicy &policy) {
  if (llvm::Error error = requireAtLeast(policy.presentPressureInitial, 1,
                                         "present_pressure_initial"))
    return error;
  if (llvm::Error error = requireAtLeast(policy.historyPressureIncrement, 1,
                                         "history_pressure_increment"))
    return error;
  if (llvm::Error error = requireAtLeast(policy.growthDenominator, 1,
                                         "present_pressure_growth_denominator"))
    return error;
  if (policy.growthNumerator < policy.growthDenominator)
    return invalidPolicy("present_pressure growth ratio ",
                         policy.growthNumerator, "/", policy.growthDenominator,
                         " is below one; negotiation pressure must not decay");
  if (std::gcd(policy.growthNumerator, policy.growthDenominator) != 1)
    return invalidPolicy("present_pressure growth ratio ",
                         policy.growthNumerator, "/", policy.growthDenominator,
                         " is not in canonical reduced form");
  return llvm::Error::success();
}

llvm::Error
loom::pnr::validateDualDirectionKernel(const DualDirectionKernel &kernel) {
  const auto *momentum = std::get_if<MomentumDeflectedDirection>(&kernel);
  if (momentum == nullptr)
    return llvm::Error::success();
  if (momentum->betaDenominator == 0)
    return invalidPolicy("momentum beta_denominator must be positive");
  if (momentum->betaNumerator >= momentum->betaDenominator)
    return invalidPolicy("momentum beta ", momentum->betaNumerator, "/",
                         momentum->betaDenominator,
                         " must satisfy 0 <= numerator < denominator");
  return llvm::Error::success();
}

llvm::Error
loom::pnr::validateDualStepSchedule(const DualStepSchedule &schedule) {
  if (const auto *constant = std::get_if<ConstantStepSchedule>(&schedule))
    return requireAtLeast(constant->step, 1, "constant step");

  if (const auto *geometric =
          std::get_if<GeometricDecayStepSchedule>(&schedule)) {
    if (llvm::Error error =
            requireAtLeast(geometric->minimumStep, 1, "geometric minimum_step"))
      return error;
    // An initial step at or below the floor never decays, so the schedule is
    // Constant and the resolver must have encoded it as such.
    if (geometric->initialStep <= geometric->minimumStep)
      return invalidPolicy("geometric initial_step ", geometric->initialStep,
                           " must exceed minimum_step ", geometric->minimumStep,
                           "; a nondecaying schedule is Constant");
    if (geometric->decayNumerator == 0)
      return invalidPolicy("geometric decay_numerator must be positive");
    if (geometric->decayNumerator >= geometric->decayDenominator)
      return invalidPolicy("geometric decay ratio ", geometric->decayNumerator,
                           "/", geometric->decayDenominator,
                           " must be strictly between zero and one");
    if (std::gcd(geometric->decayNumerator, geometric->decayDenominator) != 1)
      return invalidPolicy("geometric decay ratio ", geometric->decayNumerator,
                           "/", geometric->decayDenominator,
                           " is not in canonical reduced form");
    return llvm::Error::success();
  }

  const auto &harmonic = std::get<HarmonicDecayStepSchedule>(schedule);
  if (llvm::Error error =
          requireAtLeast(harmonic.numerator, 1, "harmonic numerator"))
    return error;
  if (llvm::Error error = requireAtLeast(harmonic.offset, 1, "harmonic offset"))
    return error;
  if (llvm::Error error =
          requireAtLeast(harmonic.minimumStep, 1, "harmonic minimum_step"))
    return error;
  // The harmonic term is maximal at iteration zero and only decreases, so a
  // first term that already sits at the floor is Constant at minimum_step.
  auto initialTerm = scaleMagnitude(harmonic.numerator, 1, harmonic.offset,
                                    "harmonic decay step");
  if (!initialTerm)
    return initialTerm.takeError();
  if (*initialTerm <= harmonic.minimumStep)
    return invalidPolicy("harmonic numerator/offset ", harmonic.numerator, "/",
                         harmonic.offset, " never exceeds minimum_step ",
                         harmonic.minimumStep,
                         "; a schedule pinned to its floor is Constant");
  return llvm::Error::success();
}

llvm::Expected<std::int64_t>
loom::pnr::scaleTowardZero(std::int64_t value, std::uint64_t numerator,
                           std::uint64_t denominator) {
  const bool negative = value < 0;
  const std::uint64_t magnitude =
      negative ? std::uint64_t{0} - static_cast<std::uint64_t>(value)
               : static_cast<std::uint64_t>(value);
  const std::uint64_t limit =
      negative ? int64MinMagnitude
               : static_cast<std::uint64_t>(
                     std::numeric_limits<std::int64_t>::max());
  auto scaled = scaleCore(negative, magnitude, numerator, denominator, limit,
                          "scale_toward_zero");
  if (!scaled)
    return scaled.takeError();
  if (!scaled->negative)
    return static_cast<std::int64_t>(scaled->magnitude);
  if (scaled->magnitude == int64MinMagnitude)
    return std::numeric_limits<std::int64_t>::min();
  return -static_cast<std::int64_t>(scaled->magnitude);
}

llvm::Expected<std::uint64_t> loom::pnr::ceilMulDiv(std::uint64_t value,
                                                    std::uint64_t numerator,
                                                    std::uint64_t denominator) {
  if (denominator == 0)
    return invalidPolicy("ceil_mul_div: denominator must be positive");
  const unsigned __int128 product =
      static_cast<unsigned __int128>(value) * numerator;
  const unsigned __int128 quotient = product / denominator;
  const unsigned __int128 result =
      quotient + (product % denominator != 0 ? 1 : 0);
  if (result > std::numeric_limits<std::uint64_t>::max())
    return arithmeticOverflow("ceil_mul_div: ", value, " * ", numerator, " / ",
                              denominator, " is not representable in uint64_t");
  return static_cast<std::uint64_t>(result);
}

llvm::Expected<RouteCost> loom::pnr::pathFinderResourceCost(
    PathFinderPriceKernel kernel, std::uint64_t claim, std::uint64_t usage,
    std::uint64_t capacity, std::uint64_t presentPressure,
    std::uint64_t historyPressure) {
  if (llvm::Error error = requireAtLeast(claim, 1, "normalized claim"))
    return std::move(error);
  if (llvm::Error error =
          requireAtLeast(presentPressure, 1, "present_pressure"))
    return std::move(error);
  auto projected = checkedAdd(usage, claim, "PathFinder projected usage");
  if (!projected)
    return projected.takeError();
  const std::uint64_t excess =
      *projected > capacity ? *projected - capacity : 0;
  auto pressureProduct = checkedMultiply(presentPressure, excess,
                                         "PathFinder present-pressure product");
  if (!pressureProduct)
    return pressureProduct.takeError();

  switch (kernel) {
  case PathFinderPriceKernel::Multiplicative: {
    auto pressureFactor =
        checkedAdd(*pressureProduct, 1, "PathFinder present-pressure factor");
    if (!pressureFactor)
      return pressureFactor.takeError();
    auto historyFactor =
        checkedAdd(historyPressure, 1, "PathFinder history factor");
    if (!historyFactor)
      return historyFactor.takeError();
    auto claimed = checkedMultiply(claim, *pressureFactor,
                                   "PathFinder multiplicative claim product");
    if (!claimed)
      return claimed.takeError();
    return narrowFiniteCost(static_cast<unsigned __int128>(*claimed) *
                                *historyFactor,
                            "PathFinder multiplicative cost");
  }
  case PathFinderPriceKernel::Additive: {
    auto historyProduct = checkedMultiply(
        claim, historyPressure, "PathFinder additive history product");
    if (!historyProduct)
      return historyProduct.takeError();
    auto withPressure =
        checkedAdd(claim, *pressureProduct, "PathFinder additive pressure sum");
    if (!withPressure)
      return withPressure.takeError();
    return narrowFiniteCost(static_cast<unsigned __int128>(*withPressure) +
                                *historyProduct,
                            "PathFinder additive cost");
  }
  }
  llvm_unreachable("invalid PathFinder price kernel");
}

llvm::Expected<RouteCost> loom::pnr::accumulateRouteCost(RouteCost accumulated,
                                                         RouteCost term) {
  return narrowFiniteCost(static_cast<unsigned __int128>(accumulated) + term,
                          "route cost accumulation");
}

llvm::Expected<std::uint64_t>
loom::pnr::pathFinderHistoryUpdate(std::uint64_t history,
                                   std::uint64_t historyPressureIncrement,
                                   std::uint64_t overuse) {
  if (llvm::Error error = requireAtLeast(historyPressureIncrement, 1,
                                         "history_pressure_increment"))
    return std::move(error);
  auto increment = checkedMultiply(historyPressureIncrement, overuse,
                                   "PathFinder history increment product");
  if (!increment)
    return increment.takeError();
  return checkedAdd(history, *increment, "PathFinder history update");
}

llvm::Expected<RouteCost> loom::pnr::dualArcResourceCost(std::uint64_t claim,
                                                         DualPrice price) {
  if (llvm::Error error = requireAtLeast(claim, 1, "normalized claim"))
    return std::move(error);
  // The claim factor (1 + price) and the product are formed in the 128-bit
  // intermediate, so the maximum price is priced exactly rather than wrapped.
  return narrowFiniteCost(static_cast<unsigned __int128>(claim) *
                              (static_cast<unsigned __int128>(price) + 1),
                          "dual arc resource cost");
}

llvm::Expected<DualDirection>
loom::pnr::dualResidual(std::uint64_t aggregatedUsage,
                        std::uint64_t effectiveCapacity) {
  if (aggregatedUsage >= effectiveCapacity) {
    const std::uint64_t magnitude = aggregatedUsage - effectiveCapacity;
    if (magnitude >
        static_cast<std::uint64_t>(std::numeric_limits<DualDirection>::max()))
      return arithmeticOverflow("dual residual: ", aggregatedUsage, " - ",
                                effectiveCapacity,
                                " is not representable in int64_t");
    return static_cast<DualDirection>(magnitude);
  }
  const std::uint64_t magnitude = effectiveCapacity - aggregatedUsage;
  if (magnitude > int64MinMagnitude)
    return arithmeticOverflow("dual residual: ", aggregatedUsage, " - ",
                              effectiveCapacity,
                              " is not representable in int64_t");
  if (magnitude == int64MinMagnitude)
    return std::numeric_limits<DualDirection>::min();
  return -static_cast<DualDirection>(magnitude);
}

llvm::Expected<DualDirection>
loom::pnr::dualDirectionFromResidual(const DualDirectionKernel &kernel,
                                     DualDirection residual,
                                     DualDirection previousDirection) {
  if (llvm::Error error = validateDualDirectionKernel(kernel))
    return std::move(error);
  const auto *momentum = std::get_if<MomentumDeflectedDirection>(&kernel);
  // The prior direction is MomentumDeflected session state. Under the other
  // kernels it is an inactive field, so only zero encodes their absent state.
  if (momentum == nullptr && previousDirection != 0)
    return invalidPolicy("previous_direction ", previousDirection,
                         " is inactive outside MomentumDeflected");
  if (std::holds_alternative<ProjectedSignedDirection>(kernel))
    return residual;
  if (std::holds_alternative<PositiveViolationOnlyDirection>(kernel))
    return std::max<DualDirection>(residual, 0);
  auto deflection = scaleTowardZero(previousDirection, momentum->betaNumerator,
                                    momentum->betaDenominator);
  if (!deflection)
    return deflection.takeError();
  return checkedAddSigned(residual, *deflection,
                          "momentum-deflected direction");
}

llvm::Expected<DualStep> loom::pnr::dualStepAt(const DualStepSchedule &schedule,
                                               std::uint64_t iteration) {
  if (llvm::Error error = validateDualStepSchedule(schedule))
    return std::move(error);
  if (const auto *constant = std::get_if<ConstantStepSchedule>(&schedule))
    return constant->step;

  if (const auto *geometric =
          std::get_if<GeometricDecayStepSchedule>(&schedule)) {
    DualStep step = geometric->initialStep;
    for (std::uint64_t k = 0; k < iteration; ++k) {
      auto scaled =
          scaleMagnitude(step, geometric->decayNumerator,
                         geometric->decayDenominator, "geometric decay step");
      if (!scaled)
        return scaled.takeError();
      // The validated decay ratio is below one, so each scaling strictly
      // decreases the step until it clamps at the floor and stays there.
      const DualStep next = std::max(geometric->minimumStep, *scaled);
      if (next == step)
        break;
      step = next;
    }
    return step;
  }

  const auto &harmonic = std::get<HarmonicDecayStepSchedule>(schedule);
  auto divisor =
      checkedAdd(harmonic.offset, iteration, "harmonic decay divisor");
  if (!divisor)
    return divisor.takeError();
  auto scaled =
      scaleMagnitude(harmonic.numerator, 1, *divisor, "harmonic decay step");
  if (!scaled)
    return scaled.takeError();
  return std::max(harmonic.minimumStep, *scaled);
}

llvm::Expected<DualPrice> loom::pnr::dualPriceUpdate(DualPrice price,
                                                     DualStep step,
                                                     DualDirection direction) {
  if (llvm::Error error = requireAtLeast(step, 1, "dual step"))
    return std::move(error);
  // The 128-bit product is exact by construction for 64-bit operands; the
  // projected sum is clamped at zero and must be representable as a price.
  const __int128 product =
      static_cast<__int128>(step) * static_cast<__int128>(direction);
  const __int128 updated = static_cast<__int128>(price) + product;
  if (updated <= 0)
    return DualPrice{0};
  if (updated > std::numeric_limits<DualPrice>::max())
    return arithmeticOverflow("dual price update: ", price, " + ", step, " * ",
                              direction, " is not representable in uint64_t");
  return static_cast<DualPrice>(updated);
}
