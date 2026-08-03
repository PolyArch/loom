#include "PnR/RoutingNegotiation.h"

#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <array>
#include <cassert>
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

// The single checked-math path. Kernels use fixed-width widened intermediates
// and narrow through these helpers; nothing wraps, saturates, or changes
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

// Exact ceil(lhs * middle * rhs / Q^2) without staged rounding. Q^2 is 2^64,
// so the quotient and remainder are recovered directly from fixed-width limbs;
// no division, heap allocation, or widened persistent representation enters
// the routing hot path.
llvm::Expected<RouteCost>
scaledRouteTripleProduct64(RouteCost lhs, RouteCost middle, RouteCost rhs,
                           llvm::StringRef operation) {
  static_assert(routeCostScale == (std::uint64_t{1} << 32));
  if (lhs == 0 || middle == 0 || rhs == 0)
    return RouteCost{0};

  const unsigned __int128 first = static_cast<unsigned __int128>(lhs) * middle;
  const std::uint64_t firstLow = static_cast<std::uint64_t>(first);
  const std::uint64_t firstHigh = static_cast<std::uint64_t>(first >> 64);
  const unsigned __int128 lowProduct =
      static_cast<unsigned __int128>(firstLow) * rhs;
  const unsigned __int128 quotient =
      static_cast<unsigned __int128>(firstHigh) * rhs + (lowProduct >> 64) +
      (static_cast<std::uint64_t>(lowProduct) != 0 ? 1 : 0);
  return narrowFiniteCost(quotient, operation);
}

template <std::size_t LhsSize, std::size_t RhsSize>
std::array<std::uint64_t, LhsSize + RhsSize>
multiplyLimbs(const std::array<std::uint64_t, LhsSize> &lhs,
              const std::array<std::uint64_t, RhsSize> &rhs) {
  std::array<std::uint64_t, LhsSize + RhsSize> result{};
  for (std::size_t lhsIndex = 0; lhsIndex < LhsSize; ++lhsIndex) {
    for (std::size_t rhsIndex = 0; rhsIndex < RhsSize; ++rhsIndex) {
      const unsigned __int128 product =
          static_cast<unsigned __int128>(lhs[lhsIndex]) * rhs[rhsIndex];
      std::size_t resultIndex = lhsIndex + rhsIndex;
      const unsigned __int128 lowSum =
          static_cast<unsigned __int128>(result[resultIndex]) +
          static_cast<std::uint64_t>(product);
      result[resultIndex++] = static_cast<std::uint64_t>(lowSum);
      unsigned __int128 carry = (product >> 64) + (lowSum >> 64);
      while (carry != 0) {
        assert(resultIndex < result.size());
        const unsigned __int128 next =
            static_cast<unsigned __int128>(result[resultIndex]) + carry;
        result[resultIndex++] = static_cast<std::uint64_t>(next);
        carry = next >> 64;
      }
    }
  }
  return result;
}

llvm::Expected<RouteCost> scaledRouteTripleProduct(RouteCost lhs,
                                                   unsigned __int128 middle,
                                                   unsigned __int128 rhs,
                                                   llvm::StringRef operation) {
  if (lhs == 0 || middle == 0 || rhs == 0)
    return RouteCost{0};
  if ((middle >> 64) == 0 && (rhs >> 64) == 0)
    return scaledRouteTripleProduct64(lhs, static_cast<std::uint64_t>(middle),
                                      static_cast<std::uint64_t>(rhs),
                                      operation);

  const std::array<std::uint64_t, 1> lhsLimbs = {lhs};
  const std::array<std::uint64_t, 2> middleLimbs = {
      static_cast<std::uint64_t>(middle),
      static_cast<std::uint64_t>(middle >> 64)};
  const std::array<std::uint64_t, 2> rhsLimbs = {
      static_cast<std::uint64_t>(rhs), static_cast<std::uint64_t>(rhs >> 64)};
  const auto first = multiplyLimbs(lhsLimbs, middleLimbs);
  auto product = multiplyLimbs(first, rhsLimbs);

  if (product[0] != 0) {
    std::size_t index = 1;
    while (index < product.size() && ++product[index] == 0)
      ++index;
    if (index == product.size())
      return arithmeticOverflow(operation,
                                ": rounded quotient exceeds fixed limbs");
  }
  for (std::size_t index = 2; index < product.size(); ++index)
    if (product[index] != 0)
      return arithmeticOverflow(operation, ": exceeds the largest finite cost ",
                                maxFiniteRouteCost);
  return narrowFiniteCost(product[1], operation);
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

llvm::Expected<RouteCost>
loom::pnr::normalizedRouteClaimCost(std::uint64_t amount,
                                    std::uint64_t capacity) {
  if (amount == 0)
    return 0;
  if (capacity == 0)
    return invalidPolicy(
        "normalized route claim has positive amount and zero capacity");
  auto result = ceilMulDiv(amount, routeCostScale, capacity);
  if (!result)
    return result.takeError();
  return narrowFiniteCost(*result, "normalized route claim cost");
}

llvm::Expected<RouteCost> loom::pnr::normalizedRouteOveruseCost(
    std::uint64_t usageBefore, std::uint64_t amount, std::uint64_t capacity) {
  auto projected =
      checkedAdd(usageBefore, amount, "PathFinder projected raw usage");
  if (!projected)
    return projected.takeError();
  const std::uint64_t excess =
      *projected > capacity ? *projected - capacity : 0;
  if (excess == 0)
    return 0;
  if (capacity == 0)
    return invalidPolicy(
        "normalized route overuse has positive amount and zero capacity");
  auto result = ceilMulDiv(excess, routeCostScale, capacity);
  if (!result)
    return result.takeError();
  return narrowFiniteCost(*result, "normalized route overuse cost");
}

llvm::Expected<RouteCost> loom::pnr::scaledRouteProduct(RouteCost lhs,
                                                        RouteCost rhs) {
  const unsigned __int128 product = static_cast<unsigned __int128>(lhs) * rhs;
  const unsigned __int128 quotient = product / routeCostScale;
  const unsigned __int128 result =
      quotient + (product % routeCostScale != 0 ? 1 : 0);
  return narrowFiniteCost(result, "scaled route product");
}

llvm::Expected<RouteCost> loom::pnr::pathFinderResourceCost(
    ResolvedPathFinderPriceKernel kernel, RouteCost qCost, RouteCost xCost,
    std::uint64_t presentPressure, std::uint64_t historyPressure) {
  if (llvm::Error error = requireAtLeast(qCost, 1, "normalized claim cost"))
    return std::move(error);
  if (llvm::Error error =
          requireAtLeast(presentPressure, 1, "present_pressure"))
    return std::move(error);

  switch (kernel) {
  case ResolvedPathFinderPriceKernel::Multiplicative: {
    const unsigned __int128 pressureFactor =
        static_cast<unsigned __int128>(presentPressure) * xCost +
        routeCostScale;
    const unsigned __int128 historyFactor =
        static_cast<unsigned __int128>(historyPressure) + routeCostScale;
    return scaledRouteTripleProduct(qCost, pressureFactor, historyFactor,
                                    "PathFinder multiplicative cost");
  }
  case ResolvedPathFinderPriceKernel::Additive: {
    auto pressureProduct = checkedMultiply(
        presentPressure, xCost, "PathFinder present-pressure product");
    if (!pressureProduct)
      return pressureProduct.takeError();
    auto historyTerm = scaledRouteProduct(qCost, historyPressure);
    if (!historyTerm)
      return historyTerm.takeError();
    return narrowFiniteCost(static_cast<unsigned __int128>(qCost) +
                                *pressureProduct + *historyTerm,
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
  auto priceTerm = scaledRouteProduct(claim, price);
  if (!priceTerm)
    return priceTerm.takeError();
  return narrowFiniteCost(static_cast<unsigned __int128>(claim) + *priceTerm,
                          "dual arc resource cost");
}

llvm::Expected<DualDirection>
loom::pnr::dualResidual(std::uint64_t aggregatedUsage,
                        std::uint64_t effectiveCapacity) {
  if (aggregatedUsage == effectiveCapacity)
    return DualDirection{0};
  if (effectiveCapacity == 0)
    return invalidPolicy(
        "dual residual has positive usage and zero effective capacity");

  if (aggregatedUsage >= effectiveCapacity) {
    const std::uint64_t magnitude = aggregatedUsage - effectiveCapacity;
    auto normalized = ceilMulDiv(magnitude, routeCostScale, effectiveCapacity);
    if (!normalized)
      return normalized.takeError();
    if (*normalized >
        static_cast<std::uint64_t>(std::numeric_limits<DualDirection>::max()))
      return arithmeticOverflow("dual residual: ", aggregatedUsage, " - ",
                                effectiveCapacity,
                                " normalized by Q is not "
                                "representable in int64_t");
    return static_cast<DualDirection>(*normalized);
  }
  const std::uint64_t magnitude = effectiveCapacity - aggregatedUsage;
  auto normalized = ceilMulDiv(magnitude, routeCostScale, effectiveCapacity);
  if (!normalized)
    return normalized.takeError();
  if (*normalized > int64MinMagnitude)
    return arithmeticOverflow(
        "dual residual: ", aggregatedUsage, " - ", effectiveCapacity,
        " normalized by Q is not representable in int64_t");
  if (*normalized == int64MinMagnitude)
    return std::numeric_limits<DualDirection>::min();
  return -static_cast<DualDirection>(*normalized);
}

llvm::Expected<DualDirection> loom::pnr::dualDirectionFromResidual(
    const ResolvedDualSubgradientPolicy &policy, DualDirection residual,
    DualDirection previousDirection) {
  if (llvm::Error error = validateResolvedDualSubgradientPolicy(policy))
    return std::move(error);
  // The prior direction is MomentumDeflected session state. Under the other
  // kernels it is an inactive field, so only zero encodes their absent state.
  if (policy.directionKernel !=
          ResolvedDualDirectionKernel::MomentumDeflected &&
      previousDirection != 0)
    return invalidPolicy("previous_direction ", previousDirection,
                         " is inactive outside MomentumDeflected");
  if (policy.directionKernel == ResolvedDualDirectionKernel::ProjectedSigned)
    return residual;
  if (policy.directionKernel ==
      ResolvedDualDirectionKernel::PositiveViolationOnly)
    return std::max<DualDirection>(residual, 0);
  auto deflection =
      scaleTowardZero(previousDirection, policy.momentum->numerator,
                      policy.momentum->denominator);
  if (!deflection)
    return deflection.takeError();
  return checkedAddSigned(residual, *deflection,
                          "momentum-deflected direction");
}

llvm::Expected<DualStep>
loom::pnr::dualStepAt(const ResolvedDualStepSchedule &schedule,
                      std::uint64_t iteration) {
  if (llvm::Error error = validateResolvedDualStepSchedule(schedule))
    return std::move(error);
  if (schedule.kind == ResolvedDualStepScheduleKind::Constant)
    return schedule.first;

  if (schedule.kind == ResolvedDualStepScheduleKind::GeometricDecay) {
    DualStep step = schedule.first;
    for (std::uint64_t k = 0; k < iteration; ++k) {
      auto scaled = scaleMagnitude(step, schedule.third, schedule.fourth,
                                   "geometric decay step");
      if (!scaled)
        return scaled.takeError();
      // The validated decay ratio is below one, so each scaling strictly
      // decreases the step until it clamps at the floor and stays there.
      const DualStep next = std::max(schedule.second, *scaled);
      if (next == step)
        break;
      step = next;
    }
    return step;
  }

  auto divisor =
      checkedAdd(schedule.second, iteration, "harmonic decay divisor");
  if (!divisor)
    return divisor.takeError();
  auto scaled =
      scaleMagnitude(schedule.first, 1, *divisor, "harmonic decay step");
  if (!scaled)
    return scaled.takeError();
  return std::max(schedule.third, *scaled);
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
