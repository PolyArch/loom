#ifndef LOOM_PNR_ROUTINGNEGOTIATION_H
#define LOOM_PNR_ROUTINGNEGOTIATION_H

#include "Common/ResolvedPnrPolicy.h"

#include "llvm/Support/Error.h"

#include <cstdint>
#include <limits>
#include <string>
#include <system_error>

namespace loom::pnr {

// Checked numeric protocol shared by both negotiation algorithms. These are
// the sole representations; there is no fixed-point, rational, or
// floating-point variant.
using RouteCost = std::uint64_t;
using DualPrice = std::uint64_t;
using DualDirection = std::int64_t;
using DualStep = std::uint64_t;

// The maximum RouteCost is reserved exclusively as the A* infinity sentinel,
// so every finite cost is strictly smaller. Kernels report a mathematical
// result of routeCostInfinity as an arithmetic overflow rather than emitting a
// finite cost indistinguishable from an unreachable arc.
constexpr RouteCost routeCostInfinity = std::numeric_limits<RouteCost>::max();
constexpr RouteCost maxFiniteRouteCost = routeCostInfinity - 1;

class RoutingNegotiationError final
    : public llvm::ErrorInfo<RoutingNegotiationError> {
public:
  enum class Kind {
    InvalidPolicy,
    ArithmeticOverflow,
  };

  static char ID;

  RoutingNegotiationError(Kind kind, std::string message);

  Kind kind() const { return kind_; }

  void log(llvm::raw_ostream &stream) const override;
  std::error_code convertToErrorCode() const override;

private:
  Kind kind_;
  std::string message_;
};

// Sole rounding authority for rational scaling: the exact
// trunc-toward-zero quotient of value * numerator / denominator, formed in
// a checked widened intermediate. The denominator must be positive.
llvm::Expected<std::int64_t> scaleTowardZero(std::int64_t value,
                                             std::uint64_t numerator,
                                             std::uint64_t denominator);

// Exact ceil(value * numerator / denominator) over nonnegative integers.
// The product is formed in a checked widened intermediate before division;
// this is the PathFinder present-pressure update protocol.
llvm::Expected<std::uint64_t> ceilMulDiv(std::uint64_t value,
                                         std::uint64_t numerator,
                                         std::uint64_t denominator);

// PathFinder cost of one normalized claim against the frozen snapshot:
//   X = max(0, usage + claim - capacity)
//   Multiplicative: claim * (1 + presentPressure * X) * (1 + historyPressure)
//   Additive:       claim + presentPressure * X + claim * historyPressure
// The kernel prices one claimed resource, so the claim must be positive, and
// present pressure is active in both price kernels, so it must be positive
// too. History pressure starts at zero and may legitimately be zero.
llvm::Expected<RouteCost>
pathFinderResourceCost(ResolvedPathFinderPriceKernel kernel,
                       std::uint64_t claim, std::uint64_t usage,
                       std::uint64_t capacity, std::uint64_t presentPressure,
                       std::uint64_t historyPressure);

// Checked arc-cost accumulation over the per-resource costs of one traversal.
// Callers fold the complete arc locally and publish the result atomically; a
// pure structural traversal claims no resource and is the empty fold, so it
// costs zero without entering a resource kernel.
llvm::Expected<RouteCost> accumulateRouteCost(RouteCost accumulated,
                                              RouteCost term);

// Proportional history update: history + increment * overuse. The increment is
// an active policy value and must be positive; zero overuse is the ordinary
// uncongested case and leaves history unchanged.
llvm::Expected<std::uint64_t>
pathFinderHistoryUpdate(std::uint64_t history,
                        std::uint64_t historyPressureIncrement,
                        std::uint64_t overuse);

// Dual fixed-price cost of one normalized claim: claim * (1 + price). Both
// algorithms price the same normalized claim, so it must be positive here too.
llvm::Expected<RouteCost> dualArcResourceCost(std::uint64_t claim,
                                              DualPrice price);

// Signed constraint residual: aggregatedUsage - effectiveCapacity.
llvm::Expected<DualDirection> dualResidual(std::uint64_t aggregatedUsage,
                                           std::uint64_t effectiveCapacity);

// Maps a residual to an update direction. previousDirection is session state
// owned by MomentumDeflected alone; it must be zero under the other kernels
// and on the first iteration, where no prior direction exists.
llvm::Expected<DualDirection>
dualDirectionFromResidual(const ResolvedDualSubgradientPolicy &policy,
                          DualDirection residual,
                          DualDirection previousDirection);

// Step produced by the owner-typed schedule at the given zero-based
// iteration.
llvm::Expected<DualStep> dualStepAt(const ResolvedDualStepSchedule &schedule,
                                    std::uint64_t iteration);

// Projected price update: max(0, price + step * direction). Every scheduled
// step is positive; a zero direction is the ordinary balanced residual and
// leaves the price unchanged.
llvm::Expected<DualPrice> dualPriceUpdate(DualPrice price, DualStep step,
                                          DualDirection direction);

} // namespace loom::pnr

#endif // LOOM_PNR_ROUTINGNEGOTIATION_H
