#include "SpatialPhysicalTiming.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <array>
#include <cstdlib>
#include <string>

namespace {

[[noreturn]] void fail(const std::string &message) {
  llvm::errs() << "Spatial physical timing test: " << message << '\n';
  std::exit(EXIT_FAILURE);
}

template <typename T> T take(llvm::Expected<T> value) {
  if (!value)
    fail(llvm::toString(value.takeError()));
  return *value;
}

void registeredDestinationCutsArrival() {
  using loom::fabric::FabricPhysicalTimingBoundaryKind;
  loom::pnr::detail::SpatialLogicalNetPhysicalTiming timing;
  std::uint64_t arrival = take(loom::pnr::detail::advanceSpatialPhysicalTiming(
      6, FabricPhysicalTimingBoundaryKind::Combinational, 0, 8, timing));
  if (arrival != 6 || timing.worstArrivalDelayQuanta != 0 ||
      timing.totalNegativeSlackQuanta != 0)
    fail("combinational traversal observed a premature endpoint");

  arrival = take(loom::pnr::detail::advanceSpatialPhysicalTiming(
      5, FabricPhysicalTimingBoundaryKind::RegisteredDestination, arrival, 8,
      timing));
  if (arrival != 0 || timing.worstArrivalDelayQuanta != 11 ||
      timing.totalNegativeSlackQuanta != 3)
    fail("registered destination did not close and reset its segment");

  arrival = take(loom::pnr::detail::advanceSpatialPhysicalTiming(
      7, FabricPhysicalTimingBoundaryKind::Combinational, arrival, 8, timing));
  if (llvm::Error error =
          loom::pnr::detail::observeSpatialPhysicalTimingEndpoint(arrival, 8,
                                                                  timing))
    fail(llvm::toString(std::move(error)));
  if (timing.worstArrivalDelayQuanta != 11 ||
      timing.totalNegativeSlackQuanta != 3)
    fail("post-register segment inherited the prior arrival");
}

void criticalityWeightsProviderDelay() {
  const loom::pnr::RouteCost base =
      take(loom::pnr::detail::physicalTimingDrivenTraversalCost(3, 8, 0));
  const loom::pnr::RouteCost critical =
      take(loom::pnr::detail::physicalTimingDrivenTraversalCost(3, 8, 4));
  if (base == 0 || critical != base * 5)
    fail("structural criticality did not weight provider traversal delay");
}

loom::pnr::RouteCost routeCost(const std::array<std::uint64_t, 2> &delays) {
  loom::pnr::RouteCost total = 0;
  for (std::uint64_t delay : delays) {
    const loom::pnr::RouteCost term =
        take(loom::pnr::detail::physicalTimingDrivenTraversalCost(delay, 8, 2));
    auto next = loom::pnr::accumulateRouteCost(total, term);
    if (!next)
      fail(llvm::toString(next.takeError()));
    total = *next;
  }
  return total;
}

void providerProfileChangesRoutePreference() {
  const std::array<std::uint64_t, 2> leftBaseline{{2, 2}};
  const std::array<std::uint64_t, 2> leftCharacterized{{4, 4}};
  const std::array<std::uint64_t, 2> right{{1, 5}};
  const loom::pnr::RouteCost characterized = routeCost(leftCharacterized);
  if (!(routeCost(leftBaseline) < routeCost(right)) ||
      !(characterized > routeCost(right)) ||
      characterized != routeCost(leftCharacterized))
    fail("provider delay change did not replay a changed route preference");
}

} // namespace

int main() {
  registeredDestinationCutsArrival();
  criticalityWeightsProviderDelay();
  providerProfileChangesRoutePreference();
  return EXIT_SUCCESS;
}
