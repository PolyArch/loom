#include "Fabric/Identity/FabricTemporalSwitchRoute.h"

#include "llvm/ADT/APInt.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <array>
#include <cstdlib>
#include <optional>
#include <string>
#include <utility>
#include <vector>

using namespace loom::fabric;

namespace {

[[noreturn]] void fail(const std::string &message) {
  llvm::errs() << "Temporal switch route test: " << message << "\n";
  std::exit(1);
}

void require(bool condition, const std::string &message) {
  if (!condition)
    fail(message);
}

template <typename T> T take(llvm::Expected<T> result) {
  if (!result)
    fail(llvm::toString(result.takeError()));
  return std::move(*result);
}

llvm::APInt tag(std::uint64_t value) { return llvm::APInt(4, value); }

void exactRowsAreTagKeyed() {
  const FabricSwitchOccurrenceRef occurrence(7);
  const std::array<FabricOrdinal, 2> outputs01 = {1, 2};
  const std::array<FabricOrdinal, 1> output2 = {3};
  const std::array<FabricTemporalSwitchRouteSignatureView, 1> first = {
      FabricTemporalSwitchRouteSignatureView{occurrence, 0, outputs01}};
  const std::array<FabricTemporalSwitchRouteSignatureView, 1> repeated = {
      FabricTemporalSwitchRouteSignatureView{occurrence, 0, outputs01}};
  const std::array<FabricTemporalSwitchRouteSignatureView, 1> disjoint = {
      FabricTemporalSwitchRouteSignatureView{occurrence, 1, output2}};

  const std::array<FabricTemporalSwitchTaggedRouteDemandView, 4> demands = {
      FabricTemporalSwitchTaggedRouteDemandView{{first}, tag(5)},
      FabricTemporalSwitchTaggedRouteDemandView{{repeated}, tag(5)},
      FabricTemporalSwitchTaggedRouteDemandView{{disjoint}, tag(5)},
      FabricTemporalSwitchTaggedRouteDemandView{{first}, tag(1)}};
  const auto rows = take(projectFabricTemporalSwitchRouteRows(demands));
  require(rows.size() == 2, "distinct tags did not own distinct rows");
  require(rows[0].tag == tag(1) && rows[0].demandOrdinals ==
                                            std::vector<std::uint64_t>{3},
          "exact rows are not ordered by unsigned tag");
  require(rows[1].tag == tag(5) && rows[1].compatible &&
              rows[1].demandOrdinals ==
                  std::vector<std::uint64_t>({0, 1, 2}),
          "compatible equal-tag demands did not share one row");
}

void incompatibleEqualTagRemainsObservable() {
  const FabricSwitchOccurrenceRef occurrence(11);
  const std::array<FabricOrdinal, 1> output1 = {1};
  const std::array<FabricOrdinal, 1> output2 = {2};
  const std::array<FabricTemporalSwitchRouteSignatureView, 1> first = {
      FabricTemporalSwitchRouteSignatureView{occurrence, 0, output1}};
  const std::array<FabricTemporalSwitchRouteSignatureView, 1> conflicting = {
      FabricTemporalSwitchRouteSignatureView{occurrence, 0, output2}};
  const std::array<FabricTemporalSwitchTaggedRouteDemandView, 2> demands = {
      FabricTemporalSwitchTaggedRouteDemandView{{first}, tag(3)},
      FabricTemporalSwitchTaggedRouteDemandView{{conflicting}, tag(3)}};
  const auto rows = take(projectFabricTemporalSwitchRouteRows(demands));
  require(rows.size() == 1 && !rows.front().compatible &&
              rows.front().demandOrdinals ==
                  std::vector<std::uint64_t>({0, 1}),
          "an incompatible equal-tag collision changed physical row identity");
}

void candidateRowsPreserveAssignedIdentity() {
  const FabricSwitchOccurrenceRef occurrence(13);
  const std::array<FabricOrdinal, 1> output1 = {1};
  const std::array<FabricOrdinal, 1> output2 = {2};
  const std::array<FabricOrdinal, 1> output3 = {3};
  const std::array<FabricOrdinal, 1> output4 = {4};
  const std::array<FabricTemporalSwitchRouteSignatureView, 1> assignedLow = {
      FabricTemporalSwitchRouteSignatureView{occurrence, 0, output1}};
  const std::array<FabricTemporalSwitchRouteSignatureView, 1> assignedHigh = {
      FabricTemporalSwitchRouteSignatureView{occurrence, 1, output2}};
  const std::array<FabricTemporalSwitchRouteSignatureView, 1> joinsHigh = {
      FabricTemporalSwitchRouteSignatureView{occurrence, 0, output3}};
  const std::array<FabricTemporalSwitchRouteSignatureView, 2> joinsNeither = {
      FabricTemporalSwitchRouteSignatureView{occurrence, 0, output3},
      FabricTemporalSwitchRouteSignatureView{occurrence, 1, output4}};
  const std::array<FabricTemporalSwitchCandidateRouteDemandView, 4> demands = {
      FabricTemporalSwitchCandidateRouteDemandView{{assignedLow}, tag(0)},
      FabricTemporalSwitchCandidateRouteDemandView{{assignedHigh}, tag(1)},
      FabricTemporalSwitchCandidateRouteDemandView{{joinsHigh}, std::nullopt},
      FabricTemporalSwitchCandidateRouteDemandView{{joinsNeither},
                                                    std::nullopt}};
  const auto rows =
      take(projectFabricTemporalSwitchCandidateRouteRows(demands));
  require(rows.size() == 3, "candidate projection changed its row lower bound");
  require(rows[0].tag == tag(0) &&
              rows[0].demandOrdinals == std::vector<std::uint64_t>{0},
          "candidate projection changed the lower assigned row");
  require(rows[1].tag == tag(1) &&
              rows[1].demandOrdinals ==
                  std::vector<std::uint64_t>({1, 2}),
          "unassigned demand did not join the first compatible assigned row");
  require(!rows[2].tag && rows[2].compatible &&
              rows[2].demandOrdinals == std::vector<std::uint64_t>{3},
          "incompatible unassigned demand did not open a provisional row");
}

} // namespace

int main() {
  exactRowsAreTagKeyed();
  incompatibleEqualTagRemainsObservable();
  candidateRowsPreserveAssignedIdentity();
  llvm::outs() << "Temporal switch route tests passed\n";
  return 0;
}
