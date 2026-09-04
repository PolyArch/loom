#include "Fabric/IR/PhysicalTag.h"
#include "Fabric/Identity/FabricTemporalSwitchRoute.h"

#include "../TestAllocationProbe.h"

#include "llvm/ADT/APInt.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <array>
#include <cstdlib>
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

void requireSuccess(llvm::Error error) {
  if (error)
    fail(llvm::toString(std::move(error)));
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
  require(::fabric::comparePhysicalTagValues(rows[0].tag, tag(1)) == 0 &&
              rows[0].demandOrdinals == std::vector<std::uint64_t>{3},
          "exact rows are not ordered by unsigned tag");
  require(::fabric::comparePhysicalTagValues(rows[1].tag, tag(5)) == 0 &&
              rows[1].compatible &&
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
  const llvm::APInt assignedLowTag = tag(0);
  const llvm::APInt assignedHighTag = tag(1);
  const std::array<FabricTemporalSwitchCandidateRouteDemandView, 4> demands = {
      FabricTemporalSwitchCandidateRouteDemandView{{assignedLow},
                                                   &assignedLowTag},
      FabricTemporalSwitchCandidateRouteDemandView{{assignedHigh},
                                                   &assignedHighTag},
      FabricTemporalSwitchCandidateRouteDemandView{{joinsHigh}, nullptr},
      FabricTemporalSwitchCandidateRouteDemandView{{joinsNeither}, nullptr}};
  const auto rows =
      take(projectFabricTemporalSwitchCandidateRouteRows(demands));
  require(rows.size() == 3, "candidate projection changed its row lower bound");
  require(rows[0].tag &&
              ::fabric::comparePhysicalTagValues(*rows[0].tag, tag(0)) == 0 &&
              rows[0].demandOrdinals == std::vector<std::uint64_t>{0},
          "candidate projection changed the lower assigned row");
  require(rows[1].tag &&
              ::fabric::comparePhysicalTagValues(*rows[1].tag, tag(1)) == 0 &&
              rows[1].demandOrdinals ==
                  std::vector<std::uint64_t>({1, 2}),
          "unassigned demand did not join the first compatible assigned row");
  require(!rows[2].tag && rows[2].compatible &&
              rows[2].demandOrdinals == std::vector<std::uint64_t>{3},
          "incompatible unassigned demand did not open a provisional row");
}

void wideTagCandidateProjectionReusesScratch() {
  const FabricSwitchOccurrenceRef occurrence(17);
  const std::array<FabricOrdinal, 1> output0 = {0};
  const std::array<FabricOrdinal, 1> output1 = {1};
  const std::array<FabricTemporalSwitchRouteSignatureView, 1> first = {
      FabricTemporalSwitchRouteSignatureView{occurrence, 0, output0}};
  const std::array<FabricTemporalSwitchRouteSignatureView, 1> second = {
      FabricTemporalSwitchRouteSignatureView{occurrence, 1, output1}};
  const std::array<FabricTemporalSwitchRouteSignatureView, 1> third = {
      FabricTemporalSwitchRouteSignatureView{occurrence, 2, output0}};
  llvm::APInt lower(129, 0);
  lower.setBit(100);
  llvm::APInt equalWithWiderStorage(193, 0);
  equalWithWiderStorage.setBit(100);
  llvm::APInt higher(193, 0);
  higher.setBit(101);
  const std::array<FabricTemporalSwitchCandidateRouteDemandView, 3> demands = {
      FabricTemporalSwitchCandidateRouteDemandView{{first}, &lower},
      FabricTemporalSwitchCandidateRouteDemandView{{second},
                                                   &equalWithWiderStorage},
      FabricTemporalSwitchCandidateRouteDemandView{{third}, &higher}};
  const std::array<FabricTemporalSwitchCandidateRouteDemandView, 3> reordered =
      {FabricTemporalSwitchCandidateRouteDemandView{{second},
                                                    &equalWithWiderStorage},
       FabricTemporalSwitchCandidateRouteDemandView{{first}, &lower},
       FabricTemporalSwitchCandidateRouteDemandView{{third}, &higher}};

  const auto materialized =
      take(projectFabricTemporalSwitchCandidateRouteRows(demands));
  const auto reorderedMaterialized =
      take(projectFabricTemporalSwitchCandidateRouteRows(reordered));
  const llvm::APInt canonicalLower = ::fabric::canonicalPhysicalTagValue(lower);
  require(materialized.size() == 2 && reorderedMaterialized.size() == 2 &&
              materialized[0].tag == canonicalLower &&
              reorderedMaterialized[0].tag == canonicalLower &&
              materialized[0].tag->getBitWidth() == 101 &&
              reorderedMaterialized[0].tag->getBitWidth() == 101,
          "equal numeric tags retained input-order-dependent storage width");

  FabricTemporalSwitchCandidateRouteProjectionScratch scratch;
  scratch.prepare(demands.size());
  FabricTemporalSwitchRouteRowMemberSpans spans;
  spans.rowOffsets.reserve(demands.size() + 1);
  spans.demandOrdinals.reserve(demands.size());
  const std::size_t retainedBytes =
      scratch.retainedStorageBytes() +
      spans.rowOffsets.capacity() * sizeof(std::uint64_t) +
      spans.demandOrdinals.capacity() * sizeof(std::uint64_t);
  require(loom::test::allocationProbeIsCalibrated(),
          "allocation probe is not calibrated");
  loom::test::startAllocationProbe();
  requireSuccess(projectFabricTemporalSwitchCandidateRouteRowMemberSpans(
      demands, spans, scratch));
  const std::size_t allocations = loom::test::stopAllocationProbe();
  require(spans.rowOffsets == std::vector<std::uint64_t>({0, 2, 3}) &&
              spans.demandOrdinals == std::vector<std::uint64_t>({0, 1, 2}),
          "wide numeric tag identity changed candidate row membership");
  require(allocations == 0 &&
              scratch.retainedStorageBytes() +
                      spans.rowOffsets.capacity() * sizeof(std::uint64_t) +
                      spans.demandOrdinals.capacity() * sizeof(std::uint64_t) ==
                  retainedBytes,
          "first prepared wide-tag candidate projection allocated storage");
}

} // namespace

int main() {
  exactRowsAreTagKeyed();
  incompatibleEqualTagRemainsObservable();
  candidateRowsPreserveAssignedIdentity();
  wideTagCandidateProjectionReusesScratch();
  llvm::outs() << "Temporal switch route tests passed\n";
  return 0;
}
