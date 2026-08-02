#include "SpatialPnrTransferIndex.h"

#include "PnR/PnrIndex.h"

#include "llvm/Support/Error.h"

#include <cstddef>
#include <cstdint>

using namespace loom;
using namespace loom::mapping;
using namespace loom::pnr;

namespace {

constexpr llvm::StringLiteral frozenArtifact = "FrozenSpatialPnrProblem";
constexpr PnrCapacityContext netCountContext{
    frozenArtifact, "logical_nets", "logical_nets", PnrCapacityMeasure::Count};
constexpr PnrCapacityContext sinkOffsetContext{frozenArtifact, "logical_nets",
                                               "logical_net_sinks",
                                               PnrCapacityMeasure::Offset};
constexpr PnrCapacityContext sinkCountContext{
    frozenArtifact, "logical_net_sinks", "logical_net_sinks",
    PnrCapacityMeasure::Count};

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::make_error<SpatialPnrFreezeFailure>(
      SpatialPnrFreezeFailureKind::Invalid, message.str());
}

llvm::Expected<PnrIndex> checked(PnrCapacityContext context,
                                 std::size_t value) {
  return checkedPnrIndex(context, static_cast<std::uint64_t>(value));
}

} // namespace

class loom::pnr::FrozenSpatialTransferIndexBuilder final {
public:
  static llvm::Expected<FrozenSpatialTransferIndex>
  build(const TechMappingView &techMapping) {
    FrozenSpatialTransferIndex result;
    const auto nets = techMapping.residualLogicalNets();
    if (llvm::Error error =
            preflightPnrIndexCapacity(netCountContext, nets.size()))
      return std::move(error);
    result.logicalNets_.reserve(nets.size());

    for (const TechResidualLogicalNetView &net : nets) {
      if (net.sinks.empty())
        return invalid("residual logical net has no sink obligation");
      auto sinkOffset =
          checked(sinkOffsetContext, result.logicalNetSinks_.size());
      if (!sinkOffset)
        return sinkOffset.takeError();
      auto sinkEnd = checkedPnrIndexAdd(
          sinkCountContext, result.logicalNetSinks_.size(), net.sinks.size());
      if (!sinkEnd)
        return sinkEnd.takeError();
      (void)sinkEnd;
      result.logicalNetSinks_.insert(result.logicalNetSinks_.end(),
                                     net.sinks.begin(), net.sinks.end());
      auto sinkCount = checked(sinkCountContext, net.sinks.size());
      if (!sinkCount)
        return sinkCount.takeError();
      result.logicalNets_.push_back({net.producer, *sinkOffset, *sinkCount});
    }
    return result;
  }
};

llvm::Expected<FrozenSpatialTransferIndex>
loom::pnr::detail::buildFrozenSpatialTransferIndex(
    const TechMappingView &techMapping) {
  return FrozenSpatialTransferIndexBuilder::build(techMapping);
}
