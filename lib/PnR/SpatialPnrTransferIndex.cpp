#include "SpatialPnrTransferIndex.h"

#include "Mapping/Artifact/MappingProgressAnalysis.h"
#include "PnR/PnrIndex.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <vector>

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
constexpr PnrCapacityContext progressDependencyContext{
    frozenArtifact, "logical_net_sinks", "progress_dependencies",
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
  build(const ::dataflow::CanonicalDataflowProgramView &dataflow,
        const TechMappingView &techMapping) {
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
    if (llvm::Error error =
            buildProgressDependencies(dataflow, techMapping, result))
      return std::move(error);
    return result;
  }

private:
  static llvm::Error buildProgressDependencies(
      const ::dataflow::CanonicalDataflowProgramView &dataflow,
      const TechMappingView &techMapping, FrozenSpatialTransferIndex &result) {
    auto projected = ::loom::mapping::deriveSpatialRouteProgressDependencies(
        dataflow, techMapping);
    if (!projected)
      return projected.takeError();
    std::vector<std::vector<std::vector<PnrIndex>>> dependencies;
    dependencies.reserve(result.logicalNets_.size());
    for (const FrozenSpatialLogicalNet &net : result.logicalNets_)
      dependencies.emplace_back(net.sinkCount);
    for (const auto &dependency : *projected) {
      if (dependency.logicalNetOrdinal >= dependencies.size() ||
          dependency.dependentSinkOrdinal >=
              dependencies[dependency.logicalNetOrdinal].size())
        return invalid("progress dependency is outside the frozen net index");
      auto prerequisite = checked(progressDependencyContext,
                                  dependency.prerequisiteSinkOrdinal);
      if (!prerequisite)
        return prerequisite.takeError();
      dependencies[dependency.logicalNetOrdinal]
                  [dependency.dependentSinkOrdinal]
                      .push_back(*prerequisite);
    }
    result.sinkProgressDependencyOffsets_.clear();
    result.sinkProgressDependencies_.clear();
    result.sinkProgressDependencyOffsets_.reserve(
        result.logicalNetSinks_.size() + 1);
    result.sinkProgressDependencyOffsets_.push_back(0);
    for (const auto [netOrdinal, net] : llvm::enumerate(result.logicalNets_)) {
      for (PnrIndex dependent = 0; dependent < net.sinkCount; ++dependent) {
        auto &sinkDependencies = dependencies[netOrdinal][dependent];
        llvm::sort(sinkDependencies);
        sinkDependencies.erase(
            std::unique(sinkDependencies.begin(), sinkDependencies.end()),
            sinkDependencies.end());
        result.sinkProgressDependencies_.insert(
            result.sinkProgressDependencies_.end(), sinkDependencies.begin(),
            sinkDependencies.end());
        auto end = checked(progressDependencyContext,
                           result.sinkProgressDependencies_.size());
        if (!end)
          return end.takeError();
        result.sinkProgressDependencyOffsets_.push_back(*end);
      }
    }
    if (result.sinkProgressDependencyOffsets_.size() !=
        result.logicalNetSinks_.size() + 1)
      return invalid("progress dependency CSR has the wrong shape");
    return llvm::Error::success();
  }
};

llvm::Expected<FrozenSpatialTransferIndex>
loom::pnr::detail::buildFrozenSpatialTransferIndex(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const TechMappingView &techMapping) {
  return FrozenSpatialTransferIndexBuilder::build(dataflow, techMapping);
}
