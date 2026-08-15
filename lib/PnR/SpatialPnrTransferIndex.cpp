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
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>
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
      auto type = dataflow.tokenType(net.producer);
      if (!type)
        return type.takeError();
      auto payloadWidth = dataflow.transportPayloadBitWidth(*type);
      if (!payloadWidth)
        return payloadWidth.takeError();
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
      result.logicalNets_.push_back(
          {net.producer, *payloadWidth, *sinkOffset, *sinkCount});
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
    std::vector<std::vector<std::vector<FrozenSpatialProgressPrerequisite>>>
        dependencies;
    dependencies.reserve(result.logicalNets_.size());
    for (const FrozenSpatialLogicalNet &net : result.logicalNets_)
      dependencies.emplace_back(net.sinkCount);
    for (const auto &dependency : *projected) {
      if (dependency.logicalNetOrdinal >= dependencies.size() ||
          dependency.dependentSinkOrdinal >=
              dependencies[dependency.logicalNetOrdinal].size())
        return invalid("progress dependency is outside the frozen net index");
      FrozenSpatialProgressPrerequisite prerequisite;
      if (const auto *external =
              std::get_if<SpatialRouteExternalSinkPrerequisite>(
                  &dependency.prerequisite)) {
        auto sink = checked(progressDependencyContext, external->sinkOrdinal);
        if (!sink)
          return sink.takeError();
        prerequisite = FrozenSpatialExternalSinkPrerequisite{*sink};
      } else {
        const auto &internal =
            std::get<SpatialRouteInternalMemoryConnectionPrerequisite>(
                dependency.prerequisite);
        const auto realizations = techMapping.memoryRealizations();
        if (internal.memoryRealizationOrdinal >= realizations.size() ||
            internal.internalEdgeOrdinal >=
                realizations[internal.memoryRealizationOrdinal]
                    .internalEdges.size() ||
            realizations[internal.memoryRealizationOrdinal]
                    .internalEdges[internal.internalEdgeOrdinal]
                    .producer !=
                result.logicalNets_[dependency.logicalNetOrdinal].producer)
          return invalid(
              "progress dependency internal connection does not resolve");
        auto realization = checked(progressDependencyContext,
                                   internal.memoryRealizationOrdinal);
        if (!realization)
          return realization.takeError();
        auto edge =
            checked(progressDependencyContext, internal.internalEdgeOrdinal);
        if (!edge)
          return edge.takeError();
        prerequisite = FrozenSpatialInternalMemoryConnectionPrerequisite{
            *realization, *edge};
      }
      dependencies[dependency.logicalNetOrdinal]
                  [dependency.dependentSinkOrdinal]
                      .push_back(std::move(prerequisite));
    }
    result.sinkProgressDependencyOffsets_.clear();
    result.sinkProgressDependencies_.clear();
    result.sinkProgressDependencyOffsets_.reserve(
        result.logicalNetSinks_.size() + 1);
    result.sinkProgressDependencyOffsets_.push_back(0);
    for (const auto [netOrdinal, net] : llvm::enumerate(result.logicalNets_)) {
      for (PnrIndex dependent = 0; dependent < net.sinkCount; ++dependent) {
        auto &sinkDependencies = dependencies[netOrdinal][dependent];
        const auto key = [](const FrozenSpatialProgressPrerequisite &value) {
          return std::visit(
              [](const auto &typed) {
                using T = std::decay_t<decltype(typed)>;
                if constexpr (std::is_same_v<
                                  T, FrozenSpatialExternalSinkPrerequisite>)
                  return std::tuple<std::uint8_t, PnrIndex, PnrIndex>{
                      0, typed.sink, 0};
                else
                  return std::tuple<std::uint8_t, PnrIndex, PnrIndex>{
                      1, typed.memoryRealization, typed.internalEdge};
              },
              value);
        };
        llvm::sort(sinkDependencies, [&](const auto &lhs, const auto &rhs) {
          return key(lhs) < key(rhs);
        });
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
