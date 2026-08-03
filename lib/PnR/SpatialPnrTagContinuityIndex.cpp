#include "PnR/SpatialPnrProblem.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <system_error>
#include <vector>

using namespace loom::fabric;
using namespace loom::pnr;

namespace {

constexpr llvm::StringLiteral frozenTagContinuity =
    "FrozenSpatialTagContinuityIndex";
constexpr PnrCapacityContext pointCountContext{frozenTagContinuity, "points",
                                               "boundary_occurrences",
                                               PnrCapacityMeasure::Count};
constexpr PnrCapacityContext pointIndexContext{frozenTagContinuity, "points",
                                               "boundary_occurrences",
                                               PnrCapacityMeasure::Index};
constexpr PnrCapacityContext traversalCountContext{
    frozenTagContinuity, "traversal_point_ordinals", "physical_traversals",
    PnrCapacityMeasure::Count};

llvm::Error invalid(llvm::Twine message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      "invalid Frozen Spatial tag-continuity index: %s", message.str().c_str());
}

} // namespace

llvm::Expected<FrozenSpatialTagContinuityIndex>
loom::pnr::freezeSpatialTagContinuityIndex(const FabricArtifactView &fabric) {
  const auto boundaries = fabric.boundaryOccurrences();
  const auto traversals = fabric.physicalTraversals();
  if (llvm::Error error =
          preflightPnrIndexCapacity(pointCountContext, boundaries.size()))
    return std::move(error);
  if (llvm::Error error =
          preflightPnrIndexCapacity(traversalCountContext, traversals.size()))
    return std::move(error);

  FrozenSpatialTagContinuityIndex result;
  result.points_.reserve(boundaries.size());
  llvm::DenseMap<FabricEntityId, PnrIndex> pointByBoundary;
  pointByBoundary.reserve(boundaries.size());
  for (auto [ordinal, boundary] : llvm::enumerate(boundaries)) {
    const auto projection = fabric.boundaryTagContinuityPoint(boundary);
    if (!projection)
      return invalid("a canonical boundary has no typed continuity point");
    auto point = checkedPnrIndex(pointIndexContext, ordinal);
    if (!point)
      return point.takeError();
    if (!pointByBoundary.try_emplace(boundary.id(), *point).second)
      return invalid("the canonical boundary inventory contains a duplicate");
    result.points_.push_back({boundary, projection->kind,
                              projection->inputTagWidthBits,
                              projection->outputTagWidthBits});
  }

  result.traversalPointOrdinals_.assign(traversals.size(),
                                        getInvalidPnrIndex());
  std::vector<std::uint8_t> pointObserved(result.points_.size(), 0);
  for (auto [ordinal, traversal] : llvm::enumerate(traversals)) {
    if (traversal.reference.kind() !=
        FabricPhysicalTraversalKind::BoundaryTraversal)
      continue;
    const auto owner =
        std::get<FabricBoundaryTraversalPayload>(traversal.reference.payload)
            .owner;
    const auto point = pointByBoundary.find(owner.id());
    if (point == pointByBoundary.end())
      return invalid("a boundary traversal names an absent continuity point");
    result.traversalPointOrdinals_[ordinal] = point->second;
    pointObserved[point->second] = 1;
  }
  if (llvm::is_contained(pointObserved, std::uint8_t{0}))
    return invalid("a continuity point has no physical traversal");
  return result;
}
