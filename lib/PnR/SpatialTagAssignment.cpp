#include "PnR/SpatialTagAssignment.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Errc.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <system_error>
#include <variant>
#include <vector>

using namespace loom;
using namespace loom::mapping;
using namespace loom::pnr;

namespace {

constexpr llvm::StringLiteral tagAssignment = "SpatialTagAssignmentProjection";
constexpr PnrCapacityContext segmentCountContext{tagAssignment, "segments",
                                                 "tag_continuity_segments",
                                                 PnrCapacityMeasure::Count};
constexpr PnrCapacityContext segmentOffsetContext{
    tagAssignment, "net_segment_offsets", "tag_continuity_segments",
    PnrCapacityMeasure::Offset};
constexpr PnrCapacityContext incidenceCountContext{
    tagAssignment, "segment_domains", "segment_domain_incidence",
    PnrCapacityMeasure::Count};

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::errc::invalid_argument,
                                 "invalid Spatial tag assignment: %s",
                                 message.str().c_str());
}

llvm::Error unsupported(const llvm::Twine &message) {
  return llvm::createStringError(llvm::errc::not_supported,
                                 "Spatial tag assignment unavailable: %s",
                                 message.str().c_str());
}

int compareUnsigned(const llvm::APInt &lhs, const llvm::APInt &rhs) {
  const unsigned width = std::max(lhs.getBitWidth(), rhs.getBitWidth());
  const llvm::APInt left = lhs.zext(width);
  const llvm::APInt right = rhs.zext(width);
  if (left.ult(right))
    return -1;
  if (right.ult(left))
    return 1;
  return 0;
}

llvm::APInt canonicalUnsigned(const llvm::APInt &value) {
  const unsigned width = std::max(1u, value.getActiveBits());
  return value.zextOrTrunc(width);
}

llvm::Expected<llvm::APInt> nextUnsigned(const llvm::APInt &value) {
  if (value.getBitWidth() == std::numeric_limits<unsigned>::max())
    return invalid("Physical Tag candidate width overflows");
  llvm::APInt next =
      value.isAllOnes() ? value.zext(value.getBitWidth() + 1) : value;
  ++next;
  return canonicalUnsigned(next);
}

bool isFree(llvm::ArrayRef<PnrIndex> domains, const llvm::APInt &value,
            llvm::ArrayRef<llvm::DenseMap<llvm::APInt, PnrIndex>> occupancy) {
  return llvm::all_of(domains, [&](PnrIndex domain) {
    return occupancy[domain].lookup(value) == 0;
  });
}

llvm::Expected<std::optional<llvm::APInt>>
chooseValue(std::uint32_t tagWidthBits, llvm::ArrayRef<PnrIndex> domains,
            const std::optional<llvm::ArrayRef<SpatialConstraintDomainValue>>
                &restriction,
            llvm::ArrayRef<llvm::DenseMap<llvm::APInt, PnrIndex>> occupancy) {
  std::optional<llvm::APInt> firstAllowed;
  const auto considerRange = [&](llvm::APInt candidate,
                                 const llvm::APInt *upper)
      -> llvm::Expected<std::optional<llvm::APInt>> {
    candidate = canonicalUnsigned(candidate);
    while ((!upper || compareUnsigned(candidate, *upper) < 0) &&
           ::fabric::isRepresentablePhysicalTagValue(tagWidthBits, candidate)) {
      if (!firstAllowed)
        firstAllowed = candidate;
      if (isFree(domains, candidate, occupancy))
        return std::optional<llvm::APInt>(std::move(candidate));
      auto next = nextUnsigned(candidate);
      if (!next)
        return next.takeError();
      candidate = std::move(*next);
    }
    return std::optional<llvm::APInt>();
  };

  if (!restriction) {
    auto selected = considerRange(llvm::APInt(1, 0), nullptr);
    if (!selected)
      return selected.takeError();
    if (*selected)
      return std::move(*selected);
    return std::move(firstAllowed);
  }

  for (const SpatialConstraintDomainValue &domainValue : *restriction) {
    const auto *interval =
        std::get_if<SpatialConstraintUnsignedInterval>(&domainValue);
    if (!interval)
      return invalid("tag restriction contains a non-interval value");
    auto selected = considerRange(interval->lower, &interval->upper);
    if (!selected)
      return selected.takeError();
    if (*selected)
      return std::move(*selected);
    if (!::fabric::isRepresentablePhysicalTagValue(tagWidthBits,
                                                   interval->lower))
      break;
  }
  return std::move(firstAllowed);
}

llvm::Expected<PnrIndex> checkedIndex(PnrCapacityContext context,
                                      std::size_t value) {
  return checkedPnrIndex(context, static_cast<std::uint64_t>(value));
}

} // namespace

llvm::ArrayRef<PnrIndex>
SpatialTagAssignmentProjection::domainSegments(PnrIndex domain) const {
  assert(domain + 1 < domainSegmentOffsets_.size());
  return llvm::ArrayRef<PnrIndex>(domainSegments_)
      .slice(domainSegmentOffsets_[domain],
             domainSegmentOffsets_[domain + 1] - domainSegmentOffsets_[domain]);
}

llvm::Expected<SpatialTagAssignmentProjection>
loom::pnr::deriveCanonicalSpatialTagAssignments(
    const FrozenSpatialPnrProblem &problem,
    llvm::ArrayRef<const RouteTreeState *> routes) {
  const auto logicalNets = problem.transfers().logicalNets();
  if (routes.size() != logicalNets.size())
    return invalid("route count does not match the frozen logical nets");

  const FrozenConstraintShard &tagConstraints = problem.constraints().shard(
      ::mapping::SpatialConstraintProjection::NetAssignedTagValues);
  if (!tagConstraints.equalityClasses().empty() ||
      !tagConstraints.disjointGroups().empty())
    return unsupported(
        "tag equality and disjointness require their relation owner");

  SpatialTagAssignmentProjection result;
  result.netSegmentOffsets_.reserve(routes.size() + 1);
  result.netSegmentOffsets_.push_back(0);
  result.segmentDomainOffsets_.push_back(0);
  for (auto [logicalNet, route] : llvm::enumerate(routes)) {
    if (!route || &route->routingGraph() != &problem.routing())
      return invalid("route does not belong to the frozen routing graph");
    auto continuity = deriveSpatialTagContinuity(*route);
    if (!continuity)
      return continuity.takeError();
    if (llvm::Error error = preflightPnrIndexCapacity(
            segmentCountContext,
            result.segments_.size() + continuity->segments().size()))
      return std::move(error);
    result.segments_.insert(result.segments_.end(),
                            continuity->segments().begin(),
                            continuity->segments().end());
    const auto localOffsets = continuity->segmentDomainOffsets();
    const auto localDomains = continuity->segmentDomains();
    for (PnrIndex segment = 0; segment < continuity->segments().size();
         ++segment) {
      const PnrIndex begin = localOffsets[segment];
      const PnrIndex end = localOffsets[segment + 1];
      if (llvm::Error error = preflightPnrIndexCapacity(
              incidenceCountContext,
              result.segmentDomains_.size() + (end - begin)))
        return std::move(error);
      result.segmentDomains_.insert(result.segmentDomains_.end(),
                                    localDomains.begin() + begin,
                                    localDomains.begin() + end);
      auto offset =
          checkedIndex(incidenceCountContext, result.segmentDomains_.size());
      if (!offset)
        return offset.takeError();
      result.segmentDomainOffsets_.push_back(*offset);
    }
    auto netEnd = checkedIndex(segmentOffsetContext, result.segments_.size());
    if (!netEnd)
      return netEnd.takeError();
    result.netSegmentOffsets_.push_back(*netEnd);
    (void)logicalNet;
  }

  const std::size_t domainCount =
      problem.routing().tagContinuity().matchDomains().size();
  std::vector<PnrIndex> domainCounts(domainCount, 0);
  for (PnrIndex domain : result.segmentDomains_) {
    if (domain >= domainCount)
      return invalid("segment names an out-of-range tag match domain");
    if (domainCounts[domain] == getInvalidPnrIndex())
      return invalid("tag match-domain incidence count overflows PnrIndex");
    ++domainCounts[domain];
  }
  result.domainSegmentOffsets_.reserve(domainCount + 1);
  result.domainSegmentOffsets_.push_back(0);
  for (PnrIndex count : domainCounts) {
    auto end = checkedPnrIndexAdd(incidenceCountContext,
                                  result.domainSegmentOffsets_.back(), count);
    if (!end)
      return end.takeError();
    result.domainSegmentOffsets_.push_back(*end);
  }
  result.domainSegments_.resize(result.segmentDomains_.size());
  std::vector<PnrIndex> cursors(result.domainSegmentOffsets_.begin(),
                                result.domainSegmentOffsets_.end() - 1);
  for (PnrIndex segment = 0; segment < result.segments_.size(); ++segment)
    for (PnrIndex incidence = result.segmentDomainOffsets_[segment];
         incidence < result.segmentDomainOffsets_[segment + 1]; ++incidence) {
      const PnrIndex domain = result.segmentDomains_[incidence];
      result.domainSegments_[cursors[domain]++] = segment;
    }

  std::vector<llvm::DenseMap<llvm::APInt, PnrIndex>> occupancy(domainCount);
  result.values_.reserve(result.segments_.size());
  for (PnrIndex logicalNet = 0; logicalNet < logicalNets.size(); ++logicalNet) {
    const auto restriction = tagConstraints.restrictedDomain(
        SpatialConstraintSubject{logicalNets[logicalNet].producer});
    for (PnrIndex segment = result.netSegmentOffsets_[logicalNet];
         segment < result.netSegmentOffsets_[logicalNet + 1]; ++segment) {
      const auto domains =
          llvm::ArrayRef<PnrIndex>(result.segmentDomains_)
              .slice(result.segmentDomainOffsets_[segment],
                     result.segmentDomainOffsets_[segment + 1] -
                         result.segmentDomainOffsets_[segment]);
      for (PnrIndex domain : domains)
        if (problem.routing()
                .tagContinuity()
                .matchDomains()[domain]
                .tagWidthBits != result.segments_[segment].tagWidthBits)
          return invalid("segment and local match-domain widths disagree");
      auto value = chooseValue(result.segments_[segment].tagWidthBits, domains,
                               restriction, occupancy);
      if (!value)
        return value.takeError();
      if (!*value) {
        ++result.unassignedCount_;
        result.values_.push_back(std::nullopt);
        continue;
      }
      result.values_.push_back(std::move(**value));
      const llvm::APInt &selected = *result.values_.back();
      for (PnrIndex domain : domains) {
        PnrIndex &count = occupancy[domain][selected];
        if (count != 0)
          ++result.conflictCount_;
        if (count == getInvalidPnrIndex())
          return invalid("tag match-domain occupancy overflows PnrIndex");
        ++count;
      }
    }
  }
  return result;
}
