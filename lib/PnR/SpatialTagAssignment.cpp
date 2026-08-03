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
#include <tuple>
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

bool valueAllowed(
    const llvm::APInt &value,
    const std::optional<llvm::ArrayRef<SpatialConstraintDomainValue>>
        &restriction) {
  if (!restriction)
    return true;
  return llvm::any_of(*restriction, [&](const auto &domainValue) {
    const auto *interval =
        std::get_if<SpatialConstraintUnsignedInterval>(&domainValue);
    return interval && compareUnsigned(interval->lower, value) <= 0 &&
           compareUnsigned(value, interval->upper) < 0;
  });
}

llvm::Error addAssignment(
    llvm::MutableArrayRef<llvm::DenseMap<llvm::APInt, PnrIndex>> occupancy,
    std::uint64_t &unassignedCount, std::uint64_t &conflictCount,
    llvm::ArrayRef<PnrIndex> domains, const std::optional<llvm::APInt> &value) {
  if (!value) {
    if (unassignedCount == std::numeric_limits<std::uint64_t>::max())
      return invalid("unassigned count overflows u64");
    ++unassignedCount;
    return llvm::Error::success();
  }

  std::uint64_t addedConflicts = 0;
  for (PnrIndex domain : domains) {
    if (domain >= occupancy.size())
      return invalid("assignment names an out-of-range tag match domain");
    const PnrIndex count = occupancy[domain].lookup(*value);
    if (count >= getInvalidPnrIndex() - PnrIndex{1})
      return invalid("tag match-domain occupancy overflows PnrIndex");
    addedConflicts += count != 0;
  }
  if (addedConflicts >
      std::numeric_limits<std::uint64_t>::max() - conflictCount)
    return invalid("tag conflict count overflows u64");
  conflictCount += addedConflicts;
  for (PnrIndex domain : domains)
    ++occupancy[domain][*value];
  return llvm::Error::success();
}

void removeAssignment(
    llvm::MutableArrayRef<llvm::DenseMap<llvm::APInt, PnrIndex>> occupancy,
    std::uint64_t &unassignedCount, std::uint64_t &conflictCount,
    llvm::ArrayRef<PnrIndex> domains,
    const std::optional<llvm::APInt> &value) noexcept {
  if (!value) {
    assert(unassignedCount != 0);
    --unassignedCount;
    return;
  }
  for (PnrIndex domain : domains) {
    assert(domain < occupancy.size());
    auto found = occupancy[domain].find(*value);
    assert(found != occupancy[domain].end() && found->second != 0);
    if (found->second > 1) {
      assert(conflictCount != 0);
      --conflictCount;
      --found->second;
    } else {
      occupancy[domain].erase(found);
    }
  }
}

} // namespace

namespace loom::pnr::detail {

struct SpatialTagNetState final {
  SpatialTagContinuityProjection continuity;
  std::vector<std::optional<llvm::APInt>> values;
};

struct SpatialTagAssignmentScratchStorage final {
  const FrozenSpatialPnrProblem *problem = nullptr;
  std::vector<SpatialTagNetState> stagedNets;
  std::vector<PnrIndex> touchedRoutes;
  SpatialTagContinuityScratch continuityScratch;
  std::size_t proposedRouteCount = 0;
  bool active = false;
};

struct SpatialTagAssignmentStateStorage final {
  const FrozenSpatialPnrProblem *problem = nullptr;
  std::vector<SpatialTagNetState> nets;
  std::vector<llvm::DenseMap<llvm::APInt, PnrIndex>> occupancy;
  std::uint64_t unassignedCount = 0;
  std::uint64_t conflictCount = 0;
};

} // namespace loom::pnr::detail

namespace {

using TagNetState = loom::pnr::detail::SpatialTagNetState;
using TagStateStorage = loom::pnr::detail::SpatialTagAssignmentStateStorage;

llvm::ArrayRef<PnrIndex> segmentDomains(const TagNetState &net,
                                        PnrIndex segment) {
  const auto offsets = net.continuity.segmentDomainOffsets();
  assert(segment + 1 < offsets.size());
  return net.continuity.segmentDomains().slice(
      offsets[segment], offsets[segment + 1] - offsets[segment]);
}

void removeNet(TagStateStorage &storage, const TagNetState &net) noexcept {
  for (PnrIndex segment = 0; segment < net.values.size(); ++segment)
    removeAssignment(storage.occupancy, storage.unassignedCount,
                     storage.conflictCount, segmentDomains(net, segment),
                     net.values[segment]);
}

std::optional<llvm::APInt>
preservedValue(const TagNetState *oldNet,
               const SpatialTagContinuitySegment &segment) {
  if (!oldNet)
    return std::nullopt;
  const auto oldSegments = oldNet->continuity.segments();
  const auto found =
      llvm::lower_bound(oldSegments, segment,
                        [](const SpatialTagContinuitySegment &lhs,
                           const SpatialTagContinuitySegment &rhs) {
                          return std::tie(lhs.originKind, lhs.origin) <
                                 std::tie(rhs.originKind, rhs.origin);
                        });
  if (found == oldSegments.end() || found->originKind != segment.originKind ||
      found->origin != segment.origin)
    return std::nullopt;
  const auto index = static_cast<std::size_t>(found - oldSegments.begin());
  if (index >= oldNet->values.size())
    return std::nullopt;
  return oldNet->values[index];
}

llvm::Error buildNet(TagStateStorage &storage, PnrIndex logicalNet,
                     const TagNetState *oldNet, TagNetState &result) {
  result.values.clear();
  result.values.reserve(result.continuity.segments().size());

  const auto logicalNets = storage.problem->transfers().logicalNets();
  if (logicalNet >= logicalNets.size())
    return invalid("logical net ordinal is out of range");
  const FrozenConstraintShard &tagConstraints =
      storage.problem->constraints().shard(
          ::mapping::SpatialConstraintProjection::NetAssignedTagValues);
  const auto restriction = tagConstraints.restrictedDomain(
      SpatialConstraintSubject{logicalNets[logicalNet].producer});
  const auto matchDomains =
      storage.problem->routing().tagContinuity().matchDomains();

  for (PnrIndex ordinal = 0; ordinal < result.continuity.segments().size();
       ++ordinal) {
    const SpatialTagContinuitySegment &segment =
        result.continuity.segments()[ordinal];
    const auto domains = segmentDomains(result, ordinal);
    for (PnrIndex domain : domains)
      if (domain >= matchDomains.size() ||
          matchDomains[domain].tagWidthBits != segment.tagWidthBits)
        return invalid("segment and local match-domain widths disagree");

    std::optional<llvm::APInt> selected = preservedValue(oldNet, segment);
    if (selected && (!::fabric::isRepresentablePhysicalTagValue(
                         segment.tagWidthBits, *selected) ||
                     !valueAllowed(*selected, restriction) ||
                     !isFree(domains, *selected, storage.occupancy)))
      selected.reset();
    if (!selected) {
      auto chosen = chooseValue(segment.tagWidthBits, domains, restriction,
                                storage.occupancy);
      if (!chosen)
        return chosen.takeError();
      selected = std::move(*chosen);
    }
    result.values.push_back(std::move(selected));
    if (llvm::Error error =
            addAssignment(storage.occupancy, storage.unassignedCount,
                          storage.conflictCount, domains, result.values.back()))
      return error;
  }
  return llvm::Error::success();
}

std::size_t retainedBytes(const TagNetState &net) {
  return net.continuity.retainedStorageBytes() +
         net.values.capacity() * sizeof(std::optional<llvm::APInt>);
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
    if (domainCounts[domain] >= getInvalidPnrIndex() - PnrIndex{1})
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
        if (count >= getInvalidPnrIndex() - PnrIndex{1})
          return invalid("tag match-domain occupancy overflows PnrIndex");
        ++count;
      }
    }
  }
  return result;
}

SpatialTagAssignmentScratch::SpatialTagAssignmentScratch()
    : storage_(std::make_unique<detail::SpatialTagAssignmentScratchStorage>()) {
}

SpatialTagAssignmentScratch::~SpatialTagAssignmentScratch() {
  assert(!storage_->active &&
         "destroying Physical Tag scratch during an active transaction");
}

llvm::Error
SpatialTagAssignmentScratch::prepare(const FrozenSpatialPnrProblem &problem) {
  if (storage_->active)
    return invalid("cannot prepare scratch during an active transaction");
  storage_->problem = &problem;
  storage_->stagedNets.resize(problem.transfers().logicalNets().size());
  storage_->touchedRoutes.clear();
  storage_->touchedRoutes.reserve(problem.transfers().logicalNets().size());
  return llvm::Error::success();
}

std::size_t SpatialTagAssignmentScratch::retainedStorageBytes() const {
  std::size_t bytes =
      storage_->stagedNets.capacity() * sizeof(detail::SpatialTagNetState) +
      storage_->touchedRoutes.capacity() * sizeof(PnrIndex) +
      storage_->continuityScratch.retainedStorageBytes();
  for (const auto &net : storage_->stagedNets)
    bytes += retainedBytes(net);
  return bytes;
}

SpatialTagAssignmentState::SpatialTagAssignmentState(
    std::unique_ptr<detail::SpatialTagAssignmentStateStorage> storage)
    : storage_(std::move(storage)) {}

SpatialTagAssignmentState::SpatialTagAssignmentState(
    SpatialTagAssignmentState &&) noexcept = default;

SpatialTagAssignmentState::~SpatialTagAssignmentState() = default;

llvm::Expected<SpatialTagAssignmentState>
SpatialTagAssignmentState::create(const FrozenSpatialPnrProblem &problem,
                                  llvm::ArrayRef<RouteTreeStateHandle> routes) {
  if (routes.size() != problem.transfers().logicalNets().size())
    return invalid("route count does not match the frozen logical nets");
  const FrozenConstraintShard &tagConstraints = problem.constraints().shard(
      ::mapping::SpatialConstraintProjection::NetAssignedTagValues);
  if (!tagConstraints.equalityClasses().empty() ||
      !tagConstraints.disjointGroups().empty())
    return unsupported(
        "tag equality and disjointness require their relation owner");

  auto storage = std::make_unique<detail::SpatialTagAssignmentStateStorage>();
  storage->problem = &problem;
  storage->nets.resize(routes.size());
  storage->occupancy.resize(
      problem.routing().tagContinuity().matchDomains().size());
  for (PnrIndex logicalNet = 0; logicalNet < routes.size(); ++logicalNet) {
    if (!routes[logicalNet] ||
        &routes[logicalNet]->routingGraph() != &problem.routing())
      return invalid("route does not belong to the frozen routing graph");
    auto continuity = deriveSpatialTagContinuity(*routes[logicalNet]);
    if (!continuity)
      return continuity.takeError();
    storage->nets[logicalNet].continuity = std::move(*continuity);
    if (llvm::Error error =
            buildNet(*storage, logicalNet, nullptr, storage->nets[logicalNet]))
      return std::move(error);
  }
  return SpatialTagAssignmentState(std::move(storage));
}

llvm::ArrayRef<SpatialTagContinuitySegment>
SpatialTagAssignmentState::segments(PnrIndex logicalNet) const {
  assert(logicalNet < storage_->nets.size());
  return storage_->nets[logicalNet].continuity.segments();
}

llvm::ArrayRef<std::optional<llvm::APInt>>
SpatialTagAssignmentState::values(PnrIndex logicalNet) const {
  assert(logicalNet < storage_->nets.size());
  return storage_->nets[logicalNet].values;
}

llvm::ArrayRef<PnrIndex>
SpatialTagAssignmentState::segmentDomains(PnrIndex logicalNet,
                                          PnrIndex segment) const {
  assert(logicalNet < storage_->nets.size());
  return ::segmentDomains(storage_->nets[logicalNet], segment);
}

std::uint64_t SpatialTagAssignmentState::unassignedCount() const {
  return storage_->unassignedCount;
}

std::uint64_t SpatialTagAssignmentState::conflictCount() const {
  return storage_->conflictCount;
}

llvm::Error SpatialTagAssignmentState::stageRouteUpdates(
    llvm::ArrayRef<RouteTreeStateHandle> routes,
    llvm::ArrayRef<std::optional<RouteTreeTransaction>> routeTransactions,
    llvm::ArrayRef<PnrIndex> touchedRoutes,
    SpatialTagAssignmentScratch &scratch) {
  auto &transaction = *scratch.storage_;
  if (transaction.active)
    return invalid("Physical Tag transaction is already active");
  if (transaction.problem != storage_->problem ||
      transaction.stagedNets.size() != storage_->nets.size() ||
      routes.size() != storage_->nets.size() ||
      routeTransactions.size() != storage_->nets.size())
    return invalid("Physical Tag scratch belongs to another frozen problem");

  transaction.touchedRoutes.assign(touchedRoutes.begin(), touchedRoutes.end());
  llvm::sort(transaction.touchedRoutes);
  transaction.touchedRoutes.erase(std::unique(transaction.touchedRoutes.begin(),
                                              transaction.touchedRoutes.end()),
                                  transaction.touchedRoutes.end());
  if (transaction.touchedRoutes.empty())
    return llvm::Error::success();
  for (PnrIndex logicalNet : transaction.touchedRoutes)
    if (logicalNet >= routes.size() || !routes[logicalNet] ||
        &routes[logicalNet]->routingGraph() != &storage_->problem->routing())
      return invalid("touched route does not belong to the frozen problem");

  transaction.active = true;
  transaction.proposedRouteCount = 0;
  for (PnrIndex logicalNet : transaction.touchedRoutes) {
    removeNet(*storage_, storage_->nets[logicalNet]);
    std::swap(storage_->nets[logicalNet], transaction.stagedNets[logicalNet]);
  }
  for (PnrIndex logicalNet : transaction.touchedRoutes) {
    if (!routeTransactions[logicalNet]) {
      rollback(scratch);
      return invalid("touched route has no prepared transaction");
    }
    if (llvm::Error error =
            rebuildSpatialTagContinuity(*routeTransactions[logicalNet],
                                        storage_->nets[logicalNet].continuity,
                                        transaction.continuityScratch)) {
      rollback(scratch);
      return error;
    }
    ++transaction.proposedRouteCount;
    if (llvm::Error error =
            buildNet(*storage_, logicalNet, &transaction.stagedNets[logicalNet],
                     storage_->nets[logicalNet])) {
      rollback(scratch);
      return error;
    }
  }
  return llvm::Error::success();
}

void SpatialTagAssignmentState::commit(
    SpatialTagAssignmentScratch &scratch) noexcept {
  auto &transaction = *scratch.storage_;
  if (!transaction.active)
    return;
  transaction.active = false;
  transaction.proposedRouteCount = 0;
  transaction.touchedRoutes.clear();
}

void SpatialTagAssignmentState::rollback(
    SpatialTagAssignmentScratch &scratch) noexcept {
  auto &transaction = *scratch.storage_;
  if (!transaction.active)
    return;
  for (PnrIndex logicalNet : llvm::ArrayRef<PnrIndex>(transaction.touchedRoutes)
                                 .take_front(transaction.proposedRouteCount))
    removeNet(*storage_, storage_->nets[logicalNet]);
  for (PnrIndex logicalNet : transaction.touchedRoutes) {
    std::swap(storage_->nets[logicalNet], transaction.stagedNets[logicalNet]);
    const TagNetState &restored = storage_->nets[logicalNet];
    for (PnrIndex segment = 0; segment < restored.values.size(); ++segment)
      llvm::cantFail(addAssignment(
          storage_->occupancy, storage_->unassignedCount,
          storage_->conflictCount, ::segmentDomains(restored, segment),
          restored.values[segment]));
  }
  transaction.active = false;
  transaction.proposedRouteCount = 0;
  transaction.touchedRoutes.clear();
}

llvm::Error SpatialTagAssignmentState::verify(
    llvm::ArrayRef<RouteTreeStateHandle> routes) const {
  if (routes.size() != storage_->nets.size())
    return invalid("candidate route count changed after Tag initialization");
  std::vector<llvm::DenseMap<llvm::APInt, PnrIndex>> expectedOccupancy(
      storage_->occupancy.size());
  std::uint64_t expectedUnassigned = 0;
  std::uint64_t expectedConflicts = 0;
  const auto logicalNets = storage_->problem->transfers().logicalNets();
  const FrozenConstraintShard &tagConstraints =
      storage_->problem->constraints().shard(
          ::mapping::SpatialConstraintProjection::NetAssignedTagValues);
  const auto matchDomains =
      storage_->problem->routing().tagContinuity().matchDomains();

  for (PnrIndex logicalNet = 0; logicalNet < routes.size(); ++logicalNet) {
    if (!routes[logicalNet] ||
        &routes[logicalNet]->routingGraph() != &storage_->problem->routing())
      return invalid("candidate route belongs to another routing graph");
    auto continuity = deriveSpatialTagContinuity(*routes[logicalNet]);
    if (!continuity)
      return continuity.takeError();
    const TagNetState &net = storage_->nets[logicalNet];
    const auto diverged = [&](llvm::StringRef field) {
      return invalid("cached Tag continuity " + field +
                     " diverges from RouteTree state for logical net " +
                     llvm::Twine(logicalNet));
    };
    if (!llvm::equal(net.continuity.segments(), continuity->segments()))
      return diverged("segments");
    if (!llvm::equal(net.continuity.nodeSegments(),
                     continuity->nodeSegments())) {
      const auto cached = net.continuity.nodeSegments();
      const auto derived = continuity->nodeSegments();
      const std::size_t common = std::min(cached.size(), derived.size());
      std::size_t first = 0;
      while (first != common && cached[first] == derived[first])
        ++first;
      return invalid("cached Tag continuity node segments diverge from "
                     "RouteTree state for logical net " +
                     llvm::Twine(logicalNet) + " at node slot " +
                     llvm::Twine(first) + " (cached size " +
                     llvm::Twine(cached.size()) + ", derived size " +
                     llvm::Twine(derived.size()) + ", cached value " +
                     (first == cached.size() ? llvm::Twine("absent")
                                             : llvm::Twine(cached[first])) +
                     ", derived value " +
                     (first == derived.size() ? llvm::Twine("absent")
                                              : llvm::Twine(derived[first])) +
                     ")");
    }
    if (!llvm::equal(net.continuity.segmentDomainOffsets(),
                     continuity->segmentDomainOffsets()))
      return diverged("segment-domain offsets");
    if (!llvm::equal(net.continuity.segmentDomains(),
                     continuity->segmentDomains()))
      return diverged("segment domains");
    if (!llvm::equal(net.continuity.domainSegmentOffsets(),
                     continuity->domainSegmentOffsets()))
      return diverged("domain-segment offsets");
    if (!llvm::equal(net.continuity.domainSegments(),
                     continuity->domainSegments()))
      return diverged("domain segments");
    if (net.values.size() != continuity->segments().size())
      return diverged("value cardinality");
    const auto restriction = tagConstraints.restrictedDomain(
        SpatialConstraintSubject{logicalNets[logicalNet].producer});
    for (PnrIndex segment = 0; segment < net.values.size(); ++segment) {
      const auto domains = ::segmentDomains(net, segment);
      for (PnrIndex domain : domains)
        if (domain >= matchDomains.size() ||
            matchDomains[domain].tagWidthBits !=
                net.continuity.segments()[segment].tagWidthBits)
          return invalid("cached Tag domain has an incompatible width");
      if (net.values[segment] &&
          (!::fabric::isRepresentablePhysicalTagValue(
               net.continuity.segments()[segment].tagWidthBits,
               *net.values[segment]) ||
           !valueAllowed(*net.values[segment], restriction)))
        return invalid("candidate Physical Tag value is outside its domain");
      if (llvm::Error error =
              addAssignment(expectedOccupancy, expectedUnassigned,
                            expectedConflicts, domains, net.values[segment]))
        return error;
    }
  }
  if (expectedUnassigned != storage_->unassignedCount ||
      expectedConflicts != storage_->conflictCount)
    return invalid("cached Physical Tag violation counts have drifted");
  for (PnrIndex domain = 0; domain < storage_->occupancy.size(); ++domain) {
    if (storage_->occupancy[domain].size() != expectedOccupancy[domain].size())
      return invalid("cached Physical Tag occupancy has drifted");
    for (const auto &entry : storage_->occupancy[domain])
      if (expectedOccupancy[domain].lookup(entry.first) != entry.second)
        return invalid("cached Physical Tag occupancy has drifted");
  }
  return llvm::Error::success();
}
