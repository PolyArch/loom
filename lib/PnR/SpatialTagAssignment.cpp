#include "PnR/SpatialTagAssignment.h"

#include "SpatialTagConstraintModel.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/Errc.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
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

bool unsignedLess(const llvm::APInt &lhs, const llvm::APInt &rhs) {
  return compareUnsigned(lhs, rhs) < 0;
}

void normalizeValues(std::vector<llvm::APInt> &values) {
  for (llvm::APInt &value : values)
    value = canonicalUnsigned(value);
  llvm::sort(values, unsignedLess);
  values.erase(std::unique(values.begin(), values.end(),
                           [](const llvm::APInt &lhs, const llvm::APInt &rhs) {
                             return compareUnsigned(lhs, rhs) == 0;
                           }),
               values.end());
}

bool equalUnsignedValues(llvm::ArrayRef<llvm::APInt> lhs,
                         llvm::ArrayRef<llvm::APInt> rhs) {
  return lhs.size() == rhs.size() &&
         llvm::equal(lhs, rhs,
                     [](const llvm::APInt &left, const llvm::APInt &right) {
                       return compareUnsigned(left, right) == 0;
                     });
}

bool containsValue(llvm::ArrayRef<llvm::APInt> values,
                   const llvm::APInt &value) {
  const auto found = llvm::lower_bound(values, value, unsignedLess);
  return found != values.end() && compareUnsigned(*found, value) == 0;
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
            llvm::ArrayRef<llvm::APInt> forbidden,
            llvm::ArrayRef<llvm::DenseMap<llvm::APInt, PnrIndex>> occupancy) {
  std::optional<llvm::APInt> firstAllowed;
  const auto considerRange = [&](llvm::APInt candidate,
                                 const llvm::APInt *upper)
      -> llvm::Expected<std::optional<llvm::APInt>> {
    candidate = canonicalUnsigned(candidate);
    while ((!upper || compareUnsigned(candidate, *upper) < 0) &&
           ::fabric::isRepresentablePhysicalTagValue(tagWidthBits, candidate)) {
      if (containsValue(forbidden, candidate)) {
        auto next = nextUnsigned(candidate);
        if (!next)
          return next.takeError();
        candidate = std::move(*next);
        continue;
      }
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
  std::vector<PnrIndex> routedNets;
  std::vector<PnrIndex> rebuiltNets;
  SpatialTagContinuityScratch continuityScratch;
  bool active = false;
};

struct SpatialTagAssignmentStateStorage final {
  const FrozenSpatialPnrProblem *problem = nullptr;
  const SpatialTagConstraintModel *constraints = nullptr;
  std::vector<SpatialTagNetState> nets;
  std::vector<llvm::DenseMap<llvm::APInt, PnrIndex>> occupancy;
  std::vector<PnrIndex> residentCounts;
  std::vector<std::uint8_t> classBuilt;
  std::uint64_t unassignedCount = 0;
  std::uint64_t conflictCount = 0;
  std::uint64_t residentCapacityOveruse = 0;
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

std::uint64_t residentOveruse(PnrIndex count,
                              std::optional<std::uint64_t> capacity) {
  return capacity && count > *capacity ? count - *capacity : 0;
}

llvm::Error addDomainResidency(TagStateStorage &storage,
                               llvm::ArrayRef<PnrIndex> domains) {
  const auto matchDomains =
      storage.problem->routing().tagContinuity().matchDomains();
  std::uint64_t addedOveruse = 0;
  for (PnrIndex domain : domains) {
    if (domain >= storage.residentCounts.size() ||
        domain >= matchDomains.size())
      return invalid("segment names an out-of-range tag match domain");
    const PnrIndex count = storage.residentCounts[domain];
    if (count >= getInvalidPnrIndex() - PnrIndex{1})
      return invalid("tag match-domain residency overflows PnrIndex");
    const auto capacity = matchDomains[domain].residentEntryCapacity;
    addedOveruse +=
        residentOveruse(count + 1, capacity) - residentOveruse(count, capacity);
  }
  if (addedOveruse > std::numeric_limits<std::uint64_t>::max() -
                         storage.residentCapacityOveruse)
    return invalid("tag match-domain capacity overuse exceeds u64");
  for (PnrIndex domain : domains)
    ++storage.residentCounts[domain];
  storage.residentCapacityOveruse += addedOveruse;
  return llvm::Error::success();
}

void removeDomainResidency(TagStateStorage &storage,
                           llvm::ArrayRef<PnrIndex> domains) noexcept {
  const auto matchDomains =
      storage.problem->routing().tagContinuity().matchDomains();
  for (PnrIndex domain : domains) {
    assert(domain < storage.residentCounts.size() &&
           domain < matchDomains.size());
    const PnrIndex count = storage.residentCounts[domain];
    assert(count != 0);
    const auto capacity = matchDomains[domain].residentEntryCapacity;
    const std::uint64_t removedOveruse =
        residentOveruse(count, capacity) - residentOveruse(count - 1, capacity);
    assert(removedOveruse <= storage.residentCapacityOveruse);
    storage.residentCapacityOveruse -= removedOveruse;
    --storage.residentCounts[domain];
  }
}

llvm::Error addSegmentState(TagStateStorage &storage,
                            llvm::ArrayRef<PnrIndex> domains,
                            const std::optional<llvm::APInt> &value) {
  if (llvm::Error error =
          addAssignment(storage.occupancy, storage.unassignedCount,
                        storage.conflictCount, domains, value))
    return error;
  if (llvm::Error error = addDomainResidency(storage, domains)) {
    removeAssignment(storage.occupancy, storage.unassignedCount,
                     storage.conflictCount, domains, value);
    return error;
  }
  return llvm::Error::success();
}

void removeSegmentState(TagStateStorage &storage,
                        llvm::ArrayRef<PnrIndex> domains,
                        const std::optional<llvm::APInt> &value) noexcept {
  removeDomainResidency(storage, domains);
  removeAssignment(storage.occupancy, storage.unassignedCount,
                   storage.conflictCount, domains, value);
}

void removeNet(TagStateStorage &storage, const TagNetState &net) noexcept {
  for (PnrIndex segment = 0; segment < net.values.size(); ++segment)
    removeSegmentState(storage, segmentDomains(net, segment),
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
                     const TagNetState *oldNet, TagNetState &result,
                     std::optional<llvm::ArrayRef<llvm::APInt>> requiredValues,
                     llvm::ArrayRef<llvm::APInt> forbiddenValues) {
  result.values.assign(result.continuity.segments().size(), std::nullopt);
  std::vector<std::uint8_t> added(result.values.size(), 0);
  bool committed = false;
  const llvm::scope_exit rollback([&] {
    if (committed)
      return;
    for (PnrIndex segment = 0; segment < result.values.size(); ++segment)
      if (added[segment])
        removeSegmentState(storage, segmentDomains(result, segment),
                           result.values[segment]);
  });

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
  }

  if (requiredValues) {
    if (requiredValues->size() > result.values.size())
      return unsupported(
          "tag equality requires more values than one member has segments");
    if (requiredValues->empty()) {
      if (!result.values.empty())
        return unsupported(
            "nonempty tagged route cannot realize an empty equality set");
      committed = true;
      return llvm::Error::success();
    }

    std::vector<PnrIndex> valueForSegment(result.values.size(),
                                          getInvalidPnrIndex());
    const auto compatible = [&](PnrIndex value, PnrIndex segment) {
      const llvm::APInt &candidate = (*requiredValues)[value];
      const auto &descriptor = result.continuity.segments()[segment];
      return ::fabric::isRepresentablePhysicalTagValue(descriptor.tagWidthBits,
                                                       candidate) &&
             valueAllowed(candidate, restriction) &&
             isFree(segmentDomains(result, segment), candidate,
                    storage.occupancy);
    };
    std::function<bool(PnrIndex, std::vector<std::uint8_t> &)> matchValue =
        [&](PnrIndex value, std::vector<std::uint8_t> &visited) {
          for (PnrIndex segment = 0; segment < result.values.size();
               ++segment) {
            if (visited[segment] || !compatible(value, segment))
              continue;
            visited[segment] = 1;
            if (valueForSegment[segment] == getInvalidPnrIndex() ||
                matchValue(valueForSegment[segment], visited)) {
              valueForSegment[segment] = value;
              return true;
            }
          }
          return false;
        };
    for (PnrIndex value = 0; value < requiredValues->size(); ++value) {
      std::vector<std::uint8_t> visited(result.values.size(), 0);
      if (!matchValue(value, visited))
        return unsupported(
            "tag equality set has no representative segment matching");
    }
    for (PnrIndex segment = 0; segment < valueForSegment.size(); ++segment) {
      const PnrIndex value = valueForSegment[segment];
      if (value == getInvalidPnrIndex())
        continue;
      result.values[segment] = (*requiredValues)[value];
      if (llvm::Error error = addSegmentState(
              storage, segmentDomains(result, segment), result.values[segment]))
        return error;
      added[segment] = 1;
    }
    std::vector<PnrIndex> remaining;
    remaining.reserve(result.values.size() - requiredValues->size());
    for (PnrIndex segment = 0; segment < result.values.size(); ++segment)
      if (!result.values[segment])
        remaining.push_back(segment);
    llvm::sort(remaining, [&](PnrIndex lhs, PnrIndex rhs) {
      const auto left = segmentDomains(result, lhs).size();
      const auto right = segmentDomains(result, rhs).size();
      if (left != right)
        return left > right;
      return lhs < rhs;
    });
    for (PnrIndex segment : remaining) {
      const auto &descriptor = result.continuity.segments()[segment];
      const auto domains = segmentDomains(result, segment);
      for (const llvm::APInt &candidate : *requiredValues)
        if (::fabric::isRepresentablePhysicalTagValue(descriptor.tagWidthBits,
                                                      candidate) &&
            valueAllowed(candidate, restriction) &&
            isFree(domains, candidate, storage.occupancy)) {
          result.values[segment] = candidate;
          break;
        }
      if (!result.values[segment])
        return unsupported(
            "tag equality set cannot color one continuity segment");
      if (llvm::Error error =
              addSegmentState(storage, domains, result.values[segment]))
        return error;
      added[segment] = 1;
    }
    committed = true;
    return llvm::Error::success();
  }

  for (PnrIndex ordinal = 0; ordinal < result.continuity.segments().size();
       ++ordinal) {
    const SpatialTagContinuitySegment &segment =
        result.continuity.segments()[ordinal];
    const auto domains = segmentDomains(result, ordinal);
    std::optional<llvm::APInt> selected = preservedValue(oldNet, segment);
    if (selected && (!::fabric::isRepresentablePhysicalTagValue(
                         segment.tagWidthBits, *selected) ||
                     !valueAllowed(*selected, restriction) ||
                     containsValue(forbiddenValues, *selected) ||
                     !isFree(domains, *selected, storage.occupancy)))
      selected.reset();
    if (!selected) {
      auto chosen = chooseValue(segment.tagWidthBits, domains, restriction,
                                forbiddenValues, storage.occupancy);
      if (!chosen)
        return chosen.takeError();
      selected = std::move(*chosen);
    }
    result.values[ordinal] = std::move(selected);
    if (llvm::Error error =
            addSegmentState(storage, domains, result.values[ordinal]))
      return error;
    added[ordinal] = 1;
  }
  committed = true;
  return llvm::Error::success();
}

std::vector<llvm::APInt> projectedValues(const TagNetState &net) {
  std::vector<llvm::APInt> result;
  result.reserve(net.values.size());
  for (const auto &value : net.values)
    if (value)
      result.push_back(*value);
  normalizeValues(result);
  return result;
}

std::vector<llvm::APInt> disjointForbiddenValues(const TagStateStorage &storage,
                                                 PnrIndex equalityClass) {
  std::vector<llvm::APInt> result;
  for (PnrIndex group : storage.constraints->classDisjointGroups(equalityClass))
    for (PnrIndex peer : storage.constraints->disjointGroupMembers(group)) {
      if (peer == equalityClass || !storage.classBuilt[peer])
        continue;
      for (PnrIndex net : storage.constraints->classMembers(peer))
        for (const auto &value : storage.nets[net].values)
          if (value)
            result.push_back(*value);
    }
  normalizeValues(result);
  return result;
}

llvm::Error buildClass(TagStateStorage &storage, PnrIndex equalityClass,
                       const std::vector<TagNetState> *oldNets,
                       std::vector<PnrIndex> *builtNets = nullptr) {
  const auto members = storage.constraints->classMembers(equalityClass);
  if (members.empty())
    return invalid("Physical Tag equality class has no member");
  PnrIndex leader = members.front();
  for (PnrIndex net : members)
    if (storage.nets[net].continuity.segments().size() <
            storage.nets[leader].continuity.segments().size() ||
        (storage.nets[net].continuity.segments().size() ==
             storage.nets[leader].continuity.segments().size() &&
         net < leader))
      leader = net;
  std::vector<PnrIndex> order;
  order.reserve(members.size());
  order.push_back(leader);
  for (PnrIndex net : members)
    if (net != leader)
      order.push_back(net);

  const std::vector<llvm::APInt> forbidden =
      disjointForbiddenValues(storage, equalityClass);
  const TagNetState *oldLeader = oldNets ? &(*oldNets)[leader] : nullptr;
  if (llvm::Error error =
          buildNet(storage, leader, oldLeader, storage.nets[leader],
                   std::nullopt, forbidden))
    return error;
  if (builtNets)
    builtNets->push_back(leader);
  const std::vector<llvm::APInt> required =
      projectedValues(storage.nets[leader]);
  for (PnrIndex net : llvm::drop_begin(order)) {
    if (llvm::Error error = buildNet(storage, net, nullptr, storage.nets[net],
                                     required, forbidden))
      return error;
    if (builtNets)
      builtNets->push_back(net);
  }
  storage.classBuilt[equalityClass] = 1;
  return llvm::Error::success();
}

llvm::Error verifyRelations(const TagStateStorage &storage) {
  if (!storage.constraints->hasRelations())
    return llvm::Error::success();
  std::vector<std::vector<llvm::APInt>> classValues(
      storage.constraints->classCount());
  for (PnrIndex equalityClass = 0;
       equalityClass < storage.constraints->classCount(); ++equalityClass) {
    const auto members = storage.constraints->classMembers(equalityClass);
    if (members.empty())
      return invalid("Physical Tag equality class has no member");
    classValues[equalityClass] = projectedValues(storage.nets[members.front()]);
    for (PnrIndex net : members.drop_front())
      if (!equalUnsignedValues(projectedValues(storage.nets[net]),
                               classValues[equalityClass]))
        return invalid("Physical Tag value-set equality is violated");
  }
  for (PnrIndex equalityClass = 0;
       equalityClass < storage.constraints->classCount(); ++equalityClass)
    for (PnrIndex group :
         storage.constraints->classDisjointGroups(equalityClass)) {
      const auto members = storage.constraints->disjointGroupMembers(group);
      if (members.empty() || members.front() != equalityClass)
        continue;
      std::vector<llvm::APInt> observed;
      for (PnrIndex member : members) {
        for (const llvm::APInt &value : classValues[member])
          if (containsValue(observed, value))
            return invalid("Physical Tag value-set disjointness is violated");
        observed.insert(observed.end(), classValues[member].begin(),
                        classValues[member].end());
        normalizeValues(observed);
      }
    }
  return llvm::Error::success();
}

llvm::Expected<SpatialTagContinuityProjection>
deriveVerifiedSpatialTagContinuity(const RouteTreeState &route) {
  SpatialTagContinuityProjection result;
  SpatialTagContinuityScratch scratch;
  if (llvm::Error error =
          ::loom::pnr::detail::rebuildSpatialTagContinuityUnchecked(
              route, result, scratch))
    return std::move(error);
  return result;
}

enum class RouteReadMode : std::uint8_t {
  Stable,
  AlreadyVerified,
};

llvm::Expected<std::unique_ptr<TagStateStorage>>
buildStorage(const FrozenSpatialPnrProblem &problem,
             llvm::ArrayRef<const RouteTreeState *> routes,
             const std::vector<TagNetState> *oldNets = nullptr,
             RouteReadMode routeReadMode = RouteReadMode::Stable) {
  if (routes.size() != problem.transfers().logicalNets().size())
    return invalid("route count does not match the frozen logical nets");
  if (oldNets && oldNets->size() != routes.size())
    return invalid("preserved assignment count does not match the routes");
  auto storage = std::make_unique<TagStateStorage>();
  storage->problem = &problem;
  storage->constraints = &problem.tagConstraints();
  storage->nets.resize(routes.size());
  storage->occupancy.resize(
      problem.routing().tagContinuity().matchDomains().size());
  storage->residentCounts.assign(
      problem.routing().tagContinuity().matchDomains().size(), 0);
  storage->classBuilt.assign(storage->constraints->classCount(), 0);
  for (PnrIndex logicalNet = 0; logicalNet < routes.size(); ++logicalNet) {
    if (!routes[logicalNet] ||
        &routes[logicalNet]->routingGraph() != &problem.routing())
      return invalid("route does not belong to the frozen routing graph");
    auto continuity =
        routeReadMode == RouteReadMode::Stable
            ? deriveSpatialTagContinuity(*routes[logicalNet])
            : deriveVerifiedSpatialTagContinuity(*routes[logicalNet]);
    if (!continuity)
      return continuity.takeError();
    storage->nets[logicalNet].continuity = std::move(*continuity);
  }
  for (PnrIndex equalityClass = 0;
       equalityClass < storage->constraints->classCount(); ++equalityClass)
    if (llvm::Error error = buildClass(*storage, equalityClass, oldNets))
      return std::move(error);
  if (llvm::Error error = verifyRelations(*storage))
    return std::move(error);
  return storage;
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
  auto storage = buildStorage(problem, routes);
  if (!storage)
    return storage.takeError();

  SpatialTagAssignmentProjection result;
  result.netSegmentOffsets_.reserve(routes.size() + 1);
  result.netSegmentOffsets_.push_back(0);
  result.segmentDomainOffsets_.push_back(0);
  for (const TagNetState &net : (*storage)->nets) {
    const auto &continuity = net.continuity;
    if (llvm::Error error = preflightPnrIndexCapacity(
            segmentCountContext,
            result.segments_.size() + continuity.segments().size()))
      return std::move(error);
    result.segments_.insert(result.segments_.end(),
                            continuity.segments().begin(),
                            continuity.segments().end());
    result.values_.insert(result.values_.end(), net.values.begin(),
                          net.values.end());
    const auto localOffsets = continuity.segmentDomainOffsets();
    const auto localDomains = continuity.segmentDomains();
    for (PnrIndex segment = 0; segment < continuity.segments().size();
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

  result.unassignedCount_ = (*storage)->unassignedCount;
  result.conflictCount_ = (*storage)->conflictCount;
  result.residentCapacityOveruse_ = (*storage)->residentCapacityOveruse;
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
  storage_->routedNets.clear();
  storage_->routedNets.reserve(problem.transfers().logicalNets().size());
  storage_->rebuiltNets.clear();
  storage_->rebuiltNets.reserve(problem.transfers().logicalNets().size());
  return llvm::Error::success();
}

std::size_t SpatialTagAssignmentScratch::retainedStorageBytes() const {
  std::size_t bytes =
      storage_->stagedNets.capacity() * sizeof(detail::SpatialTagNetState) +
      storage_->touchedRoutes.capacity() * sizeof(PnrIndex) +
      storage_->routedNets.capacity() * sizeof(PnrIndex) +
      storage_->rebuiltNets.capacity() * sizeof(PnrIndex) +
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
  std::vector<const RouteTreeState *> rawRoutes;
  rawRoutes.reserve(routes.size());
  for (const RouteTreeStateHandle &route : routes)
    rawRoutes.push_back(route.get());
  auto storage = buildStorage(problem, rawRoutes);
  if (!storage)
    return storage.takeError();
  return SpatialTagAssignmentState(std::move(*storage));
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

std::uint64_t SpatialTagAssignmentState::residentCapacityOveruse() const {
  return storage_->residentCapacityOveruse;
}

std::uint64_t
SpatialTagAssignmentState::domainResidentCount(PnrIndex domain) const {
  assert(domain < storage_->residentCounts.size());
  return storage_->residentCounts[domain];
}

std::uint64_t SpatialTagAssignmentState::domainResidentCapacityOveruse(
    PnrIndex domain) const {
  assert(domain < storage_->residentCounts.size());
  const auto matchDomains =
      storage_->problem->routing().tagContinuity().matchDomains();
  assert(domain < matchDomains.size());
  return residentOveruse(storage_->residentCounts[domain],
                         matchDomains[domain].residentEntryCapacity);
}

std::uint64_t
SpatialTagAssignmentState::domainConflictCount(PnrIndex domain) const {
  assert(domain < storage_->occupancy.size());
  std::uint64_t conflicts = 0;
  for (const auto &entry : storage_->occupancy[domain]) {
    assert(entry.second != 0);
    conflicts += entry.second - 1;
  }
  assert(conflicts <= storage_->conflictCount);
  return conflicts;
}

bool SpatialTagAssignmentState::domainValueConflicts(
    PnrIndex domain, const llvm::APInt &value) const {
  assert(domain < storage_->occupancy.size());
  return storage_->occupancy[domain].lookup(value) > 1;
}

llvm::Expected<SpatialTagAssignmentSummary>
SpatialTagAssignmentState::projectVerifiedRoutes(
    llvm::ArrayRef<const RouteTreeState *> routes,
    bool includeDomainDetails) const {
  auto projected = buildStorage(*storage_->problem, routes, &storage_->nets,
                                RouteReadMode::AlreadyVerified);
  if (!projected)
    return projected.takeError();
  SpatialTagAssignmentSummary summary;
  summary.unassignedCount = (*projected)->unassignedCount;
  summary.conflictCount = (*projected)->conflictCount;
  summary.residentCapacityOveruse = (*projected)->residentCapacityOveruse;
  if (!includeDomainDetails)
    return summary;
  summary.domainResidentCounts.reserve((*projected)->residentCounts.size());
  summary.domainConflictCounts.reserve((*projected)->occupancy.size());
  for (PnrIndex domain = 0; domain < (*projected)->residentCounts.size();
       ++domain) {
    summary.domainResidentCounts.push_back(
        (*projected)->residentCounts[domain]);
    std::uint64_t conflicts = 0;
    for (const auto &entry : (*projected)->occupancy[domain])
      conflicts += entry.second - 1;
    summary.domainConflictCounts.push_back(conflicts);
  }

  summary.netDomainUseOffsets.reserve((*projected)->nets.size() + 1);
  summary.netUnassignedCounts.reserve((*projected)->nets.size());
  summary.netDomainUseOffsets.push_back(0);
  std::vector<PnrIndex> localDomains;
  for (const TagNetState &net : (*projected)->nets) {
    summary.netUnassignedCounts.push_back(llvm::count_if(
        net.values, [](const auto &value) { return !value.has_value(); }));
    localDomains.assign(net.continuity.segmentDomains().begin(),
                        net.continuity.segmentDomains().end());
    llvm::sort(localDomains);
    for (std::size_t begin = 0; begin < localDomains.size();) {
      std::size_t end = begin + 1;
      while (end < localDomains.size() &&
             localDomains[end] == localDomains[begin])
        ++end;
      summary.netDomainUseDomains.push_back(localDomains[begin]);
      summary.netDomainUseCounts.push_back(end - begin);
      begin = end;
    }
    summary.netDomainUseOffsets.push_back(summary.netDomainUseDomains.size());
  }
  return summary;
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

  transaction.routedNets.assign(touchedRoutes.begin(), touchedRoutes.end());
  llvm::sort(transaction.routedNets);
  transaction.routedNets.erase(
      std::unique(transaction.routedNets.begin(), transaction.routedNets.end()),
      transaction.routedNets.end());
  if (transaction.routedNets.empty())
    return llvm::Error::success();
  for (PnrIndex logicalNet : transaction.routedNets)
    if (logicalNet >= routes.size() || !routes[logicalNet] ||
        &routes[logicalNet]->routingGraph() != &storage_->problem->routing() ||
        !routeTransactions[logicalNet])
      return invalid("touched route does not belong to the frozen problem");

  std::vector<PnrIndex> affectedClasses;
  affectedClasses.reserve(transaction.routedNets.size());
  for (PnrIndex logicalNet : transaction.routedNets)
    affectedClasses.push_back(storage_->constraints->classOfNet(logicalNet));
  llvm::sort(affectedClasses);
  affectedClasses.erase(
      std::unique(affectedClasses.begin(), affectedClasses.end()),
      affectedClasses.end());
  transaction.touchedRoutes.clear();
  for (PnrIndex equalityClass : affectedClasses) {
    const auto members = storage_->constraints->classMembers(equalityClass);
    transaction.touchedRoutes.insert(transaction.touchedRoutes.end(),
                                     members.begin(), members.end());
  }
  llvm::sort(transaction.touchedRoutes);
  transaction.touchedRoutes.erase(std::unique(transaction.touchedRoutes.begin(),
                                              transaction.touchedRoutes.end()),
                                  transaction.touchedRoutes.end());

  transaction.active = true;
  transaction.rebuiltNets.clear();
  for (PnrIndex logicalNet : transaction.touchedRoutes) {
    removeNet(*storage_, storage_->nets[logicalNet]);
    std::swap(storage_->nets[logicalNet], transaction.stagedNets[logicalNet]);
  }
  for (PnrIndex equalityClass : affectedClasses)
    storage_->classBuilt[equalityClass] = 0;
  for (PnrIndex logicalNet : transaction.touchedRoutes) {
    if (std::binary_search(transaction.routedNets.begin(),
                           transaction.routedNets.end(), logicalNet)) {
      if (llvm::Error error =
              rebuildSpatialTagContinuity(*routeTransactions[logicalNet],
                                          storage_->nets[logicalNet].continuity,
                                          transaction.continuityScratch)) {
        rollback(scratch);
        return error;
      }
    } else {
      storage_->nets[logicalNet].continuity =
          transaction.stagedNets[logicalNet].continuity;
    }
  }
  for (PnrIndex equalityClass : affectedClasses)
    if (llvm::Error error =
            buildClass(*storage_, equalityClass, &transaction.stagedNets,
                       &transaction.rebuiltNets)) {
      rollback(scratch);
      return error;
    }
  if (llvm::Error error = verifyRelations(*storage_)) {
    rollback(scratch);
    return error;
  }
  return llvm::Error::success();
}

void SpatialTagAssignmentState::commit(
    SpatialTagAssignmentScratch &scratch) noexcept {
  auto &transaction = *scratch.storage_;
  if (!transaction.active)
    return;
  transaction.active = false;
  transaction.touchedRoutes.clear();
  transaction.routedNets.clear();
  transaction.rebuiltNets.clear();
}

void SpatialTagAssignmentState::rollback(
    SpatialTagAssignmentScratch &scratch) noexcept {
  auto &transaction = *scratch.storage_;
  if (!transaction.active)
    return;
  for (PnrIndex logicalNet : transaction.rebuiltNets)
    removeNet(*storage_, storage_->nets[logicalNet]);
  for (PnrIndex logicalNet : transaction.touchedRoutes) {
    std::swap(storage_->nets[logicalNet], transaction.stagedNets[logicalNet]);
    const TagNetState &restored = storage_->nets[logicalNet];
    for (PnrIndex segment = 0; segment < restored.values.size(); ++segment)
      llvm::cantFail(addSegmentState(*storage_,
                                     ::segmentDomains(restored, segment),
                                     restored.values[segment]));
  }
  for (PnrIndex logicalNet : transaction.touchedRoutes)
    storage_->classBuilt[storage_->constraints->classOfNet(logicalNet)] = 1;
  transaction.active = false;
  transaction.touchedRoutes.clear();
  transaction.routedNets.clear();
  transaction.rebuiltNets.clear();
}

llvm::Error SpatialTagAssignmentState::verify(
    llvm::ArrayRef<RouteTreeStateHandle> routes) const {
  if (routes.size() != storage_->nets.size())
    return invalid("candidate route count changed after Tag initialization");
  std::vector<llvm::DenseMap<llvm::APInt, PnrIndex>> expectedOccupancy(
      storage_->occupancy.size());
  std::vector<PnrIndex> expectedResidentCounts(storage_->occupancy.size(), 0);
  std::uint64_t expectedUnassigned = 0;
  std::uint64_t expectedConflicts = 0;
  std::uint64_t expectedResidentOveruse = 0;
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
      for (PnrIndex domain : domains) {
        if (expectedResidentCounts[domain] >=
            getInvalidPnrIndex() - PnrIndex{1})
          return invalid("tag match-domain residency overflows PnrIndex");
        ++expectedResidentCounts[domain];
      }
    }
  }
  for (PnrIndex domain = 0; domain < expectedResidentCounts.size(); ++domain) {
    const std::uint64_t overuse =
        residentOveruse(expectedResidentCounts[domain],
                        matchDomains[domain].residentEntryCapacity);
    if (overuse >
        std::numeric_limits<std::uint64_t>::max() - expectedResidentOveruse)
      return invalid("tag match-domain capacity overuse exceeds u64");
    expectedResidentOveruse += overuse;
  }
  if (expectedUnassigned != storage_->unassignedCount ||
      expectedConflicts != storage_->conflictCount ||
      expectedResidentCounts != storage_->residentCounts ||
      expectedResidentOveruse != storage_->residentCapacityOveruse)
    return invalid("cached Physical Tag violation counts have drifted");
  for (PnrIndex domain = 0; domain < storage_->occupancy.size(); ++domain) {
    if (storage_->occupancy[domain].size() != expectedOccupancy[domain].size())
      return invalid("cached Physical Tag occupancy has drifted");
    for (const auto &entry : storage_->occupancy[domain])
      if (expectedOccupancy[domain].lookup(entry.first) != entry.second)
        return invalid("cached Physical Tag occupancy has drifted");
  }
  if (llvm::Error error = verifyRelations(*storage_))
    return error;
  return llvm::Error::success();
}
